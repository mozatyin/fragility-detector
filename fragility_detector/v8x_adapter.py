"""V8x adapter: dual-trigger activation, emotion-enhanced detection, crisis feed.

Bridges fragility detector to V8x shared-understanding pipeline.
Handles: activation logic, emotion context integration, session history
enhancement (masked/defensive), crisis scorer output, degradation paths.
"""

from __future__ import annotations

import time
from typing import Optional

from fragility_detector.behavioral_features import extract_features, classify_from_features
from fragility_detector.models import (
    CrisisFeedPayload,
    FragilityDetectorInput,
    FragilityDetectorOutput,
    FragilityPattern,
)


class V8xFragilityAdapter:
    """Adapter bridging fragility detector to V8x shared-understanding pipeline."""

    DISTRESS_THRESHOLD = 0.3

    def __init__(self, api_key: str = ""):
        self._api_key = api_key
        self._detector = None  # lazy init

    def _get_detector(self):
        if self._detector is None:
            from fragility_detector.detector import FragilityDetector
            self._detector = FragilityDetector(self._api_key)
        return self._detector

    @staticmethod
    def should_activate(
        vulnerability_signals: bool,
        emotion_distress: float,
        activation_trigger: str = "",
    ) -> tuple[bool, str]:
        """Dual trigger: activate if vulnerability signals present OR emotion distress > 0.3."""
        reasons: list[str] = []

        if vulnerability_signals:
            reasons.append("vulnerability signals detected")
        if emotion_distress > V8xFragilityAdapter.DISTRESS_THRESHOLD:
            reasons.append(f"emotion distress {emotion_distress:.2f} > {V8xFragilityAdapter.DISTRESS_THRESHOLD}")

        if reasons:
            trigger = "both" if len(reasons) > 1 else (
                "vulnerability_signals" if vulnerability_signals else "distress_threshold"
            )
            return True, trigger
        return False, "none"

    def detect(self, inp: FragilityDetectorInput) -> FragilityDetectorOutput:
        """Full V8.x detection pipeline with emotion context + session history.

        Degradation: on any exception, returns timeout_unknown so crisis scorer stays alert.
        """
        start_ms = time.monotonic_ns() // 1_000_000

        # 1. Activation check
        activated, trigger = self.should_activate(
            inp.vulnerability_signals, inp.emotion_distress
        )
        if not activated:
            return FragilityDetectorOutput.not_activated(inp.session_id, inp.turn_id)

        try:
            # 2. Run core detection
            detector = self._get_detector()
            snapshot = detector.detect(
                conversation=inp.conversation,
                turn=inp.turn_id,
                emotion_distress=inp.emotion_distress,
            )

            # 3. Masked/defensive enhancement from session history + emotion context
            pattern = snapshot.pattern.value
            score = snapshot.pattern_scores.get(pattern, 0.0)
            confidence = snapshot.confidence

            pattern, score, confidence = self._enhance_with_context(
                pattern=pattern,
                score=score,
                confidence=confidence,
                pattern_scores=snapshot.pattern_scores,
                emotion_distress=inp.emotion_distress,
                session_history=inp.session_fragility_history,
            )

            # 4. Build crisis feed
            crisis_feed = self._build_crisis_feed(
                fragility_score=score,
                fragility_type=pattern,
                emotion_distress=inp.emotion_distress,
                session_history=inp.session_fragility_history,
            )

            elapsed = time.monotonic_ns() // 1_000_000 - start_ms

            return FragilityDetectorOutput(
                session_id=inp.session_id,
                turn_id=inp.turn_id,
                fragility_detected=score > 0.2 and confidence > 0.3,
                fragility_type=pattern,
                fragility_score=round(score, 3),
                confidence=round(confidence, 3),
                pattern_scores=snapshot.pattern_scores,
                crisis_feed=crisis_feed,
                activated=True,
                activation_trigger=trigger,
                latency_ms=elapsed,
                evidence=snapshot.evidence,
                llm_skipped="llm_skipped" in snapshot.evidence,
            )

        except Exception:
            elapsed = time.monotonic_ns() // 1_000_000 - start_ms
            out = FragilityDetectorOutput.timeout_unknown(inp.session_id, inp.turn_id)
            out.latency_ms = elapsed
            return out

    @staticmethod
    def _enhance_with_context(
        pattern: str,
        score: float,
        confidence: float,
        pattern_scores: dict,
        emotion_distress: float,
        session_history: Optional[list],
    ) -> tuple[str, float, float]:
        """Enhance masked/defensive detection using emotion context + session history.

        Key insight from V8.x spec:
        - distress high but text "fine" → possible masked/defensive
        - prior turn was open, current normalized → possible masked
        - distress_corroboration=False is strong masked signal
        """
        # Emotion-fragility discrepancy: high emotion distress but low fragility display
        discrepancy = emotion_distress - score if emotion_distress > 0.3 else 0.0

        if discrepancy > 0.2:
            # Strong discrepancy: emotion says distressed, fragility says calm
            # This is a strong signal for masked or defensive
            masked_score = pattern_scores.get("masked", 0.0)
            defensive_score = pattern_scores.get("defensive", 0.0)
            if masked_score > defensive_score:
                if pattern != "masked":
                    pattern = "masked"
                    score = max(score, 0.3 + discrepancy * 0.5)
                    confidence = max(confidence, 0.5)
            else:
                if pattern != "defensive" and pattern != "open":
                    pattern = "defensive"
                    score = max(score, 0.3 + discrepancy * 0.5)
                    confidence = max(confidence, 0.5)

        # Session history: detect normalization after open fragility
        if session_history:
            prev_types = [
                h.get("fragility_type", "none") if isinstance(h, dict)
                else getattr(h, "fragility_type", "none")
                for h in session_history[-3:]  # last 3 turns
            ]
            prev_scores = [
                h.get("fragility_score", 0.0) if isinstance(h, dict)
                else getattr(h, "fragility_score", 0.0)
                for h in session_history[-3:]
            ]

            # Was open/high before, now suddenly low → possible masked/defensive
            if prev_types and prev_types[-1] in ("open",) and score < 0.3:
                prev_max = max(prev_scores) if prev_scores else 0.0
                if prev_max > 0.5:
                    # Normalization after open fragility
                    if pattern_scores.get("masked", 0) >= pattern_scores.get("defensive", 0):
                        pattern = "masked"
                    else:
                        pattern = "defensive"
                    score = max(score, 0.35)
                    confidence = max(confidence, 0.45)

        return pattern, min(score, 1.0), min(confidence, 1.0)

    @staticmethod
    def _build_crisis_feed(
        fragility_score: float,
        fragility_type: str,
        emotion_distress: float,
        session_history: Optional[list],
    ) -> CrisisFeedPayload:
        """Build crisis scorer payload."""
        # Corroboration: both fragility and emotion distress elevated
        corroboration = fragility_score > 0.3 and emotion_distress > 0.3

        # Escalation: current score exceeds historical max by > 0.1
        escalation = False
        if session_history:
            historical_scores = [
                h.get("fragility_score", 0.0) if isinstance(h, dict)
                else getattr(h, "fragility_score", 0.0)
                for h in session_history
            ]
            if historical_scores:
                historical_max = max(historical_scores)
                if fragility_score > historical_max + 0.1:
                    escalation = True

        # Alert level
        if fragility_score > 0.7 and corroboration:
            alert = "critical"
        elif fragility_score > 0.7 or (fragility_score > 0.5 and corroboration):
            alert = "warn"
        elif fragility_score > 0.3:
            alert = "monitor"
        else:
            alert = "none"

        # Escalation overrides: at least "warn" if escalating with corroboration
        if escalation and corroboration and alert in ("none", "monitor"):
            alert = "warn"

        return CrisisFeedPayload(
            fragility_score=fragility_score,
            fragility_type=fragility_type,
            distress_corroboration=corroboration,
            escalation_detected=escalation,
            recommended_alert_level=alert,
        )

    # Legacy compatibility methods

    @staticmethod
    def from_shared_understanding(
        shared_understanding: dict,
        behavioral_features: dict,
        emotion_distress: float = 0.0,
        session_history: Optional[list] = None,
    ) -> dict:
        """Build DetectorResult-compatible dict from shared understanding."""
        vuln = shared_understanding.get("vulnerability_signals", False)
        activated, trigger = V8xFragilityAdapter.should_activate(vuln, emotion_distress)

        if not activated:
            return {
                "detector": "fragility",
                "activated": False,
                "reason": f"Not activated: {trigger}",
                "results": {"open": 0.0, "defensive": 0.0, "masked": 0.0, "denial": 0.0},
                "derived": {"fragility_score": 0.0, "dominant_type": "none"},
            }

        emotion_keywords = shared_understanding.get("emotion_keywords", [])
        estimate = V8xFragilityAdapter.estimate_from_signals(
            vulnerability_signals=vuln,
            emotion_keywords=emotion_keywords,
            behavioral=behavioral_features,
        )

        return {
            "detector": "fragility",
            "activated": True,
            "reason": f"Activated: {trigger}",
            "results": estimate["results"],
            "derived": estimate["derived"],
        }

    @staticmethod
    def build_crisis_feed_legacy(
        fragility_result: dict,
        emotion_distress: float,
        session_history: Optional[list] = None,
    ) -> CrisisFeedPayload:
        """Legacy: build crisis feed from dict result."""
        derived = fragility_result.get("derived", {})
        return V8xFragilityAdapter._build_crisis_feed(
            fragility_score=derived.get("fragility_score", 0.0),
            fragility_type=derived.get("dominant_type", "none"),
            emotion_distress=emotion_distress,
            session_history=[
                {"fragility_score": h.get("derived", {}).get("fragility_score", 0.0)}
                for h in (session_history or [])
            ],
        )

    @staticmethod
    def estimate_from_signals(
        vulnerability_signals: bool,
        emotion_keywords: list,
        behavioral: dict,
    ) -> dict:
        """Quick pattern estimation without full LLM call."""
        hedging = behavioral.get("hedging_ratio", 0.0)
        neg_emo = behavioral.get("neg_emotion_ratio", 0.0)
        pos_emo = behavioral.get("pos_emotion_ratio", 0.0)

        scores = {"open": 0.0, "defensive": 0.0, "masked": 0.0, "denial": 0.0}

        if vulnerability_signals:
            if hedging > 0.1:
                scores["defensive"] = 0.3 + hedging
            if neg_emo > 0.1 and hedging <= 0.1:
                scores["open"] = 0.3 + neg_emo
            if pos_emo > 0 and (neg_emo > 0 or len(emotion_keywords) > 0):
                scores["masked"] = 0.2 + pos_emo
            if all(v < 0.2 for v in scores.values()):
                scores["denial"] = 0.2
        else:
            if neg_emo > 0.1:
                scores["open"] = 0.1 + neg_emo * 0.5

        scores = {k: max(0.0, min(1.0, v)) for k, v in scores.items()}
        max_score = max(scores.values())
        dominant = max(scores, key=scores.get) if max_score > 0 else "none"

        return {
            "results": scores,
            "derived": {
                "fragility_score": round(max_score, 3),
                "dominant_type": dominant,
            },
        }
