"""V8x dual-trigger adapter for fragility detection + CrisisFeedPayload."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class CrisisFeedPayload:
    """Payload sent to crisis scorer combining fragility + emotion distress."""

    fragility_score: float  # 0-1
    fragility_type: str  # "open" | "defensive" | "masked" | "denial" | "none"
    distress_corroboration: bool  # True if fragility aligns with emotion distress
    escalation_detected: bool  # True if fragility increased vs last session
    recommended_alert_level: str  # "crisis" | "high" | "medium" | "calm"


class V8xFragilityAdapter:
    """Adapter bridging fragility detector to V8x shared-understanding pipeline."""

    DISTRESS_THRESHOLD = 0.3

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
            return True, "Activated: " + "; ".join(reasons)
        return False, "Not activated: no vulnerability signals and low distress"

    @staticmethod
    def from_shared_understanding(
        shared_understanding: dict,
        behavioral_features: dict,
        emotion_distress: float = 0.0,
        session_history: Optional[list] = None,
    ) -> dict:
        """Build DetectorResult-compatible dict from shared understanding."""
        vuln = shared_understanding.get("vulnerability_signals", False)
        activated, reason = V8xFragilityAdapter.should_activate(vuln, emotion_distress)

        if not activated:
            return {
                "detector": "fragility",
                "activated": False,
                "reason": reason,
                "results": {"open": 0.0, "defensive": 0.0, "masked": 0.0, "denial": 0.0},
                "derived": {"fragility_score": 0.0, "dominant_type": "none"},
            }

        # Use signal estimation when activated
        emotion_keywords = shared_understanding.get("emotion_keywords", [])
        estimate = V8xFragilityAdapter.estimate_from_signals(
            vulnerability_signals=vuln,
            emotion_keywords=emotion_keywords,
            behavioral=behavioral_features,
        )

        return {
            "detector": "fragility",
            "activated": True,
            "reason": reason,
            "results": estimate["results"],
            "derived": estimate["derived"],
        }

    @staticmethod
    def build_crisis_feed(
        fragility_result: dict,
        emotion_distress: float,
        session_history: Optional[list] = None,
    ) -> CrisisFeedPayload:
        """Combine fragility output with emotion distress for crisis scorer."""
        derived = fragility_result.get("derived", {})
        frag_score = derived.get("fragility_score", 0.0)
        frag_type = derived.get("dominant_type", "none")

        # Corroboration: both fragility and emotion distress elevated
        corroboration = frag_score > 0.3 and emotion_distress > 0.3

        # Escalation: current score exceeds historical max by > 0.1
        escalation = False
        if session_history:
            historical_max = max(
                h.get("derived", {}).get("fragility_score", 0.0)
                for h in session_history
            )
            if frag_score > historical_max + 0.1:
                escalation = True

        # Alert level
        if frag_score > 0.7 and corroboration:
            alert = "high"
        elif frag_score > 0.5:
            alert = "medium"
        else:
            alert = "calm"

        return CrisisFeedPayload(
            fragility_score=frag_score,
            fragility_type=frag_type,
            distress_corroboration=corroboration,
            escalation_detected=escalation,
            recommended_alert_level=alert,
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
            # Defensive: hedging high
            if hedging > 0.1:
                scores["defensive"] = 0.3 + hedging
            # Open: negative emotion high, low hedging
            if neg_emo > 0.1 and hedging <= 0.1:
                scores["open"] = 0.3 + neg_emo
            # Masked: positive emotion in negative context
            if pos_emo > 0 and (neg_emo > 0 or len(emotion_keywords) > 0):
                scores["masked"] = 0.2 + pos_emo
            # Denial baseline when vulnerability present but nothing else fires strongly
            if all(v < 0.2 for v in scores.values()):
                scores["denial"] = 0.2
        else:
            # Low baseline from distress alone
            if neg_emo > 0.1:
                scores["open"] = 0.1 + neg_emo * 0.5

        # Clamp to [0, 1]
        scores = {k: max(0.0, min(1.0, v)) for k, v in scores.items()}

        # Derive dominant
        max_score = max(scores.values())
        dominant = max(scores, key=scores.get) if max_score > 0 else "none"
        fragility_score = max_score

        return {
            "results": scores,
            "derived": {
                "fragility_score": round(fragility_score, 3),
                "dominant_type": dominant,
            },
        }
