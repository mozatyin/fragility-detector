"""Session-level fragility detection.

Accumulates per-turn signals across a conversation session, then derives
the overall fragility pattern from the trajectory. This solves the core
limitation of single-message detection:

- Defensive = "was distressed, then suddenly deflected" (requires trajectory)
- Masked = "humor markers co-occur with distressing topics" (requires topic context)
- Denial = "consistently low vulnerability despite distressing context" (requires baseline)
- Open = "vulnerability increases or stays high" (trajectory confirms)

Key insight: fragility pattern is about HOW someone CHANGES or MAINTAINS
their emotional disclosure across a conversation, not what one message says.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from fragility_detector.api_retry import make_client, retry_api_call
from fragility_detector.behavioral_features import extract_features
from fragility_detector.models import (
    FragilityPattern,
    FragilitySignals,
    FragilitySnapshot,
    STAR_LABELS,
)


@dataclass
class TurnSignal:
    """Signals extracted from a single turn."""
    turn: int
    distress: float = 0.0
    vulnerability_display: float = 0.0
    humor_as_shield: float = 0.0
    denial_strength: float = 0.0
    deflection_strength: float = 0.0
    text_length: int = 0
    has_humor_markers: bool = False
    has_vulnerability_words: bool = False
    has_deflection_phrases: bool = False


@dataclass
class SessionState:
    """Accumulated state across a session."""
    turns: list[TurnSignal] = field(default_factory=list)
    topic_is_distressing: bool = False  # has the conversation touched painful topics?
    peak_distress: float = 0.0         # highest distress seen in session
    peak_vulnerability: float = 0.0     # highest vulnerability display seen

    @property
    def n_turns(self) -> int:
        return len(self.turns)

    def add_turn(self, signal: TurnSignal):
        self.turns.append(signal)
        if signal.distress > self.peak_distress:
            self.peak_distress = signal.distress
        if signal.vulnerability_display > self.peak_vulnerability:
            self.peak_vulnerability = signal.vulnerability_display
        # R4: Topic is distressing if ANY signal is above threshold
        # (not just distress — defensive/denial suppress distress expression)
        max_signal = max(
            signal.distress,
            signal.vulnerability_display,
            signal.deflection_strength,
            signal.denial_strength,
            signal.humor_as_shield,
        )
        if max_signal > 0.25:
            self.topic_is_distressing = True


class SessionFragilityDetector:
    """Session-level fragility detection.

    Usage:
        detector = SessionFragilityDetector(api_key)
        # Feed conversation turns one by one or in batches
        result = detector.detect_session(conversation)
    """

    # Minimum turns to make a session-level call
    MIN_TURNS_FOR_SESSION = 3
    # Confidence threshold below which we say "insufficient data"
    INSUFFICIENT_CONFIDENCE = 0.3

    def __init__(self, api_key: str):
        self._client = make_client(api_key)

    def detect_session(
        self,
        conversation: list[dict],
        detect_interval: int = 3,
    ) -> FragilitySnapshot:
        """Detect fragility pattern from full conversation session.

        Processes conversation in windows, accumulates signals,
        then derives session-level pattern from trajectory.

        Args:
            conversation: Full conversation [{"role": str, "text": str}, ...]
            detect_interval: Run LLM detection every N speaker turns
        """
        state = SessionState()

        # Extract signals from each speaker turn
        speaker_turn_idx = 0
        for i, turn in enumerate(conversation):
            if turn.get("role") != "speaker":
                continue

            speaker_turn_idx += 1
            text = turn["text"]
            features = extract_features(text)

            # Run LLM detection at intervals or on first/last speaker turn
            is_first = speaker_turn_idx == 1
            is_interval = speaker_turn_idx % detect_interval == 0
            # Count remaining speaker turns
            remaining_speaker = sum(
                1 for t in conversation[i + 1:] if t.get("role") == "speaker"
            )
            is_last = remaining_speaker == 0

            # Fast path: skip LLM if behavioral says no signal
            total_signal = features.get("total_signal", 0)
            if (is_first or is_interval or is_last) and total_signal >= 0.08:
                # Use window around current turn for LLM detection
                window_start = max(0, i - 3)
                window = conversation[window_start:i + 1]
                llm_scores = self._llm_detect_signals(window)
            else:
                # Behavioral-only: no LLM needed
                llm_scores = {
                    "distress": 0.0, "vulnerability_display": 0.0,
                    "humor_as_shield": 0.0, "denial_strength": 0.0,
                    "deflection_strength": 0.0,
                }

            signal = TurnSignal(
                turn=i,
                distress=llm_scores.get("distress", 0.0),
                vulnerability_display=llm_scores.get("vulnerability_display", 0.0),
                humor_as_shield=llm_scores.get("humor_as_shield", 0.0),
                denial_strength=llm_scores.get("denial_strength", 0.0),
                deflection_strength=llm_scores.get("deflection_strength", 0.0),
                text_length=len(text.split()),
                has_humor_markers=features["humor_markers"] > 0.05,
                has_vulnerability_words=features["vulnerability_ratio"] > 0.03,
                has_deflection_phrases=features["deflection_ratio"] > 0.2,
            )
            state.add_turn(signal)

        # Derive session-level pattern from accumulated signals
        return self._derive_session_pattern(state)

    def _derive_session_pattern(self, state: SessionState) -> FragilitySnapshot:
        """Derive session-level fragility pattern from accumulated turn signals."""
        if not state.turns:
            return self._empty_snapshot()

        n = len(state.turns)

        # Aggregate signals across session
        avg_distress = sum(t.distress for t in state.turns) / n
        avg_vuln = sum(t.vulnerability_display for t in state.turns) / n
        avg_humor = sum(t.humor_as_shield for t in state.turns) / n
        avg_denial = sum(t.denial_strength for t in state.turns) / n
        avg_deflection = sum(t.deflection_strength for t in state.turns) / n

        # R4: Enhanced trajectory analysis
        if n >= 2:
            first_half = state.turns[:n // 2]
            second_half = state.turns[n // 2:]

            vuln_trend = (
                sum(t.vulnerability_display for t in second_half) / len(second_half)
                - sum(t.vulnerability_display for t in first_half) / len(first_half)
            )
            deflection_trend = (
                sum(t.deflection_strength for t in second_half) / len(second_half)
                - sum(t.deflection_strength for t in first_half) / len(first_half)
            )
            humor_trend = (
                sum(t.humor_as_shield for t in second_half) / len(second_half)
                - sum(t.humor_as_shield for t in first_half) / len(first_half)
            )
            denial_trend = (
                sum(t.denial_strength for t in second_half) / len(second_half)
                - sum(t.denial_strength for t in first_half) / len(first_half)
            )
        else:
            vuln_trend = 0.0
            deflection_trend = 0.0
            humor_trend = 0.0
            denial_trend = 0.0

        # R4: Transition detection — find "vulnerability drop" events
        # This is THE key signal for defensive: vulnerability was present, then disappeared
        vuln_drop = 0.0
        defl_spike = 0.0
        if n >= 2:
            for i in range(1, n):
                prev = state.turns[i - 1]
                curr = state.turns[i]
                # Vulnerability drop: previous turn had some vuln, current doesn't
                drop = prev.vulnerability_display - curr.vulnerability_display
                if drop > vuln_drop:
                    vuln_drop = drop
                # Deflection spike: current turn has much more deflection
                spike = curr.deflection_strength - prev.deflection_strength
                if spike > defl_spike:
                    defl_spike = spike

        # R4: Consistency metrics — how stable is each signal?
        if n >= 2:
            vuln_variance = sum((t.vulnerability_display - avg_vuln) ** 2 for t in state.turns) / n
            humor_consistency = 1.0 - (sum((t.humor_as_shield - avg_humor) ** 2 for t in state.turns) / max(n, 1))
        else:
            vuln_variance = 0.0
            humor_consistency = 0.0

        # Count turns with specific patterns
        humor_turns = sum(1 for t in state.turns if t.humor_as_shield > 0.3)
        deflection_turns = sum(1 for t in state.turns if t.deflection_strength > 0.3)
        denial_turns = sum(1 for t in state.turns if t.denial_strength > 0.3)
        open_turns = sum(1 for t in state.turns if t.vulnerability_display > 0.3)

        # R4: Session-level pattern scoring with trajectory and transition features
        # OPEN: sustained/increasing vulnerability display
        open_score = (
            avg_vuln * 0.35
            + avg_distress * 0.15
            + max(0, vuln_trend) * 0.2        # increasing vulnerability over time
            + (open_turns / n) * 0.2          # proportion of open turns
            - vuln_drop * 0.15                # penalty if vulnerability dropped (→ defensive)
            - avg_deflection * 0.1
        )

        # DEFENSIVE: deflection + vulnerability transitions
        # R4: key insight — defensive IS the transition from open to closed
        defensive_score = (
            avg_deflection * 0.3
            + max(0, avg_distress - avg_vuln) * 0.2    # distress hidden
            + vuln_drop * 0.2                           # R4: vulnerability DROP is key signal
            + defl_spike * 0.15                         # R4: sudden deflection increase
            + max(0, deflection_trend) * 0.1            # increasing deflection over time
            + (deflection_turns / n) * 0.1
            - avg_vuln * 0.1                            # penalty for actual openness
        )

        # MASKED: consistent humor across distressing topics
        masked_score = (
            avg_humor * 0.4
            + min(avg_distress, avg_humor) * 0.2        # distress + humor co-occurrence
            + (humor_turns / n) * 0.2                   # proportion of humor turns
            + max(0, humor_consistency) * 0.1           # R4: consistent humor pattern
            - avg_vuln * 0.15                           # penalty for genuine openness
        )

        # DENIAL: persistent denial signals
        denial_score = (
            avg_denial * 0.4
            + (denial_turns / n) * 0.25
            + max(0, state.peak_distress - avg_vuln) * 0.15  # context suggests pain, person denies
            - avg_deflection * 0.1
        )

        # Normalize
        scores = {
            "open": open_score,
            "defensive": defensive_score,
            "masked": masked_score,
            "denial": denial_score,
        }
        min_s = min(scores.values())
        shifted = {k: v - min_s + 0.01 for k, v in scores.items()}
        total = sum(shifted.values())
        normalized = {k: v / total for k, v in shifted.items()}

        best_pattern = max(normalized, key=normalized.get)
        confidence = normalized[best_pattern]

        # Confidence adjustments
        # More turns = more confidence
        turn_factor = min(1.0, n / 5)  # max confidence boost at 5+ turns
        confidence *= (0.5 + 0.5 * turn_factor)

        # If no distressing context at all, low confidence
        if not state.topic_is_distressing:
            confidence *= 0.3

        # Build signals summary
        signals = FragilitySignals(
            distress=avg_distress,
            vulnerability_display=avg_vuln,
            humor_markers=avg_humor,
            negation_ratio=avg_denial,
            deflection_ratio=avg_deflection,
        )

        return FragilitySnapshot(
            turn=state.turns[-1].turn if state.turns else 0,
            pattern=FragilityPattern(best_pattern),
            pattern_scores=normalized,
            signals=signals,
            confidence=round(min(1.0, confidence), 3),
            evidence={
                "n_turns_analyzed": str(n),
                "peak_distress": f"{state.peak_distress:.3f}",
                "vulnerability_trend": f"{vuln_trend:+.3f}",
                "vuln_drop": f"{vuln_drop:.3f}",
                "defl_spike": f"{defl_spike:.3f}",
                "topic_is_distressing": str(state.topic_is_distressing),
                "open_turns": str(open_turns),
                "deflection_turns": str(deflection_turns),
                "humor_turns": str(humor_turns),
                "denial_turns": str(denial_turns),
            },
        )

    def _empty_snapshot(self) -> FragilitySnapshot:
        return FragilitySnapshot(
            turn=0,
            pattern=FragilityPattern.OPEN,
            pattern_scores={"open": 0.25, "defensive": 0.25, "masked": 0.25, "denial": 0.25},
            signals=FragilitySignals(),
            confidence=0.0,
            evidence={"error": "no speaker turns found"},
        )

    def _llm_detect_signals(self, window: list[dict]) -> dict:
        """Run LLM signal detection on a conversation window. Uses compressed shared prompt."""
        from fragility_detector.detector import SYSTEM_PROMPT

        conv_text = "\n".join(
            f"[{t.get('role', 'unknown')}]: {t['text']}" for t in window
        )

        import json as _json
        for attempt in range(3):
            response = retry_api_call(
                lambda: self._client.messages.create(
                    model="anthropic/claude-sonnet-4",
                    max_tokens=400,
                    temperature=0.0,
                    system=SYSTEM_PROMPT,
                    messages=[{
                        "role": "user",
                        "content": conv_text,
                    }],
                )
            )
            try:
                return self._parse_response(response.content[0].text)
            except (_json.JSONDecodeError, KeyError, IndexError):
                if attempt == 2:
                    raise
                continue

    @staticmethod
    def _parse_response(raw: str) -> dict:
        """Parse LLM JSON response."""
        import json as _json
        import re as _re
        text = raw.strip()
        json_marker = text.find("JSON:")
        if json_marker != -1:
            text = text[json_marker + 5:].strip()
        if text.startswith("```"):
            text = _re.sub(r'^```\w*\n?', '', text)
            text = _re.sub(r'\n?```$', '', text)
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1:
            text = text[start:end + 1]
        data = _json.loads(text)
        for key in ["distress", "vulnerability_display", "humor_as_shield",
                     "denial_strength", "deflection_strength"]:
            if key in data:
                data[key] = max(0.0, min(1.0, float(data[key])))
        return data
