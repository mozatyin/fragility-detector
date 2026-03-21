"""FragilityDetector: LLM-based fragility pattern detection from conversation text."""

from __future__ import annotations

import json
import re

from fragility_detector.api_retry import make_client, retry_api_call
from fragility_detector.behavioral_features import extract_features, classify_from_features
from fragility_detector.models import (
    FragilityPattern,
    FragilitySignals,
    FragilitySnapshot,
)


# Text anchors for each fragility pattern at 2 intensity levels (0.2 and 0.6; interpolate for 0.4/0.8)
FRAGILITY_ANCHORS = {
    "open": (
        "**Open** (directly expresses vulnerability):\n"
        "- 0.2: one brief admission, 'it's been hard'\n"
        "- 0.6: sustained disclosure, 'I've been crying all day', multiple vulnerable statements"
    ),
    "defensive": (
        "**Defensive** (deflects/minimizes vulnerability):\n"
        "- 0.2: one quick subject change, 'anyway...'\n"
        "- 0.6: persistent minimizing, 'it doesn't matter', 'I don't want to talk about it'"
    ),
    "masked": (
        "**Masked** (hides behind humor/casualness):\n"
        "- 0.2: one self-deprecating joke in emotional context, 'haha yeah that sucks'\n"
        "- 0.6: sustained humor over serious pain, 'at least I'm consistent at failing haha'"
    ),
    "denial": (
        "**Denial** (denies vulnerability exists):\n"
        "- 0.2: one claim of being unbothered, 'whatever'\n"
        "- 0.6: aggressive independence, 'feelings are weakness', 'I can handle anything alone'"
    ),
}

# Signals the LLM should rate
SIGNAL_DEFINITIONS = """
## Signals (0.0–1.0) for SPEAKER's message:
1. **distress**: Emotional pain. 0=neutral/casual, 0.3=mild, 0.6=suffering, 0.9=crisis
2. **vulnerability_display**: Openness of vulnerability. 0=guarded/absent, 0.3=hints, 0.6=clear, 0.9=raw
3. **humor_as_shield**: Humor deflecting FROM PAIN (not just being funny). 0=none, 0.3=occasional, 0.6=persistent, 0.9=constant. Genuine humor without pain=0.0.
4. **denial_strength**: ACTIVE rejection of vulnerability/emotions. 0=none, 0.3=mild, 0.6=assertive, 0.9=aggressive. Low vulnerability alone ≠ denial; requires EXPLICIT claims of invulnerability.
5. **deflection_strength**: Subject changes, minimizing, redirecting. 0=none, 0.3=one redirect, 0.6=persistent, 0.9=shutdown
"""

# Key disambiguation rules added in R2
DISAMBIGUATION_RULES = """
## Disambiguation Rules

### Defensive vs Denial (DIFFERENT patterns)
- **Defensive**: KNOWS vulnerability exists, AVOIDS it — subject changes, minimizing, hedging, "I don't want to talk about it"
- **Denial**: REJECTS vulnerability exists — claims invulnerability, attacks emotions as weakness, "I don't HAVE feelings about it"
- Angry denial (attacking sentiment, "Bah! Humbug!") = denial, not defensive — they ATTACK the premise feelings matter

### Open Anger vs Denial
- "I hate you" = OPEN (intense emotional engagement). "I don't care about you" = denial (claims no emotion).
- Open anger expresses raw emotion through rage. Denial dismisses emotion as irrelevant.

### Masked vs Genuine Humor vs Open-with-humor
- **Masked**: pain + humor TOGETHER, humor AVOIDS the pain. "haha I got fired, at least I can sleep in lol"
- **Genuine humor**: no underlying pain → humor_as_shield = 0.0
- **Open-with-humor**: names pain directly even if dramatic — "I'm desperate for love!" = open, not masked
- Key test: does humor AVOID or EXPOSE the pain? Avoid=masked, expose=open

### Special Cases
- **Literary/formal language**: rate EMOTIONAL CONTENT, not register. Ornate pain is still pain.
- **Shock/dissociation**: flat tone + severe trauma = open (overwhelming vulnerability), NOT defensive. Flat affect alone ≠ deflection.
- **Gratitude/farewell**: "thank you for listening" = 0.0 vulnerability. Don't infer prior hopelessness.
- **Neutral text**: no emotional distress → all signals < 0.2. Don't force a pattern.
"""


class FragilityDetector:
    """Detect fragility patterns from conversation text using LLM + behavioral features."""

    # R2: Minimum distress threshold for meaningful classification
    DISTRESS_THRESHOLD = 0.15

    def __init__(self, api_key: str):
        self._client = make_client(api_key)

    def detect(
        self,
        conversation: list[dict],
        turn: int,
        window_size: int = 6,
    ) -> FragilitySnapshot:
        """Detect fragility pattern from conversation window.

        Args:
            conversation: List of {"role": str, "text": str} dicts
            turn: Current turn number
            window_size: How many recent turns to analyze
        """
        window = conversation[-window_size:]

        # Extract behavioral features from speaker turns (zero cost)
        speaker_texts = [t["text"] for t in window if t.get("role") == "speaker"]
        combined_text = " ".join(speaker_texts) if speaker_texts else window[-1]["text"]
        features = extract_features(combined_text)
        behavioral_scores = classify_from_features(features)
        beh_signal = features.get("total_signal", 0)
        beh_is_uniform = all(abs(v - 0.25) < 0.02 for v in behavioral_scores.values())
        beh_best = max(behavioral_scores, key=behavioral_scores.get)
        beh_conf = behavioral_scores[beh_best]

        # R9: Behavioral skip — if behavioral is very confident, skip LLM
        if not beh_is_uniform and beh_signal >= 0.2 and beh_conf >= 0.6:
            # Strong behavioral signal for open/masked/denial → skip LLM
            signals = FragilitySignals(
                self_ref_ratio=features["self_ref_ratio"],
                hedging_ratio=features["hedging_ratio"],
                humor_markers=features["humor_markers"],
                negation_ratio=features.get("negation_ratio", 0.0),
                deflection_ratio=features.get("deflection_ratio", 0.0),
            )
            total = sum(behavioral_scores.values())
            merged = {k: v / total for k, v in behavioral_scores.items()}
            best = max(merged, key=merged.get)
            return FragilitySnapshot(
                turn=turn,
                pattern=FragilityPattern(best),
                pattern_scores=merged,
                signals=signals,
                confidence=round(merged[best] * 0.85, 3),  # slight penalty vs LLM
                evidence={"llm_skipped": "true", "beh_signal": f"{beh_signal:.3f}"},
                raw_llm_scores={},
            )

        # LLM detection (full prompt, only when behavioral is uncertain)
        llm_scores = self._llm_detect(window)

        # Build signals from LLM + behavioral features
        signals = FragilitySignals(
            distress=llm_scores.get("distress", 0.0),
            self_ref_ratio=features["self_ref_ratio"],
            hedging_ratio=features["hedging_ratio"],
            vulnerability_display=llm_scores.get("vulnerability_display", 0.0),
            humor_markers=features["humor_markers"],
            negation_ratio=features["negation_ratio"],
            deflection_ratio=features["deflection_ratio"],
        )

        # R2: Check if there's enough vulnerability signal to classify
        max_signal = max(
            llm_scores.get("distress", 0.0),
            llm_scores.get("vulnerability_display", 0.0),
            llm_scores.get("humor_as_shield", 0.0),
            llm_scores.get("denial_strength", 0.0),
            llm_scores.get("deflection_strength", 0.0),
        )
        insufficient_signal = max_signal < self.DISTRESS_THRESHOLD

        # Classify pattern: combine LLM pattern scores with behavioral features
        llm_pattern_scores = self._derive_pattern_scores(llm_scores)
        # behavioral_scores already computed above (before LLM skip check)

        if beh_is_uniform or beh_signal < 0.08:
            # No behavioral signal → 100% LLM
            llm_weight, beh_weight = 1.0, 0.0
        elif beh_signal > 0.3:
            # Strong behavioral signal → 50/50
            llm_weight, beh_weight = 0.5, 0.5
        else:
            # Moderate signal → 70/30 (original)
            llm_weight, beh_weight = 0.7, 0.3

        merged = {}
        for pattern in ["open", "defensive", "masked", "denial"]:
            merged[pattern] = (
                llm_weight * llm_pattern_scores.get(pattern, 0.25)
                + beh_weight * behavioral_scores.get(pattern, 0.25)
            )

        # Normalize
        total = sum(merged.values())
        if total > 0:
            merged = {k: v / total for k, v in merged.items()}

        # Pick winner
        best_pattern = max(merged, key=merged.get)
        confidence = merged[best_pattern]

        # R2: Penalize confidence when signal is insufficient
        if insufficient_signal:
            confidence *= 0.3

        return FragilitySnapshot(
            turn=turn,
            pattern=FragilityPattern(best_pattern),
            pattern_scores=merged,
            signals=signals,
            confidence=confidence,
            evidence=llm_scores.get("evidence", {}),
            raw_llm_scores=llm_scores,
        )

    # R4: Data-driven weights from LogisticRegression on 30 labeled cases
    # LOO accuracy: 100%. Replaces hand-tuned R2 formula.
    _DERIVE_WEIGHTS = None

    @classmethod
    def _load_derive_weights(cls):
        """Load learned weights from JSON. Falls back to hand-tuned if not found."""
        if cls._DERIVE_WEIGHTS is not None:
            return cls._DERIVE_WEIGHTS

        import json as _json
        from pathlib import Path
        weights_path = Path(__file__).parent / "data" / "derive_weights.json"
        if weights_path.exists():
            with open(weights_path) as f:
                cls._DERIVE_WEIGHTS = _json.load(f)
        else:
            cls._DERIVE_WEIGHTS = {}
        return cls._DERIVE_WEIGHTS

    def _derive_pattern_scores(self, llm_scores: dict) -> dict[str, float]:
        """Derive pattern classification from LLM signal scores.

        R4: Uses learned LogisticRegression weights (LOO 100% on 30 cases).
        Falls back to simple max-signal heuristic if weights not found.
        """
        signals = [
            llm_scores.get("distress", 0.0),
            llm_scores.get("vulnerability_display", 0.0),
            llm_scores.get("humor_as_shield", 0.0),
            llm_scores.get("denial_strength", 0.0),
            llm_scores.get("deflection_strength", 0.0),
        ]

        model = self._load_derive_weights()
        if not model or "weights" not in model:
            return self._derive_pattern_scores_fallback(llm_scores)

        # Compute logit for each pattern
        import math
        logits = {}
        for cls_name in model["classes"]:
            w = model["weights"][cls_name]
            b = model["intercepts"][cls_name]
            logit = b + sum(w[i] * signals[i] for i in range(len(signals)))
            logits[cls_name] = logit

        # Softmax
        max_logit = max(logits.values())
        exp_logits = {k: math.exp(v - max_logit) for k, v in logits.items()}
        total = sum(exp_logits.values())
        return {k: v / total for k, v in exp_logits.items()}

    @staticmethod
    def _derive_pattern_scores_fallback(llm_scores: dict) -> dict[str, float]:
        """Fallback: simple signal-based classification when no trained weights."""
        scores = {
            "open": llm_scores.get("vulnerability_display", 0.0),
            "defensive": llm_scores.get("deflection_strength", 0.0),
            "masked": llm_scores.get("humor_as_shield", 0.0),
            "denial": llm_scores.get("denial_strength", 0.0),
        }
        min_s = min(scores.values())
        shifted = {k: v - min_s + 0.01 for k, v in scores.items()}
        total = sum(shifted.values())
        return {k: v / total for k, v in shifted.items()}

    def _llm_detect(self, window: list[dict]) -> dict:
        """Run LLM detection on conversation window."""
        conv_text = "\n".join(
            f"[{t.get('role', 'unknown')}]: {t['text']}" for t in window
        )

        anchors_text = "\n\n".join(FRAGILITY_ANCHORS.values())

        system_prompt = f"""Analyze vulnerability patterns in conversation.

## Patterns (interpolate between anchors)
{anchors_text}

{SIGNAL_DEFINITIONS}

{DISAMBIGUATION_RULES}

## Calibration
- "I'm fine" is often NOT fine. "haha" after pain = mask, not genuine amusement.
- Dismissive responses ("ok","whatever") in emotional context = defensive or denial.
- High self-ref + hedging = Open. Humor + pain = Masked. "I don't need anyone" = Denial. Subject changes = Defensive.
- Mixed patterns common — rate all signals honestly. No emotional content → all signals near 0.0.

## Two-Step Analysis (CRITICAL)
1. **CONTENT**: What emotional topic? This tells you IF vulnerability is present.
2. **DELIVERY**: HOW they say it → determines PATTERN:
   - Raw/unguarded → open. Detached/cold → denial or deflection. Jokes/sarcasm → masked. Subject change/minimizing → defensive.

DELIVERY determines pattern, NOT content. Vulnerable content + detached delivery → denial, not open.
- "I found out it was easier to be him than to start over" → cold delivery = denial
- "I will be calm. I will be mistress of myself" → controlling emotions = defensive (not denial — managing, not rejecting)
- Defensive ACKNOWLEDGES feelings but avoids them. Denial REJECTS that feelings exist.

## Output
REASONING:
- Content: [topic]
- Delivery: [raw/detached/humorous/deflecting]

JSON:
{{"distress": 0.0, "vulnerability_display": 0.0, "humor_as_shield": 0.0, "denial_strength": 0.0, "deflection_strength": 0.0, "evidence": {{"most_revealing_quote": "...", "pattern_indicator": "..."}}}}"""

        for attempt in range(3):
            response = retry_api_call(
                lambda: self._client.messages.create(
                    model="anthropic/claude-sonnet-4",
                    max_tokens=400,  # R9: reduced from 800 (only need brief reasoning + JSON)
                    temperature=0.0,
                    system=system_prompt,
                    messages=[{
                        "role": "user",
                        "content": f"Analyze the SPEAKER's vulnerability expression pattern:\n\n{conv_text}",
                    }],
                )
            )
            try:
                return self._parse_response(response.content[0].text)
            except (json.JSONDecodeError, KeyError, IndexError):
                if attempt == 2:
                    raise
                continue

    @staticmethod
    def _parse_response(raw: str) -> dict:
        """Parse LLM JSON response."""
        text = raw.strip()

        # Extract JSON after "JSON:" marker
        json_marker = text.find("JSON:")
        if json_marker != -1:
            text = text[json_marker + 5:].strip()

        # Strip markdown fences
        if text.startswith("```"):
            text = re.sub(r'^```\w*\n?', '', text)
            text = re.sub(r'\n?```$', '', text)

        # Find first { to last }
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1:
            text = text[start:end + 1]

        data = json.loads(text)

        # Clamp signal values to [0, 1]
        for key in ["distress", "vulnerability_display", "humor_as_shield",
                     "denial_strength", "deflection_strength"]:
            if key in data:
                data[key] = max(0.0, min(1.0, float(data[key])))

        return data
