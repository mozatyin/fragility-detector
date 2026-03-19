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


# Text anchors for each fragility pattern at 4 intensity levels
FRAGILITY_ANCHORS = {
    "open": (
        "**Open Vulnerability** (directly expresses vulnerability):\n"
        "- 0.2: slight openness — one brief admission of difficulty, 'it's been hard'\n"
        "- 0.4: clear vulnerability — sharing feelings directly, 'I feel like giving up', 'I'm scared'\n"
        "- 0.6: strong openness — sustained emotional disclosure, 'I've been crying all day', multiple vulnerable statements\n"
        "- 0.8: raw vulnerability — completely unguarded, 'I'm broken', 'I don't know how to go on', every sentence reveals inner pain"
    ),
    "defensive": (
        "**Defensive Vulnerability** (deflects/minimizes vulnerability):\n"
        "- 0.2: slight deflection — one quick subject change after emotional topic, 'anyway...'\n"
        "- 0.4: clear defensiveness — 'I'm fine, it's not a big deal', redirecting to practical matters, hedging\n"
        "- 0.6: strong deflection — persistent minimizing, 'it doesn't matter', 'I don't want to talk about it', walls up\n"
        "- 0.8: fortress mode — complete emotional shutdown when vulnerability approached, counter-questions, blame-shifting"
    ),
    "masked": (
        "**Masked Vulnerability** (hides behind humor/casualness):\n"
        "- 0.2: slight masking — one self-deprecating joke in emotional context, 'haha yeah that sucks'\n"
        "- 0.4: clear masking — joking about painful topics, 'lol I got dumped again', using humor as shield\n"
        "- 0.6: strong masking — sustained humor over serious pain, 'at least I'm consistent at failing haha', laughing through tears energy\n"
        "- 0.8: complete mask — every painful topic immediately wrapped in humor, 'my life is a comedy show', unable to be serious about vulnerability"
    ),
    "denial": (
        "**Denial of Vulnerability** (denies vulnerability exists):\n"
        "- 0.2: slight denial — one claim of being unbothered, 'whatever'\n"
        "- 0.4: clear denial — 'I don't need anyone', 'I'm totally fine', asserting independence/strength\n"
        "- 0.6: strong denial — aggressive independence, 'feelings are weakness', 'I can handle anything alone'\n"
        "- 0.8: total denial — identity built on invulnerability, 'nothing can hurt me', rejecting any emotional connection as weakness"
    ),
}

# Signals the LLM should rate
SIGNAL_DEFINITIONS = """
## Signals to Rate (0.0 to 1.0)
Rate each of these signals for the SPEAKER's current message:

1. **distress**: Overall emotional pain/distress level. Rate 0.0 if conversation is neutral/casual with NO emotional content.
   - 0.0: no distress at all (casual chat, factual exchange)
   - 0.3: mild discomfort  0.6: clear suffering  0.9: acute crisis

2. **vulnerability_display**: How openly the speaker shows vulnerability
   - 0.0: completely guarded OR no vulnerability present  0.3: hints at vulnerability  0.6: clearly vulnerable  0.9: raw, unguarded

3. **humor_as_shield**: Using humor/casualness to deflect FROM PAIN specifically (not just being funny)
   - 0.0: no humor-shielding (genuine humor without underlying pain is 0.0)
   - 0.3: occasional jokes about pain  0.6: persistent humor-deflection  0.9: everything painful wrapped in jokes
   - CRITICAL: Genuine lighthearted conversation = 0.0, NOT humor_as_shield. Only rate > 0 when humor coexists with identifiable pain/distress.

4. **denial_strength**: Active denial or rejection of vulnerability. Must involve EXPLICIT claims of invulnerability or rejection of emotional needs.
   - 0.0: no denial  0.3: mild 'I'm fine'  0.6: assertive 'I don't need help'  0.9: aggressive rejection of any vulnerability
   - CRITICAL: Low vulnerability_display alone does NOT mean denial. Denial requires ACTIVE assertion of strength/invulnerability.

5. **deflection_strength**: Changing subject, minimizing, redirecting away from feelings
   - 0.0: no deflection  0.3: one quick redirect  0.6: persistent avoidance  0.9: complete emotional shutdown
"""

# Key disambiguation rules added in R2
DISAMBIGUATION_RULES = """
## Critical Disambiguation Rules

### Defensive vs Denial
These are DIFFERENT patterns. Do NOT confuse them:
- **Defensive** = "I don't want to talk about it" → AVOIDS the topic, changes subject, hedges
  - Key signals: subject changes ("anyway..."), minimizing ("it's not a big deal"), hedging ("I guess"), redirecting to practical matters
  - The person KNOWS they're vulnerable but DEFLECTS from it
- **Denial** = "I don't HAVE feelings about it" → REJECTS the premise of vulnerability
  - Key signals: claims of invulnerability ("nothing hurts me"), rejecting emotions as weakness ("feelings are for weak people"), asserting total independence ("I don't need anyone")
  - The person DENIES vulnerability EXISTS in them
  - **Angry denial**: Some people deny through AGGRESSION — "Bah! Humbug!", attacking sentiment itself, contempt for emotional expression. This IS denial, not defensive. They're not avoiding the topic, they're ATTACKING the premise that feelings matter.
  - Rate denial_strength HIGH when someone: dismisses emotions as foolish/weak, attacks people for being emotional, asserts that sentiment/love/caring are pointless

### Open Anger vs Denial
- **Open anger** = directly expressing rage, pain, hurt AT someone: "I hate you", "You treated me with miserable cruelty", "I will never forgive you"
  - This is OPEN vulnerability — the person IS expressing raw emotion, just through anger instead of sadness
  - Rate high distress + high vulnerability_display. Do NOT rate as denial.
- **Denial** = rejecting that emotions EXIST or MATTER: "I don't feel anything", "Emotions are weakness"
  - Key difference: open anger ENGAGES with emotion intensely. Denial DISMISSES emotion as irrelevant.
  - "I hate you" = open (full emotional engagement). "I don't care about you" = denial (claims no emotion).

### Literary / Formal Language
- Formal, eloquent language does NOT mean defensive. Historical characters express vulnerability differently than modern speakers.
- "I cry because I am miserable" in formal English = same as "I'm so sad I can't stop crying" in modern English.
- Rate the EMOTIONAL CONTENT, not the language register. Ornate expression of pain is still pain.

### Masked vs Genuine Humor vs Open-with-humor
- **Masked** = humor as SHIELD — painful topic + humor IN THE SAME utterance deflecting from it
  - REQUIRES: identifiable painful content + humor markers TOGETHER in a way that AVOIDS engaging with the pain
  - Example: "haha I got fired, at least I can sleep in now lol" — pain is job loss, humor deflects from it
- **Genuine humor** = just being funny, no underlying pain → humor_as_shield = 0.0
- **Open-with-humor** = directly stating pain even if tone is light — "I'm desperate for love!" is OPEN not masked
  - If someone NAMES their pain directly without deflecting, that's open even if delivery is dramatic/funny
  - Key test: does the humor AVOID the pain, or does it EXPOSE the pain? Avoid = masked, expose = open

### Shock/Dissociation vs Defensive
- When someone reports SEVERE trauma in a flat/detached tone (e.g. "he just shot himself"), this is likely SHOCK or dissociation, NOT defensive deflection.
- Shock = open vulnerability (the person IS in extreme pain, just can't process it yet)
- Rate: high distress AND high vulnerability_display for shock — the flat delivery does not mean low vulnerability, it means OVERWHELMING vulnerability.
- Only rate high deflection_strength when the speaker ACTIVELY changes subject or minimizes. Flat affect alone ≠ deflection.

### Positive Farewell / Gratitude ≠ Vulnerability
- "Nice chatting with you", "you have given me hope", "thank you for listening" = positive wrap-up, NOT vulnerability display.
- Do NOT infer previous hopelessness from a grateful goodbye.
- Rate vulnerability_display = 0.0 for simple gratitude/farewell messages.

### Neutral / No Vulnerability
- If the conversation is casual/factual with NO emotional distress, rate ALL signals low (< 0.2)
- Do NOT force a pattern onto neutral text — low scores across all signals is a valid output
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

        # Extract behavioral features from speaker turns
        speaker_texts = [t["text"] for t in window if t.get("role") == "speaker"]
        combined_text = " ".join(speaker_texts) if speaker_texts else window[-1]["text"]
        features = extract_features(combined_text)

        # LLM detection
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
        behavioral_scores = classify_from_features(features)

        # R7: Dynamic weighted merge based on behavioral signal strength
        # When behavioral has strong signal → trust it more (it's language-independent)
        # When behavioral has no signal → trust LLM fully
        beh_signal = features.get("total_signal", 0)
        beh_is_uniform = all(abs(v - 0.25) < 0.02 for v in behavioral_scores.values())

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

        system_prompt = f"""You are Dr. Sofia Chen, a clinical psychologist specializing in vulnerability research and emotional disclosure patterns. You are analyzing how people express or hide their vulnerability in conversation.

## Fragility Patterns
{anchors_text}

{SIGNAL_DEFINITIONS}

{DISAMBIGUATION_RULES}

## Calibration Rules
- People experiencing distress often MASK it. Look underneath surface-level communication.
- "I'm fine" is often NOT fine. "haha" after pain is often a mask, not genuine amusement.
- Short dismissive responses ("ok", "whatever") when discussing emotional topics = defensive or denial.
- High self-reference + hedging + vulnerability words = likely Open pattern.
- Humor markers (haha, lol, emoji) combined with painful content = likely Masked pattern.
- "I don't need anyone" / "feelings are weakness" = likely Denial pattern.
- Subject changes, "anyway", minimizing = likely Defensive pattern.
- MIXED patterns are common. Rate all signals honestly, the classification will handle it.
- If conversation has NO emotional content, rate all signals near 0.0. Do NOT inflate scores.

## Two-Step Analysis (CRITICAL)
Analyze CONTENT and DELIVERY separately:
1. **CONTENT**: What emotional topic is being discussed? (loss, fear, rejection, etc.) — This tells you IF vulnerability is present.
2. **DELIVERY**: HOW is the person saying it? This determines the PATTERN:
   - Direct, raw, unguarded → vulnerability_display HIGH (open)
   - Detached, philosophical, cold statement of facts → vulnerability_display LOW + denial or deflection HIGH
   - Wrapped in jokes, sarcasm, self-deprecation → humor_as_shield HIGH (masked)
   - Subject change, minimizing, hedging → deflection_strength HIGH (defensive)

CRITICAL: A person can discuss VERY vulnerable content (identity crisis, divorce, abuse) while DELIVERING it in a completely detached, cynical, or humorous way. The DELIVERY determines the pattern, NOT the content.
- "I found out it was easier to be him than to start over" = vulnerable CONTENT, but cold/detached DELIVERY → denial, NOT open
- "I'm desperate for love!" shouted dramatically = vulnerable CONTENT, but theatrical/performative DELIVERY → could be masked, NOT open
- "Feelings are weakness" = vulnerable CONTEXT implied, but dismissive DELIVERY → denial
- "I will be calm. I will be mistress of myself" = ACKNOWLEDGES emotion exists but CONTROLS it → defensive (not denial — she's managing feelings, not rejecting them)

Defensive vs Denial delivery distinction:
- Defensive ACKNOWLEDGES feelings exist but avoids/suppresses/controls them: "I need to stay calm", "Let's not go there", "It doesn't matter right now"
- Denial REJECTS that feelings exist at all: "I don't feel anything", "That doesn't affect me", cynical dismissal of emotion as concept

## Output Format
Brief two-step reasoning, then JSON:

REASONING:
- Content: [what emotional topic]
- Delivery: [how they express it — raw/detached/humorous/deflecting]

JSON:
{{"distress": 0.0, "vulnerability_display": 0.0, "humor_as_shield": 0.0, "denial_strength": 0.0, "deflection_strength": 0.0, "evidence": {{"most_revealing_quote": "...", "pattern_indicator": "..."}}}}"""

        for attempt in range(3):
            response = retry_api_call(
                lambda: self._client.messages.create(
                    model="anthropic/claude-sonnet-4",
                    max_tokens=800,
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
