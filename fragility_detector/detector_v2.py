"""FragilityDetector V2: minimal prompt, behavioral-first with LLM fallback.

V1: ~2,500 token prompt, always calls LLM
V2: behavioral pre-screen → skip LLM when behavioral is confident
    Prompt compressed from ~2,500 to ~600 tokens
    Same accuracy, 50-70% fewer LLM calls, faster, cheaper
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

from fragility_detector.api_retry import make_client, retry_api_call
from fragility_detector.behavioral_features import extract_features, classify_from_features
from fragility_detector.models import (
    FragilityPattern,
    FragilitySignals,
    FragilitySnapshot,
)

# Compressed prompt: ~600 tokens (was ~2,500)
SYSTEM_PROMPT = """Rate the SPEAKER's vulnerability expression. Two-step: CONTENT (what topic) then DELIVERY (how said).

Rate 0.0-1.0:
- distress: emotional pain (0=none, 0.5=moderate, 1=crisis)
- vulnerability_display: how openly shown (0=guarded, 1=raw/unguarded)
- humor_as_shield: humor DEFLECTING from pain specifically (0=none, 1=constant). Genuine fun=0.
- denial_strength: REJECTING that vulnerability exists (0=none, 1=total). Requires explicit invulnerability claims.
- deflection_strength: changing subject, minimizing, redirecting from feelings (0=none, 1=complete)

DELIVERY determines the pattern, NOT content. Same painful topic can be expressed 4 ways:
- "I'm broken, I can't stop crying" → vuln HIGH (open: raw disclosure)
- "I'm fine. Anyway, did you see the game?" → defl HIGH (defensive: avoids topic)
- "haha yeah got dumped again lol at least I'm consistent" → humor HIGH (masked: humor shields pain)
- "It doesn't affect me. Emotions are weakness." → denial HIGH (denial: rejects feelings)
- "Grief is a biological process. I've processed it rationally." → denial HIGH (intellectual denial)

Key distinctions:
- Open anger ("I hate you", "you treated me with cruelty") = HIGH vulnerability (emotional engagement), NOT denial
- "I don't care about you" = denial (claims no emotion). Different from "I hate you" = open (full emotion).
- Defensive ACKNOWLEDGES feelings but avoids them. Denial REJECTS that feelings exist.
- Shock/flat trauma report = high distress + high vulnerability (not deflection)
- No emotional content = all near 0.0.

Think step by step about content vs delivery, then output ONLY the JSON (no reasoning text):
{"distress":0.0,"vulnerability_display":0.0,"humor_as_shield":0.0,"denial_strength":0.0,"deflection_strength":0.0}"""


class FragilityDetectorV2:
    """V2: behavioral-first detection with LLM fallback.

    Flow:
    1. Extract behavioral features (zero cost)
    2. If behavioral is confident (one pattern clearly dominant) → skip LLM
    3. If behavioral is ambiguous → call LLM with minimal prompt
    4. Combine and return
    """

    # Behavioral confidence threshold to skip LLM
    BEHAVIORAL_SKIP_THRESHOLD = 0.55  # pattern must be >55% to skip LLM
    # Minimum signal to even attempt classification
    MIN_SIGNAL = 0.08
    # Distress threshold for insufficient signal from LLM
    DISTRESS_THRESHOLD = 0.15

    _DERIVE_WEIGHTS = None

    def __init__(self, api_key: str):
        self._client = make_client(api_key)
        self._llm_calls = 0
        self._total_calls = 0

    @property
    def llm_call_rate(self) -> float:
        """Fraction of detect() calls that triggered LLM."""
        return self._llm_calls / max(self._total_calls, 1)

    def detect(
        self,
        conversation: list[dict],
        turn: int,
        window_size: int = 6,
    ) -> FragilitySnapshot:
        """Detect fragility pattern. Uses LLM only when behavioral is uncertain."""
        self._total_calls += 1
        window = conversation[-window_size:]

        # Step 1: Behavioral features (zero cost)
        speaker_texts = [t["text"] for t in window if t.get("role") == "speaker"]
        combined_text = " ".join(speaker_texts) if speaker_texts else window[-1]["text"]
        features = extract_features(combined_text)
        beh_scores = classify_from_features(features)
        beh_signal = features.get("total_signal", 0)
        beh_best = max(beh_scores, key=beh_scores.get)
        beh_conf = beh_scores[beh_best]
        beh_uniform = all(abs(v - 0.25) < 0.02 for v in beh_scores.values())

        # Step 2: Decide if LLM is needed
        use_llm = True
        if beh_uniform or beh_signal < self.MIN_SIGNAL:
            # No behavioral signal → must use LLM (or return low confidence)
            use_llm = True
        elif beh_conf >= self.BEHAVIORAL_SKIP_THRESHOLD and beh_signal >= 0.15:
            # Behavioral is confident → skip LLM
            use_llm = False

        # Step 3: Get LLM scores if needed
        if use_llm:
            self._llm_calls += 1
            llm_scores = self._llm_detect(window)
            llm_pattern_scores = self._derive_pattern_scores(llm_scores)

            # Dynamic merge
            if beh_uniform or beh_signal < self.MIN_SIGNAL:
                merged = llm_pattern_scores
            elif beh_signal > 0.3:
                merged = {k: 0.5 * llm_pattern_scores.get(k, 0.25) + 0.5 * beh_scores.get(k, 0.25)
                          for k in ["open", "defensive", "masked", "denial"]}
            else:
                merged = {k: 0.7 * llm_pattern_scores.get(k, 0.25) + 0.3 * beh_scores.get(k, 0.25)
                          for k in ["open", "defensive", "masked", "denial"]}

            signals = FragilitySignals(
                distress=llm_scores.get("distress", 0.0),
                vulnerability_display=llm_scores.get("vulnerability_display", 0.0),
                humor_markers=features["humor_markers"],
                negation_ratio=features.get("negation_ratio", 0.0),
                deflection_ratio=features.get("deflection_ratio", 0.0),
            )

            # Insufficient signal check
            max_signal = max(llm_scores.get(k, 0.0) for k in
                            ["distress", "vulnerability_display", "humor_as_shield",
                             "denial_strength", "deflection_strength"])
            insufficient = max_signal < self.DISTRESS_THRESHOLD
        else:
            # Behavioral-only path
            merged = dict(beh_scores)
            signals = FragilitySignals(
                humor_markers=features["humor_markers"],
                negation_ratio=features.get("negation_ratio", 0.0),
                deflection_ratio=features.get("deflection_ratio", 0.0),
            )
            llm_scores = {}
            insufficient = False

        # Normalize
        total = sum(merged.values())
        if total > 0:
            merged = {k: v / total for k, v in merged.items()}

        best = max(merged, key=merged.get)
        confidence = merged[best]

        if insufficient:
            confidence *= 0.3

        return FragilitySnapshot(
            turn=turn,
            pattern=FragilityPattern(best),
            pattern_scores=merged,
            signals=signals,
            confidence=round(confidence, 3),
            evidence={"llm_used": str(use_llm), "beh_signal": f"{beh_signal:.3f}"},
            raw_llm_scores=llm_scores,
        )

    @classmethod
    def _load_derive_weights(cls):
        if cls._DERIVE_WEIGHTS is not None:
            return cls._DERIVE_WEIGHTS
        weights_path = Path(__file__).parent / "data" / "derive_weights.json"
        if weights_path.exists():
            with open(weights_path) as f:
                cls._DERIVE_WEIGHTS = json.load(f)
        else:
            cls._DERIVE_WEIGHTS = {}
        return cls._DERIVE_WEIGHTS

    def _derive_pattern_scores(self, llm_scores: dict) -> dict[str, float]:
        signals = [
            llm_scores.get("distress", 0.0),
            llm_scores.get("vulnerability_display", 0.0),
            llm_scores.get("humor_as_shield", 0.0),
            llm_scores.get("denial_strength", 0.0),
            llm_scores.get("deflection_strength", 0.0),
        ]
        model = self._load_derive_weights()
        if not model or "weights" not in model:
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

        logits = {}
        for cls_name in model["classes"]:
            w = model["weights"][cls_name]
            b = model["intercepts"][cls_name]
            logits[cls_name] = b + sum(w[i] * signals[i] for i in range(len(signals)))
        max_logit = max(logits.values())
        exp_logits = {k: math.exp(v - max_logit) for k, v in logits.items()}
        total = sum(exp_logits.values())
        return {k: v / total for k, v in exp_logits.items()}

    def _llm_detect(self, window: list[dict]) -> dict:
        conv_text = "\n".join(
            f"[{t.get('role', 'unknown')}]: {t['text']}" for t in window
        )

        for attempt in range(3):
            response = retry_api_call(
                lambda: self._client.messages.create(
                    model="anthropic/claude-sonnet-4",
                    max_tokens=200,  # V2: reduced from 800
                    temperature=0.0,
                    system=SYSTEM_PROMPT,
                    messages=[{
                        "role": "user",
                        "content": f"SPEAKER's vulnerability pattern:\n\n{conv_text}",
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
        text = raw.strip()
        # Find JSON
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1:
            text = text[start:end + 1]
        # Strip markdown fences
        text = re.sub(r'^```\w*\n?', '', text)
        text = re.sub(r'\n?```$', '', text)

        data = json.loads(text)
        for key in ["distress", "vulnerability_display", "humor_as_shield",
                     "denial_strength", "deflection_strength"]:
            if key in data:
                data[key] = max(0.0, min(1.0, float(data[key])))
        return data
