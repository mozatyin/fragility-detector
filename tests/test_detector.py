"""Tests for FragilityDetector (unit tests, no LLM calls)."""

import pytest
from fragility_detector.detector import FragilityDetector
from fragility_detector.models import FragilityPattern


class TestParseResponse:
    def test_parse_basic_json(self):
        raw = """REASONING: The speaker shows clear distress.

JSON:
{"distress": 0.7, "vulnerability_display": 0.6, "humor_as_shield": 0.1, "denial_strength": 0.0, "deflection_strength": 0.1, "evidence": {"most_revealing_quote": "I feel broken"}}"""
        result = FragilityDetector._parse_response(raw)
        assert result["distress"] == 0.7
        assert result["vulnerability_display"] == 0.6

    def test_parse_with_markdown_fences(self):
        raw = """REASONING: Analysis here.

JSON:
```json
{"distress": 0.5, "vulnerability_display": 0.3, "humor_as_shield": 0.0, "denial_strength": 0.4, "deflection_strength": 0.2}
```"""
        result = FragilityDetector._parse_response(raw)
        assert result["distress"] == 0.5
        assert result["denial_strength"] == 0.4

    def test_clamps_values(self):
        raw = '{"distress": 1.5, "vulnerability_display": -0.3, "humor_as_shield": 0.5, "denial_strength": 0.0, "deflection_strength": 0.0}'
        result = FragilityDetector._parse_response(raw)
        assert result["distress"] == 1.0
        assert result["vulnerability_display"] == 0.0

    def test_parse_no_json_marker(self):
        raw = '{"distress": 0.4, "vulnerability_display": 0.2, "humor_as_shield": 0.0, "denial_strength": 0.0, "deflection_strength": 0.0}'
        result = FragilityDetector._parse_response(raw)
        assert result["distress"] == 0.4


class TestDerivePatternScores:
    def setup_method(self):
        # Create detector without client (won't make API calls)
        self.detector = FragilityDetector.__new__(FragilityDetector)

    def test_high_vulnerability_gives_open(self):
        scores = self.detector._derive_pattern_scores({
            "distress": 0.8,
            "vulnerability_display": 0.9,
            "humor_as_shield": 0.0,
            "denial_strength": 0.0,
            "deflection_strength": 0.0,
        })
        assert scores["open"] == max(scores.values())

    def test_high_humor_gives_masked(self):
        scores = self.detector._derive_pattern_scores({
            "distress": 0.5,
            "vulnerability_display": 0.1,
            "humor_as_shield": 0.8,
            "denial_strength": 0.0,
            "deflection_strength": 0.1,
        })
        assert scores["masked"] == max(scores.values())

    def test_high_denial_gives_denial(self):
        scores = self.detector._derive_pattern_scores({
            "distress": 0.2,
            "vulnerability_display": 0.0,
            "humor_as_shield": 0.0,
            "denial_strength": 0.9,
            "deflection_strength": 0.1,
        })
        assert scores["denial"] == max(scores.values())

    def test_high_deflection_gives_defensive(self):
        scores = self.detector._derive_pattern_scores({
            "distress": 0.5,
            "vulnerability_display": 0.1,
            "humor_as_shield": 0.0,
            "denial_strength": 0.0,
            "deflection_strength": 0.9,
        })
        assert scores["defensive"] == max(scores.values())

    def test_scores_sum_to_one(self):
        scores = self.detector._derive_pattern_scores({
            "distress": 0.5,
            "vulnerability_display": 0.3,
            "humor_as_shield": 0.2,
            "denial_strength": 0.1,
            "deflection_strength": 0.4,
        })
        assert abs(sum(scores.values()) - 1.0) < 0.001

    def test_all_zeros_gives_valid_scores(self):
        scores = self.detector._derive_pattern_scores({
            "distress": 0.0,
            "vulnerability_display": 0.0,
            "humor_as_shield": 0.0,
            "denial_strength": 0.0,
            "deflection_strength": 0.0,
        })
        # R4: with learned weights, all-zeros gives defensive (highest bias)
        # but all scores should be positive and sum to 1
        assert abs(sum(scores.values()) - 1.0) < 0.001
        for v in scores.values():
            assert v >= 0


class TestFastPaths:
    """Test LLM skip fast paths — no API calls needed."""

    def setup_method(self):
        self.detector = FragilityDetector.__new__(FragilityDetector)

    def test_no_signal_low_distress_skips_llm(self):
        """Neutral text + no emotion distress → skip LLM, return uniform."""
        conv = [{"role": "speaker", "text": "okay sounds good"}]
        result = self.detector.detect(conv, turn=0, emotion_distress=0.0)
        assert result.confidence == 0.0
        assert result.evidence.get("llm_skipped") == "true"
        assert result.evidence.get("reason") == "no_signal"

    def test_strong_behavioral_skips_llm(self):
        """Strong vulnerability words → behavioral skip, no LLM."""
        conv = [{"role": "speaker", "text": "I'm so hurt and broken, I've been crying all day, I feel lost and helpless and scared"}]
        result = self.detector.detect(conv, turn=0)
        assert result.evidence.get("llm_skipped") == "true"
        assert result.pattern == FragilityPattern.OPEN

    def test_strong_denial_behavioral_skip(self):
        """Strong denial signals → behavioral skip."""
        conv = [{"role": "speaker", "text": "I'm completely fine. I don't need anyone. I don't care. I'm strong. Feelings are weakness. Whatever."}]
        result = self.detector.detect(conv, turn=0)
        assert result.evidence.get("llm_skipped") == "true"
        assert result.pattern == FragilityPattern.DENIAL

    def test_high_distress_bypasses_no_signal_skip(self):
        """Even with neutral text, high emotion distress prevents no-signal skip."""
        conv = [{"role": "speaker", "text": "ok"}]
        # With high emotion_distress, should NOT hit no_signal fast path
        # It will try LLM, so we verify it raises (no client) rather than returning no_signal
        try:
            result = self.detector.detect(conv, turn=0, emotion_distress=0.5)
            # If it returns, it must not be the no_signal path
            assert result.evidence.get("reason") != "no_signal"
        except AttributeError:
            # Expected: tries LLM but no client → proves it bypassed no_signal skip
            pass

    def test_humor_behavioral_skip(self):
        """Humor markers → masked pattern via behavioral skip."""
        conv = [{"role": "speaker", "text": "haha yeah that really hurt me lol 😂 at least I'm consistent at failing hahahaha"}]
        result = self.detector.detect(conv, turn=0)
        assert result.evidence.get("llm_skipped") == "true"
        assert result.pattern == FragilityPattern.MASKED
