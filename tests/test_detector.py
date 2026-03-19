"""Tests for FragilityDetector (unit tests, no LLM calls)."""

import pytest
from fragility_detector.detector import FragilityDetector


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
