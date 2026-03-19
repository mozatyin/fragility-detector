"""Tests for star map generation."""

import pytest
from fragility_detector.models import (
    FragilityPattern,
    FragilitySignals,
    FragilitySnapshot,
)
from fragility_detector.star_map import generate_star_map


class TestGenerateStarMap:
    def _make_snapshot(self, pattern: FragilityPattern, confidence: float) -> FragilitySnapshot:
        return FragilitySnapshot(
            turn=0,
            pattern=pattern,
            pattern_scores={p.value: 0.25 for p in FragilityPattern},
            signals=FragilitySignals(),
            confidence=confidence,
        )

    def test_open_pattern(self):
        star = generate_star_map(self._make_snapshot(FragilityPattern.OPEN, 0.8))
        assert star.dimension == "fragility"
        assert star.type == "open"
        assert star.star_label == "敢于面对真实"
        assert star.star_color == "rose"

    def test_defensive_pattern(self):
        star = generate_star_map(self._make_snapshot(FragilityPattern.DEFENSIVE, 0.5))
        assert star.star_label == "用坚强保护柔软"
        assert star.star_color == "amber"

    def test_masked_pattern(self):
        star = generate_star_map(self._make_snapshot(FragilityPattern.MASKED, 0.7))
        assert star.star_label == "选择性敞开"
        assert star.star_color == "teal"

    def test_denial_pattern(self):
        star = generate_star_map(self._make_snapshot(FragilityPattern.DENIAL, 0.6))
        assert star.star_label == "隐藏的温柔"
        assert star.star_color == "indigo"

    def test_brightness_bright(self):
        star = generate_star_map(self._make_snapshot(FragilityPattern.OPEN, 0.8))
        assert star.star_brightness == "bright"

    def test_brightness_medium(self):
        star = generate_star_map(self._make_snapshot(FragilityPattern.OPEN, 0.5))
        assert star.star_brightness == "medium"

    def test_brightness_dim(self):
        star = generate_star_map(self._make_snapshot(FragilityPattern.OPEN, 0.3))
        assert star.star_brightness == "dim"
