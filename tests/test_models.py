"""Tests for fragility detection models."""

import pytest
from fragility_detector.models import (
    FragilityPattern,
    FragilitySignals,
    FragilitySnapshot,
    StarMapOutput,
    STAR_LABELS,
)


class TestFragilityPattern:
    def test_enum_values(self):
        assert FragilityPattern.OPEN.value == "open"
        assert FragilityPattern.DEFENSIVE.value == "defensive"
        assert FragilityPattern.MASKED.value == "masked"
        assert FragilityPattern.DENIAL.value == "denial"

    def test_enum_from_string(self):
        assert FragilityPattern("open") == FragilityPattern.OPEN
        assert FragilityPattern("denial") == FragilityPattern.DENIAL

    def test_all_patterns_have_star_labels(self):
        for pattern in FragilityPattern:
            assert pattern in STAR_LABELS
            label = STAR_LABELS[pattern]
            assert "star_label" in label
            assert "star_sublabel" in label
            assert "star_color" in label


class TestFragilitySignals:
    def test_default_values(self):
        signals = FragilitySignals()
        assert signals.distress == 0.0
        assert signals.self_ref_ratio == 0.0

    def test_clamping(self):
        signals = FragilitySignals(distress=1.5, hedging_ratio=-0.3)
        assert signals.distress == 1.0
        assert signals.hedging_ratio == 0.0

    def test_valid_values(self):
        signals = FragilitySignals(
            distress=0.7,
            self_ref_ratio=0.15,
            hedging_ratio=0.08,
            vulnerability_display=0.6,
        )
        assert signals.distress == 0.7
        assert signals.vulnerability_display == 0.6


class TestFragilitySnapshot:
    def test_basic_creation(self):
        snap = FragilitySnapshot(
            turn=0,
            pattern=FragilityPattern.OPEN,
            pattern_scores={"open": 0.6, "defensive": 0.2, "masked": 0.1, "denial": 0.1},
            signals=FragilitySignals(distress=0.7),
            confidence=0.6,
        )
        assert snap.pattern == FragilityPattern.OPEN
        assert snap.confidence == 0.6

    def test_confidence_clamped(self):
        snap = FragilitySnapshot(
            turn=0,
            pattern=FragilityPattern.DENIAL,
            pattern_scores={"open": 0.1, "defensive": 0.1, "masked": 0.1, "denial": 0.7},
            signals=FragilitySignals(),
            confidence=1.5,
        )
        assert snap.confidence == 1.0


class TestStarMapOutput:
    def test_creation(self):
        output = StarMapOutput(
            dimension="fragility",
            type="open",
            confidence=0.82,
            star_label="敢于面对真实",
            star_sublabel="脆弱性模式",
            star_brightness="bright",
            star_color="rose",
        )
        assert output.dimension == "fragility"
        assert output.type == "open"

    def test_all_star_labels_have_required_fields(self):
        required = {"star_label", "star_sublabel", "star_color", "description_zh", "description_en"}
        for pattern, label in STAR_LABELS.items():
            for field in required:
                assert field in label, f"Missing {field} in {pattern.value}"
