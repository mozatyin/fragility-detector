"""Tests for V8x dual-trigger adapter, CrisisFeedPayload, and V8.x interfaces."""

from fragility_detector.models import (
    CrisisFeedPayload,
    FragilityDetectorInput,
    FragilityDetectorOutput,
)
from fragility_detector.v8x_adapter import V8xFragilityAdapter


class TestShouldActivate:
    def test_should_activate_on_vulnerability(self):
        activated, trigger = V8xFragilityAdapter.should_activate(
            vulnerability_signals=True, emotion_distress=0.1
        )
        assert activated is True
        assert trigger == "vulnerability_signals"

    def test_should_activate_on_high_distress(self):
        activated, trigger = V8xFragilityAdapter.should_activate(
            vulnerability_signals=False, emotion_distress=0.5
        )
        assert activated is True
        assert trigger == "distress_threshold"

    def test_should_activate_both_triggers(self):
        activated, trigger = V8xFragilityAdapter.should_activate(
            vulnerability_signals=True, emotion_distress=0.5
        )
        assert activated is True
        assert trigger == "both"

    def test_should_not_activate_when_calm(self):
        activated, trigger = V8xFragilityAdapter.should_activate(
            vulnerability_signals=False, emotion_distress=0.1
        )
        assert activated is False
        assert trigger == "none"


class TestCrisisFeed:
    def test_crisis_feed_corroboration(self):
        payload = V8xFragilityAdapter._build_crisis_feed(
            fragility_score=0.6, fragility_type="open",
            emotion_distress=0.5, session_history=None,
        )
        assert isinstance(payload, CrisisFeedPayload)
        assert payload.distress_corroboration is True
        assert payload.fragility_score == 0.6
        assert payload.fragility_type == "open"
        assert payload.recommended_alert_level == "warn"

    def test_crisis_feed_escalation_detected(self):
        history = [
            {"fragility_score": 0.3},
            {"fragility_score": 0.5},
        ]
        payload = V8xFragilityAdapter._build_crisis_feed(
            fragility_score=0.8, fragility_type="open",
            emotion_distress=0.8, session_history=history,
        )
        assert payload.escalation_detected is True
        assert payload.recommended_alert_level == "critical"

    def test_crisis_feed_no_escalation(self):
        history = [{"fragility_score": 0.5}]
        payload = V8xFragilityAdapter._build_crisis_feed(
            fragility_score=0.4, fragility_type="open",
            emotion_distress=0.2, session_history=history,
        )
        assert payload.escalation_detected is False

    def test_crisis_feed_alert_levels(self):
        # none: low score
        p = V8xFragilityAdapter._build_crisis_feed(0.1, "none", 0.1, None)
        assert p.recommended_alert_level == "none"
        # monitor: moderate score
        p = V8xFragilityAdapter._build_crisis_feed(0.4, "open", 0.1, None)
        assert p.recommended_alert_level == "monitor"
        # warn: high score without corroboration
        p = V8xFragilityAdapter._build_crisis_feed(0.8, "open", 0.1, None)
        assert p.recommended_alert_level == "warn"
        # critical: high score + corroboration
        p = V8xFragilityAdapter._build_crisis_feed(0.8, "open", 0.8, None)
        assert p.recommended_alert_level == "critical"

    def test_escalation_overrides_alert(self):
        history = [{"fragility_score": 0.1}]
        p = V8xFragilityAdapter._build_crisis_feed(0.35, "open", 0.5, history)
        assert p.escalation_detected is True
        assert p.distress_corroboration is True
        assert p.recommended_alert_level == "warn"  # escalation + corroboration → warn


class TestEnhanceWithContext:
    def test_emotion_discrepancy_detects_masked(self):
        """High emotion distress + low fragility → masked/defensive."""
        pattern, score, conf = V8xFragilityAdapter._enhance_with_context(
            pattern="open", score=0.2, confidence=0.3,
            pattern_scores={"open": 0.2, "masked": 0.15, "defensive": 0.1, "denial": 0.05},
            emotion_distress=0.7,
            session_history=None,
        )
        assert pattern == "masked"  # masked > defensive in pattern_scores
        assert score >= 0.3
        assert conf >= 0.5

    def test_session_normalization_detects_defensive(self):
        """Was open, now suddenly calm → defensive."""
        history = [
            {"fragility_type": "open", "fragility_score": 0.7},
        ]
        pattern, score, conf = V8xFragilityAdapter._enhance_with_context(
            pattern="denial", score=0.15, confidence=0.2,
            pattern_scores={"open": 0.1, "masked": 0.1, "defensive": 0.15, "denial": 0.15},
            emotion_distress=0.1,
            session_history=history,
        )
        assert pattern == "defensive"
        assert score >= 0.35

    def test_no_enhancement_when_no_discrepancy(self):
        """No discrepancy → no change."""
        pattern, score, conf = V8xFragilityAdapter._enhance_with_context(
            pattern="open", score=0.6, confidence=0.7,
            pattern_scores={"open": 0.6, "masked": 0.1, "defensive": 0.1, "denial": 0.1},
            emotion_distress=0.1,
            session_history=None,
        )
        assert pattern == "open"
        assert score == 0.6


class TestFragilityDetectorOutput:
    def test_not_activated(self):
        out = FragilityDetectorOutput.not_activated("sess1", 3)
        assert out.activated is False
        assert out.fragility_type == "none"
        assert out.fragility_score == 0.0

    def test_timeout_unknown(self):
        out = FragilityDetectorOutput.timeout_unknown("sess1", 3)
        assert out.activated is True
        assert out.activation_trigger == "timeout"
        assert out.crisis_feed is not None
        assert out.crisis_feed.recommended_alert_level == "monitor"


class TestFragilityDetectorInput:
    def test_input_creation(self):
        inp = FragilityDetectorInput(
            session_id="s1", turn_id=1, raw_text="I'm fine",
            conversation=[{"role": "speaker", "text": "I'm fine"}],
            vulnerability_signals=True, emotion_distress=0.5,
        )
        assert inp.activation_trigger == "none"
        assert inp.session_fragility_history is None


class TestEstimateFromSignals:
    def test_estimate_defensive_pattern(self):
        result = V8xFragilityAdapter.estimate_from_signals(
            vulnerability_signals=True,
            emotion_keywords=["fine", "whatever"],
            behavioral={"hedging_ratio": 0.2, "neg_emotion_ratio": 0.0, "pos_emotion_ratio": 0.0},
        )
        assert result["derived"]["dominant_type"] == "defensive"

    def test_estimate_open_pattern(self):
        result = V8xFragilityAdapter.estimate_from_signals(
            vulnerability_signals=True,
            emotion_keywords=["sad", "hurt", "crying"],
            behavioral={"hedging_ratio": 0.05, "neg_emotion_ratio": 0.3, "pos_emotion_ratio": 0.0},
        )
        assert result["derived"]["dominant_type"] == "open"


class TestFromSharedUnderstanding:
    def test_from_shared_understanding_basic(self):
        shared = {
            "vulnerability_signals": True,
            "emotion_distress": 0.5,
            "text_features": {
                "hedging_ratio": 0.05,
                "neg_emotion_ratio": 0.2,
                "pos_emotion_ratio": 0.0,
            },
        }
        behavioral = {"hedging_ratio": 0.05, "neg_emotion_ratio": 0.2, "pos_emotion_ratio": 0.0}
        result = V8xFragilityAdapter.from_shared_understanding(
            shared_understanding=shared,
            behavioral_features=behavioral,
            emotion_distress=0.5,
        )
        assert result["detector"] == "fragility"
        assert result["activated"] is True
        assert "results" in result
        assert "derived" in result
        assert "fragility_score" in result["derived"]
        assert "dominant_type" in result["derived"]

    def test_not_activated_returns_zeros(self):
        shared = {"vulnerability_signals": False}
        result = V8xFragilityAdapter.from_shared_understanding(
            shared_understanding=shared,
            behavioral_features={},
            emotion_distress=0.1,
        )
        assert result["activated"] is False
        assert result["derived"]["fragility_score"] == 0.0


class TestLegacyCrisisFeed:
    def test_build_crisis_feed_legacy(self):
        result = {
            "results": {"open": 0.6},
            "derived": {"fragility_score": 0.6, "dominant_type": "open"},
        }
        payload = V8xFragilityAdapter.build_crisis_feed_legacy(
            fragility_result=result, emotion_distress=0.5,
        )
        assert isinstance(payload, CrisisFeedPayload)
        assert payload.fragility_score == 0.6
