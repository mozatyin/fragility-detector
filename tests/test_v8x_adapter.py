"""Tests for V8x dual-trigger adapter and CrisisFeedPayload."""

from fragility_detector.v8x_adapter import CrisisFeedPayload, V8xFragilityAdapter


class TestShouldActivate:
    def test_should_activate_on_vulnerability(self):
        activated, reason = V8xFragilityAdapter.should_activate(
            vulnerability_signals=True, emotion_distress=0.1
        )
        assert activated is True
        assert "vulnerability" in reason.lower()

    def test_should_activate_on_high_distress(self):
        activated, reason = V8xFragilityAdapter.should_activate(
            vulnerability_signals=False, emotion_distress=0.5
        )
        assert activated is True
        assert "distress" in reason.lower()

    def test_should_not_activate_when_calm(self):
        activated, reason = V8xFragilityAdapter.should_activate(
            vulnerability_signals=False, emotion_distress=0.1
        )
        assert activated is False
        assert reason  # should still have a reason


class TestCrisisFeed:
    def test_crisis_feed_corroboration(self):
        result = {
            "results": {"open": 0.6, "defensive": 0.1, "masked": 0.05, "denial": 0.05},
            "derived": {"fragility_score": 0.6, "dominant_type": "open"},
        }
        payload = V8xFragilityAdapter.build_crisis_feed(
            fragility_result=result, emotion_distress=0.5
        )
        assert isinstance(payload, CrisisFeedPayload)
        assert payload.distress_corroboration is True
        assert payload.fragility_score == 0.6
        assert payload.fragility_type == "open"
        assert payload.recommended_alert_level == "medium"

    def test_crisis_feed_escalation_detected(self):
        result = {
            "results": {"open": 0.8, "defensive": 0.1, "masked": 0.0, "denial": 0.0},
            "derived": {"fragility_score": 0.8, "dominant_type": "open"},
        }
        history = [
            {"derived": {"fragility_score": 0.3}},
            {"derived": {"fragility_score": 0.5}},
        ]
        payload = V8xFragilityAdapter.build_crisis_feed(
            fragility_result=result, emotion_distress=0.8, session_history=history
        )
        assert payload.escalation_detected is True
        assert payload.recommended_alert_level == "high"

    def test_crisis_feed_no_escalation(self):
        result = {
            "results": {"open": 0.4, "defensive": 0.1, "masked": 0.0, "denial": 0.0},
            "derived": {"fragility_score": 0.4, "dominant_type": "open"},
        }
        history = [
            {"derived": {"fragility_score": 0.5}},
        ]
        payload = V8xFragilityAdapter.build_crisis_feed(
            fragility_result=result, emotion_distress=0.2, session_history=history
        )
        assert payload.escalation_detected is False


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
