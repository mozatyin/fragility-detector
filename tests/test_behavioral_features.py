"""Tests for behavioral feature extraction."""

import pytest
from fragility_detector.behavioral_features import extract_features, classify_from_features


class TestExtractFeatures:
    def test_empty_text(self):
        features = extract_features("")
        assert features["self_ref_ratio"] == 0.0

    def test_self_reference(self):
        features = extract_features("I think I am feeling bad about myself")
        assert features["self_ref_ratio"] > 0.3  # "I", "I", "myself" out of 8 words

    def test_hedging(self):
        features = extract_features("I think maybe I should probably go, I guess")
        assert features["hedging_ratio"] > 0.2  # think, maybe, probably, guess

    def test_humor_markers(self):
        features = extract_features("haha yeah that's so funny lol I'm dying hahaha")
        assert features["humor_markers"] > 0.2

    def test_denial_phrases(self):
        features = extract_features("I'm fine. It's nothing. No big deal. Whatever.")
        assert features["negation_ratio"] > 0.5  # multiple denial phrases

    def test_deflection(self):
        features = extract_features("Anyway, let's talk about something else. Never mind. It is what it is.")
        assert features["deflection_ratio"] > 0.5

    def test_vulnerability_words(self):
        features = extract_features("I feel broken and helpless, I've been crying all day")
        assert features["vulnerability_ratio"] > 0.1

    def test_strength_words(self):
        features = extract_features("I'm strong and independent, I can handle anything")
        assert features["strength_ratio"] > 0.1

    def test_open_pattern_text(self):
        text = "I feel like I'm falling apart. Everything hurts and I'm scared. I don't know how much longer I can take this."
        features = extract_features(text)
        assert features["vulnerability_ratio"] > 0
        assert features["self_ref_ratio"] > 0.1

    def test_defensive_pattern_text(self):
        text = "I'm fine. It's not a big deal. Anyway, did you see that movie? Let's talk about something else."
        features = extract_features(text)
        assert features["negation_ratio"] > 0
        assert features["deflection_ratio"] > 0

    def test_masked_pattern_text(self):
        text = "haha yeah I got dumped again lol. My love life is a comedy show. At least I'm consistent 😂"
        features = extract_features(text)
        assert features["humor_markers"] > 0

    def test_denial_pattern_text(self):
        text = "I don't need anyone. I'm strong and I can handle everything on my own. Feelings are just weakness."
        features = extract_features(text)
        assert features["strength_ratio"] > 0
        assert features["negation_ratio"] > 0


class TestClassifyFromFeatures:
    def test_returns_all_patterns(self):
        features = extract_features("hello world")
        scores = classify_from_features(features)
        assert set(scores.keys()) == {"open", "defensive", "masked", "denial"}

    def test_scores_sum_to_one(self):
        features = extract_features("I feel broken and scared")
        scores = classify_from_features(features)
        assert abs(sum(scores.values()) - 1.0) < 0.001

    def test_open_pattern_scores_highest(self):
        text = "I feel broken and helpless. I've been crying all night. I'm scared and I don't know what to do."
        features = extract_features(text)
        scores = classify_from_features(features)
        assert scores["open"] == max(scores.values())

    def test_masked_pattern_scores_highest(self):
        text = "haha yeah my life is a joke lol. Got dumped again hahaha. At least I'm consistent lmao"
        features = extract_features(text)
        scores = classify_from_features(features)
        assert scores["masked"] == max(scores.values())

    def test_denial_pattern_scores_highest(self):
        text = "I'm fine. I'm strong. I can handle this. I don't need anyone's help. Whatever."
        features = extract_features(text)
        scores = classify_from_features(features)
        # Denial should score high due to strength + negation words
        assert scores["denial"] > scores["open"]

    def test_all_scores_positive(self):
        features = extract_features("some random text about nothing in particular")
        scores = classify_from_features(features)
        for v in scores.values():
            assert v > 0
