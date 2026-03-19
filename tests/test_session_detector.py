"""Tests for session-level fragility detection (unit tests, no LLM)."""

import pytest
from fragility_detector.session_detector import SessionState, TurnSignal, SessionFragilityDetector


class TestSessionState:
    def test_empty(self):
        state = SessionState()
        assert state.n_turns == 0
        assert state.peak_distress == 0.0

    def test_add_turn_updates_peaks(self):
        state = SessionState()
        state.add_turn(TurnSignal(turn=0, distress=0.3, vulnerability_display=0.5))
        assert state.peak_distress == 0.3
        assert state.peak_vulnerability == 0.5

        state.add_turn(TurnSignal(turn=1, distress=0.8, vulnerability_display=0.2))
        assert state.peak_distress == 0.8
        assert state.peak_vulnerability == 0.5  # didn't increase

    def test_distressing_topic_flag(self):
        state = SessionState()
        state.add_turn(TurnSignal(turn=0, distress=0.1))
        assert not state.topic_is_distressing

        state.add_turn(TurnSignal(turn=1, distress=0.5))
        assert state.topic_is_distressing


class TestDeriveSessionPattern:
    def setup_method(self):
        self.detector = SessionFragilityDetector.__new__(SessionFragilityDetector)

    def test_open_pattern_high_vulnerability(self):
        state = SessionState()
        for i in range(4):
            state.add_turn(TurnSignal(
                turn=i,
                distress=0.7,
                vulnerability_display=0.6 + i * 0.05,
            ))
        snap = self.detector._derive_session_pattern(state)
        assert snap.pattern.value == "open"
        assert snap.confidence > 0.3

    def test_defensive_pattern_deflection_over_time(self):
        state = SessionState()
        # First turn: some distress, some vulnerability
        state.add_turn(TurnSignal(turn=0, distress=0.5, vulnerability_display=0.3))
        # Later turns: deflection increases, vulnerability drops
        state.add_turn(TurnSignal(turn=1, distress=0.4, vulnerability_display=0.1, deflection_strength=0.5))
        state.add_turn(TurnSignal(turn=2, distress=0.3, vulnerability_display=0.0, deflection_strength=0.7))
        state.add_turn(TurnSignal(turn=3, distress=0.2, vulnerability_display=0.0, deflection_strength=0.8))
        snap = self.detector._derive_session_pattern(state)
        assert snap.pattern.value == "defensive"

    def test_masked_pattern_humor_with_distress(self):
        state = SessionState()
        for i in range(4):
            state.add_turn(TurnSignal(
                turn=i,
                distress=0.5,
                humor_as_shield=0.6,
                vulnerability_display=0.1,
            ))
        snap = self.detector._derive_session_pattern(state)
        assert snap.pattern.value == "masked"

    def test_denial_pattern_persistent_denial(self):
        state = SessionState()
        for i in range(4):
            state.add_turn(TurnSignal(
                turn=i,
                distress=0.3,
                denial_strength=0.7,
                vulnerability_display=0.0,
            ))
        snap = self.detector._derive_session_pattern(state)
        assert snap.pattern.value == "denial"

    def test_empty_state_returns_snapshot(self):
        snap = self.detector._derive_session_pattern(SessionState())
        assert snap.confidence == 0.0

    def test_low_distress_lowers_confidence(self):
        state = SessionState()
        # All turns are low-distress (no distressing topic)
        for i in range(3):
            state.add_turn(TurnSignal(turn=i, distress=0.1, vulnerability_display=0.1))
        snap = self.detector._derive_session_pattern(state)
        assert snap.confidence < 0.3  # low because topic not distressing

    def test_more_turns_increases_confidence(self):
        state_short = SessionState()
        state_long = SessionState()
        for i in range(2):
            state_short.add_turn(TurnSignal(turn=i, distress=0.7, vulnerability_display=0.6))
        for i in range(6):
            state_long.add_turn(TurnSignal(turn=i, distress=0.7, vulnerability_display=0.6))

        snap_short = self.detector._derive_session_pattern(state_short)
        snap_long = self.detector._derive_session_pattern(state_long)
        assert snap_long.confidence > snap_short.confidence
