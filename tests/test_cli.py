"""Tests for CLI argument parsing."""

import pytest
from fragility_detector.cli import parse_args


class TestParseArgs:
    def test_smoke_flag(self):
        args = parse_args(["--smoke"])
        assert args.smoke is True

    def test_detect_text(self):
        args = parse_args(["--detect", "I feel broken"])
        assert args.detect == "I feel broken"

    def test_features_only(self):
        args = parse_args(["--features-only", "--detect", "test"])
        assert args.features_only is True

    def test_default_output(self):
        args = parse_args(["--smoke"])
        assert args.output == "results/"
