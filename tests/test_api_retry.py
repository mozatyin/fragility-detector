"""Tests for API retry utilities."""

import pytest
from fragility_detector.api_retry import make_client


class TestMakeClient:
    def test_openrouter_key_sets_base_url(self):
        client = make_client("sk-or-test-key")
        assert "openrouter" in str(client.base_url)

    def test_regular_key_no_base_url(self):
        client = make_client("sk-ant-test-key")
        assert "openrouter" not in str(client.base_url)
