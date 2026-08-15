"""API-level tests via FastAPI's TestClient — exercises the actual served
/health and /analyze endpoints end to end (real model, real pipeline, no
mocking), the same code path the frontend calls."""
from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


class TestAnalyze:
    def test_returns_200_with_valid_shape(self, sample_essay):
        resp = client.post("/analyze", json={"essay_text": sample_essay})
        assert resp.status_code == 200
        data = resp.json()
        assert "sentences" in data
        assert "transitions" in data
        assert "verdict" in data
        assert "essay_level_features" in data

    def test_sentence_count_matches_input(self, sample_essay):
        resp = client.post("/analyze", json={"essay_text": sample_essay})
        data = resp.json()
        assert len(data["sentences"]) == 3  # sample_essay has 3 sentences

    def test_each_sentence_has_expected_fields(self, sample_essay):
        resp = client.post("/analyze", json={"essay_text": sample_essay})
        data = resp.json()
        for s in data["sentences"]:
            assert 0.0 <= s["ai_score"] <= 1.0
            assert isinstance(s["context_merged"], bool)
            assert "top_features" in s
            assert "token_ranks" in s
            for f in s["top_features"]:
                assert f["direction"] in ("ai-like", "human-like")
                assert "magnitude" in f

    def test_verdict_label_is_valid(self, sample_essay):
        resp = client.post("/analyze", json={"essay_text": sample_essay})
        verdict = resp.json()["verdict"]
        assert verdict["label"] in ("Likely Human", "Inconclusive", "Likely AI")
        assert 0.0 <= verdict["confidence"] <= 1.0

    def test_essay_level_features_present(self, sample_essay):
        resp = client.post("/analyze", json={"essay_text": sample_essay})
        elf = resp.json()["essay_level_features"]
        assert "burstiness" in elf
        assert "sentence_length_variance" in elf
        assert "score_volatility" in elf

    def test_empty_essay_does_not_crash(self):
        resp = client.post("/analyze", json={"essay_text": ""})
        assert resp.status_code == 200
        data = resp.json()
        assert data["sentences"] == []
        assert data["verdict"]["label"] == "Inconclusive"

    def test_single_short_sentence_essay_has_nothing_to_merge_with(self):
        # A lone short sentence has no neighbor to merge with, so it's
        # scored from itself alone — context_merged stays False.
        resp = client.post("/analyze", json={"essay_text": "Yes."})
        data = resp.json()
        assert len(data["sentences"]) == 1
        assert data["sentences"][0]["context_merged"] is False

    def test_short_sentence_among_others_gets_merged(self):
        resp = client.post("/analyze", json={"essay_text": "Yes. This is a much longer sentence with plenty of words."})
        data = resp.json()
        assert len(data["sentences"]) == 2
        assert data["sentences"][0]["context_merged"] is True

    def test_missing_field_returns_422(self):
        resp = client.post("/analyze", json={})
        assert resp.status_code == 422
