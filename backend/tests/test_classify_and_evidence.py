"""Tests for classify.py and evidence.py against the actual trained
production model — these verify the shipped classifier.joblib loads and
behaves sanely, not just that the code runs. If these fail, the deployed
model is broken."""
from __future__ import annotations

from app.pipeline import classify, evidence, featurize


class TestClassify:
    def test_predict_proba_is_a_probability(self, sample_essay):
        results = featurize.featurize_essay(sample_essay)
        for r in results:
            score = classify.predict_proba(r.features)
            assert 0.0 <= score <= 1.0

    def test_scaled_value_matches_scale_vector(self, sample_essay):
        # scaled_value(name, raw) should agree with scale_vector's own
        # scaling of that same feature — both go through the same scaler.
        results = featurize.featurize_essay(sample_essay)
        features = results[0].features
        scaled = classify.scale_vector(features)
        for i, name in enumerate(featurize.FEATURE_NAMES):
            assert classify.scaled_value(name, features[name]) == scaled[i]

    def test_predict_proba_from_scaled_matches_predict_proba(self, sample_essay):
        results = featurize.featurize_essay(sample_essay)
        for r in results:
            scaled = classify.scale_vector(r.features)
            assert classify.predict_proba_from_scaled(scaled) == classify.predict_proba(r.features)

    def test_scale_vector_length_matches_feature_names(self, sample_essay):
        results = featurize.featurize_essay(sample_essay)
        scaled = classify.scale_vector(results[0].features)
        assert len(scaled) == len(featurize.FEATURE_NAMES)

    def test_reference_stats_has_all_feature_names(self):
        stats = classify.reference_stats()
        assert set(stats.keys()) == set(featurize.FEATURE_NAMES)
        for name in featurize.FEATURE_NAMES:
            assert len(stats[name]) > 0


class TestEvidence:
    def test_top_features_returns_at_most_three(self, sample_essay):
        results = featurize.featurize_essay(sample_essay)
        for r in results:
            contributions = evidence.top_features(r.features)
            assert len(contributions) <= 3

    def test_sorted_by_magnitude_descending(self, sample_essay):
        results = featurize.featurize_essay(sample_essay)
        contributions = evidence.top_features(results[0].features, top_n=len(featurize.FEATURE_NAMES))
        magnitudes = [c.magnitude for c in contributions]
        assert magnitudes == sorted(magnitudes, reverse=True)

    def test_direction_matches_contribution_sign(self, sample_essay):
        results = featurize.featurize_essay(sample_essay)
        for c in evidence.top_features(results[0].features):
            assert c.direction in ("ai-like", "human-like")

    def test_percentile_in_valid_range(self, sample_essay):
        results = featurize.featurize_essay(sample_essay)
        for r in results:
            for c in evidence.top_features(r.features):
                assert 0.0 <= c.percentile <= 100.0
