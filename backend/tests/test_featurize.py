"""Tests for the featurize/gltr/segment/stylometry pipeline. These load
spaCy + GPT-2 (via the pipeline modules' own caching), so they're slower
than test_aggregate.py but still run in seconds after the first load."""
from __future__ import annotations

from app.pipeline import featurize, segment


class TestSegment:
    def test_splits_into_sentences(self):
        sents = segment.segment("This is one sentence. This is another.")
        assert len(sents) == 2
        assert sents[0].text == "This is one sentence."
        assert sents[1].text == "This is another."

    def test_offsets_match_original_text(self):
        text = "First sentence here. Second one follows."
        sents = segment.segment(text)
        for s in sents:
            assert text[s.start:s.end] == s.text

    def test_empty_text(self):
        assert segment.segment("") == []


class TestFeaturizeEssay:
    def test_returns_one_result_per_sentence(self, sample_essay):
        results = featurize.featurize_essay(sample_essay)
        expected_sentences = segment.segment(sample_essay)
        assert len(results) == len(expected_sentences)

    def test_all_feature_names_present(self, sample_essay):
        results = featurize.featurize_essay(sample_essay)
        for r in results:
            assert set(r.features.keys()) == set(featurize.FEATURE_NAMES) | {
                # gltr_pct_top10 is still computed (used by the token
                # heatmap indirectly via rank buckets) even though it's
                # excluded from FEATURE_NAMES — see featurize.py's comment.
                "gltr_pct_top10"
            }

    def test_gltr_pct_top10_excluded_from_classifier_features(self):
        # Regression test for the fairness-mitigation feature drop —
        # this must stay out of FEATURE_NAMES specifically.
        assert "gltr_pct_top10" not in featurize.FEATURE_NAMES

    def test_feature_vector_matches_feature_names_order(self, sample_essay):
        results = featurize.featurize_essay(sample_essay)
        vec = featurize.feature_vector(results[0].features)
        assert len(vec) == len(featurize.FEATURE_NAMES)
        for name, value in zip(featurize.FEATURE_NAMES, vec):
            assert value == results[0].features[name]

    def test_token_stats_are_within_sentence_bounds(self, sample_essay):
        results = featurize.featurize_essay(sample_essay)
        for r in results:
            for t in r.token_stats:
                assert r.start <= t.start
                assert t.end <= r.end

    def test_perplexity_is_positive(self, sample_essay):
        results = featurize.featurize_essay(sample_essay)
        for r in results:
            assert r.features["perplexity"] > 0
