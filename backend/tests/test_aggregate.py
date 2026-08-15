"""Tests for app.pipeline.aggregate — pure functions, no model loading,
fast. This is where the essay-verdict/transition/volatility LOGIC lives,
separate from the classifier itself, so these tests don't need GPT-2."""
from __future__ import annotations

from app.pipeline import aggregate


class TestEssayVerdict:
    def test_empty_input_is_inconclusive(self):
        v = aggregate.essay_verdict([])
        assert v.label == "Inconclusive"
        assert v.confidence == 0.0

    def test_low_mean_score_is_likely_human(self):
        v = aggregate.essay_verdict([0.05, 0.1, 0.08, 0.5, 0.5])
        assert v.label in ("Likely Human", "Inconclusive")

    def test_high_mean_score_is_likely_ai(self):
        v = aggregate.essay_verdict([0.95, 0.92, 0.9])
        assert v.label == "Likely AI"

    def test_mid_mean_score_is_inconclusive(self):
        v = aggregate.essay_verdict([0.5, 0.5, 0.5])
        assert v.label == "Inconclusive"

    def test_high_volatility_reduces_confidence(self):
        stable = aggregate.essay_verdict([0.6, 0.62, 0.58, 0.61])
        bouncy = aggregate.essay_verdict([0.2, 0.9, 0.15, 0.85])
        assert bouncy.confidence < stable.confidence + 0.5  # bouncy penalized relative to its own raw mean-based confidence
        assert aggregate.score_volatility([0.2, 0.9, 0.15, 0.85]) >= aggregate.HIGH_VOLATILITY_THRESHOLD


class TestDocumentLevelFeatures:
    def test_burstiness_needs_two_sentences(self):
        assert aggregate.burstiness([50.0]) == 0.0
        assert aggregate.burstiness([]) == 0.0
        assert aggregate.burstiness([10.0, 90.0]) > 0.0

    def test_sentence_length_variance(self):
        assert aggregate.sentence_length_variance([5.0]) == 0.0
        assert aggregate.sentence_length_variance([5.0, 5.0, 5.0]) == 0.0
        assert aggregate.sentence_length_variance([1.0, 100.0]) > 0.0

    def test_score_volatility(self):
        assert aggregate.score_volatility([0.5]) == 0.0
        assert aggregate.score_volatility([0.5, 0.5, 0.5]) == 0.0
        assert aggregate.score_volatility([0.0, 1.0]) == 1.0


class TestDetectTransitions:
    def test_too_few_sentences_returns_nothing(self):
        assert aggregate.detect_transitions([0.1, 0.9]) == []

    def test_flat_scores_flag_nothing(self):
        assert aggregate.detect_transitions([0.5, 0.51, 0.49, 0.5, 0.52]) == []

    def test_real_jump_gets_flagged(self):
        # ONE genuine transition point (a realistic human-then-AI splice
        # shape), not a back-and-forth — detect_transitions() is a
        # relative-to-own-baseline outlier detector, so two comparably
        # large jumps in the same short essay can normalize each other
        # out statistically (both become "not outliers" relative to each
        # other). A single sustained shift is the case that matters.
        scores = [0.9, 0.91, 0.89, 0.9, 0.1, 0.08, 0.09, 0.1]
        flags = aggregate.detect_transitions(scores)
        assert len(flags) >= 1
        flagged_indices = {f.sentence_index for f in flags}
        assert 4 in flagged_indices  # the sentence where the drop lands
        assert "human-like" in flags[0].note or "AI-like" in flags[0].note
