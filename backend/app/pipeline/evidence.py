"""
Evidence for a sentence's ai_score — plan §5's evidence math.

Round 6: the classifier switched from LogisticRegression to
GradientBoostingClassifier (real accuracy gain, see
docs/IMPLEMENTATION.md), which has no per-feature coefficients, so
the original `coefficient_i x standardized_value_i` contribution math
(kept for the git history / old versions of this docstring) no longer
applies. Replaced with single-feature perturbation, still fully
deterministic and still model-agnostic:

    contribution_i = P(AI | actual vector)
                      - P(AI | actual vector, but feature i swapped to
                             the median value for that feature among
                             HUMAN training essays)

i.e. "how much does this one feature's actual value move the score,
relative to a typical human sentence?" Every other feature is held at
its real value while feature i is perturbed, so this is a genuine local
(per-sentence) marginal contribution, not a fixed global weight — arguably
more honest for a nonlinear model, where a feature's effect can
legitimately depend on the other feature values around it. Sorted by
|contribution| descending, top 3, each phrased against its percentile in
the human-only training distribution (reference_stats, built in
train_classifier.py from the human class only) — that part is unchanged.

This is the ONLY source of the "why" shown in the UI. Every number here
traces back to a specific computed value — never free text from a model
(plan §0).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import classify
from .featurize import FEATURE_NAMES

# Human-readable labels for the UI — the only non-computed strings in this
# module, and they're fixed labels for feature identity, not generated
# explanations of what the model found.
FEATURE_LABELS = {
    "perplexity": "sentence perplexity",
    "gltr_pct_top10": "% tokens in GPT-2's top-10 predictions",
    "gltr_pct_top100": "% tokens in GPT-2's top-100 predictions",
    "gltr_pct_top1000": "% tokens in GPT-2's top-1,000 predictions",
    "gltr_mean_rank": "average predicted-token rank",
    "sent_length": "sentence length",
    "avg_word_length": "average word length",
    "rolling_ttr": "vocabulary variety (rolling type-token ratio)",
    "punctuation_rate": "punctuation rate",
    "function_word_ratio": "function-word ratio",
    "adjective_ratio": "adjective ratio",
    "has_contraction": "contraction use",
    "passive_voice_rate": "passive-voice rate",
}


@dataclass
class FeatureContribution:
    name: str
    percentile: float  # 0-100: this value vs. the human training distribution
    direction: str  # "ai-like" | "human-like"
    magnitude: float  # |contribution| — raw, for relative bar-width scaling in the UI


def _percentile(value: float, reference_sorted: np.ndarray) -> float:
    if len(reference_sorted) == 0:
        return 50.0
    rank = np.searchsorted(reference_sorted, value, side="right")
    return 100.0 * rank / len(reference_sorted)


def top_features(features: dict[str, float], top_n: int = 3) -> list[FeatureContribution]:
    x_scaled = classify.scale_vector(features)
    reference = classify.reference_stats()
    baseline_proba = classify.predict_proba_from_scaled(x_scaled)

    contributions = []
    for i, name in enumerate(FEATURE_NAMES):
        ref_values = reference[name]
        median_raw = float(np.median(ref_values)) if len(ref_values) else features[name]
        perturbed = x_scaled.copy()
        perturbed[i] = classify.scaled_value(name, median_raw)
        contribution = baseline_proba - classify.predict_proba_from_scaled(perturbed)
        contributions.append(FeatureContribution(
            name=FEATURE_LABELS.get(name, name),
            percentile=_percentile(features[name], ref_values),
            direction="ai-like" if contribution > 0 else "human-like",
            magnitude=abs(float(contribution)),
        ))

    contributions.sort(key=lambda c: c.magnitude, reverse=True)
    return contributions[:top_n]
