"""
Sentence scores -> essay-level verdict + document-level features (plan §4)
+ transition detection (plan §6). These are all POST-HOC over the
classifier's own per-sentence scores — none of them are classifier
features themselves (not in featurize.FEATURE_NAMES), so adding them
never requires retraining.
"""
from __future__ import annotations

import json
import statistics
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

# Below this many words, a sentence's score computed from itself ALONE is
# empirically unreliable: on the eval-model's held-out predictions,
# human/AI mean-score separation was near zero (or inverted) for 1-3 word
# sentences, then stabilized from 4 words on — see
# calibrate_thresholds.py and DOCUMENTS/IMPLEMENTATION.md for the actual
# numbers this is based on, not a guess. Used by
# featurize._scoring_spans() to decide when to merge a short sentence's
# span with a neighbor for feature computation — by the time scores reach
# essay_verdict() below, every sentence has already been given a
# reliably-sized context to be scored from, so there's nothing left to
# exclude here (an earlier version of this function did exclude short
# sentences from the mean directly; superseded by the upstream merge,
# which recovers signal instead of discarding it).
MIN_RELIABLE_TOKENS = 4

THRESHOLDS_PATH = Path(__file__).resolve().parents[1] / "models" / "thresholds.json"
_DEFAULT_THRESHOLDS = {"likely_human_max": 0.35, "likely_ai_min": 0.65}


@lru_cache(maxsize=1)
def _thresholds() -> dict:
    """Essay-verdict cutoffs. Loaded from thresholds.json if
    calibrate_thresholds.py has been run; falls back to the original
    Phase 2 placeholder values (symmetric around 0.5) otherwise, so the
    app still works before calibration exists."""
    if THRESHOLDS_PATH.exists():
        return json.loads(THRESHOLDS_PATH.read_text())
    return _DEFAULT_THRESHOLDS


@dataclass
class Verdict:
    label: str
    confidence: float


@dataclass
class TransitionFlag:
    sentence_index: int
    note: str


# score_volatility integration, calibrated empirically before wiring in
# (not guessed): compared volatility on 327 PURE single-class training
# essays (feature_table_phase2.csv) against 30 genuinely MIXED hybrid
# essays. The separation is weak at the median (pure 0.192 vs hybrid
# 0.204 — heavily overlapping, NOT a strong general-purpose signal) but
# real at the tail (pure p95=0.256, hybrid max=0.321). So this only acts
# as a high-volatility tripwire, not a graded adjustment across the whole
# range — the data doesn't support anything stronger than that, and
# claiming otherwise would be the same "oversell a weak signal" mistake
# this project has deliberately avoided elsewhere (see the rejected
# sentence-threshold recalibration in DOCUMENTS/IMPLEMENTATION.md).
HIGH_VOLATILITY_THRESHOLD = 0.26
HIGH_VOLATILITY_CONFIDENCE_PENALTY = 0.7  # confidence *= this, when tripped


def essay_verdict(sentence_scores: list[float]) -> Verdict:
    """Mean-score thresholding against calibrated cutoffs (_thresholds()).
    No length-based exclusion here — every sentence's score already
    reflects a reliably-sized span by the time it reaches this function
    (featurize._scoring_spans() merges short sentences with a neighbor
    upstream, at feature-computation time). Confidence (not the label) is
    dampened when score_volatility crosses HIGH_VOLATILITY_THRESHOLD —
    see that constant's comment for why this is deliberately a mild,
    tail-only effect rather than a strong one."""
    if not sentence_scores:
        return Verdict(label="Inconclusive", confidence=0.0)

    scores = sentence_scores
    thresholds = _thresholds()
    mean_score = sum(scores) / len(scores)
    if mean_score <= thresholds["likely_human_max"]:
        label = "Likely Human"
    elif mean_score >= thresholds["likely_ai_min"]:
        label = "Likely AI"
    else:
        label = "Inconclusive"
    confidence = min(abs(mean_score - 0.5) * 2, 1.0)
    if score_volatility(scores) >= HIGH_VOLATILITY_THRESHOLD:
        confidence *= HIGH_VOLATILITY_CONFIDENCE_PENALTY
    return Verdict(label=label, confidence=confidence)


def burstiness(perplexities: list[float]) -> float:
    """Plan §4: stdev of per-sentence perplexity across the essay."""
    if len(perplexities) < 2:
        return 0.0
    return statistics.stdev(perplexities)


def sentence_length_variance(sentence_lengths: list[float]) -> float:
    """Plan §4: variance of per-sentence word count across the essay."""
    if len(sentence_lengths) < 2:
        return 0.0
    return statistics.variance(sentence_lengths)


def score_volatility(sentence_scores: list[float]) -> float:
    """Plan §4: mean absolute difference between consecutive sentence
    ai_scores. Feeds essay_verdict()'s confidence (high-volatility
    tripwire only, see HIGH_VOLATILITY_THRESHOLD) and detect_transitions()
    below."""
    if len(sentence_scores) < 2:
        return 0.0
    diffs = [abs(sentence_scores[i] - sentence_scores[i - 1]) for i in range(1, len(sentence_scores))]
    return sum(diffs) / len(diffs)


# A jump has to clear both an absolute floor and a per-essay statistical
# bar to count as a transition — the floor stops near-constant essays
# (tiny baseline volatility) from flagging every trivial wiggle as a
# "transition."
MIN_ABS_JUMP = 0.25
STD_MULTIPLIER = 1.25


def detect_transitions(sentence_scores: list[float]) -> list[TransitionFlag]:
    """Flag sentence indices where the score jump from the previous
    sentence is a local outlier relative to this essay's own volatility
    baseline — plan §4/§6's "possible change in writing pattern here"
    signal, e.g. a human paragraph an AI later polished."""
    if len(sentence_scores) < 3:
        return []

    diffs = [sentence_scores[i] - sentence_scores[i - 1] for i in range(1, len(sentence_scores))]
    abs_diffs = [abs(d) for d in diffs]
    mean_diff = sum(abs_diffs) / len(abs_diffs)
    std_diff = statistics.pstdev(abs_diffs) if len(abs_diffs) > 1 else 0.0
    cutoff = max(mean_diff + STD_MULTIPLIER * std_diff, MIN_ABS_JUMP)

    flags = []
    for i, (d, abs_d) in enumerate(zip(diffs, abs_diffs)):
        if abs_d >= cutoff:
            direction = "more AI-like" if d > 0 else "more human-like"
            flags.append(TransitionFlag(
                sentence_index=i + 1,  # the sentence where the jump lands
                note=f"Score shifted {direction} here, a bigger jump than the rest of this essay — "
                     f"possible change in writing pattern.",
            ))
    return flags
