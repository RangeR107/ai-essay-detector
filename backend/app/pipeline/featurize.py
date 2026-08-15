"""
Combines segmentation + the single GPT-2 pass into a per-sentence feature
vector. Phase 2 used perplexity alone; Phase 3 added GLTR rank-bucket
aggregates (% of tokens in top-10/top-100/top-1000, mean rank) from the
same forward pass — no second pass, no new model. Phase 4 adds stylometry
(spaCy POS/dependency features, no GPT-2 involved) the same way: extend
the feature dict, no pipeline restructuring.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from . import gltr, segment, stylometry
from .aggregate import MIN_RELIABLE_TOKENS
from .textclean import strip_markdown

# Ordered feature names -> keep this list and the vector-building order in
# sync; the classifier's StandardScaler/coefficients are indexed positionally.
#
# gltr_pct_top10 deliberately excluded from the classifier's input vector
# (still computed below and present in the per-sentence features dict, just
# not fed to the model): investigate_fairness_bias.py found it was the
# single dominant driver of the measured ELL/non-ELL false-positive-rate
# gap (docs/EVALUATION.md §5, docs/LIMITATIONS.md #1) — mean contribution
# delta +0.50, ~3x the next-biggest feature. Root cause: the model had
# learned "high top-10 % = human-like" specifically from this project's
# training data's register (EssayForum's plainer prose vs Gemini's more
# elaborate prose), not a reliable general AI-vs-human signal — and ELL
# writers land in the same "low top-10%" zone for an unrelated reason
# (non-native phrasing patterns), so they absorbed a penalty that wasn't
# actually measuring what it claimed to. Tested directly
# (experiment_drop_feature.py): removing it roughly HALVED the FPR gap
# (14.7pp -> 7.3pp) for a small in-theme accuracy cost (73.6% -> 72.0%).
# Adopted given the fairness gap is this project's most consequential
# limitation. See DOCUMENTS/IMPLEMENTATION.md for the full before/after.
FEATURE_NAMES = [
    "perplexity",
    "gltr_pct_top100",
    "gltr_pct_top1000",
    "gltr_mean_rank",
    *stylometry.FEATURE_NAMES,
]


def _gltr_aggregates(sent_token_stats: list[gltr.TokenStat]) -> dict[str, float]:
    n = len(sent_token_stats)
    if n == 0:
        return {"gltr_pct_top10": 0.0, "gltr_pct_top100": 0.0, "gltr_pct_top1000": 0.0, "gltr_mean_rank": 0.0}
    buckets = [gltr.rank_bucket(s.rank) for s in sent_token_stats]
    return {
        "gltr_pct_top10": buckets.count("top10") / n,
        "gltr_pct_top100": buckets.count("top100") / n,
        "gltr_pct_top1000": buckets.count("top1000") / n,
        "gltr_mean_rank": sum(s.rank for s in sent_token_stats) / n,
    }


@dataclass
class SentenceFeatures:
    text: str
    start: int
    end: int
    features: dict[str, float]
    token_stats: list[gltr.TokenStat] = field(default_factory=list)
    # True if this sentence was under MIN_RELIABLE_TOKENS words and its
    # features/score come from a merged span with a neighboring sentence
    # rather than itself alone — see _scoring_spans() below. Kept as
    # transparency info for the UI, not an exclusion flag (that was the
    # old design; this sentence still gets a real, context-informed
    # score now, nothing is thrown away).
    context_merged: bool = False


def _scoring_spans(sentences: list[segment.Sentence]) -> list[tuple[int, int, bool]]:
    """One (start, end, was_merged) per input sentence, in the same
    order. A sentence under MIN_RELIABLE_TOKENS words gets its span
    extended to include a neighbor (next sentence if available, else
    previous) for feature computation — replaces the old approach of
    just excluding short sentences from scoring, which threw away
    signal instead of trying to recover it. Measured motivation: on the
    eval-model's held-out predictions, human/AI mean-score separation
    was near-zero or inverted for 1-3 word sentences alone (see
    aggregate.MIN_RELIABLE_TOKENS's docstring for the actual numbers) —
    giving those sentences surrounding context is the more honest fix
    than silently dropping them from the essay-level verdict."""
    word_counts = [len(s.text.split()) for s in sentences]
    n = len(sentences)
    spans = []
    for i, s in enumerate(sentences):
        if word_counts[i] >= MIN_RELIABLE_TOKENS or n == 1:
            spans.append((s.start, s.end, False))
        elif i + 1 < n:
            spans.append((s.start, sentences[i + 1].end, True))
        else:
            spans.append((sentences[i - 1].start, s.end, True))
    return spans


def featurize_essay(essay_text: str) -> list[SentenceFeatures]:
    # Strip markdown before segmentation/GLTR, not after — the training
    # data went through the same cleanup (ingest_ai_essays.py), and a
    # pasted "**bold**"/"# Header" left in place both looks nothing like
    # what the classifier trained on AND can corrupt sentence
    # segmentation (a stray "**" gluing onto the next sentence). See
    # textclean.py's docstring for the concrete case that surfaced this.
    essay_text = strip_markdown(essay_text)
    doc = segment.parse(essay_text)
    sentences = segment.sentences_from_doc(doc)
    stats = gltr.token_stats(essay_text)
    scoring_spans = _scoring_spans(sentences)

    results = []
    for sent, (span_start, span_end, was_merged) in zip(sentences, scoring_spans):
        span_token_stats = [s for s in stats if s.start >= span_start and s.end <= span_end]
        perplexity = gltr.perplexity_for_span(stats, span_start, span_end)
        if perplexity is None:
            continue
        scoring_span = doc.char_span(span_start, span_end, alignment_mode="expand")
        features = {"perplexity": perplexity}
        features.update(_gltr_aggregates(span_token_stats))
        features.update(stylometry.compute(scoring_span, doc))
        # token_stats (for the UI's per-token heatmap) still reflect only
        # THIS sentence's own tokens, not the merged neighbor's — the
        # heatmap should highlight what's actually in the sentence shown,
        # even when the aggregate score/features behind it came from a
        # wider context.
        own_token_stats = [s for s in stats if s.start >= sent.start and s.end <= sent.end]
        results.append(SentenceFeatures(
            text=sent.text,
            start=sent.start,
            end=sent.end,
            features=features,
            token_stats=own_token_stats,
            context_merged=was_merged,
        ))
    return results


def feature_vector(features: dict[str, float]) -> list[float]:
    return [features[name] for name in FEATURE_NAMES]
