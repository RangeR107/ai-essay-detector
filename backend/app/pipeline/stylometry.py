"""
Per-sentence stylometric features (plan §4) from spaCy's POS/dependency
parse — no GPT-2 involved here, purely linguistic/structural signal,
computed alongside (not instead of) the GLTR/perplexity features in
featurize.py.
"""
from __future__ import annotations

import re

from spacy.tokens import Doc, Span

FUNCTION_POS = {"DET", "ADP", "PRON", "CCONJ", "SCONJ", "AUX", "PART"}
# spaCy's English models label passive constituents with these deps
# (not the Universal Dependencies "nsubj:pass" form).
PASSIVE_DEPS = {"nsubjpass", "csubjpass", "auxpass"}
CONTRACTION_RE = re.compile(r"\w+'(s|re|ve|ll|d|m|t)\b", re.IGNORECASE)

ROLLING_WINDOW_TOKENS = 50

FEATURE_NAMES = [
    "sent_length",
    "avg_word_length",
    "rolling_ttr",
    "punctuation_rate",
    "function_word_ratio",
    "adjective_ratio",
    "has_contraction",
    "passive_voice_rate",
]


def _rolling_ttr(doc_tokens: list, end_idx: int, window: int = ROLLING_WINDOW_TOKENS) -> float:
    start_idx = max(0, end_idx - window)
    window_words = [t.text.lower() for t in doc_tokens[start_idx:end_idx] if t.is_alpha]
    if not window_words:
        return 0.0
    return len(set(window_words)) / len(window_words)


def compute(span: Span, doc: Doc) -> dict[str, float]:
    tokens = [t for t in span if not t.is_space]
    words = [t for t in tokens if t.is_alpha]
    n_tokens = len(tokens)
    n_words = len(words)

    avg_word_length = sum(len(t.text) for t in words) / n_words if n_words else 0.0
    punctuation_rate = sum(1 for t in tokens if t.is_punct) / n_tokens if n_tokens else 0.0

    function_word_ratio = sum(1 for t in words if t.pos_ in FUNCTION_POS) / n_words if n_words else 0.0
    adjective_ratio = sum(1 for t in words if t.pos_ == "ADJ") / n_words if n_words else 0.0

    has_contraction = 1.0 if CONTRACTION_RE.search(span.text) else 0.0

    verbs = [t for t in span if t.pos_ in ("VERB", "AUX")]
    passive_markers = sum(1 for t in span if t.dep_ in PASSIVE_DEPS)
    passive_voice_rate = passive_markers / len(verbs) if verbs else 0.0

    return {
        "sent_length": float(n_words),
        "avg_word_length": avg_word_length,
        "rolling_ttr": _rolling_ttr(list(doc), span.end),
        "punctuation_rate": punctuation_rate,
        "function_word_ratio": function_word_ratio,
        "adjective_ratio": adjective_ratio,
        "has_contraction": has_contraction,
        "passive_voice_rate": passive_voice_rate,
    }
