"""Pydantic request/response models — see plan §6 for the target shape."""
from __future__ import annotations

from pydantic import BaseModel


class AnalyzeRequest(BaseModel):
    essay_text: str


class FeatureContribution(BaseModel):
    name: str
    percentile: float
    direction: str  # "ai-like" | "human-like"
    # |P(AI) with this feature actual| - |P(AI) with this feature swapped to
    # its human-reference median| (evidence.py — perturbation-based since
    # round 6's move to GradientBoostingClassifier). Not in plan §6's
    # original shape, added in Phase 5 so the UI can size the "small bar for
    # contribution magnitude" plan §7 asks for. Only meaningful relative to
    # the other entries in the same sentence's top_features list, not across
    # sentences.
    magnitude: float


class TokenRank(BaseModel):
    token: str
    rank_bucket: str  # "top10" | "top100" | "top1000" | "rest"


class SentenceResult(BaseModel):
    text: str
    start: int
    end: int
    ai_score: float
    top_features: list[FeatureContribution] = []
    token_ranks: list[TokenRank] = []
    # True for sentences under aggregate.MIN_RELIABLE_TOKENS words — their
    # score/evidence comes from a merged span with a neighboring sentence
    # (featurize._scoring_spans()), not from this sentence alone, because
    # evaluation showed human/AI score separation is near-zero (or
    # inverted) below that length. Still a real, context-informed score —
    # nothing is excluded or hidden, this is transparency info for the UI
    # to show "this score reflects nearby context," not a warning to
    # discount it.
    context_merged: bool = False


class Transition(BaseModel):
    sentence_index: int
    note: str


class VerdictResult(BaseModel):
    label: str
    confidence: float


class EssayLevelFeatures(BaseModel):
    burstiness: float
    # Beyond plan §6's original single-field shape — the plan's own §4
    # names all three of these as the document-level feature set
    # (burstiness was just the one already wired in Phase 2).
    sentence_length_variance: float
    score_volatility: float


class AnalyzeResponse(BaseModel):
    sentences: list[SentenceResult]
    transitions: list[Transition] = []
    verdict: VerdictResult
    essay_level_features: EssayLevelFeatures
