"""
GPT-2 is used ONLY as an instrument here: one forward pass over the essay
produces per-token log-probabilities. Nothing in this module (or anywhere
downstream) asks GPT-2 for a verdict, label, or explanation — see plan §0.

A single forward pass yields both perplexity (Phase 2) and GLTR rank buckets
(Phase 3) — don't compute them with separate passes/models.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

MODEL_NAME = "gpt2"


@dataclass
class TokenStat:
    token: str
    start: int
    end: int
    rank: int  # 1-indexed rank of the actual token in the predicted distribution
    logprob: float


@lru_cache(maxsize=1)
def _tokenizer() -> GPT2TokenizerFast:
    return GPT2TokenizerFast.from_pretrained(MODEL_NAME)


@lru_cache(maxsize=1)
def _model() -> GPT2LMHeadModel:
    m = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
    m.eval()
    return m


@torch.no_grad()
def token_stats(text: str, max_tokens: int = 1024) -> list[TokenStat]:
    """One GPT-2 forward pass -> per-token (rank, logprob) with char offsets.

    The first token has no preceding context to predict it from, so it's
    skipped (matches standard perplexity/GLTR convention).
    """
    if not text.strip():
        # Empty/whitespace-only input tokenizes to a 0-length tensor,
        # which GPT2LMHeadModel's forward pass can't reshape (crashes,
        # not just "no results") — caught by test_api.py's empty-essay
        # test. Nothing to score either way, so short-circuit.
        return []

    tok = _tokenizer()
    encoding = tok(text, return_offsets_mapping=True, return_tensors="pt", truncation=True, max_length=max_tokens)
    input_ids = encoding["input_ids"]
    offsets = encoding["offset_mapping"][0].tolist()

    logits = _model()(input_ids).logits[0]  # (seq_len, vocab_size)
    log_probs = torch.log_softmax(logits, dim=-1)  # (seq_len, vocab_size)

    stats: list[TokenStat] = []
    seq_len = input_ids.shape[1]
    for i in range(1, seq_len):
        actual_id = input_ids[0, i].item()
        predicted_dist = log_probs[i - 1]  # distribution predicting position i
        token_logprob = predicted_dist[actual_id].item()
        rank = int((predicted_dist > token_logprob).sum().item()) + 1
        start, end = offsets[i]
        if start == end:  # special/empty token
            continue
        stats.append(TokenStat(
            token=tok.convert_ids_to_tokens(actual_id),
            start=start,
            end=end,
            rank=rank,
            logprob=token_logprob,
        ))
    return stats


def perplexity_for_span(stats: list[TokenStat], start: int, end: int) -> float | None:
    span_logprobs = [s.logprob for s in stats if s.start >= start and s.end <= end]
    if not span_logprobs:
        return None
    mean_neg_logprob = -sum(span_logprobs) / len(span_logprobs)
    return math.exp(mean_neg_logprob)


def rank_bucket(rank: int) -> str:
    if rank <= 10:
        return "top10"
    if rank <= 100:
        return "top100"
    if rank <= 1000:
        return "top1000"
    return "rest"
