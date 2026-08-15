from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .pipeline import aggregate, classify, evidence, featurize, gltr
from .schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    EssayLevelFeatures,
    FeatureContribution,
    SentenceResult,
    TokenRank,
    Transition,
    VerdictResult,
)

app = FastAPI(title="AI Admissions Essay Detector")

app.add_middleware(
    CORSMiddleware,
    # Vite picks the next free port (5173, 5174, ...) if 5173 is already
    # taken by another running instance — a hardcoded single origin broke
    # local dev with a "Failed to fetch" the moment that happened. Allowing
    # the whole localhost/127.0.0.1 dev-port range is safe here (this is a
    # local-only dev server, not a deployed API with real user sessions).
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1):\d+",
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(request: AnalyzeRequest):
    sentence_feats = featurize.featurize_essay(request.essay_text)

    sentence_results = []
    scores = []
    perplexities = []
    sentence_lengths = []
    for sf in sentence_feats:
        score = classify.predict_proba(sf.features)
        scores.append(score)
        perplexities.append(sf.features["perplexity"])
        sentence_lengths.append(sf.features["sent_length"])
        token_ranks = [
            TokenRank(
                # GPT-2's byte-level BPE marks a leading space with 'Ġ' —
                # swap it for a real space so the UI renders readable text.
                token=t.token.replace("Ġ", " ").replace("Ċ", "\n"),
                rank_bucket=gltr.rank_bucket(t.rank),
            )
            for t in sf.token_stats
        ]
        top_features = [
            FeatureContribution(name=c.name, percentile=c.percentile, direction=c.direction, magnitude=c.magnitude)
            for c in evidence.top_features(sf.features)
        ]
        sentence_results.append(SentenceResult(
            text=sf.text,
            start=sf.start,
            end=sf.end,
            ai_score=score,
            top_features=top_features,
            token_ranks=token_ranks,
            context_merged=sf.context_merged,
        ))

    verdict = aggregate.essay_verdict(scores)
    transitions = [
        Transition(sentence_index=t.sentence_index, note=t.note)
        for t in aggregate.detect_transitions(scores)
    ]

    return AnalyzeResponse(
        sentences=sentence_results,
        transitions=transitions,
        verdict=VerdictResult(label=verdict.label, confidence=verdict.confidence),
        essay_level_features=EssayLevelFeatures(
            burstiness=aggregate.burstiness(perplexities),
            sentence_length_variance=aggregate.sentence_length_variance(sentence_lengths),
            score_volatility=aggregate.score_volatility(scores),
        ),
    )
