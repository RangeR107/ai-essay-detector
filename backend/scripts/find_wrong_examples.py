"""
Phase 7 — find the three examples plan §8 asks for: one confident false
positive, one confident false negative, one hybrid-essay boundary miss.
Each gets its actual top-contributing features (evidence.py) attached so
the write-up in docs/EVALUATION.md is grounded in real numbers, not
speculation, per the plan's explicit instruction.

Reuses feature_table_phase2.csv (already-computed sentence-level features
for all 1000 training essays) rather than re-running the pipeline — this
is post-hoc analysis of the existing production model, no new GPT-2/spaCy
passes needed.

Usage:
    cd backend && .venv/bin/python scripts/find_wrong_examples.py
"""
from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from app.pipeline import classify, evidence, featurize  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = REPO_ROOT / "data" / "processed"


def load_essay_texts() -> dict[str, str]:
    texts = {}
    for fname in ["human_essays.csv", "ai_essays.csv"]:
        with (PROCESSED_DIR / fname).open() as f:
            for r in csv.DictReader(f):
                texts[r["id"]] = r["text"]
    return texts


def main() -> None:
    with (PROCESSED_DIR / "feature_table_phase2.csv").open() as f:
        rows = list(csv.DictReader(f))

    by_essay = defaultdict(list)
    for r in rows:
        by_essay[r["essay_id"]].append(r)

    essay_scores = []
    for essay_id, sent_rows in by_essay.items():
        scores = []
        for r in sent_rows:
            features = {name: float(r[name]) for name in featurize.FEATURE_NAMES}
            scores.append((classify.predict_proba(features), r, features))
        mean_score = sum(s for s, _, _ in scores) / len(scores)
        essay_scores.append((essay_id, sent_rows[0]["class"], mean_score, scores))

    human_essays = sorted(
        (e for e in essay_scores if e[1] == "human"), key=lambda e: e[2], reverse=True
    )
    ai_essays = sorted(
        (e for e in essay_scores if e[1] == "ai"), key=lambda e: e[2]
    )

    texts = load_essay_texts()

    print("=" * 70)
    print("FALSE POSITIVE CANDIDATE (real human essay, highest mean AI-score)")
    print("=" * 70)
    fp_id, _, fp_mean, fp_scores = human_essays[0]
    print(f"essay_id={fp_id}  mean_score={fp_mean:.3f}")
    worst_sent = max(fp_scores, key=lambda t: t[0])
    print(f"Most damning sentence (score={worst_sent[0]:.3f}): {worst_sent[1]['sentence_idx']}")
    for c in evidence.top_features(worst_sent[2]):
        print(f"  {c.name}: {c.percentile:.1f}th percentile, {c.direction}, magnitude={c.magnitude:.3f}")
    print(f"\nFull essay text ({fp_id}):\n{texts.get(fp_id, '[not found]')}\n")

    print("=" * 70)
    print("FALSE NEGATIVE CANDIDATE (real AI essay, lowest mean AI-score)")
    print("=" * 70)
    fn_id, _, fn_mean, fn_scores = ai_essays[0]
    print(f"essay_id={fn_id}  mean_score={fn_mean:.3f}")
    calmest_sent = min(fn_scores, key=lambda t: t[0])
    print(f"Most human-reading sentence (score={calmest_sent[0]:.3f}): {calmest_sent[1]['sentence_idx']}")
    for c in evidence.top_features(calmest_sent[2]):
        print(f"  {c.name}: {c.percentile:.1f}th percentile, {c.direction}, magnitude={c.magnitude:.3f}")
    print(f"\nFull essay text ({fn_id}):\n{texts.get(fn_id, '[not found]')}\n")

    print("=" * 70)
    print("HYBRID-ESSAY BOUNDARY MISS (worst localization)")
    print("=" * 70)
    with (PROCESSED_DIR / "hybrid_essay_scores.csv").open() as f:
        hybrid_scores = list(csv.DictReader(f))
    # Worst = highest false-negative rate on the AI span specifically
    # (missed the most of the actual spliced-in AI content).
    def fn_rate(r):
        tp, fn = int(r["tp"]), int(r["fn"])
        total_ai = tp + fn
        return fn / total_ai if total_ai else -1
    worst_hybrid = max(hybrid_scores, key=fn_rate)
    print(f"id={worst_hybrid['id']}  tp={worst_hybrid['tp']} fp={worst_hybrid['fp']} "
          f"fn={worst_hybrid['fn']} tn={worst_hybrid['tn']}  fn_rate={fn_rate(worst_hybrid):.2f}")

    with (PROCESSED_DIR / "hybrid_essays.csv").open() as f:
        hybrid_essays = {r["id"]: r for r in csv.DictReader(f)}
    hybrid_row = hybrid_essays[worst_hybrid["id"]]
    ai_span_start = int(hybrid_row["ai_span_start"])
    sentence_feats = featurize.featurize_essay(hybrid_row["text"])
    print(f"\nai_span_start={ai_span_start}, {hybrid_row['n_human_sentences']} human + "
          f"{hybrid_row['n_ai_sentences']} ai sentences")
    print("\nPer-sentence breakdown:")
    for sf in sentence_feats:
        true_ai = sf.start >= ai_span_start
        score = classify.predict_proba(sf.features)
        pred_ai = score >= 0.5
        marker = "MISS" if true_ai != pred_ai else "ok"
        print(f"  [{marker}] true={'ai' if true_ai else 'human'} pred={'ai' if pred_ai else 'human'} "
              f"score={score:.3f} | {sf.text[:80]}")
        if true_ai and not pred_ai:
            for c in evidence.top_features(sf.features):
                print(f"      {c.name}: {c.percentile:.1f}th pct, {c.direction}, mag={c.magnitude:.3f}")


if __name__ == "__main__":
    main()
