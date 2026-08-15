"""
Scores the mid-paragraph hybrid essays (build_hybrid_essays_midparagraph.py)
against their ground-truth AI span — same methodology as
score_hybrid_essays.py (sentence threshold 0.5; short sentences are
merged with a neighbor upstream in featurize.py rather than excluded, so
every sentence here gets counted), run separately here to see whether
localization performs differently on a human-AI-human splice than on the
continuation (human-then-AI) splice.

Usage:
    cd backend && .venv/bin/python scripts/score_hybrid_essays_midparagraph.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from app.pipeline import classify, featurize  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
SENTENCE_THRESHOLD = 0.5


def main() -> None:
    with (PROCESSED_DIR / "hybrid_essays_midparagraph.csv").open() as f:
        rows = list(csv.DictReader(f))
    print(f"Scoring {len(rows)} mid-paragraph hybrid essays...")

    tp = fp = fn = tn = 0
    merged = 0
    per_essay_rows = []
    for i, r in enumerate(rows):
        ai_span_start, ai_span_end = int(r["ai_span_start"]), int(r["ai_span_end"])
        sentence_feats = featurize.featurize_essay(r["text"])
        essay_tp = essay_fp = essay_fn = essay_tn = essay_merged = 0
        for sf in sentence_feats:
            if sf.context_merged:
                essay_merged += 1
            true_ai = sf.start >= ai_span_start and sf.start < ai_span_end
            score = classify.predict_proba(sf.features)
            pred_ai = score >= SENTENCE_THRESHOLD
            if true_ai and pred_ai:
                essay_tp += 1
            elif not true_ai and pred_ai:
                essay_fp += 1
            elif true_ai and not pred_ai:
                essay_fn += 1
            else:
                essay_tn += 1
        tp += essay_tp
        fp += essay_fp
        fn += essay_fn
        tn += essay_tn
        merged += essay_merged
        per_essay_rows.append({
            "id": r["id"], "n_sentences": len(sentence_feats),
            "tp": essay_tp, "fp": essay_fp, "fn": essay_fn, "tn": essay_tn,
            "context_merged": essay_merged,
        })
        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{len(rows)}]")

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    print(f"\n=== Mid-paragraph span-localization result, sentence-level, {len(rows)} essays ===")
    print(f"TP={tp} FP={fp} FN={fn} TN={tn}  ({merged} sentences scored from a merged span)")
    print(f"Precision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}")
    print(f"F1:        {f1:.3f}")

    out_path = PROCESSED_DIR / "hybrid_essay_midparagraph_scores.csv"
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "n_sentences", "tp", "fp", "fn", "tn", "context_merged"])
        w.writeheader()
        w.writerows(per_essay_rows)
    print(f"\nPer-essay breakdown written to {out_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
