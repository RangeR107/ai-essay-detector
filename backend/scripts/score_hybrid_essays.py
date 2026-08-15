"""
Score the 100 hybrid essays (build_hybrid_essays.py) against their known
ground-truth AI span — plan §8's passage-localization deliverable:
"report precision and recall on the flagged span itself against the
known-replaced span, not just an essay-level right/wrong."

Sentences under aggregate.MIN_RELIABLE_TOKENS words are no longer
excluded from these counts. Earlier this script dropped them entirely
because a short sentence's score computed from itself alone was
unreliable; now featurize._scoring_spans() merges those sentences with
a neighbor at feature-computation time, so every sentence's score is
already context-informed by the time it reaches this script. The
`context_merged` flag on each SentenceFeatures is still tracked below
(informational only) so the per-essay breakdown shows how many
sentences in each essay relied on a merged span.

The sentence-level decision threshold stays at the standard 0.5, NOT the
0.42 value calibrate_thresholds.py computes for sentence-level balanced
accuracy — that threshold was tested here too and made things worse for
this specific task (P 0.400->0.357, R 0.777->0.874, F1 0.528->0.507): it
trades precision for recall by flagging more sentences as AI overall,
which helps balanced accuracy on independently-sampled essays but hurts
precision when the actual question is "which specific sentences in a
mixed essay are AI." In this application, a false "AI-written" flag on a
student's real writing is the costlier error, so precision is weighted
higher here than the balanced-accuracy objective calibration optimized
for — deliberately not adopting that threshold for this evaluation.

Ground truth per sentence: a sentence counts as truly-AI if its start
offset falls at/after ai_span_start (the splice always lands exactly on a
sentence boundary by construction — see build_hybrid_essays.py — so this
is exact, not an approximation).

Usage:
    cd backend && .venv/bin/python scripts/score_hybrid_essays.py
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
    threshold = SENTENCE_THRESHOLD
    path = PROCESSED_DIR / "hybrid_essays.csv"
    with path.open() as f:
        rows = list(csv.DictReader(f))
    print(f"Scoring {len(rows)} hybrid essays (sentence threshold={threshold})...")

    tp = fp = fn = tn = 0
    merged = 0
    per_essay_rows = []
    for i, r in enumerate(rows):
        ai_span_start = int(r["ai_span_start"])
        sentence_feats = featurize.featurize_essay(r["text"])
        essay_tp = essay_fp = essay_fn = essay_tn = essay_merged = 0
        for sf in sentence_feats:
            if sf.context_merged:
                essay_merged += 1
            true_ai = sf.start >= ai_span_start
            score = classify.predict_proba(sf.features)
            pred_ai = score >= threshold
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
            "id": r["id"],
            "n_sentences": len(sentence_feats),
            "tp": essay_tp, "fp": essay_fp, "fn": essay_fn, "tn": essay_tn,
            "context_merged": essay_merged,
        })
        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{len(rows)}]")

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    print(f"\n=== Span-localization result, sentence-level, {len(rows)} hybrid essays ===")
    print(f"TP={tp} FP={fp} FN={fn} TN={tn}  ({merged} sentences scored from a merged span)")
    print(f"Precision (of sentences flagged AI, how many were truly the spliced-in AI span): {precision:.3f}")
    print(f"Recall (of the truly-AI spliced sentences, how many got flagged):                {recall:.3f}")
    print(f"F1: {f1:.3f}")

    out_path = PROCESSED_DIR / "hybrid_essay_scores.csv"
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "n_sentences", "tp", "fp", "fn", "tn", "context_merged"])
        writer.writeheader()
        writer.writerows(per_essay_rows)
    print(f"\nPer-essay breakdown written to {out_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
