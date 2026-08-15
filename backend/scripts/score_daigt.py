"""
Phase 7 — score a DAIGT-v2 subset (data/processed/daigt_eval.csv) against
the trained classifier: the cross-genre generalization check from plan
§3b/§8. DAIGT is argumentative/persuasive writing, not admissions personal
statements — this measures whether the classifier's signal holds up
outside the genre it was trained on, not whether it's "as accurate" (a
distribution shift is expected and informative either way).

Samples a subset rather than all 44,868 rows — running the full GPT-2 +
spaCy pipeline over the whole dataset would take hours for a check the
plan itself calls optional/subset-sized. Stratified by class so the
sample isn't accidentally almost-all-human (DAIGT is ~61% human overall).

Metrics reported at the SENTENCE level, matching the in-theme/unseen-theme
numbers from train_classifier.py, so all three are directly comparable.

Usage:
    cd backend && .venv/bin/python scripts/score_daigt.py
"""
from __future__ import annotations

import csv
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from app.pipeline import classify, featurize  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = REPO_ROOT / "data" / "processed"

N_PER_CLASS = 100
SEED = 42


def main() -> None:
    with (PROCESSED_DIR / "daigt_eval.csv").open() as f:
        rows = list(csv.DictReader(f))
    human_rows = [r for r in rows if r["class"] == "human"]
    ai_rows = [r for r in rows if r["class"] == "ai"]
    print(f"DAIGT pool: {len(human_rows)} human, {len(ai_rows)} ai")

    rng = random.Random(SEED)
    sample = rng.sample(human_rows, N_PER_CLASS) + rng.sample(ai_rows, N_PER_CLASS)
    rng.shuffle(sample)
    print(f"Scoring stratified sample: {N_PER_CLASS} human + {N_PER_CLASS} ai essays")

    tp = fp = fn = tn = 0
    essay_correct = 0
    essay_total = 0
    for i, r in enumerate(sample):
        true_ai = r["class"] == "ai"
        sentence_feats = featurize.featurize_essay(r["text"])
        if not sentence_feats:
            continue
        scores = [classify.predict_proba(sf.features) for sf in sentence_feats]
        for score in scores:
            pred_ai = score >= 0.5
            if true_ai and pred_ai:
                tp += 1
            elif not true_ai and pred_ai:
                fp += 1
            elif true_ai and not pred_ai:
                fn += 1
            else:
                tn += 1
        essay_verdict_ai = (sum(scores) / len(scores)) >= 0.5
        essay_correct += int(essay_verdict_ai == true_ai)
        essay_total += 1
        if (i + 1) % 40 == 0:
            print(f"  [{i+1}/{len(sample)}]")

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    accuracy = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) else 0.0
    human_recall = tn / (tn + fp) if (tn + fp) else 0.0

    print(f"\n=== DAIGT-v2 cross-genre check, sentence-level, {len(sample)} essays ===")
    print(f"TP={tp} FP={fp} FN={fn} TN={tn}")
    print(f"Accuracy:        {accuracy:.3f}")
    print(f"AI precision:    {precision:.3f}")
    print(f"AI recall:       {recall:.3f}")
    print(f"Human recall:    {human_recall:.3f}")
    print(f"F1 (ai class):   {f1:.3f}")
    print(f"\nEssay-level verdict (mean-score >= 0.5) accuracy: {essay_correct}/{essay_total} = {essay_correct/essay_total:.3f}")


if __name__ == "__main__":
    main()
