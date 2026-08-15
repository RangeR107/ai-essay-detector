"""
Calibrate the essay-verdict thresholds (aggregate.py's Likely-Human /
Likely-AI cutoffs) against real held-out data instead of the Phase 2
placeholder values (0.35/0.65, symmetric around 0.5, never tuned against
anything).

Uses data/processed/eval_model_predictions.csv — sentence-level
predictions from the EVALUATION model (trained on the 80% seen-theme
split in train_classifier.py) on its own held-out test/unseen rows. This
is deliberately NOT the production classifier.joblib's predictions on
these same essays: the production model IS trained on them, so its
predictions here would be optimistic (the model has seen the answer).
Tuning on an unbiased model's held-out predictions and then applying the
resulting threshold VALUE to the production model at inference time is
standard practice and avoids that leakage.

Honest limitation: this calibration set is only ~1000 essays' worth of
sentences from the same single-generator/single-source pool everything
else in this project comes from (see docs/LIMITATIONS.md #1) — it tells
us where the decision boundary sits for essays LIKE these, not
universally.

Method: aggregate sentence scores to essay-level mean (every sentence's
score already reflects a reliably-sized span by this point — short
sentences were merged with a neighbor upstream at feature-computation
time, see featurize._scoring_spans() — so this matches exactly what
essay_verdict() does in production), sweep a candidate decision
threshold t from 0.05 to 0.95, pick t* maximizing
balanced accuracy (average of human recall and AI recall at that cutoff —
chosen over raw accuracy specifically because raw accuracy rewards
degenerate "always predict the bigger class" thresholds, exactly the
failure mode this whole project has been fighting since the placeholder-
data phases). The Likely-Human/Likely-AI band width (0.30, i.e. +-0.15)
is kept the same as the original Phase 2 design, just recentered on t*
instead of assuming t*=0.5.

Also reports (but does not wire in) the best SENTENCE-level threshold, for
future evaluation scripts that want it instead of the sklearn-default 0.5.

Output: backend/app/models/thresholds.json

Usage:
    cd backend && .venv/bin/python scripts/calibrate_thresholds.py
"""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from app.pipeline.aggregate import MIN_RELIABLE_TOKENS  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
MODELS_DIR = REPO_ROOT / "backend" / "app" / "models"

BAND_HALF_WIDTH = 0.08  # replaces the original inherited-but-never-calibrated
# 0.15. experiment_threshold_band_width.py swept half-width directly against
# this same held-out data: 0.15 left 73.4% of essays "Inconclusive" (100.0%
# accuracy on the 26.6% that got a definitive verdict) — a lot of usable
# signal thrown away for a fairly small accuracy gain over narrower widths.
# 0.08 gives a 27.4% Inconclusive rate at 98.9% definitive accuracy, chosen
# by the user as the balance point after seeing the full width-vs-accuracy
# tradeoff table (see that script's docstring for the important caveat:
# this tradeoff is measured on in-distribution essays only — a narrower
# band is riskier specifically on cross-genre/cross-generator essays, where
# the wider band's caution was doing real protective work).


def balanced_accuracy_at(threshold: float, items: list[tuple[str, float]]) -> float:
    human = [s for c, s in items if c == "human"]
    ai = [s for c, s in items if c == "ai"]
    human_recall = sum(1 for s in human if s < threshold) / len(human) if human else 0.0
    ai_recall = sum(1 for s in ai if s >= threshold) / len(ai) if ai else 0.0
    return (human_recall + ai_recall) / 2


def sweep_best_threshold(items: list[tuple[str, float]]) -> tuple[float, float]:
    best_t, best_acc = 0.5, -1.0
    t = 0.05
    while t <= 0.95:
        acc = balanced_accuracy_at(t, items)
        if acc > best_acc:
            best_t, best_acc = t, acc
        t = round(t + 0.01, 2)
    return best_t, best_acc


def main() -> None:
    path = PROCESSED_DIR / "eval_model_predictions.csv"
    if not path.exists():
        print(f"{path} not found — run train_classifier.py first.")
        return
    with path.open() as f:
        rows = list(csv.DictReader(f))
    print(f"Loaded {len(rows)} eval-model held-out sentence predictions "
          f"(MIN_RELIABLE_TOKENS={MIN_RELIABLE_TOKENS})")

    # --- Sentence-level threshold (informational) ---
    sentence_items = [(r["class"], float(r["score"])) for r in rows]
    sent_t, sent_acc = sweep_best_threshold(sentence_items)
    print(f"\nBest sentence-level threshold: {sent_t:.2f} (balanced accuracy {sent_acc:.3f}, vs 0.50 default)")
    default_acc = balanced_accuracy_at(0.5, sentence_items)
    print(f"  (balanced accuracy at 0.50: {default_acc:.3f})")

    # --- Essay-level threshold (operational — this is what aggregate.py uses) ---
    by_essay: dict[str, list[tuple[str, float, float]]] = defaultdict(list)
    for r in rows:
        by_essay[r["essay_id"]].append((r["class"], float(r["score"]), float(r["sent_length"])))

    essay_items = []
    for essay_id, sent_rows in by_essay.items():
        true_class = sent_rows[0][0]
        scores_to_use = [score for _, score, _ in sent_rows]
        mean_score = sum(scores_to_use) / len(scores_to_use)
        essay_items.append((true_class, mean_score))

    essay_t, essay_acc = sweep_best_threshold(essay_items)
    default_essay_acc = balanced_accuracy_at(0.5, essay_items)
    print(f"\n{len(essay_items)} essays. Best essay-level decision point: {essay_t:.2f} "
          f"(balanced accuracy {essay_acc:.3f}, vs {default_essay_acc:.3f} at 0.50)")

    likely_human_max = round(max(0.0, essay_t - BAND_HALF_WIDTH), 2)
    likely_ai_min = round(min(1.0, essay_t + BAND_HALF_WIDTH), 2)
    print(f"\nCalibrated essay-verdict thresholds: "
          f"Likely Human <= {likely_human_max}, Likely AI >= {likely_ai_min} "
          f"(was 0.35 / 0.65)")

    thresholds = {
        "likely_human_max": likely_human_max,
        "likely_ai_min": likely_ai_min,
        "_calibration_decision_point": essay_t,
        "_calibration_essay_balanced_accuracy": round(essay_acc, 3),
        "_calibration_sentence_threshold_fyi": sent_t,
        "_calibration_n_essays": len(essay_items),
        "_calibration_n_sentences": len(rows),
        "_note": "See backend/scripts/calibrate_thresholds.py docstring for method and the leakage-avoidance reasoning.",
    }
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = MODELS_DIR / "thresholds.json"
    out_path.write_text(json.dumps(thresholds, indent=2))
    print(f"\nSaved {out_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
