"""
Experiment, NOT wired into production: calibrate_thresholds.py has only
ever tuned the CENTER of the Likely-Human/Likely-AI band (the decision
point that maximizes balanced accuracy) — the band's HALF-WIDTH itself
(+-0.15, giving the current 0.36/0.66 cutoffs) was inherited unchanged
from the original Phase 2 placeholder design and never actually
calibrated against data.

This script sweeps that half-width directly against the same eval-model
held-out essay-level data calibrate_thresholds.py uses (never trained on
by the eval model — see that script's docstring for the leakage-avoidance
reasoning, unchanged here), holding the center point fixed at the
already-calibrated decision point. For each half-width, reports:

  - Inconclusive rate: what fraction of essays fall inside the band
    (get no definitive verdict)
  - Definitive accuracy: of essays that DO get a definitive Likely-
    Human/Likely-AI verdict, what fraction are correct
  - Overall correct-and-decisive rate: definitive_accuracy x (1 -
    inconclusive_rate) — the single number that trades off both costs
    (a wrong confident verdict vs. a correct one buried under
    "Inconclusive")

This is measured only on in-genre, in-training-distribution essays (the
same data thresholds are calibrated on) — see the script's printed
caveat about why that matters for interpreting the result.

Usage:
    cd backend && .venv/bin/python scripts/experiment_threshold_band_width.py
"""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

REPO_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
MODELS_DIR = REPO_ROOT / "backend" / "app" / "models"

HALF_WIDTHS = [0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25]


def main() -> None:
    thresholds = json.loads((MODELS_DIR / "thresholds.json").read_text())
    center = thresholds["_calibration_decision_point"]
    print(f"Calibrated decision point (center): {center}")
    print(f"Current production half-width: 0.15 (band: "
          f"{thresholds['likely_human_max']}-{thresholds['likely_ai_min']})\n")

    with (PROCESSED_DIR / "eval_model_predictions.csv").open() as f:
        rows = list(csv.DictReader(f))

    by_essay: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for r in rows:
        by_essay[r["essay_id"]].append((r["class"], float(r["score"])))

    essay_items = []
    for essay_id, sent_rows in by_essay.items():
        true_class = sent_rows[0][0]
        mean_score = sum(s for _, s in sent_rows) / len(sent_rows)
        essay_items.append((true_class, mean_score))

    print(f"{len(essay_items)} held-out essays (never trained on by the eval model)\n")
    print(f"{'Half-width':>10s} {'Band':>13s} {'Inconclusive %':>15s} {'Definitive acc':>16s} {'Correct-and-decisive':>22s}")
    print("-" * 82)

    for hw in HALF_WIDTHS:
        lo, hi = round(max(0.0, center - hw), 2), round(min(1.0, center + hw), 2)
        n_total = len(essay_items)
        n_inconclusive = 0
        n_definitive = 0
        n_definitive_correct = 0
        for true_class, mean_score in essay_items:
            if lo < mean_score < hi:
                n_inconclusive += 1
                continue
            n_definitive += 1
            pred = "ai" if mean_score >= hi else "human"
            if pred == true_class:
                n_definitive_correct += 1

        inconclusive_rate = n_inconclusive / n_total
        definitive_acc = n_definitive_correct / n_definitive if n_definitive else 0.0
        correct_and_decisive = n_definitive_correct / n_total

        marker = "  <- current" if hw == 0.15 else ""
        print(f"{hw:>10.2f} {lo:>6.2f}-{hi:<6.2f} {inconclusive_rate:>14.1%} "
              f"{definitive_acc:>16.1%} {correct_and_decisive:>21.1%}{marker}")

    print("\nCaveat: this is measured on in-genre, in-training-distribution\n"
          "essays only (the same population thresholds are calibrated on).\n"
          "A narrower band will look better here but is more likely to produce\n"
          "confidently-wrong verdicts specifically on OUT-of-distribution essays\n"
          "(different genre/generator) — this sweep can't see that risk, since\n"
          "no cross-genre essays have calibration-quality ground truth here.")


if __name__ == "__main__":
    main()
