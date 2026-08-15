"""
Phase 7 (unblocked) — the fairness/ESL false-positive check plan §8 asks
for: "False-positive rate specifically on ELLIPSE, and on PERSUADE essays
broken out by ELL status... Subgroup breakdown on PERSUADE using its
economic-disadvantage, race/ethnicity, and disability fields — report
whether false-positive rate differs meaningfully across groups."

Both PERSUADE and ELLIPSE are entirely real human writing (no AI class in
either source) — every essay's ground truth is "human," so any essay
scored "Likely AI" is by definition a false positive. Uses the exact same
production pipeline (featurize -> classify -> aggregate.essay_verdict,
including short-sentence context-merging and calibrated thresholds) that
the served app uses — this measures the actual deployed behavior, not a
different evaluation-only configuration.

Small-n subgroups are reported but flagged, not silently included as if
equally reliable — a 5-essay subgroup FPR is not the same kind of number
as a 150-essay one.

Usage:
    cd backend && .venv/bin/python scripts/score_fairness.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from app.pipeline import aggregate, classify, featurize  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = REPO_ROOT / "data" / "processed"

MIN_SUBGROUP_N = 20


def score_essay(text: str) -> tuple[str, float]:
    sentence_feats = featurize.featurize_essay(text)
    scores = [classify.predict_proba(sf.features) for sf in sentence_feats]
    verdict = aggregate.essay_verdict(scores)
    return verdict.label, sum(scores) / len(scores) if scores else 0.0


def fpr(labels: list[str]) -> tuple[float, float]:
    n = len(labels)
    if n == 0:
        return 0.0, 0.0
    strict = sum(1 for l in labels if l == "Likely AI") / n
    broad = sum(1 for l in labels if l != "Likely Human") / n
    return strict, broad


def main() -> None:
    print("=== ELLIPSE (ESL human writing) ===")
    with (PROCESSED_DIR / "ellipse_eval.csv").open() as f:
        ellipse_rows = list(csv.DictReader(f))
    ellipse_results = []
    for i, r in enumerate(ellipse_rows):
        label, mean_score = score_essay(r["text"])
        ellipse_results.append({"id": r["id"], "label": label, "mean_score": mean_score})
        if (i + 1) % 40 == 0:
            print(f"  [{i+1}/{len(ellipse_rows)}]")
    strict, broad = fpr([r["label"] for r in ellipse_results])
    print(f"\nELLIPSE false-positive rate (n={len(ellipse_results)}): "
          f"{strict:.3f} strict (Likely AI) / {broad:.3f} broad (not Likely Human)")

    with (PROCESSED_DIR / "ellipse_scores.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "label", "mean_score"])
        w.writeheader()
        w.writerows(ellipse_results)

    print("\n=== PERSUADE (human, demographic breakdown) ===")
    with (PROCESSED_DIR / "persuade_eval.csv").open() as f:
        persuade_rows = list(csv.DictReader(f))
    persuade_results = []
    for i, r in enumerate(persuade_rows):
        label, mean_score = score_essay(r["text"])
        persuade_results.append({
            "id": r["id"], "label": label, "mean_score": mean_score,
            "ell_status": r["ell_status"],
            "race_ethnicity": r["race_ethnicity"],
            "economically_disadvantaged": r["economically_disadvantaged"],
            "student_disability_status": r["student_disability_status"],
        })
        if (i + 1) % 40 == 0:
            print(f"  [{i+1}/{len(persuade_rows)}]")

    with (PROCESSED_DIR / "persuade_scores.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "label", "mean_score", "ell_status", "race_ethnicity",
                                           "economically_disadvantaged", "student_disability_status"])
        w.writeheader()
        w.writerows(persuade_results)

    overall_strict, overall_broad = fpr([r["label"] for r in persuade_results])
    print(f"\nPERSUADE overall false-positive rate (n={len(persuade_results)}): "
          f"{overall_strict:.3f} strict / {overall_broad:.3f} broad")

    for field in ["ell_status", "race_ethnicity", "economically_disadvantaged", "student_disability_status"]:
        print(f"\n-- by {field} --")
        groups: dict[str, list[str]] = {}
        for r in persuade_results:
            groups.setdefault(r[field] or "(blank)", []).append(r["label"])
        for group, labels in sorted(groups.items(), key=lambda kv: -len(kv[1])):
            s, b = fpr(labels)
            flag = "" if len(labels) >= MIN_SUBGROUP_N else "  <-- n too small to trust"
            print(f"  {group:45s} n={len(labels):4d}  strict_fpr={s:.3f}  broad_fpr={b:.3f}{flag}")


if __name__ == "__main__":
    main()
