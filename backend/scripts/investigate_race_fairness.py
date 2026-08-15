"""
Root-cause investigation for the race/ethnicity fairness gap (Task #52),
same methodology as investigate_fairness_bias.py's ELL investigation:
score_fairness.py found Black/African American PERSUADE essays flagged
"Inconclusive/Likely AI" (broad FPR) at a meaningfully higher rate than
White essays (0.417 vs 0.296, n=36 vs n=81 — both above MIN_SUBGROUP_N).
That result told us THAT a gap exists; this script investigates WHY —
which of the classifier's features actually differ in value between the
two groups, and which features' coefficient contributions are doing the
most work pushing Black/African American essays toward "ai-like."

Re-featurizes the same 300-essay PERSUADE sample (persuade_eval.csv) —
score_fairness.py only persisted the final label/mean_score, not the
underlying per-feature values, so this needs its own pass.

Two views computed, same as the ELL investigation:
  1. Raw feature means per group + standardized difference (Cohen's-d-like)
     — which features actually differ in value between the groups.
  2. Mean signed contribution (coefficient x standardized_value) per group
     — which features the classifier's own learned weights make
     load-bearing for this specific disparity.

Usage:
    cd backend && .venv/bin/python scripts/investigate_race_fairness.py
"""
from __future__ import annotations

import csv
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from app.pipeline import classify, featurize  # noqa: E402
from app.pipeline.featurize import FEATURE_NAMES  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = REPO_ROOT / "data" / "processed"

GROUP_A = "Black/African American"
GROUP_B = "White"


def main() -> None:
    with (PROCESSED_DIR / "persuade_eval.csv").open() as f:
        rows = list(csv.DictReader(f))
    print(f"Re-featurizing PERSUADE essays for '{GROUP_A}' vs '{GROUP_B}' per-feature comparison...")

    by_group: dict[str, dict[str, list[float]]] = {
        GROUP_A: {name: [] for name in FEATURE_NAMES},
        GROUP_B: {name: [] for name in FEATURE_NAMES},
    }
    contrib_by_group: dict[str, dict[str, list[float]]] = {
        GROUP_A: {name: [] for name in FEATURE_NAMES},
        GROUP_B: {name: [] for name in FEATURE_NAMES},
    }

    coefs = classify.coefficients()

    n_used = {GROUP_A: 0, GROUP_B: 0}
    for i, r in enumerate(rows):
        group = r["race_ethnicity"].strip()
        if group not in (GROUP_A, GROUP_B):
            continue
        sentence_feats = featurize.featurize_essay(r["text"])
        use = sentence_feats
        if not use:
            continue
        n_used[group] += 1

        for name_idx, name in enumerate(FEATURE_NAMES):
            raw_vals = [sf.features[name] for sf in use]
            by_group[group][name].append(sum(raw_vals) / len(raw_vals))

            contribs = []
            for sf in use:
                x_scaled = classify.scale_vector(sf.features)
                contribs.append(coefs[name_idx] * x_scaled[name_idx])
            contrib_by_group[group][name].append(sum(contribs) / len(contribs))

        if (i + 1) % 40 == 0:
            print(f"  [{i+1}/{len(rows)}]")

    print(f"\n{n_used[GROUP_A]} {GROUP_A} essays, {n_used[GROUP_B]} {GROUP_B} essays used.\n")

    print(f"=== 1. Raw feature values: {GROUP_A} vs {GROUP_B} (standardized difference) ===")
    diffs = []
    for name in FEATURE_NAMES:
        a_vals = by_group[GROUP_A][name]
        b_vals = by_group[GROUP_B][name]
        a_mean, b_mean = statistics.mean(a_vals), statistics.mean(b_vals)
        pooled_std = statistics.pstdev(a_vals + b_vals) or 1e-9
        std_diff = (a_mean - b_mean) / pooled_std
        diffs.append((name, a_mean, b_mean, std_diff))
    diffs.sort(key=lambda t: abs(t[3]), reverse=True)
    for name, a_mean, b_mean, std_diff in diffs:
        print(f"  {name:25s} {GROUP_A}={a_mean:8.3f}  {GROUP_B}={b_mean:8.3f}  std_diff={std_diff:+.3f}")

    print(f"\n=== 2. Mean signed contribution (coef x standardized value): {GROUP_A} vs {GROUP_B} ===")
    print("     (positive = pushes toward 'ai-like'; this is what actually drives the verdict)")
    contrib_diffs = []
    for name in FEATURE_NAMES:
        a_c = statistics.mean(contrib_by_group[GROUP_A][name])
        b_c = statistics.mean(contrib_by_group[GROUP_B][name])
        contrib_diffs.append((name, a_c, b_c, a_c - b_c))
    contrib_diffs.sort(key=lambda t: abs(t[3]), reverse=True)
    for name, a_c, b_c, delta in contrib_diffs:
        print(f"  {name:25s} {GROUP_A}_contrib={a_c:+.4f}  {GROUP_B}_contrib={b_c:+.4f}  delta={delta:+.4f}")

    total_a = sum(statistics.mean(contrib_by_group[GROUP_A][n]) for n in FEATURE_NAMES)
    total_b = sum(statistics.mean(contrib_by_group[GROUP_B][n]) for n in FEATURE_NAMES)
    print(f"\nTotal mean essay-level contribution sum: {GROUP_A}={total_a:+.4f}  {GROUP_B}={total_b:+.4f}  "
          f"gap={total_a - total_b:+.4f}")


if __name__ == "__main__":
    main()
