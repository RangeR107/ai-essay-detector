"""
Root-cause investigation for the measured fairness gap (docs/EVALUATION.md
§5, docs/LIMITATIONS.md #1): ELL/non-White/economically-disadvantaged
PERSUADE essays get flagged "Inconclusive" at ~1.7-2x the rate of their
peers. That result told us THAT a gap exists; this script investigates
WHY — which of the 13 classifier features actually differ between groups,
and which features' coefficient contributions are doing the most work
pushing ELL essays toward "ai-like."

Re-featurizes the same 300-essay PERSUADE sample (persuade_eval.csv) —
score_fairness.py only persisted the final label/mean_score, not the
underlying per-feature values, so this needs its own pass.

Two views computed:
  1. Raw feature means per group + standardized difference (Cohen's-d-like:
     (mean_ELL - mean_nonELL) / pooled_stdev) — which features actually
     differ in value between the groups, regardless of whether the
     classifier weights them heavily.
  2. Mean |contribution| (coefficient x standardized_value, evidence.py's
     own formula) per group, and specifically the mean SIGNED contribution
     difference — which features are pushing ELL essays toward "ai-like"
     more than non-ELL essays, i.e. which features the classifier's own
     learned weights make load-bearing for this specific disparity.

Usage:
    cd backend && .venv/bin/python scripts/investigate_fairness_bias.py
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


def main() -> None:
    with (PROCESSED_DIR / "persuade_eval.csv").open() as f:
        rows = list(csv.DictReader(f))
    print(f"Re-featurizing {len(rows)} PERSUADE essays for per-feature group comparison...")

    # essay-level mean of each raw feature, and mean signed contribution,
    # per essay, grouped by ell_status.
    by_group: dict[str, dict[str, list[float]]] = {
        "Yes": {name: [] for name in FEATURE_NAMES},
        "No": {name: [] for name in FEATURE_NAMES},
    }
    contrib_by_group: dict[str, dict[str, list[float]]] = {
        "Yes": {name: [] for name in FEATURE_NAMES},
        "No": {name: [] for name in FEATURE_NAMES},
    }

    coefs = classify.coefficients()

    for i, r in enumerate(rows):
        status = r["ell_status"].strip()
        if status not in ("Yes", "No"):
            continue
        sentence_feats = featurize.featurize_essay(r["text"])
        use = sentence_feats
        if not use:
            continue

        for name_idx, name in enumerate(FEATURE_NAMES):
            raw_vals = [sf.features[name] for sf in use]
            by_group[status][name].append(sum(raw_vals) / len(raw_vals))

            contribs = []
            for sf in use:
                x_scaled = classify.scale_vector(sf.features)
                contribs.append(coefs[name_idx] * x_scaled[name_idx])
            contrib_by_group[status][name].append(sum(contribs) / len(contribs))

        if (i + 1) % 40 == 0:
            print(f"  [{i+1}/{len(rows)}]")

    print(f"\n{len(by_group['Yes'][FEATURE_NAMES[0]])} ELL essays, "
          f"{len(by_group['No'][FEATURE_NAMES[0]])} non-ELL essays used.\n")

    print("=== 1. Raw feature values: ELL vs non-ELL (standardized difference) ===")
    diffs = []
    for name in FEATURE_NAMES:
        ell_vals = by_group["Yes"][name]
        non_vals = by_group["No"][name]
        ell_mean, non_mean = statistics.mean(ell_vals), statistics.mean(non_vals)
        pooled_std = statistics.pstdev(ell_vals + non_vals) or 1e-9
        std_diff = (ell_mean - non_mean) / pooled_std
        diffs.append((name, ell_mean, non_mean, std_diff))
    diffs.sort(key=lambda t: abs(t[3]), reverse=True)
    for name, ell_mean, non_mean, std_diff in diffs:
        print(f"  {name:25s} ELL={ell_mean:8.3f}  non-ELL={non_mean:8.3f}  std_diff={std_diff:+.3f}")

    print("\n=== 2. Mean signed contribution (coef x standardized value): ELL vs non-ELL ===")
    print("     (positive = pushes toward 'ai-like'; this is what actually drives the verdict)")
    contrib_diffs = []
    for name in FEATURE_NAMES:
        ell_c = statistics.mean(contrib_by_group["Yes"][name])
        non_c = statistics.mean(contrib_by_group["No"][name])
        contrib_diffs.append((name, ell_c, non_c, ell_c - non_c))
    contrib_diffs.sort(key=lambda t: abs(t[3]), reverse=True)
    for name, ell_c, non_c, delta in contrib_diffs:
        print(f"  {name:25s} ELL_contrib={ell_c:+.4f}  non-ELL_contrib={non_c:+.4f}  delta={delta:+.4f}")

    total_ell = sum(statistics.mean(contrib_by_group["Yes"][n]) for n in FEATURE_NAMES)
    total_non = sum(statistics.mean(contrib_by_group["No"][n]) for n in FEATURE_NAMES)
    print(f"\nTotal mean essay-level contribution sum: ELL={total_ell:+.4f}  non-ELL={total_non:+.4f}  "
          f"gap={total_ell - total_non:+.4f}")


if __name__ == "__main__":
    main()
