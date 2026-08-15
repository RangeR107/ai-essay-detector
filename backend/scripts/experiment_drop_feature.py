"""
Experiment, NOT wired into the production pipeline: investigate_fairness_bias.py
found gltr_pct_top10 is the dominant driver of the ELL/non-ELL fairness
gap (mean contribution delta +0.50, ~3x the next-biggest feature). This
script tests the direct hypothesis — retrain a variant classifier
excluding that one feature, using the exact same data/methodology as
train_classifier.py, and compare:
  1. Core accuracy (in-theme / unseen-theme) — did removing it cost us
     real signal?
  2. The PERSUADE ELL/non-ELL broad-FPR gap — did removing it help?

Reuses feature_table_phase2.csv (no new GPT-2/spaCy pass for training)
but needs a fresh pass over the 300 PERSUADE essays to re-measure fairness
with the variant model (persuade_eval.csv doesn't have cached features).

This is a standalone experiment. It does NOT overwrite classifier.joblib
— see the script's output / docs/IMPLEMENTATION.md for the adoption
decision after seeing the numbers.

Usage:
    cd backend && .venv/bin/python scripts/experiment_drop_feature.py
"""
from __future__ import annotations

import csv
import statistics
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent.parent))
from app.pipeline import featurize as featurize_mod  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = REPO_ROOT / "data" / "processed"

DROP_FEATURE = "gltr_pct_top10"
VARIANT_FEATURES = [f for f in featurize_mod.FEATURE_NAMES if f != DROP_FEATURE]
HELD_OUT_THEMES = {"background_identity", "captivating_topic", "gratitude"}


def load_feature_table() -> list[dict]:
    with (PROCESSED_DIR / "feature_table_phase2.csv").open() as f:
        return list(csv.DictReader(f))


def to_xy(rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    X = np.array([[float(r[name]) for name in VARIANT_FEATURES] for r in rows])
    y = np.array([1 if r["class"] == "ai" else 0 for r in rows])
    return X, y


def main() -> None:
    print(f"Variant feature set: {len(VARIANT_FEATURES)} features (dropped '{DROP_FEATURE}')\n")
    rows = load_feature_table()

    essays = {}
    for r in rows:
        essays.setdefault(r["essay_id"], {"theme_id": r["theme_id"], "class": r["class"]})
    essay_ids = list(essays)
    essay_themes = [essays[e]["theme_id"] for e in essay_ids]

    seen_essays = [e for e, t in zip(essay_ids, essay_themes) if t not in HELD_OUT_THEMES]
    unseen_essays = [e for e, t in zip(essay_ids, essay_themes) if t in HELD_OUT_THEMES]
    seen_classes = [essays[e]["class"] for e in seen_essays]

    train_essays, test_essays = train_test_split(seen_essays, test_size=0.2, random_state=42, stratify=seen_classes)
    train_essays, test_essays, unseen_essays_set = set(train_essays), set(test_essays), set(unseen_essays)

    train_rows = [r for r in rows if r["essay_id"] in train_essays]
    test_rows = [r for r in rows if r["essay_id"] in test_essays]
    unseen_rows = [r for r in rows if r["essay_id"] in unseen_essays_set]

    X_train, y_train = to_xy(train_rows)
    X_test, y_test = to_xy(test_rows)
    X_unseen, y_unseen = to_xy(unseen_rows)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    clf = LogisticRegression(class_weight="balanced", max_iter=1000)
    clf.fit(X_train_scaled, y_train)

    in_theme_acc = accuracy_score(y_test, clf.predict(scaler.transform(X_test)))
    unseen_theme_acc = accuracy_score(y_unseen, clf.predict(scaler.transform(X_unseen)))
    print("=== 1. Core accuracy: variant (no gltr_pct_top10) vs original ===")
    print(f"In-theme:    variant={in_theme_acc:.3f}   original=0.736")
    print(f"Unseen-theme: variant={unseen_theme_acc:.3f}   original=0.713")
    print(classification_report(y_test, clf.predict(scaler.transform(X_test)), target_names=["human", "ai"]))

    # Final production-shaped model (all data) for the fairness re-check
    X_all, y_all = to_xy(rows)
    final_scaler = StandardScaler()
    X_all_scaled = final_scaler.fit_transform(X_all)
    final_clf = LogisticRegression(class_weight="balanced", max_iter=1000)
    final_clf.fit(X_all_scaled, y_all)

    print("\n=== 2. Re-measuring PERSUADE ELL/non-ELL fairness gap with variant model ===")
    with (PROCESSED_DIR / "persuade_eval.csv").open() as f:
        persuade_rows = list(csv.DictReader(f))

    def variant_predict(features: dict[str, float]) -> float:
        x = np.array([[features[name] for name in VARIANT_FEATURES]])
        x_scaled = final_scaler.transform(x)
        return float(final_clf.predict_proba(x_scaled)[0, 1])

    # Same essay-verdict thresholds as production for a fair comparison.
    import json
    thresholds_path = REPO_ROOT / "backend" / "app" / "models" / "thresholds.json"
    thresholds = json.loads(thresholds_path.read_text()) if thresholds_path.exists() else {"likely_human_max": 0.35, "likely_ai_min": 0.65}

    def verdict_label(mean_score: float) -> str:
        if mean_score <= thresholds["likely_human_max"]:
            return "Likely Human"
        if mean_score >= thresholds["likely_ai_min"]:
            return "Likely AI"
        return "Inconclusive"

    fpr_by_group = {"Yes": [], "No": []}
    for i, r in enumerate(persuade_rows):
        status = r["ell_status"].strip()
        if status not in fpr_by_group:
            continue
        sentence_feats = featurize_mod.featurize_essay(r["text"])
        use = sentence_feats
        if not use:
            continue
        scores = [variant_predict(sf.features) for sf in use]
        label = verdict_label(sum(scores) / len(scores))
        fpr_by_group[status].append(label)
        if (i + 1) % 60 == 0:
            print(f"  [{i+1}/{len(persuade_rows)}]")

    for group, labels in fpr_by_group.items():
        broad = sum(1 for l in labels if l != "Likely Human") / len(labels) if labels else 0.0
        strict = sum(1 for l in labels if l == "Likely AI") / len(labels) if labels else 0.0
        print(f"  ell_status={group:4s} n={len(labels):3d}  broad_fpr={broad:.3f}  strict_fpr={strict:.3f}")

    no_broad = sum(1 for l in fpr_by_group["No"] if l != "Likely Human") / len(fpr_by_group["No"])
    yes_broad = sum(1 for l in fpr_by_group["Yes"] if l != "Likely Human") / len(fpr_by_group["Yes"])
    print(f"\nGap: variant={yes_broad - no_broad:+.3f}  (original model gap was +0.147: 0.367 - 0.220)")


if __name__ == "__main__":
    main()
