"""
Round 9 (accuracy lever): GradientBoostingClassifier has been running on
its round-6 defaults (n_estimators=200, max_depth=3, learning_rate=0.1
sklearn default) ever since it was adopted — never actually tuned, the
same way LogisticRegression's C was never tuned before
experiment_tune_regularization.py checked it (round 6, found flat).
This sweeps max_depth and learning_rate (n_estimators held at 200) on
the CURRENT full training set (2,246 essays: admissions + DAIGT +
OpenAI/Anthropic, matching train_classifier.py's round-8 state exactly)
and reports admissions-only in-theme/unseen-theme accuracy plus DAIGT
cross-genre accuracy for each combination, so a real winner (if one
exists) is visible on all the metrics that matter, not just one.

Usage:
    cd backend && .venv/bin/python scripts/experiment_tune_gbm_hyperparams.py
"""
from __future__ import annotations

import csv
import random
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

sys.path.insert(0, str(Path(__file__).parent.parent))
from app.pipeline import featurize as featurize_mod  # noqa: E402
from app.pipeline.featurize import FEATURE_NAMES  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = REPO_ROOT / "data" / "processed"

HELD_OUT_THEMES = {"challenging_belief", "gratitude", "obstacle_setback"}
DAIGT_N_PER_CLASS = 100
DAIGT_SEED = 42

GRID = [
    # (n_estimators, max_depth, learning_rate)
    (200, 3, 0.1),   # current production
    (200, 4, 0.1),
    (200, 5, 0.1),
    (300, 3, 0.1),
    (200, 3, 0.05),
    (200, 3, 0.2),
    (300, 4, 0.05),
]


def load_feature_table() -> list[dict]:
    with (PROCESSED_DIR / "feature_table_phase2.csv").open() as f:
        return list(csv.DictReader(f))


def to_xy(rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    X = np.array([[float(r[name]) for name in FEATURE_NAMES] for r in rows])
    y = np.array([1 if r["class"] == "ai" else 0 for r in rows])
    return X, y


def main() -> None:
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
    admissions_test_rows = [r for r in test_rows if r["theme_id"] != "cross_genre_daigt"]

    X_train, y_train = to_xy(train_rows)
    X_test, y_test = to_xy(admissions_test_rows)
    X_unseen, y_unseen = to_xy(unseen_rows)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_unseen_scaled = scaler.transform(X_unseen)
    train_weights = compute_sample_weight("balanced", y_train)

    print(f"{len(train_rows)} train sentences, {len(admissions_test_rows)} admissions-only in-theme test sentences, "
          f"{len(unseen_rows)} unseen-theme sentences\n")

    print("Featurizing DAIGT cross-genre sample (shared across all configs)...")
    with (PROCESSED_DIR / "daigt_eval.csv").open() as f:
        daigt_rows = list(csv.DictReader(f))
    human_rows = [r for r in daigt_rows if r["class"] == "human"]
    ai_rows = [r for r in daigt_rows if r["class"] == "ai"]
    rng = random.Random(DAIGT_SEED)
    daigt_sample = rng.sample(human_rows, DAIGT_N_PER_CLASS) + rng.sample(ai_rows, DAIGT_N_PER_CLASS)
    rng.shuffle(daigt_sample)

    daigt_sentence_rows = []
    for i, r in enumerate(daigt_sample):
        true_ai = r["class"] == "ai"
        for sf in featurize_mod.featurize_essay(r["text"]):
            daigt_sentence_rows.append((true_ai, sf.features))
        if (i + 1) % 40 == 0:
            print(f"  [{i+1}/{len(daigt_sample)}]")
    X_daigt = np.array([[sf[name] for name in FEATURE_NAMES] for _, sf in daigt_sentence_rows])
    y_daigt = np.array([1 if true_ai else 0 for true_ai, _ in daigt_sentence_rows])
    X_daigt_scaled = scaler.transform(X_daigt)
    print(f"{len(daigt_sentence_rows)} DAIGT sentences featurized.\n")

    print(f"{'n_est':>6s} {'depth':>6s} {'lr':>6s} {'Admissions in-theme':>20s} {'Unseen-theme':>14s} {'DAIGT':>8s}")
    print("-" * 70)
    for n_est, depth, lr in GRID:
        weights = train_weights
        clf = GradientBoostingClassifier(n_estimators=n_est, max_depth=depth, learning_rate=lr, random_state=42)
        clf.fit(X_train_scaled, y_train, sample_weight=weights)
        adm_acc = accuracy_score(y_test, clf.predict(X_test_scaled))
        unseen_acc = accuracy_score(y_unseen, clf.predict(X_unseen_scaled))
        daigt_acc = accuracy_score(y_daigt, clf.predict(X_daigt_scaled))
        marker = "  <- current production" if (n_est, depth, lr) == (200, 3, 0.1) else ""
        print(f"{n_est:>6d} {depth:>6d} {lr:>6.2f} {adm_acc:>20.3f} {unseen_acc:>14.3f} {daigt_acc:>8.3f}{marker}")


if __name__ == "__main__":
    main()
