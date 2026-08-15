"""
Experiment, NOT wired into the production pipeline: tests whether a
nonlinear classifier finds usable signal in feature *interactions* that
the current `LogisticRegression` (linear, by design, so its per-sentence
evidence is a simple `coefficient x standardized_value` decomposition)
structurally cannot represent — e.g. "high perplexity is only suspicious
when vocabulary variety is also low" is a pattern a linear model can't
express no matter how much data it gets.

Same features, same essay-level train/test/unseen split methodology as
train_classifier.py (so numbers are directly comparable), same
stratified DAIGT-v2 cross-genre sample as score_daigt.py (same seed).
Trains RandomForestClassifier and GradientBoostingClassifier alongside
the existing production LogisticRegression and reports all three
side by side, at both the sentence-level in-theme/unseen-theme splits
and the DAIGT cross-genre check.

This is a standalone comparison. It does NOT overwrite classifier.joblib
— see the script's output for whether either nonlinear model actually
won, and by how much, before considering adoption. Note: if a nonlinear
model is ever adopted, the evidence.py `coefficient x standardized_value`
per-sentence explanation would need to change to a different (but still
fully deterministic) attribution method — e.g. permutation or
tree-based feature importance — since it's linear-model-specific.

Usage:
    cd backend && .venv/bin/python scripts/experiment_nonlinear_classifier.py
"""
from __future__ import annotations

import csv
import random
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
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


EXCLUDED_AI_ID_PREFIXES = ("openai-", "anthropic-")  # match train_classifier.py's shipped model


def load_feature_table() -> list[dict]:
    with (PROCESSED_DIR / "feature_table_phase2.csv").open() as f:
        rows = list(csv.DictReader(f))
    return [r for r in rows if not r["essay_id"].startswith(EXCLUDED_AI_ID_PREFIXES)]


def to_xy(rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    X = np.array([[float(r[name]) for name in FEATURE_NAMES] for r in rows])
    y = np.array([1 if r["class"] == "ai" else 0 for r in rows])
    return X, y


def build_models() -> dict:
    return {
        "LogisticRegression (current production)": LogisticRegression(class_weight="balanced", max_iter=1000),
        "RandomForest": RandomForestClassifier(n_estimators=300, max_depth=8, class_weight="balanced", random_state=42),
        "GradientBoosting": GradientBoostingClassifier(n_estimators=200, max_depth=3, random_state=42),
    }


def fit(name: str, model, X_train, y_train):
    if name == "GradientBoosting":
        # GradientBoostingClassifier has no class_weight param; emulate
        # class_weight='balanced' with sample weights instead.
        weights = compute_sample_weight("balanced", y_train)
        model.fit(X_train, y_train, sample_weight=weights)
    else:
        model.fit(X_train, y_train)
    return model


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

    X_train, y_train = to_xy(train_rows)
    X_test, y_test = to_xy(test_rows)
    X_unseen, y_unseen = to_xy(unseen_rows)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_unseen_scaled = scaler.transform(X_unseen)

    print(f"{len(train_rows)} train sentences, {len(test_rows)} in-theme test sentences, "
          f"{len(unseen_rows)} unseen-theme sentences\n")

    # --- Compute DAIGT sentence features ONCE, shared across all 3 models ---
    print("Featurizing DAIGT cross-genre sample (shared across all models)...")
    with (PROCESSED_DIR / "daigt_eval.csv").open() as f:
        daigt_rows = list(csv.DictReader(f))
    human_rows = [r for r in daigt_rows if r["class"] == "human"]
    ai_rows = [r for r in daigt_rows if r["class"] == "ai"]
    rng = random.Random(DAIGT_SEED)
    daigt_sample = rng.sample(human_rows, DAIGT_N_PER_CLASS) + rng.sample(ai_rows, DAIGT_N_PER_CLASS)
    rng.shuffle(daigt_sample)

    daigt_sentence_rows = []  # (true_ai, feature_dict)
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

    print(f"{'Model':45s} {'In-theme':>10s} {'Unseen-theme':>14s} {'DAIGT cross-genre':>18s}")
    print("-" * 90)
    for name, model in build_models().items():
        fit(name, model, X_train_scaled, y_train)
        in_theme_acc = accuracy_score(y_test, model.predict(X_test_scaled))
        unseen_acc = accuracy_score(y_unseen, model.predict(X_unseen_scaled))
        daigt_acc = accuracy_score(y_daigt, model.predict(X_daigt_scaled))
        print(f"{name:45s} {in_theme_acc:>10.3f} {unseen_acc:>14.3f} {daigt_acc:>18.3f}")


if __name__ == "__main__":
    main()
