"""
Round 6, follow-up to the nonlinear-classifier experiment: the AI-generator
diversification (OpenAI/Anthropic) and the nonlinear-classifier swap
(RandomForest/GradientBoosting) were both tested and declined — diversity
didn't help either model's cross-genre number, and nonlinear models would
require reworking evidence.py's coefficient-based explanation anyway. This
is the one remaining lever that changes nothing structural: sweep
LogisticRegression's C (inverse regularization strength) on the SAME
Gemini-only feature set the shipped model trains on, keeping the model
linear (evidence.py's `coefficient x standardized_value` math stays valid
unchanged) and checking whether the current default (C=1.0, sklearn's
default) is actually the best choice or just never tuned.

Same essay-level seen/unseen split as train_classifier.py, same DAIGT
cross-genre check, so numbers are directly comparable to
docs/EVALUATION.md and experiment_nonlinear_classifier.py.

Usage:
    cd backend && .venv/bin/python scripts/experiment_tune_regularization.py
"""
from __future__ import annotations

import csv
import random
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent.parent))
from app.pipeline import featurize as featurize_mod  # noqa: E402
from app.pipeline.featurize import FEATURE_NAMES  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = REPO_ROOT / "data" / "processed"

HELD_OUT_THEMES = {"challenging_belief", "gratitude", "obstacle_setback"}
EXCLUDED_AI_ID_PREFIXES = ("openai-", "anthropic-")  # match train_classifier.py
DAIGT_N_PER_CLASS = 100
DAIGT_SEED = 42
C_VALUES = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0]


def load_feature_table() -> list[dict]:
    with (PROCESSED_DIR / "feature_table_phase2.csv").open() as f:
        rows = list(csv.DictReader(f))
    return [r for r in rows if not r["essay_id"].startswith(EXCLUDED_AI_ID_PREFIXES)]


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

    X_train, y_train = to_xy(train_rows)
    X_test, y_test = to_xy(test_rows)
    X_unseen, y_unseen = to_xy(unseen_rows)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_unseen_scaled = scaler.transform(X_unseen)

    print(f"{len(train_rows)} train sentences, {len(test_rows)} in-theme test sentences, "
          f"{len(unseen_rows)} unseen-theme sentences (Gemini-only AI, matches shipped model)\n")

    print("Featurizing DAIGT cross-genre sample (shared across all C values)...")
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

    print(f"{'C':>8s} {'In-theme':>10s} {'Unseen-theme':>14s} {'DAIGT cross-genre':>18s}")
    print("-" * 55)
    for c in C_VALUES:
        clf = LogisticRegression(C=c, class_weight="balanced", max_iter=1000)
        clf.fit(X_train_scaled, y_train)
        in_theme_acc = accuracy_score(y_test, clf.predict(X_test_scaled))
        unseen_acc = accuracy_score(y_unseen, clf.predict(X_unseen_scaled))
        daigt_acc = accuracy_score(y_daigt, clf.predict(X_daigt_scaled))
        marker = "  <- current default" if c == 1.0 else ""
        print(f"{c:>8.2f} {in_theme_acc:>10.3f} {unseen_acc:>14.3f} {daigt_acc:>18.3f}{marker}")


if __name__ == "__main__":
    main()
