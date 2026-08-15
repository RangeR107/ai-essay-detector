"""
Loads the trained classifier/scaler and turns feature vectors into P(AI).
This is the ONLY place a per-sentence AI-likelihood number is produced —
deterministically, from our own trained classifier, never from GPT-2 or
any hosted model (plan §0).

Round 6: swapped from LogisticRegression to GradientBoostingClassifier
(real, tested accuracy gain — see DOCUMENTS/IMPLEMENTATION.md). Since a
tree ensemble has no `.coef_`, `coefficients()` is gone; evidence.py now
computes per-feature contribution via single-feature perturbation instead
(see predict_proba_from_scaled/scaled_value below and evidence.py's
docstring) — still fully deterministic, still model-agnostic to whatever
classifier.joblib actually is.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np

from .featurize import FEATURE_NAMES

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"


@lru_cache(maxsize=1)
def _artifacts():
    clf = joblib.load(MODELS_DIR / "classifier.joblib")
    scaler = joblib.load(MODELS_DIR / "scaler.joblib")
    reference_stats = joblib.load(MODELS_DIR / "reference_stats.joblib")
    return clf, scaler, reference_stats


def predict_proba(features: dict[str, float]) -> float:
    clf, scaler, _ = _artifacts()
    x = np.array([[features[name] for name in FEATURE_NAMES]])
    x_scaled = scaler.transform(x)
    return float(clf.predict_proba(x_scaled)[0, 1])


def predict_proba_from_scaled(x_scaled_row: np.ndarray) -> float:
    """Same as predict_proba, but takes an already-standardized feature
    row — lets evidence.py score perturbed vectors without re-scaling."""
    clf, _, _ = _artifacts()
    return float(clf.predict_proba(x_scaled_row.reshape(1, -1))[0, 1])


def scale_vector(features: dict[str, float]) -> np.ndarray:
    """Standardized feature vector, in FEATURE_NAMES order — used by
    evidence.py to compute per-feature contributions."""
    _, scaler, _ = _artifacts()
    x = np.array([[features[name] for name in FEATURE_NAMES]])
    return scaler.transform(x)[0]


def scaled_value(name: str, raw_value: float) -> float:
    """Standardize a single named feature's raw value the same way
    scale_vector does — used by evidence.py to build a perturbed vector
    with one feature swapped to a reference value."""
    _, scaler, _ = _artifacts()
    i = FEATURE_NAMES.index(name)
    return float((raw_value - scaler.mean_[i]) / scaler.scale_[i])


def reference_stats() -> dict[str, np.ndarray]:
    _, _, stats = _artifacts()
    return stats
