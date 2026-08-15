"""
Run the pipeline (segmentation + single GPT-2 pass -> perplexity/GLTR +
spaCy stylometry) over every training essay and write a per-sentence
feature table.

Reads the user's real AI essays (data/processed/ai_essays.csv, 500 essays
via Gemini API — see DOCUMENTS/IMPLEMENTATION.md Phase 1 addendum 4) rather
than the Phase 2 placeholder set
(backend/scripts/build_placeholder_ai_essays.py's ai_essays_placeholder.csv).
That switch is intentionally a one-line change in load_essays() below, not
automatic, so nobody accidentally trains on real+placeholder mixed
together without noticing.

500 human (huggingface:nid989/EssayFroum-Dataset) + 500 AI (Gemini) is
enough depth per theme that the plan's §3c theme-held-out split is now
worth doing properly — not yet wired into train_classifier.py as of this
script's last edit; check that script's own docstring for current status.

Usage:
    cd backend && .venv/bin/python scripts/extract_features.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.pipeline import featurize  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = REPO_ROOT / "data" / "processed"


def load_essays() -> list[dict]:
    # Round 7: daigt_training_slice.csv (build_daigt_training_slice.py) adds
    # a genre/generator-diverse DAIGT slice to training — see that script's
    # docstring for the two correctness safeguards (score_daigt.py eval
    # isolation, persuade_eval.csv overlap exclusion) that make this safe.
    essays = []
    for fname in ["human_essays.csv", "ai_essays.csv", "daigt_training_slice.csv"]:
        path = PROCESSED_DIR / fname
        if not path.exists():
            continue
        with path.open() as f:
            essays.extend(csv.DictReader(f))
    return essays


def main() -> None:
    essays = load_essays()
    print(f"Loaded {len(essays)} essays ({sum(1 for e in essays if e['class']=='human')} human, "
          f"{sum(1 for e in essays if e['class']=='ai')} ai)")

    rows = []
    for i, essay in enumerate(essays):
        print(f"  [{i+1}/{len(essays)}] {essay['id']} ({len(essay['text'].split())} words)")
        sentence_feats = featurize.featurize_essay(essay["text"])
        for j, sf in enumerate(sentence_feats):
            row = {
                "essay_id": essay["id"],
                "sentence_idx": j,
                "class": essay["class"],
                "source": essay["source"],
                "theme_id": essay["theme_id"],
            }
            row.update({name: sf.features[name] for name in featurize.FEATURE_NAMES})
            rows.append(row)

    out_path = PROCESSED_DIR / "feature_table_phase2.csv"
    fieldnames = ["essay_id", "sentence_idx", "class", "source", "theme_id"] + featurize.FEATURE_NAMES
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {len(rows)} sentence-level rows to {out_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
