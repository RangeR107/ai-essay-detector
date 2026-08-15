"""
Phase 1 (optional per plan §3b) — download DAIGT-v2 for the cross-genre
evaluation check in Phase 7.

EVAL-ONLY. This script writes to data/processed/daigt_eval.csv and NOTHING
else in the pipeline reads that file — build_dataset.py, ingest_ai_essays.py,
extract_features.py, and train_classifier.py are all wired to
human_essays.csv + ai_essays.csv only. DAIGT never touches training; it
only gets read later by a Phase 7 evaluation script that loads the
already-trained classifier.joblib and scores it against this file.

Source: Yunij/kaggle-comp-daigt on Hugging Face — a mirror of the
thedrcat/daigt-v2-train-dataset Kaggle competition dataset. Using the HF
mirror sidesteps the Kaggle-auth blocker (same approach as
hf_essayforum.py). No license tag on the HF repo; the dataset is mostly
PERSUADE-corpus human text (an openly-published academic corpus) plus
synthetic AI generations (no personal-privacy dimension), a materially
lower-risk composition than the personal-essay sources in this project —
still treated as local-only/gitignored for consistency, not because it
specifically needs that protection.

Usage:
    cd backend && .venv/bin/python scripts/download_daigt.py
"""
from __future__ import annotations

import csv
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = REPO_ROOT / "data" / "processed"

HF_DATASET_ID = "Yunij/kaggle-comp-daigt"


def main() -> None:
    from datasets import load_dataset

    print(f"Downloading {HF_DATASET_ID}...")
    ds = load_dataset(HF_DATASET_ID)["train"]

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / "daigt_eval.csv"
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "class", "prompt_name", "source", "word_count", "text"])
        writer.writeheader()
        for i, row in enumerate(ds):
            text = " ".join(row["cleaned_text"].split())
            writer.writerow({
                "id": f"daigt-{i}",
                "class": "ai" if row["label"] == 1 else "human",
                "prompt_name": row["prompt_name"],
                "source": row["source"],
                "word_count": len(text.split()),
                "text": text,
            })

    print(f"Wrote {len(ds)} rows to {out_path.relative_to(REPO_ROOT)}")
    from collections import Counter
    labels = Counter("ai" if r["label"] == 1 else "human" for r in ds)
    sources = Counter(r["source"] for r in ds)
    print("By class:", dict(labels))
    print("By source (top 10):", dict(sources.most_common(10)))


if __name__ == "__main__":
    main()
