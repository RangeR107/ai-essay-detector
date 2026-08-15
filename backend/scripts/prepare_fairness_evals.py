"""
Phase 7 (unblocked) — sample PERSUADE and ELLIPSE into eval-only CSVs for
the fairness/ESL-false-positive check (plan §3b/§8) now that Kaggle auth
is set up. Reads the raw downloads from download_eval_datasets.py
(data/raw/persuade/, data/raw/ellipse/).

EVAL-ONLY, same isolation as daigt_eval.csv: nothing in the training
pipeline (build_dataset.py, ingest_ai_essays.py, extract_features.py,
train_classifier.py) reads these files. Both source datasets are
human-only (PERSUADE 2.0's "human_scores" file; ELLIPSE is entirely
ESL-student writing) — there is no AI class here, these exist purely to
measure false-positive rate on real human writing outside the training
distribution.

PERSUADE sampling: stratified by ell_status specifically, not a plain
random sample — plan §8 explicitly wants "PERSUADE essays broken out by
ELL status," and ELL='Yes' is only ~8.6% of the raw pool (2,244/25,996),
so a plain random 200-essay sample would carry too few ELL essays (~17)
to say anything meaningful about that specific breakout. Takes up to 150
from each of ell_status in {Yes, No} instead.

Output: data/processed/persuade_eval.csv, data/processed/ellipse_eval.csv

Usage:
    cd backend && .venv/bin/python scripts/prepare_fairness_evals.py
"""
from __future__ import annotations

import csv
import random
from pathlib import Path

csv.field_size_limit(10_000_000)

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "data" / "raw"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"

PERSUADE_PATH = RAW_DIR / "persuade" / "persuade_2.0_human_scores_demo_id_github.csv"
ELLIPSE_PATH = RAW_DIR / "ellipse" / "train.csv"

N_PER_ELL_GROUP = 150
N_ELLIPSE = 200
MIN_WORDS = 150
SEED = 42


def prepare_persuade() -> None:
    with PERSUADE_PATH.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    rows = [r for r in rows if len(r["full_text"].split()) >= MIN_WORDS]

    by_ell = {"Yes": [], "No": []}
    for r in rows:
        status = r["ell_status"].strip()
        if status in by_ell:
            by_ell[status].append(r)

    rng = random.Random(SEED)
    sample = []
    for status, pool in by_ell.items():
        take = min(N_PER_ELL_GROUP, len(pool))
        sample.extend(rng.sample(pool, take))
    rng.shuffle(sample)

    out_path = PROCESSED_DIR / "persuade_eval.csv"
    fieldnames = ["id", "text", "word_count", "ell_status", "race_ethnicity",
                  "economically_disadvantaged", "student_disability_status", "prompt_name"]
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in sample:
            text = " ".join(r["full_text"].split())
            writer.writerow({
                "id": f"persuade-{r['essay_id_comp']}",
                "text": text,
                "word_count": len(text.split()),
                "ell_status": r["ell_status"].strip(),
                "race_ethnicity": r["race_ethnicity"].strip(),
                "economically_disadvantaged": r["economically_disadvantaged"].strip(),
                "student_disability_status": r["student_disability_status"].strip(),
                "prompt_name": r["prompt_name"],
            })
    print(f"PERSUADE: wrote {len(sample)} essays to {out_path.relative_to(REPO_ROOT)} "
          f"({sum(1 for r in sample if r['ell_status'].strip()=='Yes')} ELL, "
          f"{sum(1 for r in sample if r['ell_status'].strip()=='No')} non-ELL)")


def prepare_ellipse() -> None:
    with ELLIPSE_PATH.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    rows = [r for r in rows if len(r["full_text"].split()) >= MIN_WORDS]

    rng = random.Random(SEED)
    sample = rng.sample(rows, min(N_ELLIPSE, len(rows)))

    out_path = PROCESSED_DIR / "ellipse_eval.csv"
    fieldnames = ["id", "text", "word_count", "grade", "prompt"]
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in sample:
            text = " ".join(r["full_text"].split())
            writer.writerow({
                "id": f"ellipse-{r['text_id']}",
                "text": text,
                "word_count": len(text.split()),
                "grade": r["grade"],
                "prompt": r["prompt"],
            })
    print(f"ELLIPSE: wrote {len(sample)} essays to {out_path.relative_to(REPO_ROOT)}")


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    prepare_persuade()
    prepare_ellipse()


if __name__ == "__main__":
    main()
