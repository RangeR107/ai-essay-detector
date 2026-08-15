"""
Round 7 (cross-genre lever): mix a genuinely diverse genre/generator slice
of DAIGT-v2 into TRAINING data, not just eval. Every prior lever for
cross-genre generalization (AI-generator diversity within the admissions
genre, the GradientBoosting swap) never actually exposed the classifier
to a different genre or to DAIGT's own generators — this is the first one
that does.

Two correctness safeguards, both load-bearing for the honesty of this
project's own numbers:

1. score_daigt.py's cross-genre eval sample (100 human + 100 ai,
   random.Random(42), sampled in that exact call order) must stay
   genuinely held-out. This script reproduces that exact sampling first
   and excludes those specific essay ids from anything it writes. The
   "cross-genre" number's meaning shifts as a result — from "never seen
   this genre or these generators at all" to "never seen these specific
   essays, but has seen the genre/generators broadly during training" —
   a real, disclosed methodology change, not a hidden regression in
   rigor. See docs/EVALUATION.md's round 7 section.

2. DAIGT-v2's "human" class (source == "persuade_corpus") IS the
   PERSUADE 2.0 corpus — the same corpus score_fairness.py's
   persuade_eval.csv (300 essays) is sampled from. Any DAIGT human essay
   whose text matches a persuade_eval.csv essay is excluded from the
   training slice by text-match (no shared id column between the two
   CSVs to join on directly), so the fairness eval doesn't end up
   partially trained-on — that would silently invalidate the headline
   fairness numbers in docs/EVALUATION.md §5.

AI side stratified by generator (the `source` column) and capped per
generator, so one huge generator (mistral7binstruct_v2 has 2,421 essays
in the full pool) doesn't dominate what "diverse" means here — the
point is generator diversity, not volume from one model.

Usage:
    cd backend && .venv/bin/python scripts/build_daigt_training_slice.py
"""
from __future__ import annotations

import csv
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

REPO_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = REPO_ROOT / "data" / "processed"

EVAL_N_PER_CLASS = 100  # must match score_daigt.py exactly
EVAL_SEED = 42  # must match score_daigt.py exactly
TRAIN_SEED = 43  # distinct from the eval seed
AI_CAP_PER_GENERATOR = 35


def _normalize(text: str) -> str:
    return " ".join(text.split()).lower()[:300]


def main() -> None:
    with (PROCESSED_DIR / "daigt_eval.csv").open() as f:
        rows = list(csv.DictReader(f))
    human_rows = [r for r in rows if r["class"] == "human"]
    ai_rows = [r for r in rows if r["class"] == "ai"]

    # Reproduce score_daigt.py's exact eval sample so we know what to exclude.
    eval_rng = random.Random(EVAL_SEED)
    eval_human = eval_rng.sample(human_rows, EVAL_N_PER_CLASS)
    eval_ai = eval_rng.sample(ai_rows, EVAL_N_PER_CLASS)
    eval_ids = {r["id"] for r in eval_human} | {r["id"] for r in eval_ai}
    print(f"Reproduced score_daigt.py's held-out eval sample: {len(eval_ids)} essays excluded from training.")

    with (PROCESSED_DIR / "persuade_eval.csv").open() as f:
        persuade_eval_texts = {_normalize(r["text"]) for r in csv.DictReader(f)}
    print(f"Loaded {len(persuade_eval_texts)} PERSUADE fairness-eval texts to exclude.")

    candidate_human = [
        r for r in human_rows
        if r["id"] not in eval_ids and _normalize(r["text"]) not in persuade_eval_texts
    ]
    candidate_ai = [r for r in ai_rows if r["id"] not in eval_ids]
    n_excluded_overlap = len(human_rows) - len(eval_human) - len(candidate_human)
    print(f"  -> {n_excluded_overlap} additional DAIGT-human essays excluded for PERSUADE-fairness-eval overlap.")

    by_generator: dict[str, list[dict]] = {}
    for r in candidate_ai:
        by_generator.setdefault(r["source"], []).append(r)

    train_rng = random.Random(TRAIN_SEED)
    ai_slice = []
    for gen, gen_rows in sorted(by_generator.items()):
        take = min(AI_CAP_PER_GENERATOR, len(gen_rows))
        ai_slice.extend(train_rng.sample(gen_rows, take))
    print(f"AI training slice: {len(ai_slice)} essays across {len(by_generator)} generators (cap {AI_CAP_PER_GENERATOR}/generator).")

    human_slice = train_rng.sample(candidate_human, min(len(ai_slice), len(candidate_human)))
    print(f"Human training slice: {len(human_slice)} essays (matched to AI slice size).")

    out_rows = []
    for r in human_slice:
        out_rows.append({
            "id": r["id"],
            "source": f"daigt-train:{r['source']}",
            "class": "human",
            "program_type": "persuasive-prompt",
            "theme_id": "cross_genre_daigt",
            "source_category": r["prompt_name"],
            "generator_model": "human",
            "word_count": r["word_count"],
            "text": " ".join(r["text"].split()),
        })
    for r in ai_slice:
        out_rows.append({
            "id": r["id"],
            "source": f"daigt-train:{r['source']}",
            "class": "ai",
            "program_type": "persuasive-prompt",
            "theme_id": "cross_genre_daigt",
            "source_category": r["prompt_name"],
            "generator_model": r["source"],
            "word_count": r["word_count"],
            "text": " ".join(r["text"].split()),
        })

    out_path = PROCESSED_DIR / "daigt_training_slice.csv"
    fieldnames = ["id", "source", "class", "program_type", "theme_id", "source_category", "generator_model", "word_count", "text"]
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)
    print(f"\nWrote {len(out_rows)} DAIGT training-slice essays to {out_path.relative_to(REPO_ROOT)}")
    print("AI by generator:", dict(Counter(r["source"] for r in ai_slice)))


if __name__ == "__main__":
    main()
