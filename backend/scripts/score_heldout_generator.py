"""
The precise cross-generator check DAIGT-v2 can't give on its own: DAIGT
conflates genre (persuasive, not admissions) and generator (9+ non-Gemini
models) into a single number, so a low DAIGT score doesn't say whether
it's the genre shift or the generator shift causing the miss.

This script scores `data/processed/ai_essays_heldout_generator.csv` — 20
OpenAI + 20 Anthropic admissions essays, same genre and prompt set as
training, held out entirely from `ingest_ai_essays.py` specifically so
they were never seen during training (see that script's docstring).
Genre and prompt distribution held constant; only the generator changes.
**All essays here are real AI text** (ai_essays_heldout_generator.csv
carries no human class), so this measures recall specifically — of AI
essays from a generator never trained on, how many get correctly flagged
as AI — not precision (that's covered by the human-only PERSUADE/ELLIPSE
fairness checks elsewhere, `score_fairness.py`).

Usage:
    cd backend && .venv/bin/python scripts/score_heldout_generator.py
"""
from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from app.pipeline import aggregate, classify, featurize  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = REPO_ROOT / "data" / "processed"

SENTENCE_THRESHOLD = 0.5


def main() -> None:
    path = PROCESSED_DIR / "ai_essays_heldout_generator.csv"
    if not path.exists():
        print(f"{path.relative_to(REPO_ROOT)} not found — run ingest_ai_essays.py first.")
        return
    with path.open() as f:
        rows = list(csv.DictReader(f))
    print(f"Scoring {len(rows)} held-out-generator AI essays (never seen in training)...")

    by_generator: dict[str, dict[str, int]] = defaultdict(lambda: {"sent_total": 0, "sent_flagged": 0, "essay_total": 0, "essay_flagged": 0})

    for i, r in enumerate(rows):
        gen = r["generator_model"]
        sentence_feats = featurize.featurize_essay(r["text"])
        if not sentence_feats:
            continue
        scores = [classify.predict_proba(sf.features) for sf in sentence_feats]

        by_generator[gen]["sent_total"] += len(scores)
        by_generator[gen]["sent_flagged"] += sum(1 for s in scores if s >= SENTENCE_THRESHOLD)

        verdict = aggregate.essay_verdict(scores)
        by_generator[gen]["essay_total"] += 1
        by_generator[gen]["essay_flagged"] += int(verdict.label == "Likely AI")

        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(rows)}]")

    print(f"\n=== Held-out-generator recall, sentence + essay level, {len(rows)} essays ===")
    total_sent_total = total_sent_flagged = total_essay_total = total_essay_flagged = 0
    for gen, counts in by_generator.items():
        sent_recall = counts["sent_flagged"] / counts["sent_total"] if counts["sent_total"] else 0.0
        essay_recall = counts["essay_flagged"] / counts["essay_total"] if counts["essay_total"] else 0.0
        print(f"  {gen:35s} sentence recall={sent_recall:.3f} ({counts['sent_flagged']}/{counts['sent_total']})  "
              f"essay-level Likely-AI recall={essay_recall:.3f} ({counts['essay_flagged']}/{counts['essay_total']})")
        total_sent_total += counts["sent_total"]
        total_sent_flagged += counts["sent_flagged"]
        total_essay_total += counts["essay_total"]
        total_essay_flagged += counts["essay_flagged"]

    overall_sent_recall = total_sent_flagged / total_sent_total if total_sent_total else 0.0
    overall_essay_recall = total_essay_flagged / total_essay_total if total_essay_total else 0.0
    print(f"\n  {'OVERALL':35s} sentence recall={overall_sent_recall:.3f} ({total_sent_flagged}/{total_sent_total})  "
          f"essay-level Likely-AI recall={overall_essay_recall:.3f} ({total_essay_flagged}/{total_essay_total})")
    print("\n(Essay-level uses the calibrated 'Likely AI' threshold, so an essay landing in\n"
          "'Inconclusive' counts as a miss here, same standard as everywhere else in this project.)")


if __name__ == "__main__":
    main()
