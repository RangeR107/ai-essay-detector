"""
Phase 6 — build the 100 hybrid essays for passage-localization testing
(plan §3b): "take real EssayForum essays, splice in an AI-generated
paragraph or continuation, and record the exact sentence-offset span
that's AI-written as ground truth."

Construction used here: take a real human essay, truncate after K
sentences, then append an AI essay's opening M sentences as a
continuation. Ground truth is the exact character span covering the
appended AI text. This is the "continuation" variant the plan explicitly
allows ("an AI rewrite OR continuation") — simpler to build correctly
than mid-essay paragraph replacement, which requires re-stitching text
around a removed middle section without disturbing sentence boundaries.

Sentence segmentation reuses app.pipeline.segment (the same spaCy
splitting the served app uses) — no GPT-2 pass needed here, this script
only manipulates text, it doesn't score anything.

Output: data/processed/hybrid_essays.csv

Usage:
    cd backend && .venv/bin/python scripts/build_hybrid_essays.py
"""
from __future__ import annotations

import csv
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from app.pipeline import segment  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = REPO_ROOT / "data" / "processed"

N_HYBRID = 100
MIN_HUMAN_SENTENCES = 3
MIN_AI_SENTENCES = 2
MAX_AI_SENTENCES = 6
HUMAN_KEEP_FRACTION = 0.7  # keep at most this fraction of the human essay
SEED = 42


def load_essays(fname: str) -> list[dict]:
    with (PROCESSED_DIR / fname).open() as f:
        return list(csv.DictReader(f))


def build_hybrid(human_text: str, ai_text: str, rng: random.Random) -> dict | None:
    human_sents = segment.segment(human_text)
    ai_sents = segment.segment(ai_text)
    if len(human_sents) < MIN_HUMAN_SENTENCES + 1 or len(ai_sents) < MIN_AI_SENTENCES:
        return None

    max_k = max(MIN_HUMAN_SENTENCES, int(len(human_sents) * HUMAN_KEEP_FRACTION))
    k = rng.randint(MIN_HUMAN_SENTENCES, max_k)
    human_prefix = human_text[:human_sents[k - 1].end].rstrip()

    m = rng.randint(MIN_AI_SENTENCES, min(MAX_AI_SENTENCES, len(ai_sents)))
    ai_suffix = ai_text[ai_sents[0].start:ai_sents[m - 1].end].strip()

    combined = human_prefix + " " + ai_suffix
    ai_span_start = len(human_prefix) + 1
    ai_span_end = ai_span_start + len(ai_suffix)

    return {
        "text": combined,
        "ai_span_start": ai_span_start,
        "ai_span_end": ai_span_end,
        "n_human_sentences": k,
        "n_ai_sentences": m,
    }


def main() -> None:
    human_essays = load_essays("human_essays.csv")
    ai_essays = load_essays("ai_essays.csv")
    # Round 6: match the shipped classifier's Gemini-only training data
    # (see train_classifier.py's EXCLUDED_AI_ID_PREFIXES) so this eval
    # measures the model actually served, not a hypothetical one.
    ai_essays = [r for r in ai_essays if not r["id"].startswith(("openai-", "anthropic-"))]
    print(f"Pool: {len(human_essays)} human essays, {len(ai_essays)} AI essays")

    rng = random.Random(SEED)
    human_sample = rng.sample(human_essays, N_HYBRID)
    ai_sample = rng.sample(ai_essays, N_HYBRID)

    rows = []
    skipped = 0
    for i, (h, a) in enumerate(zip(human_sample, ai_sample)):
        built = build_hybrid(h["text"], a["text"], rng)
        if built is None:
            skipped += 1
            continue
        rows.append({
            "id": f"hybrid-{i:03d}",
            "text": built["text"],
            "ai_span_start": built["ai_span_start"],
            "ai_span_end": built["ai_span_end"],
            "human_essay_id": h["id"],
            "ai_essay_id": a["id"],
            "n_human_sentences": built["n_human_sentences"],
            "n_ai_sentences": built["n_ai_sentences"],
        })

    if skipped:
        print(f"Skipped {skipped} pair(s) — essay too short to splice.")

    out_path = PROCESSED_DIR / "hybrid_essays.csv"
    fieldnames = ["id", "text", "ai_span_start", "ai_span_end", "human_essay_id",
                  "ai_essay_id", "n_human_sentences", "n_ai_sentences"]
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} hybrid essays to {out_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
