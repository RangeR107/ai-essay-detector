"""
Second hybrid-essay construction method — mid-paragraph AI replacement,
the variant plan §3b describes first ("replace one or two paragraphs
with an AI rewrite or continuation") and this project's original
build_hybrid_essays.py deliberately deferred in favor of the simpler
continuation method. This is that deferred variant: a middle span of a
real human essay's sentences is removed and replaced with real AI
sentences, keeping genuine human text both before AND after the AI
splice (continuation-style only ever has human-then-AI, never
human-AI-human) — the "a human paragraph an AI later polished" case the
plan's brief calls the realistic one.

Construction: pick a contiguous span of 2-5 sentences starting somewhere
in the middle third of the essay (never touching the first 2 or last 2
sentences, so both real-human anchors survive), remove it, splice in 2-6
AI sentences in its place. Ground truth is the exact character span of
the inserted AI text, same as the continuation variant.

Output: data/processed/hybrid_essays_midparagraph.csv — kept SEPARATE
from hybrid_essays.csv (the continuation set) rather than merged, so
score_hybrid_essays.py's existing numbers stay comparable to their own
history; a combined view can be built later if wanted.

Usage:
    cd backend && .venv/bin/python scripts/build_hybrid_essays_midparagraph.py
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
MIN_ANCHOR_SENTENCES = 2  # untouched real-human sentences required at both start and end
MIN_REPLACED_SENTENCES = 2
MAX_REPLACED_SENTENCES = 5
MIN_AI_SENTENCES = 2
MAX_AI_SENTENCES = 6
SEED = 43  # different from build_hybrid_essays.py's 42, so this is a distinct sample of pairs


def load_essays(fname: str) -> list[dict]:
    with (PROCESSED_DIR / fname).open() as f:
        return list(csv.DictReader(f))


def build_hybrid(human_text: str, ai_text: str, rng: random.Random) -> dict | None:
    human_sents = segment.segment(human_text)
    ai_sents = segment.segment(ai_text)
    min_len = 2 * MIN_ANCHOR_SENTENCES + MIN_REPLACED_SENTENCES
    if len(human_sents) < min_len or len(ai_sents) < MIN_AI_SENTENCES:
        return None

    # Replaced span must start at/after MIN_ANCHOR_SENTENCES and leave at
    # least MIN_ANCHOR_SENTENCES real sentences after it ends.
    last_possible_start = len(human_sents) - MIN_ANCHOR_SENTENCES - MIN_REPLACED_SENTENCES
    if last_possible_start < MIN_ANCHOR_SENTENCES:
        return None
    start_idx = rng.randint(MIN_ANCHOR_SENTENCES, last_possible_start)
    max_k = min(MAX_REPLACED_SENTENCES, len(human_sents) - MIN_ANCHOR_SENTENCES - start_idx)
    k = rng.randint(MIN_REPLACED_SENTENCES, max(MIN_REPLACED_SENTENCES, max_k))
    end_idx = start_idx + k  # exclusive; human_sents[start_idx:end_idx] gets dropped

    prefix = human_text[:human_sents[start_idx - 1].end].rstrip()
    suffix = human_text[human_sents[end_idx].start:].lstrip() if end_idx < len(human_sents) else ""

    m = rng.randint(MIN_AI_SENTENCES, min(MAX_AI_SENTENCES, len(ai_sents)))
    ai_middle = ai_text[ai_sents[0].start:ai_sents[m - 1].end].strip()

    combined = prefix + " " + ai_middle + (" " + suffix if suffix else "")
    ai_span_start = len(prefix) + 1
    ai_span_end = ai_span_start + len(ai_middle)

    return {
        "text": combined,
        "ai_span_start": ai_span_start,
        "ai_span_end": ai_span_end,
        "n_prefix_sentences": start_idx,
        "n_replaced_sentences": k,
        "n_ai_sentences": m,
        "n_suffix_sentences": len(human_sents) - end_idx,
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
            "id": f"hybrid-mid-{i:03d}",
            "text": built["text"],
            "ai_span_start": built["ai_span_start"],
            "ai_span_end": built["ai_span_end"],
            "human_essay_id": h["id"],
            "ai_essay_id": a["id"],
            "n_prefix_sentences": built["n_prefix_sentences"],
            "n_replaced_sentences": built["n_replaced_sentences"],
            "n_ai_sentences": built["n_ai_sentences"],
            "n_suffix_sentences": built["n_suffix_sentences"],
        })

    if skipped:
        print(f"Skipped {skipped} pair(s) — human essay too short to carve a mid-paragraph span from.")

    out_path = PROCESSED_DIR / "hybrid_essays_midparagraph.csv"
    fieldnames = ["id", "text", "ai_span_start", "ai_span_end", "human_essay_id", "ai_essay_id",
                  "n_prefix_sentences", "n_replaced_sentences", "n_ai_sentences", "n_suffix_sentences"]
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} mid-paragraph hybrid essays to {out_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
