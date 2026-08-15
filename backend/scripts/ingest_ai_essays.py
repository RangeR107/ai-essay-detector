"""
Phase 1 — ingest the user's real AI-generated admissions essays (plan §3a),
replacing the Phase 2 placeholder set (backend/scripts/build_placeholder_ai_essays.py).

Sources, all sharing the same schema (prompt_id, category, prompt, essay,
model, label) since they're all generated from the same
500_admissions_prompts.csv prompt set:
  - 500 essays from the Gemini API (data/incoming/ai_admissions_essays_gemini.csv),
    499/500 from gemini-flash-lite-latest, 1 from models/gemini-2.5-flash.
  - 100 essays from OpenAI (data/incoming/ai_admissions_essays_openai.csv),
    generator_model='gpt-5.6-luna' — added this round to move beyond a
    single-generator training set (docs/LIMITATIONS.md #2).
  - 100 essays from Anthropic (data/incoming/ai_admissions_essays_anthropic.csv),
    generator_model='claude-haiku-4-5-20251001' — same reason.

**Markdown formatting stripped from the two new sources before ingestion**
(_strip_markdown() below) — a real, measured formatting artifact, not a
hypothetical one: 100/100 Anthropic essays opened with a literal
"# Title" markdown header (0/100 Gemini essays have any markdown at
all), and both new sources kept "\n\n" paragraph breaks that Gemini's
output never had. Left in, either would be a trivial, ungenuine "tell"
letting the classifier learn "has a markdown header" or "has literal
newline characters" instead of any real stylistic signal — the same
category of problem the EssayForum meta-commentary strip (round 3)
fixed on the human side. Whitespace/newlines are handled by the existing
generic `" ".join(text.split())` normalization below (already ran on the
Gemini set); the markdown header/bold/italic markers needed a dedicated
strip since they're not just whitespace.

**Held-out-generator split, new this round**: `HELD_OUT_GENERATOR_FRACTION`
of the OpenAI and Anthropic essays (20 each, seed=44) are set aside into
a separate eval-only file, `ai_essays_heldout_generator.csv` — same
isolation pattern as DAIGT/PERSUADE/ELLIPSE (nothing in the training
pipeline reads it). This is a more precise cross-generator check than
DAIGT: DAIGT conflates genre (persuasive, not admissions) and generator
(9+ non-Gemini models) into one number, so a low DAIGT score doesn't say
whether it's the genre shift or the generator shift doing the damage.
Holding out admissions-genre essays from GPT/Claude specifically isolates
the generator effect alone. Scored by
`backend/scripts/score_heldout_generator.py`.

theme_id: uses theme_mapping.py's curated category->theme lookup
directly for all three sources (all draw from the same 50 categories,
verified covered), not the keyword-guessing heuristic used for human
essays — see that module's docstring for why ground truth beats a guess
here.

Usage:
    cd backend && .venv/bin/python scripts/ingest_ai_essays.py
"""
from __future__ import annotations

import csv
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
from build_dataset import infer_theme  # noqa: E402
from theme_mapping import theme_for_category  # noqa: E402
from app.pipeline.textclean import strip_markdown as _strip_markdown_body  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
INCOMING_DIR = REPO_ROOT / "data" / "incoming"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"

MIN_WORDS = 150
HELD_OUT_GENERATOR_FRACTION = 20  # per new generator, not a fraction — a fixed count
HELD_OUT_SEED = 44  # distinct from build_hybrid_essays.py's 42/43

SOURCES = [
    ("ai_admissions_essays_gemini.csv", "gemini", False),
    ("ai_admissions_essays_openai.csv", "openai", True),
    ("ai_admissions_essays_anthropic.csv", "anthropic", True),
]


def _strip_markdown(text: str) -> str:
    # Shared with the live-served pipeline — see
    # backend/app/pipeline/textclean.py's docstring. This wrapper just
    # adds the lstrip() this ingestion path has always done (titles sit
    # right at the very start of these generated essays).
    return _strip_markdown_body(text.lstrip())


def _load_source(fname: str, id_prefix: str, strip_md: bool) -> list[dict]:
    path = INCOMING_DIR / fname
    if not path.exists():
        print(f"{path.relative_to(REPO_ROOT)} not found. Skipping.")
        return []
    with path.open(newline="", encoding="utf-8") as f:
        source_rows = list(csv.DictReader(f))

    rows = []
    skipped = 0
    for r in source_rows:
        essay = _strip_markdown(r["essay"]) if strip_md else r["essay"]
        text = " ".join(essay.split())
        if len(text.split()) < MIN_WORDS:
            skipped += 1
            continue
        theme_id = theme_for_category(r["category"]) or infer_theme(text)
        rows.append({
            "id": f"{id_prefix}-{r['prompt_id']}",
            "source": f"user-generated:{id_prefix}-api",
            "class": r["label"].strip().lower(),
            "program_type": "admissions-prompt",
            "theme_id": theme_id,
            "source_category": r["category"],
            "generator_model": r["model"],
            "word_count": len(text.split()),
            "text": text,
        })
    if skipped:
        print(f"  Skipped {skipped} {id_prefix} essay(s) under {MIN_WORDS} words.")
    return rows


def main() -> None:
    all_rows: list[dict] = []
    heldout_rows: list[dict] = []
    rng = random.Random(HELD_OUT_SEED)

    for fname, id_prefix, strip_md in SOURCES:
        print(f"Loading {fname}...")
        rows = _load_source(fname, id_prefix, strip_md)
        print(f"  -> {len(rows)} essays")

        if id_prefix == "gemini" or not rows:
            all_rows.extend(rows)
            continue

        heldout_ids = set(rng.sample([r["id"] for r in rows], min(HELD_OUT_GENERATOR_FRACTION, len(rows))))
        for r in rows:
            (heldout_rows if r["id"] in heldout_ids else all_rows).append(r)
        print(f"  -> {len(heldout_ids)} held out entirely from training for the unseen-generator check")

    if not all_rows:
        print("No valid essays ingested.")
        return

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    fieldnames = ["id", "source", "class", "program_type", "theme_id", "source_category", "generator_model", "word_count", "text"]

    out_path = PROCESSED_DIR / "ai_essays.csv"
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nWrote {len(all_rows)} real AI essays to {out_path.relative_to(REPO_ROOT)}")

    heldout_path = PROCESSED_DIR / "ai_essays_heldout_generator.csv"
    with heldout_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(heldout_rows)
    print(f"Wrote {len(heldout_rows)} held-out-generator essays (eval-only) to {heldout_path.relative_to(REPO_ROOT)}")

    from collections import Counter
    print("By theme:", dict(Counter(r["theme_id"] for r in all_rows)))
    print("By generator:", dict(Counter(r["generator_model"] for r in all_rows)))
    print("Held-out by generator:", dict(Counter(r["generator_model"] for r in heldout_rows)))


if __name__ == "__main__":
    main()
