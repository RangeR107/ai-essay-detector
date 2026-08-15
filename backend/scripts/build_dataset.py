"""
Phase 1 — build the human-essay pool of the core training set (see plan §3a).

Sources (see data/README.md for the full provenance/licensing discussion):
  - huggingface:nid989/EssayFroum-Dataset (500 essays, third-party scrape of
    essayforum.com; no per-row author field, self-applied apache-2.0 tag
    doesn't establish the original posters licensed their essays that way —
    treated as local-only)
  - blog.emoryadmission.com, conncoll.edu, openessays.org — re-added this
    round (were EssayForum-only for a while, by user decision, to keep a
    single consistent register; re-added deliberately to counter that same
    single-source skew, which turned out to be a real contributor to the
    ELL/race fairness gap — see docs/LIMITATIONS.md #1, #3). These three are
    real published/curated "essays that worked" pages, not a peer-feedback
    forum — a different, complementary population (polished, already-
    admitted) rather than more of the same (drafts seeking critique).
    Real, low volume per source (dozens, not hundreds) — not a EssayForum-
    scale fix, a deliberate second population, not a bigger first one.

Two of the three (conncoll.edu, openessays.org) publish essays under real
individual names; author names are redacted from the essay text before
writing (see _redact_name() below) using each adapter's captured
author_name field. blog.emoryadmission.com publishes anonymously — nothing
to redact for that source. openessays.org additionally states
"License: UNKNOWN" on every page — same treatment already applied to the
EssayForum mirror: local-only, gitignored, never committed or
redistributed (see data/README.md).

Output: data/processed/human_essays.csv
Both data/raw/ and data/processed/ are gitignored — this is local-only
data, never committed (see data/README.md).

Usage:
    cd backend && .venv/bin/python scripts/build_dataset.py
"""
from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from sources import conncoll, emoryadmission, hf_essayforum, openessays  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "data" / "raw"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"

# Also catches emails/phone numbers that might appear in essay text
# regardless of source, same safety net hf_essayforum.py already applies.
_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_PHONE_RE = re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b")


def _redact_name(text: str, author_name: str) -> str:
    """Strip the author's full name and each individual name part (>=3
    chars, to skip initials/suffixes) from the essay text, word-boundary
    matched, case-insensitive — a first-name-only or last-name-only
    mention elsewhere in the text is just as identifying as the full name."""
    if not author_name:
        return text
    parts = [author_name] + [p for p in author_name.split() if len(p) >= 3]
    for part in sorted(set(parts), key=len, reverse=True):
        text = re.sub(rf"\b{re.escape(part)}\b", "[name redacted]", text, flags=re.IGNORECASE)
    return text


def _scrub_pii(text: str) -> str:
    text = _EMAIL_RE.sub("[email redacted]", text)
    text = _PHONE_RE.sub("[phone redacted]", text)
    return text

# Best-effort keyword rules for the 7 personal-statement themes from plan §3a.
# Only used for human essays now — AI essays use theme_mapping.py's direct
# category lookup instead, since they carry a known true category (see
# ingest_ai_essays.py). Expanded from the original list (which was thin on
# gratitude/captivating_topic/background_identity specifically) and switched
# from substring/first-match-wins to word-boundary-regex/most-matches-wins
# — see infer_theme() below for why that change.
THEME_KEYWORDS = {
    "obstacle_setback": [
        "overcome", "overcame", "struggle", "diagnosed", "injury", "failed", "failure",
        "setback", "hardship", "lost my", "loss of", "difficult time", "challenge i faced",
        "rock bottom", "gave up", "recover", "illness", "disability", "adversity",
    ],
    "challenging_belief": [
        "used to believe", "changed my mind", "questioned", "challenged my", "i realized i was wrong",
        "reconsidered", "my assumption", "i was wrong about", "shifted my perspective",
        "opened my eyes", "preconceived notion", "i no longer believe",
    ],
    "gratitude": [
        "grateful", "gratitude", "thankful", "appreciate", "blessed to", "indebted",
        "owe so much", "means the world", "i am lucky", "i was fortunate",
    ],
    "growth_accomplishment": [
        "i learned", "taught me", "grew as a", "proudest", "accomplishment", "achieved",
        "growth", "matured", "i became more", "developed the skill", "i am now able",
        "leadership", "responsibility", "accomplished",
    ],
    "captivating_topic": [
        "fascinated by", "obsessed with", "passion for", "captivated", "curious about",
        "love of", "drawn to", "i am passionate", "my curiosity", "endlessly interesting",
        "lose track of time", "i can't stop thinking about",
    ],
    "background_identity": [
        "my family", "my culture", "my heritage", "grew up", "immigrant", "my identity",
        "where i come from", "my community", "my hometown", "my parents", "my upbringing",
        "native language", "first generation", "my neighborhood",
    ],
    "open_topic": [],  # fallback bucket
}

_WORD_RE_CACHE: dict[str, re.Pattern] = {}


def _keyword_count(lowered_text: str, keyword: str) -> int:
    """Word-boundary match for single words, plain substring count for
    multi-word phrases (word-boundary regex on a full phrase is
    unnecessarily strict about internal whitespace/punctuation)."""
    if " " in keyword:
        return lowered_text.count(keyword)
    pattern = _WORD_RE_CACHE.setdefault(keyword, re.compile(rf"\b{re.escape(keyword)}\b"))
    return len(pattern.findall(lowered_text))


def infer_theme(text: str) -> str:
    """Most-keyword-hits-wins, not first-match-wins: counts every keyword
    occurrence per theme across the whole essay and picks the theme with
    the highest total, falling back to open_topic only if nothing matches
    at all. First-match-wins (the original approach) meant a single early,
    possibly-incidental keyword decided the whole essay's theme even if a
    different theme's vocabulary dominated the rest of the text — this is
    a strictly better signal from the same keyword lists, no new data."""
    lowered = text.lower()
    counts = {
        theme: sum(_keyword_count(lowered, kw) for kw in keywords)
        for theme, keywords in THEME_KEYWORDS.items()
        if keywords
    }
    best_theme = max(counts, key=counts.get)
    return best_theme if counts[best_theme] > 0 else "open_topic"


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print("Fetching huggingface:nid989/EssayFroum-Dataset (admissions-genre subset)...")
    essayforum_essays = hf_essayforum.fetch_essays(max_essays=500)
    print(f"  -> {len(essayforum_essays)} essays")

    print("Fetching blog.emoryadmission.com...")
    emory_essays = emoryadmission.fetch_essays()
    print(f"  -> {len(emory_essays)} essays")

    print("Fetching conncoll.edu 'Essays that Worked'...")
    conncoll_essays = conncoll.fetch_essays()
    print(f"  -> {len(conncoll_essays)} essays")

    print("Fetching openessays.org (undergrad genre)...")
    openessays_essays = openessays.fetch_undergrad_essays()
    print(f"  -> {len(openessays_essays)} essays")

    all_essays = essayforum_essays + emory_essays + conncoll_essays + openessays_essays

    raw_records = [e.__dict__ for e in all_essays]
    (RAW_DIR / "human_essays_raw.json").write_text(json.dumps(raw_records, indent=2))

    rows = []
    for e in all_essays:
        text = e.text
        author_name = getattr(e, "author_name", "")
        if author_name:
            text = _redact_name(text, author_name)
        text = _scrub_pii(text)
        rows.append({
            "id": e.id,
            "source": e.source,
            "class": "human",
            "program_type": e.program_type,
            "theme_id": infer_theme(text),
            "word_count": len(text.split()),
            "text": text,
        })

    out_path = PROCESSED_DIR / "human_essays.csv"
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "source", "class", "program_type", "theme_id", "word_count", "text"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {len(rows)} human essays to {out_path.relative_to(REPO_ROOT)}")
    from collections import Counter
    print("By source:", dict(Counter(r["source"] for r in rows)))
    print("By theme:", dict(Counter(r["theme_id"] for r in rows)))


if __name__ == "__main__":
    main()
