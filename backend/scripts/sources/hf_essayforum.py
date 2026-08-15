"""
Adapter for the nid989/EssayFroum-Dataset on Hugging Face — a third-party
scrape of essayforum.com, already collected and published by someone else.
Using this sidesteps the bot-protection blocker documented in
data/README.md's "Why this is 30, not 500" section: we're not scraping
essayforum.com ourselves (no anti-bot circumvention involved), we're
downloading an already-public HF dataset via HF's own CDN.

Licensing note (see data/README.md for the full discussion): the uploader
tags this dataset `apache-2.0`, but that's a self-applied tag on scraped
forum content — it doesn't establish that the original EssayForum posters
licensed their personal essays that way. Treated with the same conservative
policy as every other source here: local-only, gitignored, names/PII
stripped where detectable, never committed or redistributed.

No per-row author/username field exists in this dataset (unlike
openessays.org/conncoll.edu) — nothing to redact by field. A lightweight
regex safety net still runs over the text for self-identifying patterns
("My name is ..."), emails, and phone numbers, since redaction-by-field
isn't available here.

~25,571 rows total, spanning every essayforum.com sub-forum (IELTS/TOEFL
prep, opinion essays, task-based writing, admissions, scholarships, etc.).
Filtered here to admissions/personal-statement genre via keyword markers.
"""
from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from app.pipeline import segment  # noqa: E402

HF_DATASET_ID = "nid989/EssayFroum-Dataset"

ADMISSION_MARKERS = [
    "personal statement", "common app", "admission", "scholarship", "statement of purpose",
]
MIN_WORDS = 300

# EssayForum posts are feedback requests — some essays include sentences
# addressed to forum reviewers, not part of the essay itself ("Please
# review this and let me know what you think!"). Found by inspecting real
# data: 69/500 essays in an earlier build had at least one such sentence.
# Stripped at SENTENCE granularity (not whole-essay filtering) to avoid
# false positives on essays that legitimately discuss feedback/critique as
# a topic. Patterns are deliberately specific/directive rather than broad
# — an earlier, looser draft of this list (bare "let me know") false-
# positived on "My mom ... called to let me know I had been admitted,"
# genuine narrative content, not meta-commentary.
META_COMMENTARY_PATTERNS = [
    r"\bplease (review|check|correct|proofread|critique)\b",
    r"\b(any|some)\s+(feedback|thoughts?|suggestions?|comments?|critiques?)\b",
    r"\bthanks?\s+(in advance|for reading|for your (time|help))\b",
    r"\blet me know what you think\b",
    r"\bplease let me know\b",
    r"\bi would appreciate\b",
    r"\b(rough|messy)\s+draft\b",
    r"\bfeel free to (correct|criticize|point out)\b",
    r"\bproofread\b",
    r"\bcan (you|someone|anyone) (please )?(check|review|read)\b",
]
_META_COMMENTARY_RE = re.compile("|".join(META_COMMENTARY_PATTERNS), re.IGNORECASE)


def _strip_meta_commentary(text: str) -> str:
    sentences = segment.segment(text)
    kept = [s.text for s in sentences if not _META_COMMENTARY_RE.search(s.text)]
    return " ".join(kept)


@dataclass
class RawEssay:
    id: str
    source: str
    url: str
    program_type: str
    author_name: str  # always "" — no per-row attribution in this dataset
    title: str
    text: str


_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_PHONE_RE = re.compile(r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b")
_SELF_NAME_RE = re.compile(r"\bmy name is\s+[A-Z][a-zA-Z'-]+(?:\s+[A-Z][a-zA-Z'-]+)?", re.IGNORECASE)


def _scrub(text: str) -> str:
    text = _EMAIL_RE.sub("[EMAIL]", text)
    text = _PHONE_RE.sub("[PHONE]", text)
    text = _SELF_NAME_RE.sub("my name is [NAME]", text)
    return text


def fetch_essays(max_essays: int = 500, seed: int = 42) -> list[RawEssay]:
    from datasets import load_dataset

    ds = load_dataset(HF_DATASET_ID)["train"]

    candidates = []
    for i, row in enumerate(ds):
        text = " ".join(row["Cleaned Essay"].split())
        lowered = text.lower()
        if not any(m in lowered for m in ADMISSION_MARKERS):
            continue
        # Meta-commentary stripping happens BEFORE the word-count filter,
        # not after — otherwise an essay padded to >=300 words partly by
        # reviewer-directed sentences could pass the filter pre-strip and
        # then fall short post-strip, silently shrinking the final pool.
        stripped = _strip_meta_commentary(text)
        if len(stripped.split()) < MIN_WORDS:
            continue
        candidates.append((i, stripped))

    # Deterministic sample so re-runs are reproducible.
    import random
    rng = random.Random(seed)
    rng.shuffle(candidates)
    candidates = candidates[:max_essays]

    essays = []
    for i, text in candidates:
        essays.append(RawEssay(
            id=f"hf-essayforum-{i}",
            source=f"huggingface:{HF_DATASET_ID}",
            url=f"https://huggingface.co/datasets/{HF_DATASET_ID}",
            program_type="unknown",
            author_name="",
            title="",
            text=_scrub(text),
        ))
    return essays
