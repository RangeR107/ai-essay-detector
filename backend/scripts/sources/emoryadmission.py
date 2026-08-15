"""
Adapter for blog.emoryadmission.com — Emory's official undergraduate
admissions blog, which periodically publishes full personal statements from
admitted students with admissions-staff commentary.

Unlike openessays.org/conncoll.edu, these posts carry NO author name at
all (essays are presented anonymously, attributed only to "students now
enrolled at Emory University") — nothing to redact for this source.

Every post on this blog follows the same fixed structure regardless of how
many essays it contains: an <h5> holding the Common App prompt text, essay
paragraphs, then an <h5> reading "Feedback from Admission Staff", then
commentary paragraphs, repeating for however many essays are in the post.
We walk the article's <h5>/<p> children in document order and use that
structure directly, rather than guessing content patterns.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass

import requests
from bs4 import BeautifulSoup

BASE = "https://blog.emoryadmission.com"
POST_SITEMAP = f"{BASE}/post-sitemap.xml"
HEADERS = {"User-Agent": "Mozilla/5.0 (research; contact: arsalaankhan47000@gmail.com)"}

# Only posts in this family publish full student essays; the rest of the
# blog (financial aid updates, staff profiles, etc.) is out of scope.
SLUG_MARKER = "strong-personal-statement"

_END_MARKERS = ("related posts", "comments", "leave a reply")


@dataclass
class RawEssay:
    id: str
    source: str
    url: str
    program_type: str
    author_name: str  # always "" for this source — nothing to redact
    title: str
    text: str


def _list_post_urls() -> list[str]:
    resp = requests.get(POST_SITEMAP, headers=HEADERS, timeout=20)
    resp.raise_for_status()
    urls = re.findall(r"<loc><!\[CDATA\[(https://blog\.emoryadmission\.com/[^\]]+)\]\]></loc>", resp.text)
    if not urls:
        urls = re.findall(r"<loc>(https://blog\.emoryadmission\.com/[^<]+)</loc>", resp.text)
    return sorted(u for u in set(urls) if SLUG_MARKER in u.lower())


def _parse_post(url: str) -> list[RawEssay]:
    resp = requests.get(url, headers=HEADERS, timeout=20)
    if resp.status_code != 200:
        return []
    soup = BeautifulSoup(resp.text, "lxml")
    article = soup.find("article")
    if not article:
        return []

    essays: list[RawEssay] = []
    current_prompt = ""
    current_paragraphs: list[str] = []
    mode = "seeking_prompt"

    def flush():
        if current_paragraphs:
            essays.append((current_prompt, list(current_paragraphs)))

    # Heading level for the prompt/"Feedback from Admission Staff" markers
    # varies by post (h4 in 2021 posts, h5 in 2025 posts) — match any of
    # them rather than relying on a fixed level.
    for el in article.find_all(["h2", "h3", "h4", "h5", "p"]):
        text = el.get_text(" ", strip=True)
        if not text:
            continue
        if el.name != "p":
            lowered = text.lower()
            if lowered.startswith("feedback from admission"):
                flush()
                current_paragraphs = []
                mode = "commentary"
            elif any(lowered.startswith(m) for m in _END_MARKERS):
                break
            else:
                current_prompt = text
                mode = "essay"
        elif mode == "essay":
            current_paragraphs.append(text)

    slug = url.rstrip("/").rsplit("/", 1)[-1]
    results = []
    for i, (prompt, paragraphs) in enumerate(essays):
        text = re.sub(r"\s+", " ", " ".join(paragraphs)).strip()
        results.append(RawEssay(
            id=f"emoryadmission-{slug}-{i}",
            source="blog.emoryadmission.com",
            url=url,
            program_type=f"Common App Essay ({prompt[:80]})",
            author_name="",
            title=prompt,
            text=text,
        ))
    return results


def fetch_essays(rate_limit_s: float = 1.0) -> list[RawEssay]:
    urls = _list_post_urls()
    all_essays: list[RawEssay] = []
    for i, url in enumerate(urls):
        for essay in _parse_post(url):
            if len(essay.text.split()) >= 150:
                all_essays.append(essay)
        if i < len(urls) - 1:
            time.sleep(rate_limit_s)
    return all_essays
