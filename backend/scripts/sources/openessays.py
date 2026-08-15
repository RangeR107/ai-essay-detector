"""
Adapter for openessays.org (an open, crawlable database of admission essays).

Scope note: essays here are attributed to real, named individuals and each
essay page states "License: UNKNOWN". Per project decision, we only pull
undergrad-genre essays, strip the author's name from the text, and never
commit the raw text to git (see data/README.md).
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass

import requests
from bs4 import BeautifulSoup

BASE = "https://openessays.org"
SITEMAP = f"{BASE}/sitemap.xml"
HEADERS = {"User-Agent": "Mozilla/5.0 (research; contact: arsalaankhan47000@gmail.com)"}

# Only these slug-type markers count as undergrad personal-statement genre.
# (PhD/MS/MBA/LLB/GSE statements-of-purpose are a different genre and excluded.)
UNDERGRAD_MARKERS = ("-bs-", "-ba-", "-bachelors-")


@dataclass
class RawEssay:
    id: str
    source: str
    url: str
    program_type: str
    author_name: str
    title: str
    text: str


def _list_essay_urls() -> list[str]:
    resp = requests.get(SITEMAP, headers=HEADERS, timeout=20)
    resp.raise_for_status()
    urls = re.findall(r"<loc>(https://openessays\.org/essays/[a-zA-Z0-9-]+)</loc>", resp.text)
    return sorted(set(urls))


def _is_undergrad(slug: str) -> bool:
    s = f"-{slug}-"
    return any(m in s for m in UNDERGRAD_MARKERS)


def _parse_essay(url: str) -> RawEssay | None:
    resp = requests.get(url, headers=HEADERS, timeout=20, allow_redirects=True)
    if resp.status_code != 200:
        return None
    soup = BeautifulSoup(resp.text, "lxml")
    body_text = soup.get_text(" ", strip=True)

    title_match = re.search(r"^(.*?) Essay Sample \| ", soup.title.string if soup.title else "")
    title = title_match.group(1).strip() if title_match else (soup.title.string or "").strip()

    program_match = re.search(r"Program:\s*([^:]+?)\s*Type:", body_text)
    program_type = program_match.group(1).strip() if program_match else "unknown"

    source_match = re.search(r"Source:\s*(.*?)\s*View Original", body_text)
    author_name = ""
    if source_match:
        name_match = re.search(r"\(([^)]+)\)\s*$", source_match.group(1))
        if name_match:
            author_name = name_match.group(1).strip()

    # Essay body sits after "View Original" marker in the rendered text.
    body_start = body_text.find("View Original")
    essay_text = body_text[body_start + len("View Original"):].strip() if body_start != -1 else ""
    essay_text = re.sub(r"\s+", " ", essay_text).strip()

    slug = url.rstrip("/").rsplit("/", 1)[-1]
    return RawEssay(
        id=f"openessays-{slug}",
        source="openessays.org",
        url=url,
        program_type=program_type,
        author_name=author_name,
        title=title,
        text=essay_text,
    )


def fetch_undergrad_essays(rate_limit_s: float = 1.0) -> list[RawEssay]:
    urls = _list_essay_urls()
    undergrad_urls = [u for u in urls if _is_undergrad(u.rsplit("/", 1)[-1])]

    essays: list[RawEssay] = []
    for i, url in enumerate(undergrad_urls):
        essay = _parse_essay(url)
        if essay and len(essay.text.split()) >= 150:
            essays.append(essay)
        if i < len(undergrad_urls) - 1:
            time.sleep(rate_limit_s)
    return essays
