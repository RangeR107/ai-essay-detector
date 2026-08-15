"""
Adapter for Connecticut College's "Essays that Worked" page — personal
statements the college itself publishes for prospective applicants.

Essays are attributed to real, named individuals. Per project decision, we
strip the author's name from the text and never commit the raw text to git
(see data/README.md).
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass

import requests
from bs4 import BeautifulSoup

BASE = "https://www.conncoll.edu"
LISTING = f"{BASE}/admission/apply/essays-that-worked/"
HEADERS = {"User-Agent": "Mozilla/5.0 (research; contact: arsalaankhan47000@gmail.com)"}


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
    resp = requests.get(LISTING, headers=HEADERS, timeout=20)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "lxml")
    urls = set()
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if "/essays-that-worked/" in href and href.rstrip("/") != LISTING.rstrip("/"):
            if href.startswith("/"):
                href = BASE + href
            if href.startswith(BASE):
                urls.add(href)
    return sorted(urls)


def _parse_essay(url: str) -> RawEssay | None:
    resp = requests.get(url, headers=HEADERS, timeout=20)
    if resp.status_code != 200:
        return None
    soup = BeautifulSoup(resp.text, "lxml")

    heading = soup.find(["h1", "h2"])
    author_name = heading.get_text(" ", strip=True) if heading else ""
    # Strip the trailing class-year mark, e.g. "Elle Yarborough '28" -> "Elle Yarborough"
    author_name = re.sub(r"[’']\d{2}\s*$", "", author_name).strip()

    # Every page on this listing follows the same fixed template: essay
    # paragraphs, then exactly one admissions-office commentary paragraph
    # (free-form text, no distinguishing markup), then a literal "Read more
    # Essays that worked." paragraph, a CTA button, and a footer. Rather than
    # pattern-matching the commentary's free-form wording (which varies
    # essay to essay), anchor on the fixed "Read more" marker and drop the
    # paragraph immediately before it.
    all_paragraphs = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
    all_paragraphs = [p for p in all_paragraphs if p]
    stop_idx = next(
        (i for i, p in enumerate(all_paragraphs) if p.lower().startswith("read more")),
        len(all_paragraphs),
    )
    essay_paragraphs = all_paragraphs[: max(stop_idx - 1, 0)]

    text = " ".join(essay_paragraphs)
    text = re.sub(r"\s+", " ", text).strip()

    slug = url.rstrip("/").rsplit("/", 1)[-1]
    return RawEssay(
        id=f"conncoll-{slug}",
        source="conncoll.edu",
        url=url,
        program_type="Bachelors Common App Essay",
        author_name=author_name,
        title=slug.replace("-", " ").title(),
        text=text,
    )


def fetch_essays(rate_limit_s: float = 1.0) -> list[RawEssay]:
    urls = _list_essay_urls()
    essays: list[RawEssay] = []
    for i, url in enumerate(urls):
        essay = _parse_essay(url)
        if essay and len(essay.text.split()) >= 150:
            essays.append(essay)
        if i < len(urls) - 1:
            time.sleep(rate_limit_s)
    return essays
