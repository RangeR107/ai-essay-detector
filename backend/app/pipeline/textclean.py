"""
Markdown-syntax stripping, shared between training-data ingestion
(backend/scripts/ingest_ai_essays.py) and the live-served pipeline
(featurize.py).

Originally written only for training ingestion, after finding 100/100
Anthropic-generated training essays opened with a literal "# Title"
header — left in, that's a trivial, ungenuine "tell" the classifier
could learn instead of real stylistic signal, so it was stripped before
those essays ever reached the feature extractor.

That same problem exists for a live user pasting AI output that still
has its markdown intact (headers, **bold**, *italic* — ChatGPT and
Claude both format admissions-essay drafts this way by default). Left
unstripped at serve time, two things go wrong: sentence segmentation can
glue a stray "**" onto the start of the next sentence (producing a
malformed "sentence" that isn't representative of anything in training),
and the surviving features are computed on text that looks nothing like
what the classifier actually trained on (which was markdown-free).
Stripping this at the single shared entry point (featurize.featurize_essay)
keeps segmentation, GLTR token stats, and the UI's displayed sentence
text all consistent with each other and with what training saw.
"""
from __future__ import annotations

import re

_MD_HEADER_RE = re.compile(r"^#{1,6}\s+.*$", re.MULTILINE)
_MD_BOLD_RE = re.compile(r"\*\*(.+?)\*\*")
_MD_ITALIC_RE = re.compile(r"(?<!\*)\*([^*\n]+?)\*(?!\*)")


def strip_markdown(text: str) -> str:
    text = _MD_HEADER_RE.sub("", text)
    text = _MD_BOLD_RE.sub(r"\1", text)
    text = _MD_ITALIC_RE.sub(r"\1", text)
    return text
