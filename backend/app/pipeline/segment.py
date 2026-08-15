"""Sentence segmentation via spaCy. One shared nlp instance for the process."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import spacy
from spacy.tokens import Doc, Span


@dataclass
class Sentence:
    text: str
    start: int
    end: int
    span: Span  # spaCy Span for this sentence — POS/dep tags for stylometry.py


@lru_cache(maxsize=1)
def _nlp():
    return spacy.load("en_core_web_sm", exclude=["ner", "lemmatizer"])


def parse(essay_text: str) -> Doc:
    return _nlp()(essay_text)


def sentences_from_doc(doc: Doc) -> list[Sentence]:
    sentences = []
    for sent in doc.sents:
        stripped = sent.text.strip()
        if not stripped:
            continue
        leading_ws = len(sent.text) - len(sent.text.lstrip())
        start = sent.start_char + leading_ws
        end = start + len(stripped)
        sentences.append(Sentence(text=stripped, start=start, end=end, span=sent))
    return sentences


def segment(essay_text: str) -> list[Sentence]:
    return sentences_from_doc(parse(essay_text))
