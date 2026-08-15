"""Shared fixtures. Model-loading tests (featurize/classify/evidence) are
inherently slower — spaCy + GPT-2 load once per session via the pipeline
modules' own @lru_cache, not per-test."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

SAMPLE_ESSAY = (
    "The clinking of measuring spoons always fills me with joy. "
    "I learned to code when I was six years old, and it changed the way I see the world. "
    "Every day I am on the lookout for the next opportunity to create something that helps others."
)


@pytest.fixture(scope="session")
def sample_essay() -> str:
    return SAMPLE_ESSAY
