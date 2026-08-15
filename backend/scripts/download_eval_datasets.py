"""
Phase 1 — download PERSUADE and ELLIPSE (plan §3b) via the Kaggle API.

Requires Kaggle auth to already be set up (this script doesn't handle
credentials itself — the `kaggle` package reads them directly). Easiest
path: run `kaggle auth login` once in this venv (OAuth, no token to
manage). Alternatively, generate a token at kaggle.com/settings/api and
either `export KAGGLE_API_TOKEN=...` or save it to ~/.kaggle/access_token.

Dataset slugs below are third-party Kaggle re-uploads of the official
PERSUADE 2.0 / ELLIPSE corpora (found via search, not the original
competition datasets) — SANITY-CHECK THE OUTPUT after first download:
PERSUADE should have ~25,000 rows with demographic columns (ELL status,
economic disadvantage, race/ethnicity, disability); ELLIPSE should be
ESL-writer scored essays. If either looks wrong, search Kaggle directly
for a better match and update PERSUADE_SLUG / ELLIPSE_SLUG below.

Official source of truth for PERSUADE 2.0, if the Kaggle mirror is ever
stale or wrong: https://github.com/scrosseye/persuade_corpus_2.0 (data
itself is Google-Drive-hosted, linked from that repo's README).

Usage:
    cd backend && .venv/bin/python scripts/download_eval_datasets.py
"""
from __future__ import annotations

import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "data" / "raw"

PERSUADE_SLUG = "nbroad/persaude-corpus-2"
ELLIPSE_SLUG = "matthewjansen/ellipse-corpus"


def _download(slug: str, dest_subdir: str) -> None:
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()

    dest = RAW_DIR / dest_subdir
    dest.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {slug} -> {dest.relative_to(REPO_ROOT)}...")
    api.dataset_download_files(slug, path=str(dest), unzip=False)

    zips = list(dest.glob("*.zip"))
    for z in zips:
        with zipfile.ZipFile(z) as zf:
            zf.extractall(dest)
        z.unlink()

    print(f"  -> {[p.name for p in dest.iterdir()]}")


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    _download(PERSUADE_SLUG, "persuade")
    _download(ELLIPSE_SLUG, "ellipse")
    print("\nDone. Sanity-check row counts and demographic columns before using — see module docstring.")


if __name__ == "__main__":
    main()
