# Data provenance

This file documents where every essay in `data/processed/` comes from and what it can and can't support. It's written phase by phase as data gets added, not reconstructed afterward.

`data/raw/`, `data/processed/`, and `data/incoming/` are gitignored and never committed — see "Licensing and privacy" below. This file is the durable record of what's actually in them.

## Data at a glance

| Source | Count | Role | Genre |
|---|---|---|---|
| EssayForum (HF mirror) plus three curated sources | 530 human | Training | Admissions essays |
| Gemini API | 500 AI | Training | Admissions essays |
| OpenAI + Anthropic APIs | 160 AI (200 generated, 40 held out for eval) | Training | Admissions essays |
| DAIGT-v2 (16 AI generators, PERSUADE-corpus human) | 1,056 (528 human / 528 AI) | Training | Persuasive essays |
| **Total training set** | **2,246 essays** | | |
| DAIGT-v2, remaining ~43,600 rows | 200-essay fixed sample | Eval only | Persuasive, cross-genre check |
| PERSUADE 2.0 | 300 essays | Eval only | Fairness / ESL check |
| ELLIPSE | 200 essays | Eval only | Fairness / ESL check |
| Hybrid-spliced essays, two constructions | 200 essays | Eval only | Passage-localization check |
| Held-out generator essays | 40 (20 OpenAI + 20 Anthropic) | Eval only | Novel-generator recall check |

What this doesn't cover, in full detail in `docs/LIMITATIONS.md` #8: non-English essays, STEM or technical supplement essays, graduate statements of purpose, and admissions essays under roughly 150 to 300 words depending on the source. The single biggest gap is AI tools not represented in the table at all — anything released after this data was collected, or that we simply never got around to sourcing. The source-by-source detail below, including licensing and how PII was handled, is kept in full because how each source was vetted matters as much as the final count.

`DOCUMENTS/IMPLEMENTATION.md` has the full build history, and `docs/EVALUATION.md` and `docs/LIMITATIONS.md` cover what each addition actually changed about detection accuracy.

## Human essays (`data/processed/human_essays.csv`) — 530 essays

**EssayForum-only for several rounds, by user decision — reversed this
round.** The single-source EssayForum pool was a real contributor to the
ELL/race fairness gap (`docs/LIMITATIONS.md` #1, #3): EssayForum is a
peer-feedback forum skewing toward international/ESL applicants posting
*drafts*, and the classifier's notion of "human-like" was calibrated
entirely against that one register. Three previously-built, previously-
dropped adapters (openessays.org, conncoll.edu, blog.emoryadmission.com —
real published/curated "essays that worked" pages, a genuinely different,
complementary population: polished, already-admitted, not draft/
feedback-seeking) were re-enabled in `build_dataset.py` to counter that
skew directly.

**Real combined yield from the three added sources: 30 essays, not the
200-300 initially hoped for** — these are small curated showcase pages
(a handful of essays per college, refreshed occasionally), not bulk
databases, and this project already learned that once before (the same
three sources combined only yielded ~30 essays the first time they were
tried, before the EssayForum HF mirror was found). A live check before
committing to this round's plan confirmed the same ceiling still holds.
Proceeding anyway: `class_weight='balanced'` in training means the value
here is a second population to calibrate "human-like" against, not
matching EssayForum's volume — a real, if modest, diversity gain.

| Source | Count | What it is |
|---|---|---|
| `huggingface:nid989/EssayFroum-Dataset` | 500 | A third-party scrape of essayforum.com, already collected and published as an HF dataset (see "How we got past the EssayForum block" below — this is how we got real EssayForum content without scraping the site ourselves). Filtered from 25,571 total rows down to essays matching admissions/personal-statement/scholarship keyword markers and >=300 words (1,657 such essays exist; 500 sampled deterministically, seed=42). |
| `blog.emoryadmission.com` | 7 | Emory's official admissions blog, publishing full personal statements from admitted students with staff commentary. Anonymous — no author name on any post, nothing to redact. |
| `conncoll.edu` | 9 | Connecticut College's "Essays that Worked" page. Real named authors — see redaction note below. |
| `openessays.org` | 14 | An open, crawlable admissions-essay database. Real named authors, and every page states "License: UNKNOWN" — see the licensing note below. |
| **Total** | **530** | |

**Name redaction (conncoll.edu, openessays.org only — 23 essays).** Both
adapters capture each essay's `author_name` field at scrape time.
`build_dataset.py`'s `_redact_name()` replaces the full name and each
individual name part (≥3 characters, so a first-name-only or last-name-
only mention elsewhere in the text is caught too) with
`[name redacted]`, word-boundary matched and case-insensitive, before the
text is written anywhere. Verified with a real word-boundary check
against all 23 essays with a captured author name: zero leaks (an
earlier naive substring check flagged 3 false positives — e.g. author
name "Das" matching inside "coronavirus" — that a proper `\b`-bounded
check confirmed weren't real). A generic email/phone regex safety net
(same pattern already used for the EssayForum mirror) runs across all
four sources regardless of author-name presence.

**Reviewer meta-commentary stripped during ingestion.** EssayForum posts
are drafts submitted for feedback, and some open or close with text
directed at forum reviewers rather than the essay itself ("Let me know
what you think of my essay (first draft)...") — this reads as
stylistically anomalous and was found to trigger false AI-flags
(`docs/EVALUATION.md` §6). `hf_essayforum.py` now segments each essay
into sentences and drops any sentence matching a curated set of
review-request phrasings, before the word-count filter (so an essay that
only clears the 300-word floor because of stripped meta-commentary is
correctly excluded, not silently shortened below the intended floor).
Verified zero remaining matches across all 500 essays against those
specific patterns.

Theme tags — **updated**, `infer_theme()` in `backend/scripts/
build_dataset.py` was improved (word-boundary regex instead of substring
matching, most-keyword-hits-wins instead of first-match-wins) after the
original heuristic was found to produce a badly lopsided AI-side
distribution (see the AI essays section below). Still a heuristic, still
not human-verified — see `docs/LIMITATIONS.md` #4.

| theme_id | count |
|---|---|
| growth_accomplishment | 160 |
| background_identity | 119 |
| obstacle_setback | 91 |
| open_topic (fallback / unclassified) | 78 |
| captivating_topic | 41 |
| gratitude | 35 |
| challenging_belief | 6 |

(All 530 essays now run through the same `infer_theme()` heuristic,
including the 30 newly-added ones — counts include all four sources.)

### Why EssayForum-only was reversed this round

Originally: a single, consistent source rather than blending four sites
with different registers and conventions (a college's curated "essays
that worked" showcase reads differently than an
unfiltered forum post asking for feedback, independent of AI-vs-human
questions — mixing them risks the classifier partly learning
"which website" instead of "human vs AI"). openessays.org, conncoll.edu,
and blog.emoryadmission.com were dropped from `build_dataset.py`'s output
for this reason, not because anything was wrong with them.

Reversed because the tradeoff flipped: the "which website" risk is real
but small at 30/530 essays (5.7% of the pool), while the single-source
skew was measurably contributing to the fairness gap in
`docs/LIMITATIONS.md` #1 — worth the small mixing risk to get a second,
different population to calibrate against. See `DOCUMENTS/IMPLEMENTATION.md`
for the full before/after evaluation numbers.

### How we got past the EssayForum block

EssayForum itself is still bot-blocked (see "Why this took several
attempts" below) — that hasn't changed. What changed: someone else already
scraped essayforum.com and published the result as an open HF dataset
(`nid989/EssayFroum-Dataset`, 25,571 rows, no per-row author/username
field). Downloading that via `datasets.load_dataset()` isn't scraping
EssayForum ourselves and doesn't involve working around anything — it's
pulling from HF's own CDN, same as any other public dataset. See
`backend/scripts/sources/hf_essayforum.py`.

**Licensing nuance, stated plainly:** the uploader tagged this dataset
`apache-2.0`. That's a self-applied tag on scraped forum content — it does
not establish that the individual EssayForum posters licensed their own
essays under Apache 2.0, since the uploader almost certainly doesn't hold
the right to relicense someone else's writing on their behalf. This is a
real, common gap on Hugging Face (uploaders tagging scraped content with an
open-source license that only accurately describes their own scraping
code/structure, not the underlying text). Treated with the exact same
policy already applied to openessays.org and conncoll.edu: local-only,
gitignored, never committed or redistributed, PII-scrubbed. The one
practical difference from those two sources: this dataset has no per-row
author field to redact by, so `hf_essayforum.py` instead runs a regex
safety net over the text (emails, phone numbers, "my name is ___"
patterns) — weaker than exact-name redaction, but appropriate given
there's no name to target in the first place.

### History: how we got here (kept for context, not current state)

The plan's original approach — scraping EssayForum's admissions
sub-forum directly — hit bot-protection that returns HTTP 403 to every
request including `robots.txt` itself; this project doesn't work around
anti-bot defenses, so that path stayed closed throughout. Three other
sources (openessays.org, conncoll.edu, blog.emoryadmission.com) were
scraped directly and got the pool to 30 essays before the Hugging Face
dataset was found. Once it was, the user chose EssayForum-only for
consistency (see above) and those 30 essays were replaced, not kept
alongside the 500. Their licensing situations, if this decision is ever
revisited: openessays.org essays carry real names and `License: UNKNOWN`;
conncoll.edu and blog.emoryadmission.com are institutional publications
without an explicit open-reuse license (the latter is anonymous, no
author name at all). All three adapters remain in
`backend/scripts/sources/` and worked correctly when last run.

Other candidates checked and rejected along the way: JHU/Wellesley
"essays that worked" pages (403/bot-blocked), several other
college-specific URLs that no longer resolve (404), and University of
Iowa's admissions blog (publishes students' *advice about* writing their
essay, not the essay text itself — not usable as a training example).

The scraper uses a per-site adapter pattern in `backend/scripts/sources/`
— see `hf_essayforum.py` for the shape a new adapter should take if more
sources are ever needed.

### Licensing and privacy (current source)

`huggingface:nid989/EssayFroum-Dataset` has no per-row author/username
field, so there's no byline to redact — but the underlying essay text can
still self-identify (an essay mentioning its own author's name inline,
an email signed off with, a phone number). `hf_essayforum.py` runs a
regex safety net over every essay for emails, phone numbers, and
"my name is ___" patterns; verified zero matches remaining across all 500
essays post-scrub. This is weaker than exact-name redaction (no NER, so
a name mentioned without that lead-in phrase would survive), consistent
with the same non-guarantee noted for the other sources: essays may still
contain distinguishing details (schools, hometowns, other people's names)
that a determined reader could use to re-identify the author.

`data/` (including `data/incoming/`) is gitignored — raw and processed
essay text is never committed to this repo or otherwise redistributed.
Only aggregate statistics (counts, feature distributions, the trained
model) leave this local dataset.

### Known bias / coverage gaps (carried forward into `docs/LIMITATIONS.md` at Phase 7)

- EssayForum is a peer-feedback forum, not a showcase of already-admitted
  essays — this is a **different** skew than the openessays.org/conncoll.edu
  mix had (which skewed toward already-successful, published-as-exemplary
  applicants). EssayForum is heavily used by international/ESL applicants
  seeking feedback before submitting, so this pool may itself skew ESL —
  flagged as an open question per the original plan §3d, not assumed
  neutral. Worth checking directly against ELLIPSE (Phase 7) once
  available, since that's exactly the population ELLIPSE covers too.
  - These are unedited/pre-submission drafts seeking feedback, not
    polished final essays — likely rougher and more variable in quality
    than a curated "essays that worked" sample would be. This changes
    what the reference-percentile tables (§4) calibrate "human-like"
    against, compared to the earlier 30-essay pool.
- All essays are English-language admissions/scholarship personal
  statements sourced from one forum. No coverage of: non-English essays,
  STEM/technical supplement essays, or essays under 300 words (this
  source's filter, stricter than the plan's ~150-word floor).
- Genre filtering is keyword-based (see `ADMISSION_MARKERS` in
  `hf_essayforum.py`), not human-verified — some false positives
  (tangentially mentioning "scholarship") or false negatives (genuine
  admissions essays that happen not to use any marker phrase) are likely.
## AI essays (`data/processed/ai_essays.csv`) — 500 essays

Generated by the user via the Gemini API: one essay per prompt in a
`500_admissions_prompts.csv` file they wrote (50 categories x 10 prompts
each — Identity & self-understanding, Family & relationships, Challenges
& resilience, Growth & change, etc. — a much finer-grained scheme than
this project's 7-theme bucketing). Ingested via
`backend/scripts/ingest_ai_essays.py` from
`data/incoming/ai_admissions_essays_gemini.csv`.

| Field | Value |
|---|---|
| Count | 500 |
| Generator | `gemini-flash-lite-latest` (499/500), `models/gemini-2.5-flash` (1/500, likely a retry/fallback) |
| Word count | min 398, max 761, median ~560 |
| Source categories | 50, 10 essays each — preserved per-row as `source_category`, not used for the 7-theme split (see below) |

**theme_id was originally computed with the same keyword heuristic used
for human essays — fixed, since it doesn't need to be a guess.** AI
essays carry a *known* true category (Gemini's 50 generation categories),
so `theme_mapping.py` maps that directly to the 7 themes instead
(`ingest_ai_essays.py`). This produces an exact distribution matching the
known category counts (10 essays/category):

| theme_id | human | ai | ai (old, keyword-guessed) |
|---|---|---|---|
| growth_accomplishment | 160 | 100 | 193 |
| background_identity | 119 | 70 | 1 |
| obstacle_setback | 91 | 40 | 254 |
| open_topic | 78 | 120 | 31 |
| captivating_topic | 41 | 100 | 8 |
| gratitude | 35 | 20 | 7 |
| challenging_belief | 6 | 50 | 6 |

**The old AI-side distribution was badly lopsided** (~89% in 2 of 7
buckets) because the keyword heuristic wasn't designed against Gemini's
actual output register — fixed by using ground truth instead of guessing
from text. This had a large downstream effect on the unseen-theme
evaluation split (`docs/EVALUATION.md` §2) and on which 3 themes get
held out (`HELD_OUT_THEMES` in `train_classifier.py`, now
`challenging_belief`/`gratitude`/`obstacle_setback` — previously
`background_identity`/`captivating_topic`/`gratitude`). Full discussion
in `DOCUMENTS/IMPLEMENTATION.md`.

### Known limitation: single-generator (as of this batch — since broadened, see below)

Plan §3a wants generation "spread across several different LLMs and a
couple of temperature settings... a classifier trained on one model's
quirks will just learn that model's fingerprint, not 'AI-ness' in
general." This batch is Gemini-only (99.8% one specific model
variant). Using it is still the right call — real data massively
outweighs the placeholder set it replaces — but this is a real,
load-bearing limitation, not a formality: expect this classifier to be
better at detecting Gemini's writing specifically than "AI writing"
generally.

**Update: the held-out-generator batch this section anticipated as
"more valuable than originally scoped" was in fact built** (100 more
essays each from OpenAI and Anthropic, see the section below) — the
project's AI training data now spans 3 generators directly (Gemini,
OpenAI, Anthropic) plus 16 more via DAIGT-v2 (persuasive genre). The
core limitation isn't fully closed, just precisely measured now: recall
on genuinely novel generators (never in training anywhere) is 30% at
the essay level — real progress from a from a true single-generator
7.5%, still the weakest number in this project. See
`docs/LIMITATIONS.md` #2 for the full, current picture.

### Licensing and privacy

AI-generated text has no personal-privacy dimension the way real people's
essays do (no author to redact, no re-identification risk). Kept in the
same gitignored `data/` tree as everything else for consistency, not
because it needs the protection.

## More AI essays (`data/processed/ai_essays.csv`, appended) — 160 essays, round 8

**Directly answers the single-generator limitation flagged above.** 200
more real essays were generated by the user via the OpenAI and Anthropic
APIs (`gpt-5.6-luna`, `claude-haiku-4-5-20251001`), from the same
500-prompt set as the Gemini batch, via a separate pipeline
(`other_ai_pipeline/`, outside this repo). Ingested by
`backend/scripts/ingest_ai_essays.py` (rewritten to handle 3 sources).

| Field | Value |
|---|---|
| Count | 100 OpenAI + 100 Anthropic = 200 generated, 160 used in training (80 each), 40 held out entirely (20 each) |
| Word count | 500-612 words |
| Markdown stripped before ingestion | Yes — 100/100 Anthropic essays opened with a literal `# Title` header, both sources kept `\n\n` breaks Gemini's output never had; `_strip_markdown()` removes headers/bold/italic syntax so the classifier can't learn "has markdown" as a trivial, ungenuine tell |

**Held-out-generator split (`data/processed/ai_essays_heldout_generator.csv`, 40 essays, eval-only, `HELD_OUT_SEED=44`)**: the more precise cross-generator check DAIGT can't give alone, since DAIGT conflates genre and generator into one number — this isolates the generator effect specifically, same admissions genre/prompts as training. Scored by `backend/scripts/score_heldout_generator.py`.

**History, since the same lever was tested twice with opposite results depending on scale — worth documenting honestly, not just the number that stuck:**
- Round 6: trained on the 160 essays against a ~1,030-essay base. Made things worse (in-genre accuracy and passage-localization recall both dropped) and held-out-generator recall was only 7.5% even on essays from the *same* generators — declined, reverted to Gemini-only training.
- Round 7: DAIGT-v2 training slice added (above) — didn't touch this data (OpenAI/Anthropic aren't in DAIGT's 16 generators), held-out-generator recall unchanged at 7.5%.
- Round 8: retested training on the same 160 essays, now against the round-7-grown 2,246-essay base. Recall **jumped to 30.0%** (essay-level), a real, 4x improvement — the round-6 rejection wasn't wrong, it was correct for that dataset size; round 8's acceptance is the same lever at 2x the data. **Adopted.** Full numbers in `docs/EVALUATION.md` §3/`docs/LIMITATIONS.md` #2b.

**Still the weakest number in this project's evaluation**: 30% essay-level recall on genuinely novel-to-training essays from these exact two generators, and worse than that (~unmeasured, presumed similar-or-worse) on any AI tool not represented anywhere in training at all. Not solved, disclosed.

## Evaluation sets

**DAIGT-v2 (`data/processed/daigt_eval.csv`) — 44,868 rows, built.**
Downloaded via `backend/scripts/download_daigt.py` from the
`Yunij/kaggle-comp-daigt` HF mirror (sidesteps the Kaggle-auth blocker
below, same approach as `hf_essayforum.py`). 27,371 human (mostly
`persuade_corpus`), 17,497 AI across 16 generator models (`mistral7binstruct`
v1/v2, `chat_gpt_moth`, `llama2_chat`, `kingki19_palm`, `llama_70b_v1`,
`falcon_180b_v1`, `darragh_claude_v6`/`v7`, `cohere-command`,
`palm-text-bison1`, and more — verified the label convention directly
against real rows rather than assuming: label 0 = human, label 1 = AI).

**Correction, round 7: no longer eval-only.** For six rounds this file
was correctly described as "not used in training" — a fixed 200-essay
sample (`score_daigt.py`, `random.Random(42)`) was the only thing read
from it, purely for the cross-genre check. **Round 7 changed this**:
`backend/scripts/build_daigt_training_slice.py` now also samples a
1,056-essay training slice (528 human / 528 AI, capped 35/generator for
diversity) from the *rest* of the pool and writes it to
`data/processed/daigt_training_slice.csv`, which `extract_features.py`
and `train_classifier.py` both read. This was the single biggest lever
for cross-genre accuracy in this project (DAIGT sentence-level accuracy
51%->77-78%, essay-level 51%->93%, `docs/EVALUATION.md` §3) — training
on genre/generator-diverse data, not just testing against it.

Two correctness safeguards, both load-bearing, documented in full in
`build_daigt_training_slice.py`'s own docstring: (1) the exact
`score_daigt.py` eval sample is reproduced and excluded from the
training slice, so the cross-genre check stays genuinely essay-level
held-out; (2) DAIGT's "human" class *is* the PERSUADE 2.0 corpus — the
same corpus the fairness-eval set below is sampled from — so every
DAIGT-human essay whose text matches a `persuade_eval.csv` essay is
excluded by normalized-text match (301 essays were caught and excluded
by this check; without it, the fairness numbers in `docs/EVALUATION.md`
§5 would have been silently invalid).

Covers the cross-genre check from plan §3b; does NOT cover the
fairness/ESL check (no demographic fields) — that's PERSUADE's/ELLIPSE's
job specifically, see "Fairness evaluation sets" below (built long ago,
this paragraph used to incorrectly say otherwise — removed).

## Hybrid essays — 200 essays across 2 constructions, built (Phase 6 + post-Phase-7)

Passage-localization test sets (plan §3b), both eval-only (same isolation
as DAIGT above), both built from the 530/500 human/AI pools:

- **`hybrid_essays.csv` (100, continuation)** — a real human essay
  prefix (3 sentences to 70% of the essay) with a real AI essay's
  opening sentences (2-6) appended. seed=42.
- **`hybrid_essays_midparagraph.csv` (100, mid-paragraph replacement,
  added after the first evaluation round)** — a contiguous span of a
  real human essay's sentences (2-5, starting in the middle third)
  removed and replaced with real AI sentences (2-6), keeping genuine
  human text on *both* sides — the case plan §3b's brief calls most
  realistic ("a human paragraph an AI later polished"). seed=43 (a
  distinct sample of pairs from the continuation set).

Both verified directly against sample rows (span offsets checked to
contain exactly the intended spliced text, not just trusted from the
construction arithmetic) before scoring. **Rebuilt twice already**:
once when the meta-commentary-stripping fix refreshed `human_essays.csv`,
and again when the human pool expanded from 500 to 530 (see the Human
essays section above) — both builders are deterministic given the same
seed and read directly from `human_essays.csv`/`ai_essays.csv`, so each
rebuild required no code change, just a re-run.

Scored via `backend/scripts/score_hybrid_essays.py` and
`score_hybrid_essays_midparagraph.py` (sentence-level, against the known
span, short sentences context-merged rather than excluded, threshold
fixed at 0.5):

| | Continuation | Mid-paragraph |
|---|---|---|
| F1 (current, round 10) | 0.536 | 0.367 |

**Mid-paragraph is meaningfully harder** (lower precision — a
false-alarm surface on both sides of the AI splice instead of one) and
is plausibly the more realistic case to weight if picking a single
number. Both hybrid sets are still built only from the original
Gemini/EssayForum pool (unaffected by the OpenAI/Anthropic or DAIGT
additions in rounds 7-8). Full numbers and round-by-round history in
`docs/EVALUATION.md` §4.

## Fairness evaluation sets (`data/processed/persuade_eval.csv`, `ellipse_eval.csv`) — 300 + 200 essays, built

PERSUADE 2.0 and ELLIPSE, downloaded via `backend/scripts/
download_eval_datasets.py` using the user's own Kaggle credentials
(`nbroad/persaude-corpus-2`, `matthewjansen/ellipse-corpus` — third-party
Kaggle re-uploads of the official research corpora, sanity-checked: 25,996
real PERSUADE rows with all 4 demographic fields present, matching the
plan's expected size; 6,482 real ELLIPSE rows). Sampled via
`backend/scripts/prepare_fairness_evals.py`: PERSUADE stratified 150/150
by `ell_status` (a plain random sample would carry too few ELL essays to
break out meaningfully — only ~8.6% of the raw pool); ELLIPSE a plain
random 200. **Both eval-only** — same isolation as DAIGT above, nothing
in the training pipeline reads these files.

Both sources are entirely genuine human writing (no AI class), so this
measures **false-positive rate**, not accuracy. Scored via
`backend/scripts/score_fairness.py`. The most consequential finding in
this project's evaluation, investigated across two rounds, not just
measured once: `investigate_fairness_bias.py` traced the ELL/economic
gap to one dominant feature (`gltr_pct_top10`), and
`experiment_drop_feature.py` tested removing it before adopting the fix
— it shrank both gaps to small residuals (ELL vs. non-ELL: 14.7 -> 4.7
points; economically disadvantaged vs. not: 14.7 -> 1.0 point) at
essentially no accuracy cost. `investigate_race_fairness.py` then traced
the separate Black/African American vs. White gap (12.1 points) to a
related but distinct mechanism spread across four features; a mitigation
was tested (`experiment_drop_feature_race.py`) and found to nearly close
that gap too, but only at a real cost (3 points of unseen-theme
accuracy) — **that fix was deliberately not adopted**, disclosed instead
of engineered away. Diversifying the human training pool beyond EssayForum-only (round 5)
further shrank most gaps as a side effect, and the round-6 classifier
swap shrank them further still — both without any fairness-targeted
model change. **Round 7-8's large training-data additions (DAIGT,
OpenAI/Anthropic) moved fairness numbers in mixed directions instead of
uniformly improving** (PERSUADE overall broad FPR: 9.7% round 5 -> 3.7%
round 6 -> 8.3% round 7 -> 10.7% round 8/current) — a real, disclosed
finding that these numbers aren't a fixed property of "the model," they
move with any large training-data change, fairness-targeted or not.
Full numbers, both investigations' methodology, and the current mixed
picture are in `docs/EVALUATION.md` §5 and `docs/LIMITATIONS.md` #1.

### Licensing

PERSUADE and ELLIPSE were obtained properly via Kaggle with the user's
own account/credentials, following each dataset's normal terms of
access — no equivalent concern to the HF-mirror sources above. Kaggle
credentials are stored at `~/.kaggle/kaggle.json` on the local machine
only, outside this repository entirely (not just gitignored — literally
not inside the project directory), and were never committed or logged in
full.
