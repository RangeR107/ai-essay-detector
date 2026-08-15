# AI Admissions Essay Detector — Implementation Log

This file is a running, phase-by-phase log of what has actually been built,
what decisions were made (and why), and what's still open. It's updated at
the end of every phase in `docs/AI_ESSAY_DETECTOR_PLAN.md`'s build order.
For dataset-specific detail (counts, sources, licensing), see
[`data/README.md`](../data/README.md) — that file is the source of truth
for dataset state; this file summarizes it and tracks everything else.

---

## Phase 1 — Dataset foundation (human essays)

**Status: partially complete.** Human essays collected at reduced scale. AI
essays, eval sets, and hybrid essays not yet built.

### What was built

- Repo scaffolding: `backend/{app,scripts,tests}`, `frontend/src/components`,
  `data/{raw,processed}`, `docs/` per the plan's §2 layout.
- `backend/scripts/sources/openessays.py` — adapter for openessays.org.
  Fetches the sitemap, filters to undergrad-genre essays only (excludes the
  site's PhD statement-of-purpose essays as a different genre), parses
  title/program/author/body from each page.
- `backend/scripts/sources/conncoll.py` — adapter for Connecticut College's
  official "Essays that Worked" page. Notably, every essay page on that site
  follows a fixed template (essay paragraphs → one admissions-office
  commentary paragraph → "Read more" CTA → footer) with no distinguishing
  HTML markup between the commentary and the essay itself; the adapter
  anchors on the literal "Read more Essays that worked." marker and drops
  the paragraph immediately before it, rather than pattern-matching the
  commentary's free-form wording (which varies essay to essay and isn't
  reliably catchable by regex).
- `backend/scripts/build_dataset.py` — orchestrates both adapters,
  redacts each essay's byline name from its own text (word-boundary regex,
  verified zero leaks — see below), assigns a best-effort `theme_id` via
  keyword heuristics across the plan's 7 personal-statement themes, and
  writes `data/processed/human_essays.csv`.
- `data/README.md` — full provenance, licensing, and coverage-gap writeup
  (this is the plan's §3d deliverable, written now while fresh rather than
  retroactively).
- `.gitignore` — excludes `data/raw/` and `data/processed/` entirely (see
  decisions below), plus standard Python/Node/OS entries.
- `backend/requirements.txt` — full stack per plan §1 (not yet installed
  beyond the scraping subset used so far: `requests`, `beautifulsoup4`,
  `lxml`, in a `backend/.venv`).

### Result

23 human admissions essays in `data/processed/human_essays.csv`
(14 from openessays.org, 9 from conncoll.edu). Author names verified
redacted with zero leaks (checked programmatically post-build). Full
breakdown — sources, themes, word counts, licensing caveats — is in
[`data/README.md`](../data/README.md).

### Decisions made with the user this phase

1. **EssayForum scraping blocked.** The plan's original human-essay source
   returns HTTP 403 to every request (including `robots.txt`) via bot
   protection. Rather than attempt to defeat that protection, this was
   flagged to the user, who chose to source from an existing open corpus
   instead of EssayForum.
2. **openessays.org licensing.** Essays there are real, named individuals
   with each page stating `License: UNKNOWN` and no site-wide terms/license
   page. User decision: use the content, but strip the author's name from
   the text and keep `data/` local-only — gitignored, never committed or
   published. This governs the `.gitignore` and the redaction step in
   `build_dataset.py`.
3. **Core training set size.** openessays.org's undergrad-genre pool is
   only ~14 essays (the site's 144 total essays are 81% PhD
   statements-of-purpose, a different genre) — nowhere near the plan's 500
   target. User decision: scale the core set down to roughly 100-150 per
   class instead of 500/500, using all usable essays from openessays.org
   plus other small clean sources. Actual result after adding conncoll.edu
   is 23 — still short of that revised target; this gap is called out
   explicitly in `data/README.md` rather than closed by lowering the bar
   further silently. More source adapters can be added later using the
   same pattern.
4. **AI essay generation — deferred to the user.** Per the user's
   instruction, the 500-essay AI-generation step (plan §3a,
   `generate_ai_essays.py`) is not being automated in this environment.
   The user will generate matched AI admissions essays themselves (e.g.
   via Claude or another model) and hand back the results for the
   pipeline to ingest.
5. **Eval datasets (PERSUADE/ELLIPSE/DAIGT-v2) — blocked on credentials.**
   These are Kaggle-hosted. User agreed to provide Kaggle API credentials;
   not yet supplied. **Do not paste an API key/token directly into chat.**
   The safer path: create a token at
   `kaggle.com → Account → API → Create New Token` (downloads
   `kaggle.json`), then either place it yourself at `~/.kaggle/kaggle.json`
   on this machine, or tell me to do it and I'll write the file locally
   (it never needs to touch git or a form field).

### Known gaps going into Phase 2+

- Human essay count (23) is well short of even the revised ~100-150/class
  target. Not blocking Phase 2 (MVP skeleton doesn't need the full set),
  but blocking a meaningful Phase 5+ evaluation. Revisit by either adding
  more adapters (same pattern as `openessays.py`/`conncoll.py`) or waiting
  on essays the user can supply directly.
- No AI essays yet — the classifier can't be trained until the user hands
  back generated AI essays (see decision 4 above).
- No PERSUADE/ELLIPSE/DAIGT-v2 — blocked on Kaggle credentials (decision 5).
- Hybrid essays (plan §3b, the passage-localization eval set) need AI text
  to splice in, so they're blocked on the same thing as the classifier.
- Theme tagging is a rough keyword heuristic, not human-verified — flagged
  in `data/README.md`; revisit before relying on it for the theme-held-out
  split in §3c.

### Next step

Waiting on: (a) the user's generated AI admissions essays, (b) Kaggle
credentials for eval-set download. Either can unblock the next chunk of
Phase 1 independently — AI essays unblock classifier training in Phase 2;
Kaggle data unblocks Phase 7's fairness/generalization evaluation. Neither
blocks starting Phase 2's MVP skeleton (segmentation → perplexity →
logistic regression → `/analyze` → basic React page) against the current
23-essay human pool plus placeholder AI text, if the user wants to proceed
on the app side while data collection continues in parallel.

---

## Phase 2 — MVP skeleton

**Status: complete.** Something clickable end-to-end, per the plan's Phase 2
goal. Built in parallel with Phase 1 data collection still pending (AI
essays from the user, Kaggle credentials for eval sets) — this phase
doesn't depend on either.

### What was built

**Backend** (`backend/app/`):
- `pipeline/segment.py` — spaCy (`en_core_web_sm`) sentence segmentation
  with char offsets.
- `pipeline/gltr.py` — one GPT-2 (`gpt2`, CPU) forward pass over the essay,
  producing per-token `(rank, logprob)` with char offsets. Built to serve
  both perplexity (used now) and GLTR rank buckets (Phase 3, no second pass
  needed) — matches plan §0/§1's "single forward pass" requirement.
  GPT-2 only ever returns numbers here; nothing downstream asks it for a
  verdict.
- `pipeline/featurize.py` — combines segmentation + the forward pass into
  per-sentence feature dicts. Phase 2 uses one feature, `perplexity`.
- `pipeline/classify.py` — loads the trained `LogisticRegression` +
  `StandardScaler` (joblib) and returns `P(AI)` per sentence. This is the
  only place a score is produced — deterministically, from our own trained
  model.
- `pipeline/aggregate.py` — mean sentence score → essay verdict
  (`Likely Human` / `Inconclusive` / `Likely AI`) against placeholder,
  uncalibrated thresholds (0.35 / 0.65), plus `burstiness` (stdev of
  per-sentence perplexity).
- `schemas.py` / `main.py` — FastAPI app, `POST /analyze`, `GET /health`,
  CORS open to the Vite dev server. Response schema matches plan §6 in
  full, including `top_features` / `token_ranks` fields — they're wired as
  empty lists for now and get populated in Phases 3/5 without a schema
  change.

**Training scripts** (`backend/scripts/`):
- `build_placeholder_ai_essays.py` — **temporary, Phase 2 only.** 20
  short Claude-written admissions essays across the same 7 themes as the
  human set, explicitly tagged `is_placeholder=true` and
  `source=placeholder-self-gen`. This is *not* the plan's real 500-essay
  AI set (§3a) — the user is generating that separately, across multiple
  models/temperatures, and will hand it back to replace this file's
  output. It exists only so the classifier has both classes to train on
  while that's pending.
- `extract_features.py` — runs the pipeline over all 43 essays (23 human +
  20 placeholder-AI), writes a 1,053-row sentence-level feature table.
- `train_classifier.py` — trains `LogisticRegression(class_weight='balanced')`
  on standardized `perplexity`, saves `classifier.joblib`, `scaler.joblib`,
  `reference_stats.joblib` (per-feature human-class percentile table, for
  Phase 5 evidence — not used in classification).

**Frontend** (`frontend/`, Vite + React + TS + Tailwind v4):
- `EssayInput`, `SentenceHighlighter` (continuous color scale by
  `ai_score`, not binary — per plan §7), `VerdictBadge`. No
  `EvidencePanel`/`TokenHeatmap` yet (Phase 5/3).
- `api.ts` typed client for `/analyze`.
- `.claude/launch.json` added so the dev server can be previewed
  (`npm --prefix frontend run dev`, port 5173).

### Verified working

Ran both servers, submitted a real essay through the browser UI: sentences
segmented correctly, each got a continuously-shaded highlight by score, the
verdict badge and burstiness number rendered, no console/network errors.
Confirmed via direct `curl` against `/analyze` first, then via the actual
browser UI.

### Important caveat — read before trusting any number from this phase

**The trained classifier's accuracy is not meaningful yet and shouldn't be
quoted anywhere.** It's trained on 23 real human essays vs. 20 placeholder
AI essays from a single generator (Claude) at default settings — exactly
the "classifier just learns one generator's fingerprint" failure mode the
plan's §3a warns about, except here it's not even the real 500-essay set.
Measured on a random 80/20 split: 33% held-out accuracy, i.e. *worse* than
always guessing "human" (which would score ~79% given the class imbalance
in the sentence table). Perplexity alone barely separates the two classes
at this scale (human sentence-level median perplexity 30.8 vs. AI 36.1,
heavily overlapping distributions).

This isn't a bug — it's the expected result of (a) one weak feature, (b)
placeholder AI text written in a plain, fluent register similar to the
already-polished "essays that worked" human set, and (c) too little data.
It's exactly why Phases 3 (GLTR) and 4 (stylometry) add more features, and
why the real classifier needs the user's actual multi-model AI set and more
human essays before any accuracy number belongs in `docs/EVALUATION.md`.

### Next step

Phase 3 (GLTR token-rank features + token heatmap UI) can start now — it
doesn't need new data, just extends the existing single forward pass
already built in `gltr.py`. Retraining on real data (once the user's AI
essays and/or more human sources arrive) should happen before any
accuracy claim is trusted, independent of which pipeline phase is current.

---

## Phase 1 addendum — added a third human-essay source (Emory)

User asked for more than 23 human essays. Searched for more crawl-permitted,
low-risk sources following the same adapter pattern.

### What was checked and the outcome

- **blog.emoryadmission.com** (Emory's official admissions blog) — added.
  7 essays across 6 blog posts that publish full personal statements with
  admissions-staff commentary. Notably these are **anonymous** — no author
  name anywhere on the page — so there's nothing to redact and no
  re-identification risk, the cleanest source found so far. Two different
  heading levels across post years (`h4` in 2021 posts, `h5` in 2025 posts)
  required the parser to match on content ("Feedback from Admission Staff")
  rather than a fixed tag level — see `backend/scripts/sources/emoryadmission.py`.
- **JHU, Wellesley** "essays that worked" pages — bot-blocked (403), same
  failure mode as EssayForum. Not pursued further (see Phase 1's original
  decision not to work around anti-bot protection).
- **Several other college-specific URLs** surfaced by search (Tufts,
  Gettysburg, William & Mary, older Emory posts from 2018-2020) — 404,
  no longer live.
- **University of Iowa admissions blog** — live and accessible, but the
  post found is a current student's *advice about* writing a personal
  statement (paraphrasing their own essay's topic), not the submitted
  essay text itself. Not usable as a training example; excluded.

### Result

Human pool grew from 23 → **30** essays (14 openessays.org + 9 conncoll.edu
+ 7 blog.emoryadmission.com). Verified zero name leaks across all 30
post-redaction. Full counts/theme breakdown updated in
[`data/README.md`](../data/README.md). Re-ran `extract_features.py` and
`train_classifier.py` on the larger pool — still against the 20 placeholder
AI essays, so the caveat in the Phase 2 section above still applies in
full: held-out accuracy is now 33.5% (was 33.2%), essentially unchanged and
still not meaningful. Growing the human side alone doesn't fix a
single-weak-feature-vs-placeholder-data problem; that needs Phase 3/4's
additional features and the user's real AI essay set.

**Still short of the ~100-150/class target** discussed earlier. More
sources can be added the same way if useful, but returns are diminishing —
most remaining candidates found so far are either bot-blocked, gone, or
don't actually contain reusable essay text. The more efficient path to
real volume is still the user's own AI-essay generation (which also
determines how big the human side needs to be, since the two should stay
roughly balanced).

### Intake tooling added for the two remaining user-blocked items

- `backend/scripts/ingest_ai_essays.py` — the handoff path for the user's
  real AI essays. Drop one `.txt` file per essay into
  `data/incoming/ai_essays/` (gitignored), named
  `<theme_id>__<generator_model>__<temperature>__<index>.txt`; the script
  validates the theme against the plan's 7 categories, filters anything
  under 150 words, and writes `data/processed/ai_essays.csv`. Deliberately
  writes a separate file from the Phase 2 placeholder output rather than
  overwriting it, so switching `extract_features.py` from placeholder to
  real data is an explicit one-line change, not something that happens by
  accident.
- `backend/scripts/download_eval_datasets.py` — downloads PERSUADE +
  ELLIPSE via the Kaggle API once auth is set up. Note: while wiring this
  up, discovered the modern `kaggle` CLI (v2.2.4, installed in
  `backend/.venv`) no longer uses the classic `kaggle.json` file by
  default — recommended path is now `kaggle auth login` (OAuth, no token
  to manage). The script docstring and `data/README.md`/user guidance were
  written against this current flow, not the older `kaggle.json` one
  mentioned earlier in this log.
- Dataset slugs used (`nbroad/persaude-corpus-2`,
  `matthewjansen/ellipse-corpus`) are third-party Kaggle re-uploads found
  via search, not the original competition datasets — the script docstring
  flags this and says what to sanity-check (row count, demographic
  columns) after first download.

---

## Phase 1 addendum 2 — huggingface EssayForum dataset closes the 500-essay gap

User pushed back on the 30-essay count and specifically named a Hugging
Face EssayForum dataset they'd been counting on. Verified it directly
(loaded it, inspected real rows/counts — not just the HF card summary)
rather than assuming the search result was accurate.

### What was found and done

- `nid989/EssayFroum-Dataset`: 25,571 rows, a third-party scrape of
  essayforum.com already published on HF. No per-row author/username
  field. Filtered to admissions/personal-statement/scholarship genre via
  keyword markers + 300-word minimum: 1,657 unique, first-person essays
  survive. Sampled 500 deterministically (seed=42).
- Using this doesn't reopen the anti-bot question from the original Phase
  1 decision — we're not scraping essayforum.com, we're downloading an
  already-public HF dataset via HF's CDN. New adapter:
  `backend/scripts/sources/hf_essayforum.py`.
- **Licensing nuance flagged, not glossed over:** the dataset is tagged
  `apache-2.0` by the uploader, but that's a self-applied tag on scraped
  forum content — it doesn't establish the original posters licensed their
  essays that way. Same conservative treatment as every other source
  (local-only, gitignored, never committed, PII-scrubbed via regex since
  there's no author field to redact by). Full nuance written into
  `data/README.md` rather than just noted here, since that's the durable
  record for anyone deciding whether to trust/reuse this data later.
- Human essay pool: **30 → 530** (14 openessays.org + 9 conncoll.edu + 7
  blog.emoryadmission.com + 500 huggingface:nid989/EssayFroum-Dataset).
  Verified zero email/phone-pattern leaks post-scrub across all 530.

### Retraining result — and an important, non-obvious side effect

Re-ran `extract_features.py` (17,004 sentence rows from 550 essays) and
`train_classifier.py` against the still-placeholder 20-essay AI set.
Held-out accuracy on the random split: **17%** (was 33.5% before this
addendum) — precision on the "ai" class collapsed to 0.01.

**This is not a regression in data quality — it's severe class imbalance.**
Human sentences outnumber AI sentences 75:1 now (16,781 vs 223), instead of
the earlier ~4:1. `LogisticRegression(class_weight='balanced')` compensates
for imbalance by upweighting the minority class in the loss function, but
at 75:1 that overcorrection is strong enough to make the model predict
"ai" on most sentences just to avoid missing the rare true positives —
tanking overall accuracy even though it's "trying" to do the statistically
correct thing given the class ratio it sees.

**Practical implication:** growing the human pool without growing the AI
pool in proportion doesn't help and can make interim numbers look worse,
even though the underlying human data is fine. This is a concrete argument
for the user's real AI-essay batch being sized roughly to match whatever
the human pool ends up at (530, not the original ~100-150 estimate) —
worth keeping in mind when generating, rather than stopping at a smaller
AI count than the human side.

---

## Phase 1 addendum 3 — EssayForum-only by user decision

User decided the 530-essay mixed pool should instead be single-source:
drop openessays.org, conncoll.edu, and blog.emoryadmission.com, keep only
the huggingface EssayForum data. Rationale given: consistency — a
college's curated "essays that worked" showcase and an unfiltered forum
post asking for feedback read differently independent of AI-vs-human
signal, so mixing them risked the classifier partly learning "which
website" rather than "human vs AI."

### What changed

- `backend/scripts/build_dataset.py` now calls only `hf_essayforum`. The
  other three adapters (`openessays.py`, `conncoll.py`,
  `emoryadmission.py`) are left in place, fully working, in case this
  decision is revisited — just not called from the build script anymore.
  Removed the now-dead `redact_name()` helper from `build_dataset.py`
  (name redaction lives inside `hf_essayforum.py` itself, since it's the
  only active source and has no per-row author field to redact by in the
  first place).
- Human pool: 530 → **500** (EssayForum-only).
- Re-ran `extract_features.py` (15,931 sentence rows) and
  `train_classifier.py`. Result: 20% held-out accuracy, same severe
  class-imbalance pattern as addendum 2 (still 20 placeholder AI essays
  against 500 human essays, ~70:1 sentence ratio) — expected, not a new
  finding, see that addendum for the explanation. Still not a meaningful
  number; still waiting on the user's real AI essay set.
- `data/README.md` rewritten to describe the 500-essay EssayForum-only
  pool as current state, with the three dropped sources' licensing
  situations kept as historical context (not restated as if still active)
  and a new coverage-gap note: EssayForum's population (peer-feedback
  forum, heavily international/ESL, pre-submission drafts) is a
  **different** skew than the old 30-essay "already-admitted, polished"
  pool was — worth checking directly against ELLIPSE in Phase 7 since
  that's the same population ELLIPSE covers.

---

## Phase 3 — GLTR

**Status: complete.** Confirmed Phase 1 leftovers (user's AI essays,
Kaggle auth) don't block this phase — GLTR only extends the pipeline
already built in Phase 2, no new training data needed.

### What was built

- `backend/app/pipeline/featurize.py` — added 4 GLTR aggregate features
  from the *same* GPT-2 forward pass already computed for perplexity (no
  second pass): `gltr_pct_top10`, `gltr_pct_top100`, `gltr_pct_top1000`
  (fraction of a sentence's tokens whose predicted rank falls in each
  bucket) and `gltr_mean_rank`. `FEATURE_NAMES` grew from 1 to 5.
- `backend/app/main.py` — `/analyze` now populates `token_ranks` per
  sentence (was an empty list placeholder since Phase 2's schema).
  Cleans up GPT-2's byte-level BPE artifacts (`Ġ` → space, `Ċ` → newline)
  so the token text is human-readable.
- `frontend/src/components/TokenHeatmap.tsx` — new component, word-level
  shading by rank bucket (green→red) with a legend, shown for whichever
  sentence is currently selected in `SentenceHighlighter`. Wired into
  `App.tsx` below the existing sentence view.
- Re-ran `extract_features.py` (15,931 rows, now 5 features instead of 1)
  and `train_classifier.py`.

### Retrain result — a real, if modest, improvement

Held-out accuracy on the random split: **60%** (was 20% with perplexity
alone against the same 500-human/20-placeholder-AI pool). More
informative than the raw accuracy number, given the severe class
imbalance discussed in the last two addenda: **macro-avg F1 moved from
~0.50 (chance) to 0.58**, and AI-class recall is 0.56. The added GLTR
features are carrying real separating signal beyond perplexity alone,
even against weak placeholder data — this is the first result in the
project that isn't indistinguishable from noise.

Precision on the AI class is still bad (0.02) — expected, same ~70:1
class-imbalance cause as before, not a new problem. Still not a number
for `docs/EVALUATION.md`; still waiting on the user's real AI essay set
for that.

### Verified working

Backend: `curl`'d `/analyze` directly, confirmed `token_ranks` populated
with real rank buckets per token. Frontend: loaded the app in-browser,
submitted an essay, clicked a sentence, confirmed the token heatmap
renders with correct color-coded shading and legend, no console errors.

### Next step

Phase 4 (stylometry — sentence length, POS ratios, function-word ratio,
etc.) is next per the plan's build order, and also doesn't need new
training data. Phase 5 (evidence/percentile panel) after that. Real
accuracy numbers still wait on the user's AI essays; Kaggle auth still
wait on `kaggle auth login`.

---

## Phase 4 — Stylometry

**Status: complete.** Confirmed no Phase 1 blocker before starting (same
reasoning as Phase 3 — stylometry is spaCy POS/dependency parsing over
text already in hand, no new essays or Kaggle data needed).

### What was built

- `backend/app/pipeline/segment.py` — refactored to expose the parsed
  spaCy `Doc`/`Span` objects (`parse()`, `sentences_from_doc()`), not just
  plain text spans, so stylometry can read POS/dependency tags without a
  second parse. `segment()` still works as before for existing callers.
- `backend/app/pipeline/stylometry.py` — new module, 8 features per
  sentence, all spaCy-derived (no GPT-2): `sent_length`,
  `avg_word_length`, `rolling_ttr` (type-token ratio over the last 50
  tokens of the essay up to this sentence — a genuine rolling window
  across sentence boundaries, not just this sentence in isolation),
  `punctuation_rate`, `function_word_ratio`, `adjective_ratio`,
  `has_contraction`, `passive_voice_rate`. Passive voice detected via
  spaCy's English dep labels (`nsubjpass`/`csubjpass`/`auxpass` — spaCy
  doesn't use the Universal Dependencies `xxx:pass` form for English,
  worth knowing if this is ever ported to a different spaCy model).
- `featurize.py` — merged in, `FEATURE_NAMES` grew from 5 to 13.
- No UI changes this phase — plan §Phase 4 is "add the features, retrain,"
  the evidence panel that surfaces these to a user is Phase 5.

### Retrain result — best so far, and now both classes have real recall

Held-out accuracy: **72%** (was 60% with GLTR+perplexity alone). More
tellingly: macro-F1 **0.46** (up from 0.39), and for the first time both
classes have decent recall — human 0.72, AI 0.80 (Phase 3 had human 0.60,
AI 0.56). Precision on the AI class is still poor (0.04) — same ~70:1
class-imbalance cause flagged in every phase since addendum 2, still not
fixed by adding features, only by balancing the AI-essay count. Verified
`/analyze` still returns correctly with the 13-feature vector (curl
smoke test, 200 response, sensible per-sentence scores).

### Next step

Phase 5 (evidence/percentile panel — the real `EvidencePanel`, using
`reference_stats.joblib`'s human-only percentile lookup + logistic
regression coefficient × standardized-value contributions) is next, and
also doesn't need new data. After that, Phase 6 (document-level features,
transition detection, hybrid-essay scoring) starts needing the user's AI
essays for the hybrid-essay sub-task specifically — everything else in
Phase 6 can proceed without it.

---

## Phase 1 addendum 4 — user's real AI essay set (generation done, handoff pending)

User has generated the plan's §3a AI-essay set themselves: **500
AI-written college admissions essays, one per prompt in a
`500_admissions_prompts.csv` file, via the Gemini API.** This is the
single biggest open item from Phase 1 — actual files haven't been handed
off yet ("I will share the details for your reference"), so this entry
records the decision/method; counts and theme breakdown will be added
once the files land and `ingest_ai_essays.py` (or a replacement for it —
see below) runs.

### Known caveat, stated honestly now rather than discovered later

Plan §3a is explicit that AI essays should be "spread across several
different LLMs and a couple of temperature settings — a classifier
trained on one model's quirks will just learn that model's fingerprint,
not 'AI-ness' in general." This batch is Gemini-only. That doesn't block
using it — it's real, substantial data, a massive improvement over the
20-essay placeholder set — but it means:
- The classifier's "AI-like" signal risks partly being "Gemini-like." A
  held-out batch from a *different* generator (plan §3b's 100-essay
  generalization check) becomes more important, not less, given this.
- This should be stated plainly in `docs/LIMITATIONS.md` at Phase 7, not
  glossed over — "single-generator AI training data" is exactly the kind
  of honest limitation the plan's evaluation deliverables are supposed to
  surface.

### Handoff format — needs to be resolved before ingesting

`backend/scripts/ingest_ai_essays.py` (built earlier, unused so far)
currently expects one `.txt` file per essay in `data/incoming/ai_essays/`,
named `<theme_id>__<generator_model>__<temperature>__<index>.txt`. Given
the user's actual output is almost certainly a CSV (paired with
`500_admissions_prompts.csv`, one row per prompt/essay) rather than 500
hand-split text files, that script's input format likely needs to change
to read a CSV directly instead. Confirming the real file's columns before
writing that adapter, rather than guessing.

### Next step

Waiting on the user to share the actual file(s). Once received: adapt or
replace `ingest_ai_essays.py` for the real format, write
`data/processed/ai_essays.csv`, re-run `extract_features.py` and
`train_classifier.py` against real AI data for the first time — this is
the point where accuracy numbers in this log stop being placeholder
caveats and start being real (though still not §3c-split "unseen-theme"
numbers, and still short of a full `docs/EVALUATION.md` until PERSUADE/
ELLIPSE are in via Kaggle). Also unblocks the 100 hybrid essays (plan
§3b) once there's real AI text to splice into the human essays.

---

## Phase 1 addendum 5 — real AI essays ingested, first real training run

User's files were at `/Users/arsalaankhan/Desktop/essays/gemini_pipeline/`
(outside this repo). Located and inspected directly rather than guessing
at the format — `ai_admissions_essays.csv`, columns `prompt_id, category,
prompt, essay, model, label`, 500 rows (the file's raw line count, 9354,
is misleading — essay text contains embedded newlines inside quoted CSV
fields, so line count != row count; parsed properly with `csv.DictReader`
to get the real number).

### What was done

- Copied the source file into `data/incoming/ai_admissions_essays_gemini.csv`
  (gitignored, non-destructive — the user's original file untouched).
- Rewrote `backend/scripts/ingest_ai_essays.py` — the version built
  earlier assumed one `.txt` file per essay with metadata encoded in the
  filename, which didn't match the real CSV format at all. Now reads the
  actual CSV directly: `theme_id` computed via the same `infer_theme()`
  heuristic used for human essays (not the source's 50-category scheme,
  kept separately as `source_category`) so theme-tagging methodology
  stays consistent across both classes.
- Ran it: 500 AI essays → `data/processed/ai_essays.csv`. Switched
  `extract_features.py`'s `load_essays()` from `ai_essays_placeholder.csv`
  to `ai_essays.csv` (the deliberately-manual one-line change flagged back
  in addendum 4).
- **Fixed a real methodology bug while rewriting `train_classifier.py`
  anyway:** every prior phase split train/test at the sentence-row level,
  which lets sentences from the same essay land on both sides of a split
  — leakage (same author/topic/generation-run), inflating apparent
  accuracy. All splits are now done at the essay level first, sentence
  rows gathered up afterward by essay-id membership.
- **Implemented the plan §3c theme-held-out evaluation properly** for the
  first time (explicitly deferred in every earlier phase for lack of
  data volume). Held out 3 themes — `background_identity`,
  `captivating_topic`, `gratitude` — chosen as the 3 smallest combined
  (human+AI) categories. `train_classifier.py` now trains one evaluation
  model on the 80% seen-theme training split and reports both plan §3c
  numbers from it, then trains a SEPARATE final model on all 1000 essays
  (seen + unseen) for the actual deployed `classifier.joblib` — held-out
  themes are an evaluation technique, not a reason to permanently shrink
  the production model's training data.

### Results — first real numbers, not placeholder caveats

| | Accuracy | Macro-F1 | Sentences | Essays |
|---|---|---|---|---|
| In-theme held-out | 73.6% | 0.73 | 5,751 | 169 |
| Unseen-theme | 71.3% | **0.58** | 5,008 | 158 |

Raw accuracy gap is small (+2.3 points), which on its own would read as
"generalizes well." **That's misleading here — read the macro-F1 gap
instead (0.73 → 0.58).** The unseen-theme test set is itself
class-imbalanced 8:1 (human:AI, 4449 vs 559 sentences), because the three
held-out themes are exactly where the Gemini batch is thinnest
(`background_identity` has 1 AI essay total, `captivating_topic` has 8,
`gratitude` has 7 — see `data/README.md`'s theme table). On that
imbalanced set, AI-class precision drops to 0.24 even though recall stays
reasonable (0.70) — the same "always lean toward predicting the minority
class" pattern seen in every earlier addendum, now confined to the
unseen-theme slice specifically instead of the whole dataset. So: real
signal, real generalization gap, but the honest headline number is the
macro-F1 drop, not the accuracy numbers, and both come with the
single-generator caveat from `data/README.md` attached.

In-theme numbers are the first balanced, sane-looking classification
report in this project: human P=0.65/R=0.75/F1=0.70, AI P=0.81/R=0.72/
F1=0.77 — no more precision collapsing to 0.01-0.04 like every placeholder-
data run before this. The 1:1 human/AI balance (500/500) is doing exactly
what addendum 2/3 predicted it would.

### Verified working

`/analyze` retested via curl against the new production model (trained on
all 1000 essays) — 200 response, sensible per-sentence scores. One
anecdotal note, not a real evaluation: a real human sentence from the
essay pool the project *used to have* (Connecticut College's "measuring
spoons" essay, dropped in addendum 3's EssayForum-only pivot, so not in
current training data) scored 88% AI on its own. Single-sentence,
out-of-context, not remotely a claim about false-positive rate — real
FPR numbers wait for Phase 7's proper evaluation — but noted here rather
than only mentioning results that looked good.

### Next step

`data/README.md` fully updated (AI essays section, theme table, single-
generator limitation). Phase 6's hybrid-essay sub-task (splice real AI
paragraphs into real human essays, plan §3b) is now unblocked — there's
real AI text to splice with. Phase 7's full evaluation still waits on
Kaggle (PERSUADE/ELLIPSE) for the fairness/generalization sections, but
the classifier itself is no longer placeholder-only.

---

## Phase 1 addendum 6 — decisions on remaining gaps, DAIGT-v2 built

Two things resolved in conversation, not code:

1. **Multi-model AI essay diversification: deferred.** User decided not
   to generate essays from additional models right now (the single-Gemini
   limitation from addendum 5 stands, revisit later if time allows).
2. **Testing composition clarified.** Before this addendum, "testing"
   meant only internal splits of the same single-source data (in-theme
   and unseen-theme, both EssayForum+Gemini) — no independent test set.
   User asked whether DAIGT-v2 could fill that gap.

### DAIGT-v2 — built, same day, same technique as the EssayForum fix

Same move as addendum 2: rather than fight the Kaggle-auth blocker,
checked for (and found) an HF mirror — `Yunij/kaggle-comp-daigt`, 44,868
rows. Verified directly rather than trusting the dataset card: loaded it,
checked real label/source distribution before trusting the label
convention (label 0 = human, mostly `persuade_corpus`; label 1 = AI,
9+ generator models — `mistral7binstruct` v1/v2, `chat_gpt_moth`,
`llama2_chat`, `kingki19_palm`, `llama_70b_v1`, `falcon_180b_v1`,
`darragh_claude_v7`, etc.).

**Kept strictly eval-only, on purpose, to avoid exactly the kind of
conflict the user asked about:** new script `download_daigt.py` writes to
`data/processed/daigt_eval.csv`, a file nothing else in the pipeline
reads. `build_dataset.py` and `ingest_ai_essays.py` don't touch it;
`extract_features.py`/`train_classifier.py` are wired to
`human_essays.csv` + `ai_essays.csv` only, unchanged. DAIGT won't be
consumed until a Phase 7 evaluation script loads the already-trained
`classifier.joblib` and scores it against this file — mirrors how
`download_eval_datasets.py` already keeps PERSUADE/ELLIPSE separate in
`data/raw/`, never merged into training.

**What DAIGT does and doesn't close, stated plainly:** it gives Phase 7 a
real cross-genre generalization check (plan §3b) — accessible right now,
no Kaggle needed. It does NOT provide the demographic fields (ELL status,
race/ethnicity, disability) that PERSUADE carries, or ELLIPSE's
ESL-specific human samples — the subgroup-fairness section of
`docs/EVALUATION.md` still needs Kaggle auth regardless of DAIGT.

### Updated remaining-work picture

- Phase 5 (evidence panel): no blockers, next up.
- Phase 6: transitions/document features have no blockers; hybrid-essay
  span-scoring specifically needs the 100 hybrid essays (plan §3b) built
  first — unblocked (real AI text exists) but not yet done.
- Phase 7: accuracy + DAIGT cross-genre + hybrid-localization sections are
  now all achievable without further external input; only the
  subgroup-fairness section stays blocked on Kaggle.

---

## Phase 5 — Evidence

**Status: complete.** No blockers — pure backend/frontend work against the
model already trained in addendum 5.

### What was built

- `backend/app/pipeline/classify.py` — added `scale_vector()` and
  `coefficients()` accessors (previously only `predict_proba()` and
  `reference_stats()` existed) so `evidence.py` doesn't duplicate the
  scaling logic or reach into the private `_artifacts()` cache directly.
- `backend/app/pipeline/evidence.py` — new module, implements plan §5's
  evidence math exactly: `contribution_i = coefficient_i x
  standardized_value_i`, sorted by `|contribution|` descending, top 3.
  Percentile is computed against `reference_stats` (human-training-class
  only, built in `train_classifier.py`) via `np.searchsorted` — a sentence's
  raw feature value's rank within that distribution, 0-100. Direction
  (`ai-like`/`human-like`) comes from the contribution's sign, not the
  percentile. Every number here traces back to the trained model's own
  coefficients — no generated text (plan §0).
- `schemas.py` — added `magnitude: float` to `FeatureContribution` (not in
  the plan's original §6 shape; needed so the UI can size the "contribution
  magnitude" bar plan §7 asks for). Documented as only meaningful relative
  to the other 2 entries in the same sentence's list, not comparable
  across sentences (magnitudes aren't on a fixed 0-1 scale).
- `main.py` — `/analyze` now populates `top_features` per sentence (was an
  empty-list placeholder since Phase 2).
- `frontend/src/components/EvidencePanel.tsx` — new component. Phrases
  percentile as "higher/lower than X% of human training sentences" (the
  phrasing logic itself is template code, not model output — consistent
  with plan §0). Contribution bars scaled relative to the max magnitude
  *within that sentence's own top-3* (no global reference range exists,
  so cross-sentence bar-width comparison isn't meaningful and the UI
  doesn't imply it is).
- `App.tsx` — `EvidencePanel` and `TokenHeatmap` now render side by side
  in a 2-column grid, both keyed off the same `selectedIndex`. Updated the
  header copy to state the real training composition (500 human + 500 AI)
  and point at this log for the single-generator caveat, replacing the
  stale "Phase 3, placeholder data" text.

### Verified working

Backend: curl'd `/analyze`, confirmed `top_features` populated with
sensible percentile/direction/magnitude values. Frontend: loaded in
browser, submitted an essay ("The cake was eaten by John yesterday..." —
deliberately includes an obvious passive-voice clause to sanity-check the
passive_voice_rate feature surfaces in evidence), clicked a sentence,
confirmed both EvidencePanel and TokenHeatmap render correctly side by
side with real values (e.g. "% tokens in GPT-2's top-10 predictions:
ai-like, lower than 98% of human training sentences"). No console errors.

### Next step

Phase 6 (document-level features/transitions + the 100 hybrid essays for
span-localization testing) is next. The hybrid-essay construction doesn't
depend on anything built this phase — it only needs the human_essays.csv
and ai_essays.csv already in hand.

---

## Phase 6 — Document-level features, transitions, hybrid essays

**Status: complete.** No blockers going in (sanity-checked before
starting — backend healthy, data intact, model current).

### What was built

- `aggregate.py` — added `sentence_length_variance()` and
  `score_volatility()` (plan §4's other two document-level features;
  `burstiness()` was already there since Phase 2). All three are
  post-hoc over the classifier's own sentence scores/features, not
  classifier inputs themselves — adding them needed no retraining.
- `detect_transitions()` — flags a sentence index when its score jump
  from the previous sentence is a statistical outlier relative to that
  essay's *own* volatility baseline (mean + 1.25 x stdev of consecutive
  diffs), with an absolute floor (0.25) so low-volatility essays don't
  flag trivial noise. This is plan §4/§6's "possible change in writing
  pattern" signal. Tested two ways: a synthetic casual-to-academic style
  essay produced no flags (diffs too small to clear the floor — arguably
  correct, or arguably the classifier just isn't sensitive to that
  contrast yet, noted honestly rather than claimed as validated); a real
  human-sentence + real-AI-sentence splice correctly flagged the exact
  sentence where score jumped from 0.908 to 0.363.
- `schemas.py`/`main.py` — `EssayLevelFeatures` gained
  `sentence_length_variance` and `score_volatility` (beyond plan §6's
  original single-`burstiness`-field shape, same kind of documented
  deviation as Phase 5's `magnitude` field); `transitions` is no longer
  a hardcoded empty list.
- Frontend: `api.ts`, `VerdictBadge.tsx` (now shows score volatility
  too), `App.tsx` updated for the expanded types — `SentenceHighlighter`
  needed no changes, it already had unused transition-flag rendering
  wired up since Phase 2.
- `backend/scripts/build_hybrid_essays.py` — the 100 hybrid essays (plan
  §3b). Construction: real human essay truncated after K sentences (3 to
  70% of the essay), then an AI essay's opening M sentences (2-6)
  appended as a continuation — the "continuation" variant plan §3b
  explicitly allows, chosen over mid-essay paragraph replacement because
  it's simpler to get the ground-truth span exactly right. Verified the
  span offsets directly (not just trusted the arithmetic): sliced
  `text[ai_span_start:ai_span_end]` on a sample row and confirmed it's
  precisely the spliced-in AI sentences, splice boundary sitting cleanly
  on a sentence break. 100/100 pairs succeeded (no essays too short to
  splice). Sampled from the full 500/500 pools, seed=42, reproducible.
- `backend/scripts/score_hybrid_essays.py` — runs the full served
  pipeline over each hybrid essay, compares per-sentence predictions
  (ai_score >= 0.5) against the ground-truth span, reports
  precision/recall/F1 on the flagged span itself — exactly what plan §8
  asks for ("precision and recall on the flagged span itself against the
  known-replaced span, not just an essay-level right/wrong").

### Span-localization result

Sentence-level, all 100 hybrid essays combined: **precision 0.400,
recall 0.777, F1 0.528** (TP=290, FP=435, FN=83, TN=1043).

Read plainly: the model catches most of the spliced-in AI content
(78% recall) but over-flags a lot of the genuine human portion too (60%
of everything flagged AI is actually a false positive). This is
consistent with what addendum 5 already surfaced anecdotally (a real
human essay opener scored 88% AI on its own) — the classifier's current
decision boundary leans toward "AI" more readily than it should on
human text. Not a new problem, a second independent measurement of the
same one. This is exactly the kind of honest, load-bearing result plan
§8 wants surfaced, not softened — it'll be one of the concrete numbers in
`docs/EVALUATION.md` at Phase 7, alongside the in-theme/unseen-theme
numbers from addendum 5 and whatever DAIGT cross-genre check comes next.

### Next step

Phase 7 (full evaluation) is next. Everything it needs except the
subgroup-fairness section (blocked on Kaggle/PERSUADE/ELLIPSE) is now in
hand: the trained classifier, the in-theme/unseen-theme split numbers
(addendum 5), DAIGT for cross-genre (addendum 6), and now the
hybrid-essay localization numbers above. `docs/LIMITATIONS.md` can also
be written now — every limitation in it already exists somewhere in this
log, just needs consolidating into the plan's deliverable format.

---

## Phase 7 — Full evaluation

**Status: complete, except the one item that was blocked from the
start.** PERSUADE/ELLIPSE (Kaggle) never got unblocked this project, so
the subgroup-fairness section of `docs/EVALUATION.md` is honestly marked
N/A rather than faked. Everything else the plan's §8 asks for is done.

### What was built

- `backend/scripts/score_daigt.py` — stratified 200-essay sample (100
  human + 100 AI, seed=42) from the 44,868-row DAIGT pool, scored
  sentence-level against the trained classifier. Full 44,868 rows would
  take hours for a check the plan itself scopes as "optional subset."
- `backend/scripts/find_wrong_examples.py` — reused the existing
  feature_table_phase2.csv (no new GPT-2/spaCy passes needed for the
  FP/FN search) to find the most-confident false positive (highest-scoring
  real human essay) and false negative (lowest-scoring real AI essay),
  plus the worst-localized hybrid essay from Phase 6's per-essay
  breakdown. Attached each one's real `evidence.py` output so the
  write-up is grounded in actual coefficients, not speculation — the
  plan's explicit requirement for this deliverable.
- `docs/EVALUATION.md` — all 5 metrics sections plus the 3 wrong
  examples, written as the actual deliverable.
- `docs/LIMITATIONS.md` — consolidated from every addendum in this log
  plus what the evaluation itself surfaced, prioritized by what would
  close each gap.

### The headline result: DAIGT confirmed the single-generator risk empirically

This is the most important finding of the whole project so far. DAIGT
(different genre, different AI generators than Gemini) scored **51.5%
accuracy, AI recall 16.6%** — barely above chance. Every addendum since
Phase 1 flagged the single-generator limitation as a *risk*; this is the
first time it was actually *measured*, and it's worse than the in-theme/
unseen-theme numbers alone would have suggested. In-genre, same-generator
performance (73.6% in-theme) does not predict cross-generator performance
at all. Documented as the top item in `docs/LIMITATIONS.md`.

### Second finding: a traced (not hypothesized) fairness mechanism

The false-positive example (`hf-essayforum-4768`) isn't just "the model
was wrong" — its evidence output shows punctuation rate (98.8th
percentile, comma-heavy ESL-style listing) is what pushed it to a
false-positive-leaning score, outweighing a genuinely human-reading token-
rank signal. This turns the ESL-skew concern `data/README.md` flagged as
an open question back in Phase 1 into an observed mechanism — still not
a measured rate (that needs ELLIPSE, still blocked), but no longer purely
hypothetical either.

### Other results, briefly

- Unseen-theme split: accuracy 71.3% looks close to in-theme's 73.6%, but
  macro-F1 drops to 0.58 (from 0.73) — the honest number, given the
  unseen slice is itself 8:1 class-imbalanced for structural reasons
  (addendum 5's theme-selection tradeoff compounding here).
- Hybrid-essay localization: precision 0.40, recall 0.78, F1 0.53 —
  catches most spliced AI content but over-flags human text at a similar
  rate to what the false-positive example shows independently.
- Hybrid boundary-miss example surfaced a new, previously undocumented
  finding: short sentences (3-6 words) produce noisy, extreme-percentile
  features and are wrong in both directions — not something any earlier
  phase's addenda called out, discovered specifically by digging into a
  concrete failure case rather than aggregate metrics.

### Next step

Phase 8 is explicitly optional in the plan (Ghostbuster data, Binoculars
as an additional signal). The `docs/LIMITATIONS.md` priority list is the
more useful next-step guide if this project continues: Kaggle access
first (closes the biggest gap), then a held-out-generator AI batch to
turn the DAIGT finding into a precise in-genre measurement.

---

## Post-Phase-7 — threshold calibration and short-sentence handling

User asked, after seeing the evaluation results: "shouldn't we be fixing
our model if it can be made efficient?" Two fixes were identified as
achievable with data already in hand (no new essays, no Kaggle) — see
that conversation for the full options considered (multi-model AI data
and a LightGBM swap were also discussed and explicitly deferred, not
done here).

### What was built

- `train_classifier.py` now also saves the eval model's held-out
  predictions (`data/processed/eval_model_predictions.csv`) — sentence
  score, true class, sentence length, essay id, for both the in-theme and
  unseen-theme test splits. Deliberately the EVAL model's predictions
  (trained on 80% seen-theme data), not the production model's — the
  production model is trained on these same essays, so calibrating
  thresholds against its own predictions on them would be leakage. The
  resulting threshold *values* get applied to the production model
  afterward, which is standard practice and avoids the leakage.
- `aggregate.py` — added `MIN_RELIABLE_TOKENS = 4` and
  `is_low_confidence()`, empirically derived (not guessed): binned the
  eval predictions by sentence length and found human/AI mean-score
  separation was near-zero or inverted at 1-3 words, then stable from 4
  words on (see the length-vs-separation table computed during this
  session — 1 word: +0.08 separation, 2 words: -0.02, 3 words: +0.10, 4
  words: +0.19, climbing steadily after that). `essay_verdict()` now
  excludes low-confidence sentences from the mean-score calculation
  (falls back to using all sentences if every sentence in an essay is
  short, so it never returns on empty input).
- `schemas.py`/`main.py` — `SentenceResult.low_confidence: bool` added
  and populated; `essay_verdict()` now takes `sentence_lengths` too.
- `SentenceHighlighter.tsx` — low-confidence sentences render at reduced
  opacity with a dotted underline, tooltip explains why. Verified live in
  browser: a synthetic 1-word sentence ("Yes.") showed the visual
  treatment and was confirmed excluded from the essay verdict
  computation (verdict matched the other sentence's score alone).
- `calibrate_thresholds.py` — new script. Aggregates the eval
  predictions to essay-level mean (excluding low-confidence sentences,
  matching production logic exactly), sweeps a decision threshold for
  best *balanced* accuracy (chosen over raw accuracy specifically
  because raw accuracy rewards degenerate "always predict the majority
  class" thresholds — precisely the failure mode every placeholder-data
  phase of this project fought). Recenters the existing 0.30-wide
  Likely-Human/Likely-AI band on the empirically-best point instead of
  assuming 0.5. Also computes (but doesn't wire in) a best sentence-level
  threshold, "for future evaluation scripts" — see below for why that
  turned out to be the interesting part.
- Output: `backend/app/models/thresholds.json`. `aggregate.py` loads it
  with a fallback to the original 0.35/0.65 if the file doesn't exist,
  so the app still works pre-calibration.

### Result 1: essay-level threshold calibration — small, real, not dramatic

Essay-level balanced accuracy on the calibration set: 96.0% (original,
unfiltered, threshold=0.5) -> 97.2% (short-sentence filter only) ->
98.4% (filter + recentered threshold 0.52). Calibrated thresholds:
Likely Human <= 0.37, Likely AI >= 0.67 (was 0.35/0.65) — a small shift,
because the original placeholder thresholds turned out to already be
close to optimal once short-sentence noise is filtered out. Essay-level
numbers look dramatically better than the sentence-level ones throughout
this project (73.6% in-theme) because averaging ~15-20 sentences per
essay cancels out a lot of per-sentence noise — a normal, expected effect
of aggregation, not evidence the underlying problem (single-generator
training data, per DAIGT) is fixed.

### Result 2: sentence-level threshold recalibration — tested, and rejected

This is the more interesting finding. Re-scored the 100 hybrid essays
(`score_hybrid_essays.py`) with the calibrated sentence threshold (0.42)
to check whether calibration actually helped the task that matters most
(passage localization). It didn't, cleanly:

| Config | Precision | Recall | F1 |
|---|---|---|---|
| Baseline (t=0.50, all sentences) | 0.400 | 0.777 | 0.528 |
| Short-sentence exclusion only (t=0.50) | **0.409** | 0.777 | **0.536** |
| Threshold recalibration only (t=0.42, all sentences) | 0.357 | 0.874 | 0.507 |
| Both combined (t=0.42, excluded) | 0.363 | 0.874 | 0.512 |

Isolated the two effects with a one-off script (computed once per
sentence, evaluated at all 4 threshold/exclusion combinations from the
same pass — avoided 4 separate GPT-2 scoring runs) before deciding
anything, then deleted that script since it's not part of the permanent
pipeline. The short-sentence fix is a clean, unambiguous win in
isolation. The threshold change is not: lowering the bar from 0.5 to
0.42 flags more sentences as AI overall, which raises recall (catches
more of the true AI splice) but drops precision (more false alarms on
the human portion) — a real precision/recall tradeoff, not a bug.
Balanced accuracy (the objective calibration optimized for) barely
moved at the sentence level either (0.724 -> 0.729), so the tradeoff
wasn't even buying much on its own terms.

**Decision: adopted the short-sentence exclusion everywhere (including
`score_hybrid_essays.py`), explicitly kept the sentence-level threshold
at 0.5 rather than the calibrated 0.42.** Reasoning stated in
`score_hybrid_essays.py`'s docstring now: a false "this is AI-written"
flag on genuine student writing is the costlier error for this
application, so precision is weighted above the balanced-accuracy
objective threshold calibration was optimizing for. This is a judgment
call about error costs, not a purely statistical one — worth being
explicit that a different application (e.g. one where missing AI content
is costlier than false-flagging) might reasonably make the opposite
choice with the same numbers.

### Final adopted hybrid-essay numbers

Precision 0.409 (was 0.400), recall 0.777 (unchanged), F1 0.536 (was
0.528) — modest but real and honestly earned, not the result of picking
whichever number looked best after the fact. `docs/EVALUATION.md` §4
updated with these numbers and the full reasoning above; its "Model
under test" section now notes the post-calibration state and what was
tested-and-rejected, so a reader doesn't have to reconstruct this from
git history.

### What this means for "should we fix the model"

Both free-data fixes were worth doing, but neither is a substitute for
the thing that actually caps this classifier's real-world accuracy: the
single-generator training data (`docs/LIMITATIONS.md` #1, confirmed by
the DAIGT cross-genre result). Threshold tuning and noise-floor handling
are genuine, worthwhile engineering — they made the existing signal
slightly more precise and, more valuably, revealed that "optimize for
one metric" doesn't automatically transfer to a different downstream
task. They don't create new signal. That still requires either more
AI-generator diversity in training or accepting the measured limitation.

---

## Phase 1 addendum 7 — Kaggle unblocked, PERSUADE/ELLIPSE fairness measured

User provided a Kaggle API token and username directly in chat. Wrote
`~/.kaggle/kaggle.json` locally (outside the repo entirely, 600
permissions), never echoed the key value back in any response. Verified
auth worked (`kaggle datasets list`) and that both dataset slugs already
configured in `download_eval_datasets.py` (from back when this was
written pre-emptively, still blocked) were real, before running the real
download.

### What was built

- Ran `download_eval_datasets.py` for real for the first time. Got
  PERSUADE 2.0 (3 files; the one with demographics is
  `persuade_2.0_human_scores_demo_id_github.csv`) and ELLIPSE
  (`train.csv`). Sanity-checked immediately per that script's own
  docstring instructions: PERSUADE parsed to 25,996 real rows (raw
  `wc -l` showed 275,550 — misleading again, same embedded-newlines
  issue as the EssayForum CSV — parsed properly with `csv.DictReader`)
  with all 4 demographic fields present (`ell_status`, `race_ethnicity`,
  `economically_disadvantaged`, `student_disability_status`) and
  reasonable-looking value distributions; ELLIPSE parsed to 6,482 rows
  with ESL-writer holistic scoring fields. Both matched the plan's
  expectations closely enough to trust.
- `backend/scripts/prepare_fairness_evals.py` — new script. PERSUADE
  sampled stratified 150/150 by `ell_status` specifically (not a plain
  random sample) because ELL='Yes' is only ~8.6% of the raw pool
  (2,244/25,996) — a random 200-sample would carry ~17 ELL essays, too
  few to break out meaningfully against plan §8's explicit ask.
  ELLIPSE: plain random 200. Output: `persuade_eval.csv` (300 rows),
  `ellipse_eval.csv` (200 rows) — both eval-only, verified gitignored,
  same isolation as `daigt_eval.csv`.
- `backend/scripts/score_fairness.py` — new script. Both sources are
  100% genuine human writing (no AI class in either), so every non-
  "Likely Human" verdict is a real measured false positive. Uses the
  actual production pipeline end to end (same `aggregate.essay_verdict`
  the served app calls, including the low-confidence exclusion and
  calibrated thresholds from the prior addendum) — this measures
  deployed behavior, not a separate evaluation-only configuration.
  Reports both "strict" FPR (worst verdict, Likely AI) and "broad" FPR
  (anything short of Likely Human) since Inconclusive is a real cost
  too. Small subgroups (n<20) flagged in output, not silently averaged
  in as if equally reliable.

### The result: the biggest finding of the whole project

Strict FPR is 0.000 everywhere — the calibrated Likely-AI threshold
(0.67) is conservative enough that nothing in either out-of-genre human
sample crosses it outright. The broad FPR is where it lives:

| Group | Broad FPR |
|---|---|
| ELLIPSE overall | 7.5% |
| PERSUADE, non-ELL | 22.0% |
| PERSUADE, **ELL** | **36.7%** |
| PERSUADE, White | 18.5% |
| PERSUADE, Hispanic/Latino | 33.8% |
| PERSUADE, Black/African American | 36.1% |
| PERSUADE, Asian/Pacific Islander | 33.3% |
| PERSUADE, not economically disadvantaged | 18.4% |
| PERSUADE, **economically disadvantaged** | **33.1%** |
| PERSUADE, disability status | no gap (21.6% vs 28.4%, reversed) |

Every comparison except disability status points the same direction, at
1.7-2x. This is a real, load-bearing, measured result, not a projection
from the single traced false-positive example in addendum 5/6 — it
confirms that example was symptomatic of a general pattern, not a
one-off. Reordered `docs/LIMITATIONS.md` to put this first (previously
the biggest *open question*; now the biggest *measured finding*), with
the old #1 (single-generator/cross-genre) moved to #2.

Two caveats stated plainly in both `docs/EVALUATION.md` and
`docs/LIMITATIONS.md`, not hidden: this is a univariate breakdown
(demographic fields are likely correlated with each other, not isolated
as independent effects), and it measures cross-genre human writing
(persuasive, not admissions) rather than a direct in-genre measurement —
though there's no specific reason to expect the mechanism behaves
differently in-genre.

### What's actually left now

`docs/EVALUATION.md` is complete on every axis the plan's §8 asks for
except the held-out-generator precision measurement (user-deferred, not
blocked). `docs/LIMITATIONS.md`'s priority list was rewritten: closing
#1 (the fairness gap, now that it's measured and not hypothetical) is
the highest-value next step if this project continues, ahead of the
generator-diversity item that used to top the list.

---

## Post-Phase-7, round 2 — "finish everything, no limiting factor"

User's instruction: finish score_volatility integration and the fairness
investigation, then keep going on everything else remaining
(mid-paragraph hybrid variant, theme-tagging heuristic, top-level
README, test suite) before moving to further testing/tuning. Two items
explicitly NOT done, flagged rather than silently skipped: more AI
generator diversity (needs API keys not available; user already deferred
this) and Phase 8 (Ghostbuster/Binoculars — explicitly optional in the
plan, a large scope increase, held off pending explicit confirmation).

### score_volatility — wired in, deliberately weak

Before integrating, checked empirically whether volatility actually
separates "pure" essays from genuinely mixed ones: 327 pure training
essays vs. 30 hybrid (mixed) essays. Separation was weak at the median
(0.192 vs 0.204 — heavily overlapping) but real at the tail (pure
p95=0.256, hybrid max=0.321). Integrated as a confidence-only tripwire
(`aggregate.HIGH_VOLATILITY_THRESHOLD`, `essay_verdict()`) — above that
threshold, confidence is multiplied by 0.7. Deliberately did NOT make
this a stronger or graded signal, since the data doesn't support one;
doing so would repeat the exact "oversell a weak signal" mistake the
rejected sentence-threshold recalibration already taught this project to
avoid.

### The fairness investigation — the most substantial work this round

`investigate_fairness_bias.py`: re-featurized the 300-essay PERSUADE
sample, compared raw feature values and per-feature classifier
contributions between ELL and non-ELL groups. Raw differences were
mostly GLTR statistics (ELL essays less predictable to GPT-2 generally —
higher perplexity, fewer top-10-predicted tokens). The *contribution*
view (coefficient x standardized value — what actually drives the
verdict) isolated one feature, `gltr_pct_top10`, as dominant: contribution
delta +0.50 between groups, ~3x the next-largest feature. Mechanism: the
classifier had learned "high top-10% = human-like" from this project's
training data's specific register (EssayForum's plainer prose vs.
Gemini's more elaborate prose) — not a generally reliable signal — and
ELL writers land in the low-top-10% zone for an unrelated, legitimate
reason (non-native phrasing patterns), absorbing a penalty the feature
was never measuring correctly for them.

`experiment_drop_feature.py` tested the direct hypothesis — retrain
without that one feature, using the exact same methodology as
`train_classifier.py` — **before** touching the production model:
- Core accuracy cost: small (in-theme 73.6% -> 72.0%, unseen-theme
  71.3% -> 70.0%).
- Fairness gap: roughly halved (ELL gap 14.7 -> 7.3 points in the
  isolated experiment).

Judged worth adopting given the fairness gap is this project's most
consequential limitation. Retrained the actual production model
(`featurize.py`'s `FEATURE_NAMES` now excludes `gltr_pct_top10`,
documented in place with the full reasoning, not just removed silently).

### Theme-tagging fix — larger and more consequential than expected

Separately, while investigating fairness, revisited the long-standing
"theme-tagging is an unverified heuristic" limitation. Realized the AI
essays never needed a heuristic at all — they carry a *known* true
category (Gemini's 50 generation categories, captured as
`source_category` since addendum 4 but never used for `theme_id`). Built
`theme_mapping.py`, a curated 50-category -> 7-theme lookup, covering
every category (verified against the real data, zero missing). Also
improved the human-essay keyword heuristic (`infer_theme()` in
`build_dataset.py`): word-boundary regex instead of substring matching,
most-keyword-hits-wins instead of first-match-wins, expanded keyword
lists.

Result: AI-side theme distribution went from ~89% concentrated in 2 of 7
themes to exactly matching the known category counts. Human-side
improved substantially but not fully (`background_identity` 30 -> 93;
`growth_accomplishment`/`obstacle_setback` remain the two largest
categories by a wide margin, plausibly a real property of what people
write about, not just a tagging artifact).

**This forced re-picking `HELD_OUT_THEMES`** (the old smallest-3 —
`background_identity`/`captivating_topic`/`gratitude` — were smallest
specifically because the AI side was broken; recomputed smallest-3 under
the fixed distribution: `challenging_belief`/`gratitude`/
`obstacle_setback`) **and retraining/recalibrating everything
downstream**, since `feature_table_phase2.csv`'s `theme_id` column and
the eval-model's train/test split both depend on it. Patched
`theme_id` in the existing feature table via essay-id lookup rather than
re-running the full GPT-2/spaCy pass (only labels changed, not text —
re-extracting features would have been pure wasted compute).

**The result of fixing this was the single most surprising finding in
this whole round:** the unseen-theme evaluation's macro-F1 gap (0.73
in-theme vs. 0.58 unseen, previously read as "a genuine, if
data-artifact-influenced, generalization gap") nearly disappeared (0.71
vs. 0.71) once theme tagging was fixed. The original gap was
*substantially* a class-imbalance artifact — the old held-out themes
happened to be exactly where the broken AI tagging left almost no AI
essays (`background_identity` had 1 AI essay out of 500), making the
unseen-theme test set 8:1 imbalanced by accident, not by anything about
the model's real topic-generalization ability. **A data-labeling bug
had been read as a model-capability finding.** Worth stating plainly as
a lesson, not just a fixed number: check the labels before trusting a
generalization-gap result.

### Full re-evaluation chain, in order

Because the production model changed (feature set) and the theme labels
changed (affecting the eval-model split), the entire evaluation suite
needed re-running for internal consistency — a doc describing a model
that isn't actually deployed would be worse than not having the doc:
1. `train_classifier.py` (retrain eval model + production model,
   regenerate `eval_model_predictions.csv`) — production model itself
   is trained on all 1000 essays regardless of theme, so this step
   doesn't change classifier.joblib's weights vs. the gltr-drop
   experiment, only the reported in-theme/unseen-theme numbers.
2. `calibrate_thresholds.py` — thresholds landed back at almost exactly
   0.35/0.65 (the original Phase 2 placeholder values) once both fixes
   were in place — a mild, reassuring result suggesting the placeholders
   weren't badly wrong to begin with, just uncalibrated.
3. `score_daigt.py`, `score_hybrid_essays.py`,
   `score_hybrid_essays_midparagraph.py` (new), `score_fairness.py` (run
   twice — once mid-flight with stale thresholds for a sanity-check
   comparison, once fresh with final thresholds for the authoritative
   numbers).
4. `find_wrong_examples.py` re-run for fresh, consistent examples (the
   old ones referenced `gltr_pct_top10` percentiles that no longer
   exist in the model's feature set).

### Net result across the whole re-evaluation

- DAIGT cross-genre: improved (51.5% -> 54.6% accuracy, AI F1 0.246 ->
  0.333) — the fairness fix helped general robustness too, not only the
  fairness axis specifically.
- Unseen-theme: transformed from an apparent generalization problem
  (macro-F1 0.58) to near-parity with in-theme (0.71 vs 0.71) — mostly
  the theme-tagging fix, not the feature-drop.
- Hybrid localization: both variants (continuation and the new
  mid-paragraph one) got slightly worse (continuation F1 0.536 -> 0.513)
  — a real, disclosed cost of the feature-drop on this specific task.
- Fairness: ELL/economic-disadvantage gaps shrank ~80%, but — the
  important complication — **every group's absolute FPR rose**,
  including the "advantaged" baseline (non-ELL 22.0% -> 36.7%, White
  18.5% -> 34.6%). The gap closed mostly because the baseline rose to
  meet the disadvantaged rate, not because disadvantaged outcomes
  improved in absolute terms. Race/ethnicity gaps (Black, Asian/Pacific
  Islander vs. White) only partially closed — `gltr_pct_top10` was
  clearly not the whole mechanism for the racial disparity specifically.
  Stated plainly in every doc touching this number, not smoothed over:
  this is a real, disclosed tradeoff, not a clean win.

### Mid-paragraph hybrid essay variant

`build_hybrid_essays_midparagraph.py` — the plan's other-described
splice method (AI text replacing a middle span, human text on both
sides), deferred since the original Phase 6 build in favor of the
simpler continuation method. Same span-offset verification discipline as
before (checked a sample row's `text[ai_span_start:ai_span_end]` directly
against the construction). Result: meaningfully harder than continuation
(F1 0.405 vs 0.513) — a false-alarm surface on both sides of the AI
splice instead of one. Read as the more realistic real-world case.

### What's genuinely left now

`docs/LIMITATIONS.md`'s priority list was rewritten again to reflect
this round: resolving the overall-FPR-vs-disparity tradeoff (a
genuinely open, disclosed, unresolved question — not a bug to fix, a
values question about what to optimize for) is now the top item, ahead
of generator diversity. Two items outside this session's reach:
multi-model AI data (needs API keys) and Phase 8 (explicitly optional,
awaiting explicit go-ahead).

### Top-level README.md and pytest suite

`README.md` rewritten per plan §10: what it does, the LM-as-instrument
design principle, stack, local run instructions, an evaluation summary
with the actual current numbers, and the two most important limitations
stated plainly up front rather than buried in a linked doc.

`backend/tests/` was completely empty before this round — added 41 tests
across 4 files: `test_aggregate.py` (pure logic, no model loading, fast —
verdict thresholding, low-confidence exclusion, volatility penalty,
transition detection), `test_featurize.py` (segmentation, feature
extraction, the `gltr_pct_top10`-excluded-from-`FEATURE_NAMES` regression
guard), `test_classify_and_evidence.py` (against the actual shipped
`classifier.joblib`, not a mock — verifies the deployed model loads and
behaves sanely), `test_api.py` (FastAPI `TestClient`, full `/analyze`
request/response shape).

**The suite caught a real bug on its first run, not a test-writing
mistake:** `POST /analyze` with empty essay text crashed with a 500 —
`gltr.token_stats()` tokenizes empty text to a 0-length tensor, and
GPT2LMHeadModel's forward pass can't reshape that (`RuntimeError: cannot
reshape tensor of 0 elements`). Fixed with an early return in
`token_stats()` for empty/whitespace-only input. This was a real,
previously-undetected crash path in the served API — the kind of thing
manual browser testing doesn't reliably hit but a test suite does. Also
caught a wrong assumption in the tests themselves while writing them:
`detect_transitions()`'s relative-to-own-baseline algorithm can miss a
transition when there are two comparably large jumps in the same short
essay (they normalize each other out statistically) — not a bug, a real
property of the algorithm worth knowing, documented in the test that
found it rather than silently worked around.

All 41 tests pass. `pytest`/`httpx` added to `requirements.txt`.

## Post-Phase-7, round 3 — "finish everything unfinished, then move to testing"

User's instruction: finish every remaining fixable item (not the
open FPR-vs-disparity values question, already surfaced as a deliberate
non-fix) before moving to a testing-cases phase. Three substantive items
this round: the real short-sentence fix (context-merging, replacing the
round-2 exclusion workaround), meta-commentary stripping at the
ingestion source, and the race/ethnicity fairness investigation.

### Short sentences: merging replaces exclusion

Round 2's fix (`aggregate.MIN_RELIABLE_TOKENS`-based exclusion) fixed the
essay-level verdict but left short sentences literally unscored in the
UI — no color, no evidence, nothing for the user to look at. The real
fix, `featurize._scoring_spans()`: for each sentence under the
reliability floor, compute its features from a span merged with the next
sentence (or the previous one, at the essay's end) instead of itself
alone; a lone short sentence with no neighbor (single-sentence essay) is
left unmerged since there's nothing to merge with. `token_stats` (the
UI's token heatmap) and the sentence's own displayed boundaries are
unaffected — only score/evidence computation uses the wider span.
`aggregate.essay_verdict()` lost its `sentence_lengths` parameter and all
exclusion logic entirely; every sentence's score is real and
context-informed by the time it reaches that function. `schemas.py`'s
`SentenceResult.low_confidence` became `context_merged`, with revised UI
tooltip text ("too short to score alone, reflects a merged span") rather
than an implicit "don't trust this" framing.

Swept every script that had its own `is_low_confidence`-based exclusion
copy (`calibrate_thresholds.py`, `score_hybrid_essays.py`,
`score_hybrid_essays_midparagraph.py`, `investigate_fairness_bias.py`,
`experiment_drop_feature.py`, `score_fairness.py` — the last one missed
by an initial grep since it called `essay_verdict(scores, lengths)`
directly rather than `is_low_confidence`, caught when it crashed with a
`TypeError` on the new one-argument signature) — all now just use every
sentence's score, no exclusion. Deleted `aggregate.is_low_confidence()`
itself once nothing called it anymore.

Fixed a real bug found while re-running `extract_features.py` after this
change: it writes `row.update(sf.features)` (the full raw feature dict,
which still contains `gltr_pct_top10` even though that's excluded from
`FEATURE_NAMES`) against a `DictWriter` whose fieldnames come from
`FEATURE_NAMES` — a pre-existing crash that had simply never been hit
since `extract_features.py` wasn't re-run after the `gltr_pct_top10`
removal in round 2 (that round's model was trained via
`experiment_drop_feature.py`'s own separate feature computation, not
this pipeline). Fixed by filtering the row to `FEATURE_NAMES` explicitly.

### Meta-commentary stripping

EssayForum posts are drafts submitted for feedback; some open or close
with text addressed to forum reviewers rather than the essay itself
("Let me know what you think of my essay (first draft)... Hopefully it
isn't too honest and makes me look bad."). `hf_essayforum.py` now
sentence-segments each essay (reusing `app.pipeline.segment`) and drops
any sentence matching a curated regex of review-request phrasings,
*before* the word-count filter — an essay that only clears 300 words
because of the stripped text is correctly excluded rather than silently
shortened below the intended floor. One false-positive pattern caught
and tightened during development: an early loose "let me know" pattern
matched a legitimate narrative sentence ("my mom called to let me know I
had been automatically admitted"); narrowed to specific directive
phrasings ("let me know what you think", "please let me know", etc.).
Verified zero remaining matches across the final 500 essays (was 69/500
before the fix). Re-ran `build_dataset.py`: still 500 essays, theme
distribution shifted slightly since a different subset now clears the
word-count floor.

**This fix's effect wasn't limited to training data** — `hybrid_essays.csv`
and `hybrid_essays_midparagraph.csv` are built by splicing directly from
`human_essays.csv`/`ai_essays.csv`, and both had been built *before* this
fix refreshed the human pool. `find_wrong_examples.py`'s worst-hybrid
example surfaced this directly: the essay literally opened with the
un-stripped meta-commentary text, which the classifier correctly flagged
as anomalous. Since both hybrid builders are deterministic given their
seed, rebuilding them required no code change — just a re-run against
the now-current source data. Re-scoring after the rebuild recovered
precision on both variants (continuation F1 0.513 -> 0.536, mid-paragraph
0.392 -> 0.398) — the previous round's small fairness-fix cost on this
task turned out to be partly a stale-data artifact, not purely the
feature removal.

### Race/ethnicity fairness investigation

Same methodology as round 2's ELL investigation, applied to the
Black/African American vs. White PERSUADE gap (both groups above the
reliable-sample-size floor, unlike the two smallest race categories).
New script `investigate_race_fairness.py`: re-featurized the 300
PERSUADE essays, compared per-feature raw values and per-feature
classifier contributions between the two groups. The contribution view
(coefficient x standardized value — what actually drives the verdict)
isolated four features doing most of the work: `gltr_pct_top1000`
(contribution delta +0.122), `perplexity` (+0.101), `rolling_ttr`
(+0.098), `gltr_mean_rank` (+0.078) — the same underlying mechanism as
the ELL finding (GPT-2 predictability gap), spread across features
instead of concentrated in one clearly-redundant one like
`gltr_pct_top10` was.

New script `experiment_drop_feature_race.py` tested two variants:
dropping `gltr_pct_top1000` alone shrank the gap from 12.1 to 6.5 points
at a 2.7-point unseen-theme accuracy cost (0.720 -> 0.693); dropping it
together with `gltr_mean_rank` nearly closed the gap (1.2 points) at a
3.0-point cost (0.720 -> 0.690). Unlike the `gltr_pct_top10` removal,
neither variant is a clean, low-cost win.

**Presented this tradeoff to the user directly rather than deciding
unilaterally**, since it's a genuine values question this project had
already established shouldn't be resolved silently (the earlier-rejected
sentence-threshold recalibration set that precedent). The user pointed
back to the original project brief and asked for the decision that best
fits it. Two lines in the brief are directly on point: it frames spotting
and disclosing ELL-style bias as the valued outcome ("these detectors
have a habit of flagging writers who learned English as a second
language... we'd like to know you spotted it") rather than a mandate to
engineer it away regardless of cost, and it separately rewards honest
accuracy reporting over chasing a metric. **Decision: keep the current
production model (12 features, `gltr_pct_top1000`/`gltr_mean_rank`
retained), document the race/ethnicity gap and the declined mitigation
plainly in the docs.** No retraining needed — production was already the
post-context-merging round-3 model.

### Full re-evaluation

Re-ran the full chain end to end since both feature computation
(context-merging, meta-commentary stripping) and training data changed:
`extract_features.py` -> `train_classifier.py` -> `calibrate_thresholds.py`
-> `score_hybrid_essays.py` / `score_hybrid_essays_midparagraph.py` (after
rebuilding those sets) -> `score_daigt.py` -> `score_fairness.py` (fixed
to drop its own stale `essay_verdict(scores, lengths)` call) ->
`find_wrong_examples.py`. Net movement, all improvements:

- In-theme / unseen-theme: 0.722 / 0.720 (gap 0.2pp, was 0.715/0.715 —
  noise-level difference, not a regression; theme-tagging's fix already
  did the real work here in round 2).
- DAIGT cross-genre: 54.6% -> 56.1% accuracy, AI F1 0.333 -> 0.370 —
  third consecutive round of improvement, most plausibly from cleaner
  training signal (meta-commentary stripped, short sentences no longer
  discarding information) rather than any feature-set change.
- Hybrid localization (continuation): F1 0.513 -> 0.536, recovering
  essentially all of round 2's small fairness-fix cost, once the stale
  contaminated eval essays were replaced.
- Hybrid localization (mid-paragraph): F1 0.392 -> 0.398, smaller move,
  same direction.
- Fairness: ELL gap 4.7pp, economic-disadvantage gap ~1.0pp (both small
  and stable versus round 2's post-fix numbers), Black-vs-White gap
  12.1pp (down from round 2's 15.4pp, but not from this round's
  changes specifically — same untouched feature set, most likely
  reflects the recalibrated thresholds and cleaner short-sentence
  scoring shifting the whole score distribution slightly).

Updated `docs/EVALUATION.md`, `docs/LIMITATIONS.md`, `data/README.md`,
and the top-level `README.md` throughout with every number above, the
race/ethnicity investigation and declined-mitigation reasoning, and the
data-staleness discovery/fix for the hybrid essay sets. Full pytest
suite re-run after every pipeline change (context-merging, then again
after retraining) — 38 tests pass (3 fewer than round 2's 41: two
`low_confidence`-exclusion-specific tests in `test_aggregate.py` no
longer apply now that merging replaced exclusion, and were removed
rather than kept as dead assertions against a removed code path;
`test_api.py` gained a replacement pair of context-merging tests).

## Post-Phase-7, round 4 — accuracy investigation and threshold-width fix

User asked directly: how is the verdict decided, what's the actual
accuracy, is PERSUADE/ELLIPSE doing its job, and is the Inconclusive
threshold too wide. Explained the pipeline and current numbers, then
proposed a safe-first approach to improving accuracy: test a nonlinear
classifier and sweep the threshold band width (both reversible, no data
risk) before considering folding DAIGT's multi-generator AI essays into
training (a bigger lever with a real in-genre-accuracy risk). User
agreed to the safe experiments first and is separately collecting more
diverse AI-generated essays for a future round.

### Nonlinear classifier: tested, not adopted

New script `experiment_nonlinear_classifier.py`: same 12 features, same
essay-level train/test/unseen split as `train_classifier.py`, same
stratified DAIGT-v2 sample as `score_daigt.py` (features computed once,
shared across all three models for a fair, efficient comparison).
RandomForest and GradientBoosting both beat the current
LogisticRegression in-genre (0.722 -> 0.756 / 0.765 in-theme; 0.720 ->
0.741 / 0.756 unseen-theme) but got *worse* cross-genre (DAIGT: 0.553 ->
0.502 / 0.507, close to chance) — real signal in feature interactions
that helps within the training distribution, but overfits to Gemini's
specific quirks in a way that hurts transfer to a genuinely different
genre/generator more than the simpler linear model does. Not adopted:
cross-genre is the more consequential weakness, and this trade makes it
worse. Flagged as worth retesting once the user's more diverse
AI-generator data is in — a nonlinear model might generalize better with
real cross-generator patterns to learn from instead of overfitting to
one generator's noise.

### Threshold band width: a real, previously-uncalibrated parameter

Every prior threshold recalibration only ever tuned the band's *center*
— the ±0.15 half-width itself (giving the 0.35/0.65-ish cutoffs) was the
original Phase 2 placeholder, carried forward unchanged and unquestioned
through every subsequent recalibration. New script
`experiment_threshold_band_width.py` swept half-width directly against
the same held-out eval-model data `calibrate_thresholds.py` uses: at
±0.15, **73.4% of held-out essays got no definitive verdict at all**
(100% accuracy on the 26.6% that did). Narrower widths trade a small
amount of that near-perfect definitive accuracy for a much higher
decision rate (±0.05: 12.4% Inconclusive at 98.2% accuracy; ±0.08: 27.4%
at 98.9%; ±0.10: 40.6% at 99.6%). Presented the full tradeoff table to
the user with the important caveat that this is measured on
in-distribution essays only — a narrower band is riskier specifically on
the cross-genre essays the model is already weak on, since it can't
detect at inference time whether a given essay is in- or
out-of-distribution. **User chose ±0.08.**

`calibrate_thresholds.py`'s `BAND_HALF_WIDTH` constant updated from 0.15
to 0.08 (with the reasoning and the tradeoff-table numbers in a comment,
not just the bare value) and re-run: new band is 0.43-0.59 (was
0.36-0.66). Backend restarted to pick up the new `thresholds.json`
(`aggregate._thresholds()` is `lru_cache`d at process start, so a
running server doesn't see a regenerated file without a restart — caught
this by curling `/analyze` against a stale-looking response before
realizing the old uvicorn process, leftover from before this session's
edits, was still bound to port 8000 and needed to be killed first).

**Re-scored PERSUADE/ELLIPSE fairness with the new band**
(`score_fairness.py`, which calls `aggregate.essay_verdict()` directly
and so is the only eval script affected by this change — the hybrid and
DAIGT scripts use a fixed sentence-level 0.5 threshold, not the essay-
level band). Large, mostly-unplanned side benefit: broad FPR (the
"Inconclusive counts as a cost" metric) dropped sharply everywhere
(PERSUADE overall 31.7% -> 10.7%, ELLIPSE 7.5% -> 1.5%) simply because
far fewer essays now fall into the wide dead zone. Most fairness gaps
compressed as a side effect too (Black-vs-White 12.1 -> 4.0 points,
Asian-vs-White 6.8 -> 2.2, disability reversal -18.8 -> -4.6). Strict FPR
(wrongly-confident "Likely AI" on real human writing) stayed effectively
zero — 0.000 on ELLIPSE, 0.003 on PERSUADE (about 1 essay of 300 moved
from Inconclusive into a wrong confident call) — the real, small,
disclosed cost of narrowing the band. One gap moved the "wrong" way in
relative terms even as absolute numbers improved: economically-
disadvantaged vs. not went from a 1.0-point gap to 4.2 points (11.0% vs.
6.8%, both much smaller than the ~30%+ pre-ELL-fix numbers, but the
disadvantaged group's rate didn't fall quite as fast as the baseline's)
— flagged rather than smoothed over.

Updated `docs/EVALUATION.md`, `docs/LIMITATIONS.md`, and the top-level
`README.md` with the nonlinear-classifier comparison, the threshold-width
tradeoff table and decision, and the refreshed fairness numbers. pytest
suite re-run after the threshold change (band width doesn't touch the
model or features, only `thresholds.json`) — still 38/38 passing. Full
end-to-end browser verification: confirmed an essay whose mean score
used to fall inside the old dead zone now gets a definitive "Likely AI"
verdict instead of "Inconclusive."

## Post-Phase-7, round 5 — diversifying the human training pool

User asked me to source additional human essays myself (they're
separately getting Mistral/Llama AI essays), explicitly requesting every
blocker be surfaced up front this time rather than drip-fed — a direct
response to round 4's advice ("get a second human source, optional, for
later") turning out to need real investigation before it could be acted
on.

### Checked real yield before promising a number, this time

Round 4 estimated "~200-300 more human essays from a second source"
without checking actual availability. Before repeating that mistake,
live-tested the three previously-built adapters
(`sources/openessays.py`, `conncoll.py`, `emoryadmission.py`):
9 essay pages on conncoll.edu, 6 posts on emoryadmission.com
(potentially multiple essays per post), 14 undergrad essays on
openessays.org. This matched `data/README.md`'s own already-recorded
history — these three sources combined only yielded ~30 essays the
first time they were tried, years before EssayForum was found — a fact
that was already written down in this project's own docs and should
have been checked before round 4's estimate, not after. Also searched
for a larger published-essay dataset (HF/Kaggle) following the pattern
that solved this exact problem for EssayForum — found nothing usable
(several college-admissions-adjacent datasets exist but none contain
real personal-statement essay text at volume).

Surfaced the real number (30, not 200-300) to the user directly with
the reasoning, rather than silently scaling back the plan. **User chose
to proceed with 30 rather than chase more sources** — `class_weight=
'balanced'` means the value is a second population to calibrate
"human-like" against, not matching EssayForum's volume.

### Re-enabled three sources, added name redaction that wasn't fully verified before

`build_dataset.py` now calls all four source adapters (previously
EssayForum-only) and merges them. Two of the three re-added sources
(conncoll.edu, openessays.org) publish under real individual names;
`emoryadmission.com` is anonymous by design. Added `_redact_name()`:
replaces the full captured author name AND each individual name part
(≥3 characters, so a first-name-only or last-name-only mention elsewhere
in the essay is caught too) with `[name redacted]`, word-boundary
matched, case-insensitive, applied before any text is written to disk.
This closes a gap that existed but was never fully closed before —
`docs/LIMITATIONS.md` had flagged "openessays.org essays carry real
names" as an open concern from when these sources were first used, and
redaction hadn't actually been verified end-to-end. Verified this round
with a proper `\b`-bounded check across all 23 essays with a captured
author name: zero leaks (an initial naive substring-only check flagged 3
false positives — e.g. author name "Das" inside "coronavirus" — that
the real word-boundary-matched redaction had already handled correctly).
Also added a generic email/phone regex safety net across all four
sources, matching the pattern already used for the EssayForum mirror.

Real yield: EssayForum 500 (unchanged) + emoryadmission.com 7 +
conncoll.edu 9 + openessays.org 14 = **530 total human essays**.

### Full pipeline re-run

`build_dataset.py` -> rebuilt both hybrid essay sets (they sample from
`human_essays.csv`, deterministic given the same seed, no code change
needed) -> `extract_features.py` (1030 essays) -> `train_classifier.py`
-> `calibrate_thresholds.py` (kept the user's ±0.08 band width from
round 4) -> re-scored hybrid/DAIGT/fairness -> `find_wrong_examples.py`.
`HELD_OUT_THEMES` unchanged (`challenging_belief`, `gratitude`,
`obstacle_setback` remain the 3 smallest combined human+AI themes after
the addition — checked, not assumed).

Net movement:
- In-theme / unseen-theme: 0.722 / 0.711 (gap widened slightly from 0.2
  to 1.1 points — still small, not a regression worth acting on).
- DAIGT cross-genre: 56.1% -> 56.0%, essentially flat — expected, since
  this round targets the human reference distribution and fairness, not
  AI-generator diversity, which is what actually drives cross-genre AI
  detection.
- Hybrid localization: both variants essentially flat (continuation F1
  0.536 -> 0.538, mid-paragraph 0.398 -> 0.392) — same reasoning.
- Fairness: the round-4 threshold-narrowing improvement held and
  extended further on top of it. ELLIPSE broad FPR 1.5% -> 0.5%,
  PERSUADE overall 10.7% -> 9.7%. ELL gap 4.0 -> 3.3 points, Asian-vs-
  White gap reversed (Asian now *below* White, -3.8 points, was +2.2),
  disability reversal -4.6 -> -3.3 points. **Black-vs-White gap
  unchanged at 4.0 points** — the same 36-essay subgroup's individual
  verdicts didn't cross the decision boundary this round; none of the
  three newly-added sources carry race/ethnicity fields to have
  influenced it directly in the first place. Economically-disadvantaged
  gap essentially flat (4.2 -> 4.0 points).

Updated `data/README.md` (new source provenance, redaction
verification, real-vs-hoped-for yield), `docs/EVALUATION.md` (round 5
section, all numeric tables), `docs/LIMITATIONS.md` (#1 fairness numbers,
#3 rewritten — population is no longer single-source but still
EssayForum-dominated), and the top-level `README.md`. Full pytest suite
re-run after retraining — 38/38 still passing. Backend server restarted
(killed a stale leftover uvicorn process from earlier in the session that
had briefly caused a confusing mismatch between curl results and the
current code/model) and re-verified end-to-end in the browser.

## Round 6 — AI-generator diversification (declined), classifier swap (adopted)

User's explicit priorities this round, stated directly: get real
accuracy up ("we need to get our accuracy up srsly"), and later,
narrower: "a little bit more accuracy... presentable to judges," ahead
of a hackathon deadline. Three levers were tried in sequence, each
tested before being adopted or declined — no change shipped on
assumption alone.

### Lever 1: AI-generator diversity (declined)

User sourced 200 real essays: 100 from OpenAI (`gpt-5.6-luna`), 100 from
Anthropic (`claude-haiku-4-5-20251001`), same 500-prompt admissions set
as the existing Gemini essays, delivered via an external pipeline folder
(`other_ai_pipeline/`). `ingest_ai_essays.py` rewritten to handle all
three sources: stripped a real markdown-artifact contamination
(100/100 Anthropic essays opened with a literal `# Title` header — a
trivial, ungenuine "tell" that would have let the classifier learn
"has markdown" instead of real style) via `_strip_markdown()`, and
built a held-out-generator split (20 OpenAI + 20 Anthropic held out
entirely from training, seed=44) into
`data/processed/ai_essays_heldout_generator.csv` — eval-only, same
isolation pattern as DAIGT/PERSUADE/ELLIPSE. New script
`score_heldout_generator.py` measures sentence- and essay-level AI
recall on those 40 never-trained-on essays specifically, broken out by
generator — more precise than DAIGT for the "does more generator
diversity help" question, since DAIGT conflates genre and generator
into one number.

Full pipeline re-run on the 660-AI-essay (3-generator) set: in-theme
72.2% -> 71.0%, unseen-theme 71.1% -> 69.8%, DAIGT 56.0% -> 56.5%
(~flat), hybrid localization recall dropped ~12pt on both splice
variants, most fairness gaps got worse (Black-vs-White 4.0 -> 6.8pt),
ELL gap improved (3.3 -> 0.6pt). Held-out-generator recall (first-ever
run): only 30% essay-level recall overall (10% GPT, 50% Claude) — even
essays from generators *represented* in training weren't reliably
caught on new examples. **Declined** — reverted the shipped model to
Gemini-only AI training data (`train_classifier.py`'s
`EXCLUDED_AI_ID_PREFIXES`, `build_hybrid_essays*.py` filtered to match)
which cleanly restored the exact round-5 baseline numbers, confirming
the revert was correct. The OpenAI/Anthropic data and the held-out-
generator methodology stay in the repo, re-scored against every
subsequent model change below rather than deleted.

### Lever 2: regularization tuning (dead end)

New script `experiment_tune_regularization.py` swept
`LogisticRegression`'s C from 0.01 to 30 on the Gemini-only data: in-
theme/unseen-theme/DAIGT accuracy were flat to three decimal places
across the entire range (0.720-0.722 / 0.711 / 0.552-0.553). Not a lever
at all with 12 features and this much training data — no change made.

### Lever 3: classifier swap, LogisticRegression -> GradientBoostingClassifier (adopted)

`experiment_nonlinear_classifier.py` (built two rounds ago, re-run this
round with a matching Gemini-only-data filter for a clean comparison)
had already shown RandomForest/GradientBoosting beating the linear model
in-genre but losing cross-genre. Re-run specifically to check whether
the (declined) generator-diversity data would change that calculus for
a nonlinear model — it didn't (same ~4-5pt cross-genre cost either way,
confirming the in-genre/cross-genre tradeoff is about the classifier
family, not the AI-generator diversity of the training data).

Given that neither lever above delivered accuracy without a real cost,
and the round's explicit priority was accuracy for a demo,
`GradientBoostingClassifier(n_estimators=200, max_depth=3)` (sample-
weight-balanced, since it has no `class_weight` param) was adopted for
both the eval and production model in `train_classifier.py`, trained on
the Gemini-only data. This is a real architecture change, not a
hyperparameter tweak, and it broke one thing that needed a real fix, not
a workaround: `evidence.py`'s per-sentence "why" explanation was
`coefficient x standardized_value`, meaningless for a model with no
`.coef_`. Replaced with a perturbation-based method — swap each feature
to the human-training-class median, re-score, take the change in P(AI)
as that feature's contribution (`classify.py` gained
`predict_proba_from_scaled()` and `scaled_value()`; `classify.coefficients()`
removed). Still fully deterministic, still model-agnostic. Two tests in
`test_classify_and_evidence.py` that asserted on the now-removed
`coefficients()` were replaced with tests on the new functions; the
existing evidence-shape tests (`top_features` returns <=3, sorted by
magnitude, valid percentile range) needed no change since the
`FeatureContribution` interface didn't change. Full pytest suite: 39/39
passing after the swap (38 + 2 new − 1 removed).

Full pipeline re-run: train -> calibrate -> rebuild/rescore both hybrid
sets -> DAIGT -> fairness -> held-out-generator -> find_wrong_examples ->
pytest, all chained in one background run. Net movement, all real and
measured, none assumed:

- In-theme / unseen-theme: 72.2%/71.1% -> **76.3%/74.1%** (+4.1/+3.0pt) —
  the direct, intended gain.
- DAIGT cross-genre: 56.0% -> **51.2%** (essay-level 55.5% -> 51.0%,
  barely above chance) — the direct, disclosed cost, the expected
  overfit-to-training-distribution tradeoff.
- Hybrid localization: continuation F1 0.538 -> **0.606**, mid-paragraph
  0.392 -> **0.427** — unplanned but real improvement, same in-genre
  sharpening that hurt DAIGT helps here since these essays are Gemini-
  sourced AI spliced into EssayForum-sourced human text.
- Fairness: large, across-the-board improvement, unplanned. PERSUADE
  overall broad FPR 9.7% -> **3.7%**, ELLIPSE 0.5% -> **0.0%**. Every
  demographic gap in `docs/EVALUATION.md` §5 shrank to ≤2 points or
  reversed, including Black-vs-White (was 4.0pt, now reversed -2.1pt) —
  a gap that had specifically plateaued through round 5's targeted
  fixes moved substantially here, as a side effect of a different model
  family, not a targeted fix. Flagged explicitly in
  `docs/LIMITATIONS.md` as "measured small," not "provably fixed,"
  since the mechanism isn't individually traced the way the linear
  model's `gltr_pct_top10` fix was.
- Held-out-generator recall (re-scored against the Gemini-only GBM
  model): 62.7% sentence-level, **7.5%** essay-level (worse than the
  10% under the linear Gemini-only model, and worse than the declined
  3-generator model's 30% — a sharper classifier generalizes even less
  to genuinely unseen generators).

Verified end-to-end, not just via the eval scripts: restarted the
backend uvicorn process (thresholds are `lru_cache`d, need a fresh
process), pasted a real essay in the browser, confirmed the evidence
panel renders sane percentiles/directions/magnitudes sorted correctly
via both the raw `/analyze` JSON and a screenshot of the rendered panel.
Fixed two small stale UI strings found during that check: `EvidencePanel.tsx`
still said "trained-classifier coefficients" (no longer true), and
`App.tsx`'s header banner still said "500 human" (stale since round 5's
530-essay expansion) — both corrected, verified live via HMR.

Updated `README.md`, `docs/EVALUATION.md` (round 6 section, all numeric
tables, evidence-methodology note, rewritten "three wrong examples"
section with fresh GBM-era examples), `docs/LIMITATIONS.md` (#1 fairness
numbers and caveat, #2 rewritten with the full held-out-generator
methodology and numbers, nonlinear-classifier note updated from
"declined" to "retested and adopted," priority list reordered with
cross-genre now the top open item), and this file.

## Round 7 — cross-genre fix via DAIGT training-slice (the biggest single-round result)

User's explicit ask: "we need to get to 80s in accuracy and also how
will cross genre accuracy increase." Round 6 had already established
that every lever tried so far (AI-generator diversity within the
admissions genre, nonlinear-classifier retraining on that diverse data)
traded in-genre accuracy against cross-genre accuracy without ever
teaching the model anything about a different *genre* — every prior
attempt only varied generator identity while staying entirely within
admissions-essay writing. Round 7's hypothesis: genre exposure, not just
generator exposure, is the actual missing ingredient.

### Design: `build_daigt_training_slice.py`

DAIGT-v2 (`data/processed/daigt_eval.csv`, 44,868 rows, previously
eval-only) has 27,371 human + 17,497 AI essays across 16 AI generators
in the persuasive-essay genre. New script samples a training slice from
it — 528 human + 528 AI (capped 35/generator across 16 generators, for
diversity over volume) — with two correctness safeguards built in from
the start, not bolted on after a problem was found:

1. Reproduces `score_daigt.py`'s exact eval sample (`random.Random(42)`,
   same call order: 100 human then 100 AI) and excludes those specific
   essay ids from the training slice, so the DAIGT cross-genre check
   stays genuinely essay-level held-out.
2. DAIGT's "human" class *is* the PERSUADE 2.0 corpus — the same corpus
   `score_fairness.py`'s 300-essay PERSUADE fairness-eval sample is
   drawn from. No shared id column exists between the two CSVs, so
   overlap was checked by normalized-text match. **301 essays were
   caught and excluded** — a real, sizeable overlap that would have
   silently trained on ~100 of the 300 fairness-eval essays if missed,
   invalidating that whole section's numbers without any error or
   warning.

`extract_features.py` extended to also load `daigt_training_slice.csv`.
`train_classifier.py` extended to report an admissions-genre-only subset
accuracy (filtering out the new `cross_genre_daigt` theme tag) alongside
the now-blended headline numbers, specifically so round 7's numbers stay
comparable to rounds 1-6's admissions-only baseline rather than silently
changing what "in-theme accuracy" means.

### Full pipeline re-run (2,246 essays total, up from 1,190 the prior largest run)

`extract_features.py` -> `train_classifier.py` -> `calibrate_thresholds.py`
-> rebuild/rescore both hybrid sets -> `score_daigt.py` -> `score_fairness.py`
-> `score_heldout_generator.py` -> `find_wrong_examples.py` -> pytest, all
chained in one background run (~40 minutes wall-clock, mostly GPT-2
feature extraction).

**Results, the headline ones first:**

- **DAIGT cross-genre: 51.2% -> 78.4% sentence-level, 51.0% -> 93.5%
  essay-level.** The single biggest number change across all 7 rounds of
  this project. Directly answers the user's "how will cross-genre
  accuracy increase" question.
- **Held-out-generator recall (OpenAI/Anthropic, genuinely absent from
  both admissions training data and DAIGT's 16 generators): 7.5%
  essay-level, unchanged from round 6.** This is the honest asterisk on
  the headline number — genre/generator diversity helps enormously
  *when the classifier has training exposure*, and does nothing for
  generators it's never seen anywhere. Reframes this project's biggest
  limitation from "cross-genre" (now substantially addressed for
  covered genres/generators) to "cross-generator for genuinely novel
  generators" (untouched).
- **Admissions-genre-only accuracy: 76.3%/74.1% (round 6 peak) ->
  70.6%/71.7%** — a real, disclosed cost, roughly back to round 5's
  original 72.2%/71.1% baseline. The user's "80s" target was hit for
  DAIGT cross-genre (78-93%) but not for pure admissions in-genre
  accuracy (70.6-73.8% depending on blended-vs-admissions-only framing).
- **Hybrid localization gave back most of round 6's gain** (continuation
  F1 0.606 -> 0.541, mid-paragraph 0.427 -> 0.384) — these essays are
  still built only from the original Gemini/EssayForum pool, so this is
  the same admissions-genre-specialization tradeoff as above.
- **Fairness moved in mixed directions for the first time** — PERSUADE
  overall broad FPR rose from round 6's 3.7% back to 8.3% (still better
  than round 5's 9.7%), with some subgroup gaps improving further
  (Black/African American) and others reversing or worsening (Asian/
  Pacific Islander, ELL direction flip). Documented as evidence that
  these fairness numbers are not stable under unrelated training-data
  changes and should be re-checked after any future retrain, not a
  claim that round 7 specifically harmed fairness.

Verified end-to-end: backend restarted, a fresh (not-in-any-dataset)
persuasive-genre essay pasted into the browser scored sensibly
(landed Inconclusive with mixed ai-like/human-like sentence coloring,
consistent with genuinely mixed signal on real AI-generated persuasive
text — the kind of input round 6's model would very likely have called
confidently human). 39/39 pytest passing. Updated the frontend header
banner (`App.tsx`) to describe the new training composition.

Updated `README.md`, `docs/EVALUATION.md` (round 7 section, full
numeric tables incl. a 3-column round 7/6/5 comparison, the DAIGT
methodology-shift caveat, rewritten wrong-examples section), and
`docs/LIMITATIONS.md` (#2 split into 2a/2b — genre-diversity-fixed vs.
generator-novelty-unfixed — and #1's fairness section reframed around
round 7's noisier numbers, priority list reordered around the new #2b
as the top open item).

## Round 8 — retesting the round-6-declined lever against a bigger base

User asked directly what happened to the 200 OpenAI/Anthropic essays
they sourced, after round 7's summary reiterated that the 7.5%
held-out-generator recall was unchanged. Answer given: round 6 tested
training on 160 of those essays against a ~1,030-essay base and it made
things worse across the board (declined); round 7's DAIGT-v2 addition
never touched that data since it's a different set of generators. But
round 7 also just demonstrated that a large, well-structured training
addition can work dramatically where round 6's smaller one didn't — a
direct, testable reason to reconsider the round-6 decision now that the
base is 2,086 essays instead of ~1,030. Proposed retesting; user agreed.

### Change

`train_classifier.py`'s `EXCLUDED_AI_ID_PREFIXES` (added round 6 to
exclude `openai-`/`anthropic-` essay ids from training) set to empty —
those essays flow back into training from the same `ai_essays.csv` that
already contained them (`ingest_ai_essays.py`'s held-out-generator split
was untouched: the same 160 essays go to training, the same 40 stay in
`ai_essays_heldout_generator.csv`, eval-only, exactly as round 6 set it
up). No new data collection, no `extract_features.py` re-run needed —
sentence-level features for these essays were already sitting in
`feature_table_phase2.csv` from round 6's original extraction; only the
training-time filter changed. This made the pipeline re-run fast (~5
minutes instead of ~40, since the ~35-40 minute GPT-2 feature-extraction
step was skippable entirely).

### Result

- **Held-out-generator recall: 7.5% -> 30.0% essay-level (4x), 58.4% ->
  68.5% sentence-level.** GPT-5.6-luna 0%->10%, Claude-haiku-4-5
  15%->50%. This is the number the user's data was meant to move, and
  it moved.
- Admissions-only in-theme/unseen-theme: 70.6%/71.7% -> 70.8%/71.6% —
  noise-level, not a real change.
- DAIGT cross-genre: 78.4%/93.5% -> 77.5%/93.0% — also noise-level.
- Hybrid localization: 0.541/0.384 -> 0.536/0.367 — small, real dip on
  the mid-paragraph variant, otherwise flat.
- Fairness: PERSUADE overall broad FPR 8.3% -> 10.7% — a real, modest
  cost, continuing round 7's finding that large training-data changes
  move these numbers unpredictably. Some individual subgroup gaps
  widened further (White broad FPR up to 19.8%, ELL-vs-non-ELL gap
  widened to 8pt).
- 39/39 pytest still passing. Trained on 2,246 essays total.

**Adopted.** Verified end-to-end: backend restarted, a direct API call
confirmed the evidence panel still produces well-formed, correctly
sorted output. The round-6 decision to decline this exact training-data
change wasn't wrong — it was correct for a ~1,030-essay base. The
round-8 decision to adopt it isn't a reversal — it's the same
lever, correctly re-evaluated at 2x the data. Documented as a case study
in `docs/LIMITATIONS.md` #2b: the same training-data change can fail at
one dataset scale and succeed at another, and neither result generalizes
to "this always helps" or "this never helps" — it needs re-testing when
the underlying data changes materially, not assumed stable.

Updated `README.md`, `docs/EVALUATION.md` (round 8 intro item, updated
model-under-test description, Summary table now a 4-column round
8/7/6/5 comparison), and `docs/LIMITATIONS.md` (#2b rewritten with the
full round 6->7->8 trajectory and the "same essays never trained on"
framing that makes this a genuinely apples-to-apples before/after
comparison, not a new eval set).

## Round 9 — GBM hyperparameter tuning (tested, declined)

User asked to push accuracy further again. `GradientBoostingClassifier`
had run on its round-6 defaults (`n_estimators=200, max_depth=3`) ever
since adoption — never actually tuned, unlike `LogisticRegression`'s C
(swept and found flat in round 6). New `experiment_tune_gbm_hyperparams.py`
swept `max_depth` (3-5) and `learning_rate` (0.05-0.2) against the
current 2,246-essay training set, reporting admissions in-theme,
unseen-theme, and DAIGT accuracy for each. `max_depth=5` looked like a
free win on all three (+0.8pt / +0.5pt / +0.2pt, nothing regressed).

**Adopted provisionally, then caught by the held-out-generator check
before being kept.** Retrained the production model at depth=5 and ran
the full re-scoring pipeline, specifically checking
`score_heldout_generator.py`'s 40-essay genuinely-novel-generator
recall — the metric round 8 had just moved 7.5%->30.0%, and the one
most likely to reveal overfitting invisible to the other three metrics.
It dropped to 17.5% (GPT-5.6-luna essay-level recall: 0/20). More tree
depth = more capacity to fit training-specific patterns = worse
generalization to genuinely unseen essays — the same overfit pattern
seen with RandomForest/GradientBoosting depth increases earlier in this
project (round 4's nonlinear-classifier experiment), now reproduced
within GBM's own hyperparameters rather than just at the LogisticRegression-
vs-GBM model-family level.

**Reverted to `max_depth=3`** — confirmed via a full re-run that this
exactly restores round 8's numbers (70.8%/71.6% admissions, 77.5% DAIGT,
30.0% held-out-generator, 39/39 tests). Net effect of round 9: zero
change to the shipped model, but a real, disclosed finding — the
held-out-generator check is now established as the canary metric for
this specific failure mode, and any future capacity-increasing change
(more estimators, more depth, a different model family) should be
checked against it specifically, not just the three metrics that looked
clean here.

## Round 10 — a genuine no-tradeoff win

User's framing this round, explicitly: more accuracy, but nothing else
allowed to regress. After round 9 demonstrated that not all "small
tuning wins" are actually free (depth increase looked clean on 3
metrics, broke a 4th), the obvious next test was whether a
*structurally different* kind of capacity change would behave
differently: more boosting rounds (`n_estimators` 200->300) at the same
`max_depth=3`, rather than deeper trees. More estimators refines the
existing decision boundary; more depth adds capacity to fit new,
potentially training-specific patterns — different mechanisms, worth
testing separately rather than assuming they'd behave the same.

Ran the exact same full-pipeline check as round 9 (train -> calibrate ->
rescore hybrid/DAIGT/fairness/held-out-generator -> pytest) before
adopting anything. Result: admissions in-theme/unseen-theme both +0.4pt,
and every other tracked metric — DAIGT (77.5%/93.0%), hybrid
localization (F1 0.536/0.367), ELLIPSE FPR (2.5%), PERSUADE FPR (10.7%),
held-out-generator recall (30.0%/68.5%) — identical to three decimal
places to round 8. Adopted. 39/39 tests passing.

This is the first change across 10 rounds that improved something
without measurably moving anything else — worth naming as a real
methodological finding, not just a lucky number: the two "similar-
looking" hyperparameter changes (depth vs. estimator count) had opposite
outcomes because they change the model in fundamentally different ways,
and only checking against the full metric suite (specifically
held-out-generator recall, the canary this project established in round
9) told them apart. A quick 3-metric sweep would have recommended round
9's depth increase just as confidently as it did round 10's estimator
increase.

Updated `README.md` (round 9/10 bullets), `docs/EVALUATION.md` (model-
under-test line, round 9/10 Summary-section notes, 5-column comparison
table). Restarted backend, confirmed healthy.

## Round 11 — cleanup pass: dead code removed, stale docs fixed, one real bug caught

User asked for a cleanup pass: remove unnecessary code, fix anything in
the docs that's no longer true. Went through the codebase and every doc
systematically rather than relying on memory of what should be current.

### Dead code removed

`backend/scripts/build_placeholder_ai_essays.py` (36KB) and
`data/processed/ai_essays_placeholder.csv` — Phase 2 placeholder AI
essays, superseded by the real Gemini batch back in Phase 1's addendum.
Confirmed zero references anywhere in the active pipeline or tests
before deleting (only docstring mentions elsewhere explaining *why*
`ingest_ai_essays.py` exists, which are accurate history, not dead
code — left alone).

### Stale docs fixed

- `data/README.md` hadn't been substantively updated since round 5 — it
  was missing the OpenAI/Anthropic essays entirely, and **incorrectly
  claimed DAIGT-v2 was "eval-only, not used in training"**, which became
  false the moment round 7 shipped. Also still said "PERSUADE / ELLIPSE
  — not yet built. Blocked on Kaggle API credentials," directly
  contradicting a correct, fully-built section later in the same
  document — these have been built and central to fairness evaluation
  since round 2. Rewrote the affected sections with current, accurate
  data lineage.
- `docs/LIMITATIONS.md` said "eighth full version" while actually being
  the tenth. Fixed.

### Real bug found: the production model silently diverged from the eval model for two rounds

`train_classifier.py` trains two separate `GradientBoostingClassifier`
instances by design — an eval model (80% split, used for reporting
in-theme/unseen-theme accuracy) and a production model (trained on
everything, saved to `classifier.joblib`, the one actually served).
Round 10's hyperparameter edit (`n_estimators` 200->300) was written as
a single Edit call that matched and updated only the eval model's
instantiation. The production model's call site, textually identical
apart from context, was never touched — `classifier.joblib` silently
kept shipping the round-8 (`n_estimators=200`) config for two full
rounds while every reported eval number came from the correctly-updated
model.

Consequence, worse than just "one wrong number": `calibrate_thresholds.py`
recalibrates the essay-verdict decision thresholds against the eval
model's score distribution, then those thresholds get applied to the
production model's raw scores when serving requests (and when
`score_daigt.py`/`score_fairness.py`/`score_heldout_generator.py` scored
the "current" state). So round 9's "declined, dropped held-out-generator
recall 30.0%->17.5%" and round 10's "adopted, moved nothing else"
conclusions were both comparing threshold-recalibration effects applied
to an *unchanged* underlying production model, not genuine like-for-like
comparisons of different classifiers. The qualitative decisions (decline
round 9's depth increase, keep checking held-out-generator recall before
adopting anything) hold up regardless — but the stated *mechanism*
("more capacity overfits") in round 9's write-up was likely imprecise,
and round 10's "identical to three decimal places" claim was flatly an
artifact of the bug, not a real property of the change.

Found by systematically re-checking every hardcoded hyperparameter
reference across the docs against the actual code during this cleanup
pass — not caught by the extensive full-pipeline testing done in rounds
9-10, because that testing only ever inspected the eval model's
in-process numbers plus the served model's downstream eval-script
output, and never diffed the two `GradientBoostingClassifier(...)` call
sites against each other directly.

**Fix**: updated the production model's call site to match
(`n_estimators=300, max_depth=3`), re-ran the full pipeline (train,
calibrate, rescore hybrid/DAIGT/fairness/held-out-generator, pytest).
Corrected numbers came back **equal or better** than what had been
reported, not worse: held-out-generator essay-level recall 30.0% ->
**35.0%**, PERSUADE fairness broad FPR 10.7% -> **9.7%**, ELLIPSE
2.5% -> 2.0%, hybrid localization and DAIGT within rounding. Admissions
in-theme/unseen-theme accuracy (71.2%/72.0%) were unaffected the whole
time, since those numbers come directly from the eval model, which was
never buggy.

Restarted backend with the corrected model, verified end-to-end in the
browser, 39/39 pytest passing. Updated `README.md`, `docs/EVALUATION.md`
(Summary section rewritten with a correction notice and the corrected
comparison table), and `docs/LIMITATIONS.md` (#2b's headline number and
the priority list) with the real numbers, and added a durable comment
in `train_classifier.py` at the fixed call site explaining what went
wrong so the two-call-site pattern doesn't silently diverge again.

**Why this is disclosed in full rather than quietly corrected**: this
project's whole methodology has been "test before adopting, disclose
costs and mistakes rather than smooth them over" — silently fixing this
and updating only the numbers, without explaining that two rounds of
"verified, no tradeoff" claims were built on a bug, would be exactly the
kind of thing this project's own `docs/LIMITATIONS.md` criticizes other
projects for not doing.
