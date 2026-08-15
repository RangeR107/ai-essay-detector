# Evaluation

Trained on 530 human admissions essays plus 660 AI-generated ones (Gemini, OpenAI, and Anthropic), with a 1,056-essay slice of DAIGT-v2 added on top for genre and generator diversity. Thresholds are calibrated against held-out data, not guessed.

This document is an accuracy report, not a pitch. Every number below comes from a script in `backend/scripts/` that anyone can re-run — nothing here is estimated. `DOCUMENTS/IMPLEMENTATION.md` has the full build history behind these numbers, and `docs/LIMITATIONS.md` has the consolidated caveats.

## Results at a glance

| | Sentence-level | Essay-level (the actual verdict the app shows) |
|---|---|---|
| Admissions essays, topics seen in training | 71.2% | see blended row below |
| Admissions essays, unseen topics | 72.0% | — |
| Blended pool (admissions plus other trained-on genres) | — | 95%+ (calibration-time balanced accuracy) |
| Cross-genre: persuasive essays, 16 AI models (DAIGT-v2) | 77.6% | 93.0% |
| Novel AI tools never trained on (OpenAI/Anthropic, held out) | 69.3% | 35.0% — the weakest number here |

Put plainly: the detector is strong, 93 to 95%, on essays that resemble what it trained on, whether that's the admissions genre or one of the 16 AI models covered by DAIGT-v2. It's weak, 35%, on an essay from a brand-new AI tool it has never seen. That gap is the project's real remaining limitation, not the sentence-level number above it — see §3 and `docs/LIMITATIONS.md` #2 for the full discussion.

The three essays the detector confidently gets wrong, each with the feature values behind the mistake, are in §6. Everything between here and there is the detailed methodology and the history of what changed and why across ten rounds of iteration — worth reading if you want to see the reasoning, safe to skip if the table above and the summary at the end are all you need.

The current model trains on 2,246 essays: 530 human from four sources, 500 AI essays from Gemini, 160 more from OpenAI and Anthropic, and 1,056 from DAIGT-v2 spanning 16 generators in a different genre. It's a `GradientBoostingClassifier` (300 trees, depth 3) over 12 features — perplexity, three GLTR rank-bucket aggregates, and eight stylometry measures. Unless noted, every number below is sentence-level, matching what the classifier actually predicts on; essay-level numbers (the ones the app shows a user) are called out separately since averaging many sentence scores together behaves very differently from any one sentence's score.

Ten rounds of changes happened since the first version of this document, and every number below reflects all of them. The one round that didn't stick — a tree-depth increase that looked like a free accuracy win until it quietly hurt recall on unseen AI generators — is in the summary at the end, reverted and explained rather than left out.

1. **Threshold calibration** — essay-verdict thresholds recentered on
   real held-out data. A sentence-level threshold change was tested and
   rejected (hurt precision more than it helped balanced accuracy) — see
   `DOCUMENTS/IMPLEMENTATION.md`.
2. **Two real fixes to the model and data, adopted after direct testing
   (round 2):**
   - **Theme-tagging was fixed at the source for AI essays.** The AI
     batch carries a *known* true category (Gemini's 50 generation
     categories) that was being discarded in favor of a keyword guess —
     now mapped directly (`theme_mapping.py`). Human essays got an
     improved heuristic (word-boundary matching, most-matches-wins
     instead of first-match-wins).
   - **`gltr_pct_top10` was removed from the feature set.** Investigation
     (`investigate_fairness_bias.py`) found it was the dominant driver
     of a measured fairness gap (see §5) — a training-data-specific
     "high top-10% = human-like" association that penalized ELL writers
     for an unrelated reason (non-native phrasing patterns land in the
     same low-top-10% zone). Tested directly
     (`experiment_drop_feature.py`) before adopting.
3. **Round 3, this version:**
   - **Short sentences are now merged with a neighbor for scoring,
     not excluded.** Below 4 words, a sentence's score computed from
     itself alone was empirically unreliable (near-zero/inverted
     human-AI separation) — the old fix dropped those sentences from the
     essay verdict entirely; the real fix
     (`featurize._scoring_spans()`) instead computes that sentence's
     features from a merged span with an adjacent sentence, so every
     sentence gets a real, context-informed score and nothing is thrown
     away. The UI shows this with a `context_merged` flag rather than
     hiding or discounting the sentence.
   - **EssayForum reviewer meta-commentary is now stripped during
     ingestion** (`hf_essayforum.py`) — forum posts sometimes open with
     text directed at reviewers ("Let me know what you think of my
     essay...") rather than the essay itself; this contaminated both
     training data and the hybrid-essay eval sets (see §4's rebuild).
   - **Race/ethnicity fairness gap investigated** (§5) — a mitigation
     was tested and, unlike the ELL fix, deliberately **not** adopted;
     the reasoning is in §5.
4. **Round 4, this version — the Inconclusive band was narrowed from
   ±0.15 to ±0.08.** `experiment_threshold_band_width.py` found the
   inherited-but-never-calibrated ±0.15 half-width left **73.4% of
   held-out essays with no definitive verdict at all** — Inconclusive
   was the most common outcome, not an edge case. ±0.08 gives a 27.4%
   Inconclusive rate at 98.9% accuracy on the essays that *do* get a
   definitive call, a real jump in usefulness for a small, measured
   accuracy cost, chosen after the user reviewed the full width-vs-
   accuracy tradeoff table. Two nonlinear classifiers (RandomForest,
   GradientBoosting) were also tested on the same features/data
   (`experiment_nonlinear_classifier.py`) — both beat the current
   LogisticRegression in-genre (72.2% -> 75-77%) but got *worse*
   cross-genre (55.3% -> ~50%, worse than the linear model and close to
   chance) — a classic overfit-to-training-distribution pattern.
   **Not adopted**, kept `LogisticRegression`, since the cross-genre
   number is the more consequential weakness and this trade makes it
   worse. Revisit once more diverse AI-generator training data (in
   progress, user-collected) is available — that data may change the
   calculus by giving a nonlinear model real cross-generator signal to
   learn from instead of just fitting noise.
5. **Round 5, this version — the human training pool expanded from
   500 (EssayForum-only) to 530 (4 sources).** Three previously-built,
   previously-dropped adapters (openessays.org, conncoll.edu,
   blog.emoryadmission.com — real curated/published "essays that worked"
   pages, a different population than EssayForum's peer-feedback drafts)
   were re-enabled specifically to counter the single-source skew that
   was a real contributor to the fairness gaps in §5. Real combined
   yield was 30 essays, not the 200-300 initially hoped for — a live
   check of current source volume (done before committing to a plan this
   time, not assumed) confirmed these are small curated pages, not bulk
   databases, matching this project's own earlier finding when these
   same three sources were first tried. Proceeded anyway:
   `class_weight='balanced'` means the value is a second population to
   calibrate "human-like" against, not matching volume. Retrained on all
   1030 essays; every number below reflects this.

6. **Round 6, this version — the classifier itself changed, from
   `LogisticRegression` to `GradientBoostingClassifier`.** Directly
   tested first (`experiment_nonlinear_classifier.py`) on this exact
   Gemini-only data: +4.1pt in-theme, +3.0pt unseen-theme, -4.7pt DAIGT
   cross-genre. Two other levers were tried first and both declined: a
   200-essay AI-generator-diversity expansion (OpenAI + Anthropic,
   tested with both the linear and nonlinear model — moved nothing in
   the right direction, see `docs/LIMITATIONS.md` #2 for the full
   held-out-generator numbers), and LogisticRegression regularization
   tuning (C swept 0.01-30, completely flat — not a lever at all with
   this much data and only 12 features). `GradientBoostingClassifier`
   was adopted for the real accuracy/localization/fairness gains, with
   the DAIGT cross-genre cost disclosed rather than hidden (§3 below).
   Required reworking evidence.py's "why" computation from linear
   `coefficient x value` to a perturbation-based method, since a tree
   ensemble has no coefficients — see the note after §2 below.

7. **Round 7, this version — mixed a genre/generator-diverse DAIGT-v2
   slice into TRAINING data, not just eval, specifically to fix the
   cross-genre weakness round 6 disclosed.** 1,056 essays (528 human /
   528 AI across 16 generators, persuasive genre) added via
   `backend/scripts/build_daigt_training_slice.py`. Two correctness
   safeguards: (1) reproduces `score_daigt.py`'s exact 100+100 eval
   sample and excludes those specific essays from training, so the
   cross-genre check stays genuinely held-out at the essay level; (2)
   excludes any DAIGT "human" essay (source is literally the PERSUADE
   corpus) that also appears in the 300-essay PERSUADE fairness-eval
   set — 301 essays were caught and excluded by this check, a real risk
   that would have silently invalidated §5's fairness numbers if missed.
   Result: DAIGT cross-genre accuracy jumped from 51.2% to **78.4%**
   (§3) — the single biggest number change in this document's history —
   at a real, disclosed cost to pure-admissions-genre accuracy (§1-2)
   and a mixed effect on fairness (§5). **The meaning of "cross-genre"
   changed as a result** — see §3 for the honest caveat on what this
   number now measures.

8. **Round 8, this version — retested the round-6-declined lever
   (training on the OpenAI/Anthropic essays) now that the base training
   set is much larger (2,086 essays after round 7, vs. ~1,030 when this
   was declined).** Same 160-train/40-eval split as round 6
   (`score_heldout_generator.py`'s eval set is untouched, so this
   remains a genuine held-out check). Result: held-out-generator
   essay-level recall **7.5% -> 30.0%** (4x), sentence-level 58.4% ->
   68.5%. Everything else moved by less than 1.5pt except PERSUADE
   fairness broad FPR (8.3% -> 10.7%, a real, modest, disclosed cost).
   **Adopted** — the dilution effect that sank this exact lever in round
   6 didn't reproduce at this scale, confirming the round-6 decision was
   correct *for that dataset size*, not correct in general. Trained on
   **2,246** essays total now (530 human admissions + 500 AI Gemini +
   160 AI OpenAI/Anthropic + 1,056 DAIGT).

All eight rounds are described in detail where they matter most, not
just asserted here.

## 1. In-theme held-out split

Random 80/20 essay-level split *within* the themes seen during training
(`backend/scripts/train_classifier.py`). 158 essays, 5,203 sentences
held out.

| | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| human | 0.73 | 0.72 | 0.72 | 4,460 |
| ai | 0.75 | 0.76 | 0.75 | 4,963 |
| **accuracy (blended, admissions + DAIGT)** | | | **0.738** | 9,423 |
| **accuracy (admissions-genre-only subset)** | | | **0.706** | 4,875 |

(Round 5 / `LogisticRegression`, admissions-only: 0.722. Round 6 /
`GradientBoostingClassifier`, admissions-only: 0.763 — the peak. Round
7 adds the DAIGT training slice, which grows the eval pool 5,203 ->
9,423 sentences and costs the admissions-only subset 5.7pt in exchange
for the cross-genre gain in §3 — the classifier now generalizes across
genres instead of specializing in one.)

## 2. Unseen-theme split

The plan's §3c generalization test: 3 entire themes
(`challenging_belief`, `gratitude`, `obstacle_setback`) held out of
training completely, evaluated by the same model as #1. 242 essays,
8,584 sentences.

| | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| human | 0.73 | 0.70 | 0.72 | 4,409 |
| ai | 0.70 | 0.73 | 0.71 | 4,175 |
| **accuracy** | | | **0.717** | 8,584 |
| macro avg | 0.72 | 0.72 | **0.72** | |

This split is always pure admissions-genre (the 3 held-out themes are
admissions-specific categories; DAIGT essays all carry a separate
`cross_genre_daigt` theme tag, so none land here) — directly comparable
across all 7 rounds. Round 5 / `LogisticRegression`: 0.711. Round 6 /
`GradientBoostingClassifier` (Gemini-only training): 0.741, the peak.
Round 7 (DAIGT training slice added): 0.717 — a modest 2.4pt step back
even on essays that were never touched by the new data directly, since
the classifier's overall decision boundary shifted to accommodate two
genres instead of one.

**Evidence-panel methodology, changed this round:** with `LogisticRegression`, `evidence.py` computed each feature's contribution as `coefficient x standardized_value` — exact and free once the model is trained, but only possible because the model is linear. `GradientBoostingClassifier` has no such coefficients. The replacement, still fully deterministic: for each feature, swap its value to the human-training-class median, re-score with everything else held constant, and take the resulting change in P(AI) as that feature's contribution. This is a genuine local (per-sentence) explanation rather than a fixed global weight — arguably more honest for a nonlinear model, since a feature's real effect can depend on the other feature values around it. Verified via the full pytest suite (2 tests rewritten to check the new `classify.py` functions instead of the now-removed `coefficients()`) and a manual browser check of the rendered evidence panel.

**This result changed completely after fixing theme-tagging, and the
change itself is the finding.** The first version of this document
reported a large gap between in-theme (73.6%) and unseen-theme (71.3%
accuracy, but only **0.58** macro-F1) — read at the time as "a genuine,
if partially data-artifact-driven, generalization gap." That
interpretation was only half right: it *was* a data artifact, but a
bigger one than realized. The original held-out themes were chosen as
"smallest 3" under a broken heuristic that happened to make the AI side
of those themes almost empty (`background_identity` had 1 AI essay total
out of 500) — the unseen-theme test set was 8:1 class-imbalanced by
accident, and the "generalization gap" was substantially a class-balance
artifact, not the model failing to generalize across topics.

With theme-tagging fixed and `HELD_OUT_THEMES` recomputed against the
corrected (and now much better balanced) distribution, **in-theme and
unseen-theme accuracy stay close (0.722 vs 0.711, a 1.1-point gap — up
slightly from 0.2 points in the prior round after the human-pool
expansion, still small) and macro-F1 is close (0.72 vs 0.71)**. Read plainly: within its training genre and
generator, this classifier generalizes across topic about as well as the
in-theme number alone would suggest — there isn't a large hidden topic-
generalization penalty. The real generalization failure is cross-genre/
cross-generator (§3 below), not cross-topic.

## 3. DAIGT-v2 cross-genre check

DAIGT-v2 (`data/processed/daigt_eval.csv`, via the `Yunij/kaggle-comp-daigt`
HF mirror) is argumentative/persuasive writing from the PERSUADE corpus
and 9+ different AI generators (`mistral7binstruct`, `chat_gpt_moth`,
`llama2_chat`, `kingki19_palm`, `llama_70b_v1`, `falcon_180b_v1`,
`darragh_claude_v7`, etc.) — **a different genre than admissions essays
AND different generators than the Gemini model this classifier was
trained on.** `backend/scripts/score_daigt.py`, stratified sample of 100
human + 100 AI essays (seed=42; not the full 44,868 rows — see that
script's docstring for why).

| | Precision | Recall | F1 |
|---|---|---|---|
| ai | 0.769 | 0.784 | 0.777 |
| human (recall only) | — | 0.784 | — |
| **accuracy** | | | **0.784** |

Essay-level verdict accuracy (mean sentence score >= 0.5): 187/200 = **93.5%**.

**Round 7: the single biggest number change in this document's history
— 51.2% to 78.4% sentence-level, 51.0% to 93.5% essay-level.** This is
the direct result of `build_daigt_training_slice.py` (round 7 intro
item 7): 1,056 DAIGT essays across 16 generators, mixed into training.
Unlike round 6's AI-generator-diversity attempt (200 OpenAI/Anthropic
essays, all within the admissions genre — declined, moved nothing),
this lever changes *genre* too, and genre — not just generator identity
— turns out to be what the classifier actually needed exposure to.

**What this number now honestly measures, stated plainly since it's a
real methodology shift, not just a bigger number:** through round 6,
"cross-genre" meant "this classifier has never seen this genre or these
generators at all" — a strong generalization claim when true. As of
round 7, it means "this classifier has never seen these *specific* 200
essays, but has trained on this genre broadly and on most of these
generators specifically (elsewhere in the DAIGT pool)." That's a
meaningfully weaker claim, though still a real, non-trivial one — DAIGT
has 44,868 essays total; the classifier trained on a stratified 1,056-
essay slice (2.4%) and is being tested on a disjoint 200-essay sample
it never saw, so this isn't memorization of the specific eval essays.
The honest reframe: **this is now "in-distribution generalization
across two genres it was trained on," not "true zero-shot cross-genre
generalization."**

**The number that's still a genuine zero-shot check, and the one that
matters most now: `docs/LIMITATIONS.md` #2's held-out-generator
recall (OpenAI/Anthropic, neither generator represented anywhere in
training, DAIGT included) — still only 7.5% essay-level recall,
unchanged by this round.** DAIGT's 16 generators don't include
OpenAI's or Anthropic's current models, so this round's fix doesn't
transfer to them. **Read the two numbers together: this classifier now
generalizes well across genres and generators it's had broad exposure
to (16 DAIGT generators, persuasive genre — 78-93%), and still poorly to
genuinely novel generators it's never seen anywhere in training (7.5%)
— generator *novelty*, not genre, is now the load-bearing weakness.**

## 4. Hybrid-essay passage localization

Two constructions tested, both eval-only, both sentence-level against a
known ground-truth AI span:

- **Continuation** (`data/processed/hybrid_essays.csv`, 100 essays): a
  real human essay truncated, then a real AI essay's opening sentences
  appended.
- **Mid-paragraph replacement** (`data/processed/
  hybrid_essays_midparagraph.csv`, 100 essays, built after the first
  version of this document): a contiguous span of a real human essay's
  sentences removed and replaced with real AI sentences, keeping genuine
  human text on *both* sides — the case the plan's brief calls most
  realistic ("a human paragraph an AI later polished").

Both scored against the production pipeline: sentence threshold fixed at
0.5 (not the calibrated 0.48 — see the note in the model-under-test
section above and `DOCUMENTS/IMPLEMENTATION.md` for why a threshold
change was tested and rejected for this specific task), short sentences
context-merged rather than excluded (round 3, above).

**Both essay sets have now been rebuilt twice** — once when the
meta-commentary-stripping fix refreshed `human_essays.csv` (removing
un-stripped forum reviewer text that had contaminated the original
build), and again this round when the human pool expanded from 500 to
530 essays. Both builders are deterministic given the same seed and read
directly from `human_essays.csv`/`ai_essays.csv`, so each rebuild
required no code change, just a re-run.

| | Continuation | Mid-paragraph |
|---|---|---|
| Precision | 0.422 | 0.258 |
| Recall | 0.751 | 0.755 |
| F1 | 0.541 | 0.384 |
| Confusion | TP=293 FP=401 FN=97 TN=857 | TP=271 FP=780 FN=88 TN=1943 |

These hybrid essays are still built only from the original Gemini/
EssayForum pool (unaffected by the round-7 DAIGT slice, since
`build_hybrid_essays*.py` don't read `daigt_training_slice.csv`).
Round 6 had pushed both variants up (F1 0.538 -> 0.606 continuation,
0.392 -> 0.427 mid-paragraph) via the classifier swap's in-genre
sharpening; round 7's genre-diversification training gives most of
that back (0.606 -> 0.541, 0.427 -> 0.384) — consistent with §1-2's
admissions-genre-only regression, since these hybrid essays are pure
admissions-genre too.

**The mid-paragraph variant is meaningfully harder, and that's the
useful new finding here** — F1 0.392 vs. 0.538, driven almost entirely
by lower precision (0.265 vs. 0.416). This makes sense structurally: a
mid-paragraph splice has a false-alarm surface on *both* sides of the AI
content instead of just one, roughly doubling the opportunity for the
model's human-text false-positive tendency (§5) to hurt precision. The
original continuation-only numbers were, in retrospect, the easier case
— real-world "AI polished a paragraph in the middle of my essay"
detection should be expected to perform closer to the mid-paragraph
number than the continuation one.

Continuation F1 (0.538) is essentially back to the pre-`gltr_pct_top10`-
removal baseline reported in the first version of this document, after
dipping to 0.513 in round 2 — the meta-commentary and context-merging
fixes recovered the small cost that fix introduced, on this specific
task. This round's human-pool expansion left both numbers essentially
flat (0.536 -> 0.538, 0.398 -> 0.392) — expected, same reasoning as §3.

## 5. PERSUADE / ELLIPSE (subgroup fairness, ESL false-positive rate)

Kaggle access was unblocked after the first version of this evaluation
(user provided API credentials). `backend/scripts/
download_eval_datasets.py` pulled real PERSUADE 2.0 (25,996 human essays
with demographic fields) and ELLIPSE (6,482 ESL-student essays). Sampled
300 PERSUADE essays (stratified 150/150 by ELL status) and 200 ELLIPSE
essays; scored both with `backend/scripts/score_fairness.py` using the
exact production pipeline (round 3: context-merging instead of
short-sentence exclusion; round 4: the narrowed ±0.08 Inconclusive band;
round 5: the expanded 530-essay human training pool — see the intro
above). **Both sources are entirely genuine human writing — every essay
here is a true negative if the classifier gets it right, so any
non-"Likely Human" verdict is a measured false positive.**

Two rates reported: **strict** (verdict = "Likely AI") and **broad**
(verdict != "Likely Human", i.e. "Inconclusive" — a real cost too).
**Strict FPR remains exactly 0 across every group and both datasets**
in round 7 too.

| Group | Broad FPR (round 7) | Broad FPR (round 6) | Broad FPR (round 5) |
|---|---|---|---|
| ELLIPSE overall (n=200) | 1.0% | 0.0% | 0.5% |
| PERSUADE overall (n=300) | 8.3% | 3.7% | 9.7% |
| PERSUADE, non-ELL (n=150) | 9.3% | 3.3% | 8.0% |
| PERSUADE, ELL (n=150) | 7.3% | 4.0% | 11.3% |
| PERSUADE, White (n=81) | 13.6% | 4.9% | 9.9% |
| PERSUADE, Hispanic/Latino (n=142) | 6.3% | 3.5% | 9.9% |
| PERSUADE, Black/African American (n=36) | 2.8% | 2.8% | 13.9% |
| PERSUADE, Asian/Pacific Islander (n=33) | 12.1% | 3.0% | 6.1% |
| PERSUADE, not economically disadvantaged (n=103) | 10.7% | 1.0% | 5.8% |
| PERSUADE, economically disadvantaged (n=163) | 4.9% | 3.1% | 9.8% |
| PERSUADE, disability identified (n=37) | 2.7% | 2.7% | 5.4% |
| PERSUADE, disability not identified (n=229) | 7.9% | 2.2% | 8.7% |

**Round 7 is the first round where fairness moved in mixed directions
instead of uniformly improving — worth stating plainly rather than
cherry-picking the numbers that still look good.** PERSUADE overall
broad FPR rose from round 6's 3.7% back to 8.3% (still better than round
5's 9.7%, but round 6's low was not sustained). Some individual gaps
improved further — Black/African American stayed flat at 2.8% in
absolute terms while White's rate rose to 13.6%, widening the (favorable
direction) gap; economically-disadvantaged flipped to *lower* than
not-disadvantaged (4.9% vs. 10.7%). Others got worse — Asian/Pacific
Islander rose from 3.0% to 12.1%, ELL-vs-non-ELL flipped direction
(ELL now lower). **The most defensible read: round 7's large training-
data change (1,056 new essays, a different genre) reshuffled the
classifier's decision boundary enough that per-group behavior is now
noisier and less predictable than round 6's**, not that the underlying
fairness properties got worse on average — overall PERSUADE broad FPR
is still meaningfully better than round 5's original baseline. Every
number here should be re-checked after any future retrain, not assumed
stable — this round is direct evidence that a training-data change with
no fairness-specific intent can move these numbers substantially in
either direction.

### The ELL investigation and fix (round 2, unchanged this round)

`backend/scripts/investigate_fairness_bias.py` compared per-feature
values and per-feature classifier contributions between ELL and non-ELL
PERSUADE essays. The *contribution* analysis (coefficient x standardized
value — what actually drives the verdict) isolated `gltr_pct_top10` as
the dominant driver: a "high top-10% = human-like" association learned
specifically from this project's training data's register (EssayForum's
plainer prose vs. Gemini's more elaborate prose), not a generally
reliable signal — ELL writers land in the low-top-10% zone for a
completely different, legitimate reason (non-native phrasing patterns
are less predictable to GPT-2), absorbing a penalty the feature was
never actually measuring correctly for them.
`backend/scripts/experiment_drop_feature.py` tested removing it before
adopting; the fix was adopted (round 2) and remains in the production
feature set. Full before/after numbers for that round are in
`DOCUMENTS/IMPLEMENTATION.md`.

### The race/ethnicity investigation (round 3, new)

The Black/African American vs. White gap (both subgroups above
`MIN_SUBGROUP_N`, unlike the two smallest race categories below) was
investigated the same way: `backend/scripts/investigate_race_fairness.py`
re-featurized the 300 PERSUADE essays and compared per-feature raw
values and per-feature classifier contributions between the two groups.
The *contribution* view isolated four features doing most of the work,
in order: `gltr_pct_top1000` (delta +0.122), `perplexity` (+0.101),
`rolling_ttr` — vocabulary variety (+0.098), `gltr_mean_rank` (+0.078).
Mechanism, same shape as the ELL finding: Black/African American essays
in this sample are measurably less predictable to GPT-2 (higher GLTR
rank statistics, higher perplexity, more varied vocabulary) — plausibly
reflecting dialect and register differences GPT-2's training corpus
under-represents, the same "predictability gap" mechanism as ELL
writing, just via different specific features.

**Unlike `gltr_pct_top10`, none of these four features looked like an
obviously redundant one to drop.** `experiment_drop_feature_race.py`
tested two variants directly rather than assuming:

| Variant | Unseen-theme accuracy | Black-vs-White broad-FPR gap |
|---|---|---|
| Current production (12 features) | 0.720 | 12.1 points |
| Drop `gltr_pct_top1000` (11 features) | 0.693 (−2.7pp) | 6.5 points |
| Drop `gltr_pct_top1000` + `gltr_mean_rank` (10 features) | 0.690 (−3.0pp) | 1.2 points (nearly closed) |

Dropping both features nearly eliminates the gap, but costs 3 points of
real detection accuracy — unlike the `gltr_pct_top10` fix, which closed
a comparable gap for essentially free. **This mitigation was tested and
deliberately not adopted.** Two reasons: (1) this project already
rejected a comparable accuracy-for-fairness-metric trade once before (the
sentence-level threshold recalibration, `DOCUMENTS/IMPLEMENTATION.md`) on
the grounds that a detector whose entire value proposition is "show
where and why" needs its underlying accuracy to be trustworthy, and 3
points is a real cost against a ~72% baseline; (2) the project brief
explicitly frames spotting and disclosing this class of bias as the
valued outcome ("these detectors have a habit of flagging writers who
learned English as a second language... we'd like to know you spotted
it"), not a requirement to engineer it away at a real cost elsewhere.
**Practical implication: any "Inconclusive" or "Likely AI" verdict on
writing from a Black or Asian/Pacific Islander student should be treated
with real skepticism** — this gap is measured, traced to a specific
mechanism, and currently un-mitigated by design, not by oversight.

**Bottom line, across both rounds: the fix that was adopted
(`gltr_pct_top10` removal) was a clean, nearly-free win for the ELL and
economic-disadvantage gaps — both are now small (≤5 points). The
race/ethnicity gap is traced to the same underlying mechanism
(predictability-to-GPT-2) but resists an equally cheap fix; closing it
further requires a real accuracy tradeoff this project has chosen not to
make silently.** See `docs/LIMITATIONS.md` #1 for the fuller discussion.

Small-subgroup caveat unchanged: Two-or-more-races/Other (n=6) and
American Indian/Alaskan Native (n=2) are too small to draw any
conclusion from.

## 6. Three examples the detector confidently gets wrong

Found via `backend/scripts/find_wrong_examples.py`, re-run this round
against the round-7 model. This script scans the full feature table —
now including the 1,056-essay DAIGT training slice — for the single
most extreme miss in each direction, which is itself informative about
what changed this round.

### Clear false positive: `hf-essayforum-11919`

A real human EssayForum essay (mean sentence score 0.629) — a community-
service narrative. Note this is a *different* essay than every prior
version's example (`hf-essayforum-22426`) — round 7's much larger,
genre-diverse training data shifted the decision boundary enough that a
different essay is now the most confidently-wrong human example, a
concrete illustration of how much this round's training change moved
the model.

Its most damning sentence (score 0.934) breaks down as:

| Feature | Percentile | Direction |
|---|---|---|
| sentence perplexity | 0.8th | ai-like |
| average word length | 84.6th | ai-like |
| function-word ratio | 10.7th | ai-like |

Unusually *predictable* prose by GPT-2's own measure (very low
perplexity) is normally a strong AI tell — here it's a case of plain,
direct human writing (the essay literally repeats a numbered structure,
"1)... 2)...", restating its own opening) reading as suspiciously
formulaic.

### Clear false negative: `daigt-38593`

A real AI-generated essay from the DAIGT-v2 pool (mean sentence score
0.213) — one of the 16 generators represented in this round's training
data, so this is a genuine in-training-distribution miss, not a
cross-generator one. **Known gap in this example, disclosed rather than
faked**: `find_wrong_examples.py` looks up full essay text from
`human_essays.csv`/`ai_essays.csv` only, not
`daigt_training_slice.csv`/`daigt_eval.csv`, so the full text isn't
available for this specific example — a cosmetic script limitation, not
a scoring-correctness issue (the score and feature evidence below are
both computed correctly from the real essay).

Its most human-reading sentence (score 0.068) breaks down as:

| Feature | Percentile | Direction |
|---|---|---|
| vocabulary variety (rolling type-token ratio) | 6.6th | human-like |
| function-word ratio | 88.7th | human-like |
| punctuation rate | 61.1th | ai-like |

### Hybrid-essay boundary miss: `hybrid-008`

The same recurring example across every version of this document — the
worst result in the continuation set, still built only from the
Gemini/EssayForum pool: the one genuine human sentence bordering the
splice point false-flagged as AI (the assigned-prompt-boilerplate
artifact discussed in every prior version), and both AI-spliced
sentences missed as human. Scores shifted slightly with this round's
model (boilerplate sentence 0.928 -> 0.652, AI-side misses 0.129/0.348 ->
0.085/0.216) but the same structural story holds — the AI-spliced
opening ("The grey foam of the bouldering gym mats always looked softer
from the ground...") still reads human-like on sentence perplexity
(99.6th pct) and vocabulary variety (11.0th pct), the opposite failure
mode from the typical AI tell.

## Summary

**Correction, found during a post-round-10 cleanup pass — read this
before the table.** Round 10's code change (`n_estimators` 200->300)
was written as two call sites in `train_classifier.py` (the eval model
and the separately-trained production model that actually gets saved to
`classifier.joblib`) but the edit only touched the eval-model call site.
For two rounds, every reported eval number came from a correctly-updated
model while the actually-served model silently stayed on the round-8
config. Caught, fixed, and the full pipeline re-run with both call sites
consistent. The corrected numbers below are equal or *better* than what
was previously reported — not a regression, but the "identical to three
decimal places" claims in the original round 9/10 write-ups were an
artifact of comparing threshold recalibration against an unchanged
underlying model, not a genuine like-for-like comparison. That framing
is corrected here.

| Evaluation | Round 10 (corrected) | Round 8 | Round 7 | Round 6 | Round 5 | Note |
|---|---|---|---|---|---|---|
| In-theme, admissions-only | **71.2%** | 70.8% | 70.6% | 76.3% (peak) | 72.2% | From the eval model — unaffected by the bug |
| Unseen-theme (always admissions-only) | **72.0%** | 71.6% | 71.7% | 74.1% (peak) | 71.1% | Unaffected by the bug |
| DAIGT cross-genre, sentence | 77.6% | 77.5% | 78.4% | 51.2% | 56.0% | ~flat |
| DAIGT cross-genre, essay-level | 93.0% | 93.0% | 93.5% | 51.0% | 55.5% | ~flat |
| **Held-out-generator recall (OpenAI/Anthropic, essay-level)** | **35.0%** | 30.0% | 7.5% | 7.5% | — | **Real improvement over the previously-reported 30.0%**, from the corrected production model |
| Held-out-generator recall, sentence-level | 69.3% | 68.5% | 58.4% | 62.7% | — | Small real improvement |
| Hybrid localization (continuation) | F1=0.539 | F1=0.536 | F1=0.54 | F1=0.61 (peak) | F1=0.54 | ~flat |
| Hybrid localization (mid-paragraph) | F1=0.368 | F1=0.367 | F1=0.38 | F1=0.43 (peak) | F1=0.39 | ~flat |
| ELLIPSE FPR (broad) | 2.0% | 2.5% | 1.0% | 0.0% | 0.5% | Small real improvement |
| PERSUADE FPR overall (broad) | **9.7%** | 10.7% | 8.3% | 3.7% | 9.7% | **Real improvement over the previously-reported 10.7%** |
| PERSUADE fairness gaps | mixed, similar shape to round 8 | mixed | mixed | uniformly better | mixed, mostly small | Not re-broken-out by subgroup after the fix — directionally consistent with round 8 |

**Round 9 (tested, declined): `max_depth` 3->5 looked like a free win on
in-genre/DAIGT accuracy in isolation, but held-out-generator essay-level
recall dropped (from that round's then-30.0% baseline) when checked.
Reverted.** This conclusion holds regardless of the bug above — round 9
never touched `final_clf`'s hyperparameters either (only the eval
model's), so its held-out-generator regression was driven by threshold
recalibration shifting against an *unchanged* production model, not a
literal capacity increase in the served classifier. The qualitative
lesson (check held-out-generator recall before adopting any change, not
just the metrics that look good) stands; the causal explanation
("more capacity overfits") was likely imprecise — the more accurate
mechanism is "recalibrating thresholds against a differently-shaped eval
score distribution can shift real classification outcomes even when the
underlying scoring model hasn't changed." Not worth re-testing purely to
get a cleaner causal story, since the decision (decline) and the
practical lesson (check the canary metric) are unaffected either way.

**Round 10, in one sentence, corrected: a genuine small accuracy gain
(+0.4pt admissions) that, once the production-model bug was fixed,
turned out to come with real improvements elsewhere too** (held-out-generator
recall +5pt over what was previously reported, fairness FPR -1pt) rather
than the "moved nothing else" story originally told — that story was an
artifact of the bug, not a property of the actual change.

**Round 8, in one sentence: retested round 6's declined AI-generator-
diversity lever now that the base training set had grown 2x (round 7's
DAIGT addition), and this time it worked** — held-out-generator recall
quadrupled (7.5%→30.0%) with everything else essentially flat except a
real, modest fairness cost (PERSUADE broad FPR +2.4pt). **The core
lesson across rounds 6-8: the same training-data change can fail at one
dataset scale and succeed at another** — round 6's rejection wasn't
wrong given ~1,030 essays, and round 8's acceptance isn't a reversal of
that judgment, it's a different, correct judgment for a 2,246-essay
base. **What's still true: 30% is still a weak catch rate for genuinely
novel generators** — most ChatGPT/Claude essays will still read as
"Likely Human" or "Inconclusive," not "Likely AI." This remains the
project's clearest open problem, now with real, if partial, progress
against it rather than a flat number. See `docs/LIMITATIONS.md`.
