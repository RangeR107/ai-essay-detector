# Limitations

Every limitation surfaced across `data/README.md`, `docs/IMPLEMENTATION.md`, and the numbers in `docs/EVALUATION.md` lives here, grouped by how much it should change your trust in a verdict. Most consequential first.

## The two that matter most right now

**It doesn't reliably catch AI tools it's never trained on.** Essay-level recall on genuinely novel generators sits at 35%. That's real, measured progress from where it started (7.5%, before we trained on more essays from that same generator family — see #2b), but most essays from a brand-new AI tool will still come back Likely Human or Inconclusive, not Likely AI. This is the single biggest gap in the project, and any verdict on an unfamiliar or very recent AI tool deserves real skepticism.

**There's a measured false-positive bias against ELL, Black, and economically disadvantaged writers.** We went looking for this specific problem rather than waiting for someone to find it. It traces to a concrete mechanism: GPT-2 finds their writing statistically less predictable for reasons that have nothing to do with whether it was AI-written. One fix cost nothing and got adopted; a second, costlier fix got tested and deliberately turned down rather than shipped quietly. The current numbers are smaller than they used to be but noisier than we'd like — the full picture is in #1 below.

Everything past this point is the detailed methodology behind those two findings, plus seven smaller, lower-severity gaps. It's kept in full because how we found each of these matters as much as the current number does, but the two above are the ones that should actually change how much you trust a given verdict.

This is the file's tenth revision. The short version of how it got here: an AI-generator-diversity change that failed at one dataset size worked once the training set grew (see #2b for that whole story), a tree-depth increase that looked promising got reverted after it quietly hurt recall on unseen generators, and a bug in how the production model was trained briefly made a later change look better than it was — caught during a cleanup pass and corrected, documented in `docs/IMPLEMENTATION.md`.

Round 6 tested two levers: adding 200 OpenAI/Anthropic essays to AI
training data (declined — moved nothing) and switching the classifier
from `LogisticRegression` to `GradientBoostingClassifier` (adopted — real
in-genre gain, real cross-genre cost). **Round 7 then took a different
approach to the cross-genre problem — not more generator diversity
within the admissions genre, but genuine genre diversity via a DAIGT-v2
training slice — and it worked dramatically**: DAIGT cross-genre
accuracy went from 51.2% to 78.4% (93.5% essay-level). This is the
single biggest number change in this project's history, and it
reframes what "the cross-genre problem" even means: it's now split into
two separable findings, one substantially fixed and one not moved at
all. See #2 below for both, stated with equal weight rather than leading
with the good news.

## 1. Fairness bias — investigated across two axes; round 7 made this noisier, not cleanly better or worse

`docs/EVALUATION.md` §5 has the full numbers and methodology; summary
here. **Round 7 update, most current:** the DAIGT training-slice change
(#2 below) — a large training-data change with no fairness-specific
intent — moved fairness numbers in mixed directions instead of the
uniform improvement round 6 produced. PERSUADE overall broad FPR rose
from round 6's 3.7% back to 8.3% (still better than round 5's 9.7%).
Some individual gaps improved further (Black/African American held flat
at 2.8% while White rose to 13.6%, widening the favorable gap;
economically-disadvantaged flipped to *lower* than not-disadvantaged);
others reversed direction or got worse (Asian/Pacific Islander rose from
3.0% to 12.1%; the ELL-vs-non-ELL gap flipped which group is higher).
**The clearest lesson from round 7, stated plainly: these fairness
numbers are not a stable property of "the model" in some fixed sense —
they move, sometimes substantially, with training-data changes that have
nothing to do with fairness, and every number in this section should be
re-checked after any future retrain rather than assumed to hold.** The
paragraphs below describe the round-6 state (still useful for
understanding the *mechanism*, which round 7 didn't re-investigate) —
read the round-7 numbers above as current, the round-6 narrative below
as historical context for how we got here.

**Round 6 (historical — mechanism context, numbers superseded above):**
switching the classifier to
`GradientBoostingClassifier` (see #2) shrank every fairness gap in this
section substantially, as a side effect of the accuracy gain, not a
targeted fairness fix — worth naming plainly since it means the
improvement isn't fully understood mechanistically, just measured.
PERSUADE overall broad FPR: 9.7% -> 3.7%. ELLIPSE broad FPR: 0.5% -> 0.0%.
Black/African American vs. White broad-FPR gap: was 4.0 points, now
**reversed** (Black essays flagged *less* often, 2.8% vs. 4.9%). Asian/
Pacific Islander vs. White: also reversed (3.0% vs. 4.9%). ELL vs.
non-ELL: 0.7 points (was 3.3). Economically disadvantaged vs. not: 2.1
points (was 4.0). **This is real, measured improvement, but it came from
a different, less-interpretable model** (#2's tradeoff) — treat it as
"currently measured small," not "provably fixed," since a tree ensemble's
fairness behavior is harder to trace to one root cause the way
`gltr_pct_top10` was traced below.

**ELL and economic-disadvantage gaps: investigated, fixed, and the fix
held up.** `investigate_fairness_bias.py` traced these to one feature,
`gltr_pct_top10` (% of tokens in GPT-2's top-10 predictions) — a "high
top-10% = human-like" association learned specifically from this
project's training data's register (EssayForum's plainer prose vs.
Gemini's more elaborate prose), not a generally reliable signal. ELL
writers land in the low-top-10% zone for an unrelated, legitimate reason
(non-native phrasing is less predictable to GPT-2) and absorbed a
penalty the feature was never actually measuring correctly for them.
Removing it (`experiment_drop_feature.py`, tested before adopting) cost
essentially no accuracy and shrank both gaps substantially. Two later,
unrelated changes compressed both further: narrowing the Inconclusive
band's width (#6 below), and diversifying the human training pool
beyond EssayForum-only (#3 above). ELL vs. non-ELL is now 3.3 points
(was 14.7 pre-fix, 4.7 after the band narrowed, 3.3 after the data
change), economically-disadvantaged vs. not is now 4.0 points (9.8% vs.
5.8%, both much smaller than the 33%+ pre-fix numbers, though the gap
narrowed only slightly in relative terms from the band-narrowing round —
worth tracking rather than treating as fully resolved).

**Race/ethnicity gap: investigated across several rounds, largely
resolved as of round 6 — but by a less-interpretable mechanism than the
ELL fix.** `investigate_race_fairness.py` (round 3) found the
Black/African American vs. White gap traced to the same underlying
phenomenon as the ELL gap — GPT-2 predictability — but spread across
four features instead of one clearly-redundant one; the tested fix
(dropping those features) cost 3 points of real accuracy and was
declined at the time. Rounds 4-5 (threshold narrowing, human-pool
diversification) compressed the gap from 12.1 to 4.0 points as a side
effect. **Round 6's classifier swap (`GradientBoostingClassifier`, see
#2) compressed it further, to reversed** — Black essays now flagged
*less* often than White (2.8% vs. 4.9% broad FPR), Asian/Pacific
Islander also reversed (3.0% vs. 4.9%). This is real, measured
improvement, but came bundled with a different, harder-to-interpret
model, not from a standalone fairness fix — the underlying mechanism
that caused the gap in the first place was never actually removed, just
diluted by whatever the tree ensemble learned instead. Don't read this
as "the bias is solved," read it as "currently measured small."

Disability status remains reversed (disabled students flagged *less*
often than not-identified students) — still completely unexplained, not
investigated at all, though the round-6 numbers are small enough
(2.7% vs. 10.5% broad FPR pre-round-6, now near-zero across the board)
that it's a low priority to chase further.

**Practical implication, updated for round 6: every group's gap is now
small (≤2 points, several reversed), a genuine improvement over every
prior version of this document — but it rides on a less interpretable
model (#2), and small subgroups (n<10: Two-or-more-races, American
Indian/Alaskan Native) still can't be trusted either way.** Continue
treating any "Inconclusive" or "Likely AI" verdict on writing from any
demographic group with the same baseline skepticism this whole project
asks for, not because the numbers currently look bad, but because
"currently measured small on a 300-essay sample, scored by a model whose
fairness behavior isn't individually traceable the way the linear
model's was" is not the same claim as "fixed."

## 2. Generalization to unseen genres/generators — now two SEPARATE findings, one substantially fixed, one not moved at all

Through round 6, this document treated "cross-genre" and "cross-
generator" as one problem, both measured by DAIGT-v2 (which conflates
genre and generator — it's simultaneously a different genre *and*
different generators than the admissions/Gemini training data). **Round
7 split them apart, and the results are strikingly different for each.**

### 2a. Genre + generators WITH training exposure: substantially fixed

`backend/scripts/build_daigt_training_slice.py` (round 7) mixed 1,056
DAIGT-v2 essays — 528 human (real PERSUADE-corpus persuasive essays) and
528 AI across 16 generators (Mistral, ChatGPT, Llama2, PaLM, Falcon,
Claude-via-wrapper, Cohere, and more) — into training. Two correctness
safeguards make this a legitimate result, not a leak: the classifier
never trains on the same 200 essays `score_daigt.py` evaluates against
(exact seed/sampling reproduced and excluded), and 301 DAIGT "human"
essays that overlapped with the PERSUADE fairness-eval set were excluded
by text-match so §1's fairness numbers stay honest.

Result: DAIGT cross-genre accuracy **51.2% -> 78.4%** sentence-level,
**51.0% -> 93.5%** essay-level (`docs/EVALUATION.md` §3) — genuinely
this project's biggest single-round improvement. **The honest caveat
that makes this a disclosure, not a victory lap**: this number now
means "generalizes to unseen *essays* from genres/generators it had
broad training exposure to," not "generalizes to genuinely unseen AI
text" the way it did (weakly) through round 6. It's real generalization
— the classifier wasn't shown these specific 200 essays — but it's a
narrower claim than the same number implied in every prior version of
this document.

### 2b. Genuinely novel generators (no training exposure anywhere) — real progress in round 8, but still the weakest number here

A held-out set of 40 real essays (20 OpenAI `gpt-5.6-luna`, 20 Anthropic
`claude-haiku-4-5`, admissions-prompt genre, `data/processed/
ai_essays_heldout_generator.csv`, scored by
`backend/scripts/score_heldout_generator.py`) has stayed genuinely
held-out across every round since it was built (round 6) — the 40
essays here have never been trained on, only the *other* 160 essays
from the same two generators have (and only as of round 8).

**Round 6** (declined): training on 160 OpenAI/Anthropic essays against
a ~1,030-essay base gave 7.5% essay-level recall on the 40 held-out —
not meaningfully better than not training on them at all, and it cost
accuracy elsewhere. **Round 7** (DAIGT slice added, OpenAI/Anthropic
still excluded from training): recall unchanged at 7.5%, since neither
generator is in DAIGT's 16. **Round 8** (retested training on the same
160 essays, now against the round-7-grown 2,246-essay base): recall
**jumped to 30.0% (12/40)** — 10% for GPT-5.6-luna (2/20), 50% for
Claude-haiku-4-5 (10/20) — a real, 4x improvement, at a modest fairness
cost (PERSUADE broad FPR +2.4pt, `docs/EVALUATION.md` §5). Sentence-level
recall also improved, 58.4% -> 68.5%.

**Read this plainly, without overstating it either direction: the
round-6 rejection of this exact training-data change wasn't a wrong
call — it was the correct call for the dataset size that existed then.
The same change became a good call once the base dataset roughly
doubled (round 7's DAIGT addition).** This is genuine evidence that
"more diverse training data helps generalization" is true here, just
gated by having enough total data for the addition not to be pure
dilution. **Correction: a production-model training bug (round 10, found
and fixed during cleanup — see `docs/IMPLEMENTATION.md`) meant the
actually-served model differed from what was tested for two rounds;
fixed, and the real current number is 35.0% (14/40), slightly better
than the 30.0% originally reported, not worse.** Still a weak catch
rate at 35%. Most
essays from these two specific tools will still read as "Likely Human"
or "Inconclusive," not "Likely AI." Extrapolating cautiously: this
result suggests further generator diversity (either more essays from
OpenAI/Anthropic, or entirely new generators not yet represented at
all) might continue to help now that the dataset is large enough to
absorb it without dilution — an untested but reasonably-motivated next
step, not a guarantee.

**Practical implication, updated: do not trust any verdict on an essay
that might be from a recent AI tool not well-represented in this
project's training data (currently: Gemini, OpenAI's gpt-5.6-luna,
Anthropic's claude-haiku-4-5, plus DAIGT's 16 generators) — even the
generators with the most training exposure here are still caught well
under half the time on essays they weren't literally trained on.**

## 3. Human training data was a specific, non-neutral population — now broadened, but only modestly

**Updated this round.** For several rounds, the human class was 500
EssayForum posts only — a peer-feedback forum where people post *drafts*
seeking critique, not polished final submissions, skewing
international/ESL. This was very likely part of *why* #1's fairness gap
existed. Three real published/curated sources (openessays.org,
conncoll.edu, blog.emoryadmission.com — "essays that worked" showcases,
polished and already-admitted rather than draft/feedback-seeking) were
re-enabled to counter that skew, adding **30 essays** (530 total). This
genuinely helped — most fairness gaps in #1 moved further in the right
direction with no model change at all — but 30/530 (5.7%) is a modest
dilution of a 500-essay single-source pool, not a rebalancing. The
reference-percentile tables that power the evidence panel, and
"human-like" throughout this app, still mean predominantly "like an
EssayForum draft" — a polished final-draft human essay, or a human essay
in a completely different register, still may not match this baseline
as well as EssayForum-register writing does. Real, further-diversifying
volume from these three specific sources isn't available — they're
small curated pages, not bulk databases (`data/README.md` has the full
accounting of what was actually tried and found).

A concrete instance of this, found and fixed this round: some
EssayForum posts open with meta-commentary directed at forum reviewers
("Let me know what you think of my essay (first draft)...") rather than
the essay itself, which the classifier reliably flagged as AI-like —
stylistically anomalous relative to genuine narrative prose. Now stripped
during ingestion (`hf_essayforum.py`, sentence-level regex match against
a set of directive-phrase patterns); verified zero remaining matches
across all 500 essays against those specific patterns. Residual risk:
the pattern list is a curated set of common review-request phrasings, not
exhaustive — a differently-worded request for feedback could still slip
through uncaught.

## 4. Theme-tagging — fixed for AI essays, improved (not solved) for human essays

**Was:** both classes theme-tagged by the same crude keyword heuristic,
never verified, producing a badly lopsided result for AI essays (~89%
landing in 2 of 7 themes) despite the source data being well-balanced.

**Now:** AI essays use a direct, curated mapping from Gemini's 50 known
generation categories to the 7 themes (`theme_mapping.py`) — since the
true category was always known, the old approach was discarding ground
truth in favor of a worse guess. This produces an exact, predictable
distribution matching the known category counts. Human essays got an
improved heuristic (word-boundary regex instead of substring matching,
most-keyword-hits-wins instead of first-match-wins) — a real
improvement (e.g. `background_identity` went from 30 to 93 essays) but
still a heuristic, still not human-verified, and still imbalanced
(`growth_accomplishment` and `obstacle_setback` remain the two largest
human-side categories by a wide margin — plausibly a real property of
what people write about in admissions essays, not just a tagging
artifact, but unverified either way).

**This fix had a large, mostly positive, and somewhat confusing
downstream effect:** the plan's §3c unseen-theme evaluation
(`docs/EVALUATION.md` §2) went from a large apparent generalization gap
(macro-F1 0.73 in-theme vs. 0.58 unseen) to almost no gap (0.71 vs 0.71)
— not because the model got better at generalizing, but because the old
gap was itself substantially a class-imbalance artifact from the broken
theme tagging (the old held-out themes happened to be AI-thin). **The
lesson generalizes beyond this one metric: a data-labeling bug can look
exactly like a model-capability finding if you don't check the labels.**

## 5. Short sentences — fixed with a real mechanism, not a workaround

Under 4 words, a sentence's score computed from itself alone was
empirically near-zero or inverted separation between human and AI
(measured directly from held-out predictions) — there just isn't enough
text in "Yes." or "I agree." for perplexity or stylometry to say
anything reliable. **The original fix (round 2) excluded these sentences
from the essay verdict and from evaluation scoring** —
`aggregate.MIN_RELIABLE_TOKENS`, a real fix for the aggregate verdict but
a non-fix for the sentence itself: it left short sentences unscored (no
color, no evidence) in the UI, discarding signal rather than recovering
it.

**Round 3 replaces exclusion with context-merging**
(`featurize._scoring_spans()`): a sentence below the reliability floor
has its features computed from a merged span with an adjacent sentence
instead of itself alone (preferring the next sentence, falling back to
the previous one at the essay's end; a lone short sentence with no
neighbor — e.g. a one-sentence essay — is left unmerged, since there's
nothing to merge with). The UI's sentence boundaries, highlighted text,
and heatmap tokens are unaffected — only the score/evidence computation
uses the wider span. Every sentence in an essay now gets a real,
context-informed score; `aggregate.essay_verdict()` no longer needs
special-case exclusion logic at all. The `context_merged` field on each
sentence result tells the UI (and a user) which scores reflect a merged
span, framed as transparency information, not a caveat to distrust.

## 6. Threshold calibration and score_volatility — done, including a real fix to the Inconclusive band width

Essay-verdict thresholds recalibrated against real held-out data;
landed back at almost exactly the original placeholder values (0.35/0.65)
once the theme-tagging and feature-set fixes were in place — a mild
result suggesting the original placeholders weren't badly wrong, just
uncalibrated. A candidate change to the *sentence-level* decision
threshold (0.5 -> 0.42, better balanced accuracy) was tested directly
against hybrid-essay localization and **made precision worse** (flagging
more sentences as AI overall) — deliberately not adopted, since a false
AI accusation on real student writing is judged the costlier error here.

**A separate, previously-unquestioned parameter turned out to matter
much more: the Inconclusive band's half-width.** Every recalibration
above only ever tuned the band's *center*; the ±0.15 half-width itself
was inherited unchanged from the very first placeholder version and
never tested against data. `experiment_threshold_band_width.py` swept it
directly: at ±0.15, **73.4% of held-out essays got no definitive verdict
at all** — Inconclusive was the most common outcome, not a rare hedge.
Narrowed to ±0.08 (user's choice from the measured tradeoff table: 27.4%
Inconclusive at 98.9% accuracy on definitive calls, vs. e.g. 12.4%
Inconclusive at 98.2% accuracy for ±0.05). This is measured on
in-distribution essays only — a narrower band is riskier specifically on
cross-genre/cross-generator essays, where the old wide band's caution
was doing real protective work the model can't replicate on its own,
since it has no way to detect at inference time whether an essay is
out-of-distribution.

**Round 6: this was retested and adopted.** The original test found a
nonlinear classifier (RandomForest/GradientBoosting, same 12 features)
beats `LogisticRegression` in-genre (72.2% -> 75-77%) but gets *worse*
cross-genre (55.3% -> ~50%, close to chance) —
`experiment_nonlinear_classifier.py` — and was declined at the time on
the same in-genre-vs-cross-genre tradeoff logic as the race/ethnicity
fix above. It was retested once the OpenAI/Anthropic data existed
(hoping generator diversity would let a nonlinear model learn real
cross-generator signal instead of overfitting): **that hope didn't pan
out** — RandomForest/GradientBoosting still won in-genre and still lost
cross-genre by almost exactly the same margin with the diverse data as
without it (see #2's held-out-generator numbers). Given that, and given
the explicit priority "we need accuracy up" for this round, the tradeoff
was made deliberately: `GradientBoostingClassifier` was adopted anyway,
on Gemini-only training data (76.3%/74.1% in-genre, 51.2% DAIGT — #2
above), with the cross-genre cost disclosed rather than hidden. This
also required reworking the "why" evidence panel from linear
`coefficient x value` contributions to a perturbation-based method
(`evidence.py`, `classify.py`) — still fully deterministic, verified via
the full pytest suite and a manual browser check after the swap.

`score_volatility` **is now wired into `essay_verdict()`**, but only as
a mild, tail-only confidence penalty: comparing 327 pure-class training
essays against 30 genuinely mixed hybrid essays found weak separation at
the median (0.192 vs 0.204 — heavily overlapping) and real separation
only at the extreme tail (pure p95=0.256, hybrid max=0.321). The
integration reflects exactly that weak evidence — a confidence
reduction above a high threshold, not a change to the verdict label
itself. Don't expect this to catch most mixed essays; it wasn't shown to
be strong enough for that.

One counterintuitive, still-unexplained finding from evaluation: low
vocabulary variety (rolling type-token ratio) reads as *human-like* to
this model, cutting against the common assumption that repetitive text
is the AI tell. Not investigated.

## 7. Hybrid-essay construction — both variants now built, mid-paragraph is harder

Originally only the "continuation" splice (human essay truncated, AI
text appended) was built. **Now both plan-described variants exist**:
continuation and mid-paragraph replacement (AI text spliced into the
middle, human text on both sides). The mid-paragraph variant scores
meaningfully worse (F1 0.38 vs. 0.54 as of round 7, `docs/EVALUATION.md`
§4) — plausibly the more realistic real-world case, and the harder,
more-honest number to use if picking one. Round 6's classifier swap
pushed both up substantially (0.392/0.538 -> 0.427/0.606); round 7's
genre-diversification training gave most of that back (-> 0.384/0.541),
consistent with the same admissions-genre-only accuracy tradeoff
described in #2.

## 8. Coverage gaps

- English-language only. No non-English essays.
- Admissions/scholarship personal statements specifically — no STEM/technical
  supplement essays, no graduate statements of purpose.
- Word-count floors differ slightly by source (300 words for the
  EssayForum/DAIGT sources, 150 elsewhere) — essays shorter than that
  were filtered out during collection, so very short admissions essays
  are unrepresented in training.

## 9. Licensing notes (not a correctness limitation, but load-bearing for reuse)

The human training data (`nid989/EssayFroum-Dataset`) and the DAIGT eval
set (`Yunij/kaggle-comp-daigt`) are Hugging Face mirrors of scraped
content, self-tagged by their uploaders with permissive licenses that
don't clearly establish the underlying authors (EssayForum posters)
consented to that licensing. Treated as local-only throughout this
project (gitignored, never committed or redistributed) — see
`data/README.md`. PERSUADE/ELLIPSE were obtained properly via Kaggle
with the user's own credentials and are the official research corpora,
no equivalent concern there.

## What would close these gaps, roughly in priority order

1. **Push the genuinely-novel-generator recall in #2b past 35%** —
   still the single biggest open weakness, though round 8 made real
   progress (7.5%→30.0% via retraining on the 160-essay OpenAI/Anthropic
   training portion against a larger base). The next lever suggested by
   that result: more essays from these or other generators, now that
   2,246 essays has proven big enough to absorb new data without
   dilution — untested how much further this scales. A feature that
   captures generator-invariant "AI-ness" rather than exposure-dependent
   style would be the more fundamental fix, and
   wasn't found this round.
2. Understand *why* fairness (#1) moved the way it did in both rounds 6
   and 7 — currently just measured, not mechanistically traced the way
   the ELL/`gltr_pct_top10` fix was. Round 7's noisier numbers make this
   more urgent, not less: a mechanism nobody understands can't be
   predicted to hold on the next retrain either.
3. Recover some of the admissions-genre-only accuracy round 7 traded
   away (76.3%→70.6% in-theme) — likely means rebalancing the training
   mix (currently roughly 50/50 admissions/DAIGT by essay count) rather
   than accepting genre diversity and in-genre sharpness as a strict
   tradeoff.
4. Human verification for theme tagging on the human-essay side — AI
   essays are now solved via ground truth; human essays still rely on
   an unverified (if improved) heuristic.
5. A real fix for the mid-paragraph localization gap in #7 — currently
   the harder, more realistic case is also the one that lost the most
   ground in round 7 (F1 0.427 -> 0.384).
6. Investigate the unexplained disability-status reversal in #1
   (disabled students flagged *less* often) — not investigated at all
   so far, unlike the other three demographic axes.
