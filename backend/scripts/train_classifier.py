"""
Train the classifier — LogisticRegression(class_weight='balanced') on
standardized features (plan §5), evaluated the way plan §3c asks for:
two numbers, not one.

  1. In-theme held-out accuracy: random essay-level 80/20 split WITHIN
     themes seen during training.
  2. Unseen-theme accuracy: the same trained model evaluated on themes
     held out of training entirely.

The gap between the two is itself a result — it's the honest measure of
whether the model learned "AI-ness" or just topic cues. Both numbers come
from ONE model (trained on the 80% "seen" split) so they're directly
comparable. The model actually saved to classifier.joblib for the app is
a SEPARATE, final model retrained on all available data (seen + unseen) —
holding out themes is an evaluation technique, not a reason to permanently
shrink the deployed model's training set.

Splits are done at the ESSAY level, not the sentence level. Splitting
individual sentence-rows randomly (what this script did through Phase 4)
lets sentences from the same essay land on both sides of a split, which
leaks information (same author/topic/generation-run) and inflates
apparent accuracy — a real methodology bug, not just a placeholder-data
caveat, fixed here now that there's enough essay-level data to do the
split properly in the first place.

Trained on data/processed/ai_essays.csv (the user's real 500-essay Gemini
set — see docs/IMPLEMENTATION.md Phase 1 addendum 4) and
human_essays.csv (500 EssayForum essays). First run against real,
non-placeholder data on both sides.

Usage:
    cd backend && .venv/bin/python scripts/train_classifier.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

sys.path.insert(0, str(Path(__file__).parent.parent))
from app.pipeline.featurize import FEATURE_NAMES  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
MODELS_DIR = REPO_ROOT / "backend" / "app" / "models"

# The 3 smallest combined (human+AI) themes, recomputed after the theme-
# tagging heuristic improvement (docs/IMPLEMENTATION.md) — AI essays
# now use theme_mapping.py's direct category lookup instead of a keyword
# guess, and human essays use an improved word-boundary/most-matches-wins
# heuristic, both of which changed the theme distribution substantially.
# Previously background_identity/captivating_topic/gratitude (chosen when
# the AI side was badly lopsided); now challenging_belief(58)/gratitude(59)/
# obstacle_setback(134) are the genuine 3 smallest by the same "smallest
# combined count" rule.
HELD_OUT_THEMES = {"challenging_belief", "gratitude", "obstacle_setback"}

# Round 6: 200 essays from OpenAI (gpt-5.6-luna) and Anthropic
# (claude-haiku-4-5) were added to ai_essays.csv to test whether AI-
# generator diversity improves cross-genre/cross-generator detection.
# Tested against a ~1030-essay base: it didn't help (in-genre accuracy
# down, localization recall down ~12pts, held-out-generator recall only
# 30%) — declined, reverted to Gemini-only training.
#
# Round 7 added a 1,056-essay DAIGT-v2 slice (16 generators) and that
# DID fix cross-genre generalization for genres/generators with training
# exposure (DAIGT: 51%->78-93%) while leaving OpenAI/Anthropic recall
# unchanged at 7.5% (they're not in DAIGT either). Retesting the
# round-6-declined lever now that the base is 2,086 essays instead of
# ~1,030 — the dilution concern that sank it in round 6 may not apply at
# this scale. EXCLUDED_AI_ID_PREFIXES is empty again: openai-/anthropic-
# essays are back in training. score_heldout_generator.py's 40-essay
# split (still eval-only, untouched by this) is the number that tells us
# whether this round's retest actually worked.
EXCLUDED_AI_ID_PREFIXES = ()


def load_feature_table() -> list[dict]:
    path = PROCESSED_DIR / "feature_table_phase2.csv"
    with path.open() as f:
        rows = list(csv.DictReader(f))
    return [r for r in rows if not r["essay_id"].startswith(EXCLUDED_AI_ID_PREFIXES)]


def to_xy(rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    X = np.array([[float(r[name]) for name in FEATURE_NAMES] for r in rows])
    y = np.array([1 if r["class"] == "ai" else 0 for r in rows])
    return X, y


def main() -> None:
    rows = load_feature_table()
    print(f"{len(rows)} sentences | {sum(r['class']=='ai' for r in rows)} ai-labeled | "
          f"{sum(r['class']=='human' for r in rows)} human-labeled")

    # Essay-level metadata (one row per essay) drives every split below —
    # sentence rows just get gathered up afterward by essay_id membership.
    essays = {}
    for r in rows:
        essays.setdefault(r["essay_id"], {"theme_id": r["theme_id"], "class": r["class"]})
    essay_ids = list(essays)
    essay_themes = [essays[e]["theme_id"] for e in essay_ids]
    essay_classes = [essays[e]["class"] for e in essay_ids]

    seen_essays = [e for e, t in zip(essay_ids, essay_themes) if t not in HELD_OUT_THEMES]
    unseen_essays = [e for e, t in zip(essay_ids, essay_themes) if t in HELD_OUT_THEMES]
    seen_classes = [essays[e]["class"] for e in seen_essays]
    print(f"\n{len(seen_essays)} essays in seen themes, {len(unseen_essays)} essays in held-out "
          f"themes ({sorted(HELD_OUT_THEMES)})")

    train_essays, test_essays = train_test_split(
        seen_essays, test_size=0.2, random_state=42, stratify=seen_classes
    )
    train_essays, test_essays, unseen_essays = set(train_essays), set(test_essays), set(unseen_essays)

    train_rows = [r for r in rows if r["essay_id"] in train_essays]
    test_rows = [r for r in rows if r["essay_id"] in test_essays]
    unseen_rows = [r for r in rows if r["essay_id"] in unseen_essays]

    X_train, y_train = to_xy(train_rows)
    X_test, y_test = to_xy(test_rows)
    X_unseen, y_unseen = to_xy(unseen_rows)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_unseen_scaled = scaler.transform(X_unseen)

    # Round 6: GradientBoostingClassifier, not LogisticRegression — tested
    # directly (experiment_nonlinear_classifier.py) on this same Gemini-only
    # data: +4.1pt in-theme, +3.0pt unseen-theme, -4.7pt DAIGT cross-genre.
    # Adopted for the accuracy gain; the DAIGT cost is disclosed in
    # docs/LIMITATIONS.md. No class_weight param on this model, so balance
    # is emulated with sample_weight instead (same as the experiment script).
    # Round 9 (tested, declined): max_depth 5 looked like a small, free win
    # on admissions in-theme/unseen-theme/DAIGT (experiment_tune_gbm_
    # hyperparams.py showed +0.8/+0.5/+0.2pt, none regressed on THOSE three
    # metrics) — but score_heldout_generator.py's 40-essay genuinely-novel-
    # generator check (the metric round 8 specifically fixed, 7.5%->30%)
    # dropped to 17.5% under depth=5, the classic overfit-with-more-capacity
    # pattern seen before with RandomForest/GradientBoosting depth increases
    # elsewhere in this project. Reverted to depth=3. Lesson for future
    # tuning: always check held-out-generator recall specifically before
    # adopting any capacity increase — it's the canary metric for this
    # exact failure mode, and the other three metrics alone don't catch it.
    # Round 10 (adopted): n_estimators 200 -> 300, max_depth held at 3 —
    # unlike round 9's depth increase (more per-tree capacity, reverted
    # after it hurt held-out-generator recall), this adds more shallow
    # trees. Checked against the full metric suite before adopting.
    #
    # Bug, found and fixed during a later cleanup pass: this edit updated
    # ONLY this eval-model instantiation, not the final_clf instantiation
    # below (which stayed at n_estimators=200 for two full rounds without
    # anyone noticing, since it silently kept training and shipping a
    # model that didn't match what the eval numbers were reporting).
    # Every "round 10 numbers" claim reported afterward came from this
    # eval model, so those specific numbers were accurate — but the
    # actually-served classifier.joblib was still the round-8 config
    # until the fix. Re-running the full pipeline with both instantiations
    # correctly at (300, 3) changed several downstream eval numbers that
    # depend on the real production model (held-out-generator recall,
    # fairness FPR) — see docs/EVALUATION.md and docs/LIMITATIONS.md for
    # the corrected figures. Lesson: train_classifier.py has two
    # `GradientBoostingClassifier(...)` call sites by design (eval model
    # vs. production model, see the module docstring) — any hyperparameter
    # change must update both, and the full pipeline re-run is the only
    # thing that would have caught this sooner (it did, just later than
    # it should have).
    train_weights = compute_sample_weight("balanced", y_train)
    clf = GradientBoostingClassifier(n_estimators=300, max_depth=3, random_state=42)
    clf.fit(X_train_scaled, y_train, sample_weight=train_weights)

    in_theme_acc = accuracy_score(y_test, clf.predict(X_test_scaled))
    unseen_theme_acc = accuracy_score(y_unseen, clf.predict(X_unseen_scaled))

    print(f"\n=== Evaluation model (trained on {len(train_essays)} seen-theme essays) ===")
    print(f"1. In-theme held-out accuracy:   {in_theme_acc:.3f}  ({len(test_rows)} sentences, {len(test_essays)} essays)")
    print(f"2. Unseen-theme accuracy:        {unseen_theme_acc:.3f}  ({len(unseen_rows)} sentences, {len(unseen_essays)} essays)")
    print(f"   Gap (1 - 2):                  {in_theme_acc - unseen_theme_acc:+.3f}")
    print("\n-- In-theme classification report --")
    print(classification_report(y_test, clf.predict(X_test_scaled), target_names=["human", "ai"]))
    print("-- Unseen-theme classification report --")
    print(classification_report(y_unseen, clf.predict(X_unseen_scaled), target_names=["human", "ai"]))

    # Round 7: since daigt_training_slice.csv (theme_id="cross_genre_daigt")
    # is now mixed into "seen" themes, #1/#2 above are BLENDED across
    # admissions + persuasive genre. Report the admissions-genre-only
    # subset too, so this round's numbers stay comparable to rounds 1-6
    # (which were pure admissions-genre) rather than silently changing
    # what the headline numbers mean.
    admissions_test_rows = [r for r in test_rows if r["theme_id"] != "cross_genre_daigt"]
    admissions_unseen_rows = [r for r in unseen_rows if r["theme_id"] != "cross_genre_daigt"]
    if admissions_test_rows and admissions_unseen_rows:
        X_adm_test, y_adm_test = to_xy(admissions_test_rows)
        X_adm_unseen, y_adm_unseen = to_xy(admissions_unseen_rows)
        adm_in_theme_acc = accuracy_score(y_adm_test, clf.predict(scaler.transform(X_adm_test)))
        adm_unseen_acc = accuracy_score(y_adm_unseen, clf.predict(scaler.transform(X_adm_unseen)))
        print(f"\n-- Admissions-genre-only subset (comparable to rounds 1-6) --")
        print(f"   In-theme:    {adm_in_theme_acc:.3f}  ({len(admissions_test_rows)} sentences)")
        print(f"   Unseen-theme: {adm_unseen_acc:.3f}  ({len(admissions_unseen_rows)} sentences)")

    # Save the eval model's own held-out predictions (test + unseen rows —
    # never trained on by THIS model) for threshold calibration
    # (calibrate_thresholds.py). Deliberately not the production model's
    # predictions on these same rows, which would be trained-on-and-tested-on
    # leakage for calibration purposes even though it's fine for the
    # production model to also be trained on them.
    eval_predictions_path = PROCESSED_DIR / "eval_model_predictions.csv"
    with eval_predictions_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["essay_id", "sentence_idx", "class", "sent_length", "score", "split"])
        writer.writeheader()
        for split_name, split_rows, X_split_scaled in [("in_theme", test_rows, X_test_scaled), ("unseen_theme", unseen_rows, X_unseen_scaled)]:
            scores = clf.predict_proba(X_split_scaled)[:, 1]
            for r, score in zip(split_rows, scores):
                writer.writerow({
                    "essay_id": r["essay_id"],
                    "sentence_idx": r["sentence_idx"],
                    "class": r["class"],
                    "sent_length": r["sent_length"],
                    "score": score,
                    "split": split_name,
                })
    print(f"\nSaved eval-model held-out predictions to {eval_predictions_path.relative_to(REPO_ROOT)}")

    # Final production model: retrained on ALL essays (seen + unseen) — the
    # held-out-theme split above is an evaluation technique, not a reason to
    # ship a model that's missing three themes' worth of training data.
    X_all, y_all = to_xy(rows)
    final_scaler = StandardScaler()
    X_all_scaled = final_scaler.fit_transform(X_all)
    all_weights = compute_sample_weight("balanced", y_all)
    final_clf = GradientBoostingClassifier(n_estimators=300, max_depth=3, random_state=42)
    final_clf.fit(X_all_scaled, y_all, sample_weight=all_weights)

    # Reference stats: per-feature percentile lookup from the HUMAN TRAINING
    # class only (plan §4) — used for evidence phrasing in later phases, not
    # for classification.
    human_mask = y_all == 0
    reference_stats = {
        name: np.sort(X_all[human_mask, i])
        for i, name in enumerate(FEATURE_NAMES)
    }

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_clf, MODELS_DIR / "classifier.joblib")
    joblib.dump(final_scaler, MODELS_DIR / "scaler.joblib")
    joblib.dump(reference_stats, MODELS_DIR / "reference_stats.joblib")
    print(f"\nSaved production classifier.joblib/scaler.joblib/reference_stats.joblib "
          f"(trained on all {len(essay_ids)} essays) to {MODELS_DIR.relative_to(REPO_ROOT)}")
    print("(The two evaluation numbers above are from a separate held-out model, not this one.)")


if __name__ == "__main__":
    main()
