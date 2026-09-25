# Task Specs — Business Entity Resolution Challenge

Self-contained specs, one per module, so an agent (or teammate) can pick up a single
ticket without needing the full `architecture.md` in context every time. Each links
back to the relevant architecture section for depth. Skeleton code for every module
already exists in `src/business_entity_resolution/` — most tasks below are "extend
this" rather than "start from scratch."

---

## 1. Data Ingestion & Validation
**Owner focus:** EDA + Normalization member (architecture.md §20)
**Module:** `src/business_entity_resolution/data/ingest.py` (implemented — extend, don't replace)

- **Goal:** load every TSV correctly and surface data-quality issues before anything downstream trusts the data.
- **Inputs:** paths to `train_source{1,2,3}.tsv`, `train_ground_truth.tsv`, `test_source{1,2,3}.tsv`.
- **Outputs:** validated DataFrames + a written EDA report (`notebooks/01_eda.ipynb` output, or a `docs/data_dictionary.md` filled from the template).
- **Already implemented:** `load_tsv`, `validate_schema`, `validate_entity_id_prefix`, `country_value_counts`.
- **To do:**
  - Run `validate_schema` on all 7 files; fix or log every issue found (don't silently drop rows).
  - Fill in `docs/data_dictionary_template.md` with real stats (this becomes `docs/data_dictionary.md`).
  - Compute and save the ground-truth EDA stats from architecture.md §2 (singleton rate, one-to-one vs one-to-many, per-country match rates).
- **Acceptance criteria:** `validate_schema` returns `[]` on all files (or every non-empty issue is explicitly triaged and documented, not ignored); `docs/data_dictionary.md` exists and is filled in, not the placeholder template.
- **Test cases:** run against `fixtures/train/` and `fixtures/test/` first (fast, no full dataset needed) before running against real data.
- **Gotchas:** never let `country_value_counts` output become a hardcoded filter anywhere downstream (CLAUDE.md §2).

---

## 2. Normalization
**Owner focus:** EDA + Normalization member
**Module:** `src/business_entity_resolution/normalization/normalize.py` (core transforms implemented — dictionaries are empty placeholders)

- **Goal:** produce the multi-representation normalized fields (architecture.md §3) and populate real suffix/abbreviation dictionaries mined from the data.
- **Inputs:** raw `business_name`/`business_address`/`country` strings.
- **Outputs:** `NormalizedRecord` per record; populated `configs/normalization/suffixes.json` and `abbreviations.json`.
- **To do:**
  1. From EDA (`01_eda.ipynb`), extract the frequency table of name-tail tokens (candidate legal suffixes) and address-token abbreviations.
  2. Build `suffixes.json` / `abbreviations.json` as `{"observed": "canonical"}` maps — canonicalize to whichever form is more frequent, or to a single consistent choice; what matters is that both variants map to the *same* canonical string.
  3. Extend `normalize_field`/`NormalizedRecord` with the `address_components` soft extraction described in architecture.md §3 (house number, street, city-candidate, state-candidate, postal-code, landmark-phrase) — needed by several blocking channels and address features.
  4. Add unit tests confirming known noisy pairs from the fixtures normalize to matching or near-matching representations (e.g. "Sharma Traders Pvt Ltd" vs "Sharma Traders Private Limited").
- **Acceptance criteria:** dictionaries are non-empty and demonstrably derived from real data (cite the frequency counts in a comment or a short `docs/normalization_notes.md`); `address_components` extraction implemented and unit-tested.
- **Gotchas:** never delete digit tokens; keep legal-suffix tokens in the `normalized`/`tokenized` representations even after mapping them to a canonical form (map, don't strip) — see CLAUDE.md §7.

---

## 3. Blocking / Candidate Generation
**Owner focus:** Blocking member
**Module:** `src/business_entity_resolution/blocking/block.py` (Channel A implemented — B through G are stubs)

- **Goal:** implement Channels B–G (architecture.md §4), each measured independently and cumulatively via `evaluate_blocking_recall` (already implemented).
- **Inputs:** normalized S1/S2/S3 records.
- **Outputs:** `candidate_pairs_df` via `generate_candidates`; a recall table (architecture.md §5) logged per channel addition.
- **To do, in priority order** (validate recall after each before moving to the next):
  1. **Channel B** — name token inverted index, excluding high-DF tokens (`max_token_df` param already threaded through).
  2. **Channel C** — name character n-gram TF-IDF retrieval (typo/transliteration safety net). Use `sklearn.feature_extraction.text.TfidfVectorizer(analyzer="char", ngram_range=(3,4))` fit on the combined name corpus; retrieve top-K via sparse cosine similarity, or an ANN index if S2∪S3 is large (architecture.md §22).
  3. **Channel D** — address token inverted index (same pattern as B).
  4. **Channel E** — PIN/postal exact match, using `address_components.postal_code` from task 2.
  5. **Channel F** — city/state co-occurrence, using `address_components.city_candidate`/`state_candidate`.
  6. **Channel G** — address character n-gram retrieval (same pattern as C).
  7. Flip each channel to `true` in `configs/pipeline.yaml` only after its standalone + cumulative recall is measured and logged.
- **Acceptance criteria:** full-union cumulative recall ≥0.97 on the validation split (CLAUDE.md §5); every true pair missed by the full union manually audited (architecture.md §5) and the audit notes saved.
- **Test cases:** `fixtures/train/` first — note the current fixture only recovers candidates through exact-name matching (Channel A), so implementing B/C/D should visibly increase recall on this fixture (e.g., Channel D should surface S1-0001↔S3-0001 despite the very different name strings).
- **Gotchas:** `candidate_pairs.tsv` (via `generate_candidates`'s output, formatted) must be the *exact* final candidate set scored downstream — don't add a filtering step after this that isn't reflected in the file (CLAUDE.md §2).

---

## 4. Pair Features
**Owner focus:** Pairwise Features + Classical ML member
**Module:** `src/business_entity_resolution/features/pair_features.py` (name/address/country/cross-field subset implemented)

- **Goal:** complete the feature set from architecture.md §6.
- **To do:**
  - Address: PIN exact-match + missingness flags, city/state token match, house-number compatibility, landmark-token overlap (depends on `address_components` from task 2).
  - Name: legal-suffix compatibility flag, rare-token overlap (weight by inverse document frequency, not just raw overlap).
  - Cross-field: blocking-provenance one-hot features from the `channels` column in `candidate_pairs_df` (join it into `build_pair_features`'s output).
- **Acceptance criteria:** every new feature has a one-line docstring naming the noise pattern it targets (CLAUDE.md §5); no feature assumes a fixed country vocabulary.
- **Test cases:** confirm the deliberate hard-negative fixture pair (`S1-0004`/`S3-0003`, "45 Oak Ave" vs "47 Oak Ave") gets a *lower* score than the true positive (`S1-0004`/`S2-0004`) once house-number-compatibility is added — this fixture pair currently scores ambiguously high with only the base feature set, which is exactly the gap this task closes.
- **Gotchas:** consume `NormalizedRecord` only — never re-derive similarity from raw strings inline here.

---

## 5. Model & Training
**Owner focus:** Advanced Matching + Tuning member (with Pairwise Features member on the baseline/GBM path)
**Module:** `src/business_entity_resolution/models/model.py` (`RuleBasedBaseline` and `LightGBMMatcher` implemented)

- **Goal:** train the primary GBM matcher; construct proper train pairs with hard negatives (architecture.md §7).
- **To do:**
  1. Implement hard-negative mining: rank blocked-but-negative pairs by a cheap composite score, take the top slice as hard negatives, oversample relative to easy negatives.
  2. Train `LightGBMMatcher` on the grouped train split (from `validation/split.py`), evaluate with `macro_f_beta` on the held-out validation split — never plain pairwise accuracy/AUC as the deciding number.
  3. Log every run to `experiments/experiment_log.csv` (schema already in the file header).
  4. Pull feature importances (`LightGBMMatcher.feature_importance()`) to confirm/refute the "likely most discriminative" features guessed in architecture.md §6.
- **Acceptance criteria:** GBM beats `RuleBasedBaseline` on validation entity-level F₀.₅ (if it doesn't, that's a valid, important finding — log it and investigate why, don't just ship the baseline silently).
- **Gotchas:** never fit anything (including any encoder/embedding vocabulary) on the full train+test country set in a way that would look different if France were absent from train.

---

## 6. Threshold & Decision Logic
**Owner focus:** Advanced Matching + Tuning member
**Modules:** `src/business_entity_resolution/inference/predict.py` (threshold applied here), a new `evaluation/threshold_search.py` (create this)

- **Goal:** implement the threshold sweep from architecture.md §11 and update `configs/pipeline.yaml`'s placeholder `threshold.value`.
- **To do:** write a function that, given validation predictions and true labels, sweeps threshold values and returns the table from §11 (precision/recall/F₀.₅/singleton accuracy/false-merge rate/avg matches per entity); select and persist the value that maximizes entity-level F₀.₅.
- **Acceptance criteria:** the chosen threshold is *not* 0.5 by default and is justified by a saved sweep table, not eyeballed.
- **Gotchas:** only add segmented thresholds (by source, channel-provenance, or country) if the sweep shows a material, reproducible difference — don't add complexity speculatively (architecture.md §11).

---

## 7. Validation & Evaluation
**Owner focus:** Evaluation + Ensemble + Submission member
**Modules:** `src/business_entity_resolution/validation/split.py`, `evaluation/metrics.py` (both implemented — extend as needed), `evaluation/experiment_log.py` (create this)

- **Goal:** own the shared metric, the grouped split, and the experiment ledger; make sure every other module's claims are measured consistently.
- **To do:**
  - Write a small `experiment_log.py` helper (`append_experiment_row(...)`) so every experiment is logged the same way instead of hand-edited CSV rows.
  - Build the error-analysis dashboard from architecture.md §23 (false positive/negative breakdown by country/similarity bucket/candidate rank/etc.) as a notebook (`06_error_analysis.ipynb`) calling into `src/`.
  - Run K-fold grouped validation (`validation.split.grouped_kfold_entity_splits`) for any change under serious consideration, not just a single holdout.
- **Acceptance criteria:** every number quoted in a PR or standup traces back to `macro_f_beta`/`precision_recall_f_beta_breakdown` output that's actually saved somewhere (log file, notebook output), not just remembered/reported verbally.

---

## 8. Inference & Submission
**Owner focus:** Evaluation + Ensemble + Submission member
**Module:** `src/business_entity_resolution/inference/predict.py` (implemented — orchestration only, extend as models change), `validation/local_checks.py` (implemented)

- **Goal:** produce final `output/matching_results.tsv` and `output/candidate_pairs.tsv`, validated.
- **To do:**
  - Wire in the trained (non-baseline) model and tuned threshold from tasks 5–6.
  - Run `local_checks.check_matching_results` and `check_matches_are_subset_of_candidates` immediately after generating output (fast, in-process).
  - Then run the **official** `utils/validate_submission.py` before any leaderboard upload (CLAUDE.md §3, §9) — the local checks are a convenience, not a substitute.
- **Acceptance criteria:** official validator prints `PASS` with exit code 0.
