# CLAUDE.md — Business Entity Resolution Challenge

This file is the standing context for any AI agent (Claude Code or otherwise) working
in this repository. Read this before touching code. If anything you're about to do
conflicts with a rule below, stop and flag it rather than proceeding — these rules
map directly to hard disqualification/rejection conditions in the official challenge.

Full technical plan: `docs/architecture.md` (the design doc this file summarizes into
actionable rules). Official rules: `docs/problem_statement.pdf`.

---

## 1. What this project is

Entity resolution across three business-record sources (S1 = deduplicated reference,
S2/S3 = noisy). For every Source 1 entity, predict which S2/S3 records refer to the
same real-world business — zero, one, or many. Scored by F₀.₅, macro-averaged per
Source 1 entity, on a precision-weighted formula. This is a two-level system:
**Level 1 (blocking) optimizes recall. Level 2 (matching) optimizes precision.**
Never blur these two into one step.

---

## 2. Hard constraints — never violate these

- **File I/O**: all data is TSV. Always `pd.read_csv(path, sep="\t", dtype=str,
  keep_default_na=False)`. Never let `entity_id`, PIN codes, or house numbers be cast
  to numeric types. Never read a TSV without an explicit `sep="\t"`.
- **No external data of any kind.** No business-registry lookups, no geocoding APIs,
  no commercial entity-resolution services, no internet-sourced augmentation, no
  external gazetteers/dictionaries. Every normalization dictionary, city/state list,
  or abbreviation map must be **mined from the provided train/test files themselves**.
  If you (the agent) are about to fetch something from the web to help resolve
  entities or enrich records, don't — flag it instead.
- **Country is an open set.** Never write `if country in ["US", "India"]`, never
  one-hot encode country against a fixed 2-value vocabulary, never fit any
  categorical encoder for country on train alone and apply it unchanged at inference.
  Test data includes France, which never appears in train. Country must only feed
  compatibility/missingness/unseen-value features (see `docs/architecture.md` §6).
- **Model license/size**: final matching model must be MIT or Apache-2.0 licensed and
  ≤8B parameters. Classical ML (LightGBM/XGBoost/CatBoost) trivially satisfies this
  and is the default choice — don't reach for a large LLM for this task, it's a
  classification/scoring problem, not a generation problem.
- **`candidate_pairs.tsv` must be the actual, final candidate set scored by the
  matching model** — not a wider, earlier blocking pass you filter further downstream.
  If there's a cheap pre-filter stage before the real classifier, its *output* is what
  goes in this file.
- **Output format rules** (violating any of these fails validation and the run is not
  scored):
  - Every Source-1 test entity appears in `matching_results.tsv` exactly once.
  - No duplicate `source1_entity_id` rows.
  - No duplicate IDs within any single match/candidate list.
  - Matched/candidate IDs must be S2-/S3- prefixed and must exist in the **test**
    source files. Never include an S1- ID or an ID absent from test data.
  - Empty match list → empty string in the second column, not `"nan"`/`"None"`.
  - Every ID in `matching_results.tsv` for a given S1 entity must also appear in that
    entity's row in `candidate_pairs.tsv`.

---

## 3. Single sources of truth — never reimplement these

- **F₀.₅ metric**: `src/business_entity_resolution/evaluation/metrics.py::entity_f_beta`
  and `macro_f_beta`. Every experiment, every notebook, every model comparison must
  import and call these — do not write a second copy inline, and do not substitute
  `sklearn.metrics.fbeta_score`, plain F1, or pairwise AUC as a stand-in. Those are
  fine as secondary diagnostics, never as the number a decision is based on.
- **Normalization**: `src/business_entity_resolution/normalization/normalize.py`. All
  downstream code (blocking, features) consumes its output — never normalize inline
  in a blocking or feature function.
- **Validation**: `python3 utils/validate_submission.py --matching
  output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir
  dataset/test`. Run this after any change that touches inference or output writing,
  before considering that work done.

---

## 4. Module contracts

Code against these signatures; don't invent parallel ad hoc versions.

```python
# normalization
def normalize(record: dict) -> NormalizedRecord: ...
# returns: raw, normalized, tokenized, char_ngrams, digits_only, sorted_tokens
# (see docs/architecture.md §3 for the full representation spec)

# blocking
def generate_candidates(s1_df, s2_df, s3_df, config: dict) -> pd.DataFrame:
    """Returns candidate_pairs_df: [source1_entity_id, candidate_entity_id, channels]"""
def evaluate_blocking_recall(candidate_pairs_df, ground_truth_df) -> dict:
    """Returns per-channel and cumulative recall stats (docs/architecture.md §5)"""

# features
def build_pair_features(pairs_df, s1_df, s2_df, s3_df) -> pd.DataFrame:
    """Consumes NORMALIZED records only. Returns one feature row per candidate pair."""

# models — common interface across baseline / GBM / any neural variant
def fit(X, y) -> Model: ...
def predict_proba(model, X) -> np.ndarray: ...

# evaluation
def entity_f_beta(true_set: set[str], pred_set: set[str], beta: float = 0.5) -> float: ...
def macro_f_beta(all_true: dict[str, set[str]], all_pred: dict[str, set[str]], beta=0.5) -> float: ...
```

---

## 5. Definition of done, per module

- **Blocking**: cumulative recall ≥0.97 on the held-out validation split (measured via
  `evaluate_blocking_recall`, not eyeballed); recall table logged in
  `experiments/experiment_log.csv`.
- **Features**: every new feature has a one-line docstring explaining what noise
  pattern it targets; no feature silently assumes country ∈ {US, India}.
- **Model/threshold**: entity-level F₀.₅ on validation reported via `macro_f_beta`
  (not pairwise accuracy/AUC alone); threshold chosen by sweeping and maximizing this
  number, logged in the experiment log.
- **Inference/output**: `utils/validate_submission.py` passes with no errors before a
  PR is opened or a leaderboard file is uploaded.
- Any PR touching `src/` must include an updated row in
  `experiments/experiment_log.csv` if it changes blocking, features, model, or
  threshold behavior.

---

## 6. Validation split — do not break this

Split at the **Source 1 entity level** (grouped, stratified by singleton-vs-match and
roughly by country), never at the pair level. Never let pairs derived from the same
`source1_entity_id` appear in both train and validation — this is the most common way
an agent accidentally inflates validation numbers. See `docs/architecture.md` §8.

---

## 7. Things agents commonly get wrong on this task — don't do these

- Don't drop or infer `NaN` in address/name fields as if a fully missing field —
  distinguish "empty string present" from "field genuinely absent" where the data
  makes that distinguishable, and treat missingness itself as a feature, not just 0.
- Don't force one-to-one S1↔candidate matching. Zero, one, or many is correct by
  spec — apply independent per-pair thresholding by default (§13 of the architecture
  doc), don't add a top-1-only constraint without validation evidence for it.
- Don't sample training negatives uniformly at random from the whole corpus — sample
  from your own blocker's output (including a dedicated hard-negative slice), so
  negative distribution at train time matches what the model sees at inference.
- Don't treat a false merge on a true singleton the same as a missed match on a
  multi-match entity when reasoning about errors — under this metric a false merge on
  a singleton is a full 1.0→0.0 swing; singleton precision deserves disproportionate
  care (§12).
- Don't hardcode legal-suffix/abbreviation dictionaries from general knowledge of
  English/Hindi business naming conventions — mine them from the actual train/test
  corpus (this keeps the pipeline both rule-compliant and France-ready).
- Don't fetch anything from the internet to help resolve an entity, normalize an
  address, or validate a business name. If a task seems to need this, stop and say so
  instead of working around it.

---

## 8. Repo map

See `docs/architecture.md` §19 for the full structure. Quick pointers:
- `src/business_entity_resolution/` — all real logic, organized by module (§4 above).
- `notebooks/` — thin orchestration only; call into `src/`, don't inline logic, so
  diffs stay reviewable.
- `configs/normalization/` — mined dictionaries (suffixes.json, abbreviations.json),
  versioned, never hardcoded inline in code.
- `experiments/experiment_log.csv` — the shared experiment ledger; every experiment
  gets a row (schema in `docs/architecture.md` §15).
- `output/` — `matching_results.tsv` and `candidate_pairs.tsv`, always regenerated by
  `src/business_entity_resolution/inference/`, never hand-edited.

---

## 9. Before you consider any task finished

1. Run `python3 utils/validate_submission.py ...` if output files changed.
2. Confirm `macro_f_beta` (not a substitute metric) was used to justify any modeling
   or threshold decision.
3. Confirm no new code branches on a hardcoded country list.
4. Add/update the relevant row in `experiments/experiment_log.csv`.
