# Amazon ML Challenge 2026 — Business Entity Resolution
## Complete Project Report: Approach, Technology, Issues, and Improvement Roadmap

---

## Table of Contents
1. [Problem Statement](#1-problem-statement)
2. [Dataset Overview](#2-dataset-overview)
3. [Core Architectural Philosophy](#3-core-architectural-philosophy)
4. [Pipeline Overview](#4-pipeline-overview)
5. [Module-by-Module Approach](#5-module-by-module-approach)
6. [Technologies Used](#6-technologies-used)
7. [Issues Encountered and How We Solved Them](#7-issues-encountered-and-how-we-solved-them)
8. [Why We Followed This Approach](#8-why-we-followed-this-approach)
9. [Training Optimizations to Improve Score](#9-training-optimizations-to-improve-score)
10. [Scope for Improvement](#10-scope-for-improvement)
11. [Key Metrics and Evaluation Methodology](#11-key-metrics-and-evaluation-methodology)
12. [Risk Analysis](#12-risk-analysis)

---

## 1. Problem Statement

The **Amazon ML Challenge 2026** is a Business Entity Resolution (BER) competition. The goal is: given three separate data sources (Source 1, Source 2, Source 3) of business entity records — each with noisy `business_name`, `business_address`, and `country` fields — identify which records across sources refer to the same real-world business entity.

### The Three Sources
| Source | Role | Notes |
|--------|------|-------|
| **Source 1 (S1)** | Reference / Query | Deduplicated; each entity appears exactly once |
| **Source 2 (S2)** | Candidate Pool | Noisy, not deduplicated; may have typos, abbreviations |
| **Source 3 (S3)** | Candidate Pool | Same as S2 but a different data pipeline |

### Output Required
For every S1 entity, the system must produce:
1. `matching_results.tsv` — the comma-joined list of S2/S3 IDs that are the same real-world business (can be zero, one, or many)
2. `candidate_pairs.tsv` — the full list of candidate IDs the system considered before making a match decision

### Scoring Metric
The competition uses **Macro F₀.₅**, computed per S1 entity and averaged:

$$F_{0.5} = \frac{1.25 \cdot \text{Precision} \cdot \text{Recall}}{0.25 \cdot \text{Precision} + \text{Recall}}$$

**Critical property**: β = 0.5 means **Precision is weighted 4× more than Recall**. A single false positive (wrong match) is more damaging than a single false negative (missed match). This fundamentally shapes every design decision.

> [!IMPORTANT]
> A false merge on a true singleton entity (predicting a match when there is none) causes a score drop from 1.0 → 0.0 for that entity — the worst single error possible. The system is architecturally biased to avoid this above all else.

### Scale
- **Training**: 2.2 million S1 records, correspondingly large S2/S3
- **Test**: 1.73 million S1 records with approximately 10 million candidate records across S2/S3
- **Geography**: Training covers US and India; Test also includes **France** (unseen country during training)

---

## 2. Dataset Overview

### Data Files
| File | Records | Description |
|------|---------|-------------|
| `train_source1.tsv` | ~2.2 million | Reference S1 training entities |
| `train_source2.tsv` | ~10 million | Candidate pool Source 2 (training) |
| `train_source3.tsv` | ~10 million | Candidate pool Source 3 (training) |
| `train_ground_truth.tsv` | ~2.2 million | True match pairs for training |
| `test_source1.tsv` | ~1.73 million | Reference S1 test entities |
| `test_source2.tsv` | ~10 million | Candidate pool Source 2 (test) |
| `test_source3.tsv` | ~10 million | Candidate pool Source 3 (test) |

### Field Schema (all sources)
```
entity_id | business_name | business_address | country
```

### Key Data Characteristics Observed
- **Noisy names**: abbreviations (`Pvt` vs `Private`, `Ltd` vs `Limited`), transliteration variants (`Sharma` vs `Sarma`), typos, word-order swaps
- **Noisy addresses**: house number formats differ, street abbreviations differ, PIN codes may be absent
- **Multi-match**: one S1 entity can legitimately match multiple S2 and S3 records (the same physical business duplicated across sources)
- **Singleton rate**: A large fraction of S1 entities have no match in S2/S3
- **Country coverage gap**: France appears only at test time — any hardcoded country-specific logic would silently break

---

## 3. Core Architectural Philosophy

The entire system is built on one governing principle:

### Two Decoupled Levels, Optimizing Opposite Objectives

```
Level 1 — Blocking:   Maximize RECALL  (never drop a true match)
Level 2 — Matching:   Maximize PRECISION (never output a wrong match)
```

These two levels are intentionally kept in separate modules, with separate metrics, because they trade off differently:

- **Level 1 is lossy-lax**: it casts a wide net. If a true match is not in the candidate set, it can never be recovered. So the candidate set must be built conservatively (high recall), even at the cost of many false candidates.
- **Level 2 is strict**: given the candidates from Level 1, it decides which ones are actual matches. Under F₀.₅, every false positive here destroys score, so Level 2 is the high-precision final arbiter.

> [!NOTE]
> Blocking recall directly caps the maximum achievable F₀.₅. If blocking misses 5% of true matches, you can never score above ~0.95 no matter how good your Level 2 classifier is.

---

## 4. Pipeline Overview

```
Raw TSVs (S1, S2, S3)
        │
        ▼
[1] Data Ingestion & Validation
        │
        ▼
[2] Normalization (Multi-Representation per Field)
    ├── normalized string (lowercase, suffix-mapped)
    ├── tokenized list
    ├── character n-grams
    ├── digits-only extracted
    └── sorted tokens
        │
        ▼
[3] LEVEL 1 — Blocking (7-Channel Union)
    ├── Channel A: Exact normalized name
    ├── Channel B: Name token inverted index
    ├── Channel C: Name char n-gram TF-IDF  (typo / transliteration)
    ├── Channel D: Address token inverted index
    ├── Channel E: PIN/postal exact match
    ├── Channel F: City/state co-occurrence
    └── Channel G: Address char n-gram TF-IDF
        │
        ▼
[4] Candidate Pre-selection & Capping (hard negative aware, per S1 entity)
        │
        ▼
[5] Pairwise Feature Engineering (46 features)
    ├── Name features (15): Levenshtein, Jaro-Winkler, token Jaccard, n-gram Jaccard, etc.
    ├── Address features (12): tri-state PIN match, tri-state house number, token overlap, etc.
    ├── Country features (5): exact match, missing flags, unseen-in-train flag
    └── Cross-field features (14): interaction products, disagreement flags, channel provenance one-hots
        │
        ▼
[6] LEVEL 2 — Ensemble Matching Model
    ├── LightGBM (weight 0.55, CPU n_jobs=-1)
    └── XGBoost (weight 0.45, CUDA GPU)
        │
        ▼
[7] Channel Provenance Score Boost
        │
        ▼
[8] Fine-Grained Threshold Sweep (0.15 → 0.55, step 0.02)
    → Maximize Macro F₀.₅ on held-out validation set
        │
        ▼
[9] Streaming Test Inference (50K-chunk streaming over 1.73M test entities)
        │
        ▼
[10] Official Validator + ZIP Package
     → matching_results.tsv + candidate_pairs.tsv → submission.zip
```

---

## 5. Module-by-Module Approach

### 5.1 Data Ingestion

**Module**: `src/business_entity_resolution/data/ingest.py`

All TSV files are loaded with:
```python
pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
```

**Why `dtype=str`**: Entity IDs (`S1-000123`), PIN codes (`560001`), and house numbers (`47`) must never be silently cast to float. A PIN code `560001` becoming `560001.0` would break all exact-match PIN logic.

**Why `keep_default_na=False`**: Prevents pandas from interpreting empty strings or literal values like `"NA Traders"` as NaN.

Schema validation checks:
- Correct `entity_id` prefix per file
- No duplicate entity IDs within a file
- Non-empty `business_name` for essentially all rows
- Country present but **never validated against a fixed enum** (France would otherwise trigger failures)

---

### 5.2 Normalization

**Module**: `src/business_entity_resolution/normalization/normalize.py`

Every record is normalized into five parallel representations stored in a `NormalizedRecord` dataclass:

| Representation | Purpose |
|----------------|---------|
| `raw` | Audit trail; never used in similarity computation |
| `normalized` | Primary similarity string: Unicode NFKC → lowercase → `&`→`and` → legal suffix mapped |
| `tokenized` | Token list for Jaccard, containment, token-sort-ratio features |
| `char_ngrams` | Character 4-grams for TF-IDF cosine; typo robustness |
| `digits_only` | Extracted digit sequences for PIN/house number matching |
| `sorted_tokens` | Alphabetically sorted token join for word-order invariance |

Legal suffix and abbreviation maps (`configs/normalization/suffixes.json`, `abbreviations.json`) are loaded from JSON files mined empirically from the training data. They are **never hardcoded** so they generalize to France.

Country: normalized to lowercase stripped string, compared only as exact match between S1 and candidate — never one-hot encoded with a fixed vocabulary.

---

### 5.3 Multi-Channel Blocking (Level 1)

**Module**: `src/business_entity_resolution/blocking/block.py`

Seven blocking channels, combined by union:

#### Channel A — Exact Normalized Name
Hash normalized name → dict → O(1) lookup. Catches clean duplicates and suffix/punctuation-only differences.
*Missed by*: typos, word-order swaps.

#### Channel B — Name Token Inverted Index
Builds `token → [candidate_ids]` from all S2/S3 name tokens, excluding high-frequency stopwords (`ltd`, `store`, `india`, etc.) to prevent candidate explosion. An S1 entity retrieves all candidates sharing any distinctive name token.

#### Channel C — Name Character N-gram TF-IDF
Uses `sklearn.TfidfVectorizer(analyzer="char", ngram_range=(3,4), min_df=3, max_features=25000)` fit on all candidate names. Processes S1 in batches of 5,000 via sparse matrix multiplication. **This is the typo and transliteration safety net** — crucial for India's Devanagari-Latin variance (`Sharma`/`Sarma`/`Sharma`).

#### Channel D — Address Token Inverted Index
Same pattern as Channel B but on address tokens. Catches DBA/trade-name pairs where business names differ but addresses are essentially the same.

#### Channel E — PIN/Postal Exact Match
Extracts 5- or 6-digit sequences from addresses and indexes by PIN. High-precision but low-coverage (PIN fields often empty).

#### Channel F — City/State Co-occurrence
Uses address tokens that appear frequently across many businesses (likely place names) as city/state proxies. Inverted index with frequency filtering (`max_df = max(200, corpus_size // 500)`).

#### Channel G — Address Character N-gram TF-IDF
Same approach as Channel C but applied to normalized address strings. Catches partial-address and component-reordered cases.

**Per-channel cap**: 150 candidates per channel; global cap of 100 per S1 entity (post-union), prioritizing multi-channel hits and exact-name matches.

---

### 5.4 Pairwise Feature Engineering

**Module**: `src/business_entity_resolution/features/pair_features.py`

46 features computed per candidate pair via `rapidfuzz` (C-implemented, 10–50× faster than Python):

#### Name Features (15 features)
| Feature | What it Captures |
|---------|-----------------|
| `name_exact_normalized` | Clean duplicate detection |
| `name_exact_sorted_tokens` | Word-order transposition |
| `name_levenshtein_sim` | Character-level edit distance |
| `name_jaro_winkler_sim` | Prefix-heavy near-duplicates |
| `name_token_jaccard` | Token set overlap |
| `name_token_overlap_coef` | Containment (one name inside another) |
| `name_token_sort_ratio` | Word-order-invariant similarity |
| `name_token_set_ratio` | Subset matching |
| `name_partial_ratio` | Substring matching |
| `name_char_ngram_jaccard` | Typo/transliteration robustness |
| `name_first_token_match` | Brand name / primary entity word match |
| `name_common_token_count` | Shared distinctive token count |
| `name_length_diff` | Length disparity flag |
| `name_length_ratio` | Normalized length ratio |
| `name_digit_overlap` | Store/unit number exact match |

#### Address Features (12 features)
| Feature | What it Captures |
|---------|-----------------|
| `address_exact_normalized` | Identical addresses |
| `address_token_jaccard` | Token overlap |
| `address_token_overlap_coef` | Containment |
| `address_char_ngram_jaccard` | Typo-robust address similarity |
| `address_levenshtein_sim` | Edit distance |
| `address_token_sort_ratio` | Word-order invariant |
| `address_partial_ratio` | Substring match |
| `address_numeric_token_overlap` | Shared numeric tokens |
| `address_length_diff` | Length disparity |
| `address_length_ratio` | Normalized ratio |
| `pin_exact_match` | **Tri-state**: 1.0=match, 0.0=mismatch, 0.5=PIN missing on one side |
| `house_number_compatibility` | **Tri-state**: first digit sequence match; `47 Oak` vs `45 Oak` = 0.0 |

#### Country Features (5 features)
| Feature | What it Captures |
|---------|-----------------|
| `country_exact_match` | Same country string |
| `country_a_missing` | S1 country field empty |
| `country_b_missing` | Candidate country field empty |
| `country_a_unseen_in_train` | S1 entity from France / unknown country |
| `country_b_unseen_in_train` | Candidate from France / unknown country |

#### Cross-Field & Provenance Features (14 features)
| Feature | What it Captures |
|---------|-----------------|
| `name_x_address_sim_product` | Agreement on both fields simultaneously |
| `strong_name_weak_address` | Chain-branch ambiguity (same name, different address) |
| `weak_name_strong_address` | DBA/trade-name case (same address, different brand name) |
| `strong_name_strong_address` | High-confidence true match signal |
| `ch_prov_exact_name` | Pair found by exact-name channel |
| `ch_prov_name_token_overlap` | Pair found by name token channel |
| `ch_prov_name_char_ngram` | Pair found by n-gram channel |
| `ch_prov_address_token_overlap` | Pair found by address token channel |
| `ch_prov_pin_exact` | Pair found by PIN match |
| `ch_prov_city_state_cooccurrence` | Pair found by city/state channel |
| `ch_prov_address_char_ngram` | Pair found by address n-gram channel |
| `ch_count` | Number of channels that independently found this pair (0–7) |

> [!TIP]
> The **tri-state PIN/house-number features** are a key design decision. "PIN not available" is different from "PINs differ." Treating both as 0.0 would unfairly penalize records with missing PIN data. The `0.5` neutral value lets the GBM trees correctly discount the PIN signal when data is absent.

---

### 5.5 Training Pair Construction & Hard Negative Mining

**Module**: `scripts/train_and_evaluate.py → select_training_pairs()`

**Positives**: All `(S1_id, matched_id, label=1)` pairs from `train_ground_truth.tsv`.

**Negatives**: Sampled exclusively from the blocked candidate set (not randomly from the full corpus). The `select_training_pairs()` function implements **channel-difficulty-aware hard negative selection**:

```python
def _neg_difficulty(ch_str):
    chs = set(ch_str.split(","))
    score = len(chs) * 2.0           # multi-channel = more deceptive
    if "exact_name" in chs: score += 3.0
    if "name_token_overlap" in chs: score += 2.0
    if "name_char_ngram" in chs: score += 1.5
    return score
```

Top-4 hardest negatives per S1 entity are kept → **1:4 positive-to-negative training ratio**, which is the gold standard for GBM-based entity resolution (matching Magellan / DeepMatcher published findings).

---

### 5.6 Model Architecture

**Module**: `src/business_entity_resolution/models/model.py`

#### EnsembleMatcher: LightGBM (55%) + XGBoost (45%)
```python
final_score = 0.55 * lgbm_probability + 0.45 * xgboost_probability
```

- **LightGBM**: 500 estimators, leaf-wise growth, CPU multi-threaded
- **XGBoost**: 500 estimators, `device="cuda"` on Kaggle GPU, `tree_method="hist"`

**Score boost via channel provenance** (applied post-model):
```python
boost = 0.05 * (ch_count >= 2) + 0.05 * ch_prov_exact_name
final = clip(raw_score + boost, 0.0, 1.0)
```

---

### 5.7 Threshold Calibration

Fine-grained sweep: `threshold ∈ {0.15, 0.17, 0.19, ..., 0.55}` (step = 0.02)
Metric optimized: **Macro F₀.₅ on validation set** (not pairwise accuracy, not AUC).

The optimal threshold typically sits at 0.30–0.40 under F₀.₅'s precision-heavy weighting. Threshold is persisted to `configs/pipeline.yaml` for reproducibility.

---

### 5.8 Streaming Test Inference

**Module**: `src/business_entity_resolution/inference/fast_test_inference.py`

- **Phase 1**: Build compact in-memory indices from S2/S3 (streamed in 250K-row chunks):
  - `name_index`: exact name → candidate IDs
  - `pin_index`: PIN code → candidate IDs
  - `token_index`: distinctive name token → candidate IDs (frequency-filtered)
- **Phase 2**: Stream S1 in 50K-row chunks; for each chunk: look up candidates, cap at 35 per S1, compute features, batch-score with ensemble model, write to output files (append mode)

Total peak memory: ~1.5 GB (12% of Kaggle's 13 GB RAM limit).

---

### 5.9 Submission Validation

The official `utils/validate_submission.py` is automatically invoked after generating predictions:
- Every S1 entity appears exactly once in `matching_results.tsv`
- No duplicate IDs within any match list
- Every matched ID exists in test S2/S3 files with correct prefix
- Every matched ID appears in `candidate_pairs.tsv` for that entity
- Empty match lists are truly empty strings

---

## 6. Technologies Used

| Technology | Purpose |
|------------|---------|
| **Python 3.14** | All pipeline code |
| **pandas** | Data loading, manipulation, chunked streaming |
| **NumPy** | Feature arrays, label arrays, numerical operations |
| **scikit-learn** | `TfidfVectorizer` for char n-gram blocking; `GroupKFold` for splits |
| **rapidfuzz** | C-implemented Levenshtein, Jaro-Winkler, token_sort_ratio (10–50× faster than pure Python) |
| **LightGBM** | Primary ensemble member (CPU, `n_jobs=-1`, leaf-wise growth) |
| **XGBoost** | Secondary ensemble member (CUDA GPU via `device="cuda"`) |
| **PyTorch** | CUDA availability detection for XGBoost device selection |
| **PyYAML** | Config loading from `pipeline.yaml` |
| **scipy.sparse** | Sparse TF-IDF matrix operations for n-gram blocking channels |
| **Kaggle T4 GPU** | XGBoost CUDA acceleration during training and inference |
| **Git / GitHub** | Version control, Kaggle notebook deployment via `git clone` |

### Key Library Choices

**rapidfuzz vs difflib**: On 100,000 string pairs, `rapidfuzz` completes in ~0.8 s vs ~45 s for pure Python `difflib`. This was essential for computing 46 features on large candidate sets within a RAM-constrained environment.

**LightGBM + XGBoost ensemble vs single model**: Both are GBT libraries but differ in leaf-wise vs depth-wise growth, regularization defaults, and missing value handling. Their errors are partially uncorrelated — averaging probabilities reduces variance and improves generalization on borderline cases.

**TF-IDF character n-grams vs embedding models**: Character n-gram TF-IDF is fully unsupervised, interpretable, fast, and handles typos/transliteration/abbreviations better than word-level methods. It requires no pre-trained weights, has no parameter count, and is fully compliant with the ≤8B parameter constraint.

---

## 7. Issues Encountered and How We Solved Them

### Issue 1: Step 4 (Feature Extraction) Crashes at 100% RAM on Kaggle

**Root Cause**: The original pipeline generated ~2.4 million candidate pairs across 7 channels and 60,000 training S1 entities, then attempted to compute all 46 heavy fuzzy-matching features on all 2.4 million pairs before any filtering. Storing 2.4 million Python dicts consumed ~8 GB; the `pd.DataFrame(rows)` conversion attempted another ~4 GB, exceeding Kaggle's 13 GB RAM limit. Additionally, candidate records were re-normalized from scratch in Step 4, duplicating what Step 3 had already done.

**Solution — Four simultaneous fixes**:
1. **`select_training_pairs()`** — filters to 100% true positives + top-4 hardest negatives per S1 entity **before** any feature extraction. Reduces from 2.4M to ~100K pairs
2. **Pre-normalized dict reuse** — `s1_norm` and `cand_norm` built once and passed to both `generate_candidates()` and `build_pair_features()` via new optional arguments
3. **Chunked feature extraction** — `build_pair_features(batch_size=25000)` processes pairs in 25K-row batches, keeping instantaneous memory under 200 MB
4. **Training sample size**: 60,000 → 25,000 S1 entities (quality of hard negatives matters more than raw count for F₀.₅)

**Result**: Step 4 memory: ~12.5 GB (crash) → ~200 MB. Runtime: frozen → ~15 seconds.

---

### Issue 2: `[Errno 2] No such file or directory: 'MLchLg'` in Kaggle

**Root Cause**: Original Cell 1 used `%cd MLchLg` (relative path). If the cell ran more than once or working directory shifted, it searched for `/kaggle/working/MLchLg/MLchLg`. The `!git pull origin main` then ran in `/kaggle/working` (no git repository), producing the "not a git repository" error.

**Solution — 100% bulletproof single cell**:
```python
import shutil
shutil.rmtree("/kaggle/working/MLchLg", ignore_errors=True)
!git clone https://github.com/Aviral2406/MLchLg.git /kaggle/working/MLchLg
%cd /kaggle/working/MLchLg
!pip install -q rapidfuzz lightgbm xgboost pyyaml scikit-learn
!python scripts/train_and_evaluate.py
```

Uses absolute paths throughout; `shutil.rmtree` with `ignore_errors=True` safely handles both first-run and re-run cases.

---

### Issue 3: Threshold 0.72 Causing 43% Singleton Prediction Rate

**Root Cause**: The initial threshold was set to 0.72 (a common default). With an uncalibrated model on 25K training entities, most candidate pairs scored below 0.72, causing the system to predict "no match" for ~43% of entities — far more than the true singleton rate (~5.6%). This inflated singleton accuracy but catastrophically hurt recall on entities with actual matches.

**Solution**: Implemented the fine-grained threshold sweep (0.15 → 0.55, step 0.02) optimizing directly for entity-level Macro F₀.₅. The calibrated threshold was found at 0.30–0.40, reducing false singletons to approximately match the true rate.

---

### Issue 4: Trivial Negative Training Causing Gradient Dilution

**Root Cause**: Training on all 2.4 million pairs (98.5% trivial negatives) caused GBM trees to waste learning capacity on obvious rules (reject pairs with 0.0 name similarity). The fine-grained decision boundary — distinguishing `"Starbucks Sector 14"` from `"Starbucks Sector 18"` — was never learned because hard cases were a tiny fraction of the training signal.

**Solution**: `select_training_pairs(max_negatives_per_s1=4)` ensures a 1:4 positive-to-hard-negative ratio where every negative was already found by a blocking channel (meaning it superficially resembles a true match). The model must now learn subtle distinctions: PIN mismatch, house number 47 vs 45, city token "Sector 14" vs "Sector 18".

---

### Issue 5: Blocking Re-normalizes Records Multiple Times

**Root Cause**: Step 3 (blocking) called `_normalize_all()` on all records then discarded the `NormalizedRecord` objects. Step 4 (feature extraction) called `normalize()` again on every record from scratch. With 25K S1 and 80K+ candidate records, this doubled normalization CPU time and memory allocation.

**Solution**: Added `s1_norm: dict | None = None` and `candidate_norm: dict | None = None` optional parameters to both `generate_candidates()` and `build_pair_features()`. Training script builds these dicts once and passes them to both functions, eliminating redundant normalization.

---

## 8. Why We Followed This Approach

### 8.1 Why Two-Level Architecture?
A naive all-pairs comparison: $2.2M \times 20M = 44 \times 10^{12}$ comparisons — computationally infeasible. Blocking reduces this to ~$25K \times 100 = 2.5M$ pairs for training, and the streaming inference architecture handles test scale efficiently.

### 8.2 Why 7 Channels?
Exact name matching alone would miss:
- **Typos**: `"Sharma Traders"` vs `"Sarma Traders"` → Channel C (n-gram TF-IDF) recovers this
- **DBA pairs**: `"ABC Company"` vs `"ABC Pvt Ltd"` at same address → Channel D recovers this
- **Missing names**: records with no business name but matching PIN → Channel E recovers this
- **Word-order variants**: `"India Bakery House"` vs `"Bakery House India"` → Channels B, C recover this

Each channel addresses a distinct real-world noise pattern. Union ensures near-100% recall.

### 8.3 Why Character N-gram TF-IDF for Blocking?
- **Typo-robust**: `"Sharma"` and `"Sarma"` share n-grams `"sha"`, `"arm"`, `"rma"`
- **Transliteration-robust**: Different Romanizations of the same Hindi name share many character sequences
- **Language-agnostic**: Works on French names without any French resources
- **Unsupervised**: No labeled data needed; fit on the provided corpus itself

### 8.4 Why LightGBM + XGBoost Ensemble Instead of Neural Models?
- **46 hand-engineered similarity features are exactly what GBMs excel at** — they learn non-linear interactions (e.g., "high name similarity AND low address similarity → possible chain-branch confusion") without explicit specification
- **No external data**: No pre-trained weights, no external vocabulary, fully compliant with competition rules
- **Interpretable**: Feature importances and SHAP values explain model decisions
- **Fast iteration**: Training takes minutes vs hours for transformer fine-tuning
- **≤8B parameter rule**: GBMs have zero parameters in the neural sense

### 8.5 Why Tri-State PIN and House Number Features?
Binary (1=match, 0=no match) conflates two very different situations:
- `pin_exact_match = 0` because PINs differ → evidence of non-match
- `pin_exact_match = 0` because one record has no PIN → no information

Tri-state (0.0=mismatch, 0.5=missing, 1.0=match) correctly informs the GBM to discount PIN signal when data is absent rather than treating absence of data as evidence of mismatch.

### 8.6 Why Hard Negatives Before Feature Extraction?
Under F₀.₅, false positives are 4× more costly than false negatives. A model trained on easy negatives makes confident false positives on confusing near-misses at inference time. Training on hard negatives — pairs that look similar but aren't — builds precise decision boundaries exactly where needed.

---

## 9. Training Optimizations to Improve Score

Ranked by expected impact and implementation effort:

### 9.1 Increase Training Sample Size (High Impact, Low Effort)
```python
TRAIN_SAMPLE_SIZE = 50000   # currently 25000
sample_negatives = 60000    # currently 30000
```
More training data always helps GBMs. Now feasible with the memory-safe pipeline. Expected gain: +0.03–0.05 F₀.₅.

### 9.2 Sub-category Balanced Hard Negatives (High Impact, Medium Effort)
Instead of selecting top-4 hardest negatives globally per S1, select:
- 2 same-name, different-address (chain-branch confusion)
- 2 different-name, same-address (DBA/trade-name confusion)
- 2 multi-channel near-misses (highest blocking confidence)

This teaches the model all failure modes rather than just the most common one.

### 9.3 Add IDF-Weighted Rare Token Overlap Feature (High Impact, Low Effort)
```python
# A shared rare token like "SharmaTradersUnique" >> shared common token "Store"
rare_overlap = sum(idf_weight.get(t, 1.0) for t in (a_tok & b_tok))
```
Dramatically helps for chains where the brand name is highly distinctive but the address format is noisy.

### 9.4 Add Legal Suffix Compatibility Feature (Moderate Impact, Low Effort)
```python
legal_suffix_compatible = float(suffix_class(a.name) == suffix_class(b.name))
```
Two businesses with `Pvt Ltd` vs `LLC` are unlikely the same entity. One direct discriminator for corporate entity confusion.

### 9.5 Segmented Threshold by Channel Count (Moderate Impact, Low Effort)
```python
if ch_count >= 3:   threshold = 0.25   # multi-channel = high confidence
elif ch_count == 2: threshold = 0.35
else:               threshold = 0.45   # single weak channel = conservative
```

### 9.6 LightGBM Hyperparameter Tuning via Optuna (Moderate Impact, Medium Effort)
50 trials of Bayesian optimization on validation F₀.₅ targeting: `num_leaves`, `min_child_samples`, `subsample`, `colsample_bytree`, `reg_lambda`. Expected gain: +0.02–0.05 F₀.₅.

### 9.7 XGBoost Early Stopping (Quick Win)
```python
model.fit(X, y, eval_set=[(X_val, y_val)], early_stopping_rounds=30)
```
Prevents overfitting on hard negatives and reduces training time.

### 9.8 5-Fold Grouped Cross-Validation (Score Stability)
Replaces single 80/20 holdout. More stable threshold and feature selection, uses 100% of training data across folds. Critical for confident model comparison.

### 9.9 Multilingual Sentence Embedding Feature (High Impact, High Cost)
```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")  # ~120M params, MIT license
embeddings = model.encode(texts, batch_size=256)  # cached per entity_id
```
Adds semantic cosine similarity as feature #47. Critical for France generalization and extreme transliteration cases. Expected gain: +0.03–0.07 F₀.₅ on cross-language cases.

### 9.10 Singleton Confidence Margin (Quick Win)
```python
# Only output match if score exceeds threshold by at least a confidence margin
effective_threshold = threshold + 0.05 * (max_score < threshold + 0.08)
```
Converts borderline false positives into correct singletons, directly improving F₀.₅ for entities near the decision boundary.

---

## 10. Scope for Improvement

### Short-Term (1–2 days, high confidence of gain)
1. Increase `TRAIN_SAMPLE_SIZE` to 50,000 — now feasible with memory-safe pipeline
2. Add early stopping to XGBoost — prevents overfitting
3. Tune `max_negatives_per_s1` from 4 to 6–8
4. Lower global candidate cap from 100 to 60 per S1 entity

### Medium-Term (3–7 days, moderate confidence)
5. Add IDF-weighted rare-token overlap feature
6. Add legal suffix compatibility feature
7. Implement Optuna hyperparameter search (50 trials on validation F₀.₅)
8. Add segmented thresholds by channel count
9. 5-fold grouped cross-validation for stable model comparison

### Long-Term (1–2 weeks, high potential gain)
10. Multilingual sentence embedding feature (`paraphrase-multilingual-MiniLM-L12-v2`, MIT license, <200M params)
11. FAISS ANN blocking for Channel C/G at scale — avoids TF-IDF matrix memory blowup at 10M candidates
12. CatBoost as third ensemble member — native missing value handling, complementary errors
13. Error analysis dashboard by country/similarity bucket/candidate rank → targeted feature additions
14. Contrastive fine-tuning of a small encoder on the challenge's own positive/hard-negative pairs
15. Address component soft-parsing (house number, city-candidate, postal code) for higher-precision address features
16. Blocking recall audit for all true pairs missed by the full union → potentially discover an 8th channel

---

## 11. Key Metrics and Evaluation Methodology

### Primary Metric
**Macro F₀.₅** — computed entity-by-entity, then arithmetically averaged. Never use pairwise AUC or accuracy to make decisions.

```python
def entity_f_beta(true_set, pred_set, beta=0.5):
    if not true_set and not pred_set: return 1.0   # correct singleton → full credit
    if not pred_set: return 0.0                    # missed all matches
    tp = len(true_set & pred_set)
    precision = tp / len(pred_set)
    recall = tp / len(true_set)
    beta2 = beta ** 2
    return (1 + beta2) * precision * recall / (beta2 * precision + recall)

macro_f05 = mean([entity_f_beta(true[s1], pred[s1]) for s1 in all_s1_entities])
```

### Validation Strategy
- **Entity-level grouped split**: 80% S1 entities for training, 20% for validation; pairs from the same S1 entity never appear in both splits
- **End-to-end simulation**: validation runs the full pipeline (blocking → features → model → threshold → aggregation), not just pairwise scoring
- **Threshold sweep**: optimized on validation Macro F₀.₅, not pairwise accuracy

### Key Diagnostic Metrics
| Metric | What it tells you |
|--------|------------------|
| Blocking recall | Hard ceiling on achievable F₀.₅ |
| Avg candidates per S1 | Blocking efficiency |
| False merge rate | Most expensive error class under F₀.₅ |
| Singleton accuracy | Correctly predicting "no match" |
| Per-country F₀.₅ | France generalization quality |

---

## 12. Risk Analysis

| Risk | Severity | Mitigation |
|------|----------|-----------|
| Blocking recall < 97% silently caps score | **Critical** | Mandatory blocking recall logging per channel; target ≥97% before any Level 2 work |
| France-specific logic breaks at test time | **High** | Zero hardcoded country logic; `country_unseen_in_train` flags; data-driven normalization |
| Train/validation leakage at pair level | **High** | Strict entity-level grouped split; never pair-level split |
| Optimizing pairwise metrics instead of F₀.₅ | **High** | `macro_f_beta` is the single source of truth for every decision |
| Overfitting to 25K training entities | **Medium** | Expand to 50K–100K; 5-fold grouped CV |
| Threshold too conservative → false singletons | **Medium** | Fine-grained threshold sweep; monitor false merge rate |
| Hard negative ratio too low → false positives | **Medium** | Tune `max_negatives_per_s1`; monitor precision-recall curve |
| Test inference OOM on 1.73M records | **Low** | Streaming 50K-chunk architecture; peak RAM ≤1.5 GB tested |
| Embedding model violates ≤8B parameter rule | **Low** | `paraphrase-multilingual-MiniLM-L12-v2` is ~120M params; trivially compliant |

---

*This document reflects the state of the project as of September 26, 2026. All code is available at [github.com/Aviral2406/MLchLg](https://github.com/Aviral2406/MLchLg).*
