# 🏢 Business Entity Resolution — Amazon ML Challenge

> **Match noisy business records across three sources using a two-level ML pipeline.**
> Scored by **F₀.₅** (macro-averaged per Source 1 entity, precision-weighted 2×).

---

## 📖 Read These First (in order)

| Doc | What it tells you |
|---|---|
| [`CLAUDE.md`](CLAUDE.md) | ⚠️ Hard constraints — rules no code may violate. Read before touching code. |
| [`docs/task_specs.md`](task_specs.md) | Per-module task assignments with acceptance criteria |
| [`docs/architecture.md`](business_entity_resolution_architecture.md) | Full system design (blocking → features → GBM → threshold → submission) |
| [`CONTRIBUTING.md`](CONTRIBUTING.md) | Branching strategy, PR checklist, how to run things |

---

## 🗺️ Team Ownership

| Member | Owns | Primary modules |
|---|---|---|
| 1 | EDA + Normalization | `notebooks/01,02`, `src/.../data`, `src/.../normalization` |
| 2 | Blocking | `notebooks/03`, `src/.../blocking`, `candidate_pairs.tsv` |
| 3 | Pairwise Features + Classical ML | `notebooks/04,05`, `src/.../features`, `src/.../models` |
| 4 | Advanced Matching + Tuning | Encoder/neural model, threshold optimization |
| 5 | Evaluation + Ensemble + Submission | `src/.../validation`, `experiments/`, final output packaging |

---

## 🚀 Quick Start

```bash
# 1. Clone the repo
git clone <repo-url>
cd business-entity-resolution

# 2. Install dependencies
pip install -r requirements.txt

# 3. Place the dataset (shared via Google Drive — do NOT commit it)
#    data/train/train_source{1,2,3}.tsv
#    data/train/train_ground_truth.tsv
#    data/test/test_source{1,2,3}.tsv

# 4. Fast sanity check (uses fixtures, no full dataset needed)
python -c "
import sys; sys.path.insert(0, 'src')
from business_entity_resolution.data.ingest import load_tsv, validate_schema
s1 = load_tsv('fixtures/train/sample_source1.tsv', expected_prefix='S1')
print(validate_schema(s1))
"

# 5. Run notebooks in order: 01_eda → 02_normalization → 03_blocking → 04_features → 05_modeling

# 6. Validate submission output before any leaderboard upload
python utils/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir data/test
```

---

## 🏗️ System Architecture

```
Raw TSVs (S1, S2, S3)
        │
        ▼
   [EDA + Normalization]
        │
        ▼
┌──────────────────────────────┐
│  LEVEL 1: BLOCKING           │  ← optimize RECALL (≥97%)
│  7 channels (A–G), unioned   │
└──────────────┬───────────────┘
               │  candidate_pairs.tsv
               ▼
┌──────────────────────────────┐
│  LEVEL 2: MATCHING           │  ← optimize PRECISION (F₀.₅ weights it 2×)
│  LightGBM on pair features   │
└──────────────┬───────────────┘
               │
        [Threshold sweep]
               │
        matching_results.tsv
```

---

## 📂 Project Structure

```
business-entity-resolution/
├── data/               # ← NOT in git (too large; share via Drive)
│   ├── train/
│   └── test/
├── fixtures/           # Small sample data for fast testing ✅ in git
├── notebooks/          # 01_eda → 02_norm → 03_blocking → 04_features → 05_modeling → 06_error_analysis
├── src/
│   └── business_entity_resolution/
│       ├── data/           # ingestion, schema validation
│       ├── normalization/  # multi-representation normalizers
│       ├── blocking/       # channels A–G, recall diagnostics
│       ├── features/       # pairwise feature generators
│       ├── models/         # baseline, GBM, optional neural
│       ├── validation/     # grouped split, local pre-checks
│       ├── inference/      # end-to-end predict pipeline
│       └── evaluation/     # metrics, experiment logging
├── configs/
│   ├── normalization/  # suffixes.json, abbreviations.json (mined from data)
│   └── pipeline.yaml   # channel toggles, thresholds, feature flags
├── experiments/
│   └── experiment_log.csv   # every experiment gets a row here
├── output/             # matching_results.tsv, candidate_pairs.tsv (generated, not committed)
├── utils/
│   └── validate_submission.py
├── CLAUDE.md           # ⚠️ Hard rules — read first
├── CONTRIBUTING.md
└── README.md
```

---

## ⚠️ Critical Rules (from `CLAUDE.md`)

1. **No external data** — every dictionary/list must be mined from the provided TSV files only.
2. **Country is an open set** — train has US + India; test adds **France**. Never hardcode `if country in ["US", "India"]`.
3. **Always use `macro_f_beta`** from `evaluation/metrics.py` as your metric — never plain F1 or AUC.
4. **TSV loading:** always `pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)`.
5. **Run** `utils/validate_submission.py` before every leaderboard upload — no exceptions.

---

## 📊 Experiment Tracking

Every experiment (blocking, feature, model, threshold) gets a row in [`experiments/experiment_log.csv`](experiments/experiment_log.csv).  
This is required before any PR into `main`. See `CONTRIBUTING.md` for the schema.

---

## 🔗 Dataset

The full dataset (~1 GB) is shared via **[Google Drive / OneDrive — add link here]**.  
Do **not** commit it to this repo.
