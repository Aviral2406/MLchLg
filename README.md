# Business Entity Resolution Challenge

Two-level entity-resolution pipeline: a multi-channel blocking stage optimized for
recall, feeding a pairwise matching model optimized for precision, scored by
macro-averaged, per-entity F₀.₅. See `docs/architecture.md` for the full design.

## Start here

1. **`CLAUDE.md`** — hard constraints and conventions any contributor (human or AI
   agent) must follow. Read this first.
2. **`docs/architecture.md`** — the full technical design (25 sections: blocking,
   features, modeling, evaluation, team/repo structure, Mermaid diagrams).
3. **`docs/task_specs.md`** — per-module task tickets, pick one up directly.
4. **`docs/data_dictionary_template.md`** — fill this in as the first EDA deliverable
   once real data is loaded.
5. **`CONTRIBUTING.md`** — git workflow, PR checklist, how to run things.

## Repo layout

```
data/            real train/test TSVs go here (gitignored — never commit real data)
fixtures/        small hand-crafted sample TSVs for fast, network-free dev/testing
notebooks/       thin orchestration notebooks (01_eda -> 06_error_analysis)
src/business_entity_resolution/
  data/          TSV ingestion + schema validation
  normalization/ multi-representation field normalization
  blocking/      Level 1 — multi-channel candidate generation (optimize recall)
  features/      pairwise feature engineering
  models/        Level 2 — matching models (baseline + GBM), common fit/predict_proba
  validation/    grouped entity-level split, local pre-submission checks
  inference/     end-to-end predict pipeline, output writing
  ensemble/      stacking utilities (only if adopted — see architecture.md §16)
  evaluation/    the single source-of-truth F0.5 metric implementation
configs/         pipeline.yaml (channel/model/threshold toggles), mined normalization dicts
experiments/     experiment_log.csv — every experiment gets a row
models/          serialized trained models (gitignored)
output/          matching_results.tsv + candidate_pairs.tsv (regenerated, gitignored)
utils/           drop the organizer-provided validate_submission.py here
docs/            architecture.md, task_specs.md, data_dictionary_template.md
```

## Setup

```bash
pip install -r requirements.txt --break-system-packages   # or use a virtualenv
```

## Quick smoke test (no real data needed)

The skeleton has already been validated end-to-end against `fixtures/`:

```bash
python3 -c "
import sys; sys.path.insert(0, 'src')
from business_entity_resolution.data.ingest import load_tsv
from business_entity_resolution.blocking.block import generate_candidates, evaluate_blocking_recall
import yaml
s1 = load_tsv('fixtures/train/sample_source1.tsv', expected_prefix='S1')
s2 = load_tsv('fixtures/train/sample_source2.tsv', expected_prefix='S2')
s3 = load_tsv('fixtures/train/sample_source3.tsv', expected_prefix='S3')
gt = load_tsv('fixtures/train/sample_ground_truth.tsv')
config = yaml.safe_load(open('configs/pipeline.yaml'))
cand = generate_candidates(s1, s2, s3, config)
print(evaluate_blocking_recall(cand, gt))
"
```

With only Channel A (exact-name blocking) implemented so far, recall on the fixture is
low by design — that's the starting point `docs/task_specs.md` §3 (Blocking) picks up
from.

## Running the real pipeline

1. Drop the real challenge data into `data/train/` and `data/test/`.
2. Run `notebooks/01_eda.ipynb` → fill in `docs/data_dictionary_template.md` and
   `configs/normalization/{suffixes,abbreviations}.json`.
3. Work through `docs/task_specs.md` tasks 3–6 (blocking channels, features, model,
   threshold), logging each experiment in `experiments/experiment_log.csv`.
4. Generate `output/matching_results.tsv` and `output/candidate_pairs.tsv` via
   `src/business_entity_resolution/inference/predict.py`.
5. Validate:
   ```bash
   python3 utils/validate_submission.py \
     --matching output/matching_results.tsv \
     --candidate output/candidate_pairs.tsv \
     --test-dir data/test
   ```
6. Upload `output/matching_results.tsv` to the leaderboard; package the full submission
   zip per the problem statement's Final Submission Package structure, including
   `docs/architecture.md`-informed methodology notes in `Documentation_template.md`.

---

## 📦 Getting the Dataset

The full dataset (~1 GB) is **NOT in this repo** (gitignored — too large for GitHub).

### Step 1 — Download the zip
Get it from **one of these sources**:
- **Amazon ML Challenge portal** — log in and download the student resource zip directly
- **Google Drive (team shared link)** — ask the team lead for the link

### Step 2 — Extract into the right folders

**Windows (PowerShell):**
```powershell
Expand-Archive -Path "student_resource.zip" -DestinationPath "." -Force
```

**Mac/Linux:**
```bash
unzip student_resource.zip
```

### Step 3 — Place TSVs like this:
```
data/
├── train/
│   ├── train_source1.tsv
│   ├── train_source2.tsv
│   ├── train_source3.tsv
│   └── train_ground_truth.tsv
└── test/
    ├── test_source1.tsv
    ├── test_source2.tsv
    └── test_source3.tsv
```

> ⚠️ **Never `git add` files inside `data/train/` or `data/test/`** — they are in `.gitignore` and must never be committed to this repo.
