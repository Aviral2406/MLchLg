# Contributing

Team workflow for this repo — 3–5 people (or people + AI agents) working in parallel
across the modules defined in `docs/architecture.md` §20 and `docs/task_specs.md`.

## Before you start any task

1. Read `CLAUDE.md` (constraints no code may violate) and the relevant section of
   `docs/task_specs.md` for your task.
2. Check `experiments/experiment_log.csv` for related prior work — don't re-run an
   experiment someone already logged.

## Branching

- `main` is protected. Branch per module + experiment:
  `blocking/channel-e-pin`, `features/address-components`, `model/lgbm-v3`,
  `eval/threshold-sweep-v2`.
- Keep branches scoped to one module where possible — cross-module branches (e.g. a
  blocking change that also touches features) are fine when the change genuinely
  requires it, but call this out explicitly in the PR description so reviewers know to
  re-check downstream modules.

## Commits

- Reference the task-spec section in the commit message where useful, e.g.
  `blocking: implement Channel C (name char n-gram retrieval) — task_specs.md §3`.

## Pull requests — required checklist

Every PR into `main` must include:

- [ ] Updated `experiments/experiment_log.csv` row(s) for any change affecting
      blocking, features, model, or threshold behavior.
- [ ] A passing run of `src/business_entity_resolution/validation/local_checks.py`
      against `fixtures/` (fast) — and against real validation output if the change
      touches inference.
- [ ] If blocking changed: an updated recall table (architecture.md §5) in the PR
      description or a linked notebook cell output.
- [ ] If model/threshold changed: the validation entity-level `macro_f_beta` before
      and after, from `evaluation/metrics.py` — never a pairwise-only metric as the
      justification.
- [ ] No new hardcoded country list, suffix/abbreviation dictionary entry sourced from
      outside the provided data, or external API/lookup call (CLAUDE.md §2).

## Code style

- Notebooks stay thin — call into `src/business_entity_resolution/`, don't inline
  logic. This is what keeps notebook diffs reviewable and avoids merge conflicts on
  notebook JSON.
- New modules follow the existing contract pattern (see `docs/architecture.md` §4 /
  `CLAUDE.md` §4) — e.g. a new blocking channel is a function
  `channel_x_name(s1_norm, candidate_norm) -> dict[str, set[str]]` registered in
  `blocking.block.CHANNELS`, not a one-off script.
- Type hints on public function signatures; docstrings explaining *why* a
  transformation/feature exists, not just what it does (see existing modules for the
  expected level of detail).

## Reproducing someone else's experiment

Every `experiments/experiment_log.csv` row should have enough in its
`hyperparams`/`cv_strategy`/`git_commit` columns that another person can:

```bash
git checkout <commit_hash>
# re-run the relevant notebook or script with the logged config
```

If a row can't be reproduced this way, treat that as a bug in the logging, not just a
gap — fix the logging going forward.

## Running things

```bash
pip install -r requirements.txt --break-system-packages   # or a venv, either is fine

# fast sanity check against the small fixture set (no full dataset needed)
python3 -c "
import sys; sys.path.insert(0, 'src')
from business_entity_resolution.data.ingest import load_tsv, validate_schema
s1 = load_tsv('fixtures/train/sample_source1.tsv', expected_prefix='S1')
print(validate_schema(s1))
"

# once the real dataset is in data/train and data/test:
# run notebooks/01_eda.ipynb -> ... -> 05_modeling.ipynb in order, then:
python3 utils/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir data/test
```

## Weekly sync

Blocking and matching are decoupled (docs/architecture.md §0), so both can usually
proceed in parallel — but a blocking-channel change shifts the negative-sampling pool
(architecture.md §7), so re-run model training after any blocking change lands on
`main`. Use the weekly sync to catch this rather than discovering it from a confusing
metric regression later.
