# Data Dictionary — Business Entity Resolution Challenge

> **Template.** Fill this in as the first real output of EDA (`notebooks/01_eda.ipynb`,
> task spec §1 in `docs/task_specs.md`), using the actual train/test files — not from
> memory or assumption. Rename to `docs/data_dictionary.md` once filled in, and keep
> `data_dictionary_template.md` as the reusable blank template.
>
> Every number below should come from code, not estimation — this file is meant to be
> regenerable by re-running the EDA notebook.

---

## 1. File Inventory

| File | Rows | Columns | Unique `entity_id` | Notes |
|---|---|---|---|---|
| `train_source1.tsv` | | | | |
| `train_source2.tsv` | | | | |
| `train_source3.tsv` | | | | |
| `train_ground_truth.tsv` | | | | |
| `test_source1.tsv` | | | | |
| `test_source2.tsv` | | | | |
| `test_source3.tsv` | | | | |

---

## 2. Column-Level Stats (per file)

Repeat this table for each of the 6 source files.

### `<file_name>.tsv`

| Column | dtype | Null/empty rate | Distinct values | Example values |
|---|---|---|---|---|
| entity_id | str | | | |
| business_name | str | | | |
| business_address | str | | | |
| country | str | | | |

---

## 3. Business Name Profile

- Length (chars) distribution: min / p25 / median / p75 / max — per source, per country.
- Token count distribution: same breakdown.
- **Legal-suffix frequency table** (mined, not assumed) — feeds `configs/normalization/suffixes.json`:

  | Observed form | Frequency | Proposed canonical form |
  |---|---|---|
  | | | |

- **Abbreviation candidates in names** (e.g. punctuation/ampersand variants):

  | Observed form | Frequency | Proposed canonical form |
  |---|---|---|
  | | | |

- Transliteration variant clusters observed (list a few concrete examples found via
  manual sampling of near-duplicate names):
- Numeric-token rate (store/unit numbers) in names: ___%
- Case pattern distribution (ALL CAPS / Title Case / mixed): ___

## 4. Business Address Profile

- Length / token count distribution — per source, per country.
- PIN/postal code presence rate — per country:

  | Country | PIN/postal presence rate |
  |---|---|
  | | |

- **Address abbreviation frequency table** — feeds `configs/normalization/abbreviations.json`:

  | Observed form | Frequency | Proposed canonical form |
  |---|---|---|
  | | | |

- Landmark-phrase frequency (`near `, `opp `, `behind `, etc.) and rate of addresses
  containing one: ___%
- Missing-component rate (no digits at all in the address): ___%
- City/state candidate tokens (top N by frequency, used for Channel F blocking):

  | Candidate token | Frequency | Likely city or state? |
  |---|---|---|

## 5. Country Profile

| Country value (as it appears in data) | Train frequency | Test frequency | Canonical form |
|---|---|---|---|
| | | | |

- Distinct country spellings collapsing to the same canonical value (if any):
- Confirmed: no part of the codebase filters or one-hot-encodes on a fixed
  {US, India} vocabulary — checked in `src/business_entity_resolution/features/pair_features.py`
  and `normalization/normalize.py`. (Check this box once verified: ☐)

## 6. Ground Truth Profile

- Total S1 entities: ___
- Singleton rate (no match): ___%
- One-to-one match rate: ___%
- One-to-many match rate: ___% (distribution of match count: )
- Matches only in S2: ___% / only in S3: ___% / in both: ___%
- Country-conditioned singleton rate:

  | Country | Singleton rate |
  |---|---|

- **Manual sample findings** (30–50 hand-reviewed positive pairs, architecture.md §2):
  categorize the dominant matching signal per pair and summarize the distribution here
  (e.g., "62% exact-name-different-address-format, 20% abbreviation-only-difference,
  10% address-only-match-different-name, 8% landmark-based-address-match").

---

## 7. EDA-Derived Recommendations

Fill in after completing the sections above — this is the bridge from raw stats to
pipeline decisions (architecture.md §2's "recommendations" step):

1.
2.
3.
