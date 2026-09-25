# Data Dictionary & Empirical EDA Report

Generated automatically from dataset inspection on `data/train/` and `data/test/`.

---

## 1. File Schema & Record Counts

| Dataset File | Rows | Schema Status | Unique Entity IDs | Prefix Sanity |
| :--- | :--- | :--- | :--- | :--- |
| `train_source1.tsv` | 2,206,821 | Valid | 2,206,821 | S1- |
| `train_source2.tsv` | 5,034,616 | Valid | 5,034,616 | S2- |
| `train_source3.tsv` | 5,285,603 | Valid | 5,285,603 | S3- |
| `train_ground_truth.tsv` | 2,206,821 | Valid | N/A | S1 / Match |
| `test_source1.tsv` | 1,732,544 | Valid | 1,732,544 | S1- |
| `test_source2.tsv` | 4,887,273 | Valid | 4,887,273 | S2- |
| `test_source3.tsv` | 5,082,316 | Valid | 5,082,316 | S3- |

---

## 2. Ground Truth & Matching Statistics

- **Total S1 Entities in Train**: 2,206,821
- **Matched S1 Entities**: 2,206,821 (100.00%)
- **Singleton S1 Entities (No matches)**: 0 (0.00%)
- **Total Ground Truth Match Pairs**: 2,206,821
- **S1 <-> S2 Matches**: 1,919,076
- **S1 <-> S3 Matches**: 164,498
- **Average Matches per Matched S1 Entity**: 1.00

---

## 3. Country Distribution (Open-Set Verification)

- **Train Unique Countries**: 2
- **Test Unique Countries**: 3
- **France in Train**: `False`
- **France in Test**: `True`

> **Rule Confirmation**: Country is confirmed to be an open-set field. Never hardcode country lists in model logic or normalization pipelines.

---

## 4. Normalization Mined Dictionaries Summary

- `configs/normalization/suffixes.json`: Mined **30** candidate legal suffix mappings (e.g., `pvt ltd` -> `private limited`).
- `configs/normalization/abbreviations.json`: Mined **35** candidate address abbreviation mappings (e.g., `rd` -> `road`, `hno` -> `house number`).

---
