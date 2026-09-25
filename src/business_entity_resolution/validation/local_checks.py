"""
Local pre-submission checks. This is a DEVELOPER CONVENIENCE, not a replacement for
the organizer-provided utils/validate_submission.py - always run the official script
before uploading (CLAUDE.md §3). This module lets you catch obvious issues immediately
after generating output, without a subprocess round-trip, while iterating.
"""
from __future__ import annotations
import pandas as pd


def check_matching_results(matching_df: pd.DataFrame, test_s1_ids: set,
                            valid_candidate_ids: set) -> list:
    """matching_df: [source1_entity_id, matched_entity_ids]. Returns a list of issues;
    empty list == locally OK (still run the official validator before submitting)."""
    issues = []

    dup_s1 = matching_df["source1_entity_id"].duplicated()
    if dup_s1.any():
        issues.append(f"{int(dup_s1.sum())} duplicate source1_entity_id rows")

    present_s1 = set(matching_df["source1_entity_id"])
    missing_s1 = test_s1_ids - present_s1
    if missing_s1:
        issues.append(f"{len(missing_s1)} test S1 entities missing from matching_results (e.g. {list(missing_s1)[:5]})")
    extra_s1 = present_s1 - test_s1_ids
    if extra_s1:
        issues.append(f"{len(extra_s1)} rows reference S1 entities not in the test set (e.g. {list(extra_s1)[:5]})")

    for _, row in matching_df.iterrows():
        cell = row.get("matched_entity_ids", "")
        if not cell or (isinstance(cell, float)):
            continue
        ids = [x.strip() for x in str(cell).split(",") if x.strip()]
        if len(ids) != len(set(ids)):
            issues.append(f"{row['source1_entity_id']}: duplicate IDs within its match list")
        bad_prefix = [i for i in ids if not (i.startswith("S2-") or i.startswith("S3-"))]
        if bad_prefix:
            issues.append(f"{row['source1_entity_id']}: non S2-/S3- IDs in match list: {bad_prefix}")
        unknown = [i for i in ids if i not in valid_candidate_ids]
        if unknown:
            issues.append(f"{row['source1_entity_id']}: matched IDs not present in test data: {unknown}")

    return issues


def check_matches_are_subset_of_candidates(matching_df: pd.DataFrame, candidate_df: pd.DataFrame) -> list:
    """Every ID in matching_results.tsv must also appear in that S1 entity's row in
    candidate_pairs.tsv - a mismatch signals a pipeline bug (problem statement, Output
    Format section)."""
    issues = []
    cand_by_s1 = candidate_df.groupby("source1_entity_id")["candidate_entity_ids"].first().to_dict() \
        if "candidate_entity_ids" in candidate_df.columns else {}

    for _, row in matching_df.iterrows():
        sid = row["source1_entity_id"]
        cell = row.get("matched_entity_ids", "")
        if not cell or isinstance(cell, float):
            continue
        matched_ids = set(x.strip() for x in str(cell).split(",") if x.strip())
        cand_cell = cand_by_s1.get(sid, "")
        cand_ids = set(x.strip() for x in str(cand_cell).split(",") if x.strip()) if cand_cell else set()
        orphaned = matched_ids - cand_ids
        if orphaned:
            issues.append(f"{sid}: matched IDs never appeared as candidates: {orphaned}")

    return issues
