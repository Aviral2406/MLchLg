"""
Data ingestion: TSV loading and schema validation.
See docs/architecture.md §1 and CLAUDE.md §2 for the rules this module must enforce.
"""
from __future__ import annotations
import re
import pandas as pd

REQUIRED_COLUMNS = ["entity_id", "business_name", "business_address", "country"]
ENTITY_ID_PATTERN = re.compile(r"^(S1|S2|S3)-\d+$")


def load_tsv(path: str, expected_prefix: str | None = None) -> pd.DataFrame:
    """Load a challenge TSV file with the correct, mandatory read settings.

    expected_prefix: e.g. "S1" for a *_source1.tsv file - used for a prefix sanity check.
    Never remove dtype=str or keep_default_na=False (see CLAUDE.md §2): entity IDs,
    PIN codes, and house numbers must never be silently cast to numeric.
    """
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    if expected_prefix is not None:
        issues = validate_entity_id_prefix(df, expected_prefix)
        for issue in issues:
            print(f"[ingest warning] {path}: {issue}")
    return df


def validate_schema(df: pd.DataFrame, required_columns: list[str] = REQUIRED_COLUMNS) -> list[str]:
    """Returns a list of human-readable issues; empty list == schema OK."""
    issues = []
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        issues.append(f"Missing required columns: {missing}")
    if "entity_id" in df.columns and df["entity_id"].duplicated().any():
        n = int(df["entity_id"].duplicated().sum())
        issues.append(f"{n} duplicate entity_id values found")
    if "entity_id" in df.columns:
        bad_mask = ~df["entity_id"].str.match(ENTITY_ID_PATTERN)
        bad_ids = df.loc[bad_mask, "entity_id"].tolist()
        if bad_ids:
            issues.append(f"{len(bad_ids)} entity_id values do not match ^(S1|S2|S3)-\\d+$: {bad_ids[:5]}")
    if "business_name" in df.columns:
        empty_names = int((df["business_name"].str.strip() == "").sum())
        if empty_names:
            issues.append(f"{empty_names} rows have an empty business_name")
    return issues


def validate_entity_id_prefix(df: pd.DataFrame, expected_prefix: str) -> list[str]:
    """Confirms every entity_id in this file carries the expected source prefix."""
    issues = []
    bad = df.loc[~df["entity_id"].str.startswith(f"{expected_prefix}-"), "entity_id"].tolist()
    if bad:
        issues.append(f"{len(bad)} entity_id values do not start with {expected_prefix}-: {bad[:5]}")
    return issues


def country_value_counts(df: pd.DataFrame) -> pd.Series:
    """Descriptive only - NEVER use this output to build a fixed country vocabulary.
    Country is an open set; France appears only in test (CLAUDE.md §2)."""
    return df["country"].value_counts(dropna=False)
