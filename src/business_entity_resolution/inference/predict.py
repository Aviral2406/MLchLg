"""
End-to-end inference: blocking -> features -> model -> threshold -> aggregation ->
output files. See docs/architecture.md §17 for the full diagram this implements.

This wires the other modules together; it should contain orchestration logic only -
put any new similarity/feature/blocking logic in its own module, not here.
"""
from __future__ import annotations
import pandas as pd

from business_entity_resolution.blocking.block import generate_candidates
from business_entity_resolution.features.pair_features import build_pair_features
from business_entity_resolution.models.model import build_model


def predict(s1_df: pd.DataFrame, s2_df: pd.DataFrame, s3_df: pd.DataFrame,
            model, config: dict, known_train_countries: set = frozenset()) -> tuple:
    """Returns (matching_results_df, candidate_pairs_df), both in the exact output
    schema required by the problem statement (source1_entity_id + comma-joined IDs).
    `model` must already be fit (see models.model.build_model / .fit)."""

    candidate_pairs_long = generate_candidates(s1_df, s2_df, s3_df, config)

    feature_df = build_pair_features(candidate_pairs_long, s1_df, s2_df, s3_df,
                                      known_train_countries=known_train_countries)

    if len(feature_df) > 0:
        feature_cols = [c for c in feature_df.columns
                         if c not in ("source1_entity_id", "candidate_entity_id")]
        feature_df["match_proba"] = model.predict_proba(feature_df[feature_cols])
    else:
        feature_df["match_proba"] = []

    threshold = config.get("threshold", {}).get("value", 0.5)
    scored = feature_df.merge(
        candidate_pairs_long[["source1_entity_id", "candidate_entity_id"]].dropna(),
        on=["source1_entity_id", "candidate_entity_id"], how="right",
    )

    matches = (
        scored[scored["match_proba"] >= threshold]
        .groupby("source1_entity_id")["candidate_entity_id"]
        .apply(lambda ids: ",".join(sorted(set(ids))))
        .reset_index()
        .rename(columns={"candidate_entity_id": "matched_entity_ids"})
    )

    matching_results = s1_df[["entity_id"]].rename(columns={"entity_id": "source1_entity_id"})
    matching_results = matching_results.merge(matches, on="source1_entity_id", how="left")
    matching_results["matched_entity_ids"] = matching_results["matched_entity_ids"].fillna("")

    candidate_pairs = (
        candidate_pairs_long.dropna(subset=["candidate_entity_id"])
        .groupby("source1_entity_id")["candidate_entity_id"]
        .apply(lambda ids: ",".join(sorted(set(ids))))
        .reset_index()
        .rename(columns={"candidate_entity_id": "candidate_entity_ids"})
    )
    candidate_pairs = s1_df[["entity_id"]].rename(columns={"entity_id": "source1_entity_id"}) \
        .merge(candidate_pairs, on="source1_entity_id", how="left")
    candidate_pairs["candidate_entity_ids"] = candidate_pairs["candidate_entity_ids"].fillna("")

    return matching_results, candidate_pairs


def write_outputs(matching_results: pd.DataFrame, candidate_pairs: pd.DataFrame,
                   output_dir: str = "output") -> None:
    matching_results.to_csv(f"{output_dir}/matching_results.tsv", sep="\t", index=False)
    candidate_pairs.to_csv(f"{output_dir}/candidate_pairs.tsv", sep="\t", index=False)
