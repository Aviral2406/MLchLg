"""
Train/validation split at the SOURCE-1 ENTITY level (never at the pair level).
See docs/architecture.md §8 and CLAUDE.md §6 - the most common way an agent
accidentally inflates validation numbers is by leaking pairs from the same
source1_entity_id across train and validation.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def grouped_entity_split(s1_df: pd.DataFrame, ground_truth_df: pd.DataFrame,
                          val_fraction: float = 0.2, random_state: int = 42) -> tuple:
    """Returns (train_entity_ids, val_entity_ids) - disjoint sets of source1_entity_id.

    Stratifies by singleton-vs-has-matches and, where present, country, so both splits
    preserve the true singleton rate and country mix (architecture.md §8 step 2).
    """
    rng = np.random.RandomState(random_state)

    gt = ground_truth_df.set_index("source1_entity_id")["matched_entity_ids"].to_dict()
    s1_ids = s1_df["entity_id"].tolist()
    is_singleton = {sid: (not gt.get(sid, "") or str(gt.get(sid, "")).strip() == "") for sid in s1_ids}

    country_map = {}
    if "country" in s1_df.columns:
        country_map = s1_df.set_index("entity_id")["country"].to_dict()

    strata = {}
    for sid in s1_ids:
        key = (is_singleton[sid], country_map.get(sid, "unknown"))
        strata.setdefault(key, []).append(sid)

    train_ids, val_ids = [], []
    for key, ids in strata.items():
        ids = list(ids)
        rng.shuffle(ids)
        n_val = max(1, int(round(len(ids) * val_fraction))) if len(ids) > 1 else 0
        val_ids.extend(ids[:n_val])
        train_ids.extend(ids[n_val:])

    return set(train_ids), set(val_ids)


def grouped_kfold_entity_splits(s1_df: pd.DataFrame, ground_truth_df: pd.DataFrame,
                                 n_folds: int = 5, random_state: int = 42) -> list:
    """Returns a list of n_folds (train_entity_ids, val_entity_ids) tuples for grouped
    cross-validation, preferable to a single split when time allows (architecture.md §8)."""
    s1_ids = np.array(s1_df["entity_id"].tolist())
    rng = np.random.RandomState(random_state)
    shuffled = s1_ids.copy()
    rng.shuffle(shuffled)
    folds = np.array_split(shuffled, n_folds)

    splits = []
    for i in range(n_folds):
        val_ids = set(folds[i].tolist())
        train_ids = set(s1_ids.tolist()) - val_ids
        splits.append((train_ids, val_ids))
    return splits
