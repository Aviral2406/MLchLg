"""
Multi-channel blocking / candidate generation. See docs/architecture.md §4-§5.

LEVEL 1 of the two-level system: optimize RECALL. Never let this module become the
place recall quietly gets lost - always measure with evaluate_blocking_recall after
any change (definition of done: cumulative recall >= 0.97 on the validation split,
see CLAUDE.md §5).

Channel A (exact/near-exact normalized name) is implemented below as a worked example.
Channels B-G are stubs - see docs/task_specs.md "Blocking + Candidate Generation" for
the full spec of each. Do not implement any channel using an external gazetteer,
geocoding service, or lookup - see CLAUDE.md §2.
"""
from __future__ import annotations
from collections import defaultdict
import pandas as pd

from business_entity_resolution.normalization.normalize import normalize


def _normalize_all(df: pd.DataFrame) -> dict:
    """entity_id -> NormalizedRecord, for a source dataframe."""
    return {row["entity_id"]: normalize(row.to_dict()) for _, row in df.iterrows()}


def channel_a_exact_name(s1_norm: dict, candidate_norm: dict) -> dict:
    """S1 entity_id -> set of candidate entity_ids sharing the same normalized name.

    Catches: clean duplicates, pure suffix/punctuation differences.
    Misses: typos, transliteration, heavy word-order changes (covered by other channels).
    """
    index = defaultdict(set)
    for cid, rec in candidate_norm.items():
        index[rec.name.normalized].add(cid)

    result = defaultdict(set)
    for sid, rec in s1_norm.items():
        result[sid] |= index.get(rec.name.normalized, set())
    return result


def channel_b_name_token_overlap(s1_norm: dict, candidate_norm: dict, max_token_df: int = 200) -> dict:
    """TODO: inverted index on name tokens, excluding globally very-high-DF tokens.
    See docs/architecture.md §4 Channel B."""
    raise NotImplementedError("See docs/task_specs.md - Blocking module, Channel B")


def channel_c_name_char_ngram(s1_norm: dict, candidate_norm: dict, top_k: int = 50) -> dict:
    """TODO: TF-IDF char n-gram retrieval on name (typo/transliteration safety net).
    See docs/architecture.md §4 Channel C. Use scipy sparse + top-k cosine, or an ANN
    index (hnswlib) once corpus size makes brute-force retrieval slow (see §22)."""
    raise NotImplementedError("See docs/task_specs.md - Blocking module, Channel C")


def channel_d_address_token_overlap(s1_norm: dict, candidate_norm: dict, max_token_df: int = 200) -> dict:
    """TODO: same pattern as Channel B, applied to address tokens."""
    raise NotImplementedError("See docs/task_specs.md - Blocking module, Channel D")


def channel_e_pin_exact(s1_norm: dict, candidate_norm: dict) -> dict:
    """TODO: exact match on extracted PIN/postal digit sequence. High precision,
    partial coverage only (many addresses have no PIN - see data dictionary)."""
    raise NotImplementedError("See docs/task_specs.md - Blocking module, Channel E")


def channel_f_city_state_cooccurrence(s1_norm: dict, candidate_norm: dict) -> dict:
    """TODO: block on shared city+state candidate tokens (mined via frequency, not an
    external gazetteer - see docs/architecture.md §3 address_components)."""
    raise NotImplementedError("See docs/task_specs.md - Blocking module, Channel F")


def channel_g_address_char_ngram(s1_norm: dict, candidate_norm: dict, top_k: int = 50) -> dict:
    """TODO: same pattern as Channel C, applied to address char n-grams."""
    raise NotImplementedError("See docs/task_specs.md - Blocking module, Channel G")


CHANNELS = {
    "exact_name": channel_a_exact_name,
    # "name_token_overlap": channel_b_name_token_overlap,
    # "name_char_ngram": channel_c_name_char_ngram,
    # "address_token_overlap": channel_d_address_token_overlap,
    # "pin_exact": channel_e_pin_exact,
    # "city_state_cooccurrence": channel_f_city_state_cooccurrence,
    # "address_char_ngram": channel_g_address_char_ngram,
    # uncomment each as it's implemented (configs/pipeline.yaml toggles which run)
}


def generate_candidates(s1_df: pd.DataFrame, s2_df: pd.DataFrame, s3_df: pd.DataFrame,
                         config: dict) -> pd.DataFrame:
    """Returns candidate_pairs_df: [source1_entity_id, candidate_entity_id, channels].

    `channels` records which channel(s) surfaced each pair - a legitimate, strongly
    discriminative feature downstream (docs/architecture.md §6, "blocking provenance").
    This function's output IS candidate_pairs.tsv (after formatting) - see CLAUDE.md §2:
    it must be the exact final candidate set, not a wider pre-filter cut later.
    """
    s1_norm = _normalize_all(s1_df)
    candidate_df = pd.concat([s2_df, s3_df], ignore_index=True)
    candidate_norm = _normalize_all(candidate_df)

    enabled = config.get("blocking", {}).get("channels", {})
    pair_channels = defaultdict(set)  # (s1_id, cand_id) -> set of channel names

    for name, fn in CHANNELS.items():
        if not enabled.get(name, False):
            continue
        result = fn(s1_norm, candidate_norm)
        for sid, cand_ids in result.items():
            for cid in cand_ids:
                pair_channels[(sid, cid)].add(name)

    rows = [
        {"source1_entity_id": sid, "candidate_entity_id": cid, "channels": ",".join(sorted(chs))}
        for (sid, cid), chs in pair_channels.items()
    ]
    out = pd.DataFrame(rows, columns=["source1_entity_id", "candidate_entity_id", "channels"])

    cap = config.get("blocking", {}).get("candidate_cap_per_s1")
    if cap:
        out = out.groupby("source1_entity_id", group_keys=False).head(cap)

    # Ensure every S1 entity has at least a (possibly empty) presence for downstream steps
    missing = set(s1_df["entity_id"]) - set(out["source1_entity_id"])
    if missing:
        out = pd.concat([out, pd.DataFrame({"source1_entity_id": list(missing),
                                             "candidate_entity_id": None, "channels": None})],
                         ignore_index=True)
    return out


def evaluate_blocking_recall(candidate_pairs_df: pd.DataFrame, ground_truth_df: pd.DataFrame) -> dict:
    """Per-channel and cumulative recall stats. See docs/architecture.md §5.

    ground_truth_df: [source1_entity_id, matched_entity_ids] (comma-separated string,
    as in train_ground_truth.tsv).
    """
    true_pairs = set()
    for _, row in ground_truth_df.iterrows():
        matched = row["matched_entity_ids"]
        if not matched:
            continue
        for mid in matched.split(","):
            true_pairs.add((row["source1_entity_id"], mid))

    candidate_pairs = set(
        zip(candidate_pairs_df["source1_entity_id"], candidate_pairs_df["candidate_entity_id"])
    )
    recovered = true_pairs & candidate_pairs
    recall = len(recovered) / len(true_pairs) if true_pairs else 1.0

    per_s1_counts = candidate_pairs_df.groupby("source1_entity_id").size()
    zero_candidate_s1 = int((per_s1_counts == 0).sum()) if len(per_s1_counts) else 0

    return {
        "recall": recall,
        "true_pairs_total": len(true_pairs),
        "true_pairs_recovered": len(recovered),
        "candidate_pairs_total": len(candidate_pairs),
        "avg_candidates_per_s1": float(per_s1_counts.mean()) if len(per_s1_counts) else 0.0,
        "median_candidates_per_s1": float(per_s1_counts.median()) if len(per_s1_counts) else 0.0,
        "max_candidates_per_s1": int(per_s1_counts.max()) if len(per_s1_counts) else 0,
        "zero_candidate_s1_count": zero_candidate_s1,
        "missed_pairs": list(true_pairs - candidate_pairs)[:50],  # sample for manual audit, §5
    }
