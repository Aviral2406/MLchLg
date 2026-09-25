"""
Multi-channel blocking / candidate generation. See docs/architecture.md §4-§5.

LEVEL 1 of the two-level system: optimize RECALL. Never let this module become the
place recall quietly gets lost - always measure with evaluate_blocking_recall after
any change (definition of done: cumulative recall >= 0.97 on the validation split,
see CLAUDE.md §5).
"""
from __future__ import annotations
from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from business_entity_resolution.normalization.normalize import normalize, NormalizedRecord


def _normalize_all(df: pd.DataFrame) -> dict[str, NormalizedRecord]:
    """entity_id -> NormalizedRecord, for a source dataframe."""
    return {row["entity_id"]: normalize(row.to_dict()) for _, row in df.iterrows()}


def channel_a_exact_name(s1_norm: dict[str, NormalizedRecord], candidate_norm: dict[str, NormalizedRecord]) -> dict[str, set[str]]:
    """S1 entity_id -> set of candidate entity_ids sharing the exact normalized name."""
    index = defaultdict(set)
    for cid, rec in candidate_norm.items():
        if rec.name.normalized:
            index[rec.name.normalized].add(cid)

    result = defaultdict(set)
    for sid, rec in s1_norm.items():
        if rec.name.normalized in index:
            result[sid] |= index[rec.name.normalized]
    return result


def channel_b_name_token_overlap(s1_norm: dict[str, NormalizedRecord], candidate_norm: dict[str, NormalizedRecord], max_token_df: int = 500) -> dict[str, set[str]]:
    """Inverted index on name tokens, excluding tokens with document frequency > max_token_df."""
    token_df = defaultdict(int)
    for rec in candidate_norm.values():
        for token in set(rec.name.tokenized):
            token_df[token] += 1

    token_to_cids = defaultdict(set)
    for cid, rec in candidate_norm.items():
        for token in set(rec.name.tokenized):
            if token_df[token] <= max_token_df:
                token_to_cids[token].add(cid)

    result = defaultdict(set)
    for sid, rec in s1_norm.items():
        for token in set(rec.name.tokenized):
            if token in token_to_cids:
                result[sid] |= token_to_cids[token]
    return result


def channel_c_name_char_ngram(s1_norm: dict[str, NormalizedRecord], candidate_norm: dict[str, NormalizedRecord], top_k: int = 50) -> dict[str, set[str]]:
    """TF-IDF character n-gram cosine similarity candidate retrieval for business names."""
    if not s1_norm or not candidate_norm:
        return defaultdict(set)

    s1_ids = list(s1_norm.keys())
    cand_ids = list(candidate_norm.keys())

    s1_texts = [s1_norm[sid].name.normalized for sid in s1_ids]
    cand_texts = [candidate_norm[cid].name.normalized for cid in cand_ids]

    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(3, 4), min_df=1)
    try:
        cand_matrix = vectorizer.fit_transform(cand_texts)
        s1_matrix = vectorizer.transform(s1_texts)
    except ValueError:
        return defaultdict(set)

    sim_matrix = s1_matrix.dot(cand_matrix.T)

    result = defaultdict(set)
    for idx, sid in enumerate(s1_ids):
        row = sim_matrix[idx]
        if row.nnz == 0:
            continue
        # Retrieve indices of top_k similarity scores
        n_cands = min(top_k, row.nnz)
        top_indices = np.argsort(row.data)[-n_cands:]
        cand_indices = row.indices[top_indices]
        for c_idx in cand_indices:
            result[sid].add(cand_ids[c_idx])
    return result


def channel_d_address_token_overlap(s1_norm: dict[str, NormalizedRecord], candidate_norm: dict[str, NormalizedRecord], max_token_df: int = 500) -> dict[str, set[str]]:
    """Inverted index on address tokens, excluding high-DF tokens."""
    token_df = defaultdict(int)
    for rec in candidate_norm.values():
        for token in set(rec.address.tokenized):
            token_df[token] += 1

    token_to_cids = defaultdict(set)
    for cid, rec in candidate_norm.items():
        for token in set(rec.address.tokenized):
            if token_df[token] <= max_token_df:
                token_to_cids[token].add(cid)

    result = defaultdict(set)
    for sid, rec in s1_norm.items():
        for token in set(rec.address.tokenized):
            if token in token_to_cids:
                result[sid] |= token_to_cids[token]
    return result


def channel_e_pin_exact(s1_norm: dict[str, NormalizedRecord], candidate_norm: dict[str, NormalizedRecord]) -> dict[str, set[str]]:
    """Exact match on extracted PIN / postal code digits (5-6 digits)."""
    pin_index = defaultdict(set)
    for cid, rec in candidate_norm.items():
        pins = [d for d in rec.address.digits_only if len(d) in (5, 6)]
        for pin in pins:
            pin_index[pin].add(cid)

    result = defaultdict(set)
    for sid, rec in s1_norm.items():
        pins = [d for d in rec.address.digits_only if len(d) in (5, 6)]
        for pin in pins:
            if pin in pin_index:
                result[sid] |= pin_index[pin]
    return result


def channel_f_city_state_cooccurrence(s1_norm: dict[str, NormalizedRecord], candidate_norm: dict[str, NormalizedRecord]) -> dict[str, set[str]]:
    """Block on shared token pairs in address."""
    pair_index = defaultdict(set)
    for cid, rec in candidate_norm.items():
        tokens = sorted(set(rec.address.tokenized))
        if len(tokens) >= 2:
            for i in range(len(tokens)):
                for j in range(i + 1, min(i + 5, len(tokens))):
                    pair_index[(tokens[i], tokens[j])].add(cid)

    result = defaultdict(set)
    for sid, rec in s1_norm.items():
        tokens = sorted(set(rec.address.tokenized))
        if len(tokens) >= 2:
            for i in range(len(tokens)):
                for j in range(i + 1, min(i + 5, len(tokens))):
                    pair = (tokens[i], tokens[j])
                    if pair in pair_index:
                        result[sid] |= pair_index[pair]
    return result


def channel_g_address_char_ngram(s1_norm: dict[str, NormalizedRecord], candidate_norm: dict[str, NormalizedRecord], top_k: int = 50) -> dict[str, set[str]]:
    """TF-IDF character n-gram cosine similarity candidate retrieval for addresses."""
    if not s1_norm or not candidate_norm:
        return defaultdict(set)

    s1_ids = list(s1_norm.keys())
    cand_ids = list(candidate_norm.keys())

    s1_texts = [s1_norm[sid].address.normalized for sid in s1_ids]
    cand_texts = [candidate_norm[cid].address.normalized for cid in cand_ids]

    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(3, 4), min_df=1)
    try:
        cand_matrix = vectorizer.fit_transform(cand_texts)
        s1_matrix = vectorizer.transform(s1_texts)
    except ValueError:
        return defaultdict(set)

    sim_matrix = s1_matrix.dot(cand_matrix.T)

    result = defaultdict(set)
    for idx, sid in enumerate(s1_ids):
        row = sim_matrix[idx]
        if row.nnz == 0:
            continue
        n_cands = min(top_k, row.nnz)
        top_indices = np.argsort(row.data)[-n_cands:]
        cand_indices = row.indices[top_indices]
        for c_idx in cand_indices:
            result[sid].add(cand_ids[c_idx])
    return result


CHANNELS = {
    "exact_name": channel_a_exact_name,
    "name_token_overlap": channel_b_name_token_overlap,
    "name_char_ngram": channel_c_name_char_ngram,
    "address_token_overlap": channel_d_address_token_overlap,
    "pin_exact": channel_e_pin_exact,
    "city_state_cooccurrence": channel_f_city_state_cooccurrence,
    "address_char_ngram": channel_g_address_char_ngram,
}


def generate_candidates(s1_df: pd.DataFrame, s2_df: pd.DataFrame, s3_df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Returns candidate_pairs_df: [source1_entity_id, candidate_entity_id, channels]."""
    s1_norm = _normalize_all(s1_df)
    candidate_df = pd.concat([s2_df, s3_df], ignore_index=True)
    candidate_norm = _normalize_all(candidate_df)

    enabled = config.get("blocking", {}).get("channels", {})
    pair_channels = defaultdict(set)

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
    if cap and not out.empty:
        out = out.groupby("source1_entity_id", group_keys=False).head(cap)

    missing = set(s1_df["entity_id"]) - set(out["source1_entity_id"])
    if missing:
        out = pd.concat([out, pd.DataFrame({"source1_entity_id": list(missing),
                                             "candidate_entity_id": None, "channels": None})],
                         ignore_index=True)
    return out


def evaluate_blocking_recall(candidate_pairs_df: pd.DataFrame, ground_truth_df: pd.DataFrame) -> dict:
    """Per-channel and cumulative recall stats. See docs/architecture.md §5."""
    true_pairs = set()
    gt_s1_col = "source1_entity_id" if "source1_entity_id" in ground_truth_df.columns else ground_truth_df.columns[0]
    gt_match_col = "matched_entity_ids" if "matched_entity_ids" in ground_truth_df.columns else ground_truth_df.columns[1]

    for _, row in ground_truth_df.iterrows():
        matched = str(row[gt_match_col] or "")
        if not matched or matched == "nan":
            continue
        for mid in matched.split(","):
            if mid.strip():
                true_pairs.add((str(row[gt_s1_col]), mid.strip()))

    valid_cands = candidate_pairs_df.dropna(subset=["candidate_entity_id"])
    candidate_pairs = set(zip(valid_cands["source1_entity_id"], valid_cands["candidate_entity_id"]))

    recovered = true_pairs & candidate_pairs
    recall = len(recovered) / len(true_pairs) if true_pairs else 1.0

    per_s1_counts = valid_cands.groupby("source1_entity_id").size()
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
        "missed_pairs": list(true_pairs - candidate_pairs)[:50],
    }
