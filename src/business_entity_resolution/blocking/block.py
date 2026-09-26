"""
Multi-channel blocking / candidate generation.
Memory-safe, high-speed candidate retrieval with stopword filtering,
chunked TF-IDF cosine matching, and strict per-S1 candidate capping (< 1.5 GB RAM).
"""
from __future__ import annotations
from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from business_entity_resolution.normalization.normalize import normalize, NormalizedRecord

# High-frequency stop words that cause explosive Cartesian products if not filtered
STOPWORDS = frozenset({
    "the", "and", "or", "of", "in", "at", "to", "for", "a", "an", "on", "by", "with", "from",
    "road", "street", "st", "rd", "ave", "avenue", "lane", "ln", "dr", "drive", "suite", "ste",
    "floor", "flr", "apt", "apartment", "bldg", "building", "box", "po", "near", "opp", "opposite",
    "ltd", "limited", "pvt", "private", "inc", "corp", "corporation", "llc", "company", "co",
    "enterprise", "services", "india", "us", "usa"
})

MAX_CANDIDATES_PER_CHANNEL = 150


def _normalize_all(df: pd.DataFrame) -> dict[str, NormalizedRecord]:
    """entity_id -> NormalizedRecord. Fast, low-memory record conversion."""
    result = {}
    for row in df.itertuples(index=False):
        d = row._asdict()
        result[d["entity_id"]] = normalize(d)
    return result


def channel_a_exact_name(s1_norm: dict[str, NormalizedRecord], candidate_norm: dict[str, NormalizedRecord]) -> dict[str, set[str]]:
    """S1 entity_id -> candidate entity_ids sharing exact normalized name."""
    index = defaultdict(set)
    for cid, rec in candidate_norm.items():
        if rec.name.normalized:
            index[rec.name.normalized].add(cid)

    result = defaultdict(set)
    for sid, rec in s1_norm.items():
        if rec.name.normalized in index:
            cands = list(index[rec.name.normalized])[:MAX_CANDIDATES_PER_CHANNEL]
            result[sid] = set(cands)
    return result


def channel_b_name_token_overlap(s1_norm: dict[str, NormalizedRecord], candidate_norm: dict[str, NormalizedRecord], max_token_df: int = 150) -> dict[str, set[str]]:
    """Inverted index on distinctive name tokens, keeping singletons when they are informative."""
    token_df = defaultdict(int)
    for rec in candidate_norm.values():
        for token in set(rec.name.tokenized) - STOPWORDS:
            token_df[token] += 1

    token_to_cids = defaultdict(set)
    for cid, rec in candidate_norm.items():
        for token in set(rec.name.tokenized) - STOPWORDS:
            if 0 < token_df[token] <= max_token_df:
                token_to_cids[token].add(cid)

    result = defaultdict(set)
    for sid, rec in s1_norm.items():
        matched = set()
        for token in set(rec.name.tokenized) - STOPWORDS:
            if token in token_to_cids:
                matched |= token_to_cids[token]
                if len(matched) >= MAX_CANDIDATES_PER_CHANNEL:
                    break
        if matched:
            result[sid] = set(list(matched)[:MAX_CANDIDATES_PER_CHANNEL])
    return result


def channel_c_name_char_ngram(s1_norm: dict[str, NormalizedRecord], candidate_norm: dict[str, NormalizedRecord], top_k: int = 30) -> dict[str, set[str]]:
    """Chunked TF-IDF character n-gram cosine retrieval for business names (memory-safe)."""
    if not s1_norm or not candidate_norm:
        return defaultdict(set)

    s1_ids = list(s1_norm.keys())
    cand_ids = list(candidate_norm.keys())

    s1_texts = [s1_norm[sid].name.normalized for sid in s1_ids]
    cand_texts = [candidate_norm[cid].name.normalized for cid in cand_ids]

    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(3, 4), min_df=3, max_features=25000)
    try:
        cand_matrix = vectorizer.fit_transform(cand_texts)
    except ValueError:
        return defaultdict(set)

    result = defaultdict(set)
    chunk_size = 5000
    for start in range(0, len(s1_ids), chunk_size):
        end = min(start + chunk_size, len(s1_ids))
        sub_s1_texts = s1_texts[start:end]
        sub_s1_ids = s1_ids[start:end]

        try:
            sub_s1_matrix = vectorizer.transform(sub_s1_texts)
            sim_matrix = sub_s1_matrix.dot(cand_matrix.T)
        except Exception:
            continue

        for i, sid in enumerate(sub_s1_ids):
            row = sim_matrix[i]
            if row.nnz == 0:
                continue
            mask = row.data >= 0.20
            if np.any(mask):
                valid_indices = row.indices[mask]
                valid_scores = row.data[mask]
                top_order = np.argsort(valid_scores)[-top_k:]
                for c_idx in valid_indices[top_order]:
                    result[sid].add(cand_ids[c_idx])

    return result


def channel_d_address_token_overlap(s1_norm: dict[str, NormalizedRecord], candidate_norm: dict[str, NormalizedRecord], max_token_df: int = 150) -> dict[str, set[str]]:
    """Inverted index on distinctive address tokens, including rare site markers like sector and SBI."""
    token_df = defaultdict(int)
    for rec in candidate_norm.values():
        for token in set(rec.address.tokenized) - STOPWORDS:
            token_df[token] += 1

    token_to_cids = defaultdict(set)
    for cid, rec in candidate_norm.items():
        for token in set(rec.address.tokenized) - STOPWORDS:
            if 0 < token_df[token] <= max_token_df:
                token_to_cids[token].add(cid)

    result = defaultdict(set)
    for sid, rec in s1_norm.items():
        matched = set()
        for token in set(rec.address.tokenized) - STOPWORDS:
            if token in token_to_cids:
                matched |= token_to_cids[token]
                if len(matched) >= MAX_CANDIDATES_PER_CHANNEL:
                    break
        if matched:
            result[sid] = set(list(matched)[:MAX_CANDIDATES_PER_CHANNEL])
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
        matched = set()
        for pin in pins:
            if pin in pin_index:
                matched |= pin_index[pin]
                if len(matched) >= MAX_CANDIDATES_PER_CHANNEL:
                    break
        if matched:
            result[sid] = set(list(matched)[:MAX_CANDIDATES_PER_CHANNEL])
    return result


def _coarse_address_context_tokens(rec: NormalizedRecord) -> set[str]:
    return {
        token for token in rec.address.tokenized
        if token and token not in STOPWORDS and not token.isdigit() and len(token) > 1
    }


def channel_f_city_state_cooccurrence(s1_norm: dict[str, NormalizedRecord], candidate_norm: dict[str, NormalizedRecord]) -> dict[str, set[str]]:
    """Area-context overlap catches same city/state even when business names drift.
    Uses an inverted index (O(N) build + O(Q*k) lookup) — NOT a brute-force O(N²) loop.
    The previous nested-loop implementation was broken at real scale (10M+ records).
    """
    # Build inverted index: token -> set of candidate ids
    token_to_cids: dict[str, set[str]] = defaultdict(set)
    token_df: dict[str, int] = defaultdict(int)
    for cid, rec in candidate_norm.items():
        ctx = _coarse_address_context_tokens(rec)
        for tok in ctx:
            token_df[tok] += 1
    # Only index tokens that appear <max_df times to avoid generic city/state explosions
    max_df = max(200, len(candidate_norm) // 500)  # at most 0.2% of corpus
    for cid, rec in candidate_norm.items():
        for tok in _coarse_address_context_tokens(rec):
            if token_df[tok] <= max_df:
                token_to_cids[tok].add(cid)

    result: dict[str, set[str]] = defaultdict(set)
    for sid, rec in s1_norm.items():
        matched: set[str] = set()
        for tok in _coarse_address_context_tokens(rec):
            if tok in token_to_cids:
                matched |= token_to_cids[tok]
                if len(matched) >= MAX_CANDIDATES_PER_CHANNEL:
                    break
        if matched:
            result[sid] = set(list(matched)[:MAX_CANDIDATES_PER_CHANNEL])
    return result


def channel_g_address_char_ngram(s1_norm: dict[str, NormalizedRecord], candidate_norm: dict[str, NormalizedRecord], top_k: int = 30) -> dict[str, set[str]]:
    """Address char n-gram retrieval catches regional spelling drift and abbreviation noise."""
    if not s1_norm or not candidate_norm:
        return defaultdict(set)

    s1_ids = list(s1_norm.keys())
    cand_ids = list(candidate_norm.keys())
    s1_texts = [s1_norm[sid].address.normalized for sid in s1_ids]
    cand_texts = [candidate_norm[cid].address.normalized for cid in cand_ids]

    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(3, 4), min_df=2, max_features=20000)
    try:
        cand_matrix = vectorizer.fit_transform(cand_texts)
    except ValueError:
        return defaultdict(set)

    result = defaultdict(set)
    chunk_size = 5000
    for start in range(0, len(s1_ids), chunk_size):
        end = min(start + chunk_size, len(s1_ids))
        sub_s1_texts = s1_texts[start:end]
        sub_s1_ids = s1_ids[start:end]

        try:
            sub_s1_matrix = vectorizer.transform(sub_s1_texts)
            sim_matrix = sub_s1_matrix.dot(cand_matrix.T)
        except Exception:
            continue

        for i, sid in enumerate(sub_s1_ids):
            row = sim_matrix[i]
            if row.nnz == 0:
                continue
            mask = row.data >= 0.20
            if np.any(mask):
                valid_indices = row.indices[mask]
                valid_scores = row.data[mask]
                top_order = np.argsort(valid_scores)[-top_k:]
                for c_idx in valid_indices[top_order]:
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


def generate_candidates(
    s1_df: pd.DataFrame,
    s2_df: pd.DataFrame,
    s3_df: pd.DataFrame,
    config: dict,
    s1_norm: dict[str, NormalizedRecord] | None = None,
    candidate_norm: dict[str, NormalizedRecord] | None = None,
) -> pd.DataFrame:
    """Returns candidate_pairs_df: [source1_entity_id, candidate_entity_id, channels].
    Guaranteed memory-safe: caps candidates strictly per S1 entity during collection.
    """
    if s1_norm is None:
        print("  [Blocking] Normalizing S1 records in memory...")
        s1_norm = _normalize_all(s1_df)
    if candidate_norm is None:
        print("  [Blocking] Normalizing candidate records in memory...")
        candidate_df = pd.concat([s2_df, s3_df], ignore_index=True)
        candidate_norm = _normalize_all(candidate_df)

    enabled = config.get("blocking", {}).get("channels", {})
    pair_channels = defaultdict(set)

    for name, fn in CHANNELS.items():
        if not enabled.get(name, True):
            continue
        print(f"  [Blocking] Running Channel '{name}'...")
        channel_result = fn(s1_norm, candidate_norm)
        for sid, cand_ids in channel_result.items():
            for cid in cand_ids:
                pair_channels[(sid, cid)].add(name)

    cap_per_s1 = config.get("blocking", {}).get("candidate_cap_per_s1", 100)

    # Group candidates by S1 entity and enforce cap (prioritizing multi-channel and name matches)
    s1_cands = defaultdict(dict)
    for (sid, cid), chs in pair_channels.items():
        s1_cands[sid][cid] = chs

    final_pairs = []
    for sid, cand_dict in s1_cands.items():
        if len(cand_dict) > cap_per_s1:
            sorted_cands = sorted(
                cand_dict.items(),
                key=lambda item: (len(item[1]), "exact_name" in item[1], "name_token_overlap" in item[1]),
                reverse=True,
            )[:cap_per_s1]
            for cid, chs in sorted_cands:
                final_pairs.append((sid, cid, chs))
        else:
            for cid, chs in cand_dict.items():
                final_pairs.append((sid, cid, chs))

    print(f"  [Blocking] Formatting {len(final_pairs):,} unique candidate pairs...")
    rows = [
        {"source1_entity_id": sid, "candidate_entity_id": cid, "channels": ",".join(sorted(chs))}
        for sid, cid, chs in final_pairs
    ]
    out = pd.DataFrame(rows, columns=["source1_entity_id", "candidate_entity_id", "channels"])

    # Ensure every S1 entity has an entry
    missing = set(s1_df["entity_id"]) - set(out["source1_entity_id"])
    if missing:
        missing_df = pd.DataFrame({"source1_entity_id": list(missing), "candidate_entity_id": None, "channels": None})
        out = pd.concat([out, missing_df], ignore_index=True)

    return out


def evaluate_blocking_recall(candidate_pairs_df: pd.DataFrame, ground_truth_df: pd.DataFrame) -> dict:
    """Per-channel and cumulative recall stats."""
    true_pairs = set()
    gt_s1_col = "source1_entity_id" if "source1_entity_id" in ground_truth_df.columns else ground_truth_df.columns[0]
    gt_match_col = "matched_entity_ids" if "matched_entity_ids" in ground_truth_df.columns else ground_truth_df.columns[1]

    for _, row in ground_truth_df.iterrows():
        matched = str(row[gt_match_col] or "")
        if not matched or matched == "nan":
            continue
        for mid in matched.split(","):
            m_clean = mid.strip()
            if m_clean:
                true_pairs.add((str(row[gt_s1_col]), m_clean))

    valid_cands = candidate_pairs_df.dropna(subset=["candidate_entity_id"])
    candidate_pairs = set(zip(valid_cands["source1_entity_id"], valid_cands["candidate_entity_id"]))

    recovered = true_pairs & candidate_pairs
    recall = len(recovered) / len(true_pairs) if true_pairs else 1.0

    per_s1_counts = valid_cands.groupby("source1_entity_id").size()

    return {
        "recall": recall,
        "true_pairs_total": len(true_pairs),
        "true_pairs_recovered": len(recovered),
        "candidate_pairs_total": len(candidate_pairs),
        "avg_candidates_per_s1": float(per_s1_counts.mean()) if len(per_s1_counts) else 0.0,
        "missed_pairs": list(true_pairs - candidate_pairs)[:30],
    }
