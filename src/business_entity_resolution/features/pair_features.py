"""
Pairwise feature engineering. See docs/architecture.md §6.

Consumes NORMALIZED records only (via normalization.normalize) - never re-derive
similarity from raw strings here.
"""
from __future__ import annotations
import pandas as pd
from rapidfuzz import fuzz, distance

from business_entity_resolution.normalization.normalize import normalize, NormalizedRecord


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _overlap_coefficient(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


def name_features(a: NormalizedRecord, b: NormalizedRecord) -> dict:
    a_tok, b_tok = set(a.name.tokenized), set(b.name.tokenized)
    a_ng, b_ng = set(a.name.char_ngrams), set(b.name.char_ngrams)

    # First token is usually the primary brand/entity name
    a_first = a.name.tokenized[0] if a.name.tokenized else ""
    b_first = b.name.tokenized[0] if b.name.tokenized else ""
    first_token_match = float(a_first == b_first and bool(a_first))

    len_a = len(a.name.normalized)
    len_b = len(b.name.normalized)
    len_ratio = min(len_a, len_b) / max(len_a, len_b, 1)

    return {
        "name_exact_normalized": float(a.name.normalized == b.name.normalized),
        "name_exact_sorted_tokens": float(a.name.sorted_tokens == b.name.sorted_tokens),
        "name_levenshtein_sim": distance.Levenshtein.normalized_similarity(a.name.normalized, b.name.normalized),
        "name_jaro_winkler_sim": distance.JaroWinkler.similarity(a.name.normalized, b.name.normalized),
        "name_token_jaccard": _jaccard(a_tok, b_tok),
        "name_token_overlap_coef": _overlap_coefficient(a_tok, b_tok),
        "name_token_sort_ratio": fuzz.token_sort_ratio(a.name.normalized, b.name.normalized) / 100.0,
        "name_token_set_ratio": fuzz.token_set_ratio(a.name.normalized, b.name.normalized) / 100.0,
        "name_partial_ratio": fuzz.partial_ratio(a.name.normalized, b.name.normalized) / 100.0,
        "name_char_ngram_jaccard": _jaccard(a_ng, b_ng),
        "name_first_token_match": first_token_match,
        "name_common_token_count": len(a_tok & b_tok),
        "name_length_diff": abs(len_a - len_b),
        "name_length_ratio": len_ratio,
        "name_digit_overlap": float(set(a.name.digits_only) == set(b.name.digits_only) and bool(a.name.digits_only)),
    }


def address_features(a: NormalizedRecord, b: NormalizedRecord) -> dict:
    a_tok, b_tok = set(a.address.tokenized), set(b.address.tokenized)
    a_ng, b_ng = set(a.address.char_ngrams), set(b.address.char_ngrams)
    a_digits, b_digits = set(a.address.digits_only), set(b.address.digits_only)

    a_pins = set([d for d in a.address.digits_only if len(d) in (5, 6)])
    b_pins = set([d for d in b.address.digits_only if len(d) in (5, 6)])

    if a_pins and b_pins:
        pin_status = 1.0 if bool(a_pins & b_pins) else 0.0
    else:
        pin_status = 0.5  # Neutral missing value

    a_hno = a.address.digits_only[0] if a.address.digits_only else None
    b_hno = b.address.digits_only[0] if b.address.digits_only else None

    if a_hno and b_hno:
        house_number_status = 1.0 if a_hno == b_hno else 0.0
    else:
        house_number_status = 0.5

    len_a = len(a.address.normalized)
    len_b = len(b.address.normalized)
    len_ratio = min(len_a, len_b) / max(len_a, len_b, 1)

    return {
        "address_exact_normalized": float(a.address.normalized == b.address.normalized),
        "address_token_jaccard": _jaccard(a_tok, b_tok),
        "address_token_overlap_coef": _overlap_coefficient(a_tok, b_tok),
        "address_char_ngram_jaccard": _jaccard(a_ng, b_ng),
        "address_levenshtein_sim": distance.Levenshtein.normalized_similarity(a.address.normalized, b.address.normalized),
        "address_token_sort_ratio": fuzz.token_sort_ratio(a.address.normalized, b.address.normalized) / 100.0,
        "address_partial_ratio": fuzz.partial_ratio(a.address.normalized, b.address.normalized) / 100.0,
        "address_numeric_token_overlap": _jaccard(a_digits, b_digits),
        "address_length_diff": abs(len_a - len_b),
        "address_length_ratio": len_ratio,
        "pin_exact_match": pin_status,
        "house_number_compatibility": house_number_status,
    }


def country_features(a: NormalizedRecord, b: NormalizedRecord, known_train_countries: set = frozenset()) -> dict:
    return {
        "country_exact_match": float(a.country_normalized == b.country_normalized),
        "country_a_missing": float(a.country_normalized == ""),
        "country_b_missing": float(b.country_normalized == ""),
        "country_a_unseen_in_train": float(bool(known_train_countries) and a.country_normalized not in known_train_countries),
        "country_b_unseen_in_train": float(bool(known_train_countries) and b.country_normalized not in known_train_countries),
    }


ALL_CHANNEL_NAMES = [
    "exact_name", "name_token_overlap", "name_char_ngram",
    "address_token_overlap", "pin_exact", "city_state_cooccurrence", "address_char_ngram"
]


def cross_field_features(name_feats: dict, address_feats: dict, channel_str: str = "") -> dict:
    name_sim = name_feats["name_char_ngram_jaccard"]
    addr_sim = address_feats["address_char_ngram_jaccard"]

    ch_set = set(channel_str.split(",")) if channel_str else set()
    provenance_feats = {f"ch_prov_{ch}": float(ch in ch_set) for ch in ALL_CHANNEL_NAMES}
    provenance_feats["ch_count"] = len(ch_set)

    return {
        "name_x_address_sim_product": name_sim * addr_sim,
        "strong_name_weak_address": float(name_sim > 0.7 and addr_sim < 0.3),
        "weak_name_strong_address": float(name_sim < 0.3 and addr_sim > 0.7),
        "strong_name_strong_address": float(name_sim > 0.7 and addr_sim > 0.7),
        **provenance_feats,
    }


def build_pair_features(
    pairs_df: pd.DataFrame,
    s1_df: pd.DataFrame,
    s2_df: pd.DataFrame,
    s3_df: pd.DataFrame,
    known_train_countries: set = frozenset(),
    s1_norm: dict | None = None,
    cand_norm: dict | None = None,
    batch_size: int = 25000,
) -> pd.DataFrame:
    """pairs_df: [source1_entity_id, candidate_entity_id, channels].
    Returns one feature row per pair in memory-safe chunks.

    Accepts pre-built norm dicts (s1_norm, cand_norm) to avoid re-normalizing
    when called repeatedly (e.g. batched inference). If not provided, normalizes
    inline from the dataframe arguments.
    """
    if pairs_df.empty:
        return pd.DataFrame()

    if s1_norm is None:
        s1_norm = {row.entity_id: normalize(row._asdict()) for row in s1_df.itertuples(index=False)}
    if cand_norm is None:
        cand_all = pd.concat([s2_df, s3_df], ignore_index=True)
        cand_norm = {row.entity_id: normalize(row._asdict()) for row in cand_all.itertuples(index=False)}

    chunk_dfs = []
    chunk_rows = []

    for row in pairs_df.itertuples(index=False):
        sid = row.source1_entity_id
        cid = row.candidate_entity_id
        ch_str = str(getattr(row, "channels", "") or "")
        if not cid or cid != cid:  # None / NaN check
            continue
        a = s1_norm.get(sid)
        b = cand_norm.get(cid)
        if a is None or b is None:
            continue
        nf = name_features(a, b)
        af = address_features(a, b)
        cf = country_features(a, b, known_train_countries)
        xf = cross_field_features(nf, af, ch_str)
        chunk_rows.append({"source1_entity_id": sid, "candidate_entity_id": cid, **nf, **af, **cf, **xf})

        if len(chunk_rows) >= batch_size:
            chunk_dfs.append(pd.DataFrame(chunk_rows))
            chunk_rows = []

    if chunk_rows:
        chunk_dfs.append(pd.DataFrame(chunk_rows))
        chunk_rows = []

    if not chunk_dfs:
        return pd.DataFrame()

    out_df = pd.concat(chunk_dfs, ignore_index=True)
    del chunk_dfs
    return out_df
