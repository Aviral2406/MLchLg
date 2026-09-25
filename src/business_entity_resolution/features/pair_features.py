"""
Pairwise feature engineering. See docs/architecture.md §6.

Consumes NORMALIZED records only (via normalization.normalize) - never re-derive
similarity from raw strings here. A representative subset of features is implemented
below (name + address string/token similarity, country compatibility, one cross-field
interaction). Extend following the same pattern - see docs/task_specs.md "Pair
Features" for the full target feature list and priority order.
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
    return {
        "name_exact_normalized": float(a.name.normalized == b.name.normalized),
        "name_exact_sorted_tokens": float(a.name.sorted_tokens == b.name.sorted_tokens),
        "name_levenshtein_sim": distance.Levenshtein.normalized_similarity(a.name.normalized, b.name.normalized),
        "name_jaro_winkler_sim": distance.JaroWinkler.similarity(a.name.normalized, b.name.normalized),
        "name_token_jaccard": _jaccard(a_tok, b_tok),
        "name_token_overlap_coef": _overlap_coefficient(a_tok, b_tok),
        "name_token_sort_ratio": fuzz.token_sort_ratio(a.name.normalized, b.name.normalized) / 100.0,
        "name_token_set_ratio": fuzz.token_set_ratio(a.name.normalized, b.name.normalized) / 100.0,
        "name_char_ngram_jaccard": _jaccard(a_ng, b_ng),
        "name_common_token_count": len(a_tok & b_tok),
        "name_length_diff": abs(len(a.name.normalized) - len(b.name.normalized)),
        "name_digit_overlap": float(set(a.name.digits_only) == set(b.name.digits_only) and bool(a.name.digits_only)),
    }


def address_features(a: NormalizedRecord, b: NormalizedRecord) -> dict:
    a_tok, b_tok = set(a.address.tokenized), set(b.address.tokenized)
    a_ng, b_ng = set(a.address.char_ngrams), set(b.address.char_ngrams)
    a_digits, b_digits = set(a.address.digits_only), set(b.address.digits_only)
    return {
        "address_exact_normalized": float(a.address.normalized == b.address.normalized),
        "address_token_jaccard": _jaccard(a_tok, b_tok),
        "address_token_overlap_coef": _overlap_coefficient(a_tok, b_tok),
        "address_char_ngram_jaccard": _jaccard(a_ng, b_ng),
        "address_levenshtein_sim": distance.Levenshtein.normalized_similarity(a.address.normalized, b.address.normalized),
        "address_numeric_token_overlap": _jaccard(a_digits, b_digits),
        "address_length_diff": abs(len(a.address.normalized) - len(b.address.normalized)),
        # TODO: pin_exact_match, pin_missing_flags, city_token_match, state_token_match,
        # house_number_compatibility, landmark_token_overlap - see docs/task_specs.md.
        # These depend on the address_components soft-extraction from architecture.md §3,
        # which should be implemented alongside normalization.normalize_field.
    }


def country_features(a: NormalizedRecord, b: NormalizedRecord, known_train_countries: set = frozenset()) -> dict:
    """known_train_countries: pass the set of country_normalized values seen in TRAIN
    only, so the unseen-country flag is meaningful at test time (e.g. France).
    Never fit a fixed-vocabulary encoder on this - see CLAUDE.md §2."""
    return {
        "country_exact_match": float(a.country_normalized == b.country_normalized),
        "country_a_missing": float(a.country_normalized == ""),
        "country_b_missing": float(b.country_normalized == ""),
        "country_a_unseen_in_train": float(bool(known_train_countries) and a.country_normalized not in known_train_countries),
        "country_b_unseen_in_train": float(bool(known_train_countries) and b.country_normalized not in known_train_countries),
    }


def cross_field_features(name_feats: dict, address_feats: dict) -> dict:
    name_sim = name_feats["name_char_ngram_jaccard"]
    addr_sim = address_feats["address_char_ngram_jaccard"]
    return {
        "name_x_address_sim_product": name_sim * addr_sim,
        "strong_name_weak_address": float(name_sim > 0.7 and addr_sim < 0.3),
        "weak_name_strong_address": float(name_sim < 0.3 and addr_sim > 0.7),
        "strong_name_strong_address": float(name_sim > 0.7 and addr_sim > 0.7),
        # TODO: add blocking-provenance one-hot features once candidate_pairs_df's
        # `channels` column is joined in by build_pair_features - see architecture.md §6.
    }


def build_pair_features(pairs_df: pd.DataFrame, s1_df: pd.DataFrame,
                         s2_df: pd.DataFrame, s3_df: pd.DataFrame,
                         known_train_countries: set = frozenset()) -> pd.DataFrame:
    """pairs_df: [source1_entity_id, candidate_entity_id, channels] (candidate_pairs_df
    from blocking.block.generate_candidates). Returns one feature row per pair."""
    s1_norm = {row["entity_id"]: normalize(row.to_dict()) for _, row in s1_df.iterrows()}
    cand_all = pd.concat([s2_df, s3_df], ignore_index=True)
    cand_norm = {row["entity_id"]: normalize(row.to_dict()) for _, row in cand_all.iterrows()}

    rows = []
    for _, row in pairs_df.iterrows():
        sid, cid = row["source1_entity_id"], row["candidate_entity_id"]
        if cid is None or (isinstance(cid, float)):
            continue  # S1 entity with no candidates - handled downstream as a forced singleton
        a, b = s1_norm[sid], cand_norm[cid]
        nf = name_features(a, b)
        af = address_features(a, b)
        cf = country_features(a, b, known_train_countries)
        xf = cross_field_features(nf, af)
        rows.append({"source1_entity_id": sid, "candidate_entity_id": cid, **nf, **af, **cf, **xf})

    return pd.DataFrame(rows)
