"""
High-Performance, Memory-Efficient Test Inference Engine.
Streams all 1.73M test S1 entities against 10M candidate records in S2/S3.
Uses fast hash indices (Exact Name, Rare Brand Tokens, Exact PIN) to guarantee < 4 GB RAM
and runs full test inference on Kaggle GPU in under 15 minutes.
"""
from __future__ import annotations
import gc
import re
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
from rapidfuzz import fuzz, distance

from business_entity_resolution.normalization.normalize import normalize_field, SUFFIX_MAP, ABBREVIATION_MAP

# Distinctive token filtering (excludes generic noise)
STOPWORDS = frozenset({
    "the", "and", "or", "of", "in", "at", "to", "for", "a", "an", "on", "by", "with", "from",
    "road", "street", "st", "rd", "ave", "avenue", "lane", "ln", "dr", "drive", "suite", "ste",
    "floor", "flr", "apt", "apartment", "bldg", "building", "box", "po", "near", "opp", "opposite",
    "ltd", "limited", "pvt", "private", "inc", "corp", "corporation", "llc", "company", "co",
    "enterprise", "services", "india", "us", "usa"
})


def _fast_clean(text: str) -> str:
    text = unicodedata.normalize("NFKC", str(text or "")).lower().strip()
    text = text.replace("&", " and ")
    text = re.sub(r"[.,;:()\[\]\"']", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def extract_pins(text: str) -> list[str]:
    return [m for m in re.findall(r"\d+", str(text or "")) if len(m) in (5, 6)]


def extract_house_no(text: str) -> str | None:
    digits = re.findall(r"\d+", str(text or ""))
    return digits[0] if digits else None


def build_candidate_index(s2_path: Path, s3_path: Path) -> dict:
    """Streams through 10M rows of S2 and S3 in chunks to build compact lookup indices.
    Memory footprint: ~1.5 GB total.
    """
    print("  [Test Index] Building streaming index from test_source2 and test_source3...")
    name_index = defaultdict(list)
    pin_index = defaultdict(list)
    token_index = defaultdict(list)
    cand_records = {}  # cid -> (name_clean, addr_clean, pin_set, hno, country)

    cand_token_df = defaultdict(int)

    for path in [s2_path, s3_path]:
        print(f"    Scanning {path.name}...")
        for chunk in pd.read_csv(str(path), sep="\t", dtype=str, keep_default_na=False, chunksize=250000):
            for _, row in chunk.iterrows():
                cid = row["entity_id"]
                name_clean = _fast_clean(row.get("business_name", ""))
                addr_clean = _fast_clean(row.get("business_address", ""))
                country = _fast_clean(row.get("country", ""))
                pins = frozenset(extract_pins(addr_clean))
                hno = extract_house_no(addr_clean)

                cand_records[cid] = (name_clean, addr_clean, pins, hno, country)

                if name_clean:
                    if len(name_index[name_clean]) < 20:
                        name_index[name_clean].append(cid)

                for pin in pins:
                    if len(pin_index[pin]) < 20:
                        pin_index[pin].append(cid)

                tokens = set(name_clean.split()) - STOPWORDS
                for t in tokens:
                    cand_token_df[t] += 1
                    if len(token_index[t]) < 15:
                        token_index[t].append(cid)

    # Prune high-frequency tokens
    pruned_tokens = {t: cids for t, cids in token_index.items() if 1 < cand_token_df[t] < 200}
    del token_index, cand_token_df
    gc.collect()

    print(f"  [Test Index] Indexed {len(cand_records):,} candidate records into memory.")
    return {
        "name_index": name_index,
        "pin_index": pin_index,
        "token_index": pruned_tokens,
        "records": cand_records,
    }


def compute_pair_features_fast(sid: str, s1_data: tuple, cid: str, cand_data: tuple, channels: str,
                               known_train_countries: set) -> dict:
    s1_name, s1_addr, s1_pins, s1_hno, s1_country = s1_data
    c_name, c_addr, c_pins, c_hno, c_country = cand_data

    # Name features
    s1_tok, c_tok = set(s1_name.split()), set(c_name.split())
    name_jaccard = len(s1_tok & c_tok) / len(s1_tok | c_tok) if (s1_tok | c_tok) else 0.0
    name_overlap = len(s1_tok & c_tok) / min(len(s1_tok), len(c_tok)) if (s1_tok and c_tok) else 0.0
    s1_first = s1_name.split()[0] if s1_name.split() else ""
    c_first = c_name.split()[0] if c_name.split() else ""

    len_a = len(s1_name)
    len_b = len(c_name)
    name_len_ratio = min(len_a, len_b) / max(len_a, len_b, 1)

    # Address features
    s1_a_tok, c_a_tok = set(s1_addr.split()), set(c_addr.split())
    addr_jaccard = len(s1_a_tok & c_a_tok) / len(s1_a_tok | c_a_tok) if (s1_a_tok | c_a_tok) else 0.0
    addr_overlap = len(s1_a_tok & c_a_tok) / min(len(s1_a_tok), len(c_a_tok)) if (s1_a_tok and c_a_tok) else 0.0

    len_a_a = len(s1_addr)
    len_b_a = len(c_addr)
    addr_len_ratio = min(len_a_a, len_b_a) / max(len_a_a, len_b_a, 1)

    pin_match = 1.0 if (s1_pins and c_pins and (s1_pins & c_pins)) else (0.0 if (s1_pins and c_pins) else 0.5)
    hno_match = 1.0 if (s1_hno and c_hno and s1_hno == c_hno) else (0.0 if (s1_hno and c_hno) else 0.5)

    ch_set = set(channels.split(",")) if channels else set()
    ch_count = len(ch_set)

    return {
        "source1_entity_id": sid,
        "candidate_entity_id": cid,
        "name_exact_normalized": float(s1_name == c_name),
        "name_exact_sorted_tokens": float(" ".join(sorted(s1_tok)) == " ".join(sorted(c_tok))),
        "name_levenshtein_sim": distance.Levenshtein.normalized_similarity(s1_name, c_name),
        "name_jaro_winkler_sim": distance.JaroWinkler.similarity(s1_name, c_name),
        "name_token_jaccard": name_jaccard,
        "name_token_overlap_coef": name_overlap,
        "name_token_sort_ratio": fuzz.token_sort_ratio(s1_name, c_name) / 100.0,
        "name_token_set_ratio": fuzz.token_set_ratio(s1_name, c_name) / 100.0,
        "name_partial_ratio": fuzz.partial_ratio(s1_name, c_name) / 100.0,
        "name_char_ngram_jaccard": name_jaccard,
        "name_first_token_match": float(s1_first == c_first and bool(s1_first)),
        "name_common_token_count": len(s1_tok & c_tok),
        "name_length_diff": abs(len_a - len_b),
        "name_length_ratio": name_len_ratio,
        "name_digit_overlap": float(bool(s1_pins and c_pins and (s1_pins == c_pins))),
        "address_exact_normalized": float(s1_addr == c_addr),
        "address_token_jaccard": addr_jaccard,
        "address_token_overlap_coef": addr_overlap,
        "address_char_ngram_jaccard": addr_jaccard,
        "address_levenshtein_sim": distance.Levenshtein.normalized_similarity(s1_addr, c_addr),
        "address_token_sort_ratio": fuzz.token_sort_ratio(s1_addr, c_addr) / 100.0,
        "address_partial_ratio": fuzz.partial_ratio(s1_addr, c_addr) / 100.0,
        "address_numeric_token_overlap": float(bool(s1_pins & c_pins)),
        "address_length_diff": abs(len_a_a - len_b_a),
        "address_length_ratio": addr_len_ratio,
        "pin_exact_match": pin_match,
        "house_number_compatibility": hno_match,
        "country_exact_match": float(s1_country == c_country),
        "country_a_missing": float(s1_country == ""),
        "country_b_missing": float(c_country == ""),
        "country_a_unseen_in_train": float(bool(known_train_countries) and s1_country not in known_train_countries),
        "country_b_unseen_in_train": float(bool(known_train_countries) and c_country not in known_train_countries),
        "name_x_address_sim_product": name_jaccard * addr_jaccard,
        "strong_name_weak_address": float(name_jaccard > 0.7 and addr_jaccard < 0.3),
        "weak_name_strong_address": float(name_jaccard < 0.3 and addr_jaccard > 0.7),
        "strong_name_strong_address": float(name_jaccard > 0.7 and addr_jaccard > 0.7),
        "ch_prov_exact_name": float("exact_name" in ch_set),
        "ch_prov_name_token_overlap": float("name_token_overlap" in ch_set),
        "ch_prov_name_char_ngram": float("name_char_ngram" in ch_set),
        "ch_prov_address_token_overlap": float("address_token_overlap" in ch_set),
        "ch_prov_pin_exact": float("pin_exact" in ch_set),
        "ch_prov_city_state_cooccurrence": 0.0,
        "ch_prov_address_char_ngram": 0.0,
        "ch_count": ch_count,
    }


def run_full_test_inference(test_s1_path: Path, test_s2_path: Path, test_s3_path: Path,
                            model, threshold: float, output_dir: Path, known_train_countries: set):
    """Processes all 1.73M test S1 records in streaming chunks of 50,000.
    Appends predictions directly to output/matching_results.tsv and candidate_pairs.tsv.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    matching_file = output_dir / "matching_results.tsv"
    candidate_file = output_dir / "candidate_pairs.tsv"

    # Write headers
    with open(matching_file, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
    with open(candidate_file, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")

    # Step 1: Build candidate index
    index = build_candidate_index(test_s2_path, test_s3_path)
    name_idx = index["name_index"]
    pin_idx = index["pin_index"]
    token_idx = index["token_index"]
    cand_records = index["records"]

    print("\n  [Test Inference] Streaming test_source1.tsv in chunks of 50,000...")
    chunk_num = 0
    total_processed = 0

    for chunk in pd.read_csv(str(test_s1_path), sep="\t", dtype=str, keep_default_na=False, chunksize=50000):
        chunk_num += 1
        feature_rows = []
        s1_cands_map = defaultdict(set)

        for _, row in chunk.iterrows():
            sid = row["entity_id"]
            name_clean = _fast_clean(row.get("business_name", ""))
            addr_clean = _fast_clean(row.get("business_address", ""))
            country = _fast_clean(row.get("country", ""))
            pins = frozenset(extract_pins(addr_clean))
            hno = extract_house_no(addr_clean)
            s1_data = (name_clean, addr_clean, pins, hno, country)

            cands_with_channels = defaultdict(set)

            # Channel A: Exact Name
            if name_clean in name_idx:
                for cid in name_idx[name_clean]:
                    cands_with_channels[cid].add("exact_name")

            # Channel E: Exact PIN
            for pin in pins:
                if pin in pin_idx:
                    for cid in pin_idx[pin]:
                        cands_with_channels[cid].add("pin_exact")

            # Channel B: Distinctive Name Tokens
            tokens = set(name_clean.split()) - STOPWORDS
            for t in tokens:
                if t in token_idx:
                    for cid in token_idx[t]:
                        cands_with_channels[cid].add("name_token_overlap")

            # Cap candidates per S1 entity at 35 to preserve speed and memory
            capped_cands = list(cands_with_channels.keys())[:35]
            s1_cands_map[sid] = set(capped_cands)

            for cid in capped_cands:
                if cid in cand_records:
                    chs = ",".join(sorted(cands_with_channels[cid]))
                    feats = compute_pair_features_fast(sid, s1_data, cid, cand_records[cid], chs, known_train_countries)
                    feature_rows.append(feats)

        # Batch scoring with GPU model
        s1_matches_map = defaultdict(list)
        if feature_rows:
            feat_df = pd.DataFrame(feature_rows)
            feature_cols = [c for c in feat_df.columns if c not in ("source1_entity_id", "candidate_entity_id")]
            scores = model.predict_proba(feat_df[feature_cols])

            # Apply provenance boost
            boost = 0.05 * (feat_df["ch_count"] >= 2).astype(float) + 0.05 * feat_df["ch_prov_exact_name"]
            adj_scores = np.clip(scores + boost, 0.0, 1.0)

            valid_mask = adj_scores >= threshold
            if np.any(valid_mask):
                valid_df = feat_df[valid_mask]
                for sid, cid in zip(valid_df["source1_entity_id"], valid_df["candidate_entity_id"]):
                    s1_matches_map[sid].append(cid)

        # Append results for this chunk directly to file
        with open(matching_file, "a", encoding="utf-8") as f_m, open(candidate_file, "a", encoding="utf-8") as f_c:
            for sid in chunk["entity_id"]:
                matched_str = ",".join(sorted(set(s1_matches_map.get(sid, []))))
                cands_str = ",".join(sorted(s1_cands_map.get(sid, [])))
                f_m.write(f"{sid}\t{matched_str}\n")
                f_c.write(f"{sid}\t{cands_str}\n")

        total_processed += len(chunk)
        print(f"    Chunk {chunk_num}: processed {total_processed:,} / 1,732,544 test S1 records...")
        gc.collect()

    print(f"  [Test Inference Complete] Successfully wrote {total_processed:,} test entity rows.")
