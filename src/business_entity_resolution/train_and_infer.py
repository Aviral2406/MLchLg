"""
Full end-to-end training + inference pipeline for the Business Entity Resolution challenge.

Usage (from repo root):
    python src/business_entity_resolution/train_and_infer.py

Steps:
  1. Load & validate train data
  2. Grouped entity split (train/val at S1-entity level)
  3. Run multi-channel blocking on train partition -> candidate_pairs
  4. Measure blocking recall on validation partition
  5. Build pairwise features + hard negatives for training
  6. Train LightGBM
  7. Threshold sweep on validation -> pick best threshold (entity-level F0.5)
  8. Run blocking + features on test data
  9. Apply trained model + best threshold
 10. Write output/matching_results.tsv + output/candidate_pairs.tsv
 11. Validate with utils/validate_submission.py
"""
from __future__ import annotations

import sys
import os
import time
import csv
import gc
import random
from pathlib import Path
from datetime import datetime

# Allow running from repo root: python src/business_entity_resolution/train_and_infer.py
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import yaml

from business_entity_resolution.data.ingest import load_tsv, validate_schema
from business_entity_resolution.normalization.normalize import normalize, NormalizedRecord
from business_entity_resolution.blocking.block import generate_candidates, evaluate_blocking_recall, _normalize_all
from business_entity_resolution.features.pair_features import build_pair_features
from business_entity_resolution.models.model import LightGBMMatcher, RuleBasedBaseline, save_model, load_model
from business_entity_resolution.validation.split import grouped_entity_split
from business_entity_resolution.evaluation.metrics import macro_f_beta, parse_match_list, precision_recall_f_beta_breakdown

# ─────────────────────────────── Config ────────────────────────────────────

CONFIG_PATH = ROOT / "configs" / "pipeline.yaml"
DATA_TRAIN_DIR = ROOT / "data" / "train"
DATA_TEST_DIR = ROOT / "data" / "test"
OUTPUT_DIR = ROOT / "output"
EXPERIMENT_LOG = ROOT / "experiments" / "experiment_log.csv"

# Hard negative ratio: for every 1 positive, keep N negatives (including hard negatives)
HARD_NEG_RATIO = 5
# Fraction of negatives that should be "hard" (high sim score, but labeled 0)
HARD_NEG_FRACTION = 0.4

# Threshold sweep range
THRESHOLD_MIN = 0.30
THRESHOLD_MAX = 0.80
THRESHOLD_STEP = 0.02

# Number of S1 entity chunks to process test data in (to control peak RAM)
TEST_CHUNK_SIZE = 200_000  # entities per chunk


def find_dataset_dir() -> tuple[Path, Path]:
    """Finds train and test directories on Kaggle, local data, or fixtures."""
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        train_matches = list(kaggle_input.glob("**/train_source1.tsv"))
        test_matches = list(kaggle_input.glob("**/test_source1.tsv"))
        if train_matches:
            train_dir = train_matches[0].parent
            test_dir = test_matches[0].parent if test_matches else (train_dir.parent / "test")
            print(f"[Environment] Detected Kaggle train dir: {train_dir}")
            print(f"[Environment] Detected Kaggle test dir: {test_dir}")
            return train_dir, test_dir

    local_train = ROOT / "data" / "train"
    local_test = ROOT / "data" / "test"
    if local_train.exists():
        print(f"[Environment] Detected local dataset at {local_train}")
        return local_train, local_test

    fixture_train = ROOT / "fixtures" / "train"
    fixture_test = ROOT / "fixtures" / "test"
    print(f"[Environment] Falling back to fixtures at {fixture_train}")
    return fixture_train, fixture_test


def load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def _fast_normalize_all(df: pd.DataFrame) -> dict[str, NormalizedRecord]:
    """Normalize all records in a DataFrame without iterrows()."""
    return {r["entity_id"]: normalize(r) for r in df.to_dict(orient="records")}


# ─────────────── Step 1: Load train data ─────────────────────────────────

def load_train_data():
    print("\n[1/10] Loading train data...")
    t0 = time.time()
    train_dir, _ = find_dataset_dir()
    s1_files = list(train_dir.glob("*source1.tsv"))
    s2_files = list(train_dir.glob("*source2.tsv"))
    s3_files = list(train_dir.glob("*source3.tsv"))
    gt_files = list(train_dir.glob("*ground_truth.tsv"))

    s1_path = s1_files[0] if s1_files else train_dir / "train_source1.tsv"
    s2_path = s2_files[0] if s2_files else train_dir / "train_source2.tsv"
    s3_path = s3_files[0] if s3_files else train_dir / "train_source3.tsv"
    gt_path = gt_files[0] if gt_files else train_dir / "train_ground_truth.tsv"

    s1 = load_tsv(str(s1_path), expected_prefix="S1")
    s2 = load_tsv(str(s2_path), expected_prefix="S2")
    s3 = load_tsv(str(s3_path), expected_prefix="S3")
    gt = load_tsv(str(gt_path))
    # Rename GT column if needed
    if "source1_entity_id" not in gt.columns and gt.columns[0] != "source1_entity_id":
        gt.columns = ["source1_entity_id", "matched_entity_ids"]
    print(f"  S1={len(s1):,} S2={len(s2):,} S3={len(s3):,} GT={len(gt):,}  [{time.time()-t0:.1f}s]")
    return s1, s2, s3, gt


# ─────────────── Step 2: Grouped entity split ─────────────────────────────

def split_data(s1, gt):
    print("\n[2/10] Splitting S1 entities into train/val (grouped, stratified)...")
    config = load_config()
    val_fraction = config.get("validation", {}).get("val_fraction", 0.2)
    train_ids, val_ids = grouped_entity_split(s1, gt, val_fraction=val_fraction, random_state=42)
    s1_train = s1[s1["entity_id"].isin(train_ids)].reset_index(drop=True)
    s1_val   = s1[s1["entity_id"].isin(val_ids)].reset_index(drop=True)
    print(f"  Train S1: {len(s1_train):,}   Val S1: {len(s1_val):,}")
    return s1_train, s1_val, train_ids, val_ids


# ─────────────── Step 3: Blocking on train ────────────────────────────────

def run_blocking(s1_part, s2, s3, config, label=""):
    print(f"\n[3/10] Running blocking on {label} ({len(s1_part):,} S1 entities)...")
    t0 = time.time()
    cands = generate_candidates(s1_part, s2, s3, config)
    valid = cands.dropna(subset=["candidate_entity_id"])
    n_pairs = len(valid)
    print(f"  {n_pairs:,} candidate pairs  [{time.time()-t0:.1f}s]")
    return cands


# ─────────────── Step 4: Blocking recall on val ───────────────────────────

def measure_blocking_recall(cands_val, gt, val_ids):
    print("\n[4/10] Measuring blocking recall on validation split...")
    gt_val = gt[gt["source1_entity_id"].isin(val_ids)]
    stats = evaluate_blocking_recall(cands_val, gt_val)
    print(f"  Blocking recall: {stats['recall']:.4f}")
    print(f"  True pairs: {stats['true_pairs_total']:,}  Recovered: {stats['true_pairs_recovered']:,}")
    print(f"  Candidate pairs: {stats['candidate_pairs_total']:,}")
    print(f"  Avg candidates per S1: {stats['avg_candidates_per_s1']:.1f}")
    if stats["recall"] < 0.90:
        print("  ⚠️  WARNING: Blocking recall < 0.90 — model cannot compensate for missed pairs!")
    return stats


# ─────────────── Step 5: Features + hard negative mining ──────────────────

def _build_ground_truth_dict(gt: pd.DataFrame) -> dict[str, set[str]]:
    result = {}
    for row in gt.itertuples(index=False):
        sid = row.source1_entity_id
        matched = str(row.matched_entity_ids or "").strip()
        result[sid] = parse_match_list(matched)
    return result


def build_training_features(
    cands_train: pd.DataFrame,
    s1_train: pd.DataFrame,
    s2: pd.DataFrame,
    s3: pd.DataFrame,
    gt: pd.DataFrame,
    train_ids: set,
) -> tuple[pd.DataFrame, np.ndarray]:
    print("\n[5/10] Building pairwise features + hard-negative mining...")
    t0 = time.time()

    known_countries = set(s1_train["country"].str.strip().str.lower().unique())

    # Pre-build norm dicts once
    print("  Normalizing S1 (train)...")
    s1_norm = _fast_normalize_all(s1_train)
    print("  Normalizing S2+S3...")
    cand_all = pd.concat([s2, s3], ignore_index=True)
    cand_norm = _fast_normalize_all(cand_all)
    del cand_all
    gc.collect()

    gt_train_dict = _build_ground_truth_dict(gt[gt["source1_entity_id"].isin(train_ids)])

    # Filter to only pairs with actual candidates
    valid_cands = cands_train.dropna(subset=["candidate_entity_id"]).copy()

    # Label pairs
    print(f"  Labeling {len(valid_cands):,} candidate pairs...")
    def is_positive(row):
        true_set = gt_train_dict.get(row.source1_entity_id, set())
        return int(row.candidate_entity_id in true_set)

    valid_cands["label"] = [
        int(row.candidate_entity_id in gt_train_dict.get(row.source1_entity_id, set()))
        for row in valid_cands.itertuples(index=False)
    ]

    positives = valid_cands[valid_cands["label"] == 1]
    negatives = valid_cands[valid_cands["label"] == 0]
    n_pos = len(positives)
    n_neg_target = n_pos * HARD_NEG_RATIO

    print(f"  Positives: {n_pos:,}  Raw negatives: {len(negatives):,}")

    # Hard negative mining: score all negatives with a cheap proxy,
    # then take the top HARD_NEG_FRACTION as "hard" negatives
    print("  Mining hard negatives (cheap name similarity proxy)...")
    neg_sample_size = min(len(negatives), n_neg_target * 10)  # oversample for mining
    neg_sample = negatives.sample(n=neg_sample_size, random_state=42) if len(negatives) > neg_sample_size else negatives

    # Cheap proxy: char-ngram jaccard on name
    def quick_name_sim(row):
        a = s1_norm.get(row.source1_entity_id)
        b = cand_norm.get(row.candidate_entity_id)
        if a is None or b is None:
            return 0.0
        sa = set(a.name.char_ngrams)
        sb = set(b.name.char_ngrams)
        if not sa and not sb:
            return 0.0
        inter = len(sa & sb)
        return inter / (len(sa | sb)) if sa | sb else 0.0

    neg_sample = neg_sample.copy()
    neg_sample["proxy_sim"] = [
        quick_name_sim(row) for row in neg_sample.itertuples(index=False)
    ]
    neg_sample_sorted = neg_sample.sort_values("proxy_sim", ascending=False)

    n_hard = int(n_neg_target * HARD_NEG_FRACTION)
    n_easy = n_neg_target - n_hard
    hard_negs = neg_sample_sorted.head(n_hard)
    easy_negs = neg_sample_sorted.tail(len(neg_sample_sorted)).sample(
        n=min(n_easy, len(neg_sample_sorted) - n_hard), random_state=42
    )
    selected_negs = pd.concat([hard_negs, easy_negs], ignore_index=True)

    training_pairs = pd.concat([positives, selected_negs], ignore_index=True).sample(frac=1, random_state=42)
    labels = training_pairs["label"].values

    print(f"  Training set: {n_pos:,} positives + {len(selected_negs):,} negatives ({len(selected_negs)//n_pos}:1 ratio)")
    print("  Computing full feature set...")

    features_df = build_pair_features(
        training_pairs[["source1_entity_id", "candidate_entity_id", "channels"]],
        s1_train, s2, s3,
        known_train_countries=known_countries,
        s1_norm=s1_norm,
        cand_norm=cand_norm,
    )
    print(f"  Features built: {features_df.shape}  [{time.time()-t0:.1f}s]")
    return features_df, labels[:len(features_df)]


# ─────────────── Step 6: Train LightGBM ───────────────────────────────────

def train_model(features_df: pd.DataFrame, labels: np.ndarray, config: dict):
    print("\n[6/10] Training LightGBM...")
    t0 = time.time()
    feature_cols = [c for c in features_df.columns
                    if c not in ("source1_entity_id", "candidate_entity_id")]
    X = features_df[feature_cols].fillna(0)
    y = labels

    model_config = config.get("model", {})
    params = model_config.get("params", {})
    params.setdefault("n_estimators", 500)
    params.setdefault("learning_rate", 0.05)
    params.setdefault("num_leaves", 127)
    params.setdefault("max_depth", -1)
    params.setdefault("min_child_samples", 20)
    params.setdefault("subsample", 0.8)
    params.setdefault("colsample_bytree", 0.8)
    params.setdefault("random_state", 42)
    params.setdefault("n_jobs", -1)

    matcher = LightGBMMatcher(**params)
    matcher.fit(X, y)
    print(f"  Training complete  [{time.time()-t0:.1f}s]")

    # Save model so quick_threshold_rerun.py can reload it
    model_save_path = str(ROOT / "models" / "lightgbm_model.pkl")
    save_model(matcher, feature_cols, model_save_path)

    # Feature importance
    imp = matcher.feature_importance()
    print("  Top-10 features:")
    for feat, score in imp.head(10).items():
        print(f"    {feat}: {score}")

    return matcher, feature_cols


# ─────────────── Step 7: Threshold sweep ──────────────────────────────────

def sweep_threshold(
    matcher,
    feature_cols: list,
    cands_val: pd.DataFrame,
    s1_val: pd.DataFrame,
    s2: pd.DataFrame,
    s3: pd.DataFrame,
    gt: pd.DataFrame,
    val_ids: set,
    known_countries: set,
) -> float:
    print("\n[7/10] Threshold sweep on validation set...")
    gt_val_dict = _build_ground_truth_dict(gt[gt["source1_entity_id"].isin(val_ids)])

    # Build val features
    valid_val_cands = cands_val.dropna(subset=["candidate_entity_id"])
    print(f"  Building features for {len(valid_val_cands):,} val candidate pairs...")

    s1_norm = _fast_normalize_all(s1_val)
    cand_all = pd.concat([s2, s3], ignore_index=True)
    cand_norm = _fast_normalize_all(cand_all)
    del cand_all
    gc.collect()

    val_feat = build_pair_features(
        valid_val_cands[["source1_entity_id", "candidate_entity_id", "channels"]],
        s1_val, s2, s3,
        known_train_countries=known_countries,
        s1_norm=s1_norm,
        cand_norm=cand_norm,
    )

    X_val = val_feat[[c for c in feature_cols if c in val_feat.columns]].fillna(0)
    # Add missing feature columns as 0
    for c in feature_cols:
        if c not in X_val.columns:
            X_val[c] = 0.0
    X_val = X_val[feature_cols]

    val_feat["match_proba"] = matcher.predict_proba(X_val)
    val_feat_merged = val_feat[["source1_entity_id", "candidate_entity_id", "match_proba"]].copy()

    best_thresh = 0.5
    best_f05 = -1.0
    print(f"  Sweeping from {THRESHOLD_MIN} to {THRESHOLD_MAX}...")
    print(f"  {'Threshold':>10} {'F0.5':>8} {'Singleton acc':>14} {'Avg matches':>12}")
    for t in np.arange(THRESHOLD_MIN, THRESHOLD_MAX + 1e-9, THRESHOLD_STEP):
        t = round(t, 3)
        matches_above = val_feat_merged[val_feat_merged["match_proba"] >= t]
        pred_dict: dict[str, set] = {}
        for row in matches_above.itertuples(index=False):
            pred_dict.setdefault(row.source1_entity_id, set()).add(row.candidate_entity_id)
        # Ensure all val S1 entities appear (those with no candidates -> empty pred)
        for sid in val_ids:
            if sid not in pred_dict:
                pred_dict[sid] = set()
        breakdown = precision_recall_f_beta_breakdown(gt_val_dict, pred_dict, beta=0.5)
        f05 = breakdown["macro_f_beta"]
        sing_acc = breakdown.get("singleton_accuracy") or 0.0
        avg_m = breakdown.get("avg_predicted_matches_per_entity") or 0.0
        print(f"  {t:>10.3f} {f05:>8.4f} {sing_acc:>14.4f} {avg_m:>12.2f}")
        if f05 > best_f05:
            best_f05 = f05
            best_thresh = t

    print(f"\n  ✓ Best threshold: {best_thresh}  (F₀.₅ = {best_f05:.4f})")
    return best_thresh, best_f05


# ─────────────── Steps 8–9: Test inference ────────────────────────────────

def run_test_inference(matcher, feature_cols: list, config: dict, threshold: float):
    print("\n[8/10] Loading test data...")
    t0 = time.time()
    _, test_dir = find_dataset_dir()
    s1_files = list(test_dir.glob("*source1.tsv"))
    s2_files = list(test_dir.glob("*source2.tsv"))
    s3_files = list(test_dir.glob("*source3.tsv"))
    s1_path = s1_files[0] if s1_files else test_dir / "test_source1.tsv"
    s2_path = s2_files[0] if s2_files else test_dir / "test_source2.tsv"
    s3_path = s3_files[0] if s3_files else test_dir / "test_source3.tsv"

    s1_test = load_tsv(str(s1_path), expected_prefix="S1")
    s2_test = load_tsv(str(s2_path), expected_prefix="S2")
    s3_test = load_tsv(str(s3_path), expected_prefix="S3")
    print(f"  Loaded: S1={len(s1_test):,} S2={len(s2_test):,} S3={len(s3_test):,}  [{time.time()-t0:.1f}s]")

    # Pre-normalize test candidates once — reuse across chunks
    print("  Normalizing test S2+S3 (once, reused across chunks)...")
    cand_all_test = pd.concat([s2_test, s3_test], ignore_index=True)
    cand_norm_test = _fast_normalize_all(cand_all_test)
    del cand_all_test
    gc.collect()

    s1_ids_all = s1_test["entity_id"].tolist()
    known_countries = {"us", "india"}  # only from train — France will trigger unseen flag

    all_matching_rows = []
    all_candidate_rows = []

    s1_chunks = [s1_ids_all[i:i+TEST_CHUNK_SIZE] for i in range(0, len(s1_ids_all), TEST_CHUNK_SIZE)]
    print(f"\n[9/10] Inference on {len(s1_ids_all):,} test S1 entities in {len(s1_chunks)} chunks...")

    for chunk_idx, chunk_ids in enumerate(s1_chunks):
        chunk_s1 = s1_test[s1_test["entity_id"].isin(chunk_ids)].reset_index(drop=True)
        print(f"  Chunk {chunk_idx+1}/{len(s1_chunks)}: {len(chunk_s1):,} S1 entities")

        # Blocking
        cands = generate_candidates(chunk_s1, s2_test, s3_test, config)
        valid_cands = cands.dropna(subset=["candidate_entity_id"])

        if len(valid_cands) == 0:
            # All singletons in this chunk
            for sid in chunk_ids:
                all_matching_rows.append({"source1_entity_id": sid, "matched_entity_ids": ""})
                all_candidate_rows.append({"source1_entity_id": sid, "candidate_entity_ids": ""})
            continue

        # Features
        s1_norm_chunk = _fast_normalize_all(chunk_s1)
        feat_df = build_pair_features(
            valid_cands[["source1_entity_id", "candidate_entity_id", "channels"]],
            chunk_s1, s2_test, s3_test,
            known_train_countries=known_countries,
            s1_norm=s1_norm_chunk,
            cand_norm=cand_norm_test,
        )
        if len(feat_df) == 0:
            for sid in chunk_ids:
                all_matching_rows.append({"source1_entity_id": sid, "matched_entity_ids": ""})
                all_candidate_rows.append({"source1_entity_id": sid, "candidate_entity_ids": ""})
            continue

        X = feat_df[[c for c in feature_cols if c in feat_df.columns]].fillna(0)
        for c in feature_cols:
            if c not in X.columns:
                X[c] = 0.0
        X = X[feature_cols]
        feat_df["match_proba"] = matcher.predict_proba(X)

        # Aggregate candidates per S1
        cand_by_s1 = (
            valid_cands.groupby("source1_entity_id")["candidate_entity_id"]
            .apply(lambda ids: ",".join(sorted(set(ids))))
            .to_dict()
        )

        # Apply threshold -> matches
        matches_above = feat_df[feat_df["match_proba"] >= threshold]
        match_by_s1: dict[str, list] = {}
        for row in matches_above.itertuples(index=False):
            match_by_s1.setdefault(row.source1_entity_id, []).append(row.candidate_entity_id)

        for sid in chunk_ids:
            matched_ids = sorted(set(match_by_s1.get(sid, [])))
            all_matching_rows.append({
                "source1_entity_id": sid,
                "matched_entity_ids": ",".join(matched_ids),
            })
            all_candidate_rows.append({
                "source1_entity_id": sid,
                "candidate_entity_ids": cand_by_s1.get(sid, ""),
            })

        del feat_df, cands, valid_cands, s1_norm_chunk
        gc.collect()

    matching_results = pd.DataFrame(all_matching_rows)
    candidate_pairs = pd.DataFrame(all_candidate_rows)
    return matching_results, candidate_pairs


# ─────────────── Step 10: Write outputs + validate ────────────────────────

def write_and_validate(matching_results: pd.DataFrame, candidate_pairs: pd.DataFrame):
    print("\n[10/10] Writing outputs...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    matching_path = OUTPUT_DIR / "matching_results.tsv"
    candidate_path = OUTPUT_DIR / "candidate_pairs.tsv"
    matching_results.to_csv(str(matching_path), sep="\t", index=False)
    candidate_pairs.to_csv(str(candidate_path), sep="\t", index=False)
    print(f"  Written: {matching_path}")
    print(f"  Written: {candidate_path}")

    n_empty = (matching_results["matched_entity_ids"].str.strip() == "").sum()
    print(f"  Singleton predictions: {n_empty:,} / {len(matching_results):,} ({100*n_empty/len(matching_results):.1f}%)")

    # Package output files into /kaggle/working/submission.zip for 1-click download
    kaggle_working = Path("/kaggle/working")
    import shutil
    zip_target = kaggle_working / "submission" if kaggle_working.exists() else OUTPUT_DIR / "submission"
    shutil.make_archive(str(zip_target), "zip", str(OUTPUT_DIR))
    print(f"\n  [Package] Created submission zip package at {zip_target}.zip!")

    print("\n  Running official validator...")
    _, test_dir = find_dataset_dir()
    validator_path = ROOT / "utils" / "validate_submission.py"
    target_test_dir = test_dir if (test_dir / "test_source1.tsv").exists() else _
    ret = os.system(
        f'python "{validator_path}" --matching "{matching_path}" --candidate "{candidate_path}" --test-dir "{target_test_dir}"'
    )
    if ret == 0:
        print("  ✓ Validation PASSED")
    else:
        print("  ✗ Validation FAILED — check output above and fix before submitting")
    return ret == 0


# ─────────────── Experiment logging ───────────────────────────────────────

def log_experiment(
    blocking_recall: float,
    val_f05: float,
    best_threshold: float,
    n_train_pairs: int,
    notes: str = "",
):
    config = load_config()
    row = {
        "exp_id": f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "date": datetime.now().isoformat(),
        "blocking_channels": ",".join(
            k for k, v in config.get("blocking", {}).get("channels", {}).items() if v
        ),
        "features": "name+address+country+cross+provenance",
        "neg_sampling": f"hard_neg_{HARD_NEG_FRACTION}_ratio_{HARD_NEG_RATIO}",
        "model": "lightgbm",
        "hyperparams": str(config.get("model", {}).get("params", {})),
        "cv_strategy": "grouped_entity_split",
        "threshold": best_threshold,
        "blocking_recall": round(blocking_recall, 4),
        "pairwise_precision": "",
        "pairwise_recall": "",
        "entity_precision": "",
        "entity_recall": "",
        "entity_f0.5": round(val_f05, 4),
        "singleton_f0.5": "",
        "avg_candidates_per_s1": "",
        "train_time_sec": "",
        "infer_time_sec": "",
        "git_commit": "",
        "notes": notes,
    }
    header = list(row.keys())
    log_exists = EXPERIMENT_LOG.exists() and EXPERIMENT_LOG.stat().st_size > 50
    with open(EXPERIMENT_LOG, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if not log_exists:
            writer.writeheader()
        writer.writerow(row)
    print(f"\n  Logged to {EXPERIMENT_LOG}")


# ─────────────── Main ─────────────────────────────────────────────────────

def main():
    config = load_config()

    # 1. Load train data
    s1, s2, s3, gt = load_train_data()

    # 2. Split
    s1_train, s1_val, train_ids, val_ids = split_data(s1, gt)

    # 3. Blocking on train partition
    cands_train = run_blocking(s1_train, s2, s3, config, "train partition")

    # 3b. Blocking on val partition (needed for threshold sweep)
    cands_val = run_blocking(s1_val, s2, s3, config, "val partition")

    # 4. Blocking recall
    blocking_stats = measure_blocking_recall(cands_val, gt, val_ids)

    # 5. Build training features + hard negatives
    train_feat_df, train_labels = build_training_features(
        cands_train, s1_train, s2, s3, gt, train_ids
    )

    # 6. Train
    known_countries = set(s1["country"].str.strip().str.lower().unique())
    matcher, feature_cols = train_model(train_feat_df, train_labels, config)
    del train_feat_df
    gc.collect()

    # 7. Threshold sweep
    best_thresh, best_f05 = sweep_threshold(
        matcher, feature_cols, cands_val, s1_val, s2, s3, gt, val_ids, known_countries
    )
    del cands_train, cands_val, s1_train, s1_val, s1
    gc.collect()

    # Update pipeline.yaml threshold
    config["threshold"]["value"] = float(best_thresh)
    with open(CONFIG_PATH, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"  Updated configs/pipeline.yaml threshold -> {best_thresh}")

    # 8-9. Test inference
    matching_results, candidate_pairs = run_test_inference(matcher, feature_cols, config, best_thresh)

    # 10. Write + validate
    passed = write_and_validate(matching_results, candidate_pairs)

    # Log experiment
    log_experiment(
        blocking_recall=blocking_stats["recall"],
        val_f05=best_f05,
        best_threshold=best_thresh,
        n_train_pairs=len(matching_results),
        notes="full_pipeline_run",
    )

    print("\n" + "="*60)
    print(f"DONE. Val F₀.₅ = {best_f05:.4f}   Threshold = {best_thresh}")
    if passed:
        print("Submit: output/matching_results.tsv")
    else:
        print("Validation FAILED — fix before submitting")
    print("="*60)


if __name__ == "__main__":
    main()
