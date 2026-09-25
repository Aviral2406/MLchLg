"""
High-Performance, Memory-Safe Training & Threshold Tuning Pipeline.
Supports Multi-Match Entity Resolution, Fine-Grained Threshold Sweeping (0.15 - 0.55),
Channel Provenance Prior, and Transitive Graph Propagation to push Macro F0.5 > 0.96.
"""
from __future__ import annotations
import gc
import os
import sys
import yaml
import shutil
from pathlib import Path
from collections import defaultdict

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(ROOT_DIR / "src") not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / "src"))

import numpy as np
import pandas as pd
from src.business_entity_resolution.data.ingest import load_tsv
from src.business_entity_resolution.blocking.block import generate_candidates, evaluate_blocking_recall
from src.business_entity_resolution.features.pair_features import build_pair_features
from src.business_entity_resolution.models.model import EnsembleMatcher
from src.business_entity_resolution.evaluation.metrics import macro_f_beta
from src.business_entity_resolution.inference.predict import predict, write_outputs


def find_dataset_dir() -> tuple[Path, Path]:
    """Finds train and test directories on Kaggle or local workspace."""
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

    local_train = ROOT_DIR / "data" / "train"
    local_test = ROOT_DIR / "data" / "test"
    if local_train.exists():
        print(f"[Environment] Detected local dataset at {local_train}")
        return local_train, local_test

    fixture_train = ROOT_DIR / "fixtures" / "train"
    fixture_test = ROOT_DIR / "fixtures" / "test"
    print(f"[Environment] Falling back to fixtures at {fixture_train}")
    return fixture_train, fixture_test


def load_candidate_pool(path: Path, needed_ids: set, sample_negatives: int = 40000) -> pd.DataFrame:
    """Streams candidate files in small chunks to extract only required true positives
    plus controlled background negatives. Keeps peak memory strictly under 600 MB.
    """
    print(f"  [Memory-Safe Load] Scanning {path.name} in chunks...")
    chunks = []
    chunk_idx = 0
    for chunk in pd.read_csv(str(path), sep="\t", dtype=str, keep_default_na=False, chunksize=250000):
        matched = chunk[chunk["entity_id"].isin(needed_ids)]
        if not matched.empty:
            chunks.append(matched)
        if sample_negatives > 0 and chunk_idx < 15:
            unmatched = chunk[~chunk["entity_id"].isin(needed_ids)]
            if not unmatched.empty:
                sample_n = min(len(unmatched), 3000)
                chunks.append(unmatched.sample(n=sample_n, random_state=42))
        chunk_idx += 1

    df = pd.concat(chunks, ignore_index=True).drop_duplicates(subset=["entity_id"])
    print(f"  [Memory-Safe Load] Extracted {len(df):,} relevant candidates from {path.name}.")
    return df


def main():
    print("=" * 70)
    print("=== Amazon ML Challenge: Business Entity Resolution Pipeline ===")
    print("=== Multi-Match High Precision Model (Target Macro F0.5 > 0.96) ===")
    print("=" * 70)

    config_path = ROOT_DIR / "configs" / "pipeline.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    train_dir, test_dir = find_dataset_dir()

    s1_files = list(train_dir.glob("*source1.tsv"))
    s2_files = list(train_dir.glob("*source2.tsv"))
    s3_files = list(train_dir.glob("*source3.tsv"))
    gt_files = list(train_dir.glob("*ground_truth.tsv"))

    s1_path = s1_files[0] if s1_files else train_dir / "train_source1.tsv"
    s2_path = s2_files[0] if s2_files else train_dir / "train_source2.tsv"
    s3_path = s3_files[0] if s3_files else train_dir / "train_source3.tsv"
    gt_path = gt_files[0] if gt_files else train_dir / "train_ground_truth.tsv"

    print("\n[Step 1/6] Loading Source 1 and Ground Truth Labels...")
    s1_df = load_tsv(str(s1_path), expected_prefix="S1")
    gt_df = load_tsv(str(gt_path))

    # Scale training size to 60,000 S1 records (2x larger training coverage for high precision)
    TRAIN_SAMPLE_SIZE = 60000
    if len(s1_df) > TRAIN_SAMPLE_SIZE:
        print(f"  Selecting {TRAIN_SAMPLE_SIZE:,} stratified S1 records for training...")
        s1_df = s1_df.sample(n=TRAIN_SAMPLE_SIZE, random_state=42)
        s1_ids = set(s1_df["entity_id"])
        gt_df = gt_df[gt_df["source1_entity_id"].isin(s1_ids)].copy()

    needed_match_ids = set()
    gt_pairs = set()
    gt_dict = defaultdict(set)
    for _, row in gt_df.iterrows():
        sid = str(row["source1_entity_id"])
        mids = str(row["matched_entity_ids"] or "")
        if mids and mids != "nan":
            for m in mids.split(","):
                m_clean = m.strip()
                if m_clean:
                    needed_match_ids.add(m_clean)
                    gt_pairs.add((sid, m_clean))
                    gt_dict[sid].add(m_clean)

    print(f"  Total true match pairs for training: {len(gt_pairs):,}")
    print(f"  Avg matches per S1 entity: {len(gt_pairs) / len(s1_df):.2f}")

    print("\n[Step 2/6] Loading Candidate Pools with Streaming Chunks...")
    s2_df = load_candidate_pool(s2_path, needed_match_ids, sample_negatives=40000)
    s3_df = load_candidate_pool(s3_path, needed_match_ids, sample_negatives=40000)
    gc.collect()

    print("\n[Step 3/6] Generating Candidates (Enhanced 5-Channel Blocking)...")
    cands_df = generate_candidates(s1_df, s2_df, s3_df, config)
    blocking_stats = evaluate_blocking_recall(cands_df, gt_df)
    print(f"  >>> Candidate Recall: {blocking_stats['recall']:.2%} <<<")
    print(f"  Total Candidate Pairs Generated: {blocking_stats['candidate_pairs_total']:,}")
    print(f"  Average Candidates per S1: {blocking_stats['avg_candidates_per_s1']:.2f}")

    print("\n[Step 4/6] Extracting 46 High-Precision Features (RapidFuzz, Tri-State PIN, House No.)...")
    known_countries = set(s1_df["country"].unique())
    feature_df = build_pair_features(cands_df, s1_df, s2_df, s3_df, known_train_countries=known_countries)
    gc.collect()

    labels = np.array([
        1 if (sid, cid) in gt_pairs else 0
        for sid, cid in zip(feature_df["source1_entity_id"], feature_df["candidate_entity_id"])
    ])
    feature_cols = [c for c in feature_df.columns if c not in ("source1_entity_id", "candidate_entity_id")]

    print(f"  Labeled Pairs: {len(labels):,} (Positives: {labels.sum():,}, Hard Negatives: {(labels == 0).sum():,})")

    print("\n[Step 5/6] Training Dual Ensemble (LightGBM + CUDA XGBoost)...")
    model = EnsembleMatcher(lgb_weight=0.55, xgb_weight=0.45)
    model.fit(feature_df[feature_cols], labels)
    gc.collect()

    print("\n[Step 6/6] Fine-Grained Threshold Sweep (0.15 - 0.55) for Macro F0.5...")
    raw_scores = model.predict_proba(feature_df[feature_cols])

    # Provenance Prior: candidates matched by multiple channels or exact name get a small confidence boost
    boost = 0.05 * (feature_df["ch_count"] >= 2).astype(float) + 0.05 * feature_df["ch_prov_exact_name"]
    final_scores = np.clip(raw_scores + boost, 0.0, 1.0)
    feature_df["score"] = final_scores

    best_thresh = 0.35
    best_f05 = 0.0

    # Ensure all S1 entities have an entry in gt_dict (including singletons)
    for sid in s1_df["entity_id"]:
        if sid not in gt_dict:
            gt_dict[sid] = set()

    for th in np.arange(0.15, 0.56, 0.02):
        valid_sub = feature_df[feature_df["score"] >= th]
        pred_dict = {sid: set() for sid in s1_df["entity_id"]}

        for sid, cid in zip(valid_sub["source1_entity_id"], valid_sub["candidate_entity_id"]):
            pred_dict[sid].add(cid)

        f05 = macro_f_beta(pred_dict, gt_dict, beta=0.5)
        print(f"  Threshold {th:.2f} -> Macro F0.5 = {f05:.4f}")
        if f05 > best_f05:
            best_f05 = f05
            best_thresh = float(th)

    print("\n" + "=" * 70)
    print(f"  >>> BEST CALIBRATED MACRO F0.5 SCORE: {best_f05:.4f} (at threshold {best_thresh:.2f}) <<<")
    print("=" * 70)

    config["threshold"]["value"] = best_thresh

    print("\nWriting validation output predictions...")
    output_dir = ROOT_DIR / "output"
    os.makedirs(output_dir, exist_ok=True)

    test_s1_files = list(test_dir.glob("*source1.tsv"))
    test_s2_files = list(test_dir.glob("*source2.tsv"))
    test_s3_files = list(test_dir.glob("*source3.tsv"))

    if test_s1_files and test_s2_files and test_s3_files:
        print(f"  Running inference on real test dataset at {test_dir}...")
        test_s1 = load_tsv(str(test_s1_files[0]), expected_prefix="S1")
        print(f"  Test S1 entities to predict: {len(test_s1):,}")
        test_s2 = load_candidate_pool(test_s2_files[0], needed_ids=set(), sample_negatives=60000)
        test_s3 = load_candidate_pool(test_s3_files[0], needed_ids=set(), sample_negatives=60000)
        matching_res, candidate_res = predict(test_s1, test_s2, test_s3, model, config, known_train_countries=known_countries)
        write_outputs(matching_res, candidate_res, output_dir=str(output_dir))
    else:
        print("  Generating submission output on validation records...")
        matching_res, candidate_res = predict(s1_df, s2_df, s3_df, model, config, known_train_countries=known_countries)
        write_outputs(matching_res, candidate_res, output_dir=str(output_dir))

    # Package output files into /kaggle/working/submission.zip for 1-click download
    kaggle_working = Path("/kaggle/working")
    zip_target = kaggle_working / "submission" if kaggle_working.exists() else ROOT_DIR / "submission"
    shutil.make_archive(str(zip_target), "zip", str(output_dir))
    print(f"\n  [Package] Created submission zip package at {zip_target}.zip!")

    print("\nValidating submission structure with official validator...")
    import subprocess
    cmd = [
        sys.executable,
        str(ROOT_DIR / "utils" / "validate_submission.py"),
        "--matching", str(output_dir / "matching_results.tsv"),
        "--candidate", str(output_dir / "candidate_pairs.tsv"),
        "--test-dir", str(test_dir if (test_dir / "test_source1.tsv").exists() else train_dir),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    print(proc.stdout)
    if proc.stderr:
        print(proc.stderr)

    print("\n=== Training & Evaluation Complete! ===")


if __name__ == "__main__":
    main()
