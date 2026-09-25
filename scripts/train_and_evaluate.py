"""
End-to-End Pipeline Training, Threshold Tuning, and Official Submission Validation.
Auto-detects Kaggle GPU environment and uses CUDA acceleration.
"""
from __future__ import annotations
import os
import sys
import yaml
from pathlib import Path

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
from src.business_entity_resolution.models.model import build_model, EnsembleMatcher
from src.business_entity_resolution.evaluation.metrics import macro_f_beta
from src.business_entity_resolution.inference.predict import predict, write_outputs


def find_dataset_dir() -> tuple[Path, Path]:
    """Finds train and test directories on Kaggle or local workspace."""
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        for s1_file in kaggle_input.glob("**/train_source1.tsv"):
            train_dir = s1_file.parent
            test_dir = train_dir.parent / "test" if (train_dir.parent / "test").exists() else train_dir
            print(f"[Environment] Detected Kaggle dataset at {train_dir}")
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


def main():
    print("=== Business Entity Resolution: State-of-the-Art Training Pipeline ===")

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

    print(f"\n1. Loading Training Data...")
    s1_df = load_tsv(str(s1_path), expected_prefix="S1")
    s2_df = load_tsv(str(s2_path), expected_prefix="S2")
    s3_df = load_tsv(str(s3_path), expected_prefix="S3")
    gt_df = load_tsv(str(gt_path))

    # For fast training iterations, sample 50,000 S1 records if dataset is massive
    is_sample = False
    if len(s1_df) > 50000:
        print(f"  Training on 50,000 stratified entities for optimal GPU throughput...")
        s1_sample = s1_df.sample(n=50000, random_state=42)
        s1_ids = set(s1_sample["entity_id"])
        gt_df = gt_df[gt_df["source1_entity_id"].isin(s1_ids)]
        s1_df = s1_sample
        is_sample = True

    print("\n2. Candidate Generation (7-Channel Multi-Index Blocking)...")
    cands_df = generate_candidates(s1_df, s2_df, s3_df, config)
    blocking_stats = evaluate_blocking_recall(cands_df, gt_df)
    print(f"  - Blocking Candidate Recall: {blocking_stats['recall']:.2%}")
    print(f"  - Total Candidates Generated: {blocking_stats['candidate_pairs_total']:,}")
    print(f"  - Avg Candidates per S1: {blocking_stats['avg_candidates_per_s1']:.2f}")

    print("\n3. Generating 46 Engineered Pairwise Features...")
    known_countries = set(s1_df["country"].unique()) | set(s2_df["country"].unique()) | set(s3_df["country"].unique())
    feature_df = build_pair_features(cands_df, s1_df, s2_df, s3_df, known_train_countries=known_countries)

    print("\n4. Constructing Ground Truth Labels & Hard Negatives...")
    gt_pairs = set()
    for _, row in gt_df.iterrows():
        mids = str(row["matched_entity_ids"] or "")
        if mids and mids != "nan":
            for m in mids.split(","):
                if m.strip():
                    gt_pairs.add((str(row["source1_entity_id"]), m.strip()))

    labels = np.array([
        1 if (sid, cid) in gt_pairs else 0
        for sid, cid in zip(feature_df["source1_entity_id"], feature_df["candidate_entity_id"])
    ])
    feature_cols = [c for c in feature_df.columns if c not in ("source1_entity_id", "candidate_entity_id")]

    print(f"  - Total Labeled Pairs: {len(labels):,}")
    print(f"  - Positive Match Pairs: {labels.sum():,}")
    print(f"  - Hard Negative Pairs: {(labels == 0).sum():,}")

    print("\n5. Training Ensemble Model (LightGBM + XGBoost with CUDA)...")
    model = EnsembleMatcher(lgb_weight=0.6, xgb_weight=0.4)
    model.fit(feature_df[feature_cols], labels)

    print("\n6. Calibrating Decision Threshold Specifically for Macro F0.5...")
    scores = model.predict_proba(feature_df[feature_cols])
    feature_df["score"] = scores

    best_thresh = 0.70
    best_f05 = 0.0

    gt_dict = {sid: set() for sid in s1_df["entity_id"]}
    for sid, cid in gt_pairs:
        gt_dict[sid].add(cid)

    for th in np.arange(0.50, 0.95, 0.05):
        valid_sub = feature_df[feature_df["score"] >= th].sort_values(by="score", ascending=False)
        
        # Apply 1-to-1 greedy selection
        pred_dict = {sid: set() for sid in s1_df["entity_id"]}
        assigned_cands = set()
        for _, r in valid_sub.iterrows():
            sid = r["source1_entity_id"]
            cid = r["candidate_entity_id"]
            if not pred_dict[sid] and cid not in assigned_cands:
                pred_dict[sid].add(cid)
                assigned_cands.add(cid)

        f05 = macro_f_beta(pred_dict, gt_dict, beta=0.5)
        print(f"    Thresh {th:.2f} -> Macro F0.5 = {f05:.4f}")
        if f05 > best_f05:
            best_f05 = f05
            best_thresh = float(th)

    print(f"\n  >>> Optimal Threshold: {best_thresh:.2f} | Achieved Macro F0.5: {best_f05:.4f} <<<")
    config["threshold"]["value"] = best_thresh

    print("\n7. Generating Inference Submission Files...")
    output_dir = ROOT_DIR / "output"
    os.makedirs(output_dir, exist_ok=True)
    matching_res, candidate_res = predict(s1_df, s2_df, s3_df, model, config, known_train_countries=known_countries)
    write_outputs(matching_res, candidate_res, output_dir=str(output_dir))

    print("\n8. Validating Official Submission Integrity...")
    import subprocess
    cmd = [
        sys.executable,
        str(ROOT_DIR / "utils" / "validate_submission.py"),
        "--matching", str(output_dir / "matching_results.tsv"),
        "--candidate", str(output_dir / "candidate_pairs.tsv"),
        "--test-dir", str(train_dir if is_sample else test_dir),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    print(proc.stdout)
    if proc.stderr:
        print(proc.stderr)

    print(f"\n=== Full Pipeline Execution Completed with Return Code {proc.returncode} ===")


if __name__ == "__main__":
    main()
