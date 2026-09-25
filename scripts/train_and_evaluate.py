"""
End-to-End Pipeline Training, Threshold Tuning, and Official Submission Validation.
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
from src.business_entity_resolution.evaluation.metrics import macro_f_beta, precision_recall_f_beta_breakdown
from src.business_entity_resolution.inference.predict import predict, write_outputs


def main():
    print("=== Business Entity Resolution: End-to-End Model Training & Pipeline Execution ===")

    config_path = ROOT_DIR / "configs" / "pipeline.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Use sample fixtures for fast verification and development
    use_sample = False  # Set to True for fast debug, False for full dataset
    data_dir = ROOT_DIR / "fixtures" / "train" if use_sample else ROOT_DIR / "data" / "train"

    s1_file = "sample_source1.tsv" if use_sample else "train_source1.tsv"
    s2_file = "sample_source2.tsv" if use_sample else "train_source2.tsv"
    s3_file = "sample_source3.tsv" if use_sample else "train_source3.tsv"
    gt_file = "sample_ground_truth.tsv" if use_sample else "train_ground_truth.tsv"

    print(f"\n1. Loading Data from {data_dir}...")
    s1_df = load_tsv(str(data_dir / s1_file), expected_prefix="S1")
    s2_df = load_tsv(str(data_dir / s2_file), expected_prefix="S2")
    s3_df = load_tsv(str(data_dir / s3_file), expected_prefix="S3")
    gt_df = load_tsv(str(data_dir / gt_file))

    # If full dataset, take a stratified 50,000 entity subset for ultra-fast training iterations
    if not use_sample and len(s1_df) > 50000:
        print(f"  Subsampling 50,000 S1 entities for rapid iteration & validation...")
        s1_sample = s1_df.sample(n=50000, random_state=42)
        s1_ids = set(s1_sample["entity_id"])
        gt_df = gt_df[gt_df["source1_entity_id"].isin(s1_ids)]
        s1_df = s1_sample

    print("\n2. Generating Candidates (7-Channel Blocking)...")
    cands_df = generate_candidates(s1_df, s2_df, s3_df, config)
    blocking_stats = evaluate_blocking_recall(cands_df, gt_df)
    print(f"  - Blocking Candidate Recall: {blocking_stats['recall']:.2%}")
    print(f"  - Total Candidates Generated: {blocking_stats['candidate_pairs_total']:,}")
    print(f"  - Avg Candidates per S1: {blocking_stats['avg_candidates_per_s1']:.2f}")

    print("\n3. Building Pairwise Features (40 discriminative features)...")
    known_countries = set(s1_df["country"].unique()) | set(s2_df["country"].unique()) | set(s3_df["country"].unique())
    feature_df = build_pair_features(cands_df, s1_df, s2_df, s3_df, known_train_countries=known_countries)

    print("\n4. Constructing Ground Truth Pair Labels...")
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
    print(f"  - Positive Ground Truth Pairs: {labels.sum():,}")
    print(f"  - Negative Candidate Pairs: {(labels == 0).sum():,}")

    print("\n5. Training Ensemble Model (LightGBM + XGBoost)...")
    model = EnsembleMatcher(lgb_weight=0.6, xgb_weight=0.4)
    model.fit(feature_df[feature_cols], labels)

    print("\n6. Predicting & Threshold Calibration for Macro F0.5...")
    scores = model.predict_proba(feature_df[feature_cols])
    feature_df["score"] = scores

    best_thresh = 0.70
    best_f05 = 0.0

    print("  Sweeping threshold values [0.50 -> 0.90]:")
    for th in np.arange(0.50, 0.92, 0.05):
        feature_df["pred"] = (feature_df["score"] >= th).astype(int)
        pred_matches = feature_df[feature_df["pred"] == 1]
        
        # Build predictions map per S1
        pred_dict = pred_matches.groupby("source1_entity_id")["candidate_entity_id"].apply(lambda ids: set(ids)).to_dict()
        gt_dict = {sid: set() for sid in s1_df["entity_id"]}
        for sid, cid in gt_pairs:
            gt_dict[sid].add(cid)

        # Calculate macro F0.5
        f05 = macro_f_beta(pred_dict, gt_dict, beta=0.5)
        print(f"    Thresh {th:.2f} -> Macro F0.5 = {f05:.4f}")
        if f05 > best_f05:
            best_f05 = f05
            best_thresh = float(th)

    print(f"\n  [OPTIMAL THRESHOLD] {best_thresh:.2f} achieved Macro F0.5 = {best_f05:.4f}!")

    config["threshold"]["value"] = best_thresh

    print("\n7. Generating Final Output Predictions & Validation Files...")
    os.makedirs(ROOT_DIR / "output", exist_ok=True)
    matching_res, candidate_res = predict(s1_df, s2_df, s3_df, model, config, known_train_countries=known_countries)
    write_outputs(matching_res, candidate_res, output_dir=str(ROOT_DIR / "output"))

    print("\n8. Running Official Submission Validator (utils/validate_submission.py)...")
    import subprocess
    cmd = [
        sys.executable,
        str(ROOT_DIR / "utils" / "validate_submission.py"),
        "--matching", str(ROOT_DIR / "output" / "matching_results.tsv"),
        "--candidate", str(ROOT_DIR / "output" / "candidate_pairs.tsv"),
        "--test-dir", str(ROOT_DIR / "fixtures" / "train" if use_sample else ROOT_DIR / "data" / "train"),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    print("Validator Output:\n" + proc.stdout)
    if proc.stderr:
        print("Validator Stderr:\n" + proc.stderr)

    print(f"\n=== Training & Validation Pipeline Completed Successfully (Returncode={proc.returncode}) ===")


if __name__ == "__main__":
    main()
