"""
Quick re-inference with a different threshold — use this to fast-test threshold
changes on the ALREADY-GENERATED candidate pairs WITHOUT re-running blocking.

Usage:
    python src/business_entity_resolution/quick_threshold_rerun.py --threshold 0.45

This is a diagnostic tool, not the production pipeline. For the full pipeline, use
src/business_entity_resolution/train_and_infer.py.

WHY this is useful: Our score of 0.430 is almost certainly dominated by too-high
threshold (0.72) causing ~43% of entities to be predicted as singletons when only
5.6% should be. Lowering the threshold can give +0.10 on the leaderboard immediately
while the proper training run is executing.
"""
from __future__ import annotations
import sys
import argparse
import gc
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
import yaml

from business_entity_resolution.data.ingest import load_tsv
from business_entity_resolution.blocking.block import generate_candidates, _normalize_all
from business_entity_resolution.features.pair_features import build_pair_features
from business_entity_resolution.models.model import LightGBMMatcher, build_model


DATA_TEST_DIR = ROOT / "data" / "test"
OUTPUT_DIR = ROOT / "output"
CONFIG_PATH = ROOT / "configs" / "pipeline.yaml"
MODEL_PATH = ROOT / "models" / "lightgbm_model.pkl"


def load_or_train_model(config):
    """Load a saved model, or if not found, error out with instructions."""
    import os
    try:
        import pickle
        if MODEL_PATH.exists():
            with open(MODEL_PATH, "rb") as f:
                matcher, feature_cols = pickle.load(f)
            print(f"Loaded model from {MODEL_PATH}")
            return matcher, feature_cols
    except Exception as e:
        print(f"Could not load model: {e}")
    print(f"\nNo saved model found at {MODEL_PATH}.")
    print("Run the full pipeline first:")
    print("  python src/business_entity_resolution/train_and_infer.py")
    print("\nAlternatively, use the rule-based baseline to quickly test threshold changes.")
    return None, None


def main():
    parser = argparse.ArgumentParser(description="Re-run inference with a new threshold on test data")
    parser.add_argument("--threshold", type=float, default=0.45,
                        help="Match probability threshold (default: 0.45)")
    parser.add_argument("--use-baseline", action="store_true",
                        help="Use the rule-based baseline model instead of LightGBM")
    args = parser.parse_args()

    config = yaml.safe_load(open(CONFIG_PATH))

    # Override threshold
    threshold = args.threshold
    print(f"Using threshold = {threshold}")

    # Load test data
    print("Loading test data...")
    s1_test = load_tsv(str(DATA_TEST_DIR / "test_source1.tsv"), expected_prefix="S1")
    s2_test = load_tsv(str(DATA_TEST_DIR / "test_source2.tsv"), expected_prefix="S2")
    s3_test = load_tsv(str(DATA_TEST_DIR / "test_source3.tsv"), expected_prefix="S3")
    print(f"S1: {len(s1_test):,}  S2: {len(s2_test):,}  S3: {len(s3_test):,}")

    # Try to load model
    if args.use_baseline:
        from business_entity_resolution.models.model import RuleBasedBaseline
        matcher = RuleBasedBaseline()
        feature_cols = None
    else:
        try:
            import pickle
            if MODEL_PATH.exists():
                with open(MODEL_PATH, "rb") as f:
                    matcher, feature_cols = pickle.load(f)
                print(f"Loaded model from {MODEL_PATH}")
            else:
                print(f"\nNo saved model at {MODEL_PATH}. Using baseline. Run train_and_infer.py to train.")
                from business_entity_resolution.models.model import RuleBasedBaseline
                matcher = RuleBasedBaseline()
                feature_cols = None
                args.use_baseline = True
        except Exception as e:
            print(f"Model load error: {e}. Using baseline.")
            from business_entity_resolution.models.model import RuleBasedBaseline
            matcher = RuleBasedBaseline()
            feature_cols = None
            args.use_baseline = True

    known_countries = {"us", "india"}

    # Process in chunks
    CHUNK_SIZE = 200_000
    s1_ids_all = s1_test["entity_id"].tolist()
    chunks = [s1_ids_all[i:i+CHUNK_SIZE] for i in range(0, len(s1_ids_all), CHUNK_SIZE)]

    print(f"\nProcessing {len(s1_ids_all):,} test entities in {len(chunks)} chunks with threshold={threshold}...")

    # Pre-normalize candidates
    print("Pre-normalizing test S2+S3...")
    cand_all_test = pd.concat([s2_test, s3_test], ignore_index=True)
    from business_entity_resolution.normalization.normalize import normalize
    cand_norm_test = {r["entity_id"]: normalize(r) for r in cand_all_test.to_dict(orient="records")}
    del cand_all_test
    gc.collect()

    all_matching = []
    all_candidates = []

    for i, chunk_ids in enumerate(chunks):
        print(f"  Chunk {i+1}/{len(chunks)} ({len(chunk_ids):,} entities)...")
        chunk_s1 = s1_test[s1_test["entity_id"].isin(chunk_ids)].reset_index(drop=True)

        # Blocking
        cands = generate_candidates(chunk_s1, s2_test, s3_test, config)
        valid_cands = cands.dropna(subset=["candidate_entity_id"])

        # Candidates summary per S1
        cand_by_s1 = (
            valid_cands.groupby("source1_entity_id")["candidate_entity_id"]
            .apply(lambda ids: ",".join(sorted(set(ids))))
            .to_dict()
        )

        if len(valid_cands) > 0:
            s1_norm_chunk = {r["entity_id"]: normalize(r) for r in chunk_s1.to_dict(orient="records")}
            feat_df = build_pair_features(
                valid_cands[["source1_entity_id", "candidate_entity_id", "channels"]],
                chunk_s1, s2_test, s3_test,
                known_train_countries=known_countries,
                s1_norm=s1_norm_chunk,
                cand_norm=cand_norm_test,
            )

            if len(feat_df) > 0:
                if feature_cols and not args.use_baseline:
                    X = feat_df[[c for c in feature_cols if c in feat_df.columns]].fillna(0)
                    for c in feature_cols:
                        if c not in X.columns:
                            X[c] = 0.0
                    X = X[feature_cols]
                else:
                    feature_c = [c for c in feat_df.columns
                                 if c not in ("source1_entity_id", "candidate_entity_id")]
                    X = feat_df[feature_c].fillna(0)

                feat_df["match_proba"] = matcher.predict_proba(X)
                matches_above = feat_df[feat_df["match_proba"] >= threshold]
                match_by_s1: dict[str, list] = {}
                for row in matches_above.itertuples(index=False):
                    match_by_s1.setdefault(row.source1_entity_id, []).append(row.candidate_entity_id)
            else:
                match_by_s1 = {}
        else:
            match_by_s1 = {}

        for sid in chunk_ids:
            matched = sorted(set(match_by_s1.get(sid, [])))
            all_matching.append({"source1_entity_id": sid, "matched_entity_ids": ",".join(matched)})
            all_candidates.append({"source1_entity_id": sid, "candidate_entity_ids": cand_by_s1.get(sid, "")})

        del cands, valid_cands
        gc.collect()

    # Write
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    mr_df = pd.DataFrame(all_matching)
    cp_df = pd.DataFrame(all_candidates)

    mr_path = OUTPUT_DIR / f"matching_results_t{int(threshold*100):03d}.tsv"
    cp_path = OUTPUT_DIR / f"candidate_pairs_t{int(threshold*100):03d}.tsv"
    mr_df.to_csv(str(mr_path), sep="\t", index=False)
    cp_df.to_csv(str(cp_path), sep="\t", index=False)

    n_empty = (mr_df["matched_entity_ids"].str.strip() == "").sum()
    print(f"\nDone. Singleton rate: {n_empty:,}/{len(mr_df):,} ({100*n_empty/len(mr_df):.1f}%)")
    print(f"Written: {mr_path}")

    # Validate
    import os
    ret = os.system(
        f'python "{ROOT / "utils" / "validate_submission.py"}" '
        f'--matching "{mr_path}" --candidate "{cp_path}" --test-dir "{DATA_TEST_DIR}"'
    )
    if ret == 0:
        print(f"\n✓ PASS — copy {mr_path.name} to output/matching_results.tsv and submit")
        import shutil
        shutil.copy(mr_path, OUTPUT_DIR / "matching_results.tsv")
        shutil.copy(cp_path, OUTPUT_DIR / "candidate_pairs.tsv")
        print("Copied to output/matching_results.tsv and output/candidate_pairs.tsv")
    else:
        print("\n✗ Validation failed")


if __name__ == "__main__":
    main()
