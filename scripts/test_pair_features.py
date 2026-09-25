import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(ROOT_DIR / "src") not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / "src"))

import yaml
from src.business_entity_resolution.data.ingest import load_tsv
from src.business_entity_resolution.blocking.block import generate_candidates
from src.business_entity_resolution.features.pair_features import build_pair_features

def test_pair_features():
    print("Testing pair feature generation on fixtures...")
    s1 = load_tsv(str(ROOT_DIR / "fixtures" / "train" / "sample_source1.tsv"))
    s2 = load_tsv(str(ROOT_DIR / "fixtures" / "train" / "sample_source2.tsv"))
    s3 = load_tsv(str(ROOT_DIR / "fixtures" / "train" / "sample_source3.tsv"))

    with open(ROOT_DIR / "configs" / "pipeline.yaml") as f:
        config = yaml.safe_load(f)

    cands = generate_candidates(s1, s2, s3, config)
    feats = build_pair_features(cands, s1, s2, s3)
    
    print(f"Generated {len(feats)} pair feature rows with {len(feats.columns)} columns.")
    print("Feature columns:", list(feats.columns))
    
    # Check S1-0004 vs S3-0003 (hard negative, house number 47 vs 45) vs S2-0004 (true match)
    hn_sub = feats[(feats["source1_entity_id"] == "S1-0004")]
    print("\nFeatures for S1-0004 pairs:")
    print(hn_sub[["source1_entity_id", "candidate_entity_id", "house_number_compatibility", "name_jaro_winkler_sim", "address_token_jaccard"]])

    print("\n[PASS] Pair feature extraction succeeded!")

if __name__ == "__main__":
    test_pair_features()
