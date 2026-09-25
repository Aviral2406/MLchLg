import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(ROOT_DIR / "src") not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / "src"))

import yaml
from src.business_entity_resolution.data.ingest import load_tsv
from src.business_entity_resolution.blocking.block import generate_candidates, evaluate_blocking_recall

def test_fixture_blocking():
    print("Testing 7-channel blocking on fixtures...")
    s1 = load_tsv(str(ROOT_DIR / "fixtures" / "train" / "sample_source1.tsv"))
    s2 = load_tsv(str(ROOT_DIR / "fixtures" / "train" / "sample_source2.tsv"))
    s3 = load_tsv(str(ROOT_DIR / "fixtures" / "train" / "sample_source3.tsv"))
    gt = load_tsv(str(ROOT_DIR / "fixtures" / "train" / "sample_ground_truth.tsv"))

    with open(ROOT_DIR / "configs" / "pipeline.yaml") as f:
        config = yaml.safe_load(f)

    cands = generate_candidates(s1, s2, s3, config)
    res = evaluate_blocking_recall(cands, gt)
    
    print("Evaluation Results on Fixtures:")
    print(f"  - Recall: {res['recall']:.2%}")
    print(f"  - Total True Pairs: {res['true_pairs_total']}")
    print(f"  - Recovered True Pairs: {res['true_pairs_recovered']}")
    print(f"  - Total Candidates Generated: {res['candidate_pairs_total']}")
    print(f"  - Avg Candidates per S1: {res['avg_candidates_per_s1']:.2f}")

    assert res['recall'] == 1.0, f"Expected 100% recall on fixtures, got {res['recall']}"
    print("\n[PASS] All true fixture pairs successfully recovered by 7-channel blocking!")

if __name__ == "__main__":
    test_fixture_blocking()
