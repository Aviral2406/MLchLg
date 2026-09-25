import pandas as pd

from src.business_entity_resolution.models.model import mine_hard_negatives


def test_mine_hard_negatives_prefers_confusing_negatives():
    df = pd.DataFrame(
        {
            "source1_entity_id": ["S1-1", "S1-1", "S1-1", "S1-2", "S1-2"],
            "candidate_entity_id": ["S2-1", "S2-2", "S2-3", "S2-4", "S2-5"],
            "label": [0, 0, 1, 0, 0],
            "name_char_ngram_jaccard": [0.95, 0.20, 0.97, 0.55, 0.10],
            "address_token_jaccard": [0.80, 0.00, 0.90, 0.65, 0.05],
            "name_token_overlap_coef": [0.90, 0.10, 0.95, 0.40, 0.05],
            "pin_exact_match": [0.0, 0.0, 1.0, 0.0, 0.0],
            "country_exact_match": [1.0, 0.0, 1.0, 1.0, 1.0],
            "ch_count": [3, 1, 3, 2, 1],
        }
    )

    hard = mine_hard_negatives(df, label_col="label", hard_fraction=0.5)

    assert len(hard) >= 1
    assert "S2-1" in hard["candidate_entity_id"].tolist()
    assert "S2-3" not in hard["candidate_entity_id"].tolist()
    assert "composite_score" in hard.columns
