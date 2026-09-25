"""
Matching models - common fit(X, y) / predict_proba(model, X) interface across
baseline, GBM, and ensemble models. Automatically utilizes CUDA GPU if present.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def is_cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def mine_hard_negatives(feature_df: pd.DataFrame, label_col: str = "label",
                        hard_fraction: float = 0.35, score_columns: list | None = None,
                        min_candidates: int = 1) -> pd.DataFrame:
    """Return the top blocked-but-negative pairs that look most like true matches.

    This is the repo's hard-negative mining primitive: it ranks negative candidate pairs
    by a cheap composite score using the same pairwise features that dominate precision
    failures (name similarity, address similarity, provenance), then keeps the most
    confusing negatives for training. The implementation intentionally avoids external
    data and stays compatible with the challenge's two-stage API.
    """
    if feature_df.empty:
        return feature_df.copy()

    base = feature_df.copy()
    if label_col not in base.columns:
        raise KeyError(f"Label column '{label_col}' not found in feature frame")

    negatives = base[base[label_col] == 0].copy()
    if negatives.empty:
        return negatives.copy()

    if score_columns is None:
        score_columns = [
            "name_char_ngram_jaccard",
            "address_token_jaccard",
            "name_token_overlap_coef",
            "pin_exact_match",
            "country_exact_match",
            "ch_count",
        ]

    composite = pd.DataFrame({"tmp_score": 0.0}, index=negatives.index)
    for col in score_columns:
        if col in negatives.columns:
            composite["tmp_score"] += negatives[col].fillna(0.0)

    # Give a slight boost to candidates surfaced by multiple channels or exact-name
    # matches; those are the near-misses that most often poison F0.5 precision.
    if "ch_count" in negatives.columns:
        composite["tmp_score"] += 0.05 * negatives["ch_count"].fillna(0.0)
    if "name_exact_normalized" in negatives.columns:
        composite["tmp_score"] += 0.10 * negatives["name_exact_normalized"].fillna(0.0)

    negatives = negatives.assign(composite_score=composite["tmp_score"].to_numpy())
    negatives = negatives.sort_values(["composite_score", "candidate_entity_id"], ascending=[False, True])

    n_keep = max(min_candidates, int(round(len(negatives) * max(0.0, min(1.0, hard_fraction)))))
    n_keep = min(n_keep, len(negatives))
    return negatives.head(n_keep).copy()


class RuleBasedBaseline:
    """Score = simple weighted combination of name/address similarity."""

    def __init__(self, name_col: str = "name_char_ngram_jaccard",
                 address_col: str = "address_token_jaccard",
                 name_weight: float = 0.6, address_weight: float = 0.4):
        self.name_col = name_col
        self.address_col = address_col
        self.name_weight = name_weight
        self.address_weight = address_weight

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "RuleBasedBaseline":
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        score = self.name_weight * X[self.name_col] + self.address_weight * X[self.address_col]
        return np.clip(score.to_numpy(), 0.0, 1.0)


class LightGBMMatcher:
    """LightGBM Binary Classifier with multi-threading."""

    def __init__(self, **lgbm_params):
        import lightgbm as lgb
        default_params = dict(
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=63,
            objective="binary",
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
        default_params.update(lgbm_params)
        self.model = lgb.LGBMClassifier(**default_params)
        self.feature_columns_: list | None = None

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "LightGBMMatcher":
        self.feature_columns_ = list(X.columns)
        self.model.fit(X[self.feature_columns_], y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        cols = self.feature_columns_ or list(X.columns)
        return self.model.predict_proba(X[cols])[:, 1]

    def feature_importance(self) -> pd.Series:
        if self.feature_columns_ is None:
            raise RuntimeError("Call fit() before feature_importance().")
        return pd.Series(self.model.feature_importances_, index=self.feature_columns_).sort_values(ascending=False)


class XGBoostMatcher:
    """XGBoost Binary Classifier with CUDA GPU Acceleration."""

    def __init__(self, **xgb_params):
        import xgboost as xgb
        has_gpu = is_cuda_available()
        default_params = dict(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=8,
            eval_metric="logloss",
            random_state=42,
            tree_method="hist",
            device="cuda" if has_gpu else "cpu",
        )
        default_params.update(xgb_params)
        print(f"[Model] Initialized XGBoost with device='{default_params['device']}'")
        self.model = xgb.XGBClassifier(**default_params)
        self.feature_columns_: list | None = None

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "XGBoostMatcher":
        self.feature_columns_ = list(X.columns)
        self.model.fit(X[self.feature_columns_], y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        cols = self.feature_columns_ or list(X.columns)
        return self.model.predict_proba(X[cols])[:, 1]

    def feature_importance(self) -> pd.Series:
        if self.feature_columns_ is None:
            raise RuntimeError("Call fit() before feature_importance().")
        return pd.Series(self.model.feature_importances_, index=self.feature_columns_).sort_values(ascending=False)


class EnsembleMatcher:
    """Blend of LightGBM + XGBoost with GPU acceleration for maximum generalization."""

    def __init__(self, lgb_weight: float = 0.6, xgb_weight: float = 0.4):
        self.lgb = LightGBMMatcher()
        self.xgb = XGBoostMatcher()
        self.lgb_weight = lgb_weight
        self.xgb_weight = xgb_weight

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "EnsembleMatcher":
        print("  [Training] Fitting LightGBM Classifier...")
        self.lgb.fit(X, y)
        print("  [Training] Fitting XGBoost Classifier on GPU...")
        self.xgb.fit(X, y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        p_lgb = self.lgb.predict_proba(X)
        p_xgb = self.xgb.predict_proba(X)
        return self.lgb_weight * p_lgb + self.xgb_weight * p_xgb


MODEL_REGISTRY = {
    "baseline": RuleBasedBaseline,
    "lightgbm": LightGBMMatcher,
    "xgboost": XGBoostMatcher,
    "ensemble": EnsembleMatcher,
}


def build_model(config: dict):
    model_type = config.get("type", "baseline")
    params = config.get("params", {})
    cls = MODEL_REGISTRY[model_type]
    return cls(**params) if model_type != "baseline" else cls()


def save_model(matcher, feature_cols: list, path: str) -> None:
    """Persist a trained matcher + its feature column list to disk (pickle)."""
    import pickle, os
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump((matcher, feature_cols), f, protocol=4)
    print(f"  Model saved to {path}")


def load_model(path: str):
    """Load a previously saved (matcher, feature_cols) tuple."""
    import pickle
    with open(path, "rb") as f:
        matcher, feature_cols = pickle.load(f)
    print(f"  Model loaded from {path}")
    return matcher, feature_cols
