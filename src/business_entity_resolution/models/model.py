"""
Matching models - common fit(X, y) / predict_proba(model, X) interface across
baseline, GBM, and any future neural variant. See docs/architecture.md §9.

RuleBasedBaseline is fully implemented as the sanity floor every real model must beat
(architecture.md §24-M, step 5). LightGBMMatcher is a thin, working wrapper - tune
hyperparameters via configs/pipeline.yaml, not by editing this file per experiment.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


class RuleBasedBaseline:
    """Score = simple weighted combination of name/address similarity.
    No training required; `fit` is a no-op. This is the floor to beat (architecture.md
    §9-A, §24-M step 5) - always keep this comparison in the experiment log."""

    def __init__(self, name_col: str = "name_char_ngram_jaccard",
                 address_col: str = "address_token_jaccard",
                 name_weight: float = 0.6, address_weight: float = 0.4):
        self.name_col = name_col
        self.address_col = address_col
        self.name_weight = name_weight
        self.address_weight = address_weight

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "RuleBasedBaseline":
        return self  # nothing to learn

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        score = self.name_weight * X[self.name_col] + self.address_weight * X[self.address_col]
        return np.clip(score.to_numpy(), 0.0, 1.0)


class LightGBMMatcher:
    """Thin wrapper around lightgbm.LGBMClassifier satisfying the fit/predict_proba
    contract. MIT-licensed, far under the 8B parameter ceiling (CLAUDE.md §2)."""

    def __init__(self, **lgbm_params):
        import lightgbm as lgb
        default_params = dict(n_estimators=500, learning_rate=0.05, num_leaves=63,
                               objective="binary", random_state=42)
        default_params.update(lgbm_params)
        self.model = lgb.LGBMClassifier(**default_params)
        self.feature_columns_: list | None = None

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "LightGBMMatcher":
        self.feature_columns_ = list(X.columns)
        self.model.fit(X, y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        cols = self.feature_columns_ or list(X.columns)
        return self.model.predict_proba(X[cols])[:, 1]

    def feature_importance(self) -> pd.Series:
        if self.feature_columns_ is None:
            raise RuntimeError("Call fit() before feature_importance().")
        return pd.Series(self.model.feature_importances_, index=self.feature_columns_).sort_values(ascending=False)


MODEL_REGISTRY = {
    "baseline": RuleBasedBaseline,
    "lightgbm": LightGBMMatcher,
    # "xgboost": XGBoostMatcher,    # TODO if adopted - same fit/predict_proba contract
    # "catboost": CatBoostMatcher,  # TODO if adopted
}


def build_model(config: dict):
    """config: the `model` block of configs/pipeline.yaml, e.g. {"type": "lightgbm", "params": {...}}."""
    model_type = config.get("type", "baseline")
    params = config.get("params", {})
    cls = MODEL_REGISTRY[model_type]
    return cls(**params) if model_type != "baseline" else cls()
