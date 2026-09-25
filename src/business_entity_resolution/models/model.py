"""
Matching models - common fit(X, y) / predict_proba(model, X) interface across
baseline, GBM, and ensemble models. See docs/architecture.md §9.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


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
    """LightGBM Binary Classifier."""

    def __init__(self, **lgbm_params):
        import lightgbm as lgb
        default_params = dict(n_estimators=500, learning_rate=0.05, num_leaves=63,
                               objective="binary", random_state=42, verbose=-1)
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
    """XGBoost Binary Classifier."""

    def __init__(self, **xgb_params):
        import xgboost as xgb
        default_params = dict(n_estimators=500, learning_rate=0.05, max_depth=8,
                               eval_metric="logloss", random_state=42)
        default_params.update(xgb_params)
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
    """Blend of LightGBM + XGBoost for maximum generalization and precision."""

    def __init__(self, lgb_weight: float = 0.6, xgb_weight: float = 0.4):
        self.lgb = LightGBMMatcher()
        self.xgb = XGBoostMatcher()
        self.lgb_weight = lgb_weight
        self.xgb_weight = xgb_weight

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "EnsembleMatcher":
        self.lgb.fit(X, y)
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
