"""Isolation Forest anomaly score per variable.

Owner: M2

One IsolationForest per variable, trained on clean train-split data.
score() returns a percentile in [0, 1] relative to normal validation data,
so 0.99 means "more unusual than 99 % of normal readings".
"""

from pathlib import Path

import numpy as np
import pandas as pd

from skyguard.config import MODELS_DIR, load_config


class IFModel:
    def __init__(self, cfg: dict | None = None):
        self.cfg = cfg or load_config()
        self.models = {}          # var -> fitted IsolationForest
        self.scalers = {}         # var -> fitted StandardScaler
        self.val_scores = {}      # var -> sorted anomaly scores on normal validation rows

    def fit(self, train_df: pd.DataFrame, val_df: pd.DataFrame, clim) -> "IFModel":
        """TODO:
          - features via models.features.make_feature_frame
          - IsolationForest(n_estimators, max_samples from config, random_state=seed)
          - store -score_samples() of normal validation rows for percentile lookup
        """
        raise NotImplementedError

    def score(self, features: np.ndarray, var: str) -> float:
        """Percentile of this reading's anomaly score among normal validation scores (0..1)."""
        raise NotImplementedError

    def save(self, path: Path = MODELS_DIR / "iforest.joblib") -> None:
        raise NotImplementedError

    @classmethod
    def load(cls, path: Path = MODELS_DIR / "iforest.joblib") -> "IFModel":
        raise NotImplementedError
