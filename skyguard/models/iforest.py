"""Isolation Forest anomaly score per variable.

Owner: M2

One IsolationForest per variable, trained on clean train-split data.
score() returns a percentile in [0, 1] relative to normal validation data,
so 0.99 means "more unusual than 99 % of normal readings".
"""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from skyguard.config import MODELS_DIR, load_config
from skyguard.models.features import FEATURE_NAMES, make_feature_frame
from skyguard.schemas import VARIABLES

logger = logging.getLogger(__name__)


class IFModel:
    """Isolation-Forest anomaly detector — one forest per weather variable.

    Workflow
    --------
    1. ``fit(train_df, val_df, clim)`` — extract features, scale, train one
       IsolationForest per variable, then record the raw anomaly-score
       distribution on the **clean** validation split so that ``score()`` can
       return a meaningful percentile.
    2. ``score(features, var)`` — scale a single feature vector, run the forest,
       and return a **percentile** (0 = perfectly normal, 1 = most anomalous
       reading ever seen in validation data).
    3. ``save()`` / ``load()`` — persist everything with joblib.
    """

    def __init__(self, cfg: dict | None = None):
        self.cfg = cfg or load_config()
        self.models: dict[str, IsolationForest] = {}   # var -> fitted IsolationForest
        self.scalers: dict[str, StandardScaler] = {}   # var -> fitted StandardScaler
        self.val_scores: dict[str, np.ndarray] = {}    # var -> sorted raw anomaly scores on normal val rows

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, train_df: pd.DataFrame, val_df: pd.DataFrame, clim) -> "IFModel":
        """Train one IsolationForest per variable on clean data.

        Parameters
        ----------
        train_df : pd.DataFrame
            Clean (no faults) long-format data for the **train** split
            (2022-01-01 to 2023-12-31).  Columns: station_id, ts, temp_c,
            pressure_hpa, rh_pct.
        val_df : pd.DataFrame
            Clean long-format data for the **validation** split
            (2024-01-01 to 2024-06-30).  Used only to build the percentile
            lookup table — never for threshold tuning or testing.
        clim : Climatology
            Fitted on the train split.

        Returns
        -------
        self, so you can chain: ``model = IFModel().fit(train, val, clim)``.
        """
        ml_cfg = self.cfg["ml"]
        seed = self.cfg.get("seed", 42)

        for var in VARIABLES:
            logger.info("IFModel: fitting %s …", var)

            # --- extract features (NaN-safe, returns 0-filled DataFrame) ---
            train_feats = make_feature_frame(train_df, clim, var)
            val_feats = make_feature_frame(val_df, clim, var)

            # --- scale ---
            scaler = StandardScaler()
            X_train = scaler.fit_transform(train_feats.values)
            X_val = scaler.transform(val_feats.values)

            # --- train Isolation Forest ---
            iforest = IsolationForest(
                n_estimators=ml_cfg["n_estimators"],
                max_samples=min(ml_cfg["max_samples"], X_train.shape[0]),
                random_state=seed,
                contamination="auto",       # default; we use our own percentile
            )
            iforest.fit(X_train)

            # --- build percentile lookup from validation scores ---
            # IsolationForest.score_samples() returns negative scores where
            # *more negative = more anomalous*.  We negate so that
            # higher = more anomalous, making np.searchsorted work naturally
            # for percentile computation.
            raw_val_scores = -iforest.score_samples(X_val)
            raw_val_scores.sort()

            # --- store ---
            self.models[var] = iforest
            self.scalers[var] = scaler
            self.val_scores[var] = raw_val_scores

            logger.info(
                "  %s: train %d rows, val %d rows, val score range [%.4f, %.4f]",
                var, X_train.shape[0], X_val.shape[0],
                raw_val_scores[0], raw_val_scores[-1],
            )

        return self

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def score(self, features: np.ndarray, var: str) -> float:
        """Percentile of this reading's anomaly score among normal validation scores.

        Parameters
        ----------
        features : np.ndarray
            Feature vector of shape ``(n_features,)`` from ``make_features()``.
        var : str
            Variable name (``temp_c``, ``pressure_hpa``, ``rh_pct``).

        Returns
        -------
        float in [0, 1].  0.0 = perfectly normal.  1.0 = more anomalous than
        every validation reading.  0.99 = more anomalous than 99 % of
        validation data.
        """
        if var not in self.models:
            logger.warning("IFModel.score: no model for '%s', returning 0.0", var)
            return 0.0

        # Scale using the same scaler used during training
        X = self.scalers[var].transform(features.reshape(1, -1))

        # Raw anomaly score (higher = more anomalous, after negation)
        raw = -self.models[var].score_samples(X)[0]

        # Percentile: fraction of validation scores <= this score
        sorted_scores = self.val_scores[var]
        rank = np.searchsorted(sorted_scores, raw, side="right")
        percentile = rank / len(sorted_scores)

        # Clip to [0, 1] for safety (should already be in range)
        return float(np.clip(percentile, 0.0, 1.0))

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path = MODELS_DIR / "iforest.joblib") -> None:
        """Persist all models, scalers, and validation scores to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        bundle = {
            "models": self.models,
            "scalers": self.scalers,
            "val_scores": self.val_scores,
            "cfg_ml": self.cfg["ml"],
        }
        joblib.dump(bundle, path)
        logger.info("IFModel saved to %s (%.1f KB)", path, path.stat().st_size / 1024)

    @classmethod
    def load(cls, path: Path = MODELS_DIR / "iforest.joblib") -> "IFModel":
        """Load a previously saved IFModel."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found.  Train the model first:\n"
                f"  python -m skyguard.models.train"
            )
        bundle = joblib.load(path)
        obj = cls()
        obj.models = bundle["models"]
        obj.scalers = bundle["scalers"]
        obj.val_scores = bundle["val_scores"]
        return obj

