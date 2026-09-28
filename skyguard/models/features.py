"""Features for one variable at the latest timestamp of a station's history.

Owner: M2

The same function is used for training (applied row by row, or a vectorised
version over a whole DataFrame) and for live replay.
"""

import numpy as np
import pandas as pd

# Suggested starting set; adjust, but keep names and plain-English labels in sync.
FEATURE_NAMES = [
    "value",
    "delta_1h",
    "delta_3h",
    "roll_std_6h",
    "roll_std_24h",
    "dev_sigma",          # (value - climatology mean) / climatology std
    "same_value_run",     # consecutive identical readings
    "dewpoint",
    "dewpoint_delta_1h",
    "hour_sin",
    "hour_cos",
]

# Used by explain() to turn SHAP feature names into readable reasons.
FEATURE_LABELS = {
    "value": "reported value",
    "delta_1h": "sudden 1-hour change",
    "delta_3h": "3-hour change",
    "roll_std_6h": "short-term variability",
    "roll_std_24h": "daily variability",
    "dev_sigma": "distance from normal for this hour and season",
    "same_value_run": "identical readings in a row",
    "dewpoint": "implied dewpoint",
    "dewpoint_delta_1h": "change in moisture content",
    "hour_sin": "time of day",
    "hour_cos": "time of day",
}


def make_features(station_hist: pd.DataFrame, clim, var: str) -> np.ndarray:
    """Feature vector (len(FEATURE_NAMES),) for the last row of station_hist.

    station_hist  one station's rows up to and including the current ts, sorted by ts
    TODO: NaN-safe; missing inputs -> 0 plus rely on the physics MISSING rule.
    """
    raise NotImplementedError


def make_feature_frame(df: pd.DataFrame, clim, var: str) -> pd.DataFrame:
    """Vectorised version over a whole long-format DataFrame (for training)."""
    raise NotImplementedError
