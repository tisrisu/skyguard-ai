"""What is normal for a station at a given month and hour of day.

Owner: M1

Used by the neighbour check (to remove altitude and local-climate differences),
by severity (deviation in standard deviations) and by imputation.
"""

import pandas as pd


class Climatology:
    def __init__(self):
        self.table: pd.DataFrame | None = None   # index (station_id, month, hour), columns (var, "mean"/"std")

    def fit(self, df: pd.DataFrame) -> "Climatology":
        """TODO: mean and std of each variable per (station_id, month, hour of day).
        Fit on the train split only. Use a std floor (e.g. 0.3 °C / 0.5 hPa / 2 %) to avoid dividing by ~0.
        """
        raise NotImplementedError

    def expected(self, station_id: str, ts: pd.Timestamp, var: str) -> tuple[float, float]:
        """(mean, std) for this station, month and hour."""
        raise NotImplementedError

    def anomaly(self, df: pd.DataFrame, var: str) -> pd.Series:
        """Vectorised: df[var] minus the expected mean for each row."""
        raise NotImplementedError


def climatology(df: pd.DataFrame) -> Climatology:
    return Climatology().fit(df)
