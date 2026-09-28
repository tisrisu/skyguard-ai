"""What is normal for a station at a given month and hour of day.

Owner: M1

Used by the neighbour check (to remove altitude and local-climate differences),
by severity (deviation in standard deviations) and by imputation.
Fit on the train split only.
"""

import pandas as pd

from skyguard.config import load_config
from skyguard.schemas import VARIABLES


def _utc(ts) -> pd.Timestamp:
    ts = pd.Timestamp(ts)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


class Climatology:
    def __init__(self):
        self.table: pd.DataFrame | None = None   # index (station_id, month, hour), columns (var, "mean"/"std")
        self._rows: dict = {}                    # same data as plain tuples, for fast single lookups
        self._cols: dict = {}

    def fit(self, df: pd.DataFrame) -> "Climatology":
        """Mean and std per (station_id, month, hour). std is floored (config climatology.std_floor)."""
        floors = load_config()["climatology"]["std_floor"]
        ts = pd.to_datetime(df["ts"], utc=True)
        grouped = df.groupby([df["station_id"], ts.dt.month.rename("month"), ts.dt.hour.rename("hour")])

        parts = {}
        for var in VARIABLES:
            stats = grouped[var].agg(["mean", "std"])
            stats["std"] = stats["std"].fillna(floors[var]).clip(lower=floors[var])
            parts[var] = stats
        self.table = pd.concat(parts, axis=1)

        self._cols = {col: i for i, col in enumerate(self.table.columns)}
        self._rows = dict(zip(self.table.index, self.table.itertuples(index=False, name=None)))
        return self

    def expected(self, station_id: str, ts, var: str) -> tuple[float, float]:
        """(mean, std) for this station at the month and hour of ts. (NaN, NaN) if unknown."""
        if self.table is None:
            raise RuntimeError("Climatology is not fitted, call fit() first")
        ts = _utc(ts)
        row = self._rows.get((station_id, ts.month, ts.hour))
        if row is None:
            return float("nan"), float("nan")
        return float(row[self._cols[(var, "mean")]]), float(row[self._cols[(var, "std")]])

    def _stats_for(self, df: pd.DataFrame, var: str) -> pd.DataFrame:
        """Climatology mean and std for every row of df, in df's order."""
        if self.table is None:
            raise RuntimeError("Climatology is not fitted, call fit() first")
        ts = pd.to_datetime(df["ts"], utc=True)
        keys = pd.MultiIndex.from_arrays([df["station_id"], ts.dt.month, ts.dt.hour])
        stats = self.table[var].reindex(keys)
        stats.index = df.index
        return stats

    def anomaly(self, df: pd.DataFrame, var: str) -> pd.Series:
        """Vectorised: df[var] minus the expected mean for each row."""
        return df[var] - self._stats_for(df, var)["mean"]

    def deviation_sigma(self, df: pd.DataFrame, var: str) -> pd.Series:
        """(value - mean) / std for each row."""
        stats = self._stats_for(df, var)
        return (df[var] - stats["mean"]) / stats["std"]


def climatology(df: pd.DataFrame) -> Climatology:
    return Climatology().fit(df)
