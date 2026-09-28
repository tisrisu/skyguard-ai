"""What is normal for a station at a given month and hour of day.

Owner: M1

Used by the neighbour check (to remove altitude and local-climate differences),
by severity (deviation in standard deviations) and by imputation.
"""

import numpy as np
import pandas as pd

VARIABLES = ["temp_c", "pressure_hpa", "rh_pct"]

# Minimum std floor to avoid dividing by ≈ 0
_STD_FLOOR = {
    "temp_c": 0.3,
    "pressure_hpa": 0.5,
    "rh_pct": 2.0,
}


class Climatology:
    """Mean and std of each variable per (station_id, month, hour-of-day).

    Fit on the training split only so that the model never sees test data.
    """

    def __init__(self):
        # Index: (station_id, month, hour)
        # Columns: MultiIndex (var, "mean") and (var, "std")
        self.table: pd.DataFrame | None = None

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------
    def fit(self, df: pd.DataFrame) -> "Climatology":
        """Compute mean and std of each variable per (station_id, month, hour).

        Parameters
        ----------
        df : DataFrame with columns station_id, ts, temp_c, pressure_hpa, rh_pct.
             ts must be a timezone-aware (UTC) datetime.
        """
        df = df.copy()
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
        df["month"] = df["ts"].dt.month
        df["hour"] = df["ts"].dt.hour

        grouped = df.groupby(["station_id", "month", "hour"])

        records = {}
        for var in VARIABLES:
            if var not in df.columns:
                continue
            agg = grouped[var].agg(["mean", "std"])
            # apply std floor
            floor = _STD_FLOOR.get(var, 0.3)
            agg["std"] = agg["std"].clip(lower=floor)
            records[var] = agg

        # combine into a single DataFrame with MultiIndex columns: (var, mean/std)
        self.table = pd.concat(records, axis=1)
        return self

    # ------------------------------------------------------------------
    # Scalar lookup
    # ------------------------------------------------------------------
    def expected(self, station_id: str, ts: pd.Timestamp, var: str) -> tuple[float, float]:
        """(mean, std) for this station, month, and hour.

        Returns (NaN, NaN) if the key is missing (unseen station / sparse data).
        """
        if self.table is None:
            raise RuntimeError("Climatology not fitted. Call .fit() first.")

        ts = pd.Timestamp(ts, tz="UTC") if ts.tzinfo is None else ts
        key = (station_id, ts.month, ts.hour)
        try:
            row = self.table.loc[key]
            return float(row[(var, "mean")]), float(row[(var, "std")])
        except KeyError:
            return float("nan"), float("nan")

    # ------------------------------------------------------------------
    # Vectorised anomaly
    # ------------------------------------------------------------------
    def anomaly(self, df: pd.DataFrame, var: str) -> pd.Series:
        """df[var] minus the expected mean for each row (vectorised)."""
        if self.table is None:
            raise RuntimeError("Climatology not fitted. Call .fit() first.")

        df = df.copy()
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
        df["_month"] = df["ts"].dt.month
        df["_hour"] = df["ts"].dt.hour

        # merge the climatology mean for each row
        means = self.table[(var, "mean")].rename("_clim_mean")
        merged = df.join(means, on=["station_id", "_month", "_hour"])
        return df[var] - merged["_clim_mean"]

    def deviation_sigma(self, df: pd.DataFrame, var: str) -> pd.Series:
        """(value - mean) / std — how many standard deviations away."""
        if self.table is None:
            raise RuntimeError("Climatology not fitted. Call .fit() first.")

        df = df.copy()
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
        df["_month"] = df["ts"].dt.month
        df["_hour"] = df["ts"].dt.hour

        clim = self.table[[var]].copy()
        clim.columns = ["_clim_mean", "_clim_std"]
        merged = df.join(clim, on=["station_id", "_month", "_hour"])
        return (df[var] - merged["_clim_mean"]) / merged["_clim_std"].replace(0, np.nan)


def climatology(df: pd.DataFrame) -> Climatology:
    """Convenience function matching the signature in DEMO_PLAN §4.2."""
    return Climatology().fit(df)
