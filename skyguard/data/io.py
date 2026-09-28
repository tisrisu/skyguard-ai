"""Loading the prepared data files."""

import json

import pandas as pd

from skyguard.config import HOURLY_FILE, STATIONS_FILE, load_config


def load_data(path=None) -> pd.DataFrame:
    """Hourly data in long format: station_id, ts (UTC), temp_c, pressure_hpa, rh_pct."""
    path = path or HOURLY_FILE
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run: python -m skyguard.data.fetch && python -m skyguard.data.clean"
        )
    df = pd.read_parquet(path)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df.sort_values(["station_id", "ts"]).reset_index(drop=True)


def load_stations(path=None) -> list[dict]:
    """List of {station_id, name, lat, lon, elevation_m}."""
    path = path or STATIONS_FILE
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run: python -m skyguard.data.fetch")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def select_split(df: pd.DataFrame, name: str, cfg: dict | None = None) -> pd.DataFrame:
    """Rows of df in the named split from config.yaml ("train", "val" or "test").

    Both dates in the config are inclusive, so "2023-12-31" keeps that whole day.
    """
    cfg = cfg or load_config()
    start, end = cfg["splits"][name]
    start = pd.Timestamp(start, tz="UTC")
    end = pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)
    return df[(df["ts"] >= start) & (df["ts"] < end)].reset_index(drop=True)
