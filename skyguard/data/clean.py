"""Clean raw downloads into one long-format hourly table.

Owner: M1

Run:  python -m skyguard.data.clean
Writes: data/processed/ncr_hourly.parquet  (station_id, ts, temp_c, pressure_hpa, rh_pct)
"""

import pandas as pd

from skyguard.config import load_config


def clean_station(df: pd.DataFrame, max_gap_h: int) -> pd.DataFrame:
    """TODO:
      - reindex to a complete hourly UTC index over the station's period
      - drop duplicate timestamps
      - linearly interpolate gaps of <= max_gap_h hours; longer gaps stay NaN
    """
    raise NotImplementedError


def main() -> None:
    """TODO: read every data/raw/*.parquet, clean, add station_id, concat, save."""
    cfg = load_config()
    raise NotImplementedError


if __name__ == "__main__":
    main()
