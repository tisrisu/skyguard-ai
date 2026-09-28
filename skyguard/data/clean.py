"""Clean raw downloads into one long-format hourly table.

Owner: M1

Run:  python -m skyguard.data.clean
Writes:
  data/processed/ncr_hourly.parquet   station_id, ts, temp_c, pressure_hpa, rh_pct
  data/sample_2stations.parquet       first 2 stations, first 31 days (small file for quick tests)
"""

import pandas as pd

from skyguard.config import DATA_DIR, HOURLY_FILE, PROCESSED_DIR, RAW_DIR, load_config
from skyguard.schemas import DATA_COLUMNS, VARIABLES


def clean_station(df: pd.DataFrame, max_gap_h: int) -> pd.DataFrame:
    """Put one station on a complete hourly UTC grid and fill short gaps.

    Duplicate timestamps keep the first reading. Gaps of up to max_gap_h hours are
    filled by linear interpolation; longer gaps and gaps at the edges stay NaN.
    """
    df = df.copy()
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df = df.drop_duplicates("ts").set_index("ts").sort_index()
    df = df.reindex(pd.date_range(df.index.min(), df.index.max(), freq="h"))
    df.index.name = "ts"

    for var in VARIABLES:
        values = df[var].astype(float)
        missing = values.isna()
        gap_len = missing.groupby((missing != missing.shift()).cumsum()).transform("sum")
        filled = values.interpolate(limit_area="inside")
        df[var] = filled.where(~missing | (gap_len <= max_gap_h))

    return df.reset_index()


def main() -> None:
    cfg = load_config()
    max_gap_h = cfg["clean"]["max_interp_gap_h"]

    raw_files = sorted(RAW_DIR.glob("*.parquet"))
    if not raw_files:
        raise FileNotFoundError(f"No parquet files in {RAW_DIR}. Run: python -m skyguard.data.fetch")

    frames = []
    for path in raw_files:
        df = pd.read_parquet(path)
        sid = df["station_id"].iloc[0]
        df = clean_station(df, max_gap_h)
        df["station_id"] = sid
        frames.append(df[DATA_COLUMNS])
        print(f"{sid}: {len(df):,} rows")

    combined = pd.concat(frames, ignore_index=True).sort_values(["station_id", "ts"])
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(HOURLY_FILE, index=False)
    print(f"Saved {len(combined):,} rows to {HOURLY_FILE}")

    first_two = sorted(combined["station_id"].unique())[:2]
    sample = combined[combined["station_id"].isin(first_two)]
    sample = sample[sample["ts"] < sample["ts"].min() + pd.Timedelta(days=31)]
    sample.to_parquet(DATA_DIR / "sample_2stations.parquet", index=False)
    print(f"Saved sample ({len(sample):,} rows) to {DATA_DIR / 'sample_2stations.parquet'}")


if __name__ == "__main__":
    main()
