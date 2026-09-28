"""Clean raw downloads into one long-format hourly table.

Owner: M1

Run:  python -m skyguard.data.clean
Writes: data/processed/ncr_hourly.parquet  (station_id, ts, temp_c, pressure_hpa, rh_pct)
        data/sample_2stations.parquet      (first 2 stations, one month — unblocks teammates)
"""

import pandas as pd

from skyguard.config import load_config, RAW_DIR, PROCESSED_DIR, HOURLY_FILE, DATA_DIR

VARIABLES = ["temp_c", "pressure_hpa", "rh_pct"]


def clean_station(df: pd.DataFrame, max_gap_h: int) -> pd.DataFrame:
    """Reindex to a complete hourly UTC grid, then interpolate short gaps.

    Steps:
      1. Ensure 'ts' is a proper DatetimeIndex in UTC.
      2. Drop duplicate timestamps (keep the first).
      3. Build a complete hourly range covering the station's data span.
      4. Reindex to that range — new rows are NaN.
      5. Linearly interpolate gaps of <= max_gap_h consecutive NaN hours;
         longer gaps stay NaN.
    """
    df = df.copy()
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df = df.drop_duplicates(subset="ts", keep="first")
    df = df.set_index("ts").sort_index()

    # complete hourly index
    full_idx = pd.date_range(df.index.min(), df.index.max(), freq="h", tz="UTC")
    df = df.reindex(full_idx)
    df.index.name = "ts"

    # interpolate only short gaps (limit = max_gap_h consecutive NaNs)
    for col in VARIABLES:
        if col in df.columns:
            df[col] = df[col].interpolate(method="linear", limit=max_gap_h)

    return df.reset_index()


def main() -> None:
    """Read every data/raw/*.parquet, clean, add station_id, concat, save."""
    cfg = load_config()
    max_gap_h = cfg.get("clean", {}).get("max_interp_gap_h", 2)

    raw_files = sorted(RAW_DIR.glob("*.parquet"))
    if not raw_files:
        raise FileNotFoundError(
            f"No parquet files in {RAW_DIR}. Run: python -m skyguard.data.fetch"
        )

    frames = []
    for path in raw_files:
        df = pd.read_parquet(path)
        # station_id was stamped during fetch
        if "station_id" not in df.columns:
            print(f"  SKIP {path.name}: no station_id column")
            continue

        sid = df["station_id"].iloc[0]
        print(f"  Cleaning {sid} ({path.name}) …")

        df = clean_station(df, max_gap_h)
        df["station_id"] = sid

        # keep only the canonical columns
        df = df[["station_id", "ts"] + [c for c in VARIABLES if c in df.columns]]
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values(["station_id", "ts"]).reset_index(drop=True)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(HOURLY_FILE, index=False)
    print(f"\n[OK] Saved {len(combined):,} rows -> {HOURLY_FILE}")

    # --- sample file for teammates (first 2 stations, one month) ---
    station_ids = sorted(combined["station_id"].unique())[:2]
    if len(station_ids) >= 2:
        sample = combined[combined["station_id"].isin(station_ids)]
        # one month: pick the first month with decent data
        first_ts = sample["ts"].min()
        sample = sample[sample["ts"] < first_ts + pd.Timedelta(days=31)]
        sample_path = DATA_DIR / "sample_2stations.parquet"
        sample.to_parquet(sample_path, index=False)
        print(f"[OK] Sample file ({len(sample):,} rows) -> {sample_path}")


if __name__ == "__main__":
    main()
