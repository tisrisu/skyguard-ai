"""Download hourly weather data for Delhi-NCR stations.

Owner: M1

Run:  python -m skyguard.data.fetch
Writes:
  data/raw/<station_id>.parquet       one file per station
  data/stations.json                  list of {station_id, name, lat, lon, elevation_m}

Data source: Open-Meteo Archive API (free, no key, reanalysis + observations).
Falls back from Meteostat (bulk endpoint returns 403 as of 2026).
"""

import json
import time
import urllib.request

import pandas as pd

from skyguard.config import load_config, RAW_DIR, DATA_DIR, STATIONS_FILE

# Real Delhi-NCR meteorological station sites.
# Coordinates and elevations from public WMO / IMD / Wikipedia sources.
_DEFAULT_STATIONS = [
    {"station_id": "DEL-01", "name": "Safdarjung",           "lat": 28.585, "lon": 77.206, "elevation_m": 216},
    {"station_id": "DEL-02", "name": "Palam (IGI Airport)",  "lat": 28.566, "lon": 77.103, "elevation_m": 230},
    {"station_id": "DEL-03", "name": "Ayanagar",             "lat": 28.470, "lon": 77.109, "elevation_m": 231},
    {"station_id": "DEL-04", "name": "Hindon (Ghaziabad)",   "lat": 28.697, "lon": 77.364, "elevation_m": 211},
    {"station_id": "DEL-05", "name": "Noida",                "lat": 28.535, "lon": 77.391, "elevation_m": 200},
    {"station_id": "DEL-06", "name": "Gurugram",             "lat": 28.457, "lon": 77.026, "elevation_m": 234},
    {"station_id": "DEL-07", "name": "Faridabad",            "lat": 28.408, "lon": 77.310, "elevation_m": 198},
    {"station_id": "DEL-08", "name": "Rohtak",               "lat": 28.895, "lon": 76.606, "elevation_m": 219},
]

VARIABLES = ["temp_c", "pressure_hpa", "rh_pct"]

# Mapping from Open-Meteo field names → our column names
_OM_MAP = {
    "temperature_2m": "temp_c",
    "surface_pressure": "pressure_hpa",
    "relative_humidity_2m": "rh_pct",
}


def _fetch_open_meteo(lat: float, lon: float, start: str, end: str) -> pd.DataFrame:
    """Fetch hourly T, P, RH from the Open-Meteo Archive API for one point.

    Parameters
    ----------
    lat, lon : float
    start, end : ISO date strings like "2022-01-01"

    Returns
    -------
    DataFrame with columns ts (UTC datetime), temp_c, pressure_hpa, rh_pct.
    """
    params = (
        f"latitude={lat}&longitude={lon}"
        f"&start_date={start}&end_date={end}"
        "&hourly=temperature_2m,relative_humidity_2m,surface_pressure"
        "&timezone=UTC"
    )
    url = f"https://archive-api.open-meteo.com/v1/archive?{params}"

    req = urllib.request.Request(url, headers={"User-Agent": "SkyGuardAI/1.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read())

    hourly = data["hourly"]
    df = pd.DataFrame({
        "ts": pd.to_datetime(hourly["time"], utc=True),
        "temp_c": hourly.get("temperature_2m"),
        "pressure_hpa": hourly.get("surface_pressure"),
        "rh_pct": hourly.get("relative_humidity_2m"),
    })
    return df


def main() -> None:
    """Fetch all stations, compute coverage, save raw parquets + stations.json."""
    cfg = load_config()

    period = cfg.get("period", {})
    start = period.get("start", "2022-01-01")
    end = period.get("end", "2024-12-31")
    min_cov = cfg.get("region", {}).get("min_coverage", 0.80)
    max_stations = cfg.get("region", {}).get("max_stations", 8)

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    station_defs = _DEFAULT_STATIONS[:max_stations]

    valid_meta: list[dict] = []
    coverage_rows: list[dict] = []

    for stn in station_defs:
        sid = stn["station_id"]
        print(f"  Fetching {sid} ({stn['name']}) …", end=" ", flush=True)

        try:
            df = _fetch_open_meteo(stn["lat"], stn["lon"], start, end)
        except Exception as exc:
            print(f"FAILED: {exc}")
            continue

        if df.empty:
            print("no data")
            continue

        total = len(df)
        t_cov = df["temp_c"].notna().sum() / total
        p_cov = df["pressure_hpa"].notna().sum() / total
        rh_cov = df["rh_pct"].notna().sum() / total

        print(f"{total:,} rows  T={t_cov:.0%}  P={p_cov:.0%}  RH={rh_cov:.0%}")

        if t_cov >= min_cov and p_cov >= min_cov and rh_cov >= min_cov:
            df["station_id"] = sid
            df.to_parquet(RAW_DIR / f"{sid}.parquet", index=False)

            valid_meta.append({
                "station_id": sid,
                "name": stn["name"],
                "lat": stn["lat"],
                "lon": stn["lon"],
                "elevation_m": stn["elevation_m"],
            })
            coverage_rows.append({
                "ID": sid, "Name": stn["name"],
                "T%": f"{t_cov:.1%}", "P%": f"{p_cov:.1%}", "RH%": f"{rh_cov:.1%}",
            })
        else:
            print(f"    -> SKIPPED (coverage < {min_cov:.0%})")

        # be polite to the free API
        time.sleep(1.0)

    # save stations.json
    with open(STATIONS_FILE, "w", encoding="utf-8") as f:
        json.dump(valid_meta, f, indent=2)

    print(f"\n[OK] {len(valid_meta)} stations saved to {STATIONS_FILE}")
    if coverage_rows:
        print(pd.DataFrame(coverage_rows).to_string(index=False))


if __name__ == "__main__":
    main()