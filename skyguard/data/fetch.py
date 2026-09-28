"""Download hourly weather data for the Delhi-NCR stations.

Owner: M1

Run:  python -m skyguard.data.fetch
Writes:
  data/raw/<station_id>.parquet   one file per station
  data/stations.json              list of {station_id, name, lat, lon, elevation_m}

Source: Open-Meteo historical weather API (free, no key). This is ECMWF reanalysis
at the station coordinates, not raw station observations. Meteostat, the original
plan, returned HTTP 403 for bulk downloads.
"""

import json
import time
import urllib.parse
import urllib.request

import pandas as pd

from skyguard.config import RAW_DIR, STATIONS_FILE, load_config
from skyguard.schemas import VARIABLES

API_URL = "https://archive-api.open-meteo.com/v1/archive"

# Open-Meteo field -> our column
FIELDS = {
    "temperature_2m": "temp_c",
    "pressure_msl": "pressure_hpa",
    "relative_humidity_2m": "rh_pct",
}

# IMD / WMO station sites in Delhi-NCR (coordinates and elevations from public listings)
STATIONS = [
    {"station_id": "DEL-01", "name": "Safdarjung",          "lat": 28.585, "lon": 77.206, "elevation_m": 216},
    {"station_id": "DEL-02", "name": "Palam (IGI Airport)", "lat": 28.566, "lon": 77.103, "elevation_m": 230},
    {"station_id": "DEL-03", "name": "Ayanagar",            "lat": 28.470, "lon": 77.109, "elevation_m": 231},
    {"station_id": "DEL-04", "name": "Hindon (Ghaziabad)",  "lat": 28.697, "lon": 77.364, "elevation_m": 211},
    {"station_id": "DEL-05", "name": "Noida",               "lat": 28.535, "lon": 77.391, "elevation_m": 200},
    {"station_id": "DEL-06", "name": "Gurugram",            "lat": 28.457, "lon": 77.026, "elevation_m": 234},
    {"station_id": "DEL-07", "name": "Faridabad",           "lat": 28.408, "lon": 77.310, "elevation_m": 198},
    {"station_id": "DEL-08", "name": "Rohtak",              "lat": 28.895, "lon": 76.606, "elevation_m": 219},
]


def fetch_hourly(lat: float, lon: float, start: str, end: str) -> pd.DataFrame:
    """Hourly temp_c, pressure_hpa (sea level) and rh_pct for one point, ts in UTC."""
    query = urllib.parse.urlencode({
        "latitude": lat,
        "longitude": lon,
        "start_date": start,
        "end_date": end,
        "hourly": ",".join(FIELDS),
        "timezone": "UTC",
    })
    req = urllib.request.Request(f"{API_URL}?{query}", headers={"User-Agent": "skyguard-ai"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        hourly = json.loads(resp.read())["hourly"]

    df = pd.DataFrame({col: hourly[field] for field, col in FIELDS.items()}, dtype=float)
    df.insert(0, "ts", pd.to_datetime(hourly["time"], utc=True))
    return df


def main() -> None:
    cfg = load_config()
    start, end = cfg["period"]["start"], cfg["period"]["end"]
    min_cov = cfg["region"]["min_coverage"]

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    kept = []
    for stn in STATIONS[: cfg["region"]["max_stations"]]:
        sid = stn["station_id"]
        print(f"{sid} {stn['name']}: ", end="", flush=True)
        try:
            df = fetch_hourly(stn["lat"], stn["lon"], start, end)
        except Exception as exc:
            print(f"failed ({exc})")
            continue

        coverage = df[list(VARIABLES)].notna().mean()
        print(f"{len(df):,} rows, coverage " + ", ".join(f"{v} {coverage[v]:.1%}" for v in VARIABLES))
        if (coverage >= min_cov).all():
            df.insert(0, "station_id", sid)
            df.to_parquet(RAW_DIR / f"{sid}.parquet", index=False)
            kept.append(stn)
        else:
            print(f"  skipped, coverage below {min_cov:.0%}")

        time.sleep(1)  # free API, keep the request rate low

    with open(STATIONS_FILE, "w", encoding="utf-8") as f:
        json.dump(kept, f, indent=2)
        f.write("\n")
    print(f"Saved {len(kept)} stations to {STATIONS_FILE}")


if __name__ == "__main__":
    main()
