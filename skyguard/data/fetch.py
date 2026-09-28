"""Download hourly station data from Meteostat.

Owner: M1

Run:  python -m skyguard.data.fetch
Writes:
  data/raw/<meteostat_id>.parquet    one file per station
  data/stations.json                 list of {station_id, name, lat, lon, elevation_m, meteostat_id}
"""

import pandas as pd

from skyguard.config import load_config


def find_stations(cfg: dict) -> pd.DataFrame:
    """Nearest stations to cfg["region"] center.

    TODO:
      - meteostat.Stations().nearby(lat, lon).fetch(n) with n a bit larger than max_stations
      - drop stations further than region.radius_km
    """
    raise NotImplementedError


def fetch_hourly(meteostat_id: str, start, end) -> pd.DataFrame:
    """Hourly observations for one station.

    TODO:
      - meteostat.Hourly(meteostat_id, start, end, model=False).fetch()
        (model=False: real observations only, no model-filled values)
      - rename temp -> temp_c, pres -> pressure_hpa, rhum -> rh_pct
      - index -> "ts" column, localised to UTC
    """
    raise NotImplementedError


def main() -> None:
    """TODO:
      - find stations, fetch each, keep those with >= region.min_coverage for all three variables
      - keep at most region.max_stations; assign ids DEL-01, DEL-02, ... (nearest first)
      - save raw parquet files and data/stations.json
      - print a coverage table
    """
    cfg = load_config()
    raise NotImplementedError


if __name__ == "__main__":
    main()
