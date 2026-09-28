"""Do neighbouring stations agree with this reading?

Owner: M3

Method (thresholds in config.yaml -> spatial):
  1. anomaly a = value - climatology mean (removes altitude and local-climate differences)
  2. z = (a_station - median(a_neighbours)) / (1.4826 * MAD(a_neighbours) + mad_floor[var])
  3. a front reaches stations at different times: evaluate neighbours at every hour in
     ts ± time_window_h and keep the smallest |z|
  4. consistent = |z| <= z_threshold; None if fewer than min_neighbours have data
"""

import pandas as pd


def spatial_check(df_window: pd.DataFrame, stations: list[dict], clim, station_id: str,
                  ts: pd.Timestamp, var: str, cfg: dict | None = None) -> dict:
    """df_window: all stations' rows around ts (at least ts ± time_window_h).

    Returns:
      {
        "z": float | None,
        "n": int,                      # neighbours with data
        "consistent": bool | None,
        "reason": "5 of 5 neighbouring stations show normal temperature (z = 9.4)",
      }
    """
    raise NotImplementedError
