"""Corrected value for a faulty reading.

Owner: M1
"""

import pandas as pd


def impute(history: pd.DataFrame, neighbours_now: pd.DataFrame, clim, station_id: str,
           ts: pd.Timestamp, var: str) -> float:
    """Best estimate of what the sensor should have read.

    history         this station's rows before ts (faulty rows already excluded if possible)
    neighbours_now  neighbour rows at ts (± spatial.time_window_h)

    TODO:
      spatial  = clim mean for this station at ts + median neighbour anomaly
      temporal = linear extrapolation from the last good values
      return a weighted blend (weights from each method's error on the validation split);
      fall back to whichever one is available.
    """
    raise NotImplementedError
