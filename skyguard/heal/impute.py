"""Corrected value for a faulty reading.

Owner: M1
"""

import numpy as np
import pandas as pd


def impute(history: pd.DataFrame, neighbours_now: pd.DataFrame, clim, station_id: str,
           ts: pd.Timestamp, var: str) -> float:
    """Best estimate of what the sensor should have read.

    history         this station's rows before ts (faulty rows already excluded if possible)
    neighbours_now  neighbour rows at ts (+/- spatial.time_window_h)
    """
    # 1. Spatial prediction: expected mean for this station + median neighbour anomaly
    spatial_pred = np.nan
    expected_mean, _ = clim.expected(station_id, ts, var)
    
    if pd.notna(expected_mean) and not neighbours_now.empty:
        anomalies = []
        for _, row in neighbours_now.iterrows():
            n_mean, _ = clim.expected(row["station_id"], row["ts"], var)
            if pd.notna(n_mean):
                anomalies.append(row[var] - n_mean)
        
        if anomalies:
            med_anomaly = float(np.median(anomalies))
            spatial_pred = expected_mean + med_anomaly

    # 2. Temporal prediction: simple forward fill from the last known good value
    temporal_pred = np.nan
    if not history.empty:
        hist_valid = history[history[var].notna()]
        if not hist_valid.empty:
            temporal_pred = hist_valid.iloc[-1][var]
            
    # 3. Blend them (we trust spatial more for things like storms and sudden changes)
    if pd.notna(spatial_pred) and pd.notna(temporal_pred):
        return float(0.7 * spatial_pred + 0.3 * temporal_pred)
    elif pd.notna(spatial_pred):
        return float(spatial_pred)
    elif pd.notna(temporal_pred):
        return float(temporal_pred)
    elif pd.notna(expected_mean):
        return float(expected_mean)
    
    return float("nan")
