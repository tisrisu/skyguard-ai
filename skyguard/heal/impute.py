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
            if pd.isna(row[var]) or row[var] == -9999.0:
                continue
            n_mean, _ = clim.expected(row["station_id"], row["ts"], var)
            if pd.notna(n_mean):
                anomalies.append(row[var] - n_mean)
        
        if anomalies:
            med_anomaly = float(np.median(anomalies))
            spatial_pred = expected_mean + med_anomaly

    # 2. Temporal prediction: linear trend from the last known good values
    temporal_pred = np.nan
    if not history.empty:
        hist_valid = history[history[var].notna()]
        if len(hist_valid) >= 2:
            y1, y2 = hist_valid.iloc[-2][var], hist_valid.iloc[-1][var]
            temporal_pred = y2 + (y2 - y1)  # simple linear extrapolation for 1 hour
        elif len(hist_valid) == 1:
            temporal_pred = hist_valid.iloc[-1][var]
            
    # 3. Blend them (we trust spatial more for things like storms and sudden changes)
    from skyguard.config import load_config
    cfg = load_config()
    w_spatial = cfg.get("impute", {}).get("spatial_weight", 0.7)
    w_temporal = cfg.get("impute", {}).get("temporal_weight", 0.3)
    
    if pd.notna(spatial_pred) and pd.notna(temporal_pred):
        return float(w_spatial * spatial_pred + w_temporal * temporal_pred)
    elif pd.notna(spatial_pred):
        return float(spatial_pred)
    elif pd.notna(temporal_pred):
        return float(temporal_pred)
    elif pd.notna(expected_mean):
        return float(expected_mean)
    
    return float("nan")
