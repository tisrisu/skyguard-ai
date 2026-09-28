import numpy as np
import pandas as pd

from skyguard.data.climatology import Climatology
from skyguard.data.clean import clean_station


def test_clean_fills_short_gaps_only():
    ts = pd.date_range("2024-01-01", periods=12, freq="h", tz="UTC")
    temp = np.arange(12, dtype=float)
    temp[[2, 3]] = np.nan          # 2 h gap -> filled
    temp[[6, 7, 8]] = np.nan       # 3 h gap -> stays empty
    raw = pd.DataFrame({"ts": ts, "temp_c": temp, "pressure_hpa": 1000.0, "rh_pct": 50.0})
    raw = raw.drop(index=10)       # missing timestamp -> row added, 1 h gap -> filled

    out = clean_station(raw, max_gap_h=2)

    assert len(out) == 12
    assert out["temp_c"].iloc[[2, 3, 10]].tolist() == [2.0, 3.0, 10.0]
    assert out["temp_c"].iloc[6:9].isna().all()


def test_climatology_expected_and_anomaly(hourly):
    clim = Climatology().fit(hourly)

    mean, std = clim.expected("A", "2024-08-05T06:00:00Z", "temp_c")
    assert np.isclose(mean, 35.0)
    assert std >= 0.3                                    # floored, data has no spread
    assert np.isnan(clim.expected("X", "2024-08-05T06:00:00Z", "temp_c")[0])

    assert np.allclose(clim.anomaly(hourly, "temp_c"), 0)
    shifted = hourly.assign(temp_c=hourly["temp_c"] + 3)
    assert np.allclose(clim.anomaly(shifted, "temp_c"), 3)
