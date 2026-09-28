import numpy as np
import pandas as pd

from skyguard.inject.injector import build_test_set, inject_fault, inject_storm

T0 = pd.Timestamp("2024-08-10 12:00", tz="UTC")


def rows(df, station, start, hours):
    return (df["station_id"] == station) & (df["ts"] >= start) & (df["ts"] < start + pd.Timedelta(hours=hours))


def test_spike_keeps_sign_of_magnitude_and_labels(hourly):
    out = inject_fault(hourly, "A", "temp_c", "SPIKE", T0, 1, magnitude=23, seed=1)
    m = rows(out, "A", T0, 1)

    assert np.allclose(out.loc[m, "temp_c"] - out.loc[m, "orig_temp_c"], 23)
    assert (out.loc[m, "label_temp_c"] == "SPIKE").all()
    assert (out.loc[m, "event_id"] == "SPIKE_001").all()
    assert (out.loc[~m, "label_temp_c"] == "NONE").all()


def test_second_live_injection_gets_next_event_id(hourly):
    out = inject_fault(hourly, "A", "temp_c", "SPIKE", T0, 2, magnitude=10)
    out = inject_fault(out, "B", "temp_c", "SPIKE", T0, 1, magnitude=10)
    assert set(out.loc[out["event_id"] != "", "event_id"]) == {"SPIKE_001", "SPIKE_002"}


def test_noise_is_scaled_to_normal_hourly_change(hourly):
    # a dropout already in the data must not inflate the noise
    out = inject_fault(hourly, "A", "temp_c", "DROPOUT", T0 - pd.Timedelta(days=3), 3, seed=3)
    out = inject_fault(out, "A", "temp_c", "NOISE", T0, 24, magnitude=4, seed=0)
    m = rows(out, "A", T0, 24)
    assert (out.loc[m, "temp_c"] - out.loc[m, "orig_temp_c"]).abs().max() < 15


def test_out_of_range_is_outside_valid_range(hourly):
    for seed in range(5):
        out = inject_fault(hourly, "A", "rh_pct", "OUT_OF_RANGE", T0, 6, seed=seed)
        assert not out.loc[rows(out, "A", T0, 6), "rh_pct"].between(0, 102).any()


def test_storm_is_genuine_and_reaches_neighbours(hourly, stations):
    out = inject_storm(hourly, stations, "A", T0, seed=0)
    storm = out["event_id"] == "STORM_001"

    assert set(out.loc[storm, "station_id"]) == {"A", "B", "C"}
    assert (out.loc[storm, ["label_temp_c", "label_pressure_hpa", "label_rh_pct"]] == "NONE").all().all()
    assert (out.loc[storm, "temp_c"] < out.loc[storm, "orig_temp_c"]).all()
    assert (out.loc[storm, "pressure_hpa"] > out.loc[storm, "orig_pressure_hpa"]).all()


def test_test_set_has_no_overlapping_events(hourly, stations):
    out = build_test_set(hourly, stations, seed=5)
    labels = out[["label_temp_c", "label_pressure_hpa", "label_rh_pct"]] != "NONE"

    assert (labels.sum(axis=1) <= 1).all()
    assert not (labels.any(axis=1) & out["event_id"].str.startswith("STORM")).any()
    assert out["event_id"].str.startswith("STORM").any()
