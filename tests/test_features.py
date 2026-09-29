"""Tests for M2 task L1 — feature engineering.

Verifies make_features (single-row) and make_feature_frame (vectorised) against
the sample data and synthetic fixtures.
"""

import math

import numpy as np
import pandas as pd
import pytest

from skyguard.data.climatology import Climatology
from skyguard.models.features import (
    FEATURE_NAMES,
    _same_value_run_length,
    _same_value_run_vectorised,
    make_feature_frame,
    make_features,
)
from skyguard.physics.dewpoint import dewpoint_c


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def clim(hourly):
    """Climatology fitted on the synthetic hourly fixture from conftest."""
    return Climatology().fit(hourly)


# ---------------------------------------------------------------------------
# Helper: _same_value_run_length
# ---------------------------------------------------------------------------

class TestSameValueRunLength:
    def test_basic_run(self):
        s = pd.Series([30.0, 30.0, 31.0, 31.0, 31.0])
        assert _same_value_run_length(s) == 3

    def test_no_repeat(self):
        s = pd.Series([1.0, 2.0, 3.0])
        assert _same_value_run_length(s) == 1

    def test_single_element(self):
        s = pd.Series([42.0])
        assert _same_value_run_length(s) == 1

    def test_all_same(self):
        s = pd.Series([5.0, 5.0, 5.0, 5.0])
        assert _same_value_run_length(s) == 4

    def test_nan_at_end(self):
        s = pd.Series([1.0, 2.0, np.nan])
        assert _same_value_run_length(s) == 1

    def test_empty(self):
        s = pd.Series([], dtype=float)
        assert _same_value_run_length(s) == 0


class TestSameValueRunVectorised:
    def test_basic(self):
        s = pd.Series([30.0, 30.0, 31.0, 31.0, 31.0])
        result = _same_value_run_vectorised(s)
        expected = [1, 2, 1, 2, 3]
        np.testing.assert_array_equal(result.values, expected)

    def test_no_repeats(self):
        s = pd.Series([1.0, 2.0, 3.0])
        result = _same_value_run_vectorised(s)
        np.testing.assert_array_equal(result.values, [1, 1, 1])


# ---------------------------------------------------------------------------
# make_features (single-row)
# ---------------------------------------------------------------------------

class TestMakeFeatures:
    def test_output_shape(self, hourly, clim):
        """Feature vector has exactly len(FEATURE_NAMES) elements."""
        station_a = hourly[hourly["station_id"] == "A"].reset_index(drop=True)
        feat = make_features(station_a, clim, "temp_c")
        assert feat.shape == (len(FEATURE_NAMES),), f"Expected {len(FEATURE_NAMES)} features, got {feat.shape}"

    def test_no_nans(self, hourly, clim):
        """No NaN values in output — all are filled with 0.0."""
        station_a = hourly[hourly["station_id"] == "A"].reset_index(drop=True)
        feat = make_features(station_a, clim, "temp_c")
        assert not np.any(np.isnan(feat)), f"Found NaN in features: {feat}"

    def test_no_infs(self, hourly, clim):
        station_a = hourly[hourly["station_id"] == "A"].reset_index(drop=True)
        feat = make_features(station_a, clim, "temp_c")
        assert np.all(np.isfinite(feat)), f"Found inf in features: {feat}"

    def test_value_feature(self, hourly, clim):
        """Feature[0] ('value') equals the last temp_c in the history."""
        station_a = hourly[hourly["station_id"] == "A"].reset_index(drop=True)
        feat = make_features(station_a, clim, "temp_c")
        expected_value = station_a["temp_c"].iloc[-1]
        assert abs(feat[0] - expected_value) < 1e-9

    def test_delta_1h(self, hourly, clim):
        """Feature[1] ('delta_1h') equals diff(1) of the last row."""
        station_a = hourly[hourly["station_id"] == "A"].reset_index(drop=True)
        feat = make_features(station_a, clim, "temp_c")
        expected = station_a["temp_c"].iloc[-1] - station_a["temp_c"].iloc[-2]
        assert abs(feat[1] - expected) < 1e-9

    def test_dewpoint_correctness(self, hourly, clim):
        """Dewpoint feature matches manual Magnus calculation."""
        station_a = hourly[hourly["station_id"] == "A"].reset_index(drop=True)
        feat = make_features(station_a, clim, "temp_c")

        last_temp = station_a["temp_c"].iloc[-1]
        last_rh = station_a["rh_pct"].iloc[-1]
        expected_dp = dewpoint_c(last_temp, last_rh)

        # dewpoint is feature index 7
        assert abs(feat[7] - expected_dp) < 1e-6, (
            f"Dewpoint mismatch: got {feat[7]}, expected {expected_dp}"
        )

    def test_hour_sin_cos(self, hourly, clim):
        """Hour sin/cos cycle correctly for the last timestamp."""
        station_a = hourly[hourly["station_id"] == "A"].reset_index(drop=True)
        feat = make_features(station_a, clim, "temp_c")

        last_ts = pd.Timestamp(station_a["ts"].iloc[-1])
        hour = last_ts.hour + last_ts.minute / 60.0
        expected_sin = math.sin(2 * math.pi * hour / 24.0)
        expected_cos = math.cos(2 * math.pi * hour / 24.0)

        assert abs(feat[9] - expected_sin) < 1e-9
        assert abs(feat[10] - expected_cos) < 1e-9

    def test_single_row_history(self, hourly, clim):
        """Works with only 1 row of history (edge case during replay start)."""
        station_a = hourly[hourly["station_id"] == "A"].head(1).reset_index(drop=True)
        feat = make_features(station_a, clim, "temp_c")
        assert feat.shape == (len(FEATURE_NAMES),)
        assert not np.any(np.isnan(feat))

    def test_same_value_run_frozen(self, hourly, clim):
        """Frozen sensor (repeated values) should give a run > 1."""
        station_a = hourly[hourly["station_id"] == "A"].copy().reset_index(drop=True)
        # Freeze the last 5 values
        frozen_val = station_a["temp_c"].iloc[-6]
        station_a.loc[station_a.index[-5:], "temp_c"] = frozen_val
        feat = make_features(station_a, clim, "temp_c")
        # same_value_run is feature index 6
        assert feat[6] >= 5, f"Expected run >= 5 for frozen sensor, got {feat[6]}"

    def test_works_for_all_variables(self, hourly, clim):
        """make_features works for all three variables."""
        station_a = hourly[hourly["station_id"] == "A"].reset_index(drop=True)
        for var in ("temp_c", "pressure_hpa", "rh_pct"):
            feat = make_features(station_a, clim, var)
            assert feat.shape == (len(FEATURE_NAMES),), f"Failed for {var}"
            assert not np.any(np.isnan(feat)), f"NaN for {var}"


# ---------------------------------------------------------------------------
# make_feature_frame (vectorised)
# ---------------------------------------------------------------------------

class TestMakeFeatureFrame:
    def test_output_shape(self, hourly, clim):
        """Output has same number of rows as input and columns match FEATURE_NAMES."""
        ff = make_feature_frame(hourly, clim, "temp_c")
        assert ff.shape[0] == len(hourly)
        assert list(ff.columns) == FEATURE_NAMES

    def test_no_nans(self, hourly, clim):
        """No NaN values anywhere in the frame."""
        ff = make_feature_frame(hourly, clim, "temp_c")
        nan_count = ff.isna().sum().sum()
        assert nan_count == 0, f"Found {nan_count} NaNs in feature frame"

    def test_index_preserved(self, hourly, clim):
        """Output index matches input index."""
        ff = make_feature_frame(hourly, clim, "temp_c")
        pd.testing.assert_index_equal(ff.index, hourly.index)

    def test_value_column_matches_input(self, hourly, clim):
        """'value' column is exactly the input variable."""
        ff = make_feature_frame(hourly, clim, "temp_c")
        pd.testing.assert_series_equal(
            ff["value"], hourly["temp_c"].rename("value"), check_names=True
        )

    def test_consistency_with_single_row(self, hourly, clim):
        """Last row of make_feature_frame should match make_features for that station."""
        station_a = hourly[hourly["station_id"] == "A"].reset_index(drop=True)
        ff = make_feature_frame(station_a, clim, "temp_c")
        feat_single = make_features(station_a, clim, "temp_c")

        ff_last = ff.iloc[-1].values
        np.testing.assert_allclose(
            ff_last, feat_single, atol=1e-9,
            err_msg="Vectorised and single-row features differ for last row"
        )

    def test_works_for_all_variables(self, hourly, clim):
        for var in ("temp_c", "pressure_hpa", "rh_pct"):
            ff = make_feature_frame(hourly, clim, var)
            assert ff.shape == (len(hourly), len(FEATURE_NAMES))
            assert ff.isna().sum().sum() == 0


# ---------------------------------------------------------------------------
# Integration: run on real sample data
# ---------------------------------------------------------------------------

class TestOnSampleData:
    @pytest.fixture
    def sample_df(self):
        from skyguard.data.io import load_data
        try:
            return load_data(path=pd.io.common.Path("data/sample_2stations.parquet"))
        except FileNotFoundError:
            pytest.skip("sample_2stations.parquet not available")

    @pytest.fixture
    def sample_clim(self, sample_df):
        return Climatology().fit(sample_df)

    def test_make_features_on_real_data(self, sample_df, sample_clim):
        station = sample_df[sample_df["station_id"] == "DEL-01"].reset_index(drop=True)
        feat = make_features(station, sample_clim, "temp_c")
        assert feat.shape == (len(FEATURE_NAMES),)
        assert not np.any(np.isnan(feat))
        # Sanity: value should be a plausible Delhi temperature
        assert -10 < feat[0] < 55, f"Value {feat[0]} not plausible"

    def test_make_feature_frame_on_real_data(self, sample_df, sample_clim):
        ff = make_feature_frame(sample_df, sample_clim, "temp_c")
        assert ff.shape == (len(sample_df), len(FEATURE_NAMES))
        assert ff.isna().sum().sum() == 0
