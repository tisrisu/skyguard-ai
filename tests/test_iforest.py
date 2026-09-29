"""Tests for M2 task L2 — Isolation Forest model.

Verifies IFModel fit, score, save, and load using synthetic data from conftest
and (when available) the real sample data.
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from skyguard.data.climatology import Climatology
from skyguard.models.features import FEATURE_NAMES, make_features
from skyguard.models.iforest import IFModel
from skyguard.schemas import VARIABLES


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def clim(hourly):
    """Climatology fitted on the synthetic hourly fixture."""
    return Climatology().fit(hourly)


@pytest.fixture
def trained_model(hourly, clim):
    """IFModel fitted on synthetic data (train=val=hourly for simplicity)."""
    model = IFModel()
    model.fit(hourly, hourly, clim)
    return model


# ---------------------------------------------------------------------------
# fit()
# ---------------------------------------------------------------------------

class TestFit:
    def test_fit_returns_self(self, hourly, clim):
        """fit() returns the model itself (for chaining)."""
        model = IFModel()
        result = model.fit(hourly, hourly, clim)
        assert result is model

    def test_models_created_for_all_variables(self, trained_model):
        """One IsolationForest per variable."""
        for var in VARIABLES:
            assert var in trained_model.models, f"Missing model for {var}"
            assert var in trained_model.scalers, f"Missing scaler for {var}"
            assert var in trained_model.val_scores, f"Missing val_scores for {var}"

    def test_val_scores_are_sorted(self, trained_model):
        """Validation scores must be sorted for searchsorted to work."""
        for var in VARIABLES:
            scores = trained_model.val_scores[var]
            assert np.all(scores[:-1] <= scores[1:]), (
                f"val_scores for {var} are not sorted"
            )

    def test_val_scores_length_matches_data(self, hourly, trained_model):
        """Number of validation scores equals number of rows in val_df."""
        expected_len = len(hourly)  # we used hourly as both train and val
        for var in VARIABLES:
            assert len(trained_model.val_scores[var]) == expected_len, (
                f"val_scores for {var} has wrong length"
            )


# ---------------------------------------------------------------------------
# score()
# ---------------------------------------------------------------------------

class TestScore:
    def test_score_returns_float_in_range(self, hourly, clim, trained_model):
        """score() returns a float in [0, 1]."""
        station = hourly[hourly["station_id"] == "A"].reset_index(drop=True)
        for var in VARIABLES:
            feat = make_features(station, clim, var)
            s = trained_model.score(feat, var)
            assert isinstance(s, float), f"Expected float, got {type(s)}"
            assert 0.0 <= s <= 1.0, f"Score {s} out of [0,1] for {var}"

    def test_normal_data_scores_low(self, hourly, clim, trained_model):
        """Normal data (identical distribution) should mostly score < 0.9."""
        station = hourly[hourly["station_id"] == "A"].reset_index(drop=True)
        scores = []
        # Sample a handful of rows
        for i in range(24, len(station), 24):  # one per day
            feat = make_features(station.iloc[:i+1].reset_index(drop=True), clim, "temp_c")
            scores.append(trained_model.score(feat, "temp_c"))
        median_score = np.median(scores)
        assert median_score < 0.9, (
            f"Median score on normal data is {median_score}, expected < 0.9"
        )

    def test_extreme_spike_scores_high(self, hourly, clim, trained_model):
        """A 55°C spike should score much higher than normal data."""
        station = hourly[hourly["station_id"] == "A"].copy().reset_index(drop=True)
        # Inject a massive spike at the last row
        station.loc[station.index[-1], "temp_c"] = 55.0
        feat = make_features(station, clim, "temp_c")
        spike_score = trained_model.score(feat, "temp_c")
        # It should be above the median normal score
        assert spike_score > 0.5, (
            f"Spike score {spike_score} is not elevated enough"
        )

    def test_frozen_sensor_scores_elevated(self, hourly, clim, trained_model):
        """A frozen sensor (many identical values) should score higher than normal."""
        station = hourly[hourly["station_id"] == "A"].copy().reset_index(drop=True)
        # Freeze the last 12 readings
        frozen_val = station["temp_c"].iloc[-13]
        station.loc[station.index[-12:], "temp_c"] = frozen_val
        feat = make_features(station, clim, "temp_c")
        frozen_score = trained_model.score(feat, "temp_c")
        # Should be at least somewhat elevated
        assert frozen_score > 0.3, (
            f"Frozen sensor score {frozen_score} is not elevated"
        )

    def test_score_unknown_variable_returns_zero(self, trained_model):
        """score() with an unknown variable gracefully returns 0.0."""
        feat = np.zeros(len(FEATURE_NAMES))
        s = trained_model.score(feat, "nonexistent_var")
        assert s == 0.0

    def test_score_deterministic(self, hourly, clim, trained_model):
        """Same input produces same score (no randomness in scoring)."""
        station = hourly[hourly["station_id"] == "A"].reset_index(drop=True)
        feat = make_features(station, clim, "temp_c")
        s1 = trained_model.score(feat, "temp_c")
        s2 = trained_model.score(feat, "temp_c")
        assert s1 == s2, f"Non-deterministic: {s1} != {s2}"


# ---------------------------------------------------------------------------
# save() / load()
# ---------------------------------------------------------------------------

class TestPersistence:
    def test_save_and_load_roundtrip(self, trained_model, hourly, clim):
        """Model produces identical scores after save→load."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_model.joblib"
            trained_model.save(path)

            # File should exist and be non-empty
            assert path.exists()
            assert path.stat().st_size > 0

            loaded = IFModel.load(path)

        # Loaded model should have the same variables
        assert set(loaded.models.keys()) == set(trained_model.models.keys())
        assert set(loaded.scalers.keys()) == set(trained_model.scalers.keys())
        assert set(loaded.val_scores.keys()) == set(trained_model.val_scores.keys())

        # Scores should be identical
        station = hourly[hourly["station_id"] == "A"].reset_index(drop=True)
        for var in VARIABLES:
            feat = make_features(station, clim, var)
            s_orig = trained_model.score(feat, var)
            s_loaded = loaded.score(feat, var)
            assert s_orig == s_loaded, (
                f"Score mismatch for {var}: {s_orig} vs {s_loaded}"
            )

    def test_load_missing_file_raises(self):
        """load() raises FileNotFoundError for a non-existent path."""
        with pytest.raises(FileNotFoundError, match="not found"):
            IFModel.load(Path("/nonexistent/model.joblib"))

    def test_save_creates_parent_dirs(self, trained_model):
        """save() creates parent directories if they don't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "subdir" / "deep" / "model.joblib"
            trained_model.save(path)
            assert path.exists()


# ---------------------------------------------------------------------------
# Integration: verify on real sample data (if available)
# ---------------------------------------------------------------------------

class TestOnSampleData:
    @pytest.fixture
    def sample_df(self):
        from skyguard.data.io import load_data
        try:
            return load_data(path=Path("data/sample_2stations.parquet"))
        except FileNotFoundError:
            pytest.skip("sample_2stations.parquet not available")

    @pytest.fixture
    def sample_clim(self, sample_df):
        return Climatology().fit(sample_df)

    @pytest.fixture
    def sample_model(self, sample_df, sample_clim):
        """Train on sample data (same for train and val since it's just 1 month)."""
        model = IFModel()
        model.fit(sample_df, sample_df, sample_clim)
        return model

    def test_fit_on_real_data(self, sample_model):
        """Model trains without errors on real Delhi data."""
        for var in VARIABLES:
            assert var in sample_model.models

    def test_score_on_real_data(self, sample_df, sample_clim, sample_model):
        """Scoring works on real data and produces valid percentiles."""
        station = sample_df[sample_df["station_id"] == "DEL-01"].reset_index(drop=True)
        for var in VARIABLES:
            feat = make_features(station, sample_clim, var)
            s = sample_model.score(feat, var)
            assert 0.0 <= s <= 1.0, f"Score {s} out of range for {var}"

    def test_spike_vs_normal_on_real_data(self, sample_df, sample_clim, sample_model):
        """A 55°C spike on real data should score higher than the normal last row."""
        station = sample_df[sample_df["station_id"] == "DEL-01"].copy().reset_index(drop=True)

        # Normal score
        feat_normal = make_features(station, sample_clim, "temp_c")
        s_normal = sample_model.score(feat_normal, "temp_c")

        # Inject spike
        station.loc[station.index[-1], "temp_c"] = 55.0
        feat_spike = make_features(station, sample_clim, "temp_c")
        s_spike = sample_model.score(feat_spike, "temp_c")

        assert s_spike > s_normal, (
            f"Spike score ({s_spike}) should exceed normal score ({s_normal})"
        )
