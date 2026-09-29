"""Tests for M2 task L4 — fault-type fingerprinting.

Verifies fingerprint() against every rule in the docstring, edge cases, and
the exact fault patterns produced by M1's injector.
"""

import numpy as np
import pandas as pd
import pytest

from skyguard.fusion.fingerprint import _recent_baseline, fingerprint
from skyguard.schemas import FaultType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_hist(values: list[float], var: str = "temp_c",
               station_id: str = "DEL-01",
               start: str = "2024-08-14T00:00:00Z") -> pd.DataFrame:
    """Build a minimal station_hist DataFrame from a list of values."""
    n = len(values)
    ts = pd.date_range(start, periods=n, freq="h", tz="UTC")
    data = {
        "station_id": [station_id] * n,
        "ts": ts,
        "temp_c": [30.0] * n,
        "pressure_hpa": [1013.0] * n,
        "rh_pct": [60.0] * n,
    }
    data[var] = values
    return pd.DataFrame(data)


def _physics(fault_hint=None, hard=False, soft=False, rule_ids=None, reasons=None):
    """Build a physics dict matching check_physics() output."""
    return {
        "hard": hard,
        "soft": soft,
        "rule_ids": rule_ids or [],
        "reasons": reasons or [],
        "fault_hint": fault_hint,
    }


def _spatial(z=0.0, n=5, consistent=True, reason=""):
    return {"z": z, "n": n, "consistent": consistent, "reason": reason}


# ---------------------------------------------------------------------------
# Rule 1: physics fault_hint
# ---------------------------------------------------------------------------

class TestPhysicsFaultHint:
    def test_frozen_hint(self):
        hist = _make_hist([30.0] * 10)
        assert fingerprint(hist, "temp_c", _physics(fault_hint="FROZEN")) == FaultType.FROZEN

    def test_dropout_hint(self):
        hist = _make_hist([30.0] * 10)
        assert fingerprint(hist, "temp_c", _physics(fault_hint="DROPOUT")) == FaultType.DROPOUT

    def test_out_of_range_hint(self):
        hist = _make_hist([30.0] * 10)
        assert fingerprint(hist, "temp_c", _physics(fault_hint="OUT_OF_RANGE")) == FaultType.OUT_OF_RANGE

    def test_no_hint(self):
        """No fault_hint → fingerprint relies on data patterns."""
        hist = _make_hist([30.0, 30.5, 31.0, 30.8, 31.2])
        assert fingerprint(hist, "temp_c", _physics(fault_hint=None)) is None

    def test_empty_physics_dict(self):
        """Empty dict (M3's stub not ready) → no crash."""
        hist = _make_hist([30.0, 30.5, 31.0, 30.8, 31.2])
        assert fingerprint(hist, "temp_c", {}) is None


# ---------------------------------------------------------------------------
# Rule 2: DROPOUT
# ---------------------------------------------------------------------------

class TestDropout:
    def test_nan_value(self):
        hist = _make_hist([30.0, 31.0, float("nan")])
        assert fingerprint(hist, "temp_c", _physics()) == FaultType.DROPOUT

    def test_sentinel_minus_9999(self):
        hist = _make_hist([30.0, 31.0, -9999.0])
        assert fingerprint(hist, "temp_c", _physics()) == FaultType.DROPOUT

    def test_sentinel_minus_999(self):
        hist = _make_hist([30.0, 31.0, -999.0])
        assert fingerprint(hist, "temp_c", _physics()) == FaultType.DROPOUT

    def test_pressure_zero(self):
        hist = _make_hist([1013.0, 1012.0, 0.0], var="pressure_hpa")
        assert fingerprint(hist, "pressure_hpa", _physics()) == FaultType.DROPOUT

    def test_temp_zero_is_not_dropout(self):
        """0 °C is a valid temperature, should NOT be DROPOUT."""
        hist = _make_hist([5.0, 3.0, 0.0])
        assert fingerprint(hist, "temp_c", _physics()) != FaultType.DROPOUT


# ---------------------------------------------------------------------------
# Rule 3: OUT_OF_RANGE
# ---------------------------------------------------------------------------

class TestOutOfRange:
    def test_temp_above_range(self):
        """Config range for temp_c is [-30, 55]. 60 °C → OUT_OF_RANGE."""
        hist = _make_hist([30.0, 31.0, 60.0])
        assert fingerprint(hist, "temp_c", _physics()) == FaultType.OUT_OF_RANGE

    def test_temp_below_range(self):
        hist = _make_hist([5.0, 0.0, -35.0])
        assert fingerprint(hist, "temp_c", _physics()) == FaultType.OUT_OF_RANGE

    def test_pressure_above_range(self):
        """Config: pressure_hpa range [870, 1085]. 1095 → OUT_OF_RANGE."""
        hist = _make_hist([1013.0, 1013.0, 1095.0], var="pressure_hpa")
        assert fingerprint(hist, "pressure_hpa", _physics()) == FaultType.OUT_OF_RANGE

    def test_rh_above_range(self):
        """Config: rh_pct range [0, 102]. 107 → OUT_OF_RANGE."""
        hist = _make_hist([60.0, 65.0, 107.0], var="rh_pct")
        assert fingerprint(hist, "rh_pct", _physics()) == FaultType.OUT_OF_RANGE

    def test_boundary_value_is_not_out_of_range(self):
        """Exactly 55.0 °C is the boundary — should NOT be OUT_OF_RANGE."""
        hist = _make_hist([30.0, 31.0, 55.0])
        # 55.0 is not > 55, so it stays within range
        assert fingerprint(hist, "temp_c", _physics()) != FaultType.OUT_OF_RANGE


# ---------------------------------------------------------------------------
# Rule 4: FROZEN
# ---------------------------------------------------------------------------

class TestFrozen:
    def test_temp_flatline_6_steps(self):
        """Config: flatline_steps.temp_c = 6. Six identical readings → FROZEN."""
        hist = _make_hist([30.0, 31.0, 32.0] + [32.0] * 6)
        assert fingerprint(hist, "temp_c", _physics()) == FaultType.FROZEN

    def test_pressure_flatline_12_steps(self):
        """Config: flatline_steps.pressure_hpa = 12."""
        hist = _make_hist([1013.0, 1012.0] + [1012.0] * 12, var="pressure_hpa")
        assert fingerprint(hist, "pressure_hpa", _physics()) == FaultType.FROZEN

    def test_not_enough_identical(self):
        """Five identical temp readings is below the threshold of 6."""
        hist = _make_hist([30.0, 31.0, 33.0] + [32.0] * 5)
        assert fingerprint(hist, "temp_c", _physics()) != FaultType.FROZEN

    def test_rh_saturated_ignored(self):
        """RH at 100 % can sit unchanged legitimately (fog). Not FROZEN."""
        hist = _make_hist([95.0, 98.0] + [100.0] * 6, var="rh_pct")
        assert fingerprint(hist, "rh_pct", _physics()) != FaultType.FROZEN

    def test_rh_lower_flatline_is_frozen(self):
        """RH at 50 % sitting flat for 6 hours IS suspicious."""
        hist = _make_hist([55.0, 52.0] + [50.0] * 6, var="rh_pct")
        assert fingerprint(hist, "rh_pct", _physics()) == FaultType.FROZEN


# ---------------------------------------------------------------------------
# Rule 5: SPIKE
# ---------------------------------------------------------------------------

class TestSpike:
    def test_jump_and_return_3pt(self):
        """Classic 3-point spike: normal → jump → return."""
        # soft_limit for temp_c is 6 °C/h
        hist = _make_hist([30.0, 31.0, 30.5, 45.0, 30.8])
        # At the last row: v[-3:] = [45.0, 30.8] — wait, let me be explicit
        # v[-3] = 30.5, v[-2] = 45.0, v[-1] = 30.8
        # jump_in = |45 - 30.5| = 14.5 >= 6, returned = |30.8 - 30.5| = 0.3 < 6
        assert fingerprint(hist, "temp_c", _physics()) == FaultType.SPIKE

    def test_currently_at_peak(self):
        """Just jumped: normal readings then a big spike at the latest row."""
        hist = _make_hist([30.0, 30.5, 31.0, 30.8, 55.0])
        # delta_1h = |55.0 - 30.8| = 24.2 >= 6
        assert fingerprint(hist, "temp_c", _physics()) == FaultType.SPIKE

    def test_small_change_not_spike(self):
        """A change of 3 °C/h is below the 6 °C soft limit."""
        hist = _make_hist([30.0, 31.0, 32.0, 33.0, 34.0])
        assert fingerprint(hist, "temp_c", _physics()) != FaultType.SPIKE

    def test_pressure_spike(self):
        """Pressure soft limit is 3 hPa/h."""
        hist = _make_hist([1013.0, 1013.0, 1013.0, 1013.0, 1020.0],
                          var="pressure_hpa")
        assert fingerprint(hist, "pressure_hpa", _physics()) == FaultType.SPIKE


# ---------------------------------------------------------------------------
# Rule 6: NOISE
# ---------------------------------------------------------------------------

class TestNoise:
    def test_high_variability_no_trend(self):
        """Inject NOISE-like pattern: high std in last 6 hours, no trend."""
        # Build a stable baseline followed by noisy readings
        rng = np.random.default_rng(42)
        baseline = [30.0 + 0.3 * rng.standard_normal() for _ in range(20)]
        # NOISE: 6 hours of high variability (4× normal)
        noisy = [30.0 + 4.0 * rng.standard_normal() for _ in range(6)]
        hist = _make_hist(baseline + noisy)
        result = fingerprint(hist, "temp_c", _physics())
        assert result == FaultType.NOISE

    def test_trending_is_not_noise(self):
        """Steadily rising values have high std but a clear trend — not NOISE."""
        # 20 stable hours, then 6 hours of a 1°C/h rise (total 5°C)
        baseline = [30.0 + 0.2 * np.random.default_rng(42).standard_normal() for _ in range(20)]
        trending = [30.0 + 1.0 * i for i in range(6)]
        hist = _make_hist(baseline + trending)
        result = fingerprint(hist, "temp_c", _physics())
        # A pure linear trend has zero residual std → should NOT be NOISE
        assert result != FaultType.NOISE


# ---------------------------------------------------------------------------
# Rule 7: DRIFT
# ---------------------------------------------------------------------------

class TestDrift:
    def test_gradual_offset_with_spatial_disagreement(self):
        """Slow drift: 0.2 °C/h over 24 hours, total 4.8 °C, no big single step."""
        # 24 hours of normal variation before the drift + 24 hours of drifting
        rng = np.random.default_rng(42)
        normal = [30.0 + 0.5 * rng.standard_normal() for _ in range(24)]
        drifting = [30.0 + 0.2 * i for i in range(24)]  # ends at 34.6
        hist = _make_hist(normal + drifting)
        spatial = _spatial(z=5.0, consistent=False)  # neighbours disagree
        result = fingerprint(hist, "temp_c", _physics(), spatial=spatial)
        assert result == FaultType.DRIFT

    def test_drift_not_detected_without_spatial(self):
        """Without spatial context, DRIFT cannot be confirmed."""
        normal = [30.0] * 24
        drifting = [30.0 + 0.2 * i for i in range(24)]
        hist = _make_hist(normal + drifting)
        result = fingerprint(hist, "temp_c", _physics(), spatial=None)
        assert result != FaultType.DRIFT

    def test_drift_not_detected_when_neighbours_agree(self):
        """Spatial z below threshold → not drift."""
        normal = [30.0] * 24
        drifting = [30.0 + 0.2 * i for i in range(24)]
        hist = _make_hist(normal + drifting)
        spatial = _spatial(z=1.0, consistent=True)  # neighbours agree
        result = fingerprint(hist, "temp_c", _physics(), spatial=spatial)
        assert result != FaultType.DRIFT


# ---------------------------------------------------------------------------
# Rule 8: Fallback SPIKE
# ---------------------------------------------------------------------------

class TestFallbackSpike:
    def test_sustained_spike_hour2(self):
        """Hour 2 of a 2-hour spike: delta_1h is 0 but deviation from baseline
        is still large → should be caught by the fallback rule."""
        # 10 hours of normal, then 2 hours at 55°C (spiked)
        normal = [30.0, 30.5, 31.0, 30.8, 30.2, 30.5, 31.0, 30.8, 30.2, 30.5]
        hist = _make_hist(normal + [55.0, 55.0])
        result = fingerprint(hist, "temp_c", _physics())
        assert result == FaultType.SPIKE


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_history(self):
        hist = _make_hist([])
        assert fingerprint(hist, "temp_c", _physics()) is None

    def test_single_row(self):
        hist = _make_hist([30.0])
        assert fingerprint(hist, "temp_c", _physics()) is None

    def test_normal_reading(self):
        """A normal reading with no anomaly → None."""
        hist = _make_hist([30.0, 30.5, 31.0, 30.8, 30.2])
        assert fingerprint(hist, "temp_c", _physics()) is None

    def test_all_variables(self):
        """Fingerprint works for every variable in schemas.VARIABLES."""
        from skyguard.schemas import VARIABLES
        for var in VARIABLES:
            hist = _make_hist([30.0, 30.5, 31.0], var=var)
            result = fingerprint(hist, var, _physics())
            assert result is None or result in FaultType.FAULTS


# ---------------------------------------------------------------------------
# Demo scenarios
# ---------------------------------------------------------------------------

class TestDemoScenarios:
    def test_55c_spike_demo(self):
        """The PS's 55 °C example: should be classified as SPIKE."""
        # Normal Delhi temperature then sudden spike
        hist = _make_hist([31.0, 32.0, 31.5, 31.8, 32.0, 55.0])
        result = fingerprint(hist, "temp_c", _physics(hard=True, rule_ids=["RANGE"]))
        # Note: 55.0 is at the boundary of physics.range [-30, 55], not outside
        # So it should be caught as SPIKE via delta_1h = |55.0 - 32.0| = 23.0 >= 6
        assert result == FaultType.SPIKE

    def test_frozen_sensor_demo(self):
        """Frozen sensor stuck at 32.0 °C for 6 hours."""
        hist = _make_hist([30.0, 31.0, 32.0] + [32.0] * 6)
        result = fingerprint(hist, "temp_c", _physics())
        assert result == FaultType.FROZEN

    def test_storm_no_fault_type(self):
        """Storm readings: temperature drops naturally, all values plausible,
        no flatline, no huge single-step jump → should return None."""
        # Realistic storm: gentle 1.5°C drops per hour for 4 hours, then recovery
        # With natural baseline variation to avoid triggering NOISE
        rng = np.random.default_rng(42)
        normal = [30.0 + 0.5 * rng.standard_normal() for _ in range(20)]
        storm = [30.0, 28.5, 27.0, 25.5, 24.0, 24.5, 25.5, 26.5]
        hist = _make_hist(normal + storm)
        result = fingerprint(hist, "temp_c", _physics())
        # 1.5°C/h is below the 6°C/h soft limit → not SPIKE, and the storm
        # is a steady trend → not NOISE either
        assert result is None


# ---------------------------------------------------------------------------
# Helper: _recent_baseline
# ---------------------------------------------------------------------------

class TestRecentBaseline:
    def test_normal_case(self):
        col = pd.Series([30.0, 31.0, 32.0, 30.5, 31.0, 55.0, 55.0, 55.0])
        result = _recent_baseline(col, exclude_last=3)
        # Uses indices 0..4 (excluding last 3), median of [30.0, 31.0, 32.0, 30.5, 31.0]
        assert result == pytest.approx(31.0)

    def test_too_short(self):
        col = pd.Series([30.0, 55.0, 55.0])
        assert _recent_baseline(col, exclude_last=3) is None

    def test_with_nans(self):
        col = pd.Series([30.0, np.nan, 31.0, np.nan, 32.0, 55.0, 55.0, 55.0])
        result = _recent_baseline(col, exclude_last=3)
        # Valid values: [30.0, 31.0, 32.0], median = 31.0
        assert result == pytest.approx(31.0)
