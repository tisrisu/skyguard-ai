"""Tests for M2 task L3 — decision fusion.

Verifies decide() against every rule in the docstring, edge cases, and the
key demo scenarios (55°C spike, storm, frozen sensor, normal reading).
"""

import math

import pytest

from skyguard.fusion.decide import _severity, decide
from skyguard.schemas import Severity, Status


# ---------------------------------------------------------------------------
# Helpers for constructing inputs
# ---------------------------------------------------------------------------

def physics(hard=False, soft=False, rule_ids=None, reasons=None):
    return {
        "hard": hard,
        "soft": soft,
        "rule_ids": rule_ids or [],
        "reasons": reasons or [],
    }


def spatial(z=0.0, n=5, consistent=True, reason=""):
    return {"z": z, "n": n, "consistent": consistent, "reason": reason}


# Shared config matching config.yaml
CFG = {
    "ml": {"high": 0.99, "suspect": 0.95, "n_estimators": 200, "max_samples": 256},
    "severity_sigma": [3, 6, 10],
}


# ---------------------------------------------------------------------------
# Severity helper
# ---------------------------------------------------------------------------

class TestSeverity:
    def test_low(self):
        assert _severity(2.5, [3, 6, 10], physics_hard=False) == Severity.LOW

    def test_medium(self):
        assert _severity(4.0, [3, 6, 10], physics_hard=False) == Severity.MEDIUM

    def test_high(self):
        assert _severity(8.0, [3, 6, 10], physics_hard=False) == Severity.HIGH

    def test_critical_by_sigma(self):
        assert _severity(12.0, [3, 6, 10], physics_hard=False) == Severity.CRITICAL

    def test_boundary_3(self):
        """Exactly 3.0 → MEDIUM (≥ 3)."""
        assert _severity(3.0, [3, 6, 10], physics_hard=False) == Severity.MEDIUM

    def test_boundary_6(self):
        assert _severity(6.0, [3, 6, 10], physics_hard=False) == Severity.HIGH

    def test_boundary_10(self):
        assert _severity(10.0, [3, 6, 10], physics_hard=False) == Severity.CRITICAL

    def test_negative_dev(self):
        """Negative dev_sigma uses absolute value."""
        assert _severity(-8.0, [3, 6, 10], physics_hard=False) == Severity.HIGH

    def test_physics_hard_overrides(self):
        """Physics hard → CRITICAL even if dev_sigma is tiny."""
        assert _severity(0.5, [3, 6, 10], physics_hard=True) == Severity.CRITICAL

    def test_nan_dev(self):
        """NaN dev_sigma → LOW (treated as 0)."""
        assert _severity(float("nan"), [3, 6, 10], physics_hard=False) == Severity.LOW

    def test_inf_dev(self):
        """Inf dev_sigma → LOW (treated as 0 after isfinite check)."""
        assert _severity(float("inf"), [3, 6, 10], physics_hard=False) == Severity.LOW


# ---------------------------------------------------------------------------
# Rule 1: Physics hard violation → SENSOR_FAULT
# ---------------------------------------------------------------------------

class TestRule1PhysicsHard:
    def test_status_is_sensor_fault(self):
        status, conf, sev = decide(
            physics(hard=True, reasons=["55°C exceeds range"]),
            spatial(z=9.4, consistent=False),
            ml_score=0.998, dev_sigma=12.5, cfg=CFG,
        )
        assert status == Status.SENSOR_FAULT

    def test_severity_is_critical(self):
        """Physics hard → always CRITICAL."""
        _, _, sev = decide(
            physics(hard=True),
            spatial(consistent=True),
            ml_score=0.0, dev_sigma=0.5, cfg=CFG,
        )
        assert sev == Severity.CRITICAL

    def test_confidence_range(self):
        """Confidence should be in [0.95, 0.99]."""
        _, conf_low, _ = decide(
            physics(hard=True), spatial(), ml_score=0.0, dev_sigma=5.0, cfg=CFG,
        )
        _, conf_high, _ = decide(
            physics(hard=True), spatial(), ml_score=1.0, dev_sigma=5.0, cfg=CFG,
        )
        assert 0.95 <= conf_low <= 0.99
        assert 0.95 <= conf_high <= 0.99
        assert conf_high > conf_low  # ML agreement increases confidence

    def test_physics_hard_overrides_neighbours_agree(self):
        """Even if neighbours agree, physics hard → SENSOR_FAULT."""
        status, _, _ = decide(
            physics(hard=True),
            spatial(consistent=True),
            ml_score=0.5, dev_sigma=5.0, cfg=CFG,
        )
        assert status == Status.SENSOR_FAULT


# ---------------------------------------------------------------------------
# Rule 2a: ML high + neighbours disagree → SENSOR_FAULT
# ---------------------------------------------------------------------------

class TestRule2aMLHighDisagree:
    def test_status(self):
        status, _, _ = decide(
            physics(), spatial(z=8.0, consistent=False),
            ml_score=0.998, dev_sigma=10.5, cfg=CFG,
        )
        assert status == Status.SENSOR_FAULT

    def test_confidence_formula(self):
        """confidence = 0.6 + 0.4 * mean(ml, min(|z|/6, 1))"""
        ml = 0.998
        z = 8.0
        z_signal = min(z / 6.0, 1.0)
        expected_conf = 0.6 + 0.4 * (ml + z_signal) / 2.0
        _, conf, _ = decide(
            physics(), spatial(z=z, consistent=False),
            ml_score=ml, dev_sigma=5.0, cfg=CFG,
        )
        assert abs(conf - round(expected_conf, 4)) < 1e-6

    def test_physics_soft_also_triggers(self):
        """Rule 2a triggers on physics soft even if consistent is True."""
        status, _, _ = decide(
            physics(soft=True), spatial(consistent=True),
            ml_score=0.995, dev_sigma=5.0, cfg=CFG,
        )
        assert status == Status.SENSOR_FAULT


# ---------------------------------------------------------------------------
# Rule 2b: ML high + neighbours agree → GENUINE_EVENT
# ---------------------------------------------------------------------------

class TestRule2bGenuineEvent:
    def test_status(self):
        status, _, _ = decide(
            physics(), spatial(z=1.5, consistent=True),
            ml_score=0.995, dev_sigma=5.0, cfg=CFG,
        )
        assert status == Status.GENUINE_EVENT

    def test_confidence_above_0_7(self):
        _, conf, _ = decide(
            physics(), spatial(consistent=True),
            ml_score=0.995, dev_sigma=5.0, cfg=CFG,
        )
        assert conf >= 0.7

    def test_severity_from_dev_sigma(self):
        """Storm: dev_sigma might be 4 → MEDIUM severity (not CRITICAL)."""
        _, _, sev = decide(
            physics(), spatial(consistent=True),
            ml_score=0.995, dev_sigma=4.0, cfg=CFG,
        )
        assert sev == Severity.MEDIUM


# ---------------------------------------------------------------------------
# Rule 2c: ML high + no neighbours → SUSPECT
# ---------------------------------------------------------------------------

class TestRule2cNoNeighbours:
    def test_status(self):
        status, _, _ = decide(
            physics(), spatial(consistent=None),
            ml_score=0.995, dev_sigma=5.0, cfg=CFG,
        )
        assert status == Status.SUSPECT

    def test_confidence_at_most_0_6(self):
        _, conf, _ = decide(
            physics(), spatial(consistent=None),
            ml_score=0.999, dev_sigma=5.0, cfg=CFG,
        )
        assert conf <= 0.6


# ---------------------------------------------------------------------------
# Rule 3: ML suspect + neighbours disagree → SUSPECT
# ---------------------------------------------------------------------------

class TestRule3Suspect:
    def test_status(self):
        status, _, _ = decide(
            physics(), spatial(z=4.0, consistent=False),
            ml_score=0.96, dev_sigma=4.0, cfg=CFG,
        )
        assert status == Status.SUSPECT

    def test_confidence_range(self):
        """Confidence should be between 0.3 and 0.5."""
        _, conf_lo, _ = decide(
            physics(), spatial(consistent=False),
            ml_score=0.95, dev_sigma=4.0, cfg=CFG,
        )
        _, conf_hi, _ = decide(
            physics(), spatial(consistent=False),
            ml_score=0.989, dev_sigma=4.0, cfg=CFG,
        )
        assert 0.3 <= conf_lo <= 0.5
        assert 0.3 <= conf_hi <= 0.5
        assert conf_hi > conf_lo

    def test_not_triggered_when_neighbours_agree(self):
        """ML suspect but neighbours agree → NORMAL (not suspect)."""
        status, _, _ = decide(
            physics(), spatial(consistent=True),
            ml_score=0.96, dev_sigma=2.0, cfg=CFG,
        )
        assert status == Status.NORMAL


# ---------------------------------------------------------------------------
# Rule 4: Everything else → NORMAL
# ---------------------------------------------------------------------------

class TestRule4Normal:
    def test_normal_reading(self):
        status, conf, sev = decide(
            physics(), spatial(consistent=True),
            ml_score=0.3, dev_sigma=0.5, cfg=CFG,
        )
        assert status == Status.NORMAL
        assert conf == 0.0
        assert sev is None

    def test_moderate_ml_but_neighbours_agree(self):
        """ML at 0.96 (above suspect) but neighbours agree → NORMAL."""
        status, _, sev = decide(
            physics(), spatial(consistent=True),
            ml_score=0.96, dev_sigma=1.0, cfg=CFG,
        )
        assert status == Status.NORMAL
        assert sev is None

    def test_low_ml_and_neighbours_disagree(self):
        """ML below suspect → NORMAL even if neighbours disagree."""
        status, _, _ = decide(
            physics(), spatial(consistent=False),
            ml_score=0.5, dev_sigma=1.0, cfg=CFG,
        )
        assert status == Status.NORMAL


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_nan_ml_score(self):
        """NaN ml_score → treated as 0.0 → NORMAL."""
        status, _, _ = decide(
            physics(), spatial(), ml_score=float("nan"), dev_sigma=1.0, cfg=CFG,
        )
        assert status == Status.NORMAL

    def test_inf_ml_score(self):
        """Inf ml_score → treated as 0.0 → NORMAL."""
        status, _, _ = decide(
            physics(), spatial(), ml_score=float("inf"), dev_sigma=1.0, cfg=CFG,
        )
        assert status == Status.NORMAL

    def test_empty_physics_and_spatial(self):
        """Minimal dicts with just defaults → NORMAL."""
        status, _, _ = decide({}, {}, ml_score=0.3, dev_sigma=1.0, cfg=CFG)
        assert status == Status.NORMAL

    def test_output_types(self):
        """Return types are correct."""
        status, conf, sev = decide(
            physics(), spatial(), ml_score=0.5, dev_sigma=1.0, cfg=CFG,
        )
        assert isinstance(status, str)
        assert isinstance(conf, float)
        assert sev is None or isinstance(sev, str)

    def test_all_statuses_are_valid(self):
        """Every returned status is in Status.ALL."""
        scenarios = [
            (physics(hard=True), spatial(), 0.9, 12.0),        # Rule 1
            (physics(), spatial(consistent=False), 0.995, 7.0), # Rule 2a
            (physics(), spatial(consistent=True), 0.995, 5.0),  # Rule 2b
            (physics(), spatial(consistent=None), 0.995, 5.0),  # Rule 2c
            (physics(), spatial(consistent=False), 0.96, 4.0),  # Rule 3
            (physics(), spatial(consistent=True), 0.3, 1.0),    # Rule 4
        ]
        for p, s, ml, ds in scenarios:
            status, conf, sev = decide(p, s, ml, ds, cfg=CFG)
            assert status in Status.ALL, f"Invalid status: {status}"
            assert 0.0 <= conf <= 1.0, f"Confidence out of range: {conf}"
            if status == Status.NORMAL:
                assert sev is None
            else:
                assert sev in Severity.ALL, f"Invalid severity: {sev}"


# ---------------------------------------------------------------------------
# Demo scenarios (the 5 moments from the video)
# ---------------------------------------------------------------------------

class TestDemoScenarios:
    def test_55c_spike(self):
        """The PS's 55°C example: SENSOR_FAULT, CRITICAL, high confidence."""
        status, conf, sev = decide(
            physics(hard=True, soft=True,
                    rule_ids=["RANGE", "DEWPOINT_MAX"],
                    reasons=["55°C exceeds physical maximum",
                             "Implied dewpoint 50.8°C exceeds ~35°C"]),
            spatial(z=9.4, n=5, consistent=False,
                    reason="only this station shows the anomaly"),
            ml_score=0.998, dev_sigma=12.5, cfg=CFG,
        )
        assert status == Status.SENSOR_FAULT
        assert conf >= 0.95
        assert sev == Severity.CRITICAL

    def test_storm_genuine_event(self):
        """Storm across multiple stations: GENUINE_EVENT, NOT sensor fault."""
        status, conf, sev = decide(
            physics(hard=False, soft=False),
            spatial(z=1.2, n=5, consistent=True,
                    reason="4 of 5 neighbours show similar drop"),
            ml_score=0.995, dev_sigma=4.5, cfg=CFG,
        )
        assert status == Status.GENUINE_EVENT
        assert status != Status.SENSOR_FAULT  # this is the key demo point
        assert sev == Severity.MEDIUM

    def test_frozen_sensor(self):
        """Frozen sensor: flagged by ML, neighbours disagree."""
        status, _, sev = decide(
            physics(hard=False, soft=True, rule_ids=["FLATLINE"]),
            spatial(z=3.5, n=5, consistent=False),
            ml_score=0.993, dev_sigma=3.5, cfg=CFG,
        )
        assert status == Status.SENSOR_FAULT
        assert sev in (Severity.MEDIUM, Severity.HIGH)

    def test_normal_green(self):
        """Normal operating conditions: NORMAL, no alarm."""
        status, conf, sev = decide(
            physics(hard=False, soft=False),
            spatial(z=0.3, n=5, consistent=True),
            ml_score=0.15, dev_sigma=0.8, cfg=CFG,
        )
        assert status == Status.NORMAL
        assert conf == 0.0
        assert sev is None
