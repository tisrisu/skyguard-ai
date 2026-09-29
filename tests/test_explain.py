"""Tests for M2 task L5 — plain-English explanations.

Verifies explain() output structure, priority ordering, reason caps,
SHAP integration (mocked), and edge cases.
"""

from __future__ import annotations
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from skyguard.fusion.explain import (
    MAX_REASONS,
    SHAP_TOP_N,
    _shap_contributions,
    explain,
)
from skyguard.models.features import FEATURE_LABELS, FEATURE_NAMES


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _physics(reasons=None, hard=False, soft=False, rule_ids=None):
    return {
        "hard": hard,
        "soft": soft,
        "rule_ids": rule_ids or [],
        "reasons": reasons or [],
        "fault_hint": None,
    }


def _spatial(reason="", z=0.0, n=5, consistent=True):
    return {"z": z, "n": n, "consistent": consistent, "reason": reason}


def _features():
    """Dummy feature vector matching FEATURE_NAMES length."""
    return np.zeros(len(FEATURE_NAMES), dtype=np.float64)


def _mock_model(shap_values=None):
    """Create a mock IFModel with a mock IsolationForest and StandardScaler."""
    model = MagicMock()
    model.models = {"temp_c": MagicMock(), "pressure_hpa": MagicMock(), "rh_pct": MagicMock()}
    # StandardScaler that returns data as-is
    for var in model.models:
        scaler = MagicMock()
        scaler.transform.return_value = np.zeros((1, len(FEATURE_NAMES)))
        model.scalers = {var: scaler for var in model.models}

    return model


# ---------------------------------------------------------------------------
# Output structure
# ---------------------------------------------------------------------------

class TestOutputStructure:
    def test_returns_tuple_of_two(self):
        result = explain(_physics(), _spatial(), 0.5, _features(), "temp_c")
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_reasons_is_list_of_strings(self):
        reasons, _ = explain(_physics(reasons=["Test reason"]), _spatial(), 0.5, _features(), "temp_c")
        assert isinstance(reasons, list)
        assert all(isinstance(r, str) for r in reasons)

    def test_shap_top_is_list_of_tuples(self):
        _, shap_top = explain(_physics(), _spatial(), 0.5, _features(), "temp_c")
        assert isinstance(shap_top, list)
        # Without a model, shap_top should be empty
        assert shap_top == []

    def test_at_least_one_reason(self):
        """Every non-NORMAL result must have >= 1 reason."""
        reasons, _ = explain({}, {}, 0.5, _features(), "temp_c")
        assert len(reasons) >= 1

    def test_max_4_reasons(self):
        """Even with many inputs, should never exceed MAX_REASONS."""
        physics = _physics(reasons=[
            "Reason 1", "Reason 2", "Reason 3", "Reason 4", "Reason 5"
        ])
        spatial = _spatial(reason="Spatial reason too")
        reasons, _ = explain(physics, spatial, 0.99, _features(), "temp_c")
        assert len(reasons) <= MAX_REASONS


# ---------------------------------------------------------------------------
# Priority ordering
# ---------------------------------------------------------------------------

class TestPriorityOrdering:
    def test_physics_first(self):
        """Physics reasons should appear before spatial."""
        physics = _physics(reasons=["Dewpoint exceeds physical max"])
        spatial = _spatial(reason="Neighbours disagree")
        reasons, _ = explain(physics, spatial, 0.5, _features(), "temp_c")
        assert reasons[0] == "Dewpoint exceeds physical max"
        assert reasons[1] == "Neighbours disagree"

    def test_spatial_after_physics(self):
        physics = _physics(reasons=["Physics reason A", "Physics reason B"])
        spatial = _spatial(reason="5 of 5 neighbours show normal temperature")
        reasons, _ = explain(physics, spatial, 0.5, _features(), "temp_c")
        assert reasons[0] == "Physics reason A"
        assert reasons[1] == "Physics reason B"
        assert reasons[2] == "5 of 5 neighbours show normal temperature"

    def test_no_duplicate_reasons(self):
        """Same reason from physics and spatial should not be duplicated."""
        same_reason = "Temperature is abnormally high"
        physics = _physics(reasons=[same_reason])
        spatial = _spatial(reason=same_reason)
        reasons, _ = explain(physics, spatial, 0.5, _features(), "temp_c")
        assert reasons.count(same_reason) == 1


# ---------------------------------------------------------------------------
# Physics reasons
# ---------------------------------------------------------------------------

class TestPhysicsReasons:
    def test_multiple_physics_reasons(self):
        physics = _physics(reasons=[
            "Implied dewpoint 50.4 °C exceeds the physical maximum (~35 °C)",
            "Temperature rose 23.1 °C in 1 h while pressure and humidity barely changed",
        ])
        reasons, _ = explain(physics, _spatial(), 0.5, _features(), "temp_c")
        assert len(reasons) >= 2
        assert "Implied dewpoint 50.4 °C exceeds the physical maximum (~35 °C)" in reasons

    def test_empty_physics_reasons(self):
        """No physics reasons → should not crash."""
        reasons, _ = explain(_physics(reasons=[]), _spatial(), 0.5, _features(), "temp_c")
        assert isinstance(reasons, list)

    def test_empty_strings_filtered(self):
        """Empty strings in reasons list should be ignored."""
        physics = _physics(reasons=["", "Valid reason", ""])
        reasons, _ = explain(physics, _spatial(), 0.5, _features(), "temp_c")
        assert "" not in reasons
        assert "Valid reason" in reasons


# ---------------------------------------------------------------------------
# Spatial reason
# ---------------------------------------------------------------------------

class TestSpatialReason:
    def test_spatial_reason_included(self):
        spatial = _spatial(reason="5 of 5 neighbouring stations show normal temperature (z = 9.4)")
        reasons, _ = explain(_physics(), spatial, 0.5, _features(), "temp_c")
        assert "5 of 5 neighbouring stations show normal temperature (z = 9.4)" in reasons

    def test_empty_spatial_reason(self):
        """Empty spatial reason string → should not add empty entry."""
        spatial = _spatial(reason="")
        reasons, _ = explain(_physics(reasons=["Physics only"]), spatial, 0.5, _features(), "temp_c")
        assert "" not in reasons

    def test_no_spatial_dict(self):
        """Empty dict for spatial → no crash."""
        reasons, _ = explain(_physics(reasons=["Test"]), {}, 0.5, _features(), "temp_c")
        assert "Test" in reasons


# ---------------------------------------------------------------------------
# SHAP contributions (mocked)
# ---------------------------------------------------------------------------

class TestShapContributions:
    def test_no_model_returns_empty(self):
        result = _shap_contributions(None, _features(), "temp_c")
        assert result == []

    def test_wrong_var_returns_empty(self):
        model = _mock_model()
        result = _shap_contributions(model, _features(), "nonexistent_var")
        assert result == []

    @patch("skyguard.fusion.explain._get_explainer")
    def test_shap_top_3_format(self, mock_get_explainer):
        """With a mocked explainer, should return top 3 contributions."""
        mock_explainer = MagicMock()
        # Create fake SHAP values — one per feature
        fake_sv = np.array([[0.1, 0.41, 0.05, 0.08, 0.03, 0.27, 0.02, 0.12, 0.06, 0.01, 0.004]])
        mock_explainer.shap_values.return_value = fake_sv
        mock_get_explainer.return_value = mock_explainer

        model = _mock_model()
        result = _shap_contributions(model, _features(), "temp_c")

        assert len(result) == SHAP_TOP_N
        # Should be sorted by |value| descending
        assert result[0][1] >= result[1][1] >= result[2][1]
        # Labels should be from FEATURE_LABELS
        for label, val in result:
            assert label in FEATURE_LABELS.values()
            assert isinstance(val, float)

    @patch("skyguard.fusion.explain._get_explainer")
    def test_shap_reasons_in_explain(self, mock_get_explainer):
        """SHAP reasons should appear as 'ML flag: ...' in the reasons list."""
        mock_explainer = MagicMock()
        fake_sv = np.array([[0.0, 0.41, 0.0, 0.0, 0.0, 0.27, 0.0, 0.12, 0.0, 0.0, 0.0]])
        mock_explainer.shap_values.return_value = fake_sv
        mock_get_explainer.return_value = mock_explainer

        model = _mock_model()
        reasons, shap_top = explain(_physics(), _spatial(), 0.99, _features(), "temp_c", model=model)

        # Should have SHAP-derived reasons
        ml_reasons = [r for r in reasons if r.startswith("ML flag:")]
        assert len(ml_reasons) >= 1
        # shap_top should have entries
        assert len(shap_top) > 0

    @patch("skyguard.fusion.explain._get_explainer")
    def test_explainer_failure_graceful(self, mock_get_explainer):
        """If SHAP throws, should return empty and not crash."""
        mock_get_explainer.return_value = None  # explainer init failed

        model = _mock_model()
        result = _shap_contributions(model, _features(), "temp_c")
        assert result == []


# ---------------------------------------------------------------------------
# Fallback guarantee
# ---------------------------------------------------------------------------

class TestFallbackGuarantee:
    def test_high_ml_score_fallback(self):
        """No physics, no spatial, no model → ML score fallback."""
        reasons, _ = explain({}, {}, 0.99, _features(), "temp_c")
        assert len(reasons) == 1
        assert "ML anomaly score unusually high" in reasons[0]
        assert "99%" in reasons[0]

    def test_low_ml_score_fallback(self):
        """No reasons at all, low ML score → generic fallback."""
        reasons, _ = explain({}, {}, 0.3, _features(), "temp_c")
        assert len(reasons) == 1
        assert reasons[0] == "Flagged by the detection pipeline"

    def test_fallback_not_used_when_reasons_exist(self):
        """If physics gives a reason, no fallback needed."""
        reasons, _ = explain(_physics(reasons=["Real reason"]), {}, 0.3, _features(), "temp_c")
        assert "Flagged by the detection pipeline" not in reasons
        assert "ML anomaly score" not in reasons[0]


# ---------------------------------------------------------------------------
# Demo scenarios (matching mock_results.json)
# ---------------------------------------------------------------------------

class TestDemoScenarios:
    def test_55c_spike_reasons(self):
        """The 55 °C spike demo: physics + spatial reasons, no SHAP (model=None)."""
        physics = _physics(
            hard=True,
            rule_ids=["DEWPOINT_MAX", "RATE_OF_CHANGE"],
            reasons=[
                "Implied dewpoint 50.4 °C exceeds the physical maximum (~35 °C)",
                "Temperature rose 23.1 °C in 1 h while pressure and humidity barely changed",
            ],
        )
        spatial = _spatial(
            reason="5 of 5 neighbouring stations show normal temperature (z = 9.4)",
            z=9.4,
            consistent=False,
        )
        reasons, shap_top = explain(physics, spatial, 0.998, _features(), "temp_c")

        # Should have the physics reasons first, then spatial
        assert reasons[0].startswith("Implied dewpoint")
        assert reasons[1].startswith("Temperature rose")
        assert reasons[2].startswith("5 of 5")
        assert len(reasons) <= MAX_REASONS
        # No model → no SHAP
        assert shap_top == []

    def test_storm_genuine_event_reasons(self):
        """Storm: spatial says neighbours agree. Still get physics reasons."""
        physics = _physics(
            soft=True,
            reasons=["Temperature fell 8.2 °C in 1 h"],
        )
        spatial = _spatial(
            reason="4 of 5 neighbouring stations show the same drop within 2 h (z = 0.7)",
            z=0.7,
            consistent=True,
        )
        reasons, _ = explain(physics, spatial, 0.995, _features(), "temp_c")
        assert "Temperature fell 8.2 °C in 1 h" in reasons
        assert any("neighbouring stations" in r for r in reasons)

    def test_frozen_sensor_reasons(self):
        """Frozen sensor: physics reason about consecutive identical readings."""
        physics = _physics(
            hard=True,
            reasons=[
                "Humidity has reported exactly 64 % for 9 consecutive hours",
                "Temperature changed 5.3 °C over the same period, so humidity should have moved",
            ],
        )
        reasons, _ = explain(physics, _spatial(), 0.97, _features(), "rh_pct")
        assert len(reasons) >= 2
        assert reasons[0].startswith("Humidity has reported")

    def test_all_variables_work(self):
        """explain() should work for all three variables."""
        from skyguard.schemas import VARIABLES
        for var in VARIABLES:
            reasons, shap_top = explain(
                _physics(reasons=["Test"]),
                _spatial(),
                0.5,
                _features(),
                var,
            )
            assert len(reasons) >= 1
