"""Plain-English reasons for a decision.

Owner: M2

Order: physics reasons first (most decisive), then the neighbour reason,
then the top SHAP features translated with models.features.FEATURE_LABELS.
At most 4 reasons. Only called for non-NORMAL results (SHAP is slower than detection).
"""

from __future__ import annotations

import logging

import numpy as np

from skyguard.models.features import FEATURE_LABELS, FEATURE_NAMES

logger = logging.getLogger(__name__)

MAX_REASONS = 4
SHAP_TOP_N = 3

# Cache one TreeExplainer per (model_id, var) so we don't rebuild each call.
_explainer_cache: dict[tuple[int, str], object] = {}


# ---------------------------------------------------------------------------
# SHAP helpers
# ---------------------------------------------------------------------------

def _get_explainer(model, var: str):
    """Lazily build and cache a ``shap.TreeExplainer`` for one variable's
    IsolationForest.  Returns ``None`` on failure (logged, not raised)."""
    key = (id(model), var)
    if key not in _explainer_cache:
        try:
            import shap
            _explainer_cache[key] = shap.TreeExplainer(model.models[var])
        except Exception as exc:
            logger.warning("SHAP TreeExplainer init failed for %s: %s", var, exc)
            _explainer_cache[key] = None
    return _explainer_cache[key]


def _shap_contributions(model, features: np.ndarray, var: str,
                        ) -> list[tuple[str, float]]:
    """Top-3 SHAP feature contributions as ``[(plain-English label, |value|), ...]``.

    Returns an empty list if SHAP is unavailable, the model is ``None``,
    or computation fails for any reason.
    """
    if model is None or var not in getattr(model, "models", {}):
        return []

    explainer = _get_explainer(model, var)
    if explainer is None:
        return []

    try:
        # Scale features the same way as during training
        X = model.scalers[var].transform(features.reshape(1, -1))
        shap_values = explainer.shap_values(X)

        # shap_values shape: (1, n_features) — contribution per feature
        sv = np.asarray(shap_values).flatten()

        # Pair with feature names, sort by |contribution|, keep top N
        pairs = sorted(
            zip(FEATURE_NAMES, sv),
            key=lambda p: abs(p[1]),
            reverse=True,
        )

        result = []
        for fname, val in pairs[:SHAP_TOP_N]:
            label = FEATURE_LABELS.get(fname, fname)
            result.append((label, round(float(abs(val)), 4)))
        return result

    except Exception as exc:
        logger.warning("SHAP computation failed for %s: %s", var, exc)
        return []


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

def explain(physics: dict, spatial: dict, ml_score: float, features: np.ndarray, var: str,
            model=None) -> tuple[list[str], list[tuple[str, float]]]:
    """Return ``(reasons, shap_top)`` for a non-NORMAL result.

    Parameters
    ----------
    physics : dict
        From ``check_physics()``:
        ``{"hard": bool, "soft": bool, "rule_ids": [...], "reasons": [...]}``.
        An empty dict ``{}`` is safe.
    spatial : dict
        From ``spatial_check()``:
        ``{"z": float, "n": int, "consistent": bool|None, "reason": str}``.
        An empty dict ``{}`` is safe.
    ml_score : float
        Percentile from ``IFModel.score()`` — 0 = normal, 1 = most anomalous.
    features : np.ndarray
        Feature vector from ``make_features()`` — shape ``(n_features,)``.
    var : str
        Variable name (one of ``schemas.VARIABLES``).
    model : IFModel | None
        Fitted ``IFModel``.  If ``None``, SHAP is skipped and only
        physics + spatial reasons are returned.

    Returns
    -------
    ``(reasons, shap_top)`` where

    reasons
        ``list[str]`` — up to 4 plain-English reasons, ordered:
        physics first (most decisive) → spatial → SHAP-derived.
    shap_top
        ``list[tuple[str, float]]`` — top 3 SHAP feature contributions
        as ``(plain-English label, |contribution|)``.  Empty when SHAP
        is unavailable or the model is ``None``.
    """
    reasons: list[str] = []

    # --- 1. Physics reasons (highest priority, most interpretable) --------
    for r in physics.get("reasons", []):
        if r and r not in reasons:
            reasons.append(r)

    # --- 2. Spatial / neighbour reason ------------------------------------
    spatial_reason = spatial.get("reason", "")
    if spatial_reason and spatial_reason not in reasons:
        reasons.append(spatial_reason)

    # --- 3. SHAP-derived reasons (translated to plain English) ------------
    shap_top = _shap_contributions(model, features, var)
    for label, contrib in shap_top:
        if len(reasons) >= MAX_REASONS:
            break
        reason = f"ML flag: {label} (contribution {contrib:.2f})"
        if reason not in reasons:
            reasons.append(reason)

    # --- Truncate to MAX_REASONS ------------------------------------------
    reasons = reasons[:MAX_REASONS]

    # --- Guarantee at least 1 reason for every non-NORMAL result ----------
    if not reasons:
        if ml_score >= 0.95:
            reasons.append(f"ML anomaly score unusually high ({ml_score:.0%})")
        else:
            reasons.append("Flagged by the detection pipeline")

    return reasons, shap_top

