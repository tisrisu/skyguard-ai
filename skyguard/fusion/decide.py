"""Combine physics, neighbours and ML into a final status.

Owner: M2

Starting rules (tune thresholds on the validation split):

  ml = ML percentile score (0..1)
  1. physics hard violation                       -> SENSOR_FAULT, confidence 0.95-0.99
  2. ml >= ml.high:
       neighbours disagree or physics soft          -> SENSOR_FAULT, confidence 0.6 + 0.4 * mean(ml, min(|z|/6, 1))
       neighbours agree                             -> GENUINE_EVENT
       not enough neighbours                        -> SUSPECT, confidence <= 0.6
  3. ml >= ml.suspect and neighbours disagree       -> SUSPECT
  4. otherwise                                      -> NORMAL

  Severity from dev_sigma (|value - expected| / std) using config severity_sigma;
  any hard physics violation is CRITICAL. NORMAL has severity None.
"""

from __future__ import annotations

import math

from skyguard.config import load_config
from skyguard.schemas import Severity, Status


# ---------------------------------------------------------------------------
# Severity helper
# ---------------------------------------------------------------------------

def _severity(dev_sigma: float, thresholds: list[float], physics_hard: bool) -> str | None:
    """Map |dev_sigma| to a severity level.

    thresholds is config ``severity_sigma`` = [3, 6, 10] meaning:
        |dev_sigma| <  3  →  LOW
        3 ≤ |dev_sigma| <  6  →  MEDIUM
        6 ≤ |dev_sigma| < 10  →  HIGH
       10 ≤ |dev_sigma|       →  CRITICAL

    A physics hard violation always returns CRITICAL regardless of dev_sigma.
    """
    if physics_hard:
        return Severity.CRITICAL

    abs_dev = abs(dev_sigma) if (dev_sigma is not None and math.isfinite(dev_sigma)) else 0.0

    if abs_dev >= thresholds[2]:      # >= 10
        return Severity.CRITICAL
    elif abs_dev >= thresholds[1]:    # >= 6
        return Severity.HIGH
    elif abs_dev >= thresholds[0]:    # >= 3
        return Severity.MEDIUM
    else:
        return Severity.LOW


# ---------------------------------------------------------------------------
# Main decision function
# ---------------------------------------------------------------------------

def decide(physics: dict | None, spatial: dict | None, ml_score: float, dev_sigma: float,
           cfg: dict | None = None) -> tuple[str, float, str | None]:
    """Fuse physics, spatial and ML signals into (status, confidence, severity).

    Parameters
    ----------
    physics : dict | None
        From ``check_physics()``:
        ``{"hard": bool, "soft": bool, "rule_ids": [...], "reasons": [...]}``.
    spatial : dict | None
        From ``spatial_check()``:
        ``{"z": float | None, "n": int, "consistent": bool | None, "reason": str}``.
    ml_score : float
        Percentile from ``IFModel.score()`` — 0 = normal, 1 = most anomalous.
    dev_sigma : float
        ``(value - climatology_mean) / climatology_std`` — how many σ from normal.
    cfg : dict | None
        Full config dict; loaded from ``config.yaml`` if not provided.

    Returns
    -------
    (status, confidence, severity) where
        status     : one of ``Status.ALL``
        confidence : float 0..1 (how sure we are of this status)
        severity   : one of ``Severity.ALL`` or ``None`` when status is NORMAL
    """
    cfg = cfg or load_config()
    ml_cfg = cfg["ml"]
    ml_high = ml_cfg["high"]         # 0.99
    ml_suspect = ml_cfg["suspect"]   # 0.95
    sev_thresholds = cfg["severity_sigma"]   # [3, 6, 10]

    physics = physics or {}
    spatial = spatial or {}

    # Sanitise inputs
    if ml_score is None or not math.isfinite(ml_score):
        ml_score = 0.0

    physics_hard = bool(physics.get("hard", False))
    physics_soft = bool(physics.get("soft", False))
    z_raw = spatial.get("z")
    spatial_z = abs(z_raw) if (z_raw is not None and math.isfinite(z_raw)) else 0.0
    consistent = spatial.get("consistent")       # True / False / None

    # Convenience: neighbours disagree = consistent is explicitly False
    neighbours_disagree = consistent is False
    neighbours_agree = consistent is True
    no_neighbours = consistent is None

    # ------------------------------------------------------------------
    # Rule 1: Physics hard violation → SENSOR_FAULT (highest priority)
    # ------------------------------------------------------------------
    if physics_hard:
        confidence = 0.95 + 0.04 * ml_score       # 0.95 .. 0.99
        severity = _severity(dev_sigma, sev_thresholds, physics_hard=True)
        return Status.SENSOR_FAULT, round(confidence, 4), severity

    # ------------------------------------------------------------------
    # Rule 2: ML score >= high threshold (0.99)
    # ------------------------------------------------------------------
    if ml_score >= ml_high:

        # 2a: neighbours disagree OR physics soft → SENSOR_FAULT
        if neighbours_disagree or physics_soft:
            z_signal = min(spatial_z / 6.0, 1.0)
            confidence = 0.6 + 0.4 * (ml_score + z_signal) / 2.0
            severity = _severity(dev_sigma, sev_thresholds, physics_hard=False)
            return Status.SENSOR_FAULT, round(confidence, 4), severity

        # 2b: neighbours agree → GENUINE_EVENT (real weather, no alarm)
        if neighbours_agree:
            confidence = 0.7 + 0.3 * ml_score
            severity = _severity(dev_sigma, sev_thresholds, physics_hard=False)
            return Status.GENUINE_EVENT, round(confidence, 4), severity

        # 2c: not enough neighbours → SUSPECT (can't confirm either way)
        if no_neighbours:
            confidence = min(0.6, 0.3 + 0.3 * ml_score)
            severity = _severity(dev_sigma, sev_thresholds, physics_hard=False)
            return Status.SUSPECT, round(confidence, 4), severity

    # ------------------------------------------------------------------
    # Rule 3: ML score >= suspect threshold AND neighbours disagree
    # ------------------------------------------------------------------
    if ml_score >= ml_suspect and neighbours_disagree:
        # Confidence scales between 0.3 and 0.5 as ml_score approaches ml_high
        span = max(ml_high - ml_suspect, 1e-9)
        frac = (ml_score - ml_suspect) / span
        confidence = 0.3 + 0.2 * frac
        severity = _severity(dev_sigma, sev_thresholds, physics_hard=False)
        return Status.SUSPECT, round(confidence, 4), severity

    # ------------------------------------------------------------------
    # Rule 4: Everything else → NORMAL
    # ------------------------------------------------------------------
    return Status.NORMAL, 0.0, None

