"""Name the type of sensor fault from its signature.

Owner: M2

Rules (first match wins):
  1. physics fault_hint set (FROZEN / DROPOUT / OUT_OF_RANGE)  -> that type
  2. missing / sentinel / pressure-zero                         -> DROPOUT
  3. value outside physics.range                                -> OUT_OF_RANGE
  4. consecutive identical readings >= flatline_steps            -> FROZEN
  5. large jump that returns to normal within <= 2 steps         -> SPIKE
  6. high 6-hour variability without a trend                     -> NOISE
  7. persistent offset from neighbours, growing gradually        -> DRIFT
  8. any remaining large deviation from recent baseline           -> SPIKE
  9. otherwise                                                    -> None
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from skyguard.config import load_config
from skyguard.schemas import FaultType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _recent_baseline(col: pd.Series, exclude_last: int = 3,
                     window: int = 24) -> float | None:
    """Median of readings *before* a potential fault window.

    Skips the last ``exclude_last`` values (which may be part of the
    anomaly) and takes up to ``window`` values before that.
    Returns ``None`` if fewer than 3 valid readings are available.
    """
    if len(col) <= exclude_last:
        return None
    lookback = min(window, len(col) - exclude_last)
    baseline = col.iloc[-(lookback + exclude_last):-exclude_last]
    valid = baseline.dropna()
    return float(valid.median()) if len(valid) >= 3 else None


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

def fingerprint(station_hist: pd.DataFrame, var: str, physics: dict | None = None,
                spatial: dict | None = None, cfg: dict | None = None) -> str | None:
    """Classify the type of sensor fault from the recent signal pattern.

    Parameters
    ----------
    station_hist : pd.DataFrame
        One station's rows up to and including the current *ts*, sorted by ts.
        Must contain columns: ``ts``, ``temp_c``, ``pressure_hpa``, ``rh_pct``.
    var : str
        The variable under inspection (one of ``schemas.VARIABLES``).
    physics : dict | None
        Output of ``check_physics()`` for this reading.  May contain
        ``fault_hint`` (``"FROZEN"`` / ``"DROPOUT"`` / ``"OUT_OF_RANGE"``
        / ``None``).  An empty dict is safe (M3's stub not ready yet).
    spatial : dict | None
        Output of ``spatial_check()``.  Used for DRIFT detection.
        ``None`` is safe.
    cfg : dict | None
        Full config dict; loaded from ``config.yaml`` if not provided.

    Returns
    -------
    One of ``FaultType.FAULTS`` or ``None`` if the pattern doesn't match
    any known fault type.
    """
    cfg = cfg or load_config()
    phys = cfg["physics"]
    physics = physics or {}
    col = station_hist[var]

    if col.empty:
        return None

    current = float(col.iloc[-1]) if pd.notna(col.iloc[-1]) else np.nan

    # ── 1. Physics fault_hint (authoritative when available) ──────────────
    hint = physics.get("fault_hint")
    if hint in FaultType.FAULTS:
        return hint

    # ── 2. DROPOUT: NaN, sentinel, or pressure == 0 ──────────────────────
    if np.isnan(current):
        return FaultType.DROPOUT
    sentinels = phys.get("sentinels", [-9999, -999])
    if current in sentinels:
        return FaultType.DROPOUT
    if var == "pressure_hpa" and current == 0.0:
        return FaultType.DROPOUT

    # ── 3. OUT_OF_RANGE: outside physics bounds ──────────────────────────
    lo, hi = phys["range"][var]
    if current < lo or current > hi:
        return FaultType.OUT_OF_RANGE

    # ── 4. FROZEN: flatline of >= N identical readings ────────────────────
    flatline_n = phys["flatline_steps"][var]
    if len(col) >= flatline_n:
        tail = col.iloc[-flatline_n:]
        if tail.notna().all() and tail.nunique() == 1:
            # Saturated RH (near 100 %) can legitimately sit unchanged
            ignore_above = phys.get("flatline_ignore_rh_above", 98)
            if not (var == "rh_pct" and current >= ignore_above):
                return FaultType.FROZEN

    # ── Need >= 2 rows for rate-of-change checks ─────────────────────────
    if len(col) < 2:
        return None

    roc_limits = phys["rate_of_change_per_h"].get(var, [None, None])
    soft_limit = roc_limits[0]                    # may be None

    # ── 5. SPIKE: jump-and-return within <= 2 steps ──────────────────────
    if soft_limit is not None:
        # 5a. Classic 3-point jump-and-return (spike already passed)
        if len(col) >= 3 and col.iloc[-3:].notna().all():
            v = col.iloc[-3:].values
            if abs(v[1] - v[0]) >= soft_limit and abs(v[2] - v[0]) < soft_limit:
                return FaultType.SPIKE

        # 5b. 4-point window (spike spanning 2 hours in the middle)
        if len(col) >= 4 and col.iloc[-4:].notna().all():
            v = col.iloc[-4:].values
            if abs(v[1] - v[0]) >= soft_limit and abs(v[3] - v[0]) < soft_limit:
                return FaultType.SPIKE

        # 5c. Currently at the peak of a spike (just jumped)
        prev = col.iloc[-2]
        if pd.notna(prev) and abs(current - prev) >= soft_limit:
            return FaultType.SPIKE

    # ── 6. NOISE: high 6-hour std relative to history, no clear trend ────
    if len(col) >= 6:
        recent_6h = col.iloc[-6:]
        if recent_6h.notna().all():
            std_6h = float(recent_6h.std())

            # Baseline variability from earlier history
            if len(col) > 12:
                baseline_std = float(
                    col.iloc[:-6].rolling(6, min_periods=4).std().median()
                )
            else:
                baseline_std = float(col.std())
            baseline_std = max(baseline_std, 1e-6)

            # Detrend: remove the linear fit and measure the residual std.
            # NOISE has high residual variability; a steady trend does not.
            x = np.arange(6, dtype=np.float64)
            coeffs = np.polyfit(x, recent_6h.values, 1)
            residuals = recent_6h.values - np.polyval(coeffs, x)
            residual_std = float(np.std(residuals, ddof=1)) if len(residuals) > 1 else 0.0

            # Noise: residual variability much higher than baseline AND
            # the raw std is dominated by jitter, not by a trend
            if residual_std > 3.0 * baseline_std and std_6h > 3.0 * baseline_std:
                return FaultType.NOISE

    # ── 7. DRIFT: gradual offset, neighbours disagree ────────────────────
    if spatial is not None and len(col) >= 24:
        z = spatial.get("z")
        z_thresh = cfg["spatial"]["z_threshold"]          # 3.0
        if z is not None and abs(z) > z_thresh:
            drift_tol = cfg.get("drift_tolerance", {}).get(var)
            recent_24 = col.iloc[-24:]
            if drift_tol is not None and recent_24.notna().all():
                max_step = float(recent_24.diff().dropna().abs().max())
                total_drift = abs(float(recent_24.iloc[-1] - recent_24.iloc[0]))
                step_thresh = soft_limit if soft_limit is not None else float("inf")
                # Many small steps adding up to a large offset → DRIFT
                if max_step < step_thresh and total_drift > drift_tol:
                    return FaultType.DRIFT

    # ── 8. Fallback SPIKE: notable deviation from pre-anomaly baseline ───
    if soft_limit is not None:
        baseline = _recent_baseline(col)
        if baseline is not None and abs(current - baseline) >= soft_limit * 1.5:
            return FaultType.SPIKE

    # ── 9. No pattern matched ────────────────────────────────────────────
    return None
