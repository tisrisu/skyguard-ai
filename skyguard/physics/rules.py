"""Physics plausibility checks for one variable at the latest timestamp.

Owner: M3
"""

from __future__ import annotations

import math
import pandas as pd

from skyguard.config import load_config
from skyguard.physics.dewpoint import dewpoint_c


def _finite(x) -> bool:
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def check_physics(station_hist: pd.DataFrame, var: str, cfg: dict | None = None) -> dict:
    """Check the latest reading against configured physical/signature rules."""
    cfg = cfg or load_config()
    p = cfg["physics"]
    hist = station_hist.sort_values("ts").reset_index(drop=True)
    result = {"hard": False, "soft": False, "rule_ids": [], "reasons": [], "fault_hint": None}
    if hist.empty or var not in hist.columns:
        return result

    row = hist.iloc[-1]
    value = row[var]
    missing = pd.isna(value)
    try:
        value_f = float(value)
    except (TypeError, ValueError):
        value_f = float("nan")

    sentinels = set(p.get("sentinels", [-9999, -999]))
    is_missing = missing or not _finite(value_f) or value_f in sentinels or (
        var == "pressure_hpa" and value_f == 0.0
    )
    if is_missing:
        result["hard"] = True
        result["rule_ids"].append("MISSING")
        result["fault_hint"] = "DROPOUT"
        result["reasons"].append(f"{var} is missing or contains a dropout sentinel ({value!r})")
        return result

    lo, hi = p["range"][var]
    if value_f < lo or value_f > hi:
        result["hard"] = True
        result["rule_ids"].append("RANGE")
        result["fault_hint"] = "OUT_OF_RANGE"
        result["reasons"].append(f"{var}={value_f:.2f} is outside the physical range [{lo}, {hi}]")
    elif value_f == hi and var == "temp_c":
        # 55 C is the project's deliberate demo boundary; dewpoint logic below
        # decides whether it is physically implausible.
        pass

    # One-hour rate of change. Data is hourly by contract; if timestamps are
    # irregular, normalize the threshold by elapsed hours.
    if len(hist) >= 2:
        prev = hist.iloc[-2]
        prev_val = prev[var]
        if pd.notna(prev_val):
            try:
                delta = abs(value_f - float(prev_val))
                dt_h = max((pd.Timestamp(row["ts"]) - pd.Timestamp(prev["ts"])).total_seconds() / 3600.0, 1e-9)
                soft, hard = p["rate_of_change_per_h"].get(var, [None, None])
                if soft is not None and delta / dt_h > soft:
                    result["soft"] = True
                    result["rule_ids"].append("ROC")
                    result["reasons"].append(
                        f"{var} changed by {delta:.2f} in {dt_h:.1f} h (soft limit {soft}/h)"
                    )
                if hard is not None and delta / dt_h > hard:
                    result["hard"] = True
                    if "ROC" not in result["rule_ids"]:
                        result["rule_ids"].append("ROC")
                    result["reasons"].append(
                        f"{var} changed by {delta:.2f} in {dt_h:.1f} h (hard limit {hard}/h)"
                    )
            except (TypeError, ValueError):
                pass

    # Dewpoint checks use the current T/RH pair.
    if "temp_c" in hist and "rh_pct" in hist:
        t = row["temp_c"]
        rh = row["rh_pct"]
        if pd.notna(t) and pd.notna(rh):
            try:
                dp = float(dewpoint_c(float(t), float(rh)))
                max_dp = p["dewpoint_max_c"]
                if dp > max_dp:
                    result["hard"] = True
                    result["rule_ids"].append("DEWPOINT_MAX")
                    result["reasons"].append(
                        f"Implied dewpoint {dp:.1f} °C exceeds the physical maximum (~{max_dp:g} °C)"
                    )
                if len(hist) >= 2:
                    prev_t, prev_rh = hist.iloc[-2]["temp_c"], hist.iloc[-2]["rh_pct"]
                    if pd.notna(prev_t) and pd.notna(prev_rh):
                        prev_dp = float(dewpoint_c(float(prev_t), float(prev_rh)))
                        jump = abs(dp - prev_dp)
                        if jump > p["dewpoint_jump_per_h"]:
                            result["soft"] = True
                            result["rule_ids"].append("DEWPOINT_JUMP")
                            result["reasons"].append(
                                f"Implied dewpoint changed by {jump:.1f} °C in 1 h (limit {p['dewpoint_jump_per_h']} °C/h)"
                            )
            except (TypeError, ValueError):
                pass

    # Flatline signature.
    n = int(p["flatline_steps"][var])
    if len(hist) >= n:
        tail = hist[var].iloc[-n:]
        if tail.notna().all() and tail.nunique() == 1:
            ignore = p.get("flatline_ignore_rh_above", 98)
            if not (var == "rh_pct" and float(tail.iloc[-1]) >= ignore):
                result["hard"] = True
                result["rule_ids"].append("FLATLINE")
                result["fault_hint"] = "FROZEN"
                result["reasons"].append(
                    f"{var} has stayed at {float(tail.iloc[-1]):.2f} for {n} consecutive readings (flatline)"
                )

    # Temperature decoupled from pressure/RH.
    dcfg = p.get("decoupled", {})
    if var == "temp_c" and len(hist) >= 2:
        prev = hist.iloc[-2]
        if all(pd.notna(prev[c]) and pd.notna(row[c]) for c in ("temp_c", "pressure_hpa", "rh_pct")):
            dt = abs(float(row["temp_c"]) - float(prev["temp_c"]))
            dp = abs(float(row["pressure_hpa"]) - float(prev["pressure_hpa"]))
            drh = abs(float(row["rh_pct"]) - float(prev["rh_pct"]))
            if dt > dcfg.get("temp_jump", 6) and dp <= dcfg.get("pressure_still", 0.5) and drh <= dcfg.get("rh_still", 3):
                result["soft"] = True
                result["rule_ids"].append("DECOUPLED")
                result["reasons"].append(
                    f"Temperature jumped {dt:.1f} °C while pressure changed only {dp:.2f} hPa and RH {drh:.1f}%"
                )

    # De-duplicate rule IDs/reasons while preserving priority order.
    result["rule_ids"] = list(dict.fromkeys(result["rule_ids"]))
    result["reasons"] = list(dict.fromkeys(result["reasons"]))
    return result
