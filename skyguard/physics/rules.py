"""Physics plausibility checks for one variable at the latest timestamp.

Owner: M3

Thresholds: config.yaml -> physics.

| Rule          | Check                                                        | Kind      | fault_hint   |
|---------------|--------------------------------------------------------------|-----------|--------------|
| RANGE         | value outside physics.range                                  | hard      | OUT_OF_RANGE |
| MISSING       | NaN, a sentinel (-9999, -999) or pressure == 0               | hard      | DROPOUT      |
| ROC           | |change in 1 h| above soft / hard limit                      | soft/hard |              |
| DEWPOINT_MAX  | implied dewpoint > physics.dewpoint_max_c (T and RH only)    | hard      |              |
| DEWPOINT_JUMP | |dewpoint change in 1 h| > physics.dewpoint_jump_per_h      | soft      |              |
| FLATLINE      | last N readings identical (N = physics.flatline_steps[var]); | hard      | FROZEN       |
|               | skip RH when >= flatline_ignore_rh_above                     |           |              |
| DECOUPLED     | T jumps > temp_jump while P and RH barely move               | soft      |              |

Keep the per-reading logic simple (plain comparisons over the last few values)
so it can be ported to C for the ESP32 later.
"""

import pandas as pd


def check_physics(station_hist: pd.DataFrame, var: str, cfg: dict | None = None) -> dict:
    """Checks for `var` at the last row of station_hist (one station, sorted by ts).

    Returns:
      {
        "hard": bool,
        "soft": bool,
        "rule_ids": ["DEWPOINT_MAX", ...],
        "reasons": ["Implied dewpoint 50.4 °C exceeds the physical maximum (~35 °C)", ...],
        "fault_hint": "FROZEN" | "DROPOUT" | "OUT_OF_RANGE" | None,
      }
    Reasons must be readable by a non-expert and include the actual numbers.
    """
    raise NotImplementedError
