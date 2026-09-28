"""Name the type of sensor fault from its signature.

Owner: M2

Rules (first match wins):
  physics fault_hint set (FROZEN / DROPOUT / OUT_OF_RANGE)  -> that type
  large jump that returns to normal within <= 2 steps       -> SPIKE
  high short-term variability without a trend               -> NOISE
  persistent offset from neighbours growing over days       -> DRIFT
  otherwise                                                  -> SPIKE if a jump, else None
"""

import pandas as pd


def fingerprint(station_hist: pd.DataFrame, var: str, physics: dict,
                spatial: dict | None = None) -> str | None:
    """Fault type (schemas.FaultType) or None."""
    raise NotImplementedError
