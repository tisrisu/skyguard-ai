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


def decide(physics: dict, spatial: dict, ml_score: float, dev_sigma: float,
           cfg: dict | None = None) -> tuple[str, float, str | None]:
    """Return (status, confidence, severity). See module docstring."""
    raise NotImplementedError
