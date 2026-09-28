"""Trust score (0-100) per station and sensor.

Owner: M3

Rules (config.yaml -> trust):
  start at trust.start
  SENSOR_FAULT: subtract trust.penalty[severity]
  GENUINE_EVENT, SUSPECT: no change
  NORMAL: add trust.recovery_per_clean, capped at trust.start
Bands: HEALTHY >= 80, WATCH >= 50, DEGRADED >= 25, else FAILED.
"""


class TrustTracker:
    def __init__(self, cfg: dict | None = None):
        self.scores: dict[tuple[str, str], float] = {}

    def update(self, station_id: str, var: str, status: str, severity: str | None) -> float:
        """Apply one result and return the new score."""
        raise NotImplementedError

    def get(self, station_id: str, var: str) -> float:
        raise NotImplementedError

    def band(self, score: float) -> str:
        """"HEALTHY" | "WATCH" | "DEGRADED" | "FAILED"."""
        raise NotImplementedError
