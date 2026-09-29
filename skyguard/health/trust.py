"""Trust score (0-100) per station and sensor.
"""

from __future__ import annotations

from skyguard.config import load_config
from skyguard.schemas import Severity, Status


class TrustTracker:
    def __init__(self, cfg: dict | None = None):
        self.cfg = cfg or load_config()
        self.scores: dict[tuple[str, str], float] = {}

    def update(self, station_id: str, var: str, status: str, severity: str | None) -> float:
        key = (station_id, var)
        score = self.scores.get(key, float(self.cfg["trust"]["start"]))
        if status == Status.SENSOR_FAULT:
            penalty = self.cfg["trust"]["penalty"].get(severity or Severity.LOW, 0)
            score -= float(penalty)
        elif status == Status.NORMAL:
            score += float(self.cfg["trust"]["recovery_per_clean"])
        score = max(0.0, min(float(self.cfg["trust"]["start"]), score))
        self.scores[key] = score
        return score

    def get(self, station_id: str, var: str) -> float:
        return float(self.scores.get((station_id, var), self.cfg["trust"]["start"]))

    def band(self, score: float) -> str:
        bands = self.cfg["trust"]["bands"]
        if score >= bands["HEALTHY"]:
            return "HEALTHY"
        if score >= bands["WATCH"]:
            return "WATCH"
        if score >= bands["DEGRADED"]:
            return "DEGRADED"
        return "FAILED"
