"""Shared names and the result format used by the engine and the dashboard.

Change anything here only after telling the whole team.
"""

from dataclasses import asdict, dataclass, field

VARIABLES = ("temp_c", "pressure_hpa", "rh_pct")

UNITS = {"temp_c": "°C", "pressure_hpa": "hPa", "rh_pct": "%"}
NAMES = {"temp_c": "Temperature", "pressure_hpa": "Pressure", "rh_pct": "Humidity"}

DATA_COLUMNS = ["station_id", "ts", *VARIABLES]
LABEL_COLUMNS = {v: f"label_{v}" for v in VARIABLES}
EVENT_COLUMN = "event_id"


class Status:
    NORMAL = "NORMAL"
    GENUINE_EVENT = "GENUINE_EVENT"
    SUSPECT = "SUSPECT"
    SENSOR_FAULT = "SENSOR_FAULT"
    ALL = (NORMAL, GENUINE_EVENT, SUSPECT, SENSOR_FAULT)


class FaultType:
    NONE = "NONE"  # label value for clean rows
    SPIKE = "SPIKE"
    FROZEN = "FROZEN"
    DRIFT = "DRIFT"
    DROPOUT = "DROPOUT"
    NOISE = "NOISE"
    OUT_OF_RANGE = "OUT_OF_RANGE"
    FAULTS = (SPIKE, FROZEN, DRIFT, DROPOUT, NOISE, OUT_OF_RANGE)


class Severity:
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"
    ALL = (LOW, MEDIUM, HIGH, CRITICAL)


# Higher = worse. Used to pick the overall status of a station.
STATUS_RANK = {
    Status.NORMAL: 0,
    Status.GENUINE_EVENT: 1,
    Status.SUSPECT: 2,
    Status.SENSOR_FAULT: 3,
}


def worst_status(statuses) -> str:
    return max(statuses, key=STATUS_RANK.__getitem__, default=Status.NORMAL)


@dataclass
class Result:
    """Outcome for one station, one timestamp, one variable."""

    station_id: str
    ts: str                                  # ISO-8601 UTC, e.g. "2024-08-14T09:00:00Z"
    variable: str                            # one of VARIABLES
    value: float | None
    status: str = Status.NORMAL
    fault_type: str | None = None            # one of FaultType.FAULTS, or None
    confidence: float = 0.0                  # 0..1
    severity: str | None = None              # one of Severity.ALL, or None when NORMAL
    reasons: list[str] = field(default_factory=list)
    shap_top: list[tuple[str, float]] = field(default_factory=list)
    corrected_value: float | None = None
    trust: float = 100.0
    scores: dict = field(default_factory=dict)   # e.g. {"physics": 1.0, "iforest": 0.99, "spatial_z": 9.4}

    def to_dict(self) -> dict:
        return asdict(self)
