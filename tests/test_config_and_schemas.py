import json

from skyguard.config import MOCK_DIR, load_config
from skyguard.schemas import VARIABLES, Result, Status, worst_status


def test_config_has_all_sections():
    cfg = load_config()
    for key in ("region", "splits", "physics", "spatial", "ml", "trust"):
        assert key in cfg
    for var in VARIABLES:
        assert var in cfg["physics"]["range"]


def test_worst_status():
    assert worst_status([Status.NORMAL, Status.GENUINE_EVENT]) == Status.GENUINE_EVENT
    assert worst_status([Status.SUSPECT, Status.SENSOR_FAULT, Status.NORMAL]) == Status.SENSOR_FAULT
    assert worst_status([]) == Status.NORMAL


def test_mock_results_match_result_format():
    fields = set(Result("X", "t", "temp_c", 1.0).to_dict())
    for row in json.loads((MOCK_DIR / "mock_results.json").read_text(encoding="utf-8")):
        assert set(row) == fields
        assert row["status"] in Status.ALL
