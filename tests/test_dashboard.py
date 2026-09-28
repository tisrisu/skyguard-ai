import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dashboard"))

import summary  # noqa: E402
from replay import Replay  # noqa: E402
from skyguard.config import load_config  # noqa: E402

START = pd.Timestamp("2024-08-05", tz="UTC")


@pytest.fixture
def replay(hourly, stations):
    return Replay(hourly, stations, load_config(), START)


def test_jump_warms_up_and_advance_moves_one_hour(replay):
    assert replay.now == START
    assert replay.results["ts"].nunique() == 25          # 24 h warm-up + now
    replay.advance(3)
    assert replay.now == START + pd.Timedelta(hours=3)
    assert replay.results["ts"].max() == replay.now


def test_spike_becomes_one_fault_card(replay):
    at = replay.inject_spike("A")
    replay.advance(2)

    feed = summary.alert_feed(replay.results, replay.now - pd.Timedelta(hours=48))
    assert len(feed) == 1
    card = feed[0]
    assert (card["station_id"], card["variable"], card["fault_type"]) == ("A", "temp_c", "SPIKE")
    assert card["value"] == pytest.approx(55.0)
    assert card["start"] == at
    assert card["trust_after"] < card["trust_before"]


def test_storm_is_one_weather_card_for_all_stations(replay):
    replay.inject_storm("A")
    replay.advance(12)

    feed = summary.alert_feed(replay.results, replay.now - pd.Timedelta(hours=48))
    assert [c["kind"] for c in feed] == ["weather"]
    assert sorted(feed[0]["stations"]) == ["A", "B", "C"]
    assert summary.kpis(replay.results, replay.now, replay.step_ms)["faults_24h"] == 0


def test_clear_injections_restores_recorded_data(replay):
    replay.inject_freeze("B")
    replay.clear_injections()
    replay.advance(12)
    assert (replay.results["status"] == "NORMAL").all()


def test_second_storm_on_the_same_station_is_refused(replay):
    assert replay.inject_storm("A") is not None
    assert replay.inject_storm("A") is None
