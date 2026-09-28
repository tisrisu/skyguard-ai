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

    active, earlier = summary.alert_feed(replay.results, replay.now)
    assert active == [] and len(earlier) == 1
    card = earlier[0]
    assert (card["station_id"], card["variable"], card["fault_type"]) == ("A", "temp_c", "SPIKE")
    assert card["value"] == pytest.approx(55.0)
    assert card["start"] == at
    assert card["trust_after"] < card["trust_before"]


def test_storm_is_one_weather_card_for_all_stations(replay):
    replay.inject_storm("A")
    replay.advance(12)

    active, earlier = summary.alert_feed(replay.results, replay.now)
    feed = active + earlier
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


def test_evidence_separates_fault_from_weather(replay):
    replay.inject_spike("A")
    replay.advance(1)
    replay.inject_storm("B")
    replay.advance(4)

    active, earlier = summary.alert_feed(replay.results, replay.now)
    feed = active + earlier
    fault = next(c for c in feed if c["kind"] == "fault")
    weather = next(c for c in feed if c["kind"] == "weather")
    z_limit = load_config()["spatial"]["z_threshold"]
    assert fault["scores"]["physics"] == 1.0                   # 55 °C at this humidity: impossible dewpoint
    assert abs(fault["scores"]["spatial_z"]) > z_limit
    assert weather["scores"]["physics"] == 0.0
    assert abs(weather["scores"]["spatial_z"]) <= z_limit


def test_alert_log_lists_every_event(replay):
    replay.inject_spike("A")
    replay.advance(1)
    replay.inject_storm("B")
    replay.advance(4)

    log = summary.alert_log(replay.results, {"A": "Alpha"}, "Asia/Kolkata")
    assert set(log["status"]) == {"SENSOR_FAULT", "GENUINE_EVENT"}
    assert log.loc[log["status"] == "SENSOR_FAULT", "station"].item() == "A Alpha"


def test_cards_keep_their_order_while_an_event_goes_on(replay):
    replay.inject_spike("A")
    replay.advance(2)
    replay.inject_freeze("B", hours=6)
    replay.advance(1)
    first = [c["start"] for c in sum(summary.alert_feed(replay.results, replay.now), [])]
    replay.advance(3)
    active, earlier = summary.alert_feed(replay.results, replay.now)
    assert [c["start"] for c in active + earlier] == first     # frozen sensor still first, spike below it
    assert active[0]["fault_type"] == "FROZEN"


def test_lowest_trust_tile_reacts_to_a_fault(replay):
    before = summary.kpis(replay.results, replay.now, replay.step_ms)["lowest_trust"]
    replay.inject_spike("C")
    replay.advance(1)
    k = summary.kpis(replay.results, replay.now, replay.step_ms)
    assert k["lowest_trust"] < before
    assert k["lowest_sensor"] == ("C", "temp_c")
