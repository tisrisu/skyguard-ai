"""SkyGuard dashboard: replays the Delhi-NCR network hour by hour and shows what the detector finds.

Owner: M4

Run:  streamlit run dashboard/app.py   (or run_demo.bat)
"""

import json
import sys
import time
from pathlib import Path

import pandas as pd
import streamlit as st

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]

import cards  # noqa: E402
import charts  # noqa: E402
import summary  # noqa: E402
from replay import Replay  # noqa: E402
from skyguard.config import REPORTS_DIR, load_config  # noqa: E402
from skyguard.data.io import load_data, load_stations, select_split  # noqa: E402
from skyguard.geo import neighbours  # noqa: E402
from skyguard.schemas import VARIABLES  # noqa: E402

START = "2024-08-01"            # replay starts here (test split, monsoon)
TICK_SECONDS = 1.0
SPEEDS = {"1 h/s": 1, "3 h/s": 3, "6 h/s": 6}
WINDOW_HOURS = 48

st.set_page_config(page_title="SkyGuard", page_icon=":material/thunderstorm:", layout="wide")
st.html(f"<style>{(HERE / 'style.css').read_text(encoding='utf-8')}</style>")


@st.cache_resource(show_spinner="Loading station data")
def load():
    cfg = load_config()
    return select_split(load_data(), "test", cfg), load_stations(), cfg


def get_replay() -> Replay:
    if "replay" not in st.session_state:
        df, stations, cfg = load()
        st.session_state.replay = Replay(df, stations, cfg, pd.Timestamp(START, tz="UTC"))
        st.session_state.playing = False
        st.session_state.station = stations[0]["station_id"]
        st.session_state.jumped_to = pd.Timestamp(START).date()
    return st.session_state.replay


def toggle_play():
    st.session_state.playing = not st.session_state.playing


def injected(r: Replay, at, message: str, refused: str = "No reading at that hour to change"):
    if at is None:
        st.toast(refused)
        return
    if not st.session_state.playing:
        r.advance(1)
    st.toast(f"{message}, {cards.when(at)}")


def sidebar(r: Replay, names: dict):
    with st.sidebar:
        st.html(cards.section("Replay"))
        playing = st.session_state.playing
        st.button("Pause" if playing else "Play", type="primary", width="stretch", on_click=toggle_play,
                  icon=":material/pause:" if playing else ":material/play_arrow:",
                  disabled=r.at_end and not playing)
        st.segmented_control("Speed", list(SPEEDS), default="1 h/s", key="speed")
        # no widget key on purpose: keyed state can be reset to a default between reruns,
        # which silently jumped the replay. The chosen day lives in jumped_to instead.
        day = st.date_input("Jump to", value=st.session_state.jumped_to, format="DD/MM/YYYY",
                            min_value=r.hours[0].date(),
                            max_value=r.hours[-1].date())
        if day != st.session_state.jumped_to:
            st.session_state.jumped_to = day
            r.jump(pd.Timestamp(day, tz="UTC"))

        st.html(cards.section("Station"))
        st.selectbox("Station", list(names), format_func=lambda s: f"{s}  {names[s]}",
                     key="station", label_visibility="collapsed")

        st.html(cards.section("Demo"))
        st.caption("Each action changes the data from the next hour.")
        sid = st.session_state.station
        if st.button("Spike to 55 °C", icon=":material/bolt:", width="stretch"):
            injected(r, r.inject_spike(sid), f"Temperature spike at {sid}")
        if st.button("Freeze humidity sensor", icon=":material/ac_unit:", width="stretch"):
            injected(r, r.inject_freeze(sid), f"Humidity sensor frozen at {sid}")
        if st.button("Thunderstorm", icon=":material/thunderstorm:", width="stretch"):
            injected(r, r.inject_storm(sid), f"Thunderstorm starting near {sid}",
                     refused=f"A storm is already passing {sid}")
        if st.button("Remove injected faults", type="tertiary", width="stretch"):
            r.clear_injections()
            st.toast("Back to the recorded data")

        st.divider()
        st.caption(f"Detector: {r.source.name}" + (f". {r.notice}." if r.notice else ""))


def tick(r: Replay):
    if not st.session_state.playing:
        return
    if r.at_end:
        st.session_state.playing = False
        st.rerun()
    now = time.monotonic()
    # a button click also reruns the page; don't let it skip an extra hour
    if now - st.session_state.get("last_tick", 0.0) >= TICK_SECONDS * 0.8:
        r.advance(SPEEDS.get(st.session_state.speed, 1))
        st.session_state.last_tick = now


def station_window(r: Replay, sid: str):
    since = r.now - pd.Timedelta(hours=WINDOW_HOURS)
    window = r.df[(r.df["ts"] > since) & (r.df["ts"] <= r.now)]
    values = window[window["station_id"] == sid]
    results = r.results[(r.results["station_id"] == sid) & (r.results["ts"] > since)]

    nearby = [n["station_id"] for n in neighbours(r.stations, sid, r.cfg["spatial"]["radius_km"])]
    others = window[window["station_id"].isin(nearby)].copy()
    for var in VARIABLES:
        lo, hi = r.cfg["physics"]["range"][var]
        others[var] = others[var].where(others[var].between(lo, hi))
    median = others.groupby("ts")[list(VARIABLES)].median()
    return values, results, median


def live_tab(r: Replay, names: dict):
    sid = st.session_state.station
    left, right = st.columns([1.6, 1], gap="large")
    with left:
        st.html(cards.section("Network"))
        status = summary.station_status(r.results, r.now)
        st.plotly_chart(charts.network_map(r.stations, status, sid), key="map",
                        config=charts.PLOT_CONFIG, width="stretch")
        st.html(cards.LEGEND)
    with right:
        st.html(cards.section("Alerts"))
        active, earlier = summary.alert_feed(r.results, r.now)
        with st.container(height=470, border=False):
            if not active and not earlier:
                st.html(cards.empty("Nothing flagged yet. Use the demo buttons on the left to add a fault."))
            for title, items in (("Active now", active), ("Earlier", earlier)):
                if items:
                    st.html(f'<p class="sg-group">{title}</p>')
                for item in items:
                    st.html(cards.alert_card(item, names, r.now))

    st.html(cards.section(f"{sid} {names[sid]}, last {WINDOW_HOURS} h (UTC)"))
    values, results, median = station_window(r, sid)
    st.plotly_chart(charts.station_chart(values, results, median, r.cfg), key="station_chart",
                    config=charts.PLOT_CONFIG, width="stretch")


def health_tab(r: Replay, names: dict):
    st.html(cards.section("Sensor status, last 72 h"))
    st.plotly_chart(charts.health_strip(r.results, r.stations, r.now), key="health",
                    config=charts.PLOT_CONFIG, width="stretch")
    st.html(cards.LEGEND)

    st.html(cards.section("Needs attention"))
    todo = summary.maintenance(r.results, r.now)
    if todo.empty:
        st.html(cards.empty("Every sensor has a trust score of 50 or more."))
    else:
        st.html(cards.attention(todo, names))

    st.html(cards.section("Trust score now"))
    st.html(cards.trust_grid(summary.trust_now(r.results, r.now), r.stations))

    st.html(cards.section("Alert log, last 7 days (UTC)"))
    log = summary.alert_log(r.results, names, charts.TZ)
    if log.empty:
        st.html(cards.empty("Nothing flagged yet."))
        return
    st.dataframe(log, hide_index=True, width="stretch", height=min(38 + 35 * len(log), 320))
    st.download_button("Download CSV", log.to_csv(index=False).encode("utf-8"), icon=":material/download:",
                       file_name=f"skyguard_alerts_{r.now:%Y%m%d_%H%M}.csv",
                       mime="text/csv")


def score_tab():
    path = REPORTS_DIR / "metrics.json"
    if not path.exists():
        st.html(cards.empty("No evaluation results yet. Once the detection engine is ready, run "
                            "<code>python eval/run_eval.py --split test</code>."))
        return
    metrics = json.loads(path.read_text(encoding="utf-8"))
    st.html(cards.score_tiles(metrics))
    st.html(cards.section("Detection by fault type"))
    st.plotly_chart(charts.scorecard_chart(metrics["per_fault"]), key="score",
                    config=charts.PLOT_CONFIG, width="stretch")
    st.caption("Test split, July to December 2024, with injected faults and storms. "
               "Recall: share of fault events caught. Precision: share of alerts that were real faults.")


def page():
    r = get_replay()
    tick(r)
    names = {s["station_id"]: s["name"] for s in r.stations}

    st.html(cards.header(r.now, st.session_state.playing, r.source.name, r.notice))
    st.html(cards.tiles(summary.kpis(r.results, r.now, r.step_ms), names))

    live, health, score = st.tabs(["Live monitor", "Sensor health", "Scorecard"])
    with live:
        live_tab(r, names)
    with health:
        health_tab(r, names)
    with score:
        score_tab()


replay = get_replay()
sidebar(replay, {s["station_id"]: s["name"] for s in replay.stations})
st.fragment(run_every=TICK_SECONDS if st.session_state.playing else None)(page)()
