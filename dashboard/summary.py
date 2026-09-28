"""Turns raw per-hour results into what the views show: alert runs, KPIs, station status."""

import pandas as pd

from skyguard.schemas import NAMES, Severity, Status, worst_status

FLAGGED = (Status.SENSOR_FAULT, Status.SUSPECT)
SEVERITY_ORDER = {s: i for i, s in enumerate(Severity.ALL)}
HOUR = pd.Timedelta(hours=1)


def _new_run(frame: pd.DataFrame, keys: list[str]) -> pd.Series:
    """True where a row starts a new run of consecutive hours with the same keys."""
    changed = (frame[keys] != frame[keys].shift()).any(axis=1)
    return changed | (frame["ts"].diff() != HOUR)


def _worst_severity(values) -> str | None:
    values = [v for v in values if v in SEVERITY_ORDER]
    return max(values, key=SEVERITY_ORDER.get) if values else None


def fault_runs(results: pd.DataFrame) -> list[dict]:
    """One entry per run of consecutive flagged hours on one sensor."""
    flagged = results[results["status"].isin(FLAGGED)].copy()
    if flagged.empty:
        return []
    flagged["fault_type"] = flagged["fault_type"].fillna("")
    flagged = flagged.sort_values(["station_id", "variable", "ts"])
    flagged["run"] = _new_run(flagged, ["station_id", "variable", "status", "fault_type"]).cumsum()

    trust = results.set_index(["station_id", "variable", "ts"])["trust"]
    runs = []
    for _, g in flagged.groupby("run"):
        peak = g.loc[g["confidence"].idxmax()]
        start, end = g["ts"].iloc[0], g["ts"].iloc[-1]
        runs.append({
            "kind": "fault",
            "station_id": peak["station_id"],
            "variable": peak["variable"],
            "status": peak["status"],
            "fault_type": peak["fault_type"] or None,
            "severity": _worst_severity(g["severity"]),
            "confidence": float(peak["confidence"]),
            "value": peak["value"],
            "corrected": peak["corrected_value"],
            "reasons": list(peak["reasons"]),
            "scores": peak["scores"] if isinstance(peak["scores"], dict) else {},
            "start": start,
            "end": end,
            "trust_before": float(trust.get((peak["station_id"], peak["variable"], start - HOUR), 100.0)),
            "trust_after": float(g["trust"].iloc[-1]),
        })
    return runs


def weather_events(results: pd.DataFrame) -> list[dict]:
    """Genuine events that overlap in time are one weather event, however many stations it reached."""
    genuine = results[results["status"] == Status.GENUINE_EVENT]
    if genuine.empty:
        return []
    per_station = []
    for sid, g in genuine.groupby("station_id"):
        hours = g.drop_duplicates("ts").sort_values("ts")
        hours = hours.assign(run=_new_run(hours, ["station_id"]).cumsum())
        for _, run in hours.groupby("run"):
            first = genuine[(genuine["station_id"] == sid) & (genuine["ts"] == run["ts"].iloc[0])]
            first = first.sort_values("variable", key=lambda v: v != "temp_c")   # temperature tells the story best
            # evidence from the latest hour: by then the front has reached the neighbours too
            latest = genuine[(genuine["station_id"] == sid) & (genuine["ts"] == run["ts"].iloc[-1])]
            latest = latest.sort_values("variable", key=lambda v: v != "temp_c")
            scores = latest.iloc[0]["scores"]
            per_station.append({"station_id": sid, "start": run["ts"].iloc[0], "end": run["ts"].iloc[-1],
                                "confidence": float(g["confidence"].max()),
                                "reasons": list(first.iloc[0]["reasons"]),
                                "scores": scores if isinstance(scores, dict) else {}})

    events = []
    for part in sorted(per_station, key=lambda p: p["start"]):
        if events and part["start"] <= events[-1]["end"] + HOUR:
            event = events[-1]
            event["end"] = max(event["end"], part["end"])
            event["stations"].append(part["station_id"])
        else:
            events.append({"kind": "weather", "status": Status.GENUINE_EVENT, "start": part["start"],
                           "end": part["end"], "stations": [part["station_id"]],
                           "station_id": part["station_id"], "confidence": part["confidence"],
                           "reasons": part["reasons"], "scores": part["scores"]})
    return events


def alert_feed(results: pd.DataFrame, now: pd.Timestamp, limit: int = 30) -> tuple[list[dict], list[dict]]:
    """(active, earlier): alerts still going at `now`, and the rest of the kept history.

    Newest start first in both, so a card keeps its place while its event goes on.
    """
    items = sorted(fault_runs(results) + weather_events(results), key=lambda x: x["start"], reverse=True)
    active = [x for x in items if x["end"] == now]
    earlier = [x for x in items if x["end"] != now]
    return active, earlier[: max(limit - len(active), 0)]


def station_status(results: pd.DataFrame, now: pd.Timestamp) -> dict[str, dict]:
    """Worst status and lowest trust of each station at `now`."""
    current = results[results["ts"] == now]
    return {
        sid: {"status": worst_status(g["status"]), "trust": float(g["trust"].min())}
        for sid, g in current.groupby("station_id")
    }


def kpis(results: pd.DataFrame, now: pd.Timestamp, step_ms: list[float]) -> dict:
    day = results[results["ts"] > now - pd.Timedelta(hours=24)]
    current = results[results["ts"] == now]
    runs = fault_runs(day)
    lowest = current.loc[current["trust"].idxmin()] if not current.empty else None
    return {
        "lowest_trust": None if lowest is None else float(lowest["trust"]),
        "lowest_sensor": None if lowest is None else (lowest["station_id"], lowest["variable"]),
        "open_faults": sum(r["end"] == now for r in runs),
        "faults_24h": len(runs),
        "events_24h": len(weather_events(day)),
        "step_ms": sum(step_ms[-24:]) / len(step_ms[-24:]) if step_ms else None,
    }


def trust_now(results: pd.DataFrame, now: pd.Timestamp) -> dict[tuple[str, str], float]:
    """Trust score of every sensor at `now`, keyed by (station_id, variable)."""
    current = results[results["ts"] == now]
    return {(r.station_id, r.variable): float(r.trust) for r in current.itertuples(index=False)}


def maintenance(results: pd.DataFrame, now: pd.Timestamp, threshold: float = 50) -> pd.DataFrame:
    """Sensors whose trust is below the threshold, lowest first, with the last fault seen."""
    current = results[(results["ts"] == now) & (results["trust"] < threshold)]
    flagged = results[results["status"] == Status.SENSOR_FAULT].sort_values("ts")
    last_fault = flagged.groupby(["station_id", "variable"])["fault_type"].last()
    rows = [{
        "station_id": r.station_id,
        "variable": r.variable,
        "trust": r.trust,
        "last_fault": last_fault.get((r.station_id, r.variable)),
    } for r in current.itertuples(index=False)]
    return pd.DataFrame(rows, columns=["station_id", "variable", "trust", "last_fault"]).sort_values("trust")


def alert_log(results: pd.DataFrame, names: dict, tz: str) -> pd.DataFrame:
    """Every fault run and weather event in the kept history, newest first, for export."""
    rows = []
    for item in fault_runs(results) + weather_events(results):
        weather = item["kind"] == "weather"
        rows.append({
            "start": item["start"].tz_convert(tz).strftime("%Y-%m-%d %H:%M"),
            "end": item["end"].tz_convert(tz).strftime("%Y-%m-%d %H:%M"),
            "hours": int((item["end"] - item["start"]) / HOUR) + 1,
            "station": ", ".join(item["stations"]) if weather else f"{item['station_id']} {names.get(item['station_id'], '')}",
            "sensor": "" if weather else NAMES[item["variable"]],
            "status": item["status"],
            "fault_type": "" if weather else (item["fault_type"] or ""),
            "severity": "" if weather else (item["severity"] or ""),
            "confidence": round(item["confidence"], 2),
            "reported": None if weather else item["value"],
            "corrected": None if weather else item["corrected"],
        })
    columns = ["start", "end", "hours", "station", "sensor", "status", "fault_type", "severity",
               "confidence", "reported", "corrected"]
    return pd.DataFrame(rows, columns=columns).sort_values("start", ascending=False, ignore_index=True)
