"""HTML for the header, KPI tiles and alert cards. Styling lives in style.css."""

from html import escape

import pandas as pd

from charts import IST, STATUS_COLOR, STATUS_LABEL
from skyguard.schemas import NAMES, UNITS, Status

LEGEND = '<div class="sg-legend">' + "".join(
    f'<span><i style="background:{STATUS_COLOR[s]}"></i>{STATUS_LABEL[s]}</span>' for s in STATUS_COLOR
) + "</div>"


def empty(message: str) -> str:
    return f'<div class="sg-empty">{message}</div>'


def section(title: str) -> str:
    return f'<p class="sg-section">{escape(title)}</p>'


def when(ts: pd.Timestamp) -> str:
    return ts.tz_convert(IST).strftime("%d %b, %H:%M")


def header(now: pd.Timestamp, playing: bool, source: str, notice: str | None) -> str:
    local = now.tz_convert(IST)
    note = ""
    if notice:
        note = (f'<div class="sg-note">{escape(source)} data: showing the fault injector\'s labels. '
                f'{escape(notice)}.</div>')
    return f"""
<div class="sg-header">
  <div>
    <div class="sg-brand">Sky<span>Guard</span></div>
    <div class="sg-sub">Automatic weather station monitor &middot; Delhi-NCR, 8 stations</div>
  </div>
  <div class="sg-clock">
    <span class="sg-pill{' live' if playing else ''}">{'Replaying' if playing else 'Paused'}</span>
    <div class="sg-time">{local:%H:%M} <span>IST</span></div>
    <div class="sg-date">{local:%a %d %b %Y}</div>
  </div>
</div>{note}"""


def tiles(k: dict) -> str:
    step = f"{k['step_ms']:.0f} ms" if k["step_ms"] is not None else "&ndash;"
    items = [
        (f"{k['healthy']}<small>/{k['sensors']}</small>", "Sensors healthy", ""),
        (k["open_faults"], "Open faults", "alert" if k["open_faults"] else ""),
        (k["faults_24h"], "Faults, last 24 h", ""),
        (k["events_24h"], "Weather events, last 24 h", "weather" if k["events_24h"] else ""),
        (step, "Check time per hour", ""),
    ]
    cells = "".join(f'<div class="sg-tile {cls}"><b>{value}</b><span>{label}</span></div>'
                    for value, label, cls in items)
    return f'<div class="sg-tiles">{cells}</div>'


def _reading(value, unit: str) -> str:
    if value is None or pd.isna(value):
        return "no reading"
    return f"{value:g} {unit}" if abs(value) >= 1000 else f"{value:.1f} {unit}"


def _reasons(reasons) -> str:
    if not reasons:
        return ""
    return "<ul>" + "".join(f"<li>{escape(str(r))}</li>" for r in reasons) + "</ul>"


def alert_card(item: dict, names: dict, now: pd.Timestamp) -> str:
    if item["kind"] == "weather":
        return _weather_card(item, now)

    sid, var = item["station_id"], item["variable"]
    unit = UNITS[var]
    hours = int((item["end"] - item["start"]) / pd.Timedelta(hours=1)) + 1
    live = '<span class="sg-live">now</span>' if item["end"] == now else ""
    kind = STATUS_LABEL[item["status"]]
    if item["fault_type"]:
        kind += f" &middot; {item['fault_type'].replace('_', ' ').capitalize()}"

    values = f'<span class="bad">{_reading(item["value"], unit)}</span>'
    if item["corrected"] is not None and not pd.isna(item["corrected"]):
        values += f' &rarr; <b>{item["corrected"]:.1f} {unit}</b> <small>corrected</small>'

    severity = item["severity"] or "LOW"
    css = "fault" if item["status"] == Status.SENSOR_FAULT else "suspect"
    return f"""
<div class="sg-card {css}">
  <div class="sg-top"><span class="sg-badge {severity}">{severity}</span>
    <span>{when(item['start'])}{f' &middot; {hours} h' if hours > 1 else ''} {live}</span></div>
  <div class="sg-title">{escape(sid)} {escape(names.get(sid, ''))} &middot; {NAMES[var]}</div>
  <div class="sg-kind">{kind}</div>
  <div class="sg-values">{values}</div>
  {_reasons(item['reasons'])}
  <div class="sg-foot"><span>Confidence {item['confidence']:.0%}</span>
    <span>Trust {item['trust_before']:.0f} &rarr; {item['trust_after']:.0f}</span></div>
</div>"""


def _weather_card(item: dict, now: pd.Timestamp) -> str:
    stations = item["stations"]
    live = '<span class="sg-live">now</span>' if item["end"] == now else ""
    return f"""
<div class="sg-card weather">
  <div class="sg-top"><span class="sg-badge WEATHER">WEATHER</span>
    <span>{when(item['start'])} &ndash; {item['end'].tz_convert(IST):%H:%M} {live}</span></div>
  <div class="sg-title">Weather event at {len(stations)} station{'s' if len(stations) > 1 else ''}</div>
  <div class="sg-kind">Genuine event &middot; no action needed</div>
  <div class="sg-values"><small>{escape(', '.join(stations))}</small></div>
  {_reasons(item['reasons'])}
  <div class="sg-foot"><span>Confidence {item['confidence']:.0%}</span>
    <span>Readings kept as reported</span></div>
</div>"""


def score_tiles(m: dict) -> str:
    def pct(x):
        return "&ndash;" if x is None else f"{x:.1%}"

    latency = m.get("latency_ms_mean")
    items = [
        (pct(m.get("false_alarm_rate_storm")), "False alarms during storms"),
        (pct(m.get("false_alarm_rate_normal")), "False alarms, normal weather"),
        (pct(m.get("fault_type_accuracy")), "Fault type named correctly"),
        ("&ndash;" if latency is None else f"{latency:.0f} ms", "Mean check time"),
    ]
    cells = "".join(f'<div class="sg-tile"><b>{v}</b><span>{label}</span></div>' for v, label in items)
    return f'<div class="sg-tiles four">{cells}</div>'
