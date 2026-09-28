"""HTML for the header, KPI tiles and alert cards. Styling lives in style.css."""

from html import escape

import pandas as pd

from charts import IST, STATUS_COLOR, STATUS_LABEL, TZ
from skyguard.config import load_config
from skyguard.schemas import NAMES, UNITS, Status

LEGEND = '<div class="sg-legend">' + "".join(
    f'<span><i style="background:{STATUS_COLOR[s]}"></i>{STATUS_LABEL[s]}</span>' for s in STATUS_COLOR
) + "</div>"


def empty(message: str) -> str:
    return f'<div class="sg-empty">{message}</div>'


def section(title: str) -> str:
    return f'<p class="sg-section">{escape(title)}</p>'


def when(ts: pd.Timestamp) -> str:
    return ts.tz_convert(TZ).strftime("%d %b, %H:%M")


def header(now: pd.Timestamp, playing: bool, source: str, notice: str | None) -> str:
    utc, local = now.tz_convert(TZ), now.tz_convert(IST)
    note = ""
    if notice:
        note = (f'<div class="sg-note">{escape(source)}: faults come from the fault injector and are '
                f'confirmed with simple physics and neighbour checks. {escape(notice)}.</div>')
    return f"""
<div class="sg-header">
  <div>
    <div class="sg-brand">Sky<span>Guard</span></div>
    <div class="sg-sub">Sensor integrity monitor for automatic weather stations &middot; Delhi-NCR, 8 stations</div>
  </div>
  <div class="sg-clock">
    <span class="sg-pill{' live' if playing else ''}">{'Replaying' if playing else 'Paused'}</span>
    <div class="sg-time">{utc:%H:%M} <span>UTC</span></div>
    <div class="sg-date">{utc:%a %d %b %Y} &middot; {local:%H:%M} IST</div>
  </div>
</div>{note}"""


def band(score: float) -> str:
    """HEALTHY, WATCH, DEGRADED or FAILED, from config.yaml -> trust.bands."""
    bands = load_config()["trust"]["bands"]
    for name in ("HEALTHY", "WATCH", "DEGRADED"):
        if score >= bands[name]:
            return name
    return "FAILED"


def tiles(k: dict) -> str:
    step = f"{k['step_ms']:.0f} ms" if k["step_ms"] is not None else "&ndash;"
    if k["lowest_trust"] is None:
        lowest, lowest_label, lowest_css = "&ndash;", "Lowest sensor trust", ""
    elif k["lowest_trust"] >= load_config()["trust"]["start"]:
        lowest, lowest_label, lowest_css = "100<small>/100</small>", "Every sensor at full trust", "trust-healthy"
    else:
        sid, var = k["lowest_sensor"]
        lowest = f"{k['lowest_trust']:.0f}<small>/100</small>"
        lowest_label = f"Lowest trust &middot; {escape(sid)} {NAMES[var].lower()}"
        lowest_css = "trust-" + band(k["lowest_trust"]).lower()
    items = [
        (lowest, lowest_label, lowest_css),
        (k["open_faults"], "Open faults", "alert" if k["open_faults"] else ""),
        (k["faults_24h"], "Faults, last 24 h", ""),
        (k["events_24h"], "Weather events, last 24 h", "weather" if k["events_24h"] else ""),
        (step, "Check time per hour", ""),
    ]
    cells = "".join(f'<div class="sg-tile {cls}"><b>{value}</b><span>{label}</span></div>'
                    for value, label, cls in items)
    return f'<div class="sg-tiles">{cells}</div>'


def trust_grid(trust: dict, stations: list[dict]) -> str:
    """Stations down, sensors across, each cell a score and a bar coloured by its band."""
    head = "".join(f"<th>{NAMES[v]}</th>" for v in ("temp_c", "pressure_hpa", "rh_pct"))
    rows = []
    for s in stations:
        cells = []
        for var in ("temp_c", "pressure_hpa", "rh_pct"):
            score = trust.get((s["station_id"], var))
            if score is None:
                cells.append('<td class="na">&ndash;</td>')
                continue
            css = band(score).lower()
            cells.append(f'<td class="{css}"><b>{score:.0f}</b>'
                         f'<span class="bar"><i style="width:{score:.0f}%"></i></span></td>')
        rows.append(f'<tr><th><b>{escape(s["station_id"])}</b> {escape(s["name"])}</th>{"".join(cells)}</tr>')
    return f'<table class="sg-trust"><thead><tr><th></th>{head}</tr></thead><tbody>{"".join(rows)}</tbody></table>'


def attention(todo, names: dict) -> str:
    """Sensors below trust 50, lowest first."""
    items = []
    for r in todo.itertuples(index=False):
        fault = f" &middot; last fault {r.last_fault.replace('_', ' ').lower()}" if isinstance(r.last_fault, str) else ""
        items.append(f'<li class="{band(r.trust).lower()}"><b>{r.trust:.0f}</b>'
                     f'<span>{escape(r.station_id)} {escape(names.get(r.station_id, ""))} &middot; '
                     f'{NAMES[r.variable]}<small>{band(r.trust).capitalize()}{fault}</small></span></li>')
    return f'<ul class="sg-attention">{"".join(items)}</ul>'


def _reading(value, unit: str) -> str:
    if value is None or pd.isna(value):
        return "no reading"
    return f"{value:g} {unit}" if abs(value) >= 1000 else f"{value:.1f} {unit}"


def _reasons(reasons) -> str:
    if not reasons:
        return ""
    return "<ul>" + "".join(f"<li>{escape(str(r))}</li>" for r in reasons) + "</ul>"


PHYSICS_FAIL = {
    "FLATLINE": "stuck",
    "RANGE": "out of range",
    "MISSING": "no reading",
    "DEWPOINT_MAX": "impossible dewpoint",
}
CHECK_HELP = {
    "Physics": "Valid range, impossible dewpoint, and the same value repeated for hours",
    "Neighbours": "Compared with the median of nearby stations; z is how many typical spreads away",
    "Pattern": "Isolation Forest score from the detection engine",
}


def _evidence(scores: dict) -> str:
    """The three checks behind a decision: pass, fail, or not available yet."""
    cfg = load_config()
    physics, z, ml = scores.get("physics"), scores.get("spatial_z"), scores.get("iforest")
    checks = [
        ("Physics", None if physics is None else physics < 0.5, "plausible",
         PHYSICS_FAIL.get(scores.get("physics_rule"), "implausible")),
        ("Neighbours", None if z is None else abs(z) <= cfg["spatial"]["z_threshold"],
         "in line" if z is None else f"in line, z {z:.1f}", "" if z is None else f"out of line, z {z:.1f}"),
        ("Pattern", None if ml is None else ml < cfg["ml"]["high"], "typical", "unusual"),
    ]
    pills = []
    for name, ok, good, bad in checks:
        tip = escape(CHECK_HELP[name])
        if ok is None:
            pills.append(f'<span class="sg-check na" title="{tip}">{name} <i>&ndash;</i></span>')
        else:
            mark, css, text = ("&#10003;", "pass", good) if ok else ("&#10007;", "fail", bad)
            pills.append(f'<span class="sg-check {css}" title="{tip}">{mark} {name} <i>{text}</i></span>')
    return f'<div class="sg-evidence">{"".join(pills)}</div>'


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
  {_evidence(item.get('scores') or {})}
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
    <span>{when(item['start'])} &ndash; {item['end'].tz_convert(TZ):%H:%M} {live}</span></div>
  <div class="sg-title">Weather event at {len(stations)} station{'s' if len(stations) > 1 else ''}</div>
  <div class="sg-kind">Genuine event &middot; no action needed</div>
  <div class="sg-values"><small>{escape(', '.join(stations))}</small></div>
  {_evidence(item.get('scores') or {})}
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
