"""Plotly figures for the dashboard.

The look borrows from the thermohygrograph drum charts weather stations used to
record on: cream graph paper, violet recorder ink, red for trouble.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from skyguard.geo import haversine_km
from skyguard.physics.dewpoint import dewpoint_c
from skyguard.schemas import FaultType, NAMES, UNITS, VARIABLES, Status

TZ = "UTC"               # observations are timed in UTC, as IMD reports them
IST = "Asia/Kolkata"

PAPER = "#F4EFE3"
PAPER_DEEP = "#EAE3D2"
INK = "#2A251F"
MUTED = "#7B705F"
GRID = "#DDD4C1"
GRID_FINE = "#EAE3D4"
VIOLET = "#5B3F95"       # recorder ink: what the sensor reported
SAFFRON = "#D98E1C"      # corrected values, suspects
VERMILION = "#D2452B"    # sensor faults
MONSOON = "#2E7D5B"      # genuine weather
GRAPHITE = "#8C8272"     # normal

MONO = '"Source Code Pro", "SFMono-Regular", Consolas, monospace'
SERIF = '"Source Serif 4", "Source Serif Pro", Georgia, serif'

STATUS_COLOR = {
    Status.NORMAL: GRAPHITE,
    Status.GENUINE_EVENT: MONSOON,
    Status.SUSPECT: SAFFRON,
    Status.SENSOR_FAULT: VERMILION,
}
STATUS_LABEL = {
    Status.NORMAL: "Normal",
    Status.GENUINE_EVENT: "Weather event",
    Status.SUSPECT: "Suspect",
    Status.SENSOR_FAULT: "Sensor fault",
}

PLOT_CONFIG = {"displayModeBar": False}


def _style(fig: go.Figure, height: int) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=8, r=8, t=30, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=INK, size=12, family=MONO),
        hoverlabel=dict(bgcolor=PAPER_DEEP, bordercolor=GRID, font=dict(color=INK, family=MONO)),
    )
    grid = dict(gridcolor=GRID, zeroline=False, linecolor=GRID,
                minor=dict(showgrid=True, gridcolor=GRID_FINE))
    fig.update_xaxes(**grid)
    fig.update_yaxes(**grid)
    return fig


def _links(stations: list[dict], per_station: int = 3) -> list[tuple[int, int]]:
    """Each station joined to its nearest few neighbours."""
    links = set()
    for i, s in enumerate(stations):
        nearest = sorted((haversine_km(s["lat"], s["lon"], t["lat"], t["lon"]), j)
                         for j, t in enumerate(stations) if j != i)
        links.update(tuple(sorted((i, j))) for _, j in nearest[:per_station])
    return sorted(links)


def network_map(stations: list[dict], status: dict, selected: str | None) -> go.Figure:
    """Stations at their true positions, joined to their nearest neighbours.

    Drawn without map tiles: a tile map reloads on every refresh and flashes during
    the replay, and this also works without internet.
    """
    ids = [s["station_id"] for s in stations]
    lat0 = np.radians(np.mean([s["lat"] for s in stations]))
    x = [s["lon"] * np.cos(lat0) for s in stations]      # true scale around Delhi
    y = [s["lat"] for s in stations]
    state = [status.get(i, {}).get("status", Status.NORMAL) for i in ids]
    colors = [STATUS_COLOR[s] for s in state]

    fig = go.Figure()
    # links light up when one end is faulty or both ends see the same weather event
    groups = {}
    for i, j in _links(stations):
        if Status.SENSOR_FAULT in (state[i], state[j]):
            style = ("rgba(210, 69, 43, 0.55)", 1.6)
        elif state[i] == state[j] == Status.GENUINE_EVENT:
            style = ("rgba(46, 125, 91, 0.75)", 2.6)
        else:
            style = ("rgba(42, 37, 31, 0.16)", 1)
        xs, ys = groups.setdefault(style, ([], []))
        xs += [x[i], x[j], None]
        ys += [y[i], y[j], None]
    for (color, width), (xs, ys) in groups.items():
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=color, width=width),
                                 hoverinfo="skip"))

    glow = [i for i, s in enumerate(state) if s != Status.NORMAL]
    fig.add_trace(go.Scatter(x=[x[i] for i in glow], y=[y[i] for i in glow], mode="markers",
                             marker=dict(size=46, color=[colors[i] for i in glow], opacity=0.2),
                             hoverinfo="skip"))
    fig.add_trace(go.Scatter(
        x=x, y=y, mode="markers+text", text=ids, textposition="top center",
        textfont=dict(color=INK, size=12, family=MONO),
        marker=dict(size=[20 if i == selected else 15 for i in ids], color=colors,
                    line=dict(width=3, color=[INK if i == selected else "rgba(0,0,0,0)" for i in ids])),
        customdata=[[s["name"], STATUS_LABEL[st], status.get(s["station_id"], {}).get("trust", 100)]
                    for s, st in zip(stations, state)],
        hovertemplate="<b>%{text}</b> %{customdata[0]}<br>%{customdata[1]}"
                      "<br>Lowest sensor trust %{customdata[2]:.0f}<extra></extra>",
    ))

    # 10 km scale bar, bottom right
    ten_km = 10 / 111.32
    right, bottom = max(x), min(y) - 0.05
    fig.add_trace(go.Scatter(x=[right - ten_km, right], y=[bottom, bottom], mode="lines",
                             line=dict(color=MUTED, width=2), hoverinfo="skip"))
    fig.add_annotation(x=right - ten_km / 2, y=bottom, text="10 km", showarrow=False, yshift=10,
                       font=dict(color=MUTED, size=11))

    fig.update_xaxes(visible=False, showgrid=False, minor_showgrid=False, range=[min(x) - 0.06, max(x) + 0.06])
    fig.update_yaxes(visible=False, showgrid=False, minor_showgrid=False, range=[min(y) - 0.09, max(y) + 0.07],
                     scaleanchor="x", scaleratio=1)
    fig.update_layout(showlegend=False, uirevision="network")
    fig = _style(fig, 430).update_layout(margin=dict(l=0, r=0, t=0, b=0))
    return fig.update_xaxes(showgrid=False, minor_showgrid=False).update_yaxes(showgrid=False, minor_showgrid=False)


def _hour_runs(stamps: pd.Series):
    """(first, last) of each run of consecutive hours."""
    stamps = pd.Series(sorted(stamps.unique()))
    if stamps.empty:
        return []
    starts = stamps.diff() != pd.Timedelta(hours=1)
    return [(g.iloc[0], g.iloc[-1]) for _, g in stamps.groupby(starts.cumsum())]


def station_chart(values: pd.DataFrame, results: pd.DataFrame, median: pd.DataFrame, cfg: dict) -> go.Figure:
    """Reported, corrected and neighbour median for one station, one panel per variable.

    values   the station's rows (as the sensor reported them) for the time window
    results  the station's results for the same window
    median   neighbour median per hour, index ts, one column per variable

    The temperature panel also carries the dewpoint implied by the reported
    temperature and humidity, against the highest dewpoint ever observed.
    """
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                        subplot_titles=[f"{NAMES[v]} ({UNITS[v]})" for v in VARIABLES])
    ts = values["ts"]
    x = ts.dt.tz_convert(TZ)
    weather = results.loc[results["status"] == Status.GENUINE_EVENT, "ts"]
    valid = {}

    for row, var in enumerate(VARIABLES, start=1):
        unit = UNITS[var]
        first = row == 1
        lo, hi = cfg["physics"]["range"][var]
        reported = values[var].where(values[var].between(lo, hi))
        valid[var] = reported
        res = results[results["variable"] == var].set_index("ts").reindex(ts)
        res.index = values.index
        flagged = res["status"].isin([Status.SENSOR_FAULT, Status.SUSPECT]).to_numpy()
        corrected = res["corrected_value"].where(flagged)

        fig.add_trace(go.Scatter(
            x=median.index.tz_convert(TZ), y=median[var], name="Neighbour median",
            line=dict(color=MUTED, width=1.3, dash="dot"), legendgroup="median", showlegend=first,
            hovertemplate=f"Neighbours %{{y:.1f}} {unit}<extra></extra>",
        ), row=row, col=1)
        fig.add_trace(go.Scatter(
            x=x, y=reported, name="Reported", line=dict(color=VIOLET, width=2.2, shape="spline", smoothing=0.4),
            legendgroup="reported", showlegend=first,
            hovertemplate=f"Reported %{{y:.1f}} {unit}<extra></extra>",
        ), row=row, col=1)
        fig.add_trace(go.Scatter(
            x=x, y=corrected, name="Corrected", mode="lines+markers",
            line=dict(color=SAFFRON, width=2.2), marker=dict(size=6, color=SAFFRON),
            legendgroup="corrected", showlegend=first,
            hovertemplate=f"Corrected %{{y:.1f}} {unit}<extra></extra>",
        ), row=row, col=1)

        # mark flagged hours at the reported value, or at the corrected one when the
        # reading itself can't be drawn (missing, -9999, off the scale)
        marks = reported.where(reported.notna(), corrected)[flagged]
        raw = values[var][flagged]
        fig.add_trace(go.Scatter(
            x=x[flagged], y=marks, mode="markers", name="Flagged",
            marker=dict(symbol="x-thin", size=12, color=VERMILION, line=dict(width=2.8, color=VERMILION)),
            customdata=np.column_stack([raw.map(lambda v: "no data" if pd.isna(v) else f"{v:g} {unit}"),
                                        res["fault_type"][flagged].fillna("").to_numpy()]),
            legendgroup="flagged", showlegend=first,
            hovertemplate="Flagged %{customdata[1]}: %{customdata[0]}<extra></extra>",
        ), row=row, col=1)

        for start, end in _hour_runs(weather):
            fig.add_vrect(x0=start.tz_convert(TZ) - pd.Timedelta(minutes=30),
                          x1=end.tz_convert(TZ) + pd.Timedelta(minutes=30),
                          fillcolor=MONSOON, opacity=0.12, line_width=0, row=row, col=1)

    limit = cfg["physics"]["dewpoint_max_c"]
    fig.add_trace(go.Scatter(
        x=x, y=dewpoint_c(valid["temp_c"], valid["rh_pct"]), name="Dewpoint",
        line=dict(color=MONSOON, width=1.6, dash="dash"),
        hovertemplate="Dewpoint %{y:.1f} °C<extra></extra>",
    ), row=1, col=1)
    fig.add_hline(y=limit, row=1, col=1, line=dict(color=VERMILION, width=1, dash="dot"),
                  annotation=dict(text=f"no dewpoint above ~{limit} °C has ever been observed",
                                  font=dict(size=10, color=VERMILION), xanchor="right", x=1, yanchor="bottom"))

    for title in (f"{NAMES[v]} ({UNITS[v]})" for v in VARIABLES):
        fig.update_annotations(selector=dict(text=title), font=dict(size=13, color=INK, family=SERIF),
                               x=0, xanchor="left")
    fig.update_layout(hovermode="x unified", uirevision="station",
                      legend=dict(orientation="h", x=0, xanchor="left", y=-0.12, yanchor="top",
                                  font=dict(color=MUTED)))
    return _style(fig, 620).update_layout(margin=dict(b=80))


def health_strip(results: pd.DataFrame, stations: list[dict], now: pd.Timestamp, hours: int = 72) -> go.Figure:
    """Status of every sensor for the last `hours`, one row per sensor."""
    order = list(STATUS_COLOR)
    recent = results[results["ts"] > now - pd.Timedelta(hours=hours)]
    if recent.empty:
        return _style(go.Figure(), 120)
    grid = (recent.assign(code=recent["status"].map({s: i for i, s in enumerate(order)}))
            .pivot_table(index=["station_id", "variable"], columns="ts", values="code", aggfunc="max"))
    rows = [(s["station_id"], v) for s in stations for v in VARIABLES]
    grid = grid.reindex(rows)
    labels = [f"{sid}  {NAMES[v]}" for sid, v in rows]
    names = np.vectorize(lambda c: "No data" if np.isnan(c) else STATUS_LABEL[order[int(c)]],
                         otypes=[str])(grid.to_numpy(dtype=float))

    # normal hours are pale so the few that matter stand out
    steps = ["#D9D0BC", MONSOON, SAFFRON, VERMILION]
    scale = []
    for i, color in enumerate(steps):
        scale += [[i / len(steps), color], [(i + 1) / len(steps), color]]

    fig = go.Figure(go.Heatmap(
        z=grid.to_numpy(), x=grid.columns.tz_convert(TZ), y=labels,
        zmin=-0.5, zmax=len(steps) - 0.5, colorscale=scale, showscale=False,
        xgap=1, ygap=3, customdata=names,
        hovertemplate="%{y}<br>%{x|%d %b %H:%M}<br>%{customdata}<extra></extra>",
    ))
    fig = _style(fig, 26 * len(rows) + 40).update_layout(margin=dict(t=4))
    fig.update_yaxes(autorange="reversed", showgrid=False, minor_showgrid=False, tickfont=dict(size=11, color=MUTED))
    fig.update_xaxes(showgrid=False, minor_showgrid=False)
    return fig


def scorecard_chart(per_fault: dict) -> go.Figure:
    kinds = [k for k in FaultType.FAULTS if k in per_fault]
    labels = [k.replace("_", " ").capitalize() for k in kinds]
    fig = go.Figure([
        go.Bar(y=labels, x=[per_fault[k]["recall"] for k in kinds], name="Recall",
               orientation="h", marker_color=VIOLET,
               hovertemplate="Recall %{x:.0%}<extra></extra>"),
        go.Bar(y=labels, x=[per_fault[k]["precision"] for k in kinds], name="Precision",
               orientation="h", marker_color=SAFFRON,
               hovertemplate="Precision %{x:.0%}<extra></extra>"),
    ])
    fig.update_layout(barmode="group", bargap=0.35,
                      legend=dict(orientation="h", x=1, xanchor="right", y=1.12, font=dict(color=MUTED)))
    fig = _style(fig, 360)
    fig.update_xaxes(range=[0, 1], tickformat=".0%")
    fig.update_yaxes(autorange="reversed", showgrid=False, minor_showgrid=False)
    return fig
