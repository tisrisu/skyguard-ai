"""Plotly figures for the dashboard. Colours match the presentation."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from skyguard.geo import haversine_km
from skyguard.schemas import FaultType, NAMES, UNITS, VARIABLES, Status

IST = "Asia/Kolkata"

INK = "#E4EBF3"
MUTED = "#8FA3BF"
GRID = "#1A2E4C"
PANEL = "#10223D"
LINE = "#7FB6DA"
SKY = "#2CA6E0"
AMBER = "#F5A623"
RED = "#E5484D"
GREEN = "#2FA37C"

STATUS_COLOR = {
    Status.NORMAL: GREEN,
    Status.GENUINE_EVENT: SKY,
    Status.SUSPECT: AMBER,
    Status.SENSOR_FAULT: RED,
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
        font=dict(color=INK, size=12),
        hoverlabel=dict(bgcolor=PANEL, bordercolor=GRID, font_color=INK),
    )
    fig.update_xaxes(gridcolor=GRID, zeroline=False, linecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zeroline=False, linecolor=GRID)
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
            style = ("rgba(229, 72, 77, 0.5)", 1.5)
        elif state[i] == state[j] == Status.GENUINE_EVENT:
            style = ("rgba(44, 166, 224, 0.7)", 2.5)
        else:
            style = ("rgba(127, 182, 218, 0.2)", 1)
        xs, ys = groups.setdefault(style, ([], []))
        xs += [x[i], x[j], None]
        ys += [y[i], y[j], None]
    for (color, width), (xs, ys) in groups.items():
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=color, width=width),
                                 hoverinfo="skip"))

    glow = [i for i, s in enumerate(state) if s != Status.NORMAL]
    fig.add_trace(go.Scatter(x=[x[i] for i in glow], y=[y[i] for i in glow], mode="markers",
                             marker=dict(size=44, color=[colors[i] for i in glow], opacity=0.22),
                             hoverinfo="skip"))
    fig.add_trace(go.Scatter(
        x=x, y=y, mode="markers+text", text=ids, textposition="top center",
        textfont=dict(color=INK, size=12),
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

    fig.update_xaxes(visible=False, range=[min(x) - 0.06, max(x) + 0.06])
    fig.update_yaxes(visible=False, range=[min(y) - 0.09, max(y) + 0.07], scaleanchor="x", scaleratio=1)
    fig.update_layout(showlegend=False, uirevision="network")
    return _style(fig, 430).update_layout(margin=dict(l=0, r=0, t=0, b=0))


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
    """
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                        subplot_titles=[f"{NAMES[v]} ({UNITS[v]})" for v in VARIABLES])
    ts = values["ts"]
    x = ts.dt.tz_convert(IST)
    weather = results.loc[results["status"] == Status.GENUINE_EVENT, "ts"]

    for row, var in enumerate(VARIABLES, start=1):
        unit = UNITS[var]
        first = row == 1
        lo, hi = cfg["physics"]["range"][var]
        reported = values[var].where(values[var].between(lo, hi))
        res = results[results["variable"] == var].set_index("ts").reindex(ts)
        res.index = values.index
        flagged = res["status"].isin([Status.SENSOR_FAULT, Status.SUSPECT]).to_numpy()
        corrected = res["corrected_value"].where(flagged)

        fig.add_trace(go.Scatter(
            x=median.index.tz_convert(IST), y=median[var], name="Neighbour median",
            line=dict(color=MUTED, width=1.2, dash="dot"), legendgroup="median", showlegend=first,
            hovertemplate=f"Neighbours %{{y:.1f}} {unit}<extra></extra>",
        ), row=row, col=1)
        fig.add_trace(go.Scatter(
            x=x, y=reported, name="Reported", line=dict(color=LINE, width=2),
            legendgroup="reported", showlegend=first,
            hovertemplate=f"Reported %{{y:.1f}} {unit}<extra></extra>",
        ), row=row, col=1)
        fig.add_trace(go.Scatter(
            x=x, y=corrected, name="Corrected", mode="lines+markers",
            line=dict(color=AMBER, width=2), marker=dict(size=5),
            legendgroup="corrected", showlegend=first,
            hovertemplate=f"Corrected %{{y:.1f}} {unit}<extra></extra>",
        ), row=row, col=1)

        # mark flagged hours at the reported value, or at the corrected one when the
        # reading itself can't be drawn (missing, -9999, off the scale)
        marks = reported.where(reported.notna(), corrected)[flagged]
        raw = values[var][flagged]
        fig.add_trace(go.Scatter(
            x=x[flagged], y=marks, mode="markers", name="Flagged",
            marker=dict(symbol="x-thin", size=11, color=RED, line=dict(width=2.5, color=RED)),
            customdata=np.column_stack([raw.map(lambda v: "no data" if pd.isna(v) else f"{v:g} {unit}"),
                                        res["fault_type"][flagged].fillna("").to_numpy()]),
            legendgroup="flagged", showlegend=first,
            hovertemplate="Flagged %{customdata[1]}: %{customdata[0]}<extra></extra>",
        ), row=row, col=1)

        for start, end in _hour_runs(weather):
            fig.add_vrect(x0=start.tz_convert(IST) - pd.Timedelta(minutes=30),
                          x1=end.tz_convert(IST) + pd.Timedelta(minutes=30),
                          fillcolor=SKY, opacity=0.13, line_width=0, row=row, col=1)

    fig.update_annotations(font=dict(size=12, color=MUTED), x=0, xanchor="left")
    fig.update_layout(hovermode="x unified", uirevision="station",
                      legend=dict(orientation="h", x=1, xanchor="right", y=1.1, font=dict(color=MUTED)))
    return _style(fig, 560)


def health_strip(results: pd.DataFrame, stations: list[dict], now: pd.Timestamp, hours: int = 72) -> go.Figure:
    """Status of every sensor for the last `hours`, one row per sensor."""
    codes = {s: i for i, s in enumerate(STATUS_COLOR)}
    recent = results[results["ts"] > now - pd.Timedelta(hours=hours)]
    if recent.empty:
        return _style(go.Figure(), 120)
    grid = (recent.assign(code=recent["status"].map(codes))
            .pivot_table(index=["station_id", "variable"], columns="ts", values="code", aggfunc="max"))
    rows = [(s["station_id"], v) for s in stations for v in VARIABLES]
    grid = grid.reindex(rows)
    labels = [f"{sid}  {NAMES[v]}" for sid, v in rows]
    order = list(STATUS_COLOR)
    names = np.vectorize(lambda c: "No data" if np.isnan(c) else STATUS_LABEL[order[int(c)]],
                         otypes=[str])(grid.to_numpy(dtype=float))

    steps = list(STATUS_COLOR.values())
    scale = []
    for i, color in enumerate(steps):
        scale += [[i / len(steps), color], [(i + 1) / len(steps), color]]

    fig = go.Figure(go.Heatmap(
        z=grid.to_numpy(), x=grid.columns.tz_convert(IST), y=labels,
        zmin=-0.5, zmax=len(steps) - 0.5, colorscale=scale, showscale=False,
        xgap=1, ygap=3, customdata=names,
        hovertemplate="%{y}<br>%{x|%d %b %H:%M}<br>%{customdata}<extra></extra>",
    ))
    fig.update_yaxes(autorange="reversed", showgrid=False, tickfont=dict(size=11, color=MUTED))
    fig.update_xaxes(showgrid=False)
    return _style(fig, 26 * len(rows) + 40).update_layout(margin=dict(t=4))


def scorecard_chart(per_fault: dict) -> go.Figure:
    kinds = [k for k in FaultType.FAULTS if k in per_fault]
    labels = [k.replace("_", " ").capitalize() for k in kinds]
    fig = go.Figure([
        go.Bar(y=labels, x=[per_fault[k]["recall"] for k in kinds], name="Recall",
               orientation="h", marker_color=SKY,
               hovertemplate="Recall %{x:.0%}<extra></extra>"),
        go.Bar(y=labels, x=[per_fault[k]["precision"] for k in kinds], name="Precision",
               orientation="h", marker_color=AMBER,
               hovertemplate="Precision %{x:.0%}<extra></extra>"),
    ])
    fig.update_layout(barmode="group", bargap=0.35,
                      legend=dict(orientation="h", x=1, xanchor="right", y=1.12, font=dict(color=MUTED)))
    fig.update_xaxes(range=[0, 1], tickformat=".0%")
    fig.update_yaxes(autorange="reversed", showgrid=False)
    return _style(fig, 360)
