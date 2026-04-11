"""Plotly figure builder functions — network map and time-series charts."""
import numpy as np
import plotly.graph_objects as go
from dash import html

from environment import HandoverEnv

# Design tokens — match values in assets/style.css
BS_COLORS   = ["#6366F1", "#3B82F6", "#10B981"]   # indigo / blue / emerald
USER_COLORS = {
    "pedestrian": "#F97316",   # orange-500
    "vehicle":    "#A855F7",   # purple-500
    "emergency":  "#E11D48",   # rose-600
}
USER_SYMBOLS = {
    "pedestrian": "circle-dot",
    "vehicle":    "arrow-up",    # rotated per-user to show movement direction
    "emergency":  "star",
}
USER_SIZES = {
    "pedestrian": 9,
    "vehicle":    13,
    "emergency":  14,
}

FONT_FAMILY = "Inter, system-ui, -apple-system, sans-serif"


def _sinr_color(sinr: float) -> str:
    """Map SINR (dB) to a colour: red (weak) → amber (medium) → green (strong)."""
    if sinr > 15:
        return "#10B981"   # emerald — good
    if sinr > 5:
        return "#F59E0B"   # amber  — medium
    return "#EF4444"       # red    — weak
GRID_COLOR  = "#F1F5F9"   # slate-100 — very light
AXIS_COLOR  = "#94A3B8"   # slate-400


def build_network_figure(env: HandoverEnv, step: int) -> go.Figure:
    fig = go.Figure()

    # Coverage circles (decorative — radius ~30% of area)
    radius = env.area_size * 0.3
    for bs in env.base_stations:
        theta = np.linspace(0, 2 * np.pi, 80)
        cx = bs.position[0] + radius * np.cos(theta)
        cy = bs.position[1] + radius * np.sin(theta)
        fig.add_trace(go.Scatter(
            x=cx, y=cy, mode="lines",
            line=dict(color=BS_COLORS[bs.bs_id], width=1.5, dash="dot"),
            opacity=0.35,
            showlegend=False, hoverinfo="skip",
        ))

    # User to BS connection lines — coloured by SINR quality
    for user in env.users:
        if user.connected_bs is not None:
            bs   = env.base_stations[user.connected_bs]
            sinr = bs.calculate_sinr(user.position)
            fig.add_trace(go.Scatter(
                x=[user.position[0], bs.position[0]],
                y=[user.position[1], bs.position[1]],
                mode="lines",
                line=dict(color=_sinr_color(sinr), width=1.2),
                opacity=0.7,
                showlegend=False, hoverinfo="skip",
            ))

    # Base stations
    for bs in env.base_stations:
        fig.add_trace(go.Scatter(
            x=[bs.position[0]],
            y=[bs.position[1]],
            mode="markers+text",
            marker=dict(
                symbol="square",
                size=18,
                color=BS_COLORS[bs.bs_id],
                line=dict(width=2, color="white"),
            ),
            text=[f"BS{bs.bs_id + 1}  {len(bs.connected_users)}u"],
            textposition="bottom center",
            textfont=dict(size=11, color=BS_COLORS[bs.bs_id],
                          family=FONT_FAMILY),
            name=f"BS{bs.bs_id + 1}",
            hovertemplate=(
                f"<b>BS{bs.bs_id + 1}</b><br>"
                f"Users: {len(bs.connected_users)}<br>"
                f"Load: {bs.get_load() * 100:.0f}%"
                "<extra></extra>"
            ),
        ))

    # Users (grouped by type)
    for utype in ("pedestrian", "vehicle", "emergency"):
        users = [u for u in env.users if u.user_type == utype]
        if not users:
            continue

        symbol = USER_SYMBOLS[utype]
        color  = USER_COLORS[utype]
        size   = USER_SIZES[utype]

        # Vehicles: rotate arrow marker to show actual movement direction
        # Plotly angle=0 → arrow points up; convert math radians → clockwise-from-north
        if utype == "vehicle":
            angles = [90 - float(np.degrees(u.direction)) for u in users]
        else:
            angles = 0

        speeds = {"pedestrian": "5 km/h", "vehicle": "60 km/h", "emergency": "120 km/h"}
        fig.add_trace(go.Scatter(
            x=[u.position[0] for u in users],
            y=[u.position[1] for u in users],
            mode="markers",
            marker=dict(
                symbol=symbol,
                size=size,
                color=color,
                angle=angles,
                line=dict(width=1.5, color="white"),
            ),
            name=f"{utype.capitalize()} ({speeds[utype]})",
            hovertemplate=(
                f"<b>{utype.capitalize()}</b><br>"
                "x: %{x:.0f}  y: %{y:.0f}"
                "<extra></extra>"
            ),
        ))

    # Layout — axes scale with env.area_size
    area = env.area_size
    fig.update_layout(
        title=dict(
            text=f"Step {step}",
            font=dict(size=12, color=AXIS_COLOR, family=FONT_FAMILY),
            x=0.01, y=0.99, xanchor="left", yanchor="top",
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(
            range=[0, area], autorange=False,
            showgrid=True, gridcolor=GRID_COLOR, gridwidth=1,
            zeroline=False,
            tickfont=dict(size=10, color=AXIS_COLOR, family=FONT_FAMILY),
            showline=False,
        ),
        yaxis=dict(
            range=[0, area], autorange=False,
            showgrid=True, gridcolor=GRID_COLOR, gridwidth=1,
            zeroline=False,
            tickfont=dict(size=10, color=AXIS_COLOR, family=FONT_FAMILY),
            showline=False,
            scaleanchor="x",
        ),
        margin=dict(l=8, r=8, t=28, b=28),
        legend=dict(
            orientation="h", y=-0.05, x=0,
            font=dict(size=11, color=AXIS_COLOR, family=FONT_FAMILY),
            bgcolor="rgba(0,0,0,0)",
            borderwidth=0,
        ),
        hoverlabel=dict(
            bgcolor="white",
            bordercolor=GRID_COLOR,
            font=dict(size=12, family=FONT_FAMILY),
        ),
    )
    return fig


def build_chart(values: list, _title: str = "", color: str = "#4F46E5") -> go.Figure:
    """Minimal area-filled time-series chart."""
    # Hex to rgba conversion for fill
    r, g, b = tuple(int(color.lstrip("#")[i:i+2], 16) for i in (0, 2, 4))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=values,
        mode="lines",
        line=dict(color=color, width=1.75),
        fill="tozeroy",
        fillcolor=f"rgba({r},{g},{b},0.06)",
        hovertemplate="Step %{x}: %{y:.2f}<extra></extra>",
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=8, r=8, t=8, b=8),
        showlegend=False,
        xaxis=dict(
            showgrid=False, zeroline=False,
            tickfont=dict(size=9, color=AXIS_COLOR, family=FONT_FAMILY),
            showline=False,
        ),
        yaxis=dict(
            showgrid=True, gridcolor=GRID_COLOR, gridwidth=1,
            zeroline=False,
            tickfont=dict(size=9, color=AXIS_COLOR, family=FONT_FAMILY),
            showline=False,
        ),
        hoverlabel=dict(
            bgcolor="white", bordercolor=GRID_COLOR,
            font=dict(size=11, family=FONT_FAMILY),
        ),
    )
    return fig


def build_bs_load_bars(env: HandoverEnv) -> list:
    """Base station load bars — custom HTML, styled via CSS."""
    bars = []
    for bs in env.base_stations:
        pct = int(bs.get_load() * 100)
        bars.append(html.Div([
            html.Div([
                html.Span(f"BS{bs.bs_id + 1}", className="bs-name"),
                html.Span(f"{pct}%", className="bs-pct"),
            ], className="bs-label-row"),
            html.Div(
                html.Div(className=f"bs-fill bs-fill-{bs.bs_id}",
                         style={"width": f"{pct}%"}),
                className="bs-track",
            ),
        ]))
    return bars
