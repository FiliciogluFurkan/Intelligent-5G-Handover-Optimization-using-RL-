"""Plotly figure builder functions — network map and time-series charts."""
import numpy as np
import plotly.graph_objects as go
from dash import html
import dash_bootstrap_components as dbc

from environment import HandoverEnv

BS_COLORS = ["red", "blue", "green"]
BS_COLORS_RGB = {
    "red":   "220,53,69",
    "blue":  "13,110,253",
    "green": "25,135,84",
}


def build_network_figure(env: HandoverEnv, step: int) -> go.Figure:
    """Canlı ağ haritası — BS'ler, kullanıcılar, bağlantı çizgileri."""
    fig = go.Figure()

    # Coverage circles
    for bs in env.base_stations:
        theta = np.linspace(0, 2 * np.pi, 60)
        cx = bs.position[0] + 150 * np.cos(theta)
        cy = bs.position[1] + 150 * np.sin(theta)
        fig.add_trace(go.Scatter(
            x=cx, y=cy, mode="lines",
            line=dict(color=BS_COLORS[bs.bs_id], width=1, dash="dot"),
            showlegend=False, hoverinfo="skip",
        ))

    # User → BS connection lines
    for user in env.users:
        if user.connected_bs is not None:
            bs = env.base_stations[user.connected_bs]
            fig.add_trace(go.Scatter(
                x=[user.position[0], bs.position[0]],
                y=[user.position[1], bs.position[1]],
                mode="lines",
                line=dict(color=BS_COLORS[user.connected_bs], width=0.5),
                showlegend=False, hoverinfo="skip",
            ))

    # Base stations
    for bs in env.base_stations:
        fig.add_trace(go.Scatter(
            x=[bs.position[0]], y=[bs.position[1]],
            mode="markers+text",
            marker=dict(symbol="square", size=22, color=BS_COLORS[bs.bs_id]),
            text=[f"BS{bs.bs_id + 1}<br>{len(bs.connected_users)} users"],
            textposition="bottom center",
            name=f"BS{bs.bs_id + 1}",
        ))

    # Users by type
    for utype, symbol, color in [
        ("pedestrian", "circle",      "orange"),
        ("vehicle",    "square",      "purple"),
        ("emergency",  "triangle-up", "crimson"),
    ]:
        users = [u for u in env.users if u.user_type == utype]
        if users:
            fig.add_trace(go.Scatter(
                x=[u.position[0] for u in users],
                y=[u.position[1] for u in users],
                mode="markers",
                marker=dict(symbol=symbol, size=10, color=color,
                            line=dict(width=2, color="white")),
                name=utype.capitalize(),
            ))

    fig.update_layout(
        title=dict(text=f"Network Map — Step {step}", font=dict(size=14)),
        xaxis=dict(showgrid=True, gridcolor="#e9ecef"),
        yaxis=dict(range=[0, 600], showgrid=True, gridcolor="#e9ecef",
                   autorange=False, scaleanchor="x"),
        plot_bgcolor="#f8f9fa",
        paper_bgcolor="white",
        margin=dict(l=20, r=20, t=40, b=60),
        legend=dict(orientation="h", y=-0.12),
        height=560,
    )
    return fig


def build_chart(values: list, title: str, color: str) -> go.Figure:
    """Tek renkli alan doldurmalı zaman serisi grafiği."""
    rgb = BS_COLORS_RGB.get(color, "100,100,100")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=values, mode="lines",
        line=dict(color=color, width=2),
        fill="tozeroy",
        fillcolor=f"rgba({rgb},0.08)",
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(size=13)),
        height=220,
        margin=dict(l=30, r=10, t=40, b=30),
        plot_bgcolor="#f8f9fa",
        paper_bgcolor="white",
        showlegend=False,
        xaxis=dict(showgrid=True, gridcolor="#e9ecef", title="Steps"),
        yaxis=dict(showgrid=True, gridcolor="#e9ecef"),
    )
    return fig


def build_bs_load_bars(env: HandoverEnv) -> list:
    """BS yük progress barları."""
    colors = ["danger", "primary", "success"]
    bars = []
    for bs in env.base_stations:
        load_pct = int(bs.get_load() * 100)
        bars.append(html.Div([
            html.Small(
                f"BS{bs.bs_id + 1}  —  {len(bs.connected_users)} users  ({load_pct}%)",
                className="text-muted",
            ),
            dbc.Progress(
                value=load_pct,
                color=colors[bs.bs_id],
                className="mb-2",
                style={"height": "10px"},
            ),
        ]))
    return bars
