"""Dash layout — all UI components are defined here."""
from dash import dcc, html
import dash_bootstrap_components as dbc

from environment import HandoverEnv
from figures import build_network_figure, build_chart


def create_layout(env: HandoverEnv) -> html.Div:
    empty_net   = build_network_figure(env, 0)

    return html.Div([

        # ── Top App Bar ────────────────────────────────────────────────────
        html.Div([
            html.Div(className="topbar-dot"),
            html.H1("5G Handover Optimization", className="topbar-title"),
            html.Span("Live Simulation", className="topbar-badge"),
        ], className="topbar"),

        # ── Page Content ───────────────────────────────────────────────────
        html.Div([

            # State & Timer
            dcc.Store(id="sim-state", data={
                "running": False, "step": 0, "algorithm": "baseline",
                "num_users": 15,
                "episode_reward": 0.0,
                "run_results": {},
                "history": {"handovers": [], "sinr": [], "energy": [],
                            "handover_log": []},
            }),
            dcc.Interval(id="interval", interval=500, n_intervals=0, disabled=True),

            # Error alert
            dbc.Alert(id="error-alert", is_open=False,
                      color="danger", dismissable=True,
                      className="mb-3"),

            # ── Toolbar ────────────────────────────────────────────────────
            html.Div([

                html.Div([
                    html.Span("Algorithm", className="input-label"),
                    dcc.Dropdown(
                        id="algo-dropdown",
                        options=[
                            {"label": "Baseline — Greedy SINR", "value": "baseline"},
                            {"label": "DQN — Deep Q-Network",   "value": "dqn"},
                            {"label": "PPO — Proximal Policy",  "value": "ppo"},
                        ],
                        value="baseline",
                        clearable=False,
                    ),
                ], className="toolbar-group wide"),

                html.Div(className="toolbar-divider"),

                html.Div([
                    html.Span("Simulation Speed", className="input-label"),
                    dcc.Slider(
                        id="speed-slider", min=1, max=10, step=1, value=3,
                        marks={i: str(i) for i in range(1, 11)},
                        tooltip={"always_visible": False},
                    ),
                ], className="toolbar-group wider"),

                html.Div(className="toolbar-divider"),

                html.Div([
                    html.Span("Users", className="input-label"),
                    dcc.Slider(
                        id="users-slider", min=5, max=30, step=5, value=15,
                        marks={v: str(v) for v in [5, 10, 15, 20, 25, 30]},
                        tooltip={"always_visible": False},
                    ),
                ], className="toolbar-group wider"),

                html.Div(className="toolbar-divider"),

                html.Div([
                    html.Button("▶  Start",  id="btn-start",
                                className="btn-primary-custom"),
                    html.Button("⏸  Stop",   id="btn-stop",
                                className="btn-outline-custom"),
                    html.Button("↺  Reset",  id="btn-reset",
                                className="btn-outline-custom"),
                ], className="btn-group"),

            ], className="toolbar mb-4"),

            # ── Episode Progress Bar ───────────────────────────────────────
            html.Div([
                html.Div([
                    html.Span("Episode Progress", className="input-label",
                              style={"marginRight": "12px", "whiteSpace": "nowrap"}),
                    html.Div(
                        html.Div(id="progress-fill",
                                 style={"width": "0%", "height": "100%",
                                        "background": "linear-gradient(90deg,#6366F1,#A855F7)",
                                        "borderRadius": "4px",
                                        "transition": "width 0.3s ease"}),
                        style={"flex": "1", "height": "8px",
                               "background": "#E2E8F0", "borderRadius": "4px"},
                    ),
                    html.Span(id="progress-label", children="0 / 1000",
                              style={"marginLeft": "12px", "fontSize": "12px",
                                     "color": "#94A3B8", "whiteSpace": "nowrap"}),
                ], style={"display": "flex", "alignItems": "center",
                          "padding": "10px 16px"}),
            ], className="card-flat mb-4"),

            # ── Main content ───────────────────────────────────────────────
            dbc.Row([

                # Network map
                dbc.Col(
                    html.Div([
                        html.Div([
                            html.Span("📡", style={"fontSize": "14px"}),
                            html.P("Live Network Map", className="card-header-label"),
                        ], className="card-header-flat"),
                        html.Div(
                            dcc.Graph(id="network-map", figure=empty_net,
                                      config={"displayModeBar": False},
                                      style={"height": "520px"}),
                            className="card-body-p0",
                        ),
                    ], className="card-flat"),
                    width=8,
                ),

                # Right panel
                dbc.Col([

                    # Metrics grid
                    html.Div([
                        _metric_cell("Total Handovers", "metric-handovers",
                                     "0",   "c-indigo"),
                        _metric_cell("Avg SINR (dB)",   "metric-sinr",
                                     "0.0", "c-emerald"),
                        _metric_cell("Ping-Pong",        "metric-pingpong",
                                     "0",   "c-amber"),
                        _metric_cell("Emergency Disc.",  "metric-emergency",
                                     "0",   "c-rose"),
                    ], className="card-flat metrics-grid mb-3"),

                    # BS Load
                    html.Div([
                        html.Div([
                            html.Span("📶", style={"fontSize": "14px"}),
                            html.P("Base Station Load",
                                   className="card-header-label"),
                        ], className="card-header-flat"),
                        html.Div(id="bs-load-bars", className="bs-row"),
                    ], className="card-flat mb-3"),

                    # Handover Log
                    html.Div([
                        html.Div([
                            html.Span("🔀", style={"fontSize": "14px"}),
                            html.P("Handover Log", className="card-header-label"),
                            html.Span("t = time step",
                                      style={"fontSize": "10px", "color": "#94A3B8",
                                             "marginLeft": "auto"}),
                        ], className="card-header-flat"),
                        html.Div(id="handover-log", children=[
                            html.P("No handovers yet.",
                                   style={"color": "#94A3B8", "fontSize": "12px",
                                          "padding": "8px 12px", "margin": 0}),
                        ]),
                    ], className="card-flat mb-3"),

                    # Legend
                    html.Div([
                        html.Div([
                            html.Span("◎", style={"fontSize": "14px",
                                                   "color": "#94A3B8"}),
                            html.P("User Types", className="card-header-label"),
                        ], className="card-header-flat"),
                        html.Div([
                            html.Div([
                                html.Span("●", style={"color": "#F97316",
                                                       "fontSize": "16px",
                                                       "lineHeight": "1"}),
                                html.Span("  Pedestrian  5 km/h"),
                            ], className="legend-item"),
                            html.Div([
                                html.Span("➤", style={"color": "#A855F7",
                                                       "fontSize": "14px",
                                                       "lineHeight": "1"}),
                                html.Span("  Vehicle  60 km/h"),
                            ], className="legend-item"),
                            html.Div([
                                html.Span("★", style={"color": "#E11D48",
                                                       "fontSize": "16px",
                                                       "lineHeight": "1"}),
                                html.Span("  Emergency  120 km/h"),
                            ], className="legend-item"),
                        ], className="legend-row"),
                    ], className="card-flat"),

                ], width=4),
            ], className="mb-4 g-3"),

            # ── Time-series charts ─────────────────────────────────────────
            dbc.Row([
                dbc.Col(_chart_card("chart-handovers",
                                    "Handovers", "📈"), width=4),
                dbc.Col(_chart_card("chart-sinr",
                                    "Avg SINR (dB)", "📊"), width=4),
                dbc.Col(_chart_card("chart-energy",
                                    "Energy Consumption",""), width=4),
            ], className="g-3 mb-4"),

            # ── Run comparison panel ────────────────────────────────────────
            html.Div([
                html.Div([
                    html.Span("📊", style={"fontSize": "14px"}),
                    html.P("Run Comparison", className="card-header-label"),
                    html.Span("Updates at end of each episode",
                              style={"fontSize": "10px", "color": "#94A3B8",
                                     "marginLeft": "auto"}),
                ], className="card-header-flat"),
                html.Div(id="comparison-panel", children=[
                    html.P("Run at least one episode to see results.",
                           style={"color": "#94A3B8", "fontSize": "12px",
                                  "padding": "12px 16px", "margin": 0}),
                ]),
            ], className="card-flat"),

        ], className="page-wrapper"),
    ])


# ── Helpers ────────────────────────────────────────────────────────────────

def _metric_cell(label, element_id, init, color_cls):
    return html.Div([
        html.P(label, className="metric-label"),
        html.P(init,  id=element_id,
               className=f"metric-value {color_cls}"),
    ], className="metrics-cell")


def _chart_card(graph_id, title, icon):
    return html.Div([
        html.Div([
            html.Span(icon, style={"fontSize": "14px"}),
            html.P(title, className="card-header-label"),
        ], className="card-header-flat"),
        html.Div(
            dcc.Graph(id=graph_id,
                      figure=build_chart([], title, "#4F46E5"),
                      config={"displayModeBar": False},
                      style={"height": "200px"}),
            className="card-body-p0",
        ),
    ], className="card-flat")
