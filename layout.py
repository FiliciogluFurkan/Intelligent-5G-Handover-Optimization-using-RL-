"""Dash layout — tüm UI bileşenleri burada tanımlı."""
from dash import dcc, html
import dash_bootstrap_components as dbc

from environment import HandoverEnv
from figures import build_network_figure, build_chart


def create_layout(env: HandoverEnv) -> dbc.Container:
    """Ana layout'u oluştur ve döndür."""

    # Başlangıç figürleri
    empty_net   = build_network_figure(env, 0)
    empty_chart = build_chart([], "", "blue")

    return dbc.Container(fluid=True, children=[

        # ── State & Timer ─────────────────────────────────────────────────
        dcc.Store(id="sim-state", data={
            "running":   False,
            "step":      0,
            "algorithm": "baseline",
            "history":   {"handovers": [], "sinr": [], "energy": []},
        }),
        dcc.Interval(id="interval", interval=500, n_intervals=0, disabled=True),

        # ── Header ────────────────────────────────────────────────────────
        dbc.Row(
            dbc.Col(html.H3(
                "5G Handover Optimization Dashboard",
                className="header-title",
            )),
            className="mb-3 mt-2",
        ),

        # ── Hata Bildirimi ────────────────────────────────────────────────
        dbc.Row(dbc.Col(
            dbc.Alert(id="error-alert", is_open=False,
                      color="danger", dismissable=True, className="mb-2"),
        )),

        # ── Kontrol Çubuğu ────────────────────────────────────────────────
        dbc.Row([
            dbc.Col([
                html.Label("Algorithm", className="control-label"),
                dcc.Dropdown(
                    id="algo-dropdown",
                    options=[
                        {"label": "Baseline (Greedy SINR)", "value": "baseline"},
                        {"label": "DQN — Deep Q-Network",  "value": "dqn"},
                        {"label": "PPO — Proximal Policy",  "value": "ppo"},
                    ],
                    value="baseline",
                    clearable=False,
                    className="algo-dropdown",
                ),
            ], width=3),

            dbc.Col([
                html.Label("Simulation Speed", className="control-label"),
                dcc.Slider(
                    id="speed-slider", min=1, max=10, step=1, value=3,
                    marks={i: str(i) for i in range(1, 11)},
                    tooltip={"always_visible": False},
                ),
            ], width=4),

            dbc.Col([
                dbc.Button("▶ Start", id="btn-start",
                           color="success", className="control-btn me-2"),
                dbc.Button("⏸ Stop", id="btn-stop",
                           color="warning", className="control-btn me-2"),
                dbc.Button(" Reset", id="btn-reset",
                           color="secondary", className="control-btn"),
            ], width=5, className="d-flex align-items-end pb-1"),
        ], className="control-bar mb-3 align-items-end"),

        # ── Harita + Metrikler ────────────────────────────────────────────
        dbc.Row([

            # Canlı ağ haritası
            dbc.Col(
                dbc.Card([
                    dbc.CardHeader("📡 Live Network Map",
                                   className="card-header-custom"),
                    dbc.CardBody(
                        dcc.Graph(id="network-map", figure=empty_net,
                                  config={"displayModeBar": False}),
                        className="p-1",
                    ),
                ], className="dashboard-card"),
                width=8,
            ),

            # Metrik paneli
            dbc.Col([

                # 4 metrik kartı
                dbc.Row([
                    dbc.Col(_metric_card("Total Handovers", "metric-handovers",
                                         "0", "metric-blue"),  width=6),
                    dbc.Col(_metric_card("Avg SINR (dB)",   "metric-sinr",
                                         "0.0", "metric-green"), width=6),
                ], className="mb-2"),

                dbc.Row([
                    dbc.Col(_metric_card("Ping-Pong",       "metric-pingpong",
                                         "0", "metric-orange"), width=6),
                    dbc.Col(_metric_card("Emergency Disc.", "metric-emergency",
                                         "0", "metric-red"),    width=6),
                ], className="mb-2"),

                # BS yük barları
                dbc.Card([
                    dbc.CardHeader("📶 Baz İstasyonu Yükü",
                                   className="card-header-custom"),
                    dbc.CardBody(id="bs-load-bars"),
                ], className="dashboard-card mb-2"),

                # Kullanıcı tipi açıklaması
                dbc.Card(
                    dbc.CardBody([
                        html.Small("User Types",
                                   className="fw-bold d-block mb-2 text-muted"),
                        html.Span("● Pedestrian (5 km/h)   ",
                                  style={"color": "orange", "fontSize": "12px"}),
                        html.Br(),
                        html.Span("■ Vehicle (60 km/h)   ",
                                  style={"color": "purple", "fontSize": "12px"}),
                        html.Br(),
                        html.Span("▲ Emergency (120 km/h)",
                                  style={"color": "crimson", "fontSize": "12px"}),
                    ]),
                    className="dashboard-card",
                ),

            ], width=4),

        ], className="mb-3"),

        # ── Zaman Serisi Grafikleri ───────────────────────────────────────
        dbc.Row([
            dbc.Col(
                dbc.Card([
                    dbc.CardHeader("📈 Handover Sayısı",
                                   className="card-header-custom"),
                    dbc.CardBody(
                        dcc.Graph(id="chart-handovers",
                                  figure=build_chart([], "Handovers Over Time", "blue"),
                                  config={"displayModeBar": False}),
                        className="p-1",
                    ),
                ], className="dashboard-card"),
                width=4,
            ),
            dbc.Col(
                dbc.Card([
                    dbc.CardHeader("📊 Average SINR (dB)",
                                   className="card-header-custom"),
                    dbc.CardBody(
                        dcc.Graph(id="chart-sinr",
                                  figure=build_chart([], "Avg SINR (dB) Over Time", "green"),
                                  config={"displayModeBar": False}),
                        className="p-1",
                    ),
                ], className="dashboard-card"),
                width=4,
            ),
            dbc.Col(
                dbc.Card([
                    dbc.CardHeader("⚡ Energy Consumption",
                                   className="card-header-custom"),
                    dbc.CardBody(
                        dcc.Graph(id="chart-energy",
                                  figure=build_chart([], "Energy Over Time", "red"),
                                  config={"displayModeBar": False}),
                        className="p-1",
                    ),
                ], className="dashboard-card"),
                width=4,
            ),
        ]),

    ], style={"maxWidth": "1400px", "margin": "0 auto"})


# ── Yardımcı ──────────────────────────────────────────────────────────────────

def _metric_card(label: str, element_id: str,
                 init_value: str, color_class: str) -> dbc.Card:
    return dbc.Card(
        dbc.CardBody([
            html.P(label, className="metric-label"),
            html.H3(init_value, id=element_id, className=f"metric-value {color_class}"),
        ]),
        className="dashboard-card metric-card",
    )
