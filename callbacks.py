"""Dash callbacks — tüm simülasyon mantığı burada."""
import os
import numpy as np
import dash
from dash import Input, Output, State, callback_context

from environment import HandoverEnv
import agents
from figures import build_network_figure, build_chart, build_bs_load_bars

# Modül seviyesinde env (tek process — thread-safe değil ama demo için yeterli)
env: HandoverEnv = HandoverEnv(area_size=600)
env.reset()


def register_callbacks(app: dash.Dash) -> None:
    """Tüm callback'leri verilen Dash uygulamasına kaydet."""

    # ── Kontrol butonları ─────────────────────────────────────────────────────
    @app.callback(
        Output("sim-state",   "data"),
        Output("interval",    "disabled"),
        Output("interval",    "interval"),
        Output("error-alert", "children"),
        Output("error-alert", "is_open"),
        Input("btn-start",     "n_clicks"),
        Input("btn-stop",      "n_clicks"),
        Input("btn-reset",     "n_clicks"),
        Input("algo-dropdown", "value"),
        State("sim-state",    "data"),
        State("speed-slider", "value"),
        prevent_initial_call=True,
    )
    def handle_controls(start, stop, reset, algorithm, state, speed):
        global env
        triggered   = callback_context.triggered_id
        alert_msg   = ""
        alert_open  = False
        interval_ms = max(100, 1000 // (speed or 3))

        if triggered == "btn-reset":
            env = HandoverEnv(area_size=600)
            env.reset()
            state = {
                "running":   False,
                "step":      0,
                "algorithm": algorithm,
                "history":   {"handovers": [], "sinr": [], "energy": []},
            }
            return state, True, interval_ms, alert_msg, alert_open

        if triggered == "btn-start":
            if algorithm in ("dqn", "ppo"):
                path = f"models/{algorithm}_handover.zip"
                if not os.path.exists(path):
                    alert_msg  = (f"Model bulunamadı: {path} — "
                                  f"önce train.py çalıştırın.")
                    alert_open = True
                    state["running"] = False
                    return state, True, interval_ms, alert_msg, alert_open
            state["running"]   = True
            state["algorithm"] = algorithm

        if triggered == "btn-stop":
            state["running"] = False

        if triggered == "algo-dropdown":
            state["algorithm"] = algorithm

        disabled = not state["running"]
        return state, disabled, interval_ms, alert_msg, alert_open

    # ── Simülasyon tick'i (her interval'da çalışır) ───────────────────────────
    @app.callback(
        Output("network-map",      "figure"),
        Output("metric-handovers", "children"),
        Output("metric-sinr",      "children"),
        Output("metric-pingpong",  "children"),
        Output("metric-emergency", "children"),
        Output("bs-load-bars",     "children"),
        Output("chart-handovers",  "figure"),
        Output("chart-sinr",       "figure"),
        Output("chart-energy",     "figure"),
        Output("sim-state",        "data",      allow_duplicate=True),
        Output("interval",         "disabled",  allow_duplicate=True),
        Output("error-alert",      "children",  allow_duplicate=True),
        Output("error-alert",      "is_open",   allow_duplicate=True),
        Input("interval", "n_intervals"),
        State("sim-state", "data"),
        prevent_initial_call=True,
    )
    def tick(n_intervals, state):
        global env

        if not state.get("running", False):
            raise dash.exceptions.PreventUpdate

        algorithm  = state.get("algorithm", "baseline")
        alert_msg  = ""
        alert_open = False

        try:
            action = agents.get_action(algorithm, env)
            _, _, terminated, _, _ = env.step(action)
        except Exception as exc:
            state["running"] = False
            alert_msg  = str(exc)
            alert_open = True
            return _pack_outputs(env, state, True, alert_msg, alert_open)

        if terminated:
            env.reset()

        # Geçmişi güncelle
        state["step"] += 1
        hist = state["history"]
        hist["handovers"].append(env.total_handovers)
        hist["sinr"].append(_avg_sinr(env))
        hist["energy"].append(_total_energy(env))

        return _pack_outputs(env, state, False, alert_msg, alert_open)


# ── Yardımcı fonksiyonlar ──────────────────────────────────────────────────────

def _avg_sinr(env: HandoverEnv) -> float:
    sinrs = [
        env.base_stations[u.connected_bs].calculate_sinr(u.position)
        for u in env.users if u.connected_bs is not None
    ]
    return float(np.mean(sinrs)) if sinrs else 0.0


def _total_energy(env: HandoverEnv) -> float:
    return sum(bs.total_energy for bs in env.base_stations)


def _pack_outputs(env, state, interval_disabled, alert_msg, alert_open):
    hist = state["history"]
    return (
        build_network_figure(env, state["step"]),
        str(env.total_handovers),
        f"{_avg_sinr(env):.1f}",
        str(env.ping_pong_count),
        str(env.emergency_disconnections),
        build_bs_load_bars(env),
        build_chart(hist["handovers"], "Handovers Over Time",      "blue"),
        build_chart(hist["sinr"],      "Avg SINR (dB) Over Time",  "green"),
        build_chart(hist["energy"],    "Energy Consumption",       "red"),
        state,
        interval_disabled,
        alert_msg,
        alert_open,
    )
