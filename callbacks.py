"""Dash callbacks — all simulation logic is handled here."""
import logging
import os
import numpy as np
import dash
from dash import Input, Output, State, callback_context

from environment import HandoverEnv
import agents
from figures import build_network_figure, build_chart, build_bs_load_bars

# Module-level env instance (single-process demo only — not thread-safe)
env: HandoverEnv = HandoverEnv()
env.reset()


def register_callbacks(app: dash.Dash) -> None:
    """Register all callbacks with the given Dash application."""

    # Control buttons: Start / Stop / Reset / Algorithm switch
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
        State("users-slider", "value"),
        prevent_initial_call=True,
    )
    def handle_controls(start, stop, reset, algorithm, state, speed, num_users):
        global env
        triggered   = callback_context.triggered_id
        alert_msg   = ""
        alert_open  = False
        interval_ms = max(100, 1000 // (speed or 3))
        num_users   = int(num_users or 15)

        if triggered == "btn-reset":
            env = HandoverEnv(num_users=num_users)
            env.reset()
            state = {
                "running":   False,
                "step":      0,
                "algorithm": algorithm,
                "num_users": num_users,
                "history":   {"handovers": [], "sinr": [], "energy": [],
                              "handover_log": []},
            }
            return state, True, interval_ms, alert_msg, alert_open

        if triggered == "btn-start":
            if algorithm in ("dqn", "ppo"):
                path = f"models/{algorithm}_handover.zip"
                if not os.path.exists(path):
                    alert_msg  = (f"Model not found: {path} — run train.py first.")
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

    # Simulation tick (fires on every interval)
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
        Output("progress-fill",    "style"),
        Output("progress-label",   "children"),
        Output("handover-log",     "children"),
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
            # Capture acting user before step for handover logging
            acting_idx  = env.current_user_idx
            acting_user = env.users[acting_idx]
            old_bs      = acting_user.connected_bs
            old_ho_count = env.total_handovers

            action = agents.get_action(algorithm, env)
            _, _, terminated, _, _ = env.step(action)

            # Record handover event if one occurred
            if env.total_handovers > old_ho_count:
                new_bs = acting_user.connected_bs
                log    = state["history"].setdefault("handover_log", [])
                log.insert(0, {
                    "step":      env.time_step,
                    "type":      acting_user.user_type,
                    "from_bs":   f"BS{old_bs + 1}" if old_bs is not None else "—",
                    "to_bs":     f"BS{new_bs + 1}" if new_bs is not None else "—",
                })
                state["history"]["handover_log"] = log[:10]   # keep last 10

        except Exception as exc:
            logging.exception("Simulation step error")
            state["running"] = False
            alert_msg  = "Simulation error — check server logs."
            alert_open = True
            return _pack_outputs(env, state, True, alert_msg, alert_open)

        if terminated:
            env.reset()

        # Update history
        state["step"] += 1
        hist = state["history"]
        hist["handovers"].append(env.total_handovers)
        hist["sinr"].append(_avg_sinr(env))
        hist["energy"].append(_energy_per_step(env, state["step"]))

        return _pack_outputs(env, state, False, alert_msg, alert_open)


# Helper functions

def _avg_sinr(env: HandoverEnv) -> float:
    sinrs = [
        env.base_stations[u.connected_bs].calculate_sinr(u.position)
        for u in env.users if u.connected_bs is not None
    ]
    return float(np.mean(sinrs)) if sinrs else 0.0


def _energy_per_step(env: HandoverEnv, step: int) -> float:
    """Return average energy per step to keep chart interpretable."""
    total = sum(bs.total_energy for bs in env.base_stations)
    return total / max(step, 1)


def _pack_outputs(env, state, interval_disabled, alert_msg, alert_open):
    from dash import html as _html
    hist      = state["history"]
    max_steps = env.max_steps
    cur_step  = env.time_step
    pct       = min(cur_step / max_steps * 100, 100)
    progress_style = {
        "width": f"{pct:.1f}%", "height": "100%",
        "background": "linear-gradient(90deg,#6366F1,#A855F7)",
        "borderRadius": "4px", "transition": "width 0.3s ease",
    }
    return (
        build_network_figure(env, state["step"]),
        str(env.total_handovers),
        f"{_avg_sinr(env):.1f}",
        str(env.ping_pong_count),
        str(env.emergency_disconnections),
        build_bs_load_bars(env),
        build_chart(hist["handovers"], "Handovers Over Time",      "#4F46E5"),
        build_chart(hist["sinr"],      "Avg SINR (dB) Over Time",  "#059669"),
        build_chart(hist["energy"],    "Energy / Step",            "#E11D48"),
        state,
        interval_disabled,
        alert_msg,
        alert_open,
        progress_style,
        f"{cur_step} / {max_steps}",
        _build_handover_log(hist.get("handover_log", [])),
    )


_TYPE_COLORS = {
    "pedestrian": "#F97316",
    "vehicle":    "#A855F7",
    "emergency":  "#E11D48",
}

def _build_handover_log(log: list):
    from dash import html as _html
    if not log:
        return [_html.P("No handovers yet.",
                        style={"color": "#94A3B8", "fontSize": "12px",
                               "padding": "8px 12px", "margin": 0})]
    rows = []
    for entry in log:
        color = _TYPE_COLORS.get(entry["type"], "#6B7280")
        rows.append(_html.Div([
            _html.Span(f"t{entry['step']}",
                       style={"color": "#94A3B8", "fontSize": "11px",
                               "minWidth": "36px", "fontFamily": "monospace"}),
            _html.Span(entry["type"].capitalize(),
                       style={"color": color, "fontSize": "11px",
                               "minWidth": "80px", "fontWeight": "600"}),
            _html.Span(f"{entry['from_bs']} → {entry['to_bs']}",
                       style={"color": "#475569", "fontSize": "11px",
                               "fontFamily": "monospace"}),
        ], style={"display": "flex", "gap": "8px", "padding": "3px 12px",
                  "borderBottom": "1px solid #F1F5F9"}))
    return rows
