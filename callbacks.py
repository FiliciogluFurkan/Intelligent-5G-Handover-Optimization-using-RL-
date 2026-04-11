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
DASHBOARD_MAX_STEPS = 300
env: HandoverEnv = HandoverEnv(max_steps=DASHBOARD_MAX_STEPS)
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
            env = HandoverEnv(num_users=num_users, max_steps=DASHBOARD_MAX_STEPS)
            env.reset()
            state = {
                "running":        False,
                "step":           0,
                "algorithm":      algorithm,
                "num_users":      num_users,
                "episode_reward": 0.0,
                "run_results":    {},
                "history":        {"handovers": [], "sinr": [], "energy": [],
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
        Output("progress-fill",      "style"),
        Output("progress-label",     "children"),
        Output("handover-log",       "children"),
        Output("comparison-panel",   "children"),
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

        # Accumulate episode reward (reward not returned by step — proxy via SINR)
        state["episode_reward"] = state.get("episode_reward", 0.0) + _avg_sinr(env)

        if terminated:
            # Save this run's metrics before resetting
            algo_label = algorithm.upper()
            state.setdefault("run_results", {})[algo_label] = {
                "ho_rate":    round(env.total_handovers / max(env.time_step, 1), 3),
                "pp_rate":    round(env.ping_pong_count / max(env.total_handovers, 1), 3),
                "avg_sinr":   round(_avg_sinr(env), 1),
                "em_disc":    env.emergency_disconnections,
            }
            state["episode_reward"] = 0.0
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
        _build_comparison_panel(state.get("run_results", {})),
    )


_ALGO_COLORS = {
    "BASELINE": "#6B7280",
    "DQN":      "#3B82F6",
    "PPO":      "#10B981",
}

def _build_comparison_panel(run_results: dict):
    from dash import html as _html
    if not run_results:
        return [_html.P("Run at least one episode to see results.",
                        style={"color": "#94A3B8", "fontSize": "12px",
                               "padding": "12px 16px", "margin": 0})]

    _S = {"fontSize": "12px", "padding": "6px 12px", "fontFamily": "monospace"}
    _H = {"fontSize": "11px", "color": "#94A3B8", "padding": "4px 12px",
          "fontWeight": "600", "textTransform": "uppercase", "letterSpacing": "0.05em"}

    header = _html.Div([
        _html.Span("Algorithm",  style={**_H, "minWidth": "100px"}),
        _html.Span("HO / step",  style={**_H, "minWidth": "80px"}),
        _html.Span("Ping-Pong",  style={**_H, "minWidth": "80px"}),
        _html.Span("Avg SINR",   style={**_H, "minWidth": "80px"}),
        _html.Span("Em. Disc.",  style={**_H, "minWidth": "70px"}),
    ], style={"display": "flex", "borderBottom": "2px solid #E2E8F0"})

    rows = [header]
    for algo in ["BASELINE", "DQN", "PPO"]:
        if algo not in run_results:
            continue
        r     = run_results[algo]
        color = _ALGO_COLORS.get(algo, "#6B7280")
        pp_pct = f"{r['pp_rate']*100:.1f}%"
        rows.append(_html.Div([
            _html.Span(algo, style={**_S, "minWidth": "100px",
                                    "color": color, "fontWeight": "700",
                                    "fontFamily": "Inter, sans-serif"}),
            _html.Span(f"{r['ho_rate']:.3f}",  style={**_S, "minWidth": "80px"}),
            _html.Span(pp_pct,                  style={**_S, "minWidth": "80px",
                                                        "color": "#EF4444" if r["pp_rate"] > 0 else "#10B981",
                                                        "fontWeight": "600"}),
            _html.Span(f"{r['avg_sinr']} dB",  style={**_S, "minWidth": "80px"}),
            _html.Span(str(r["em_disc"]),       style={**_S, "minWidth": "70px"}),
        ], style={"display": "flex", "borderBottom": "1px solid #F1F5F9",
                  "alignItems": "center"}))
    return rows


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
