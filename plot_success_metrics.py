"""
Focused success-metric visualization:
  1. Handover rate reduction vs Baseline
  2. Ping-pong rate reduction vs Baseline
  3. Emergency disconnections (connection continuity)
  4. Improvement summary (% reduction)
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from stable_baselines3 import DQN, PPO
from environment import HandoverEnv

EVAL_SEEDS = [42, 123, 456, 789, 1337, 1111, 2222, 3333, 4444, 5555]
NUM_EPISODES = 10


def evaluate_baseline():
    rewards, handover_rates, ping_pong_rates, emergency_disc = [], [], [], []
    env = HandoverEnv()
    for ep in range(NUM_EPISODES):
        obs, _ = env.reset(seed=EVAL_SEEDS[ep])
        done = False
        while not done:
            user = env.users[env.current_user_idx]
            sinrs = [bs.calculate_sinr(user.position) for bs in env.base_stations]
            action = int(np.argmax(sinrs))
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
        handover_rates.append(env.total_handovers / max(env.time_step, 1))
        ping_pong_rates.append(env.ping_pong_count / max(env.total_handovers, 1))
        emergency_disc.append(env.emergency_disconnections)
    env.close()
    return handover_rates, ping_pong_rates, emergency_disc


def evaluate_agent(model):
    handover_rates, ping_pong_rates, emergency_disc = [], [], []
    env = HandoverEnv()
    for ep in range(NUM_EPISODES):
        obs, _ = env.reset(seed=EVAL_SEEDS[ep])
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
        handover_rates.append(env.total_handovers / max(env.time_step, 1))
        ping_pong_rates.append(env.ping_pong_count / max(env.total_handovers, 1))
        emergency_disc.append(env.emergency_disconnections)
    env.close()
    return handover_rates, ping_pong_rates, emergency_disc


print("Evaluating Baseline...")
b_ho, b_pp, b_ed = evaluate_baseline()

print("Evaluating DQN (best)...")
dqn = DQN.load("models/best_dqn/best_model.zip" if os.path.exists("models/best_dqn/best_model.zip") else "models/dqn_handover")
d_ho, d_pp, d_ed = evaluate_agent(dqn)

print("Evaluating PPO (best)...")
ppo = PPO.load("models/best_ppo/best_model.zip" if os.path.exists("models/best_ppo/best_model.zip") else "models/ppo_handover")
p_ho, p_pp, p_ed = evaluate_agent(ppo)

# ---- compute means / stds ----
methods  = ["Baseline", "DQN", "PPO"]
ho_means = [np.mean(b_ho), np.mean(d_ho), np.mean(p_ho)]
ho_stds  = [np.std(b_ho),  np.std(d_ho),  np.std(p_ho)]
pp_means = [np.mean(b_pp), np.mean(d_pp), np.mean(p_pp)]
pp_stds  = [np.std(b_pp),  np.std(d_pp),  np.std(p_pp)]
ed_means = [np.mean(b_ed), np.mean(d_ed), np.mean(p_ed)]
ed_stds  = [np.std(b_ed),  np.std(d_ed),  np.std(p_ed)]

# % improvement vs baseline (lower is better for HO and PP)
dqn_ho_pct = (ho_means[0] - ho_means[1]) / ho_means[0] * 100
ppo_ho_pct = (ho_means[0] - ho_means[2]) / ho_means[0] * 100
dqn_pp_pct = (pp_means[0] - pp_means[1]) / pp_means[0] * 100 if pp_means[0] > 0 else 0
ppo_pp_pct = (pp_means[0] - pp_means[2]) / pp_means[0] * 100 if pp_means[0] > 0 else 0

print(f"\nDQN Handover reduction vs Baseline: {dqn_ho_pct:.1f}%")
print(f"PPO Handover reduction vs Baseline: {ppo_ho_pct:.1f}%")
print(f"DQN Ping-Pong reduction vs Baseline: {dqn_pp_pct:.1f}%")
print(f"PPO Ping-Pong reduction vs Baseline: {ppo_pp_pct:.1f}%")
print(f"Emergency disconnections — Baseline: {ed_means[0]:.1f}, DQN: {ed_means[1]:.1f}, PPO: {ed_means[2]:.1f}")

# ---- plot ----
colors = ["#6B7280", "#3B82F6", "#10B981"]
fig, axes = plt.subplots(1, 3, figsize=(15, 6))
fig.suptitle("5G Handover Optimization — RL vs Greedy Baseline\nSuccess Metrics",
             fontsize=14, fontweight="bold", y=1.02)


def bar_panel(ax, means, stds, title, ylabel, lower_is_better=True, note=None):
    bars = ax.bar(methods, means, yerr=stds, capsize=7,
                  color=colors, alpha=0.88, edgecolor="white", linewidth=1.5,
                  error_kw={"elinewidth": 2, "ecolor": "#374151"})
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for bar, mean, std in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + std + max(means) * 0.015,
                f"{mean:.3f}" if mean < 1 else f"{mean:.1f}",
                ha="center", va="bottom", fontsize=9, fontweight="bold")
    if lower_is_better:
        ax.text(0.97, 0.97, "lower = better", transform=ax.transAxes,
                ha="right", va="top", fontsize=8, color="#6B7280", style="italic")
    if note:
        ax.text(0.5, -0.18, note, transform=ax.transAxes,
                ha="center", va="top", fontsize=8, color="#EF4444",
                style="italic", wrap=True)


bar_panel(axes[0], ho_means, ho_stds,
          "Handover Rate\n(HOs / time-step)",
          "HOs / time-step",
          lower_is_better=True,
          note=f"DQN: {dqn_ho_pct:.0f}% less  |  PPO: {ppo_ho_pct:.0f}% less than Baseline")

bar_panel(axes[1], pp_means, pp_stds,
          "Ping-Pong Rate\n(fraction of HOs)",
          "Fraction",
          lower_is_better=True,
          note=f"DQN: {dqn_pp_pct:.0f}% less  |  PPO: {ppo_pp_pct:.0f}% less than Baseline")

bar_panel(axes[2], ed_means, ed_stds,
          "Emergency Disconnections\n(connection continuity)",
          "Count per episode",
          lower_is_better=True,
          note="Baseline=0: RL agents sacrifice some continuity\nto avoid unnecessary handovers")

plt.tight_layout()
os.makedirs("figures", exist_ok=True)
out = "figures/success_metrics.png"
plt.savefig(out, dpi=300, bbox_inches="tight")
print(f"\nSaved to '{out}'")
