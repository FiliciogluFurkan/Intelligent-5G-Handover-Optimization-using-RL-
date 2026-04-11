import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from stable_baselines3 import DQN, PPO
from environment import HandoverEnv

# Fixed seeds for reproducible, fair comparison across all algorithms
EVAL_SEEDS = [42, 123, 456, 789, 1337, 1111, 2222, 3333, 4444, 5555]
NUM_EPISODES = 10


def evaluate_agent(model, num_episodes=NUM_EPISODES):
    """Evaluate a trained RL agent over multiple seeded episodes."""
    rewards, handover_rates, avg_sinrs, ping_pong_rates, emergency_disc = [], [], [], [], []

    env = HandoverEnv()
    for ep in range(num_episodes):
        obs, _ = env.reset(seed=EVAL_SEEDS[ep])
        episode_reward = 0.0
        sinr_values = []
        done = False

        while not done:
            acting_user_idx = env.current_user_idx
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward

            user = env.users[acting_user_idx]
            if user.connected_bs is not None:
                sinr = env.base_stations[user.connected_bs].calculate_sinr(user.position)
                sinr_values.append(sinr)

            done = terminated or truncated

        rewards.append(episode_reward)
        handover_rates.append(env.total_handovers / max(env.time_step, 1))
        avg_sinrs.append(float(np.mean(sinr_values)) if sinr_values else 0.0)
        ping_pong_rates.append(env.ping_pong_count / max(env.total_handovers, 1))
        emergency_disc.append(env.emergency_disconnections)

    env.close()
    return {
        "rewards": rewards,
        "handover_rates": handover_rates,
        "avg_sinrs": avg_sinrs,
        "ping_pong_rates": ping_pong_rates,
        "emergency_disc": emergency_disc,
    }


def evaluate_baseline(num_episodes=NUM_EPISODES):
    """Evaluate the greedy SINR baseline over multiple seeded episodes."""
    rewards, handover_rates, avg_sinrs, ping_pong_rates, emergency_disc = [], [], [], [], []

    env = HandoverEnv()
    for ep in range(num_episodes):
        obs, _ = env.reset(seed=EVAL_SEEDS[ep])
        episode_reward = 0.0
        sinr_values = []
        done = False

        while not done:
            acting_user_idx = env.current_user_idx
            user = env.users[acting_user_idx]
            sinrs = [bs.calculate_sinr(user.position) for bs in env.base_stations]
            action = int(np.argmax(sinrs))

            obs, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            sinr_values.append(max(sinrs))

            done = terminated or truncated

        rewards.append(episode_reward)
        handover_rates.append(env.total_handovers / max(env.time_step, 1))
        avg_sinrs.append(float(np.mean(sinr_values)) if sinr_values else 0.0)
        ping_pong_rates.append(env.ping_pong_count / max(env.total_handovers, 1))
        emergency_disc.append(env.emergency_disconnections)

    env.close()
    return {
        "rewards": rewards,
        "handover_rates": handover_rates,
        "avg_sinrs": avg_sinrs,
        "ping_pong_rates": ping_pong_rates,
        "emergency_disc": emergency_disc,
    }


def _mean_std(values):
    return float(np.mean(values)), float(np.std(values))


def _load_model(algo: str, use_best: bool):
    """Load final or best checkpoint for a given algorithm."""
    cls = DQN if algo == "dqn" else PPO
    if use_best:
        path = f"models/best_{algo}/best_model.zip"
        if not os.path.exists(path):
            print(f"  [warn] best model not found at '{path}', falling back to final.")
            path = f"models/{algo}_handover"
    else:
        path = f"models/{algo}_handover"
    return cls.load(path)


def compare_methods():
    """Compare baseline vs DQN vs PPO (final and best) with statistical reporting."""
    os.makedirs("figures", exist_ok=True)

    print("Evaluating Baseline (greedy SINR)...")
    baseline = evaluate_baseline()

    print("Evaluating DQN — final model (200k steps)...")
    dqn_final = evaluate_agent(_load_model("dqn", use_best=False))

    print("Evaluating DQN — best checkpoint...")
    dqn_best = evaluate_agent(_load_model("dqn", use_best=True))

    print("Evaluating PPO — final model (200k steps)...")
    ppo_final = evaluate_agent(_load_model("ppo", use_best=False))

    print("Evaluating PPO — best checkpoint...")
    ppo_best = evaluate_agent(_load_model("ppo", use_best=True))

    results = {
        "Baseline":  baseline,
        "DQN\n(final)": dqn_final,
        "DQN\n(best)":  dqn_best,
        "PPO\n(final)": ppo_final,
        "PPO\n(best)":  ppo_best,
    }

    # Print comparison table with mean ± std
    print("\n" + "=" * 75)
    print(f"PERFORMANCE COMPARISON — {NUM_EPISODES} Episodes Each, Fixed Seeds")
    print("=" * 75)
    labels = {
        "Baseline":       "Baseline    ",
        "DQN\n(final)":   "DQN (final) ",
        "DQN\n(best)":    "DQN (best)  ",
        "PPO\n(final)":   "PPO (final) ",
        "PPO\n(best)":    "PPO (best)  ",
    }
    for name, r in results.items():
        rm,  rs  = _mean_std(r["rewards"])
        hrm, hrs = _mean_std(r["handover_rates"])
        sm,  ss  = _mean_std(r["avg_sinrs"])
        ppm, pps = _mean_std(r["ping_pong_rates"])
        em,  es  = _mean_std(r["emergency_disc"])
        print(f"\n{labels[name]}:")
        print(f"  Avg Reward:        {rm:8.1f} ± {rs:.1f}")
        print(f"  Handover Rate:     {hrm:8.2f} ± {hrs:.2f}  (HOs/time-step)")
        print(f"  Avg SINR:          {sm:8.2f} ± {ss:.2f}  dB")
        print(f"  Ping-Pong Rate:    {ppm:8.3f} ± {pps:.3f}  (fraction of HOs)")
        print(f"  Emergency Disc.:   {em:8.1f} ± {es:.1f}")

    plot_comparison(results)
    plot_training_curves()


def plot_comparison(results: dict):
    """Plot 2×2 comparison bar chart with error bars."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Algorithm Comparison — Baseline vs DQN vs PPO",
                 fontsize=14, fontweight="bold", y=1.01)

    methods = list(results.keys())
    colors = ["#6B7280", "#3B82F6", "#93C5FD", "#10B981", "#6EE7B7"]

    metrics = [
        ("rewards",        "Average Reward",                  "Reward"),
        ("handover_rates", "Handover Rate (HOs/step)",        "HOs / time-step"),
        ("avg_sinrs",      "Average SINR (dB)",               "dB"),
        ("ping_pong_rates","Ping-Pong Rate\n(frac. of HOs)",  "Fraction"),
    ]

    for idx, (key, title, ylabel) in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]
        means = [np.mean(results[m][key]) for m in methods]
        stds  = [np.std(results[m][key])  for m in methods]
        bars = ax.bar(methods, means, yerr=stds, capsize=6,
                      color=colors, alpha=0.85, edgecolor="white", linewidth=1.5)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=10)
        ax.grid(axis="y", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        for bar, mean in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() * 1.01,
                    f"{mean:.2f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    out = "figures/comparison_bar_charts.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    print(f"\nComparison chart saved to '{out}'")
    plt.show()


def plot_training_curves():
    """Plot DQN and PPO training reward curves from Monitor logs."""
    os.makedirs("figures", exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Training Curves — Episode Reward Over Time",
                 fontsize=13, fontweight="bold")

    for ax, algo in zip(axes, ["dqn", "ppo"]):
        log_file = f"models/{algo}_monitor.monitor.csv"
        if not os.path.exists(log_file):
            ax.text(0.5, 0.5,
                    f"No training log found\n({log_file})\nRun train.py first.",
                    ha="center", va="center", transform=ax.transAxes,
                    color="gray", fontsize=11)
            ax.set_title(f"{algo.upper()} Training Curve", fontsize=12, fontweight="bold")
            continue

        import pandas as pd
        df = pd.read_csv(log_file, skiprows=1)
        rewards = df["r"].values
        rolling = pd.Series(rewards).rolling(5, min_periods=1).mean().values

        ax.plot(rewards, alpha=0.3, color="#3B82F6", label="Episode reward")
        ax.plot(rolling, color="#3B82F6", linewidth=2, label="5-ep rolling mean")
        ax.set_title(f"{algo.upper()} Training Curve", fontsize=12, fontweight="bold")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Episode Reward")
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()
    out = "figures/training_curves.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Training curves saved to '{out}'")
    plt.show()


if __name__ == "__main__":
    compare_methods()
