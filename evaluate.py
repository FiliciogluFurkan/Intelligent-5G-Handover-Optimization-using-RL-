import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import DQN, PPO
from environment import HandoverEnv

def evaluate_agent(model, env, num_episodes=10):
    """Evaluate trained agent"""
    total_rewards = []
    handover_counts = []
    avg_sinrs = []
    ping_pong_counts = []
    emergency_disc_counts = []

    for episode in range(num_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        sinr_values = []
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            # Fix: capture acting user index BEFORE step() increments it
            acting_user_idx = env.current_user_idx
            obs, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward

            # Track SINR for the user that actually acted
            user = env.users[acting_user_idx]
            bs = env.base_stations[user.connected_bs] if user.connected_bs is not None else env.base_stations[0]
            sinr = bs.calculate_sinr(user.position)
            sinr_values.append(sinr)

            done = terminated or truncated

        total_rewards.append(episode_reward)
        handover_counts.append(env.total_handovers)
        avg_sinrs.append(np.mean(sinr_values))
        ping_pong_counts.append(env.ping_pong_count)
        emergency_disc_counts.append(env.emergency_disconnections)

    return {
        "avg_reward": np.mean(total_rewards),
        "avg_handovers": np.mean(handover_counts),
        "avg_sinr": np.mean(avg_sinrs),
        "ping_pongs": np.mean(ping_pong_counts),
        "emergency_disconnections": np.mean(emergency_disc_counts)
    }

def baseline_evaluation(env, num_episodes=10):
    """Evaluate baseline (signal-strength only) strategy"""
    total_rewards = []
    handover_counts = []
    avg_sinrs = []
    ping_pong_counts = []
    emergency_disc_counts = []

    for episode in range(num_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        sinr_values = []
        done = False

        while not done:
            # Fix: capture acting user index BEFORE step() increments it
            acting_user_idx = env.current_user_idx
            user = env.users[acting_user_idx]
            sinrs = [bs.calculate_sinr(user.position) for bs in env.base_stations]
            action = np.argmax(sinrs)

            obs, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            sinr_values.append(max(sinrs))

            done = terminated or truncated

        total_rewards.append(episode_reward)
        handover_counts.append(env.total_handovers)
        avg_sinrs.append(np.mean(sinr_values))
        ping_pong_counts.append(env.ping_pong_count)
        emergency_disc_counts.append(env.emergency_disconnections)

    return {
        "avg_reward": np.mean(total_rewards),
        "avg_handovers": np.mean(handover_counts),
        "avg_sinr": np.mean(avg_sinrs),
        "ping_pongs": np.mean(ping_pong_counts),
        "emergency_disconnections": np.mean(emergency_disc_counts)
    }

def compare_methods():
    """Compare baseline vs RL methods"""
    # Fix: create a fresh env per method to prevent shared state / energy accumulation
    print("Evaluating Baseline...")
    baseline_results = baseline_evaluation(HandoverEnv(num_users=15), num_episodes=5)

    print("\nEvaluating DQN Agent...")
    dqn_model = DQN.load("models/dqn_handover")
    dqn_results = evaluate_agent(dqn_model, HandoverEnv(num_users=15), num_episodes=5)

    print("\nEvaluating PPO Agent...")
    ppo_model = PPO.load("models/ppo_handover")
    ppo_results = evaluate_agent(ppo_model, HandoverEnv(num_users=15), num_episodes=5)

    # Print comparison
    print("\n" + "=" * 60)
    print("PERFORMANCE COMPARISON")
    print("=" * 60)

    methods = ["Baseline", "DQN", "PPO"]
    results = [baseline_results, dqn_results, ppo_results]

    for method, result in zip(methods, results):
        print(f"\n{method}:")
        print(f"  Avg Reward: {result['avg_reward']:.2f}")
        print(f"  Avg Handovers: {result['avg_handovers']:.1f}")
        print(f"  Avg SINR: {result['avg_sinr']:.2f} dB")
        print(f"  Ping-Pongs: {result['ping_pongs']:.1f}")
        print(f"  Emergency Disconnections: {result['emergency_disconnections']:.1f}")

    # Visualization
    plot_comparison(methods, results)

def plot_comparison(methods, results):
    """Plot comparison graphs"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    metrics = ['avg_handovers', 'avg_sinr', 'ping_pongs', 'emergency_disconnections']
    titles = ['Average Handovers', 'Average SINR (dB)', 'Ping-Pong Count', 'Emergency Disconnections']

    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[idx // 2, idx % 2]
        values = [r[metric] for r in results]
        ax.bar(methods, values, color=['gray', 'blue', 'green'])
        ax.set_title(title)
        ax.set_ylabel('Value')
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig('comparison_results.png', dpi=300)
    print("\nComparison graph saved to 'comparison_results.png'")
    plt.show()

if __name__ == "__main__":
    compare_methods()
