import os
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from environment import HandoverEnv
from config.settings import TRAIN


def train_agent(algorithm="DQN", total_timesteps=200_000):
    """Train an RL agent for handover optimization with checkpointing."""
    os.makedirs("models", exist_ok=True)
    os.makedirs("models/checkpoints", exist_ok=True)

    algo = algorithm.upper()

    # Wrap environments with Monitor to log episode rewards
    env = Monitor(HandoverEnv(), filename=f"models/{algo.lower()}_monitor")
    eval_env = Monitor(HandoverEnv())

    # Validate environment (runs once)
    print("Checking environment...")
    check_env(HandoverEnv())
    print("Environment check passed!")

    # Checkpoint every 20k steps
    checkpoint_cb = CheckpointCallback(
        save_freq=20_000,
        save_path="models/checkpoints/",
        name_prefix=f"{algo.lower()}_ckpt",
        verbose=0,
    )

    # Evaluate on a separate env every 10k steps and save best model
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=f"models/best_{algo.lower()}/",
        log_path=f"models/{algo.lower()}_eval_logs/",
        eval_freq=10_000,
        n_eval_episodes=10,
        deterministic=True,
        verbose=1,
    )

    # Create agent using config constants
    if algo == "DQN":
        model = DQN(
            "MlpPolicy", env, verbose=1,
            learning_rate=TRAIN.dqn_learning_rate,
            buffer_size=TRAIN.dqn_buffer_size,
            learning_starts=TRAIN.dqn_learning_starts,
            batch_size=TRAIN.dqn_batch_size,
            gamma=TRAIN.dqn_gamma,
            exploration_fraction=TRAIN.dqn_exploration_fraction,
            exploration_final_eps=TRAIN.dqn_final_eps,
        )
    elif algo == "PPO":
        model = PPO(
            "MlpPolicy", env, verbose=1,
            learning_rate=TRAIN.ppo_learning_rate,
            n_steps=TRAIN.ppo_n_steps,
            n_epochs=TRAIN.ppo_n_epochs,
            batch_size=TRAIN.ppo_batch_size,
            gamma=TRAIN.ppo_gamma,
            gae_lambda=TRAIN.ppo_gae_lambda,
        )
    else:
        raise ValueError(f"Unknown algorithm '{algorithm}'. Choose: DQN or PPO")

    print(f"\nTraining {algo} for {total_timesteps:,} timesteps...")
    model.learn(total_timesteps=total_timesteps, callback=[checkpoint_cb, eval_cb])

    # Save final model
    model.save(f"models/{algo.lower()}_handover")
    print(f"\nModel saved to models/{algo.lower()}_handover.zip")

    env.close()
    eval_env.close()
    return model


if __name__ == "__main__":
    print("=" * 60)
    print("Training DQN Agent (400k timesteps)")
    print("=" * 60)
    train_agent("DQN", total_timesteps=400_000)

    print("\n" + "=" * 60)
    print("Training PPO Agent (400k timesteps)")
    print("=" * 60)
    train_agent("PPO", total_timesteps=400_000)
