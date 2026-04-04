from stable_baselines3 import DQN, PPO
from stable_baselines3.common.env_checker import check_env
from environment import HandoverEnv
import numpy as np

def train_agent(algorithm="DQN", total_timesteps=100000):
    """Train RL agent for handover optimization"""
    import os
    os.makedirs("models", exist_ok=True)

    # Create environment
    env = HandoverEnv(num_users=15)
    
    # Check environment
    print("Checking environment...")
    check_env(env)
    print("Environment check passed!")
    
    # Create agent
    if algorithm == "DQN":
        model = DQN("MlpPolicy", env, verbose=1, 
                    learning_rate=1e-4,
                    buffer_size=50000,
                    learning_starts=1000,
                    batch_size=64,
                    gamma=0.99,
                    exploration_fraction=0.3,
                    exploration_final_eps=0.05)
    else:  # PPO
        model = PPO("MlpPolicy", env, verbose=1,
                    learning_rate=3e-4,
                    n_steps=2048,
                    batch_size=64,
                    gamma=0.99,
                    gae_lambda=0.95)
    
    print(f"\nTraining {algorithm} agent...")
    model.learn(total_timesteps=total_timesteps)
    
    # Save model
    model.save(f"models/{algorithm.lower()}_handover")
    print(f"\nModel saved to models/{algorithm.lower()}_handover.zip")
    
    return model

if __name__ == "__main__":
    import os
    os.makedirs("models", exist_ok=True)
    
    # Train both algorithms
    print("=" * 50)
    print("Training DQN Agent")
    print("=" * 50)
    train_agent("DQN", total_timesteps=50000)
    
    print("\n" + "=" * 50)
    print("Training PPO Agent")
    print("=" * 50)
    train_agent("PPO", total_timesteps=50000)
