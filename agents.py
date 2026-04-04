"""Shared handover policy agents — single source of truth for algorithm dispatch."""
from __future__ import annotations
import os
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from environment import HandoverEnv

_model_cache: dict = {}


def _load_model(algorithm: str):
    """Load and cache a trained RL model."""
    if algorithm in _model_cache:
        return _model_cache[algorithm]

    path = f"models/{algorithm}_handover.zip"
    if not os.path.exists(path):
        return None

    if algorithm == "dqn":
        from stable_baselines3 import DQN
        _model_cache[algorithm] = DQN.load(f"models/{algorithm}_handover")
    elif algorithm == "ppo":
        from stable_baselines3 import PPO
        _model_cache[algorithm] = PPO.load(f"models/{algorithm}_handover")

    return _model_cache.get(algorithm)


def get_action(algorithm: str, env: "HandoverEnv") -> int:
    """
    Get the next action for the current user in the environment.

    Args:
        algorithm: One of 'baseline', 'dqn', 'ppo'
        env: Active HandoverEnv instance

    Returns:
        Action integer (0-3)

    Raises:
        ValueError: If algorithm is not recognized
        FileNotFoundError: If RL model file does not exist
    """
    algorithm = algorithm.lower()

    if algorithm == "baseline":
        user = env.users[env.current_user_idx]
        sinrs = [bs.calculate_sinr(user.position) for bs in env.base_stations]
        return int(np.argmax(sinrs))

    elif algorithm in ("dqn", "ppo"):
        model = _load_model(algorithm)
        if model is None:
            raise FileNotFoundError(
                f"Model not found: models/{algorithm}_handover.zip — run train.py first"
            )
        obs = env._get_observation()
        action, _ = model.predict(obs, deterministic=True)
        return int(action)

    else:
        raise ValueError(f"Unknown algorithm '{algorithm}'. Choose: baseline, dqn, ppo")


def clear_model_cache() -> None:
    """Clear the loaded model cache (useful for testing)."""
    _model_cache.clear()
