"""Central configuration for 5G Handover Optimization project."""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class SimulationConfig:
    area_size: float = 500.0
    num_base_stations: int = 3
    max_capacity_per_bs: int = 20
    max_steps: int = 1000
    default_num_users: int = 15


@dataclass(frozen=True)
class BaseStationConfig:
    tx_power_dbm: float = 43.0
    path_loss_intercept: float = 128.1
    path_loss_exponent: float = 37.6
    noise_power_w: float = 1e-13
    default_interference_w: float = 1e-10


@dataclass(frozen=True)
class UserConfig:
    pedestrian_speed_kmh: float = 5.0
    vehicle_speed_kmh: float = 60.0
    emergency_speed_kmh: float = 120.0
    max_speed_kmh: float = 120.0
    direction_change_prob: float = 0.1
    direction_change_max_rad: float = 3.14159 / 4


@dataclass(frozen=True)
class TrainingConfig:
    dqn_learning_rate: float = 1e-4
    dqn_buffer_size: int = 50000
    dqn_learning_starts: int = 1000
    dqn_batch_size: int = 64
    dqn_gamma: float = 0.99
    dqn_exploration_fraction: float = 0.3
    dqn_final_eps: float = 0.05

    ppo_learning_rate: float = 3e-4
    ppo_n_steps: int = 2048
    ppo_batch_size: int = 64
    ppo_gamma: float = 0.99
    ppo_gae_lambda: float = 0.95


@dataclass(frozen=True)
class RewardConfig:
    sinr_scale: float = 10.0
    handover_penalty: float = 1.0
    ping_pong_penalty: float = 5.0
    emergency_sinr_threshold_db: float = -5.0
    emergency_disconnect_penalty: float = 20.0
    energy_scale: float = 0.1
    ping_pong_window_steps: int = 10


# Singleton instances
SIM = SimulationConfig()
BS = BaseStationConfig()
USERS = UserConfig()
TRAIN = TrainingConfig()
REWARD = RewardConfig()

# Fixed base station positions
BS_POSITIONS = [
    [125.0, 250.0],
    [250.0, 125.0],
    [375.0, 375.0],
]
