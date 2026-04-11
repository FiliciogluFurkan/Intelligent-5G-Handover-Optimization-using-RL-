"""Central configuration for 5G Handover Optimization project."""
import math
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
    direction_change_max_rad: float = math.pi / 4


@dataclass(frozen=True)
class TrainingConfig:
    dqn_learning_rate: float = 1e-4
    dqn_buffer_size: int = 100_000
    dqn_learning_starts: int = 1000
    dqn_batch_size: int = 64
    dqn_gamma: float = 0.99
    dqn_exploration_fraction: float = 0.4
    dqn_final_eps: float = 0.05

    ppo_learning_rate: float = 3e-4
    ppo_n_steps: int = 2048
    ppo_n_epochs: int = 10
    ppo_batch_size: int = 64
    ppo_gamma: float = 0.99
    ppo_gae_lambda: float = 0.95


@dataclass(frozen=True)
class RewardConfig:
    sinr_scale: float = 10.0
    handover_penalty: float = 0.3              # base handover cost (reduced: was 0.5)
    ping_pong_penalty: float = 2.0             # reduced from 5.0 — was too punishing
    emergency_sinr_threshold_db: float = 0.0   # disconnect penalty trigger (~330m from BS)
    emergency_disconnect_penalty: float = 30.0
    emergency_sinr_weight: float = 2.5         # emergency SINR reward 2.5x stronger
    emergency_handover_factor: float = 0.1     # emergency handover base cost ×0.1
    handover_latency_factor: float = 0.05      # reduced from 0.5 — was adding ~1.0 extra cost per HO
    emergency_latency_factor: float = 0.01     # reduced from 0.05
    # Gap penalty removed: fires every step including steps where agent isn't acting on that user
    # (each user acts once per 15 steps), causing reward explosion. SINR signal is sufficient.
    emergency_sinr_gap_threshold_db: float = 3.0   # kept for reference but not used
    emergency_gap_penalty_scale: float = 0.0        # disabled
    energy_scale: float = 0.1
    ping_pong_window_steps: int = 10


# Singleton instances
SIM = SimulationConfig()
BS = BaseStationConfig()
USERS = UserConfig()
TRAIN = TrainingConfig()
REWARD = RewardConfig()

# Fixed base station positions in 500x500m area
BS_POSITIONS = [
    [150.0, 300.0],
    [300.0, 200.0],
    [450.0, 430.0],
]
