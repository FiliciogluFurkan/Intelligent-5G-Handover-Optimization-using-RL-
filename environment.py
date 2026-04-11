import gymnasium as gym
import numpy as np
from gymnasium import spaces
from base_station import BaseStation
from users import Pedestrian, Vehicle, EmergencyVehicle
from config.settings import SIM, BS_POSITIONS, REWARD


class HandoverEnv(gym.Env):
    """5G Handover Optimization Environment"""

    def __init__(self, num_users=None, area_size=None, max_steps=None):
        super().__init__()

        self.area_size = area_size if area_size is not None else SIM.area_size
        self.num_users = num_users if num_users is not None else SIM.default_num_users
        self.max_velocity = 120.0

        # Initialize base stations at fixed positions (500x500m area)
        self.base_stations = [
            BaseStation(i, BS_POSITIONS[i], max_capacity=SIM.max_capacity_per_bs)
            for i in range(SIM.num_base_stations)
        ]

        self.users = []
        self.time_step = 0
        self.user_step = 0
        self.max_steps = max_steps if max_steps is not None else SIM.max_steps

        # Action space: select BS for each user (0, 1, 2, or 3=no change)
        self.action_space = spaces.Discrete(4)

        # Observation space: SINR(3), load(3), velocity, handover_count, user_type
        self.observation_space = spaces.Box(
            low=np.array([-120, -120, -120, 0, 0, 0, 0, 0, 0], dtype=np.float32),
            high=np.array([50, 50, 50, 1, 1, 1, 1, 1, 1], dtype=np.float32),
            dtype=np.float32
        )

        self.total_handovers = 0
        self.ping_pong_count = 0
        self.emergency_disconnections = 0

    def reset(self, seed=None, options=None):
        """Reset environment to initial state."""
        super().reset(seed=seed)

        # Create diverse user mix
        self.users = []
        for i in range(self.num_users):
            pos = np.random.uniform(0, self.area_size, 2)
            if i < self.num_users // 2:
                self.users.append(Pedestrian(i, pos, area_size=self.area_size))
            elif i < self.num_users * 0.8:
                self.users.append(Vehicle(i, pos, area_size=self.area_size))
            else:
                self.users.append(EmergencyVehicle(i, pos, area_size=self.area_size))

        # Reset base stations
        for bs in self.base_stations:
            bs.connected_users = set()
            bs.total_energy = 0.0

        # Initial connection: each user connects to the closest BS
        for user in self.users:
            closest_bs = min(self.base_stations,
                             key=lambda bs: np.linalg.norm(user.position - bs.position))
            closest_bs.add_user(user)

        self.time_step = 0
        self.user_step = 0
        self.total_handovers = 0
        self.ping_pong_count = 0
        self.emergency_disconnections = 0
        self.current_user_idx = 0

        return self._get_observation(), {}

    def _get_observation(self, user_idx=None) -> np.ndarray:
        """Get current state observation for one user."""
        if user_idx is None:
            user_idx = self.current_user_idx
        user = self.users[user_idx]

        sinrs = list(np.clip(
            [bs.calculate_sinr(user.position) for bs in self.base_stations], -120, 50
        ))
        loads = [bs.get_load() for bs in self.base_stations]
        velocity_norm = user.velocity / self.max_velocity
        handover_norm = min(user.handover_count / 10.0, 1.0)
        type_norm     = {"pedestrian": 0.0, "vehicle": 0.5, "emergency": 1.0}[user.user_type]

        obs = np.array(sinrs + loads + [velocity_norm, handover_norm, type_norm], dtype=np.float32)
        return obs

    def step(self, action):
        """Execute one step in the environment."""
        user = self.users[self.current_user_idx]
        reward = 0.0

        # Action: 0-2 = select BS, 3 = no change
        if action < 3:
            target_bs = self.base_stations[action]

            if user.connected_bs != action:
                # Remove from old BS
                if user.connected_bs is not None:
                    self.base_stations[user.connected_bs].remove_user(user)

                # Add to new BS
                target_bs.add_user(user)
                user.handover_count += 1
                self.total_handovers += 1

                # Ping-pong detection — emergency vehicles are exempt
                if user.user_type != "emergency":
                    if hasattr(user, 'last_handover_time_step'):
                        if self.time_step - user.last_handover_time_step < REWARD.ping_pong_window_steps:
                            self.ping_pong_count += 1
                            reward -= REWARD.ping_pong_penalty

                user.last_handover_time_step = self.time_step

                # Emergency handovers are cheap — encourage signal tracking
                if user.user_type == "emergency":
                    reward -= REWARD.handover_penalty * REWARD.emergency_handover_factor
                else:
                    reward -= REWARD.handover_penalty

                # Handover latency: SINR reward lost during the switching instant
                # Normal users lose full SINR (LTE ~50ms), emergency lose 20% (5G NR ~5ms)
                pre_sinr = self.base_stations[action].calculate_sinr(user.position)
                latency_factor = REWARD.emergency_latency_factor if user.user_type == "emergency" \
                                 else REWARD.handover_latency_factor
                reward -= (pre_sinr / REWARD.sinr_scale) * latency_factor

        # Guard: if user has no connection (edge case), connect to best SINR BS
        if user.connected_bs is None:
            best = int(np.argmax([bs.calculate_sinr(user.position) for bs in self.base_stations]))
            self.base_stations[best].add_user(user)

        # Calculate reward based on SINR — emergency vehicles weighted higher
        current_bs  = self.base_stations[user.connected_bs]
        sinr        = current_bs.calculate_sinr(user.position)
        sinr_weight = REWARD.emergency_sinr_weight if user.user_type == "emergency" else 1.0
        reward += sinr / REWARD.sinr_scale * sinr_weight

        # Emergency vehicle with low SINR — count only on zone entry (state transition)
        was_weak = getattr(user, '_in_weak_zone', False)
        in_weak = user.user_type == "emergency" and sinr < REWARD.emergency_sinr_threshold_db
        if in_weak:
            reward -= REWARD.emergency_disconnect_penalty
            if not was_weak:
                self.emergency_disconnections += 1
        user._in_weak_zone = in_weak

        # Gap penalty disabled: accumulates across all steps including non-acting steps,
        # causing reward explosion in multi-user cycling environment.

        # Energy consumption penalty (proportional to BS load)
        energy = current_bs.get_load() * REWARD.energy_scale
        current_bs.total_energy += energy
        reward -= energy

        # Move only the acting user
        user.move(dt=1.0)

        # Capture observation before advancing user index
        obs = self._get_observation()

        # Advance to next user
        self.current_user_idx = (self.current_user_idx + 1) % self.num_users

        # Increment time step when all users have been processed
        if self.current_user_idx == 0:
            self.time_step += 1

        self.user_step += 1

        terminated = self.time_step >= self.max_steps
        truncated = False

        return obs, reward, terminated, truncated, {}

    def render(self):
        """Render the environment (optional)."""
        pass
