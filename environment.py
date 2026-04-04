import gymnasium as gym
import numpy as np
from gymnasium import spaces
from base_station import BaseStation
from users import Pedestrian, Vehicle, EmergencyVehicle

class HandoverEnv(gym.Env):
    """5G Handover Optimization Environment"""

    def __init__(self, num_users=15, area_size=500):
        super().__init__()

        self.area_size = area_size
        self.num_users = num_users
        self.max_velocity = 120.0

        # Initialize base stations at fixed positions (600x600 alana göre)
        self.base_stations = [
            BaseStation(0, [150, 300]),
            BaseStation(1, [300, 200]),
            BaseStation(2, [450, 430])
        ]

        self.users = []
        self.time_step = 0
        self.user_step = 0
        self.max_steps = 1000

        # Action space: select BS for each user (0, 1, 2, or 3=no change)
        self.action_space = spaces.Discrete(4)

        # Observation space: SINR(3), load(3), velocity, handover_count
        # Per-feature bounds: SINR in dB [-120, 50], loads [0, 1], normalized [0, 1]
        self.observation_space = spaces.Box(
            low=np.array([-120, -120, -120, 0, 0, 0, 0, 0], dtype=np.float32),
            high=np.array([50, 50, 50, 1, 1, 1, 1, 1], dtype=np.float32),
            dtype=np.float32
        )

        self.total_handovers = 0
        self.ping_pong_count = 0
        self.emergency_disconnections = 0

    def reset(self, seed=None, options=None):
        """Reset environment to initial state"""
        super().reset(seed=seed)

        # Create diverse user mix
        self.users = []
        for i in range(self.num_users):
            pos = np.random.uniform(0, self.area_size, 2)
            if i < self.num_users // 2:
                self.users.append(Pedestrian(i, pos))
            elif i < self.num_users * 0.8:
                self.users.append(Vehicle(i, pos))
            else:
                self.users.append(EmergencyVehicle(i, pos))

        # Reset base stations
        for bs in self.base_stations:
            bs.connected_users = []
            bs.total_energy = 0.0

        # Initial connection (closest BS)
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

    def _get_observation(self, user_idx=None):
        """Get current state observation for one user"""
        if user_idx is None:
            user_idx = self.current_user_idx
        user = self.users[user_idx]

        # SINR from all base stations
        sinrs = [bs.calculate_sinr(user.position) for bs in self.base_stations]

        # Load from all base stations
        loads = [bs.get_load() for bs in self.base_stations]

        # User velocity (normalized)
        velocity_norm = user.velocity / self.max_velocity

        # Handover count (normalized)
        handover_norm = min(user.handover_count / 10.0, 1.0)

        obs = np.array(sinrs + loads + [velocity_norm, handover_norm], dtype=np.float32)
        return obs

    def step(self, action):
        """Execute one step in the environment"""
        user = self.users[self.current_user_idx]
        reward = 0.0

        # Action: 0-2 = select BS, 3 = no change
        if action < 3:
            target_bs = self.base_stations[action]

            # Check if handover is needed
            if user.connected_bs != action:
                # Remove from old BS (guard against None)
                if user.connected_bs is not None:
                    old_bs = self.base_stations[user.connected_bs]
                    old_bs.remove_user(user)

                # Add to new BS
                target_bs.add_user(user)
                user.handover_count += 1
                self.total_handovers += 1

                # Ping-pong detection using per-user step counter
                if hasattr(user, 'last_handover_user_step'):
                    if self.user_step - user.last_handover_user_step < 10:
                        self.ping_pong_count += 1
                        reward -= 5  # Penalty for ping-pong

                user.last_handover_user_step = self.user_step
                reward -= 1  # Small penalty for handover

        # Calculate reward based on SINR
        current_bs = self.base_stations[user.connected_bs]
        sinr = current_bs.calculate_sinr(user.position)
        reward += sinr / 10.0  # Reward for good signal

        # Penalty for emergency vehicle with low SINR
        if user.user_type == "emergency" and sinr < -5:
            reward -= 20
            self.emergency_disconnections += 1

        # Energy consumption (proportional to load)
        energy = current_bs.get_load() * 0.1
        current_bs.total_energy += energy
        reward -= energy  # Penalty for high energy

        # Move only the acting user
        user.move(dt=1.0)

        # Capture observation for the CURRENT user before advancing index
        obs = self._get_observation()

        # Move to next user
        self.current_user_idx = (self.current_user_idx + 1) % self.num_users

        # Increment time step when all users processed
        if self.current_user_idx == 0:
            self.time_step += 1

        # Increment per-step counter every step
        self.user_step += 1

        terminated = self.time_step >= self.max_steps
        truncated = False

        return obs, reward, terminated, truncated, {}

    def render(self):
        """Render the environment (optional)"""
        pass
