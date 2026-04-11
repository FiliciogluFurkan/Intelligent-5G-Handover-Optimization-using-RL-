import numpy as np
from config.settings import BS as BSConfig


class BaseStation:
    """5G Base Station with signal and load management"""

    def __init__(self, bs_id, position, max_capacity=20):
        self.bs_id = bs_id
        self.position = np.array(position, dtype=float)
        self.max_capacity = max_capacity
        self.connected_users: set = set()
        self.total_energy = 0.0

    def calculate_sinr(self, user_position, interference_power=None) -> float:
        """Calculate Signal-to-Interference-plus-Noise Ratio (SINR) in dB."""
        if interference_power is None:
            interference_power = BSConfig.default_interference_w
        distance = np.linalg.norm(user_position - self.position)

        # Path loss model
        path_loss_db = (BSConfig.path_loss_intercept
                        + BSConfig.path_loss_exponent * np.log10(max(distance, 1) / 1000))

        # Received signal power
        rx_power_dbm = BSConfig.tx_power_dbm - path_loss_db
        rx_power_w = 10 ** ((rx_power_dbm - 30) / 10)

        # SINR calculation
        sinr = rx_power_w / (interference_power + BSConfig.noise_power_w)
        sinr_db = 10 * np.log10(sinr + 1e-10)

        return sinr_db

    def get_load(self) -> float:
        """Return current load ratio clamped to [0, 1]."""
        return min(len(self.connected_users) / self.max_capacity, 1.0)

    def add_user(self, user) -> None:
        """Connect a user to this base station."""
        if user not in self.connected_users:
            self.connected_users.add(user)
            user.connected_bs = self.bs_id

    def remove_user(self, user) -> None:
        """Disconnect a user from this base station."""
        self.connected_users.discard(user)
        if user.connected_bs == self.bs_id:
            user.connected_bs = None
