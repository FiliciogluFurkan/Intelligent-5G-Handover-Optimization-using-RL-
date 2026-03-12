import numpy as np

class BaseStation:
    """5G Base Station with signal and load management"""
    
    def __init__(self, bs_id, position, max_capacity=20):
        self.bs_id = bs_id
        self.position = np.array(position, dtype=float)
        self.max_capacity = max_capacity
        self.connected_users = []
        self.total_energy = 0.0
        
    def calculate_sinr(self, user_position, interference_power=1e-10):
        """Calculate Signal-to-Interference-plus-Noise Ratio (SINR)"""
        distance = np.linalg.norm(user_position - self.position)
        
        # Path loss model (simplified)
        path_loss_db = 128.1 + 37.6 * np.log10(max(distance, 1) / 1000)
        
        # Transmit power (dBm)
        tx_power_dbm = 43
        
        # Received signal power
        rx_power_dbm = tx_power_dbm - path_loss_db
        rx_power_w = 10 ** ((rx_power_dbm - 30) / 10)
        
        # SINR calculation
        noise_power = 1e-13
        sinr = rx_power_w / (interference_power + noise_power)
        sinr_db = 10 * np.log10(sinr + 1e-10)
        
        return sinr_db
    
    def get_load(self):
        """Return current load ratio (0-1)"""
        return len(self.connected_users) / self.max_capacity
    
    def add_user(self, user):
        """Connect a user to this base station"""
        if user not in self.connected_users:
            self.connected_users.append(user)
            user.connected_bs = self.bs_id
            
    def remove_user(self, user):
        """Disconnect a user from this base station"""
        if user in self.connected_users:
            self.connected_users.remove(user)
