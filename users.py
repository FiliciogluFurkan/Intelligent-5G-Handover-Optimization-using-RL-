import numpy as np

class User:
    """Base class for mobile users in the 5G network"""

    def __init__(self, user_id, position, velocity, user_type, area_size: float = 500.0):
        self.user_id = user_id
        self.position = np.array(position, dtype=float)
        self.velocity = velocity  # km/h
        self.user_type = user_type
        self.area_size = area_size
        self.direction = np.random.uniform(0, 2 * np.pi)
        self.connected_bs = None
        self.handover_count = 0

    def move(self, dt=1.0):
        """Move user based on velocity and direction"""
        # Convert km/h to m/s
        speed_ms = self.velocity / 3.6

        # Update position
        dx = speed_ms * np.cos(self.direction) * dt
        dy = speed_ms * np.sin(self.direction) * dt
        self.position += np.array([dx, dy])

        # Boundary reflection
        if self.position[0] < 0 or self.position[0] > self.area_size:
            self.direction = np.pi - self.direction
            self.position[0] = np.clip(self.position[0], 0, self.area_size)
        if self.position[1] < 0 or self.position[1] > self.area_size:
            self.direction = -self.direction
            self.position[1] = np.clip(self.position[1], 0, self.area_size)

        # Random direction change (10% chance)
        if np.random.random() < 0.1:
            self.direction += np.random.uniform(-np.pi/4, np.pi/4)
            self.direction = self.direction % (2 * np.pi)

class Pedestrian(User):
    def __init__(self, user_id, position, area_size: float = 500.0):
        super().__init__(user_id, position, velocity=5, user_type="pedestrian", area_size=area_size)

class Vehicle(User):
    def __init__(self, user_id, position, area_size: float = 500.0):
        super().__init__(user_id, position, velocity=60, user_type="vehicle", area_size=area_size)

class EmergencyVehicle(User):
    def __init__(self, user_id, position, area_size: float = 500.0):
        super().__init__(user_id, position, velocity=120, user_type="emergency", area_size=area_size)
