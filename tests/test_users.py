"""Unit tests for User classes."""
import numpy as np
import pytest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from users import User, Pedestrian, Vehicle, EmergencyVehicle


class TestPedestrian:
    def test_velocity_is_5(self):
        p = Pedestrian(0, np.array([100.0, 100.0]))
        assert p.velocity == 5

    def test_user_type(self):
        p = Pedestrian(0, np.array([0.0, 0.0]))
        assert p.user_type == "pedestrian"

    def test_initial_connected_bs_is_none(self):
        p = Pedestrian(0, np.array([0.0, 0.0]))
        assert p.connected_bs is None


class TestVehicle:
    def test_velocity_is_60(self):
        v = Vehicle(0, np.array([100.0, 100.0]))
        assert v.velocity == 60

    def test_user_type(self):
        v = Vehicle(0, np.array([0.0, 0.0]))
        assert v.user_type == "vehicle"


class TestEmergencyVehicle:
    def test_velocity_is_120(self):
        e = EmergencyVehicle(0, np.array([100.0, 100.0]))
        assert e.velocity == 120

    def test_user_type(self):
        e = EmergencyVehicle(0, np.array([0.0, 0.0]))
        assert e.user_type == "emergency"


class TestUserMovement:
    def test_move_changes_position(self):
        user = Vehicle(0, np.array([250.0, 250.0]))
        original_pos = user.position.copy()
        user.move(dt=1.0)
        assert not np.allclose(user.position, original_pos)

    def test_move_stays_in_bounds(self):
        """After many moves, user stays within [0, 500] x [0, 500]."""
        user = EmergencyVehicle(0, np.array([0.0, 0.0]))
        for _ in range(1000):
            user.move(dt=1.0)
        assert 0 <= user.position[0] <= 500
        assert 0 <= user.position[1] <= 500

    def test_boundary_reflection_x(self):
        """User near right boundary should reflect back."""
        user = Vehicle(0, np.array([490.0, 250.0]))
        user.direction = 0.0  # Moving right (positive x)
        for _ in range(200):
            user.move(dt=1.0)
        assert 0 <= user.position[0] <= 500

    def test_handover_count_initializes_zero(self):
        user = Pedestrian(0, np.array([100.0, 100.0]))
        assert user.handover_count == 0
