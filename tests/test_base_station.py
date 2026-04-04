"""Unit tests for BaseStation class."""
import numpy as np
import pytest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from base_station import BaseStation
from users import Pedestrian


class TestBaseStationInit:
    def test_init_stores_id_and_position(self):
        bs = BaseStation(0, [100, 200])
        assert bs.bs_id == 0
        assert np.allclose(bs.position, [100, 200])

    def test_init_default_capacity(self):
        bs = BaseStation(1, [0, 0])
        assert bs.max_capacity == 20

    def test_init_custom_capacity(self):
        bs = BaseStation(0, [0, 0], max_capacity=5)
        assert bs.max_capacity == 5

    def test_init_empty_users_and_zero_energy(self):
        bs = BaseStation(0, [0, 0])
        assert bs.connected_users == []
        assert bs.total_energy == 0.0


class TestBaseStationSINR:
    def test_sinr_returns_float(self):
        bs = BaseStation(0, [250, 250])
        sinr = bs.calculate_sinr(np.array([250, 250]))
        assert isinstance(sinr, float)

    def test_sinr_decreases_with_distance(self):
        bs = BaseStation(0, [0, 0])
        close_sinr = bs.calculate_sinr(np.array([10, 0]))
        far_sinr = bs.calculate_sinr(np.array([400, 0]))
        assert close_sinr > far_sinr

    def test_sinr_finite_for_zero_distance(self):
        """Should not return inf or nan at zero distance."""
        bs = BaseStation(0, [250, 250])
        sinr = bs.calculate_sinr(np.array([250.0, 250.0]))
        assert np.isfinite(sinr)

    def test_sinr_finite_for_large_distance(self):
        bs = BaseStation(0, [0, 0])
        sinr = bs.calculate_sinr(np.array([1000, 1000]))
        assert np.isfinite(sinr)


class TestBaseStationUserManagement:
    def setup_method(self):
        self.bs = BaseStation(0, [250, 250])
        self.user = Pedestrian(0, np.array([100.0, 100.0]))

    def test_add_user_connects_user(self):
        self.bs.add_user(self.user)
        assert self.user in self.bs.connected_users
        assert self.user.connected_bs == 0

    def test_add_user_no_duplicate(self):
        self.bs.add_user(self.user)
        self.bs.add_user(self.user)
        assert self.bs.connected_users.count(self.user) == 1

    def test_remove_user_disconnects(self):
        self.bs.add_user(self.user)
        self.bs.remove_user(self.user)
        assert self.user not in self.bs.connected_users

    def test_get_load_zero_when_empty(self):
        assert self.bs.get_load() == 0.0

    def test_get_load_proportional(self):
        bs = BaseStation(0, [0, 0], max_capacity=10)
        users = [Pedestrian(i, np.array([float(i), 0.0])) for i in range(5)]
        for u in users:
            bs.add_user(u)
        assert bs.get_load() == pytest.approx(0.5)
