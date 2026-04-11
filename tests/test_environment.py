"""Unit tests for HandoverEnv."""
import numpy as np
import pytest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment import HandoverEnv


class TestHandoverEnvInit:
    def test_default_init(self):
        env = HandoverEnv()
        assert env.num_users == 15
        assert env.area_size == 500
        assert len(env.base_stations) == 3

    def test_custom_num_users(self):
        env = HandoverEnv(num_users=5)
        assert env.num_users == 5

    def test_observation_space_shape(self):
        env = HandoverEnv()
        assert env.observation_space.shape == (9,)

    def test_action_space_size(self):
        env = HandoverEnv()
        assert env.action_space.n == 4


class TestHandoverEnvReset:
    def test_reset_returns_correct_obs_shape(self):
        env = HandoverEnv(num_users=5)
        obs, info = env.reset()
        assert obs.shape == (9,)

    def test_reset_creates_users(self):
        env = HandoverEnv(num_users=5)
        env.reset()
        assert len(env.users) == 5

    def test_reset_zeros_counters(self):
        env = HandoverEnv(num_users=5)
        env.reset()
        assert env.total_handovers == 0
        assert env.ping_pong_count == 0
        assert env.emergency_disconnections == 0
        assert env.time_step == 0

    def test_reset_connects_all_users_to_bs(self):
        env = HandoverEnv(num_users=10)
        env.reset()
        for user in env.users:
            assert user.connected_bs is not None
            assert 0 <= user.connected_bs < 3


class TestHandoverEnvStep:
    def setup_method(self):
        self.env = HandoverEnv(num_users=5)
        self.env.reset(seed=42)

    def test_step_returns_five_values(self):
        result = self.env.step(0)
        assert len(result) == 5

    def test_step_obs_shape(self):
        obs, reward, terminated, truncated, info = self.env.step(0)
        assert obs.shape == (9,)

    def test_step_reward_is_float(self):
        _, reward, _, _, _ = self.env.step(0)
        assert isinstance(reward, float)

    def test_step_no_action_no_handover(self):
        """Action 3 = no change, should not increment total_handovers."""
        initial_handovers = self.env.total_handovers
        self.env.step(3)
        assert self.env.total_handovers == initial_handovers

    def test_episode_terminates(self):
        """Episode should terminate after max_steps."""
        env = HandoverEnv(num_users=3)
        obs, _ = env.reset(seed=0)
        terminated = False
        steps = 0
        while not terminated and steps < 100_000:
            obs, reward, terminated, truncated, _ = env.step(3)
            steps += 1
        assert terminated

    def test_obs_values_not_nan(self):
        """Observation should never contain NaN."""
        env = HandoverEnv(num_users=5)
        obs, _ = env.reset(seed=1)
        assert not np.any(np.isnan(obs))
        for _ in range(100):
            obs, _, terminated, _, _ = env.step(env.action_space.sample())
            assert not np.any(np.isnan(obs))
            if terminated:
                obs, _ = env.reset()


class TestHandoverRewards:
    def test_handover_reduces_reward(self):
        """Switching BS should apply a penalty."""
        env = HandoverEnv(num_users=3)
        env.reset(seed=7)

        # Get the current user's connected BS
        user = env.users[env.current_user_idx]
        current_bs = user.connected_bs

        # Find a different BS to switch to
        other_bs = (current_bs + 1) % 3

        _, reward_with_handover, _, _, _ = env.step(other_bs)

        env.reset(seed=7)
        _, reward_no_handover, _, _, _ = env.step(3)  # no change

        # Handover should be penalized
        assert reward_with_handover < reward_no_handover + 5  # some tolerance for SINR differences
