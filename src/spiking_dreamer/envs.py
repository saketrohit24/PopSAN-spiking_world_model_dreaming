#!/usr/bin/env python3
"""
Environment helpers and custom wrappers.
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class HumanoidCompact(gym.ObservationWrapper):
    """Humanoid-v4 with compact (qpos + qvel) observation."""

    # Humanoid-v4 default obs layout:
    #   qpos[2:]       -> 22 dims
    #   qvel           -> 23 dims
    #   cinert         -> 130 dims
    #   cvel           -> 78 dims
    #   qfrc_actuator  -> 23 dims
    #   cfrc_ext       -> 84 dims
    # Keep only qpos[2:] + qvel.
    COMPACT_DIM = 45

    def __init__(self, env: gym.Env):
        super().__init__(env)
        high = np.inf * np.ones(self.COMPACT_DIM, dtype=np.float64)
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float64)

    def observation(self, obs):
        if obs.shape[0] < self.COMPACT_DIM:
            raise ValueError(
                f"Expected observation with at least {self.COMPACT_DIM} dims, got {obs.shape[0]}"
            )
        return obs[: self.COMPACT_DIM]


def _make_humanoid_compact(**kwargs) -> gym.Env:
    humanoid_kwargs = dict(kwargs)
    # Humanoid-v4 defaults to excluding root x/y from obs; force explicit behavior.
    humanoid_kwargs.setdefault("exclude_current_positions_from_observation", True)
    return HumanoidCompact(gym.make("Humanoid-v4", **humanoid_kwargs))


CUSTOM_ENVS = {
    "HumanoidCompactLite-v0": _make_humanoid_compact,
    # Backward-compatible alias
    "HumanoidCompact-v0": _make_humanoid_compact,
}


def make_env(env_name: str, **kwargs) -> gym.Env:
    """Create standard or custom environments by name."""
    if env_name in CUSTOM_ENVS:
        return CUSTOM_ENVS[env_name](**kwargs)
    return gym.make(env_name, **kwargs)
