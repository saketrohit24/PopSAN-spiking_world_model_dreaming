"""
HumanoidCompact-v0: Humanoid-v4 with compact state representation.

Keeps only qpos (22 dims) + qvel (23 dims) = 45 dims from the full
376-dim Humanoid-v4 observation. These are the minimal Markovian state
variables; the remaining dims (cinert, cvel, qfrc_actuator, cfrc_ext)
are derived quantities computable from qpos/qvel + the MuJoCo model.

The action space (17 dims) and full 3D bipedal dynamics are unchanged.
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class HumanoidCompact(gym.ObservationWrapper):
    """Humanoid-v4 with compact (qpos + qvel) observation."""

    # In Humanoid-v4, obs layout is:
    #   qpos[2:]  (22 dims) – joint positions, root x/y excluded
    #   qvel      (23 dims) – joint velocities
    #   cinert    (130 dims)
    #   cvel      (78 dims)
    #   qfrc_actuator (23 dims)
    #   cfrc_ext  (84 dims)
    # We keep only the first 45 = 22 + 23.

    COMPACT_DIM = 45  # 22 (qpos excl root x,y) + 23 (qvel)

    def __init__(self, env):
        super().__init__(env)
        high = np.inf * np.ones(self.COMPACT_DIM, dtype=np.float64)
        self.observation_space = spaces.Box(-high, high, dtype=np.float64)

    def observation(self, obs):
        return obs[: self.COMPACT_DIM]


# ---- Custom env registry (keyed by string name) ----
CUSTOM_ENVS = {
    "HumanoidCompact-v0": lambda **kw: HumanoidCompact(gym.make("Humanoid-v4", **kw)),
}


def make_env(env_name: str, **kwargs) -> gym.Env:
    """Create an environment by name, supporting both standard Gymnasium
    envs and custom wrappers defined in this module."""
    if env_name in CUSTOM_ENVS:
        return CUSTOM_ENVS[env_name](**kwargs)
    return gym.make(env_name, **kwargs)
