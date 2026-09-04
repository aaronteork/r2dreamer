import gymnasium as gym
import numpy as np
from gymnasium import spaces

from custom_env.ant_env import HomeostaticAntEnv


class HomeostaticAntR2Env(gym.Env):
    """Converts the Gymnasium Ant environment to R2Dreamer's API."""

    def __init__(self, cfg, seed=0, provide_terminal_signals=False):
        self._env = HomeostaticAntEnv(cfg)
        self._seed = seed
        self._needs_seed = True
        # A physiological-limit reset is still an episode boundary, but in a
        # continuing homeostatic task it should not necessarily imply zero
        # continuation value. Negative rewards would otherwise make death
        # appear attractive to the value function.
        self._provide_terminal_signals = provide_terminal_signals

        height, width = cfg.image_size
        obs_spaces = {
            "image": spaces.Box(0, 255, shape=(height, width, 4), dtype=np.uint8),
            "proprioception": self._env.observation_space["proprioception"],
            "internal_state": self._env.observation_space["internal_state"],
            "is_first": spaces.Box(0, 1, shape=(), dtype=np.bool_),
            "is_last": spaces.Box(0, 1, shape=(), dtype=np.bool_),
            "is_terminal": spaces.Box(0, 1, shape=(), dtype=np.bool_),
        }
        if "heat_sensor" in self._env.observation_space:
            obs_spaces["heat_sensor"] = self._env.observation_space["heat_sensor"]

        self.observation_space = spaces.Dict(obs_spaces)
        self.action_space = self._env.action_space

    def _convert(self, obs, is_first, is_last, is_terminal):
        # Your vision: C,H,W float32 in [0, 1].
        # R2Dreamer image: H,W,C uint8 in [0, 255].
        image = np.moveaxis(obs["vision"], 0, -1)
        image = np.clip(image * 255.0, 0, 255).astype(np.uint8)

        result = {
            "image": image,
            "proprioception": obs["proprioception"].astype(np.float32),
            "internal_state": obs["internal_state"].astype(np.float32),
            "is_first": np.array(is_first, dtype=np.bool_),
            "is_last": np.array(is_last, dtype=np.bool_),
            "is_terminal": np.array(is_terminal, dtype=np.bool_),
        }
        if "heat_sensor" in obs:
            result["heat_sensor"] = obs["heat_sensor"].astype(np.float32)
        return result

    def reset(self, *, seed=None, options=None):
        """Reset the adapter, optionally with an explicit evaluation seed.

        The return value intentionally remains the observation-only format used
        by R2Dreamer's training loop.
        """
        del options
        if seed is not None:
            obs, _ = self._env.reset(seed=seed)
            self._needs_seed = False
        elif self._needs_seed:
            obs, _ = self._env.reset(seed=self._seed)
            self._needs_seed = False
        else:
            obs, _ = self._env.reset()

        return self._convert(
            obs, is_first=True, is_last=False, is_terminal=False
        )

    def step(self, action):
        action = np.clip(
            action,
            self.action_space.low,
            self.action_space.high,
        ).astype(np.float32, copy=False)

        obs, reward, terminated, truncated, info = self._env.step(action)

        # A physiological limit still triggers an environment reset. By
        # default, however, do not expose it as an RL terminal: is_last keeps
        # the RSSM episode boundary while is_terminal controls Dreamer's
        # continuation target.
        done = bool(terminated)
        converted = self._convert(
            obs,
            is_first=False,
            is_last=done,
            is_terminal=done and self._provide_terminal_signals,
        )
        info["discount"] = np.array(
            0.0 if done and self._provide_terminal_signals else 1.0,
            dtype=np.float32,
        )

        return converted, np.float32(reward), done, info
