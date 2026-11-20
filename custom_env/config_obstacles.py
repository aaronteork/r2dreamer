"""Configuration for zero-shot finite-resource navigation through clutter."""

import math
from dataclasses import dataclass

from custom_env.config_env import EnvConfig


@dataclass(frozen=True, kw_only=True)
class ObstaclesConfig(EnvConfig):
    env_name: str = "ObstaclesEnv"
    is_training: bool = False
    max_steps: float = float("inf")
    num_food: int = 1
    num_water: int = 1
    num_heat: int = 0
    initial_hunger: float = -0.30
    initial_thirst: float = -0.30
    pillar_count: int = 0
    pillar_diameter: float = 1.4
    wall_count: int = 2
    wall_length: float = 3.0
    wall_thickness: float = 0.5
    obstacle_height: float = 2.0
    spawn_x: float = 0.0
    spawn_y: float = 0.0
    spawn_yaw: float = 0.0  # Radians: zero faces world +X.
    # Sampling/reachability margins only; no collision shells or padding.
    path_clearance: float = 0.8
    obstacle_spacing: float = 1.4
    boundary_obstacle_gap: float = 1.1
    spawn_obstacle_gap: float = 1.1
    resource_obstacle_gap: float = 1.0
    resource_boundary_gap: float = 1.1
    spawn_resource_gap: float = 1.8
    connectivity_grid_step: float = 0.15
    min_detour_ratio: float = 1.15
    layout_mode: str = "random"
    chicane_primary_need: float = -0.30
    chicane_secondary_need: float = -0.15
    chicane_doorway_min: float = 3.0
    chicane_doorway_max: float = 3.8
    chicane_baffle_jitter_y: float = 0.4
    layout_attempts: int = 1000
    placement_attempts: int = 200
    resource_placement_attempts: int = 10000

    def __post_init__(self):
        if self.layout_mode not in {"random", "chicane", "t_divider"}:
            raise ValueError("layout_mode must be 'random', 'chicane', or 't_divider'.")
        for name in ("num_food", "num_water", "pillar_count", "wall_count"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer.")
        if self.num_food + self.num_water == 0:
            raise ValueError("At least one resource is required.")
        for name in ("initial_hunger", "initial_thirst", "chicane_primary_need", "chicane_secondary_need"):
            if not -0.99999 < getattr(self, name) < 0.99999:
                raise ValueError(f"{name} must be strictly within the death boundaries.")
        if not (0.0 < self.chicane_doorway_min <= self.chicane_doorway_max < 2.0 * self.arena_size):
            raise ValueError("chicane doorway bounds must satisfy 0 < min <= max < arena width.")
        if not math.isfinite(self.chicane_baffle_jitter_y) or self.chicane_baffle_jitter_y < 0.0:
            raise ValueError("chicane_baffle_jitter_y must be finite and nonnegative.")
        for name in (
            "arena_size", "pillar_diameter", "wall_length", "wall_thickness",
            "obstacle_height", "path_clearance", "connectivity_grid_step",
            "hunger_decay", "thirst_decay", "replenish_rate", "object_spacing",
            "object_interaction_dist", "boundary_obstacle_gap",
            "resource_boundary_gap", "spawn_obstacle_gap", "resource_obstacle_gap",
            "spawn_resource_gap", "obstacle_spacing",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive.")
        if not math.isfinite(self.min_detour_ratio) or self.min_detour_ratio < 1.0:
            raise ValueError("min_detour_ratio must be finite and at least 1.0.")
        for name in ("layout_attempts", "placement_attempts", "resource_placement_attempts"):
            if not isinstance(getattr(self, name), int) or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if self.num_heat != 0 or self.obs_space_dim != 26 or self.imagine:
            raise ValueError("Obstacles requires the two-state training observation contract.")
        if not 0 < self.camera_fovy < 180:
            raise ValueError("camera_fovy must be between zero and 180 degrees.")
        if not all(math.isfinite(v) for v in (self.spawn_x, self.spawn_y, self.spawn_yaw)):
            raise ValueError("Spawn pose must be finite.")
        if max(abs(self.spawn_x), abs(self.spawn_y)) >= self.arena_size - 0.1 - self.path_clearance:
            raise ValueError("Spawn must leave path_clearance to the arena boundary.")

    @property
    def natural_step_bound(self) -> int:
        """Conservative death bound even if every available resource is consumed."""
        return math.ceil(min(
            (1 + self.initial_hunger + self.num_food * self.replenish_rate) / self.hunger_decay,
            (1 + self.initial_thirst + self.num_water * self.replenish_rate) / self.thirst_decay,
        )) + 1

    @classmethod
    def chicane(cls, **overrides) -> "ObstaclesConfig":
        """Build a 1-resource target-primed parameterized Staggered Chicane config."""
        defaults = dict(
            layout_mode="chicane",
            num_food=1,
            num_water=0,
            initial_hunger=-0.30,
            initial_thirst=-0.15,
            pillar_count=0,
            wall_count=4,
            wall_length=4.5,
            wall_thickness=0.5,
            min_detour_ratio=1.5,
        )
        defaults.update(overrides)
        return cls(**defaults)

    @classmethod
    def t_divider(cls, **overrides) -> "ObstaclesConfig":
        """Build a 2-resource T-divider layout config with 1.0 m-thick walls."""
        defaults = dict(
            layout_mode="t_divider",
            num_food=1,
            num_water=1,
            initial_hunger=-0.30,
            initial_thirst=-0.30,
            pillar_count=0,
            wall_count=3,
            wall_length=2.8,
            wall_thickness=1.0,
            min_detour_ratio=1.15,
        )
        defaults.update(overrides)
        return cls(**defaults)
