from dataclasses import dataclass

from custom_env.config_env import EnvConfig


@dataclass(frozen=True, kw_only=True)
class PartitionConfig(EnvConfig):
    """Configuration for the zero-shot partitioned landmark-recall task."""

    env_name: str = "PartitionRecallEnv"
    max_steps: int = 10_000
    num_food: int = 1
    num_water: int = 1
    num_heat: int = 0
    render_mode: str = "rgb_array"

    # A vertical partition attached to the north wall. MuJoCo box sizes are
    # half extents. The outer boundary is identical to the training arena.
    partition_x: float = 0.0
    partition_lower_end_y: float = -1.5
    partition_thickness: float = 0.2
    partition_height: float = 2.0
    path_clearance: float = 0.75

    # Fixed preview pose. The training camera faces body +X, so the torso is
    # yawed +90 degrees to look along world +Y.
    spawn_x: float = 0.0
    spawn_y: float = -4.0

    # Candidate sites vary the metric target position while retaining a wide,
    # symmetric arena. One site is sampled independently on each side.
    left_resource_sites: tuple[tuple[float, float], ...] = (
        (-3.0, 2.5),
        (-3.0, 3.5),
        (-3.8, 2.5),
        (-3.8, 3.5),
    )
    right_resource_sites: tuple[tuple[float, float], ...] = (
        (3.0, 2.5),
        (3.0, 3.5),
        (3.8, 2.5),
        (3.8, 3.5),
    )

    # Initial needs match the held-out selective-foraging evaluation and lie
    # within the range encountered by the training environment.
    primary_need_low: float = -0.15
    primary_need_high: float = -0.11
    secondary_need_low: float = -0.10
    secondary_need_high: float = -0.05
