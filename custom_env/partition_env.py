"""Zero-shot partitioned open-field landmark-recall environment."""

from __future__ import annotations

import math
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from gymnasium import spaces
from gymnasium.envs.mujoco.ant_v5 import AntEnv
from gymnasium.utils import EzPickle

from custom_env.ant_env import HomeostaticAntEnv
from custom_env.config_partition import PartitionConfig


class PartitionRecallEnv(HomeostaticAntEnv):
    """Two-resource evaluation with a single opaque central partition.

    The task preserves the training observation, action, homeostatic, reward,
    and resource-replenishment contracts. It changes only the held-out layout:
    one food and one water are previewed, do not respawn, and the wall occludes
    the unvisited resource after the agent enters the opposite side.
    """

    def __init__(self, cfg: PartitionConfig, **kwargs):
        self.cfg = cfg
        if cfg.num_heat != 0 or cfg.obs_space_dim != 26:
            raise ValueError(
                "Partition recall requires two internal states and "
                "26-value training proprioception."
            )
        if not 0.0 < cfg.camera_fovy < 180.0:
            raise ValueError("camera_fovy must be between 0 and 180 degrees.")
        if not cfg.left_resource_sites or not cfg.right_resource_sites:
            raise ValueError("At least one resource site is required on each side.")

        xml_file_path = Path(__file__).parent / cfg.xml_path
        tree = ET.parse(xml_file_path)
        worldbody = tree.find(".//worldbody")
        if worldbody is None:
            raise ValueError(f"{xml_file_path} does not contain a worldbody.")

        # Match the training environment's square boundary exactly. The base
        # class normally adds these dynamically, but this held-out environment
        # builds its own XML so that it can insert the central partition too.
        wall_attrs = {
            "type": "box",
            "rgba": "0.5 0.5 0.5 1",
            "conaffinity": "1",
            "condim": "3",
        }
        for name, pos, size in (
            (
                "wall_n",
                f"0 {cfg.arena_size} 0.5",
                f"{cfg.arena_size} 0.1 2.0",
            ),
            (
                "wall_s",
                f"0 {-cfg.arena_size} 0.5",
                f"{cfg.arena_size} 0.1 2.0",
            ),
            (
                "wall_e",
                f"{cfg.arena_size} 0 0.5",
                f"0.1 {cfg.arena_size} 2.0",
            ),
            (
                "wall_w",
                f"{-cfg.arena_size} 0 0.5",
                f"0.1 {cfg.arena_size} 2.0",
            ),
        ):
            ET.SubElement(
                worldbody,
                "geom",
                dict(wall_attrs, name=name, pos=pos, size=size),
            )
        partition_upper_end_y = cfg.arena_size - 0.1
        if cfg.partition_lower_end_y >= partition_upper_end_y:
            raise ValueError("partition_lower_end_y must be inside the arena.")
        partition_y = (cfg.partition_lower_end_y + partition_upper_end_y) / 2.0
        partition_half_length = (
            partition_upper_end_y - cfg.partition_lower_end_y
        ) / 2.0
        ET.SubElement(
            worldbody,
            "geom",
            {
                "name": "partition_wall",
                "type": "box",
                "pos": (
                    f"{cfg.partition_x} {partition_y} "
                    f"{cfg.partition_height / 2.0}"
                ),
                "size": (
                    f"{cfg.partition_thickness / 2.0} "
                    f"{partition_half_length} "
                    f"{cfg.partition_height / 2.0}"
                ),
                "rgba": "0.5 0.5 0.5 1",
                "conaffinity": "1",
                "condim": "3",
            },
        )

        with tempfile.NamedTemporaryFile(suffix=".xml", delete=False) as file:
            temp_xml_path = file.name
        tree.write(temp_xml_path)
        try:
            AntEnv.__init__(
                self,
                xml_file=temp_xml_path,
                width=cfg.image_size[0],
                height=cfg.image_size[1],
                render_mode=cfg.render_mode,
                **kwargs,
            )
        finally:
            Path(temp_xml_path).unlink(missing_ok=True)

        self.pov_camera_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_CAMERA, "pov"
        )
        if self.pov_camera_id == -1:
            raise ValueError("ant_env.xml must define a camera named 'pov'.")
        self.model.cam_fovy[self.pov_camera_id] = cfg.camera_fovy
        self.ant_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "torso"
        )
        self.partition_geom_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "partition_wall"
        )

        self.hunger = 0.0
        self.thirst = 0.0
        self.temperature = 0.0
        self.posture = 0.0
        self.prev_drive = 0.0
        self.current_step = 0
        self.food_consumed = 0
        self.water_consumed = 0
        self.heat_exposed_time = 0.0
        self.object: list[tuple[str, float, float]] = []

        self.resources_consumed: list[str] = []
        self.first_resource_step = -1
        self.second_resource_step = -1
        self.target_reacquisition_step = -1
        self.second_leg_distance = 0.0
        self.occluded_second_leg_distance = 0.0
        self.shortest_second_leg_distance = float("nan")
        self.initial_target_bearing_error_rad = float("nan")
        self.wall_contact_steps = 0
        self.second_leg_stall_steps = 0
        self._previous_ant_pos: np.ndarray | None = None
        self._initial_hunger = 0.0
        self._initial_thirst = 0.0

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(8,), dtype=np.float32
        )
        self.observation_space = spaces.Dict(
            {
                "proprioception": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(cfg.obs_space_dim,),
                    dtype=np.float32,
                ),
                "vision": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(4, cfg.image_size[0], cfg.image_size[1]),
                    dtype=np.float32,
                ),
                "internal_state": spaces.Box(
                    low=-1.0, high=1.0, shape=(2,), dtype=np.float32
                ),
            }
        )
        EzPickle.__init__(self, cfg, **kwargs)

    @property
    def _wall_bounds(self) -> tuple[float, float, float, float]:
        half_width = self.cfg.partition_thickness / 2.0
        return (
            self.cfg.partition_x - half_width,
            self.cfg.partition_x + half_width,
            self.cfg.partition_lower_end_y,
            self.cfg.arena_size - 0.1,
        )

    def _segment_intersects_partition(
        self, start: np.ndarray, end: np.ndarray, *, margin: float = 0.0
    ) -> bool:
        """Return whether a 2-D segment crosses the wall's expanded AABB."""
        xmin, xmax, ymin, ymax = self._wall_bounds
        xmin -= margin
        xmax += margin
        ymin -= margin
        ymax += margin
        direction = np.asarray(end, dtype=np.float64) - np.asarray(
            start, dtype=np.float64
        )
        t_min, t_max = 0.0, 1.0
        for origin, delta, lower, upper in zip(start, direction, (xmin, ymin), (xmax, ymax)):
            if abs(delta) < 1e-12:
                if origin < lower or origin > upper:
                    return False
                continue
            near = (lower - origin) / delta
            far = (upper - origin) / delta
            if near > far:
                near, far = far, near
            t_min = max(t_min, near)
            t_max = min(t_max, far)
            if t_min > t_max:
                return False
        return True

    def _has_line_of_sight(self, target: np.ndarray) -> bool:
        camera_xy = self.data.cam_xpos[self.pov_camera_id][:2]
        return not self._segment_intersects_partition(camera_xy, target)

    def _resource_visible(self, resource: tuple[str, float, float]) -> bool:
        target = np.asarray(resource[1:3], dtype=np.float64)
        return self._has_line_of_sight(target) and self._is_in_camera_fov(target)

    def _remaining_resource(self) -> tuple[str, float, float] | None:
        return self.object[0] if len(self.object) == 1 else None

    def _approximate_shortest_path(
        self, start: np.ndarray, target: np.ndarray
    ) -> float:
        """Shortest collision-free polyline around the partition's open end."""
        if not self._segment_intersects_partition(
            start, target, margin=self.cfg.path_clearance
        ):
            return float(np.linalg.norm(target - start))
        xmin, xmax, ymin, _ = self._wall_bounds
        clearance = self.cfg.path_clearance
        left_x = xmin - clearance
        right_x = xmax + clearance
        if start[0] < self.cfg.partition_x:
            start_x, target_x = left_x, right_x
        else:
            start_x, target_x = right_x, left_x
        # The upper end meets the north arena wall, so the only valid route is
        # around the lower end of the partition.
        end_y = ymin - clearance
        first = np.array([start_x, end_y])
        second = np.array([target_x, end_y])
        return float(
            np.linalg.norm(first - start)
            + np.linalg.norm(second - first)
            + np.linalg.norm(target - second)
        )

    def reset_model(self):
        self.current_step = 0
        self.food_consumed = 0
        self.water_consumed = 0
        self.resources_consumed = []
        self.first_resource_step = -1
        self.second_resource_step = -1
        self.target_reacquisition_step = -1
        self.second_leg_distance = 0.0
        self.occluded_second_leg_distance = 0.0
        self.shortest_second_leg_distance = float("nan")
        self.initial_target_bearing_error_rad = float("nan")
        self.wall_contact_steps = 0
        self.second_leg_stall_steps = 0

        primary = self.np_random.uniform(
            self.cfg.primary_need_low, self.cfg.primary_need_high
        )
        secondary = self.np_random.uniform(
            self.cfg.secondary_need_low, self.cfg.secondary_need_high
        )
        if self.np_random.random() < 0.5:
            self.hunger, self.thirst = primary, secondary
        else:
            self.hunger, self.thirst = secondary, primary
        self.temperature = 0.0
        self._initial_hunger = self.hunger
        self._initial_thirst = self.thirst

        qpos = self.init_qpos.copy()
        qpos[0] = self.cfg.spawn_x
        qpos[1] = self.cfg.spawn_y
        qpos[3:7] = [math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)]
        qpos[7:] += self.np_random.uniform(
            low=-0.01, high=0.01, size=self.model.nq - 7
        )
        qvel = self.init_qvel.copy()
        qvel += self.np_random.uniform(
            low=-0.005, high=0.005, size=self.model.nv
        )
        self.set_state(qpos, qvel)
        self.posture = self._get_posture()
        self._previous_ant_pos = self.data.xpos[self.ant_body_id][:2].copy()

        left = self.cfg.left_resource_sites[
            int(self.np_random.integers(len(self.cfg.left_resource_sites)))
        ]
        right = self.cfg.right_resource_sites[
            int(self.np_random.integers(len(self.cfg.right_resource_sites)))
        ]
        if self.np_random.random() < 0.5:
            self.object = [("food", *left), ("water", *right)]
        else:
            self.object = [("water", *left), ("food", *right)]

        if not all(self._resource_visible(resource) for resource in self.object):
            raise RuntimeError(
                "Partition configuration must make both resources visible at reset."
            )
        self.prev_drive = self._calculate_drive()
        return self._get_obs()

    def _wall_contacting(self) -> bool:
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            if self.partition_geom_id in (contact.geom1, contact.geom2):
                return True
        return False

    def _target_bearing_error(
        self, ant_pos: np.ndarray, target: np.ndarray
    ) -> float:
        rotation = self.data.xmat[self.ant_body_id].reshape(3, 3)
        forward = rotation[:2, 0]
        target_direction = target - ant_pos
        norm = np.linalg.norm(target_direction)
        if norm < 1e-9:
            return 0.0
        target_direction /= norm
        cross = forward[0] * target_direction[1] - forward[1] * target_direction[0]
        dot = np.clip(np.dot(forward, target_direction), -1.0, 1.0)
        return float(abs(np.arctan2(cross, dot)))

    def step(self, action):
        self.hunger -= self.cfg.hunger_decay
        self.thirst -= self.cfg.thirst_decay
        self.do_simulation(action, self.frame_skip)
        self.current_step += 1
        self.posture = self._get_posture()

        ant_pos = self.data.xpos[self.ant_body_id][:2].copy()
        displacement = (
            0.0
            if self._previous_ant_pos is None
            else float(np.linalg.norm(ant_pos - self._previous_ant_pos))
        )
        self._previous_ant_pos = ant_pos.copy()
        if self.first_resource_step >= 0 and self.second_resource_step < 0:
            self.second_leg_distance += displacement
            if self.target_reacquisition_step < 0:
                self.occluded_second_leg_distance += displacement
            if displacement < 0.01:
                self.second_leg_stall_steps += 1
        if self._wall_contacting():
            self.wall_contact_steps += 1

        for resource in list(self.object):
            kind, x, y = resource
            target = np.array([x, y], dtype=np.float64)
            if (
                np.linalg.norm(ant_pos - target) < self.cfg.object_interaction_dist
                and self._resource_visible(resource)
            ):
                if kind == "food":
                    self.hunger += self.cfg.replenish_rate
                    self.food_consumed += 1
                else:
                    self.thirst += self.cfg.replenish_rate
                    self.water_consumed += 1
                self.object.remove(resource)
                self.resources_consumed.append(kind)
                if self.first_resource_step < 0:
                    self.first_resource_step = self.current_step
                    remaining = self._remaining_resource()
                    if remaining is not None:
                        remaining_pos = np.asarray(remaining[1:3], dtype=np.float64)
                        self.shortest_second_leg_distance = (
                            self._approximate_shortest_path(ant_pos, remaining_pos)
                        )
                        self.initial_target_bearing_error_rad = (
                            self._target_bearing_error(ant_pos, remaining_pos)
                        )
                        if self._resource_visible(remaining):
                            self.target_reacquisition_step = self.current_step
                else:
                    self.second_resource_step = self.current_step

        remaining = self._remaining_resource()
        target_visible = False
        target_bearing_error = float("nan")
        remaining_x = float("nan")
        remaining_y = float("nan")
        if remaining is not None:
            remaining_pos = np.asarray(remaining[1:3], dtype=np.float64)
            remaining_x, remaining_y = map(float, remaining_pos)
            target_visible = self._resource_visible(remaining)
            target_bearing_error = self._target_bearing_error(
                ant_pos, remaining_pos
            )
            if (
                self.first_resource_step >= 0
                and target_visible
                and self.target_reacquisition_step < 0
            ):
                self.target_reacquisition_step = self.current_step

        up_vector_z = self.data.xmat[self.ant_body_id][8]
        z_pos = self.data.xpos[self.ant_body_id][2]
        action_magnitude = np.linalg.norm(action)
        limit_reached = abs(self.hunger) > 0.99999 or abs(self.thirst) > 0.99999
        is_flipped = up_vector_z < 0.0
        both_consumed = self.food_consumed >= 1 and self.water_consumed >= 1
        term_reason = 2 if is_flipped else 1 if limit_reached else 4 if both_consumed else 0

        current_drive = self._calculate_drive()
        self.hunger = np.clip(self.hunger, -1.0, 1.0)
        self.thirst = np.clip(self.thirst, -1.0, 1.0)
        homeo_reward = self.cfg.reward_scale * (self.prev_drive - current_drive)
        movement_penalty = -0.5 * self.cfg.movement_penalty_weight * action_magnitude**2
        posture_penalty = -self.cfg.posture_penalty_weight * self.posture**2
        reward = homeo_reward + movement_penalty + posture_penalty
        self.prev_drive = current_drive
        observation = self._get_obs()

        path_efficiency = float("nan")
        if self.second_leg_distance > 0.0 and np.isfinite(
            self.shortest_second_leg_distance
        ):
            path_efficiency = min(
                1.0,
                self.shortest_second_leg_distance / self.second_leg_distance,
            )
        second_leg_latency = (
            self.second_resource_step - self.first_resource_step
            if self.second_resource_step >= 0
            else self.current_step - self.first_resource_step
            if self.first_resource_step >= 0
            else -1
        )
        reacquisition_latency = (
            self.target_reacquisition_step - self.first_resource_step
            if self.target_reacquisition_step >= 0
            else -1
        )
        info = {
            "timestep": np.array(self.current_step),
            "hunger": np.array(self.hunger),
            "thirst": np.array(self.thirst),
            "food_consumed": np.array(self.food_consumed),
            "water_consumed": np.array(self.water_consumed),
            "up_vector_z": np.array(up_vector_z),
            "is_flipped": np.array(is_flipped),
            "z_pos": np.array(z_pos),
            "termination_reason": np.array(term_reason),
            "posture": np.array(self.posture),
            "action_magnitude": np.array(action_magnitude),
            "reward_homeostatic": np.array(homeo_reward),
            "reward_movement_penalty": np.array(movement_penalty),
            "reward_posture_penalty": np.array(posture_penalty),
            "resources_consumed": list(self.resources_consumed),
            "initial_hunger": np.array(self._initial_hunger),
            "initial_thirst": np.array(self._initial_thirst),
            "ant_x": np.array(ant_pos[0]),
            "ant_y": np.array(ant_pos[1]),
            "remaining_resource_x": np.array(remaining_x),
            "remaining_resource_y": np.array(remaining_y),
            "remaining_resource_visible": np.array(target_visible),
            "target_bearing_error_rad": np.array(target_bearing_error),
            "initial_target_bearing_error_rad": np.array(
                self.initial_target_bearing_error_rad
            ),
            "first_resource_step": np.array(self.first_resource_step),
            "second_resource_step": np.array(self.second_resource_step),
            "target_reacquisition_step": np.array(self.target_reacquisition_step),
            "second_leg_latency": np.array(second_leg_latency),
            "target_reacquisition_latency": np.array(reacquisition_latency),
            "second_leg_distance": np.array(self.second_leg_distance),
            "occluded_second_leg_distance": np.array(
                self.occluded_second_leg_distance
            ),
            "shortest_second_leg_distance": np.array(
                self.shortest_second_leg_distance
            ),
            "second_leg_path_efficiency": np.array(path_efficiency),
            "wall_contact_steps": np.array(self.wall_contact_steps),
            "second_leg_stall_steps": np.array(self.second_leg_stall_steps),
        }
        if not self.cfg.is_training:
            environment_rgb, _ = self.mux_render(camera_name="environment")
            info["environment"] = self._add_hud(environment_rgb)
            pov_rgb, pov_depth = self.mux_render(camera_name="pov")
            info["vision"] = pov_rgb
            info["vision_depth"] = pov_depth
        return observation, reward, self.terminated, self.truncated, info

    @property
    def terminated(self):
        return bool(abs(self.hunger) > 0.99999 or abs(self.thirst) > 0.99999)

    @property
    def truncated(self):
        return self.current_step >= self.cfg.max_steps
