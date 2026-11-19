"""Held-out finite-resource foraging among unpadded physical obstacles."""

import math
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import replace
from pathlib import Path

import mujoco
import numpy as np

from custom_env.ant_env import HomeostaticAntEnv, RESOURCE_MARKER_RADIUS
from custom_env.config_obstacles import ObstaclesConfig
from custom_env.ego_motion import compute_ego_motion
from custom_env.obstacles_layout import generate_layout


class ObstaclesEnv(HomeostaticAntEnv):
    def __init__(self, cfg: ObstaclesConfig, **kwargs):
        tree = ET.parse(Path(__file__).parent / cfg.xml_path)
        worldbody = tree.find("worldbody")
        if worldbody is None:
            raise ValueError("Ant XML has no worldbody.")
        # Mocap bodies allow per-reset obstacle poses without stale static BVHs.
        # They have no action, observation, collision padding, or visible shells.
        # Compile them inside the arena, then assign sampled poses before any
        # simulation. Distant placeholders inflate model.stat.extent permanently:
        # MuJoCo scales clipping, fog and shadows by that compiled extent.
        for i in range(cfg.pillar_count + cfg.wall_count):
            pillar = i < cfg.pillar_count
            body = ET.SubElement(worldbody, "body", name=f"obstacle_{i}",
                                 mocap="true", pos=f"0 0 {cfg.obstacle_height/2}")
            size = (f"{cfg.pillar_diameter/2} {cfg.obstacle_height/2}" if pillar else
                    f"{cfg.wall_length/2} {cfg.wall_thickness/2} {cfg.obstacle_height/2}")
            ET.SubElement(body, "geom", name=f"obstacle_geom_{i}",
                          type="cylinder" if pillar else "box", size=size,
                          rgba="0.5 0.5 0.5 1", contype="1", conaffinity="1", group="2")
        with tempfile.NamedTemporaryFile(suffix=".xml", delete=False) as file:
            xml_path = Path(file.name)
        tree.write(xml_path)
        self._homeostatic_death = False
        try:
            super().__init__(replace(cfg, xml_path=str(xml_path)), **kwargs)
        finally:
            xml_path.unlink(missing_ok=True)
        self.cfg = cfg
        self.model.cam_fovy[self.pov_camera_id] = cfg.camera_fovy
        self._obstacle_mocap_ids = [int(self.model.body_mocapid[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"obstacle_{i}")
        ]) for i in range(cfg.pillar_count + cfg.wall_count)]
        self._obstacle_geom_ids = {
            int(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"obstacle_geom_{i}"))
            for i in range(cfg.pillar_count + cfg.wall_count)
        }
        self.layout = None
        self.total_distance = 0.
        self.wall_contact_steps = 0
        self.stall_steps = 0
        self.occluded_steps = 0
        self.ever_flipped = False
        self._layout_seed = cfg.seed
        # Group 2 is visible in the standard renderer; rays exclude ant/floor.
        self._obstacle_ray_groups = np.array([0, 0, 1, 0, 0, 0], dtype=np.uint8)

    def _wall_contacting(self) -> bool:
        if not self._obstacle_geom_ids:
            return False
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            if contact.geom1 in self._obstacle_geom_ids or contact.geom2 in self._obstacle_geom_ids:
                return True
        return False

    def reset(self, *, seed=None, options=None):
        if options and self.cfg.layout_mode != "chicane":
            raise ValueError("Obstacles does not use partition resource-order conditions.")
        if seed is not None:
            self._layout_seed = int(seed)
        return super().reset(seed=seed, options=options)

    def reset_model(self):
        self.layout = generate_layout(self.cfg, self._layout_seed, options=self._reset_options)
        for mocap_id, o in zip(self._obstacle_mocap_ids, self.layout["obstacles"]):
            self.data.mocap_pos[mocap_id] = [o["x"], o["y"], self.cfg.obstacle_height/2]
            self.data.mocap_quat[mocap_id] = [math.cos(o["angle"]/2), 0, 0, math.sin(o["angle"]/2)]
        spawn_x, spawn_y, spawn_yaw = self.layout["spawn"]
        qpos, qvel = self.init_qpos.copy(), self.init_qvel.copy()
        qpos[:2] = [spawn_x, spawn_y]
        qpos[3:7] = [math.cos(spawn_yaw / 2), 0, 0, math.sin(spawn_yaw / 2)]
        self.set_state(qpos, qvel)
        self._reset_ego_motion_tracking()
        self.object = [tuple(r) for r in self.layout["resources"]]
        self.hunger = float(self.layout.get("initial_hunger", self.cfg.initial_hunger))
        self.thirst = float(self.layout.get("initial_thirst", self.cfg.initial_thirst))
        self._initial_hunger, self._initial_thirst = self.hunger, self.thirst
        self.temperature = 0.
        self.current_step = self.food_consumed = self.water_consumed = 0
        self.resources_consumed = []
        self.heat_exposed_time = self.total_distance = 0.
        self.wall_contact_steps = 0
        self.stall_steps = 0
        self.occluded_steps = 0
        self.ever_flipped = False
        self._homeostatic_death = False
        self.posture = self._get_posture()
        self.prev_drive = self._calculate_drive()
        return self._get_obs()

    def _resource_visible(self, resource):
        _, x, y = resource
        if not self._is_in_camera_fov(np.array([x, y])):
            return False
        origin = self.data.cam_xpos[self.pov_camera_id].copy()
        target = np.array([x, y, RESOURCE_MARKER_RADIUS])
        offset = target - origin
        length = float(np.linalg.norm(offset))
        if length < 1e-9:
            return True
        hit = np.array([-1], dtype=np.int32)
        distance = mujoco.mj_ray(self.model, self.data, origin, offset/length,
                                 self._obstacle_ray_groups, 1, -1, hit)
        return distance < 0 or distance >= length

    def step(self, action):
        self.hunger -= self.cfg.hunger_decay
        self.thirst -= self.cfg.thirst_decay
        previous_position, previous_rotation = self._get_torso_pose()
        self.do_simulation(action[:8], self.frame_skip)
        position, rotation = self._get_torso_pose()
        self._ego_motion = compute_ego_motion(previous_position, previous_rotation, position, rotation)
        self._previous_torso_position, self._previous_torso_rotation = position, rotation
        displacement = float(np.linalg.norm(position[:2] - previous_position[:2]))
        self.total_distance += displacement
        if displacement < 0.01:
            self.stall_steps += 1
        if self._wall_contacting():
            self.wall_contact_steps += 1
        self.current_step += 1
        self.posture = self._get_posture()
        for resource in list(self.object):
            if (np.linalg.norm(position[:2] - np.asarray(resource[1:])) < self.cfg.object_interaction_dist
                    and self._resource_visible(resource)):
                if resource[0] == "food":
                    self.hunger += self.cfg.replenish_rate
                    self.food_consumed += 1
                else:
                    self.thirst += self.cfg.replenish_rate
                    self.water_consumed += 1
                self.resources_consumed.append(resource[0])
                self.object.remove(resource)
        if self.object and not any(self._resource_visible(resource) for resource in self.object):
            self.occluded_steps += 1
        self._homeostatic_death = bool(abs(self.hunger) > .99999 or abs(self.thirst) > .99999)
        success = not self.object and not self._homeostatic_death
        is_flipped = bool(self.data.xmat[self.ant_body_id][8] < 0)
        self.ever_flipped = self.ever_flipped or is_flipped
        geo_dist = float(self.layout.get("geodesic_shortest_distance", 0.0))
        euc_dist = float(self.layout.get("euclidean_shortest_distance", 0.0))
        detour_ratio = float(self.layout.get("detour_ratio", 1.0))
        if success:
            geodesic_eff = min(1.0, geo_dist / self.total_distance) if self.total_distance > 0 else 1.0
        else:
            geodesic_eff = 0.0
        current_drive = self._calculate_drive()
        self.hunger, self.thirst = np.clip([self.hunger, self.thirst], -1., 1.)
        action_magnitude = float(np.linalg.norm(action[:8]))
        homeo_reward = self.cfg.reward_scale * (self.prev_drive - current_drive)
        movement_penalty = -.5 * self.cfg.movement_penalty_weight * action_magnitude**2
        posture_penalty = -self.cfg.posture_penalty_weight * self.posture**2
        self.prev_drive = current_drive
        reward = homeo_reward + movement_penalty + posture_penalty
        info = dict(
            timestep=self.current_step, hunger=self.hunger, thirst=self.thirst,
            initial_hunger=self._initial_hunger, initial_thirst=self._initial_thirst,
            food_consumed=self.food_consumed, water_consumed=self.water_consumed,
            resources_collected=self.food_consumed + self.water_consumed,
            num_food=int(self.layout.get("num_food", self.cfg.num_food)),
            num_water=int(self.layout.get("num_water", self.cfg.num_water)),
            collection_success=int(success), total_distance=self.total_distance,
            completion_steps=self.current_step if success else float("nan"),
            completion_distance=self.total_distance if success else float("nan"),
            geodesic_shortest_distance=geo_dist,
            euclidean_shortest_distance=euc_dist,
            detour_ratio=detour_ratio,
            geodesic_path_efficiency=float(geodesic_eff),
            wall_contact_steps=float(self.wall_contact_steps),
            stall_steps=float(self.stall_steps),
            occluded_steps=float(self.occluded_steps),
            occluded_step_fraction=float(self.occluded_steps / max(1, self.current_step)),
            homeostatic_death=int(self._homeostatic_death),
            termination_reason=1 if self._homeostatic_death else 4 if success else 0,
            posture=self.posture, z_pos=float(position[2]),
            is_flipped=is_flipped,
            flipped_at_end=int(is_flipped),
            ever_flipped=int(self.ever_flipped),
            action_magnitude=action_magnitude, reward_homeostatic=homeo_reward,
            reward_movement_penalty=movement_penalty, reward_posture_penalty=posture_penalty,
            ego_motion=self._ego_motion.copy(),
        )
        observation = self._get_obs()
        if not self.cfg.is_training:
            environment_rgb, _ = self.mux_render(camera_name="environment")
            info["environment"] = self._add_hud(environment_rgb)
            info["vision"], info["vision_depth"] = self.mux_render(camera_name="pov")
        return observation, reward, self.terminated, False, info

    @property
    def terminated(self):
        return bool(self._homeostatic_death or not self.object)
