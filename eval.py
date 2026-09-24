"""Evaluate a trained R2Dreamer agent on the thesis evaluation tasks.

This standalone evaluator restores the Hydra configuration saved with a
training run, loads a single R2Dreamer
checkpoint, carries the RSSM state between environment steps, records POV and
environment videos, and writes step- and episode-level CSV statistics.

Examples
--------
Evaluate the normal foraging task from a completed run::

    python eval.py --task forage --run-dir logdir/homeostatic-ant

Evaluate the internal-state shift task without recording video::

    python eval.py --task shift \
        --run-dir logdir/homeostatic-ant \
        --no-video

Evaluate selective and sequential resource collection in the Y-maze::

    python eval.py --task ymaze --run-dir logdir/homeostatic-ant

Evaluate zero-shot landmark recall in the partitioned open field::

    python eval.py --task partition --run-dir logdir/homeostatic-ant
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
from dataclasses import fields
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import torch
from tensordict import TensorDict

from custom_env.ant_env import HomeostaticAntEnv
from custom_env.config_env import EnvConfig
from custom_env.config_partition import PartitionConfig
from custom_env.config_ymaze import YMazeConfig
from custom_env.partition_env import PartitionRecallEnv
from custom_env.ymaze_env import YMazeTestEnv
from dreamer import Dreamer
from envs.homeostatic_ant import HomeostaticAntR2Env
from tools import set_seed_everywhere


DATETIME = dt.datetime.now(ZoneInfo("Asia/Singapore")).strftime(
    "%Y-%m-%d_%H-%M-%S"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate an R2Dreamer agent.")
    parser.add_argument(
        "--task",
        choices=("forage", "shift", "ymaze", "partition"),
        required=True,
        help="Evaluation task.",
    )
    checkpoint = parser.add_mutually_exclusive_group(required=True)
    checkpoint.add_argument(
        "--run-dir",
        type=Path,
        help="Training-run directory containing latest.pt and .hydra/config.yaml.",
    )
    checkpoint.add_argument(
        "--model-path",
        "--model_path",
        dest="model_path",
        type=Path,
        help="Specific R2Dreamer checkpoint (legacy interface).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Hydra config.yaml; inferred from .hydra/config.yaml when omitted.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="Number of episodes (default: 1 for forage, 10 otherwise).",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Maximum steps per episode (default: EnvConfig.eval_max_steps).",
    )
    parser.add_argument("--seed", type=int, default=0, help="First episode seed.")
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device (default: CUDA when available, otherwise CPU).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Directory for videos and CSV files. By default, writes to "
            "<run-dir>/evaluation/<timestamp>/."
        ),
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Sample policy actions instead of using the distribution mode.",
    )
    parser.add_argument(
        "--no-video", action="store_true", help="Do not record evaluation videos."
    )
    parser.add_argument("--video-fps", type=float, default=30.0)
    parser.add_argument(
        "--video-size",
        type=int,
        default=512,
        help="Width and height of each square output video.",
    )
    args = parser.parse_args()

    if args.run_dir is not None:
        if not args.run_dir.is_dir():
            parser.error(f"Run directory not found: {args.run_dir}")
        args.model_path = args.run_dir / "latest.pt"
    else:
        args.run_dir = args.model_path.parent
    if not args.model_path.is_file():
        parser.error(f"Checkpoint not found: {args.model_path}")
    if args.config is not None and not args.config.is_file():
        parser.error(f"Configuration file not found: {args.config}")
    if args.episodes is not None and args.episodes < 1:
        parser.error("--episodes must be at least 1.")
    if args.max_steps is not None and args.max_steps < 1:
        parser.error("--max-steps must be at least 1.")
    if args.video_fps <= 0:
        parser.error("--video-fps must be positive.")
    if args.video_size < 1:
        parser.error("--video-size must be at least 1.")
    return args


def infer_config_path(model_path: Path) -> Path:
    """Find the training run's Hydra configuration above a checkpoint."""
    for directory in (model_path.parent, *model_path.parent.parents):
        candidate = directory / ".hydra" / "config.yaml"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "Could not find .hydra/config.yaml above the checkpoint; pass --config."
    )


def select_device(value: str | None) -> torch.device:
    if value is not None:
        return torch.device(value)
    if torch.accelerator.is_available():
        return torch.device(torch.accelerator.current_accelerator())
    return torch.device("cpu")


def load_config(path: Path, device: torch.device) -> Any:
    try:
        from omegaconf import OmegaConf
    except ImportError as error:
        raise RuntimeError(
            "Loading the saved run configuration requires hydra-core."
        ) from error

    config = OmegaConf.load(path)
    if "model" not in config or "env" not in config:
        raise ValueError(
            f"{path} is not an R2Dreamer Hydra config (model/env are required)."
        )
    OmegaConf.update(config, "device", str(device), force_add=True)
    OmegaConf.update(config, "model.device", str(device), force_add=True)
    # Evaluation never calls update(), so compilation only adds setup cost.
    OmegaConf.update(config, "model.compile", False, force_add=True)
    OmegaConf.resolve(config)
    return config


def make_env_config(
    config: Any,
    *,
    task: str,
    seed: int,
    render_size: tuple[int, int] | None = None,
) -> EnvConfig:
    """Translate the run configuration into the underlying Ant dataclass."""
    valid_fields = {field.name for field in fields(EnvConfig)}
    configured = {
        key: value
        for key, value in dict(config.env).items()
        if key in valid_fields
    }
    configured.update(
        seed=seed,
        device=torch.device(config.model.device),
        is_training=False,
        image_size=render_size or tuple(map(int, config.env.size)),
        shift=task == "shift",
    )
    return EnvConfig(**configured)


def make_ymaze_env(
    config: Any,
    *,
    seed: int,
    render_size: tuple[int, int] | None = None,
) -> Any:
    """Build the local held-out Y-maze with the training observation contract."""
    policy_size = tuple(map(int, config.env.size))
    valid_fields = {field.name for field in fields(YMazeConfig)}
    configured = {
        key: value for key, value in dict(config.env).items()
        if key in valid_fields
    }
    configured.update(
        seed=seed,
        device=torch.device(config.model.device),
        image_size=render_size or policy_size,
        is_training=False,
        num_heat=0,
        obs_space_dim=26,
        shift=False,
    )
    ymaze_config = YMazeConfig(**configured)
    return R2DreamerEnvAdapter(
        YMazeTestEnv(ymaze_config), seed=seed, policy_size=policy_size
    )


def make_partition_env(
    config: Any,
    *,
    seed: int,
    render_size: tuple[int, int] | None = None,
) -> Any:
    """Build the held-out partition task with the training observation contract."""
    policy_size = tuple(map(int, config.env.size))
    valid_fields = {field.name for field in fields(PartitionConfig)}
    configured = {
        key: value
        for key, value in dict(config.env).items()
        if key in valid_fields
    }
    configured.update(
        seed=seed,
        device=torch.device(config.model.device),
        image_size=render_size or policy_size,
        is_training=False,
        num_food=1,
        num_water=1,
        num_heat=0,
        obs_space_dim=26,
        max_steps=10_000,
        shift=False,
    )
    partition_config = PartitionConfig(**configured)
    return R2DreamerEnvAdapter(
        PartitionRecallEnv(partition_config),
        seed=seed,
        policy_size=policy_size,
    )


class R2DreamerEnvAdapter:
    """Convert a legacy Gymnasium visual environment to R2Dreamer's API."""

    def __init__(
        self, env: Any, *, seed: int, policy_size: tuple[int, int]
    ):
        from gymnasium import spaces

        self._env = env
        self._seed = seed
        height, width = policy_size
        self._policy_size = (width, height)
        observation_spaces = {
            "image": spaces.Box(
                0, 255, shape=(height, width, 4), dtype=np.uint8
            ),
            "proprioception": env.observation_space["proprioception"],
            "internal_state": env.observation_space["internal_state"],
            "is_first": spaces.Box(0, 1, shape=(), dtype=np.bool_),
            "is_last": spaces.Box(0, 1, shape=(), dtype=np.bool_),
            "is_terminal": spaces.Box(0, 1, shape=(), dtype=np.bool_),
        }
        self.observation_space = spaces.Dict(observation_spaces)
        self.action_space = env.action_space

    def _convert(
        self, observation: dict[str, np.ndarray], *, first: bool, last: bool
    ) -> dict[str, np.ndarray]:
        image = np.moveaxis(observation["vision"], 0, -1)
        if (image.shape[1], image.shape[0]) != self._policy_size:
            import cv2

            image = cv2.resize(
                image, self._policy_size, interpolation=cv2.INTER_AREA
            )
        image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
        return {
            "image": image,
            "proprioception": observation["proprioception"].astype(np.float32),
            "internal_state": observation["internal_state"].astype(np.float32),
            "is_first": np.array(first, dtype=np.bool_),
            "is_last": np.array(last, dtype=np.bool_),
            "is_terminal": np.array(last, dtype=np.bool_),
        }

    def reset(self, *, seed: int | None = None) -> dict[str, np.ndarray]:
        result = self._env.reset(seed=self._seed if seed is None else seed)
        observation = result[0] if isinstance(result, tuple) else result
        return self._convert(observation, first=True, last=False)

    def step(self, action: np.ndarray) -> tuple[Any, np.float32, bool, dict[str, Any]]:
        action = np.clip(
            action, self.action_space.low, self.action_space.high
        ).astype(np.float32, copy=False)
        observation, reward, terminated, truncated, info = self._env.step(action)
        done = bool(terminated or truncated)
        info["evaluation_terminated"] = bool(terminated)
        info["evaluation_truncated"] = bool(truncated)
        info.setdefault("posture", np.asarray(self._env.posture))
        return (
            self._convert(observation, first=False, last=done),
            np.float32(reward),
            done,
            info,
        )

    def close(self) -> None:
        self._env.close()


def restore_actor_distribution(config: Any, action_space: Any) -> None:
    """Undo Dreamer's constructor mutation in a resolved saved config."""
    actor_dist = config.model.actor.dist
    if "name" not in actor_dist:
        return
    if hasattr(action_space, "multi_discrete"):
        key = "multi_disc"
    elif hasattr(action_space, "discrete"):
        key = "disc"
    else:
        key = "cont"
    config.model.actor.dist = {key: actor_dist}


def load_agent(
    checkpoint_path: Path,
    config: Any,
    env: Any,
    device: torch.device,
) -> Dreamer:
    restore_actor_distribution(config, env.action_space)
    agent = Dreamer(config.model, env.observation_space, env.action_space).to(device)
    try:
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=False
        )
    except TypeError:  # PyTorch versions without weights_only.
        checkpoint = torch.load(checkpoint_path, map_location=device)
    if not isinstance(checkpoint, dict) or "agent_state_dict" not in checkpoint:
        raise ValueError(
            f"{checkpoint_path} is not an R2Dreamer checkpoint: "
            "missing agent_state_dict."
        )
    agent.load_state_dict(checkpoint["agent_state_dict"], strict=True)
    return agent.eval()


def observation_batch(
    observation: dict[str, np.ndarray], device: torch.device
) -> TensorDict:
    tensors = {
        key: torch.as_tensor(value).unsqueeze(0)
        for key, value in observation.items()
    }
    return TensorDict(tensors, batch_size=(1,)).to(device)


class VideoRecorder:
    """Write the two diagnostic views returned in the environment info."""

    def __init__(self, output_dir: Path, task: str, fps: float, size: int):
        try:
            import cv2
        except ImportError as error:
            raise RuntimeError(
                "Video recording requires opencv-python; use --no-video to skip it."
            ) from error

        self.cv2 = cv2
        self.size = (size, size)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        stem = f"r2dreamer_{DATETIME}_{task}"
        self.pov_path = output_dir / f"{stem}_pov_video.mp4"
        self.env_path = output_dir / f"{stem}_env_video.mp4"
        self.pov = cv2.VideoWriter(str(self.pov_path), fourcc, fps, self.size)
        self.environment = cv2.VideoWriter(
            str(self.env_path), fourcc, fps, self.size
        )
        if not self.pov.isOpened() or not self.environment.isOpened():
            self.close()
            raise RuntimeError("OpenCV could not initialize the MP4 video writers.")

    def write(self, info: dict[str, Any]) -> None:
        for key, writer in (
            ("vision", self.pov),
            ("environment", self.environment),
        ):
            if key not in info:
                raise KeyError(f"Evaluation environment did not provide info[{key!r}].")
            frame = np.asarray(info[key])
            if (frame.shape[1], frame.shape[0]) != self.size:
                frame = self.cv2.resize(
                    frame, self.size, interpolation=self.cv2.INTER_AREA
                )
            writer.write(self.cv2.cvtColor(frame, self.cv2.COLOR_RGB2BGR))

    def close(self) -> None:
        self.pov.release()
        self.environment.release()


def scalar(info: dict[str, Any], key: str, default: float = 0.0) -> float:
    return float(np.asarray(info.get(key, default)).item())


def episode_summary(
    episode: int,
    seed: int,
    steps: int,
    total_reward: float,
    outcome: str,
    info: dict[str, Any],
) -> dict[str, Any]:
    row = {
        "episode": episode,
        "seed": seed,
        "steps": steps,
        "total_reward": total_reward,
        "outcome": outcome,
        "termination_reason": int(scalar(info, "termination_reason")),
        "food_consumed": int(scalar(info, "food_consumed")),
        "water_consumed": int(scalar(info, "water_consumed")),
        "final_hunger": scalar(info, "hunger"),
        "final_thirst": scalar(info, "thirst"),
        "final_posture": scalar(info, "posture"),
        "final_height": scalar(info, "z_pos"),
    }
    if "temperature" in info:
        row["final_temperature"] = scalar(info, "temperature")
    if "initial_hunger" in info:
        row["initial_hunger"] = scalar(info, "initial_hunger")
    if "initial_thirst" in info:
        row["initial_thirst"] = scalar(info, "initial_thirst")
    if "resources_consumed" in info:
        resources = [str(resource) for resource in info["resources_consumed"]]
        row["resource_sequence"] = ">".join(resources)
        expected_first = (
            "food"
            if scalar(info, "initial_hunger") < scalar(info, "initial_thirst")
            else "water"
        )
        row["expected_first_resource"] = expected_first
        row["correct_first_resource"] = int(
            bool(resources) and resources[0] == expected_first
        )
    for key in (
        "first_resource_step",
        "second_resource_step",
        "target_reacquisition_step",
        "second_leg_latency",
        "target_reacquisition_latency",
        "second_leg_distance",
        "occluded_second_leg_distance",
        "shortest_second_leg_distance",
        "second_leg_path_efficiency",
        "initial_target_bearing_error_rad",
        "wall_contact_steps",
        "second_leg_stall_steps",
    ):
        if key in info:
            row[key] = scalar(info, key)
    return row


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def evaluation_summary(
    task: str, max_steps: int, rows: list[dict[str, Any]]
) -> dict[str, Any]:
    """Create a compact task-level summary alongside per-episode results."""
    result = {
        "task": task,
        "episodes": len(rows),
        "max_steps": max_steps,
        "mean_steps": float(np.mean([row["steps"] for row in rows])),
        "mean_total_reward": float(
            np.mean([row["total_reward"] for row in rows])
        ),
        "mean_food_consumed": float(
            np.mean([row["food_consumed"] for row in rows])
        ),
        "mean_water_consumed": float(
            np.mean([row["water_consumed"] for row in rows])
        ),
        "resource_collection_rate": float(
            np.mean([row["outcome"] == "resources_collected" for row in rows])
        ),
        "survival_to_cutoff_rate": float(
            np.mean([row["outcome"] == "evaluation_cutoff" for row in rows])
        ),
    }
    if any("correct_first_resource" in row for row in rows):
        choices = [
            row["correct_first_resource"]
            for row in rows
            if "correct_first_resource" in row
        ]
        result["correct_first_resource_rate"] = float(np.mean(choices))
    successful_recall = [
        row
        for row in rows
        if row["outcome"] == "resources_collected"
        and np.isfinite(row.get("second_leg_path_efficiency", np.nan))
    ]
    if successful_recall:
        result["mean_successful_second_leg_path_efficiency"] = float(
            np.mean(
                [row["second_leg_path_efficiency"] for row in successful_recall]
            )
        )
        result["mean_successful_second_leg_distance"] = float(
            np.mean([row["second_leg_distance"] for row in successful_recall])
        )
    if any("first_resource_step" in row for row in rows):
        first_collected = [row for row in rows if row["first_resource_step"] >= 0]
        result["first_resource_collection_rate"] = float(
            len(first_collected) / len(rows)
        )
        result["second_resource_given_first_rate"] = float(
            np.mean(
                [row["second_resource_step"] >= 0 for row in first_collected]
            )
        ) if first_collected else 0.0
    return result


@torch.inference_mode()
def evaluate(
    agent: Dreamer,
    env: Any,
    *,
    task: str,
    episodes: int,
    max_steps: int,
    first_seed: int,
    stochastic: bool,
    recorder: VideoRecorder | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    step_rows: list[dict[str, Any]] = []
    episode_rows: list[dict[str, Any]] = []

    for episode in range(episodes):
        seed = first_seed + episode
        observation = env.reset(seed=seed)
        state = agent.get_initial_state(1)
        total_reward = 0.0
        info: dict[str, Any] = {}
        outcome = "evaluation_cutoff"

        for step in range(1, max_steps + 1):
            action, state = agent.act(
                observation_batch(observation, agent.device),
                state,
                eval=not stochastic,
            )
            observation, reward, done, info = env.step(
                action.squeeze(0).detach().cpu().numpy()
            )
            reward_value = float(reward)
            total_reward += reward_value
            if recorder is not None:
                recorder.write(info)

            step_row = {
                "episode": episode,
                "seed": seed,
                "step": step,
                "reward": reward_value,
                "food_consumed": int(scalar(info, "food_consumed")),
                "water_consumed": int(scalar(info, "water_consumed")),
                "hunger": scalar(info, "hunger"),
                "thirst": scalar(info, "thirst"),
                "posture": scalar(info, "posture"),
                "height": scalar(info, "z_pos"),
            }
            for key in (
                "temperature",
                "action_magnitude",
                "reward_homeostatic",
                "reward_movement_penalty",
                "reward_posture_penalty",
                "heat_exposed_time",
                "sweating",
                "is_flipped",
                "ant_x",
                "ant_y",
                "remaining_resource_x",
                "remaining_resource_y",
                "remaining_resource_visible",
                "target_bearing_error_rad",
                "initial_target_bearing_error_rad",
                "first_resource_step",
                "second_resource_step",
                "target_reacquisition_step",
                "second_leg_latency",
                "target_reacquisition_latency",
                "second_leg_distance",
                "occluded_second_leg_distance",
                "shortest_second_leg_distance",
                "second_leg_path_efficiency",
                "wall_contact_steps",
                "second_leg_stall_steps",
            ):
                if key in info:
                    step_row[key] = scalar(info, key)
            step_rows.append(step_row)

            resources_collected = (
                task in {"shift", "ymaze", "partition"}
                and scalar(info, "food_consumed") >= 1
                and scalar(info, "water_consumed") >= 1
            )
            if resources_collected and (task in {"ymaze", "partition"} or not done):
                outcome = "resources_collected"
                break
            if done:
                outcome = (
                    "environment_truncation"
                    if bool(info.get("evaluation_truncated", False))
                    else "homeostatic_termination"
                )
                break
            if step % 100 == 0:
                print(
                    f"Episode {episode + 1}/{episodes}: step {step}/{max_steps}",
                    end="\r",
                    flush=True,
                )

        episode_rows.append(
            episode_summary(
                episode, seed, step, total_reward, outcome, info
            )
        )
        print(
            f"Episode {episode + 1}/{episodes}: {outcome} after {step} steps; "
            f"food={int(scalar(info, 'food_consumed'))}, "
            f"water={int(scalar(info, 'water_consumed'))}"
        )

    return step_rows, episode_rows


def main() -> None:
    args = parse_args()
    model_path = args.model_path.resolve()
    config_path = (args.config or infer_config_path(model_path)).resolve()
    device = select_device(args.device)
    config = load_config(config_path, device)
    if str(config.env.task) != "homeoant_ant":
        raise ValueError(
            f"This evaluator supports homeoant_ant runs, not {config.env.task!r}."
        )

    episodes = args.episodes or (1 if args.task == "forage" else 10)
    render_size = None if args.no_video else (args.video_size, args.video_size)
    if args.task == "ymaze":
        env = make_ymaze_env(
            config, seed=args.seed, render_size=render_size
        )
        default_max_steps = int(env._env.cfg.max_steps)
    elif args.task == "partition":
        env = make_partition_env(
            config, seed=args.seed, render_size=render_size
        )
        default_max_steps = int(env._env.cfg.max_steps)
    else:
        env_config = make_env_config(
            config,
            task=args.task,
            seed=args.seed,
            render_size=render_size,
        )
        if render_size is None:
            env = HomeostaticAntR2Env(env_config, seed=args.seed)
        else:
            # Render the environment at the requested video resolution, then
            # downsample only the policy observation to its training size.
            env = R2DreamerEnvAdapter(
                HomeostaticAntEnv(env_config),
                seed=args.seed,
                policy_size=tuple(map(int, config.env.size)),
            )
        default_max_steps = int(env_config.eval_max_steps)
    max_steps = args.max_steps or default_max_steps
    set_seed_everywhere(args.seed)

    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else args.run_dir.resolve() / "evaluation" / DATETIME
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    recorder = None
    try:
        agent = load_agent(model_path, config, env, device)
        if not args.no_video:
            recorder = VideoRecorder(
                output_dir, args.task, args.video_fps, args.video_size
            )
        step_rows, episode_rows = evaluate(
            agent,
            env,
            task=args.task,
            episodes=episodes,
            max_steps=max_steps,
            first_seed=args.seed,
            stochastic=args.stochastic,
            recorder=recorder,
        )
    finally:
        if recorder is not None:
            recorder.close()
        env.close()

    stem = f"r2dreamer_{DATETIME}_{args.task}"
    steps_path = output_dir / f"{stem}_step_stats.csv"
    episodes_path = output_dir / f"{stem}_episode_stats.csv"
    summary_path = output_dir / f"{stem}_summary.csv"
    manifest_path = output_dir / "evaluation_manifest.json"
    write_csv(steps_path, step_rows)
    write_csv(episodes_path, episode_rows)
    write_csv(
        summary_path,
        [evaluation_summary(args.task, max_steps, episode_rows)],
    )
    manifest = {
        "created_at": DATETIME,
        "task": args.task,
        "checkpoint": str(model_path),
        "config": str(config_path),
        "episodes": episodes,
        "max_steps": max_steps,
        "first_seed": args.seed,
        "stochastic_policy": args.stochastic,
        "video_recorded": recorder is not None,
        "video_fps": args.video_fps if recorder is not None else None,
        "video_resolution": (
            [args.video_size, args.video_size] if recorder is not None else None
        ),
        "policy_resolution": list(map(int, config.env.size)),
    }
    with manifest_path.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)
    print(f"Saved step statistics to {steps_path}")
    print(f"Saved episode statistics to {episodes_path}")
    print(f"Saved evaluation summary to {summary_path}")
    print(f"Saved evaluation manifest to {manifest_path}")
    if recorder is not None:
        print(f"Saved POV video to {recorder.pov_path}")
        print(f"Saved environment video to {recorder.env_path}")


if __name__ == "__main__":
    main()
