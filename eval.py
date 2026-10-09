"""Evaluate a trained R2Dreamer agent on the thesis evaluation tasks.

This standalone evaluator restores the Hydra configuration saved with a
training run, loads a single R2Dreamer
checkpoint, carries the RSSM state between environment steps, records POV and
environment videos, and writes step- and episode-level CSV statistics.

Examples
--------
Evaluate the normal foraging task from a completed run::

    python eval.py --task forage --run-dir logdir/homeostatic-ant

Evaluate notebook-style imagination from the run's latest checkpoint::

    python eval.py --task imagine \
        --run-dir logdir/homeostatic-ant \
        --episodes 100 \
        --seed 100000 \
        --no-video

Evaluate zero-shot landmark recall in the partitioned open field::

    python eval.py --task partition --run-dir logdir/homeostatic-ant

Use observed camera features for Action-Visual SRU evaluation only::

    python eval.py --task partition --run-dir logdir/homeostatic-ant \
        --sru-visual-source observed
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
from custom_env.partition_env import PartitionRecallEnv
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
        choices=("forage", "imagine", "partition"),
        required=True,
        help="Evaluation task.",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Training-run directory containing latest.pt and .hydra/config.yaml.",
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
        help="Number of episodes (default: 1 forage, 100 imagine, 100 partition).",
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
        "--sru-visual-source", choices=("predicted", "observed"), default="predicted",
        help="Action-Visual SRU modulation during real observation updates; imagination always uses predicted features.",
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
    parser.add_argument(
        "--imagination-anchor-delay",
        type=int,
        default=10,
        help="Real steps after first consumption before anchoring imagination.",
    )
    parser.add_argument(
        "--imagination-horizon",
        type=int,
        default=50,
        help="Number of future steps in each imagination rollout.",
    )
    args = parser.parse_args()

    if not args.run_dir.is_dir():
        parser.error(f"Run directory not found: {args.run_dir}")
    args.model_path = args.run_dir / "latest.pt"
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
    if args.imagination_anchor_delay < 0:
        parser.error("--imagination-anchor-delay cannot be negative.")
    if args.imagination_horizon < 1:
        parser.error("--imagination-horizon must be at least 1.")
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
        imagine=task == "imagine",
    )
    return EnvConfig(**configured)


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
        imagine=False,
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

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> dict[str, np.ndarray]:
        result = self._env.reset(
            seed=self._seed if seed is None else seed, options=options
        )
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
    for key in ("initial_food_side", "initial_water_side"):
        if key in info:
            row[key] = str(info[key])
    for key in (
        "initial_food_x",
        "initial_food_y",
        "initial_water_x",
        "initial_water_y",
    ):
        if key in info:
            row[key] = scalar(info, key)
    if "resources_consumed" in info:
        resources = [str(resource) for resource in info["resources_consumed"]]
        row["resource_sequence"] = ">".join(resources)
        expected_first = (
            "food"
            if scalar(info, "initial_hunger") < scalar(info, "initial_thirst")
            else "water"
        )
        row["expected_first_resource"] = expected_first
        expected_sequence = [
            expected_first,
            "water" if expected_first == "food" else "food",
        ]
        row["expected_resource_sequence"] = ">".join(expected_sequence)
        row["correct_first_resource"] = int(
            bool(resources) and resources[0] == expected_first
        )
        row["second_resource_collected"] = int(len(resources) >= 2)
        row["correct_resource_sequence"] = int(resources == expected_sequence)
    for key in (
        "wall_contact_steps",
        "minimum_camera_wall_distance",
        "near_clip_distance",
        "near_clip_risk_frames",
        "camera_inside_partition_frames",
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
            np.mean(
                [
                    row["food_consumed"] >= 1 and row["water_consumed"] >= 1
                    for row in rows
                ]
            )
        ) if task in {"partition", "imagine"} else float(
            np.mean([row["outcome"] == "resources_collected" for row in rows])
        ),
        "survival_to_cutoff_rate": float(
            np.mean([row["outcome"] == "evaluation_cutoff" for row in rows])
        ),
    }
    if any("correct_first_resource" in row for row in rows):
        result["first_resource_accuracy"] = float(
            np.mean([row.get("correct_first_resource", 0) for row in rows])
        )
        result["second_resource_collection_rate"] = float(
            np.mean([row.get("second_resource_collected", 0) for row in rows])
        )
        result["correct_sequence_success_rate"] = float(
            np.mean([row.get("correct_resource_sequence", 0) for row in rows])
        )
    if any("dominant_need" in row for row in rows):
        result["hunger_dominant_episodes"] = sum(
            row.get("dominant_need") == "hunger" for row in rows
        )
        result["thirst_dominant_episodes"] = sum(
            row.get("dominant_need") == "thirst" for row in rows
        )
        result["food_left_episodes"] = sum(
            row.get("food_side") == "left" for row in rows
        )
        result["food_right_episodes"] = sum(
            row.get("food_side") == "right" for row in rows
        )
    if task == "partition":
        return {
            key: result[key]
            for key in (
                "task",
                "episodes",
                "max_steps",
                "first_resource_accuracy",
                "second_resource_collection_rate",
                "correct_sequence_success_rate",
                "hunger_dominant_episodes",
                "thirst_dominant_episodes",
                "food_left_episodes",
                "food_right_episodes",
            )
        }
    if task == "imagine":
        predicted = [row for row in rows if "mean_actual_action_ssim" in row]
        result["imagination_completion_rate"] = float(
            np.mean([row.get("full_imagination_horizon", 0) for row in rows])
        )
        result["mean_actual_action_ssim"] = (
            float(
                np.mean([row["mean_actual_action_ssim"] for row in predicted])
            )
            if predicted
            else float("nan")
        )
        result["mean_imagined_action_ssim"] = (
            float(
                np.mean([row["mean_imagined_action_ssim"] for row in predicted])
            )
            if predicted
            else float("nan")
        )
    return result


def balanced_resource_conditions(
    episodes: int, seed: int
) -> list[dict[str, str]]:
    """Create a reproducible, near-factorially-balanced condition schedule."""
    combinations = [
        {"dominant_need": need, "food_side": side}
        for need in ("hunger", "thirst")
        for side in ("left", "right")
    ]
    conditions = [dict(item) for item in combinations for _ in range(episodes // 4)]
    remainder = episodes % 4
    rng = np.random.default_rng(seed)
    if remainder == 1:
        conditions.append(dict(combinations[int(rng.integers(4))]))
    elif remainder == 2:
        diagonals = ((0, 3), (1, 2))
        for index in diagonals[int(rng.integers(2))]:
            conditions.append(dict(combinations[index]))
    elif remainder == 3:
        omitted = int(rng.integers(4))
        conditions.extend(
            dict(item) for index, item in enumerate(combinations) if index != omitted
        )
    rng.shuffle(conditions)
    return conditions


def set_torch_seed(seed: int) -> None:
    """Reset latent sampling without changing an environment's private RNG."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_rollout_grid(
    path: Path,
    rollouts: list[tuple[str, list[np.ndarray]]],
    *,
    sample_interval: int = 5,
) -> None:
    """Save labelled rollout rows using the anchor and every Nth future frame."""
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError(
            "Saving imagination PNGs requires opencv-python."
        ) from error
    if sample_interval < 1:
        raise ValueError("sample_interval must be positive")
    if not rollouts or any(not frames for _, frames in rollouts):
        raise ValueError("Each rollout row must contain at least one frame")

    labelled_rows: list[tuple[str, list[np.ndarray]]] = []
    for label, frames in rollouts:
        indices = [0, *range(sample_interval, len(frames), sample_interval)]
        images = []
        for index in indices:
            image = np.asarray(frames[index])[..., :3]
            if np.issubdtype(image.dtype, np.floating):
                image = np.clip(image, 0.0, 1.0) * 255.0
            images.append(np.asarray(image, dtype=np.uint8))
        labelled_rows.append((label, images))

    frame_height, frame_width = labelled_rows[0][1][0].shape[:2]
    column_count = max(len(images) for _, images in labelled_rows)
    label_width = max(128, frame_width * 2)
    canvas = np.full(
        (frame_height * len(labelled_rows), label_width + frame_width * column_count, 3),
        255,
        dtype=np.uint8,
    )
    for row_index, (label, images) in enumerate(labelled_rows):
        y_start = row_index * frame_height
        for column_index, image in enumerate(images):
            if image.shape[:2] != (frame_height, frame_width):
                raise ValueError("All rollout images must have identical dimensions")
            x_start = label_width + column_index * frame_width
            canvas[
                y_start : y_start + frame_height,
                x_start : x_start + frame_width,
            ] = image
        cv2.putText(
            canvas,
            label,
            (8, y_start + frame_height // 2 + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
        if row_index:
            cv2.line(
                canvas,
                (0, y_start),
                (canvas.shape[1] - 1, y_start),
                (192, 192, 192),
                1,
            )

    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)):
        raise RuntimeError(f"Failed to save imagination image: {path}")


def rgb_ssim(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Return RGB structural similarity for images represented on [0, 1]."""
    try:
        from skimage.metrics import structural_similarity
    except ImportError as error:
        raise RuntimeError(
            "Computing imagination SSIM requires scikit-image."
        ) from error
    return float(
        structural_similarity(
            np.clip(actual[..., :3], 0.0, 1.0),
            np.clip(predicted[..., :3], 0.0, 1.0),
            channel_axis=-1,
            data_range=1.0,
        )
    )


@torch.inference_mode()
def evaluate_imagination(
    agent: Dreamer,
    env: Any,
    *,
    episodes: int,
    max_steps: int,
    first_seed: int,
    stochastic: bool,
    anchor_delay: int,
    horizon: int,
    image_dir: Path,
    recorder: VideoRecorder | None,
    sru_visual_source: str = "predicted",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Run the notebook protocol and save one labelled rollout grid per episode."""
    if not hasattr(agent, "decoder"):
        raise ValueError(
            "Imagination images require a checkpoint trained with rep_loss='dreamer'; "
            "this checkpoint has no observation decoder."
        )

    step_rows: list[dict[str, Any]] = []
    episode_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    conditions = balanced_resource_conditions(episodes, first_seed)

    for episode, condition in enumerate(conditions):
        environment_seed = first_seed + episode
        latent_seed = first_seed + 1_000_000_000 + episode
        observation = env.reset(seed=environment_seed, options=condition)
        set_torch_seed(latent_seed)
        state = agent.get_initial_state(1)
        precollection_frames = [observation["image"].copy()]
        total_reward = 0.0
        info: dict[str, Any] = {}
        outcome = "evaluation_cutoff"
        first_consumption_step: int | None = None
        steps_after_consumption: int | None = None
        anchor_step: int | None = None
        anchor: tuple[torch.Tensor, torch.Tensor] | None = None
        anchor_image: np.ndarray | None = None
        future_actions: list[torch.Tensor] = []
        actual_images: list[np.ndarray] = []
        previous_consumed = 0

        for step in range(1, max_steps + 1):
            action, state = agent.act(
                observation_batch(observation, agent.device),
                state,
                sru_visual_source=sru_visual_source,
            )
            if steps_after_consumption == anchor_delay and anchor is None:
                anchor = (state["stoch"].clone(), state["deter"].clone())
                anchor_image = observation["image"].copy()
                anchor_step = step - 1

            next_observation, reward, done, info = env.step(
                action.squeeze(0).detach().cpu().numpy()
            )
            if anchor is not None:
                future_actions.append(action.detach().clone())
                actual_images.append(next_observation["image"].copy())

            reward_value = float(reward)
            total_reward += reward_value
            if recorder is not None:
                recorder.write(info)
            consumed = int(
                scalar(info, "food_consumed") + scalar(info, "water_consumed")
            )
            if consumed == 0 and step % 10 == 0:
                precollection_frames.append(next_observation["image"].copy())
            step_rows.append(
                {
                    "episode": episode + 1,
                    "environment_seed": environment_seed,
                    "latent_seed": latent_seed,
                    "dominant_need": condition["dominant_need"],
                    "food_side": condition["food_side"],
                    "step": step,
                    "reward": reward_value,
                    "food_consumed": int(scalar(info, "food_consumed")),
                    "water_consumed": int(scalar(info, "water_consumed")),
                    "hunger": scalar(info, "hunger"),
                    "thirst": scalar(info, "thirst"),
                }
            )

            if previous_consumed == 0 and consumed >= 1:
                first_consumption_step = step
                steps_after_consumption = 0
            elif steps_after_consumption is not None and anchor is None:
                steps_after_consumption += 1
            previous_consumed = consumed
            observation = next_observation

            if len(future_actions) >= horizon:
                outcome = "imagination_complete"
                break
            if consumed >= 2:
                outcome = "resources_collected"
                break
            if done:
                outcome = (
                    "environment_truncation"
                    if bool(info.get("evaluation_truncated", False))
                    else "homeostatic_termination"
                )
                break

        row = episode_summary(
            episode + 1,
            environment_seed,
            step,
            total_reward,
            outcome,
            info,
        )
        row.update(
            {
                "environment_seed": environment_seed,
                "latent_seed": latent_seed,
                "dominant_need": condition["dominant_need"],
                "food_side": condition["food_side"],
                "first_consumption_step": first_consumption_step,
                "imagination_anchor_step": anchor_step,
                "imagination_steps": len(future_actions),
                "full_imagination_horizon": int(len(future_actions) == horizon),
            }
        )

        if anchor is not None and future_actions and anchor_image is not None:
            actions = torch.stack(future_actions, dim=1)
            set_torch_seed(latent_seed + 10_000_000)
            actual_stoch, actual_deter = agent._frozen_rssm.imagine_with_action(
                *anchor, actions
            )
            actual_action_prediction = (
                agent.decoder(actual_stoch, actual_deter)["image"]
                .mode()[0]
                .cpu()
                .numpy()
            )

            set_torch_seed(latent_seed + 20_000_000)
            stoch, deter = (value.clone() for value in anchor)
            imagined_stoch, imagined_deter = [], []
            for _ in range(len(future_actions)):
                feat = agent._frozen_rssm.get_feat(stoch, deter)
                imagined_action = agent._frozen_actor(feat).rsample()
                stoch, deter = agent._frozen_rssm.img_step(
                    stoch, deter, imagined_action
                )
                imagined_stoch.append(stoch)
                imagined_deter.append(deter)
            imagined_prediction = (
                agent.decoder(
                    torch.stack(imagined_stoch, dim=1),
                    torch.stack(imagined_deter, dim=1),
                )["image"]
                .mode()[0]
                .cpu()
                .numpy()
            )

            actual = np.stack(actual_images).astype(np.float32) / 255.0
            episode_predictions: list[dict[str, Any]] = []
            for index, (truth, actual_action_frame, imagined_action_frame) in enumerate(
                zip(
                    actual,
                    actual_action_prediction,
                    imagined_prediction,
                ),
                start=1,
            ):
                prediction_row = {
                    "episode": episode + 1,
                    "environment_seed": environment_seed,
                    "latent_seed": latent_seed,
                    "horizon_step": index,
                    "actual_action_ssim": rgb_ssim(
                        truth, actual_action_frame
                    ),
                    "imagined_action_ssim": rgb_ssim(
                        truth, imagined_action_frame
                    ),
                }
                episode_predictions.append(prediction_row)
                prediction_rows.append(prediction_row)
            row["mean_actual_action_ssim"] = float(
                np.mean(
                    [item["actual_action_ssim"] for item in episode_predictions]
                )
            )
            row["mean_imagined_action_ssim"] = float(
                np.mean(
                    [item["imagined_action_ssim"] for item in episode_predictions]
                )
            )
            save_rollout_grid(
                image_dir / f"ep{episode + 1}.png",
                [
                    ("actual", [anchor_image, *actual_images]),
                    ("imagine", [anchor_image, *list(imagined_prediction)]),
                    (
                        "actualimagine",
                        [anchor_image, *list(actual_action_prediction)],
                    ),
                ],
            )

        save_rollout_grid(
            image_dir / f"ep{episode + 1}_precollection.png",
            [("precollection", precollection_frames)],
            sample_interval=1,
        )

        episode_rows.append(row)
        print(
            f"Episode {episode + 1}/{episodes}: {outcome} after {step} steps; "
            f"imagination={len(future_actions)}/{horizon}",
            flush=True,
        )

    return step_rows, episode_rows, prediction_rows


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
    sru_visual_source: str = "predicted",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    step_rows: list[dict[str, Any]] = []
    episode_rows: list[dict[str, Any]] = []

    conditions: list[dict[str, str] | None]
    if task == "partition":
        conditions = balanced_resource_conditions(episodes, first_seed)
    else:
        conditions = [None] * episodes

    for episode, condition in enumerate(conditions):
        seed = first_seed + episode
        latent_seed = first_seed + 1_000_000_000 + episode
        observation = env.reset(seed=seed, options=condition)
        set_torch_seed(latent_seed)
        state = agent.get_initial_state(1)
        total_reward = 0.0
        info: dict[str, Any] = {}
        outcome = "evaluation_cutoff"

        for step in range(1, max_steps + 1):
            action, state = agent.act(
                observation_batch(observation, agent.device),
                state,
                sru_visual_source=sru_visual_source,
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
                "latent_seed": latent_seed,
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
                "wall_contact_steps",
                "camera_wall_distance",
                "minimum_camera_wall_distance",
                "near_clip_distance",
                "near_clip_risk",
                "near_clip_risk_frames",
                "camera_inside_partition",
                "camera_inside_partition_frames",
            ):
                if key in info:
                    step_row[key] = scalar(info, key)
            step_rows.append(step_row)

            resources_collected = (
                task in {"imagine", "partition"}
                and scalar(info, "food_consumed") >= 1
                and scalar(info, "water_consumed") >= 1
            )
            if resources_collected and (task == "partition" or not done):
                if task == "partition":
                    resources = [
                        str(resource)
                        for resource in info.get("resources_consumed", [])
                    ]
                    expected_first = (
                        "food"
                        if scalar(info, "initial_hunger")
                        < scalar(info, "initial_thirst")
                        else "water"
                    )
                    expected_sequence = [
                        expected_first,
                        "water" if expected_first == "food" else "food",
                    ]
                    outcome = (
                        "success"
                        if resources == expected_sequence
                        else "incorrect_resource_sequence"
                    )
                else:
                    outcome = "resources_collected"
                break
            if done:
                outcome = (
                    "environment_truncation"
                    if bool(info.get("evaluation_truncated", False))
                    else "homeostatic_termination"
                )
                break
            # if step % 100 == 0:
            #     print(
            #         f"Episode {episode + 1}/{episodes}: step {step}/{max_steps}",
            #         end="\r",
            #         flush=True,
            #     )

        episode_row = episode_summary(
            episode, seed, step, total_reward, outcome, info
        )
        episode_row["latent_seed"] = latent_seed
        if condition is not None:
            episode_row.update(condition)
        episode_rows.append(episode_row)
        print(
            f"Episode {episode + 1}/{episodes}: {outcome} after {step} steps; "
            f"food={int(scalar(info, 'food_consumed'))}, "
            f"water={int(scalar(info, 'water_consumed'))}"
        )

    return step_rows, episode_rows


def make_task_env(
    args: argparse.Namespace,
    config: Any,
    render_size: tuple[int, int] | None,
) -> tuple[Any, int]:
    if args.task == "partition":
        env = make_partition_env(config, seed=args.seed, render_size=render_size)
        return env, int(env._env.cfg.max_steps)

    env_config = make_env_config(
        config,
        task=args.task,
        seed=args.seed,
        render_size=render_size,
    )
    if render_size is None:
        env = HomeostaticAntR2Env(env_config, seed=args.seed)
    else:
        env = R2DreamerEnvAdapter(
            HomeostaticAntEnv(env_config),
            seed=args.seed,
            policy_size=tuple(map(int, config.env.size)),
        )
    return env, int(env_config.eval_max_steps)


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

    default_episodes = {"forage": 1, "imagine": 100, "partition": 100}
    episodes = args.episodes or default_episodes[args.task]
    render_size = None if args.no_video else (args.video_size, args.video_size)
    run_dir = config_path.parent.parent
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else run_dir / "evaluation" / DATETIME
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    env, default_max_steps = make_task_env(args, config, render_size)
    max_steps = args.max_steps or default_max_steps
    set_seed_everywhere(args.seed)
    recorder = None
    prediction_rows: list[dict[str, Any]] = []
    try:
        print(f"Evaluating checkpoint: {model_path.name}", flush=True)
        agent = load_agent(model_path, config, env, device)
        if args.sru_visual_source == "observed" and not agent.use_action_visual_sru:
            raise ValueError("--sru-visual-source observed requires an action_visual_sru checkpoint")
        if not args.no_video:
            recorder = VideoRecorder(
                output_dir,
                args.task,
                args.video_fps,
                args.video_size,
            )
        if args.task == "imagine":
            step_rows, episode_rows, prediction_rows = evaluate_imagination(
                agent,
                env,
                episodes=episodes,
                max_steps=max_steps,
                first_seed=args.seed,
                stochastic=args.stochastic,
                anchor_delay=args.imagination_anchor_delay,
                horizon=args.imagination_horizon,
                image_dir=output_dir / "imagination",
                recorder=recorder,
                sru_visual_source=args.sru_visual_source,
            )
        else:
            step_rows, episode_rows = evaluate(
                agent,
                env,
                task=args.task,
                episodes=episodes,
                max_steps=max_steps,
                first_seed=args.seed,
                stochastic=args.stochastic,
                recorder=recorder,
                sru_visual_source=args.sru_visual_source,
            )
    finally:
        if recorder is not None:
            recorder.close()
        env.close()

    for rows in (step_rows, episode_rows, prediction_rows):
        for row in rows:
            row["checkpoint"] = str(model_path)
            row["sru_visual_source"] = args.sru_visual_source

    stem = f"r2dreamer_{DATETIME}_{args.task}"
    steps_path = output_dir / f"{stem}_step_stats.csv"
    episodes_path = output_dir / f"{stem}_episode_stats.csv"
    prediction_path = output_dir / f"{stem}_prediction_stats.csv"
    summary_path = output_dir / f"{stem}_summary.csv"
    write_csv(steps_path, step_rows)
    write_csv(episodes_path, episode_rows)
    if prediction_rows:
        write_csv(prediction_path, prediction_rows)
    summary = evaluation_summary(args.task, max_steps, episode_rows)
    summary["checkpoint"] = str(model_path)
    summary["sru_visual_source"] = args.sru_visual_source
    write_csv(summary_path, [summary])

    manifest_path = output_dir / "evaluation_manifest.json"
    manifest = {
        "created_at": DATETIME,
        "task": args.task,
        "checkpoint": str(model_path),
        "config": str(config_path),
        "episodes": episodes,
        "max_steps": max_steps,
        "first_environment_seed": args.seed,
        "first_latent_seed": args.seed + 1_000_000_000,
        "condition_schedule": (
            "seeded_balanced_2x2"
            if args.task in {"imagine", "partition"}
            else None
        ),
        "stochastic_policy": args.stochastic,
        "sru_visual_source": args.sru_visual_source,
        "imagination_anchor_delay": (
            args.imagination_anchor_delay if args.task == "imagine" else None
        ),
        "imagination_horizon": (
            args.imagination_horizon if args.task == "imagine" else None
        ),
        "video_recorded": not args.no_video,
        "video_fps": args.video_fps if not args.no_video else None,
        "video_resolution": (
            [args.video_size, args.video_size] if not args.no_video else None
        ),
        "policy_resolution": list(map(int, config.env.size)),
    }
    with manifest_path.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)
    print(f"Saved evaluation to {output_dir}")
    print(f"Saved evaluation manifest to {manifest_path}")


if __name__ == "__main__":
    main()
