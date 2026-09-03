"""Evaluate R2Dreamer checkpoints on Homeostatic Ant survival.

The evaluator reconstructs an agent from the Hydra configuration stored in a
training run's ``.hydra/config.yaml`` and evaluates every
``checkpoint_<step>.pt`` in a directory.  It writes checkpoint aggregates,
per-episode results, and separate mean and median survival plots.  In each
plot, the light band is the full min--max range across evaluation episodes.

Example:
    python evaluate_r2dreamer_survival.py --checkpoint-dir logdir/2026-08-26/12-00-00
"""

from __future__ import annotations

import argparse
import csv
import re
from functools import partial
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from tensordict import TensorDict

from custom_env.config_env import EnvConfig
from dreamer import Dreamer
from envs.homeostatic_ant import HomeostaticAntR2Env
from tools import set_seed_everywhere


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure R2Dreamer survival by checkpoint.")
    parser.add_argument("--checkpoint-dir", type=Path, required=True, help="Directory containing checkpoint_<step>.pt files.")
    parser.add_argument("--config", type=Path, default=None, help="Hydra config.yaml; inferred from .hydra/config.yaml when omitted.")
    parser.add_argument("--episodes", type=int, default=16, help="Evaluation episodes per checkpoint.")
    parser.add_argument(
        "--parallel-envs",
        type=int,
        default=None,
        help="Concurrent episode workers (defaults to env.env_num from the training config).",
    )
    parser.add_argument("--max-steps", type=int, default=60_000, help="Administrative per-episode survival cutoff.")
    parser.add_argument("--seed", type=int, default=0, help="First episode seed; one is added for every episode.")
    parser.add_argument("--device", default=None, help="Torch device (defaults to CUDA when available).")
    parser.add_argument("--output-dir", type=Path, default=Path("survival_evaluation"), help="Output directory.")
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes must be at least 1.")
    if args.max_steps < 1:
        parser.error("--max-steps must be at least 1.")
    if args.parallel_envs is not None and args.parallel_envs < 1:
        parser.error("--parallel-envs must be at least 1.")
    if not args.checkpoint_dir.is_dir():
        parser.error(f"Checkpoint directory not found: {args.checkpoint_dir}")
    if args.config is not None and not args.config.is_file():
        parser.error(f"Configuration file not found: {args.config}")
    return args


def find_checkpoints(directory: Path) -> list[tuple[int, Path]]:
    pattern = re.compile(r"^checkpoint_(?P<step>\d+)\.pt$", re.IGNORECASE)
    found = [(int(match.group("step")), path) for path in directory.iterdir() if path.is_file() and (match := pattern.match(path.name))]
    found.sort(key=lambda item: (item[0], item[1].name))
    if not found:
        raise FileNotFoundError(f"No checkpoint_<step>.pt files found in {directory}.")
    return found


def infer_config_path(checkpoint_dir: Path) -> Path:
    for directory in (checkpoint_dir, *checkpoint_dir.parents):
        candidate = directory / ".hydra" / "config.yaml"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("Could not find .hydra/config.yaml above the checkpoint directory; pass --config explicitly.")


def load_config(path: Path, device: torch.device) -> Any:
    try:
        from omegaconf import OmegaConf
    except ImportError as error:
        raise RuntimeError("Loading an R2Dreamer run configuration requires omegaconf (installed with hydra-core).") from error

    config = OmegaConf.load(path)
    if "model" not in config or "env" not in config:
        raise ValueError(f"{path} is not an R2Dreamer Hydra configuration (expected model and env sections).")
    OmegaConf.update(config, "device", str(device), force_add=True)
    OmegaConf.update(config, "model.device", str(device), force_add=True)
    # Evaluation never calls update(), so compilation only adds needless setup.
    OmegaConf.update(config, "model.compile", False, force_add=True)
    OmegaConf.resolve(config)
    return config


def load_agent(checkpoint: Path, config: Any, device: torch.device) -> Dreamer:
    # The adapter establishes the precise observation/action spaces used in training.
    image_size = tuple(map(int, config.env.size))
    probe_env = HomeostaticAntR2Env(EnvConfig(seed=0, image_size=image_size, is_training=True, training_initial_bounds=0.0))
    try:
        # Hydra saves resolved configs with the chosen distribution directly in
        # ``actor.dist`` (for example ``{name: bounded_normal}``), whereas
        # Dreamer expects the training-time selector mapping at this point.
        # Restore that mapping only for evaluation configuration reconstruction.
        actor_dist = config.model.actor.dist
        if "name" in actor_dist:
            if hasattr(probe_env.action_space, "multi_discrete"):
                dist_key = "multi_disc"
            elif hasattr(probe_env.action_space, "discrete"):
                dist_key = "disc"
            else:
                dist_key = "cont"
            config.model.actor.dist = {dist_key: actor_dist}
        agent = Dreamer(config.model, probe_env.observation_space, probe_env.action_space).to(device)
    finally:
        probe_env.close()
    try:
        payload = torch.load(checkpoint, map_location=device, weights_only=False)
    except TypeError:
        payload = torch.load(checkpoint, map_location=device)
    if not isinstance(payload, dict) or "agent_state_dict" not in payload:
        raise ValueError(f"{checkpoint} is not an R2Dreamer checkpoint: missing agent_state_dict.")
    agent.load_state_dict(payload["agent_state_dict"], strict=True)
    return agent.eval()


def scalar(info: dict[str, Any], key: str, default: float = 0.0) -> float:
    return float(np.asarray(info.get(key, default)).item())


def unpack_step_result(result: tuple[Any, ...]) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
    """Accept either Gymnasium or R2Dreamer's legacy transition format.

    ``HomeostaticAntR2Env`` deliberately exposes the format used by the
    R2Dreamer training loop: ``(observation, reward, done, info)``.  Other
    Gymnasium environments expose ``(observation, reward, terminated,
    truncated, info)``.  Treat the legacy ``done`` flag as a termination;
    that is the adapter's homeostatic episode boundary.
    """
    if len(result) == 5:
        observation, reward, terminated, truncated, info = result
        return observation, reward, bool(terminated), bool(truncated), info
    if len(result) == 4:
        observation, reward, done, info = result
        return observation, reward, bool(done), False, info
    raise ValueError(f"Environment step() returned {len(result)} values; expected 4 or 5.")


class GymnasiumEvaluationEnv(gym.Env):
    """Expose HomeostaticAntR2Env through Gymnasium's current vector API."""

    def __init__(self, seed: int, image_size: tuple[int, int]):
        self._env = HomeostaticAntR2Env(
            EnvConfig(
                seed=seed,
                image_size=image_size,
                is_training=True,
                training_initial_bounds=0.0,
            ),
            seed=seed,
        )
        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        del seed, options
        return self._env.reset(), {}

    def step(
        self, action: np.ndarray
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        return unpack_step_result(self._env.step(action))

    def close(self) -> None:
        self._env.close()


def vector_info_at(infos: dict[str, Any], index: int) -> dict[str, Any]:
    """Recover one worker's info from Gymnasium's dict-of-arrays format."""
    result = {}
    for key, values in infos.items():
        if key.startswith("_"):
            continue
        mask = infos.get(f"_{key}")
        if mask is None or bool(mask[index]):
            result[key] = values[index]
    return result


def observation_batch(
    observations: dict[str, np.ndarray], device: torch.device
) -> TensorDict:
    tensors = {
        key: torch.as_tensor(value) for key, value in observations.items()
    }
    batch_size = next(iter(observations.values())).shape[0]
    return TensorDict(tensors, batch_size=(batch_size,)).to(device)


@torch.inference_mode()
def run_episode_batch(
    agent: Dreamer,
    image_size: tuple[int, int],
    seeds: list[int],
    max_steps: int,
) -> list[dict[str, Any]]:
    envs = gym.vector.AsyncVectorEnv(
        [partial(GymnasiumEvaluationEnv, seed, image_size) for seed in seeds],
        context="spawn",
        autoreset_mode=gym.vector.AutoresetMode.NEXT_STEP,
    )
    try:
        observations, _ = envs.reset()
        state = agent.get_initial_state(len(seeds))
        active = np.ones(len(seeds), dtype=np.bool_)
        steps = np.zeros(len(seeds), dtype=np.int64)
        flips = np.zeros(len(seeds), dtype=np.int64)
        episodes: list[dict[str, Any] | None] = [None] * len(seeds)

        while active.any():
            actions, state = agent.act(
                observation_batch(observations, agent.device), state, eval=True
            )
            observations, _, terminated, truncated, infos = envs.step(
                actions.detach().cpu().numpy()
            )
            for index in np.flatnonzero(active):
                info = vector_info_at(infos, index)
                steps[index] += 1
                flips[index] += int(bool(scalar(info, "is_flipped")))
                reached_cutoff = steps[index] >= max_steps
                if terminated[index] or truncated[index] or reached_cutoff:
                    outcome = (
                        "homeostatic_termination"
                        if terminated[index]
                        else "environment_truncation"
                        if truncated[index]
                        else "evaluation_cutoff"
                    )
                    episodes[index] = {
                        "seed": seeds[index],
                        "survival_steps": int(steps[index]),
                        "outcome": outcome,
                        "final_state_label": int(scalar(info, "termination_reason")),
                        "food_consumed": int(scalar(info, "food_consumed")),
                        "water_consumed": int(scalar(info, "water_consumed")),
                        "flipped_steps": int(flips[index]),
                        "final_hunger": scalar(info, "hunger"),
                        "final_thirst": scalar(info, "thirst"),
                        "final_posture": scalar(info, "posture"),
                        "final_height": scalar(info, "z_pos"),
                    }
                    active[index] = False
    finally:
        envs.close()

    if any(episode is None for episode in episodes):
        raise RuntimeError("An evaluation episode ended without an outcome.")
    return [episode for episode in episodes if episode is not None]


def evaluate_checkpoint(args: argparse.Namespace, step: int, checkpoint: Path, config: Any, device: torch.device) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    set_seed_everywhere(args.seed)
    agent = load_agent(checkpoint, config, device)
    image_size = tuple(map(int, config.env.size))
    configured_workers = int(getattr(config.env, "env_num", args.episodes))
    parallel_envs = min(args.parallel_envs or configured_workers, args.episodes)
    print(f"  Running {args.episodes} episodes with {parallel_envs} parallel workers.", flush=True)
    episode_seeds = [args.seed + index for index in range(args.episodes)]
    episodes = []
    for start in range(0, args.episodes, parallel_envs):
        batch_seeds = episode_seeds[start : start + parallel_envs]
        episodes.extend(run_episode_batch(agent, image_size, batch_seeds, args.max_steps))
    lengths = np.asarray([episode["survival_steps"] for episode in episodes], dtype=np.float64)
    row = {
        "checkpoint_step": step, "checkpoint": str(checkpoint), "episodes": args.episodes,
        "max_steps": args.max_steps, "seed": args.seed,
        "mean_survival_steps": float(lengths.mean()), "median_survival_steps": float(np.median(lengths)),
        "min_survival_steps": int(lengths.min()), "max_survival_steps": int(lengths.max()),
        "survival_rate_to_cutoff": float(sum(item["outcome"] == "evaluation_cutoff" for item in episodes) / args.episodes),
        "homeostatic_terminations": sum(item["outcome"] == "homeostatic_termination" for item in episodes),
        "environment_truncations": sum(item["outcome"] == "environment_truncation" for item in episodes),
        "evaluation_cutoffs": sum(item["outcome"] == "evaluation_cutoff" for item in episodes),
    }
    for episode in episodes:
        episode.update({"checkpoint_step": step, "checkpoint": str(checkpoint)})
    return row, episodes


def save_survival_plot(rows: list[dict[str, Any]], statistic: str, output_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError("Creating survival plots requires matplotlib.") from error
    ordered = sorted(rows, key=lambda row: int(row["checkpoint_step"]))
    steps = np.asarray([row["checkpoint_step"] for row in ordered])
    values = np.asarray([row[f"{statistic}_survival_steps"] for row in ordered])
    lower = np.asarray([row["min_survival_steps"] for row in ordered])
    upper = np.asarray([row["max_survival_steps"] for row in ordered])
    figure, axis = plt.subplots(figsize=(10, 5), constrained_layout=True)
    color = "C0" if statistic == "mean" else "C1"
    axis.fill_between(steps, lower, upper, color=color, alpha=0.18, label="Episode range (min--max)")
    axis.plot(steps, values, color=color, marker="o", linewidth=2, label=f"{statistic.title()} survival")
    axis.set_title(f"R2Dreamer {statistic} survival by checkpoint")
    axis.set_xlabel("Training steps")
    axis.set_ylabel(f"{statistic.title()} survival steps")
    axis.set_ylim(bottom=0)
    axis.set_xticks(steps)
    axis.ticklabel_format(style="plain", axis="x")
    axis.grid(axis="y", alpha=0.3)
    axis.legend()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    checkpoint_dir = args.checkpoint_dir.resolve()
    config_path = (args.config or infer_config_path(checkpoint_dir)).resolve()
    device = torch.device(args.device) if args.device else (torch.accelerator.current_accelerator() if torch.accelerator.is_available() else torch.device("cpu"))
    config = load_config(config_path, device)
    if str(config.env.task) != "homeoant_ant":
        raise ValueError(f"This evaluator supports homeoant_ant runs, not {config.env.task!r}.")

    rows, episode_rows = [], []
    for step, checkpoint in find_checkpoints(checkpoint_dir):
        print(f"Evaluating checkpoint {step}: {checkpoint.name}", flush=True)
        row, episodes = evaluate_checkpoint(args, step, checkpoint.resolve(), config, device)
        rows.append(row)
        episode_rows.extend(episodes)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    aggregate_path = output_dir / "survival_by_checkpoint.csv"
    episodes_path = output_dir / "survival_episodes_by_checkpoint.csv"
    mean_plot_path = output_dir / "mean_survival_by_checkpoint.png"
    median_plot_path = output_dir / "median_survival_by_checkpoint.png"
    write_csv(aggregate_path, rows)
    write_csv(episodes_path, episode_rows)
    save_survival_plot(rows, "mean", mean_plot_path)
    save_survival_plot(rows, "median", median_plot_path)
    print(f"Saved aggregate data to {aggregate_path}")
    print(f"Saved per-episode data to {episodes_path}")
    print(f"Saved mean survival plot to {mean_plot_path}")
    print(f"Saved median survival plot to {median_plot_path}")


if __name__ == "__main__":
    main()
