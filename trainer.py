import csv

import numpy as np
import torch

import tools


_SURVIVAL_FINAL_FIELDS = {
    "final_state_label": "log_termination_reason",
    "food_consumed": "log_food_consumed",
    "water_consumed": "log_water_consumed",
    "final_hunger": "log_hunger",
    "final_thirst": "log_thirst",
    "final_posture": "log_posture",
    "final_height": "log_z_pos",
}


def _summarize_survival_episodes(episodes, train_step, checkpoint, max_steps, seed):
    lengths = np.asarray(
        [episode["survival_steps"] for episode in episodes], dtype=np.float64
    )
    return {
        "checkpoint_step": train_step,
        "checkpoint": str(checkpoint),
        "episodes": len(episodes),
        "max_steps": max_steps,
        "seed": seed,
        "mean_survival_steps": float(lengths.mean()),
        "median_survival_steps": float(np.median(lengths)),
        "min_survival_steps": int(lengths.min()),
        "max_survival_steps": int(lengths.max()),
        "survival_rate_to_cutoff": float(
            sum(item["outcome"] == "evaluation_cutoff" for item in episodes)
            / len(episodes)
        ),
        "homeostatic_terminations": sum(
            item["outcome"] == "homeostatic_termination" for item in episodes
        ),
        "environment_truncations": sum(
            item["outcome"] == "environment_truncation" for item in episodes
        ),
        "evaluation_cutoffs": sum(
            item["outcome"] == "evaluation_cutoff" for item in episodes
        ),
    }


def _append_csv(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def _evaluation_step_masks(done, once_done):
    """Return environment-reset and metric-counting masks for evaluation.

    A worker that just terminated is reset exactly once because ``done`` is
    cleared by the reset transition. Workers whose result is already recorded
    may subsequently be stepped by the vector backend, but those transitions
    are ignored and cannot alter their evaluation result.
    """
    reset = done
    stepped = ~done & ~once_done
    return reset, stepped


class OnlineTrainer:
    def __init__(self, config, replay_buffer, logger, logdir, train_envs, eval_envs):
        self.replay_buffer = replay_buffer
        self.logger = logger
        self.logdir = logdir
        self.train_envs = train_envs
        self.eval_envs = eval_envs
        self.steps = int(config.steps)
        self.pretrain = int(config.pretrain)
        self.eval_every = int(config.eval_every)
        self.eval_episode_num = int(config.eval_episode_num)
        self.eval_max_steps = int(config.eval_max_steps)
        self.survival_eval = bool(config.survival_eval)
        self.eval_seed = int(config.eval_seed)
        if self.survival_eval and self.eval_max_steps < 1:
            raise ValueError("trainer.eval_max_steps must be positive for survival evaluation")
        self.video_pred_log = bool(config.video_pred_log)
        self.params_hist_log = bool(config.params_hist_log)
        self.checkpoint_every = int(config.checkpoint_every)
        if self.checkpoint_every < 0:
            raise ValueError("trainer.checkpoint_every must be zero or positive")
        self.batch_length = int(config.batch_length)
        batch_steps = int(config.batch_size * config.batch_length)
        # train_ratio is based on data steps rather than environment steps.
        self._updates_needed = tools.Every(batch_steps / config.train_ratio * config.action_repeat)
        self._should_pretrain = tools.Once()
        self._should_log = tools.Every(config.update_log_every)
        self._should_eval = tools.Every(self.eval_every)
        self._action_repeat = config.action_repeat

    def save_checkpoint(self, agent, step):
        """Save a resumable model snapshot for the given environment step."""
        items_to_save = {
            "step": step,
            "agent_state_dict": agent.state_dict(),
            "optims_state_dict": tools.recursively_collect_optim_state_dict(agent),
        }
        path = self.logdir / f"checkpoint_{step:012d}.pt"
        torch.save(items_to_save, path)
        print(f"Saved checkpoint: {path}")

    def eval(self, agent, train_step):
        """Run evaluation episodes.

        For CPU-based environments (``ParallelEnv``), stepping is executed on
        CPU and observations are moved to GPU asynchronously.  For GPU-resident
        environments (``IsaacLabVecEnv``), no device transfer is needed —
        ``.to()`` is a no-op when source and target devices match.
        """
        print("Evaluating the policy...")
        envs = self.eval_envs
        agent.eval()
        # (B,)
        done = torch.ones(envs.env_num, dtype=torch.bool, device=agent.device)
        once_done = torch.zeros(envs.env_num, dtype=torch.bool, device=agent.device)
        steps = torch.zeros(envs.env_num, dtype=torch.int32, device=agent.device)
        returns = torch.zeros(envs.env_num, dtype=torch.float32, device=agent.device)
        log_metrics = {}
        flipped_steps = torch.zeros_like(steps)
        outcomes = torch.full_like(steps, -1)
        final_metrics = {
            field: torch.zeros(envs.env_num, dtype=torch.float32, device=agent.device)
            for field in _SURVIVAL_FINAL_FIELDS
        }
        # cache is only used for video logging / open-loop prediction.
        cache = []
        agent_state = agent.get_initial_state(envs.env_num)
        # (B, A)
        act = agent_state["prev_action"].clone()
        first_reset = True
        while not once_done.all():
            reset, stepped = _evaluation_step_masks(done, once_done)
            # Step environments.  Each env backend handles device placement
            # internally (ParallelEnv converts to CPU, IsaacLabVecEnv keeps
            # on GPU).  The .to() calls below are no-ops when the data is
            # already on agent.device.
            # (B, A), (B,)
            reset_seeds = (
                [self.eval_seed + index for index in range(envs.env_num)]
                if self.survival_eval and first_reset
                else None
            )
            trans, step_done = envs.step(act.detach(), reset, reset_seeds=reset_seeds)
            first_reset = False
            # dict of (B, 1, *)
            trans = trans.to(agent.device, non_blocking=True)
            # (B,)
            done = step_done.to(agent.device)
            steps += stepped

            # Store transition.
            # We keep the observation and the action that produced it together.
            trans["action"] = act
            if len(cache) < self.batch_length:
                cache.append(trans.clone())
            # (B, A)
            act, agent_state = agent.act(trans, agent_state, eval=True)
            returns += trans["reward"][:, 0] * stepped
            for key, value in trans.items():
                if key.startswith("log_"):
                    if self.survival_eval and key in {
                        "log_termination_reason",
                        "log_food_consumed",
                        "log_water_consumed",
                        "log_is_flipped",
                        "log_hunger",
                        "log_thirst",
                        "log_posture",
                        "log_z_pos",
                    }:
                        continue
                    if key not in log_metrics:
                        log_metrics[key] = torch.zeros_like(returns)
                    log_metrics[key] += value[:, 0] * stepped

            if self.survival_eval:
                flipped_steps += trans["log_is_flipped"][:, 0].to(torch.int32) * stepped
            reached_cutoff = (
                steps >= self.eval_max_steps
                if self.eval_max_steps > 0
                else torch.zeros_like(done)
            )
            finished = stepped & (done | reached_cutoff)
            if self.survival_eval and finished.any():
                # The Homeostatic Ant adapter has no separate environment
                # truncation: its legacy done flag is physiological failure.
                outcomes[finished & done] = 0
                outcomes[finished & ~done & reached_cutoff] = 2
                for field, key in _SURVIVAL_FINAL_FIELDS.items():
                    final_metrics[field][finished] = trans[key][:, 0][finished]
            once_done |= finished
        # dict of (B, T, *)
        cache = torch.stack(cache, dim=1) if len(cache) else None
        self.logger.scalar("episode/eval_score", returns.mean())
        self.logger.scalar("episode/eval_length", steps.to(torch.float32).mean())
        for key, value in log_metrics.items():
            if key == "log_success":
                value = torch.clip(value, max=1.0)  # make sure 1.0 for success episode
            self.logger.scalar(f"episode/eval_{key[4:]}", value.mean())
        if self.survival_eval:
            checkpoint_path = self.logdir / f"checkpoint_{train_step:012d}.pt"
            # Training saves step zero before begin(), and periodic checkpoints
            # at the end of the preceding collection iteration. If evaluation
            # is invoked manually or after an unusual resume, do not claim a
            # nonexistent checkpoint as the source of the live policy.
            checkpoint = checkpoint_path if checkpoint_path.is_file() else ""
            episode_rows = []
            for index in range(envs.env_num):
                outcome = {
                    0: "homeostatic_termination",
                    1: "environment_truncation",
                    2: "evaluation_cutoff",
                }[int(outcomes[index].item())]
                episode_rows.append(
                    {
                        "seed": self.eval_seed + index,
                        "survival_steps": int(steps[index].item()),
                        "outcome": outcome,
                        "final_state_label": int(final_metrics["final_state_label"][index].item()),
                        "food_consumed": int(final_metrics["food_consumed"][index].item()),
                        "water_consumed": int(final_metrics["water_consumed"][index].item()),
                        "flipped_steps": int(flipped_steps[index].item()),
                        "final_hunger": float(final_metrics["final_hunger"][index].item()),
                        "final_thirst": float(final_metrics["final_thirst"][index].item()),
                        "final_posture": float(final_metrics["final_posture"][index].item()),
                        "final_height": float(final_metrics["final_height"][index].item()),
                        "checkpoint_step": train_step,
                        "checkpoint": str(checkpoint),
                    }
                )
            summary = _summarize_survival_episodes(
                episode_rows,
                train_step,
                checkpoint,
                self.eval_max_steps,
                self.eval_seed,
            )
            for name in (
                "mean_survival_steps",
                "median_survival_steps",
                "min_survival_steps",
                "max_survival_steps",
                "survival_rate_to_cutoff",
                "homeostatic_terminations",
                "environment_truncations",
                "evaluation_cutoffs",
            ):
                self.logger.scalar(f"episode/eval_{name}", summary[name])
            output_dir = self.logdir / "survival_evaluation"
            _append_csv(output_dir / "survival_by_checkpoint.csv", [summary])
            _append_csv(
                output_dir / "survival_episodes_by_checkpoint.csv", episode_rows
            )
        if cache is not None and "image" in cache:
            self.logger.video("eval_video", tools.to_np(cache["image"][:1]))
        if self.video_pred_log and cache is not None:
            initial = agent.get_initial_state(1)
            self.logger.video(
                "eval_open_loop",
                tools.to_np(
                    agent.video_pred(
                        cache[:1],  # give only first batch
                        (initial["stoch"], initial["deter"]),
                    )
                ),
            )
        self.logger.write(train_step)
        agent.train()

    def begin(self, agent):
        """Main online training loop.

        For CPU-based environments the loop overlaps CPU stepping and GPU
        model execution via pinned-memory async H2D transfers.  For
        GPU-resident environments (IsaacLab) no transfer is needed —
        ``.to()`` is a no-op when the data is already on the target device.
        """
        envs = self.train_envs
        video_cache = []
        step = self.replay_buffer.count() * self._action_repeat
        next_checkpoint = (
            ((step // self.checkpoint_every) + 1) * self.checkpoint_every
            if self.checkpoint_every > 0
            else None
        )
        update_count = 0
        # (B,)
        done = torch.ones(envs.env_num, dtype=torch.bool, device=agent.device)
        returns = torch.zeros(envs.env_num, dtype=torch.float32, device=agent.device)
        lengths = torch.zeros(envs.env_num, dtype=torch.int32, device=agent.device)
        episode_ids = torch.arange(
            envs.env_num, dtype=torch.int32, device=agent.device
        )  # Kept constant so short episodes (< batch_length) remain sampable; RSSM resets via is_first.
        train_metrics = {}
        agent_state = agent.get_initial_state(envs.env_num)
        # (B, A)
        act = agent_state["prev_action"].clone()
        while step < self.steps:
            # Evaluation
            if self._should_eval(step) and self.eval_episode_num > 0 and self.eval_envs is not None:
                self.eval(agent, step)
            # Save metrics
            if done.any():
                for i, d in enumerate(done):
                    if d and lengths[i] > 0:
                        if i == 0 and len(video_cache) > 0:
                            video = torch.stack(video_cache, axis=0)
                            self.logger.video("train_video", tools.to_np(video[None]))
                            video_cache = []
                        self.logger.scalar("episode/score", returns[i])
                        self.logger.scalar("episode/length", lengths[i])
                        self.logger.write(step + i)  # to show all values on tensorboard
                        returns[i] = lengths[i] = 0
            step += int((~done).sum()) * self._action_repeat  # step is based on env side
            lengths += ~done

            # Step environments.  Each env backend handles device placement
            # internally (ParallelEnv converts to CPU, IsaacLabVecEnv keeps
            # on GPU).  The .to() calls below are no-ops when the data is
            # already on agent.device.
            # (B, A), (B,)
            trans, step_done = envs.step(act.detach(), done)
            # dict of (B, 1, *)
            trans = trans.to(agent.device, non_blocking=True)
            # (B,)
            done = step_done.to(agent.device)

            # Policy inference on GPU.
            # "agent_state" is reset by the agent based on the "is_first" flag in trans.
            # (B, A)
            act, agent_state = agent.act(trans.clone(), agent_state, eval=False)

            # Store transition.
            # We keep the observation and the action that produced it together.
            # Mask actions after an episode has ended.
            trans["action"] = act * ~done.unsqueeze(-1)
            trans["stoch"] = agent_state["stoch"]
            trans["deter"] = agent_state["deter"]
            trans["episode"] = episode_ids  # Don't lift dim
            if "image" in trans:
                video_cache.append(trans["image"][0])
            self.replay_buffer.add_transition(trans.detach())
            returns += trans["reward"][:, 0]
            # Update models after enough data has accumulated
            if step // (envs.env_num * self._action_repeat) > self.batch_length + 1:
                if self._should_pretrain():
                    update_num = self.pretrain
                else:
                    update_num = self._updates_needed(step)
                for _ in range(update_num):
                    _metrics = agent.update(self.replay_buffer)
                    train_metrics = _metrics
                update_count += update_num
                # Log training metrics
                if self._should_log(step):
                    for name, value in train_metrics.items():
                        value = tools.to_np(value) if isinstance(value, torch.Tensor) else value
                        self.logger.scalar(f"train/{name}", value)
                    self.logger.scalar("train/opt/updates", update_count)
                    if self.video_pred_log:
                        data, _, initial = self.replay_buffer.sample()
                        self.logger.video("open_loop", tools.to_np(agent.video_pred(data, initial)))
                    if self.params_hist_log:
                        for name, param in agent._named_params.items():
                            self.logger.histogram(name, tools.to_np(param))
                    self.logger.write(step, fps=True)
            if next_checkpoint is not None and step >= next_checkpoint:
                self.save_checkpoint(agent, step)
                while step >= next_checkpoint:
                    next_checkpoint += self.checkpoint_every
        if self.eval_episode_num > 0 and self.eval_envs is not None:
            self.eval(agent, step)
