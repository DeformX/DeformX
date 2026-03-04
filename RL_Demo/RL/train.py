"""
Hydra training entry.

Run from RL_Demo root:
  python -m RL.train task=franka_reach algo=ppo

Example overrides:
  python -m RL.train task=franka_reach algo=ppo task.num_envs=32 algo.rollout_len=128 render=true
"""

from __future__ import annotations

import csv
import importlib
from datetime import datetime
from collections import deque
from pathlib import Path

import numpy as np
import torch
import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf


def _load(path: str):
    mod, name = path.rsplit(".", 1)
    return getattr(importlib.import_module(mod), name)


def _as_success_mask(info, num_envs: int) -> np.ndarray:
    if not isinstance(info, dict) or "success_mask" not in info:
        return np.zeros((num_envs,), dtype=np.bool_)
    arr = np.asarray(info["success_mask"]).astype(np.bool_).reshape(-1)
    if arr.size != num_envs:
        return np.zeros((num_envs,), dtype=np.bool_)
    return arr


def _as_dist_array(info, num_envs: int) -> np.ndarray | None:
    if not isinstance(info, dict) or "dist" not in info:
        return None
    arr = np.asarray(info["dist"], dtype=np.float64).reshape(-1)
    if arr.size != num_envs:
        return None
    return arr


def _write_joint_csv(path: Path, joint_traj: list[np.ndarray], dt: float, joint_names: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["t", *joint_names])
        for i, q in enumerate(joint_traj):
            writer.writerow([float(i) * float(dt), *[float(v) for v in q]])


def _trajectory_name_prefix(min_dist: float | None, target_pos: np.ndarray) -> str:
    min_tag = "na"
    if min_dist is not None and np.isfinite(min_dist):
        min_tag = f"{float(min_dist):.3f}"

    target_arr = np.asarray(target_pos, dtype=np.float64).reshape(-1)
    if target_arr.size < 3:
        padded = np.full((3,), np.nan, dtype=np.float64)
        padded[: target_arr.size] = target_arr
        target_arr = padded

    return (
        f"min_dis_{min_tag}_"
        f"target_{float(target_arr[0]):.1f}_{float(target_arr[1]):.1f}_{float(target_arr[2]):.1f}"
    )


def _save_training_configs_to_success_dir(cfg: DictConfig, success_export_dir: Path):
    success_export_dir.mkdir(parents=True, exist_ok=True)

    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = success_export_dir / f"train_config_{run_tag}"
    suffix = 1
    while run_dir.exists():
        run_dir = success_export_dir / f"train_config_{run_tag}_{suffix:02d}"
        suffix += 1
    run_dir.mkdir(parents=True, exist_ok=True)

    OmegaConf.save(config=cfg, f=str(run_dir / "config_merged.yaml"), resolve=True)
    if hasattr(cfg, "task"):
        OmegaConf.save(config=cfg.task, f=str(run_dir / "config_task.yaml"), resolve=True)
    if hasattr(cfg, "algo"):
        OmegaConf.save(config=cfg.algo, f=str(run_dir / "config_algo.yaml"), resolve=True)

    print(f"[config-export] saved training configs to {run_dir}")


def _setup_wandb(cfg: DictConfig):
    wb_cfg = getattr(cfg, "wandb", None)
    enabled = bool(getattr(wb_cfg, "enabled", False)) if wb_cfg is not None else False
    if not enabled:
        return None

    try:
        import wandb
    except Exception as exc:
        raise ImportError(
            "wandb.enabled=true but wandb is not installed. "
            "Install with Isaac Sim python: -m pip install wandb"
        ) from exc

    project = str(getattr(wb_cfg, "project", "cosseratx-rl")).strip() or "cosseratx-rl"
    entity = str(getattr(wb_cfg, "entity", "")).strip() or None
    run_name = str(getattr(wb_cfg, "run_name", "")).strip() or None
    mode = str(getattr(wb_cfg, "mode", "online")).strip() or "online"
    notes = str(getattr(wb_cfg, "notes", "")).strip() or None
    tags_cfg = getattr(wb_cfg, "tags", [])
    tags = [str(t).strip() for t in tags_cfg if str(t).strip()]

    wandb_dir = Path(to_absolute_path(str(getattr(wb_cfg, "dir", "wandb"))))
    wandb_dir.mkdir(parents=True, exist_ok=True)

    run = wandb.init(
        project=project,
        entity=entity,
        name=run_name,
        mode=mode,
        notes=notes,
        tags=tags if len(tags) > 0 else None,
        dir=str(wandb_dir),
        config=OmegaConf.to_container(cfg, resolve=True),
    )
    print(f"[wandb] enabled project={project} run={run.name} mode={mode}")
    return run


def _export_success_episode(
    *,
    env_id: int,
    step: int,
    export_index: int,
    info,
    episode_actions: list[list[np.ndarray]],
    episode_joint_cmds: list[list[np.ndarray]],
    success_export_dir: Path,
    success_export_prefix: str,
    default_joint_names: list[str],
    min_dist: float | None,
    target_pos: np.ndarray,
):
    traj_actions = episode_actions[env_id]
    traj_joint = episode_joint_cmds[env_id]
    traj_prefix = _trajectory_name_prefix(min_dist=min_dist, target_pos=target_pos)
    file_base = f"{traj_prefix}_{success_export_prefix}_{export_index:03d}_env{env_id}_step{step}"

    actions_np = np.asarray(traj_actions, dtype=np.float32)
    actions_path = success_export_dir / f"{file_base}_actions.npz"
    np.savez_compressed(
        actions_path,
        actions=actions_np,
        step=step,
        env_id=env_id,
    )

    if len(traj_joint) > 0:
        dt = float(info.get("control_dt", 0.0)) if isinstance(info, dict) else 0.0
        if dt <= 0.0:
            dt = 1.0
        joint_names = info.get("joint_names", default_joint_names) if isinstance(info, dict) else default_joint_names
        joint_csv = success_export_dir / f"{file_base}.csv"
        _write_joint_csv(joint_csv, traj_joint, dt, list(joint_names))
        print(f"[success-export] csv={joint_csv} | actions={actions_path}")
    else:
        print(f"[success-export] actions={actions_path} (no joint trajectory in env info)")


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    # Reproducibility
    torch.manual_seed(int(cfg.seed))
    np.random.seed(int(cfg.seed))

    device = torch.device(str(cfg.device))
    wandb_run = _setup_wandb(cfg)

    # ---- Env ----
    EnvCls = _load(str(cfg.task.env_cls))
    env = EnvCls(cfg.task, headless=not bool(cfg.render))

    obs_dim = int(env.obs_dim)
    act_dim = int(env.act_dim)

    # ---- Algo ----
    from RL.algos.ppo import PPO

    algo = PPO(obs_dim=obs_dim, act_dim=act_dim, cfg=cfg.algo, device=device)

    # ---- Checkpoint/export cfg ----
    save_checkpoints = bool(getattr(cfg, "save_checkpoints", True))
    checkpoint_every = int(getattr(cfg, "checkpoint_every", 5000))
    checkpoint_dir = Path(str(getattr(cfg, "checkpoint_dir", "checkpoints")))
    save_final_checkpoint = bool(getattr(cfg, "save_final_checkpoint", True))
    resume_checkpoint = str(getattr(cfg, "resume_checkpoint", "")).strip()

    export_success_actions = bool(getattr(cfg, "export_success_actions", True))
    success_export_limit = int(getattr(cfg, "success_export_limit", 1))
    success_export_dir = Path(str(getattr(cfg, "success_export_dir", "success_trajectories")))
    success_export_prefix = str(getattr(cfg, "success_export_prefix", "success_episode"))
    save_configs_to_success_dir = bool(getattr(cfg, "save_configs_to_success_dir", True))

    if save_checkpoints:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if export_success_actions or save_configs_to_success_dir:
        success_export_dir.mkdir(parents=True, exist_ok=True)
    if save_configs_to_success_dir:
        _save_training_configs_to_success_dir(cfg=cfg, success_export_dir=success_export_dir)

    if resume_checkpoint:
        ckpt = algo.load(resume_checkpoint)
        print(f"[resume] loaded checkpoint: {resume_checkpoint} (step={ckpt.get('step')})")

    obs = env.reset()
    ep_rew_acc = 0.0

    total_steps = int(cfg.total_steps)
    log_every = int(cfg.log_every)
    rollout_len = int(cfg.algo.rollout_len)
    num_envs = int(obs.shape[0])
    log_episode_metrics = bool(getattr(cfg, "log_episode_metrics", True))
    episode_log_every = int(getattr(cfg, "episode_log_every", 1))
    episode_log_avg_window = int(getattr(cfg, "episode_log_avg_window", 10))
    if episode_log_every <= 0:
        episode_log_every = 1
    if episode_log_avg_window <= 0:
        episode_log_avg_window = 1

    # Per-env episode buffers used for success export.
    episode_actions: list[list[np.ndarray]] = [[] for _ in range(num_envs)]
    episode_joint_cmds: list[list[np.ndarray]] = [[] for _ in range(num_envs)]
    success_exported_in_episode = np.zeros((num_envs,), dtype=np.bool_)
    episode_returns = np.zeros((num_envs,), dtype=np.float64)
    episode_lengths = np.zeros((num_envs,), dtype=np.int32)
    episode_min_dist = np.full((num_envs,), np.inf, dtype=np.float64)
    n_completed_episodes = 0
    recent_returns: deque[float] = deque(maxlen=episode_log_avg_window)
    recent_lengths: deque[int] = deque(maxlen=episode_log_avg_window)
    recent_final_dist: deque[float] = deque(maxlen=episode_log_avg_window)
    recent_min_dist: deque[float] = deque(maxlen=episode_log_avg_window)
    recent_success: deque[int] = deque(maxlen=episode_log_avg_window)
    n_success_exports = 0
    default_joint_names = ["shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3"]
    target_pos = np.asarray(getattr(cfg.task, "target_local", [np.nan, np.nan, np.nan]), dtype=np.float64)

    for step in range(1, total_steps + 1):
        actions, logp = algo.act(obs)  # numpy actions, torch logp (cpu)
        for i in range(num_envs):
            episode_actions[i].append(np.asarray(actions[i], dtype=np.float32).copy())

        next_obs, rew, done, info = env.step(actions)
        rew_arr = np.asarray(rew, dtype=np.float64).reshape(-1)
        dist_arr = _as_dist_array(info, num_envs)
        if rew_arr.size == num_envs:
            episode_returns += rew_arr
            episode_lengths += 1
        if dist_arr is not None:
            episode_min_dist = np.minimum(episode_min_dist, dist_arr)

        if isinstance(info, dict) and "commanded_joint_positions" in info:
            q_cmd = np.asarray(info["commanded_joint_positions"], dtype=np.float64)
            if q_cmd.shape[0] == num_envs:
                for i in range(num_envs):
                    episode_joint_cmds[i].append(q_cmd[i].copy())

        algo.store(obs, actions, logp, rew, done)
        obs = next_obs

        ep_rew_acc += float(np.mean(rew))

        done_mask = np.asarray(done, dtype=np.bool_).reshape(-1)
        done_ids = np.where(done_mask)[0].astype(np.int32)
        success_mask = _as_success_mask(info, num_envs)

        # Export immediately on first goal hit in each episode (even if env is not done yet).
        first_success_ids = np.where(success_mask & ~success_exported_in_episode)[0].astype(np.int32)
        for env_id in first_success_ids.tolist():
            if export_success_actions and n_success_exports < success_export_limit:
                min_dist_for_export = None
                if np.isfinite(episode_min_dist[env_id]):
                    min_dist_for_export = float(episode_min_dist[env_id])
                _export_success_episode(
                    env_id=env_id,
                    step=step,
                    export_index=n_success_exports,
                    info=info,
                    episode_actions=episode_actions,
                    episode_joint_cmds=episode_joint_cmds,
                    success_export_dir=success_export_dir,
                    success_export_prefix=success_export_prefix,
                    default_joint_names=default_joint_names,
                    min_dist=min_dist_for_export,
                    target_pos=target_pos,
                )
                n_success_exports += 1
            success_exported_in_episode[env_id] = True

        for env_id in done_ids.tolist():
            is_success = bool(success_mask[env_id])
            n_completed_episodes += 1
            final_dist = float("nan")
            if dist_arr is not None:
                final_dist = float(dist_arr[env_id])
            min_dist = final_dist
            if np.isfinite(episode_min_dist[env_id]):
                min_dist = float(episode_min_dist[env_id])
            ep_ret = float(episode_returns[env_id])
            ep_len = int(episode_lengths[env_id])

            recent_returns.append(ep_ret)
            recent_lengths.append(ep_len)
            recent_final_dist.append(final_dist)
            recent_min_dist.append(min_dist)
            recent_success.append(1 if is_success else 0)

            if log_episode_metrics:
                print(
                    f"[episode {n_completed_episodes}] env={env_id} "
                    f"ret={ep_ret:.3f} len={ep_len} final_dist={final_dist:.3f} "
                    f"min_dist={min_dist:.3f} success={int(is_success)}"
                )
            if wandb_run is not None:
                wandb_run.log(
                    {
                        "train/step": int(step),
                        "episode/index": int(n_completed_episodes),
                        "episode/env_id": int(env_id),
                        "episode/return": float(ep_ret),
                        "episode/len": int(ep_len),
                        "episode/final_dist": float(final_dist),
                        "episode/min_dist": float(min_dist),
                        "episode/success": int(is_success),
                    },
                    step=int(step),
                )
            if log_episode_metrics and n_completed_episodes % episode_log_every == 0:
                k = len(recent_returns)
                avg_ret = float(np.mean(np.asarray(recent_returns, dtype=np.float64))) if k > 0 else 0.0
                avg_len = float(np.mean(np.asarray(recent_lengths, dtype=np.float64))) if k > 0 else 0.0
                avg_final = float(np.nanmean(np.asarray(recent_final_dist, dtype=np.float64))) if k > 0 else float("nan")
                avg_min = float(np.nanmean(np.asarray(recent_min_dist, dtype=np.float64))) if k > 0 else float("nan")
                avg_success = float(np.mean(np.asarray(recent_success, dtype=np.float64))) if k > 0 else 0.0
                print(
                    f"[episode-summary {n_completed_episodes}] window={k} "
                    f"avg_ret={avg_ret:.3f} avg_len={avg_len:.1f} "
                    f"avg_final_dist={avg_final:.3f} avg_min_dist={avg_min:.3f} "
                    f"success_rate={avg_success:.3f}"
                )
                if wandb_run is not None:
                    wandb_run.log(
                        {
                            "train/step": int(step),
                            "episode_summary/window": int(k),
                            "episode_summary/avg_ret": float(avg_ret),
                            "episode_summary/avg_len": float(avg_len),
                            "episode_summary/avg_final_dist": float(avg_final),
                            "episode_summary/avg_min_dist": float(avg_min),
                            "episode_summary/success_rate": float(avg_success),
                        },
                        step=int(step),
                    )

            if (
                is_success
                and (not bool(success_exported_in_episode[env_id]))
                and export_success_actions
                and n_success_exports < success_export_limit
            ):
                _export_success_episode(
                    env_id=env_id,
                    step=step,
                    export_index=n_success_exports,
                    info=info,
                    episode_actions=episode_actions,
                    episode_joint_cmds=episode_joint_cmds,
                    success_export_dir=success_export_dir,
                    success_export_prefix=success_export_prefix,
                    default_joint_names=default_joint_names,
                    min_dist=min_dist,
                    target_pos=target_pos,
                )
                n_success_exports += 1

            # done envs are auto-reset by env; start new episode buffers
            episode_actions[env_id] = []
            episode_joint_cmds[env_id] = []
            success_exported_in_episode[env_id] = False
            episode_returns[env_id] = 0.0
            episode_lengths[env_id] = 0
            episode_min_dist[env_id] = np.inf

        if step % rollout_len == 0:
            algo.update()

        if save_checkpoints and checkpoint_every > 0 and step % checkpoint_every == 0:
            ckpt_path = checkpoint_dir / f"ppo_step_{step}.pt"
            algo.save(
                str(ckpt_path),
                step=step,
                extra={"task": str(cfg.task.name), "seed": int(cfg.seed)},
            )
            print(f"[checkpoint] saved {ckpt_path}")
            if wandb_run is not None:
                wandb_run.log(
                    {
                        "train/step": int(step),
                        "train/checkpoint_step": int(step),
                    },
                    step=int(step),
                )

        if step % log_every == 0:
            extra = ""
            wandb_payload = {
                "train/step": int(step),
                "train/avg_rew": float(ep_rew_acc / log_every),
            }
            if isinstance(info, dict) and "success_rate" in info:
                extra = f" | success={info['success_rate']:.3f}"
                wandb_payload["train/success_rate"] = float(info["success_rate"])
            if isinstance(info, dict) and "dist_mean" in info:
                extra += f" | dist={info['dist_mean']:.3f}"
                wandb_payload["train/dist_mean"] = float(info["dist_mean"])
            if isinstance(info, dict) and "r_progress_mean" in info:
                extra += f" | r_prog={info['r_progress_mean']:.3f}"
                wandb_payload["reward/r_progress_mean"] = float(info["r_progress_mean"])
            if isinstance(info, dict) and "r_proximity_mean" in info:
                extra += f" | r_prox={info['r_proximity_mean']:.3f}"
                wandb_payload["reward/r_proximity_mean"] = float(info["r_proximity_mean"])
            if isinstance(info, dict) and "r_joint3_velocity_mean" in info:
                extra += f" | r_j3v={info['r_joint3_velocity_mean']:.3f}"
                wandb_payload["reward/r_joint3_velocity_mean"] = float(info["r_joint3_velocity_mean"])
            if isinstance(info, dict) and "joint3_vel_mean" in info:
                extra += f" | j3v={info['joint3_vel_mean']:.3f}"
                wandb_payload["train/joint3_vel_mean"] = float(info["joint3_vel_mean"])
            print(f"[step {step}] avg_rew={ep_rew_acc/log_every:.3f}{extra}")
            if wandb_run is not None:
                wandb_run.log(wandb_payload, step=int(step))
            ep_rew_acc = 0.0

    if save_checkpoints and save_final_checkpoint:
        final_path = checkpoint_dir / "ppo_final.pt"
        algo.save(
            str(final_path),
            step=total_steps,
            extra={"task": str(cfg.task.name), "seed": int(cfg.seed)},
        )
        print(f"[checkpoint] saved {final_path}")

    # If your env owns SimulationApp, it should also close it.
    env.close()
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
