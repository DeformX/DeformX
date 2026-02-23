"""
Hydra evaluation/export entry.

Run from RL_Demo root:
  python -m RL.eval task=wire_swing algo=ppo eval.checkpoint=/abs/path/to/ppo_step_50000.pt
"""

from __future__ import annotations

import csv
import importlib
import json
from pathlib import Path

import hydra
import numpy as np
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig


CANONICAL_JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow",
    "wrist_1",
    "wrist_2",
    "wrist_3",
]


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


def _policy_actions(algo, obs: np.ndarray, deterministic: bool) -> np.ndarray:
    if deterministic:
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=algo.device)
            actions = algo.actor(obs_t).detach().cpu().numpy()
    else:
        actions, _ = algo.act(obs)
    return np.clip(np.asarray(actions, dtype=np.float32), -1.0, 1.0)


def _write_joint_csv(path: Path, times: np.ndarray, joints: np.ndarray):
    if joints.ndim != 2 or joints.shape[1] != len(CANONICAL_JOINT_NAMES):
        raise ValueError(
            "Joint trajectory must have shape (T, 6) for UR replay export. "
            f"Got shape {joints.shape}."
        )
    if times.ndim != 1 or times.shape[0] != joints.shape[0]:
        raise ValueError(f"Bad time/joint shape: times={times.shape}, joints={joints.shape}")

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["t", *CANONICAL_JOINT_NAMES])
        for i in range(joints.shape[0]):
            writer.writerow([float(times[i]), *[float(v) for v in joints[i]]])


def _resample_joints(times: np.ndarray, joints: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    if dt <= 0.0 or times.shape[0] <= 1:
        return times, joints

    t0 = float(times[0])
    t1 = float(times[-1])
    if t1 <= t0:
        return times, joints

    t_new = np.arange(t0, t1 + dt * 0.5, dt, dtype=np.float64)
    if t_new.shape[0] == 0:
        return times, joints
    if t_new[-1] < t1:
        t_new = np.concatenate([t_new, np.asarray([t1], dtype=np.float64)], axis=0)

    out = np.zeros((t_new.shape[0], joints.shape[1]), dtype=np.float64)
    for j in range(joints.shape[1]):
        out[:, j] = np.interp(t_new, times, joints[:, j])
    return t_new, out


def _select_episode(episodes: list[dict], mode: str) -> dict:
    if len(episodes) == 0:
        raise ValueError("No episodes available to select from.")

    successful = [e for e in episodes if bool(e["success"])]

    if mode == "first_success":
        return successful[0] if successful else episodes[0]
    if mode == "best_success":
        pool = successful if len(successful) > 0 else episodes
        return max(pool, key=lambda e: (float(e["return"]), -int(e["episode_index"])))
    if mode == "best_return":
        return max(episodes, key=lambda e: (float(e["return"]), -int(e["episode_index"])))
    if mode == "best_min_dist":
        valid = [e for e in episodes if e["min_dist"] is not None]
        if len(valid) == 0:
            return max(episodes, key=lambda e: (float(e["return"]), -int(e["episode_index"])))
        return min(valid, key=lambda e: (float(e["min_dist"]), int(e["episode_index"])))
    if mode == "latest":
        return episodes[-1]

    raise ValueError(
        f"Unsupported eval.export_mode='{mode}'. "
        "Use one of: first_success, best_success, best_return, best_min_dist, latest."
    )


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    torch.manual_seed(int(cfg.seed))
    np.random.seed(int(cfg.seed))

    device = torch.device(str(cfg.device))

    env = None
    try:
        EnvCls = _load(str(cfg.task.env_cls))
        env = EnvCls(cfg.task, headless=not bool(cfg.render))

        obs_dim = int(env.obs_dim)
        act_dim = int(env.act_dim)

        from RL.algos.ppo import PPO

        algo = PPO(obs_dim=obs_dim, act_dim=act_dim, cfg=cfg.algo, device=device)

        ckpt_path = str(getattr(cfg.eval, "checkpoint", "")).strip()
        if not ckpt_path:
            ckpt_path = str(getattr(cfg, "resume_checkpoint", "")).strip()
        if not ckpt_path:
            raise ValueError("No checkpoint provided. Set eval.checkpoint=/path/to/ppo_step_x.pt")
        ckpt_path = to_absolute_path(ckpt_path)
        if not Path(ckpt_path).is_file():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        ckpt = algo.load(ckpt_path)
        print(f"[eval] loaded checkpoint: {ckpt_path} (step={ckpt.get('step')})")

        deterministic = bool(getattr(cfg.eval, "deterministic", True))
        target_episodes = int(getattr(cfg.eval, "episodes", 5))
        if target_episodes <= 0:
            raise ValueError(f"eval.episodes must be >= 1, got {target_episodes}")

        default_max_steps = int(getattr(cfg.task, "max_steps", 300))
        max_total_steps = int(getattr(cfg.eval, "max_total_steps", 0))
        if max_total_steps <= 0:
            max_total_steps = max(target_episodes * max(default_max_steps, 1) * 4, 1000)

        export_dir = Path(to_absolute_path(str(getattr(cfg.eval, "export_dir", "eval_exports"))))
        export_prefix = str(getattr(cfg.eval, "export_prefix", "eval")).strip() or "eval"
        export_mode = str(getattr(cfg.eval, "export_mode", "best_success")).strip()
        export_joint_csv = bool(getattr(cfg.eval, "export_joint_csv", True))
        export_actions_npz = bool(getattr(cfg.eval, "export_actions_npz", True))
        export_episode_summary = bool(getattr(cfg.eval, "export_episode_summary", True))
        export_summary_json = bool(getattr(cfg.eval, "export_summary_json", True))
        resample_dt = float(getattr(cfg.eval, "resample_dt", 0.0))

        obs = env.reset()
        num_envs = int(obs.shape[0])

        ep_returns = np.zeros((num_envs,), dtype=np.float64)
        ep_steps = np.zeros((num_envs,), dtype=np.int32)
        ep_min_dist = np.full((num_envs,), np.inf, dtype=np.float64)
        ep_control_dt = np.full(
            (num_envs,),
            float(getattr(cfg.task, "phys_dt", 1.0)) * float(getattr(cfg.task, "num_substeps", 1)),
            dtype=np.float64,
        )
        ep_actions: list[list[np.ndarray]] = [[] for _ in range(num_envs)]
        ep_joint_cmds: list[list[np.ndarray]] = [[] for _ in range(num_envs)]

        episodes: list[dict] = []
        global_steps = 0

        while len(episodes) < target_episodes and global_steps < max_total_steps:
            actions = _policy_actions(algo, obs, deterministic=deterministic)
            next_obs, rew, done, info = env.step(actions)

            rew = np.asarray(rew, dtype=np.float64).reshape(-1)
            done = np.asarray(done, dtype=np.bool_).reshape(-1)
            if rew.size != num_envs or done.size != num_envs:
                raise RuntimeError(
                    f"Env returned inconsistent batch shapes: rew={rew.shape}, done={done.shape}, num_envs={num_envs}"
                )

            success_mask = _as_success_mask(info, num_envs)
            dist_arr = _as_dist_array(info, num_envs)

            q_cmd = None
            if isinstance(info, dict) and "commanded_joint_positions" in info:
                q_cmd = np.asarray(info["commanded_joint_positions"], dtype=np.float64)
                if q_cmd.ndim != 2 or q_cmd.shape[0] != num_envs:
                    q_cmd = None

            info_control_dt = None
            if isinstance(info, dict) and "control_dt" in info:
                try:
                    info_control_dt = float(info["control_dt"])
                except Exception:
                    info_control_dt = None

            for i in range(num_envs):
                ep_returns[i] += float(rew[i])
                ep_steps[i] += 1
                ep_actions[i].append(np.asarray(actions[i], dtype=np.float32).copy())
                if q_cmd is not None:
                    ep_joint_cmds[i].append(q_cmd[i].copy())
                if info_control_dt is not None and info_control_dt > 0.0:
                    ep_control_dt[i] = info_control_dt
                if dist_arr is not None:
                    ep_min_dist[i] = min(ep_min_dist[i], float(dist_arr[i]))

            done_ids = np.where(done)[0].astype(np.int32)
            for env_id in done_ids.tolist():
                ep_index = len(episodes)
                joint_traj = np.asarray(ep_joint_cmds[env_id], dtype=np.float64)
                action_traj = np.asarray(ep_actions[env_id], dtype=np.float32)
                min_dist = None
                if np.isfinite(ep_min_dist[env_id]):
                    min_dist = float(ep_min_dist[env_id])

                episode = {
                    "episode_index": ep_index,
                    "env_id": int(env_id),
                    "success": bool(success_mask[env_id]),
                    "return": float(ep_returns[env_id]),
                    "steps": int(ep_steps[env_id]),
                    "min_dist": min_dist,
                    "control_dt": float(ep_control_dt[env_id]),
                    "joint_traj": joint_traj,
                    "action_traj": action_traj,
                }
                episodes.append(episode)
                print(
                    f"[eval] episode={ep_index:03d} env={env_id} "
                    f"success={episode['success']} return={episode['return']:.3f} steps={episode['steps']}"
                )

                ep_returns[env_id] = 0.0
                ep_steps[env_id] = 0
                ep_min_dist[env_id] = np.inf
                ep_actions[env_id] = []
                ep_joint_cmds[env_id] = []

                if len(episodes) >= target_episodes:
                    break

            obs = next_obs
            global_steps += 1

        if len(episodes) == 0:
            raise RuntimeError("Evaluation finished without any completed episode.")
        if len(episodes) < target_episodes:
            print(
                f"[eval] warning: collected only {len(episodes)} / {target_episodes} episodes "
                f"before max_total_steps={max_total_steps}."
            )

        selected = _select_episode(episodes, mode=export_mode)
        success_rate = float(np.mean([float(e["success"]) for e in episodes]))
        mean_return = float(np.mean([float(e["return"]) for e in episodes]))

        export_dir.mkdir(parents=True, exist_ok=True)
        base = f"{export_prefix}_{export_mode}_ep{int(selected['episode_index']):03d}"

        summary_rows = [
            {
                "episode_index": int(e["episode_index"]),
                "env_id": int(e["env_id"]),
                "success": int(bool(e["success"])),
                "return": float(e["return"]),
                "steps": int(e["steps"]),
                "min_dist": "" if e["min_dist"] is None else float(e["min_dist"]),
                "control_dt": float(e["control_dt"]),
                "joint_samples": int(e["joint_traj"].shape[0]),
                "action_samples": int(e["action_traj"].shape[0]),
            }
            for e in episodes
        ]

        if export_episode_summary:
            summary_csv = export_dir / f"{export_prefix}_episode_summary.csv"
            with summary_csv.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
                writer.writeheader()
                writer.writerows(summary_rows)
            print(f"[eval-export] summary_csv={summary_csv}")

        if export_summary_json:
            summary_json = export_dir / f"{export_prefix}_summary.json"
            payload = {
                "checkpoint": str(ckpt_path),
                "checkpoint_step": ckpt.get("step"),
                "task": str(cfg.task.name),
                "algo": str(cfg.algo.name),
                "deterministic": deterministic,
                "requested_episodes": target_episodes,
                "collected_episodes": len(episodes),
                "global_steps": int(global_steps),
                "success_rate": success_rate,
                "mean_return": mean_return,
                "selected_episode_index": int(selected["episode_index"]),
                "selected_mode": export_mode,
                "episodes": summary_rows,
            }
            summary_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            print(f"[eval-export] summary_json={summary_json}")

        selected_joint_traj = np.asarray(selected["joint_traj"], dtype=np.float64)
        selected_action_traj = np.asarray(selected["action_traj"], dtype=np.float32)
        selected_dt = float(selected["control_dt"])
        if selected_dt <= 0.0:
            selected_dt = float(getattr(cfg.task, "phys_dt", 1.0)) * float(getattr(cfg.task, "num_substeps", 1))

        export_times = np.arange(selected_joint_traj.shape[0], dtype=np.float64) * selected_dt
        csv_times = export_times
        csv_joints = selected_joint_traj
        if selected_joint_traj.shape[0] > 0 and resample_dt > 0.0:
            csv_times, csv_joints = _resample_joints(export_times, selected_joint_traj, dt=resample_dt)

        if export_joint_csv:
            if selected_joint_traj.shape[0] == 0:
                print("[eval-export] warning: selected episode has no commanded_joint_positions; skip CSV export.")
            else:
                out_csv = export_dir / f"{base}.csv"
                _write_joint_csv(out_csv, csv_times, csv_joints)
                print(f"[eval-export] joint_csv={out_csv}")

        if export_actions_npz:
            out_npz = export_dir / f"{base}.npz"
            np.savez_compressed(
                out_npz,
                actions=selected_action_traj,
                joint_positions=selected_joint_traj,
                t=export_times,
                exported_joint_positions=csv_joints,
                exported_t=csv_times,
                control_dt=selected_dt,
                resample_dt=resample_dt,
                success=bool(selected["success"]),
                episode_return=float(selected["return"]),
                episode_steps=int(selected["steps"]),
            )
            print(f"[eval-export] npz={out_npz}")

        print(
            f"[eval] done: episodes={len(episodes)} success_rate={success_rate:.3f} "
            f"mean_return={mean_return:.3f} selected={selected['episode_index']} ({export_mode})"
        )
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()
