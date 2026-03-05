#!/usr/bin/env python3
"""Replay a joint trajectory CSV with the Ball-Joint wire environment backend.

Usage:
  /home/robot/isaacsim/python.sh RL_Demo/tools/replay_wire_traj_bj.py /abs/path/to/traj.csv
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import numpy.testing  # Keep numpy.testing bound before Kit mutates import paths.

from replay_wire_traj import JOINT_NAMES, NUM_ROBOT_DOFS, load_joint_csv


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[2]
RL_DEMO_ROOT = SCRIPT_PATH.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(RL_DEMO_ROOT) not in sys.path:
    sys.path.insert(0, str(RL_DEMO_ROOT))
PYELASTICA_MESH_ROOT = REPO_ROOT / "PyElastica-Mesh"
if PYELASTICA_MESH_ROOT.is_dir() and str(PYELASTICA_MESH_ROOT) not in sys.path:
    sys.path.insert(0, str(PYELASTICA_MESH_ROOT))

from RL.envs.wire_swing_bj_env import WireSwingBallJointEnv


DEFAULT_PHYS_DT = 1.0 / 500.0


def _derive_sim_dt_from_csv(times: np.ndarray, default_dt: float) -> float:
    dts = np.diff(times)
    dts = dts[np.isfinite(dts) & (dts > 1.0e-9)]
    if dts.size == 0:
        return float(default_dt)
    return float(np.median(dts))


def _load_task_cfg(task_config_path: Path | None) -> dict[str, object]:
    if task_config_path is None:
        return {}
    p = task_config_path.expanduser().resolve()
    if not p.is_file():
        raise FileNotFoundError(f"Task config not found: {p}")
    try:
        import yaml
    except Exception as exc:
        raise ImportError("PyYAML is required for --task-config.") from exc
    with p.open("r") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise RuntimeError(f"Invalid YAML structure in {p}")
    task = data.get("task", data)
    if not isinstance(task, dict):
        raise RuntimeError(f"Invalid task section in {p}")
    return dict(task)


def _auto_find_task_config(path_traj: Path) -> Path | None:
    root = path_traj.parent
    candidates = [
        p / "config_merged.yaml"
        for p in root.glob("train_config_*")
        if (p / "config_merged.yaml").is_file()
    ]
    if len(candidates) == 0:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def _infer_env_id_from_name(path_traj: Path) -> int | None:
    m = re.search(r"_env(\d+)_", path_traj.name)
    if m is None:
        return None
    return int(m.group(1))


def _build_replay_cfg(
    *,
    sim_dt: float,
    num_envs: int,
    target_local: np.ndarray,
    wire_n_elem: int,
    wire_radius: float,
    bj_joint_damping: float,
    bj_attach_stiffness: float,
    bj_attach_damping: float,
    bj_attach_rot_stiffness: float,
    bj_attach_rot_damping: float,
    bj_link_linear_damping: float,
    bj_link_angular_damping: float,
    task_overrides: dict[str, object] | None = None,
) -> SimpleNamespace:
    cfg_dict: dict[str, object] = dict(
        name="wire_swing_bj_replay",
        num_envs=int(num_envs),
        env_spacing=3.0,
        phys_dt=float(sim_dt),
        num_substeps=1,
        max_steps=500,
        init_warmup_steps=20,
        num_robot_dofs=6,
        end_effector_link="wrist_3_link",
        active_joints=[1, 2, 3],
        default_joint_positions=[-1.5935, -2.2756, -1.1913, -0.3241, 1.5707, 3.1415],
        robot_offset=[0.0, 0.0, 1.7],
        robot_orient_xyz_deg=[-90.0, 0.0, 0.0],
        target_local=np.asarray(target_local, dtype=np.float64).tolist(),
        stick_length=0.65,
        stick_radius=0.010,
        wire_base_length=1.0,
        wire_n_elem=int(max(1, wire_n_elem)),
        wire_base_radius=float(wire_radius),
        bj_joint_damping=float(bj_joint_damping),
        bj_joint_stiffness=0.0,
        bj_attach_stiffness=float(bj_attach_stiffness),
        bj_attach_damping=float(bj_attach_damping),
        bj_attach_rot_stiffness=float(bj_attach_rot_stiffness),
        bj_attach_rot_damping=float(bj_attach_rot_damping),
        bj_joint_drive_type="force",
        bj_link_linear_damping=float(bj_link_linear_damping),
        bj_link_angular_damping=float(bj_link_angular_damping),
        bj_mat_static_friction=0.6,
        bj_mat_dynamic_friction=0.5,
        bj_mat_restitution=0.0,
        bj_collision_rest_offset=0.0,
        bj_collision_contact_offset=0.002,
        bj_link_density=600.0,
        joint_pos_lower_limits=[-6.28, -6.28, -6.28, -6.28, -6.28, -6.28],
        joint_pos_upper_limits=[6.28, -1.92, -0.44, 6.28, 6.28, 6.28],
        joint_vel_limits=[1.5708, 1.5708, 1.5708, 1.5708, 1.5708, 1.5708],
        joint_delta_scale=3.0,
        positive_only_active_joints=True,
        success_thresh=0.05,
        touch_thresh=0.03,
        too_far_thresh=4.0,
        reward_clip_min=-10.0,
        reward_clip_max=200.0,
        action_penalty=0.05,
        smooth_penalty=0.02,
        w_proximity=1.0,
        w_progress=10.0,
        w_new_best=30.0,
        w_tip_velocity_toward_target=1.0,
        w_joint3_velocity=0.5,
        joint3_reward_index=3,
        joint3_velocity_abs=False,
        w_time_penalty=0.05,
        success_bonus=50.0,
        bonus_thresh_1=0.5,
        bonus_thresh_2=0.2,
        bonus_thresh_3=0.1,
        bonus_thresh_4=0.05,
        bonus_value_1=1.0,
        bonus_value_2=3.0,
        bonus_value_3=5.0,
        bonus_value_4=10.0,
        touch_bonus=50.0,
        swing_done_enabled=False,
        swing_grace_steps=300,
        swing_fallback_steps=400,
        terminate_on_success=True,
        enable_wire_visual=True,
        wire_visual_mode="debug_spheres",
        max_visual_envs=1,
        max_visual_nodes=21,
        wire_debug_sphere_radius=0.015,
        wire_debug_sphere_color=[1.0, 0.0, 0.0],
        wire_usd=(
            "/home/robot/Workspace/Siemens_Cable_Simulator/usd/"
            "wire_usdc/wire_usdc/wire_yellow_s20_r0.005_l1.usdc"
        ),
        sync_reset_ratio=1.0,
    )
    if task_overrides:
        cfg_dict.update(task_overrides)
    # Force replay/runtime controls.
    cfg_dict["num_envs"] = int(num_envs)
    cfg_dict["phys_dt"] = float(sim_dt)
    cfg_dict["target_local"] = np.asarray(target_local, dtype=np.float64).tolist()
    cfg_dict["wire_n_elem"] = int(max(1, wire_n_elem))
    cfg_dict["wire_base_radius"] = float(wire_radius)
    cfg_dict["bj_joint_damping"] = float(bj_joint_damping)
    cfg_dict["bj_attach_stiffness"] = float(bj_attach_stiffness)
    cfg_dict["bj_attach_damping"] = float(bj_attach_damping)
    cfg_dict["bj_attach_rot_stiffness"] = float(bj_attach_rot_stiffness)
    cfg_dict["bj_attach_rot_damping"] = float(bj_attach_rot_damping)
    cfg_dict["bj_link_linear_damping"] = float(bj_link_linear_damping)
    cfg_dict["bj_link_angular_damping"] = float(bj_link_angular_damping)
    # Replay runs with only one env trajectory (others are unknown), so disable
    # batched sync-reset trigger to avoid cross-env reset interference.
    cfg_dict["sync_reset_ratio"] = 2.0
    return SimpleNamespace(**cfg_dict)


def _set_robot_target(env: WireSwingBallJointEnv, joint_cmd: np.ndarray, env_idx: int) -> None:
    cmd = np.zeros((1, int(env.num_dof)), dtype=np.float32)
    cmd[0, :NUM_ROBOT_DOFS] = np.asarray(joint_cmd, dtype=np.float32)
    env.robot_view.set_joint_position_targets(cmd, indices=np.asarray([env_idx], dtype=np.int32))
    env.commanded_q[env_idx, :NUM_ROBOT_DOFS] = cmd[0, :NUM_ROBOT_DOFS]


def _reconstruct_actions_from_joint_csv(
    env: WireSwingBallJointEnv, joint_positions: np.ndarray
) -> np.ndarray:
    q_cmd = np.asarray(joint_positions, dtype=np.float32)
    if q_cmd.ndim != 2 or q_cmd.shape[1] != NUM_ROBOT_DOFS:
        raise ValueError(f"Expected joint_positions shape (T, {NUM_ROBOT_DOFS}), got {q_cmd.shape}")

    active = np.asarray(env.active_joints, dtype=np.int64)
    if active.size != int(env.act_dim):
        raise RuntimeError(
            f"active_joints size {active.size} != env.act_dim {int(env.act_dim)}"
        )

    denom = np.asarray(env.joint_max_delta[active], dtype=np.float32)
    denom = np.where(np.abs(denom) > 1.0e-9, denom, np.ones_like(denom))

    prev = np.asarray(env.default_joint_positions, dtype=np.float32)[active].copy()
    actions = np.zeros((q_cmd.shape[0], int(env.act_dim)), dtype=np.float32)
    for i in range(q_cmd.shape[0]):
        nxt = q_cmd[i, active]
        a = (nxt - prev) / denom
        if bool(env.positive_only_active_joints):
            a = np.clip(a, 0.0, 1.0)
        else:
            a = np.clip(a, -1.0, 1.0)
        actions[i] = a
        prev = nxt.copy()
    return actions


def _load_actions_npz_for_traj(path_traj: Path) -> np.ndarray | None:
    npz_path = path_traj.with_name(f"{path_traj.stem}_actions.npz")
    if not npz_path.is_file():
        return None
    data = np.load(npz_path)
    if "actions" not in data:
        return None
    actions = np.asarray(data["actions"], dtype=np.float32)
    if actions.ndim != 2:
        return None
    return actions


def replay_bj(
    path_traj: Path,
    *,
    headless: bool,
    settle_seconds: float,
    target_local: np.ndarray,
    wire_n_elem: int,
    wire_radius: float,
    bj_joint_damping: float,
    bj_attach_stiffness: float,
    bj_attach_damping: float,
    bj_attach_rot_stiffness: float,
    bj_attach_rot_damping: float,
    bj_link_linear_damping: float,
    bj_link_angular_damping: float,
    task_overrides: dict[str, object] | None,
    num_envs: int,
    replay_env_id: int,
    ignore_actions_npz: bool,
) -> tuple[Path, Path, Path]:
    times, joint_positions = load_joint_csv(path_traj)
    sim_dt_csv = _derive_sim_dt_from_csv(times, DEFAULT_PHYS_DT)
    cfg_dt = None
    if task_overrides is not None and "phys_dt" in task_overrides:
        try:
            cfg_dt = float(task_overrides["phys_dt"])
        except Exception:
            cfg_dt = None
    sim_dt = float(cfg_dt) if cfg_dt is not None else float(sim_dt_csv)
    csv_dts = np.diff(times)
    valid_csv_dts = csv_dts[np.isfinite(csv_dts) & (csv_dts > 1.0e-9)]
    median_csv_dt = float(np.median(valid_csv_dts)) if valid_csv_dts.size > 0 else float(sim_dt_csv)
    print(
        f"[replay-bj] loaded {times.shape[0]} frames | "
        f"median csv dt={median_csv_dt:.6f}s | "
        f"sim_dt={sim_dt:.6f}s | "
        "control_dt=sim_dt (1 command per sim step)"
    )
    if abs(float(sim_dt) - float(median_csv_dt)) > 1.0e-9:
        print(
            "[replay-bj] note: sim_dt differs from csv median dt. "
            f"csv={median_csv_dt:.6f}s, sim={sim_dt:.6f}s"
        )

    cfg = _build_replay_cfg(
        sim_dt=float(sim_dt),
        num_envs=int(num_envs),
        target_local=np.asarray(target_local, dtype=np.float64),
        wire_n_elem=int(wire_n_elem),
        wire_radius=float(wire_radius),
        bj_joint_damping=float(bj_joint_damping),
        bj_attach_stiffness=float(bj_attach_stiffness),
        bj_attach_damping=float(bj_attach_damping),
        bj_attach_rot_stiffness=float(bj_attach_rot_stiffness),
        bj_attach_rot_damping=float(bj_attach_rot_damping),
        bj_link_linear_damping=float(bj_link_linear_damping),
        bj_link_angular_damping=float(bj_link_angular_damping),
        task_overrides=task_overrides,
    )

    env: WireSwingBallJointEnv | None = None
    try:
        env = WireSwingBallJointEnv(cfg, headless=bool(headless))
        print("[replay-bj] WireSwingBallJointEnv initialized", flush=True)
        if replay_env_id < 0 or replay_env_id >= int(env.num_envs):
            raise ValueError(
                f"replay_env_id out of range [0, {int(env.num_envs) - 1}]: {replay_env_id}"
            )

        # Match training startup: constructor performs one reset, then RL.train calls env.reset() once.
        env.reset()
        print(f"[replay-bj] replay env index={replay_env_id} / num_envs={int(env.num_envs)}", flush=True)
        actions_replay: np.ndarray | None = None
        if not bool(ignore_actions_npz):
            actions_npz = _load_actions_npz_for_traj(path_traj)
            if actions_npz is not None and actions_npz.shape[1] == int(env.act_dim):
                if actions_npz.shape[0] == joint_positions.shape[0]:
                    actions_replay = actions_npz.astype(np.float32)
                    print(
                        f"[replay-bj] using actions from npz: "
                        f"{path_traj.with_name(f'{path_traj.stem}_actions.npz')}",
                        flush=True,
                    )
                else:
                    print(
                        "[replay-bj] actions npz length mismatch; fallback to reconstructed actions. "
                        f"npz={actions_npz.shape[0]}, csv={joint_positions.shape[0]}",
                        flush=True,
                    )
        if actions_replay is None:
            actions_replay = _reconstruct_actions_from_joint_csv(env, joint_positions)
            print("[replay-bj] using actions reconstructed from joint CSV", flush=True)

        control_dt = float(max(env.control_dt, 1.0e-8))
        sim_t: list[float] = []
        tip_trace: list[np.ndarray] = []
        joint_trace: list[np.ndarray] = []

        for frame_idx in range(times.shape[0]):
            act = np.zeros((int(env.num_envs), int(env.act_dim)), dtype=np.float32)
            act[replay_env_id] = actions_replay[frame_idx]
            env.step(act)

            tip = np.asarray(env.tip_world[replay_env_id], dtype=np.float64)
            if not np.all(np.isfinite(tip)):
                tip = tip_trace[-1].copy() if len(tip_trace) > 0 else np.zeros(3, dtype=np.float64)
            q_now = np.asarray(env.robot_view.get_joint_positions(), dtype=np.float64)[
                replay_env_id, :NUM_ROBOT_DOFS
            ]
            if not np.all(np.isfinite(q_now)):
                q_now = (
                    joint_trace[-1].copy()
                    if len(joint_trace) > 0
                    else np.asarray(joint_positions[frame_idx], dtype=np.float64)
                )

            sim_t.append(float(times[frame_idx]))
            tip_trace.append(tip.copy())
            joint_trace.append(q_now.copy())

            if frame_idx % 100 == 0:
                print(f"[replay-bj] frame {frame_idx + 1}/{times.shape[0]}", flush=True)

        settle_steps = max(0, int(round(float(settle_seconds) / control_dt)))
        print(
            f"[replay-bj] holding final pose for {float(settle_seconds):.2f}s "
            f"({settle_steps} steps)",
            flush=True,
        )
        for _ in range(settle_steps):
            act = np.zeros((int(env.num_envs), int(env.act_dim)), dtype=np.float32)
            env.step(act)
            tip = np.asarray(env.tip_world[replay_env_id], dtype=np.float64)
            if not np.all(np.isfinite(tip)):
                tip = tip_trace[-1].copy()
            q_now = np.asarray(env.robot_view.get_joint_positions(), dtype=np.float64)[
                replay_env_id, :NUM_ROBOT_DOFS
            ]
            if not np.all(np.isfinite(q_now)):
                q_now = joint_trace[-1].copy()
            next_t = float(sim_t[-1] + control_dt) if len(sim_t) > 0 else float(control_dt)
            sim_t.append(next_t)
            tip_trace.append(tip.copy())
            joint_trace.append(q_now.copy())

        sim_t_arr = np.asarray(sim_t, dtype=np.float64)
        tip_arr = np.asarray(tip_trace, dtype=np.float64)
        joints_arr = np.asarray(joint_trace, dtype=np.float64)
        target_world = np.asarray(env.target_world[replay_env_id], dtype=np.float64)

        yz_delta = tip_arr[:, 1:3] - target_world[1:3]
        yz_dist = np.linalg.norm(yz_delta, axis=1)
        min_idx = int(np.argmin(yz_dist))
        min_dist_yz = float(yz_dist[min_idx])
        min_time_yz = float(sim_t_arr[min_idx])
        min_tip = tip_arr[min_idx]
        print(
            "[replay-bj] min tip-target distance on YZ plane: "
            f"{min_dist_yz:.6f} m at t={min_time_yz:.4f}s (frame={min_idx}) | "
            f"tip=(x={min_tip[0]:+.4f}, y={min_tip[1]:+.4f}, z={min_tip[2]:+.4f})",
            flush=True,
        )

        out_prefix = path_traj.with_suffix("")
        out_csv = Path(str(out_prefix) + "_bj_wire_tip_trace.csv")
        out_png = Path(str(out_prefix) + "_bj_wire_tip_yz.png")
        out_joint_png = Path(str(out_prefix) + "_bj_arm_joints.png")

        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with out_csv.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["t", "tip_x", "tip_y", "tip_z", "target_x", "target_y", "target_z"])
            for i in range(sim_t_arr.shape[0]):
                writer.writerow(
                    [
                        float(sim_t_arr[i]),
                        float(tip_arr[i, 0]),
                        float(tip_arr[i, 1]),
                        float(tip_arr[i, 2]),
                        float(target_world[0]),
                        float(target_world[1]),
                        float(target_world[2]),
                    ]
                )

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        y = tip_arr[:, 1]
        z = tip_arr[:, 2]
        fig, ax = plt.subplots(figsize=(8, 7))
        ax.plot(y, z, color="#1f77b4", linewidth=1.7, label="wire tip path")
        ax.scatter([y[0]], [z[0]], color="green", s=40, label="start")
        ax.scatter([y[-1]], [z[-1]], color="red", s=40, label="end")
        ax.scatter([target_world[1]], [target_world[2]], color="#ff7f0e", marker="*", s=180, label="target")
        ax.set_xlabel("Y [m]")
        ax.set_ylabel("Z [m]")
        ax.set_title("BJ Wire Tip Trajectory on YZ Plane")
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal", adjustable="box")
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(out_png, dpi=220)
        plt.close(fig)

        fig_j, axes = plt.subplots(3, 2, figsize=(12, 10), sharex=True)
        axes_flat = axes.reshape(-1)
        for j in range(NUM_ROBOT_DOFS):
            ax = axes_flat[j]
            ax.plot(sim_t_arr, joints_arr[:, j], color="#2a9d8f", linewidth=1.4, label="actual")
            if sim_t_arr.shape[0] == times.shape[0]:
                cmd_t = times
            else:
                cmd_t = times - float(times[0])
            ax.plot(cmd_t, joint_positions[:, j], "--", color="#e76f51", linewidth=1.0, label="command")
            ax.set_title(JOINT_NAMES[j])
            ax.set_ylabel("rad")
            ax.grid(True, alpha=0.3)
            if j == 0:
                ax.legend(loc="best")
        axes_flat[4].set_xlabel("t [s]")
        axes_flat[5].set_xlabel("t [s]")
        fig_j.suptitle("Arm Joint Motion (6 DOF) - BJ Replay", y=0.995)
        fig_j.tight_layout()
        fig_j.savefig(out_joint_png, dpi=220)
        plt.close(fig_j)

        print(f"[replay-bj] wrote tip trace CSV: {out_csv}", flush=True)
        print(f"[replay-bj] wrote YZ figure: {out_png}", flush=True)
        print(f"[replay-bj] wrote arm joint figure: {out_joint_png}", flush=True)
        return out_csv, out_png, out_joint_png
    except Exception:
        traceback.print_exc()
        raise
    finally:
        if env is not None:
            print("[replay-bj] closing environment", flush=True)
            env.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path_traj", type=str, help="Path to trajectory CSV")
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run headless (no GUI).",
    )
    parser.add_argument(
        "--settle-seconds",
        type=float,
        default=0.0,
        help="Hold final command for this many seconds before exit.",
    )
    parser.add_argument(
        "--target-local",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        default=[0.0, 1.5, 0.5],
        help="Target position in env-local coordinates.",
    )
    parser.add_argument(
        "--task-config",
        type=str,
        default=None,
        help="Optional YAML with task config (e.g., train config_merged.yaml).",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=None,
        help="Replay world env count. Default: from --task-config task.num_envs, else 1.",
    )
    parser.add_argument(
        "--replay-env-id",
        type=int,
        default=None,
        help="Which env index receives the CSV commands. Default: inferred from filename (_envX_), else 0.",
    )
    parser.add_argument(
        "--ignore-actions-npz",
        action="store_true",
        help="Ignore sibling *_actions.npz and always reconstruct actions from joint CSV.",
    )
    parser.add_argument(
        "--wire-n-elem",
        type=int,
        default=20,
        help="Number of BJ wire links/elements.",
    )
    parser.add_argument(
        "--wire-base-radius",
        type=float,
        default=0.00635,
        help="BJ wire capsule radius [m].",
    )
    parser.add_argument(
        "--bj-joint-damping",
        type=float,
        default=100.0,
        help="Angular joint damping for BJ wire (D6 drive).",
    )
    parser.add_argument(
        "--bj-attach-stiffness",
        type=float,
        default=5000.0,
        help="Translational stiffness of the stick-tip-to-first-link attach joint. "
             "0 = rigid kinematic attach (default). >0 = soft spring (e.g. 1e3..1e5).",
    )
    parser.add_argument(
        "--bj-attach-damping",
        type=float,
        default=200.0,
        help="Translational damping of the attach joint.",
    )
    parser.add_argument(
        "--bj-attach-rot-stiffness",
        type=float,
        default=100.0,
        help="Rotational stiffness of the attach joint.",
    )
    parser.add_argument(
        "--bj-attach-rot-damping",
        type=float,
        default=10.0,
        help="Rotational damping of the attach joint.",
    )
    parser.add_argument(
        "--bj-link-linear-damping",
        type=float,
        default=1.0,
        help="Linear damping for BJ wire links.",
    )
    parser.add_argument(
        "--bj-link-angular-damping",
        type=float,
        default=0.3,
        help="Angular damping for BJ wire links.",
    )
    args = parser.parse_args()

    traj_path = Path(args.path_traj).expanduser().resolve()
    if not traj_path.is_file():
        raise FileNotFoundError(f"Trajectory file not found: {traj_path}")
    if float(args.settle_seconds) < 0.0:
        raise ValueError("--settle-seconds must be >= 0.")
    if int(args.wire_n_elem) <= 0:
        raise ValueError("--wire-n-elem must be > 0.")
    if float(args.wire_base_radius) <= 0.0:
        raise ValueError("--wire-base-radius must be > 0.")
    inferred_env_id = _infer_env_id_from_name(traj_path)
    task_cfg_path: Path | None = None
    if args.task_config:
        task_cfg_path = Path(args.task_config).expanduser().resolve()
    else:
        task_cfg_path = _auto_find_task_config(traj_path)
        if task_cfg_path is not None:
            print(f"[replay-bj] auto-loaded task config: {task_cfg_path}")
    task_overrides = _load_task_cfg(task_cfg_path) if task_cfg_path is not None else {}

    if len(task_overrides) > 0 and "num_envs" in task_overrides:
        default_num_envs = int(task_overrides.get("num_envs", 1))
    elif inferred_env_id is not None:
        default_num_envs = int(inferred_env_id) + 1
    else:
        default_num_envs = 1
    replay_num_envs = int(args.num_envs) if args.num_envs is not None else default_num_envs
    if replay_num_envs <= 0:
        raise ValueError("--num-envs must be > 0.")
    if args.replay_env_id is not None:
        replay_env_id = int(args.replay_env_id)
    else:
        if inferred_env_id is None:
            replay_env_id = 0
        elif int(inferred_env_id) < int(replay_num_envs):
            replay_env_id = int(inferred_env_id)
        else:
            replay_env_id = 0
            print(
                "[replay-bj] inferred env id from filename is out of range for current num_envs; "
                f"fallback to env 0 (inferred={int(inferred_env_id)}, num_envs={int(replay_num_envs)})."
            )

    out_csv, out_png, out_joint_png = replay_bj(
        traj_path,
        headless=bool(args.headless),
        settle_seconds=float(args.settle_seconds),
        target_local=np.asarray(args.target_local, dtype=np.float64),
        wire_n_elem=int(args.wire_n_elem),
        wire_radius=float(args.wire_base_radius),
        bj_joint_damping=float(args.bj_joint_damping),
        bj_attach_stiffness=float(args.bj_attach_stiffness),
        bj_attach_damping=float(args.bj_attach_damping),
        bj_attach_rot_stiffness=float(args.bj_attach_rot_stiffness),
        bj_attach_rot_damping=float(args.bj_attach_rot_damping),
        bj_link_linear_damping=float(args.bj_link_linear_damping),
        bj_link_angular_damping=float(args.bj_link_angular_damping),
        task_overrides=task_overrides,
        num_envs=replay_num_envs,
        replay_env_id=replay_env_id,
        ignore_actions_npz=bool(args.ignore_actions_npz),
    )
    print(f"[replay-bj] wrote tip trace CSV: {out_csv}")
    print(f"[replay-bj] wrote YZ figure: {out_png}")
    print(f"[replay-bj] wrote arm joint figure: {out_joint_png}")


if __name__ == "__main__":
    main()
