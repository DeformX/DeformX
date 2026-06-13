#!/usr/bin/env python3
"""Replay a joint trajectory CSV with Ball-Joint wire and camera video export.

Usage:
  $ISAAC_PYTHON visualization_scripts/demo3_sim/replay_wire_traj_bj.py /abs/path/to/traj.csv
"""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import numpy.testing  # Keep numpy.testing bound before Kit mutates import paths.

from replay_wire_traj import JOINT_NAMES, NUM_ROBOT_DOFS, load_joint_csv


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[2]
RL_DEMO_ROOT = REPO_ROOT / "RL_Demo"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(RL_DEMO_ROOT) not in sys.path:
    sys.path.insert(0, str(RL_DEMO_ROOT))
PYELASTICA_MESH_ROOT = REPO_ROOT / "PyElastica-Mesh"
if PYELASTICA_MESH_ROOT.is_dir() and str(PYELASTICA_MESH_ROOT) not in sys.path:
    sys.path.insert(0, str(PYELASTICA_MESH_ROOT))

from RL.envs.wire_swing_bj_env import WireSwingBallJointEnv


DEFAULT_PHYS_DT = 0.002
SETTLE_SECONDS = 2.0
DEFAULT_TARGET_LOCAL = np.array([0.0, 2.0, 2.3], dtype=np.float64)
DEFAULT_WIRE_BASE_LENGTH = 1.0
DEFAULT_WIRE_N_ELEM = 20
DEFAULT_WIRE_BASE_RADIUS = 0.00635
DEFAULT_BJ_JOINT_DAMPING = 5.0
DEFAULT_BJ_LINK_LINEAR_DAMPING = 0.6
DEFAULT_BJ_LINK_ANGULAR_DAMPING = 1.2

CAMERA_PATH = "/World/ReplayCamera"
CAMERA_POS = np.array([5.5, 1.8, 1.6], dtype=np.float64)
CAMERA_ROT_XYZ_DEG = np.array([87.0, 0.0, 90.0], dtype=np.float64)
CAMERA_FOCAL_LENGTH_MM = 20.0

PATH_TRACING_SPP = 5
VIDEO_TARGET_FPS = 60.0
VIDEO_CRF = 18
SCENE_LIGHT_INTENSITY = 900.0
SCENE_LIGHT_COLOR = (1.0, 1.0, 1.0)
GROUND_COLOR = np.array([0.0, 0.0, 0.0], dtype=np.float32)


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
    task_overrides: dict[str, object] | None = None,
) -> SimpleNamespace:
    cfg_dict: dict[str, object] = dict(
        name="wire_swing_bj_replay",
        num_envs=int(num_envs),
        env_spacing=3.0,
        phys_dt=float(sim_dt),
        num_substeps=1,
        init_warmup_steps=20,
        num_robot_dofs=6,
        end_effector_link="wrist_3_link",
        active_joints=[1, 2, 3],
        default_joint_positions=[-1.5935, -2.2756, -1.1913, -0.3241, 1.5707, 3.1415],
        robot_offset=[0.0, 0.0, 1.7],
        robot_orient_xyz_deg=[-90.0, 0.0, 0.0],
        target_local=DEFAULT_TARGET_LOCAL.tolist(),
        stick_length=0.65,
        stick_radius=0.010,
        wire_base_length=float(DEFAULT_WIRE_BASE_LENGTH),
        wire_n_elem=int(DEFAULT_WIRE_N_ELEM),
        wire_base_radius=float(DEFAULT_WIRE_BASE_RADIUS),
        bj_joint_damping=float(DEFAULT_BJ_JOINT_DAMPING),
        bj_joint_stiffness=0.0,
        bj_joint_drive_type="force",
        bj_link_linear_damping=float(DEFAULT_BJ_LINK_LINEAR_DAMPING),
        bj_link_angular_damping=float(DEFAULT_BJ_LINK_ANGULAR_DAMPING),
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
    )
    if task_overrides:
        cfg_dict.update(task_overrides)

    cfg_dict["num_envs"] = int(num_envs)
    cfg_dict["phys_dt"] = float(sim_dt)
    # Keep replay running without env auto-resets.
    cfg_dict["max_steps"] = int(1_000_000_000)
    cfg_dict["terminate_on_success"] = False
    cfg_dict["too_far_thresh"] = float(1.0e9)
    cfg_dict["swing_done_enabled"] = False
    cfg_dict["swing_grace_steps"] = int(1_000_000_000)
    cfg_dict["swing_fallback_steps"] = int(1_000_000_000)
    # Replay runs with only one env trajectory (others are unknown), so disable
    # batched sync-reset trigger to avoid cross-env reset interference.
    cfg_dict["sync_reset_ratio"] = 2.0
    return SimpleNamespace(**cfg_dict)


def _reconstruct_actions_from_joint_csv(env: WireSwingBallJointEnv, joint_positions: np.ndarray) -> np.ndarray:
    q_cmd = np.asarray(joint_positions, dtype=np.float32)
    if q_cmd.ndim != 2 or q_cmd.shape[1] != NUM_ROBOT_DOFS:
        raise ValueError(f"Expected joint_positions shape (T, {NUM_ROBOT_DOFS}), got {q_cmd.shape}")

    active = np.asarray(env.active_joints, dtype=np.int64)
    if active.size != int(env.act_dim):
        raise RuntimeError(f"active_joints size {active.size} != env.act_dim {int(env.act_dim)}")

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


def replay(
    path_traj: Path,
    *,
    headless: bool,
    settle_seconds: float,
    task_overrides: dict[str, object] | None,
    num_envs: int,
    replay_env_id: int,
    ignore_actions_npz: bool,
) -> tuple[Path, Path, Path, Path | None]:
    times, joint_positions = load_joint_csv(path_traj)
    sim_dt_csv = _derive_sim_dt_from_csv(times, DEFAULT_PHYS_DT)
    cfg_dt = None
    if task_overrides is not None and "phys_dt" in task_overrides:
        try:
            cfg_dt = float(task_overrides["phys_dt"])
        except Exception:
            cfg_dt = None
    sim_dt = float(cfg_dt) if cfg_dt is not None else float(sim_dt_csv)

    out_prefix = path_traj.with_suffix("")
    out_csv = Path(str(out_prefix) + "_bj_wire_tip_trace.csv")
    out_png = Path(str(out_prefix) + "_bj_wire_tip_yz.png")
    out_joint_png = Path(str(out_prefix) + "_bj_arm_joints.png")
    out_video = Path(str(out_prefix) + "_bj_camera.mp4")
    out_video_frames_dir = Path(str(out_prefix) + "_bj_camera_frames")

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
        task_overrides=task_overrides,
    )

    original_argv = list(sys.argv)
    sys.argv = [sys.argv[0]]

    env: WireSwingBallJointEnv | None = None
    try:
        env = WireSwingBallJointEnv(cfg, headless=bool(headless))
        print("[replay-bj] WireSwingBallJointEnv initialized", flush=True)
        if replay_env_id < 0 or replay_env_id >= int(env.num_envs):
            raise ValueError(f"replay_env_id out of range [0, {int(env.num_envs) - 1}]: {replay_env_id}")

        try:
            import carb

            settings = carb.settings.get_settings()
            settings.set_bool("/rtx/post/motionblur/enabled", False)
            settings.set_string("/rtx/rendermode", "PathTracing")
            settings.set_int("/rtx/pathtracing/spp", int(PATH_TRACING_SPP))
        except Exception:
            pass

        from pxr import Gf, Usd, UsdGeom, UsdLux

        stage = env.stage

        def create_camera(stage_):
            camera = UsdGeom.Camera.Define(stage_, CAMERA_PATH)
            camera.CreateFocalLengthAttr(float(CAMERA_FOCAL_LENGTH_MM))
            xf = UsdGeom.Xformable(camera.GetPrim())
            xf.ClearXformOpOrder()
            xf.AddTranslateOp().Set(Gf.Vec3d(*CAMERA_POS.tolist()))
            xf.AddRotateXYZOp().Set(Gf.Vec3f(*CAMERA_ROT_XYZ_DEG.tolist()))
            return CAMERA_PATH

        def colorize_default_ground(stage_):
            root = stage_.GetPrimAtPath("/World/defaultGroundPlane")
            if not root.IsValid():
                return
            color = Gf.Vec3f(float(GROUND_COLOR[0]), float(GROUND_COLOR[1]), float(GROUND_COLOR[2]))
            changed = 0
            for prim in Usd.PrimRange(root):
                if not prim.IsA(UsdGeom.Gprim):
                    continue
                gp = UsdGeom.Gprim(prim)
                attr = gp.GetDisplayColorAttr()
                if not attr.IsValid():
                    attr = gp.CreateDisplayColorAttr()
                attr.Set([color])
                changed += 1
            if changed > 0:
                print(f"[replay-bj] ground color set to {GROUND_COLOR.tolist()} on {changed} prim(s)", flush=True)

        def use_black_ground_plane():
            # Match replay_wire_traj.py behavior: explicit black ground plane.
            default_ground = stage.GetPrimAtPath("/World/defaultGroundPlane")
            if default_ground.IsValid():
                try:
                    UsdGeom.Imageable(default_ground).MakeInvisible()
                    print("[replay-bj] hid /World/defaultGroundPlane", flush=True)
                except Exception:
                    pass

            black_ground_path = "/World/ReplayGroundPlane"
            if stage.GetPrimAtPath(black_ground_path).IsValid():
                return
            try:
                env.world.scene.add_ground_plane(prim_path=black_ground_path, color=GROUND_COLOR)
            except TypeError:
                env.world.scene.add_ground_plane(color=GROUND_COLOR)
            print(f"[replay-bj] added black ground plane at {black_ground_path}", flush=True)

        dome = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
        dome.CreateIntensityAttr().Set(0.0)
        dome.CreateColorAttr().Set(Gf.Vec3f(*SCENE_LIGHT_COLOR))
        distant = UsdLux.DistantLight.Define(stage, "/World/DistantLight")
        distant.CreateIntensityAttr().Set(float(SCENE_LIGHT_INTENSITY))
        distant.CreateColorAttr().Set(Gf.Vec3f(*SCENE_LIGHT_COLOR))
        extra = UsdLux.DistantLight.Define(stage, "/World/ExtraLight")
        extra.CreateIntensityAttr().Set(float(SCENE_LIGHT_INTENSITY))
        extra.CreateColorAttr().Set(Gf.Vec3f(*SCENE_LIGHT_COLOR))
        exf = UsdGeom.Xformable(extra.GetPrim())
        exf.ClearXformOpOrder()
        exf.AddRotateXYZOp().Set(Gf.Vec3f(0.0, 90.0, 0.0))
        fill = UsdLux.DistantLight.Define(stage, "/World/FillLight")
        fill.CreateIntensityAttr().Set(float(SCENE_LIGHT_INTENSITY))
        fill.CreateColorAttr().Set(Gf.Vec3f(*SCENE_LIGHT_COLOR))
        back = UsdLux.DistantLight.Define(stage, "/World/BackLight")
        back.CreateIntensityAttr().Set(float(SCENE_LIGHT_INTENSITY))
        back.CreateColorAttr().Set(Gf.Vec3f(*SCENE_LIGHT_COLOR))
        use_black_ground_plane()
        colorize_default_ground(stage)

        camera_path = create_camera(stage)
        try:
            import omni.kit.viewport.utility as viewport_utility

            viewport = viewport_utility.get_active_viewport()
            if viewport is not None:
                viewport.set_active_camera(camera_path)
        except Exception:
            pass

        video_step_stride = max(1, int(round((1.0 / sim_dt) / float(VIDEO_TARGET_FPS))))
        video_step_count = 0
        video_frame_count = 0
        video_capture_paths: list[Path] = []
        renderer_capture = None
        capture_enabled = False

        out_video_frames_dir.mkdir(parents=True, exist_ok=True)
        for p in out_video_frames_dir.glob("frame_*.png"):
            try:
                p.unlink()
            except Exception:
                pass

        if not bool(headless):
            try:
                import omni.renderer_capture

                renderer_capture = omni.renderer_capture.acquire_renderer_capture_interface()
                capture_enabled = renderer_capture is not None
            except Exception:
                capture_enabled = False

        if capture_enabled:
            print(
                f"[replay-bj] camera capture enabled | target_fps={VIDEO_TARGET_FPS:.1f} | "
                f"step_stride={video_step_stride} | frame_dir={out_video_frames_dir}",
                flush=True,
            )
        else:
            print("[replay-bj] camera capture unavailable; skipping video export", flush=True)

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

        def maybe_capture_next_frame() -> None:
            nonlocal video_step_count, video_frame_count
            if capture_enabled and ((video_step_count % video_step_stride) == 0):
                frame_path = out_video_frames_dir / f"frame_{video_frame_count:06d}.png"
                renderer_capture.capture_next_frame_swapchain(str(frame_path))
                video_capture_paths.append(frame_path)
                video_frame_count += 1

        for frame_idx in range(times.shape[0]):
            maybe_capture_next_frame()
            act = np.zeros((int(env.num_envs), int(env.act_dim)), dtype=np.float32)
            act[replay_env_id] = actions_replay[frame_idx]
            env.step(act)
            video_step_count += 1

            tip = np.asarray(env.tip_world[replay_env_id], dtype=np.float64)
            if not np.all(np.isfinite(tip)):
                tip = tip_trace[-1].copy() if len(tip_trace) > 0 else np.zeros(3, dtype=np.float64)
            q_now = np.asarray(env.robot_view.get_joint_positions(), dtype=np.float64)[replay_env_id, :NUM_ROBOT_DOFS]
            if not np.all(np.isfinite(q_now)):
                q_now = joint_trace[-1].copy() if len(joint_trace) > 0 else np.asarray(joint_positions[frame_idx], dtype=np.float64)

            sim_t.append(float(times[frame_idx]))
            tip_trace.append(tip.copy())
            joint_trace.append(q_now.copy())

            if frame_idx % 100 == 0:
                print(f"[replay-bj] frame {frame_idx + 1}/{times.shape[0]}", flush=True)

        settle_steps = max(0, int(round(float(settle_seconds) / control_dt)))
        print(
            f"[replay-bj] holding final pose for {float(settle_seconds):.2f}s ({settle_steps} steps)",
            flush=True,
        )
        for _ in range(settle_steps):
            maybe_capture_next_frame()
            act = np.zeros((int(env.num_envs), int(env.act_dim)), dtype=np.float32)
            env.step(act)
            video_step_count += 1

            tip = np.asarray(env.tip_world[replay_env_id], dtype=np.float64)
            if not np.all(np.isfinite(tip)):
                tip = tip_trace[-1].copy()
            q_now = np.asarray(env.robot_view.get_joint_positions(), dtype=np.float64)[replay_env_id, :NUM_ROBOT_DOFS]
            if not np.all(np.isfinite(q_now)):
                q_now = joint_trace[-1].copy()
            next_t = float(sim_t[-1] + control_dt) if len(sim_t) > 0 else float(control_dt)
            sim_t.append(next_t)
            tip_trace.append(tip.copy())
            joint_trace.append(q_now.copy())

        wrote_video = False
        if capture_enabled and video_frame_count > 0:
            for _ in range(4):
                env.world.step(render=True)

            missing_paths = [p for p in video_capture_paths if not p.is_file()]
            t0 = time.time()
            while missing_paths and (time.time() - t0) < 20.0:
                env.simulation_app.update()
                time.sleep(0.02)
                missing_paths = [p for p in video_capture_paths if not p.is_file()]
            if missing_paths:
                print(
                    f"[replay-bj] warning: {len(missing_paths)} captured frame(s) still missing before encode",
                    flush=True,
                )

            ffmpeg_cmd = [
                "ffmpeg",
                "-y",
                "-framerate",
                str(float(VIDEO_TARGET_FPS)),
                "-i",
                str(out_video_frames_dir / "frame_%06d.png"),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-crf",
                str(int(VIDEO_CRF)),
                str(out_video),
            ]
            try:
                subprocess.run(
                    ffmpeg_cmd,
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                wrote_video = out_video.is_file()
            except Exception as exc:
                print(f"[replay-bj] failed to encode camera video: {exc}", flush=True)

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
            "[replay-bj] target world position: "
            f"x={target_world[0]:+.4f}, y={target_world[1]:+.4f}, z={target_world[2]:+.4f}"
        )
        print(
            "[replay-bj] minimum tip-target distance on YZ plane: "
            f"{min_dist_yz:.6f} m at t={min_time_yz:.4f}s "
            f"(tip_y={min_tip[1]:+.4f}, tip_z={min_tip[2]:+.4f})"
        )

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
        ax.scatter(
            [min_tip[1]],
            [min_tip[2]],
            color="#8c564b",
            s=50,
            label=f"min yz dist ({min_dist_yz:.3f} m)",
        )
        ax.plot(
            [min_tip[1], target_world[1]],
            [min_tip[2], target_world[2]],
            "--",
            color="#8c564b",
            linewidth=1.1,
            alpha=0.9,
        )
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
            cmd_t = times if sim_t_arr.shape[0] == times.shape[0] else times - float(times[0])
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
        if wrote_video:
            print(f"[replay-bj] wrote camera video: {out_video}", flush=True)
        elif capture_enabled:
            print(
                f"[replay-bj] camera frames saved (video encode failed): {out_video_frames_dir}",
                flush=True,
            )
            out_video = None
        else:
            out_video = None

        return out_csv, out_png, out_joint_png, out_video
    except Exception:
        traceback.print_exc()
        raise
    finally:
        if env is not None:
            print("[replay-bj] closing environment", flush=True)
            env.close()
        sys.argv = original_argv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path_traj", type=str, help="Path to trajectory CSV")
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run headless (video capture disabled in this mode).",
    )
    parser.add_argument(
        "--settle-seconds",
        type=float,
        default=float(SETTLE_SECONDS),
        help="Hold final command for this many seconds before exit.",
    )
    parser.add_argument(
        "--task-config",
        type=str,
        default=None,
        help="Optional YAML with task config (e.g., train config_merged.yaml).",
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
    args = parser.parse_args()

    traj_path = Path(args.path_traj).expanduser().resolve()
    if not traj_path.is_file():
        raise FileNotFoundError(f"Trajectory file not found: {traj_path}")
    if float(args.settle_seconds) < 0.0:
        raise ValueError("--settle-seconds must be >= 0.")
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
    replay_num_envs = int(default_num_envs)
    if replay_num_envs <= 0:
        raise ValueError("resolved replay num_envs must be > 0.")

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

    out_csv, out_png, out_joint_png, out_video = replay(
        traj_path,
        headless=bool(args.headless),
        settle_seconds=float(args.settle_seconds),
        task_overrides=task_overrides,
        num_envs=replay_num_envs,
        replay_env_id=replay_env_id,
        ignore_actions_npz=bool(args.ignore_actions_npz),
    )
    print(f"[replay-bj] wrote tip trace CSV: {out_csv}")
    print(f"[replay-bj] wrote YZ figure: {out_png}")
    print(f"[replay-bj] wrote arm joint figure: {out_joint_png}")
    if out_video is not None:
        print(f"[replay-bj] wrote camera video: {out_video}")


if __name__ == "__main__":
    main()
