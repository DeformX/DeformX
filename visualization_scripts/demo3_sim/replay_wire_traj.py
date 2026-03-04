#!/usr/bin/env python3
"""Replay a joint trajectory CSV with SkeletonRodDriver wire visualization.

Usage:
  /home/robot/isaacsim/python.sh RL_Demo/tools/replay_wire_traj.py /abs/path/to/traj.csv
"""

from __future__ import annotations

import argparse
import csv
import inspect
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import numpy.testing  # Keep numpy.testing bound before Kit mutates import paths.


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
PYELASTICA_MESH_ROOT = REPO_ROOT / "PyElastica-Mesh"
if PYELASTICA_MESH_ROOT.is_dir() and str(PYELASTICA_MESH_ROOT) not in sys.path:
    sys.path.insert(0, str(PYELASTICA_MESH_ROOT))

NUM_ROBOT_DOFS = 6
JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3"]
END_EFFECTOR_LINK = "wrist_3_link"
WIRE_ROOT_PATH = "/World/PyElasticaWire"
LONG_STICK_LENGTH = 0.65
LONG_STICK_RADIUS = 0.010

ROBOT_OFFSET = np.array([0.0, 0.0, 1.7], dtype=np.float64)
ROBOT_ORIENT_XYZ_DEG = np.array([-90.0, 0.0, 0.0], dtype=np.float64)
TARGET_LOCAL = np.array([0.0, 2.0, 2.3], dtype=np.float64)
CAMERA_PATH = "/World/ReplayCamera"
CAMERA_POS = np.array([5.2, 1.8, 1.7], dtype=np.float64)
CAMERA_ROT_XYZ_DEG = np.array([87.0, 0.0, 90], dtype=np.float64)  # typical "look toward -X" intent
CAMERA_FOCAL_LENGTH_MM = 20.0
WIRE_USD = (
    "/home/robot/Workspace/Siemens_Cable_Simulator/usd/"
    "wire_usdc/wire_usdc/wire_yellow_s20_r0.005_l1.usdc"
)

DEFAULT_PHYS_DT = 1.0 / 500.0
SETTLE_SECONDS = 1.0
PATH_TRACING_SPP = 5
VIDEO_TARGET_FPS = 60.0
VIDEO_CRF = 18
GROUND_COLOR = np.array([0.0, 0.0, 0.0], dtype=np.float32)


def _normalize_field_name(name: str) -> str:
    return "".join(ch.lower() for ch in str(name) if ch.isalnum())


def _pick_name(field_names: tuple[str, ...], candidates: tuple[str, ...]) -> str | None:
    for c in candidates:
        if c in field_names:
            return c
    normalized_map = {_normalize_field_name(n): n for n in field_names}
    for c in candidates:
        key = _normalize_field_name(c)
        if key in normalized_map:
            return normalized_map[key]
    return None


def load_joint_csv(path_traj: Path) -> tuple[np.ndarray, np.ndarray]:
    arr = np.genfromtxt(str(path_traj), delimiter=",", names=True, dtype=np.float64)
    if arr.size == 0:
        raise RuntimeError(f"Trajectory CSV is empty: {path_traj}")
    if arr.ndim == 0:
        arr = arr.reshape(1)
    if arr.dtype.names is None:
        raise RuntimeError(f"Trajectory CSV must contain a header row: {path_traj}")

    field_names = tuple(arr.dtype.names)
    t_col = _pick_name(field_names, ("t", "time", "timestamp"))
    if t_col is None:
        raise RuntimeError(
            f"Missing time column in {path_traj}. Expected one of t,time,timestamp."
        )

    joint_cols = [
        _pick_name(field_names, ("shoulder_pan", "shoulder_pan_joint")),
        _pick_name(field_names, ("shoulder_lift", "shoulder_lift_joint")),
        _pick_name(field_names, ("elbow", "elbow_joint")),
        _pick_name(field_names, ("wrist_1", "wrist_1_joint")),
        _pick_name(field_names, ("wrist_2", "wrist_2_joint")),
        _pick_name(field_names, ("wrist_3", "wrist_3_joint")),
    ]
    if any(c is None for c in joint_cols):
        raise RuntimeError(
            f"Missing one or more UR joint columns in {path_traj}. "
            f"Fields={field_names}, resolved={joint_cols}"
        )

    times = np.asarray(arr[t_col], dtype=np.float64)
    joint_positions = np.column_stack(
        [np.asarray(arr[c], dtype=np.float64) for c in joint_cols]
    ).astype(np.float32)
    if times.ndim != 1 or joint_positions.ndim != 2 or joint_positions.shape[1] != NUM_ROBOT_DOFS:
        raise RuntimeError(
            f"Bad trajectory shape: times={times.shape}, joints={joint_positions.shape}"
        )
    if times.shape[0] != joint_positions.shape[0]:
        raise RuntimeError(
            f"times length {times.shape[0]} != joints length {joint_positions.shape[0]}"
        )
    if not np.all(np.isfinite(times)) or not np.all(np.isfinite(joint_positions)):
        raise RuntimeError(f"Trajectory has NaN/inf values: {path_traj}")
    if times.shape[0] < 2:
        raise RuntimeError("Need at least 2 trajectory rows")
    return times, joint_positions


def _derive_sim_dt_from_csv(times: np.ndarray, default_dt: float) -> float:
    dts = np.diff(times)
    dts = dts[np.isfinite(dts) & (dts > 1.0e-9)]
    if dts.size == 0:
        return float(default_dt)
    return float(np.median(dts))


def _orthonormalize(R: np.ndarray) -> np.ndarray:
    u, _, vt = np.linalg.svd(R)
    rn = u @ vt
    if np.linalg.det(rn) < 0.0:
        u[:, -1] *= -1.0
        rn = u @ vt
    return rn


def _rotation_matrix_to_rotvec(R: np.ndarray) -> np.ndarray:
    R = _orthonormalize(R)
    tr = np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0)
    angle = float(np.arccos(tr))
    if angle < 1.0e-9:
        return np.zeros(3, dtype=np.float64)
    s = np.sin(angle)
    if abs(s) < 1.0e-9:
        return np.zeros(3, dtype=np.float64)
    axis = np.array(
        [
            (R[2, 1] - R[1, 2]) / (2.0 * s),
            (R[0, 2] - R[2, 0]) / (2.0 * s),
            (R[1, 0] - R[0, 1]) / (2.0 * s),
        ],
        dtype=np.float64,
    )
    return axis * angle


def _resolve_ur5e_usd_path() -> str:
    local_candidates = [
        "/home/robot/isaacsim_assets/Assets/Isaac/4.5/Isaac/Robots/UniversalRobots/ur5/ur5.usd",
        "/home/robot/isaacsim_assets/Assets/Isaac/5.1/Isaac/Robots/UniversalRobots/ur5/ur5.usd",
        "/home/robot/isaacsim_assets/Assets/Isaac/4.5/Isaac/Robots/UniversalRobots/ur5e/ur5e.usd",
        "/home/robot/isaacsim_assets/Assets/Isaac/5.1/Isaac/Robots/UniversalRobots/ur5e/ur5e.usd",
        "/home/robot/isaacsim/Assets/Isaac/4.5/Isaac/Robots/UniversalRobots/ur5e/ur5e.usd",
    ]
    for path in local_candidates:
        if os.path.exists(path):
            return path
    return (
        "https://omniverse-content-production.s3-us-west-2.amazonaws.com/"
        "Assets/Isaac/4.5/Isaac/Robots/UniversalRobots/ur5/ur5.usd"
    )


def _import_skeleton_driver():
    try:
        from tools.rod_skel_driver_sim import SkeletonRodDriver

        return SkeletonRodDriver
    except Exception:
        pass
    try:
        from rod_skel_driver_sim import SkeletonRodDriver

        return SkeletonRodDriver
    except Exception as exc:
        raise ImportError(
            "Cannot import SkeletonRodDriver from tools.rod_skel_driver_sim "
            "or rod_skel_driver_sim."
        ) from exc


def replay(path_traj: Path, cosim_overrides: dict[str, object] | None = None) -> tuple[Path, Path, Path, Path | None]:
    times, joint_positions = load_joint_csv(path_traj)
    sim_dt = _derive_sim_dt_from_csv(times, DEFAULT_PHYS_DT)
    out_prefix = path_traj.with_suffix("")
    out_csv = Path(str(out_prefix) + "_wire_tip_trace.csv")
    out_png = Path(str(out_prefix) + "_wire_tip_yz.png")
    out_joint_png = Path(str(out_prefix) + "_arm_joints.png")
    out_video = Path(str(out_prefix) + "_camera.mp4")
    out_video_frames_dir = Path(str(out_prefix) + "_camera_frames")
    csv_dts = np.diff(times)
    valid_csv_dts = csv_dts[np.isfinite(csv_dts) & (csv_dts > 1.0e-9)]
    median_csv_dt = float(np.median(valid_csv_dts)) if valid_csv_dts.size > 0 else float(sim_dt)
    print(
        f"[replay] loaded {times.shape[0]} frames | "
        f"median csv dt={median_csv_dt:.6f}s | "
        f"sim_dt={sim_dt:.6f}s | "
        "control_dt=sim_dt (1 command per sim step)"
    )

    # Prevent SimulationApp from consuming user trajectory argument as a Kit arg.
    original_argv = list(sys.argv)
    sys.argv = [sys.argv[0]]
    try:
        from isaacsim import SimulationApp
    except ImportError:
        from omni.isaac.kit import SimulationApp

    simulation_app = SimulationApp(
        {"headless": False, "physics_gpu": 0, "active_gpu": 0, "renderer": "PathTracing"}
    )
    try:
        print("[replay] SimulationApp created", flush=True)
        try:
            import carb

            settings = carb.settings.get_settings()
            settings.set_bool("/rtx/post/motionblur/enabled", False)
            settings.set_string("/rtx/rendermode", "PathTracing")
            settings.set_int("/rtx/pathtracing/spp", int(PATH_TRACING_SPP))
        except Exception:
            pass

        from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdLux, UsdPhysics
        from omni.isaac.core import World
        from omni.isaac.core.articulations import ArticulationView
        from omni.isaac.core.utils.stage import add_reference_to_stage
        import omni.usd

        from co_sim.engine import CoSimEngine
        from co_sim.models import CoSimConfig, FrameState

        print("[replay] Isaac + co_sim imports done", flush=True)
        SkeletonRodDriver = _import_skeleton_driver()
        print(f"[replay] SkeletonRodDriver imported: {SkeletonRodDriver}", flush=True)
        if not Path(WIRE_USD).is_file():
            raise FileNotFoundError(f"wire_usd not found: {WIRE_USD}")

        def get_prim_world_pose(stage, prim_path: str):
            prim = stage.GetPrimAtPath(prim_path)
            if not prim.IsValid():
                return None, None
            world_tf = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            t = world_tf.ExtractTranslation()
            R = np.array(world_tf.ExtractRotationMatrix(), dtype=np.float64)
            p = np.array([float(t[0]), float(t[1]), float(t[2])], dtype=np.float64)
            return p, _orthonormalize(R)

        def make_frame_state_from_pose(position, director, prev_kin, dt):
            pos = np.asarray(position, dtype=np.float64)
            R = _orthonormalize(np.asarray(director, dtype=np.float64))
            vel = (pos - prev_kin["position"]) / dt
            acc = (vel - prev_kin["velocity"]) / dt
            R_delta = R @ prev_kin["director"].T
            omega = _rotation_matrix_to_rotvec(R_delta) / dt
            alpha = (omega - prev_kin["omega"]) / dt
            fs = FrameState(
                position=pos,
                director=R,
                velocity=vel,
                acceleration=acc,
                omega=omega,
                alpha=alpha,
            )
            new_kin = {"position": pos, "director": R, "velocity": vel, "omega": omega}
            return fs, new_kin

        def find_link_path(stage, robot_path, link_name):
            for prim in Usd.PrimRange(stage.GetPrimAtPath(robot_path)):
                if prim.GetName() == link_name:
                    return prim.GetPath().pathString
            return None

        def create_long_stick_with_tip(stage, ee_link_path):
            stick_path = f"{ee_link_path}/LongStickVisual"
            cap = UsdGeom.Capsule.Define(stage, stick_path)
            cap.CreateHeightAttr(float(LONG_STICK_LENGTH))
            cap.CreateRadiusAttr(float(LONG_STICK_RADIUS))
            cap.CreateAxisAttr("Z")
            cap.CreateDisplayColorAttr().Set([Gf.Vec3f(0.55, 0.55, 0.6)])
            stick_xf = UsdGeom.Xformable(cap.GetPrim())
            stick_xf.ClearXformOpOrder()
            stick_xf.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, float(LONG_STICK_LENGTH * 0.5)))

            tip_path = f"{ee_link_path}/LongStickTip"
            tip = UsdGeom.Xform.Define(stage, tip_path)
            tip_xf = UsdGeom.Xformable(tip.GetPrim())
            tip_xf.ClearXformOpOrder()
            tip_xf.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, float(LONG_STICK_LENGTH)))
            return tip_path

        def create_target(stage, local_pos):
            sphere = UsdGeom.Sphere.Define(stage, "/World/Target")
            sphere.GetRadiusAttr().Set(0.03)
            sphere.GetDisplayColorAttr().Set([Gf.Vec3f(0.0, 1.0, 0.0)])
            xf = UsdGeom.Xformable(sphere.GetPrim())
            xf.ClearXformOpOrder()
            xf.AddTranslateOp().Set(
                Gf.Vec3d(float(local_pos[0]), float(local_pos[1]), float(local_pos[2]))
            )

        def create_camera(stage):
            camera = UsdGeom.Camera.Define(stage, CAMERA_PATH)
            camera.CreateFocalLengthAttr(float(CAMERA_FOCAL_LENGTH_MM))
            xf = UsdGeom.Xformable(camera.GetPrim())
            xf.ClearXformOpOrder()
            xf.AddTranslateOp().Set(Gf.Vec3d(*CAMERA_POS.tolist()))
            xf.AddRotateXYZOp().Set(Gf.Vec3f(*CAMERA_ROT_XYZ_DEG.tolist()))
            return CAMERA_PATH

        world = World(physics_dt=sim_dt, rendering_dt=sim_dt, stage_units_in_meters=1.0)
        stage = omni.usd.get_context().get_stage()
        print("[replay] World created", flush=True)
        world.scene.add_ground_plane(color=GROUND_COLOR)

        ps = UsdPhysics.Scene.Define(stage, "/physicsScene")
        ps.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
        ps.CreateGravityMagnitudeAttr().Set(9.81)
        PhysxSchema.PhysxSceneAPI.Apply(stage.GetPrimAtPath("/physicsScene")).CreateEnableGPUDynamicsAttr().Set(
            True
        )

        dome = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
        dome.CreateIntensityAttr().Set(0.0)
        dome.CreateColorAttr().Set(Gf.Vec3f(1.0, 1.0, 1.0))
        distant = UsdLux.DistantLight.Define(stage, "/World/DistantLight")
        distant.CreateIntensityAttr().Set(3800.0)
        distant.CreateColorAttr().Set(Gf.Vec3f(1.0, 1.0, 1.0))
        fill = UsdLux.DistantLight.Define(stage, "/World/FillLight")
        fill.CreateIntensityAttr().Set(1200.0)
        fill.CreateColorAttr().Set(Gf.Vec3f(0.95, 0.95, 1.0))
        back = UsdLux.DistantLight.Define(stage, "/World/BackLight")
        back.CreateIntensityAttr().Set(900.0)
        back.CreateColorAttr().Set(Gf.Vec3f(1.0, 0.98, 0.95))
        create_target(stage, TARGET_LOCAL)
        camera_path = create_camera(stage)
        try:
            import omni.kit.viewport.utility as viewport_utility

            viewport = viewport_utility.get_active_viewport()
            if viewport is not None:
                viewport.set_active_camera(camera_path)
        except Exception:
            pass
        print("[replay] Target prim created", flush=True)

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
        try:
            import omni.renderer_capture

            renderer_capture = omni.renderer_capture.acquire_renderer_capture_interface()
            capture_enabled = renderer_capture is not None
        except Exception:
            capture_enabled = False

        if capture_enabled:
            print(
                f"[replay] camera capture enabled | target_fps={VIDEO_TARGET_FPS:.1f} | "
                f"step_stride={video_step_stride} | frame_dir={out_video_frames_dir}",
                flush=True,
            )
        else:
            print("[replay] camera capture unavailable; skipping video export", flush=True)

        def step_world_with_capture(record_video: bool = True):
            nonlocal video_step_count, video_frame_count
            if record_video and capture_enabled and ((video_step_count % video_step_stride) == 0):
                frame_path = out_video_frames_dir / f"frame_{video_frame_count:06d}.png"
                renderer_capture.capture_next_frame_swapchain(str(frame_path))
                video_capture_paths.append(frame_path)
                video_frame_count += 1
            world.step(render=True)
            if record_video:
                video_step_count += 1

        robot_path = "/World/UR5e"
        add_reference_to_stage(usd_path=_resolve_ur5e_usd_path(), prim_path=robot_path)
        xf = UsdGeom.Xformable(stage.GetPrimAtPath(robot_path))
        xf.ClearXformOpOrder()
        xf.AddTranslateOp().Set(Gf.Vec3d(*ROBOT_OFFSET.tolist()))
        xf.AddRotateXYZOp().Set(Gf.Vec3f(*ROBOT_ORIENT_XYZ_DEG.tolist()))

        ee_link_path = find_link_path(stage, robot_path, END_EFFECTOR_LINK)
        if ee_link_path is None:
            raise RuntimeError(f"Cannot find end effector link: {END_EFFECTOR_LINK}")
        stick_tip_path = create_long_stick_with_tip(stage, ee_link_path)
        print(f"[replay] Stick tip path: {stick_tip_path}", flush=True)

        world.reset()
        robot_view = ArticulationView("/World/UR5e", name="robot")
        world.scene.add(robot_view)
        world.reset()
        robot_view.initialize()

        ndof = int(robot_view.num_dof)
        cmd = np.zeros((1, ndof), dtype=np.float32)
        cmd[0, :NUM_ROBOT_DOFS] = joint_positions[0]
        robot_view.set_joint_positions(cmd)
        robot_view.set_joint_velocities(np.zeros_like(cmd))
        robot_view.set_joint_position_targets(cmd)

        for _ in range(20):
            step_world_with_capture(record_video=False)

        tip_pos0, tip_dir0 = get_prim_world_pose(stage, stick_tip_path)
        if tip_pos0 is None:
            raise RuntimeError(f"Cannot read stick-tip pose: {stick_tip_path}")

        frame_init = FrameState(
            position=np.asarray(tip_pos0, dtype=np.float64),
            director=np.asarray(tip_dir0, dtype=np.float64),
            velocity=np.zeros(3, dtype=np.float64),
            acceleration=np.zeros(3, dtype=np.float64),
            omega=np.zeros(3, dtype=np.float64),
            alpha=np.zeros(3, dtype=np.float64),
        )
        wire_driver = SkeletonRodDriver(stage, skeleton_path=WIRE_ROOT_PATH)
        wire_driver.load_asset(WIRE_USD)
        print("[replay] Skeleton wire asset loaded", flush=True)

        cfg_kwargs: dict[str, object] = dict(
            base_length=1.0,
            n_elem=20,
            base_radius=0.00635,
            py_dt=1.0e-5,
            isaac_dt=sim_dt,
            final_time=1.0e9,
            render=False,
            joint_k=1500.0,
            joint_nu=20.0,
            joint_kt=10.0,
            joint_nut=0.0,
            density=700.0,
            youngs_modulus=2e6,
            shear_modulus_ratio=1.5,
            damping_constant=0.1,
            rod_direction=np.asarray(frame_init.director[2], dtype=np.float64),
            rod_normal=np.asarray(frame_init.director[0], dtype=np.float64),
            frame_initial_position=np.asarray(frame_init.position, dtype=np.float64),
            frame_initial_director=np.asarray(frame_init.director, dtype=np.float64),
            frame_initial_velocity=np.asarray(frame_init.velocity, dtype=np.float64),
            frame_initial_acceleration=np.asarray(frame_init.acceleration, dtype=np.float64),
            frame_initial_omega=np.asarray(frame_init.omega, dtype=np.float64),
            frame_initial_alpha=np.asarray(frame_init.alpha, dtype=np.float64),
            rod_start=np.asarray(frame_init.position, dtype=np.float64),
            use_ground_contact=True,
            ground_z=0.0,
            ground_contact_k=1.0e2,
            ground_contact_nu=1.0,
            ground_static_mu=np.array([1.0, 1.0, 1.0], dtype=np.float64),
            ground_kinetic_mu=np.array([0.5, 0.5, 0.5], dtype=np.float64),
            ground_slip_velocity_tol=1.0e-6,
            settle_time=1.0,
            initial_wire_theta = -0.5 * np.pi,
        )
        if cosim_overrides:
            cfg_kwargs.update(cosim_overrides)
        accepted_keys = set(inspect.signature(CoSimConfig).parameters.keys())
        dropped_keys = sorted([k for k in cfg_kwargs.keys() if k not in accepted_keys])
        if len(dropped_keys) > 0:
            print(f"[replay] CoSimConfig dropped unsupported keys: {dropped_keys}", flush=True)
        engine_cfg = CoSimConfig(**{k: v for k, v in cfg_kwargs.items() if k in accepted_keys})
        wire_engine = CoSimEngine(config=engine_cfg, frame_initial_state=frame_init)
        print("[replay] CoSimEngine initialized", flush=True)
        kin_state = {
            "position": np.asarray(frame_init.position, dtype=np.float64),
            "director": np.asarray(frame_init.director, dtype=np.float64),
            "velocity": np.zeros(3, dtype=np.float64),
            "omega": np.zeros(3, dtype=np.float64),
        }

        last_good_rod_pos = None
        last_good_rod_dir = None

        def safe_wire_snapshot(snap):
            nonlocal last_good_rod_pos, last_good_rod_dir
            rod_pos = np.asarray(snap.rod_position, dtype=np.float64)
            rod_dir = np.asarray(snap.rod_director, dtype=np.float64)
            finite = np.all(np.isfinite(rod_pos)) and np.all(np.isfinite(rod_dir))
            if finite:
                last_good_rod_pos = rod_pos.copy()
                last_good_rod_dir = rod_dir.copy()
                return rod_pos, rod_dir
            if last_good_rod_pos is not None and last_good_rod_dir is not None:
                return last_good_rod_pos, last_good_rod_dir
            rod_pos = np.nan_to_num(rod_pos, nan=0.0, posinf=0.0, neginf=0.0)
            rod_dir = np.nan_to_num(rod_dir, nan=0.0, posinf=0.0, neginf=0.0)
            return rod_pos, rod_dir

        snap0 = wire_engine.snapshot()
        rod_pos0, rod_dir0 = safe_wire_snapshot(snap0)
        wire_driver.update_skeleton(rod_pos0, rod_dir0, time_code=None)

        sim_t = [0.0]
        tip_trace = [np.asarray(rod_pos0[:, -1], dtype=np.float64)]
        q0 = (
            np.asarray(robot_view.get_joint_positions(), dtype=np.float64)[0, :NUM_ROBOT_DOFS].copy()
        )
        joint_trace = [q0]
        t_elapsed = 0.0

        for frame_idx in range(1, times.shape[0]):
            cmd[:, :] = 0.0
            cmd[0, :NUM_ROBOT_DOFS] = joint_positions[frame_idx]
            robot_view.set_joint_position_targets(cmd)

            tip_pos, tip_dir = get_prim_world_pose(stage, stick_tip_path)
            if tip_pos is not None:
                frame_cmd, kin_state = make_frame_state_from_pose(
                    tip_pos, tip_dir, kin_state, sim_dt
                )
            else:
                frame_cmd = frame_init
            wire_engine.update_frame_state(frame_cmd, duration=sim_dt)
            snap = wire_engine.snapshot()
            rod_pos, rod_dir = safe_wire_snapshot(snap)
            try:
                wire_driver.update_skeleton(rod_pos, rod_dir, time_code=None)
            except Exception:
                pass
            step_world_with_capture(record_video=True)

            tip = np.asarray(rod_pos[:, -1], dtype=np.float64)
            if not np.all(np.isfinite(tip)):
                tip = tip_trace[-1].copy()
            t_elapsed += sim_dt
            sim_t.append(t_elapsed)
            tip_trace.append(tip)
            q_now = np.asarray(robot_view.get_joint_positions(), dtype=np.float64)[0, :NUM_ROBOT_DOFS]
            if not np.all(np.isfinite(q_now)):
                q_now = joint_trace[-1].copy()
            joint_trace.append(q_now.copy())

            if frame_idx % 100 == 0:
                print(f"[replay] frame {frame_idx}/{times.shape[0] - 1}")

        settle_steps = max(1, int(round(SETTLE_SECONDS / sim_dt)))
        print(f"[replay] holding final pose for {SETTLE_SECONDS:.2f}s ({settle_steps} steps)")
        for _ in range(settle_steps):
            tip_pos, tip_dir = get_prim_world_pose(stage, stick_tip_path)
            if tip_pos is not None:
                frame_cmd, kin_state = make_frame_state_from_pose(
                    tip_pos, tip_dir, kin_state, sim_dt
                )
            else:
                frame_cmd = frame_init
            wire_engine.update_frame_state(frame_cmd, duration=sim_dt)
            snap = wire_engine.snapshot()
            rod_pos, rod_dir = safe_wire_snapshot(snap)
            try:
                wire_driver.update_skeleton(rod_pos, rod_dir, time_code=None)
            except Exception:
                pass
            step_world_with_capture(record_video=True)

            tip = np.asarray(rod_pos[:, -1], dtype=np.float64)
            if not np.all(np.isfinite(tip)):
                tip = tip_trace[-1].copy()
            t_elapsed += sim_dt
            sim_t.append(t_elapsed)
            tip_trace.append(tip)
            q_now = np.asarray(robot_view.get_joint_positions(), dtype=np.float64)[0, :NUM_ROBOT_DOFS]
            if not np.all(np.isfinite(q_now)):
                q_now = joint_trace[-1].copy()
            joint_trace.append(q_now.copy())

        wrote_video = False
        if capture_enabled and video_frame_count > 0:
            for _ in range(4):
                world.step(render=True)

            missing_paths = [p for p in video_capture_paths if not p.is_file()]
            t0 = time.time()
            while missing_paths and (time.time() - t0) < 20.0:
                simulation_app.update()
                time.sleep(0.02)
                missing_paths = [p for p in video_capture_paths if not p.is_file()]
            if missing_paths:
                print(
                    f"[replay] warning: {len(missing_paths)} captured frame(s) still missing before encode",
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
                print(f"[replay] failed to encode camera video: {exc}", flush=True)

        sim_t_arr = np.asarray(sim_t, dtype=np.float64)
        tip_arr = np.asarray(tip_trace, dtype=np.float64)
        joints_arr = np.asarray(joint_trace, dtype=np.float64)
        target_world = TARGET_LOCAL.copy()
        tip_yz = tip_arr[:, 1:3]
        target_yz = target_world[1:3]
        yz_distance = np.linalg.norm(tip_yz - target_yz[None, :], axis=1)
        min_yz_idx = int(np.argmin(yz_distance))
        min_yz_distance = float(yz_distance[min_yz_idx])
        min_yz_time = float(sim_t_arr[min_yz_idx])
        min_yz_tip = tip_arr[min_yz_idx]
        print(
            "[replay] target world position: "
            f"x={target_world[0]:+.4f}, y={target_world[1]:+.4f}, z={target_world[2]:+.4f}"
        )
        print(
            "[replay] minimum tip-target distance on YZ plane: "
            f"{min_yz_distance:.6f} m at t={min_yz_time:.4f}s "
            f"(tip_y={min_yz_tip[1]:+.4f}, tip_z={min_yz_tip[2]:+.4f})"
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
            [min_yz_tip[1]],
            [min_yz_tip[2]],
            color="#8c564b",
            s=50,
            label=f"min yz dist ({min_yz_distance:.3f} m)",
        )
        ax.plot(
            [min_yz_tip[1], target_world[1]],
            [min_yz_tip[2], target_world[2]],
            "--",
            color="#8c564b",
            linewidth=1.1,
            alpha=0.9,
        )
        ax.set_xlabel("Y [m]")
        ax.set_ylabel("Z [m]")
        ax.set_title("Wire Tip Trajectory on YZ Plane")
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
            ax.plot(times - float(times[0]), joint_positions[:, j], "--", color="#e76f51", linewidth=1.0, label="command")
            ax.set_title(JOINT_NAMES[j])
            ax.set_ylabel("rad")
            ax.grid(True, alpha=0.3)
            if j == 0:
                ax.legend(loc="best")
        axes_flat[4].set_xlabel("t [s]")
        axes_flat[5].set_xlabel("t [s]")
        fig_j.suptitle("Arm Joint Motion (6 DOF)", y=0.995)
        fig_j.tight_layout()
        fig_j.savefig(out_joint_png, dpi=220)
        plt.close(fig_j)

        print(f"[replay] wrote tip trace CSV: {out_csv}", flush=True)
        print(f"[replay] wrote YZ figure: {out_png}", flush=True)
        print(f"[replay] wrote arm joint figure: {out_joint_png}", flush=True)
        if wrote_video:
            print(f"[replay] wrote camera video: {out_video}", flush=True)
        elif capture_enabled:
            print(
                f"[replay] camera frames saved (video encode failed): {out_video_frames_dir}",
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
        print("[replay] closing SimulationApp", flush=True)
        simulation_app.close()
        sys.argv = original_argv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path_traj", type=str, help="Path to trajectory CSV")
    parser.add_argument(
        "--use-ground-contact",
        action="store_true",
        help="Enable co-sim ground contact/friction.",
    )
    parser.add_argument("--ground-z", type=float, default=0.0, help="Ground plane z height for co-sim.")
    parser.add_argument("--ground-contact-k", type=float, default=1.0e4, help="Ground contact stiffness.")
    parser.add_argument("--ground-contact-nu", type=float, default=5.0, help="Ground contact damping.")
    parser.add_argument(
        "--ground-static-mu",
        type=float,
        nargs=3,
        metavar=("MU_T", "MU_N", "MU_B"),
        default=[1.0, 1.0, 1.0],
        help="Ground static friction triplet.",
    )
    parser.add_argument(
        "--ground-kinetic-mu",
        type=float,
        nargs=3,
        metavar=("MU_T", "MU_N", "MU_B"),
        default=[0.5, 0.5, 0.5],
        help="Ground kinetic friction triplet.",
    )
    parser.add_argument(
        "--ground-slip-velocity-tol",
        type=float,
        default=1.0e-6,
        help="Slip velocity tolerance for ground friction.",
    )
    parser.add_argument(
        "--settle-duration",
        type=float,
        default=0.0,
        help="Warm-start settle duration (seconds) before replay starts.",
    )
    args = parser.parse_args()

    traj_path = Path(args.path_traj).expanduser().resolve()
    if not traj_path.is_file():
        raise FileNotFoundError(f"Trajectory file not found: {traj_path}")

    cosim_overrides = dict(
        use_ground_contact=bool(args.use_ground_contact),
        ground_z=float(args.ground_z),
        ground_contact_k=float(args.ground_contact_k),
        ground_contact_nu=float(args.ground_contact_nu),
        ground_static_mu=np.asarray(args.ground_static_mu, dtype=np.float64),
        ground_kinetic_mu=np.asarray(args.ground_kinetic_mu, dtype=np.float64),
        ground_slip_velocity_tol=float(args.ground_slip_velocity_tol),
        settle_duration=float(args.settle_duration),
    )
    out_csv, out_png, out_joint_png, out_video = replay(traj_path, cosim_overrides=cosim_overrides)
    print(f"[replay] wrote tip trace CSV: {out_csv}")
    print(f"[replay] wrote YZ figure: {out_png}")
    print(f"[replay] wrote arm joint figure: {out_joint_png}")
    if out_video is not None:
        print(f"[replay] wrote camera video: {out_video}")


if __name__ == "__main__":
    main()
