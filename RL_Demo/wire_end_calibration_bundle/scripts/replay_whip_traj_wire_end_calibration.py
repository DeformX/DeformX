#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Replay a CSV UR5e trajectory with PyElastica wire and export wire-end trajectory.

This reuses the same environment setup as replay_exported_trajectory_isaac45.py,
but loads joint commands from a CSV file and runs one replay pass.
"""

import argparse
import json
import os
import sys
import time as _time
from pathlib import Path

import numpy as np
import numpy.testing  # Keep numpy.testing bound before Kit mutates import paths.

SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_DIR = SCRIPT_PATH.parent
BUNDLE_ROOT = SCRIPT_DIR.parent
REPO_ROOT = next(
    (p for p in (BUNDLE_ROOT, *BUNDLE_ROOT.parents) if (p / "co_sim").is_dir()),
    BUNDLE_ROOT.parents[1],
)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--traj_csv",
    type=str,
    default=str(BUNDLE_ROOT / "data" / "trajectory" / "whip_traj_high.csv"),
    help="CSV trajectory file with columns: t, shoulder_pan, shoulder_lift, elbow, wrist_1, wrist_2, wrist_3",
)
parser.add_argument("--headless", action="store_true")
parser.add_argument("--physics_gpu", type=int, default=0)
parser.add_argument(
    "--playback_speed",
    type=float,
    default=1.0,
    help="1.0=real-time, >1 faster, <1 slower (only for GUI playback)",
)
parser.add_argument(
    "--sim_fps",
    type=float,
    default=0.0,
    help="World physics/render rate. <=0 derives from CSV time step.",
)
parser.add_argument(
    "--max_frames",
    type=int,
    default=0,
    help="Replay only the first N frames (0 = all).",
)
parser.add_argument(
    "--wire_usd",
    type=str,
    default="/home/robot/Workspace/Siemens_Cable_Simulator/usd/wire_usdc/wire_usdc/wire_yellow_s20_r0.005_l1.usdc",
)
parser.add_argument("--wire_base_length", type=float, default=1.0)
parser.add_argument("--wire_n_elem", type=int, default=20)
parser.add_argument("--wire_base_radius", type=float, default=0.006)
parser.add_argument("--py_dt", type=float, default=2.0e-5)
parser.add_argument(
    "--engine_cfg",
    type=str,
    default=str(BUNDLE_ROOT / "config" / "replay_whip_traj_wire_end_calibration_engine_cfg.json"),
    help="JSON config file for CoSimEngine parameters",
)
parser.add_argument(
    "--out_csv",
    type=str,
    default=str(BUNDLE_ROOT / "outputs" / "raw" / "whip_wire_end_positions.csv"),
    help="Output CSV containing wire-end and stick-tip positions",
)
parser.add_argument(
    "--out_plot",
    type=str,
    default=str(BUNDLE_ROOT / "outputs" / "plots" / "whip_wire_end_trajectory.png"),
    help="Output trajectory figure path",
)
parser.add_argument(
    "--ref_csv",
    type=str,
    default=str(BUNDLE_ROOT / "data" / "reference" / "whipping_high_1_001_stacked_transformed.csv"),
    help="Reference CSV to overlay in plots (expects Time + X,Y,Z columns)",
)
parser.add_argument(
    "--ref_pos_scale",
    type=float,
    default=1.0e-3,
    help="Scale factor applied to reference XYZ values (default 1e-3 for mm->m)",
)
parser.add_argument(
    "--ref_label",
    type=str,
    default="reference",
    help="Legend label prefix for reference trajectory",
)
parser.add_argument(
    "--compare_t_start",
    type=float,
    default=0.0,
    help="Comparison range start time [s] on elapsed timeline (t-t0)",
)
parser.add_argument(
    "--compare_t_end",
    type=float,
    default=-1.0,
    help="Comparison range end time [s] on elapsed timeline; <0 uses max overlap",
)
parser.add_argument(
    "--compare_out_csv",
    type=str,
    default=str(BUNDLE_ROOT / "outputs" / "comparison" / "whip_reference_comparison.csv"),
    help="Output aligned comparison CSV for selected time range",
)
parser.add_argument(
    "--make_video",
    action="store_true",
    help="Also render a 3D trajectory video from exported wire-end positions",
)
parser.add_argument(
    "--out_video",
    type=str,
    default=str(BUNDLE_ROOT / "outputs" / "videos" / "whip_wire_end_trajectory.mp4"),
    help="Output trajectory video path (.mp4 preferred)",
)
parser.add_argument("--video_fps", type=int, default=60)
args, _ = parser.parse_known_args()

if args.playback_speed <= 0.0:
    raise ValueError("--playback_speed must be > 0")
if args.video_fps <= 0:
    raise ValueError("--video_fps must be > 0")

traj_path = os.path.abspath(args.traj_csv)
if not os.path.isfile(traj_path):
    raise FileNotFoundError(f"Trajectory CSV not found: {traj_path}")

NUM_ROBOT_DOFS = 6
END_EFFECTOR_LINK = "wrist_3_link"
WIRE_ROOT_PATH = "/World/PyElasticaWire"
LONG_STICK_LENGTH = 0.65
LONG_STICK_RADIUS = 0.010

ROBOT_OFFSET = np.array([0.0, 0.0, 1.7], dtype=np.float64)
ROBOT_ORIENT_XYZ_DEG = np.array([-90.0, 0.0, 0.0], dtype=np.float64)
TARGET_LOCAL = np.array([0.0, 1.7, 2.5], dtype=np.float64)

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from isaacsim import SimulationApp
except ImportError:
    from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp(
    {
        "headless": args.headless,
        "physics_gpu": args.physics_gpu,
        "active_gpu": args.physics_gpu,
    }
)

try:
    import carb

    carb.settings.get_settings().set_bool("/rtx/post/motionblur/enabled", False)
except Exception:
    pass

from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdLux, UsdPhysics
from omni.isaac.core import World
from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.utils.stage import add_reference_to_stage
import omni.usd

from co_sim.engine import CoSimEngine
from co_sim.models import CoSimConfig, FrameState
from rod_skel_driver_test import SkeletonRodDriver


def _normalize_field_name(name: str) -> str:
    return "".join(ch.lower() for ch in str(name) if ch.isalnum())


def _pick_name(field_names, candidates):
    for c in candidates:
        if c in field_names:
            return c
    normalized_map = {_normalize_field_name(n): n for n in field_names}
    for c in candidates:
        key = _normalize_field_name(c)
        if key in normalized_map:
            return normalized_map[key]
    return None


def load_joint_csv(csv_path: str):
    arr = np.genfromtxt(csv_path, delimiter=",", names=True, dtype=np.float64)
    if arr.size == 0:
        raise RuntimeError(f"{csv_path} is empty")

    if arr.ndim == 0:
        arr = arr.reshape(1)

    if arr.dtype.names is None:
        raise RuntimeError(f"{csv_path} must contain a header row")

    field_names = tuple(arr.dtype.names)
    t_col = _pick_name(field_names, ("t", "time", "timestamp"))
    if t_col is None:
        raise RuntimeError(
            f"{csv_path} missing time column (expected one of: t,time,timestamp). Fields={field_names}"
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
            f"{csv_path} missing one or more joint columns. "
            f"Fields={field_names}, resolved={joint_cols}"
        )

    times = np.asarray(arr[t_col], dtype=np.float64)
    joint_positions = np.column_stack([np.asarray(arr[c], dtype=np.float64) for c in joint_cols]).astype(np.float32)

    if np.isnan(times).any() or np.isnan(joint_positions).any():
        raise RuntimeError(f"{csv_path} contains NaN values")

    if times.ndim != 1 or joint_positions.ndim != 2 or joint_positions.shape[1] != NUM_ROBOT_DOFS:
        raise RuntimeError(
            f"Bad trajectory shape: times={times.shape}, joints={joint_positions.shape}"
        )

    if len(times) != len(joint_positions):
        raise RuntimeError(
            f"times length {len(times)} != joint_positions length {len(joint_positions)}"
        )

    return times, joint_positions


def derive_sim_fps(times: np.ndarray, fallback: float = 60.0) -> float:
    if len(times) < 2:
        return fallback
    dts = np.diff(times)
    dts = dts[dts > 1.0e-9]
    if len(dts) == 0:
        return fallback
    return float(1.0 / np.median(dts))


def load_reference_xyz_csv(csv_path: str, pos_scale: float = 1.0):
    arr = np.genfromtxt(csv_path, delimiter=",", names=True, dtype=np.float64)
    if arr.size == 0:
        raise RuntimeError(f"{csv_path} is empty")
    if arr.ndim == 0:
        arr = arr.reshape(1)
    if arr.dtype.names is None:
        raise RuntimeError(f"{csv_path} must contain a header row")

    field_names = tuple(arr.dtype.names)
    time_col = _pick_name(
        field_names,
        ("Time (Seconds)", "Time_Seconds", "time_seconds", "time", "Time", "t", "timestamp"),
    )
    x_col = _pick_name(field_names, ("X", "x"))
    y_col = _pick_name(field_names, ("Y", "y"))
    z_col = _pick_name(field_names, ("Z", "z"))
    if time_col is None or x_col is None or y_col is None or z_col is None:
        raise RuntimeError(
            f"{csv_path} missing required columns. "
            f"Need time + X/Y/Z, found fields={field_names}"
        )

    t_raw = np.asarray(arr[time_col], dtype=np.float64)
    xyz = np.column_stack(
        [
            np.asarray(arr[x_col], dtype=np.float64),
            np.asarray(arr[y_col], dtype=np.float64),
            np.asarray(arr[z_col], dtype=np.float64),
        ]
    )
    mask = np.isfinite(t_raw) & np.isfinite(xyz).all(axis=1)
    t_raw = t_raw[mask]
    xyz = xyz[mask]
    if len(t_raw) == 0:
        raise RuntimeError(f"{csv_path} has no valid finite rows")

    t = t_raw - float(t_raw[0])
    xyz = xyz * float(pos_scale)
    return t.astype(np.float64), xyz.astype(np.float64)


COENGINE_CFG_DEFAULT = {
    "py_dt": 2.0e-5,
    "final_time": 1.0e9,
    "output_interval": 0.01,
    "n_elem": 20,
    "base_length": 1.0,
    "base_radius": 0.006,
    "density": 500.0,
    "youngs_modulus": 1.0e5,
    "shear_modulus_ratio": 1.5,
    "damping_constant": 1.0e-2,
    "joint_k": 500.0,
    "joint_nu": 20.0,
    "joint_kt": 10.0,
    "joint_nut": 0.0,
    "frame_base_length": 0.1,
    "frame_base_radius": 0.01,
    "frame_density": 5000.0,
    "output_name": "replay_whip_traj_wire_end_calibration",
    "output_dir": None,
    "render": False,
    "render_speed": 1.0,
    "render_fps": 100,
    "force_vector_scale": 1.0,
    "print_progress": True,
}


def _cli_flag_present(flag_name: str) -> bool:
    return any(arg == flag_name or arg.startswith(flag_name + "=") for arg in sys.argv[1:])


def _write_default_coengine_cfg(cfg_path: str):
    out_dir = os.path.dirname(os.path.abspath(cfg_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(COENGINE_CFG_DEFAULT, f, indent=2, sort_keys=True)
        f.write("\n")


def load_coengine_cfg(cfg_path: str) -> dict:
    cfg_path = os.path.abspath(cfg_path)
    if not os.path.isfile(cfg_path):
        _write_default_coengine_cfg(cfg_path)
        print(f"[ReplayCSV] CoEngine cfg not found, wrote default: {cfg_path}")
        return dict(COENGINE_CFG_DEFAULT)

    with open(cfg_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        raise RuntimeError(f"CoEngine cfg must be a JSON object: {cfg_path}")

    loaded = raw.get("coengine", raw) if isinstance(raw.get("coengine", raw), dict) else raw
    cfg = dict(COENGINE_CFG_DEFAULT)
    for k, v in loaded.items():
        if k in cfg:
            cfg[k] = v
        else:
            print(f"[ReplayCSV] Warning: unknown CoEngine cfg key ignored: {k}")

    # Backward-compatible CLI overrides if explicitly passed.
    if _cli_flag_present("--wire_base_length"):
        cfg["base_length"] = float(args.wire_base_length)
    if _cli_flag_present("--wire_n_elem"):
        cfg["n_elem"] = int(args.wire_n_elem)
    if _cli_flag_present("--wire_base_radius"):
        cfg["base_radius"] = float(args.wire_base_radius)
    if _cli_flag_present("--py_dt"):
        cfg["py_dt"] = float(args.py_dt)

    return cfg


def interpolate_xyz(query_t: np.ndarray, src_t: np.ndarray, src_xyz: np.ndarray) -> np.ndarray:
    x = np.interp(query_t, src_t, src_xyz[:, 0])
    y = np.interp(query_t, src_t, src_xyz[:, 1])
    z = np.interp(query_t, src_t, src_xyz[:, 2])
    return np.column_stack([x, y, z])


def build_comparison(
    sim_t: np.ndarray,
    sim_xyz: np.ndarray,
    ref_t: np.ndarray | None,
    ref_xyz: np.ndarray | None,
    t_start: float,
    t_end: float,
):
    if ref_t is None or ref_xyz is None or len(ref_t) < 2:
        return None
    if len(sim_t) < 2 or len(sim_xyz) < 2:
        return None

    sim_t = np.asarray(sim_t, dtype=np.float64)
    ref_t = np.asarray(ref_t, dtype=np.float64)
    sim_xyz = np.asarray(sim_xyz, dtype=np.float64)
    ref_xyz = np.asarray(ref_xyz, dtype=np.float64)

    user_t0 = float(t_start)
    user_t1 = float(t_end)
    if user_t1 < 0.0:
        user_t1 = min(float(sim_t[-1]), float(ref_t[-1]))

    cmp_t0 = max(float(sim_t[0]), float(ref_t[0]), user_t0)
    cmp_t1 = min(float(sim_t[-1]), float(ref_t[-1]), user_t1)
    if cmp_t1 <= cmp_t0:
        return None

    sim_mask = (sim_t >= cmp_t0) & (sim_t <= cmp_t1)
    if int(np.count_nonzero(sim_mask)) < 2:
        return None

    t = sim_t[sim_mask]
    sim_sel = sim_xyz[sim_mask]
    ref_sel = interpolate_xyz(t, ref_t, ref_xyz)
    err = sim_sel - ref_sel
    err_norm = np.linalg.norm(err, axis=1)
    mae_xyz = np.mean(np.abs(err), axis=0)
    rmse_xyz = np.sqrt(np.mean(err ** 2, axis=0))
    metrics = {
        "samples": int(len(t)),
        "mae_x": float(mae_xyz[0]),
        "mae_y": float(mae_xyz[1]),
        "mae_z": float(mae_xyz[2]),
        "rmse_x": float(rmse_xyz[0]),
        "rmse_y": float(rmse_xyz[1]),
        "rmse_z": float(rmse_xyz[2]),
        "mae_3d": float(np.mean(err_norm)),
        "rmse_3d": float(np.sqrt(np.mean(err_norm ** 2))),
        "max_3d": float(np.max(err_norm)),
    }
    return {
        "range_start": cmp_t0,
        "range_end": cmp_t1,
        "t": t,
        "sim": sim_sel,
        "ref": ref_sel,
        "err": err,
        "err_norm": err_norm,
        "metrics": metrics,
    }


def save_comparison_csv(output_path: str, comparison: dict):
    out_dir = os.path.dirname(os.path.abspath(output_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    t = comparison["t"]
    sim = comparison["sim"]
    ref = comparison["ref"]
    err = comparison["err"]
    err_norm = comparison["err_norm"]
    table = np.column_stack(
        [
            t,
            sim[:, 0],
            sim[:, 1],
            sim[:, 2],
            ref[:, 0],
            ref[:, 1],
            ref[:, 2],
            err[:, 0],
            err[:, 1],
            err[:, 2],
            err_norm,
        ]
    )
    np.savetxt(
        output_path,
        table,
        delimiter=",",
        comments="",
        header="t,sim_x,sim_y,sim_z,ref_x,ref_y,ref_z,err_x,err_y,err_z,err_norm",
    )


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


def _resolve_ur5e_usd_path():
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
    return "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5/Isaac/Robots/UniversalRobots/ur5/ur5.usd"


def find_link_path(stage, robot_path, link_name):
    for prim in Usd.PrimRange(stage.GetPrimAtPath(robot_path)):
        if prim.GetName() == link_name:
            return prim.GetPath().pathString
    return None


def build_wire_engine(frame_initial_state, isaac_dt, coengine_cfg: dict):
    cfg = CoSimConfig(
        py_dt=float(coengine_cfg["py_dt"]),
        isaac_dt=float(isaac_dt),
        final_time=float(coengine_cfg["final_time"]),
        output_interval=float(coengine_cfg["output_interval"]),
        n_elem=int(coengine_cfg["n_elem"]),
        base_length=float(coengine_cfg["base_length"]),
        base_radius=float(coengine_cfg["base_radius"]),
        density=float(coengine_cfg["density"]),
        youngs_modulus=float(coengine_cfg["youngs_modulus"]),
        shear_modulus_ratio=float(coengine_cfg["shear_modulus_ratio"]),
        damping_constant=float(coengine_cfg["damping_constant"]),
        joint_k=float(coengine_cfg["joint_k"]),
        joint_nu=float(coengine_cfg["joint_nu"]),
        joint_kt=float(coengine_cfg["joint_kt"]),
        joint_nut=float(coengine_cfg["joint_nut"]),
        frame_base_length=float(coengine_cfg["frame_base_length"]),
        frame_base_radius=float(coengine_cfg["frame_base_radius"]),
        frame_density=float(coengine_cfg["frame_density"]),
        output_name=str(coengine_cfg["output_name"]),
        output_dir=coengine_cfg["output_dir"],
        render=bool(coengine_cfg["render"]),
        render_speed=float(coengine_cfg["render_speed"]),
        render_fps=(None if coengine_cfg["render_fps"] is None else int(coengine_cfg["render_fps"])),
        force_vector_scale=float(coengine_cfg["force_vector_scale"]),
        print_progress=bool(coengine_cfg["print_progress"]),
        rod_direction=np.asarray(frame_initial_state.director[2], dtype=np.float64),
        rod_normal=np.asarray(frame_initial_state.director[0], dtype=np.float64),
        frame_initial_position=np.asarray(frame_initial_state.position, dtype=np.float64),
        frame_initial_director=np.asarray(frame_initial_state.director, dtype=np.float64),
        frame_initial_velocity=np.asarray(frame_initial_state.velocity, dtype=np.float64),
        frame_initial_acceleration=np.asarray(frame_initial_state.acceleration, dtype=np.float64),
        frame_initial_omega=np.asarray(frame_initial_state.omega, dtype=np.float64),
        frame_initial_alpha=np.asarray(frame_initial_state.alpha, dtype=np.float64),
        rod_start=np.asarray(frame_initial_state.position, dtype=np.float64),
    )
    return CoSimEngine(config=cfg, frame_initial_state=frame_initial_state)


def build_wire_driver(stage):
    driver = SkeletonRodDriver(stage, skeleton_path=WIRE_ROOT_PATH)
    driver.load_asset(args.wire_usd)
    return driver


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
    s = UsdGeom.Sphere.Define(stage, "/World/Target")
    s.GetRadiusAttr().Set(0.03)
    s.GetDisplayColorAttr().Set([Gf.Vec3f(0.0, 1.0, 0.0)])
    xf = UsdGeom.Xformable(s.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(float(local_pos[0]), float(local_pos[1]), float(local_pos[2])))


def make_ground_plane_gray(stage, root_path="/World/defaultGroundPlane", gray=0.45):
    root = stage.GetPrimAtPath(root_path)
    if not root.IsValid():
        return

    color = [Gf.Vec3f(float(gray), float(gray), float(gray))]
    for prim in Usd.PrimRange(root):
        if prim.IsA(UsdGeom.Gprim):
            gprim = UsdGeom.Gprim(prim)
            gprim.GetDisplayColorAttr().Set(color)
            rel = prim.GetRelationship("material:binding")
            if rel and rel.IsValid():
                rel.ClearTargets(True)


def save_end_positions_csv(
    output_path: str,
    sample_times: np.ndarray,
    wire_end_positions: np.ndarray,
    stick_tip_positions: np.ndarray,
):
    out_dir = os.path.dirname(os.path.abspath(output_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    table = np.column_stack(
        [
            sample_times,
            wire_end_positions[:, 0],
            wire_end_positions[:, 1],
            wire_end_positions[:, 2],
            stick_tip_positions[:, 0],
            stick_tip_positions[:, 1],
            stick_tip_positions[:, 2],
        ]
    )
    np.savetxt(
        output_path,
        table,
        delimiter=",",
        header="t,end_x,end_y,end_z,stick_tip_x,stick_tip_y,stick_tip_z",
        comments="",
    )


def _set_equal_3d_limits(ax, xyz: np.ndarray, pad_ratio: float = 0.10):
    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]
    x_min, x_max = float(np.min(x)), float(np.max(x))
    y_min, y_max = float(np.min(y)), float(np.max(y))
    z_min, z_max = float(np.min(z)), float(np.max(z))

    span_x = x_max - x_min
    span_y = y_max - y_min
    span_z = z_max - z_min
    max_span = max(span_x, span_y, span_z, 1e-6)
    half = 0.5 * max_span * (1.0 + float(pad_ratio))

    cx = 0.5 * (x_min + x_max)
    cy = 0.5 * (y_min + y_max)
    cz = 0.5 * (z_min + z_max)

    ax.set_xlim(cx - half, cx + half)
    ax.set_ylim(cy - half, cy + half)
    ax.set_zlim(cz - half, cz + half)


def plot_end_trajectory(
    output_path: str,
    sample_times: np.ndarray,
    wire_end_positions: np.ndarray,
    ref_times: np.ndarray | None = None,
    ref_positions: np.ndarray | None = None,
    ref_label: str = "reference",
    comparison: dict | None = None,
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = os.path.dirname(os.path.abspath(output_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    xyz = wire_end_positions
    has_cmp = comparison is not None
    fig = plt.figure(figsize=(16, 5) if has_cmp else (12, 5))

    ax3 = fig.add_subplot(1, 3, 1, projection="3d") if has_cmp else fig.add_subplot(1, 2, 1, projection="3d")
    ax3.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], color="#1f77b4", linewidth=1.6, label="sim")
    xyz_for_limits = xyz
    if ref_positions is not None and len(ref_positions) > 0:
        ax3.plot(
            ref_positions[:, 0],
            ref_positions[:, 1],
            ref_positions[:, 2],
            color="#ff7f0e",
            linewidth=1.4,
            linestyle="--",
            label=ref_label,
        )
        xyz_for_limits = np.vstack([xyz_for_limits, ref_positions])
    ax3.scatter(xyz[0, 0], xyz[0, 1], xyz[0, 2], color="green", s=30, label="start")
    ax3.scatter(xyz[-1, 0], xyz[-1, 1], xyz[-1, 2], color="red", s=30, label="end")
    ax3.set_title("Wire-End 3D Trajectory")
    ax3.set_xlabel("X [m]")
    ax3.set_ylabel("Y [m]")
    ax3.set_zlabel("Z [m]")
    _set_equal_3d_limits(ax3, xyz_for_limits, pad_ratio=0.10)
    ax3.legend(loc="best")

    ax2 = fig.add_subplot(1, 3, 2) if has_cmp else fig.add_subplot(1, 2, 2)
    ax2.plot(sample_times, xyz[:, 0], label="sim x", linewidth=1.3)
    ax2.plot(sample_times, xyz[:, 1], label="sim y", linewidth=1.3)
    ax2.plot(sample_times, xyz[:, 2], label="sim z", linewidth=1.3)
    if ref_times is not None and ref_positions is not None and len(ref_times) > 0:
        ax2.plot(ref_times, ref_positions[:, 0], "--", label=f"{ref_label} x", linewidth=1.2)
        ax2.plot(ref_times, ref_positions[:, 1], "--", label=f"{ref_label} y", linewidth=1.2)
        ax2.plot(ref_times, ref_positions[:, 2], "--", label=f"{ref_label} z", linewidth=1.2)
    ax2.set_title("Wire-End Position vs Time")
    ax2.set_xlabel("t [s]")
    ax2.set_ylabel("Position [m]")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="best")

    if has_cmp:
        axe = fig.add_subplot(1, 3, 3)
        cmp_t = comparison["t"]
        cmp_err = comparison["err"]
        cmp_norm = comparison["err_norm"]
        m = comparison["metrics"]
        axe.plot(cmp_t, cmp_err[:, 0], label="err x", linewidth=1.0)
        axe.plot(cmp_t, cmp_err[:, 1], label="err y", linewidth=1.0)
        axe.plot(cmp_t, cmp_err[:, 2], label="err z", linewidth=1.0)
        axe.plot(cmp_t, cmp_norm, "k", label="|err|", linewidth=1.5)
        axe.set_title(
            "Selected-Range Error\n"
            f"RMSE3D={m['rmse_3d']:.4f} m, MAX3D={m['max_3d']:.4f} m"
        )
        axe.set_xlabel("t [s]")
        axe.set_ylabel("Error [m]")
        axe.grid(True, alpha=0.3)
        axe.legend(loc="best")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def render_end_trajectory_video(
    output_path: str,
    sample_times: np.ndarray,
    wire_end_positions: np.ndarray,
    video_fps: int,
    ref_times: np.ndarray | None = None,
    ref_positions: np.ndarray | None = None,
    ref_label: str = "reference",
    comparison: dict | None = None,
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import animation

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    x = wire_end_positions[:, 0]
    y = wire_end_positions[:, 1]
    z = wire_end_positions[:, 2]
    n = len(x)

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title("Wire-End 3D Trajectory")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    xyz_for_limits = wire_end_positions
    if ref_positions is not None and len(ref_positions) > 0:
        xyz_for_limits = np.vstack([xyz_for_limits, ref_positions])
    _set_equal_3d_limits(ax, xyz_for_limits, pad_ratio=0.10)
    ax.grid(True, alpha=0.3)

    # Plot the full reference trajectory in the background.
    ax.plot(x, y, z, color="#d9d9d9", linewidth=1.0, label="full path")
    trail, = ax.plot([], [], [], color="#1f77b4", linewidth=2.0, label="traced")
    point, = ax.plot([], [], [], "o", color="red", markersize=5, label="current")
    ref_trail = None
    ref_point = None
    ref_x = ref_y = ref_z = None
    if ref_times is not None and ref_positions is not None and len(ref_times) > 0:
        ref_x = ref_positions[:, 0]
        ref_y = ref_positions[:, 1]
        ref_z = ref_positions[:, 2]
        ax.plot(ref_x, ref_y, ref_z, color="#f0c089", linewidth=1.0, linestyle="--", label=f"{ref_label} full")
        ref_trail, = ax.plot([], [], [], color="#ff7f0e", linewidth=1.8, linestyle="--", label=f"{ref_label} traced")
        ref_point, = ax.plot([], [], [], "o", color="#ff7f0e", markersize=4, label=f"{ref_label} current")
    ax.scatter([x[0]], [y[0]], [z[0]], color="green", s=24, label="start")
    time_text = ax.text2D(0.02, 0.95, "", transform=ax.transAxes)
    cmp_text = ax.text2D(0.02, 0.88, "", transform=ax.transAxes)
    ax.legend(loc="upper right")
    ax.view_init(elev=24.0, azim=35.0)

    def _init():
        trail.set_data([], [])
        trail.set_3d_properties([])
        point.set_data([], [])
        point.set_3d_properties([])
        if ref_trail is not None:
            ref_trail.set_data([], [])
            ref_trail.set_3d_properties([])
        if ref_point is not None:
            ref_point.set_data([], [])
            ref_point.set_3d_properties([])
        time_text.set_text("")
        cmp_text.set_text("")
        artists = [trail, point, time_text, cmp_text]
        if ref_trail is not None:
            artists.append(ref_trail)
        if ref_point is not None:
            artists.append(ref_point)
        return tuple(artists)

    def _update(i):
        trail.set_data(x[: i + 1], y[: i + 1])
        trail.set_3d_properties(z[: i + 1])
        point.set_data([x[i]], [y[i]])
        point.set_3d_properties([z[i]])
        if ref_trail is not None and ref_point is not None:
            t_now = sample_times[i]
            ref_i = int(np.searchsorted(ref_times, t_now, side="right") - 1)
            ref_i = max(0, min(ref_i, len(ref_times) - 1))
            ref_trail.set_data(ref_x[: ref_i + 1], ref_y[: ref_i + 1])
            ref_trail.set_3d_properties(ref_z[: ref_i + 1])
            ref_point.set_data([ref_x[ref_i]], [ref_y[ref_i]])
            ref_point.set_3d_properties([ref_z[ref_i]])
        ax.view_init(elev=24.0, azim=35.0 + 180.0 * float(i) / max(n - 1, 1))
        t_now = float(sample_times[i])
        time_text.set_text(f"t = {t_now:.3f} s")
        if comparison is not None and ref_times is not None and ref_positions is not None:
            t0 = comparison["range_start"]
            t1 = comparison["range_end"]
            m = comparison["metrics"]
            if t_now >= t0 and t_now <= t1 and t_now >= ref_times[0] and t_now <= ref_times[-1]:
                sim_now = np.array([x[i], y[i], z[i]], dtype=np.float64)
                ref_now = interpolate_xyz(np.array([t_now], dtype=np.float64), ref_times, ref_positions)[0]
                e_now = float(np.linalg.norm(sim_now - ref_now))
                cmp_text.set_text(
                    f"cmp[{t0:.3f},{t1:.3f}]s |e|={e_now:.4f}m  RMSE3D={m['rmse_3d']:.4f}m"
                )
            else:
                cmp_text.set_text(
                    f"cmp[{t0:.3f},{t1:.3f}]s (outside)  RMSE3D={m['rmse_3d']:.4f}m"
                )
        artists = [trail, point, time_text, cmp_text]
        if ref_trail is not None:
            artists.append(ref_trail)
        if ref_point is not None:
            artists.append(ref_point)
        return tuple(artists)

    anim = animation.FuncAnimation(
        fig,
        _update,
        init_func=_init,
        frames=n,
        interval=1000.0 / float(video_fps),
        blit=False,
    )

    saved_path = out_path
    try:
        writer = animation.FFMpegWriter(fps=int(video_fps), bitrate=1800)
        anim.save(str(out_path), writer=writer)
    except Exception:
        # Fallback to GIF if ffmpeg is unavailable.
        saved_path = out_path.with_suffix(".gif")
        writer = animation.PillowWriter(fps=int(min(video_fps, 30)))
        anim.save(str(saved_path), writer=writer)

    plt.close(fig)
    return str(saved_path)


def main():
    times, joint_positions = load_joint_csv(traj_path)
    coengine_cfg_path = os.path.abspath(args.engine_cfg)
    coengine_cfg = load_coengine_cfg(coengine_cfg_path)
    print(f"[ReplayCSV] CoEngine cfg: {coengine_cfg_path}")
    print(
        "[ReplayCSV] CoEngine params: "
        f"n_elem={coengine_cfg['n_elem']} "
        f"L={coengine_cfg['base_length']} "
        f"r={coengine_cfg['base_radius']} "
        f"py_dt={coengine_cfg['py_dt']} "
        f"E={coengine_cfg['youngs_modulus']} "
        f"density={coengine_cfg['density']} "
        f"joint(k,nu,kt,nut)=({coengine_cfg['joint_k']}, {coengine_cfg['joint_nu']}, "
        f"{coengine_cfg['joint_kt']}, {coengine_cfg['joint_nut']})"
    )
    ref_times = None
    ref_positions = None
    ref_path = os.path.abspath(args.ref_csv) if args.ref_csv else ""
    if ref_path and os.path.isfile(ref_path):
        try:
            ref_times, ref_positions = load_reference_xyz_csv(
                ref_path, pos_scale=float(args.ref_pos_scale)
            )
            print(
                f"[ReplayCSV] Loaded reference CSV: {ref_path} "
                f"(rows={len(ref_times)}, time=[{ref_times[0]:.3f},{ref_times[-1]:.3f}]s, "
                f"time rebased to t0=0)"
            )
        except Exception as e:
            print(f"[ReplayCSV] Warning: failed to load reference CSV '{ref_path}': {e}")
    elif ref_path:
        print(f"[ReplayCSV] Warning: reference CSV not found, skip overlay: {ref_path}")

    frame_limit = len(times) if args.max_frames <= 0 else min(int(args.max_frames), len(times))
    if frame_limit <= 0:
        raise RuntimeError("No frames to replay")

    sim_fps = float(args.sim_fps)
    if sim_fps <= 0.0:
        sim_fps = derive_sim_fps(times[:frame_limit], fallback=60.0)
    if sim_fps <= 0.0:
        raise ValueError("Resolved sim_fps must be > 0")

    sim_dt = 1.0 / sim_fps
    ur5e_usd_path = _resolve_ur5e_usd_path()

    print(f"[ReplayCSV] UR5e USD: {ur5e_usd_path}")
    print(f"[ReplayCSV] Trajectory: {traj_path}")
    print(
        f"[ReplayCSV] Frames: {frame_limit}/{len(times)} "
        f"Duration: {times[min(frame_limit - 1, len(times)-1)]:.3f}s"
    )
    print(f"[ReplayCSV] sim_fps={sim_fps:.3f} (sim_dt={sim_dt:.6f}s)")

    world = World(
        physics_dt=sim_dt,
        rendering_dt=sim_dt,
        stage_units_in_meters=1.0,
    )
    stage = omni.usd.get_context().get_stage()
    world.scene.add_default_ground_plane()
    make_ground_plane_gray(stage)

    ps = UsdPhysics.Scene.Define(stage, "/physicsScene")
    ps.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
    ps.CreateGravityMagnitudeAttr().Set(9.81)
    PhysxSchema.PhysxSceneAPI.Apply(stage.GetPrimAtPath("/physicsScene")).CreateEnableGPUDynamicsAttr().Set(True)

    dome = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
    dome.CreateIntensityAttr().Set(0.0)
    dome.CreateColorAttr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    distant = UsdLux.DistantLight.Define(stage, "/World/DistantLight")
    distant.CreateIntensityAttr().Set(2500.0)
    distant.CreateColorAttr().Set(Gf.Vec3f(1.0, 1.0, 1.0))
    create_target(stage, TARGET_LOCAL)

    robot_path = "/World/UR5e"
    add_reference_to_stage(usd_path=ur5e_usd_path, prim_path=robot_path)
    xf = UsdGeom.Xformable(stage.GetPrimAtPath(robot_path))
    xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set(Gf.Vec3d(*ROBOT_OFFSET.tolist()))
    xf.AddRotateXYZOp().Set(Gf.Vec3f(*ROBOT_ORIENT_XYZ_DEG.tolist()))

    joint_names = [
        "shoulder_pan_joint",
        "shoulder_lift_joint",
        "elbow_joint",
        "wrist_1_joint",
        "wrist_2_joint",
        "wrist_3_joint",
    ]
    robot_prim = stage.GetPrimAtPath(robot_path)
    for i, jname in enumerate(joint_names):
        for prim in Usd.PrimRange(robot_prim):
            if prim.GetName() == jname:
                drive = UsdPhysics.DriveAPI.Get(prim, "angular")
                if drive:
                    drive.GetTargetPositionAttr().Set(float(np.degrees(joint_positions[0, i])))
                break

    ee_link_path = find_link_path(stage, robot_path, END_EFFECTOR_LINK)
    if ee_link_path is None:
        raise RuntimeError(f"End-effector link '{END_EFFECTOR_LINK}' not found under {robot_path}")
    stick_tip_path = create_long_stick_with_tip(stage, ee_link_path)

    world.reset()
    robot_view = ArticulationView("/World/UR5e", name="robot")
    world.scene.add(robot_view)
    world.reset()
    robot_view.initialize()

    ndof = robot_view.num_dof
    jp = np.zeros((1, ndof), dtype=np.float32)
    jp[0, :NUM_ROBOT_DOFS] = joint_positions[0]
    robot_view.set_joint_positions(jp)
    robot_view.set_joint_velocities(np.zeros((1, ndof), dtype=np.float32))
    robot_view.set_joint_position_targets(jp)

    for _ in range(20):
        world.step(render=not args.headless)

    ee_pos0, ee_dir0 = get_prim_world_pose(stage, stick_tip_path)
    if ee_pos0 is None:
        raise RuntimeError(f"Unable to read world pose for long stick tip: {stick_tip_path}")

    frame_init = FrameState(
        position=ee_pos0,
        director=ee_dir0,
        velocity=np.zeros(3, dtype=np.float64),
        acceleration=np.zeros(3, dtype=np.float64),
        omega=np.zeros(3, dtype=np.float64),
        alpha=np.zeros(3, dtype=np.float64),
    )
    wire_driver = build_wire_driver(stage)
    wire_engine = build_wire_engine(frame_init, sim_dt, coengine_cfg)
    kin_state = {
        "position": np.asarray(frame_init.position, dtype=np.float64),
        "director": np.asarray(frame_init.director, dtype=np.float64),
        "velocity": np.zeros(3, dtype=np.float64),
        "omega": np.zeros(3, dtype=np.float64),
    }

    snap0 = wire_engine.snapshot()
    wire_driver.update_skeleton(snap0.rod_position, snap0.rod_director, time_code=None)

    sample_times = [float(times[0])]
    wire_end_positions = [np.asarray(snap0.rod_position[:, -1], dtype=np.float64)]
    stick_tip_positions = [np.asarray(ee_pos0, dtype=np.float64)]

    t_wall_prev = _time.perf_counter()

    for frame_idx in range(1, frame_limit):
        if not simulation_app.is_running():
            print(f"[ReplayCSV] App stopped at frame {frame_idx}.")
            break

        cmd = np.zeros((1, ndof), dtype=np.float32)
        cmd[0, :NUM_ROBOT_DOFS] = joint_positions[frame_idx]
        robot_view.set_joint_position_targets(cmd)

        ee_pos, ee_dir = get_prim_world_pose(stage, stick_tip_path)
        if ee_pos is not None:
            frame_cmd, kin_state = make_frame_state_from_pose(ee_pos, ee_dir, kin_state, sim_dt)
        else:
            frame_cmd = frame_init

        wire_engine.update_frame_state(frame_cmd, duration=sim_dt)
        snap = wire_engine.snapshot()
        wire_driver.update_skeleton(snap.rod_position, snap.rod_director, time_code=None)

        world.step(render=not args.headless)

        sample_times.append(float(times[frame_idx]))
        wire_end_positions.append(np.asarray(snap.rod_position[:, -1], dtype=np.float64))
        if ee_pos is None:
            stick_tip_positions.append(np.full(3, np.nan, dtype=np.float64))
        else:
            stick_tip_positions.append(np.asarray(ee_pos, dtype=np.float64))

        if not args.headless:
            desired_dt = (times[frame_idx] - times[frame_idx - 1]) / float(args.playback_speed)
            now = _time.perf_counter()
            elapsed = now - t_wall_prev
            t_wall_prev = now
            to_sleep = desired_dt - elapsed
            if to_sleep > 0.0:
                _time.sleep(to_sleep)
        else:
            t_wall_prev = _time.perf_counter()

        if frame_idx % 200 == 0:
            p = wire_end_positions[-1]
            print(
                f"[ReplayCSV] frame {frame_idx:05d}/{frame_limit - 1:05d} "
                f"end=({p[0]:+.4f}, {p[1]:+.4f}, {p[2]:+.4f})"
            )

    sample_times_arr = np.asarray(sample_times, dtype=np.float64)
    wire_end_arr = np.asarray(wire_end_positions, dtype=np.float64)
    stick_tip_arr = np.asarray(stick_tip_positions, dtype=np.float64)
    sim_elapsed_t = sample_times_arr - float(sample_times_arr[0])

    save_end_positions_csv(args.out_csv, sample_times_arr, wire_end_arr, stick_tip_arr)
    print(f"[ReplayCSV] Saved end-point CSV: {os.path.abspath(args.out_csv)}")

    comparison = build_comparison(
        sim_t=sim_elapsed_t,
        sim_xyz=wire_end_arr,
        ref_t=ref_times,
        ref_xyz=ref_positions,
        t_start=float(args.compare_t_start),
        t_end=float(args.compare_t_end),
    )
    if comparison is None:
        print("[ReplayCSV] Comparison skipped (missing ref data or no overlap in selected range).")
    else:
        m = comparison["metrics"]
        print(
            "[ReplayCSV] Comparison "
            f"range=[{comparison['range_start']:.3f},{comparison['range_end']:.3f}]s "
            f"samples={m['samples']} RMSE3D={m['rmse_3d']:.5f}m MAX3D={m['max_3d']:.5f}m"
        )
        save_comparison_csv(args.compare_out_csv, comparison)
        print(f"[ReplayCSV] Saved comparison CSV: {os.path.abspath(args.compare_out_csv)}")

    plot_end_trajectory(
        args.out_plot,
        sim_elapsed_t,
        wire_end_arr,
        ref_times=ref_times,
        ref_positions=ref_positions,
        ref_label=str(args.ref_label),
        comparison=comparison,
    )
    print(f"[ReplayCSV] Saved trajectory figure: {os.path.abspath(args.out_plot)}")

    if args.make_video:
        saved_video = render_end_trajectory_video(
            output_path=args.out_video,
            sample_times=sim_elapsed_t,
            wire_end_positions=wire_end_arr,
            video_fps=int(args.video_fps),
            ref_times=ref_times,
            ref_positions=ref_positions,
            ref_label=str(args.ref_label),
            comparison=comparison,
        )
        print(f"[ReplayCSV] Saved trajectory video: {os.path.abspath(saved_video)}")

    final_end = wire_end_arr[-1]
    print(
        "[ReplayCSV] Done. "
        f"Samples={len(sample_times_arr)} "
        f"Final wire-end=({final_end[0]:+.5f}, {final_end[1]:+.5f}, {final_end[2]:+.5f})"
    )


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
