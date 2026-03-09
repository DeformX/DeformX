#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Replay exported UR5e trajectory with PyElastica wire attached to the end effector."""

import argparse
import os
import sys
import time as _time
from pathlib import Path

import numpy as np
import numpy.testing  # Keep numpy.testing bound before Kit mutates import paths.

NUM_ROBOT_DOFS = 6
SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_DIR = SCRIPT_PATH.parent
BUNDLE_ROOT = SCRIPT_DIR.parent
REPO_ROOT = next(
    (p for p in (BUNDLE_ROOT, *BUNDLE_ROOT.parents) if (p / "co_sim").is_dir()),
    BUNDLE_ROOT.parents[1],
)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--traj_npz",
    type=str,
    default=str(BUNDLE_ROOT / "data" / "trajectory" / "train_isaac45_exported_trajectory.npz"),
    help="Trajectory npz from export_train_trajectory_isaac45.py",
)
parser.add_argument("--headless", action="store_true")
parser.add_argument("--physics_gpu", type=int, default=0)
parser.add_argument("--playback_speed", type=float, default=1.0,
                    help="1.0=real-time, >1 faster, <1 slower")
parser.add_argument("--sim_fps", type=float, default=60.0,
                    help="World physics and rendering rate")
parser.add_argument("--max_frames", type=int, default=0,
                    help="Replay only the first N frames (0 = all).")
parser.add_argument(
    "--wire_usd",
    type=str,
    default="/home/robot/Workspace/Siemens_Cable_Simulator/usd/wire_usdc/wire_usdc/wire_yellow_s20_r0.005_l1.usdc",
)
parser.add_argument("--wire_base_length", type=float, default=1)
parser.add_argument("--wire_n_elem", type=int, default=20)
parser.add_argument("--wire_base_radius", type=float, default=0.006)
parser.add_argument("--py_dt", type=float, default=2.0e-5)
parser.add_argument(
    "--set_joints_deg",
    type=str,
    default="",
    help="Optional one-shot target in degrees: 6 values (base/shoulder/elbow/wrist1/wrist2/wrist3), comma or space separated.",
)
parser.add_argument(
    "--set_joints_rad",
    type=str,
    default="",
    help="Optional one-shot target in radians: 6 values (base/shoulder/elbow/wrist1/wrist2/wrist3), comma or space separated.",
)
parser.add_argument(
    "--one_shot",
    action="store_true",
    help="Command one pose, print final long-stick tip pose, and exit.",
)
parser.add_argument(
    "--settle_steps",
    type=int,
    default=60,
    help="Physics steps to settle before printing one-shot tip pose.",
)
args, _ = parser.parse_known_args()

if args.playback_speed <= 0.0:
    raise ValueError("--playback_speed must be > 0")
if args.sim_fps <= 0.0:
    raise ValueError("--sim_fps must be > 0")
if args.settle_steps < 0:
    raise ValueError("--settle_steps must be >= 0")
if args.set_joints_deg and args.set_joints_rad:
    raise ValueError("Use only one of --set_joints_deg or --set_joints_rad")


def _parse_joint_list(text: str, arg_name: str) -> np.ndarray:
    toks = [tok for tok in text.replace(",", " ").split() if tok]
    if len(toks) != NUM_ROBOT_DOFS:
        raise ValueError(
            f"{arg_name} requires exactly {NUM_ROBOT_DOFS} values, got {len(toks)}: {text!r}"
        )
    try:
        vals = np.asarray([float(tok) for tok in toks], dtype=np.float64)
    except ValueError as exc:
        raise ValueError(f"{arg_name} contains non-numeric values: {text!r}") from exc
    return vals

traj_path = os.path.abspath(args.traj_npz)
if not os.path.isfile(traj_path):
    raise FileNotFoundError(f"Trajectory file not found: {traj_path}")

d = np.load(traj_path)
if "times" not in d:
    raise RuntimeError(f"{traj_path} missing key 'times'")
if "joint_positions" in d:
    joint_positions = d["joint_positions"].astype(np.float32)
elif "positions" in d:
    joint_positions = d["positions"].astype(np.float32)
else:
    raise RuntimeError(f"{traj_path} missing key 'joint_positions'/'positions'")

times = d["times"].astype(np.float64)
if joint_positions.ndim != 2 or joint_positions.shape[1] < 6:
    raise RuntimeError(
        f"joint_positions must be shape (F, >=6), got {joint_positions.shape}"
    )
joint_positions = joint_positions[:, :6]

if len(times) != len(joint_positions):
    raise RuntimeError(
        f"times length {len(times)} != joint_positions length {len(joint_positions)}"
    )

manual_joint_target = None
if args.set_joints_deg:
    manual_joint_target = np.deg2rad(_parse_joint_list(args.set_joints_deg, "--set_joints_deg")).astype(np.float32)
elif args.set_joints_rad:
    manual_joint_target = _parse_joint_list(args.set_joints_rad, "--set_joints_rad").astype(np.float32)

if manual_joint_target is not None:
    joint_positions = manual_joint_target.reshape(1, NUM_ROBOT_DOFS)
    times = np.array([0.0], dtype=np.float64)
    args.one_shot = True

if "robot_offset" in d:
    ROBOT_OFFSET = d["robot_offset"].astype(np.float64)
else:
    ROBOT_OFFSET = np.array([0.0, 0.0, 1.7], dtype=np.float64)

if "robot_orient_xyz_deg" in d:
    ROBOT_ORIENT_XYZ_DEG = d["robot_orient_xyz_deg"].astype(np.float64)
else:
    ROBOT_ORIENT_XYZ_DEG = np.array([-90.0, 0.0, 0.0], dtype=np.float64)

if "target_local" in d:
    TARGET_LOCAL = d["target_local"].astype(np.float64)
else:
    TARGET_LOCAL = np.array([0.0, 1.7, 2.5], dtype=np.float64)

END_EFFECTOR_LINK = "wrist_3_link"
WIRE_ROOT_PATH = "/World/PyElasticaWire"
LONG_STICK_LENGTH = 0.65
LONG_STICK_RADIUS = 0.010

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

from pxr import Usd, UsdGeom, UsdPhysics, Gf, PhysxSchema, UsdLux
from omni.isaac.core import World
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.articulations import ArticulationView
import omni.usd

from co_sim.engine import CoSimEngine
from co_sim.models import CoSimConfig, FrameState
from rod_skel_driver_test import SkeletonRodDriver


def _orthonormalize(R: np.ndarray) -> np.ndarray:
    u, _, vt = np.linalg.svd(R)
    Rn = u @ vt
    if np.linalg.det(Rn) < 0.0:
        u[:, -1] *= -1.0
        Rn = u @ vt
    return Rn


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


def build_wire_engine(frame_initial_state, isaac_dt):
    cfg = CoSimConfig(
        base_length=float(args.wire_base_length),
        n_elem=int(args.wire_n_elem),
        base_radius=float(args.wire_base_radius),
        py_dt=float(args.py_dt),
        isaac_dt=float(isaac_dt),
        final_time=1e9,
        render=False,
        joint_k=500,
        joint_nu=20.0,
        joint_kt=10.0,
        joint_nut=0.0,
        density=500.0,
        youngs_modulus=1.0e5,
        
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
    engine = CoSimEngine(config=cfg, frame_initial_state=frame_initial_state)
    return engine


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
            # Remove material binding if present so displayColor takes effect.
            rel = prim.GetRelationship("material:binding")
            if rel and rel.IsValid():
                rel.ClearTargets(True)


def main():
    ur5e_usd_path = _resolve_ur5e_usd_path()
    print(f"[Replay] UR5e USD: {ur5e_usd_path}")
    print(f"[Replay] Trajectory: {traj_path}")
    if manual_joint_target is not None:
        print(f"[Replay] One-shot joint target [deg]: {np.degrees(manual_joint_target)}")
    frame_limit = len(times) if args.max_frames <= 0 else min(int(args.max_frames), len(times))
    print(f"[Replay] Frames: {frame_limit}/{len(times)}  Duration: {times[min(frame_limit - 1, len(times)-1)]:.3f}s")

    sim_dt = 1.0 / float(args.sim_fps)
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
        "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
        "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
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

    if args.one_shot:
        for _ in range(args.settle_steps):
            robot_view.set_joint_position_targets(jp)
            world.step(render=not args.headless)
        tip_pos, tip_R = get_prim_world_pose(stage, stick_tip_path)
        if tip_pos is None:
            print("[Replay] One-shot done, but stick tip pose is unavailable.")
        else:
            tip_rotvec = _rotation_matrix_to_rotvec(tip_R)
            print(f"[Replay] Stick tip world xyz [m]: {tip_pos}")
            print(f"[Replay] Stick tip world xyz [mm]: {tip_pos * 1000.0}")
            print(f"[Replay] Stick tip world rotvec [rad]: {tip_rotvec}")
        return

    frame_init = FrameState(
        position=ee_pos0,
        director=ee_dir0,
        velocity=np.zeros(3, dtype=np.float64),
        acceleration=np.zeros(3, dtype=np.float64),
        omega=np.zeros(3, dtype=np.float64),
        alpha=np.zeros(3, dtype=np.float64),
    )
    wire_driver = build_wire_driver(stage)
    wire_engine = build_wire_engine(frame_init, sim_dt)
    kin_state = {
        "position": np.asarray(frame_init.position, dtype=np.float64),
        "director": np.asarray(frame_init.director, dtype=np.float64),
        "velocity": np.zeros(3, dtype=np.float64),
        "omega": np.zeros(3, dtype=np.float64),
    }

    snap0 = wire_engine.snapshot()
    wire_driver.update_skeleton(snap0.rod_position, snap0.rod_director, time_code=None)

    frame_idx = 0
    episode_count = 0
    t_wall_prev = _time.perf_counter()
    while simulation_app.is_running():
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

        if not args.headless and frame_idx > 0:
            desired_dt = (times[frame_idx] - times[frame_idx - 1]) / float(args.playback_speed)
            now = _time.perf_counter()
            elapsed = now - t_wall_prev
            t_wall_prev = now
            to_sleep = desired_dt - elapsed
            if to_sleep > 0.0:
                _time.sleep(to_sleep)
        else:
            t_wall_prev = _time.perf_counter()

        frame_idx += 1
        if frame_idx >= frame_limit:
            episode_count += 1
            frame_idx = 0

            # Hard reset: snap robot and wire to episode start state.
            jp[:, :NUM_ROBOT_DOFS] = joint_positions[0]
            robot_view.set_joint_positions(jp)
            robot_view.set_joint_velocities(np.zeros((1, ndof), dtype=np.float32))
            robot_view.set_joint_position_targets(jp)
            for _ in range(2):
                world.step(render=not args.headless)

            ee_pos0, ee_dir0 = get_prim_world_pose(stage, stick_tip_path)
            if ee_pos0 is not None:
                frame_init = FrameState(
                    position=ee_pos0,
                    director=ee_dir0,
                    velocity=np.zeros(3, dtype=np.float64),
                    acceleration=np.zeros(3, dtype=np.float64),
                    omega=np.zeros(3, dtype=np.float64),
                    alpha=np.zeros(3, dtype=np.float64),
                )
                wire_engine = build_wire_engine(frame_init, sim_dt)
                kin_state = {
                    "position": np.asarray(frame_init.position, dtype=np.float64),
                    "director": np.asarray(frame_init.director, dtype=np.float64),
                    "velocity": np.zeros(3, dtype=np.float64),
                    "omega": np.zeros(3, dtype=np.float64),
                }
                snap_reset = wire_engine.snapshot()
                wire_driver.update_skeleton(snap_reset.rod_position, snap_reset.rod_director, time_code=None)

            t_wall_prev = _time.perf_counter()
            if episode_count <= 5 or (episode_count % 20 == 0):
                print(f"[Replay] Episode {episode_count} reset to frame 0.")

    ee_pos, _ = get_prim_world_pose(stage, stick_tip_path)
    if ee_pos is None:
        print(f"[Replay] Done. Episodes={episode_count}")
    else:
        print(f"[Replay] Done. Episodes={episode_count}  Final stick tip={ee_pos}")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
