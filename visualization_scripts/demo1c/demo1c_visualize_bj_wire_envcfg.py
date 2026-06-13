#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Demo1C wire visualization using the same BJ wire construction as WireSwingBallJointEnv.

Compared to demo1c_visualize_bj_wire.py, this script keeps the same replay/export flow
but constructs the wire chain with the same parameters and D6-joint setup used in:
- RL_Demo/RL/envs/wire_swing_bj_env.py
- RL_Demo/conf/task/wire_swing_bj.yaml
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import numpy.testing  # Keep numpy.testing bound before Kit mutates import paths.

_REPO_ROOT = Path(__file__).resolve().parents[2]

parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true")
parser.add_argument("--physics_gpu", type=int, default=0)
parser.add_argument(
    "--traj_npz",
    type=str,
    default=str(_REPO_ROOT / "visualization_scripts" / "data" / "rope_demo_02212026_1.2Hz_18deg_240Hz.npz"),
    help="NPZ containing endpoint trajectory in key `pos` with shape (T,3).",
)
parser.add_argument(
    "--task-config",
    type=str,
    default=str(_REPO_ROOT / "RL_Demo" / "conf" / "task" / "wire_swing_bj.yaml"),
    help="Task config YAML used for BJ wire parameters.",
)
parser.add_argument("--warmup_steps", type=int, default=120, help="Hold moving endpoint for N steps before replay.")
parser.add_argument("--max_steps", type=int, default=0, help="Stop after N physics steps (0 = run forever).")
parser.add_argument("--ball_offset_z", type=float, default=0.003, help="Z offset added to all visualization balls.")
parser.add_argument("--ball_size_scale", type=float, default=2.0, help="Scale factor for visualization ball radii.")
args, _ = parser.parse_known_args()

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from isaacsim import SimulationApp

simulation_app = SimulationApp(
    {
        "headless": args.headless,
        "physics_gpu": args.physics_gpu,
        "active_gpu": args.physics_gpu,
    }
)

import omni.usd
from omni.isaac.core import World
from omni.isaac.core.prims import RigidPrimView
from omni.physx.scripts import physicsUtils
from pxr import Gf, PhysxSchema, UsdGeom, UsdLux, UsdPhysics, UsdShade, Usd

PHYSICS_DT = 1.0 / 240.0
WIRE_ROOT = "/World/BallJointWireDemo1C"
TARGET_BALL_PATH = "/World/TargetBall"
TRACK_BALLS_ROOT = "/World/TrackBalls"
FIRST_ANCHOR_PATH = f"{WIRE_ROOT}/first_anchor"
LAST_ANCHOR_PATH = f"{WIRE_ROOT}/last_anchor"
DEMO1C_WIRE_LENGTH = 2.06
DEMO1C_NUM_LINKS = 41
TASK_PHYS_DT = 0.002
TASK_WIRE_RADIUS = 0.00635
TASK_BJ_JOINT_DAMPING = 0.03
TASK_BJ_JOINT_STIFFNESS = 0.0
TASK_BJ_JOINT_DRIVE_TYPE = "force"
TASK_BJ_LINK_LINEAR_DAMPING = 0.05
TASK_BJ_LINK_ANGULAR_DAMPING = 0.01
TASK_BJ_MAT_STATIC_FRICTION = 0.03
TASK_BJ_MAT_DYNAMIC_FRICTION = 0.05
TASK_BJ_MAT_RESTITUTION = 0.01
TASK_BJ_COLLISION_REST_OFFSET = 0
TASK_BJ_COLLISION_CONTACT_OFFSET = 0
TASK_BJ_LINK_DENSITY = 1000.0
TASK_BJ_ATTACH_STIFFNESS = 2000.0
TASK_BJ_ATTACH_DAMPING = 20
TASK_BJ_ATTACH_ROT_STIFFNESS = 0.01
TASK_BJ_ATTACH_ROT_DAMPING = 0.01


def _resolve_task_cfg_path(path: str) -> Path:
    p = Path(path).expanduser().resolve()
    if not p.is_file():
        raise FileNotFoundError(f"Task config not found: {p}")
    return p


def load_endpoint_trajectory(npz_path: str):
    with np.load(npz_path, allow_pickle=True) as data:
        if "pos" in data:
            pos = np.asarray(data["pos"], dtype=np.float64)
        elif "position" in data:
            arr = np.asarray(data["position"], dtype=np.float64)
            if arr.ndim == 2 and arr.shape[1] >= 3:
                pos = arr[:, :3]
            elif arr.ndim == 3 and arr.shape[1] == 3:
                pos = arr[:, :, 0].T
            elif arr.ndim == 3 and arr.shape[2] == 3:
                pos = arr[:, 0, :]
            else:
                raise RuntimeError(f"Unsupported `position` shape: {arr.shape}")
        else:
            raise RuntimeError(f"{npz_path} missing key `pos` (or compatible `position`).")

        if pos.ndim != 2 or pos.shape[1] != 3:
            raise RuntimeError(f"`pos` must have shape (T,3), got {pos.shape}")

        track_pos = None
        if "track_pos" in data:
            tp = np.asarray(data["track_pos"], dtype=np.float64)
            if tp.ndim == 3 and tp.shape[2] == 3 and tp.shape[0] == pos.shape[0]:
                track_pos = tp
            else:
                raise RuntimeError(
                    f"`track_pos` must have shape (T,N,3) with T={pos.shape[0]}, got {tp.shape}"
                )
            # Include target position as part of each tracked frame.
            track_pos = np.concatenate((track_pos, pos[:, None, :]), axis=1)

        t = None
        if "time" in data:
            t_arr = np.asarray(data["time"], dtype=np.float64).reshape(-1)
            if t_arr.size == pos.shape[0]:
                t = t_arr

    return pos, t, track_pos


def sample_pos_at_time(pos: np.ndarray, time_arr: np.ndarray | None, sim_t: float, step_idx: int) -> np.ndarray:
    if time_arr is None:
        i = min(step_idx, pos.shape[0] - 1)
        return pos[i]

    if sim_t <= float(time_arr[0]):
        return pos[0]
    if sim_t >= float(time_arr[-1]):
        return pos[-1]

    j = int(np.searchsorted(time_arr, sim_t, side="right"))
    i0 = max(0, j - 1)
    i1 = min(j, time_arr.size - 1)
    t0 = float(time_arr[i0])
    t1 = float(time_arr[i1])
    if t1 <= t0:
        return pos[i0]
    a = (sim_t - t0) / (t1 - t0)
    return (1.0 - a) * pos[i0] + a * pos[i1]


def sample_track_pos_at_time(
    track_pos: np.ndarray,
    time_arr: np.ndarray | None,
    sim_t: float,
    step_idx: int,
) -> np.ndarray:
    if time_arr is None:
        i = min(step_idx, track_pos.shape[0] - 1)
        return track_pos[i]

    if sim_t <= float(time_arr[0]):
        return track_pos[0]
    if sim_t >= float(time_arr[-1]):
        return track_pos[-1]

    j = int(np.searchsorted(time_arr, sim_t, side="right"))
    i0 = max(0, j - 1)
    i1 = min(j, time_arr.size - 1)
    t0 = float(time_arr[i0])
    t1 = float(time_arr[i1])
    if t1 <= t0:
        return track_pos[i0]
    a = (sim_t - t0) / (t1 - t0)
    return (1.0 - a) * track_pos[i0] + a * track_pos[i1]


def create_env_style_bj_wire(
    stage,
    root_path: str,
    first_node: np.ndarray,
    num_links: int,
    wire_length: float,
    wire_radius: float,
):
    UsdGeom.Scope.Define(stage, root_path)

    bj_joint_damping = float(TASK_BJ_JOINT_DAMPING)
    bj_joint_stiffness = float(TASK_BJ_JOINT_STIFFNESS)
    bj_joint_drive_type = str(TASK_BJ_JOINT_DRIVE_TYPE).strip().lower()
    if bj_joint_drive_type not in ("force", "acceleration"):
        bj_joint_drive_type = "force"
    bj_link_linear_damping = float(TASK_BJ_LINK_LINEAR_DAMPING)
    bj_link_angular_damping = float(TASK_BJ_LINK_ANGULAR_DAMPING)
    bj_mat_static_friction = float(TASK_BJ_MAT_STATIC_FRICTION)
    bj_mat_dynamic_friction = float(TASK_BJ_MAT_DYNAMIC_FRICTION)
    bj_mat_restitution = float(TASK_BJ_MAT_RESTITUTION)
    bj_collision_rest_offset = float(TASK_BJ_COLLISION_REST_OFFSET)
    bj_collision_contact_offset = float(TASK_BJ_COLLISION_CONTACT_OFFSET)
    bj_link_density = float(TASK_BJ_LINK_DENSITY)

    mat_path = f"{root_path}/WireMaterial"
    UsdShade.Material.Define(stage, mat_path)
    mat = UsdPhysics.MaterialAPI.Apply(stage.GetPrimAtPath(mat_path))
    mat.CreateStaticFrictionAttr().Set(float(bj_mat_static_friction))
    mat.CreateDynamicFrictionAttr().Set(float(bj_mat_dynamic_friction))
    mat.CreateRestitutionAttr().Set(float(bj_mat_restitution))

    link_half = float(wire_length) / (2.0 * float(num_links))
    link_len = 2.0 * link_half
    x0, y0, z0 = map(float, first_node)

    link_paths = []
    for i in range(num_links):
        path = f"{root_path}/link_{i}"
        cap = UsdGeom.Capsule.Define(stage, path)
        cap.CreateHeightAttr(float(link_len))
        cap.CreateRadiusAttr(float(wire_radius))
        cap.CreateAxisAttr("Z")
        cap.CreateDisplayColorAttr().Set([Gf.Vec3f(0.92, 0.48, 0.12)])
        # Initial placement: stack along -Z (physics will settle anyway).
        UsdGeom.Xformable(cap.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(x0 - (i + 0.5) * link_len, y0, z0)
)

        prim = cap.GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(prim)
        UsdPhysics.CollisionAPI.Apply(prim)
        UsdPhysics.MassAPI.Apply(prim).CreateDensityAttr().Set(float(bj_link_density))

        rb = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
        rb.CreateLinearDampingAttr().Set(float(bj_link_linear_damping))
        rb.CreateAngularDampingAttr().Set(float(bj_link_angular_damping))
        col = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        col.CreateRestOffsetAttr().Set(float(bj_collision_rest_offset))
        col.CreateContactOffsetAttr().Set(float(bj_collision_contact_offset))

        physicsUtils.add_physics_material_to_prim(stage, prim, mat_path)
        link_paths.append(path)

    # Match WireSwingBallJointEnv: use D6 joints + locked translation + per-axis angular drives.
    for i in range(num_links - 1):
        joint_path = f"{root_path}/joint_{i}"
        joint = UsdPhysics.Joint.Define(stage, joint_path)
        joint.GetBody0Rel().SetTargets([link_paths[i]])
        joint.GetBody1Rel().SetTargets([link_paths[i + 1]])
        # Same convention as env: link_i +Z <-> link_{i+1} -Z.
        joint.CreateLocalPos0Attr().Set(Gf.Vec3f(0.0, 0.0, float(link_half)))
        joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, float(-link_half)))
        joint.CreateLocalRot0Attr().Set(Gf.Quatf(1.0))
        joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0))
        joint.CreateCollisionEnabledAttr().Set(False)

        jprim = joint.GetPrim()
        for ax in ("transX", "transY", "transZ"):
            limit = UsdPhysics.LimitAPI.Apply(jprim, ax)
            limit.CreateLowAttr(0.0)
            limit.CreateHighAttr(0.0)

        for ax in ("rotX", "rotY", "rotZ"):
            drive = UsdPhysics.DriveAPI.Apply(jprim, ax)
            drive.CreateTypeAttr(str(bj_joint_drive_type))
            drive.CreateDampingAttr(float(bj_joint_damping))
            drive.CreateStiffnessAttr(float(bj_joint_stiffness))

    return link_paths, link_half


def set_kinematic(stage, prim_path: str):
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise RuntimeError(f"Invalid prim for kinematic setup: {prim_path}")
    rb = UsdPhysics.RigidBodyAPI(prim)
    if not rb:
        rb = UsdPhysics.RigidBodyAPI.Apply(prim)
    rb.CreateKinematicEnabledAttr().Set(True)


def create_kinematic_anchor(
    stage,
    prim_path: str,
    position_xyz: np.ndarray,
    quat_wxyz: np.ndarray,
    radius: float,
    color_rgb: tuple[float, float, float],
):
    sphere = UsdGeom.Sphere.Define(stage, prim_path)
    sphere.CreateRadiusAttr().Set(float(radius))
    sphere.CreateDisplayColorAttr().Set([Gf.Vec3f(*[float(v) for v in color_rgb])])
    set_translate_op(stage, prim_path, np.asarray(position_xyz, dtype=np.float64))
    set_orient_op_wxyz(stage, prim_path, np.asarray(quat_wxyz, dtype=np.float64))

    prim = sphere.GetPrim()
    rb = UsdPhysics.RigidBodyAPI.Apply(prim)
    rb.CreateKinematicEnabledAttr().Set(True)
    return prim_path


def create_attach_joint(
    stage,
    joint_path: str,
    anchor_path: str,
    link_path: str,
    link_local_pos: np.ndarray,
    drive_rotation: bool = True,
):
    joint = UsdPhysics.Joint.Define(stage, joint_path)
    joint.GetBody0Rel().SetTargets([anchor_path])
    joint.GetBody1Rel().SetTargets([link_path])
    joint.CreateLocalPos0Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    joint.CreateLocalPos1Attr().Set(
        Gf.Vec3f(
            float(link_local_pos[0]),
            float(link_local_pos[1]),
            float(link_local_pos[2]),
        )
    )
    joint.CreateLocalRot0Attr().Set(Gf.Quatf(1.0))
    joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0))
    joint.CreateCollisionEnabledAttr().Set(False)

    drive_type = str(TASK_BJ_JOINT_DRIVE_TYPE).strip().lower()
    if drive_type not in ("force", "acceleration"):
        drive_type = "force"
    trans_k = float(TASK_BJ_ATTACH_STIFFNESS)
    trans_d = float(TASK_BJ_ATTACH_DAMPING)
    rot_k = float(TASK_BJ_ATTACH_ROT_STIFFNESS)
    rot_d = float(TASK_BJ_ATTACH_ROT_DAMPING)

    jprim = joint.GetPrim()
    for ax in ("transX", "transY", "transZ"):
        drive = UsdPhysics.DriveAPI.Apply(jprim, ax)
        drive.CreateTypeAttr(str(drive_type))
        drive.CreateStiffnessAttr(float(trans_k))
        drive.CreateDampingAttr(float(trans_d))
    if drive_rotation:
        for ax in ("rotX", "rotY", "rotZ"):
            drive = UsdPhysics.DriveAPI.Apply(jprim, ax)
            drive.CreateTypeAttr(str(drive_type))
            drive.CreateStiffnessAttr(float(rot_k))
            drive.CreateDampingAttr(float(rot_d))


def set_orient_op_wxyz(stage, prim_path: str, quat_wxyz: np.ndarray):
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise RuntimeError(f"Invalid prim for orient op: {prim_path}")

    xf = UsdGeom.Xformable(prim)
    translate_op = None
    orient_op = None
    for op in xf.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate and translate_op is None:
            translate_op = op
        if op.GetOpType() == UsdGeom.XformOp.TypeOrient and orient_op is None:
            orient_op = op

    if orient_op is None:
        orient_op = xf.AddOrientOp()

    if translate_op is not None:
        xf.SetXformOpOrder([translate_op, orient_op], resetXformStack=False)

    w, x, y, z = [float(v) for v in np.asarray(quat_wxyz, dtype=np.float64).reshape(4)]
    if orient_op.GetPrecision() == UsdGeom.XformOp.PrecisionDouble:
        orient_op.Set(Gf.Quatd(w, Gf.Vec3d(x, y, z)))
    else:
        orient_op.Set(Gf.Quatf(w, Gf.Vec3f(x, y, z)))


def set_translate_op(stage, prim_path: str, pos_xyz: np.ndarray):
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise RuntimeError(f"Invalid prim for translate op: {prim_path}")

    xf = UsdGeom.Xformable(prim)
    translate_op = None
    for op in xf.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            translate_op = op
            break
    if translate_op is None:
        translate_op = xf.AddTranslateOp()

    x, y, z = [float(v) for v in np.asarray(pos_xyz, dtype=np.float64).reshape(3)]
    if translate_op.GetPrecision() == UsdGeom.XformOp.PrecisionDouble:
        translate_op.Set(Gf.Vec3d(x, y, z))
    else:
        translate_op.Set(Gf.Vec3f(x, y, z))


def get_local_z_world_dir(stage, prim_path: str) -> np.ndarray:
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        return np.array([np.nan, np.nan, np.nan], dtype=np.float64)
    tf = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    R = np.array(tf.ExtractRotationMatrix(), dtype=np.float64)
    z_dir = R @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
    n = np.linalg.norm(z_dir)
    return z_dir if n < 1.0e-12 else (z_dir / n)


def set_ground_plane_height(stage, z_world: float):
    ground_prim = None
    for path in ("/World/defaultGroundPlane", "/World/GroundPlane", "/World/groundPlane"):
        p = stage.GetPrimAtPath(path)
        if p.IsValid():
            ground_prim = p
            break

    if ground_prim is None:
        for p in stage.Traverse():
            if "ground" in p.GetName().lower() and p.IsA(UsdGeom.Xformable):
                ground_prim = p
                break

    if ground_prim is None:
        return None

    xf = UsdGeom.Xformable(ground_prim)
    translate_op = None
    for op in xf.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            translate_op = op
            break
    if translate_op is None:
        translate_op = xf.AddTranslateOp()
        x, y = 0.0, 0.0
    else:
        t = translate_op.Get()
        if t is None:
            x, y = 0.0, 0.0
        else:
            x, y = float(t[0]), float(t[1])

    translate_op.Set(Gf.Vec3d(x, y, float(z_world)))
    return ground_prim.GetPath().pathString


def create_target_ball(stage, prim_path: str, radius: float, position: np.ndarray):
    sphere = UsdGeom.Sphere.Define(stage, prim_path)
    sphere.CreateRadiusAttr().Set(float(radius))
    sphere.CreateDisplayColorAttr().Set([Gf.Vec3f(0.95, 0.12, 0.12)])
    set_translate_op(stage, prim_path, np.asarray(position, dtype=np.float64))


def create_track_balls(stage, root_path: str, positions: np.ndarray, radius: float):
    UsdGeom.Scope.Define(stage, root_path)
    ball_paths: list[str] = []
    n = int(positions.shape[0])
    for i in range(n):
        p = np.asarray(positions[i], dtype=np.float64)
        path = f"{root_path}/pt_{i:03d}"
        sphere = UsdGeom.Sphere.Define(stage, path)
        sphere.CreateRadiusAttr().Set(float(radius))
        sphere.CreateDisplayColorAttr().Set([Gf.Vec3f(0.90, 0.10, 0.10)])
        set_translate_op(stage, path, p)
        ball_paths.append(path)
    return ball_paths


def create_joint_balls(stage, link_paths: list[str], link_half: float, radius: float):
    """Create one blue marker per internal spherical joint.

    Joint i is between link_i and link_(i+1), anchored at link_i local +Z end.
    """
    ball_paths: list[str] = []
    joint_local_pos = np.array([0.0, 0.0, float(link_half)], dtype=np.float64)
    n_links = len(link_paths)
    for i in range(max(0, n_links - 1)):
        path = f"{link_paths[i]}/joint_ball"
        sphere = UsdGeom.Sphere.Define(stage, path)
        sphere.CreateRadiusAttr().Set(float(radius))
        sphere.CreateDisplayColorAttr().Set([Gf.Vec3f(0.12, 0.35, 0.95)])
        set_translate_op(stage, path, joint_local_pos)
        ball_paths.append(path)
    return ball_paths, joint_local_pos


def get_prim_world_position(stage, prim_path: str) -> np.ndarray:
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise RuntimeError(f"Invalid prim for world-position query: {prim_path}")
    tf = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    t = tf.ExtractTranslation()
    return np.array([float(t[0]), float(t[1]), float(t[2])], dtype=np.float64)


def get_marker_world_positions(stage, marker_paths: list[str]) -> np.ndarray:
    if len(marker_paths) == 0:
        return np.zeros((0, 3), dtype=np.float64)
    out = np.zeros((len(marker_paths), 3), dtype=np.float64)
    for i, pth in enumerate(marker_paths):
        out[i] = get_prim_world_position(stage, pth)
    return out


def add_ball_offset(position_xyz: np.ndarray, z_offset: float) -> np.ndarray:
    p = np.asarray(position_xyz, dtype=np.float64).copy()
    p[2] += float(z_offset)
    return p


def get_wire_node_positions(stage, link_paths: list[str], link_half: float) -> np.ndarray:
    """Return wire node positions in chain order with shape (num_links+1, 3)."""
    n_links = len(link_paths)
    if n_links <= 0:
        raise RuntimeError("Wire has no links; cannot compute node positions.")

    plus_end = np.zeros((n_links, 3), dtype=np.float64)   # local +Z end
    minus_end = np.zeros((n_links, 3), dtype=np.float64)  # local -Z end
    z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    half = float(link_half)

    for i, path in enumerate(link_paths):
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid():
            raise RuntimeError(f"Invalid wire link prim: {path}")
        tf = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        t = tf.ExtractTranslation()
        R = np.array(tf.ExtractRotationMatrix(), dtype=np.float64)
        c = np.array([float(t[0]), float(t[1]), float(t[2])], dtype=np.float64)
        z_dir = R @ z_axis
        n = float(np.linalg.norm(z_dir))
        if n > 1.0e-12:
            z_dir = z_dir / n
        plus_end[i] = c + z_dir * half
        minus_end[i] = c - z_dir * half

    # Match env convention:
    # node0 = link_0 local -Z; internal node i = link_(i-1) local +Z; last node = link_last local +Z.
    nodes = np.zeros((n_links + 1, 3), dtype=np.float64)
    nodes[0] = minus_end[0]
    for i in range(1, n_links):
        nodes[i] = plus_end[i - 1]
    nodes[n_links] = plus_end[-1]
    return nodes


def main():
    task_cfg_path = _resolve_task_cfg_path(args.task_config)

    sim_dt = float(TASK_PHYS_DT)
    wire_radius = float(TASK_WIRE_RADIUS)
    # Match the original demo1c wire geometry so the wire can sag instead of staying taut.
    wire_length = float(DEMO1C_WIRE_LENGTH)
    num_links = int(DEMO1C_NUM_LINKS)
    if sim_dt <= 0.0:
        raise ValueError("task.phys_dt must be > 0")
    if wire_radius <= 0.0:
        raise ValueError("task.wire_base_radius must be > 0")
    if wire_length <= 0.0:
        raise ValueError("task.wire_base_length must be > 0")

    endpoint_pos, endpoint_t, track_pos = load_endpoint_trajectory(args.traj_npz)

    # Pinned endpoint position for link_0 local -Z end (env convention).
    first_node = np.array([0.0, 0.0, wire_radius], dtype=np.float64)

    world = World(physics_dt=sim_dt, rendering_dt=sim_dt, stage_units_in_meters=1.0)
    stage = omni.usd.get_context().get_stage()
    world.scene.add_default_ground_plane()
    ground_z = -wire_radius - 1.0e-3
    ground_path = set_ground_plane_height(stage, ground_z)

    ps = UsdPhysics.Scene.Define(stage, "/physicsScene")
    ps.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
    ps.CreateGravityMagnitudeAttr().Set(9.81)
    PhysxSchema.PhysxSceneAPI.Apply(stage.GetPrimAtPath("/physicsScene")).CreateEnableGPUDynamicsAttr().Set(True)

    UsdLux.DomeLight.Define(stage, "/World/DomeLight").CreateIntensityAttr().Set(1000.0)
    UsdLux.DistantLight.Define(stage, "/World/DistantLight").CreateIntensityAttr().Set(3000.0)

    # Visual reference: this ball follows NPZ `pos` strictly at sim-time.
    strict_pos0 = sample_pos_at_time(endpoint_pos, endpoint_t, 0.0, 0)
    target_ball_radius = max(0.008, 1.5 * wire_radius) * float(args.ball_size_scale)
    track_ball_radius = max(0.0035, 0.75 * wire_radius) * float(args.ball_size_scale)
    joint_ball_radius = max(0.0025, 0.45 * wire_radius) * float(args.ball_size_scale)

    create_target_ball(
        stage,
        TARGET_BALL_PATH,
        radius=target_ball_radius,
        position=add_ball_offset(strict_pos0, float(args.ball_offset_z)),
    )
    track_ball_paths = []
    if track_pos is not None and track_pos.shape[1] > 0:
        track0 = sample_track_pos_at_time(track_pos, endpoint_t, 0.0, 0)
        track0 = track0.copy()
        track0[:, 2] += float(args.ball_offset_z)
        track_ball_paths = create_track_balls(
            stage,
            TRACK_BALLS_ROOT,
            track0,
            radius=track_ball_radius,
        )

    link_paths, link_half = create_env_style_bj_wire(
        stage=stage,
        root_path=WIRE_ROOT,
        first_node=first_node,
        num_links=num_links,
        wire_length=wire_length,
        wire_radius=wire_radius,
    )
    joint_ball_paths, joint_ball_local_pos = create_joint_balls(
        stage,
        link_paths,
        link_half,
        radius=joint_ball_radius,
    )

    first_link = link_paths[0]
    last_link = link_paths[-1]
    mid_link = link_paths[len(link_paths) // 2]

    first_view = RigidPrimView(first_link, name="first_link")
    last_view = RigidPrimView(last_link, name="last_link")
    mid_view = RigidPrimView(mid_link, name="mid_link")
    world.scene.add(first_view)
    world.scene.add(last_view)
    world.scene.add(mid_view)

    # --- Endpoint constraints ---
    # Keep link_0 kinematic along +X. Drive the last endpoint position with a kinematic
    # anchor, but do not drive last-link rotation.
    q_first_wxyz = np.array([0.70710677, 0.0, -0.70710677, 0.0], dtype=np.float32)  # -90deg about Y
    q_last_wxyz = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    target_node0 = np.asarray(strict_pos0, dtype=np.float32)
    last_dir = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    set_kinematic(stage, first_link)
    set_orient_op_wxyz(stage, first_link, q_first_wxyz)
    first_dir_init = get_local_z_world_dir(stage, first_link).astype(np.float32)
    set_translate_op(
        stage,
        first_link,
        np.asarray(first_node, dtype=np.float32) + first_dir_init * float(link_half),
    )
    set_orient_op_wxyz(stage, last_link, q_last_wxyz)
    set_translate_op(
        stage,
        last_link,
        target_node0 - last_dir * float(link_half),
    )
    create_kinematic_anchor(
        stage,
        LAST_ANCHOR_PATH,
        position_xyz=target_node0,
        quat_wxyz=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        radius=max(0.5 * wire_radius, 0.004),
        color_rgb=(0.20, 0.40, 0.95),
    )
    create_attach_joint(
        stage,
        f"{WIRE_ROOT}/last_attach_joint",
        LAST_ANCHOR_PATH,
        last_link,
        np.array([0.0, 0.0, float(link_half)], dtype=np.float64),
        drive_rotation=False,
    )
    last_anchor_view = RigidPrimView(LAST_ANCHOR_PATH, name="last_anchor")
    world.scene.add(last_anchor_view)

    world.reset()

    # Re-apply endpoint poses after reset so the last endpoint stays on target.
    set_orient_op_wxyz(stage, first_link, q_first_wxyz)
    first_dir_init = get_local_z_world_dir(stage, first_link).astype(np.float32)
    first_center = np.asarray(first_node, dtype=np.float32) + first_dir_init * float(link_half)
    first_view.set_world_poses(
        positions=first_center.reshape(1, 3),
        orientations=q_first_wxyz.reshape(1, 4),
    )
    last_anchor_view.set_world_poses(
        positions=target_node0.reshape(1, 3),
    )
    set_translate_op(stage, first_link, first_center)
    set_translate_op(stage, LAST_ANCHOR_PATH, target_node0)

    # Read back the actual world direction of local +Z after initialization.
    first_dir = get_local_z_world_dir(stage, first_link).astype(np.float32)

    print(f"[Run] traj={args.traj_npz}, frames={endpoint_pos.shape[0]}, has_time={endpoint_t is not None}")
    print(f"[Run] task_config={task_cfg_path}")
    print(f"[Run] nodes={num_links + 1}, links={num_links}, length={wire_length:.3f}m, radius={wire_radius:.4f}m")
    print(
        f"[Run] joint_drive={TASK_BJ_JOINT_DRIVE_TYPE} "
        f"damping={TASK_BJ_JOINT_DAMPING:.3f} "
        f"stiffness={TASK_BJ_JOINT_STIFFNESS:.3f}"
    )
    print(
        f"[Run] link_damping=({TASK_BJ_LINK_LINEAR_DAMPING:.3f}, "
        f"{TASK_BJ_LINK_ANGULAR_DAMPING:.3f}) "
        f"mat_friction=({TASK_BJ_MAT_STATIC_FRICTION:.3f},"
        f"{TASK_BJ_MAT_DYNAMIC_FRICTION:.3f})"
    )
    print("[Run] endpoint_mode=kinematic_first_free_last_direction")
    print(f"[Run] sim_dt={sim_dt:.6f}s")
    print(f"[Run] warmup_steps={int(args.warmup_steps)}")
    if ground_path is not None:
        print(f"[Run] ground plane moved to z={ground_z:+.4f} at {ground_path}")
    print(f"[Run] target ball path={TARGET_BALL_PATH} follows NPZ pos strictly.")
    if track_ball_paths:
        print(f"[Run] track balls={len(track_ball_paths)} at {TRACK_BALLS_ROOT} follow NPZ track_pos.")
    if joint_ball_paths:
        print(
            f"[Run] wire joint balls={len(joint_ball_paths)} (blue), "
            "each parented to link_i at local position "
            f"({joint_ball_local_pos[0]:+.4f},{joint_ball_local_pos[1]:+.4f},{joint_ball_local_pos[2]:+.4f})"
        )
    print(f"[Run] export positions source=blue joint balls, count={len(joint_ball_paths)}")
    print(f"[Run] visualization balls z offset={float(args.ball_offset_z):+.3f} m")
    print(
        f"[Run] ball radii: target={target_ball_radius:.4f} m, "
        f"track={track_ball_radius:.4f} m, joint={joint_ball_radius:.4f} m "
        f"(scale={float(args.ball_size_scale):.2f})"
    )
    print(
        "[Run] pinned endpoint (link0 local -Z): "
        f"first=({first_node[0]:+.3f},{first_node[1]:+.3f},{first_node[2]:+.3f})"
    )
    print(
        "[Run] expect chain direction from pinned point ~(+1,0,0). "
        f"measured first_dir(local+Z)=({first_dir[0]:+.3f},{first_dir[1]:+.3f},{first_dir[2]:+.3f}) "
        f"=> chain_dir=first_dir=({first_dir[0]:+.3f},{first_dir[1]:+.3f},{first_dir[2]:+.3f})"
    )
    print("[Run] Running... (Ctrl+C to stop)")
    sys.stdout.flush()

    out_npz = Path(args.traj_npz).expanduser().resolve().with_suffix("")
    out_npz = out_npz.with_name(out_npz.name + "_wire_joint_positions.npz")
    recorded_positions: list[np.ndarray] = []
    recorded_time: list[float] = []
    recorded_target_pos: list[np.ndarray] = []
    recorded_track_pos: list[np.ndarray] | None = [] if track_pos is not None else None

    warmup_steps = int(args.warmup_steps)
    steps = 0
    try:
        while simulation_app.is_running():
            if steps < warmup_steps:
                replay_step = 0
                replay_t = 0.0
            else:
                replay_step = steps - warmup_steps
                replay_t = replay_step * sim_dt
            record_this_step = steps >= warmup_steps

            target_node = sample_pos_at_time(endpoint_pos, endpoint_t, replay_t, replay_step)
            strict_pos = target_node
            track_frame = None
            if track_pos is not None:
                track_frame = sample_track_pos_at_time(track_pos, endpoint_t, replay_t, replay_step)

            target_node_f32 = np.asarray(target_node, dtype=np.float32)
            first_center = np.asarray(first_node, dtype=np.float32) + first_dir * float(link_half)
            first_view.set_world_poses(
                positions=first_center.reshape(1, 3),
                orientations=q_first_wxyz.reshape(1, 4),
            )
            last_anchor_view.set_world_poses(
                positions=target_node_f32.reshape(1, 3),
            )
            set_orient_op_wxyz(stage, first_link, q_first_wxyz)
            set_translate_op(stage, first_link, first_center)
            set_translate_op(stage, LAST_ANCHOR_PATH, target_node)

            set_translate_op(stage, TARGET_BALL_PATH, add_ball_offset(strict_pos, float(args.ball_offset_z)))
            if track_frame is not None:
                for i, pth in enumerate(track_ball_paths):
                    set_translate_op(stage, pth, add_ball_offset(track_frame[i], float(args.ball_offset_z)))

            world.step(render=not args.headless)
            steps += 1
            if record_this_step:
                marker_positions = get_marker_world_positions(stage, joint_ball_paths)
                recorded_time.append(float(replay_t))
                recorded_target_pos.append(np.asarray(target_node, dtype=np.float64).copy())
                if recorded_track_pos is not None and track_frame is not None:
                    recorded_track_pos.append(np.asarray(track_frame, dtype=np.float64).copy())
                recorded_positions.append(marker_positions)

            if steps == 1:
                pos0, _ = first_view.get_world_poses()
                c0 = np.asarray(pos0[0], dtype=np.float64)
                d0 = get_local_z_world_dir(stage, first_link)
                e0 = c0 - d0 * float(link_half)  # pinned -Z endpoint
                err = np.linalg.norm(e0 - np.asarray(first_node, dtype=np.float64))
                chain_dir = d0
                print(
                    f"[Debug] link0_pinned_end=({e0[0]:+.4f},{e0[1]:+.4f},{e0[2]:+.4f}) "
                    f"target=({first_node[0]:+.4f},{first_node[1]:+.4f},{first_node[2]:+.4f}) "
                    f"dir(local+Z)=({d0[0]:+.4f},{d0[1]:+.4f},{d0[2]:+.4f}) "
                    f"chain_dir=({chain_dir[0]:+.4f},{chain_dir[1]:+.4f},{chain_dir[2]:+.4f}) "
                    f"err={err:.3e}"
                )
                sys.stdout.flush()

            if steps % 240 == 0:
                mid_pos, _ = mid_view.get_world_poses()
                p = mid_pos[0]
                print(
                    f"[Step {steps:06d}] target=({target_node[0]:+.3f},{target_node[1]:+.3f},{target_node[2]:+.3f}) "
                    f"mid=({p[0]:+.3f},{p[1]:+.3f},{p[2]:+.3f})"
                )
                sys.stdout.flush()

            if args.max_steps > 0 and steps >= args.max_steps:
                break
    except KeyboardInterrupt:
        pass

    n_samples = len(recorded_positions)
    n_markers = len(joint_ball_paths)
    if n_samples > 0:
        positions_arr = np.asarray(recorded_positions, dtype=np.float64)
        time_arr = np.asarray(recorded_time, dtype=np.float64).reshape(-1)
        target_arr = np.asarray(recorded_target_pos, dtype=np.float64)
    else:
        positions_arr = np.zeros((0, n_markers, 3), dtype=np.float64)
        time_arr = np.zeros((0,), dtype=np.float64)
        target_arr = np.zeros((0, 3), dtype=np.float64)

    if positions_arr.shape[1:] != (n_markers, 3):
        raise RuntimeError(
            f"Unexpected positions shape {positions_arr.shape}; expected (T, {n_markers}, 3)."
        )
    if time_arr.shape != (n_samples,):
        raise RuntimeError(f"Unexpected time shape {time_arr.shape}; expected ({n_samples},).")
    if target_arr.shape != (n_samples, 3):
        raise RuntimeError(f"Unexpected target_pos shape {target_arr.shape}; expected ({n_samples}, 3).")
    if len(recorded_time) != n_samples or len(recorded_target_pos) != n_samples:
        raise RuntimeError(
            f"Inconsistent record lengths: positions={n_samples}, time={len(recorded_time)}, target={len(recorded_target_pos)}."
        )

    out_payload = {
        "positions": positions_arr,
        "time": time_arr,
        "target_pos": target_arr,
    }

    if recorded_track_pos is not None:
        if n_samples > 0:
            track_arr = np.asarray(recorded_track_pos, dtype=np.float64)
        else:
            track_arr = np.zeros((0, int(track_pos.shape[1]), 3), dtype=np.float64)
        if track_arr.ndim != 3 or track_arr.shape[0] != n_samples or track_arr.shape[2] != 3:
            raise RuntimeError(
                f"Unexpected track_pos shape {track_arr.shape}; expected ({n_samples}, N, 3)."
            )
        if len(recorded_track_pos) != n_samples:
            raise RuntimeError(
                f"Inconsistent track_pos length: positions={n_samples}, track_pos={len(recorded_track_pos)}."
            )
        out_payload["track_pos"] = track_arr

    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_npz, **out_payload)
    export_shapes = (
        f"positions={positions_arr.shape}, time={time_arr.shape}, target_pos={target_arr.shape}"
        + (f", track_pos={out_payload['track_pos'].shape}" if "track_pos" in out_payload else "")
    )
    print(f"[Export] npz={out_npz} {export_shapes}")
    sys.stdout.flush()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback

        print(f"\n[ERROR] {e}")
        traceback.print_exc()
        sys.stdout.flush()
    finally:
        simulation_app.close()
