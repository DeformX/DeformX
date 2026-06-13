#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualize a ball-joint wire with endpoint constraints:
- First endpoint (FREE end of link_0, i.e., local +Z end) fixed at `first_node`.
- Second endpoint follows `pos` from an NPZ trajectory.

Goal in this version:
- The wire should extend along world -X from the pinned endpoint.
  (Because we pin the FREE end, the chain extends from the opposite end => direction = - first_dir.
   So we set first_dir (world direction of local +Z) = +X, hence chain direction = -X.)
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
parser.add_argument("--num_nodes", type=int, default=41, help="Wire node count (must be >= 2).")
parser.add_argument("--wire_length", type=float, default=2.04, help="Wire total length in meters.")
parser.add_argument("--wire_radius", type=float, default=0.006)
parser.add_argument("--joint_damping", type=float, default=5.0, help="Ball-joint angular drive damping.")
parser.add_argument("--mat_static_friction", type=float, default=0.6, help="Wire material static friction.")
parser.add_argument("--mat_dynamic_friction", type=float, default=0.5, help="Wire material dynamic friction.")
parser.add_argument("--warmup_steps", type=int, default=120, help="Hold moving endpoint for N steps before replay.")
parser.add_argument("--max_steps", type=int, default=0, help="Stop after N physics steps (0 = run forever).")
parser.add_argument("--ball_offset_z", type=float, default=0.003, help="Z offset added to all visualization balls.")
parser.add_argument("--ball_size_scale", type=float, default=2.0, help="Scale factor for visualization ball radii.")
args, _ = parser.parse_known_args()

if args.num_nodes < 2:
    raise ValueError("--num_nodes must be >= 2")
if args.wire_length <= 0.0:
    raise ValueError("--wire_length must be > 0")
if args.wire_radius <= 0.0:
    raise ValueError("--wire_radius must be > 0")
if args.joint_damping < 0.0:
    raise ValueError("--joint_damping must be >= 0")
if args.mat_static_friction < 0.0:
    raise ValueError("--mat_static_friction must be >= 0")
if args.mat_dynamic_friction < 0.0:
    raise ValueError("--mat_dynamic_friction must be >= 0")

REPO_ROOT = Path(__file__).resolve().parents[1]
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
WIRE_ROOT_A = "/World/BallJointWireDemo1C_A"
WIRE_ROOT_B = "/World/BallJointWireDemo1C_B"
WIRE_B_Y_OFFSET = 2.0
TARGET_BALL_PATH_A = "/World/TargetBallA"
TARGET_BALL_PATH_B = "/World/TargetBallB"
TRACK_BALLS_ROOT_A = "/World/TrackBallsA"
TRACK_BALLS_ROOT_B = "/World/TrackBallsB"


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


def create_rigid_wire(
    stage,
    root_path: str,
    first_node: np.ndarray,
    num_links: int,
    wire_length: float,
    wire_radius: float,
    mat_static_friction: float,
    mat_dynamic_friction: float,
    joint_damping: float,
    initial_spacing_scale: float,
):
    UsdGeom.Scope.Define(stage, root_path)

    mat_path = f"{root_path}/WireMaterial"
    UsdShade.Material.Define(stage, mat_path)
    mat = UsdPhysics.MaterialAPI.Apply(stage.GetPrimAtPath(mat_path))
    mat.CreateStaticFrictionAttr().Set(float(mat_static_friction))
    mat.CreateDynamicFrictionAttr().Set(float(mat_dynamic_friction))
    mat.CreateRestitutionAttr().Set(0.0)

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
        # Initial placement differs between wires to compare sensitivity to initial state.
        UsdGeom.Xformable(cap.GetPrim()).AddTranslateOp().Set(
            Gf.Vec3d(x0 - (i + 0.5) * link_len * float(initial_spacing_scale), y0, z0)
        )

        prim = cap.GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(prim)
        UsdPhysics.CollisionAPI.Apply(prim)
        UsdPhysics.MassAPI.Apply(prim).CreateDensityAttr().Set(600.0)

        rb = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
        rb.CreateLinearDampingAttr().Set(0.6)
        rb.CreateAngularDampingAttr().Set(1.2)
        col = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        col.CreateRestOffsetAttr().Set(0.0)
        col.CreateContactOffsetAttr().Set(0.002)

        physicsUtils.add_physics_material_to_prim(stage, prim, mat_path)
        link_paths.append(path)

    for i in range(num_links - 1):
        joint_path = f"{root_path}/joint_{i}"
        joint = UsdPhysics.SphericalJoint.Define(stage, joint_path)
        joint.GetBody0Rel().SetTargets([link_paths[i]])
        joint.GetBody1Rel().SetTargets([link_paths[i + 1]])
        # Joint is at link_i local -Z end, and link_{i+1} local +Z end.
        joint.CreateLocalPos0Attr().Set(Gf.Vec3f(0.0, 0.0, float(-link_half)))
        joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, float(link_half)))
        joint.CreateLocalRot0Attr().Set(Gf.Quatf(1.0))
        joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0))

        drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "angular")
        drive.CreateTypeAttr("force")
        drive.CreateDampingAttr(float(joint_damping))
        drive.CreateStiffnessAttr(0.0)

    return link_paths, link_half


def set_kinematic(stage, prim_path: str):
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise RuntimeError(f"Invalid prim for kinematic setup: {prim_path}")
    rb = UsdPhysics.RigidBodyAPI(prim)
    if not rb:
        rb = UsdPhysics.RigidBodyAPI.Apply(prim)
    rb.CreateKinematicEnabledAttr().Set(True)


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


def create_joint_balls(
    stage,
    link_paths: list[str],
    link_half: float,
    radius: float,
    color: Gf.Vec3f | None = None,
):
    """Create one blue marker per internal spherical joint.

    Joint i is between link_i and link_(i+1), anchored at link_i local -Z end.
    """
    ball_paths: list[str] = []
    marker_color = color if color is not None else Gf.Vec3f(0.12, 0.35, 0.95)
    joint_local_pos = np.array([0.0, 0.0, -float(link_half)], dtype=np.float64)
    n_links = len(link_paths)
    for i in range(max(0, n_links - 1)):
        path = f"{link_paths[i]}/joint_ball"
        sphere = UsdGeom.Sphere.Define(stage, path)
        sphere.CreateRadiusAttr().Set(float(radius))
        sphere.CreateDisplayColorAttr().Set([marker_color])
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

    nodes = np.zeros((n_links + 1, 3), dtype=np.float64)
    nodes[0] = plus_end[0]
    for i in range(1, n_links):
        # Internal joint is shared by link_(i-1) local -Z and link_i local +Z.
        # nodes[i] = 0.5 * (minus_end[i - 1] + plus_end[i])
        nodes[i] = (minus_end[i - 1] )
    nodes[n_links] = minus_end[-1]
    return nodes


def main():
    endpoint_pos, endpoint_t, track_pos = load_endpoint_trajectory(args.traj_npz)
    num_links = int(args.num_nodes) - 1

    # Pinned endpoint position (free end of link_0)
    first_node = np.array([0.0, 0.0, float(args.wire_radius)], dtype=np.float64)
    first_node_b = first_node + np.array([0.0, float(WIRE_B_Y_OFFSET), 0.0], dtype=np.float64)

    world = World(physics_dt=PHYSICS_DT, rendering_dt=PHYSICS_DT, stage_units_in_meters=1.0)
    stage = omni.usd.get_context().get_stage()
    world.scene.add_default_ground_plane()
    ground_z = -float(args.wire_radius) - 1.0e-3
    ground_path = set_ground_plane_height(stage, ground_z)

    ps = UsdPhysics.Scene.Define(stage, "/physicsScene")
    ps.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
    ps.CreateGravityMagnitudeAttr().Set(9.81)
    PhysxSchema.PhysxSceneAPI.Apply(stage.GetPrimAtPath("/physicsScene")).CreateEnableGPUDynamicsAttr().Set(True)

    UsdLux.DomeLight.Define(stage, "/World/DomeLight").CreateIntensityAttr().Set(1000.0)
    UsdLux.DistantLight.Define(stage, "/World/DistantLight").CreateIntensityAttr().Set(3000.0)

    # Visual reference: this ball follows NPZ `pos` strictly at sim-time.
    strict_pos0 = sample_pos_at_time(endpoint_pos, endpoint_t, 0.0, 0)
    target_ball_radius = max(0.008, 1.5 * float(args.wire_radius)) * float(args.ball_size_scale)
    track_ball_radius = max(0.0035, 0.75 * float(args.wire_radius)) * float(args.ball_size_scale)
    joint_ball_radius = max(0.0025, 0.45 * float(args.wire_radius)) * float(args.ball_size_scale)

    strict_pos0_b = strict_pos0 + np.array([0.0, float(WIRE_B_Y_OFFSET), 0.0], dtype=np.float64)
    create_target_ball(
        stage,
        TARGET_BALL_PATH_A,
        radius=target_ball_radius,
        position=add_ball_offset(strict_pos0, float(args.ball_offset_z)),
    )
    create_target_ball(
        stage,
        TARGET_BALL_PATH_B,
        radius=target_ball_radius,
        position=add_ball_offset(strict_pos0_b, float(args.ball_offset_z)),
    )
    track_ball_paths_a = []
    track_ball_paths_b = []
    if track_pos is not None and track_pos.shape[1] > 0:
        track0 = sample_track_pos_at_time(track_pos, endpoint_t, 0.0, 0)
        track0_a = track0.copy()
        track0_a[:, 2] += float(args.ball_offset_z)
        track_ball_paths_a = create_track_balls(
            stage,
            TRACK_BALLS_ROOT_A,
            track0_a,
            radius=track_ball_radius,
        )
        track0_b = track0.copy()
        track0_b[:, 1] += float(WIRE_B_Y_OFFSET)
        track0_b[:, 2] += float(args.ball_offset_z)
        track_ball_paths_b = create_track_balls(
            stage,
            TRACK_BALLS_ROOT_B,
            track0_b,
            radius=track_ball_radius,
        )

    link_paths_a, link_half_a = create_rigid_wire(
        stage=stage,
        root_path=WIRE_ROOT_A,
        first_node=first_node,
        num_links=num_links,
        wire_length=float(args.wire_length),
        wire_radius=float(args.wire_radius),
        mat_static_friction=float(args.mat_static_friction),
        mat_dynamic_friction=float(args.mat_dynamic_friction),
        joint_damping=float(args.joint_damping),
        initial_spacing_scale=1.0,
    )
    joint_ball_paths_a, joint_ball_local_pos = create_joint_balls(
        stage,
        link_paths_a,
        link_half_a,
        radius=joint_ball_radius,
        color=Gf.Vec3f(0.12, 0.35, 0.95),
    )
    link_paths_b, link_half_b = create_rigid_wire(
        stage=stage,
        root_path=WIRE_ROOT_B,
        first_node=first_node_b,
        num_links=num_links,
        wire_length=float(args.wire_length),
        wire_radius=float(args.wire_radius),
        mat_static_friction=float(args.mat_static_friction),
        mat_dynamic_friction=float(args.mat_dynamic_friction),
        joint_damping=float(args.joint_damping),
        initial_spacing_scale=2.0,
    )
    joint_ball_paths_b, _ = create_joint_balls(
        stage,
        link_paths_b,
        link_half_b,
        radius=joint_ball_radius,
        color=Gf.Vec3f(0.10, 0.85, 0.18),
    )

    if not np.isclose(link_half_a, link_half_b):
        raise RuntimeError(f"link_half mismatch: {link_half_a} vs {link_half_b}")
    link_half = float(link_half_a)

    first_link_a = link_paths_a[0]
    last_link_a = link_paths_a[-1]
    mid_link_a = link_paths_a[len(link_paths_a) // 2]

    first_link_b = link_paths_b[0]
    last_link_b = link_paths_b[-1]
    mid_link_b = link_paths_b[len(link_paths_b) // 2]

    set_kinematic(stage, first_link_a)
    set_kinematic(stage, last_link_a)
    set_kinematic(stage, first_link_b)
    set_kinematic(stage, last_link_b)

    first_view_a = RigidPrimView(first_link_a, name="first_link_a")
    last_view_a = RigidPrimView(last_link_a, name="last_link_a")
    mid_view_a = RigidPrimView(mid_link_a, name="mid_link_a")
    first_view_b = RigidPrimView(first_link_b, name="first_link_b")
    last_view_b = RigidPrimView(last_link_b, name="last_link_b")
    mid_view_b = RigidPrimView(mid_link_b, name="mid_link_b")
    world.scene.add(first_view_a)
    world.scene.add(last_view_a)
    world.scene.add(mid_view_a)
    world.scene.add(first_view_b)
    world.scene.add(last_view_b)
    world.scene.add(mid_view_b)

    world.reset()

    # --- Orientation constraints via USD orient ops (wxyz) ---
    # We want: local +Z of link_0 points to world +X
    # Then the chain extends from the pinned free end along -X (because chain direction = -first_dir).
    q_first_wxyz = np.array([0.70710677, 0.0, +0.70710677, 0.0], dtype=np.float32)  # +90deg about Y
    q_last_wxyz = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)  # 180deg about Y

    set_orient_op_wxyz(stage, first_link_a, q_first_wxyz)
    set_orient_op_wxyz(stage, last_link_a, q_last_wxyz)
    set_orient_op_wxyz(stage, first_link_b, q_first_wxyz)
    set_orient_op_wxyz(stage, last_link_b, q_last_wxyz)

    # Read back the actual world direction of local +Z after applying orient op.
    first_dir_a = get_local_z_world_dir(stage, first_link_a).astype(np.float32)
    first_dir_b = get_local_z_world_dir(stage, first_link_b).astype(np.float32)
    # For last link at 180deg about Y, local +Z maps to world -Z.
    last_dir = np.array([0.0, 0.0, -1.0], dtype=np.float32)

    # Pin FREE end (local +Z) at first_node:
    # end(+Z) = center + first_dir*half  => center = end - first_dir*half
    first_center_a = np.asarray(first_node, dtype=np.float32) - first_dir_a * float(link_half)
    first_center_b = np.asarray(first_node_b, dtype=np.float32) - first_dir_b * float(link_half)
    # Additional user-requested offset: move link_0 by one link length along -X.
    first_center_a[0] -= float(2.0 * link_half)
    first_center_b[0] -= float(2.0 * link_half)
    set_translate_op(stage, first_link_a, first_center_a)
    set_translate_op(stage, first_link_b, first_center_b)

    print(f"[Run] traj={args.traj_npz}, frames={endpoint_pos.shape[0]}, has_time={endpoint_t is not None}")
    print(f"[Run] nodes={args.num_nodes}, links={num_links}, length={args.wire_length:.3f}m, radius={args.wire_radius:.4f}m")
    print(
        f"[Run] joint_damping={args.joint_damping:.3f}, "
        f"mat_friction=({args.mat_static_friction:.3f},{args.mat_dynamic_friction:.3f})"
    )
    print(f"[Run] warmup_steps={int(args.warmup_steps)}")
    if ground_path is not None:
        print(f"[Run] ground plane moved to z={ground_z:+.4f} at {ground_path}")
    print(
        f"[Run] target ball paths=({TARGET_BALL_PATH_A}, {TARGET_BALL_PATH_B}) "
        f"with wire B y_offset={float(WIRE_B_Y_OFFSET):+.3f} m"
    )
    if track_ball_paths_a or track_ball_paths_b:
        print(
            f"[Run] track balls A={len(track_ball_paths_a)} at {TRACK_BALLS_ROOT_A}, "
            f"B={len(track_ball_paths_b)} at {TRACK_BALLS_ROOT_B} follow NPZ track_pos."
        )
    if joint_ball_paths_a or joint_ball_paths_b:
        print(
            f"[Run] wire A joint balls={len(joint_ball_paths_a)} (blue), "
            f"wire B joint balls={len(joint_ball_paths_b)} (green), "
            "each parented to link_i at local position "
            f"({joint_ball_local_pos[0]:+.4f},{joint_ball_local_pos[1]:+.4f},{joint_ball_local_pos[2]:+.4f})"
        )
    print(
        "[Run] export positions source=joint balls "
        f"(wire_a={len(joint_ball_paths_a)}, wire_b={len(joint_ball_paths_b)})"
    )
    print(f"[Run] visualization balls z offset={float(args.ball_offset_z):+.3f} m")
    print(
        f"[Run] ball radii: target={target_ball_radius:.4f} m, "
        f"track={track_ball_radius:.4f} m, joint={joint_ball_radius:.4f} m "
        f"(scale={float(args.ball_size_scale):.2f})"
    )
    print(
        "[Run] pinned free end (link0 local +Z): "
        f"first=({first_node[0]:+.3f},{first_node[1]:+.3f},{first_node[2]:+.3f})"
    )
    print(
        "[Run] wire B pinned free end (offset in y): "
        f"first_b=({first_node_b[0]:+.3f},{first_node_b[1]:+.3f},{first_node_b[2]:+.3f}), "
        f"y_offset={float(WIRE_B_Y_OFFSET):+.3f} m"
    )
    print(
        "[Run] expect chain direction from pinned point ~(-1,0,0). "
        f"measured first_dir_a(local+Z)=({first_dir_a[0]:+.3f},{first_dir_a[1]:+.3f},{first_dir_a[2]:+.3f}), "
        f"first_dir_b(local+Z)=({first_dir_b[0]:+.3f},{first_dir_b[1]:+.3f},{first_dir_b[2]:+.3f}) "
        f"=> chain_dir_a=-first_dir_a=({-first_dir_a[0]:+.3f},{-first_dir_a[1]:+.3f},{-first_dir_a[2]:+.3f})"
    )
    print("[Run] Running... (Ctrl+C to stop)")
    sys.stdout.flush()

    out_npz = Path(args.traj_npz).expanduser().resolve().with_suffix("")
    out_npz = out_npz.with_name(out_npz.name + "_wire_joint_positions.npz")
    recorded_positions_a: list[np.ndarray] = []
    recorded_positions_b: list[np.ndarray] = []
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
                replay_t = replay_step * PHYSICS_DT
            record_this_step = steps >= warmup_steps

            target_node = sample_pos_at_time(endpoint_pos, endpoint_t, replay_t, replay_step)
            strict_pos = target_node
            strict_pos_b = np.asarray(strict_pos, dtype=np.float64) + np.array(
                [0.0, float(WIRE_B_Y_OFFSET), 0.0], dtype=np.float64
            )
            track_frame = None
            if track_pos is not None:
                track_frame = sample_track_pos_at_time(track_pos, endpoint_t, replay_t, replay_step)

            # Last chain-side joint is at local +Z, so center = joint - (local+Z_world)*half.
            target_node_b = np.asarray(target_node, dtype=np.float32) + np.array(
                [0.0, float(WIRE_B_Y_OFFSET), 0.0], dtype=np.float32
            )
            last_center = np.asarray(target_node, dtype=np.float32) + last_dir * float(link_half)
            last_center_b = target_node_b + last_dir * float(link_half)

            # Kinematic targets (positions). Orientation is handled by USD orient op.
            first_view_a.set_world_poses(positions=first_center_a.reshape(1, 3))
            last_view_a.set_world_poses(positions=last_center.reshape(1, 3))
            first_view_b.set_world_poses(positions=first_center_b.reshape(1, 3))
            last_view_b.set_world_poses(positions=last_center_b.reshape(1, 3))

            # Keep inspector translate in sync.
            set_translate_op(stage, first_link_a, first_center_a)
            set_translate_op(stage, last_link_a, last_center)
            set_translate_op(stage, first_link_b, first_center_b)
            set_translate_op(stage, last_link_b, last_center_b)
            set_translate_op(stage, TARGET_BALL_PATH_A, add_ball_offset(strict_pos, float(args.ball_offset_z)))
            set_translate_op(stage, TARGET_BALL_PATH_B, add_ball_offset(strict_pos_b, float(args.ball_offset_z)))
            if track_frame is not None:
                for i, pth in enumerate(track_ball_paths_a):
                    set_translate_op(stage, pth, add_ball_offset(track_frame[i], float(args.ball_offset_z)))
                for i, pth in enumerate(track_ball_paths_b):
                    p_b = np.asarray(track_frame[i], dtype=np.float64) + np.array(
                        [0.0, float(WIRE_B_Y_OFFSET), 0.0], dtype=np.float64
                    )
                    set_translate_op(stage, pth, add_ball_offset(p_b, float(args.ball_offset_z)))

            world.step(render=not args.headless)
            steps += 1
            if record_this_step:
                marker_positions_a = get_marker_world_positions(stage, joint_ball_paths_a)
                marker_positions_b = get_marker_world_positions(stage, joint_ball_paths_b)
                recorded_time.append(float(replay_t))
                recorded_target_pos.append(np.asarray(target_node, dtype=np.float64).copy())
                if recorded_track_pos is not None and track_frame is not None:
                    recorded_track_pos.append(np.asarray(track_frame, dtype=np.float64).copy())
                recorded_positions_a.append(marker_positions_a)
                recorded_positions_b.append(marker_positions_b)

            if steps == 1:
                pos0a, _ = first_view_a.get_world_poses()
                c0a = np.asarray(pos0a[0], dtype=np.float64)
                d0a = get_local_z_world_dir(stage, first_link_a)
                e0a = c0a + d0a * float(link_half)  # free end position (local +Z)
                err_a = np.linalg.norm(e0a - np.asarray(first_node, dtype=np.float64))
                chain_dir_a = -d0a
                pos0b, _ = first_view_b.get_world_poses()
                c0b = np.asarray(pos0b[0], dtype=np.float64)
                d0b = get_local_z_world_dir(stage, first_link_b)
                e0b = c0b + d0b * float(link_half)
                err_b = np.linalg.norm(e0b - np.asarray(first_node_b, dtype=np.float64))
                chain_dir_b = -d0b
                print(
                    f"[Debug] wire_a_link0_free_end=({e0a[0]:+.4f},{e0a[1]:+.4f},{e0a[2]:+.4f}) "
                    f"target=({first_node[0]:+.4f},{first_node[1]:+.4f},{first_node[2]:+.4f}) "
                    f"dir(local+Z)=({d0a[0]:+.4f},{d0a[1]:+.4f},{d0a[2]:+.4f}) "
                    f"chain_dir=({chain_dir_a[0]:+.4f},{chain_dir_a[1]:+.4f},{chain_dir_a[2]:+.4f}) "
                    f"err={err_a:.3e}; "
                    f"wire_b_link0_free_end=({e0b[0]:+.4f},{e0b[1]:+.4f},{e0b[2]:+.4f}) "
                    f"target_b=({first_node_b[0]:+.4f},{first_node_b[1]:+.4f},{first_node_b[2]:+.4f}) "
                    f"dir(local+Z)=({d0b[0]:+.4f},{d0b[1]:+.4f},{d0b[2]:+.4f}) "
                    f"chain_dir=({chain_dir_b[0]:+.4f},{chain_dir_b[1]:+.4f},{chain_dir_b[2]:+.4f}) "
                    f"err={err_b:.3e}"
                )
                sys.stdout.flush()

            if steps % 240 == 0:
                mid_pos_a, _ = mid_view_a.get_world_poses()
                p_a = mid_pos_a[0]
                mid_pos_b, _ = mid_view_b.get_world_poses()
                p_b = mid_pos_b[0]
                print(
                    f"[Step {steps:06d}] target=({target_node[0]:+.3f},{target_node[1]:+.3f},{target_node[2]:+.3f}) "
                    f"target_b=({target_node_b[0]:+.3f},{target_node_b[1]:+.3f},{target_node_b[2]:+.3f}) "
                    f"mid_a=({p_a[0]:+.3f},{p_a[1]:+.3f},{p_a[2]:+.3f}) "
                    f"mid_b=({p_b[0]:+.3f},{p_b[1]:+.3f},{p_b[2]:+.3f})"
                )
                sys.stdout.flush()

            if args.max_steps > 0 and steps >= args.max_steps:
                break
    except KeyboardInterrupt:
        pass

    n_samples = len(recorded_positions_a)
    n_markers_a = len(joint_ball_paths_a)
    n_markers_b = len(joint_ball_paths_b)
    if n_samples > 0:
        positions_a_arr = np.asarray(recorded_positions_a, dtype=np.float64)
        positions_b_arr = np.asarray(recorded_positions_b, dtype=np.float64)
        time_arr = np.asarray(recorded_time, dtype=np.float64).reshape(-1)
        target_arr = np.asarray(recorded_target_pos, dtype=np.float64)
    else:
        positions_a_arr = np.zeros((0, n_markers_a, 3), dtype=np.float64)
        positions_b_arr = np.zeros((0, n_markers_b, 3), dtype=np.float64)
        time_arr = np.zeros((0,), dtype=np.float64)
        target_arr = np.zeros((0, 3), dtype=np.float64)

    if positions_a_arr.shape[1:] != (n_markers_a, 3):
        raise RuntimeError(
            f"Unexpected wire A positions shape {positions_a_arr.shape}; expected (T, {n_markers_a}, 3)."
        )
    if positions_b_arr.shape[1:] != (n_markers_b, 3):
        raise RuntimeError(
            f"Unexpected wire B positions shape {positions_b_arr.shape}; expected (T, {n_markers_b}, 3)."
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
        # Keep `positions` for compatibility and expose explicit A/B arrays for comparison.
        "positions": positions_a_arr,
        "positions_wire_a": positions_a_arr,
        "positions_wire_b": positions_b_arr,
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
        f"positions_wire_a={positions_a_arr.shape}, positions_wire_b={positions_b_arr.shape}, "
        f"time={time_arr.shape}, target_pos={target_arr.shape}"
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
