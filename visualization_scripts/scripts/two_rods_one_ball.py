#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Attach two rods to one GUI-movable ball:
1) PyElastica rod (CoSimEngine + SkeletonRodDriver)
2) Rigid-body rod (capsules + spherical joints)
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import numpy.testing  # Keep numpy.testing bound to the same NumPy before Kit mutates import paths.

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
import deformx_paths

parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true")
parser.add_argument("--physics_gpu", type=int, default=0)
parser.add_argument(
    "--wire_usd",
    type=str,
    default=str(deformx_paths.ASSET_ROOT / "usd" / "skeleton_mesh_rod_collisions_20_l1.5_r0.005.usdc"),
)
parser.add_argument("--num_links", type=int, default=20)
parser.add_argument("--wire_length", type=float, default=1.5)
parser.add_argument("--wire_radius", type=float, default=0.005)
parser.add_argument("--max_steps", type=int, default=0, help="Stop after N steps (0 = run forever)")
args, _ = parser.parse_known_args()

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
from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade

from co_sim.engine import CoSimEngine
from co_sim.models import CoSimConfig, FrameState
from rod_skel_driver_test import SkeletonRodDriver

PHYSICS_DT = 1.0 / 200.0
PY_DT = 2.0e-5

BALL_PATH = "/World/Ball"
PY_WIRE_ROOT = "/World/PyElasticaWire"
RIGID_WIRE_ROOT = "/World/RigidWire"

PY_WIRE_BASE_LENGTH = 1.5
PY_WIRE_N_ELEM = 20
PY_WIRE_BASE_RADIUS = 0.005

# CoSimEngine uses frame_axis = director[2]. Use -Z so wire points down.
DEFAULT_FRAME_DIRECTOR = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, -1.0],
    ],
    dtype=np.float64,
)


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


def create_ball_handle(stage, ball_path: str):
    s = UsdGeom.Sphere.Define(stage, ball_path)
    s.GetRadiusAttr().Set(0.06)
    s.GetDisplayColorAttr().Set([Gf.Vec3f(0.95, 0.2, 0.2)])
    xf = UsdGeom.Xformable(s.GetPrim())
    xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 1.1))


def get_prim_world_pose(stage, prim_path: str):
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        return None, None, None
    tf = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    t = tf.ExtractTranslation()
    q = tf.ExtractRotationQuat()
    qi = q.GetImaginary()
    R = np.array(tf.ExtractRotationMatrix(), dtype=np.float64)
    pos = np.array([float(t[0]), float(t[1]), float(t[2])], dtype=np.float64)
    quat_wxyz = np.array([float(q.GetReal()), float(qi[0]), float(qi[1]), float(qi[2])], dtype=np.float32)
    return pos, quat_wxyz, _orthonormalize(R)


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


def create_rigid_wire(stage, root_path: str, top_pos, num_links, wire_length, wire_radius):
    UsdGeom.Scope.Define(stage, root_path)

    mat_path = f"{root_path}/WireMaterial"
    UsdShade.Material.Define(stage, mat_path)
    m = UsdPhysics.MaterialAPI.Apply(stage.GetPrimAtPath(mat_path))
    m.CreateStaticFrictionAttr().Set(0.5)
    m.CreateDynamicFrictionAttr().Set(0.5)
    m.CreateRestitutionAttr().Set(0.0)

    link_half = float(wire_length) / (2.0 * float(num_links))
    link_len = 2.0 * link_half

    x0, y0, z0 = map(float, top_pos)
    link_paths = []
    for i in range(num_links):
        p = f"{root_path}/link_{i}"
        cap = UsdGeom.Capsule.Define(stage, p)
        cap.CreateHeightAttr(link_len)
        cap.CreateRadiusAttr(float(wire_radius))
        cap.CreateAxisAttr("Z")
        cap.CreateDisplayColorAttr().Set([Gf.Vec3f(0.95, 0.45, 0.1)])
        UsdGeom.Xformable(cap.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(x0, y0, z0 - (i + 0.5) * link_len))

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
        link_paths.append(p)

    for i in range(num_links - 1):
        jp = f"{root_path}/joint_{i}"
        j = UsdPhysics.SphericalJoint.Define(stage, jp)
        j.GetBody0Rel().SetTargets([link_paths[i]])
        j.GetBody1Rel().SetTargets([link_paths[i + 1]])
        j.CreateLocalPos0Attr().Set(Gf.Vec3f(0.0, 0.0, -link_half))
        j.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, link_half))
        j.CreateLocalRot0Attr().Set(Gf.Quatf(1.0))
        j.CreateLocalRot1Attr().Set(Gf.Quatf(1.0))

        d = UsdPhysics.DriveAPI.Apply(j.GetPrim(), "angular")
        d.CreateTypeAttr("force")
        d.CreateDampingAttr(5.0)
        d.CreateStiffnessAttr(0.0)

    return link_paths, link_half


def main():
    world = World(physics_dt=PHYSICS_DT, rendering_dt=PHYSICS_DT, stage_units_in_meters=1.0)
    stage = omni.usd.get_context().get_stage()
    world.scene.add_default_ground_plane()

    ps = UsdPhysics.Scene.Define(stage, "/physicsScene")
    ps.CreateGravityDirectionAttr().Set(Gf.Vec3f(0, 0, -1))
    ps.CreateGravityMagnitudeAttr().Set(9.81)
    PhysxSchema.PhysxSceneAPI.Apply(stage.GetPrimAtPath("/physicsScene")).CreateEnableGPUDynamicsAttr().Set(True)

    UsdLux.DomeLight.Define(stage, "/World/DomeLight").CreateIntensityAttr().Set(1000.0)
    UsdLux.DistantLight.Define(stage, "/World/DistantLight").CreateIntensityAttr().Set(3000.0)

    create_ball_handle(stage, BALL_PATH)
    ball_pos, ball_quat, ball_R = get_prim_world_pose(stage, BALL_PATH)
    if ball_pos is None:
        raise RuntimeError(f"Ball prim invalid: {BALL_PATH}")

    # Rigid-body rod (head anchored at same ball)
    rigid_links, rigid_link_half = create_rigid_wire(
        stage,
        RIGID_WIRE_ROOT,
        ball_pos,
        num_links=int(args.num_links),
        wire_length=float(args.wire_length),
        wire_radius=float(args.wire_radius),
    )
    rigid_first_view = RigidPrimView(rigid_links[0], name="rigid_first")
    world.scene.add(rigid_first_view)

    # Reset after rigid-body wire creation. Build the PyElastica wire after reset
    # so reset does not deactivate the skeleton-driven path.
    world.reset()
    ball_pos, _, _ = get_prim_world_pose(stage, BALL_PATH)
    if ball_pos is None:
        raise RuntimeError(f"Ball prim invalid after reset: {BALL_PATH}")

    # PyElastica rod (head anchored at ball)
    frame_init = FrameState(
        position=ball_pos,
        director=DEFAULT_FRAME_DIRECTOR.copy(),
        velocity=np.zeros(3, dtype=np.float64),
        acceleration=np.zeros(3, dtype=np.float64),
        omega=np.zeros(3, dtype=np.float64),
        alpha=np.zeros(3, dtype=np.float64),
    )

    py_driver = SkeletonRodDriver(stage, skeleton_path=PY_WIRE_ROOT)
    py_driver.load_asset(args.wire_usd)

    py_cfg = CoSimConfig(
        base_length=PY_WIRE_BASE_LENGTH,
        n_elem=PY_WIRE_N_ELEM,
        base_radius=PY_WIRE_BASE_RADIUS,
        py_dt=PY_DT,
        isaac_dt=PHYSICS_DT,
        final_time=1e9,
        render=False,
        joint_k=1.0e1,
        joint_nu=5.0,
        rod_direction=np.array([0.0, 0.0, -1.0], dtype=np.float64),
        rod_normal=np.array([1.0, 0.0, 0.0], dtype=np.float64),
        frame_initial_position=np.asarray(frame_init.position, dtype=np.float64),
        frame_initial_director=np.asarray(frame_init.director, dtype=np.float64),
        frame_initial_velocity=np.asarray(frame_init.velocity, dtype=np.float64),
        frame_initial_acceleration=np.asarray(frame_init.acceleration, dtype=np.float64),
        frame_initial_omega=np.asarray(frame_init.omega, dtype=np.float64),
        frame_initial_alpha=np.asarray(frame_init.alpha, dtype=np.float64),
        rod_start=np.asarray(frame_init.position, dtype=np.float64),
    )
    py_engine = CoSimEngine(config=py_cfg, frame_initial_state=frame_init)
    py_kin = {
        "position": np.asarray(frame_init.position, dtype=np.float64),
        "director": np.asarray(frame_init.director, dtype=np.float64),
        "velocity": np.zeros(3, dtype=np.float64),
        "omega": np.zeros(3, dtype=np.float64),
    }

    print("[Run] Move /World/Ball in GUI to drive BOTH rods.")
    print("[Run] Rod A: PyElastica, Rod B: rigid capsules with ball joints.")
    print("[Run] Running... (Ctrl+C to stop)")
    sys.stdout.flush()

    steps = 0
    try:
        while simulation_app.is_running():
            ball_pos, ball_quat, ball_R = get_prim_world_pose(stage, BALL_PATH)
            if ball_pos is None:
                break

            # PyElastica update
            py_cmd, py_kin = make_frame_state_from_pose(
                position=ball_pos,
                director=DEFAULT_FRAME_DIRECTOR,
                prev_kin=py_kin,
                dt=PHYSICS_DT,
            )
            py_engine.update_frame_state(py_cmd, duration=PHYSICS_DT)
            py_snap = py_engine.snapshot()
            py_driver.update_skeleton(py_snap.rod_position, py_snap.rod_director, time_code=None)

            # Rigid link-0 attachment to the same ball pose
            offset = ball_R @ np.array([0.0, 0.0, float(rigid_link_half)], dtype=np.float64)
            first_center = (ball_pos - offset).astype(np.float32)
            rigid_first_view.set_world_poses(
                positions=first_center.reshape(1, 3),
                orientations=ball_quat.reshape(1, 4),
            )
            rigid_first_view.set_velocities(np.zeros((1, 6), dtype=np.float32))

            world.step(render=not args.headless)
            steps += 1
            if args.max_steps > 0 and steps >= args.max_steps:
                break
    except KeyboardInterrupt:
        pass


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
