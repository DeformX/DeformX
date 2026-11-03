#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Move rigid-body wire (capsules + spherical joints) with a GUI ball handle.

Use the viewport gizmo to move /World/Ball while simulation is running.
The first wire link is attached to the ball every step.
"""

import argparse
import sys

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true")
parser.add_argument("--physics_gpu", type=int, default=0)
parser.add_argument("--num_links", type=int, default=20)
parser.add_argument("--wire_length", type=float, default=1.5)
parser.add_argument("--wire_radius", type=float, default=0.005)
parser.add_argument("--max_steps", type=int, default=0, help="Stop after N steps (0 = run forever)")
args, _ = parser.parse_known_args()

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

PHYSICS_DT = 1.0 / 200.0
BALL_PATH = "/World/Ball"
WIRE_ROOT = "/World/RigidWire"


def create_ball_handle(stage, ball_path: str):
    s = UsdGeom.Sphere.Define(stage, ball_path)
    s.GetRadiusAttr().Set(0.05)
    s.GetDisplayColorAttr().Set([Gf.Vec3f(0.95, 0.2, 0.2)])
    xf = UsdGeom.Xformable(s.GetPrim())
    xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 1.1))


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
    return pos, quat_wxyz, R


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
    ball_pos, _, _ = get_prim_world_pose(stage, BALL_PATH)
    if ball_pos is None:
        raise RuntimeError(f"Ball prim invalid: {BALL_PATH}")

    link_paths, link_half = create_rigid_wire(
        stage,
        WIRE_ROOT,
        ball_pos,
        num_links=int(args.num_links),
        wire_length=float(args.wire_length),
        wire_radius=float(args.wire_radius),
    )

    first_link_view = RigidPrimView(link_paths[0], name="first_link")
    world.scene.add(first_link_view)
    world.reset()

    print("[Run] Move /World/Ball in GUI to drive rigid wire.")
    print("[Run] Running... (Ctrl+C to stop)")
    sys.stdout.flush()

    steps = 0
    try:
        while simulation_app.is_running():
            ball_pos, ball_quat, ball_R = get_prim_world_pose(stage, BALL_PATH)
            if ball_pos is None:
                break

            offset = ball_R @ np.array([0.0, 0.0, float(link_half)], dtype=np.float64)
            first_center = (ball_pos - offset).astype(np.float32)

            first_link_view.set_world_poses(
                positions=first_center.reshape(1, 3),
                orientations=ball_quat.reshape(1, 4),
            )
            first_link_view.set_velocities(np.zeros((1, 6), dtype=np.float32))

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
