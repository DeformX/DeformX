#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Drive a PyElastica wire from a GUI-movable ball in Isaac Sim.

Move `/World/Ball` with the viewport gizmo while the simulation is running.
The wire root frame follows the ball pose each physics step.
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
from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdLux, UsdPhysics
from omni.isaac.core import World

from co_sim.engine import CoSimEngine
from co_sim.models import CoSimConfig, FrameState
from rod_skel_driver_test import SkeletonRodDriver

PHYSICS_DT = 1.0 / 200.0
PY_DT = 2.0e-5

WIRE_BASE_LENGTH = 1.5
WIRE_N_ELEM = 20
WIRE_BASE_RADIUS = 0.005
WIRE_ROOT_PATH = "/World/PyElasticaWire"
BALL_PATH = "/World/Ball"


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


def create_ball_handle(stage, ball_path: str):
    s = UsdGeom.Sphere.Define(stage, ball_path)
    s.GetRadiusAttr().Set(0.05)
    s.GetDisplayColorAttr().Set([Gf.Vec3f(0.95, 0.15, 0.15)])
    xf = UsdGeom.Xformable(s.GetPrim())
    xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.9))


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

    ball_pos, ball_dir = get_prim_world_pose(stage, BALL_PATH)
    if ball_pos is None:
        raise RuntimeError(f"Ball prim not valid: {BALL_PATH}")

    frame_init = FrameState(
        position=ball_pos,
        director=ball_dir,
        velocity=np.zeros(3, dtype=np.float64),
        acceleration=np.zeros(3, dtype=np.float64),
        omega=np.zeros(3, dtype=np.float64),
        alpha=np.zeros(3, dtype=np.float64),
    )

    driver = SkeletonRodDriver(stage, skeleton_path=WIRE_ROOT_PATH)
    driver.load_asset(args.wire_usd)

    cfg = CoSimConfig(
        base_length=WIRE_BASE_LENGTH,
        n_elem=WIRE_N_ELEM,
        base_radius=WIRE_BASE_RADIUS,
        py_dt=PY_DT,
        isaac_dt=PHYSICS_DT,
        final_time=1e9,
        render=False,
        joint_k=1.0e1,
        joint_nu=5.0,
        frame_initial_position=np.asarray(frame_init.position, dtype=np.float64),
        frame_initial_director=np.asarray(frame_init.director, dtype=np.float64),
        frame_initial_velocity=np.asarray(frame_init.velocity, dtype=np.float64),
        frame_initial_acceleration=np.asarray(frame_init.acceleration, dtype=np.float64),
        frame_initial_omega=np.asarray(frame_init.omega, dtype=np.float64),
        frame_initial_alpha=np.asarray(frame_init.alpha, dtype=np.float64),
        rod_start=np.asarray(frame_init.position, dtype=np.float64),
    )
    engine = CoSimEngine(config=cfg, frame_initial_state=frame_init)

    kin_state = {
        "position": np.asarray(frame_init.position, dtype=np.float64),
        "director": np.asarray(frame_init.director, dtype=np.float64),
        "velocity": np.zeros(3, dtype=np.float64),
        "omega": np.zeros(3, dtype=np.float64),
    }

    print("[Run] Move /World/Ball in GUI to drive the wire.")
    print("[Run] Running... (Ctrl+C to stop)")
    sys.stdout.flush()

    steps = 0
    try:
        while simulation_app.is_running():
            pos, direction = get_prim_world_pose(stage, BALL_PATH)
            if pos is None:
                break

            frame_cmd, kin_state = make_frame_state_from_pose(pos, direction, kin_state, PHYSICS_DT)
            engine.update_frame_state(frame_cmd, duration=PHYSICS_DT)
            snap = engine.snapshot()
            driver.update_skeleton(snap.rod_position, snap.rod_director, time_code=None)

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
