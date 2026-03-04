#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Minimal Isaac Sim scene:
- Loads a UR5e (USD reference)
- Adds a camera at a configurable pose
- (Optional) ground plane + simple lights
- Keeps running so you can tweak parameters and rerun quickly

Run:
  /home/robot/isaacsim/python.sh minimal_arm_camera_env.py
  /home/robot/isaacsim/python.sh minimal_arm_camera_env.py --headless
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np


# -------------------------
# User knobs (edit these)
# -------------------------
ROBOT_PATH = "/World/UR5e"
ROBOT_POS = np.array([0.0, 0.0, 1.7], dtype=np.float64)
ROBOT_ROT_XYZ_DEG = np.array([-90.0, 0.0, 0.0], dtype=np.float64)
CAMERA_PATH = "/World/MainCamera"
CAMERA_POS = np.array([4.28, 1.8, 1.7], dtype=np.float64)
CAMERA_ROT_XYZ_DEG = np.array([87.0, 0.0, 90], dtype=np.float64)  # typical "look toward -X" intent
FOCAL_LENGTH_MM = 20.0

ADD_GROUND = True
GROUND_COLOR = np.array([0.05, 0.05, 0.05], dtype=np.float32)

ADD_LIGHTS = True

PHYSICS_DT = 1.0 / 240.0
RENDER_DT = 1.0 / 60.0


def resolve_ur5e_usd_path() -> str:
    """Try local Isaac Sim assets first, otherwise fall back to an online USD."""
    local_candidates = [
        "/home/robot/isaacsim_assets/Assets/Isaac/4.5/Isaac/Robots/UniversalRobots/ur5e/ur5e.usd",
        "/home/robot/isaacsim_assets/Assets/Isaac/5.1/Isaac/Robots/UniversalRobots/ur5e/ur5e.usd",
        "/home/robot/isaacsim_assets/Assets/Isaac/4.5/Isaac/Robots/UniversalRobots/ur5/ur5.usd",
        "/home/robot/isaacsim_assets/Assets/Isaac/5.1/Isaac/Robots/UniversalRobots/ur5/ur5.usd",
        "/home/robot/isaacsim/Assets/Isaac/4.5/Isaac/Robots/UniversalRobots/ur5e/ur5e.usd",
    ]
    for p in local_candidates:
        if os.path.exists(p):
            return p
    # fallback
    return (
        "https://omniverse-content-production.s3-us-west-2.amazonaws.com/"
        "Assets/Isaac/4.5/Isaac/Robots/UniversalRobots/ur5/ur5.usd"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--renderer", type=str, default="RayTracedLighting", choices=["RayTracedLighting", "PathTracing"])
    parser.add_argument("--active-gpu", type=int, default=0)
    parser.add_argument("--physics-gpu", type=int, default=0)
    args = parser.parse_args()

    # Prevent SimulationApp from consuming unknown args
    original_argv = list(sys.argv)
    sys.argv = [sys.argv[0]]

    try:
        from isaacsim import SimulationApp
    except ImportError:
        from omni.isaac.kit import SimulationApp

    simulation_app = SimulationApp(
        {
            "headless": bool(args.headless),
            "renderer": str(args.renderer),
            "active_gpu": int(args.active_gpu),
            "physics_gpu": int(args.physics_gpu),
        }
    )

    try:
        from pxr import Gf, UsdGeom, UsdLux
        import omni.usd

        from omni.isaac.core import World
        from omni.isaac.core.utils.stage import add_reference_to_stage

        # World
        world = World(physics_dt=PHYSICS_DT, rendering_dt=RENDER_DT, stage_units_in_meters=1.0)
        stage = omni.usd.get_context().get_stage()

        # Optional ground
        if ADD_GROUND:
            world.scene.add_ground_plane(color=GROUND_COLOR)

        # Optional lights
        if ADD_LIGHTS:
            dome = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
            dome.CreateIntensityAttr().Set(200.0)
            dome.CreateColorAttr().Set(Gf.Vec3f(1.0, 1.0, 1.0))

            key = UsdLux.DistantLight.Define(stage, "/World/KeyLight")
            key.CreateIntensityAttr().Set(2500.0)
            key.CreateColorAttr().Set(Gf.Vec3f(1.0, 1.0, 1.0))

        # Load robot
        usd_path = resolve_ur5e_usd_path()
        add_reference_to_stage(usd_path=usd_path, prim_path=ROBOT_PATH)

        # Robot transform
        robot_prim = stage.GetPrimAtPath(ROBOT_PATH)
        if not robot_prim.IsValid():
            raise RuntimeError(f"Robot prim not valid: {ROBOT_PATH}")

        robot_xf = UsdGeom.Xformable(robot_prim)
        robot_xf.ClearXformOpOrder()
        robot_xf.AddTranslateOp().Set(Gf.Vec3d(*ROBOT_POS.tolist()))
        robot_xf.AddRotateXYZOp().Set(Gf.Vec3f(*ROBOT_ROT_XYZ_DEG.tolist()))

        # Create camera
        cam = UsdGeom.Camera.Define(stage, CAMERA_PATH)
        cam.CreateFocalLengthAttr(float(FOCAL_LENGTH_MM))
        cam_xf = UsdGeom.Xformable(cam.GetPrim())
        cam_xf.ClearXformOpOrder()
        cam_xf.AddTranslateOp().Set(Gf.Vec3d(*CAMERA_POS.tolist()))
        cam_xf.AddRotateXYZOp().Set(Gf.Vec3f(*CAMERA_ROT_XYZ_DEG.tolist()))

        # Set viewport active camera (if GUI)
        if not args.headless:
            try:
                import omni.kit.viewport.utility as viewport_utility

                vp = viewport_utility.get_active_viewport()
                if vp is not None:
                    vp.set_active_camera(CAMERA_PATH)
            except Exception:
                pass

        # Reset and run
        world.reset()
        print("[minimal] Scene ready.")
        print(f"[minimal] Robot USD: {usd_path}")
        print(f"[minimal] Camera: pos={CAMERA_POS.tolist()} rotXYZdeg={CAMERA_ROT_XYZ_DEG.tolist()} focal={FOCAL_LENGTH_MM}mm")
        print("[minimal] Close the window or Ctrl+C to exit.")

        while simulation_app.is_running():
            world.step(render=True)
            # small sleep to reduce CPU spin when headless
            if args.headless:
                time.sleep(0.001)

    finally:
        simulation_app.close()
        sys.argv = original_argv


if __name__ == "__main__":
    main()