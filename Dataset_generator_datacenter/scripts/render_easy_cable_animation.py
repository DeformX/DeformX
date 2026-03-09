#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from isaacsim import SimulationApp


def parse_args():
    pkg_dir = Path(__file__).resolve().parents[1]
    data_dir = pkg_dir / "data"
    npz_dir = data_dir / "npz_file"

    parser = argparse.ArgumentParser(
        description="Render an RGB cable-only animation from an NPZ trajectory and a single cable USD asset."
    )
    parser.add_argument("--headless", action="store_true", help="Run Isaac Sim headless.")
    parser.add_argument("--npz", type=str, default=str(npz_dir / "easy.npz"), help="Trajectory NPZ path.")
    parser.add_argument(
        "--cable_usd",
        type=str,
        default=str(data_dir / "data_cernter_cable.usdc"),
        help="Cable USD asset path.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=str(pkg_dir / "output" / "easy_cable_animation"),
        help="Output directory for frames and video.",
    )
    parser.add_argument("--frame_start", type=int, default=0, help="First NPZ frame to render.")
    parser.add_argument("--frame_end", type=int, default=None, help="Last NPZ frame to render (inclusive).")
    parser.add_argument("--frame_step", type=int, default=1, help="NPZ frame step.")
    parser.add_argument("--width", type=int, default=960, help="Render width.")
    parser.add_argument("--height", type=int, default=960, help="Render height.")
    parser.add_argument("--rt_subframes", type=int, default=1, help="Replicator rt_subframes per rendered frame.")
    parser.add_argument("--video_name", type=str, default="easy_cable_animation.mp4", help="Output mp4 filename.")
    parser.add_argument("--skip_video", action="store_true", help="Skip ffmpeg mp4 assembly.")
    return parser.parse_args()


ARGS = parse_args()
SIM_APP = SimulationApp({"headless": bool(ARGS.headless)})


if __package__ is None or __package__ == "":
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from Dataset_generator_datacenter.rod_skel_driver import SkeletonRodDriver
else:
    from ..rod_skel_driver import SkeletonRodDriver


import carb
import numpy as np
import omni.replicator.core as rep
import omni.usd
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdShade, UsdSkel


def set_render_quality() -> None:
    settings = carb.settings.get_settings()
    pairs = [
        ("/rtx/rendermode", "RayTracedLighting"),
        ("/rtx/pathtracing/enabled", False),
        ("/rtx/post/dlss/enabled", False),
        ("/rtx/post/taa/enabled", False),
        ("/rtx/taa/enabled", False),
        ("/rtx/post/motionBlur/enabled", False),
        ("/rtx/post/motionblur/enabled", False),
    ]
    for path, value in pairs:
        try:
            settings.set(path, value)
        except Exception:
            pass


def validate_npz(npz_path: Path):
    if not npz_path.is_file():
        raise FileNotFoundError(f"NPZ file not found: {npz_path}")

    data = np.load(npz_path)
    required = {"time", "pos", "director"}
    missing = sorted(required.difference(data.files))
    if missing:
        raise RuntimeError(f"NPZ is missing required arrays: {missing}")

    time_arr = np.asarray(data["time"])
    pos_arr = np.asarray(data["pos"])
    dir_arr = np.asarray(data["director"])

    if time_arr.ndim != 1:
        raise RuntimeError(f"'time' must be (T,), got {time_arr.shape}")
    if pos_arr.ndim != 4:
        raise RuntimeError(f"'pos' must be (T, N, 3, K), got {pos_arr.shape}")
    if dir_arr.ndim != 5:
        raise RuntimeError(f"'director' must be (T, N, 3, 3, D), got {dir_arr.shape}")

    t_pos, n_pos, pos_xyz, k_nodes = pos_arr.shape
    t_dir, n_dir, dir_a, dir_b, d_elems = dir_arr.shape

    if pos_xyz != 3:
        raise RuntimeError(f"'pos' third dimension must be 3, got {pos_xyz}")
    if dir_a != 3 or dir_b != 3:
        raise RuntimeError(f"'director' inner dimensions must be 3x3, got {(dir_a, dir_b)}")
    if len(time_arr) != t_pos:
        raise RuntimeError(f"'time' length {len(time_arr)} does not match pos T={t_pos}")
    if t_pos != t_dir:
        raise RuntimeError(f"pos T={t_pos} does not match director T={t_dir}")
    if n_pos != n_dir:
        raise RuntimeError(f"pos N={n_pos} does not match director N={n_dir}")
    if k_nodes != d_elems + 1:
        raise RuntimeError(f"Invalid NPZ geometry: pos K={k_nodes} must equal director D+1={d_elems + 1}")

    return (
        np.asarray(time_arr, dtype=np.float32),
        np.asarray(pos_arr, dtype=np.float32),
        np.asarray(dir_arr, dtype=np.float32),
    )


def get_asset_default_prim_path(asset_usd_path: str):
    asset_stage = Usd.Stage.Open(asset_usd_path)
    if asset_stage is None:
        raise RuntimeError(f"Failed to open asset stage: {asset_usd_path}")
    default_prim = asset_stage.GetDefaultPrim()
    if not default_prim.IsValid():
        raise RuntimeError(f"Asset defaultPrim missing: {asset_usd_path}")
    return default_prim.GetPath()


def find_first_skeleton_under(stage: Usd.Stage, root_path: str) -> str:
    root_prim = stage.GetPrimAtPath(root_path)
    if not root_prim.IsValid():
        raise RuntimeError(f"Root prim not valid: {root_path}")
    for prim in Usd.PrimRange(root_prim):
        if prim.IsA(UsdSkel.Skeleton):
            return prim.GetPath().pathString
    raise RuntimeError(f"No UsdSkel.Skeleton found under {root_path}")


def get_skeleton_joint_count(stage: Usd.Stage, skel_path: str) -> int:
    prim = stage.GetPrimAtPath(skel_path)
    if not prim.IsValid():
        return 0
    skel = UsdSkel.Skeleton(prim)
    joints = skel.GetJointsAttr().Get() or []
    return len(joints)


def wait_updates(n: int) -> None:
    for _ in range(int(max(0, n))):
        SIM_APP.update()


def get_world_bounds(stage: Usd.Stage, prim_path: str):
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        return None
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    aligned = bbox_cache.ComputeWorldBound(prim).ComputeAlignedRange()
    return aligned.GetMin(), aligned.GetMax()


def fit_capture_view(stage: Usd.Stage, camera_path: str) -> None:
    cables_bounds = get_world_bounds(stage, "/World/Cables")
    if cables_bounds is None:
        return

    bounds_min, bounds_max = cables_bounds
    center = np.array(
        [
            0.5 * (float(bounds_min[0]) + float(bounds_max[0])),
            0.5 * (float(bounds_min[1]) + float(bounds_max[1])),
            0.5 * (float(bounds_min[2]) + float(bounds_max[2])),
        ],
        dtype=np.float32,
    )
    span = np.array(
        [
            float(bounds_max[0]) - float(bounds_min[0]),
            float(bounds_max[1]) - float(bounds_min[1]),
            float(bounds_max[2]) - float(bounds_min[2]),
        ],
        dtype=np.float32,
    )

    backdrop_prim = stage.GetPrimAtPath("/World/Backdrop")
    if backdrop_prim.IsValid():
        width = float(max(0.6, span[1] * 1.35))
        height = float(max(0.9, span[2] * 1.15))
        xf = UsdGeom.Xformable(backdrop_prim)
        xf.ClearXformOpOrder()
        xf.AddTranslateOp().Set(Gf.Vec3f(float(center[0] - 0.24), float(center[1]), float(center[2])))
        xf.AddScaleOp().Set(Gf.Vec3f(0.005, 0.5 * width, 0.5 * height))

    position_lighting(stage, center, span)

    cam_prim = stage.GetPrimAtPath(camera_path)
    if not cam_prim.IsValid():
        return

    focal_length = 28.0
    desired_height = float(max(0.9, span[2] * 1.15))
    vfov = 2.0 * np.arctan(24.576 / (2.0 * focal_length))
    distance = float(max(0.95, 0.5 * desired_height / np.tan(0.5 * vfov) + 0.12))
    eye = Gf.Vec3d(
        float(center[0] + 0.92 * distance),
        float(center[1] - 0.38 * distance),
        float(center[2] + 0.16 * distance),
    )
    target = Gf.Vec3d(float(center[0]), float(center[1]), float(center[2] + 0.05 * span[2]))

    cam = UsdGeom.Camera(cam_prim)
    cam.GetFocalLengthAttr().Set(focal_length)
    xf = UsdGeom.Xformable(cam_prim)
    xf.ClearXformOpOrder()
    xf.AddTransformOp().Set(Gf.Matrix4d().SetLookAt(eye, target, Gf.Vec3d(0.0, 0.0, 1.0)).GetInverse())


def compute_scene_bounds(pos_arr: np.ndarray):
    mins = pos_arr.min(axis=(0, 1, 3))
    maxs = pos_arr.max(axis=(0, 1, 3))
    center = 0.5 * (mins + maxs)
    span = maxs - mins
    return mins, maxs, center, span


def make_backdrop(stage: Usd.Stage, center: np.ndarray, span: np.ndarray) -> None:
    width = float(max(0.6, span[1] * 1.3))
    height = float(max(0.9, span[2] * 1.15))
    backdrop = UsdGeom.Cube.Define(stage, "/World/Backdrop")
    xf = UsdGeom.Xformable(backdrop.GetPrim())
    xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set(Gf.Vec3f(float(center[0] - 0.24), float(center[1]), float(center[2])))
    xf.AddScaleOp().Set(Gf.Vec3f(0.005, 0.5 * width, 0.5 * height))
    backdrop.CreateDisplayColorAttr().Set([Gf.Vec3f(0.92, 0.92, 0.94)])


def create_preview_material(
    stage: Usd.Stage,
    material_path: str,
    diffuse_rgb: tuple[float, float, float],
    roughness: float,
    emissive_rgb: tuple[float, float, float] = (0.0, 0.0, 0.0),
):
    material = UsdShade.Material.Define(stage, material_path)
    shader = UsdShade.Shader.Define(stage, f"{material_path}/PreviewSurface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*diffuse_rgb))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(float(roughness))
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(1.0)
    shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*emissive_rgb))
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return material


def position_lighting(stage: Usd.Stage, center: np.ndarray, span: np.ndarray) -> None:
    scene_scale = float(max(np.max(span), 0.45))

    fill = stage.GetPrimAtPath("/World/FillLight")
    if fill.IsValid():
        fill_xf = UsdGeom.Xformable(fill)
        fill_xf.ClearXformOpOrder()
        fill_xf.AddTranslateOp().Set(
            Gf.Vec3f(
                float(center[0] + 0.70 * scene_scale),
                float(center[1] - 0.55 * scene_scale),
                float(center[2] + 0.55 * scene_scale),
            )
        )

    rim = stage.GetPrimAtPath("/World/RimLight")
    if rim.IsValid():
        rim_xf = UsdGeom.Xformable(rim)
        rim_xf.ClearXformOpOrder()
        rim_xf.AddTranslateOp().Set(
            Gf.Vec3f(
                float(center[0] - 0.20 * scene_scale),
                float(center[1] + 0.80 * scene_scale),
                float(center[2] + 0.40 * scene_scale),
            )
        )


def bind_materials(stage: Usd.Stage) -> None:
    cable_material = create_preview_material(
        stage,
        "/World/Looks/Cable",
        diffuse_rgb=(1.00, 0.42, 0.05),
        roughness=0.12,
        emissive_rgb=(0.28, 0.10, 0.01),
    )
    backdrop_material = create_preview_material(
        stage,
        "/World/Looks/Backdrop",
        diffuse_rgb=(0.08, 0.09, 0.10),
        roughness=0.98,
    )

    backdrop_prim = stage.GetPrimAtPath("/World/Backdrop")
    if backdrop_prim.IsValid():
        UsdShade.MaterialBindingAPI(backdrop_prim).Bind(backdrop_material)

    cables_root = stage.GetPrimAtPath("/World/Cables")
    if not cables_root.IsValid():
        return

    for prim in Usd.PrimRange(cables_root):
        if prim.IsA(UsdGeom.Mesh):
            mesh = UsdGeom.Mesh(prim)
            mesh.CreateDisplayColorAttr().Set([Gf.Vec3f(1.00, 0.42, 0.05)])
            UsdShade.MaterialBindingAPI(prim).Bind(
                cable_material, bindingStrength=UsdShade.Tokens.strongerThanDescendants
            )


def make_camera(stage: Usd.Stage, center: np.ndarray, span: np.ndarray) -> str:
    desired_height = float(max(0.9, span[2] * 1.15))
    focal_length = 28.0
    vfov = 2.0 * np.arctan(24.576 / (2.0 * focal_length))
    distance = float(max(0.95, 0.5 * desired_height / np.tan(0.5 * vfov) + 0.12))
    eye = Gf.Vec3d(
        float(center[0] + 0.92 * distance),
        float(center[1] - 0.38 * distance),
        float(center[2] + 0.16 * distance),
    )
    target = Gf.Vec3d(float(center[0]), float(center[1]), float(center[2] + 0.05 * span[2]))
    cam = UsdGeom.Camera.Define(stage, "/World/Camera")
    cam.GetFocalLengthAttr().Set(focal_length)
    cam.GetHorizontalApertureAttr().Set(24.576)
    cam.GetVerticalApertureAttr().Set(24.576)
    xf = UsdGeom.Xformable(cam.GetPrim())
    xf.ClearXformOpOrder()
    cam_to_world = Gf.Matrix4d().SetLookAt(eye, target, Gf.Vec3d(0.0, 0.0, 1.0)).GetInverse()
    xf.AddTransformOp().Set(cam_to_world)
    return cam.GetPath().pathString


def make_lighting(stage: Usd.Stage, center: np.ndarray, span: np.ndarray) -> None:
    key = UsdLux.DistantLight.Define(stage, "/World/KeyLight")
    key.CreateIntensityAttr(14000.0)
    key.CreateAngleAttr(0.53)
    key_xf = UsdGeom.Xformable(key.GetPrim())
    key_xf.ClearXformOpOrder()
    key_xf.AddRotateXYZOp().Set(Gf.Vec3f(-32.0, 0.0, 18.0))

    fill = UsdLux.SphereLight.Define(stage, "/World/FillLight")
    fill.CreateIntensityAttr(85000.0)
    fill.CreateRadiusAttr(0.18)

    rim = UsdLux.SphereLight.Define(stage, "/World/RimLight")
    rim.CreateIntensityAttr(42000.0)
    rim.CreateRadiusAttr(0.16)
    rim.CreateColorAttr(Gf.Vec3f(0.90, 0.94, 1.0))
    position_lighting(stage, center, span)


def build_stage(cable_usd: Path, num_wires: int, center: np.ndarray, span: np.ndarray):
    ctx = omni.usd.get_context()
    ctx.new_stage()
    wait_updates(20)
    stage = ctx.get_stage()
    if stage is None:
        raise RuntimeError("Failed to create a new stage.")

    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.Xform.Define(stage, "/World")

    make_backdrop(stage, center, span)
    make_lighting(stage, center, span)
    camera_path = make_camera(stage, center, span)

    cable_parent = UsdGeom.Xform.Define(stage, "/World/Cables")
    asset_default_path = get_asset_default_prim_path(str(cable_usd))

    skel_paths: list[str] = []
    for wire_index in range(int(num_wires)):
        root_path = f"/World/Cables/Wire_{wire_index:03d}"
        root_xf = UsdGeom.Xform.Define(stage, root_path)
        root_prim = root_xf.GetPrim()
        UsdGeom.Xformable(root_prim).ClearXformOpOrder()
        root_prim.GetReferences().AddReference(str(cable_usd), asset_default_path)

    wait_updates(20)

    for wire_index in range(int(num_wires)):
        root_path = f"/World/Cables/Wire_{wire_index:03d}"
        skel_paths.append(find_first_skeleton_under(stage, root_path))

    bind_materials(stage)
    wait_updates(12)

    return stage, camera_path, skel_paths


def build_drivers(stage: Usd.Stage, skel_paths: list[str], expected_elems: int):
    drivers: list[SkeletonRodDriver] = []
    for skel_path in skel_paths:
        num_joints = get_skeleton_joint_count(stage, skel_path)
        if num_joints != int(expected_elems):
            raise RuntimeError(
                f"Skeleton joints mismatch for {skel_path}: joints={num_joints}, expected={expected_elems}"
            )
        driver = SkeletonRodDriver(stage, skel_path, assume_chain=True)
        driver.skel_prim = stage.GetPrimAtPath(skel_path)
        driver.skeleton_path = skel_path
        driver._setup_animation()
        drivers.append(driver)
    return drivers


def render_animation(
    pos_arr: np.ndarray,
    dir_arr: np.ndarray,
    frame_indices: list[int],
    out_dir: Path,
    width: int,
    height: int,
    rt_subframes: int,
) -> None:
    _, num_wires, _, num_nodes = pos_arr.shape
    expected_elems = int(num_nodes - 1)
    _, _, center, span = compute_scene_bounds(pos_arr)

    stage, camera_path, skel_paths = build_stage(Path(ARGS.cable_usd), num_wires, center, span)
    drivers = build_drivers(stage, skel_paths, expected_elems)

    first_frame = int(frame_indices[0])
    for wire_index, driver in enumerate(drivers):
        driver.update_skeleton(pos_arr[first_frame, wire_index], dir_arr[first_frame, wire_index], Usd.TimeCode.Default())
    wait_updates(4)
    fit_capture_view(stage, camera_path)
    wait_updates(8)

    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    set_render_quality()
    render_product = rep.create.render_product(camera_path, resolution=(int(width), int(height)))
    writer = rep.WriterRegistry.get("BasicWriter")
    writer.initialize(output_dir=str(frames_dir), rgb=True)
    writer.attach([render_product])
    wait_updates(8)

    for out_index, frame in enumerate(frame_indices):
        for wire_index, driver in enumerate(drivers):
            driver.update_skeleton(pos_arr[frame, wire_index], dir_arr[frame, wire_index], Usd.TimeCode.Default())
        wait_updates(4)
        print(f"[RENDER] {out_index + 1}/{len(frame_indices)} npz_frame={frame}")
        rep.orchestrator.step(rt_subframes=int(max(1, rt_subframes)))
        SIM_APP.update()

    try:
        writer.detach()
    except Exception:
        pass


def assemble_video(out_dir: Path, fps: float, video_name: str) -> Path | None:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        print("[WARN] ffmpeg not found; leaving the frame sequence only.")
        return None

    frames_pattern = str((out_dir / "frames" / "rgb_%04d.png").resolve())
    video_path = (out_dir / video_name).resolve()
    cmd = [
        ffmpeg,
        "-y",
        "-framerate",
        f"{float(fps):.6f}",
        "-i",
        frames_pattern,
        "-pix_fmt",
        "yuv420p",
        str(video_path),
    ]
    subprocess.run(cmd, check=True)
    return video_path


def main() -> None:
    npz_path = Path(ARGS.npz).resolve()
    cable_usd = Path(ARGS.cable_usd).resolve()
    out_dir = Path(ARGS.out).resolve()

    if not cable_usd.is_file():
        raise FileNotFoundError(f"Cable USD not found: {cable_usd}")

    time_arr, pos_arr, dir_arr = validate_npz(npz_path)
    total_frames = int(pos_arr.shape[0])
    frame_start = int(max(0, ARGS.frame_start))
    frame_end = total_frames - 1 if ARGS.frame_end is None else int(ARGS.frame_end)
    frame_end = min(frame_end, total_frames - 1)
    frame_step = int(max(1, ARGS.frame_step))
    frame_indices = list(range(frame_start, frame_end + 1, frame_step))
    if not frame_indices:
        raise RuntimeError("No frames selected.")

    out_dir.mkdir(parents=True, exist_ok=True)
    fps = 1.0 / float(time_arr[1] - time_arr[0]) if len(time_arr) > 1 else 30.0

    print(f"[INFO] npz={npz_path}")
    print(f"[INFO] cable_usd={cable_usd}")
    print(f"[INFO] frames={frame_indices[0]}..{frame_indices[-1]} step={frame_step} count={len(frame_indices)}")
    print(f"[INFO] wires={pos_arr.shape[1]} output={out_dir}")

    render_animation(
        pos_arr=pos_arr,
        dir_arr=dir_arr,
        frame_indices=frame_indices,
        out_dir=out_dir,
        width=int(ARGS.width),
        height=int(ARGS.height),
        rt_subframes=int(ARGS.rt_subframes),
    )

    if not ARGS.skip_video:
        video_path = assemble_video(out_dir, fps / frame_step, ARGS.video_name)
        if video_path is not None:
            print(f"[INFO] video={video_path}")

    print("[INFO] Done.")
    SIM_APP.close()


if __name__ == "__main__":
    main()
