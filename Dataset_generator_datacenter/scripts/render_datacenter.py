#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Isaac Sim 4.5 Standalone – with RGB / Depth / Instance Segmentation capture
- Reads NPZ (pos/positions + optional director + time)
- References cable USD asset N times
- AUTO-FIXES missing SkelRoot + mesh bindings
- Author skeleton animation per wire
- Renders RGB, Depth, Instance Segmentation per frame from a specified camera

Run:
  /home/robot/isaacsim/python.sh test_drop_multi_diff_rods.py
"""

from __future__ import annotations
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import os
import numpy as np
from pxr import Usd, UsdGeom, UsdLux, UsdSkel, Gf
import omni.usd
import omni.timeline

from rod_skel_driver import SkeletonRodDriver

# ============================================================
# CONFIG
# ============================================================
CABLE_ASSET_PATH = "/home/robot/Workspace/CosseratX/Dataset_generator_datacenter/data/cables/data cernter cable_blue.usdc"
NPZ_PATH = "/home/robot/Workspace/CosseratX/Dataset_generator_datacenter/data/npz_file/hard.npz"

OFFSET_MODE = "line"
OFFSET_STEP = 0.00
GRID_COLS = 5
ASSUME_CHAIN_PARENT = True

# Unit scale: cable asset & NPZ positions are in m, background scene is in mm.
POS_SCALE = 100.0
HDR_PATH = "/home/robot/Downloads/charolettenbrunn_park_4k.hdr"
DOME_INTENSITY = 500.0
DOME_EXPOSURE = 0.0
EXPORT_STAGE_PATH = ""
SKIP_MISMATCH_WIRES = True

# Background scene USDs: list of (path, scale) tuples
BACKGROUND_USDS = [
    ("/home/robot/Workspace/CosseratX/Dataset_generator_datacenter/data/Datacenter_NVD@10012/Assets/DigitalTwin/Assets/Datacenter/Facilities/Stages/Data_Hall/DataHall_Full_01.usd", 1.0),
    ("/home/robot/Workspace/CosseratX/Dataset_generator_datacenter/data/data_center_scene.usdc", 100.0),
]

# ---- Camera & Rendering Config ----
# Camera transform (from Isaac Sim viewport)
CAMERA_TRANSLATE = (13, -60, 163)
CAMERA_ROTATE    = (102, 0, 49)  # Euler XYZ degrees
RENDER_WIDTH  = 1024
RENDER_HEIGHT = 1024

# Which frames to capture: "all", "last", or a list like [0, 10, 20]
CAPTURE_FRAMES = "all"

# Output directory for rendered images
RENDER_OUTPUT_DIR = "/home/robot/Workspace/CosseratX/Dataset_generator_datacenter/data/renders"

# Semantic class name for wire instances
WIRE_SEMANTIC_CLASS = "cable"


# ============================================================
# NPZ Loader
# ============================================================
def load_npz_data(npz_path):
    data = np.load(npz_path)
    print("NPZ keys:", data.files)
    for k in data.files:
        print(f"  {k:20s} shape={data[k].shape}, dtype={data[k].dtype}")

    if "positions" in data.files:
        positions = data["positions"]
    elif "pos" in data.files:
        positions = data["pos"]
    else:
        raise RuntimeError("NPZ must contain 'positions' or 'pos'")

    if positions.ndim != 4 or positions.shape[2] != 3:
        raise RuntimeError(f"positions must be (T,N,3,K). Got {positions.shape}")

    T, N, _, K = positions.shape

    director = None
    if "director" in data.files:
        director = data["director"]
        if director.ndim != 5 or director.shape[2] != 3 or director.shape[3] != 3:
            raise RuntimeError(f"director must be (T,N,3,3,D). Got {director.shape}")
        if director.shape[0] != T or director.shape[1] != N:
            raise RuntimeError(f"director T/N mismatch")
        D = director.shape[4]
        if K != D + 1:
            raise RuntimeError(f"K must equal D+1: K={K}, D={D}")
        print(f"[INFO] Using director from NPZ (T={T}, N={N}, D={D})")
    else:
        print(f"[INFO] No 'director' in NPZ — will compute from positions")

    if "time" in data.files:
        time_arr = data["time"]
    elif "dt" in data.files:
        time_arr = np.arange(T, dtype=np.float64) * float(data["dt"])
    else:
        raise RuntimeError("NPZ must contain 'time' or 'dt'")

    return positions, director, time_arr


# ============================================================
# Helpers
# ============================================================
def make_offsets(n):
    if OFFSET_MODE == "none":
        return [Gf.Vec3f(0, 0, 0)] * n
    if OFFSET_MODE == "line":
        return [Gf.Vec3f(0, float(OFFSET_STEP) * i, 0) for i in range(n)]
    if OFFSET_MODE == "grid":
        cols = max(1, int(GRID_COLS))
        return [Gf.Vec3f(0, float(OFFSET_STEP)*(i%cols), float(OFFSET_STEP)*(i//cols)) for i in range(n)]
    raise ValueError(f"Unknown OFFSET_MODE: {OFFSET_MODE}")


def get_asset_default_prim_path(path):
    s = Usd.Stage.Open(path)
    dp = s.GetDefaultPrim()
    if not dp.IsValid():
        raise RuntimeError(f"No defaultPrim in {path}")
    return dp.GetPath()


def find_first_skeleton_under(stage, root_path):
    for p in Usd.PrimRange(stage.GetPrimAtPath(root_path)):
        if p.IsA(UsdSkel.Skeleton):
            return p.GetPath().pathString
    raise RuntimeError(f"No Skeleton under {root_path}")


def get_skeleton_joint_count(stage, skel_path):
    prim = stage.GetPrimAtPath(skel_path)
    if not prim.IsValid():
        return 0
    return len(UsdSkel.Skeleton(prim).GetJointsAttr().Get() or [])


def find_skel_root_above(stage, skel_path):
    prim = stage.GetPrimAtPath(skel_path)
    while prim and prim.IsValid():
        if prim.GetTypeName() == "SkelRoot":
            return prim
        prim = prim.GetParent()
    return None


def promote_to_skel_root(stage, prim):
    path = prim.GetPath()
    if prim.GetTypeName() == "SkelRoot":
        return
    print(f"  [FIX] Promoting {path} from '{prim.GetTypeName()}' to 'SkelRoot'")
    edit_layer = stage.GetEditTarget().GetLayer()
    prim_spec = edit_layer.GetPrimAtPath(path)
    if prim_spec is None:
        from pxr import Sdf
        prim_spec = Sdf.CreatePrimInLayer(edit_layer, path)
    prim_spec.typeName = "SkelRoot"


def ensure_mesh_skeleton_binding(stage, wire_root_path, skel_path):
    skel_sdf_path = stage.GetPrimAtPath(skel_path).GetPath()
    bound_count = 0
    for prim in Usd.PrimRange(stage.GetPrimAtPath(wire_root_path)):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        bind = UsdSkel.BindingAPI(prim)
        ji = bind.GetJointIndicesPrimvar()
        jw = bind.GetJointWeightsPrimvar()
        ji_ok = bool(ji and ji.GetAttr().IsValid() and ji.GetAttr().HasAuthoredValue())
        jw_ok = bool(jw and jw.GetAttr().IsValid() and jw.GetAttr().HasAuthoredValue())
        if ji_ok and jw_ok:
            UsdSkel.BindingAPI.Apply(prim)
            UsdSkel.BindingAPI(prim).CreateSkeletonRel().SetTargets([skel_sdf_path])
            bound_count += 1
    if bound_count > 0:
        print(f"  [FIX] Bound {bound_count} mesh(es) to skeleton {skel_path}")
    else:
        print(f"  [WARN] No skinned meshes found under {wire_root_path}!")


def bind_head_meshes_to_first_joint(stage, wire_root_path, skel_path):
    skel_sdf_path = stage.GetPrimAtPath(skel_path).GetPath()
    head_bound = 0
    for prim in Usd.PrimRange(stage.GetPrimAtPath(wire_root_path)):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        bind = UsdSkel.BindingAPI(prim)
        ji = bind.GetJointIndicesPrimvar()
        jw = bind.GetJointWeightsPrimvar()
        ji_ok = bool(ji and ji.GetAttr().IsValid() and ji.GetAttr().HasAuthoredValue())
        jw_ok = bool(jw and jw.GetAttr().IsValid() and jw.GetAttr().HasAuthoredValue())
        if ji_ok and jw_ok:
            continue
        path_str = prim.GetPath().pathString.lower()
        if "/head" not in path_str:
            continue
        mesh = UsdGeom.Mesh(prim)
        points = mesh.GetPointsAttr().Get()
        if not points:
            continue
        num_verts = len(points)
        UsdSkel.BindingAPI.Apply(prim)
        bind_applied = UsdSkel.BindingAPI(prim)
        ji_pv = bind_applied.CreateJointIndicesPrimvar(False, elementSize=1)
        ji_pv.Set([0] * num_verts)
        jw_pv = bind_applied.CreateJointWeightsPrimvar(False, elementSize=1)
        jw_pv.Set([1.0] * num_verts)
        bind_applied.CreateSkeletonRel().SetTargets([skel_sdf_path])
        head_bound += 1
        print(f"  [FIX] Head mesh {prim.GetPath().name} ({num_verts} verts) → bone_0")
    if head_bound == 0:
        print(f"  [INFO] No unbound head meshes found under {wire_root_path}")


def build_rod_dir_from_positions(pos):
    T, W, _, N = pos.shape
    E = N - 1
    dp = pos[:, :, :, 1:] - pos[:, :, :, :-1]
    t = dp / (np.linalg.norm(dp, axis=2, keepdims=True) + 1e-12)
    up1 = np.array([0., 0., 1.])[None, None, :, None]
    up2 = np.array([0., 1., 0.])[None, None, :, None]
    n = np.cross(up1, t, axis=2)
    n_norm = np.linalg.norm(n, axis=2, keepdims=True)
    mask = (n_norm < 1e-8)
    if np.any(mask):
        n2 = np.cross(up2, t, axis=2)
        n2_norm = np.linalg.norm(n2, axis=2, keepdims=True)
        n = np.where(mask, n2, n)
        n_norm = np.where(mask, n2_norm, n_norm)
    n = n / (n_norm + 1e-12)
    b = np.cross(t, n, axis=2)
    b = b / (np.linalg.norm(b, axis=2, keepdims=True) + 1e-12)
    rod_dir = np.zeros((T, W, 3, 3, E), dtype=np.float64)
    rod_dir[:, :, :, 0, :] = t
    rod_dir[:, :, :, 1, :] = n
    rod_dir[:, :, :, 2, :] = b
    return rod_dir


# ============================================================
# Semantic labels for instance segmentation
# ============================================================
def add_semantic_labels(stage, wire_roots):
    """
    Add semantic labels to each wire for instance segmentation.
    Labels both the wire root AND all meshes underneath, because
    instance_segmentation annotator works at the mesh/prim level.
    """
    from pxr import Sdf

    for i, rp in enumerate(wire_roots):
        prim = stage.GetPrimAtPath(rp)
        if not prim.IsValid():
            continue

        label = f"{WIRE_SEMANTIC_CLASS}_{i}"

        # Label the wire root and ALL prims underneath
        for p in Usd.PrimRange(prim):
            # Each prim can have multiple semantic tags; use unique API name per wire
            api_name = f"Semantics_{WIRE_SEMANTIC_CLASS}_{i}"
            p.CreateAttribute(
                f"semantic:{api_name}:params:semanticType",
                Sdf.ValueTypeNames.String
            ).Set("class")
            p.CreateAttribute(
                f"semantic:{api_name}:params:semanticData",
                Sdf.ValueTypeNames.String
            ).Set(label)

        print(f"  [INFO] Semantic: {rp} (+ children) → {label}")


# ============================================================
# Build stage
# ============================================================
def build_stage(num_wires):
    ctx = omni.usd.get_context()
    ctx.new_stage()
    stage = ctx.get_stage()

    # Load background scenes
    if BACKGROUND_USDS:
        for idx, (bg_path, bg_scale) in enumerate(BACKGROUND_USDS):
            if not os.path.isfile(bg_path):
                print(f"[WARN] Background USD not found: {bg_path}")
                continue
            bg_prim_path = f"/World/Background_{idx}"
            bg_prim = UsdGeom.Xform.Define(stage, bg_prim_path).GetPrim()
            if bg_scale != 1.0:
                xf = UsdGeom.Xformable(bg_prim)
                xf.ClearXformOpOrder()
                xf.AddScaleOp().Set(Gf.Vec3f(bg_scale, bg_scale, bg_scale))
            bg_dp = get_asset_default_prim_path(bg_path)
            bg_prim.GetReferences().AddReference(bg_path, bg_dp)
            print(f"[INFO] Background {idx}: {os.path.basename(bg_path)} (scale={bg_scale})")
    else:
        dome = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
        if HDR_PATH and os.path.isfile(HDR_PATH):
            dome.CreateTextureFileAttr(HDR_PATH)
        dome.CreateIntensityAttr(DOME_INTENSITY)
        dome.CreateExposureAttr(DOME_EXPOSURE)

    offsets = make_offsets(num_wires)
    asset_dp = get_asset_default_prim_path(CABLE_ASSET_PATH)
    print(f"[INFO] Asset defaultPrim = {asset_dp}")

    wire_roots = []
    for i in range(num_wires):
        rp = f"/World/Wire_{i}"
        wire_roots.append(rp)
        xf = UsdGeom.Xform.Define(stage, rp)
        prim = xf.GetPrim()
        UsdGeom.Xformable(prim).ClearXformOpOrder()
        UsdGeom.Xformable(prim).AddTranslateOp().Set(offsets[i])
        if POS_SCALE != 1.0:
            UsdGeom.Xformable(prim).AddScaleOp().Set(Gf.Vec3f(POS_SCALE, POS_SCALE, POS_SCALE))
        prim.GetReferences().AddReference(CABLE_ASSET_PATH, asset_dp)

    for _ in range(8):
        simulation_app.update()

    skel_paths = []
    for i, rp in enumerate(wire_roots):
        try:
            skel_path = find_first_skeleton_under(stage, rp)
            print(f"[INFO] Wire {i} skeleton = {skel_path}")

            skel_root = find_skel_root_above(stage, skel_path)
            if skel_root is None:
                promote_to_skel_root(stage, stage.GetPrimAtPath(rp))

            ensure_mesh_skeleton_binding(stage, rp, skel_path)
            bind_head_meshes_to_first_joint(stage, rp, skel_path)
            skel_paths.append(skel_path)
        except Exception as e:
            print(f"[WARN] Wire {i}: {e}")
            skel_paths.append("")

    # Add semantic labels for instance segmentation
    add_semantic_labels(stage, wire_roots)

    for _ in range(4):
        simulation_app.update()

    # Only keep DomeLights from Background_0, disable all others
    for prim in stage.Traverse():
        if prim.IsA(UsdLux.DomeLight):
            path_str = prim.GetPath().pathString
            if "/World/Background_0" in path_str:
                continue  # keep this one
            UsdGeom.Imageable(prim).GetVisibilityAttr().Set(UsdGeom.Tokens.invisible)
            UsdLux.DomeLight(prim).CreateIntensityAttr().Set(0.0)
            print(f"  [INFO] Disabled DomeLight: {path_str}")

    return stage, wire_roots, skel_paths


# ============================================================
# Camera setup
# ============================================================
def create_camera(stage):
    """Create a camera with translate + rotate from viewport."""
    cam_path = "/World/RenderCamera"
    cam = UsdGeom.Camera.Define(stage, cam_path)

    xformable = UsdGeom.Xformable(cam.GetPrim())
    xformable.ClearXformOpOrder()

    # Translate
    xformable.AddTranslateOp().Set(
        Gf.Vec3d(*CAMERA_TRANSLATE)
    )

    # Rotate: Isaac Sim viewport uses XYZ Euler (in degrees)
    rx, ry, rz = CAMERA_ROTATE
    xformable.AddRotateXYZOp().Set(Gf.Vec3f(rx, ry, rz))

    cam.CreateFocalLengthAttr(24.0)
    cam.CreateHorizontalApertureAttr(36.0)
    cam.CreateClippingRangeAttr(Gf.Vec2f(0.1, 100000.0))

    print(f"[INFO] Camera: {cam_path}")
    print(f"  Translate: {CAMERA_TRANSLATE}")
    print(f"  Rotate:    {CAMERA_ROTATE}")
    return cam_path


# ============================================================
# Render frames: RGB, Depth, Instance Segmentation
# ============================================================
def render_frames(stage, wire_roots, positions, time_arr):
    """
    Render RGB, depth, and instance segmentation using BasicWriter
    (same approach as the reference generator.py).
    """
    import omni.replicator.core as rep

    T = positions.shape[0]
    dt = float(time_arr[1] - time_arr[0])
    fps = 1.0 / dt
    fps = 5
    # Create output dirs
    rgb_dir = os.path.join(RENDER_OUTPUT_DIR, "rgb")
    depth_dir = os.path.join(RENDER_OUTPUT_DIR, "depth")
    seg_dir = os.path.join(RENDER_OUTPUT_DIR, "instance_seg")
    for d in [rgb_dir, depth_dir, seg_dir]:
        os.makedirs(d, exist_ok=True)

    # Frames to capture
    if CAPTURE_FRAMES == "all":
        frames = list(range(T))
    elif CAPTURE_FRAMES == "last":
        frames = [T - 1]
    else:
        frames = [f for f in CAPTURE_FRAMES if 0 <= f < T]

    print(f"[INFO] Capturing {len(frames)} frames → {RENDER_OUTPUT_DIR}")

    # Create render product from our camera
    rp = rep.create.render_product("/World/RenderCamera", (RENDER_WIDTH, RENDER_HEIGHT))

    # Create depth annotator (separate from BasicWriter for raw float output)
    depth_ann = rep.AnnotatorRegistry.get_annotator("distance_to_camera")
    depth_ann.attach([rp])

    timeline = omni.timeline.get_timeline_interface()

    def safe_writer_detach(writer):
        for meth in ["detach", "clear", "reset"]:
            if hasattr(writer, meth):
                try:
                    getattr(writer, meth)()
                    return
                except Exception:
                    pass

    for fi, frame in enumerate(frames):
        # Seek timeline to target frame
        timeline.set_current_time(frame / fps)

        # Let renderer fully settle (avoid blur / ghosting)
        for _ in range(50):
            simulation_app.update()

        # ---- RGB (BasicWriter) ----
        print(f"  [RENDER] Frame {frame}: writing RGB...")
        writer = rep.WriterRegistry.get("BasicWriter")
        safe_writer_detach(writer)
        writer.initialize(output_dir=rgb_dir, rgb=True)
        writer.attach([rp])
        rep.orchestrator.step(rt_subframes=16)
        simulation_app.update()
        safe_writer_detach(writer)

        # ---- Instance Segmentation (BasicWriter) ----
        print(f"  [RENDER] Frame {frame}: writing instance_id_segmentation...")
        writer = rep.WriterRegistry.get("BasicWriter")
        safe_writer_detach(writer)
        writer.initialize(output_dir=seg_dir, instance_id_segmentation=True)
        writer.attach([rp])
        rep.orchestrator.step(rt_subframes=1)
        simulation_app.update()
        safe_writer_detach(writer)

        # ---- Depth (Annotator → npy + png) ----
        print(f"  [RENDER] Frame {frame}: capturing depth...")
        rep.orchestrator.step(rt_subframes=1)
        simulation_app.update()

        depth = depth_ann.get_data()
        if depth is not None and depth.size > 0:
            from PIL import Image
            depth = np.array(depth, dtype=np.float32)
            np.save(os.path.join(depth_dir, f"{frame:04d}.npy"), depth)
            # Visualization PNG
            valid = depth[depth < 1e6]
            if valid.size > 0:
                dmin, dmax = valid.min(), valid.max()
                vis = np.clip((depth - dmin) / (dmax - dmin + 1e-8) * 255, 0, 255).astype(np.uint8)
            else:
                vis = np.zeros_like(depth, dtype=np.uint8)
            Image.fromarray(vis).save(os.path.join(depth_dir, f"{frame:04d}.png"))
            print(f"  [RENDER] Depth saved: {frame:04d}.npy + .png")
        else:
            print(f"  [WARN] Depth data empty for frame {frame}")

        print(f"  [RENDER] Frame {frame} complete")

    print(f"[INFO] Render complete → {RENDER_OUTPUT_DIR}")
    print(f"  rgb/:          {rgb_dir}")
    print(f"  depth/:        {depth_dir}")
    print(f"  instance_seg/: {seg_dir}")


# ============================================================
# Author animation
# ============================================================
def author_animation(stage, skel_paths, positions, director, time_arr):
    T, W_data, _, N_nodes = positions.shape
    E = N_nodes - 1
    dt = float(time_arr[1] - time_arr[0])
    fps = 1.0 / dt

    if director is not None:
        rod_dir = director
        print(f"[INFO] Using NPZ director directly: {rod_dir.shape}")
    else:
        rod_dir = build_rod_dir_from_positions(positions)
        print(f"[INFO] Computed rod_dir from positions: {rod_dir.shape}")

    W_use = min(W_data, len(skel_paths))
    print(f"[INFO] W_data={W_data}, W_skel={len(skel_paths)}, W_use={W_use}")

    drivers = []
    for i in range(W_use):
        sp = skel_paths[i]
        if not sp:
            continue
        prim = stage.GetPrimAtPath(sp)
        if not prim.IsValid():
            continue
        nj = get_skeleton_joint_count(stage, sp)
        if nj != E:
            msg = f"Wire {i}: joints={nj} != E={E}"
            if SKIP_MISMATCH_WIRES:
                print(f"[WARN] {msg} => SKIP")
                continue
            raise RuntimeError(msg)
        driver = SkeletonRodDriver(stage, sp, assume_chain=ASSUME_CHAIN_PARENT)
        driver.skel_prim = prim
        driver.skeleton_path = sp
        driver._setup_animation()
        drivers.append((i, driver))

    if not drivers:
        raise RuntimeError("No compatible wires.")

    print(f"[INFO] Authoring {T} frames, fps={fps:.3f}, wires={len(drivers)}")
    for frame in range(T):
        tc = Usd.TimeCode(frame)
        for wi, drv in drivers:
            drv.update_skeleton(positions[frame, wi], rod_dir[frame, wi], tc)

    stage.SetTimeCodesPerSecond(fps)
    stage.SetFramesPerSecond(fps)

    tl = omni.timeline.get_timeline_interface()
    tl.set_start_time(0.0)
    tl.set_end_time((T - 1) / fps)
    tl.set_current_time(0.0)
    print("[INFO] Animation authored.")


# ============================================================
# Main
# ============================================================
def main():
    positions, director, time_arr = load_npz_data(NPZ_PATH)
    T, W, _, K = positions.shape
    print(f"[INFO] NPZ: T={T}, W={W}, K={K}")

    stage, wire_roots, skel_paths = build_stage(W)
    author_animation(stage, skel_paths, positions, director, time_arr)

    if EXPORT_STAGE_PATH:
        stage.GetRootLayer().Export(EXPORT_STAGE_PATH)
        print(f"[INFO] Exported: {EXPORT_STAGE_PATH}")

    # Create camera and render
    create_camera(stage)

    # Let scene settle
    for _ in range(10):
        simulation_app.update()

    # Render all frames
    render_frames(stage, wire_roots, positions, time_arr)

    # Interactive viewing
    omni.timeline.get_timeline_interface().play()
    print("Running... close window to exit.")
    while simulation_app.is_running():
        simulation_app.update()
    simulation_app.close()


if __name__ == "__main__":
    main()
