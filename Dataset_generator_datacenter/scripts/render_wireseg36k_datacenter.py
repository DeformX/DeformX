#!/usr/bin/env python3
"""Render the wireseg32k datacenter dataset.

This runner uses the new asset layout:
  asset_wireseg32k/datacenter/

Dataset rules:
  - dgrid_cX_nsYY.npz uses data center camera X.usdc
  - data center lights.usdc is shared by all configs
  - pos/director axis 0 is the trajectory-data axis
  - seed_arr stores the source trajectory seed ids
  - only the final time frame of each trajectory is rendered
  - --max_seeds selects render-randomization variants per trajectory
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import os
import random
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
DATACENTER_DIR = Path(__file__).resolve().parents[1]
DEFAULT_ASSET_ROOT = REPO_ROOT / "asset_wireseg32k" / "datacenter"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "output" / "wireseg32k" / "datacenter"
DEFAULT_DATAHALL_USD = (
    DEFAULT_ASSET_ROOT
    / "Datacenter_NVD@10012"
    / "Assets"
    / "DigitalTwin"
    / "Assets"
    / "Datacenter"
    / "Facilities"
    / "Stages"
    / "Data_Hall"
    / "DataHall_Full_01.usd"
)
DEFAULT_SCENE_USD = DATACENTER_DIR / "data" / "data_center_scene.usdc"

WIRE_SEMANTIC_CLASS = "cable"
POS_SCALE = 100.0
CAMERA_LIGHT_SCALE = 100.0
ASSUME_CHAIN_PARENT = True
SKIP_MISMATCH_WIRES = True

# Filled after SimulationApp starts. Keeping these global avoids importing omni/pxr
# during dry-run and syntax checks.
simulation_app = None
Usd = None
UsdGeom = None
UsdLux = None
UsdSkel = None
Gf = None
Sdf = None
omni = None
SkeletonRodDriver = None


@dataclass
class ReusableStage:
    stage: object
    cables: list[Path]
    all_cameras: list[str]
    all_lights: list[str]
    dome_lights: list[str]
    wire_slots: dict[Path, tuple[list[str], list[str]]]
    all_wire_roots: list[str]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--asset_root", type=Path, default=DEFAULT_ASSET_ROOT)
    p.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    p.add_argument("--datahall_usd", type=Path, default=DEFAULT_DATAHALL_USD)
    p.add_argument("--scene_usd", type=Path, default=DEFAULT_SCENE_USD)
    p.add_argument("--config", action="append", default=[], help="Config name(s), e.g. dgrid_c1_ns04. Can be repeated or comma-separated.")
    p.add_argument("--max_configs", type=int, default=None, help="Limit number of configs after filtering, for smoke tests.")
    p.add_argument("--traj_index", type=int, default=None, help="Render only one trajectory-data index from each NPZ.")
    p.add_argument("--max_trajs", type=int, default=None, help="Limit trajectory-data samples per config, for smoke tests.")
    p.add_argument("--seed_index", type=int, default=None, help="Render only one randomization variant index per trajectory.")
    p.add_argument("--seed_start", type=int, default=0, help="Start randomization variant index per trajectory.")
    p.add_argument("--max_seeds", type=int, default=1, help="Number of randomization variants per trajectory.")
    p.add_argument("--cams_sample_per_frame", "--camera_num", dest="cams_sample_per_frame", type=int, default=1)
    p.add_argument(
        "--wire_color_mode",
        choices=("mixed", "same"),
        default="mixed",
        help="mixed chooses a cable USD per wire; same chooses one cable USD for all wires in a sample.",
    )
    p.add_argument("--lights_on_per_frame", type=int, default=8)
    p.add_argument("--light_intensity_min", type=float, default=500.0)
    p.add_argument("--light_intensity_max", type=float, default=2000.0)
    p.add_argument("--dome_intensity", type=float, default=300.0)
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--do_seg", action="store_true")
    p.add_argument("--do_depth", action="store_true")
    p.add_argument(
        "--render_mode",
        choices=("RayTracedLighting", "PathTracing"),
        default="RayTracedLighting",
        help="RayTracedLighting uses much less GPU memory for the heavy datacenter scene.",
    )
    p.add_argument("--accum_steps", type=int, default=80)
    p.add_argument("--accum_subframes", type=int, default=16)
    p.add_argument("--warmup_updates", type=int, default=60)
    p.add_argument("--settle_updates", type=int, default=20)
    p.add_argument("--reuse_stage", action="store_true", help="Reuse one datacenter stage per config instead of rebuilding it per variant.")
    p.add_argument("--clean", action="store_true", help="Remove each selected config output before rendering it.")
    p.add_argument("--dry_run", action="store_true", help="Validate config selection and count renders without starting Isaac Sim.")
    p.set_defaults(headless=True)
    p.add_argument("--headless", dest="headless", action="store_true")
    p.add_argument("--no-headless", dest="headless", action="store_false")
    return p.parse_args()


def split_config_filters(raw: Iterable[str]) -> set[str]:
    out: set[str] = set()
    for item in raw:
        for part in item.split(","):
            name = part.strip()
            if not name:
                continue
            if name.endswith(".npz"):
                name = name[:-4]
            out.add(name)
    return out


def config_camera_index(npz_path: Path) -> int:
    m = re.match(r"dgrid_c([1-4])_ns\d+\.npz$", npz_path.name)
    if not m:
        raise ValueError(f"Cannot infer camera index from config name: {npz_path.name}")
    return int(m.group(1))


def discover_configs(asset_root: Path, filters: set[str], max_configs: int | None) -> list[Path]:
    traj_root = asset_root / "data_grid_clean"
    configs = sorted(traj_root.glob("dgrid_c*_ns*.npz"))
    if filters:
        configs = [p for p in configs if p.stem in filters]
    if max_configs is not None:
        configs = configs[: max(0, int(max_configs))]
    if not configs:
        raise RuntimeError(f"No datacenter configs selected under {traj_root}")
    return configs


def selected_indices(
    count: int,
    index: int | None,
    start: int,
    max_count: int | None,
    label: str,
) -> list[int]:
    if count < 1:
        raise ValueError(f"{label} count must be >= 1")
    if index is not None:
        if index < 0 or index >= count:
            raise ValueError(f"{label}_index {index} out of range [0, {count - 1}]")
        return [int(index)]
    if start < 0 or start >= count:
        raise ValueError(f"{label}_start {start} out of range [0, {count - 1}]")
    indices = list(range(int(start), count))
    if max_count is not None:
        indices = indices[: max(0, int(max_count))]
    return indices


def selected_traj_indices(num_trajs: int, traj_index: int | None, max_trajs: int | None) -> list[int]:
    return selected_indices(num_trajs, traj_index, 0, max_trajs, "traj")


def selected_variant_indices(seed_start: int, seed_index: int | None, max_seeds: int | None) -> list[int]:
    if seed_index is not None:
        if seed_index < 0:
            raise ValueError("seed_index must be >= 0")
        return [int(seed_index)]
    if seed_start < 0:
        raise ValueError("seed_start must be >= 0")
    if max_seeds is None:
        max_seeds = 1
    if max_seeds < 0:
        raise ValueError("max_seeds must be >= 0")
    return list(range(int(seed_start), int(seed_start) + int(max_seeds)))


def variant_rng_seed(config_name: str, traj_seed: int, traj_index: int, variant_index: int, mode: str) -> int:
    key = f"{config_name}|traj_seed={traj_seed}|traj_index={traj_index}|variant={variant_index}|mode={mode}"
    return int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "big")


def cable_usds(asset_root: Path) -> list[Path]:
    paths = sorted(asset_root.glob("data center cable_*.usdc"))
    if not paths:
        raise RuntimeError(f"No cable USD files found in {asset_root}")
    return paths


def camera_usd_for_config(asset_root: Path, npz_path: Path) -> Path:
    idx = config_camera_index(npz_path)
    path = asset_root / f"data center camera {idx}.usdc"
    if not path.is_file():
        raise RuntimeError(f"Missing camera USD for {npz_path.name}: {path}")
    return path


def lights_usd(asset_root: Path) -> Path:
    path = asset_root / "data center lights.usdc"
    if not path.is_file():
        raise RuntimeError(f"Missing shared datacenter lights USD: {path}")
    return path


def dry_run(args: argparse.Namespace) -> None:
    filters = split_config_filters(args.config)
    configs = discover_configs(args.asset_root, filters, args.max_configs)
    cables = cable_usds(args.asset_root)
    lights = lights_usd(args.asset_root)

    print(f"asset_root: {args.asset_root}")
    print(f"output_root: {args.output_root}")
    print(f"cable_usds: {len(cables)}")
    for p in cables:
        print(f"  {p.name}")
    print(f"lights_usd: {lights.name}")
    print(f"cams_sample_per_frame: {args.cams_sample_per_frame}")
    print(f"wire_color_mode: {args.wire_color_mode}")
    print(f"reuse_stage: {args.reuse_stage}")
    print(f"traj_index: {args.traj_index}")
    print(f"max_trajs: {args.max_trajs}")
    print(f"seed_index: {args.seed_index}")
    print(f"seed_start: {args.seed_start}")
    print(f"max_seeds: {args.max_seeds}")
    print("configs:")

    total = 0
    for npz_path in configs:
        z = np.load(npz_path)
        if "pos" not in z.files or "director" not in z.files or "seed_arr" not in z.files:
            raise RuntimeError(f"{npz_path.name} must contain pos, director, and seed_arr")
        pos = z["pos"]
        director = z["director"]
        seed_arr = z["seed_arr"]
        if pos.ndim != 5:
            raise RuntimeError(f"{npz_path.name}: pos must be (traj,T,W,3,N), got {pos.shape}")
        if director.ndim != 6:
            raise RuntimeError(f"{npz_path.name}: director must be (traj,T,W,3,3,E), got {director.shape}")
        if pos.shape[0] != director.shape[0] or pos.shape[0] != seed_arr.shape[0]:
            raise RuntimeError(f"{npz_path.name}: trajectory axis mismatch")
        if pos.shape[1] != director.shape[1]:
            raise RuntimeError(f"{npz_path.name}: time axis mismatch")
        if pos.shape[2] != director.shape[2]:
            raise RuntimeError(f"{npz_path.name}: wire axis mismatch")
        if pos.shape[3] != 3 or director.shape[3:5] != (3, 3):
            raise RuntimeError(f"{npz_path.name}: xyz/director axis mismatch")
        if pos.shape[4] != director.shape[5] + 1:
            raise RuntimeError(f"{npz_path.name}: nodes must equal segments + 1")

        traj_indices = selected_traj_indices(len(seed_arr), args.traj_index, args.max_trajs)
        variant_indices = selected_variant_indices(args.seed_start, args.seed_index, args.max_seeds)
        camera_usd = camera_usd_for_config(args.asset_root, npz_path)
        count = len(traj_indices) * len(variant_indices) * int(args.cams_sample_per_frame)
        total += count
        print(
            f"  {npz_path.name}: camera={camera_usd.name}, trajs={len(traj_indices)}/{len(seed_arr)}, "
            f"variants={variant_indices}, last_frame={pos.shape[1] - 1}, wires={pos.shape[2]}, renders={count}"
        )

    print(f"total RGB renders: {total}")
    if args.do_seg:
        print(f"total SEG renders: {total}")
    if args.do_depth:
        print(f"total DEPTH renders: {total}")


def init_isaac(headless: bool):
    global simulation_app, Usd, UsdGeom, UsdLux, UsdSkel, Gf, Sdf, omni, SkeletonRodDriver

    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    if str(DATACENTER_DIR) not in sys.path:
        sys.path.insert(0, str(DATACENTER_DIR))

    from isaacsim import SimulationApp

    simulation_app = SimulationApp({"headless": bool(headless)})

    from pxr import Gf as _Gf
    from pxr import Sdf as _Sdf
    from pxr import Usd as _Usd
    from pxr import UsdGeom as _UsdGeom
    from pxr import UsdLux as _UsdLux
    from pxr import UsdSkel as _UsdSkel
    import omni as _omni
    import omni.timeline  # noqa: F401
    import omni.usd  # noqa: F401
    from Dataset_generator_datacenter.rod_skel_driver import SkeletonRodDriver as _SkeletonRodDriver

    Usd = _Usd
    UsdGeom = _UsdGeom
    UsdLux = _UsdLux
    UsdSkel = _UsdSkel
    Gf = _Gf
    Sdf = _Sdf
    omni = _omni
    SkeletonRodDriver = _SkeletonRodDriver
    return simulation_app


def wait_updates(n: int) -> None:
    for _ in range(int(max(0, n))):
        simulation_app.update()


def safe_writer_detach(writer) -> None:
    for meth in ("detach", "clear", "reset"):
        if hasattr(writer, meth):
            try:
                getattr(writer, meth)()
                return
            except Exception:
                pass


def detach_basic_writer_best_effort(rep) -> None:
    try:
        writer = rep.WriterRegistry.get("BasicWriter")
    except Exception:
        return
    safe_writer_detach(writer)


def destroy_render_products(render_products: list[object]) -> None:
    for render_product in render_products:
        try:
            render_product.destroy()
        except Exception:
            pass
    wait_updates(1)


def try_set_setting(settings, path: str, value) -> None:
    try:
        settings.set(path, value)
    except Exception:
        pass


def disable_temporal_effects() -> None:
    import carb

    settings = carb.settings.get_settings()
    for p in [
        "/rtx/post/motionBlur/enabled",
        "/rtx/post/motionblur/enabled",
        "/rtx/post/taa/enabled",
        "/rtx/taa/enabled",
        "/app/renderer/aa/taa/enabled",
        "/rtx/denoiser/enabled",
        "/rtx/denoiser/enable",
        "/rtx/denoiser/temporal/enabled",
        "/rtx/denoiser/enableTemporal",
        "/rtx/post/denoiser/enabled",
        "/rtx/post/denoiser/temporal/enabled",
        "/rtx/post/dlss/enabled",
    ]:
        try_set_setting(settings, p, False)


def set_rgb_quality(render_mode: str) -> None:
    import carb

    settings = carb.settings.get_settings()
    path_tracing = render_mode == "PathTracing"
    for p, v in [
        ("/rtx/rendermode", render_mode),
        ("/rtx/pathtracing/enabled", path_tracing),
        ("/rtx/pathtracing/maxBounces", 6 if path_tracing else 1),
        ("/rtx/pathtracing/adaptiveSampling/enabled", False),
        ("/rtx/post/autoExposure/enabled", False),
        ("/rtx/post/autoExposure/enable", False),
        ("/rtx/post/histogram/enabled", False),
    ]:
        try_set_setting(settings, p, v)
    disable_temporal_effects()


def get_asset_default_prim_path(path: Path):
    stage = Usd.Stage.Open(str(path))
    if stage is None:
        raise RuntimeError(f"Cannot open USD: {path}")
    dp = stage.GetDefaultPrim()
    if not dp.IsValid():
        raise RuntimeError(f"No defaultPrim in {path}")
    return dp.GetPath()


def reference_usd(stage, usd_path: Path, prim_path: str, scale: float = 1.0) -> str:
    prim = UsdGeom.Xform.Define(stage, prim_path).GetPrim()
    xform = UsdGeom.Xformable(prim)
    xform.ClearXformOpOrder()
    if scale != 1.0:
        xform.AddScaleOp().Set(Gf.Vec3f(float(scale), float(scale), float(scale)))
    prim.GetReferences().AddReference(str(usd_path), get_asset_default_prim_path(usd_path))
    return prim_path


def list_cameras(stage, root_path: str) -> list[str]:
    root = stage.GetPrimAtPath(root_path)
    cams: list[str] = []
    for prim in Usd.PrimRange(root):
        if prim.IsA(UsdGeom.Camera):
            cams.append(prim.GetPath().pathString)
    return sorted(cams)


def list_sphere_lights(stage, root_path: str) -> list[str]:
    root = stage.GetPrimAtPath(root_path)
    lights: list[str] = []
    for prim in Usd.PrimRange(root):
        if prim.IsA(UsdLux.SphereLight):
            lights.append(prim.GetPath().pathString)
    return sorted(lights)


def set_random_lights(stage, light_paths: list[str], k: int, intensity_range: tuple[float, float], rng: random.Random) -> list[str]:
    for path in light_paths:
        prim = stage.GetPrimAtPath(path)
        if prim.IsValid():
            UsdLux.SphereLight(prim).CreateIntensityAttr().Set(0.0)

    if not light_paths:
        return []

    count = min(int(k), len(light_paths))
    chosen = rng.sample(light_paths, k=count)
    lo, hi = intensity_range
    for path in chosen:
        prim = stage.GetPrimAtPath(path)
        if prim.IsValid():
            UsdLux.SphereLight(prim).CreateIntensityAttr().Set(float(rng.uniform(lo, hi)))
    return chosen


def configure_dome_lights(stage, intensity: float, disabled_roots: list[str] | None = None) -> list[str]:
    disabled_roots = disabled_roots or []
    configured: list[str] = []
    active_count = 0
    disabled_count = 0
    for prim in stage.Traverse():
        if not prim.IsA(UsdLux.DomeLight):
            continue
        path = prim.GetPath().pathString
        disabled = _is_under_any(path, disabled_roots)
        imageable = UsdGeom.Imageable(prim)
        if imageable:
            imageable.GetVisibilityAttr().Set(UsdGeom.Tokens.invisible if disabled else UsdGeom.Tokens.inherited)
        UsdLux.DomeLight(prim).CreateIntensityAttr().Set(0.0 if disabled else float(intensity))
        configured.append(path)
        if disabled:
            disabled_count += 1
        else:
            active_count += 1
    print(
        f"[INFO] DomeLights intensity={float(intensity)} active={active_count} disabled={disabled_count}",
        flush=True,
    )
    return configured


def set_wire_dome_lights(
    stage,
    all_wire_roots: list[str],
    active_wire_roots: list[str],
    intensity: float,
) -> tuple[int, int]:
    active = set(active_wire_roots)
    active_count = 0
    disabled_count = 0
    for prim in stage.Traverse():
        if not prim.IsA(UsdLux.DomeLight):
            continue
        path = prim.GetPath().pathString
        matching_roots = [root for root in all_wire_roots if path == root or path.startswith(root + "/")]
        if not matching_roots:
            continue
        enabled = any(root in active for root in matching_roots)
        imageable = UsdGeom.Imageable(prim)
        if imageable:
            imageable.GetVisibilityAttr().Set(UsdGeom.Tokens.inherited if enabled else UsdGeom.Tokens.invisible)
        UsdLux.DomeLight(prim).CreateIntensityAttr().Set(float(intensity) if enabled else 0.0)
        if enabled:
            active_count += 1
        else:
            disabled_count += 1
    return active_count, disabled_count


def find_first_skeleton_under(stage, root_path: str) -> str:
    for prim in Usd.PrimRange(stage.GetPrimAtPath(root_path)):
        if prim.IsA(UsdSkel.Skeleton):
            return prim.GetPath().pathString
    raise RuntimeError(f"No Skeleton under {root_path}")


def get_skeleton_joint_count(stage, skel_path: str) -> int:
    prim = stage.GetPrimAtPath(skel_path)
    if not prim.IsValid():
        return 0
    return len(UsdSkel.Skeleton(prim).GetJointsAttr().Get() or [])


def find_skel_root_above(stage, skel_path: str):
    prim = stage.GetPrimAtPath(skel_path)
    while prim and prim.IsValid():
        if prim.GetTypeName() == "SkelRoot":
            return prim
        prim = prim.GetParent()
    return None


def promote_to_skel_root(stage, prim) -> None:
    if prim.GetTypeName() == "SkelRoot":
        return
    edit_layer = stage.GetEditTarget().GetLayer()
    prim_spec = edit_layer.GetPrimAtPath(prim.GetPath())
    if prim_spec is None:
        prim_spec = Sdf.CreatePrimInLayer(edit_layer, prim.GetPath())
    prim_spec.typeName = "SkelRoot"


def ensure_mesh_skeleton_binding(stage, wire_root_path: str, skel_path: str) -> None:
    skel_sdf_path = stage.GetPrimAtPath(skel_path).GetPath()
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


def bind_head_meshes_to_first_joint(stage, wire_root_path: str, skel_path: str) -> None:
    skel_sdf_path = stage.GetPrimAtPath(skel_path).GetPath()
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
        if "/head" not in prim.GetPath().pathString.lower():
            continue
        points = UsdGeom.Mesh(prim).GetPointsAttr().Get()
        if not points:
            continue
        num_verts = len(points)
        UsdSkel.BindingAPI.Apply(prim)
        bind_applied = UsdSkel.BindingAPI(prim)
        bind_applied.CreateJointIndicesPrimvar(False, elementSize=1).Set([0] * num_verts)
        bind_applied.CreateJointWeightsPrimvar(False, elementSize=1).Set([1.0] * num_verts)
        bind_applied.CreateSkeletonRel().SetTargets([skel_sdf_path])


def add_semantic_labels(stage, wire_roots: list[str]) -> None:
    for i, root_path in enumerate(wire_roots):
        root = stage.GetPrimAtPath(root_path)
        if not root.IsValid():
            continue
        label = f"{WIRE_SEMANTIC_CLASS}_{i}"
        api_name = f"Semantics_{WIRE_SEMANTIC_CLASS}_{i}"
        for prim in Usd.PrimRange(root):
            prim.CreateAttribute(f"semantic:{api_name}:params:semanticType", Sdf.ValueTypeNames.String).Set("class")
            prim.CreateAttribute(f"semantic:{api_name}:params:semanticData", Sdf.ValueTypeNames.String).Set(label)


def build_rod_dir_from_positions(pos_last: np.ndarray) -> np.ndarray:
    # pos_last: (W, 3, N), returns (W, 3, 3, N - 1)
    dp = pos_last[:, :, 1:] - pos_last[:, :, :-1]
    tangent = dp / (np.linalg.norm(dp, axis=1, keepdims=True) + 1e-12)
    up1 = np.array([0.0, 0.0, 1.0])[None, :, None]
    up2 = np.array([0.0, 1.0, 0.0])[None, :, None]
    normal = np.cross(up1, tangent, axis=1)
    normal_norm = np.linalg.norm(normal, axis=1, keepdims=True)
    mask = normal_norm < 1e-8
    if np.any(mask):
        normal2 = np.cross(up2, tangent, axis=1)
        normal2_norm = np.linalg.norm(normal2, axis=1, keepdims=True)
        normal = np.where(mask, normal2, normal)
        normal_norm = np.where(mask, normal2_norm, normal_norm)
    normal = normal / (normal_norm + 1e-12)
    binormal = np.cross(tangent, normal, axis=1)
    binormal = binormal / (np.linalg.norm(binormal, axis=1, keepdims=True) + 1e-12)
    rod_dir = np.zeros((pos_last.shape[0], 3, 3, pos_last.shape[2] - 1), dtype=np.float64)
    rod_dir[:, :, 0, :] = tangent
    rod_dir[:, :, 1, :] = normal
    rod_dir[:, :, 2, :] = binormal
    return rod_dir


def choose_wire_assets(cables: list[Path], num_wires: int, rng: random.Random, mode: str) -> list[Path]:
    if mode == "same":
        cable = rng.choice(cables)
        return [cable for _ in range(num_wires)]
    if mode != "mixed":
        raise ValueError(f"Unsupported wire_color_mode: {mode}")
    return [rng.choice(cables) for _ in range(num_wires)]


def usd_identifier(text: str) -> str:
    out = re.sub(r"[^A-Za-z0-9_]", "_", text)
    if not out:
        out = "x"
    if out[0].isdigit():
        out = "_" + out
    return out


def set_root_visibility(stage, root_paths: list[str], visible_roots: list[str]) -> None:
    visible = set(visible_roots)
    for root_path in root_paths:
        prim = stage.GetPrimAtPath(root_path)
        if not prim.IsValid():
            continue
        UsdGeom.Imageable(prim).GetVisibilityAttr().Set(
            UsdGeom.Tokens.inherited if root_path in visible else UsdGeom.Tokens.invisible
        )


def reference_wires(stage, wire_assets: list[Path]) -> tuple[list[str], list[str]]:
    wire_roots: list[str] = []
    skel_paths: list[str] = []
    for i, asset_path in enumerate(wire_assets):
        root_path = f"/World/Wire_{i}"
        wire_roots.append(root_path)
        prim = UsdGeom.Xform.Define(stage, root_path).GetPrim()
        xform = UsdGeom.Xformable(prim)
        xform.ClearXformOpOrder()
        xform.AddScaleOp().Set(Gf.Vec3f(POS_SCALE, POS_SCALE, POS_SCALE))
        prim.GetReferences().AddReference(str(asset_path), get_asset_default_prim_path(asset_path))

    wait_updates(8)

    for root_path in wire_roots:
        try:
            skel_path = find_first_skeleton_under(stage, root_path)
            if find_skel_root_above(stage, skel_path) is None:
                promote_to_skel_root(stage, stage.GetPrimAtPath(root_path))
            ensure_mesh_skeleton_binding(stage, root_path, skel_path)
            bind_head_meshes_to_first_joint(stage, root_path, skel_path)
            skel_paths.append(skel_path)
        except Exception as exc:
            print(f"[WARN] {root_path}: {exc}", flush=True)
            skel_paths.append("")

    add_semantic_labels(stage, wire_roots)
    wait_updates(4)
    return wire_roots, skel_paths


def reference_wire_slots(stage, cables: list[Path], num_wires: int) -> tuple[dict[Path, tuple[list[str], list[str]]], list[str]]:
    wire_slots: dict[Path, tuple[list[str], list[str]]] = {}
    all_wire_roots: list[str] = []
    slot_roots_by_asset: dict[Path, list[str]] = {}

    for asset_i, asset_path in enumerate(cables):
        roots: list[str] = []
        suffix = usd_identifier(asset_path.stem)
        for wire_i in range(num_wires):
            root_path = f"/World/WireSlot_{wire_i}_{asset_i}_{suffix}"
            roots.append(root_path)
            all_wire_roots.append(root_path)
            prim = UsdGeom.Xform.Define(stage, root_path).GetPrim()
            xform = UsdGeom.Xformable(prim)
            xform.ClearXformOpOrder()
            xform.AddScaleOp().Set(Gf.Vec3f(POS_SCALE, POS_SCALE, POS_SCALE))
            prim.GetReferences().AddReference(str(asset_path), get_asset_default_prim_path(asset_path))
        slot_roots_by_asset[asset_path] = roots

    wait_updates(8)

    for asset_path, roots in slot_roots_by_asset.items():
        skels: list[str] = []
        for root_path in roots:
            try:
                skel_path = find_first_skeleton_under(stage, root_path)
                if find_skel_root_above(stage, skel_path) is None:
                    promote_to_skel_root(stage, stage.GetPrimAtPath(root_path))
                ensure_mesh_skeleton_binding(stage, root_path, skel_path)
                bind_head_meshes_to_first_joint(stage, root_path, skel_path)
                skels.append(skel_path)
            except Exception as exc:
                print(f"[WARN] {root_path}: {exc}", flush=True)
                skels.append("")
        wire_slots[asset_path] = (roots, skels)

    add_semantic_labels(stage, all_wire_roots)
    set_root_visibility(stage, all_wire_roots, [])
    wait_updates(4)
    return wire_slots, all_wire_roots


def apply_final_pose(stage, skel_paths: list[str], pos_last: np.ndarray, director_last: np.ndarray | None) -> None:
    num_wires, _, num_nodes = pos_last.shape
    num_elems = num_nodes - 1
    rod_dir = director_last if director_last is not None else build_rod_dir_from_positions(pos_last)
    if rod_dir.shape != (num_wires, 3, 3, num_elems):
        raise RuntimeError(f"director shape mismatch: expected {(num_wires, 3, 3, num_elems)}, got {rod_dir.shape}")

    drivers = []
    for wire_i, skel_path in enumerate(skel_paths[:num_wires]):
        if not skel_path:
            continue
        prim = stage.GetPrimAtPath(skel_path)
        if not prim.IsValid():
            continue
        num_joints = get_skeleton_joint_count(stage, skel_path)
        if num_joints != num_elems:
            msg = f"Wire {wire_i}: skeleton joints={num_joints}, trajectory elems={num_elems}"
            if SKIP_MISMATCH_WIRES:
                print(f"[WARN] {msg}; skipping", flush=True)
                continue
            raise RuntimeError(msg)
        driver = SkeletonRodDriver(stage, skel_path, assume_chain=ASSUME_CHAIN_PARENT)
        driver.skel_prim = prim
        driver.skeleton_path = skel_path
        driver._setup_animation()
        drivers.append((wire_i, driver))

    if not drivers:
        raise RuntimeError("No compatible datacenter wires found for this pose")

    for wire_i, driver in drivers:
        driver.update_skeleton(pos_last[wire_i], rod_dir[wire_i], Usd.TimeCode(0))

    stage.SetTimeCodesPerSecond(30.0)
    stage.SetFramesPerSecond(30.0)
    timeline = omni.timeline.get_timeline_interface()
    timeline.set_start_time(0.0)
    timeline.set_end_time(0.0)
    timeline.set_current_time(0.0)


def _is_under_any(path: str, roots: list[str]) -> bool:
    return any(path == root or path.startswith(root + "/") for root in roots)


def ancestor_paths(path: str) -> set[str]:
    parts = [p for p in path.split("/") if p]
    out = {"/"}
    cur = ""
    for part in parts:
        cur += "/" + part
        out.add(cur)
    return out


def snapshot_visibility(stage) -> dict[str, object | None]:
    state: dict[str, object | None] = {}
    for prim in stage.Traverse():
        imageable = UsdGeom.Imageable(prim)
        if not imageable:
            continue
        attr = imageable.GetVisibilityAttr()
        if not attr or not attr.IsValid():
            continue
        state[prim.GetPath().pathString] = attr.Get() if attr.HasAuthoredValue() else None
    return state


def restore_visibility(stage, state: dict[str, object | None]) -> None:
    for prim in stage.Traverse():
        imageable = UsdGeom.Imageable(prim)
        if not imageable:
            continue
        path = prim.GetPath().pathString
        if path not in state:
            continue
        attr = imageable.GetVisibilityAttr()
        old = state[path]
        if old is None:
            attr.Clear()
        else:
            attr.Set(old)


def set_visibility_for_seg(stage, visible_roots: list[str], keep_paths: set[str]) -> None:
    for prim in stage.Traverse():
        imageable = UsdGeom.Imageable(prim)
        if not imageable:
            continue
        path = prim.GetPath().pathString
        visible = _is_under_any(path, visible_roots) or path in keep_paths
        imageable.GetVisibilityAttr().Set(UsdGeom.Tokens.inherited if visible else UsdGeom.Tokens.invisible)


def create_reusable_config_stage(args: argparse.Namespace, npz_path: Path, num_wires: int) -> ReusableStage:
    ctx = omni.usd.get_context()
    ctx.new_stage()
    wait_updates(5)
    stage = ctx.get_stage()
    if stage is None:
        raise RuntimeError("No stage after ctx.new_stage()")

    UsdGeom.Xform.Define(stage, "/World")

    if args.datahall_usd.is_file():
        reference_usd(stage, args.datahall_usd, "/World/Background_0", scale=1.0)
    else:
        print(f"[WARN] Missing datahall USD: {args.datahall_usd}", flush=True)
    if args.scene_usd.is_file():
        reference_usd(stage, args.scene_usd, "/World/Background_1", scale=POS_SCALE)
    else:
        print(f"[WARN] Missing scene USD: {args.scene_usd}", flush=True)

    camera_usd = camera_usd_for_config(args.asset_root, npz_path)
    light_usd = lights_usd(args.asset_root)
    camera_root = reference_usd(stage, camera_usd, "/World/DataCenterCameras", scale=CAMERA_LIGHT_SCALE)
    light_root = reference_usd(stage, light_usd, "/World/DataCenterLights", scale=CAMERA_LIGHT_SCALE)

    cables = cable_usds(args.asset_root)
    wire_slots, all_wire_roots = reference_wire_slots(stage, cables, int(num_wires))
    dome_lights = configure_dome_lights(stage, float(args.dome_intensity), disabled_roots=all_wire_roots)

    all_cameras = list_cameras(stage, camera_root)
    if len(all_cameras) < int(args.cams_sample_per_frame):
        raise RuntimeError(f"Need {args.cams_sample_per_frame} cameras from {camera_usd.name}, found {len(all_cameras)}")
    all_lights = list_sphere_lights(stage, light_root)

    print(
        f"[BASE_STAGE] {npz_path.stem} wires={num_wires} cable_usds={len(cables)} "
        f"wire_slots={len(all_wire_roots)} cameras={len(all_cameras)} lights={len(all_lights)} "
        f"dome_lights={len(dome_lights)}",
        flush=True,
    )
    return ReusableStage(
        stage=stage,
        cables=cables,
        all_cameras=all_cameras,
        all_lights=all_lights,
        dome_lights=dome_lights,
        wire_slots=wire_slots,
        all_wire_roots=all_wire_roots,
    )


def prepare_reused_stage_variant(
    args: argparse.Namespace,
    reusable: ReusableStage,
    npz_path: Path,
    pos_last: np.ndarray,
    director_last: np.ndarray | None,
    rng_seed: int,
    traj_seed: int,
    traj_index: int,
    variant_index: int,
) -> tuple[list[str], list[str]]:
    rng = random.Random(int(rng_seed))
    wire_assets = choose_wire_assets(reusable.cables, int(pos_last.shape[0]), rng, str(args.wire_color_mode))

    chosen_roots: list[str] = []
    chosen_skels: list[str] = []
    for wire_i, asset_path in enumerate(wire_assets):
        roots, skels = reusable.wire_slots[asset_path]
        chosen_roots.append(roots[wire_i])
        chosen_skels.append(skels[wire_i])

    set_root_visibility(reusable.stage, reusable.all_wire_roots, chosen_roots)
    active_wire_domes, disabled_wire_domes = set_wire_dome_lights(
        reusable.stage,
        reusable.all_wire_roots,
        chosen_roots,
        float(args.dome_intensity),
    )

    cameras_needed = int(args.cams_sample_per_frame)
    selected_cameras = rng.sample(reusable.all_cameras, k=cameras_needed)
    chosen_lights = set_random_lights(
        reusable.stage,
        reusable.all_lights,
        int(args.lights_on_per_frame),
        (float(args.light_intensity_min), float(args.light_intensity_max)),
        rng,
    )

    apply_final_pose(reusable.stage, chosen_skels, pos_last, director_last)
    wait_updates(args.warmup_updates)

    print(
        f"[STAGE_REUSE] {npz_path.stem} traj_index={traj_index} traj_seed={traj_seed} variant={variant_index} "
        f"rng_seed={rng_seed} wires={pos_last.shape[0]} cameras={len(selected_cameras)} "
        f"wire_color_mode={args.wire_color_mode} "
        f"wire_usds={','.join(p.stem for p in sorted(set(wire_assets)))} "
        f"lights={len(chosen_lights)} dome_lights={len(reusable.dome_lights)} "
        f"wire_domes_active={active_wire_domes} wire_domes_disabled={disabled_wire_domes}",
        flush=True,
    )
    return chosen_roots, selected_cameras


def create_stage(
    args: argparse.Namespace,
    npz_path: Path,
    pos_last: np.ndarray,
    director_last: np.ndarray,
    rng_seed: int,
    traj_seed: int,
    traj_index: int,
    variant_index: int,
    cameras_needed: int,
) -> tuple[object, list[str], list[str]]:
    ctx = omni.usd.get_context()
    ctx.new_stage()
    wait_updates(5)
    stage = ctx.get_stage()
    if stage is None:
        raise RuntimeError("No stage after ctx.new_stage()")

    UsdGeom.Xform.Define(stage, "/World")

    if args.datahall_usd.is_file():
        reference_usd(stage, args.datahall_usd, "/World/Background_0", scale=1.0)
    else:
        print(f"[WARN] Missing datahall USD: {args.datahall_usd}", flush=True)
    if args.scene_usd.is_file():
        reference_usd(stage, args.scene_usd, "/World/Background_1", scale=POS_SCALE)
    else:
        print(f"[WARN] Missing scene USD: {args.scene_usd}", flush=True)

    camera_usd = camera_usd_for_config(args.asset_root, npz_path)
    light_usd = lights_usd(args.asset_root)
    camera_root = reference_usd(stage, camera_usd, "/World/DataCenterCameras", scale=CAMERA_LIGHT_SCALE)
    light_root = reference_usd(stage, light_usd, "/World/DataCenterLights", scale=CAMERA_LIGHT_SCALE)

    cables = cable_usds(args.asset_root)
    rng = random.Random(int(rng_seed))
    wire_assets = choose_wire_assets(cables, int(pos_last.shape[0]), rng, str(args.wire_color_mode))
    wire_roots, skel_paths = reference_wires(stage, wire_assets)
    dome_lights = configure_dome_lights(stage, float(args.dome_intensity))

    all_cameras = list_cameras(stage, camera_root)
    if len(all_cameras) < cameras_needed:
        raise RuntimeError(f"Need {cameras_needed} cameras from {camera_usd.name}, found {len(all_cameras)}")
    selected_cameras = rng.sample(all_cameras, k=int(cameras_needed))

    all_lights = list_sphere_lights(stage, light_root)
    chosen_lights = set_random_lights(
        stage,
        all_lights,
        int(args.lights_on_per_frame),
        (float(args.light_intensity_min), float(args.light_intensity_max)),
        rng,
    )

    apply_final_pose(stage, skel_paths, pos_last, director_last)
    wait_updates(args.warmup_updates)

    print(
        f"[STAGE] {npz_path.stem} traj_index={traj_index} traj_seed={traj_seed} variant={variant_index} "
        f"rng_seed={rng_seed} wires={pos_last.shape[0]} "
        f"camera_usd={camera_usd.name} cameras={len(selected_cameras)} "
        f"wire_color_mode={args.wire_color_mode} "
        f"wire_usds={','.join(p.stem for p in sorted(set(wire_assets)))} "
        f"lights={len(chosen_lights)} dome_lights={len(dome_lights)}",
        flush=True,
    )
    return stage, wire_roots, selected_cameras


def render_current_stage(
    args: argparse.Namespace,
    stage,
    wire_roots: list[str],
    camera_paths: list[str],
    out_dir: Path,
) -> None:
    import omni.replicator.core as rep

    set_rgb_quality(str(args.render_mode))
    disable_temporal_effects()

    out_dir.mkdir(parents=True, exist_ok=True)
    rgb_dir = out_dir / "rgb"
    seg_dir = out_dir / "seg"
    depth_dir = out_dir / "depth"
    rgb_dir.mkdir(parents=True, exist_ok=True)
    if args.do_seg:
        seg_dir.mkdir(parents=True, exist_ok=True)
    if args.do_depth:
        depth_dir.mkdir(parents=True, exist_ok=True)

    render_products = [
        rep.create.render_product(cam, resolution=(int(args.width), int(args.height)))
        for cam in camera_paths
    ]
    if not render_products:
        raise RuntimeError("No render products created")

    detach_basic_writer_best_effort(rep)
    for _ in range(int(max(1, args.accum_steps))):
        rep.orchestrator.step(rt_subframes=int(max(1, args.accum_subframes)))
        simulation_app.update()

    writer = rep.WriterRegistry.get("BasicWriter")

    safe_writer_detach(writer)
    writer.initialize(output_dir=str(rgb_dir), rgb=True)
    writer.attach(render_products)
    rep.orchestrator.step(rt_subframes=1)
    simulation_app.update()
    wait_updates(args.settle_updates)
    safe_writer_detach(writer)

    if args.do_seg:
        keep = {"/World"}
        for cam in camera_paths:
            keep.update(ancestor_paths(cam))
        state = snapshot_visibility(stage)
        try:
            set_visibility_for_seg(stage, wire_roots, keep)
            wait_updates(5)
            safe_writer_detach(writer)
            writer.initialize(output_dir=str(seg_dir), instance_id_segmentation=True)
            writer.attach(render_products)
            rep.orchestrator.step(rt_subframes=1)
            simulation_app.update()
            wait_updates(args.settle_updates)
            safe_writer_detach(writer)
        finally:
            restore_visibility(stage, state)
            wait_updates(5)

    if args.do_depth:
        safe_writer_detach(writer)
        writer.initialize(output_dir=str(depth_dir), distance_to_camera=True)
        writer.attach(render_products)
        rep.orchestrator.step(rt_subframes=1)
        simulation_app.update()
        wait_updates(args.settle_updates)
        safe_writer_detach(writer)

    destroy_render_products(render_products)


def close_stage() -> None:
    try:
        omni.usd.get_context().close_stage()
    except Exception:
        pass
    wait_updates(10)
    gc.collect()


def render_config_reuse(
    args: argparse.Namespace,
    npz_path: Path,
    pos: np.ndarray,
    director: np.ndarray | None,
    seed_arr: np.ndarray,
) -> None:
    traj_indices = selected_traj_indices(len(seed_arr), args.traj_index, args.max_trajs)
    variant_indices = selected_variant_indices(args.seed_start, args.seed_index, args.max_seeds)
    config_out = args.output_root / npz_path.stem
    if args.clean and config_out.exists():
        shutil.rmtree(config_out)

    last_frame = int(pos.shape[1] - 1)
    try:
        reusable = create_reusable_config_stage(args, npz_path, int(pos.shape[2]))
        for traj_local_i, traj_idx in enumerate(traj_indices, start=1):
            traj_seed = int(seed_arr[traj_idx])
            pos_last = np.asarray(pos[traj_idx, last_frame], dtype=np.float64)
            director_last = None if director is None else np.asarray(director[traj_idx, last_frame], dtype=np.float64)

            for variant_local_i, variant_idx in enumerate(variant_indices, start=1):
                rng_seed = variant_rng_seed(npz_path.stem, traj_seed, int(traj_idx), int(variant_idx), str(args.wire_color_mode))
                out_dir = (
                    config_out
                    / f"traj_{traj_seed}"
                    / f"variant_{variant_idx:03d}_{args.wire_color_mode}"
                    / f"frame_{last_frame:06d}"
                )

                print(
                    f"[RUN_REUSE] {npz_path.stem} traj_index={traj_idx} ({traj_local_i}/{len(traj_indices)}) "
                    f"traj_seed={traj_seed} variant={variant_idx} ({variant_local_i}/{len(variant_indices)}) "
                    f"rng_seed={rng_seed} last_frame={last_frame} out={out_dir}",
                    flush=True,
                )
                wire_roots, camera_paths = prepare_reused_stage_variant(
                    args,
                    reusable,
                    npz_path,
                    pos_last,
                    director_last,
                    rng_seed,
                    traj_seed,
                    int(traj_idx),
                    int(variant_idx),
                )
                render_current_stage(args, reusable.stage, wire_roots, camera_paths, out_dir)
    finally:
        close_stage()


def render_config(args: argparse.Namespace, npz_path: Path) -> None:
    data = np.load(npz_path)
    pos = data["pos"]
    director = data["director"] if "director" in data.files else None
    seed_arr = data["seed_arr"]

    if pos.ndim != 5:
        raise RuntimeError(f"{npz_path.name}: expected pos (traj,T,W,3,N), got {pos.shape}")
    if director is not None and director.ndim != 6:
        raise RuntimeError(f"{npz_path.name}: expected director (traj,T,W,3,3,E), got {director.shape}")

    if args.reuse_stage:
        render_config_reuse(args, npz_path, pos, director, seed_arr)
        return

    traj_indices = selected_traj_indices(len(seed_arr), args.traj_index, args.max_trajs)
    variant_indices = selected_variant_indices(args.seed_start, args.seed_index, args.max_seeds)
    config_out = args.output_root / npz_path.stem
    if args.clean and config_out.exists():
        shutil.rmtree(config_out)

    last_frame = int(pos.shape[1] - 1)
    for traj_local_i, traj_idx in enumerate(traj_indices, start=1):
        traj_seed = int(seed_arr[traj_idx])
        pos_last = np.asarray(pos[traj_idx, last_frame], dtype=np.float64)
        director_last = None if director is None else np.asarray(director[traj_idx, last_frame], dtype=np.float64)

        for variant_local_i, variant_idx in enumerate(variant_indices, start=1):
            rng_seed = variant_rng_seed(npz_path.stem, traj_seed, int(traj_idx), int(variant_idx), str(args.wire_color_mode))
            out_dir = (
                config_out
                / f"traj_{traj_seed}"
                / f"variant_{variant_idx:03d}_{args.wire_color_mode}"
                / f"frame_{last_frame:06d}"
            )

            print(
                f"[RUN] {npz_path.stem} traj_index={traj_idx} ({traj_local_i}/{len(traj_indices)}) "
                f"traj_seed={traj_seed} variant={variant_idx} ({variant_local_i}/{len(variant_indices)}) "
                f"rng_seed={rng_seed} last_frame={last_frame} out={out_dir}",
                flush=True,
            )
            try:
                stage, wire_roots, camera_paths = create_stage(
                    args,
                    npz_path,
                    pos_last,
                    director_last,
                    rng_seed,
                    traj_seed,
                    int(traj_idx),
                    int(variant_idx),
                    int(args.cams_sample_per_frame),
                )
                render_current_stage(args, stage, wire_roots, camera_paths, out_dir)
            finally:
                close_stage()


def main() -> int:
    args = parse_args()
    args.asset_root = args.asset_root.resolve()
    args.output_root = args.output_root.resolve()
    args.datahall_usd = args.datahall_usd.resolve()
    args.scene_usd = args.scene_usd.resolve()

    if args.cams_sample_per_frame < 1:
        raise ValueError("cams_sample_per_frame must be >= 1")

    filters = split_config_filters(args.config)
    configs = discover_configs(args.asset_root, filters, args.max_configs)

    if args.dry_run:
        dry_run(args)
        return 0

    app = init_isaac(args.headless)
    try:
        disable_temporal_effects()
        for npz_path in configs:
            render_config(args, npz_path)
    finally:
        app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
