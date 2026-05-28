from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
ASSET_ROOT = REPO_ROOT / "asset_wireseg32k"
USD_ROOT = ASSET_ROOT / "usd"
DATA_ROOT = REPO_ROOT / "Dataset_generator" / "data"
OUTPUT_ROOT = REPO_ROOT / "output"


@dataclass(frozen=True)
class RenderConfig:
    """Per-run/per-frame render settings."""

    do_seg: bool = False
    do_depth: bool = False
    seed: int = 42
    accum_steps: int = 80
    accum_subframes: int = 16

    # Optional: retained for CLI parity / bookkeeping.
    frame_start: int | None = None
    frame_end: int | None = None
    frame_step: int = 1


@dataclass(frozen=True)
class GeneratorConfig:
    # Simulation
    headless: bool = True

    # Scene + data
    scene_variant_specs: tuple[tuple[str, str, int], ...] = (
        (str(USD_ROOT / "rod_drop_multi_2_plane.usdc"), str(ASSET_ROOT / "wires_traj_data" / "drop_n2_100.npz"), 2),
        (str(USD_ROOT / "rod_drop_multi_4_plane.usdc"), str(ASSET_ROOT / "wires_traj_data" / "drop_n4_100.npz"), 4),
        (str(USD_ROOT / "rod_drop_multi_8_plane.usdc"), str(ASSET_ROOT / "wires_traj_data" / "drop_n8_100.npz"), 8),
        (str(USD_ROOT / "rod_hang_flying.usdc"), str(ASSET_ROOT / "wires_traj_data" / "hang_n2_100.npz"), 2),
        (str(USD_ROOT / "rod_hang_flying.usdc"), str(ASSET_ROOT / "wires_traj_data" / "hang_n4_100.npz"), 4),
        (str(USD_ROOT / "rod_hang_flying.usdc"), str(ASSET_ROOT / "wires_traj_data" / "hang_n8_100.npz"), 8),
    )
    scene_usd: str = scene_variant_specs[0][0]
    npz_path: str = scene_variant_specs[0][1]

    table_prim_path: str = "/root/ground/Plane"   
    table_texture_dir: str = str(ASSET_ROOT / "ground")
    table_texture_exts: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".exr")
    randomize_table_material_per_frame: bool = True


    # Wires assets
    wire_asset_dir: str = str(USD_ROOT / "wire_usdc" / "wire_usdc")
    recursive_assets: bool = False
    radius_tag: str = "r0.005"
    num_wires: int = 8

    wire_parent_path: str = "/World"
    assume_chain_parent: bool = True
    skip_mismatch_wires: bool = True
    wire_world_offset_xyz: Tuple[float, float, float] = (0, 0.0, 0.0)

    # Lighting: if both hdr_path and hdr_dir are given, hdr_dir is used for randomization,if enabled
    # if hdr_dir is None, hdr_path is always used
    hdr_path: str = str(ASSET_ROOT / "background" / "boiler_room_4k.hdr")
    hdr_dir: str = str(ASSET_ROOT / "background")
    randomize_hdr_per_frame: bool = True
    
    
    dome_intensity_range: Tuple[float, float] = (100.0, 500.0)
    dome_exposure: float = 0.0

    # Randomize (tokens/prefixes)
    cam_name_token: str = "Cam_"
    light_name_token: str = "rig_light_"
    occluder_name_prefixes: Tuple[str, ...] = ("occ_",)

    cams_sample_per_frame: int = 1
    lights_on_per_frame: int = 3
    light_intensity_range: Tuple[float, float] = (200.0, 2000.0)

    # Output
    capture_out_dir: str = str(OUTPUT_ROOT / "capture_reload_each_test_13")
    capture_width: int = 1024
    capture_height: int = 1024

    camera_jitter_enabled: bool = True
    camera_jitter_pos_std_m: float = 0.0003        # 0.3 mm
    camera_jitter_rot_std_deg: float = 0.03        # 0.03 degree
    camera_jitter_focal_std_mm: float = 0.05       # 0.05 mm
    camera_jitter_aperture_std_mm: float = 0.02    # 0.02 mm
    camera_jitter_fstop_std: float = 0.2          # tiny DOF change

    # Warmups
    warmup_after_open: int = 180
    warmup_after_reference: int = 60
    warmup_after_domelight: int = 30
    warmup_after_pose: int = 20
    warmup_after_lights: int = 20
    warmup_after_cam_sync: int = 20
    warmup_before_final_rgb_write: int = 5
    warmup_before_seg_write: int = 5
    warmup_after_hide: int = 20
    warmup_after_restore: int = 5
    warmup_after_close: int = 30
    warmup_after_build_proxies: int = 10

    # Animation FPS (only for consistency/seeds; kept for parity)
    fps: float = 30.0

    # Quality / Stability
    disable_temporal_effects: bool = True

    # Proxy cameras root
    proxy_root: str = "/World/_CaptureProxies"


def default_config() -> GeneratorConfig:
    """Defaults match archive/dataset_generator_oneframe.py."""
    return GeneratorConfig()
