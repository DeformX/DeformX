#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import gc
import hashlib
import os
import random
import sys
from pathlib import Path
from typing import Any
import time
from glob import glob
from typing import List, Sequence
import numpy as np

from .config import GeneratorConfig, RenderConfig


class DatasetGenerator:
    def __init__(self, sim_app, cfg: GeneratorConfig):
        self._sim_app = sim_app
        self.cfg = cfg
        self.global_index: int = 0  # global sample id across ALL cameras & frames
        self._npz_cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, int]] = {}
        self._assets_by_rtag: dict[str, list[str]] | None = None
        self._hdr_candidates: list[str] | None = None
        self._table_textures: list[str] | None = None
        self._asset_paths: list[str] | None = None
        self._variant_ranges: list[tuple[int, int, tuple[str, str, int]]] | None = None

    # -------------------------
    # Public API
    # -------------------------
    def _scene_variants(self) -> tuple[tuple[str, str, int], ...]:
        specs = tuple(getattr(self.cfg, "scene_variant_specs", ()) or ())
        if specs:
            return tuple((str(scene), str(npz_path), int(num_wires)) for scene, npz_path, num_wires in specs)
        return ((self.cfg.scene_usd, self.cfg.npz_path, int(self.cfg.num_wires)),)

    def _stream_seed(self, seed: int, frame: int, stream: str) -> int:
        key = f"{int(seed)}:{int(frame)}:{stream}".encode("utf-8")
        return int.from_bytes(hashlib.sha256(key).digest()[:8], "big", signed=False)

    def _stream_rng(self, seed: int, frame: int, stream: str) -> random.Random:
        return random.Random(self._stream_seed(seed, frame, stream))

    def _load_npz(self, npz_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        key = os.path.abspath(npz_path)
        cached = self._npz_cache.get(key)
        if cached is not None:
            return cached

        data = np.load(key)
        pos_arr = data["pos_arr"]
        dir_arr = data["dir_arr"]
        radius_arr = data["radius_arr"]
        end_frame = int(pos_arr.shape[0] - 1)
        self._npz_cache[key] = (pos_arr, dir_arr, radius_arr, end_frame)
        return self._npz_cache[key]

    def load_npz_once(self) -> tuple[np.ndarray, np.ndarray, int]:
        _, npz_path, _ = self._scene_variants()[0]
        pos_arr, dir_arr, _, end_frame = self._load_npz(npz_path)
        return pos_arr, dir_arr, end_frame

    def _variant_ranges_once(self) -> list[tuple[int, int, tuple[str, str, int]]]:
        if self._variant_ranges is not None:
            return self._variant_ranges

        ranges: list[tuple[int, int, tuple[str, str, int]]] = []
        start = 0
        for spec in self._scene_variants():
            _, npz_path, _ = spec
            _, _, _, end_frame = self._load_npz(npz_path)
            count = int(end_frame) + 1
            ranges.append((start, start + count - 1, spec))
            start += count
        self._variant_ranges = ranges
        return self._variant_ranges

    def global_end_frame(self) -> int:
        ranges = self._variant_ranges_once()
        if not ranges:
            raise RuntimeError("No scene/data variants configured.")
        return int(ranges[-1][1])

    def _variant_for_global_frame(self, frame: int) -> tuple[str, str, int, int]:
        for start, end, spec in self._variant_ranges_once():
            if start <= int(frame) <= end:
                scene_usd, npz_path, num_wires = spec
                return scene_usd, npz_path, int(num_wires), int(frame) - start
        raise ValueError(f"Global frame {frame} out of range [0, {self.global_end_frame()}]")

    def list_assets_once(self) -> list[str]:
        if self._asset_paths is not None:
            return self._asset_paths
        self._asset_paths = self._build_assets_by_rtag_once().get("__all__", [])
        return self._asset_paths

    def _build_assets_by_rtag_once(self) -> dict[str, list[str]]:
        if self._assets_by_rtag is not None:
            return self._assets_by_rtag

        root = Path(self.cfg.wire_asset_dir)
        if not root.is_dir():
            raise RuntimeError(f"wire_asset_dir not found: {self.cfg.wire_asset_dir}")

        patterns = ["*.usdc", "*.usd", "*.usda", "*.usd*"]
        paths: list[str] = []
        if self.cfg.recursive_assets:
            for pat in patterns:
                paths += [str(p) for p in root.rglob(pat)]
        else:
            for pat in patterns:
                paths += [str(p) for p in root.glob(pat)]

        paths = sorted({p for p in paths if os.path.isfile(p)})

        by: dict[str, list[str]] = {}
        for p in paths:
            _name = os.path.basename(p)
            by.setdefault("__all__", []).append(p)

        self._assets_by_rtag = by
        return by

    def _radius_to_rtag(self, r: float) -> str:
        return f"r{float(r):.3f}"

    def _select_wire_assets_from_radius(
        self, radius_arr: np.ndarray, frame: int, num_wires: int, rng: random.Random
    ) -> list[str]:
        radius_arr = np.array(radius_arr, dtype=np.float32)
        if radius_arr.ndim != 2:
            raise RuntimeError(f"radius_arr must be 2D (T, W), got shape={radius_arr.shape}")

        T, W = radius_arr.shape
        if frame < 0 or frame >= T:
            raise ValueError(f"frame {frame} out of range [0, {T-1}]")

        n = int(num_wires)
        if W < n:
            raise RuntimeError(f"radius_arr has W={W} wires, but requested num_wires={n}")

        # Collect all asset paths once and match by radius tag in filename
        all_assets = self._build_assets_by_rtag_once().get("__all__", [])
        if not all_assets:
            raise RuntimeError(f"No USD assets found under wire_asset_dir={self.cfg.wire_asset_dir}")

        chosen_paths: list[str] = []
        radii = radius_arr[frame, :n].tolist()

        for i, r in enumerate(radii):
            rtag = self._radius_to_rtag(float(r))  # e.g. "r0.003"
            matches = [p for p in all_assets if rtag in os.path.basename(p)]

            if not matches:
                raise RuntimeError(
                    f"No asset matched for wire[{i}] radius={float(r):.6f} (tag='{rtag}'). "
                    f"Please ensure filenames contain '{rtag}'."
                )
            chosen_paths.append(rng.choice(matches))

        return chosen_paths

    def render_frame(self, frame: int, render_cfg: RenderConfig) -> None:
        scene_usd, npz_path, num_wires, local_frame = self._variant_for_global_frame(frame)
        pos_arr, dir_arr, radius_arr, _ = self._load_npz(npz_path)

        # Deterministic per-frame RNG (keeps content deterministic per frame/seed)
        np.random.seed(self._stream_seed(render_cfg.seed, frame, "numpy") % (2**32))
        asset_rng = self._stream_rng(render_cfg.seed, frame, "wire_asset")
        dome_rng = self._stream_rng(render_cfg.seed, frame, "domelight")
        light_rng = self._stream_rng(render_cfg.seed, frame, "lights")
        camera_rng = self._stream_rng(render_cfg.seed, frame, "camera_pick")
        asset_paths = self._select_wire_assets_from_radius(radius_arr, local_frame, num_wires, asset_rng)

        print(
            f"[SCENE] global_frame={frame} local_frame={local_frame} wires={num_wires} "
            f"scene={os.path.basename(scene_usd)} npz={os.path.basename(npz_path)}"
        )
        stage = self._open_scene(scene_usd)
        try:
            skel_paths = self._reference_wires(stage, asset_paths)
            self._ensure_single_domelight(stage, frame, render_cfg.seed, dome_rng)

            if self.cfg.randomize_table_material_per_frame:
                self._randomize_table_texture_in_place(stage, frame=frame, seed=int(render_cfg.seed))
            self._wait_updates(self.cfg.warmup_after_domelight)

            self._apply_pose_for_frame(stage, asset_paths, skel_paths, pos_arr, dir_arr, local_frame, num_wires)
            self._wait_updates(self.cfg.warmup_after_pose)

            real_cams_all = self._list_user_cameras(stage)
            if len(real_cams_all) < self.cfg.cams_sample_per_frame:
                raise RuntimeError(
                    f"Need at least {self.cfg.cams_sample_per_frame} user cameras, found {len(real_cams_all)}"
                )

            lights_all = self._list_lights(stage)
            occluder_roots = self._list_occluder_roots(stage)

            # Build K proxy cameras and render-products
            proxy_cam_paths = self._build_proxy_cameras(stage, self.cfg.cams_sample_per_frame)
            rep = self._rep()
            proxy_rps = [
                rep.create.render_product(c, resolution=(self.cfg.capture_width, self.cfg.capture_height))
                for c in proxy_cam_paths
            ]
            K = len(proxy_rps)
            if K == 0:
                raise RuntimeError("No proxy render products created.")

            # Allocate K global ids for this frame (one per camera output)
            ids = list(range(self.global_index, self.global_index + K))
            self.global_index += K

            if lights_all:
                self._set_only_k_lights_on(stage, lights_all, self.cfg.lights_on_per_frame, light_rng)
            self._wait_updates(self.cfg.warmup_after_lights)

            chosen_real = camera_rng.sample(real_cams_all, k=len(proxy_cam_paths))
            self._sync_real_cams_to_proxies(stage, chosen_real, proxy_cam_paths)
            self._wait_updates(self.cfg.warmup_after_cam_sync)

            if self.cfg.camera_jitter_enabled:
                self._apply_camera_pose_jitter(stage, proxy_cam_paths, frame, int(render_cfg.seed))
                self._apply_camera_intrinsics_jitter(stage, proxy_cam_paths, frame, int(render_cfg.seed))
                self._sim_app.update()

            # root
            out_root = Path(self.cfg.capture_out_dir).resolve()
            # per-frame directory
            frame_dir = out_root / f"frame_{frame:06d}"
            # Flat output dirs (no frame subfolder)
            rgb_dir = os.path.abspath(os.path.join(frame_dir, "rgb"))
            seg_dir = os.path.abspath(os.path.join(frame_dir, "seg"))
            depth_dir = os.path.abspath(os.path.join(frame_dir, "depth"))
            Path(rgb_dir).mkdir(parents=True, exist_ok=True)
            if render_cfg.do_seg:
                Path(seg_dir).mkdir(parents=True, exist_ok=True)
            if render_cfg.do_depth:
                Path(depth_dir).mkdir(parents=True, exist_ok=True)

            print(
                f"[RENDER] ids={ids[0]:06d}..{ids[-1]:06d} frame={frame} "
                f"accum={render_cfg.accum_steps}x{render_cfg.accum_subframes} "
                f"seg={render_cfg.do_seg} depth={render_cfg.do_depth}"
            )

            # RGB
            self._set_rgb_quality()
            self._wait_updates(self.cfg.warmup_before_final_rgb_write)

            self._accumulate_without_writing(render_cfg.accum_steps, render_cfg.accum_subframes)

            t_rgb = self._write_one_rgb(proxy_rps, rgb_dir)  # must return timestamp

            # SEG
            t_seg = None
            if render_cfg.do_seg:
                self._set_seg_quality()
                self._wait_updates(self.cfg.warmup_before_seg_write)

                prev_vis: dict[str, str] = {}
                if occluder_roots:
                    prev_vis = self._set_visibility_recursive(occluder_roots, visible=False)
                    self._wait_updates(self.cfg.warmup_after_hide)

                t_seg = self._write_one_seg(proxy_rps, seg_dir)  # must return timestamp

                if occluder_roots:
                    self._restore_visibility(stage, prev_vis)
                    self._wait_updates(self.cfg.warmup_after_restore)

            # DEPTH
            t_depth = None
            if render_cfg.do_depth:
                t_depth = self._write_one_depth(proxy_rps, depth_dir)  # must return timestamp

            # Rename outputs into global ids (no camera info in names)
            # self._rename_outputs_after_write_global(
            #     ids=ids,
            #     rgb_dir=rgb_dir,
            #     seg_dir=(seg_dir if render_cfg.do_seg else None),
            #     depth_dir=(depth_dir if render_cfg.do_depth else None),
            #     wrote_seg=bool(render_cfg.do_seg),
            #     wrote_depth=bool(render_cfg.do_depth),
            #     t_rgb=t_rgb,
            #     t_seg=t_seg,
            #     t_depth=t_depth,
            # )

        finally:
            self._close_scene()

    def run(self, frames: list[int], render_cfg: RenderConfig) -> None:
        if self.cfg.disable_temporal_effects:
            self._disable_temporal()

        Path(self.cfg.capture_out_dir).mkdir(parents=True, exist_ok=True)

        end_frame_global = self.global_end_frame()
        self.list_assets_once()

        for f in frames:
            if int(f) < 0 or int(f) > end_frame_global:
                raise ValueError(f"Frame {f} out of range [0, {end_frame_global}]")

        print(
            f"[INFO] Will render {len(frames)} frame(s). "
            f"Global frames: 0..{end_frame_global} across {len(self._scene_variants())} scene/data variants"
        )

        for f in frames:
            self.render_frame(int(f), render_cfg)

    # -------------------------
    # Imports (after SimulationApp)
    # -------------------------
    def _rep(self):
        import omni.replicator.core as rep

        return rep

    def _usd(self):
        from pxr import Gf, Usd, UsdGeom, UsdLux, UsdSkel

        return Usd, UsdGeom, UsdLux, UsdSkel, Gf

    def _omni_usd(self):
        import omni.usd

        return omni.usd

    def _carb(self):
        import carb

        return carb

    def _import_skeleton_driver(self):
        try:
            from rod_skel_driver import SkeletonRodDriver  # type: ignore

            return SkeletonRodDriver
        except Exception:
            archive_dir = Path(__file__).resolve().parents[1] / "archive"
            if archive_dir.is_dir() and str(archive_dir) not in sys.path:
                sys.path.insert(0, str(archive_dir))
            from rod_skel_driver import SkeletonRodDriver  # type: ignore

            return SkeletonRodDriver

    # -------------------------
    # Core scene lifecycle
    # -------------------------
    def _wait_updates(self, n: int) -> None:
        for _ in range(int(max(0, n))):
            self._sim_app.update()

    def _open_scene(self, scene_usd: str | None = None):
        omni_usd = self._omni_usd()
        ctx = omni_usd.get_context()
        scene_path = os.path.abspath(scene_usd or self.cfg.scene_usd)
        ctx.open_stage(scene_path)
        self._wait_updates(self.cfg.warmup_after_open)
        stage = ctx.get_stage()
        if stage is None:
            raise RuntimeError("Stage is not available after opening scene.")
        return stage

    def _close_scene(self) -> None:
        omni_usd = self._omni_usd()
        ctx = omni_usd.get_context()
        try:
            ctx.close_stage()
        except Exception:
            pass
        self._wait_updates(self.cfg.warmup_after_close)
        gc.collect()

    # -------------------------
    # Settings / quality
    # -------------------------
    def _try_set_setting(self, settings, path: str, value) -> None:
        try:
            settings.set(path, value)
        except Exception:
            pass

    def _disable_temporal(self) -> None:
        s = self._carb().settings.get_settings()
        for p in [
            "/rtx/post/taa/enabled",
            "/rtx/taa/enabled",
            "/app/renderer/aa/taa/enabled",
            "/rtx/denoiser/temporal/enabled",
            "/rtx/denoiser/enableTemporal",
            "/rtx/post/denoiser/temporal/enabled",
        ]:
            self._try_set_setting(s, p, False)

    def _set_rgb_quality(self) -> None:
        s = self._carb().settings.get_settings()
        for p, v in [
            ("/rtx/rendermode", "PathTracing"),
            ("/rtx/pathtracing/enabled", True),
            ("/rtx/pathtracing/maxBounces", 6),
            ("/rtx/pathtracing/adaptiveSampling/enabled", False),
        ]:
            self._try_set_setting(s, p, v)

        for p, v in [
            ("/rtx/post/autoExposure/enabled", False),
            ("/rtx/post/autoExposure/enable", False),
            ("/rtx/post/histogram/enabled", False),
            ("/rtx/post/motionBlur/enabled", False),
            ("/rtx/post/motionblur/enabled", False),
            ("/rtx/post/dlss/enabled", False),
        ]:
            self._try_set_setting(s, p, v)

        for p in [
            "/rtx/post/taa/enabled",
            "/rtx/taa/enabled",
            "/rtx/denoiser/temporal/enabled",
            "/rtx/post/denoiser/temporal/enabled",
        ]:
            self._try_set_setting(s, p, False)

        for p, v in [
            ("/rtx/denoiser/enabled", True),
            ("/rtx/post/denoiser/enabled", True),
        ]:
            self._try_set_setting(s, p, v)

    def _set_seg_quality(self) -> None:
        s = self._carb().settings.get_settings()
        for p, v in [
            ("/rtx/rendermode", "RayTracedLighting"),
            ("/rtx/pathtracing/enabled", False),
        ]:
            self._try_set_setting(s, p, v)

    # -------------------------
    # Assets + wires
    # -------------------------
    def _list_wire_assets_rtag_exact(self, n: int, rtag: str) -> list[str]:
        root = Path(self.cfg.wire_asset_dir)
        if not root.is_dir():
            raise RuntimeError(f"wire_asset_dir not found: {self.cfg.wire_asset_dir}")

        patterns = ["*.usdc", "*.usd", "*.usda", "*.usd*"]
        paths: list[str] = []
        if self.cfg.recursive_assets:
            for pat in patterns:
                paths += [str(p) for p in root.rglob(pat)]
        else:
            for pat in patterns:
                paths += [str(p) for p in root.glob(pat)]

        paths = sorted({p for p in paths if os.path.isfile(p)})
        keep = [p for p in paths if rtag in os.path.basename(p)]
        if len(keep) < n:
            raise RuntimeError(f"Need {n} assets containing '{rtag}', but found {len(keep)}")
        return keep[:n]

    def _get_asset_default_prim_path(self, asset_usd_path: str):
        Usd, _, _, _, _ = self._usd()
        asset_stage = Usd.Stage.Open(asset_usd_path)
        if asset_stage is None:
            raise RuntimeError(f"Failed to open asset stage: {asset_usd_path}")
        dp = asset_stage.GetDefaultPrim()
        if not dp.IsValid():
            raise RuntimeError(f"Asset defaultPrim missing: {asset_usd_path}")
        return dp.GetPath()

    def _find_first_skeleton_under(self, stage, root_path: str) -> str:
        Usd, _, _, UsdSkel, _ = self._usd()
        root_prim = stage.GetPrimAtPath(root_path)
        if not root_prim.IsValid():
            raise RuntimeError(f"Root prim not valid: {root_path}")
        for p in Usd.PrimRange(root_prim):
            if p.IsA(UsdSkel.Skeleton):
                return p.GetPath().pathString
        raise RuntimeError(f"No UsdSkel.Skeleton found under {root_path}")

    def _get_skeleton_joint_count(self, stage, skel_path: str) -> int:
        _, _, _, UsdSkel, _ = self._usd()
        prim = stage.GetPrimAtPath(skel_path)
        if not prim.IsValid():
            return 0
        skel = UsdSkel.Skeleton(prim)
        joints = skel.GetJointsAttr().Get() or []
        return len(joints)

    def _reference_wires(self, stage, asset_paths: list[str]) -> list[str]:
        _, UsdGeom, _, _, _ = self._usd()

        skel_paths: list[str] = []
        for i, asset_usd in enumerate(asset_paths):
            root_path = f"{self.cfg.wire_parent_path}/Wire_{i}"
            root_xf = UsdGeom.Xform.Define(stage, root_path)
            root_prim = root_xf.GetPrim()
            UsdGeom.Xformable(root_prim).ClearXformOpOrder()

            asset_default_path = self._get_asset_default_prim_path(asset_usd)
            root_prim.GetReferences().AddReference(asset_usd, asset_default_path)

        self._wait_updates(self.cfg.warmup_after_reference)

        for i in range(len(asset_paths)):
            root_path = f"{self.cfg.wire_parent_path}/Wire_{i}"
            try:
                skel_paths.append(self._find_first_skeleton_under(stage, root_path))
            except Exception:
                skel_paths.append("")
        return skel_paths

    # -------------------------
    # Lighting
    # -------------------------
    def _list_hdr_candidates_once(self) -> list[str]:
        if self._hdr_candidates is not None:
            return self._hdr_candidates

        hdr_dir = getattr(self.cfg, "hdr_dir", None)
        if not hdr_dir:
            self._hdr_candidates = []
            return self._hdr_candidates

        root = Path(hdr_dir)
        if not root.is_dir():
            self._hdr_candidates = []
            return self._hdr_candidates

        exts = (".hdr", ".exr")
        paths: list[str] = []
        for p in root.rglob("*"):
            if p.is_file() and p.suffix.lower() in exts:
                paths.append(str(p))

        self._hdr_candidates = sorted(set(paths))
        return self._hdr_candidates

    def _pick_hdr_for_frame(self, frame: int, seed: int) -> str | None:
        hdrs = self._list_hdr_candidates_once()
        if not hdrs:
            return None

        rng = self._stream_rng(seed, frame, "hdr_choice")
        return rng.choice(hdrs)

    def _ensure_single_domelight(self, stage, frame: int, seed: int, rng: random.Random) -> None:
        _, UsdGeom, UsdLux, _, _ = self._usd()

        dome = UsdLux.DomeLight.Define(stage, "/World/DomeLight")

        chosen_hdr = None
        if getattr(self.cfg, "randomize_hdr_per_frame", False):
            chosen_hdr = self._pick_hdr_for_frame(frame, seed)

        if chosen_hdr and os.path.isfile(chosen_hdr):
            dome.CreateTextureFileAttr(chosen_hdr)
        elif self.cfg.hdr_path and os.path.isfile(self.cfg.hdr_path):
            dome.CreateTextureFileAttr(self.cfg.hdr_path)

        lo, hi = self.cfg.dome_intensity_range
        dome.CreateIntensityAttr(float(float(rng.uniform(lo, hi))))
        dome.CreateExposureAttr(float(self.cfg.dome_exposure))

        for prim in stage.Traverse():
            if prim.IsA(UsdLux.DomeLight):
                p = prim.GetPath().pathString
                if p == "/World/DomeLight":
                    continue
                img = UsdGeom.Imageable(prim)
                if img:
                    img.GetVisibilityAttr().Set(UsdGeom.Tokens.invisible)
                d = UsdLux.DomeLight(prim)
                d.CreateIntensityAttr().Set(0.0)
                d.CreateExposureAttr().Set(0.0)

    # -------------------------
    # Material randomization
    # -------------------------
    def _list_table_textures_once(self) -> list[str]:
        if self._table_textures is not None:
            return self._table_textures

        tex_dir = self.cfg.table_texture_dir
        if not tex_dir:
            self._table_textures = []
            return self._table_textures

        root = Path(tex_dir)
        if not root.is_dir():
            self._table_textures = []
            return self._table_textures

        exts = set(e.lower() for e in self.cfg.table_texture_exts)
        self._table_textures = sorted(
            str(p) for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts
        )
        return self._table_textures

    def _randomize_table_texture_in_place(self, stage, frame: int, seed: int) -> None:
        from pxr import Sdf, UsdGeom, UsdShade

        # 1) Get the target prim
        table_prim = stage.GetPrimAtPath(self.cfg.table_prim_path)
        if not table_prim.IsValid():
            print(f"[Warning] table_prim_path not found: {self.cfg.table_prim_path}")
            return

        # 2) Randomly select a texture
        textures = self._list_table_textures_once()
        if not textures:
            print("[Warning] No textures found for table randomization.")
            return
        rng = self._stream_rng(seed, frame, "table_texture")
        chosen_texture = os.path.abspath(rng.choice(textures))

        # 3) Create/override a material under the table prim's Looks scope
        mat_root = f"{self.cfg.table_prim_path}/Looks"
        if not stage.GetPrimAtPath(mat_root):
            UsdGeom.Scope.Define(stage, mat_root)

        mat_path = f"{mat_root}/StandardBoardMat"
        material = UsdShade.Material.Define(stage, mat_path)

        # Build a simple UsdPreviewSurface network
        shader = UsdShade.Shader.Define(stage, f"{mat_path}/PBRShader")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.4)

        tex_node = UsdShade.Shader.Define(stage, f"{mat_path}/DiffuseTexture")
        tex_node.CreateIdAttr("UsdUVTexture")
        tex_node.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(chosen_texture))
        tex_node.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)

        uv_reader = UsdShade.Shader.Define(stage, f"{mat_path}/UVReader")
        uv_reader.CreateIdAttr("UsdPrimvarReader_float2")
        # Isaac's default plane typically uses "st" as UV primvar
        uv_reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
        uv_reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)

        # Connect the network
        tex_node.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
            uv_reader.ConnectableAPI(), "result"
        )
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
            tex_node.ConnectableAPI(), "rgb"
        )
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")

        # 4) Bind the material with a strong opinion
        UsdShade.MaterialBindingAPI(table_prim).Bind(material, UsdShade.Tokens.strongerThanDescendants)

        # 5) Rendering refresh for Isaac Sim (some cases need an extra update)
        self._sim_app.update()

    # -------------------------
    # Occluders visibility
    # -------------------------
    def _list_occluder_roots(self, stage) -> list[Any]:
        roots: list[Any] = []
        prefixes = self.cfg.occluder_name_prefixes
        for prim in stage.Traverse():
            if prim.IsValid() and any(prim.GetName().startswith(p) for p in prefixes):
                roots.append(prim)
        roots = sorted(roots, key=lambda p: p.GetPath().pathString)
        return roots

    def _set_visibility_recursive(self, root_prims: list[Any], visible: bool) -> dict[str, str]:
        Usd, UsdGeom, _, _, _ = self._usd()

        prev: dict[str, str] = {}
        token = UsdGeom.Tokens.inherited if visible else UsdGeom.Tokens.invisible
        for root in root_prims:
            for p in Usd.PrimRange(root):
                img = UsdGeom.Imageable(p)
                if not img:
                    continue
                attr = img.GetVisibilityAttr()
                old = attr.Get()
                prev[p.GetPath().pathString] = str(old) if old is not None else ""
                attr.Set(token)
        return prev

    def _restore_visibility(self, stage, prev: dict[str, str]) -> None:
        _, UsdGeom, _, _, _ = self._usd()

        for path, old in prev.items():
            prim = stage.GetPrimAtPath(path)
            if not prim.IsValid():
                continue
            img = UsdGeom.Imageable(prim)
            if not img:
                continue
            img.GetVisibilityAttr().Set(UsdGeom.Tokens.inherited if old == "" else old)

    # -------------------------
    # Cameras & lights
    # -------------------------
    def _get_default_prim_path(self, stage) -> str:
        dp = stage.GetDefaultPrim()
        if dp and dp.IsValid():
            return dp.GetPath().pathString
        return "/root"

    def _list_user_cameras(self, stage) -> list[str]:
        _, UsdGeom, _, _, _ = self._usd()

        root = self._get_default_prim_path(stage)
        cams: list[str] = []
        for prim in stage.Traverse():
            if prim.IsA(UsdGeom.Camera):
                p = prim.GetPath().pathString
                if p.startswith(root + "/") and (self.cfg.cam_name_token in p):
                    if "OmniverseKit" in p or "Viewport" in p:
                        continue
                    cams.append(p)
        return sorted(set(cams))

    def _list_lights(self, stage) -> list[str]:
        _, _, UsdLux, _, _ = self._usd()

        root = self._get_default_prim_path(stage)
        lights: list[str] = []
        for prim in stage.Traverse():
            if prim.IsA(UsdLux.SphereLight):
                p = prim.GetPath().pathString
                if p.startswith(root + "/") and (self.cfg.light_name_token in p):
                    lights.append(p)
        return sorted(set(lights))

    def _set_only_k_lights_on(
        self, stage, light_paths: list[str], k: int, rng: random.Random
    ) -> list[str]:
        _, _, UsdLux, _, _ = self._usd()

        for p in light_paths:
            prim = stage.GetPrimAtPath(p)
            if prim.IsValid():
                UsdLux.SphereLight(prim).GetIntensityAttr().Set(0.0)

        if not light_paths:
            return []

        k = min(int(k), len(light_paths))
        chosen = rng.sample(light_paths, k=k)
        lo, hi = self.cfg.light_intensity_range
        for p in chosen:
            prim = stage.GetPrimAtPath(p)
            if prim.IsValid():
                UsdLux.SphereLight(prim).GetIntensityAttr().Set(float(rng.uniform(lo, hi)))
        return chosen

    # -------------------------
    # Proxy cameras
    # -------------------------
    def _apply_camera_pose_jitter(self, stage, proxy_cam_paths: list[str], frame: int, seed: int) -> None:
        Usd, UsdGeom, _, _, Gf = self._usd()

        rng = self._stream_rng(seed, frame, "camera_pose")
        pos_std = float(getattr(self.cfg, "camera_jitter_pos_std_m", 0.0))
        rot_std_deg = float(getattr(self.cfg, "camera_jitter_rot_std_deg", 0.0))

        xcache = UsdGeom.XformCache(Usd.TimeCode.Default())

        for cam_path in proxy_cam_paths:
            cam_prim = stage.GetPrimAtPath(cam_path)
            if not cam_prim.IsValid():
                continue

            rig_prim = cam_prim.GetParent()
            if not rig_prim.IsValid():
                continue

            base_world = xcache.GetLocalToWorldTransform(rig_prim)

            dx = rng.gauss(0.0, pos_std)
            dy = rng.gauss(0.0, pos_std)
            dz = rng.gauss(0.0, pos_std)

            rx = rng.gauss(0.0, rot_std_deg)
            ry = rng.gauss(0.0, rot_std_deg)
            rz = rng.gauss(0.0, rot_std_deg)

            t = Gf.Matrix4d().SetTranslate(Gf.Vec3d(dx, dy, dz))
            r = (
                Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(1, 0, 0), rx))
                * Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(0, 1, 0), ry))
                * Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(0, 0, 1), rz))
            )

            jitter = t * r
            new_world = base_world * jitter
            self._set_xform_world_matrix(rig_prim, new_world)

    def _apply_camera_intrinsics_jitter(self, stage, proxy_cam_paths: list[str], frame: int, seed: int) -> None:
        _, UsdGeom, _, _, _ = self._usd()

        rng = self._stream_rng(seed, frame, "camera_intrinsics")

        focal_std = float(getattr(self.cfg, "camera_jitter_focal_std_mm", 0.0))
        ap_std = float(getattr(self.cfg, "camera_jitter_aperture_std_mm", 0.0))
        fstop_std = float(getattr(self.cfg, "camera_jitter_fstop_std", 0.0))

        def jitter(v: float, s: float, lo: float | None = None) -> float:
            out = float(v) + rng.gauss(0.0, float(s))
            if lo is not None:
                out = max(float(lo), out)
            return out

        for cam_path in proxy_cam_paths:
            prim = stage.GetPrimAtPath(cam_path)
            if not prim.IsValid():
                continue

            cam = UsdGeom.Camera(prim)

            try:
                v = cam.GetFocalLengthAttr().Get()
                if v is not None and focal_std > 0:
                    cam.GetFocalLengthAttr().Set(jitter(float(v), focal_std, lo=0.1))
            except Exception:
                pass

            try:
                ha = cam.GetHorizontalApertureAttr().Get()
                va = cam.GetVerticalApertureAttr().Get()
                if ha is not None and ap_std > 0:
                    cam.GetHorizontalApertureAttr().Set(jitter(float(ha), ap_std, lo=0.1))
                if va is not None and ap_std > 0:
                    cam.GetVerticalApertureAttr().Set(jitter(float(va), ap_std, lo=0.1))
            except Exception:
                pass

            try:
                v = cam.GetFStopAttr().Get()
                if v is not None and fstop_std > 0:
                    cam.GetFStopAttr().Set(jitter(float(v), fstop_std, lo=0.1))
            except Exception:
                pass

    def _copy_camera_intrinsics(self, src_cam, dst_cam) -> None:
        pairs = [
            (src_cam.GetFocalLengthAttr(), dst_cam.GetFocalLengthAttr()),
            (src_cam.GetHorizontalApertureAttr(), dst_cam.GetHorizontalApertureAttr()),
            (src_cam.GetVerticalApertureAttr(), dst_cam.GetVerticalApertureAttr()),
            (src_cam.GetClippingRangeAttr(), dst_cam.GetClippingRangeAttr()),
            (src_cam.GetFocusDistanceAttr(), dst_cam.GetFocusDistanceAttr()),
            (src_cam.GetFStopAttr(), dst_cam.GetFStopAttr()),
        ]
        for a_src, a_dst in pairs:
            try:
                v = a_src.Get()
                if v is not None:
                    a_dst.Set(v)
            except Exception:
                pass

    def _set_xform_world_matrix(self, xf_prim, world_mat) -> None:
        _, UsdGeom, _, _, _ = self._usd()
        xf = UsdGeom.Xformable(xf_prim)
        xf.ClearXformOpOrder()
        xf.AddTransformOp().Set(world_mat)

    def _build_proxy_cameras(self, stage, k: int) -> list[str]:
        _, UsdGeom, _, _, _ = self._usd()

        UsdGeom.Xform.Define(stage, self.cfg.proxy_root)
        proxy_cam_paths: list[str] = []
        for i in range(int(k)):
            rig_path = f"{self.cfg.proxy_root}/ProxyRig_{i}"
            cam_path = f"{rig_path}/Camera"
            UsdGeom.Xform.Define(stage, rig_path)
            UsdGeom.Camera.Define(stage, cam_path)
            proxy_cam_paths.append(cam_path)
        self._wait_updates(self.cfg.warmup_after_build_proxies)
        return proxy_cam_paths

    def _sync_real_cams_to_proxies(self, stage, real_cam_paths: list[str], proxy_cam_paths: list[str]) -> None:
        Usd, UsdGeom, _, _, _ = self._usd()

        xcache = UsdGeom.XformCache(Usd.TimeCode.Default())
        for real_p, proxy_cam_p in zip(real_cam_paths, proxy_cam_paths):
            real_prim = stage.GetPrimAtPath(real_p)
            proxy_cam_prim = stage.GetPrimAtPath(proxy_cam_p)
            if not real_prim.IsValid() or not proxy_cam_prim.IsValid():
                continue

            proxy_rig_prim = proxy_cam_prim.GetParent()
            if not proxy_rig_prim.IsValid():
                continue

            world_mat = xcache.GetLocalToWorldTransform(real_prim)
            self._set_xform_world_matrix(proxy_rig_prim, world_mat)
            self._copy_camera_intrinsics(UsdGeom.Camera(real_prim), UsdGeom.Camera(proxy_cam_prim))

    # -------------------------
    # Pose (single frame)
    # -------------------------
    def _apply_pose_for_frame(
        self,
        stage,
        asset_paths: list[str],
        skel_paths: list[str],
        pos_arr: np.ndarray,
        dir_arr: np.ndarray,
        frame: int,
        num_wires: int,
    ) -> None:
        SkeletonRodDriver = self._import_skeleton_driver()
        Usd, _, _, _, _ = self._usd()

        T, w_data, _, n_nodes = pos_arr.shape
        e_elems = int(n_nodes - 1)
        if frame < 0 or frame >= T:
            raise ValueError(f"frame {frame} out of range [0, {T-1}]")

        w_use = min(int(num_wires), w_data, len(asset_paths), len(skel_paths))

        drivers: list[tuple[int, Any]] = []
        for i in range(int(w_use)):
            skel_path = skel_paths[i]
            if not skel_path:
                continue
            num_joints = self._get_skeleton_joint_count(stage, skel_path)
            if num_joints != e_elems:
                if self.cfg.skip_mismatch_wires:
                    continue
                raise RuntimeError(f"wire[{i}] joints mismatch (joints={num_joints}, elems={e_elems})")

            driver = SkeletonRodDriver(stage, skel_path, assume_chain=self.cfg.assume_chain_parent)
            driver.skel_prim = stage.GetPrimAtPath(skel_path)
            driver.skeleton_path = skel_path
            driver._setup_animation()
            drivers.append((i, driver))

        if not drivers:
            raise RuntimeError("No compatible wires found for this frame.")

        tc = Usd.TimeCode.Default()
        offset = np.array(self.cfg.wire_world_offset_xyz, dtype=np.float32).reshape((3, 1))
        for wire_i, driver in drivers:
            pos_3n = pos_arr[frame, wire_i] + offset
            dir_33e = dir_arr[frame, wire_i]
            driver.update_skeleton(pos_3n, dir_33e, tc)

    # -------------------------
    # Rendering: accumulate then write ONCE
    # -------------------------
    def _accumulate_without_writing(self, accum_steps: int, accum_subframes: int) -> None:
        # No writer attached: render products update, but nothing is written to disk.
        rep = self._rep()
        self._detach_basic_writer_best_effort()
        for _ in range(int(max(1, accum_steps))):
            rep.orchestrator.step(rt_subframes=int(max(1, accum_subframes)))
            self._sim_app.update()

    def _safe_writer_detach(self, writer) -> None:
        for meth in ["detach", "clear", "reset"]:
            if hasattr(writer, meth):
                try:
                    getattr(writer, meth)()
                    return
                except Exception:
                    pass

    def _detach_basic_writer_best_effort(self) -> None:
        rep = self._rep()
        try:
            writer = rep.WriterRegistry.get("BasicWriter")
        except Exception:
            return
        self._safe_writer_detach(writer)

    def _write_one_rgb(self, proxy_rps, rgb_dir: str) -> float:
        rep = self._rep()
        Path(rgb_dir).mkdir(parents=True, exist_ok=True)

        t0 = time.time()
        writer = rep.WriterRegistry.get("BasicWriter")
        self._safe_writer_detach(writer)
        writer.initialize(output_dir=rgb_dir, rgb=True)
        writer.attach(proxy_rps)
        rep.orchestrator.step(rt_subframes=1)
        self._sim_app.update()
        self._safe_writer_detach(writer)
        return t0


    def _write_one_seg(self, proxy_rps, seg_dir: str) -> float:
        rep = self._rep()
        Path(seg_dir).mkdir(parents=True, exist_ok=True)

        t0 = time.time()
        writer = rep.WriterRegistry.get("BasicWriter")
        self._safe_writer_detach(writer)
        writer.initialize(output_dir=seg_dir, instance_id_segmentation=True)
        writer.attach(proxy_rps)
        rep.orchestrator.step(rt_subframes=1)
        self._sim_app.update()
        self._safe_writer_detach(writer)
        return t0


    def _write_one_depth(self, proxy_rps, depth_dir: str) -> float:
        rep = self._rep()
        Path(depth_dir).mkdir(parents=True, exist_ok=True)

        t0 = time.time()
        writer = rep.WriterRegistry.get("BasicWriter")
        self._safe_writer_detach(writer)
        writer.initialize(output_dir=depth_dir, distance_to_camera=True)
        writer.attach(proxy_rps)
        rep.orchestrator.step(rt_subframes=1)
        self._sim_app.update()
        self._safe_writer_detach(writer)
        return t0


    def _latest_files(self, folder: str, exts: tuple[str, ...], k: int, newer_than: float) -> List[Path]:
        p = Path(folder)
        if not p.is_dir():
            return []

        eps = 0.2
        files: List[Path] = []
        for ext in exts:
            files += list(p.rglob(f"*{ext}"))  # recursive

        files = [f for f in files if f.is_file() and f.stat().st_mtime >= (newer_than - eps)]
        files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
        return files[:k]

    

    def _rename_outputs_after_write_global(
        self,
        ids: Sequence[int],
        rgb_dir: str,
        seg_dir: str | None,
        depth_dir: str | None,
        wrote_seg: bool,
        wrote_depth: bool,
        t_rgb: float,
        t_seg: float | None = None,
        t_depth: float | None = None,
    ) -> None:
        self._rename_group_global(
            folder=rgb_dir,
            ids=ids,
            channel="rgb",
            exts=(".png", ".jpg", ".jpeg", ".exr"),
            t0=t_rgb,
        )

        if wrote_seg and seg_dir is not None:
            self._rename_group_global(
                folder=seg_dir,
                ids=ids,
                channel="seg",
                exts=(".png", ".npy", ".npz"),
                t0=(t_seg if t_seg is not None else t_rgb),
            )

        if wrote_depth and depth_dir is not None:
            self._rename_group_global(
                folder=depth_dir,
                ids=ids,
                channel="depth",
                exts=(".npy", ".npz", ".exr", ".png"),
                t0=(t_depth if t_depth is not None else t_rgb),
            )
    def _rename_group_global(
        self,
        folder: str,
        ids: Sequence[int],
        channel: str,
        exts: tuple[str, ...],
        t0: float,
    ) -> None:
        """
        Find the latest K files written into `folder` after timestamp t0,
        and rename them to global IDs: {id:06d}{ext}.

        Also, if there are mapping JSONs (common for instance_id_segmentation),
        rename the latest K mapping json to {id:06d}_mapping.json.
        """
        folder_p = Path(folder)
        folder_p.mkdir(parents=True, exist_ok=True)

        K = int(len(ids))
        if K <= 0:
            return

        # 1) collect latest K data files (png/npy/npz/exr/...)
        files = self._latest_files(folder=str(folder_p), exts=exts, k=K, newer_than=t0)

        if len(files) < K:
            # Try a slightly more permissive search window in case filesystem mtimes lag
            files = self._latest_files(folder=str(folder_p), exts=exts, k=K, newer_than=(t0 - 2.0))

        if len(files) < K:
            raise RuntimeError(
                f"[RENAME] Not enough '{channel}' outputs found to rename. "
                f"Expected {K}, found {len(files)} in {folder_p}. "
                f"(exts={exts}, t0={t0})"
            )

        # We took newest-first in _latest_files(); reorder to oldest->newest to map to ids in order
        files_sorted = sorted(files, key=lambda f: (f.stat().st_mtime, f.name))

        # 2) rename/move to global IDs
        for src, gid in zip(files_sorted, ids):
            ext = src.suffix  # keep original extension
            dst = folder_p / f"{int(gid):06d}{ext}"

            # If target exists, avoid clobber
            if dst.exists():
                # Make it unique but still traceable
                dst = folder_p / f"{int(gid):06d}_{channel}{ext}"

            try:
                src.rename(dst)
            except Exception:
                # cross-device move fallback
                import shutil
                shutil.move(str(src), str(dst))

        # 3) (optional) rename segmentation mapping jsons
        # BasicWriter often writes mapping JSON for instance_id_segmentation.
        # We try to rename the latest K json files produced around t0.
        jsons = self._latest_files(folder=str(folder_p), exts=(".json",), k=K, newer_than=t0)
        if len(jsons) < K:
            jsons = self._latest_files(folder=str(folder_p), exts=(".json",), k=K, newer_than=(t0 - 2.0))

        if len(jsons) >= K:
            jsons_sorted = sorted(jsons, key=lambda f: (f.stat().st_mtime, f.name))
            for src, gid in zip(jsons_sorted, ids):
                dst = folder_p / f"{int(gid):06d}_mapping.json"
                if dst.exists():
                    dst = folder_p / f"{int(gid):06d}_mapping_{channel}.json"
                try:
                    src.rename(dst)
                except Exception:
                    import shutil
                    shutil.move(str(src), str(dst))

    # --- keep your _write_one_rgb/_write_one_seg/_write_one_depth as "return t0" versions ---
