#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Tuple
from delete_plane import clean_plane_segmentation
IMG_EXTS = (".png", ".jpg", ".jpeg", ".exr")
DEPTH_EXTS = (".npy", ".npz", ".exr", ".png")


def parse_cam_idx(name: str) -> Optional[int]:
    # "Replicator" -> 0, "Replicator_01" -> 1, "Replicator_1" -> 1
    if name == "Replicator":
        return 0
    m = re.match(r"Replicator_(\d+)$", name)
    if m:
        return int(m.group(1))
    return None


def parse_frame_idx(frame_dir_name: str) -> Optional[int]:
    # frame_000123 -> 123
    if not frame_dir_name.startswith("frame_"):
        return None
    try:
        return int(frame_dir_name.split("_", 1)[1])
    except Exception:
        return None


def pick_single_file(folder: Path, exts: Tuple[str, ...]) -> Optional[Path]:
    if not folder.is_dir():
        return None
    files = [p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in exts]
    if not files:
        return None
    files.sort(key=lambda p: p.name)  # stable ordering
    return files[-1]


def move_or_copy(src: Path, dst: Path, do_copy: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        # Avoid clobbering an existing file: append a numeric suffix.
        stem, suf = dst.stem, dst.suffix
        i = 1
        while True:
            cand = dst.with_name(f"{stem}_{i}{suf}")
            if not cand.exists():
                dst = cand
                break
            i += 1

    if do_copy:
        shutil.copy2(src, dst)
    else:
        shutil.move(str(src), str(dst))


@dataclass
class Sample:
    seed_name: str            # e.g. seed_042 or "__noseed__"
    frame_idx: int
    cam_idx: int
    rgb: Optional[Path]
    seg: Optional[Path]
    seg_map: Optional[Path]
    depth: Optional[Path]


def collect_samples(root: Path) -> List[Sample]:
    # If seed_* dirs exist, iterate over them; otherwise treat root itself as one seed.
    seed_dirs = sorted([p for p in root.iterdir() if p.is_dir() and p.name.startswith("seed_")])
    if not seed_dirs:
        seed_dirs = [root]

    samples: List[Sample] = []

    for sd in seed_dirs:
        seed_name = sd.name if sd != root else "__noseed__"
        frame_dirs = sorted([p for p in sd.iterdir() if p.is_dir() and p.name.startswith("frame_")])

        for fd in frame_dirs:
            frame_idx = parse_frame_idx(fd.name)
            if frame_idx is None:
                continue

            # RGB camera dirs
            rgb_root = fd / "rgb"
            rgb_rep_dirs = sorted([p for p in rgb_root.iterdir() if p.is_dir() and p.name.startswith("Replicator")]) if rgb_root.is_dir() else []

            # SEG camera dirs (usually share the RGB camera names)
            seg_root = fd / "seg"
            seg_rep_dirs = sorted([p for p in seg_root.iterdir() if p.is_dir() and p.name.startswith("Replicator")]) if seg_root.is_dir() else []

            # DEPTH camera dirs
            depth_root = fd / "depth"
            depth_rep_dirs = sorted([p for p in depth_root.iterdir() if p.is_dir() and p.name.startswith("Replicator")]) if depth_root.is_dir() else []

            # Prefer the RGB camera set; fall back to the union when there is no RGB.
            rep_names = {p.name for p in rgb_rep_dirs} or {p.name for p in seg_rep_dirs} or {p.name for p in depth_rep_dirs}
            rep_names = set(rep_names) | {p.name for p in seg_rep_dirs} | {p.name for p in depth_rep_dirs}

            for rep_name in sorted(rep_names):
                cam_idx = parse_cam_idx(rep_name)
                if cam_idx is None:
                    continue

                rgb_rep = fd / "rgb" / rep_name
                seg_rep = fd / "seg" / rep_name / "instance_id_segmentation"
                depth_rep = fd / "depth" / rep_name

                rgb_file = pick_single_file(rgb_rep, IMG_EXTS)
                seg_file = pick_single_file(seg_rep, (".png", ".exr"))
                seg_map = pick_single_file(seg_rep, (".json",))
                depth_file = pick_single_file(depth_rep, DEPTH_EXTS)

                # Skip samples that have no outputs at all.
                if not any([rgb_file, seg_file, seg_map, depth_file]):
                    continue

                samples.append(Sample(
                    seed_name=seed_name,
                    frame_idx=frame_idx,
                    cam_idx=cam_idx,
                    rgb=rgb_file,
                    seg=seg_file,
                    seg_map=seg_map,
                    depth=depth_file,
                ))

    # Stable ordering: seed -> frame -> cam
    samples.sort(key=lambda s: (s.seed_name, s.frame_idx, s.cam_idx))
    return samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Raw Replicator output directory (contains seed_*/frame_*).")
    ap.add_argument("--out", default=None, help="Destination directory. Default: <root>_organized")
    ap.add_argument("--copy", action="store_true", help="Copy files instead of moving them (default: move).")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    out_root = Path(args.out).resolve() if args.out else Path(str(root) + "_organized")
    out_rgb = out_root / "rgb"
    out_seg = out_root / "seg"
    out_depth = out_root / "depth"
    out_root.mkdir(parents=True, exist_ok=True)
    out_rgb.mkdir(parents=True, exist_ok=True)
    out_seg.mkdir(parents=True, exist_ok=True)
    out_depth.mkdir(parents=True, exist_ok=True)

    samples = collect_samples(root)
    if not samples:
        print("[WARN] No samples found.")
        return

    index_path = out_root / "index.jsonl"
    with index_path.open("w", encoding="utf-8") as f:
        for gid, s in enumerate(samples):
            gid_str = f"{gid:06d}"

            rgb_dst = None
            seg_dst = None
            map_dst = None
            depth_dst = None

            if s.rgb:
                rgb_dst = out_rgb / f"rgb_{gid_str}{s.rgb.suffix.lower()}"
                move_or_copy(s.rgb, rgb_dst, do_copy=bool(args.copy))

            if s.seg:
                seg_dst = out_seg / f"seg_{gid_str}{s.seg.suffix.lower()}"
                move_or_copy(s.seg, seg_dst, do_copy=bool(args.copy))

            if s.seg_map:
                map_dst = out_seg / f"seg_{gid_str}_mapping.json"
                move_or_copy(s.seg_map, map_dst, do_copy=bool(args.copy))

            if s.depth:
                depth_dst = out_depth / f"dep_{gid_str}{s.depth.suffix.lower()}"
                move_or_copy(s.depth, depth_dst, do_copy=bool(args.copy))

            rec = {
                "gid": gid,
                "seed": s.seed_name,
                "frame": s.frame_idx,
                "camera": s.cam_idx,
                "rgb": str(rgb_dst.relative_to(out_root)) if rgb_dst else None,
                "seg": str(seg_dst.relative_to(out_root)) if seg_dst else None,
                "seg_mapping": str(map_dst.relative_to(out_root)) if map_dst else None,
                "depth": str(depth_dst.relative_to(out_root)) if depth_dst else None,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"[DONE] wrote: {out_root}")
    print(f"[DONE] index: {index_path}")

    clean_plane_segmentation(out_seg)


if __name__ == "__main__":
    main()
