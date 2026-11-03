#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Recolor segmentation images with vivid, high-contrast colors.

Supported inputs:
1) Instance-ID masks:
   - grayscale PNG
   - uint16 PNG
   - .npy mask with shape (H, W)

2) Already-colored segmentation PNG:
   - RGB/RGBA PNG
   - each unique color is treated as one segment and remapped

Usage examples:
    python recolor_seg.py \
        --input_dir /path/to/instance_seg \
        --output_dir /path/to/instance_seg_colorful

    python recolor_seg.py \
        --input_dir /path/to/instance_seg \
        --output_dir /path/to/instance_seg_colorful \
        --bg_mode black

    python recolor_seg.py \
        --input_dir /path/to/instance_seg \
        --output_dir /path/to/instance_seg_colorful \
        --alpha 255
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
from PIL import Image


# A vivid, high-contrast palette.
# Background is handled separately, so these are for instance colors.
VIVID_PALETTE = np.array(
    [
        [255,  32,  32],   # red
        [ 32, 160, 255],   # blue
        [  0, 220, 120],   # green
        [255, 190,   0],   # yellow-orange
        [180,  80, 255],   # purple
        [255,  80, 180],   # pink
        [  0, 230, 230],   # cyan
        [255, 120,   0],   # orange
        [140, 255,   0],   # lime
        [255, 255, 255],   # white
        [ 90,   0, 255],   # violet
        [255,   0, 120],   # magenta-red
        [  0, 120, 255],   # azure
        [255, 230,   0],   # bright yellow
        [  0, 255, 170],   # aqua green
        [255, 140, 140],   # light red
        [120, 255, 255],   # light cyan
        [220, 220,  40],   # olive yellow
        [255, 100, 255],   # light magenta
        [100, 255, 100],   # light green
    ],
    dtype=np.uint8,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True, help="Folder containing seg images or .npy masks")
    parser.add_argument("--output_dir", type=str, required=True, help="Folder to save recolored PNGs")
    parser.add_argument(
        "--bg_mode",
        type=str,
        default="black",
        choices=["black", "white", "transparent"],
        help="Background rendering mode",
    )
    parser.add_argument(
        "--alpha",
        type=int,
        default=255,
        help="Alpha for non-background pixels when saving RGBA output (0~255)",
    )
    return parser.parse_args()


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def get_bg_rgba(bg_mode: str) -> np.ndarray:
    if bg_mode == "black":
        return np.array([0, 0, 0, 255], dtype=np.uint8)
    if bg_mode == "white":
        return np.array([255, 255, 255, 255], dtype=np.uint8)
    if bg_mode == "transparent":
        return np.array([0, 0, 0, 0], dtype=np.uint8)
    raise ValueError(f"Unknown bg_mode: {bg_mode}")


def make_color_for_index(idx: int) -> np.ndarray:
    """
    Deterministically generate vivid colors.
    Uses the fixed palette first, then falls back to HSV-like spacing.
    """
    if idx < len(VIVID_PALETTE):
        return VIVID_PALETTE[idx]

    # Fallback: generate more colors with good spacing
    # Using a simple deterministic hue stepping
    hue = (idx * 137) % 360  # golden-angle-like spacing
    sat = 0.85
    val = 1.0
    return hsv_to_rgb_uint8(hue / 360.0, sat, val)


def hsv_to_rgb_uint8(h: float, s: float, v: float) -> np.ndarray:
    i = int(h * 6.0)
    f = h * 6.0 - i
    i = i % 6

    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)

    if i == 0:
        r, g, b = v, t, p
    elif i == 1:
        r, g, b = q, v, p
    elif i == 2:
        r, g, b = p, v, t
    elif i == 3:
        r, g, b = p, q, v
    elif i == 4:
        r, g, b = t, p, v
    else:
        r, g, b = v, p, q

    return np.array(
        [int(round(r * 255)), int(round(g * 255)), int(round(b * 255))],
        dtype=np.uint8,
    )


def is_image_file(path: Path) -> bool:
    return path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def load_mask_or_color_image(path: Path) -> Tuple[str, np.ndarray]:
    """
    Returns:
        ("id_mask", arr)   where arr shape is (H, W)
        ("color_img", arr) where arr shape is (H, W, C), C in {3,4}
    """
    if path.suffix.lower() == ".npy":
        arr = np.load(path)
        if arr.ndim != 2:
            raise ValueError(f"{path.name}: .npy mask must have shape (H, W), got {arr.shape}")
        return "id_mask", arr

    img = Image.open(path)
    arr = np.array(img)

    if arr.ndim == 2:
        return "id_mask", arr

    if arr.ndim == 3 and arr.shape[2] in (3, 4):
        # If all RGB channels are identical, treat as grayscale id-like mask
        if arr.shape[2] >= 3 and np.array_equal(arr[..., 0], arr[..., 1]) and np.array_equal(arr[..., 1], arr[..., 2]):
            return "id_mask", arr[..., 0]
        return "color_img", arr

    raise ValueError(f"{path.name}: unsupported image shape {arr.shape}")


def recolor_id_mask(mask: np.ndarray, bg_mode: str, alpha: int) -> np.ndarray:
    """
    Recolor a single-channel instance-id mask to vivid RGBA PNG.
    Convention:
    - id == 0 is background
    - each nonzero id gets a vivid color
    """
    if mask.ndim != 2:
        raise ValueError(f"Expected 2D mask, got {mask.shape}")

    mask = np.asarray(mask)
    h, w = mask.shape
    out = np.zeros((h, w, 4), dtype=np.uint8)

    bg_rgba = get_bg_rgba(bg_mode)
    out[:] = bg_rgba

    unique_ids = np.unique(mask)
    unique_ids = [x for x in unique_ids.tolist() if int(x) != 0]

    color_map: Dict[int, np.ndarray] = {}
    for color_idx, inst_id in enumerate(sorted(unique_ids)):
        color_map[int(inst_id)] = make_color_for_index(color_idx)

    for inst_id, rgb in color_map.items():
        sel = mask == inst_id
        out[sel, :3] = rgb
        out[sel, 3] = np.uint8(alpha)

    return out


def recolor_color_seg(img: np.ndarray, bg_mode: str, alpha: int) -> np.ndarray:
    """
    Recolor an already-colored segmentation image.
    Every unique non-background RGB color becomes a new vivid color.
    Background is detected as:
    - pure black [0,0,0] if present, otherwise
    - most frequent color
    """
    if img.ndim != 3 or img.shape[2] not in (3, 4):
        raise ValueError(f"Expected RGB/RGBA image, got {img.shape}")

    rgb = img[..., :3]
    h, w, _ = rgb.shape
    out = np.zeros((h, w, 4), dtype=np.uint8)
    bg_rgba = get_bg_rgba(bg_mode)
    out[:] = bg_rgba

    flat = rgb.reshape(-1, 3)
    unique_colors, counts = np.unique(flat, axis=0, return_counts=True)

    if len(unique_colors) == 0:
        return out

    black = np.array([0, 0, 0], dtype=np.uint8)
    has_black = np.any(np.all(unique_colors == black, axis=1))

    if has_black:
        bg_color = black
    else:
        bg_color = unique_colors[np.argmax(counts)]

    non_bg_colors: List[np.ndarray] = [
        c for c in unique_colors
        if not np.array_equal(c, bg_color)
    ]

    for color_idx, old_rgb in enumerate(non_bg_colors):
        new_rgb = make_color_for_index(color_idx)
        sel = np.all(rgb == old_rgb, axis=-1)
        out[sel, :3] = new_rgb
        out[sel, 3] = np.uint8(alpha)

    return out


def process_one_file(in_path: Path, out_dir: Path, bg_mode: str, alpha: int) -> None:
    kind, arr = load_mask_or_color_image(in_path)

    if kind == "id_mask":
        out = recolor_id_mask(arr, bg_mode=bg_mode, alpha=alpha)
    elif kind == "color_img":
        out = recolor_color_seg(arr, bg_mode=bg_mode, alpha=alpha)
    else:
        raise RuntimeError(f"Unknown input kind: {kind}")

    out_name = in_path.stem + "_colorful.png"
    out_path = out_dir / out_name
    Image.fromarray(out, mode="RGBA").save(out_path)
    print(f"[OK] {in_path.name} -> {out_path.name}")


def main() -> None:
    args = parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    ensure_output_dir(output_dir)

    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input dir not found: {input_dir}")

    files = sorted(
        [p for p in input_dir.iterdir() if p.is_file() and (is_image_file(p) or p.suffix.lower() == ".npy")]
    )

    if not files:
        raise RuntimeError(f"No supported files found in {input_dir}")

    print(f"[INFO] Found {len(files)} files in {input_dir}")
    print(f"[INFO] Saving recolored results to {output_dir}")

    for path in files:
        try:
            process_one_file(path, output_dir, bg_mode=args.bg_mode, alpha=args.alpha)
        except Exception as e:
            print(f"[WARN] Skip {path.name}: {e}")

    print("[DONE]")


if __name__ == "__main__":
    main()