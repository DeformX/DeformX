from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps


@dataclass(frozen=True)
class ConfigSheet:
    name: str
    start_frame: int
    count: int = 10


CONFIG_SHEETS: tuple[ConfigSheet, ...] = (
    ConfigSheet("drop_n2", 0),
    ConfigSheet("drop_n4", 100),
    ConfigSheet("drop_n8", 200),
    ConfigSheet("hang_n2", 300),
    ConfigSheet("hang_n4", 500),
    ConfigSheet("hang_n8", 700),
)


BG = (18, 20, 24)
PANEL = (30, 34, 40)
TEXT = (242, 242, 242)
SUBTEXT = (172, 177, 186)
ACCENT = (98, 178, 255)


def _font(size: int):
    candidates = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    )
    for path in candidates:
        if Path(path).is_file():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def _flatten_rgba(image: Image.Image, bg=(0, 0, 0)) -> Image.Image:
    if image.mode != "RGBA":
        return image.convert("RGB")
    base = Image.new("RGBA", image.size, (*bg, 255))
    return Image.alpha_composite(base, image).convert("RGB")


def _fit(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    return image.resize(size, Image.Resampling.LANCZOS)


def _depth_range(depth_paths: list[Path]) -> tuple[float, float]:
    values = []
    for path in depth_paths:
        arr = np.load(path)
        finite = arr[np.isfinite(arr)]
        if finite.size:
            values.append(finite.reshape(-1))
    if not values:
        return 0.0, 1.0
    merged = np.concatenate(values, axis=0)
    lo, hi = np.percentile(merged, [1.0, 99.0])
    if not np.isfinite(lo):
        lo = float(np.nanmin(merged))
    if not np.isfinite(hi):
        hi = float(np.nanmax(merged))
    if hi <= lo:
        hi = lo + 1e-6
    return float(lo), float(hi)


def _depth_preview(path: Path, size: tuple[int, int], lo: float, hi: float) -> Image.Image:
    arr = np.load(path).astype(np.float32)
    valid = np.isfinite(arr)
    norm = np.zeros_like(arr, dtype=np.float32)
    if valid.any():
        clipped = np.clip(arr[valid], lo, hi)
        norm[valid] = 1.0 - ((clipped - lo) / max(hi - lo, 1e-6))
    gray = Image.fromarray(np.clip(norm * 255.0, 0, 255).astype(np.uint8), mode="L")
    colored = ImageOps.colorize(gray, black="#0a1a3a", white="#fff59e")
    return _fit(colored, size)


def _load_preview(path: Path, size: tuple[int, int]) -> Image.Image:
    return _fit(_flatten_rgba(Image.open(path)), size)


def _frame_dir(seed_dir: Path, frame: int) -> Path:
    return seed_dir / f"frame_{frame:06d}"


def _build_sheet(seed_dir: Path, out_dir: Path, spec: ConfigSheet) -> Path:
    frames = list(range(spec.start_frame, spec.start_frame + spec.count))
    depth_paths = [_frame_dir(seed_dir, frame) / "depth" / "distance_to_camera_0000.npy" for frame in frames]
    lo, hi = _depth_range(depth_paths)

    cols = 5
    rows = 2
    sub_w = 320
    sub_h = 320
    gap = 14
    tile_pad = 18
    tile_label_h = 62
    header_h = 92
    footer_h = 18
    tile_w = tile_pad * 2 + (sub_w * 3) + (gap * 2)
    tile_h = tile_pad * 2 + tile_label_h + sub_h
    sheet_w = tile_w * cols + gap * (cols + 1)
    sheet_h = header_h + (tile_h * rows) + gap * (rows + 1) + footer_h

    sheet = Image.new("RGB", (sheet_w, sheet_h), BG)
    draw = ImageDraw.Draw(sheet)
    title_font = _font(42)
    meta_font = _font(20)
    tile_font = _font(24)
    small_font = _font(18)

    draw.text((gap, 18), spec.name, fill=TEXT, font=title_font)
    draw.text(
        (gap, 62),
        f"frames {frames[0]:06d}..{frames[-1]:06d} | RGB / SEG / DEPTH | seed_042",
        fill=SUBTEXT,
        font=meta_font,
    )
    draw.line((gap, header_h - 10, sheet_w - gap, header_h - 10), fill=ACCENT, width=3)

    for idx, frame in enumerate(frames):
        row = idx // cols
        col = idx % cols
        x0 = gap + col * (tile_w + gap)
        y0 = header_h + gap + row * (tile_h + gap)
        x1 = x0 + tile_w
        y1 = y0 + tile_h

        draw.rounded_rectangle((x0, y0, x1, y1), radius=18, fill=PANEL)
        draw.text((x0 + tile_pad, y0 + 10), f"frame {frame:06d}", fill=TEXT, font=tile_font)

        label_y = y0 + 36
        rgb_x = x0 + tile_pad
        seg_x = rgb_x + sub_w + gap
        dep_x = seg_x + sub_w + gap
        for label, lx in (("RGB", rgb_x), ("SEG", seg_x), ("DEPTH", dep_x)):
            draw.text((lx, label_y), label, fill=SUBTEXT, font=small_font)

        image_y = y0 + tile_pad + tile_label_h
        frame_dir = _frame_dir(seed_dir, frame)
        rgb = _load_preview(frame_dir / "rgb" / "rgb_0000.png", (sub_w, sub_h))
        seg = _load_preview(frame_dir / "seg" / "instance_id_segmentation_0000.png", (sub_w, sub_h))
        depth = _depth_preview(frame_dir / "depth" / "distance_to_camera_0000.npy", (sub_w, sub_h), lo, hi)

        sheet.paste(rgb, (rgb_x, image_y))
        sheet.paste(seg, (seg_x, image_y))
        sheet.paste(depth, (dep_x, image_y))

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{spec.name}_review_sheet.png"
    sheet.save(out_path)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build contact sheets for generated dataset review.")
    parser.add_argument(
        "--seed_dir",
        type=Path,
        default=Path("output/capture_reload_each_test_13/seed_042"),
        help="Seed output directory containing frame_XXXXXX folders.",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("output/capture_reload_each_test_13/seed_042/review_sheets"),
        help="Directory to write review sheets into.",
    )
    args = parser.parse_args()

    created = []
    for spec in CONFIG_SHEETS:
        created.append(_build_sheet(args.seed_dir, args.out_dir, spec))
    for path in created:
        print(path)


if __name__ == "__main__":
    main()
