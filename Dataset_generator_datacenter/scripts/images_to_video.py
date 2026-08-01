#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Assemble the RGB image sequence rendered by render_datacenter.py into a video.

Usage:
  python images_to_video.py                          # use defaults
  python images_to_video.py -i /path/to/rgb -o out.mp4 --fps 30
  python images_to_video.py --codec XVID -o out.avi  # switch codec

Requires:
  pip install opencv-python
"""

import argparse
import glob
import os
import sys
from pathlib import Path

import cv2

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
import deformx_paths


# Defaults mirror render_datacenter.py (override with DEFORMX_DATA_ROOT).
_DC_RENDERS = deformx_paths.env_path(
    "DEFORMX_DATA_ROOT", "Dataset_generator_datacenter", "data"
) / "renders"
DEFAULT_IMAGE_DIR = str(_DC_RENDERS / "rgb")
DEFAULT_OUTPUT = str(_DC_RENDERS / "output.mp4")
DEFAULT_FPS = 30  # render_datacenter.py itself uses fps = 5


def collect_images(image_dir: str) -> list[str]:
    """Collect every png/jpg in the directory, sorted by filename."""
    patterns = ["*.png", "*.jpg", "*.jpeg"]
    files = []
    for pat in patterns:
        files.extend(glob.glob(os.path.join(image_dir, pat)))
    files.sort()
    return files


def images_to_video(image_dir: str, output_path: str, fps: float, codec: str):
    images = collect_images(image_dir)
    if not images:
        print(f"[ERROR] No images (png/jpg) found under {image_dir}")
        sys.exit(1)

    print(f"[INFO] Found {len(images)} images")
    print(f"  first frame: {os.path.basename(images[0])}")
    print(f"  last frame:  {os.path.basename(images[-1])}")

    # Read the first image to determine the output resolution.
    sample = cv2.imread(images[0])
    if sample is None:
        print(f"[ERROR] Could not read image: {images[0]}")
        sys.exit(1)
    h, w = sample.shape[:2]
    print(f"[INFO] resolution: {w}x{h}, fps: {fps}, codec: {codec}")

    # Make sure the output directory exists.
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    if not writer.isOpened():
        print(f"[ERROR] Could not open video writer (codec={codec}, path={output_path})")
        print("  hint: if mp4v/avc1 is unavailable, try --codec XVID -o output.avi")
        sys.exit(1)

    for i, img_path in enumerate(images):
        frame = cv2.imread(img_path)
        if frame is None:
            print(f"  [WARN] Skipping unreadable image: {img_path}")
            continue
        # Resize to match the first frame if needed.
        if frame.shape[:2] != (h, w):
            frame = cv2.resize(frame, (w, h))
        writer.write(frame)
        if (i + 1) % 50 == 0 or (i + 1) == len(images):
            print(f"  [PROGRESS] {i + 1}/{len(images)}")

    writer.release()
    print(f"[DONE] Video written: {output_path}")
    duration = len(images) / fps
    print(f"  frames: {len(images)}, duration: {duration:.2f}s")


def main():
    parser = argparse.ArgumentParser(description="Assemble a rendered RGB image sequence into a video")
    parser.add_argument("-i", "--image-dir", default=DEFAULT_IMAGE_DIR,
                        help=f"Directory of RGB images (default: {DEFAULT_IMAGE_DIR})")
    parser.add_argument("-o", "--output", default=DEFAULT_OUTPUT,
                        help=f"Output video path (default: {DEFAULT_OUTPUT})")
    parser.add_argument("--fps", type=float, default=DEFAULT_FPS,
                        help=f"Video frame rate (default: {DEFAULT_FPS})")
    parser.add_argument("--codec", default="mp4v",
                        help="FourCC codec (default: mp4v; alternatives: XVID, avc1, MJPG)")
    args = parser.parse_args()

    images_to_video(args.image_dir, args.output, args.fps, args.codec)


if __name__ == "__main__":
    main()
