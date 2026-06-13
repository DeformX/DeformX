#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
将 render_datacenter.py 渲染输出的 RGB 图片序列合成为视频。

用法:
  python images_to_video.py                          # 使用默认参数
  python images_to_video.py -i /path/to/rgb -o out.mp4 --fps 30
  python images_to_video.py --codec XVID -o out.avi  # 换编码器

依赖:
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


# 默认值与 render_datacenter.py 保持一致（可用 DEFORMX_DATA_ROOT 覆盖）
_DC_RENDERS = deformx_paths.env_path(
    "DEFORMX_DATA_ROOT", "Dataset_generator_datacenter", "data"
) / "renders"
DEFAULT_IMAGE_DIR = str(_DC_RENDERS / "rgb")
DEFAULT_OUTPUT = str(_DC_RENDERS / "output.mp4")
DEFAULT_FPS = 30  # render_datacenter.py 中 fps = 5


def collect_images(image_dir: str) -> list[str]:
    """收集目录下所有 png/jpg 图片并按文件名排序。"""
    patterns = ["*.png", "*.jpg", "*.jpeg"]
    files = []
    for pat in patterns:
        files.extend(glob.glob(os.path.join(image_dir, pat)))
    files.sort()
    return files


def images_to_video(image_dir: str, output_path: str, fps: float, codec: str):
    images = collect_images(image_dir)
    if not images:
        print(f"[ERROR] 在 {image_dir} 下未找到任何图片 (png/jpg)")
        sys.exit(1)

    print(f"[INFO] 找到 {len(images)} 张图片")
    print(f"  首帧: {os.path.basename(images[0])}")
    print(f"  末帧: {os.path.basename(images[-1])}")

    # 读取第一张图获取尺寸
    sample = cv2.imread(images[0])
    if sample is None:
        print(f"[ERROR] 无法读取图片: {images[0]}")
        sys.exit(1)
    h, w = sample.shape[:2]
    print(f"[INFO] 分辨率: {w}x{h}, FPS: {fps}, 编码: {codec}")

    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    if not writer.isOpened():
        print(f"[ERROR] 无法创建视频写入器 (codec={codec}, path={output_path})")
        print("  提示: 若 mp4v/avc1 不可用, 尝试 --codec XVID -o output.avi")
        sys.exit(1)

    for i, img_path in enumerate(images):
        frame = cv2.imread(img_path)
        if frame is None:
            print(f"  [WARN] 跳过无法读取的图片: {img_path}")
            continue
        # 确保尺寸一致
        if frame.shape[:2] != (h, w):
            frame = cv2.resize(frame, (w, h))
        writer.write(frame)
        if (i + 1) % 50 == 0 or (i + 1) == len(images):
            print(f"  [PROGRESS] {i + 1}/{len(images)}")

    writer.release()
    print(f"[DONE] 视频已保存: {output_path}")
    duration = len(images) / fps
    print(f"  帧数: {len(images)}, 时长: {duration:.2f}s")


def main():
    parser = argparse.ArgumentParser(description="将渲染的 RGB 图片序列合成为视频")
    parser.add_argument("-i", "--image-dir", default=DEFAULT_IMAGE_DIR,
                        help=f"RGB 图片目录 (默认: {DEFAULT_IMAGE_DIR})")
    parser.add_argument("-o", "--output", default=DEFAULT_OUTPUT,
                        help=f"输出视频路径 (默认: {DEFAULT_OUTPUT})")
    parser.add_argument("--fps", type=float, default=DEFAULT_FPS,
                        help=f"视频帧率 (默认: {DEFAULT_FPS})")
    parser.add_argument("--codec", default="mp4v",
                        help="FourCC 编码器 (默认: mp4v, 可选: XVID, avc1, MJPG)")
    args = parser.parse_args()

    images_to_video(args.image_dir, args.output, args.fps, args.codec)


if __name__ == "__main__":
    main()
