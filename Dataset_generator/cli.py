#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Headless example:
  conda deactivate
  /home/robot/isaacsim/python.sh Dataset_generator/cli.py \
    --frame_start 0 --frame_end 300 --do_seg --accum_steps 80 --accum_subframes 16

Single frame:
  conda deactivate
  /home/robot/isaacsim/python.sh Dataset_generator/cli.py \
    --frame 100 --do_seg --accum_steps 120 --accum_subframes 32

Multi-variant sweep:
  conda deactivate
  /home/robot/isaacsim/python.sh Dataset_generator/cli.py \
    --do_seg --seed 42 --num_variants 10
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from dataclasses import replace
from pathlib import Path
from typing import List
from delete_plane import clean_plane_segmentation

if __package__ is None or __package__ == "":
    # Allow running as a plain script path: `python Dataset_generator/cli.py ...`
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from Dataset_generator.config import RenderConfig, default_config
    from Dataset_generator.generator import DatasetGenerator
else:
    from .config import RenderConfig, default_config
    from .generator import DatasetGenerator


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--frame", type=int, default=None, help="Render exactly one frame and exit.")
    p.add_argument("--frame_start", type=int, default=0, help="Start frame for loop.")
    p.add_argument("--frame_end", type=int, default=None, help="End frame for loop (inclusive).")
    p.add_argument("--frame_step", type=int, default=1, help="Step for loop.")
    p.add_argument("--do_seg", action="store_true", help="Also output instance_id_segmentation.")
    p.add_argument("--do_depth", action="store_true", help="Also output depth maps.")
    p.add_argument("--seed", type=int, default=42, help="Base seed for determinism.")
    p.add_argument("--accum_steps", type=int, default=80, help="How many accumulation steps (no writing).")
    p.add_argument("--accum_subframes", type=int, default=16, help="rt_subframes per accumulation step (no writing).")
    p.add_argument("--num_variants", type=int, default=1, help="How many seed variants to run.")
    return p.parse_args()


def build_frames(args, end_frame_npz: int) -> List[int]:
    if args.frame is not None:
        frames = [int(args.frame)]
    else:
        frame_end = end_frame_npz if args.frame_end is None else int(args.frame_end)
        step = int(max(1, args.frame_step))
        frames = list(range(int(args.frame_start), frame_end + 1, step))

    for f in frames:
        if int(f) < 0 or int(f) > int(end_frame_npz):
            raise ValueError(f"Frame {f} out of range [0, {end_frame_npz}]")
    return frames

def cleanup_all_replicator_dirs(base_out: str) -> None:
    print("Cleaning up Replicator dirs...")
    base = Path(base_out)
    if not base.is_dir():
        return
    for p in base.rglob("Replicator*"):
        if p.is_dir():
            shutil.rmtree(p)

def main() -> None:
    args = parse_args()

    cfg = default_config()
    base_out = Path(cfg.capture_out_dir)

    # Must be created before importing/using most omni.* APIs.
    from isaacsim import SimulationApp

    sim_app = SimulationApp({"headless": bool(cfg.headless)})
    # Build frames once using the base cfg (global sample range is independent of output dir)
    gen = DatasetGenerator(sim_app, cfg)
    end_frame_npz = gen.global_end_frame()
    frames = build_frames(args, end_frame_npz)

    for k in range(int(max(1, args.num_variants))):
        seed_k = int(args.seed) + k
        base_out_seed = base_out / f"seed_{seed_k:03d}"
        # cfg is frozen -> make a new cfg with a different output dir
        cfg_k = replace(cfg, capture_out_dir=str(base_out_seed))

        render_cfg_k = RenderConfig(
            do_seg=bool(args.do_seg),
            do_depth=bool(args.do_depth),
            seed=seed_k,
            accum_steps=int(args.accum_steps),
            accum_subframes=int(args.accum_subframes),
            frame_start=None if args.frame is not None else int(args.frame_start),
            frame_end=int(args.frame)
            if args.frame is not None
            else (None if args.frame_end is None else int(args.frame_end)),
            frame_step=int(args.frame_step),
        )

        print(f"[SWEEP] variant={k+1}/{args.num_variants} seed={seed_k} out={cfg_k.capture_out_dir}")
        sys.stdout.flush()
        gen.cfg = cfg_k
        gen.run(frames, render_cfg_k)

    # cleanup_all_replicator_dirs(str(base_out))
    # clean_plane_segmentation(str(base_out / "seg"))
    sim_app.close()
    



if __name__ == "__main__":
    main()
