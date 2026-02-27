#!/usr/bin/env python3
"""Rank successful action trajectories by joint speed.

Example:
    python RL_Demo/tools/rank_the_fast_traj.py RL_Demo/data/success_trajectories
"""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np


TRAJ_NAME_RE = re.compile(r"^success_episode_\d+_env\d+_step\d+\.csv$")
JOINT_CANDIDATES: tuple[tuple[str, ...], ...] = (
    ("shoulder_pan", "shoulder_pan_joint"),
    ("shoulder_lift", "shoulder_lift_joint"),
    ("elbow", "elbow_joint"),
    ("wrist_1", "wrist_1_joint"),
    ("wrist_2", "wrist_2_joint"),
    ("wrist_3", "wrist_3_joint"),
)
TIME_CANDIDATES = ("t", "time", "timestamp")


@dataclass
class TrajStats:
    path: Path
    samples: int
    duration: float
    mean_abs_joint_speed: np.ndarray  # (J,)
    max_abs_joint_speed: np.ndarray  # (J,)
    rms_joint_speed: np.ndarray  # (J,)
    score: float


def _normalize_field_name(name: str) -> str:
    return "".join(ch.lower() for ch in str(name) if ch.isalnum())


def _pick_name(field_names: tuple[str, ...], candidates: tuple[str, ...]) -> str | None:
    for c in candidates:
        if c in field_names:
            return c
    normalized_map = {_normalize_field_name(n): n for n in field_names}
    for c in candidates:
        key = _normalize_field_name(c)
        if key in normalized_map:
            return normalized_map[key]
    return None


def _load_joint_csv(path_csv: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    arr = np.genfromtxt(str(path_csv), delimiter=",", names=True, dtype=np.float64)
    if arr.size == 0:
        raise RuntimeError(f"CSV is empty: {path_csv}")
    if arr.ndim == 0:
        arr = arr.reshape(1)
    if arr.dtype.names is None:
        raise RuntimeError(f"CSV must contain a header row: {path_csv}")

    field_names = tuple(arr.dtype.names)
    t_col = _pick_name(field_names, TIME_CANDIDATES)
    if t_col is None:
        raise RuntimeError(
            f"Missing time column in {path_csv}. Expected one of {TIME_CANDIDATES}."
        )

    joint_cols: list[str] = []
    for candidates in JOINT_CANDIDATES:
        col = _pick_name(field_names, candidates)
        if col is None:
            raise RuntimeError(
                f"Missing joint column candidates {candidates} in {path_csv}. "
                f"Available fields: {field_names}"
            )
        joint_cols.append(col)

    times = np.asarray(arr[t_col], dtype=np.float64).reshape(-1)
    joints = np.column_stack([np.asarray(arr[c], dtype=np.float64) for c in joint_cols])

    if times.shape[0] != joints.shape[0]:
        raise RuntimeError(
            f"Times length {times.shape[0]} != joints length {joints.shape[0]} in {path_csv}"
        )
    if not np.all(np.isfinite(times)) or not np.all(np.isfinite(joints)):
        raise RuntimeError(f"NaN/inf detected in {path_csv}")
    if times.size < 2:
        raise RuntimeError(f"Need at least 2 rows in {path_csv} to compute speeds.")
    return times, joints, joint_cols


def _compute_joint_speed_stats(times: np.ndarray, joints: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dt = np.diff(times)
    dq = np.diff(joints, axis=0)
    valid = np.isfinite(dt) & (dt > 1.0e-12)

    if not np.any(valid):
        n_joint = joints.shape[1]
        zeros = np.zeros((n_joint,), dtype=np.float64)
        return zeros, zeros, zeros

    vel = dq[valid] / dt[valid, None]
    abs_vel = np.abs(vel)
    mean_abs = np.mean(abs_vel, axis=0)
    max_abs = np.max(abs_vel, axis=0)
    rms = np.sqrt(np.mean(np.square(vel), axis=0))
    return mean_abs, max_abs, rms


def _collect_traj_csvs(root_dir: Path, recursive: bool) -> list[Path]:
    if not root_dir.exists():
        raise FileNotFoundError(f"Input folder does not exist: {root_dir}")
    if not root_dir.is_dir():
        raise NotADirectoryError(f"Input must be a directory: {root_dir}")

    iterator = root_dir.rglob("*.csv") if recursive else root_dir.glob("*.csv")
    paths = [p for p in iterator if TRAJ_NAME_RE.match(p.name)]
    paths.sort()
    return paths


def _rank_trajectories(paths: list[Path]) -> tuple[list[TrajStats], list[str]]:
    all_stats: list[TrajStats] = []
    joint_names: list[str] | None = None

    for p in paths:
        times, joints, cols = _load_joint_csv(p)
        if joint_names is None:
            joint_names = cols
        elif len(cols) != len(joint_names):
            raise RuntimeError(
                f"Joint dimension mismatch for {p}. Expected {len(joint_names)} joints, got {len(cols)}."
            )

        mean_abs, max_abs, rms = _compute_joint_speed_stats(times, joints)
        score = float(np.mean(mean_abs))
        all_stats.append(
            TrajStats(
                path=p,
                samples=int(times.shape[0]),
                duration=float(times[-1] - times[0]),
                mean_abs_joint_speed=mean_abs,
                max_abs_joint_speed=max_abs,
                rms_joint_speed=rms,
                score=score,
            )
        )

    if joint_names is None:
        joint_names = []

    all_stats.sort(key=lambda s: s.score, reverse=True)
    return all_stats, joint_names


def _print_rank_table(stats: list[TrajStats], joint_names: list[str], top_k: int):
    shown = stats if top_k <= 0 else stats[:top_k]
    print(f"Ranked trajectories: {len(stats)} (showing {len(shown)})")
    print("Ranking metric: mean(abs(joint_velocity)) averaged across all joints [rad/s]")
    print("")

    if not shown:
        return

    joint_header = " | ".join(joint_names)
    print(f"{'rank':>4} | {'score':>10} | {'samples':>7} | {'duration_s':>10} | file")
    print("-" * 120)
    for i, s in enumerate(shown, start=1):
        print(
            f"{i:4d} | {s.score:10.6f} | {s.samples:7d} | {s.duration:10.4f} | {s.path}"
        )
        joint_vals = " | ".join(f"{v:10.6f}" for v in s.mean_abs_joint_speed)
        print(f"      mean_abs[{joint_header}]")
        print(f"      {joint_vals}")
    print("")


def _write_rank_csv(out_csv: Path, stats: list[TrajStats], joint_names: list[str]):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["rank", "score", "samples", "duration_s", "path"]
    fieldnames += [f"mean_abs_{jn}" for jn in joint_names]
    fieldnames += [f"max_abs_{jn}" for jn in joint_names]
    fieldnames += [f"rms_{jn}" for jn in joint_names]

    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rank, s in enumerate(stats, start=1):
            row = {
                "rank": rank,
                "score": float(s.score),
                "samples": int(s.samples),
                "duration_s": float(s.duration),
                "path": str(s.path),
            }
            row.update({f"mean_abs_{jn}": float(v) for jn, v in zip(joint_names, s.mean_abs_joint_speed)})
            row.update({f"max_abs_{jn}": float(v) for jn, v in zip(joint_names, s.max_abs_joint_speed)})
            row.update({f"rms_{jn}": float(v) for jn, v in zip(joint_names, s.rms_joint_speed)})
            writer.writerow(row)
    print(f"Saved rank CSV: {out_csv}")


def main():
    parser = argparse.ArgumentParser(
        description="Collect success action trajectory CSVs, compute per-joint speeds, and rank by speed."
    )
    parser.add_argument("folder", type=Path, help="Folder containing success_episode_*_env*_step*.csv files.")
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Only scan the top-level folder (default scans recursively).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=0,
        help="How many ranked rows to print. 0 prints all.",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=None,
        help="Optional output CSV path for the full ranking table.",
    )
    args = parser.parse_args()

    csv_paths = _collect_traj_csvs(args.folder, recursive=not args.no_recursive)
    if not csv_paths:
        print(f"No matching files found under: {args.folder}")
        print("Expected filename pattern: success_episode_<id>_env<id>_step<id>.csv")
        return

    print(f"Found {len(csv_paths)} trajectory CSV files under {args.folder}")
    stats, joint_names = _rank_trajectories(csv_paths)
    _print_rank_table(stats, joint_names, top_k=max(0, int(args.top_k)))

    if args.out_csv is not None:
        _write_rank_csv(args.out_csv, stats, joint_names)


if __name__ == "__main__":
    main()
