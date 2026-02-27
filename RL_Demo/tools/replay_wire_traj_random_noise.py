#!/usr/bin/env python3
"""Replay a joint trajectory CSV after adding random noise to commanded actions.

Usage:
  /home/robot/isaacsim/python.sh RL_Demo/tools/replay_wire_traj_random_noise.py /abs/path/to/traj.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from replay_wire_traj import JOINT_NAMES, NUM_ROBOT_DOFS, load_joint_csv, replay


def _validate_noise_joints(indices: list[int]) -> list[int]:
    unique = sorted(set(indices))
    for idx in unique:
        if idx < 0 or idx >= NUM_ROBOT_DOFS:
            raise ValueError(f"--noise-joints index out of range [0, {NUM_ROBOT_DOFS - 1}]: {idx}")
    return unique


def _validate_joint_index(name: str, idx: int | None) -> int | None:
    if idx is None:
        return None
    if idx < 0 or idx >= NUM_ROBOT_DOFS:
        raise ValueError(f"{name} out of range [0, {NUM_ROBOT_DOFS - 1}]: {idx}")
    return int(idx)


def _write_traj_csv(path: Path, times: np.ndarray, joints: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["t", *JOINT_NAMES])
        for i in range(times.shape[0]):
            writer.writerow(
                [
                    float(times[i]),
                    float(joints[i, 0]),
                    float(joints[i, 1]),
                    float(joints[i, 2]),
                    float(joints[i, 3]),
                    float(joints[i, 4]),
                    float(joints[i, 5]),
                ]
            )


def _add_action_noise(
    joint_positions: np.ndarray,
    noise_std: float,
    noise_clip: float | None,
    noise_joints: list[int],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    noisy_positions = np.asarray(joint_positions, dtype=np.float32).copy()
    noise = np.zeros_like(noisy_positions, dtype=np.float32)

    if noise_std <= 0.0 or len(noise_joints) == 0:
        return noisy_positions, noise

    sampled = rng.normal(
        loc=0.0,
        scale=float(noise_std),
        size=(joint_positions.shape[0], len(noise_joints)),
    ).astype(np.float32)
    if noise_clip is not None and noise_clip > 0.0:
        np.clip(sampled, -float(noise_clip), float(noise_clip), out=sampled)

    noise[:, noise_joints] = sampled
    noisy_positions += noise
    return noisy_positions, noise


def _add_single_joint_noise(
    joint_positions: np.ndarray,
    joint_idx: int | None,
    noise_std: float,
    noise_clip: float | None,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    noisy_positions = np.asarray(joint_positions, dtype=np.float32).copy()
    noise = np.zeros_like(noisy_positions, dtype=np.float32)
    if joint_idx is None or noise_std <= 0.0:
        return noisy_positions, noise

    sampled = rng.normal(loc=0.0, scale=float(noise_std), size=joint_positions.shape[0]).astype(np.float32)
    if noise_clip is not None and noise_clip > 0.0:
        np.clip(sampled, -float(noise_clip), float(noise_clip), out=sampled)
    noise[:, joint_idx] = sampled
    noisy_positions += noise
    return noisy_positions, noise


def _apply_joint_delay(
    joint_positions: np.ndarray,
    joint_idx: int | None,
    delay_steps: int,
) -> tuple[np.ndarray, int]:
    delayed = np.asarray(joint_positions, dtype=np.float32).copy()
    if joint_idx is None or delay_steps <= 0 or delayed.shape[0] == 0:
        return delayed, 0

    steps = int(min(delay_steps, delayed.shape[0] - 1))
    if steps <= 0:
        return delayed, 0
    original = delayed[:, joint_idx].copy()
    delayed[:steps, joint_idx] = original[0]
    delayed[steps:, joint_idx] = original[:-steps]
    return delayed, steps


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path_traj", type=str, help="Path to trajectory CSV")
    parser.add_argument(
        "--noise-std",
        type=float,
        default=0.03,
        help="Gaussian noise std (rad) added to commanded joint targets.",
    )
    parser.add_argument(
        "--noise-clip",
        type=float,
        default=None,
        help="Optional absolute clip for sampled noise (rad).",
    )
    parser.add_argument(
        "--noise-seed",
        type=int,
        default=None,
        help="Random seed for reproducible action noise.",
    )
    parser.add_argument(
        "--noise-joints",
        type=int,
        nargs="*",
        default=[0, 1, 2, 3, 4, 5],
        help="Joint indices to perturb (0-5). Default: all joints.",
    )
    parser.add_argument(
        "--single-joint-index",
        type=int,
        default=None,
        help="Optional single joint index (0-5) for slight extra perturbation.",
    )
    parser.add_argument(
        "--single-joint-std",
        type=float,
        default=0.005,
        help="Gaussian noise std (rad) for --single-joint-index.",
    )
    parser.add_argument(
        "--single-joint-clip",
        type=float,
        default=None,
        help="Optional absolute clip for --single-joint-index noise (rad).",
    )
    parser.add_argument(
        "--single-joint-only",
        action="store_true",
        help="Apply only the single-joint slight noise (skip global --noise-std noise).",
    )
    parser.add_argument(
        "--delay-joint-index",
        type=int,
        default=None,
        help="Optional joint index (0-5) whose commanded action will be delayed.",
    )
    parser.add_argument(
        "--delay-steps",
        type=int,
        default=0,
        help="Delay amount in command frames for --delay-joint-index.",
    )
    parser.add_argument(
        "--delay-seconds",
        type=float,
        default=None,
        help="Delay amount in seconds for --delay-joint-index. Overrides --delay-steps.",
    )
    parser.add_argument(
        "--output-noisy-csv",
        type=str,
        default=None,
        help="Path to write the noisy command trajectory CSV.",
    )
    parser.add_argument(
        "--use-ground-contact",
        action="store_true",
        help="Enable co-sim ground contact/friction.",
    )
    parser.add_argument("--ground-z", type=float, default=0.0, help="Ground plane z height for co-sim.")
    parser.add_argument("--ground-contact-k", type=float, default=1.0e4, help="Ground contact stiffness.")
    parser.add_argument("--ground-contact-nu", type=float, default=5.0, help="Ground contact damping.")
    parser.add_argument(
        "--ground-static-mu",
        type=float,
        nargs=3,
        metavar=("MU_T", "MU_N", "MU_B"),
        default=[1.0, 1.0, 1.0],
        help="Ground static friction triplet.",
    )
    parser.add_argument(
        "--ground-kinetic-mu",
        type=float,
        nargs=3,
        metavar=("MU_T", "MU_N", "MU_B"),
        default=[0.5, 0.5, 0.5],
        help="Ground kinetic friction triplet.",
    )
    parser.add_argument(
        "--ground-slip-velocity-tol",
        type=float,
        default=1.0e-6,
        help="Slip velocity tolerance for ground friction.",
    )
    parser.add_argument(
        "--settle-duration",
        type=float,
        default=0.0,
        help="Warm-start settle duration (seconds) before replay starts.",
    )
    args = parser.parse_args()

    traj_path = Path(args.path_traj).expanduser().resolve()
    if not traj_path.is_file():
        raise FileNotFoundError(f"Trajectory file not found: {traj_path}")

    noise_joints = _validate_noise_joints(args.noise_joints)
    single_joint_idx = _validate_joint_index("--single-joint-index", args.single_joint_index)
    delay_joint_idx = _validate_joint_index("--delay-joint-index", args.delay_joint_index)
    times, joint_positions = load_joint_csv(traj_path)
    rng = np.random.default_rng(args.noise_seed)

    noisy_joint_positions = np.asarray(joint_positions, dtype=np.float32).copy()
    total_noise = np.zeros_like(noisy_joint_positions, dtype=np.float32)

    if not args.single_joint_only:
        noisy_joint_positions, global_noise = _add_action_noise(
            joint_positions=noisy_joint_positions,
            noise_std=float(args.noise_std),
            noise_clip=args.noise_clip,
            noise_joints=noise_joints,
            rng=rng,
        )
        total_noise += global_noise

    noisy_joint_positions, slight_noise = _add_single_joint_noise(
        joint_positions=noisy_joint_positions,
        joint_idx=single_joint_idx,
        noise_std=float(args.single_joint_std),
        noise_clip=args.single_joint_clip,
        rng=rng,
    )
    total_noise += slight_noise

    if args.delay_seconds is not None:
        dts = np.diff(times)
        valid_dts = dts[np.isfinite(dts) & (dts > 1.0e-9)]
        dt_ref = float(np.median(valid_dts)) if valid_dts.size > 0 else 0.0
        delay_steps = int(round(float(args.delay_seconds) / dt_ref)) if dt_ref > 0.0 else 0
    else:
        delay_steps = int(args.delay_steps)
    if delay_steps < 0:
        raise ValueError(f"--delay-steps/--delay-seconds must resolve to >= 0, got {delay_steps}")
    noisy_joint_positions, applied_delay_steps = _apply_joint_delay(
        joint_positions=noisy_joint_positions,
        joint_idx=delay_joint_idx,
        delay_steps=delay_steps,
    )

    if args.output_noisy_csv is not None:
        noisy_csv_path = Path(args.output_noisy_csv).expanduser().resolve()
    else:
        noisy_csv_path = traj_path.with_name(f"{traj_path.stem}_noisy_actions.csv")
    _write_traj_csv(noisy_csv_path, times, noisy_joint_positions)

    all_perturbed_joints = sorted(set(noise_joints + ([] if single_joint_idx is None else [single_joint_idx])))
    affected_noise = (
        total_noise[:, all_perturbed_joints]
        if len(all_perturbed_joints) > 0
        else np.zeros((total_noise.shape[0], 0), dtype=np.float32)
    )
    mean_abs_noise = float(np.mean(np.abs(affected_noise))) if affected_noise.size > 0 else 0.0
    max_abs_noise = float(np.max(np.abs(affected_noise))) if affected_noise.size > 0 else 0.0
    print(
        "[noise] wrote noisy action CSV: "
        f"{noisy_csv_path} | seed={args.noise_seed} | global_joints={noise_joints} | "
        f"global_std={float(args.noise_std):.6f} rad | single_joint={single_joint_idx} | "
        f"single_std={float(args.single_joint_std):.6f} rad | delay_joint={delay_joint_idx} | "
        f"delay_steps={applied_delay_steps} | mean|n|={mean_abs_noise:.6f} rad | "
        f"max|n|={max_abs_noise:.6f} rad"
    )

    cosim_overrides = dict(
        use_ground_contact=bool(args.use_ground_contact),
        ground_z=float(args.ground_z),
        ground_contact_k=float(args.ground_contact_k),
        ground_contact_nu=float(args.ground_contact_nu),
        ground_static_mu=np.asarray(args.ground_static_mu, dtype=np.float64),
        ground_kinetic_mu=np.asarray(args.ground_kinetic_mu, dtype=np.float64),
        ground_slip_velocity_tol=float(args.ground_slip_velocity_tol),
        settle_duration=float(args.settle_duration),
    )
    out_csv, out_png, out_joint_png = replay(noisy_csv_path, cosim_overrides=cosim_overrides)
    print(f"[replay-noise] wrote tip trace CSV: {out_csv}")
    print(f"[replay-noise] wrote YZ figure: {out_png}")
    print(f"[replay-noise] wrote arm joint figure: {out_joint_png}")


if __name__ == "__main__":
    main()
