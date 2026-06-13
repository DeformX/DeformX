#!/usr/bin/env python3
"""Evaluate a PPO checkpoint, export a trajectory CSV, and replay it with wire visualization.

Usage:
  $ISAAC_PYTHON RL_Demo/tools/replay_checkpoint_wire_traj.py \
    --checkpoint /abs/path/to/ppo_step_50000.pt
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

from replay_wire_traj import replay


SCRIPT_PATH = Path(__file__).resolve()
RL_DEMO_ROOT = SCRIPT_PATH.parents[1]


def _bool_str(value: bool) -> str:
    return "true" if bool(value) else "false"


def _extract_joint_csv_from_eval_output(output: str) -> Path | None:
    tag = "[eval-export] joint_csv="
    for line in output.splitlines():
        if tag in line:
            raw = line.split(tag, 1)[1].strip()
            if raw:
                return Path(raw).expanduser().resolve()
    return None


def _find_latest_exported_joint_csv(export_dir: Path, export_prefix: str) -> Path | None:
    if not export_dir.is_dir():
        return None
    candidates = [
        p
        for p in export_dir.glob(f"{export_prefix}_*.csv")
        if p.is_file() and not p.name.endswith("_episode_summary.csv")
    ]
    if len(candidates) == 0:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def _run_eval_subprocess(cmd: list[str], cwd: Path) -> str:
    print(f"[checkpoint-replay] running: {' '.join(cmd)}")
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    captured_lines: list[str] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="")
        captured_lines.append(line)
    ret = proc.wait()
    if ret != 0:
        raise RuntimeError(f"RL.eval failed with exit code {ret}")
    return "".join(captured_lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to PPO checkpoint (*.pt).",
    )
    parser.add_argument("--task", type=str, default="wire_swing", help="Hydra task config name.")
    parser.add_argument("--algo", type=str, default="ppo", help="Hydra algo config name.")
    parser.add_argument("--episodes", type=int, default=8, help="Number of evaluation episodes.")
    parser.add_argument(
        "--export-mode",
        type=str,
        default="best_success",
        help="Episode selection mode: first_success|best_success|best_return|best_min_dist|latest.",
    )
    parser.add_argument(
        "--export-dir",
        type=str,
        default=str(RL_DEMO_ROOT / "eval_exports"),
        help="Directory for RL.eval exports.",
    )
    parser.add_argument(
        "--export-prefix",
        type=str,
        default="ckpt_replay",
        help="Prefix for RL.eval export files.",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use deterministic actor actions during evaluation.",
    )
    parser.add_argument(
        "--render-eval",
        action="store_true",
        help="Render during RL.eval rollout before replay.",
    )
    parser.add_argument(
        "--resample-dt",
        type=float,
        default=0.0,
        help="Optional resample dt for exported command CSV (seconds).",
    )
    parser.add_argument(
        "--skip-replay",
        action="store_true",
        help="Only export trajectory from checkpoint; skip replay visualization.",
    )
    parser.add_argument(
        "--hydra-override",
        action="append",
        default=[],
        help="Extra Hydra override for RL.eval (can be repeated).",
    )
    parser.add_argument(
        "--use-ground-contact",
        action="store_true",
        help="Enable co-sim ground contact/friction in replay stage.",
    )
    parser.add_argument("--ground-z", type=float, default=0.0, help="Ground plane z height for replay co-sim.")
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

    ckpt_path = Path(args.checkpoint).expanduser().resolve()
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    export_dir = Path(args.export_dir).expanduser().resolve()
    export_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "-m",
        "RL.eval",
        f"task={args.task}",
        f"algo={args.algo}",
        f"render={_bool_str(args.render_eval)}",
        f"eval.checkpoint={str(ckpt_path)}",
        f"eval.episodes={int(args.episodes)}",
        f"eval.export_mode={args.export_mode}",
        f"eval.export_dir={str(export_dir)}",
        f"eval.export_prefix={args.export_prefix}",
        f"eval.deterministic={_bool_str(args.deterministic)}",
        "eval.export_joint_csv=true",
        f"eval.resample_dt={float(args.resample_dt)}",
    ]
    for override in args.hydra_override:
        if str(override).strip():
            cmd.append(str(override).strip())

    eval_output = _run_eval_subprocess(cmd=cmd, cwd=RL_DEMO_ROOT)
    joint_csv = _extract_joint_csv_from_eval_output(eval_output)
    if joint_csv is None or not joint_csv.is_file():
        joint_csv = _find_latest_exported_joint_csv(export_dir=export_dir, export_prefix=args.export_prefix)
    if joint_csv is None or not joint_csv.is_file():
        raise RuntimeError(
            "Could not locate exported joint CSV from RL.eval. "
            f"Checked stdout and {export_dir} with prefix '{args.export_prefix}'."
        )

    print(f"[checkpoint-replay] selected trajectory CSV: {joint_csv}")
    if args.skip_replay:
        print("[checkpoint-replay] skip-replay enabled; stopping after export.")
        return

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

    out_csv, out_png, out_joint_png = replay(joint_csv, cosim_overrides=cosim_overrides)
    print(f"[checkpoint-replay] wrote tip trace CSV: {out_csv}")
    print(f"[checkpoint-replay] wrote YZ figure: {out_png}")
    print(f"[checkpoint-replay] wrote arm joint figure: {out_joint_png}")


if __name__ == "__main__":
    main()
