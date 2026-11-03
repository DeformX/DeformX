#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Sweep wire density/Young's modulus and summarize whole-trajectory errors."""

import argparse
import csv
import json
import math
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_float_list(text: str) -> list[float]:
    vals = []
    for tok in text.replace(";", ",").split(","):
        tok = tok.strip()
        if tok:
            vals.append(float(tok))
    if not vals:
        raise ValueError("Expected at least one value")
    return vals


def ensure_2d_structured(arr: np.ndarray) -> np.ndarray:
    if arr.shape == ():
        arr = np.array([arr], dtype=arr.dtype)
    return arr


def read_metrics_from_comparison(csv_path: Path) -> dict:
    arr = np.genfromtxt(csv_path, delimiter=",", names=True, dtype=np.float64)
    arr = ensure_2d_structured(arr)
    err_norm = np.asarray(arr["err_norm"], dtype=np.float64)
    sim_xyz = np.column_stack((arr["sim_x"], arr["sim_y"], arr["sim_z"])).astype(np.float64)
    ref_xyz = np.column_stack((arr["ref_x"], arr["ref_y"], arr["ref_z"])).astype(np.float64)

    n = int(err_norm.size)
    mae = float(np.mean(err_norm)) if n else float("nan")
    rmse = float(math.sqrt(np.mean(err_norm**2))) if n else float("nan")
    max_err = float(np.max(err_norm)) if n else float("nan")
    final_err = float(err_norm[-1]) if n else float("nan")

    # Shape-only trajectory difference (time-independent) via nearest-neighbor distances.
    dmat = np.linalg.norm(sim_xyz[:, None, :] - ref_xyz[None, :, :], axis=2)
    min_sim_to_ref = np.min(dmat, axis=1)
    min_ref_to_sim = np.min(dmat, axis=0)
    chamfer = float(0.5 * (np.mean(min_sim_to_ref) + np.mean(min_ref_to_sim)))
    hausdorff = float(max(np.max(min_sim_to_ref), np.max(min_ref_to_sim)))

    return {
        "samples": n,
        "mae": mae,
        "rmse": rmse,
        "max_err": max_err,
        "final_err": final_err,
        "chamfer": chamfer,
        "hausdorff": hausdorff,
    }


def write_summary_csv(summary_csv: Path, rows: list[dict]) -> None:
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "case",
        "density",
        "youngs_modulus",
        "samples",
        "mae",
        "rmse",
        "max_err",
        "final_err",
        "chamfer",
        "hausdorff",
        "cfg_path",
        "comparison_csv",
        "plot_path",
        "video_path",
        "raw_path",
    ]
    with summary_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_summary_plot(summary_plot: Path, rows: list[dict]) -> None:
    summary_plot.parent.mkdir(parents=True, exist_ok=True)
    labels = [r["case"] for r in rows]
    rmse = [r["rmse"] for r in rows]
    mae = [r["mae"] for r in rows]
    chamfer = [r["chamfer"] for r in rows]
    hausdorff = [r["hausdorff"] for r in rows]
    x = np.arange(len(rows))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    ax1.plot(x, rmse, "o-", label="RMSE (time-aligned)")
    ax1.plot(x, mae, "s-", label="MAE (time-aligned)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
    ax1.set_ylabel("Error [m]")
    ax1.set_title("Whole-Trajectory Error (No Error-vs-Time Plot)")
    ax1.grid(True, alpha=0.35)
    ax1.legend()

    ax2.plot(x, chamfer, "o-", label="Chamfer (shape-only)")
    ax2.plot(x, hausdorff, "s-", label="Hausdorff (shape-only)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
    ax2.set_ylabel("Distance [m]")
    ax2.set_title("Time-Independent Trajectory Difference")
    ax2.grid(True, alpha=0.35)
    ax2.legend()

    fig.savefig(summary_plot, dpi=180)
    plt.close(fig)


def main() -> int:
    here = Path(__file__).resolve().parent
    bundle_root = here.parent

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--densities",
        type=str,
        default="400,500,600",
        help="Comma-separated density values",
    )
    parser.add_argument(
        "--youngs",
        type=str,
        default="70000,130000",
        help="Comma-separated Young's modulus values",
    )
    parser.add_argument(
        "--base_cfg",
        type=str,
        default=str(bundle_root / "config" / "replay_whip_traj_wire_end_calibration_engine_cfg.json"),
    )
    parser.add_argument(
        "--traj_csv",
        type=str,
        default=str(bundle_root / "data" / "trajectory" / "whip_traj_high.csv"),
    )
    parser.add_argument(
        "--ref_csv",
        type=str,
        default=str(bundle_root / "data" / "reference" / "whipping_high_1_001_stacked_transformed (2).csv"),
    )
    parser.add_argument("--compare_t_start", type=float, default=0.0)
    parser.add_argument("--compare_t_end", type=float, default=-1.0)
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(bundle_root / "outputs" / f"sweep_{datetime.now().strftime('%Y%m%d_%H%M%S')}"),
    )
    args = parser.parse_args()

    densities = parse_float_list(args.densities)
    youngs_vals = parse_float_list(args.youngs)

    output_dir = Path(args.output_dir).resolve()
    cfg_dir = output_dir / "configs"
    cmp_dir = output_dir / "comparison"
    plot_dir = output_dir / "plots"
    video_dir = output_dir / "videos"
    raw_dir = output_dir / "raw"
    for d in (cfg_dir, cmp_dir, plot_dir, video_dir, raw_dir):
        d.mkdir(parents=True, exist_ok=True)

    with open(args.base_cfg, "r", encoding="utf-8") as f:
        base_cfg = json.load(f)

    replay_script = here / "replay_whip_traj_wire_end_calibration.py"
    isaac_python = Path(os.environ.get("ISAAC_PYTHON", "isaacsim/python.sh"))
    summary_rows = []

    total_cases = len(densities) * len(youngs_vals)
    case_idx = 0
    for density in densities:
        for youngs in youngs_vals:
            case_idx += 1
            case = f"rho{density:g}_E{youngs:g}"
            print(f"[Sweep] ({case_idx}/{total_cases}) Running {case}", flush=True)

            cfg = dict(base_cfg)
            cfg["density"] = float(density)
            cfg["youngs_modulus"] = float(youngs)
            cfg["output_name"] = case

            cfg_path = cfg_dir / f"{case}.json"
            with cfg_path.open("w", encoding="utf-8") as f:
                json.dump(cfg, f, indent=2, sort_keys=True)

            out_cmp = cmp_dir / f"{case}_comparison.csv"
            out_plot = plot_dir / f"{case}.png"
            out_video = video_dir / f"{case}.mp4"
            out_raw = raw_dir / f"{case}_wire_end_positions.csv"

            cmd = [
                str(isaac_python),
                str(replay_script),
                "--headless",
                "--traj_csv",
                str(Path(args.traj_csv).resolve()),
                "--ref_csv",
                str(Path(args.ref_csv).resolve()),
                "--engine_cfg",
                str(cfg_path),
                "--compare_t_start",
                str(args.compare_t_start),
                "--compare_t_end",
                str(args.compare_t_end),
                "--compare_out_csv",
                str(out_cmp),
                "--out_plot",
                str(out_plot),
                "--out_csv",
                str(out_raw),
                "--make_video",
                "--out_video",
                str(out_video),
            ]
            subprocess.run(cmd, cwd=str(bundle_root), check=True)

            metrics = read_metrics_from_comparison(out_cmp)
            row = {
                "case": case,
                "density": density,
                "youngs_modulus": youngs,
                "samples": metrics["samples"],
                "mae": metrics["mae"],
                "rmse": metrics["rmse"],
                "max_err": metrics["max_err"],
                "final_err": metrics["final_err"],
                "chamfer": metrics["chamfer"],
                "hausdorff": metrics["hausdorff"],
                "cfg_path": str(cfg_path),
                "comparison_csv": str(out_cmp),
                "plot_path": str(out_plot),
                "video_path": str(out_video),
                "raw_path": str(out_raw),
            }
            summary_rows.append(row)
            print(
                f"[Sweep] {case}: RMSE={row['rmse']:.4f} m, MAE={row['mae']:.4f} m, "
                f"Chamfer={row['chamfer']:.4f} m, Hausdorff={row['hausdorff']:.4f} m",
                flush=True,
            )

    summary_csv = output_dir / "summary_metrics.csv"
    summary_plot = output_dir / "summary_metrics.png"
    write_summary_csv(summary_csv, summary_rows)
    write_summary_plot(summary_plot, summary_rows)

    best_rmse = min(summary_rows, key=lambda r: r["rmse"])
    best_chamfer = min(summary_rows, key=lambda r: r["chamfer"])
    print(f"[Sweep] Summary CSV: {summary_csv}")
    print(f"[Sweep] Summary Plot: {summary_plot}")
    print(
        f"[Sweep] Best RMSE: {best_rmse['case']} (RMSE={best_rmse['rmse']:.4f} m, "
        f"density={best_rmse['density']}, E={best_rmse['youngs_modulus']})"
    )
    print(
        f"[Sweep] Best Chamfer: {best_chamfer['case']} (Chamfer={best_chamfer['chamfer']:.4f} m, "
        f"density={best_chamfer['density']}, E={best_chamfer['youngs_modulus']})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
