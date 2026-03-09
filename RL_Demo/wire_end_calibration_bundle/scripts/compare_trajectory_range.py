#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compare simulated wire-end trajectory against reference trajectory over a selected time range.

Outputs:
- metrics printed to terminal
- per-sample aligned CSV with errors
- error plot PNG
"""

import argparse
import os
from pathlib import Path

import numpy as np


def _normalize_field_name(name: str) -> str:
    return "".join(ch.lower() for ch in str(name) if ch.isalnum())


def _pick_name(field_names, candidates):
    for c in candidates:
        if c in field_names:
            return c
    normalized_map = {_normalize_field_name(n): n for n in field_names}
    for c in candidates:
        key = _normalize_field_name(c)
        if key in normalized_map:
            return normalized_map[key]
    return None


def load_sim_csv(path: str):
    arr = np.genfromtxt(path, delimiter=",", names=True, dtype=np.float64)
    if arr.size == 0:
        raise RuntimeError(f"Simulation CSV is empty: {path}")
    if arr.ndim == 0:
        arr = arr.reshape(1)
    fields = tuple(arr.dtype.names)

    t_col = _pick_name(fields, ("t", "time", "timestamp"))
    x_col = _pick_name(fields, ("end_x", "x"))
    y_col = _pick_name(fields, ("end_y", "y"))
    z_col = _pick_name(fields, ("end_z", "z"))
    if t_col is None or x_col is None or y_col is None or z_col is None:
        raise RuntimeError(
            f"Simulation CSV missing required columns. Need time + x/y/z. Found={fields}"
        )

    t = np.asarray(arr[t_col], dtype=np.float64)
    xyz = np.column_stack(
        [
            np.asarray(arr[x_col], dtype=np.float64),
            np.asarray(arr[y_col], dtype=np.float64),
            np.asarray(arr[z_col], dtype=np.float64),
        ]
    )
    m = np.isfinite(t) & np.isfinite(xyz).all(axis=1)
    t = t[m]
    xyz = xyz[m]
    if len(t) < 2:
        raise RuntimeError(f"Simulation CSV has too few valid rows: {path}")
    return t, xyz


def load_ref_csv(path: str, pos_scale: float):
    arr = np.genfromtxt(path, delimiter=",", names=True, dtype=np.float64)
    if arr.size == 0:
        raise RuntimeError(f"Reference CSV is empty: {path}")
    if arr.ndim == 0:
        arr = arr.reshape(1)
    fields = tuple(arr.dtype.names)

    t_col = _pick_name(fields, ("Time (Seconds)", "Time_Seconds", "time_seconds", "time", "t"))
    x_col = _pick_name(fields, ("X", "x"))
    y_col = _pick_name(fields, ("Y", "y"))
    z_col = _pick_name(fields, ("Z", "z"))
    if t_col is None or x_col is None or y_col is None or z_col is None:
        raise RuntimeError(
            f"Reference CSV missing required columns. Need time + X/Y/Z. Found={fields}"
        )

    t = np.asarray(arr[t_col], dtype=np.float64)
    xyz = np.column_stack(
        [
            np.asarray(arr[x_col], dtype=np.float64),
            np.asarray(arr[y_col], dtype=np.float64),
            np.asarray(arr[z_col], dtype=np.float64),
        ]
    )
    m = np.isfinite(t) & np.isfinite(xyz).all(axis=1)
    t = t[m]
    xyz = xyz[m] * float(pos_scale)
    if len(t) < 2:
        raise RuntimeError(f"Reference CSV has too few valid rows: {path}")
    return t, xyz


def interpolate_xyz(query_t: np.ndarray, src_t: np.ndarray, src_xyz: np.ndarray) -> np.ndarray:
    x = np.interp(query_t, src_t, src_xyz[:, 0])
    y = np.interp(query_t, src_t, src_xyz[:, 1])
    z = np.interp(query_t, src_t, src_xyz[:, 2])
    return np.column_stack([x, y, z])


def save_aligned_csv(path: str, t: np.ndarray, sim_xyz: np.ndarray, ref_xyz: np.ndarray, err_xyz: np.ndarray):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    err_norm = np.linalg.norm(err_xyz, axis=1)
    out = np.column_stack(
        [
            t,
            sim_xyz[:, 0],
            sim_xyz[:, 1],
            sim_xyz[:, 2],
            ref_xyz[:, 0],
            ref_xyz[:, 1],
            ref_xyz[:, 2],
            err_xyz[:, 0],
            err_xyz[:, 1],
            err_xyz[:, 2],
            err_norm,
        ]
    )
    np.savetxt(
        path,
        out,
        delimiter=",",
        comments="",
        header=(
            "t,sim_x,sim_y,sim_z,ref_x,ref_y,ref_z,"
            "err_x,err_y,err_z,err_norm"
        ),
    )


def save_error_plot(path: str, t: np.ndarray, err_xyz: np.ndarray):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    err_norm = np.linalg.norm(err_xyz, axis=1)
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    ax0.plot(t, err_xyz[:, 0], label="err x")
    ax0.plot(t, err_xyz[:, 1], label="err y")
    ax0.plot(t, err_xyz[:, 2], label="err z")
    ax0.set_ylabel("Error [m]")
    ax0.set_title("Trajectory Error by Axis")
    ax0.grid(True, alpha=0.3)
    ax0.legend(loc="best")

    ax1.plot(t, err_norm, color="k", linewidth=1.5, label="|error|")
    ax1.set_xlabel("t [s]")
    ax1.set_ylabel("Norm [m]")
    ax1.set_title("3D Position Error Norm")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="best")

    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def compute_metrics(err_xyz: np.ndarray):
    err_norm = np.linalg.norm(err_xyz, axis=1)
    mae_xyz = np.mean(np.abs(err_xyz), axis=0)
    rmse_xyz = np.sqrt(np.mean(err_xyz ** 2, axis=0))
    return {
        "samples": int(len(err_xyz)),
        "mae_x": float(mae_xyz[0]),
        "mae_y": float(mae_xyz[1]),
        "mae_z": float(mae_xyz[2]),
        "rmse_x": float(rmse_xyz[0]),
        "rmse_y": float(rmse_xyz[1]),
        "rmse_z": float(rmse_xyz[2]),
        "mae_3d": float(np.mean(err_norm)),
        "rmse_3d": float(np.sqrt(np.mean(err_norm ** 2))),
        "max_3d": float(np.max(err_norm)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sim_csv",
        type=str,
        default="/home/robot/Workspace/PyElastica-Mesh/mytest/whip_wire_end_positions.csv",
        help="Simulation CSV (e.g., whip_wire_end_positions.csv)",
    )
    parser.add_argument(
        "--ref_csv",
        type=str,
        default="/home/robot/Workspace/PyElastica-Mesh/mytest/whipping_high_1_001_stacked.csv",
        help="Reference CSV with time + X,Y,Z",
    )
    parser.add_argument("--t_start", type=float, default=0.0, help="Comparison range start time [s]")
    parser.add_argument(
        "--t_end",
        type=float,
        default=-1.0,
        help="Comparison range end time [s], <0 uses max available",
    )
    parser.add_argument(
        "--ref_pos_scale",
        type=float,
        default=1.0e-3,
        help="Scale factor for reference XYZ (default 1e-3 for mm->m)",
    )
    parser.add_argument(
        "--out_csv",
        type=str,
        default="/home/robot/Workspace/PyElastica-Mesh/mytest/trajectory_error_aligned.csv",
        help="Output aligned comparison CSV",
    )
    parser.add_argument(
        "--out_plot",
        type=str,
        default="/home/robot/Workspace/PyElastica-Mesh/mytest/trajectory_error_plot.png",
        help="Output error plot PNG",
    )
    args = parser.parse_args()

    sim_csv = os.path.abspath(args.sim_csv)
    ref_csv = os.path.abspath(args.ref_csv)
    if not os.path.isfile(sim_csv):
        raise FileNotFoundError(f"sim_csv not found: {sim_csv}")
    if not os.path.isfile(ref_csv):
        raise FileNotFoundError(f"ref_csv not found: {ref_csv}")

    sim_t, sim_xyz = load_sim_csv(sim_csv)
    ref_t, ref_xyz = load_ref_csv(ref_csv, pos_scale=float(args.ref_pos_scale))

    # Rebase both timelines to elapsed time from first sample.
    sim_t = sim_t - sim_t[0]
    ref_t = ref_t - ref_t[0]

    user_t0 = float(args.t_start)
    user_t1 = float(args.t_end)
    if user_t1 < 0.0:
        user_t1 = min(float(sim_t[-1]), float(ref_t[-1]))

    overlap_t0 = max(float(sim_t[0]), float(ref_t[0]), user_t0)
    overlap_t1 = min(float(sim_t[-1]), float(ref_t[-1]), user_t1)
    if overlap_t1 <= overlap_t0:
        raise RuntimeError(
            f"No overlap in selected range. "
            f"selected=[{user_t0:.6f},{user_t1:.6f}], "
            f"sim=[{sim_t[0]:.6f},{sim_t[-1]:.6f}], "
            f"ref=[{ref_t[0]:.6f},{ref_t[-1]:.6f}]"
        )

    sim_mask = (sim_t >= overlap_t0) & (sim_t <= overlap_t1)
    t = sim_t[sim_mask]
    sim_sel = sim_xyz[sim_mask]
    ref_interp = interpolate_xyz(t, ref_t, ref_xyz)
    err = sim_sel - ref_interp

    metrics = compute_metrics(err)
    save_aligned_csv(args.out_csv, t, sim_sel, ref_interp, err)
    save_error_plot(args.out_plot, t, err)

    print("[Compare] Range:", f"{overlap_t0:.6f}s -> {overlap_t1:.6f}s")
    print("[Compare] Samples:", metrics["samples"])
    print(
        "[Compare] MAE xyz [m]:",
        f"({metrics['mae_x']:.6f}, {metrics['mae_y']:.6f}, {metrics['mae_z']:.6f})",
    )
    print(
        "[Compare] RMSE xyz [m]:",
        f"({metrics['rmse_x']:.6f}, {metrics['rmse_y']:.6f}, {metrics['rmse_z']:.6f})",
    )
    print(
        "[Compare] 3D error [m]:",
        f"MAE={metrics['mae_3d']:.6f}, RMSE={metrics['rmse_3d']:.6f}, MAX={metrics['max_3d']:.6f}",
    )
    print("[Compare] Saved aligned CSV:", os.path.abspath(args.out_csv))
    print("[Compare] Saved error plot:", os.path.abspath(args.out_plot))


if __name__ == "__main__":
    main()
