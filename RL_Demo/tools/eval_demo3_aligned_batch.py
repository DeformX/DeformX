"""
MoCap CSV -> Local Coordinate Transform, Per-File YZ Error & 3-D Visualisation

Coordinate transform: world (x, y, z) -> local (y, -z, -x)

For each CSV file:
  1. Parse & transform Unlabeled markers to local frame
  2. Find nearest point to target **in the YZ plane**
  3. Record YZ Error (distance in YZ plane from target)
  4. Save 3-D plot

After all files: print summary table with per-file YZ Error and mean YZ Error,
and save a summary CSV + bar chart.

Usage:
  python mocap_transform_viz.py <csv_dir> [--target X Y Z] [--outdir OUTPUT_DIR]
"""

import argparse
import csv
import glob
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


# ============================================================
# Coordinate-system transform  (world -> local)
# ============================================================

def transform_point(p: np.ndarray, scale: float = 1.0) -> np.ndarray:
    s = p / scale
    return np.array([-s[2], s[0], -s[1]])


# ============================================================
# CSV parsing
# ============================================================

def _find_header_rows(rows):
    type_idx = name_idx = measure_idx = header_idx = data_idx = None
    for i, row in enumerate(rows):
        if len(row) < 3:
            continue
        cell1 = row[1].strip() if len(row) > 1 else ""
        if cell1 == "Type" and type_idx is None:
            type_idx = i
        elif cell1 == "Name" and name_idx is None:
            name_idx = i
        if measure_idx is None and any(
            c.strip() in ("Position", "Rotation") for c in row[2:]
        ):
            if cell1 not in ("Type", "Name", "ID"):
                measure_idx = i
        if row[0].strip() == "Frame" and header_idx is None:
            header_idx = i
    if header_idx is not None:
        data_idx = header_idx + 1
    return type_idx, name_idx, measure_idx, data_idx


def parse_mocap_csv(filepath: str, scale: float = 1.0) -> np.ndarray:
    with open(filepath, "r", encoding="utf-8-sig") as f:
        rows = list(csv.reader(f))

    type_idx, name_idx, measure_idx, data_idx = _find_header_rows(rows)

    if any(v is None for v in (type_idx, name_idx, measure_idx, data_idx)):
        print(f"  [WARN] {filepath}: could not locate header rows. Skipping.")
        return np.empty((0, 3))

    type_row = rows[type_idx]
    name_row = rows[name_idx]
    measure_row = rows[measure_idx]

    unlabeled_xyz_groups = []
    col = 2
    ncols = max(len(type_row), len(name_row), len(measure_row))
    while col < ncols:
        typ = type_row[col].strip() if col < len(type_row) else ""
        name = name_row[col].strip() if col < len(name_row) else ""
        meas = measure_row[col].strip() if col < len(measure_row) else ""
        if typ == "Marker" and name.startswith("Unlabeled") and meas == "Position":
            if col + 2 < ncols:
                unlabeled_xyz_groups.append((col, col + 1, col + 2))
            col += 3
        else:
            col += 1

    if not unlabeled_xyz_groups:
        print(f"  [WARN] {filepath}: no Unlabeled marker Position columns found.")
        return np.empty((0, 3))

    print(f"  {os.path.basename(filepath)}: {len(unlabeled_xyz_groups)} Unlabeled marker group(s)")

    points = []
    for row in rows[data_idx:]:
        for cx, cy, cz in unlabeled_xyz_groups:
            try:
                x_str = row[cx].strip() if cx < len(row) else ""
                y_str = row[cy].strip() if cy < len(row) else ""
                z_str = row[cz].strip() if cz < len(row) else ""
                if "" in (x_str, y_str, z_str):
                    continue
                p_old = np.array([float(x_str), float(y_str), float(z_str)])
                points.append(transform_point(p_old, scale))
            except (ValueError, IndexError):
                continue

    return np.array(points) if points else np.empty((0, 3))


# ============================================================
# Equal-axis helper
# ============================================================

def set_axes_equal(ax):
    limits = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
    centers = limits.mean(axis=1)
    max_range = (limits[:, 1] - limits[:, 0]).max() / 2.0
    ax.set_xlim3d([centers[0] - max_range, centers[0] + max_range])
    ax.set_ylim3d([centers[1] - max_range, centers[1] + max_range])
    ax.set_zlim3d([centers[2] - max_range, centers[2] + max_range])


# ============================================================
# Per-file: find YZ-nearest, compute errors, plot & save
# ============================================================

def process_one_file(points: np.ndarray, target: np.ndarray,
                     title: str, save_path: str) -> dict:
    """Return dict with nearest point info and YZ error. Save 3-D plot."""

    # --- find nearest in YZ plane ---
    dists_yz = np.linalg.norm(points[:, 1:3] - target[1:3], axis=1)
    nearest_idx = np.argmin(dists_yz)
    nearest_pt = points[nearest_idx]
    dyz = dists_yz[nearest_idx]

    dx = nearest_pt[0] - target[0]
    nearest_dist = np.linalg.norm(nearest_pt - target)

    # --- plot ---
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(points[:, 0], points[:, 1], points[:, 2],
               c="steelblue", s=1, alpha=0.15, label="Unlabeled markers")

    ax.scatter(*target, c="red", s=200, marker="o", edgecolors="darkred",
               linewidths=1.5,
               label=f"Target ({target[0]:.1f}, {target[1]:.1f}, {target[2]:.1f})",
               zorder=5)

    ax.scatter(*nearest_pt, c="lime", s=150, marker="^", edgecolors="darkgreen",
               linewidths=1.5,
               label=f"Nearest YZ (d_yz={dyz:.1f}, d_3d={nearest_dist:.1f} mm)",
               zorder=5)

    ax.plot([target[0], nearest_pt[0]],
            [target[1], nearest_pt[1]],
            [target[2], nearest_pt[2]],
            "g--", linewidth=1.5, alpha=0.8)

    ax.set_xlabel("X (mm) - local")
    ax.set_ylabel("Y (mm) - local")
    ax.set_zlabel("Z (mm) - local")
    ax.set_title(f"{title}\nYZ Error = {dyz:.3f} mm")
    ax.legend(loc="upper left", fontsize=9)

    set_axes_equal(ax)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)

    return {
        "nearest_pt": nearest_pt,
        "total_dist": nearest_dist,
        "x_error": dx,
        "yz_error": dyz,
    }


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Per-file MoCap YZ Error analysis with 3-D plots."
    )
    parser.add_argument("input", type=str,
                        help="Path to a single .csv or a directory of .csv files.")
    parser.add_argument("--target", type=float, nargs=3, default=[0.0, 0.0, 0.0],
                        metavar=("X", "Y", "Z"),
                        help="Target position in local frame (mm). Default: 0 0 0")
    parser.add_argument("--scale", type=float, default=1.0,
                        help="Divide world coords by this before remapping. Default: 1.0")
    parser.add_argument("--outdir", type=str, default=None,
                        help="Output directory for plots & summary. "
                             "Default: <input>/results")
    args = parser.parse_args()
    scale = args.scale
    target = np.array(args.target)

    # --- resolve files ---
    if os.path.isdir(args.input):
        csv_files = sorted(glob.glob(os.path.join(args.input, "*.csv")))
    elif os.path.isfile(args.input):
        csv_files = [args.input]
    else:
        print(f"ERROR: {args.input} is not a valid file or directory.")
        sys.exit(1)

    if not csv_files:
        print(f"ERROR: no .csv files found in {args.input}")
        sys.exit(1)

    # --- resolve output dir ---
    if args.outdir is not None:
        outdir = args.outdir
    elif os.path.isdir(args.input):
        outdir = os.path.join(args.input, "results")
    else:
        outdir = os.path.join(os.path.dirname(args.input) or ".", "results")
    os.makedirs(outdir, exist_ok=True)

    print(f"Processing {len(csv_files)} file(s)")
    print(f"Target (local): {target} mm")
    print(f"Output dir    : {outdir}\n")

    # --- per-file loop ---
    results = []  # list of (filename, info_dict)

    for fp in csv_files:
        basename = os.path.basename(fp)
        stem = os.path.splitext(basename)[0]
        print(f"--- {basename} ---")

        pts = parse_mocap_csv(fp, scale)
        if pts.size == 0:
            print(f"  No valid data, skipped.\n")
            continue

        print(f"  Samples: {len(pts)}")
        img_path = os.path.join(outdir, f"{stem}.png")
        info = process_one_file(pts, target, title=basename, save_path=img_path)
        results.append((basename, info))

        print(f"  YZ Error : {info['yz_error']:.3f} mm")
        print(f"  X Error  : {info['x_error']:.3f} mm")
        print(f"  Total    : {info['total_dist']:.3f} mm")
        print(f"  Plot     : {img_path}\n")

    if not results:
        print("No valid data found in any file.")
        sys.exit(1)

    # ============================================================
    # Summary
    # ============================================================
    yz_errors = [r[1]["yz_error"] for r in results]
    mean_yz = np.mean(yz_errors)
    std_yz  = np.std(yz_errors)

    print("=" * 60)
    print(f"{'File':<40} {'YZ Error (mm)':>14}")
    print("-" * 60)
    for fname, info in results:
        print(f"{fname:<40} {info['yz_error']:>14.3f}")
    print("-" * 60)
    print(f"{'Mean YZ Error':<40} {mean_yz:>14.3f}")
    print(f"{'Std  YZ Error':<40} {std_yz:>14.3f}")
    print("=" * 60)

    # --- save summary CSV ---
    summary_path = os.path.join(outdir, "summary.csv")
    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["File", "X_Error_mm", "YZ_Error_mm", "Total_Dist_mm",
                          "Nearest_X", "Nearest_Y", "Nearest_Z"])
        for fname, info in results:
            pt = info["nearest_pt"]
            writer.writerow([fname,
                             f"{info['x_error']:.4f}",
                             f"{info['yz_error']:.4f}",
                             f"{info['total_dist']:.4f}",
                             f"{pt[0]:.4f}", f"{pt[1]:.4f}", f"{pt[2]:.4f}"])
        writer.writerow([])
        writer.writerow(["Mean YZ Error (mm)", f"{mean_yz:.4f}"])
        writer.writerow(["Std  YZ Error (mm)", f"{std_yz:.4f}"])
    print(f"\nSummary CSV -> {summary_path}")

    # --- bar chart of per-file YZ errors ---
    fig, ax = plt.subplots(figsize=(max(8, len(results) * 0.8), 5))
    x_pos = np.arange(len(results))
    bars = ax.bar(x_pos, yz_errors, color="steelblue", edgecolor="white")
    ax.axhline(mean_yz, color="red", linestyle="--", linewidth=1.5,
               label=f"Mean = {mean_yz:.3f} mm")

    ax.set_xticks(x_pos)
    ax.set_xticklabels([os.path.splitext(r[0])[0] for r in results],
                       rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("YZ Error (mm)")
    ax.set_title("Per-File YZ Error (YZ-Nearest Unlabeled Marker)")
    ax.legend()

    for bar, val in zip(bars, yz_errors):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{val:.2f}", ha="center", va="bottom", fontsize=7)

    plt.tight_layout()
    bar_path = os.path.join(outdir, "yz_error_summary.png")
    plt.savefig(bar_path, dpi=150)
    plt.close(fig)
    print(f"Bar chart   -> {bar_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()