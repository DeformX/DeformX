"""
MoCap CSV -> Local Coordinate Transform, Nearest-Point Search & 3-D Visualisation

Coordinate transform: world (x, y, z) -> local (y, -z, -x)
No rotation matrix or translation needed — pure axis remapping.

Usage:
  python mocap_transform_viz.py <csv_dir_or_file> [--target X Y Z]
"""

import argparse
import csv
import glob
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


# ============================================================
# Coordinate-system transform  (world -> local)
# local = (y_world, -z_world, -x_world)
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

    print(f"  {os.path.basename(filepath)}: found {len(unlabeled_xyz_groups)} "
          f"Unlabeled marker group(s)")

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
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Transform MoCap Unlabeled markers to local frame, "
                    "find nearest in YZ plane to target, and visualise."
    )
    parser.add_argument("input", type=str,
                        help="Path to a single .csv or a directory of .csv files.")
    parser.add_argument("--target", type=float, nargs=3, default=[0.0, 0.0, 0.0],
                        metavar=("X", "Y", "Z"),
                        help="Target position in local frame (mm). Default: 0 0 0")
    parser.add_argument("--scale", type=float, default=1.0,
                        help="World coordinates are divided by this value before "
                             "axis remapping. Default: 1.0 (no scaling)")
    args = parser.parse_args()
    scale = args.scale

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

    print(f"Processing {len(csv_files)} file(s) ...\n")

    all_points = []
    file_labels = []
    for fp in csv_files:
        pts = parse_mocap_csv(fp, scale)
        if pts.size > 0:
            all_points.append(pts)
            file_labels.extend([os.path.basename(fp)] * len(pts))

    if not all_points:
        print("No valid Unlabeled marker data found across all files.")
        sys.exit(1)

    all_points = np.vstack(all_points)
    print(f"\nTotal transformed Unlabeled marker samples: {len(all_points)}")

    target = np.array(args.target)

    # --- Find nearest point in YZ plane ---
    dists_yz = np.linalg.norm(all_points[:, 1:3] - target[1:3], axis=1)
    nearest_idx = np.argmin(dists_yz)
    nearest_pt = all_points[nearest_idx]
    nearest_dist_yz = dists_yz[nearest_idx]
    nearest_dist_3d = np.linalg.norm(nearest_pt - target)
    nearest_file = file_labels[nearest_idx]

    dx = nearest_pt[0] - target[0]

    print(f"\nTarget position (local frame): {target}  (mm)")
    print(f"Nearest Unlabeled marker (by YZ distance):")
    print(f"  Position      : ({nearest_pt[0]:.3f}, {nearest_pt[1]:.3f}, {nearest_pt[2]:.3f}) mm")
    print(f"  YZ distance   : {nearest_dist_yz:.3f} mm")
    print(f"  X distance    : {dx:.3f} mm  (signed)")
    print(f"  3D distance   : {nearest_dist_3d:.3f} mm")
    print(f"  From file     : {nearest_file}")

    # --- 3-D Visualisation ---
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(all_points[:, 0], all_points[:, 1], all_points[:, 2],
               c="steelblue", s=1, alpha=0.15, label="Unlabeled markers")

    ax.scatter(*target, c="red", s=200, marker="o", edgecolors="darkred",
               linewidths=1.5,
               label=f"Target ({target[0]:.1f}, {target[1]:.1f}, {target[2]:.1f})",
               zorder=5)

    ax.scatter(*nearest_pt, c="lime", s=150, marker="^", edgecolors="darkgreen",
               linewidths=1.5, label=f"Nearest YZ (d_yz={nearest_dist_yz:.1f} mm)", zorder=5)

    ax.plot([target[0], nearest_pt[0]],
            [target[1], nearest_pt[1]],
            [target[2], nearest_pt[2]],
            "g--", linewidth=1.5, alpha=0.8)

    ax.set_xlabel("X (mm) - local")
    ax.set_ylabel("Y (mm) - local")
    ax.set_zlabel("Z (mm) - local")
    ax.set_title("MoCap Unlabeled Markers in Local Frame (Nearest by YZ)")
    ax.legend(loc="upper left", fontsize=9)

    set_axes_equal(ax)
    plt.tight_layout()
    # plt.savefig("mocap_viz.png", dpi=150)
    # print("\nSaved visualisation -> mocap_viz.png")
    plt.show()


if __name__ == "__main__":
    main()