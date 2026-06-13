import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation


def plot_frame_all_data(data, frame_idx, figsize_per_plot=(8, 2.8)):
    keys = list(data.files) if hasattr(data, "files") else list(data.keys())
    n_keys = len(keys)
    if n_keys == 0:
        raise ValueError("No arrays found to plot.")
    target_arr = None
    if "target_pos" in keys:
        tp = np.asarray(data["target_pos"])
        if tp.ndim == 2 and tp.shape[1] >= 2:
            target_arr = tp
    elif "pos" in keys:
        tp = np.asarray(data["pos"])
        if tp.ndim == 2 and tp.shape[1] >= 2:
            target_arr = tp

    fig, axes = plt.subplots(n_keys, 1, figsize=(figsize_per_plot[0], figsize_per_plot[1] * n_keys))
    if n_keys == 1:
        axes = [axes]

    for i, (ax, key) in enumerate(zip(axes, keys)):
        arr = np.asarray(data[key])
        ax.set_title(f"{key} (frame={frame_idx})")

        if arr.ndim == 0:
            ax.text(0.5, 0.5, f"{arr.item():.6g}", ha="center", va="center")
            ax.set_axis_off()
            continue

        if frame_idx < 0 or frame_idx >= arr.shape[0]:
            ax.text(0.5, 0.5, f"frame out of range [0, {arr.shape[0]-1}]", ha="center", va="center")
            ax.set_axis_off()
            continue

        frame = arr[frame_idx]
        if frame.ndim == 0:
            ax.axhline(float(frame), lw=1.25)
            ax.set_xlim(0, 1)
        elif frame.ndim == 1:
            ax.plot(frame, lw=1.0)
        elif frame.ndim == 2 and frame.shape[1] >= 3:
            sort_idx = np.argsort(frame[:, 0])
            frame_sorted = frame[sort_idx]
            subplotspec = ax.get_subplotspec()
            ax.remove()
            ax = fig.add_subplot(subplotspec, projection="3d")
            axes[i] = ax
            ax.set_title(f"{key} (frame={frame_idx})")
            ax.plot(frame_sorted[:, 0], frame_sorted[:, 1], frame_sorted[:, 2], marker="o", ms=2, lw=0.9)
            if key == "track_pos" and target_arr is not None and frame_idx < target_arr.shape[0] and target_arr.shape[1] >= 3:
                target_point = target_arr[frame_idx]
                ax.scatter(
                    target_point[0],
                    target_point[1],
                    target_point[2],
                    c="red",
                    s=45,
                    marker="*",
                    label="target_pos",
                )
                ax.legend(loc="best")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("z")
        elif frame.ndim == 2 and frame.shape[1] >= 2:
            sort_idx = np.argsort(frame[:, 0])
            frame_sorted = frame[sort_idx]
            ax.plot(frame_sorted[:, 0], frame_sorted[:, 1], marker="o", ms=2, lw=0.9)
            if key == "track_pos" and target_arr is not None and frame_idx < target_arr.shape[0]:
                target_point = target_arr[frame_idx]
                ax.scatter(target_point[0], target_point[1], c="red", s=45, marker="*", label="target_pos")
                ax.legend(loc="best")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
        else:
            flat = frame.reshape(frame.shape[0], -1) if frame.ndim > 1 else frame[None, :]
            ax.imshow(flat, aspect="auto", interpolation="nearest")
            ax.set_xlabel("features")
            ax.set_ylabel("rows")

    fig.tight_layout()
    return fig, axes


def _normalize_xyz_array(arr: np.ndarray | None, name: str) -> np.ndarray | None:
    if arr is None:
        return None
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim != 3 or arr.shape[2] < 3:
        raise ValueError(f"`{name}` must have shape (T,N,3+), got {arr.shape}")
    return arr[:, :, :3]


def _axis_bounds_xyz(*arrays: np.ndarray | None, margin_ratio: float = 0.06):
    mins = []
    maxs = []
    for arr in arrays:
        if arr is None or arr.size == 0:
            continue
        mins.append(np.min(arr.reshape(-1, 3), axis=0))
        maxs.append(np.max(arr.reshape(-1, 3), axis=0))
    if not mins:
        return np.array([-1.0, -1.0, -1.0]), np.array([1.0, 1.0, 1.0])
    min_xyz = np.min(np.vstack(mins), axis=0)
    max_xyz = np.max(np.vstack(maxs), axis=0)
    span = np.maximum(max_xyz - min_xyz, 1.0e-6)
    pad = span * float(margin_ratio)
    return min_xyz - pad, max_xyz + pad


def parse_index_pairs(spec: str) -> list[tuple[int, int]]:
    """Parse mapping like '0:3, 1:4' into [(0,3), (1,4)]."""
    pairs: list[tuple[int, int]] = []
    spec = str(spec).strip()
    if not spec:
        return pairs
    for token in spec.split(","):
        item = token.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"Invalid pair '{item}'. Expected format 'pos_idx:track_idx'.")
        pos_s, track_s = item.split(":", 1)
        pairs.append((int(pos_s.strip()), int(track_s.strip())))
    if not pairs:
        raise ValueError("No valid index pairs parsed from mapping string.")
    return pairs


def compute_positions_track_loss(
    positions: np.ndarray,
    track_pos: np.ndarray,
    index_pairs: list[tuple[int, int]],
    metric: str = "mse",
):
    """
    Compute mapped error between `positions` and `track_pos`:
      err_per_frame = sum_k ||positions[t,p_k] - track_pos[t,q_k]|| / 21

    Args:
      positions: array with shape (T, Np, 3+)
      track_pos: array with shape (T, Nt, 3+)
      index_pairs: list of (positions_index, track_pos_index)
      metric: deprecated (kept for CLI compatibility)

    Returns:
      dict with scalar and breakdown errors:
        - loss                         mean(err_per_frame)
        - loss_per_frame               shape (T,)  (err_per_frame)
        - loss_per_pair                shape (K,)  (mean pair contribution / 21)
        - loss_per_frame_per_pair      shape (T, K) (pair contribution / 21)
        - pairs                        shape (K, 2)
    """
    pos = _normalize_xyz_array(positions, "positions")
    trk = _normalize_xyz_array(track_pos, "track_pos")
    if pos is None or trk is None:
        raise ValueError("Both `positions` and `track_pos` are required.")
    if not index_pairs:
        raise ValueError("`index_pairs` must not be empty.")

    t_steps = min(int(pos.shape[0]), int(trk.shape[0]))
    if t_steps <= 0:
        raise ValueError("No overlapping frames to compute loss.")
    pos = pos[:t_steps]
    trk = trk[:t_steps]

    pairs = np.asarray(index_pairs, dtype=np.int64).reshape(-1, 2)
    pos_idx = pairs[:, 0]
    trk_idx = pairs[:, 1]

    if np.any(pos_idx < 0) or np.any(pos_idx >= pos.shape[1]):
        raise IndexError(f"positions indices out of range [0, {pos.shape[1]-1}]: {pos_idx.tolist()}")
    if np.any(trk_idx < 0) or np.any(trk_idx >= trk.shape[1]):
        raise IndexError(f"track_pos indices out of range [0, {trk.shape[1]-1}]: {trk_idx.tolist()}")

    _ = metric
    diff = pos[:, pos_idx, :] - trk[:, trk_idx, :]  # (T, K, 3)
    dist_frame_pair = np.linalg.norm(diff, axis=2)  # (T, K)
    loss_frame_pair = dist_frame_pair / 21.0
    loss_per_frame = np.sum(loss_frame_pair, axis=1)
    loss_per_pair = np.mean(loss_frame_pair, axis=0)
    loss = float(np.mean(loss_per_frame))
    return {
        "loss": loss,
        "loss_per_frame": loss_per_frame,
        "loss_per_pair": loss_per_pair,
        "loss_per_frame_per_pair": loss_frame_pair,
        "pairs": pairs,
    }


def export_mapped_loss_csv(
    loss_info: dict,
    output_path: str,
    time_arr: np.ndarray | None = None,
):
    """
    Export mapped error to CSV.

    Columns:
      frame, [time], err_per_frame, err_pair_pos{p}_trk{t}, ...
    """
    pairs = np.asarray(loss_info["pairs"], dtype=np.int64)
    loss_per_frame = np.asarray(loss_info["loss_per_frame"], dtype=np.float64).reshape(-1)
    loss_per_frame_per_pair = np.asarray(loss_info["loss_per_frame_per_pair"], dtype=np.float64)
    if loss_per_frame_per_pair.ndim != 2 or loss_per_frame_per_pair.shape[0] != loss_per_frame.size:
        raise ValueError(
            "Invalid loss_per_frame_per_pair shape: "
            f"{loss_per_frame_per_pair.shape}, expected ({loss_per_frame.size}, K)."
        )

    columns = [np.arange(loss_per_frame.size, dtype=np.int64)]
    header_parts = ["frame"]

    if time_arr is not None:
        t = np.asarray(time_arr, dtype=np.float64).reshape(-1)
        n = min(loss_per_frame.size, t.size)
        if n <= 0:
            raise ValueError("time array has no overlap with loss frames.")
        loss_per_frame = loss_per_frame[:n]
        loss_per_frame_per_pair = loss_per_frame_per_pair[:n]
        columns[0] = columns[0][:n]
        columns.append(t[:n])
        header_parts.append("time")

    columns.append(loss_per_frame)
    header_parts.append("err_per_frame")

    for k in range(pairs.shape[0]):
        p_idx = int(pairs[k, 0])
        t_idx = int(pairs[k, 1])
        columns.append(loss_per_frame_per_pair[:, k])
        header_parts.append(f"err_pair_pos{p_idx}_trk{t_idx}")

    out = np.column_stack(columns)
    out_path = Path(output_path).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        out_path,
        out,
        delimiter=",",
        header=",".join(header_parts),
        comments="",
        fmt="%.10f",
    )
    print(f"[Save] mapped-err csv: {out_path}")
    return out_path


def compute_positions_track_distance_sum(
    positions: np.ndarray,
    track_pos: np.ndarray,
    index_pairs: list[tuple[int, int]],
):
    """
    Compute mapped Euclidean distances and their per-frame sum.

    Returns:
      dict with:
        - sum_distance_per_frame          shape (T,)
        - avg_distance_per_frame          shape (T,), sum_distance_per_frame / 21
        - distance_per_frame_per_pair     shape (T, K)
        - total_distance_sum              scalar (sum over all frames)
        - mean_distance_sum               scalar (mean per frame)
        - mean_avg_distance               scalar (mean avg distance per frame)
        - pairs                           shape (K, 2)
    """
    pos = _normalize_xyz_array(positions, "positions")
    trk = _normalize_xyz_array(track_pos, "track_pos")
    if pos is None or trk is None:
        raise ValueError("Both `positions` and `track_pos` are required.")
    if not index_pairs:
        raise ValueError("`index_pairs` must not be empty.")

    t_steps = min(int(pos.shape[0]), int(trk.shape[0]))
    if t_steps <= 0:
        raise ValueError("No overlapping frames to compute distance sum.")
    pos = pos[:t_steps]
    trk = trk[:t_steps]

    pairs = np.asarray(index_pairs, dtype=np.int64).reshape(-1, 2)
    pos_idx = pairs[:, 0]
    trk_idx = pairs[:, 1]
    if np.any(pos_idx < 0) or np.any(pos_idx >= pos.shape[1]):
        raise IndexError(f"positions indices out of range [0, {pos.shape[1]-1}]: {pos_idx.tolist()}")
    if np.any(trk_idx < 0) or np.any(trk_idx >= trk.shape[1]):
        raise IndexError(f"track_pos indices out of range [0, {trk.shape[1]-1}]: {trk_idx.tolist()}")

    diff = pos[:, pos_idx, :] - trk[:, trk_idx, :]  # (T, K, 3)
    dist_frame_pair = np.linalg.norm(diff, axis=2)  # (T, K)
    sum_distance_per_frame = np.sum(dist_frame_pair, axis=1)
    avg_distance_per_frame = sum_distance_per_frame / 21.0
    total_distance_sum = float(np.sum(sum_distance_per_frame))
    mean_distance_sum = float(np.mean(sum_distance_per_frame))
    mean_avg_distance = float(np.mean(avg_distance_per_frame))
    return {
        "sum_distance_per_frame": sum_distance_per_frame,
        "avg_distance_per_frame": avg_distance_per_frame,
        "distance_per_frame_per_pair": dist_frame_pair,
        "total_distance_sum": total_distance_sum,
        "mean_distance_sum": mean_distance_sum,
        "mean_avg_distance": mean_avg_distance,
        "pairs": pairs,
    }


def plot_sum_distance_figure(
    sum_distance_per_frame: np.ndarray,
    time_arr: np.ndarray | None = None,
    output_path: str | None = None,
    title: str = "Sum of Mapped Distances per Frame",
):
    y = np.asarray(sum_distance_per_frame, dtype=np.float64).reshape(-1)
    if y.size == 0:
        raise ValueError("sum_distance_per_frame is empty.")

    if time_arr is not None:
        t = np.asarray(time_arr, dtype=np.float64).reshape(-1)
        n = min(y.size, t.size)
        x = t[:n]
        y = y[:n]
        x_label = "time [s]"
    else:
        x = np.arange(y.size, dtype=np.int64)
        x_label = "frame"

    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    ax.plot(x, y, lw=1.4, c="#0b6a98")
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel("average distance [m]")
    ax.grid(alpha=0.35)
    ax.text(
        0.01,
        0.98,
        f"mean={float(np.mean(y)):.6f} m\nmax={float(np.max(y)):.6f} m",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="#cccccc"),
    )
    fig.tight_layout()

    out_path = None
    if output_path is not None:
        out_path = Path(output_path).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=180)
        print(f"[Save] sum-distance plot: {out_path}")
    plt.close(fig)
    return out_path


def make_track_positions_video(
    npz_path: str,
    output_path: str | None = None,
    fps: int = 30,
    stride: int = 8,
    max_frames: int = 0,
    dpi: int = 140,
    draw_lines: bool = True,
    ignore_track_indices: list[int] | None = None,
):
    data = np.load(npz_path, allow_pickle=True)
    keys = list(data.files)

    positions = _normalize_xyz_array(data["positions"], "positions") if "positions" in keys else None
    track_pos = _normalize_xyz_array(data["track_pos"], "track_pos") if "track_pos" in keys else None
    if track_pos is not None and ignore_track_indices:
        n_track = track_pos.shape[1]
        valid_ignore = sorted({int(i) for i in ignore_track_indices if 0 <= int(i) < n_track})
        if valid_ignore:
            keep_mask = np.ones((n_track,), dtype=bool)
            keep_mask[valid_ignore] = False
            track_pos = track_pos[:, keep_mask, :]
            print(f"[Info] ignoring track_pos indices: {valid_ignore}")

    target_pos = None
    if "target_pos" in keys:
        tp = np.asarray(data["target_pos"], dtype=np.float64)
        if tp.ndim == 2 and tp.shape[1] >= 3:
            target_pos = tp[:, :3]

    if positions is None and track_pos is None:
        raise ValueError("Need at least one of `positions` or `track_pos` in the NPZ.")

    lengths = []
    if positions is not None:
        lengths.append(positions.shape[0])
    if track_pos is not None:
        lengths.append(track_pos.shape[0])
    if target_pos is not None:
        lengths.append(target_pos.shape[0])
    n_steps = int(min(lengths))
    if n_steps <= 0:
        raise ValueError("No frames available to animate.")

    time_arr = None
    if "time" in keys:
        t = np.asarray(data["time"], dtype=np.float64).reshape(-1)
        if t.size >= n_steps:
            time_arr = t[:n_steps]

    frame_ids = np.arange(0, n_steps, max(1, int(stride)), dtype=int)
    if max_frames > 0:
        frame_ids = frame_ids[: int(max_frames)]
    if frame_ids.size == 0:
        raise ValueError("No frames selected; check --stride/--max_frames.")

    in_path = Path(npz_path).expanduser().resolve()
    if output_path is None:
        output_path = str(in_path.with_name(in_path.stem + "_track_positions.mp4"))
    out_path = Path(output_path).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(9.5, 7.2))
    ax = fig.add_subplot(111, projection="3d")
    min_xyz, max_xyz = _axis_bounds_xyz(positions, track_pos, target_pos[:, None, :] if target_pos is not None else None)
    ax.set_xlim(float(min_xyz[0]), float(max_xyz[0]))
    ax.set_ylim(float(min_xyz[1]), float(max_xyz[1]))
    ax.set_zlim(float(min_xyz[2]), float(max_xyz[2]))
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title("Demo1C: track_pos and positions")
    ax.set_box_aspect(np.maximum(max_xyz - min_xyz, 1.0e-6))
    ax.view_init(elev=20, azim=-62)

    artists = []

    pos_scatter = None
    pos_line = None
    if positions is not None:
        pos_scatter = ax.scatter([], [], [], s=16, c="#1f77b4", label="positions", depthshade=False)
        artists.append(pos_scatter)
        if draw_lines:
            pos_line, = ax.plot([], [], [], lw=1.0, c="#1f77b4", alpha=0.85)
            artists.append(pos_line)

    track_scatter = None
    track_line = None
    if track_pos is not None:
        track_scatter = ax.scatter([], [], [], s=14, c="#d62728", label="track_pos", depthshade=False)
        artists.append(track_scatter)
        if draw_lines:
            track_line, = ax.plot([], [], [], lw=0.9, c="#d62728", alpha=0.55)
            artists.append(track_line)

    target_scatter = None
    if target_pos is not None:
        target_scatter = ax.scatter([], [], [], s=100, c="#111111", marker="*", label="target_pos", depthshade=False)
        artists.append(target_scatter)

    ax.legend(loc="upper left")
    info_text = ax.text2D(0.02, 0.98, "", transform=ax.transAxes, va="top", ha="left")
    artists.append(info_text)

    def _set_points(artist, pts):
        artist._offsets3d = (pts[:, 0], pts[:, 1], pts[:, 2])

    def _set_line(line_artist, pts):
        line_artist.set_data(pts[:, 0], pts[:, 1])
        line_artist.set_3d_properties(pts[:, 2])

    def update(frame_i):
        src_idx = int(frame_ids[frame_i])
        if positions is not None:
            p = positions[src_idx]
            _set_points(pos_scatter, p)
            if pos_line is not None:
                _set_line(pos_line, p)
        if track_pos is not None:
            tp = track_pos[src_idx]
            _set_points(track_scatter, tp)
            if track_line is not None:
                _set_line(track_line, tp)
        if target_pos is not None:
            t = target_pos[src_idx]
            _set_points(target_scatter, t.reshape(1, 3))

        if time_arr is not None:
            info_text.set_text(f"frame={src_idx}  t={time_arr[src_idx]:.3f}s")
        else:
            info_text.set_text(f"frame={src_idx}")
        return artists

    anim = animation.FuncAnimation(
        fig,
        update,
        frames=frame_ids.size,
        interval=1000.0 / float(max(1, fps)),
        blit=False,
    )

    ext = out_path.suffix.lower()
    if ext == ".gif":
        writer = animation.PillowWriter(fps=max(1, int(fps)))
    else:
        if animation.writers.is_available("ffmpeg"):
            writer = animation.FFMpegWriter(
                fps=max(1, int(fps)),
                codec="libx264",
                bitrate=5000,
                metadata={"title": "Demo1C track_pos and positions"},
            )
        else:
            out_path = out_path.with_suffix(".gif")
            writer = animation.PillowWriter(fps=max(1, int(fps)))
            print("[Info] ffmpeg not available. Saving GIF instead:", out_path)

    print(f"[Save] Writing video with {frame_ids.size} frames to: {out_path}")
    anim.save(str(out_path), writer=writer, dpi=max(80, int(dpi)))
    plt.close(fig)
    return out_path


def _build_arg_parser():
    parser = argparse.ArgumentParser(description="Plot and animate Demo1C NPZ data.")
    parser.add_argument(
        "--npz",
        type=str,
        default=str(Path(__file__).resolve().parents[2] / "visualization_scripts" / "data" / "rope_demo_02212026_1.2Hz_18deg_240Hz_wire_joint_positions.npz"),
        help="Input NPZ path.",
    )
    parser.add_argument(
        "--frame",
        type=int,
        default=1500,
        help="Frame index for single-frame plotting mode.",
    )
    parser.add_argument(
        "--video",
        action="store_true",
        help="Export animation video showing track_pos and positions movement.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output video path (.mp4 recommended, .gif also supported).",
    )
    parser.add_argument("--fps", type=int, default=30, help="Output video FPS.")
    parser.add_argument("--stride", type=int, default=8, help="Use every Nth frame.")
    parser.add_argument(
        "--max_frames",
        type=int,
        default=0,
        help="Limit number of animation frames (0 means no limit).",
    )
    parser.add_argument("--dpi", type=int, default=140, help="Video rendering DPI.")
    parser.add_argument(
        "--no_lines",
        action="store_true",
        help="Show moving points only (disable polyline overlays).",
    )
    parser.add_argument(
        "--ignore_track_indices",
        type=str,
        default="18,20",
        help="Comma-separated track_pos point indices to ignore in video (e.g. '18,20').",
    )
    parser.add_argument(
        "--loss_map",
        type=str,
        default="1:19,3:17,5:16,7:15,9:14,11:13,13:12,15:11,17:10,19:9,21:8,23:7,25:6,27:5,29:4,31:3,33:2,35:1,37:0",
        help="Mapped loss pairs in format 'pos_idx:track_idx,...' (e.g. '0:0,1:2').",
    )
    parser.add_argument(
        "--loss_metric",
        type=str,
        default="mse",
        choices=["mse", "rmse", "mae"],
        help="Metric used by mapped loss.",
    )
    parser.add_argument(
        "--plot_sum_distance",
        action="store_true",
        help="Compute mapped sum of Euclidean distance and save a figure.",
    )
    parser.add_argument(
        "--sum_distance_output",
        type=str,
        default=None,
        help="Output image path for sum-distance plot (default: <npz>_sum_distance.png).",
    )
    parser.add_argument(
        "--loss_csv_output",
        type=str,
        default=None,
        help="Output CSV path for mapped loss per frame and per pair.",
    )
    return parser


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    data = np.load(args.npz, allow_pickle=True)
    print(data.files)
    for key in data.files:
        print(key, data[key].shape)

    pairs = []
    loss_info = None
    if args.loss_map.strip():
        if "positions" not in data.files or "track_pos" not in data.files:
            raise ValueError("Need `positions` and `track_pos` in NPZ to compute mapped error.")
        pairs = parse_index_pairs(args.loss_map)
        loss_info = compute_positions_track_loss(
            positions=data["positions"],
            track_pos=data["track_pos"],
            index_pairs=pairs,
            metric=args.loss_metric,
        )
        print(
            f"[Err] pairs={pairs} "
            f"frames={loss_info['loss_per_frame'].shape[0]} mean_err={loss_info['loss']:.8f}"
        )
        for i, pair_loss in enumerate(loss_info["loss_per_pair"]):
            pos_i, trk_i = pairs[i]
            print(f"[ErrPair] positions[{pos_i}] vs track_pos[{trk_i}] = {float(pair_loss):.8f}")
        if args.loss_csv_output is not None:
            time_arr = np.asarray(data["time"], dtype=np.float64).reshape(-1) if "time" in data.files else None
            export_mapped_loss_csv(
                loss_info=loss_info,
                output_path=args.loss_csv_output,
                time_arr=time_arr,
            )

    if args.plot_sum_distance:
        if "positions" not in data.files or "track_pos" not in data.files:
            raise ValueError("Need `positions` and `track_pos` in NPZ to compute sum distance.")
        if not pairs:
            raise ValueError("Provide `--loss_map` with index pairs, e.g. --loss_map '0:0,1:2'.")
        dist_info = compute_positions_track_distance_sum(
            positions=data["positions"],
            track_pos=data["track_pos"],
            index_pairs=pairs,
        )
        time_arr = np.asarray(data["time"], dtype=np.float64).reshape(-1) if "time" in data.files else None
        out_plot = args.sum_distance_output
        if out_plot is None:
            npz_path = Path(args.npz).expanduser().resolve()
            out_plot = str(npz_path.with_name(npz_path.stem + "_sum_distance.png"))
        plot_sum_distance_figure(
            sum_distance_per_frame=dist_info["avg_distance_per_frame"],
            time_arr=time_arr,
            output_path=out_plot,
            title=f"Average Distance per Frame (sum/21, {len(pairs)} mapped pairs)",
        )
        print(
            f"[SumDistance] pairs={pairs} frames={dist_info['sum_distance_per_frame'].shape[0]} "
            f"total={dist_info['total_distance_sum']:.8f} mean_sum={dist_info['mean_distance_sum']:.8f} "
            f"mean_err={dist_info['mean_avg_distance']:.8f}"
        )

    if args.video:
        ignore_track_indices = []
        if args.ignore_track_indices.strip():
            ignore_track_indices = [int(x.strip()) for x in args.ignore_track_indices.split(",") if x.strip()]
        out_path = make_track_positions_video(
            npz_path=args.npz,
            output_path=args.output,
            fps=args.fps,
            stride=args.stride,
            max_frames=args.max_frames,
            dpi=args.dpi,
            draw_lines=not args.no_lines,
            ignore_track_indices=ignore_track_indices,
        )
        print(f"[Done] saved: {out_path}")
    elif not args.plot_sum_distance:
        plot_frame_all_data(data, frame_idx=args.frame)
        plt.show()
