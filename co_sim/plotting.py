"""Plotting and rendering helpers for co-simulation outputs."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from render_scripts import post_processing as pp

from .models import CoSimConfig


def _finalize_and_save(fig: plt.Figure, output_path: Path) -> None:
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_force_vector_with_magnitude(
    time: np.ndarray,
    mean_force: np.ndarray,
    force_mag: np.ndarray,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(time, mean_force[:, 0], label="Fx")
    ax.plot(time, mean_force[:, 1], label="Fy")
    ax.plot(time, mean_force[:, 2], label="Fz")
    ax.plot(time, force_mag, "k--", linewidth=2.0, label="|F|")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("mean force [N]")
    ax.set_title("Mean Rod-on-Frame Force Vector (Real Simulation Time)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    _finalize_and_save(fig, output_path)


def plot_force_vs_time(
    time: np.ndarray,
    force_mag: np.ndarray,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(time, force_mag, color="#1f77b4", linewidth=2.0)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("|F| [N]")
    ax.set_title("Rod-on-Frame Force Magnitude vs Time")
    ax.grid(True, alpha=0.3)
    _finalize_and_save(fig, output_path)


def render_multiview_video(
    sampled_rod_pos: np.ndarray,
    sampled_frame_pos: np.ndarray,
    sampled_frame_dir: np.ndarray,
    sampled_time: np.ndarray,
    sampled_mean_force: np.ndarray,
    cfg: CoSimConfig,
    video_path: Path,
    render_fps: int | None,
    render_speed: float,
    force_vector_scale: float,
) -> None:
    frame_span = 0.05
    frame_line = sampled_frame_pos + frame_span * np.stack(
        [-sampled_frame_dir[:, 2, :, 0], sampled_frame_dir[:, 2, :, 0]],
        axis=-1,
    )
    n_nodes = sampled_rod_pos.shape[2]
    pad_len = max(0, n_nodes - frame_line.shape[2])
    frame_line = np.pad(frame_line, ((0, 0), (0, 0), (0, pad_len)), constant_values=np.nan)

    rod_for_plot = np.concatenate([sampled_rod_pos[:, None, ...], frame_line[:, None, ...]], axis=1)

    joint_center = sampled_rod_pos[:, :, 0].mean(axis=0)
    x_window = max(0.25 * cfg.base_length, 0.2)
    y_window = 0.25
    z_window = 0.2
    bounds = (
        (float(joint_center[0] - x_window / 2), float(joint_center[0] + x_window / 2)),
        (float(joint_center[1] - y_window / 2), float(joint_center[1] + y_window / 2)),
        (float(joint_center[2] - z_window / 2), float(joint_center[2] + z_window / 2)),
    )

    pp.plot_rods_multiview(
        rod_for_plot,
        video_path=video_path,
        times=sampled_time,
        fps=render_fps,
        speed=render_speed,
        plane_z=None,
        colors=["#ff7f0e", "#1f77b4"],
        bounds=bounds,
        force_origins=sampled_frame_pos[:, :, 0],
        force_vectors=sampled_mean_force,
        force_scale=force_vector_scale,
        force_color="#d62728",
        force_label="Mean Rod->Frame Force",
        show_force_magnitude=True,
    )
