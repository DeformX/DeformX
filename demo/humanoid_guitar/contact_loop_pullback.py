"""Chronological closeup and measured camera pullback for the full strum."""
import math

import numpy as np

from .contact_loop_replay import loop_replay_spec


PULLBACK_EDIT = "closeup-pullback"
FULLBODY_EDIT = "closeup-pullback-fullbody"
PULLBACK_EDITS = (PULLBACK_EDIT, FULLBODY_EDIT)
CHAPTERS = (
    ("closeup", "right", "SLOW MOTION"),
    ("pullback", "pullback", "CAMERA PULLBACK"),
    ("overview", "performance", "NORMAL SPEED"),
)
PULLBACK_FRAMES = 64


def pullback_spec(motion, mode=PULLBACK_EDIT):
    if mode not in PULLBACK_EDITS:
        raise ValueError("Unknown pullback edit")
    slow_stride = 4 if mode == FULLBODY_EDIT else 1
    legacy = loop_replay_spec(motion)
    first, last = legacy["normal_source_indices_inclusive"]
    pair_end = legacy["slow_source_indices_inclusive"][1]
    overview_start = pair_end + PULLBACK_FRAMES * slow_stride
    stroke = 3 if mode == FULLBODY_EDIT else 2
    start, end = (round(t * legacy["record_hz"])
                  for t in motion["loop_strum"]["strokes"][stroke]["window_s"])
    if (not first < pair_end < overview_start or not start <= overview_start < end
            or (pair_end - first) % slow_stride or (last - overview_start) % 16):
        raise ValueError("Pullback boundaries do not align with the slow and normal-speed sample grids")
    return dict(
        chapter_order=[chapter for chapter, _, _ in CHAPTERS],
        fps=legacy["fps"],
        record_hz=legacy["record_hz"],
        closeup_source_indices_inclusive=[first, pair_end - slow_stride],
        pullback_source_indices_inclusive=[pair_end, overview_start - slow_stride],
        overview_source_indices_inclusive=[overview_start, last],
        closeup_source_stride=slow_stride,
        pullback_source_stride=slow_stride,
        overview_source_stride=16,
        camera_interpolation="quintic-ease-with-increasing-look-distance",
    )


def build_pullback_edit(motion, mode=PULLBACK_EDIT):
    spec = pullback_spec(motion, mode)
    rows = []
    for chapter, view, caption in CHAPTERS:
        first, last = spec[f"{chapter}_source_indices_inclusive"]
        stride = spec[f"{chapter}_source_stride"]
        rows.extend(
            dict(trajectory_index=index, view=view, chapter=chapter, edit_caption=caption,
                 playback_speed=stride * spec["fps"] / spec["record_hz"], source_replay=False)
            for index in range(first, last + 1, stride)
        )
    return rows


def build_pullback_camera_path(rows, spec, calibration, placement):
    """Ease the target, viewing direction, distance, lens and closeup fill."""
    world_from_guitar = np.asarray(placement, dtype=float)
    close_eye = (world_from_guitar @ [*calibration["eye_guitar_local_m"], 1])[:3]
    close_target = (world_from_guitar @ [*calibration["target_guitar_local_m"], 1])[:3]
    wide_eye = np.asarray(calibration["performance_eye_world_m"], dtype=float)
    wide_target = np.asarray(calibration["performance_target_world_m"], dtype=float)
    close_ray = close_eye - close_target
    wide_ray = wide_eye - wide_target
    close_distance = np.linalg.norm(close_ray)
    wide_distance = np.linalg.norm(wide_ray)
    close_focal = float(calibration["focal_length_mm"])
    fill = float(calibration["fill"]["intensity"])
    if not 0 < close_distance < wide_distance or not 0 < close_focal or not fill >= 0:
        raise ValueError("Invalid closeup or overview calibration for the pullback")
    if not all(np.isfinite(value).all() for value in (close_ray, wide_ray, close_focal, fill)):
        raise ValueError("Nonfinite camera calibration")

    close_direction = close_ray / close_distance
    wide_direction = wide_ray / wide_distance
    first, last = spec["pullback_source_indices_inclusive"]
    path = []
    for row in rows:
        t = min(1., max(0., (row["trajectory_index"] - first) / (last - first)))
        mix = t**3 * (10 + t * (-15 + 6 * t))
        if mix == 0:
            eye, target = close_eye, close_target
        elif mix == 1:
            eye, target = wide_eye, wide_target
        else:
            target = close_target + mix * (wide_target - close_target)
            direction = close_direction + mix * (wide_direction - close_direction)
            direction_norm = np.linalg.norm(direction)
            if direction_norm < 1e-6:
                raise ValueError("Pullback passes between opposing camera directions")
            distance = close_distance + mix * (wide_distance - close_distance)
            eye = target + distance * direction / direction_norm
        path.append(dict(eye_world_m=eye.tolist(), target_world_m=target.tolist(),
                         focal_length_mm=close_focal + mix * (38. - close_focal),
                         macro_fill_intensity=fill * (1. - mix)))
    return path


def camera_world_matrix(camera):
    eye = np.asarray(camera["eye_world_m"], dtype=float)
    target = np.asarray(camera["target_world_m"], dtype=float)
    backward = eye - target
    backward /= np.linalg.norm(backward)
    right = np.cross([0., 0., 1.], backward)
    horizontal = np.linalg.norm(right)
    if horizontal < 1e-6:
        raise ValueError("Pullback aims along the camera up axis")
    right /= horizontal
    result = np.eye(4)
    result[:3, :3] = np.column_stack([right, np.cross(backward, right), backward])
    result[:3, 3] = eye
    return result


def verify_pullback_edit(motion, frames, spec, calibration, placement, mode=PULLBACK_EDIT):
    """Verify both the source schedule and native USD camera readbacks."""
    expected_spec = pullback_spec(motion, mode)
    rows = build_pullback_edit(motion, mode)
    if spec != expected_spec or len(frames) != len(rows):
        raise ValueError("Pullback specification or frame count changed")
    path = build_pullback_camera_path(rows, spec, calibration, placement)
    matrices, focal_lengths, fill_intensities, distances = [], [], [], []
    max_matrix_error = max_focal_error = max_fill_error = 0.
    for i, (frame, row, camera) in enumerate(zip(frames, rows, path, strict=True)):
        if (any(frame.get(key) != value for key, value in row.items())
                or frame["source_replay"] is not False or frame.get("phase") != "HOLD"
                or frame.get("video_frame") != i
                or not math.isclose(frame.get("video_time_s", math.nan), i / spec["fps"],
                                    rel_tol=0., abs_tol=1e-10)
                or not math.isclose(frame.get("simulation_time_s", math.nan),
                                    row["trajectory_index"] / spec["record_hz"],
                                    rel_tol=0., abs_tol=1e-10)):
            raise ValueError(f"Pullback source, speed, phase or label changed at frame {i}")
        readback = frame.get("camera", {})
        matrix = np.asarray(readback.get("world_from_camera"), dtype=float)
        focal = readback.get("focal_length_mm", math.nan)
        fill = readback.get("macro_fill_intensity", math.nan)
        if matrix.shape != (4, 4) or not np.isfinite(matrix).all() or not (
                math.isfinite(focal) and math.isfinite(fill)):
            raise ValueError(f"Invalid native pullback camera readback at frame {i}")
        matrix_error = float(np.max(np.abs(matrix - camera_world_matrix(camera))))
        focal_error = abs(focal - camera["focal_length_mm"])
        fill_error = abs(fill - camera["macro_fill_intensity"])
        if matrix_error > 2e-6 or focal_error > 5e-6 or fill_error > 5e-5:
            raise ValueError(f"Pullback camera, focal length or fill differs at frame {i}")
        max_matrix_error = max(max_matrix_error, matrix_error)
        max_focal_error = max(max_focal_error, focal_error)
        max_fill_error = max(max_fill_error, fill_error)
        matrices.append(matrix)
        focal_lengths.append(focal)
        fill_intensities.append(fill)
        distances.append(np.linalg.norm(matrix[:3, 3] - camera["target_world_m"]))

    matrices = np.asarray(matrices)
    transition = np.asarray([i for i, row in enumerate(rows) if row["chapter"] == "pullback"])
    if not np.all(np.diff(np.asarray(distances)[transition]) > 0):
        raise ValueError("Native camera does not move away from its aim point during the pullback")
    rotations = matrices[:, :3, :3]
    dots = (np.einsum("nij,nij->n", rotations[:-1], rotations[1:]) - 1) / 2
    return dict(
        frames=len(frames),
        transition_frames=len(transition),
        maximum_world_matrix_error=max_matrix_error,
        maximum_focal_length_error_mm=max_focal_error,
        maximum_macro_fill_intensity_error=max_fill_error,
        opening_look_distance_m=float(distances[0]),
        closing_look_distance_m=float(distances[-1]),
        maximum_adjacent_eye_step_m=float(np.linalg.norm(np.diff(matrices[:, :3, 3], axis=0), axis=1).max()),
        maximum_adjacent_rotation_deg=float(np.degrees(np.arccos(np.clip(dots, -1, 1))).max()),
        maximum_adjacent_focal_step_mm=float(np.abs(np.diff(focal_lengths)).max()),
        maximum_adjacent_macro_fill_step=float(np.abs(np.diff(fill_intensities)).max()),
    )
