"""Verify full-body framing by projecting the rendered USD subjects' bounds."""

import numpy as np


SUBJECT_PATHS = {
    "robot": "/World/Robot",
    "head": "/World/Robot/Robot/head_link",
    "left_foot": "/World/Robot/Robot/left_ankle_roll_link",
    "right_foot": "/World/Robot/Robot/right_ankle_roll_link",
    "guitar": "/World/Guitar",
    "stool": "/World/Stool",
}
EDGES = ("left", "top", "right", "bottom")
MINIMUM_MARGIN_FRACTION = 0.035


def verify_fullbody_framing(frames, resolution):
    width, height = resolution
    if not isinstance(width, int) or not isinstance(height, int) or min(width, height) <= 0:
        raise ValueError("Invalid full-body image resolution")
    overview = [row for row in frames if row["chapter"] == "overview"]
    if not overview:
        raise ValueError("Missing full-body overview frames")
    minimum = {name: np.full(4, np.inf) for name in SUBJECT_PATHS}
    required = MINIMUM_MARGIN_FRACTION * np.array([width, height, width, height])

    for row in overview:
        evidence = row.get("framing", {})
        bounds = evidence.get("world_bound_corners_m", {})
        camera = np.asarray(row.get("camera", {}).get("world_from_camera"), dtype=float)
        intrinsics = np.asarray(evidence.get("intrinsics_px"), dtype=float)
        clip = np.asarray(evidence.get("clip_near_far_m"), dtype=float)
        if (set(bounds) != set(SUBJECT_PATHS) or camera.shape != (4, 4)
                or intrinsics.shape != (3, 3) or clip.shape != (2,)
                or not all(np.isfinite(value).all() for value in (camera, intrinsics, clip))
                or not 0 < clip[0] < clip[1]
                or min(intrinsics[0, 0], intrinsics[1, 1]) <= 0
                or not np.allclose(intrinsics[:2, 2], [width / 2, height / 2], rtol=0, atol=1e-5)):
            raise ValueError("Incomplete or invalid native full-body framing evidence")

        for name, corners in bounds.items():
            points = np.asarray(corners, dtype=float)
            if points.shape != (8, 3) or not np.isfinite(points).all():
                raise ValueError(f"Invalid native bound for {name}")
            local = (points - camera[:3, 3]) @ camera[:3, :3]
            depth = -local[:, 2]  # USD cameras look along local -Z.
            if np.any((depth <= clip[0]) | (depth >= clip[1])):
                raise ValueError(f"Full-body camera clips {name} in depth")
            x = intrinsics[0, 2] + intrinsics[0, 0] * local[:, 0] / depth
            y = intrinsics[1, 2] - intrinsics[1, 1] * local[:, 1] / depth
            margin = np.array([x.min(), y.min(), width - x.max(), height - y.max()])
            if np.any(margin < required):
                raise ValueError(f"Full-body margin too small for {name}: {margin.tolist()}")
            minimum[name] = np.minimum(minimum[name], margin)

    return dict(
        overview_frames=len(overview), image_resolution_px=[width, height],
        minimum_required_margin_fraction=MINIMUM_MARGIN_FRACTION,
        minimum_margin_px_by_subject={
            name: dict(zip(EDGES, margin.tolist(), strict=True)) for name, margin in minimum.items()
        },
    )
