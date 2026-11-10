"""Measure contact and validate a portable cascade recording."""

import argparse
from functools import lru_cache
import json
from pathlib import Path

from .model import (
    COUNT, FPS, ITERATIONS, RADIUS, SECONDS, SEED, SPHERE_CENTER, SPHERE_RADIUS,
    STRAIN_LIMIT, SUBSTEPS, VERTICES, initial_positions, sha,
)


def segment_distance(p0, p1, q0, q1):
    import numpy as np

    p0, p1, q0, q1 = [np.asarray(x, dtype=float) for x in (p0, p1, q0, q1)]
    if not all(np.isfinite(x).all() and x.shape[-1:] == (3,) for x in (p0, p1, q0, q1)):
        raise ValueError("finite XYZ segment endpoints required")
    d1, d2, r = p1 - p0, q1 - q0, p0 - q0
    a, e = np.sum(d1 * d1, axis=-1), np.sum(d2 * d2, axis=-1)
    f, c, b = np.sum(d2 * r, axis=-1), np.sum(d1 * r, axis=-1), np.sum(d1 * d2, axis=-1)
    eps = 1e-30
    den = a * e - b * b
    s = np.where(den > eps, np.clip((b * f - c * e) / np.maximum(den, eps), 0, 1), 0.0)
    t = (b * s + f) / np.maximum(e, eps)
    s = np.where(
        t < 0,
        np.clip(-c / np.maximum(a, eps), 0, 1),
        np.where(t > 1, np.clip((b - c) / np.maximum(a, eps), 0, 1), s),
    )
    t = np.clip(t, 0, 1)
    s = np.where(e <= eps, np.clip(-c / np.maximum(a, eps), 0, 1), s)
    t = np.where(e <= eps, 0.0, t)
    s = np.where(a <= eps, 0.0, s)
    t = np.where(a <= eps, np.clip(f / np.maximum(e, eps), 0, 1), t)
    return np.linalg.norm(p0 + s[..., None] * d1 - q0 - t[..., None] * d2, axis=-1)



@lru_cache(maxsize=1)
def segment_pairs(rods, vertices):
    import numpy as np

    # The solver excludes nearby segments of the same rod from self-contact.
    rod = np.repeat(np.arange(rods), vertices - 1)
    local = np.tile(np.arange(vertices - 1), rods)
    start = rod * vertices + local
    i, j = np.triu_indices(len(start), 1)
    cross_rod = rod[i] != rod[j]
    keep = cross_rod | (np.abs(local[i] - local[j]) > 2)
    return start[i[keep]], start[j[keep]], cross_rod[keep]


def measure_clearance(positions):
    import numpy as np

    positions = np.asarray(positions, dtype=float)
    p = positions.reshape(-1, 3)
    i, j, cross = segment_pairs(*positions.shape[:2])
    min_pair = min_cross = float("inf")
    for start in range(0, len(i), 131072):
        chunk = slice(start, start + 131072)
        a, b = i[chunk], j[chunk]
        distance = segment_distance(p[a], p[a + 1], p[b], p[b + 1])
        min_pair = min(min_pair, float(distance.min()))
        mask = cross[chunk]
        if mask.any():
            min_cross = min(min_cross, float(distance[mask].min()))

    # Distance from the sphere center to the nearest point on each finite segment.
    offset = positions[:, :-1] - SPHERE_CENTER
    delta = np.diff(positions, axis=1)
    length_squared = np.sum(delta * delta, axis=-1)
    fraction = np.clip(-np.sum(offset * delta, axis=-1) / length_squared, 0., 1.)
    sphere_distance = np.linalg.norm(offset + fraction[..., None] * delta, axis=-1)
    return {
        "min_nonlocal_centerline_distance_m": min_pair if np.isfinite(min_pair) else None,
        "min_cross_rod_centerline_distance_m": min_cross if np.isfinite(min_cross) else None,
        "min_cross_rod_surface_gap_m": min_cross - 2 * RADIUS if np.isfinite(min_cross) else None,
        "min_sphere_surface_gap_m": float(sphere_distance.min() - SPHERE_RADIUS - RADIUS),
        "min_floor_surface_gap_m": float(p[:, 2].min() - RADIUS),
    }


def load_trajectory(data):
    import numpy as np

    report = json.loads((data / "report.json").read_text())
    path = data / "trajectory.npz"
    if report.get("complete") is not True:
        raise ValueError("simulation is incomplete; see report.json")
    if sha(path) != report["trajectory_sha256"]:
        raise ValueError("trajectory SHA-256 disagrees with report.json")
    settings = report["settings"]
    frames = round(settings["seconds"] * FPS) + 1
    count = settings["count"]
    if not 1 <= count <= COUNT or frames < 2:
        raise ValueError("invalid trajectory dimensions in report.json")
    with np.load(path, allow_pickle=False) as trajectory:
        times = trajectory["t"]
        positions = trajectory["positions"]
        velocities = trajectory["velocities"]
        rest = trajectory["rest_lengths"]
    if (
        report["frames"] != frames
        or times.shape != (frames,)
        or not np.allclose(times, np.arange(frames) / FPS, atol=1e-12, rtol=0.)
    ):
        raise ValueError("trajectory does not follow the 30 Hz frame cadence")
    if (
        positions.shape != (frames, count, VERTICES, 3)
        or velocities.shape != positions.shape
        or not np.isfinite(positions).all()
        or not np.isfinite(velocities).all()
    ):
        raise ValueError("invalid position or velocity array")
    initial = initial_positions(count, settings["seed"])
    if not np.array_equal(initial, positions[0]):
        raise ValueError("trajectory initial state does not match the seeded model")
    expected_rest = np.linalg.norm(np.diff(initial, axis=1), axis=-1)
    if not np.array_equal(rest, expected_rest):
        raise ValueError("trajectory rest lengths do not match the seeded model")
    recorded_strain = float(np.max(np.abs(np.linalg.norm(np.diff(positions, axis=2), axis=-1) / rest - 1)))
    if recorded_strain > STRAIN_LIMIT:
        raise ValueError("recorded segment strain exceeds the numerical limit")
    return report, times, positions


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="output directory from cascade run")
    parser.add_argument("--reference", action="store_true", help="also require the reference settings and observed contact")
    args = parser.parse_args(argv)
    try:
        report, times, _ = load_trajectory(args.data)
        diagnostics = args.data / "diagnostics.json"
        if sha(diagnostics) != report["diagnostics_sha256"]:
            raise ValueError("diagnostic SHA-256 disagrees with report.json")
        if args.reference:
            reference = dict(count=COUNT, seconds=SECONDS, seed=SEED, substeps=SUBSTEPS, iterations=ITERATIONS)
            if any(report["settings"][key] != value for key, value in reference.items()):
                raise ValueError("simulation settings do not match the six-second reference")
            if not all(report["contact_proximity_frames"].values()):
                raise ValueError("expected cross-rod, sphere and floor proximity")
            distance = report["metrics"]["min_nonlocal_centerline_distance_m"]
            if distance < report["settings"]["contact_barrier_m"] - 0.0001:
                raise ValueError("sampled nonlocal rods crossed the effective contact barrier tolerance")
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    print(json.dumps({
        "frames": len(times),
        "simulation_interval_s": [float(times[0]), float(times[-1])],
        "trajectory_sha256": report["trajectory_sha256"],
        "metrics": report["metrics"],
        "contact_proximity_frames": report["contact_proximity_frames"],
        "reference_checks": args.reference,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
