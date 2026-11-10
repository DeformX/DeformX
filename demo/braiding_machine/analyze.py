"""Audit signed spatial crossing order and sampled clearance from saved states."""
import argparse
import json
from pathlib import Path

import numpy as np

from demo.showcase.diagnostics import initial_rest_lengths, segment_distance
from demo.showcase.simulate import sha
from .model import PRESETS, Settings, carriers_at, expected_word, fixtures, initial_positions

PAIRS = ((0, 1), (0, 2), (1, 2))
CRITERIA = Path(__file__).with_name("criteria.json")


def braid_order(positions, tol=1e-6):
    """Read the braid in the x/z projection, going up from the grips.

    At each x-order swap, record the one-based slot of the strand on the left
    just below the swap, signed by that strand's y minus the other's y.
    Between shared height knots both centerlines are linear, so solve each
    zero exactly on that interval. Reject projections that lack a strict
    height order, transverse crossings, or resolvable signed depth.
    """
    p = np.asarray(positions, dtype=float)[:, ::-1]
    if not np.isfinite(p).all() or np.any(np.diff(p[..., 2], axis=1) <= tol):
        return [], "nonmonotone strand height"
    floor, roof = p[:, 0, 2].max(), p[:, -1, 2].min()
    z = np.unique(np.r_[floor, p[..., 2].ravel(), roof])
    z = z[(z >= floor) & (z <= roof)]
    if len(z) < 2:
        return [], "no common height interval"
    x = np.array([np.interp(z, rod[:, 2], rod[:, 0]) for rod in p])
    candidates = []
    for a, b in PAIRS:
        dx = x[a] - x[b]
        close = np.abs(dx) <= tol
        if close[0] or close[-1] or (close[:-1] & close[1:]).any():
            return [], "ambiguous projected order"
        k = np.flatnonzero(~close)
        for lo, hi in zip(k[:-1], k[1:]):
            if np.sign(dx[lo]) == np.sign(dx[hi]):
                continue
            h = z[lo + 1] if hi > lo + 1 else z[lo] - dx[lo] * (z[hi] - z[lo]) / (dx[hi] - dx[lo])
            y = np.array([np.interp(h, p[i, :, 2], p[i, :, 1]) for i in (a, b)])
            near = y[0] - y[1] if dx[lo] < 0 else y[1] - y[0]
            if abs(near) <= tol:
                return [], "unresolved crossing depth"
            candidates.append((float(h), a, b, float(near)))
    order = list(np.argsort(x[:, 0]))
    previous_z = -np.inf
    result = []
    for h, a, b, depth in sorted(candidates):
        i, j = sorted((order.index(a), order.index(b)))
        if j != i + 1 or h - previous_z <= tol:
            return [], "simultaneous or nonadjacent projected crossing"
        result.append(dict(z_m=h, pair=[a, b], depth_m=depth,
                           generator=(i + 1) * (1 if depth > 0 else -1)))
        order[i], order[j] = order[j], order[i]
        previous_z = h
    return result, None


def frame_clearance(a0, a1, b0, b1, endpoint_bound):
    """Discard pairs whose z separation exceeds a distance between endpoints."""
    a_lo, a_hi = np.minimum(a0[..., 2], a1[..., 2]), np.maximum(a0[..., 2], a1[..., 2])
    b_lo, b_hi = np.minimum(b0[..., 2], b1[..., 2]), np.maximum(b0[..., 2], b1[..., 2])
    gap = np.maximum(a_lo, b_lo) - np.minimum(a_hi, b_hi)
    candidate = np.nonzero(gap <= np.nextafter(endpoint_bound, np.inf)[:, None, None])
    segments = [np.broadcast_to(x, (*gap.shape, 3))[candidate] for x in (a0, a1, b0, b1)]
    closest = endpoint_bound.copy()
    np.minimum.at(closest, candidate[0], segment_distance(*segments))
    return closest


def geometry(positions, rest_lengths, barrier_limit):
    """Exact centerline distances on saved polylines, sampled in time."""
    p = np.asarray(positions, dtype=float)
    p0, p1 = p[:, :, :-1], p[:, :, 1:]
    strain = np.abs(np.linalg.norm(p1 - p0, axis=-1) / rest_lengths - 1)
    self_a, self_b = np.triu_indices(p0.shape[2], k=3)
    lowest_self = float("inf")
    minima = {f"{a}-{b}": float("inf") for a, b in PAIRS}
    counts = dict.fromkeys(minima, 0)
    for lo in range(0, len(p), 24):
        s, e = p0[lo:lo + 24], p1[lo:lo + 24]
        if self_a.size:
            bound = np.linalg.norm(e[:, :, :-3] - s[:, :, 3:], axis=-1).min(axis=(1, 2))
            closest = frame_clearance(s[:, :, self_a], e[:, :, self_a],
                                      s[:, :, self_b], e[:, :, self_b], bound).min()
            lowest_self = min(lowest_self, float(closest))
        for a, b in PAIRS:
            key = f"{a}-{b}"
            bound = np.linalg.norm(s[:, a] - s[:, b], axis=-1).min(axis=1)
            closest = frame_clearance(s[:, a, :, None], e[:, a, :, None],
                                      s[:, b, None, :], e[:, b, None, :], bound)
            minima[key] = min(minima[key], float(closest.min()))
            counts[key] += int(np.count_nonzero(closest < barrier_limit))
    return dict(
        max_abs_segment_strain=float(strain.max()),
        min_interstrand_centerline_m=min(minima.values()),
        min_nonlocal_self_centerline_m=lowest_self if np.isfinite(lowest_self) else None,
        interstrand_pair_minima_m=minima,
        pairs_in_barrier=counts,
    )


def diagnostics(t, positions, cfg):
    metrics = geometry(positions, initial_rest_lengths(initial_positions(cfg)), cfg.barrier_limit)
    t_checks = [cfg.settle_s + (i + 1) * cfg.exchange_s for i in range(cfg.exchanges)]
    checkpoints = [round(x * cfg.step_hz) for x in t_checks] + [cfg.steps]
    stages = []
    for stage, idx in enumerate(checkpoints, 1):
        if idx >= len(t):
            break
        crossings, reason = braid_order(positions[idx])
        stages.append(dict(index=idx, t_s=float(t[idx]), crossings=crossings, reason=reason,
                           expected_word=expected_word(min(stage, cfg.exchanges)),
                           measured_word=[x["generator"] for x in crossings]))
    metrics["checkpoints"] = stages
    return metrics


def endpoint_error(p, roots):
    error_root = np.linalg.norm(p[..., 0, :] - roots, axis=-1).max()
    error_grip = np.linalg.norm(p[..., -1, :] - fixtures(), axis=-1).max()
    return float(max(error_root, error_grip))


def criteria():
    return json.loads(CRITERIA.read_text())


def gates(metrics, cfg, complete, drive, contact):
    bounds = criteria()
    stages = metrics.get("checkpoints", [])
    checks = [x for stage in stages for x in stage["crossings"]]
    return dict(
        numerical_completion=complete,
        prescribed_drive_and_contact=drive and contact,
        signed_checkpoint_order=len(stages) == cfg.exchanges + 1 and all(
            stage["reason"] is None and stage["measured_word"] == stage["expected_word"]
            for stage in stages
        ),
        crossing_depth=bool(checks) and min(abs(x["depth_m"]) for x in checks)
        >= bounds["min_crossing_abs_depth_m"],
        crossing_spacing=bool(checks) and all(
            b["z_m"] - a["z_m"] >= bounds["min_crossing_vertical_gap_m"]
            for stage in stages for a, b in zip(stage["crossings"], stage["crossings"][1:])
        ),
        interstrand_contact_witness=all(
            v >= bounds["min_barrier_samples_per_pair"]
            for v in metrics.get("pairs_in_barrier", {}).values()
        ) and len(metrics.get("pairs_in_barrier", {})) == 3,
        interstrand_clearance=metrics.get("min_interstrand_centerline_m", 0)
        >= bounds["min_interstrand_centerline_m"],
        nonlocal_self_clearance=metrics.get("min_nonlocal_self_centerline_m", 0)
        >= bounds["min_nonlocal_self_centerline_m"],
        segment_strain=metrics.get("max_abs_segment_strain", float("inf"))
        <= bounds["max_abs_segment_strain"],
        endpoint_binding=metrics.get("max_endpoint_error_m", float("inf"))
        <= bounds["max_endpoint_error_m"],
    )


def read_run(folder):
    """Bind report, exact state bytes, step clock, carrier inputs and endpoints."""
    report = json.loads((folder / "report.json").read_text())
    path = folder / "trajectory.npz"
    if report.get("complete") is not True:
        raise ValueError("incomplete numerical run")
    if sha(path) != report.get("trajectory_sha256"):
        raise ValueError("trajectory hash mismatch")
    cfg = Settings(**report["settings"])
    if cfg not in PRESETS.values() or not isinstance(report["drive"], bool):
        raise ValueError("unknown trajectory settings")
    with np.load(path, allow_pickle=False) as archive:
        data = {name: archive[name] for name in ("t", "positions", "carriers")}
    t, p, roots = (data[k] for k in ("t", "positions", "carriers"))
    expected_t = np.arange(cfg.steps + 1) / cfg.step_hz
    if not np.array_equal(t, expected_t):
        raise ValueError("trajectory time does not match the complete solver step grid")
    if p.shape != (len(t), 3, cfg.vertices, 3) or not np.isfinite(p).all():
        raise ValueError("trajectory positions have invalid shape or values")
    expected = carriers_at(t, cfg, report["drive"])
    if roots.shape != expected.shape or not np.array_equal(roots, expected):
        raise ValueError("recorded carrier inputs do not match the endpoint program")
    if endpoint_error(p, roots) > criteria()["max_endpoint_error_m"]:
        raise ValueError("recorded positions do not match prescribed endpoints")
    return report, data


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="completed run directory")
    args = parser.parse_args(argv)
    report, data = read_run(args.data)
    cfg = Settings(**report["settings"])
    result = diagnostics(data["t"], data["positions"], cfg)
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
