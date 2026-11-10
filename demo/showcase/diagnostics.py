"""CPU numerical diagnostics, independent of CUDA solver imports."""

import numpy as np


def segment_distance(p0, p1, q0, q1):
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


def pair_indices(nropes, nverts):
    # Same-rope |i-j|<=2 exclusion matches bruteDetectK. Cross-rope pairs stay.
    rod = np.repeat(np.arange(nropes), nverts - 1)
    local = np.tile(np.arange(nverts - 1), nropes)
    flat = rod * nverts + local
    i, j = np.triu_indices(len(flat), 1)
    keep = (rod[i] != rod[j]) | (np.abs(local[i] - local[j]) > 2)
    return flat[i[keep]], flat[j[keep]]


def object_clearance(p, objects, radius, phi, max_spacing=0.002):
    p = np.asarray(p, float)
    a = p[:, :-1].reshape(-1, 3)
    b = p[:, 1:].reshape(-1, 3)
    longest = float(np.linalg.norm(b - a, axis=1).max())
    n = max(1, int(np.ceil(longest / max_spacing)))
    points = (
        a[:, None] + np.linspace(0, 1, n + 1)[None, :, None] * (b - a)[:, None]
    ).reshape(-1, 3)
    node = sample = float("inf")
    for obj in objects:
        pos = np.asarray(obj.get("pos", [0, 0, 0]))
        node = min(node, float(phi(obj, p.reshape(-1, 3) - pos).min() - radius))
        sample = min(sample, float(phi(obj, points - pos).min() - radius))
    # Exact analytic primitive SDF is1-Lipschitz. This bound does NOT apply to
    # the separately sampled/trilinearly-interpolated solver grid.
    return dict(
        node_min_m=node,
        sample_min_m=sample,
        lower_bound_m=sample - longest / n / 2,
        max_sample_spacing_m=longest / n,
        surface="analytic geometry, not interpolated grid",
    )


def initial_rest_lengths(positions):
    # from_positions -> configure sets each lRest from the supplied float32
    # segment geometry. No private reference_state extension is required.
    p = np.asarray(positions, dtype=np.float32)
    x = np.linalg.norm(np.diff(p, axis=1), axis=-1)
    if not np.isfinite(x).all() or np.any(x <= 0):
        raise ValueError("Finite nondegenerate constructor geometry required")
    return x.astype(float)


def contact_exercised(name, near_self, object_counts):
    required = [1, 2, 3] if name == "pegs" else ([1, 2] if name == "press" else [])
    valid = (
        near_self >= 3
        if name in ("knot", "coil", "braid")
        else all(object_counts[i] >= 3 for i in required)
    )
    return bool(valid), required
