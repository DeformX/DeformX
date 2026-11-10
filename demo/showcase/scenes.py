"""Deterministic SI geometry, explicit clamps and natural rest shapes; NumPy only."""

import colorsys
import numpy as np

NAMES = ("datacenter", "knot", "coil", "loom", "cascade", "parallel")
TITLES = dict(
    datacenter="Data center / UR10e cable routing",
    knot="Open knot / tighten and self-contact",
    coil="Coil collapse / dense self-contact",
    loom="Cable loom / moving guide contact",
    cascade="Cable cascade / multi-rope contact",
    parallel="Wave array / GPU batch scaling",
)


def smooth(x):
    u = np.clip(x, 0.0, 1.0)
    return u * u * u * (10 + u * (-15 + 6 * u))


def palette(n):
    return [
        list(colorsys.hsv_to_rgb((0.54 + i * 0.78 / max(1, n - 1)) % 1, 0.70, 0.94))
        for i in range(n)
    ]


def phi(obj, p):
    p = np.asarray(p, float)
    kind = obj["type"]
    if kind == "floor":
        return p[..., 2]
    if kind == "sphere":
        return np.linalg.norm(p, axis=-1) - obj["radius"]
    if kind == "cylinder":
        d = np.stack(
            (
                np.linalg.norm(p[..., :2], axis=-1) - obj["radius"],
                np.abs(p[..., 2]) - obj["half_height"],
            ),
            axis=-1,
        )
        return np.linalg.norm(np.maximum(d, 0), axis=-1) + np.minimum(d.max(-1), 0)
    if kind == "box":
        rr = obj.get("round", 0)
        d = np.abs(p) - (np.asarray(obj["half"]) - rr)
        return np.linalg.norm(np.maximum(d, 0), axis=-1) + np.minimum(d.max(-1), 0) - rr
    raise ValueError(kind)


def bounds(obj):
    if obj["type"] == "cylinder":
        h = np.array([obj["radius"], obj["radius"], obj["half_height"]])
    elif obj["type"] == "sphere":
        h = np.repeat(obj["radius"], 3)
    else:
        h = np.asarray(obj["half"])
    return -h - 0.05, h + 0.05


def resample(points, n):
    distances = np.r_[0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    s = np.linspace(0, distances[-1], n)
    return np.stack([np.interp(s, distances, points[:, k]) for k in range(3)], axis=-1)


def build_scene(name, count=None, seed=0):
    if name not in NAMES:
        raise ValueError("Unknown scene: " + name)
    if count is not None and count < 1:
        raise ValueError("count must be positive")
    if count is not None and name not in ("cascade", "parallel", "loom"):
        raise ValueError("--count applies to cascade, parallel and loom")
    objects = [dict(type="floor", pos=[0, 0, 0], color=[0.045, 0.065, 0.095])]
    pins = None
    seconds = 6.0
    r = 0.00635
    density = 0.117
    ea = 3000.0
    ei = 0.0002
    camera = dict(eye=[2.0, -2.5, 2.0], target=[0, 0, 0.45])
    fixed = False
    straight = False
    contact = True
    worlds = 1
    if name == "knot":
        t = np.linspace(0.2, 2 * np.pi - 0.2, 129)
        p = (
            0.18
            * np.stack(
                (
                    np.sin(t) + 2 * np.sin(2 * t),
                    np.cos(t) - 2 * np.cos(2 * t),
                    -np.sin(3 * t),
                ),
                axis=1,
            )
        )[None]
        p[:, :, 2] += 0.65
        pins = np.zeros(p.shape[:2], bool)
        pins[:, [0, -1]] = True
        camera = dict(eye=[1.7, -2.2, 1.6], target=[0, 0, 0.5])
    elif name == "coil":
        t = np.linspace(0, 8 * np.pi, 193)
        p = np.stack(
            (0.16 * np.cos(t), 0.16 * np.sin(t), 0.055 + 0.04 * t / (2 * np.pi)), axis=1
        )[None]
        pins = np.zeros(p.shape[:2], bool)
        for axis in (0, 1):
            for side in (-1, 1):
                h = np.array([0.4, 0.4, 0.065])
                h[axis] = 0.02
                center = np.array([0.0, 0.0, 0.065])
                center[axis] = side * 0.4
                objects.append(
                    dict(
                        type="box",
                        half=h.tolist(),
                        round=0.01,
                        pos=center.tolist(),
                        color=[0.12, 0.20, 0.30],
                    )
                )
        camera = dict(eye=[0.9, -1.2, 1.1], target=[0, 0, 0.06])
    elif name == "loom":
        n = count or 7
        x = np.linspace(-0.75, 0.75, 81)
        p = np.zeros((n, len(x), 3))
        p[:, :, 0] = x
        p[:, :, 1] = (np.arange(n)[:, None] - (n - 1) / 2) * 0.026
        p[:, :, 2] = 0.19
        pins = np.zeros(p.shape[:2], bool)
        pins[:, [0, -1]] = True
        straight = True
        for i, x in enumerate([-0.4, 0.0, 0.4]):
            objects.append(
                dict(
                    type="cylinder",
                    radius=0.075,
                    half_height=0.20,
                    pos=[x, (-1 if i % 2 == 0 else 1) * (0.16 + n * 0.013), 0.20],
                    moving=True,
                    color=[0.19, 0.32, 0.47],
                )
            )
        camera = dict(eye=[1.25, -1.9, 1.7], target=[0, 0, 0.12])
    elif name == "cascade":
        n = count or 48
        nv = 33
        length = 1.1
        p = np.zeros((n, nv, 3))
        rng = np.random.default_rng(seed)
        for i in range(n):
            yaw = (i % 2) * np.pi / 2 + 0.07 * rng.uniform(-1, 1)
            center = np.array(
                [
                    ((i // 2) % 6 - 2.5) * 0.12,
                    ((i // 12) % 4 - 1.5) * 0.13,
                    0.65 + 0.045 * i,
                ]
            )
            direction = np.array([np.cos(yaw), np.sin(yaw), 0.02 * rng.uniform(-1, 1)])
            p[i] = (
                center + np.linspace(-length / 2, length / 2, nv)[:, None] * direction
            )
        pins = np.zeros(p.shape[:2], bool)
        straight = True
        ea = 11473.0
        ei = 0.00448
        objects.append(
            dict(type="sphere", radius=0.28, pos=[0, 0, 0.28], color=[0.16, 0.27, 0.42])
        )
        camera = dict(eye=[2.8, -3.4, 2.6], target=[0, 0, 1.1])
    elif name == "parallel":
        n = count or 64
        nv = 25
        side = int(np.ceil(np.sqrt(n)))
        roots = np.c_[
            (np.arange(n) % side - (side - 1) / 2) * 1.25,
            (np.arange(n) // side - (side - 1) / 2) * 1.25,
            np.full(n, 1.35),
        ]
        p = (
            roots[:, None, :]
            + np.c_[np.zeros(nv), np.zeros(nv), -np.linspace(0, 1, nv)][None]
        )
        fixed = True
        straight = True
        contact = False
        worlds = n
        camera = dict(eye=[side * 1.2, -side * 1.5, side * 1.1], target=[0, 0, 0.65])
    else:
        seconds = 10.0
        nv = 81
        r = 0.004
        ea = 3000.0
        ei = 0.0005
        straight = True
        fixed = True
        t = np.linspace(0, 1, 1000)[:, None]
        control = np.array(
            [
                [-0.30, -0.22, 0.48],
                [-0.10, -0.85, 0.42],
                [-0.80, -0.85, 0.16],
                [-0.58, 0.04, 0.30],
            ]
        )
        curve = (
            (1 - t) ** 3 * control[0]
            + 3 * (1 - t) ** 2 * t * control[1]
            + 3 * (1 - t) * t * t * control[2]
            + t**3 * control[3]
        )
        p = resample(curve, nv)[None]
        pins = np.zeros(p.shape[:2], bool)
        pins[:, [0, -1]] = True
        for x, y in [(-0.06, -0.08), (0.17, 0.02)]:
            objects.append(
                dict(
                    type="cylinder",
                    radius=0.035,
                    half_height=0.24,
                    pos=[x, y, 0.24],
                    color=[0.15, 0.30, 0.43],
                )
            )
        camera = dict(eye=[1.65, -2.2, 1.35], target=[0, -0.06, 0.53])
    if pins is None:
        pins = np.zeros(p.shape[:2], bool)
        pins[:, 0] = True
    s = dict(
        name=name,
        title=TITLES[name],
        positions=p.astype(np.float32),
        pinned=pins,
        objects=objects,
        radius=r,
        density=density,
        EA=ea,
        EI=ei,
        drag=0.5,
        mu=0.4,
        seconds=seconds,
        camera=camera,
        colors=palette(len(p)),
        fixed_root_orientation=fixed,
        straight_rest=straight,
        rope_contact=contact,
        worlds=worlds,
        seed=seed,
        default_substeps=200 if name == "cascade" else 50,
        default_iterations=24 if name == "cascade" else 16,
        natural_reference="straight, untwisted"
        if straight
        else "initial curve, stress-free",
        boundary="root material frame clamped; other pinned endpoint position only"
        if fixed
        else "pinned positions are free rotational pivots",
    )
    if name == "datacenter":
        s.update(
            target_position=[0.30, 0.12, 0.67],
            robot_scope="imported UR10e/Robotiq joint frames; scripted cable routing and rack approach; kinematic attachment, no contact grasp or connector mating",
            asset_profile_required=True,
        )
    return s


def roots_at(scene, t):
    roots = scene["positions"][:, 0].astype(float).copy()
    name = scene["name"]
    if name == "knot":
        d = roots[0] - scene["positions"][0, -1]
        d /= np.linalg.norm(d)
        roots[0] += 0.8 * smooth((t - 0.6) / 4.5) * d
    elif name == "parallel":
        i = np.arange(len(roots))
        phase = i * 0.17
        ramp = smooth(t / 0.8)
        roots[:, 0] += (
            0.22 * ramp * (np.sin(2 * np.pi * 0.65 * t + phase) - np.sin(phase))
        )
        roots[:, 1] += (
            0.16 * ramp * (np.cos(2 * np.pi * 0.65 * t + phase) - np.cos(phase))
        )
    elif name == "datacenter":
        times = np.array([0, 1, 3, 5, 7, 8.5, 10.0])
        points = np.array(
            [
                [-0.30, -0.22, 0.48],
                [-0.30, -0.22, 0.48],
                [-0.24, -0.10, 0.59],
                [-0.02, -0.12, 0.68],
                [0.30, -0.12, 0.67],
                [0.30, 0.035, 0.67],
                [0.30, 0.12, 0.67],
            ]
        )
        j = min(np.searchsorted(times, t, side="right") - 1, len(times) - 2)
        j = max(0, j)
        u = smooth((t - times[j]) / (times[j + 1] - times[j]))
        roots[0] = (1 - u) * points[j] + u * points[j + 1]
    return roots


def object_poses_at(scene, t):
    positions = np.array([o["pos"] for o in scene["objects"]], float)
    if scene["name"] == "loom":
        amp = (
            0.22
            * smooth((t - 0.6) / 1.5)
            * (0.5 + 0.5 * np.cos(2 * np.pi * max(0, t - 2.1) / 4.0))
            if t > 2.1
            else 0.22 * smooth((t - 0.6) / 1.5)
        )
        for i in range(1, len(positions)):
            positions[i, 1] += (1 if positions[i, 1] < 0 else -1) * amp
    return positions


def root_frames(scene):
    # Exact constructor tangent mapped from solver material +X, xyzw.
    d = scene["positions"][:, 1] - scene["positions"][:, 0]
    d = d / np.linalg.norm(d, axis=-1, keepdims=True)
    src = np.zeros_like(d)
    src[:, 0] = 1
    q = np.c_[np.cross(src, d), 1 + d[:, 0]]
    bad = np.linalg.norm(q, axis=-1) < 1e-6
    q[bad] = [0, 0, 1, 0]
    return (q / np.linalg.norm(q, axis=-1, keepdims=True)).astype(np.float32)


def serializable(scene):
    return {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in scene.items()}
