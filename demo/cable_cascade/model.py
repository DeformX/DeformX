"""SI constants and initial conditions from the recorded cascade."""

import colorsys
import hashlib
import json

FPS = 30
STEPS_PER_FRAME = 4
DT = 1 / (FPS * STEPS_PER_FRAME)
COUNT = 48
VERTICES = 33
LENGTH = 1.1
RADIUS = 0.00635
DENSITY = 0.117
EA = 11473.0
EI = 0.00448
GRAVITY = (0., 0., -9.81)
DRAG = 0.5
FRICTION = 0.4
SPHERE_RADIUS = 0.28
SPHERE_CENTER = (0., 0., SPHERE_RADIUS)
SPHERE_COLOR = (0.16, 0.27, 0.42)
GRID_SPACING = 0.004
SUBSTEPS = 200
ITERATIONS = 24
SECONDS = 6
SEED = 0
DIAGNOSTIC_STRIDE = 12
CONTACT_RADIUS = 0.9 * RADIUS
CONTACT_SKIN = 0.2 * RADIUS
STRAIN_LIMIT = 0.25


def initial_positions(count, seed):
    import numpy as np

    rng = np.random.default_rng(seed)
    positions = np.empty((count, VERTICES, 3), np.float32)
    s = np.linspace(-LENGTH / 2, LENGTH / 2, VERTICES)
    for i in range(count):
        yaw = (i % 2) * np.pi / 2 + 0.07 * rng.uniform(-1, 1)
        center = np.array([
            ((i // 2) % 6 - 2.5) * 0.12,
            ((i // 12) % 4 - 1.5) * 0.13,
            0.65 + 0.045 * i,
        ])
        direction = np.array([np.cos(yaw), np.sin(yaw), 0.02 * rng.uniform(-1, 1)])
        positions[i] = center + s[:, None] * direction
    return positions


def palette(count):
    return [
        colorsys.hsv_to_rgb((0.54 + i * 0.78 / max(1, count - 1)) % 1, 0.70, 0.94)
        for i in range(count)
    ]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
