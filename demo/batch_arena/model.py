"""Pure NumPy scene definition and auditable metrics (SI)."""
import numpy as np

DT = 1 / 120
VERTICES = 33
EIS = np.array([0.00015, 0.0003, 0.0006, 0.0012], np.float32)
COLORS = [[0.06, 0.84, 0.77], [0.13, 0.58, 0.98], [1.0, 0.58, 0.12], [0.94, 0.22, 0.45]]


def build(count):
    if count < 1:
        raise ValueError("count must be positive")
    side = int(np.ceil(np.sqrt(count)))
    i = np.arange(count)
    origin = np.c_[(i % side - (side - 1) / 2) * 1.35,
                   (i // side - (side - 1) / 2) * 0.95, np.full(count, 0.42)]
    u = np.linspace(0, 1, VERTICES)
    p = np.repeat(origin[:, None], VERTICES, axis=1)
    p[:, :, 0] += u - 0.5
    # An initially prestressed shallow arch; straight, untwisted natural rest.
    p[:, :, 2] += 0.07 * np.sin(np.pi * u)
    pins = np.zeros((count, VERTICES), bool)
    pins[:, [0, -1]] = True
    return dict(positions=p.astype(np.float32), pins=pins, family=i % 4,
                phase=(i // 4 % 8) * np.pi / 4)


def root_at(scene, t, drive=True):
    p = scene['positions'][:, 0].astype(float).copy()
    if drive:
        ramp = np.clip(t / 1.0, 0, 1)
        ramp = ramp * ramp * (3 - 2 * ramp)
        phase = scene['phase']
        p[:, 0] += 0.26 * ramp * (0.5 - 0.5 * np.cos(2 * np.pi * 0.32 * t + phase))
        p[:, 1] += 0.10 * ramp * np.sin(2 * np.pi * 0.32 * t + phase)
    return p.astype(np.float32)


def timing(samples_ms, worlds, dt=DT):
    x = np.asarray(samples_ms, float)
    if x.ndim != 1 or len(x) < 2 or not np.isfinite(x).all() or np.any(x <= 0) or worlds < 1:
        raise ValueError('positive finite timings and world count required')
    median = float(np.median(x))
    return dict(samples=len(x), median_ms=median, p95_ms=float(np.percentile(x, 95)),
                mean_ms=float(x.mean()), per_world_rtf=dt * 1000 / median,
                aggregate_world_steps_per_s=worlds * 1000 / median)


def diagnostics(p, v, q, initial, expected_root):
    finite = all(np.isfinite(x).all() for x in (p, v, q))
    if not finite:
        return dict(finite=False, max_strain=None, max_pin_error_m=None)
    natural = np.linalg.norm(np.diff(initial, axis=1), axis=-1)
    strain = np.abs(np.linalg.norm(np.diff(p, axis=1), axis=-1) / natural - 1)
    pin = max(float(np.linalg.norm(p[:, 0] - expected_root, axis=-1).max()),
              float(np.linalg.norm(p[:, -1] - initial[:, -1], axis=-1).max()))
    return dict(finite=True, max_strain=float(strain.max()), max_pin_error_m=pin)
