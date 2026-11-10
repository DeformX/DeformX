"""SI geometry and prescribed endpoint motion for the alternating carriers."""
from dataclasses import dataclass, replace

import numpy as np

STRAND_NAMES = ("CORAL", "JADE", "GOLD")
STRAND_COLORS = ((.93, .115, .085), (.035, .64, .46), (.97, .57, .04))
SLOT_PITCH_M = .107
CARRIER_DEPTH_M = .078
GRIP_Z_M = .43
CARRIER_Z_M = 1.06
RADIUS_M = .012


@dataclass(frozen=True)
class Settings:
    exchanges: int = 5
    vertices: int = 25
    step_hz: int = 60
    substeps: int = 48
    iterations: int = 12
    settle_s: float = .30
    exchange_s: float = 1.
    motion_s: float = .85
    hold_s: float = .60
    density_kg_m: float = .117
    EA_N: float = 450.
    EI_Nm2: float = .00023
    gravity_z_m_s2: float = -9.81
    drag: float = .65
    damping: float = 0.
    barrier_radius_m: float = .0108
    barrier_thickness_m: float = .0024
    collision_stiffness: float = .025
    friction: float = .4
    friction_stiffness: float = 1e7
    detection_scaler: float = 1.2
    carrier_z_m: float = CARRIER_Z_M
    smooth_carrier_acceleration: bool = False

    @property
    def steps(self):
        return round((self.settle_s + self.exchanges * self.exchange_s + self.hold_s) * self.step_hz)

    @property
    def barrier_limit(self):
        return 2 * self.barrier_radius_m + self.barrier_thickness_m


SHOWCASE = Settings()
LONG = replace(SHOWCASE, exchange_s=4.60, motion_s=4.30, hold_s=1.)
PROCESS = replace(SHOWCASE, exchanges=20, vertices=97, hold_s=1., carrier_z_m=2.95,
                  smooth_carrier_acceleration=True, drag=3.)
PRESETS = {
    "short": replace(SHOWCASE, exchanges=3, hold_s=.20),
    "showcase": SHOWCASE,
    "refined": replace(SHOWCASE, vertices=41, step_hz=120),
    "long": LONG,
    "long-refined": replace(LONG, vertices=41, step_hz=120),
    "process": PROCESS,
    "process-refined": replace(PROCESS, vertices=129, step_hz=120),
}


def fixtures():
    """Return fixed lower grip positions, left to right by strand ID."""
    return np.array([[-SLOT_PITCH_M, 0., GRIP_Z_M],
                     [0., 0., GRIP_Z_M],
                     [SLOT_PITCH_M, 0., GRIP_Z_M]])


def initial_positions(cfg):
    """Three parallel, straight, unstretched rods, root first."""
    p = np.broadcast_to(fixtures()[:, None], (3, cfg.vertices, 3)).copy()
    p[..., 2] = np.linspace(cfg.carrier_z_m, GRIP_Z_M, cfg.vertices)
    return p.astype(np.float32)


def carriers_at(t, cfg, drive=True):
    """Endpoint program, indexed by strand identity across slot permutations.

    Each pair moves on opposite halves of the same ellipse, with smooth starts
    and stops. The outer carrier takes the foreground lane on both sides.
    """
    t = np.asarray(t, dtype=float)
    p = np.broadcast_to(fixtures(), (*t.shape, 3, 3)).copy()
    p[..., 2] = cfg.carrier_z_m
    if not drive:
        return p.astype(np.float32)

    slot = [0, 1, 2]
    for exchange in range(cfg.exchanges):
        left = exchange % 2
        a, b = slot[left:left + 2]
        start = cfg.settle_s + exchange * cfg.exchange_s
        u = np.clip((t - start) / cfg.motion_s, 0., 1.)
        if cfg.smooth_carrier_acceleration:
            # Integrate a cosine speed ramp over the first and last 30%,
            # with constant speed between ramps and zero acceleration at rest.
            ramp = .30
            v = np.minimum(u, 1 - u)
            phase = np.where(v < ramp, (v - ramp / np.pi * np.sin(np.pi * v / ramp)) / 2,
                             v - ramp / 2) / (1 - ramp)
            angle = np.pi * np.where(u <= .5, phase, 1 - phase)
        else:
            angle = np.pi * u * u * (3 - 2 * u)
        dx = SLOT_PITCH_M * .5 * (1 - np.cos(angle))
        dy = (-1 if left == 0 else 1) * CARRIER_DEPTH_M * np.sin(angle)
        p[..., a, 0] += dx
        p[..., b, 0] -= dx
        p[..., a, 1] += dy
        p[..., b, 1] -= dy
        slot[left:left + 2] = [b, a]
    return p.astype(np.float32)


def expected_word(exchanges):
    return [-1 if k % 2 == 0 else 2 for k in range(exchanges)]
