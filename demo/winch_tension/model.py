"""SI dimensions, motor input and the free bearing equation for the two reels."""
from dataclasses import dataclass
import math

import numpy as np

DURATION_S = 3.8
FEED_START_S, FEED_END_S = .55, 1.8
REVERSE_START_S, REVERSE_END_S = 2.4, 3.65
CYCLE_PERIOD_S = DURATION_S - FEED_START_S
MAX_CYCLES = 8
RAMP_S = .2
REEL_RADIUS_M, BARREL_HALF_WIDTH_M = .085, .05
ROD_RADIUS_M = .004
REEL_CENTERS_M = np.array([[-.26, 0., .365], [.26, 0., .365]])
INIT_TAIL_M = .23
VERTICES = 129
INIT_RADIAL_CLEARANCE_M = .00035
SDF_SPACING_M = .002
SDF_MARGIN_M = .024
DENSITY_KG_M, EA_N, EI_NM2 = .095, 1000., 5e-5
DRAG = .012
GRAVITY_M_S2 = np.array([0., 0., -9.81])
MU_STATIC, MU_KINETIC, V_SLIP_M_S, KAPPA = .85, .68, .002, 1.
ITERATIONS = 10
TERMINAL_MASS_KG, TETHER_K_N_M, TETHER_C_NS_M, TETHER_REST_M = .012, 5., .18, .02
GUIDE_INERTIA_KG_M2, GUIDE_BRAKE_NMS = .0009, .004
STRAIN_LIMIT = .04


@dataclass(frozen=True)
class Settings:
    duration_s: float | None = None
    speed_m_s: float = .075
    exchange_hz: int = 240
    rod_hz: int = 4800
    mode: str = "driven"
    cycles: int = 1

    def __post_init__(self):
        if self.mode not in {"driven", "still", "frictionless"}:
            raise ValueError("mode must be driven, still or frictionless")
        if self.exchange_hz not in {120, 240, 480} or self.rod_hz not in {4800, 9600}:
            raise ValueError("exchange rate must be 120/240/480 Hz and rod rate 4800/9600 Hz")
        if not math.isfinite(self.speed_m_s) or not .025 <= self.speed_m_s <= .09:
            raise ValueError("motor surface speed must be in [0.025, 0.09] m/s")
        if type(self.cycles) is not int or not 1 <= self.cycles <= MAX_CYCLES:
            raise ValueError(f"cycles must be an integer in [1, {MAX_CYCLES}]")
        program_end = DURATION_S + (self.cycles - 1) * CYCLE_PERIOD_S
        if self.duration_s is None:
            object.__setattr__(self, "duration_s", program_end)
        steps = self.duration_s * self.exchange_hz
        if (not math.isfinite(steps) or not .1 <= self.duration_s <= program_end
                or abs(steps - round(steps)) > 1e-8):
            raise ValueError(f"duration must be in [0.1, {program_end:g}] s and land on an exchange step")

    @property
    def dt(self):
        return 1.0 / self.exchange_hz

    @property
    def steps(self):
        return round(self.duration_s * self.exchange_hz)

    def motor_angle(self, t):
        if self.mode == "still":
            return 0.
        t -= cycle_index(t, self.cycles) * CYCLE_PERIOD_S
        feed = pulse_distance(t - FEED_START_S)
        reverse = pulse_distance(t - REVERSE_START_S)
        return self.speed_m_s * (feed - reverse) / REEL_RADIUS_M


def pulse_distance(t):
    """Integral of a unit trapezoid with zero-velocity ramp endpoints, in s."""
    width = FEED_END_S - FEED_START_S
    t = min(max(t, 0.), width)
    ramp_in = min(t, RAMP_S)
    plateau = max(0., min(t, width - RAMP_S) - RAMP_S)
    ramp_out = max(0., t - (width - RAMP_S))
    return ramp_in ** 2 / (2 * RAMP_S) + plateau + ramp_out - ramp_out ** 2 / (2 * RAMP_S)


def cycle_index(t, cycles):
    """Zero-based cycle, with the initial settle assigned to cycle zero."""
    return min(cycles - 1, max(0, math.floor((t - FEED_START_S + 1e-9) / CYCLE_PERIOD_S)))


def phase(t, cycles=1):
    t = t - cycle_index(t, cycles) * CYCLE_PERIOD_S + 1e-9
    if t < FEED_START_S:
        return "Settle"
    if t < FEED_END_S:
        return "Feed"
    if t < REVERSE_START_S:
        return "Pause"
    if t < REVERSE_END_S:
        return "Reverse"
    return "Hold"


def initial_positions():
    """Sample two straight tails, tangent quarter wraps and the upper span."""
    left, right = REEL_CENTERS_M[:, 0]
    z = REEL_CENTERS_M[0, 2]
    radius = REEL_RADIUS_M + ROD_RADIUS_M + INIT_RADIAL_CLEARANCE_M
    arc = math.pi * radius / 2
    span = right - left
    s = np.linspace(0., 2 * INIT_TAIL_M + 2 * arc + span, VERTICES)

    def profile(u):
        if u < INIT_TAIL_M:
            return left - radius, z - INIT_TAIL_M + u
        u -= INIT_TAIL_M
        if u < arc:
            return left - radius * math.cos(u / radius), z + radius * math.sin(u / radius)
        u -= arc
        if u < span:
            return left + u, z + radius
        u -= span
        if u < arc:
            return right + radius * math.sin(u / radius), z + radius * math.cos(u / radius)
        return right + radius, z - (u - arc)

    x, z = np.array([profile(u) for u in s]).T
    return np.column_stack((x, np.zeros_like(x), z)).astype(np.float32)


def tether_anchors(positions):
    anchors = positions[[0, -1]].copy()
    anchors[:, 2] = 0.
    return anchors


def tether_forces(positions, velocities, anchors):
    """Tension-only Kelvin–Voigt lines. Slack lines exert zero force."""
    delta = anchors - positions
    length = np.linalg.norm(delta, axis=-1)
    unit = delta / np.maximum(length[:, None], 1e-12)
    axial_speed = -np.sum(velocities * unit, axis=-1)
    tension = TETHER_K_N_M * (length - TETHER_REST_M) + TETHER_C_NS_M * axial_speed
    return np.where(length > TETHER_REST_M, np.maximum(0., tension), 0.)[:, None] * unit


def drum_sdf(points):
    """Signed distance of the finite physical barrel, along world/local Y."""
    radial = np.linalg.norm(points[:, [0, 2]], axis=-1) - REEL_RADIUS_M
    axial = np.abs(points[:, 1]) - BARREL_HALF_WIDTH_M
    delta = np.column_stack((radial, axial))
    return np.linalg.norm(np.maximum(delta, 0.), axis=-1) + np.minimum(delta.max(axis=-1), 0.)


def reel_quaternions(angles):
    """Reel orientations in the native xyzw convention, about +Y."""
    half = np.asarray(angles) / 2
    return np.column_stack((np.zeros_like(half), np.sin(half), np.zeros_like(half),
                            np.cos(half))).astype(np.float32)


def axle_torque(wrench, mesh_origin, pivot):
    """Cable-on-mesh world wrench at the mesh origin -> +Y torque at axle."""
    force, torque = np.asarray(wrench, dtype=np.float64).reshape(2, 3)
    lever = np.asarray(mesh_origin) - np.asarray(pivot)
    return float((torque + np.cross(lever, force))[1])


@dataclass
class PassiveAxle:
    inertia: float = GUIDE_INERTIA_KG_M2
    brake: float = GUIDE_BRAKE_NMS
    omega: float = 0.
    angle: float = 0.

    def step(self, cable_torque, dt):
        self.omega += dt * (cable_torque - self.brake * self.omega) / self.inertia
        self.angle += dt * self.omega
