"""Core data models for co-simulation."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def as_vec3(x: np.ndarray | list[float] | tuple[float, float, float], name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=float).reshape(-1)
    if arr.shape != (3,):
        raise ValueError(f"{name} must have shape (3,), got {arr.shape}.")
    return arr


def as_rot3x3(x: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.shape != (3, 3):
        raise ValueError(f"{name} must have shape (3, 3), got {arr.shape}.")
    return arr


@dataclass
class FrameState:
    """Frame kinematics from the external simulator in world coordinates."""

    position: np.ndarray
    director: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray
    omega: np.ndarray
    alpha: np.ndarray

    def validated(self) -> "FrameState":
        return FrameState(
            position=as_vec3(self.position, "position"),
            director=as_rot3x3(self.director, "director"),
            velocity=as_vec3(self.velocity, "velocity"),
            acceleration=as_vec3(self.acceleration, "acceleration"),
            omega=as_vec3(self.omega, "omega"),
            alpha=as_vec3(self.alpha, "alpha"),
        )


@dataclass
class CoSimConfig:
    """Complete parameter set for engine construction + dummy Isaac driving."""

    py_dt: float = 1.0e-5
    isaac_dt: float = 1.0e-2
    final_time: float = 3.0
    output_interval: float = 1.0 / 100.0

    n_elem: int = 40
    base_length: float = 1.0
    base_radius: float = 2.5e-3
    density: float = 1_000.0
    youngs_modulus: float = 1.0e6
    shear_modulus_ratio: float = 1.5
    damping_constant: float = 1.0e-2
    joint_k: float = 5.0e2
    joint_nu: float = 20.0
    joint_kt: float = 1.0e1
    joint_nut: float = 0.0
    joint_model: str = "fixed"

    rod_start: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0]))
    rod_direction: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0]))
    rod_normal: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 1.0]))

    frame_base_length: float = 0.1
    frame_base_radius: float = 0.01
    frame_density: float = 5_000.0
    frame_initial_position: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0]))
    frame_initial_director: np.ndarray = field(
        default_factory=lambda: np.array(
            [
                [0.0, 0.0, 1.0],
                [0.0, -1.0, 0.0],
                [1.0, 0.0, 0.0],
            ]
        )
    )
    frame_initial_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    frame_initial_acceleration: np.ndarray = field(default_factory=lambda: np.zeros(3))
    frame_initial_omega: np.ndarray = field(default_factory=lambda: np.zeros(3))
    frame_initial_alpha: np.ndarray = field(default_factory=lambda: np.zeros(3))

    command_sine_amp: float = 0.1
    command_sine_freq: float = 1.0

    output_name: str = "cosim_test"
    output_dir: str | None = None
    render: bool = False
    render_speed: float = 1.0
    render_fps: int | None = 100
    force_vector_scale: float = 1.0
    print_progress: bool = True


@dataclass
class RodInitialState:
    """Minimal initial state needed to construct the rod."""

    start: np.ndarray
    direction: np.ndarray
    normal: np.ndarray

    def validated(self) -> "RodInitialState":
        return RodInitialState(
            start=as_vec3(self.start, "rod.start"),
            direction=as_vec3(self.direction, "rod.direction"),
            normal=as_vec3(self.normal, "rod.normal"),
        )


@dataclass
class ImpulseResult:
    """Net impulse accumulated over one external (Isaac) update."""

    linear_impulse: np.ndarray
    angular_impulse: np.ndarray
    elapsed_time: float
    sim_time: float


@dataclass
class SceneSnapshot:
    """Current rod + frame state used by logging/visualization."""

    sim_time: float
    rod_position: np.ndarray
    rod_director: np.ndarray
    frame_position: np.ndarray
    frame_director: np.ndarray
