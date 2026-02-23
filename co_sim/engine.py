"""Core co-simulation engine: one rod + one kinematic attachment frame."""

from __future__ import annotations

from collections.abc import Callable

import elastica as ea
import numpy as np
from elastica.joint import get_relative_rotation_two_systems

from .models import CoSimConfig, FrameState, ImpulseResult, RodInitialState, SceneSnapshot


def default_rod_initial_state(config: CoSimConfig | None = None) -> RodInitialState:
    cfg = CoSimConfig() if config is None else config
    return RodInitialState(
        start=np.asarray(cfg.rod_start, dtype=float),
        direction=np.asarray(cfg.rod_direction, dtype=float),
        normal=np.asarray(cfg.rod_normal, dtype=float),
    )


def default_frame_initial_state(config: CoSimConfig | None = None) -> FrameState:
    cfg = CoSimConfig() if config is None else config
    return FrameState(
        position=np.asarray(cfg.frame_initial_position, dtype=float),
        director=np.asarray(cfg.frame_initial_director, dtype=float),
        velocity=np.asarray(cfg.frame_initial_velocity, dtype=float),
        acceleration=np.asarray(cfg.frame_initial_acceleration, dtype=float),
        omega=np.asarray(cfg.frame_initial_omega, dtype=float),
        alpha=np.asarray(cfg.frame_initial_alpha, dtype=float),
    )


class FrameStateBuffer:
    """Mutable state buffer used by the kinematic frame constraint."""

    def __init__(self) -> None:
        self.position = np.zeros(3)
        self.director = np.eye(3)
        self.velocity = np.zeros(3)
        self.acceleration = np.zeros(3)
        self.omega_world = np.zeros(3)
        self.alpha_world = np.zeros(3)

    def update(self, state: FrameState, isaac_t: float = 0.0) -> None:
        s = state.validated()
        self.position[...] = s.position
        self.director[...] = s.director
        self.velocity[...] = s.velocity + s.acceleration * isaac_t
        # self.velocity[...] = s.velocity
        self.acceleration[...] = s.acceleration
        self.omega_world[...] = s.omega + s.alpha * isaac_t
        # self.omega_world[...] = s.omega
        self.alpha_world[...] = s.alpha

    def apply_to_system(self, system) -> None:
        np.copyto(system.position_collection[:, 0], self.position)
        np.copyto(system.director_collection[..., 0], self.director)
        np.copyto(system.velocity_collection[:, 0], self.velocity)
        np.copyto(system.omega_collection[:, 0], self.omega_local)
        np.copyto(system.acceleration_collection[:, 0], self.acceleration)
        np.copyto(system.alpha_collection[:, 0], self.alpha_local)
        np.copyto(system.external_forces, 0.0)
        np.copyto(system.external_torques, 0.0)

    @property
    def omega_local(self) -> np.ndarray:
        # director maps world vectors to local vectors in PyElastica.
        return self.director @ self.omega_world

    @property
    def alpha_local(self) -> np.ndarray:
        return self.director @ self.alpha_world


class RateOnlyFrameBC(ea.ConstraintBase):
    """Constrain frame rates only; position/director are integrated by the stepper."""

    def __init__(self, state: FrameStateBuffer, **kwargs):
        super().__init__(**kwargs)
        self.state = state

    def constrain_values(self, system, time: np.float64) -> None:
        # Leave position/director unconstrained so they evolve from rates.
        pass

    def constrain_rates(self, system, time: np.float64) -> None:
        np.copyto(system.velocity_collection[:, 0], self.state.velocity)
        np.copyto(system.omega_collection[:, 0], self.state.omega_local)


class _ImpulseAccumulator:
    def __init__(self) -> None:
        self.linear_impulse = np.zeros(3)
        self.angular_impulse = np.zeros(3)

    def reset(self) -> None:
        self.linear_impulse[...] = 0.0
        self.angular_impulse[...] = 0.0


class _StepSizeBuffer:
    def __init__(self, dt: float) -> None:
        self.dt = float(dt)


class RecordAndZeroFrameLoads(ea.NoForces):
    """
    Record frame loads and then clear them, so the frame remains kinematic.
    """

    def __init__(self, dt_buffer: _StepSizeBuffer, accum: _ImpulseAccumulator):
        super().__init__()
        self.dt_buffer = dt_buffer
        self.accum = accum

    def apply_forces(self, system, time: np.float64 = np.float64(0.0)) -> None:
        force_world = system.external_forces[:, 0].copy()
        self.accum.linear_impulse += force_world * self.dt_buffer.dt
        system.external_forces[...] = 0.0

    def apply_torques(self, system, time: np.float64 = np.float64(0.0)) -> None:
        torque_local = system.external_torques[:, 0].copy()
        torque_world = system.director_collection[..., 0].T @ torque_local
        self.accum.angular_impulse += torque_world * self.dt_buffer.dt
        system.external_torques[...] = 0.0


class CoSimEngine:
    """
    Build the co-sim scene and advance one Isaac update at a time.

    One call to `update_frame_state`:
    1) applies the latest frame command state
    2) advances internal PyElastica steps
    3) returns net impulse from the rod/joint on the frame over that interval
    """

    class _Simulator(
        ea.BaseSystemCollection,
        ea.Forcing,
        ea.Connections,
        ea.Constraints,
        ea.Damping,
    ):
        pass

    def __init__(
        self,
        config: CoSimConfig,
        rod_initial_state: RodInitialState | None = None,
        frame_initial_state: FrameState | None = None,
    ):
        self.config = config
        self.py_dt = float(config.py_dt)
        self.isaac_dt = float(config.isaac_dt)
        if self.py_dt <= 0.0:
            raise ValueError(f"py_dt must be positive, got {self.py_dt}.")
        if self.isaac_dt <= 0.0:
            raise ValueError(f"isaac_dt must be positive, got {self.isaac_dt}.")

        rod_init = (
            default_rod_initial_state(config)
            if rod_initial_state is None
            else rod_initial_state
        ).validated()
        frame_init = (
            default_frame_initial_state(config)
            if frame_initial_state is None
            else frame_initial_state
        ).validated()

        self.sim = self._Simulator()

        self.rod = ea.CosseratRod.straight_rod(
            n_elements=config.n_elem,
            start=rod_init.start,
            direction=rod_init.direction,
            normal=rod_init.normal,
            base_length=config.base_length,
            base_radius=config.base_radius,
            density=config.density,
            youngs_modulus=config.youngs_modulus,
            shear_modulus=config.youngs_modulus / (2.0 * config.shear_modulus_ratio),
        )
        self.sim.append(self.rod)

        frame_axis = frame_init.director[2]
        frame_normal = frame_init.director[0]
        frame_start = frame_init.position - 0.5 * config.frame_base_length * frame_axis
        self.frame = ea.Cylinder(
            start=frame_start,
            direction=frame_axis,
            normal=frame_normal,
            base_length=config.frame_base_length,
            base_radius=config.frame_base_radius,
            density=config.frame_density,
        )
        self.sim.append(self.frame)

        self.sim.dampen(self.rod).using(
            ea.AnalyticalLinearDamper,
            damping_constant=config.damping_constant,
            time_step=self.py_dt,
        )

        self.sim.add_forcing_to(self.rod).using(
            ea.GravityForces, acc_gravity=np.array([0.0, 0.0, -9.81])
        )

        rest_rot = get_relative_rotation_two_systems(self.frame, 0, self.rod, 0)
        self.sim.connect(self.frame, self.rod, first_connect_idx=0, second_connect_idx=0).using(
            ea.FixedJoint,
            k=config.joint_k,
            nu=config.joint_nu,
            kt=config.joint_kt,
            nut=config.joint_nut,
            rest_rotation_matrix=rest_rot,
        )

        self.frame_state = FrameStateBuffer()
        self.frame_state.update(frame_init, isaac_t=0.0)
        self.sim.constrain(self.frame).using(RateOnlyFrameBC, state=self.frame_state)

        # Register after joint so synchronize() sees joint loads first, then records+zeros.
        self._step_size = _StepSizeBuffer(self.py_dt)
        self.tick_impulse = _ImpulseAccumulator()
        self.sim.add_forcing_to(self.frame).using(
            RecordAndZeroFrameLoads,
            dt_buffer=self._step_size,
            accum=self.tick_impulse,
        )

        self.sim.finalize()
        self.stepper: ea.typing.StepperProtocol = ea.PositionVerlet()
        self.time = np.float64(0.0)

        # Ensure frame state arrays match the supplied initial command exactly.
        self.apply_command_state(frame_init, isaac_t=0.0)

    def apply_command_state(self, state: FrameState, isaac_t: float = 0.0) -> None:
        self.frame_state.update(state, isaac_t=isaac_t)
        self.frame_state.apply_to_system(self.frame)

    def snapshot(self) -> SceneSnapshot:
        return SceneSnapshot(
            sim_time=float(self.time),
            rod_position=self.rod.position_collection.copy(),
            rod_director=self.rod.director_collection.copy(),
            frame_position=self.frame.position_collection.copy(),
            frame_director=self.frame.director_collection.copy(),
        )

    def update_frame_state(
        self,
        frame_state: FrameState,
        duration: float | None = None,
        observer: Callable[[float, np.ndarray], None] | None = None,
    ) -> ImpulseResult:
        """
        Update frame command and advance the co-sim by one external update interval.

        Parameters
        ----------
        frame_state
            Kinematic frame command from Isaac for this interval.
        duration
            Simulated time to advance for this update.
            Defaults to `isaac_dt`.
        observer
            Optional callback called after each internal step as
            `observer(sim_time, mean_force_window)`.
        """
        update_duration = self.isaac_dt if duration is None else float(duration)
        if update_duration <= 0.0:
            raise ValueError(f"duration must be positive, got {update_duration}.")

        start_time = float(self.time)
        self.apply_command_state(frame_state, isaac_t=self.isaac_dt)
        self.tick_impulse.reset()
        target_time = start_time + update_duration
        while float(self.time) < target_time:
            current_time = float(self.time)
            remaining = target_time - current_time
            time_ulp = np.spacing(current_time)
            if remaining <= max(time_ulp, np.finfo(float).eps):
                # Remaining interval is below time resolution; snap to target.
                self.time = np.float64(target_time)
                break

            dt_step = min(self.py_dt, remaining)
            self._step_size.dt = dt_step
            self.time = self.stepper.step(self.sim, self.time, np.float64(dt_step))
            new_time = float(self.time)
            if new_time <= current_time:
                # Guard against non-progress due floating-point resolution.
                self.time = np.float64(target_time)
                break
            if observer is not None:
                elapsed = max(float(self.time) - start_time, np.finfo(float).eps)
                mean_force = self.tick_impulse.linear_impulse / elapsed
                observer(float(self.time), mean_force.copy())

        elapsed = float(self.time) - start_time
        return ImpulseResult(
            linear_impulse=self.tick_impulse.linear_impulse.copy(),
            angular_impulse=self.tick_impulse.angular_impulse.copy(),
            elapsed_time=elapsed,
            sim_time=float(self.time),
        )
