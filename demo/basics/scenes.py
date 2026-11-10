"""Initial shapes, physical settings, and support motion in SI units."""

from dataclasses import dataclass

import numpy as np

from demo.showcase.diagnostics import initial_rest_lengths
from demo.showcase.scenes import bounds, phi, root_frames


GRAVITY = (0.0, 0.0, -9.81)


@dataclass
class Scene:
    positions: np.ndarray  # [rope, vertex, xyz], m
    pinned: np.ndarray
    seconds: float
    density: float = 0.117  # kg/m
    ea: float = 1000.0  # N
    ei: tuple = (0.004,)  # N m², one value per rope
    drag: float = 1.0  # linear drag, 1/s
    clamp: bool = False
    radius: float = 0.005  # m
    floor: bool = False
    self_contact: bool = False
    obstacle: dict | None = None
    root_delta: tuple = (0.0, 0.0, 0.0)  # m
    move_seconds: float = 0.6
    sdf_spacing: float = 0.003  # m
    friction: float = 0.35

    def support(self, t):
        """Prescribe root position/velocity with zero velocity at both ends."""
        u = np.clip(t / self.move_seconds, 0.0, 1.0)
        fraction = u**3 * (10 - 15*u + 6*u*u)
        speed = 30 * u*u * (1-u)**2 / self.move_seconds
        delta = np.asarray(self.root_delta)
        p = self.positions[:, 0] + fraction * delta
        v = np.broadcast_to(speed * delta, p.shape)
        return p, v

    def settings(self):
        return dict(
            ropes=len(self.positions), vertices_per_rope=self.positions.shape[1],
            pinned_vertices=[np.flatnonzero(row).tolist() for row in self.pinned],
            density_kg_per_m=self.density, EA_N=self.ea, EI_Nm2=self.ei,
            linear_drag_per_s=self.drag, radius_m=self.radius, gravity_m_per_s2=GRAVITY,
            root_orientation_clamped=self.clamp, rest_shape="initial geometry",
            floor=self.floor, root_delta_m=self.root_delta, root_move_s=self.move_seconds,
            obstacle=self.obstacle, sdf_spacing_m=self.sdf_spacing if self.obstacle else None,
            surface_friction=self.friction if self.floor or self.obstacle else None,
            self_contact_parameters=dict(radius_m=0.9*self.radius, barrier_m=0.2*self.radius,
                                         k_collision=0.025, mu=self.friction,
                                         k_friction=1e7, detection_scaler=1.2)
            if self.self_contact else None,
        )


def line(start, end, n=33):
    return np.linspace(start, end, n, dtype=np.float32)[None]


def scene_from_positions(positions, seconds, *, pin_tip=False, **kwargs):
    pins = np.zeros(positions.shape[:2], bool)
    pins[:, 0] = True
    pins[:, -1] = pin_tip
    return Scene(positions, pins, seconds, **kwargs)


def build(name):
    if name == "hanging":
        return scene_from_positions(line([0, 0, 0.7], [0.25, 0, 0.37]), 1.5)
    if name == "endpoints":
        # Right endpoint fixed; left endpoint feeds slack and moves sideways.
        return scene_from_positions(
            line([-0.32, 0, 0.6], [0.32, 0, 0.6]), 1.4, pin_tip=True,
            root_delta=(0.14, 0.12, 0.0),
        )
    if name == "obstacle":
        return scene_from_positions(
            line([-0.32, 0, 0.6], [0.32, 0, 0.6]), 1.4, pin_tip=True,
            root_delta=(0.14, 0.0, 0.0),
            obstacle=dict(type="sphere", radius=0.080, pos=[0.04, 0.0, 0.43]),
        )
    if name == "self_contact":
        # Stress-free open helix: gravity compresses its turns against the floor.
        theta = np.linspace(0, 4*np.pi, 81)
        p = np.stack((0.1*np.cos(theta), 0.1*np.sin(theta), 0.012 + 0.02*theta/(2*np.pi)), -1)
        return scene_from_positions(
            p[None].astype(np.float32), 1.2, ei=(0.0002,), drag=0.5,
            floor=True, self_contact=True,
        )
    if name == "materials":
        # Independent cantilevers; only EI changes between the two specimens.
        p = line([0, -0.12, 0.55], [0.32, -0.12, 0.55])
        other = p + np.array([0, 0.24, 0], dtype=np.float32)
        return scene_from_positions(
            np.concatenate((p, other)), 1.5, ei=(0.006, 0.040), clamp=True, drag=3.0,
        )
    raise ValueError(f"unknown scene: {name}")


def load_cuda():
    """Only import the simulator when a run/reset needs it."""
    try:
        import torch
        import ropesim
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            f"CUDA is required; install CUDA-enabled PyTorch and the DeformX ropesim extension ({exc})."
        ) from exc
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; no CUDA device is visible to PyTorch.")
    constructor = getattr(getattr(ropesim, "RopeBatch", None), "from_positions", None)
    # pybind11 exposes the binding signature in its docstring.
    signature = (getattr(constructor, "__doc__", None) or "").partition("\n")[0]
    if "deterministic:" not in signature:
        raise RuntimeError(
            "The installed DeformX ropesim extension lacks deterministic RopeBatch support. "
            "Rebuild it in this Python environment from the repository root with "
            "python -m pip install -e Cosserat-Rod-Solver-CUDA."
        )
    return torch, ropesim


class Rod:
    """Common setup and synchronized sampling for these examples."""

    def __init__(self, scene, torch, rs, *, dt, substeps, iterations, contact=True):
        self.torch = torch
        self.dt, self.substeps = dt, substeps
        self.stream = torch.cuda.default_stream()
        self.rest_lengths = initial_rest_lengths(scene.positions)
        with torch.cuda.stream(self.stream):
            b = self.batch = rs.RopeBatch.from_positions(
                scene.positions, scene.density, scene.ea, scene.ei[0],
                dt / substeps, iterations, pinned=scene.pinned,
                fix_root_orientation=scene.clamp, deterministic=True,
            )
            if len(scene.ei) > 1:
                ei = np.repeat(np.asarray(scene.ei, np.float32)[:, None], scene.positions.shape[1]-1, axis=1)
                b.set_bend_stiffnesses(ei)
            b.damping = 0.0
            b.drag = scene.drag
            b.set_gravity(np.asarray(GRAVITY, np.float32))
            self.root_q = self._device(root_frames({"positions": scene.positions})) if scene.clamp else None
            if scene.floor:
                b.set_planes(np.array([[0, 0, 1, 0]], np.float32), radius=scene.radius, mu_s=scene.friction)
            if contact and scene.self_contact:
                b.set_rope_contact(True, 0.9*scene.radius, 0.2*scene.radius, 0.025, scene.friction, 1e7)
            self.mesh_enabled = bool(contact and scene.obstacle)
            if self.mesh_enabled:
                obj = scene.obstacle
                b.set_contact(scene.radius, scene.friction)
                grid = rs.sdf.from_function(lambda p: phi(obj, p), *bounds(obj), scene.sdf_spacing)
                b.add_mesh(*grid.as_args(), pos=np.asarray(obj["pos"], np.float32))
                b.enable_contact_forces()

    def _device(self, value):
        # Copy nonwriteable broadcasts and cast the CUDA driver input to float32.
        return self.torch.as_tensor(np.array(value, dtype=np.float32, order="C", copy=True), device="cuda")

    def advance(self, start, velocity, end):
        with self.torch.cuda.stream(self.stream):
            self.batch.step(
                self._device(start), self._device(velocity), self.dt, self.substeps,
                root_q=self.root_q, root_pos_end=self._device(end), root_q_end=self.root_q,
            )
            self.stream.synchronize()

    def sample(self):
        with self.torch.cuda.stream(self.stream):
            p, v, q = [
                self.torch.as_tensor(getter()).cpu().numpy()
                for getter in (self.batch.positions, self.batch.velocities, self.batch.quaternions)
            ]
        if self.batch.last_error or self.batch.last_warning:
            raise RuntimeError(
                f"solver flags: error={self.batch.last_error} warning={self.batch.last_warning}"
            )
        if not all(np.isfinite(a).all() for a in (p, v, q)):
            raise RuntimeError("nonfinite positions, velocities or quaternions")
        return p, v, q
