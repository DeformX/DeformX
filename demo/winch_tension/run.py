"""Drive a tensioned cable across two traction reels, with a free right axle."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import time

LIMITATIONS = [
    "Left reel angle is prescribed. No motor torque or tension feedback controller is modeled.",
    "Right reel is a constrained one-degree-of-freedom inertia and viscous brake. Its torque is the previous exchange's uncorrected world-frame cable-on-barrel mesh torque.",
    "The cable is gravity loaded and pretensioned by compliant tethers. All vertices, including terminal masses, are free. Only the barrel contacts the cable; decorative flanges are visual.",
    "This is traction transport with a single partial wrap per reel and finite travel. It models neither changing winding radius nor accumulating layers.",
    "Grid interpolation, penalty forces, regularized slip and explicit exchange affect instantaneous loads. Wrenches and computed tether forces are uncalibrated model outputs.",
]
ENDS = [0, -1]


def initial_frames(positions):
    import numpy as np

    return dict(t=[0.], positions=[positions], velocities=[np.zeros_like(positions)],
                reel_angle_rad=[np.zeros(2)], guide_angular_velocity_rad_s=[0.],
                reel_wrenches=[np.zeros((2, 6))], guide_applied_torque_Nm=[0.],
                contact_forces_N=[np.zeros_like(positions)], tether_forces_N=[np.zeros((2, 3))])


def simulate(settings, frames, report):
    import numpy as np

    from . import model
    from .analysis import sha256

    try:
        import torch
        import ropesim
        from ropesim import sdf
    except ImportError as exc:
        raise RuntimeError("Install the DeformX CUDA ropesim extension and CUDA-enabled PyTorch") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA device is required for the DeformX cable solver")

    report["backend_sha256"] = sha256(ropesim._ropesim.__file__)
    report["hardware"] = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
                          "torch_cuda": torch.version.cuda}
    positions = frames["positions"][0]
    velocities = frames["velocities"][0]
    anchors = model.tether_anchors(positions)
    rest = np.linalg.norm(np.diff(positions, axis=0), axis=-1)
    hz = settings.rod_hz
    rod = ropesim.RopeBatch.from_positions(
        positions[None], model.DENSITY_KG_M, model.EA_N, model.EI_NM2, 1 / hz, model.ITERATIONS,
        pinned=np.zeros((1, len(positions)), bool), fix_root_orientation=False, straight_rest=True,
    )
    masses = 1. / np.asarray(rod.inv_masses(), dtype=np.float64)[0]
    masses[ENDS] = model.TERMINAL_MASS_KG
    rod.set_masses(masses.astype(np.float32))
    rod.damping, rod.drag = 0., model.DRAG
    rod.set_gravity(model.GRAVITY_M_S2.astype(np.float32))
    mu_s, mu_k = (0., 0.) if settings.mode == "frictionless" else (model.MU_STATIC, model.MU_KINETIC)
    rod.set_contact(model.ROD_RADIUS_M, mu_s, mu_k, v_slip=model.V_SLIP_M_S, kappa=model.KAPPA)
    rod.enable_contact_forces(True)
    half_extent = np.array([model.REEL_RADIUS_M, model.BARREL_HALF_WIDTH_M, model.REEL_RADIUS_M]) + model.SDF_MARGIN_M
    grid = sdf.from_function(model.drum_sdf, -half_extent, half_extent, model.SDF_SPACING_M)
    centers = model.REEL_CENTERS_M.astype(np.float32)
    for center in centers:
        rod.add_mesh(*grid.as_args(), pos=center)
    report["sdf_values_sha256"] = hashlib.sha256(grid.values.tobytes()).hexdigest()
    report["rest_length_m"] = float(rest.sum())
    report["peak_abs_segment_strain"] = 0.
    report["sample_count"] = 1
    report["timing"]["setup_wall_s"] = time.perf_counter() - report["timing"].pop("_start")
    dt = settings.dt
    substeps = hz // settings.exchange_hz
    guide = model.PassiveAxle()
    exchange_torque = 0.
    started = time.perf_counter()
    try:
        for k in range(1, settings.steps + 1):
            t = k * dt
            applied_torque = exchange_torque
            guide.step(applied_torque, dt)
            reel_angle = [settings.motor_angle(t), guide.angle]
            rod.set_mesh_poses_all(centers[None], model.reel_quaternions(reel_angle)[None])
            forces = model.tether_forces(positions[ENDS], velocities[ENDS], anchors)
            kicked = velocities.copy()
            kicked[ENDS] += forces * (dt / masses[ENDS, None])
            rod.set_velocities(kicked[None])
            rod.step_free(dt, substeps)
            torch.cuda.synchronize()
            positions = torch.as_tensor(rod.positions()).cpu().numpy()[0].copy()
            velocities = torch.as_tensor(rod.velocities()).cpu().numpy()[0].copy()
            wrenches = rod.mesh_wrenches()[0].copy()
            contact = rod.contact_forces()[0].copy()
            state = [positions, velocities, wrenches, contact, reel_angle, guide.omega, forces]
            if any(not np.isfinite(value).all() for value in state):
                raise RuntimeError(f"Nonfinite state after step {k}, t={t:g} s")
            for key, value in zip(frames, (t, positions, velocities, reel_angle, guide.omega,
                                            wrenches, applied_torque, contact, forces), strict=True):
                frames[key].append(value)
            report["sample_count"] = k + 1
            strain = float(np.max(np.abs(np.linalg.norm(np.diff(positions, axis=0), axis=-1) / rest - 1.)))
            report["peak_abs_segment_strain"] = max(strain, report["peak_abs_segment_strain"])
            rejection = None
            if rod.last_warning or rod.last_error:
                rejection = f"Native error={rod.last_error} warning={rod.last_warning} at t={t:g} s"
            elif strain > model.STRAIN_LIMIT:
                rejection = f"Segment strain {strain:.4f} exceeded {model.STRAIN_LIMIT:g} at t={t:g} s"
            if rejection:
                report.update(rejected_sample_index=k, rejected_sample_time_s=t)
                raise RuntimeError(rejection)
            exchange_torque = model.axle_torque(wrenches[1], centers[1], centers[1])
    finally:
        report["timing"]["exchange_loop_wall_s"] = time.perf_counter() - started


def execute(out, settings):
    import numpy as np

    from . import model
    from .analysis import measure, sha256

    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    report_path = out / "report.json"
    positions = model.initial_positions()
    frames = initial_frames(positions)
    report = dict(complete=False, backend_sha256=None, limitations=LIMITATIONS,
                  settings=asdict(settings) | dict(
                      vertices=model.VERTICES, reel_centers_m=model.REEL_CENTERS_M.tolist(),
                      reel_radius_m=model.REEL_RADIUS_M, barrel_half_width_m=model.BARREL_HALF_WIDTH_M,
                      rod_radius_m=model.ROD_RADIUS_M, density_kg_m=model.DENSITY_KG_M, EA_N=model.EA_N,
                      EI_Nm2=model.EI_NM2, gravity_m_s2=model.GRAVITY_M_S2.tolist(), drag=model.DRAG,
                      sdf_spacing_m=model.SDF_SPACING_M, tether_anchors_m=model.tether_anchors(positions).tolist(),
                      tether_stiffness_N_m=model.TETHER_K_N_M, tether_damping_Ns_m=model.TETHER_C_NS_M,
                      tether_rest_length_m=model.TETHER_REST_M, terminal_mass_kg=model.TERMINAL_MASS_KG,
                      guide_inertia_kg_m2=model.GUIDE_INERTIA_KG_M2, guide_brake_Nms=model.GUIDE_BRAKE_NMS,
                      mu_s=0. if settings.mode == "frictionless" else model.MU_STATIC,
                      mu_k=0. if settings.mode == "frictionless" else model.MU_KINETIC,
                      slip_regularizer_m_s=model.V_SLIP_M_S, penalty_kappa=model.KAPPA,
                      rod_iterations=model.ITERATIONS, rod_substeps=settings.rod_hz // settings.exchange_hz),
                  timing=dict(_start=started, scope="Exchange loop includes solver, CUDA synchronization, wrench/pose exchange, host readback and diagnostics. Excludes setup and output."),
                  source_sha256={name: sha256(Path(__file__).with_name(name))
                                 for name in ("model.py", "run.py", "analysis.py")})
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    try:
        simulate(settings, frames, report)
        report["complete"] = len(frames["t"]) == settings.steps + 1
    except BaseException as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        report["timing"].pop("_start", None)
        report["timing"]["before_output_wall_s"] = time.perf_counter() - started
        arrays = {key: np.asarray(value) for key, value in frames.items()}
        arrays["rest_lengths_m"] = np.linalg.norm(np.diff(positions, axis=0), axis=-1)
        arrays["tether_anchors_m"] = model.tether_anchors(positions)
        metrics = measure(arrays, cycles=settings.cycles,
                          rejected_sample_index=report.get("rejected_sample_index"))
        metrics["task_success"] &= report["complete"]
        report["metrics"] = metrics
        data = out / "trajectory.npz"
        np.savez_compressed(data, **arrays)
        report["trajectory_sha256"] = sha256(data)
        partial_report = out / ".report.json.tmp"
        partial_report.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        partial_report.replace(report_path)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="Fresh output directory")
    parser.add_argument("--mode", choices=("driven", "still", "frictionless"), default="driven")
    parser.add_argument("--duration-s", type=float, help="Trial duration in seconds (default full program)")
    parser.add_argument("--speed-m-s", type=float, help="Motor barrel surface speed in m/s (default 0.075)")
    parser.add_argument("--cycles", type=int, help="Successive feed/pause/reverse cycles, 1–8 (default 1)")
    parser.add_argument("--exchange-hz", type=int, choices=(120, 240, 480), default=240)
    parser.add_argument("--rod-hz", type=int, choices=(4800, 9600), default=4800)
    args = parser.parse_args(argv)
    try:
        from . import model
    except ImportError as exc:
        print(f"Winch trial requires NumPy: {exc}", file=sys.stderr)
        return 1
    try:
        options = {key: value for key, value in vars(args).items() if key != "out" and value is not None}
        settings = model.Settings(**options)
    except ValueError as exc:
        parser.error(str(exc))
    try:
        report = execute(args.out, settings)
    except (RuntimeError, OSError) as exc:
        print(f"Winch trial failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"out": str(args.out), "complete": report["complete"],
                      "task_success": report["metrics"]["task_success"]}))
    return 0 if report["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
