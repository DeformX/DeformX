"""Run the dense interacting cable cascade using the DeformX CUDA solver."""

import argparse
import math
from pathlib import Path
import platform
import time

from .model import (
    CONTACT_RADIUS, CONTACT_SKIN, COUNT, DENSITY, DIAGNOSTIC_STRIDE, DRAG, DT,
    EA, EI, FPS, FRICTION, GRAVITY, GRID_SPACING, ITERATIONS, RADIUS, SECONDS,
    SEED, SPHERE_CENTER, SPHERE_RADIUS, STEPS_PER_FRAME, STRAIN_LIMIT, SUBSTEPS,
    VERTICES, initial_positions, sha, write_json,
)


def positive_int(text):
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def rod_count(text):
    value = positive_int(text)
    if value > COUNT:
        raise argparse.ArgumentTypeError(f"must be between 1 and {COUNT}")
    return value


def seed_value(text):
    value = int(text)
    if value < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return value


def frame_duration(text):
    value = float(text)
    frames = value * FPS
    if not math.isfinite(frames) or frames < 1 or not math.isclose(frames, round(frames), abs_tol=1e-8, rel_tol=0.):
        raise argparse.ArgumentTypeError("must be a positive multiple of 1/30 second")
    return round(frames) / FPS


def configure(positions, substeps, iterations, ropesim, sdf):
    import numpy as np

    batch = ropesim.RopeBatch.from_positions(
        positions, DENSITY, EA, EI, DT / substeps, iterations,
        pinned=np.zeros(positions.shape[:2], bool),
        fix_root_orientation=False, straight_rest=True,
    )
    batch.damping = 0.0
    batch.drag = DRAG
    batch.set_gravity(np.array(GRAVITY, np.float32))
    batch.set_planes(np.array([[0, 0, 1, 0]], np.float32), radius=RADIUS, mu_s=FRICTION, mu_k=FRICTION)
    batch.set_rope_contact(True, CONTACT_RADIUS, CONTACT_SKIN, 0.025, FRICTION, 1e7, 1.2, len(positions))
    extent = SPHERE_RADIUS + .05
    sphere = sdf.from_function(
        lambda points: np.linalg.norm(points, axis=-1) - SPHERE_RADIUS,
        np.full(3, -extent), np.full(3, extent), GRID_SPACING,
    )
    batch.add_mesh(*sphere.as_args(), pos=np.array(SPHERE_CENTER, np.float32))
    return batch


def simulate(args):
    import numpy as np
    import torch
    import ropesim
    from ropesim import sdf
    from .audit import measure_clearance

    if not torch.cuda.is_available():
        raise RuntimeError("a CUDA GPU and CUDA-enabled PyTorch are required")
    torch.cuda.set_stream(torch.cuda.default_stream())

    out = args.out
    out.mkdir(parents=True)
    initial = initial_positions(args.count, args.seed)
    rest = np.linalg.norm(np.diff(initial, axis=1), axis=-1)
    steps = round(args.seconds * FPS) * STEPS_PER_FRAME
    report_path = out / "report.json"
    report = {
        "complete": False,
        "settings": dict(
            seconds=args.seconds, count=args.count, vertices=VERTICES, seed=args.seed,
            substeps=args.substeps, iterations=args.iterations, dt_s=DT, fps=FPS,
            contact_group_size=args.count, radius_m=RADIUS, density_kg_m=DENSITY,
            EA_N=EA, EI_Nm2=EI, gravity_m_s2=GRAVITY, drag=DRAG, friction=FRICTION,
            sphere_radius_m=SPHERE_RADIUS, sphere_center_m=SPHERE_CENTER,
            sdf_spacing_m=GRID_SPACING, contact_barrier_m=2 * CONTACT_RADIUS,
            diagnostic_stride_outer_steps=DIAGNOSTIC_STRIDE,
        ),
        "runtime": dict(
            host=platform.node(), python=platform.python_version(), torch=str(torch.__version__),
            cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(),
            backend_sha256=sha(Path(ropesim._ropesim.__file__)),
            source_sha256={
                name: sha(Path(__file__).with_name(name))
                for name in ("run.py", "model.py", "audit.py")
            },
        ),
    }
    write_json(report_path, report)

    try:
        batch = configure(initial, args.substeps, args.iterations, ropesim, sdf)
        times, positions, velocities, diagnostics, latencies, flags = [], [], [], [], [], []
        max_strain = 0.
        finite = True
        stop = None
        for step in range(steps + 1):
            t = step * DT
            if step:
                torch.cuda.synchronize()
                start = time.perf_counter_ns()
                batch.step_free(DT, args.substeps)
                torch.cuda.synchronize()
                latencies.append((time.perf_counter_ns() - start) * 1e-6)
            p, v, q = [
                torch.as_tensor(read()).cpu().numpy().copy()
                for read in (batch.positions, batch.velocities, batch.quaternions)
            ]
            finite = all(np.isfinite(state).all() for state in (p, v, q))
            if not finite:
                stop = "nonfinite solver state"
                np.savez_compressed(out / "nonfinite-state.npz", t=t, positions=p, velocities=v, quaternions=q)
                break
            strain = float(np.max(np.abs(np.linalg.norm(np.diff(p, axis=1), axis=-1) / rest - 1)))
            max_strain = max(max_strain, strain)
            if batch.last_error or batch.last_warning:
                flags.append(dict(step=step, error=int(batch.last_error), warning=int(batch.last_warning)))
            if step % STEPS_PER_FRAME == 0:
                times.append(t)
                positions.append(p)
                velocities.append(v)
            # Keep the reference sample phase and include the initial and final states.
            if step in (0, steps) or step % DIAGNOSTIC_STRIDE == 1:
                diagnostics.append(dict(step=step, t_s=t, strain=strain, **measure_clearance(p)))
            if flags or strain > STRAIN_LIMIT:
                stop = "solver flags" if flags else "segment strain exceeds 25%"
                break
            if step and step % (FPS * STEPS_PER_FRAME) == 0:
                print(f"Cascade {t:.1f}/{args.seconds:.1f} s, peak strain {max_strain:.3%}", flush=True)

        trajectory = out / "trajectory.npz"
        np.savez_compressed(trajectory, t=times, positions=positions, velocities=velocities, rest_lengths=rest)
        diagnostic_path = out / "diagnostics.json"
        write_json(diagnostic_path, diagnostics)
        np.save(out / "timings_ms.npy", np.asarray(latencies))
        minimums = {
            key: min((row[key] for row in diagnostics if row[key] is not None), default=None)
            for key in diagnostics[0] if key.startswith("min_")
        } if diagnostics else {}
        warmup = min(15, max(0, len(latencies) - 1))
        timed = latencies[warmup:]
        report.update(
            complete=stop is None and len(latencies) == steps,
            stop_reason=stop, frames=len(times), outer_steps=len(latencies),
            trajectory_sha256=sha(trajectory), diagnostics_sha256=sha(diagnostic_path),
            metrics=dict(finite=finite, flags=flags, max_strain=max_strain, **minimums),
            contact_proximity_frames={
                label: sum(row[key] is not None and row[key] <= tolerance for row in diagnostics)
                for label, key, tolerance in (
                    ("cross_rods", "min_cross_rod_surface_gap_m", 0.),
                    ("sphere", "min_sphere_surface_gap_m", .001),
                    ("floor", "min_floor_surface_gap_m", .001),
                )
            },
            timing=dict(
                warmup_steps=warmup, samples=len(timed),
                median_ms=float(np.median(timed)) if timed else None,
                per_world_rtf=float(DT * 1000 / np.mean(timed)) if timed else None,
                scope="CUDA solve and synchronization",
            ),
        )
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        write_json(report_path, report)
    if not report["complete"]:
        raise RuntimeError(f"{stop}; see {report_path}")
    print(f"Saved {len(times)} states to {out}", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path, help="fresh output directory")
    parser.add_argument("--seconds", type=frame_duration, default=SECONDS, help="simulation seconds in 1/30 s increments")
    parser.add_argument("--count", type=rod_count, default=COUNT, help="number of interacting cables (1–48)")
    parser.add_argument("--seed", type=seed_value, default=SEED, help="seed for cable yaw and tilt")
    parser.add_argument("--substeps", type=positive_int, default=SUBSTEPS, help="solver substeps per 120 Hz outer step")
    parser.add_argument("--iterations", type=positive_int, default=ITERATIONS, help="solver iterations per substep")
    args = parser.parse_args(argv)
    if args.out.exists():
        parser.error("--out must be a fresh directory")
    try:
        simulate(args)
    except (ImportError, OSError, RuntimeError, ValueError) as error:
        parser.exit(1, f"Cascade simulation failed: {error}\nSee demo/cable_cascade/README.md for the CUDA setup.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
