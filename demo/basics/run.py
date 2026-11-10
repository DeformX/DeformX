"""Run one beginner case and save its measured CUDA trajectory."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import time

from . import CONTACT_CASES, DT, NAMES


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(parser, out):
    """Check the output before touching CUDA, then reserve it exclusively."""
    if out.exists() or out.is_symlink():
        parser.error(f"output already exists: {out}")
    from .scenes import load_cuda

    try:
        torch, rs = load_cuda()
        out.mkdir(parents=True)
    except (OSError, RuntimeError) as exc:
        parser.error(str(exc))
    return torch, rs


def write_report(out, report):
    (out / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


def report_header(torch, rs, settings):
    here = Path(__file__).resolve().parent
    sources = list(here.glob("*.py")) + [
        here.parent / "showcase" / name for name in ("diagnostics.py", "scenes.py")
    ]
    return dict(
        complete=False, settings=settings, metrics={}, timing={},
        source_sha256={str(p.relative_to(here.parent)): sha256(p) for p in sorted(sources)},
        backend=dict(extension_sha256=sha256(rs._ropesim.__file__),
                     torch=str(torch.__version__), device=torch.cuda.get_device_name()),
        trajectory_sha256=None, force_accuracy="not validated", model_accuracy="not validated",
    )


def finish(out, report, **arrays):
    import numpy as np

    trajectory = out / "trajectory.npz"
    np.savez_compressed(trajectory, **arrays)
    report["trajectory_sha256"] = sha256(trajectory)
    write_report(out, report)
    print(f"complete={report['complete']} {out / 'report.json'}")
    return 0 if report["complete"] else 1


def case_diagnostics(name, scene, positions, mesh_wrenches):
    """Return demonstration evidence and matching per-frame arrays."""
    import numpy as np
    from demo.showcase.diagnostics import object_clearance, pair_indices, segment_distance
    from demo.showcase.scenes import phi

    tip_drop = scene.positions[:, -1, 2] - positions[-1, :, -1, 2]
    if name == "hanging":
        drop = float(scene.positions[0, -1, 2] - positions[:, 0, -1, 2].min())
        return dict(peak_tip_drop_m=drop, demonstrated=drop > 0.03), {}
    if name == "endpoints":
        root = float(np.linalg.norm(positions[-1, 0, 0] - positions[0, 0, 0]))
        mid = scene.positions.shape[1] // 2
        interior = float(np.linalg.norm(positions[-1, 0, mid] - positions[0, 0, mid]))
        return dict(root_travel_m=root, midpoint_displacement_m=interior,
                    demonstrated=root > 0.05 and interior > 0.04), {}
    if name == "materials":
        difference = float(tip_drop[0] - tip_drop[1])
        return dict(final_tip_drop_m=tip_drop.tolist(), difference_m=difference,
                    demonstrated=difference > 0.03), {}
    if name == "obstacle":
        gaps = np.asarray([
            object_clearance(p, [scene.obstacle], scene.radius, phi)["sample_min_m"]
            for p in positions
        ])
        force = np.linalg.norm(mesh_wrenches[:, :3], axis=1)
        active = int(np.count_nonzero(force > 0.001))
        return dict(peak_mesh_force_N=float(force.max()), active_mesh_frames=active,
                    minimum_sampled_gap_m=float(gaps.min()),
                    demonstrated=active >= 3 and gaps.min() >= -0.005), dict(
            sampled_gap_m=gaps, mesh_wrench=mesh_wrenches,
        )

    # Segment pairs exclude two neighbors on either side; this is not a swept test.
    i, j = pair_indices(*scene.positions.shape[:2])
    distances = np.asarray([
        segment_distance(p[0, i], p[0, i+1], p[0, j], p[0, j+1]).min()
        for p in positions
    ])
    near = int(np.count_nonzero(distances < 2*scene.radius))
    return dict(minimum_nonlocal_distance_m=float(distances.min()), near_contact_frames=near,
                demonstrated=near >= 3 and distances.min() >= 1.8*scene.radius - 0.0001), dict(
        nonlocal_distance_m=distances,
    )


def run_case(args, scene, torch, rs):
    import numpy as np
    from .scenes import Rod

    steps = round(args.seconds / DT)
    settings = dict(case=args.case, requested_seconds=args.seconds, dt_s=DT,
                    requested_steps=steps, substeps=args.substeps, iterations=args.iterations,
                    contact_enabled=(args.contact != "off") if args.case in CONTACT_CASES else None,
                    **scene.settings())
    report = report_header(torch, rs, settings)
    report["limitations"] = [
        "Finite samples and demonstrated behavior do not validate material parameters, forces or convergence.",
        "Pinned supports are ideal position constraints. A clamped root also fixes its orientation.",
        "Obstacle clearance is sampled against analytic geometry; the solver uses a trilinear SDF.",
        "Self-contact proximity samples are geometric evidence, not a force readout or swept topology check.",
    ]
    write_report(args.out, report)
    try:
        rod = Rod(scene, torch, rs, dt=DT, substeps=args.substeps,
                  iterations=args.iterations, contact=args.contact != "off")
        p, v, q = rod.sample()
    except (RuntimeError, ValueError) as exc:
        report["failure"] = str(exc)
        write_report(args.out, report)
        print(report["failure"])
        return 1

    times, positions, velocities, orientations = [0.0], [p], [v], [q]
    step_ms = []
    mesh_wrenches = [np.zeros(6)]
    max_pin_error = 0.0
    begin = time.perf_counter()
    for k in range(steps):
        t = (k+1) * DT
        start, velocity = scene.support(k * DT)
        end, _ = scene.support(t)
        try:
            tick = time.perf_counter()
            rod.advance(start, velocity, end)
            ms = (time.perf_counter() - tick) * 1e3
            p, v, q = rod.sample()
            wrench = rod.batch.mesh_wrenches()[0, 0] if rod.mesh_enabled else np.zeros(6)
            if not np.isfinite(wrench).all():
                raise RuntimeError("nonfinite obstacle wrench")
        except (RuntimeError, ValueError) as exc:
            report["failure"] = f"step {k+1}: {exc}"
            break

        max_pin_error = max(max_pin_error, float(np.linalg.norm(p[:, 0] - end, axis=-1).max()))
        if scene.pinned[:, -1].all():
            max_pin_error = max(max_pin_error, float(np.linalg.norm(
                p[:, -1] - scene.positions[:, -1], axis=-1
            ).max()))
        times.append(t)
        positions.append(p)
        velocities.append(v)
        orientations.append(q)
        mesh_wrenches.append(wrench)
        step_ms.append(ms)

    wall = time.perf_counter() - begin
    positions = np.asarray(positions)
    strain = float(np.max(np.abs(
        np.linalg.norm(np.diff(positions, axis=2), axis=-1) / rod.rest_lengths - 1
    )))
    metrics, extra = case_diagnostics(args.case, scene, positions, np.asarray(mesh_wrenches))
    report["complete"] = len(step_ms) == steps
    # A disabled-contact run is a control even if another surface remains active.
    report["metrics"] = {
        **metrics,
        "demonstrated": bool(metrics["demonstrated"] and report["complete"]
                             and args.contact != "off" and max_pin_error <= 5e-5 and strain <= 0.25),
        "max_segment_strain": strain, "max_pin_error_m": max_pin_error,
        "recorded_steps": len(step_ms), "simulated_seconds": times[-1],
    }
    report["timing"] = dict(
        scope="loop includes host support calculation, synchronized step, readback and recording; excludes setup, post-run diagnostics and I/O",
        solve_scope="host input transfers plus solver and CUDA synchronization; no warmup excluded",
        loop_wall_s=wall,
        mean_solve_ms=float(np.mean(step_ms)) if step_ms else None,
        p95_solve_ms=float(np.percentile(step_ms, 95)) if step_ms else None,
    )
    return finish(args.out, report, t=times, positions=positions, velocities=velocities,
                  quaternions=orientations, rest_lengths=rod.rest_lengths,
                  solve_ms=step_ms, **extra)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=NAMES, required=True)
    parser.add_argument("--out", type=Path, required=True, help="new directory for the recording")
    parser.add_argument("--seconds", type=float, help="duration in seconds; default depends on the case")
    parser.add_argument("--substeps", type=int, default=50, help="per 1/120 s step (default: 50)")
    parser.add_argument("--iterations", type=int, default=16, help="solver iterations (default: 16)")
    parser.add_argument("--contact", choices=("on", "off"),
                        help="obstacle or self-contact control only (default: on)")
    args = parser.parse_args(argv)
    if args.seconds is not None and (not math.isfinite(args.seconds) or not DT <= args.seconds <= 60):
        parser.error(f"--seconds must be finite and between {DT:g} and 60")
    if not 1 <= args.substeps <= 1000 or not 1 <= args.iterations <= 100:
        parser.error("--substeps must be 1–1000 and --iterations must be 1–100")
    if args.contact is not None and args.case not in CONTACT_CASES:
        parser.error("--contact is only supported for obstacle or self_contact")
    try:
        from .scenes import build
    except ImportError as exc:
        parser.error(f"install NumPy to run these examples ({exc})")
    scene = build(args.case)
    args.seconds = scene.seconds if args.seconds is None else args.seconds
    return run_case(args, scene, *prepare(parser, args.out))


if __name__ == "__main__":
    raise SystemExit(main())
