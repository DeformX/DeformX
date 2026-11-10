"""Run the three-strand braider with the DeformX CUDA extension."""
import argparse
from dataclasses import asdict
from pathlib import Path
import socket
import time
import traceback

SOURCE = Path(__file__).parent
PRESET_NAMES = ("short", "showcase", "refined", "long", "long-refined", "process", "process-refined")
TIMING_SCOPE = "Synchronized CUDA wall time per outer solver step; excludes copy-out, diagnostics and replay."
LIMITS = [
    "Only the three root positions are prescribed by the moving carriers. The three lower grips are fixed.",
    "The machine hardware is visual. This is a finite anchored braid with no payout, winding, or modeled actuator dynamics.",
    "Rope-rope barrier candidates and responses use the native solver. Geometry is checked at saved outer steps, not continuously.",
    "The Python backend does not expose rope-pair contact forces. Barrier proximity and the matched contact-off control are evidence of interaction.",
    "The material and collision parameters are illustrative. No force calibration or physical machine validation is claimed.",
]


def execute(out, preset, cfg, drive, contact):
    import numpy as np

    from demo.showcase.diagnostics import initial_rest_lengths
    from demo.showcase.simulate import sha, write
    from .analyze import CRITERIA, diagnostics, endpoint_error, gates
    from .model import STRAND_NAMES, carriers_at, initial_positions

    report = dict(complete=False, task_success=False, preset=preset, settings=asdict(cfg),
                  drive=drive, contact=contact, host=socket.gethostname(), metrics={},
                  requested_steps=cfg.steps, completed_steps=0, flags=[], limits=LIMITS,
                  strand_ids=list(STRAND_NAMES), criteria_sha256=sha(CRITERIA),
                  source_sha256={name: sha(SOURCE / name) for name in ("run.py", "model.py", "analyze.py")},
                  shared_diagnostics_sha256=sha(SOURCE.parent / "showcase" / "diagnostics.py"),
                  timing=dict(scope=TIMING_SCOPE))
    report_file = out / "report.json"
    write(report_file, report)
    ts = np.arange(cfg.steps + 1) / cfg.step_hz
    roots = carriers_at(ts, cfg, drive)
    initial = initial_positions(cfg)
    states, velocities, quaternions, ms = [], [], [], []
    start = time.perf_counter()
    try:
        try:
            import torch
            import ropesim
        except ImportError as error:
            raise RuntimeError("Install the DeformX CUDA ropesim extension and CUDA-enabled PyTorch.") from error
        if not torch.cuda.is_available():
            raise RuntimeError("This example requires a CUDA device visible to PyTorch.")
        report["backend"] = dict(name="DeformX ropesim CUDA",
                                 extension_sha256=sha(ropesim._ropesim.__file__),
                                 extension=Path(ropesim._ropesim.__file__).name,
                                 torch=torch.__version__, cuda=torch.version.cuda,
                                 gpu=torch.cuda.get_device_name())
        pins = np.zeros(cfg.vertices, bool)
        pins[[0, -1]] = True
        dt = 1 / cfg.step_hz
        batch = ropesim.RopeBatch.from_positions(
            initial, cfg.density_kg_m, cfg.EA_N, cfg.EI_Nm2, dt / cfg.substeps,
            cfg.iterations, pinned=pins, fix_root_orientation=False, straight_rest=True,
            deterministic=True,
        )
        batch.set_gravity(np.array([0., 0., cfg.gravity_z_m_s2], np.float32))
        batch.drag = cfg.drag
        batch.damping = cfg.damping
        if contact:
            batch.set_rope_contact(True, cfg.barrier_radius_m, cfg.barrier_thickness_m,
                                   cfg.collision_stiffness, cfg.friction, cfg.friction_stiffness,
                                   cfg.detection_scaler, group_ropes=3)
        else:
            batch.set_rope_contact(False)
        target = torch.as_tensor(roots, device="cuda")
        rate = (target[1:] - target[:-1]) / dt

        def sample():
            result = [torch.as_tensor(read()).cpu().numpy().copy()
                      for read in (batch.positions, batch.velocities, batch.quaternions)]
            if not all(np.isfinite(x).all() for x in result):
                return False
            for values, value in zip((states, velocities, quaternions), result):
                values.append(value)
            return True

        if not sample():
            raise RuntimeError("The native constructor produced a nonfinite state.")
        for k in range(cfg.steps):
            torch.cuda.synchronize()
            begin = time.perf_counter_ns()
            batch.step(target[k], rate[k], dt, cfg.substeps, root_pos_end=target[k + 1])
            torch.cuda.synchronize()
            ms.append((time.perf_counter_ns() - begin) / 1e6)
            finite = sample()
            error, warning = int(batch.last_error), int(batch.last_warning)
            if finite:
                report["completed_steps"] = k + 1
            if error or warning or not finite:
                report["flags"].append(dict(step=k + 1, error=error, warning=warning, finite=finite))
                break
        report["complete"] = report["completed_steps"] == cfg.steps and not report["flags"]
    except Exception as error:
        report["error"] = dict(type=type(error).__name__, message=str(error), traceback=traceback.format_exc())
        print(str(error), flush=True)
    finally:
        count = len(states)
        # Preserve the finite prefix and its original input/timestamps, even on failure.
        if count:
            position = np.stack(states)
            path = out / "trajectory.npz"
            np.savez_compressed(path, t=ts[:count], positions=position, carriers=roots[:count],
                                velocities=velocities, quaternions=quaternions,
                                rest_lengths=initial_rest_lengths(initial))
            report["trajectory_sha256"] = sha(path)
            metrics = diagnostics(ts[:count], position, cfg)
            metrics["max_endpoint_error_m"] = endpoint_error(position, roots[:count])
            report["metrics"] = metrics
        report["gates"] = gates(report["metrics"], cfg, report["complete"], drive, contact)
        report["task_success"] = all(report["gates"].values())
        if ms:
            np.save(out / "solver_timing_ms.npy", np.asarray(ms))
            report["timing"].update(samples=len(ms), median_ms=float(np.median(ms)),
                                    p95_ms=float(np.percentile(ms, 95)),
                                    samples_sha256=sha(out / "solver_timing_ms.npy"))
        report["timing"]["run_wall_s"] = time.perf_counter() - start
        write(report_file, report)
    print(f"{preset} complete={report['complete']} task_success={report['task_success']}  {report_file}", flush=True)
    return 0 if report["complete"] else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="fresh output directory")
    parser.add_argument("--preset", choices=PRESET_NAMES, default="short",
                        help="3 exchanges (short), 5 (showcase/long), or 20 (process). "
                             "Refined presets increase the space and time resolution.")
    parser.add_argument("--no-drive", action="store_true", help="keep carriers parked as a control")
    parser.add_argument("--no-contact", action="store_true", help="disable native rope contact as a control")
    args = parser.parse_args(argv)
    try:
        args.out.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        parser.error("output already exists; choose a fresh directory")
    from .model import PRESETS

    return execute(args.out, args.preset, PRESETS[args.preset], not args.no_drive, not args.no_contact)


if __name__ == "__main__":
    raise SystemExit(main())
