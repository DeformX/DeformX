"""Real CUDA runs; lightweight imports keep outcome checks usable without a GPU."""

import hashlib
import copy
import json
import platform
from pathlib import Path
import subprocess
import time
import traceback
import numpy as np
from .scenes import roots_at, object_poses_at, root_frames, serializable, phi, bounds
from .diagnostics import (
    pair_indices,
    segment_distance,
    initial_rest_lengths,
    object_clearance,
)
from .metrics import summarize_timing

DT = 1 / 120
FPS = 30


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def outcome(
    *,
    completed,
    finite,
    flags,
    pin_error_m,
    max_strain,
    min_contact_distance_m,
    contact_radius,
    contact_enabled,
):
    if not completed or not finite or flags or pin_error_m > 5e-5 or max_strain > 0.25:
        return "FAIL"
    if (
        contact_enabled
        and min_contact_distance_m is not None
        and min_contact_distance_m < 1.8 * contact_radius - 0.0001
    ):
        return "FAIL"
    return "PASS"


def hardware(torch, rs):
    def cmd(args):
        try:
            return subprocess.check_output(
                args, text=True, stderr=subprocess.DEVNULL, timeout=5
            ).strip()
        except (OSError, subprocess.SubprocessError):
            return "unavailable"

    return dict(
        host=platform.node(),
        gpu=torch.cuda.get_device_name(),
        torch=torch.__version__,
        driver=cmd(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]
        ),
        commit=cmd(["git", "rev-parse", "HEAD"]),
        extension=str(rs._ropesim.__file__),
        extension_sha256=sha(rs._ropesim.__file__),
        source_sha256={p.name: sha(p) for p in Path(__file__).parent.glob("*.py")},
    )


def configure(scene, rs, sdf, substeps, iterations, contact=True):
    p = scene["positions"]
    r = scene["radius"]
    b = rs.RopeBatch.from_positions(
        p,
        scene["density"],
        scene["EA"],
        scene["EI"],
        DT / substeps,
        iterations,
        pinned=scene["pinned"],
        fix_root_orientation=scene["fixed_root_orientation"],
        straight_rest=scene["straight_rest"],
    )
    b.damping = 0.0
    b.drag = scene["drag"]
    b.set_gravity(np.array([0, 0, -9.81], np.float32))
    b.set_planes(
        np.array([[0, 0, 1, 0]], np.float32),
        radius=r,
        mu_s=scene["mu"],
        mu_k=scene["mu"],
    )
    enabled = contact and scene["rope_contact"]
    if enabled:
        b.set_rope_contact(True, 0.9 * r, 0.2 * r, 0.025, scene["mu"], 1e7, 1.2, len(p))
    grids = []
    for oi, obj in enumerate(scene["objects"]):
        if obj["type"] == "floor":
            continue
        lo, hi = bounds(obj)
        grid = sdf.from_function(lambda points, o=obj: phi(o, points), lo, hi, 0.004)
        mid = b.add_mesh(*grid.as_args(), pos=np.array(obj["pos"], np.float32))
        grids.append((oi, mid, grid))
    return b, grids, enabled


def execute(args, scenes):
    import torch, ropesim as rs
    from ropesim import sdf

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for run; --list and --describe work on CPU"
        )
    torch.cuda.set_stream(torch.cuda.default_stream())
    out = args.out.resolve()
    out.mkdir(parents=True)
    suite = dict(complete=False, errors=[], cases={}, hardware=hardware(torch, rs))
    write(out / "suite_report.json", suite)
    for scene in scenes:
        dest = out / scene["name"]
        dest.mkdir()
        try:
            case_args = copy.copy(args)
            case_args.substeps = (
                args.substeps
                if args.substeps is not None
                else scene["default_substeps"]
            )
            case_args.iterations = args.iterations if args.iterations is not None else scene["default_iterations"]
            report = run_case(case_args, scene, dest, rs, sdf, torch)
            suite["cases"][scene["name"]] = report
        except Exception:
            detail = traceback.format_exc()
            write(dest / "error.json", dict(error=detail))
            suite["errors"].append(dict(scene=scene["name"], error=detail))
            print(detail, flush=True)
        write(out / "suite_report.json", suite)
    suite["complete"] = not suite["errors"] and all(
        r["completed"] for r in suite["cases"].values()
    )
    write(out / "suite_report.json", suite)
    if not suite["complete"] or any(
        r["status"] == "FAIL" for r in suite["cases"].values()
    ):
        raise RuntimeError(
            "Scene validation failed; recordings and reasons preserved in " + str(out)
        )


def run_case(args, scene, dest, rs, sdf, torch):
    p0 = scene["positions"]
    nr, nv = p0.shape[:2]
    radius = scene["radius"]
    b, grids, contact = configure(
        scene, rs, sdf, args.substeps, args.iterations, args.contact == "on"
    )
    natural = initial_rest_lengths(p0)
    duration = args.seconds if args.seconds is not None else scene["seconds"]
    steps = int(round(duration / DT))
    device = lambda a: torch.as_tensor(
        np.ascontiguousarray(a, np.float32), device="cuda"
    )
    qroot = device(root_frames(scene)) if scene["fixed_root_orientation"] else None
    pairs = pair_indices(nr, nv) if contact else None
    # Pair checks at 10 Hz (not a swept/substep topology guarantee).
    ts = []
    positions = []
    velocities = []
    poses = []
    latencies = []
    diag = []
    flags = []
    finite = True
    stop = None
    maxstrain = pinerror = 0.0
    mindistance = None
    mingap = 1e9
    object_counts = np.zeros(len(scene["objects"]), int)
    minobj = np.full(len(scene["objects"]), 1e9)

    def read():
        return [
            torch.as_tensor(f()).cpu().numpy().copy()
            for f in (b.positions, b.velocities, b.quaternions)
        ]

    def record(t, state, objects):
        ts.append(t)
        positions.append(state[0])
        velocities.append(state[1])
        poses.append(objects.copy())

    state = read()
    record(0, state, object_poses_at(scene, 0))
    last = (0.0, state, object_poses_at(scene, 0))
    for k in range(steps):
        t0 = k * DT
        t = (k + 1) * DT
        objects = object_poses_at(scene, t)
        root0 = roots_at(scene, t0)
        root1 = roots_at(scene, t)
        h = 1e-5
        v0 = (roots_at(scene, t0 + h) - roots_at(scene, max(0, t0 - h))) / (
            h if t0 == 0 else 2 * h
        )
        # Previous observation copies synchronize the default CUDA stream.
        torch.cuda.synchronize()
        before = time.perf_counter_ns()
        for oi, mid, grid in grids:
            if scene["objects"][oi].get("moving"):
                b.set_mesh_poses(
                    mid,
                    np.asarray(objects[oi], np.float32),
                    np.array([0, 0, 0, 1], np.float32),
                )
        if scene["pinned"][:, 0].all():
            b.step(
                device(root0),
                device(v0),
                DT,
                args.substeps,
                root_q=qroot,
                root_pos_end=device(root1),
                root_q_end=qroot,
            )
        else:
            b.step_free(DT, args.substeps)
        torch.cuda.synchronize()
        latencies.append((time.perf_counter_ns() - before) * 1e-6)
        state = read()
        p, v, q = state
        finite = all(np.isfinite(x).all() for x in state)
        if not finite:
            np.savez_compressed(
                dest / "nonfinite_state.npz", t=t, positions=p, velocities=v, q=q
            )
            stop = "nonfinite state"
            if ts[-1] != last[0]:
                record(*last)
            break
        last = (t, state, objects)
        strain = float(
            np.max(np.abs(np.linalg.norm(np.diff(p, axis=1), axis=-1) / natural - 1))
        )
        maxstrain = max(strain, maxstrain)
        expected = p0.copy()
        expected[:, 0] = root1
        mask = scene["pinned"]
        if mask.any():
            pinerror = max(
                pinerror, float(np.linalg.norm(p[mask] - expected[mask], axis=-1).max())
            )
        if b.last_error or b.last_warning:
            flags.append(
                dict(t=t, error=int(b.last_error), warning=int(b.last_warning))
            )
        if k % 12 == 0 or k == steps - 1:
            if pairs is not None:
                a, c = pairs
                flat = p.reshape(-1, 3)
                distance = float(
                    segment_distance(flat[a], flat[a + 1], flat[c], flat[c + 1]).min()
                )
                mindistance = (
                    distance if mindistance is None else min(distance, mindistance)
                )
            objects_now = [
                dict(o, pos=objects[i].tolist()) for i, o in enumerate(scene["objects"])
            ]
            clearance = object_clearance(p, objects_now, radius, phi, max_spacing=0.004)
            mingap = min(mingap, clearance["sample_min_m"])
            for i, obj in enumerate(objects_now):
                gap = float(phi(obj, p.reshape(-1, 3) - objects[i]).min() - radius)
                minobj[i] = min(minobj[i], gap)
                object_counts[i] += gap < 0.001
            diag.append(
                dict(
                    t=t,
                    strain=strain,
                    min_distance_m=mindistance,
                    surface_gap_m=clearance["sample_min_m"],
                )
            )
        if (k + 1) % 4 == 0:
            record(t, state, objects)
        if flags or strain > 0.25:
            stop = "solver error/warning" if flags else "segment strain exceeds 25%"
            if ts[-1] != t:
                record(t, state, objects)
            break
        if (k + 1) % 120 == 0:
            print(
                f"{scene['name']}: {t:.1f}/{duration:.1f} s; max strain {maxstrain:.3%}",
                flush=True,
            )
    if ts[-1] != last[0]:
        record(*last)
    completed = stop is None and len(latencies) == steps
    np.savez_compressed(
        dest / "trajectory.npz",
        t=ts,
        positions=positions,
        velocities=velocities,
        object_positions=poses,
        rest_lengths=natural,
    )
    np.save(dest / "timings_ms.npy", np.asarray(latencies))
    write(dest / "diagnostics.json", diag)
    write(dest / "metadata.json", serializable(scene))
    status = outcome(
        completed=completed,
        finite=finite,
        flags=flags,
        pin_error_m=pinerror,
        max_strain=maxstrain,
        min_contact_distance_m=mindistance,
        contact_radius=radius,
        contact_enabled=contact,
    )
    if status == "PASS" and mingap < -0.001:
        status = "WARN"
    if status == "PASS" and scene["rope_contact"] and not contact:
        status = "CONTROL"
    report = dict(
        name=scene["name"],
        status=status,
        completed=completed,
        stop_reason=stop,
        requested_duration_s=duration,
        recorded_duration_s=ts[-1],
        frames=len(ts),
        trajectory_sha256=sha(dest / "trajectory.npz"),
        settings=dict(
            dt=DT,
            substeps=args.substeps,
            iterations=args.iterations,
            ropes=nr,
            vertices_per_rope=nv,
            rope_contact=contact,
            contact_group_size=nr if contact else None,
            diagnostic_hz=10,
            contact_barrier_wall_m=1.8 * radius,
        ),
        metrics=dict(
            max_strain=maxstrain,
            pin_error_m=pinerror,
            finite=finite,
            flags=flags,
            min_nonlocal_segment_distance_m=mindistance,
            minimum_sampled_surface_gap_m=mingap,
            object_proximity_counts=object_counts.tolist(),
            object_min_vertex_gap_m=minobj.tolist(),
        ),
        timing=summarize_timing(
            latencies, DT, scene["worlds"], min(15, max(0, len(latencies) - 1))
        ),
    )
    if scene["name"] == "datacenter" and completed:
        from .asset_robot import solve_path

        targets = np.array([roots_at(scene, t)[0] for t in ts])
        robot = solve_path(targets, scene["assets"]["robot"]["kinematics"])
        links = robot["link_transforms"][..., :3, 3]
        np.savez_compressed(
            dest / "robot.npz",
            t=ts,
            **robot,
            links=links,
            clamp=targets,
        )
        report["robot_sha256"] = sha(dest / "robot.npz")
        report["metadata_sha256"] = sha(dest / "metadata.json")
        report["task"] = dict(
            scope=scene["robot_scope"],
            robot_model="NVIDIA Isaac UR10e + Robotiq 2F-140 USD assembly",
            max_ik_error_m=float(robot["ik_error"].max()),
            max_orientation_error_rad=float(robot["orientation_error"].max()),
            max_recorded_joint_increment_rad=float(np.abs(np.diff(robot["joints"], axis=0)).max()),
            endpoint_target_error_m=float(np.linalg.norm(targets[-1] - scene["target_position"])),
            pose_success=bool(np.linalg.norm(targets[-1] - scene["target_position"]) < 0.002),
            attachment="kinematic, not contact grasp", connector_mating=False,
            robot_and_rack_collision="visual-only; cable contacts use explicit guide SDFs and floor",
        )
    write(dest / "report.json", report)
    print(scene["name"], status, report["timing"], flush=True)
    return report
