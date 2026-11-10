"""Measured transport criteria and offline comparison of bound trajectories."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from . import model

CONTACT_THRESHOLD_N = .05
CONTACT_FRACTION_MIN = .95
CABLE_TRAVEL_MIN_M = .025
GUIDE_TRAVEL_MIN_RAD = .22
PAUSE_CABLE_EXCURSION_MAX_M = .012
PAUSE_GUIDE_EXCURSION_MAX_RAD = .16
GUIDE_WORK_MIN_J = .0002


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_recording(folder):
    folder = Path(folder)
    report = json.loads((folder / "report.json").read_text())
    trajectory = folder / "trajectory.npz"
    if sha256(trajectory) != report.get("trajectory_sha256"):
        raise ValueError(f"Trajectory hash mismatch for {folder}")
    with np.load(trajectory, allow_pickle=False) as archive:
        frames = {key: archive[key] for key in archive.files}
    return report, frames


def measure(frames, cycles=1, rejected_sample_index=None):
    t = frames["t"]
    last_valid_t = t[-1] if rejected_sample_index is None else t[rejected_sample_index - 1]
    mid_x = frames["positions"][:, frames["positions"].shape[1] // 2, 0]
    angle = frames["reel_angle_rad"][:, 1]
    omega = frames["guide_angular_velocity_rad_s"]
    native = frames["reel_wrenches"]
    applied = frames["guide_applied_torque_Nm"]
    force = np.linalg.norm(native[:, :, :3], axis=-1)
    balance = frames["contact_forces_N"].sum(axis=1, dtype=np.float64) + native[:, :, :3].sum(axis=1)
    work = applied[1:] * (omega[1:] + omega[:-1]) * .5 * np.diff(t)

    result = dict(max_contact_force_N=force.max(axis=0).tolist(),
                  max_abs_guide_torque_Nm=float(np.max(np.abs(native[:, 1, 4]))),
                  max_abs_force_balance_component_N=float(np.max(np.abs(balance))),
                  force_balance_max_N=float(np.linalg.norm(balance, axis=-1).max()),
                  final_material_x_m=float(mid_x[-1]), final_guide_angle_rad=float(angle[-1]))
    def movement(values, a, b):
        return float(np.interp(b, t, values) - np.interp(a, t, values))

    def excursion(values, a, b):
        active = values[(t >= a - 1e-9) & (t <= b + 1e-9)]
        return float(np.max(np.abs(active - np.interp(a, t, values))))

    def driven_phase(a, b):
        active = (t > a + 1e-9) & (t <= b + 1e-9)
        return dict(cable_dx_m=movement(mid_x, a, b), guide_dtheta_rad=movement(angle, a, b),
                    loaded_fraction_per_reel=(force[active] > CONTACT_THRESHOLD_N).mean(axis=0).tolist(),
                    guide_contact_work_J=float(work[active[1:]].sum()))

    evaluated = np.zeros(len(t), dtype=bool)
    summaries = []
    for index in range(cycles):
        offset = index * model.CYCLE_PERIOD_S
        start = model.FEED_START_S + offset
        feed_end = model.FEED_END_S + offset
        reverse_start = model.REVERSE_START_S + offset
        stop = model.REVERSE_END_S + offset
        eligible = bool(last_valid_t >= stop - 1e-9)
        cycle = dict(cycle=index + 1, feed_start_s=start, reverse_end_s=stop,
                     evaluation_eligible=eligible, task_success=False)
        summaries.append(cycle)
        if not eligible:
            continue
        active = (t > start + 1e-9) & (t <= stop + 1e-9)
        evaluated |= active
        feed = driven_phase(start, feed_end)
        reverse = driven_phase(reverse_start, stop)
        pause = dict(cable_excursion_m=excursion(mid_x, feed_end, reverse_start),
                     guide_excursion_rad=excursion(angle, feed_end, reverse_start))
        checks = dict(
            cable_follows_program=feed["cable_dx_m"] > CABLE_TRAVEL_MIN_M and reverse["cable_dx_m"] < -CABLE_TRAVEL_MIN_M,
            guide_follows_cable=feed["guide_dtheta_rad"] > GUIDE_TRAVEL_MIN_RAD and reverse["guide_dtheta_rad"] < -GUIDE_TRAVEL_MIN_RAD,
            bounded_pause=pause["cable_excursion_m"] < PAUSE_CABLE_EXCURSION_MAX_M and pause["guide_excursion_rad"] < PAUSE_GUIDE_EXCURSION_MAX_RAD,
            both_reels_loaded=min(feed["loaded_fraction_per_reel"] + reverse["loaded_fraction_per_reel"]) > CONTACT_FRACTION_MIN,
            positive_contact_work=min(feed["guide_contact_work_J"], reverse["guide_contact_work_J"]) > GUIDE_WORK_MIN_J,
        )
        cycle.update(feed=feed, pause=pause, reverse=reverse, checks=checks, task_success=all(checks.values()),
                     active_window_guide_torque_abs_peak_Nm=float(np.max(np.abs(native[active, 1, 4]))))

    result.update(cycles=summaries, eligible_cycles=sum(c["evaluation_eligible"] for c in summaries),
                  evaluation_eligible=all(c["evaluation_eligible"] for c in summaries),
                  task_success=all(c["task_success"] for c in summaries))
    if evaluated.any():
        torque = native[evaluated, 1, 4]
        result.update(active_window_rms_force_per_reel_N=np.sqrt(np.mean(force[evaluated] ** 2, axis=0)).tolist(),
                      active_window_guide_torque_rms_Nm=float(np.sqrt(np.mean(torque ** 2))),
                      active_window_guide_torque_abs_p95_Nm=float(np.quantile(np.abs(torque), .95)),
                      active_window_guide_torque_abs_peak_Nm=float(np.max(np.abs(torque))))
    if cycles == 1 and result["evaluation_eligible"]:
        result.update({key: summaries[0][key] for key in ("feed", "pause", "reverse", "checks")})
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path, help="Run directories to compare")
    args = parser.parse_args(argv)
    results = {}
    for folder in args.runs:
        report, frames = load_recording(folder)
        metrics = measure(frames, cycles=report["settings"].get("cycles", 1),
                          rejected_sample_index=report.get("rejected_sample_index"))
        metrics["task_success"] &= report["complete"]
        results[str(folder)] = dict(complete=report["complete"], settings=report["settings"], metrics=metrics)
    print(json.dumps(results, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
