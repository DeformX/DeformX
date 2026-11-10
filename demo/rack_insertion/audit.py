"""Audit contact acquisition and insertion using the recorded 30 Hz physical states."""

import argparse
import json
from pathlib import Path

from .assets import sha
from .profile import load_profile


def verify_recording(folder):
    report_path = folder / "report.json"
    if not report_path.is_file():
        raise ValueError(f"Missing {report_path}. Run the rack insertion example first.")
    report = json.loads(report_path.read_text())
    if not report["complete"] or not report["finite"] or report["errors"] or not report["app_close_invoked"]:
        raise ValueError("Incomplete or nonfinite physical recording")
    for name, field in (
        ("scene.usda", "scene_sha256"), ("trajectory.npz", "trajectory_sha256"),
        ("settling.npz", None),
    ):
        expected = report[field] if field else report["initialization"]["settling_sha256"]
        if sha(folder / name) != expected:
            raise ValueError(f"Recording checksum mismatch for {name}")
    for name, expected in report["source_sha256"].items():
        if sha(folder / "source" / name) != expected:
            raise ValueError(f"Recording source checksum mismatch for {name}")
    profile = report["portable_profile"]
    if load_profile(profile["path"]) != profile:
        raise ValueError("Recorded asset profile no longer matches")
    return report


def audit(folder):
    report = verify_recording(folder)
    import numpy as np
    from .contracts import score_acquisition, score_insertion

    if sha(Path(__file__).with_name("contracts.py")) != report["source_sha256"]["contracts.py"]:
        raise ValueError("The active acceptance criteria differ from the recorded criteria")
    cable = report["cable"]
    seated_depth = report["connector"]["seated_depth_m"]
    seating = report["controller"]["seating_observation"] or {}
    panel_grid = next(grid for grid in cable["colliders"] if grid["path"].endswith("/PatchPanelPhysical"))
    aperture = panel_grid["aperture_grid_probe"]["phi_m"]
    with np.load(folder / "trajectory.npz", allow_pickle=False) as record:
        t = record["t"]
        if t.ndim != 1 or len(t) < 2 or len(t) != report["recorded_frames"] or not np.allclose(
            np.diff(t), 1 / 30, atol=1e-8, rtol=0
        ):
            raise ValueError("Uniform increasing 30 Hz trajectory required")
        for name in record.files:
            values = record[name]
            if values.dtype.kind in "fci" and (len(values) != len(t) or not np.isfinite(values).all()):
                raise ValueError(f"Invalid recorded values in {name}")
        phase, depth = record["phase"], record["depth"]
        left, right = record["left_force"], record["right_force"]
        acquisition = score_acquisition(t, record["plug_pos"][:, 2], left, right, phase)
        insertion = score_insertion(
            t, depth, record["lateral_error"], record["axis_error"], record["socket_force"],
            np.maximum(left, right), phase, seated_depth, tcp_z=record["tcp"][:, 2],
        )
        preload = record["seating_preload_m"]
        increments = np.diff(preload)
        insert_depth = depth[phase == "insert"]
        seating_rate = float(np.max(increments / np.diff(t)))
        separation = float(record["min_contact_separation"].min())
        phases = list(dict.fromkeys(phase.tolist()))
        fixture_forces = record["fixture_pair_force"]
        insert_or_release = np.isin(phase, ["insert", "release"])
        acquisition_matches = acquisition == report["acquisition"]
        insertion_matches = all(insertion[key] == report["insertion"][key] for key in insertion)
        result = {
            "report_sha256": sha(folder / "report.json"),
            "trajectory_sha256": report["trajectory_sha256"],
            "frames": len(t),
            "duration_s": float(t[-1]),
            "phases": phases,
            "acquisition": acquisition,
            "insertion": insertion,
            "checks": {
                "acquisition": acquisition["passed"],
                "insertion_retention_after_withdrawal": insertion["passed"],
                "routing_and_insertion": report["routing"]["attempted"] and report["insertion"]["attempted"],
                "no_controller_failure": not report["controller"]["failures"],
                "no_attachment_or_pose_writes": (
                    report["gripper_attachment_constraints"] == report["post_reset_object_pose_writes"] == 0
                    and cable["pinned_vertices"] == 0
                    and report["initialization"]["support_joint_disabled_before_trial"]
                ),
                "bounded_seating": bool(
                    np.min(increments) >= -1e-12 and seating_rate <= .00025 + 1e-10
                    and preload.max() <= .0005 + 1e-12 and len(insert_depth)
                    and insert_depth.max() <= seated_depth + .00075
                    and seating.get("stop_reason") == "stable measured socket contact"
                    and seating.get("contact_dwell_s", 0.) >= .1 - 1e-9
                    and seating.get("socket_force_N", 0.) >= .01
                ),
                "aperture_and_web": (panel_grid["closed"] and panel_grid["thickness_m"] == 0.
                                     and all(aperture[i] > .002 for i in (0, 2, 4))
                                     and all(aperture[i] < 0. for i in (1, 3))),
                "separation_and_gap": separation >= -.001 and cable["peak_termination_gap_m"] <= .005,
                "matches_recorded_scores": acquisition_matches and insertion_matches,
            },
            "maximum_preload_target_m": float(preload.max()),
            "sampled_preload_max_rate_m_s": seating_rate,
            "minimum_sampled_contact_separation_m": separation,
            "peak_termination_gap_m": cable["peak_termination_gap_m"],
            "peak_segment_strain": cable["max_strain"],
            "peak_fixture_pair_force_N": float(fixture_forces.max()),
            "insert_release_peak_fixture_force_N": float(fixture_forces[insert_or_release].max(initial=0.)),
            "simulation_loop_wall_s": report["simulation_loop_wall_s"],
        }
    result["passed"] = all(result["checks"].values())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Simulation output directory")
    parser.add_argument("--out", type=Path, help="Optional new JSON audit output")
    args = parser.parse_args()
    try:
        result = audit(args.data.resolve())
        text = json.dumps(result, indent=2, allow_nan=False) + "\n"
        if args.out:
            with args.out.open("x") as stream:
                stream.write(text)
    except (OSError, ValueError, KeyError, ImportError) as exc:
        parser.error(str(exc))
    print(text, end="")
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
