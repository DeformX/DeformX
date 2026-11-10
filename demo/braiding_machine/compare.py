"""Check the contact-off and parked-carrier controls and the time/space refinement."""
import argparse
import json
from pathlib import Path

import numpy as np

from demo.showcase.simulate import sha
from .analyze import CRITERIA, criteria, diagnostics, endpoint_error, gates, read_run
from .model import PRESETS, Settings


def compare(directories, preset="showcase"):
    refinement = {"showcase": "refined", "long": "long-refined", "process": "process-refined"}[preset]
    cases = {}
    for name, path in directories.items():
        run, data = read_run(path)
        case_preset = refinement if name == "refined" else preset
        cfg = PRESETS[case_preset]
        expected_drive, expected_contact = name != "no_drive", name != "no_contact"
        if (run["preset"], run["drive"], run["contact"]) != (case_preset, expected_drive, expected_contact):
            raise ValueError(f"wrong run mode for {name}")
        if Settings(**run["settings"]) != cfg or run["criteria_sha256"] != sha(CRITERIA):
            raise ValueError(f"wrong settings or criteria for {name}")
        metrics = diagnostics(data["t"], data["positions"], cfg)
        metrics["max_endpoint_error_m"] = endpoint_error(data["positions"], data["carriers"])
        checks = gates(metrics, cfg, True, run["drive"], run["contact"])
        if all(checks.values()) != run["task_success"]:
            raise ValueError(f"report task result does not match saved states for {name}")
        cases[name] = run, data, metrics

    nominal, disabled, parked, refined = (cases[k][2] for k in ("nominal", "no_contact", "no_drive", "refined"))
    n_run, n_data, _ = cases["nominal"]
    f_run, f_data, _ = cases["refined"]
    f_stride = PRESETS[refinement].step_hz // PRESETS[preset].step_hz
    same_sources = all(
        run["source_sha256"] == n_run["source_sha256"]
        and run["shared_diagnostics_sha256"] == n_run["shared_diagnostics_sha256"]
        for run, _, _ in cases.values()
    )
    same_backend = all(run["backend"]["extension_sha256"] == n_run["backend"]["extension_sha256"]
                       for run, _, _ in cases.values())
    paired = zip(nominal["checkpoints"], refined["checkpoints"])
    matches = all(
        a["t_s"] == b["t_s"]
        and a["measured_word"] == b["measured_word"]
        and [x["pair"] for x in a["crossings"]] == [x["pair"] for x in b["crossings"]]
        for a, b in paired
    )
    shifts = [
        abs(a["z_m"] - b["z_m"])
        for nominal_stage, refined_stage in zip(nominal["checkpoints"], refined["checkpoints"])
        for a, b in zip(nominal_stage["crossings"], refined_stage["crossings"])
    ]
    bounds = criteria()
    delta = dict(
        max_crossing_z_delta_m=max(shifts) if shifts and matches else None,
        min_interstrand_centerline_delta_m=abs(
            nominal["min_interstrand_centerline_m"] - refined["min_interstrand_centerline_m"]
        ),
        peak_strain_delta=abs(nominal["max_abs_segment_strain"] - refined["max_abs_segment_strain"]),
    )
    checks = dict(
        same_source_and_backend=same_sources and same_backend,
        nominal_success=n_run["task_success"],
        matched_contact_off_drive=np.array_equal(n_data["carriers"], cases["no_contact"][1]["carriers"]),
        contact_off_loses_order=any(
            stage["measured_word"] != stage["expected_word"] for stage in disabled["checkpoints"]
        ),
        contact_off_loses_clearance=disabled["min_interstrand_centerline_m"] <= bounds["contact_off_max_min_centerline_m"],
        parked_control_no_braid=all(
            not stage["measured_word"] and stage["reason"] is None for stage in parked["checkpoints"]
        ) and not any(parked["pairs_in_barrier"].values()),
        refinement_success=f_run["task_success"],
        matched_refinement_drive=np.array_equal(n_data["carriers"], f_data["carriers"][::f_stride]),
        matched_refinement_order=matches,
        bounded_crossing_z_delta=delta["max_crossing_z_delta_m"] is not None
        and delta["max_crossing_z_delta_m"] <= bounds["refinement_max_crossing_z_delta_m"],
        bounded_min_interstrand_centerline_delta=delta["min_interstrand_centerline_delta_m"]
        <= bounds["refinement_max_interstrand_clearance_delta_m"],
        bounded_peak_strain_delta=delta["peak_strain_delta"] <= bounds["refinement_max_peak_strain_delta"],
    )
    return dict(
        complete=True, accepted=all(checks.values()), checks=checks, refinement_delta=delta,
        criteria_sha256=sha(CRITERIA),
        inputs={name: dict(trajectory_sha256=run["trajectory_sha256"],
                           report_sha256=sha(directories[name] / "report.json"),
                           task_success=run["task_success"])
                for name, (run, _, _) in cases.items()},
        limit="One combined space/time refinement measures sensitivity for this case. It does not isolate the two error sources or establish convergence.",
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("nominal", "no_contact", "no_drive", "refined"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True, help="run directory")
    parser.add_argument("--preset", choices=("showcase", "long", "process"), default="showcase",
                        help="carrier schedule shared by the nominal and control runs")
    args = parser.parse_args(argv)
    directories = {name: getattr(args, name) for name in ("nominal", "no_contact", "no_drive", "refined")}
    result = compare(directories, args.preset)
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0 if result["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
