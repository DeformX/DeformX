"""Contracts for the traction reel transport example."""
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from demo.winch_tension import analysis, model, run


def test_native_world_wrench_has_cable_on_axle_sign_and_reference_point():
    # Cable pulling +x at the top of a roller drives positive rotation about +y.
    wrench = [4.0, 0.0, -2.0, 0.0, 0.3, 0.0]
    torque = model.axle_torque(wrench, [0.0, 0.0, 0.2], [0.0, 0.0, 0.0])
    assert torque == pytest.approx(1.1)
    axle = model.PassiveAxle(0.01, 0.2)
    axle.step(torque, 0.005)
    assert axle.omega == pytest.approx(0.55)
    assert axle.angle == pytest.approx(0.00275)
    axle.step(0.0, 0.005)
    assert axle.omega == pytest.approx(0.495)


def phase_record():
    times = np.linspace(0.0, 3.8, 761)
    knots = [0.0, 0.55, 1.8, 2.4, 3.65, 3.8]
    positions = np.zeros((len(times), 3, 3))
    positions[:, 1, 0] = np.interp(times, knots, [0, 0, .075, .077, -.002, -.002])
    angles = np.zeros((len(times), 2))
    angles[:, 1] = np.interp(times, knots, [0, 0, .80, .81, -.02, -.02])
    velocity = np.gradient(angles[:, 1], times)
    applied = .005 * np.sign(velocity)
    wrench = np.zeros((len(times), 2, 6))
    wrench[:, :, 2] = -.5
    wrench[1:, 1, 4] = applied[1:]
    forces = np.zeros_like(positions)
    forces[:, 1, 2] = 1.0
    return dict(t=times, positions=positions, reel_angle_rad=angles,
                guide_angular_velocity_rad_s=velocity, reel_wrenches=wrench,
                guide_applied_torque_Nm=applied, contact_forces_N=forces)


def test_success_uses_measured_material_travel_dynamic_guide_and_loads():
    recorded = phase_record()
    assert analysis.measure(recorded)["task_success"]
    assert analysis.measure(recorded)["force_balance_max_N"] == pytest.approx(0.0)
    no_cable_travel = dict(recorded, positions=np.zeros_like(recorded["positions"]))
    no_guide_travel = dict(recorded, reel_angle_rad=np.zeros_like(recorded["reel_angle_rad"]))
    no_applied_torque = dict(recorded, guide_applied_torque_Nm=np.zeros_like(recorded["t"]))
    no_contact = dict(recorded, reel_wrenches=np.zeros_like(recorded["reel_wrenches"]))
    incomplete = {key: value[:-80] for key, value in recorded.items()}
    for rejected in [no_cable_travel, no_guide_travel, no_applied_torque, no_contact, incomplete]:
        assert not analysis.measure(rejected)["task_success"]


def test_cycle_program_keeps_warmup_once_and_restarts_after_hold():
    settings = model.Settings(cycles=8)
    assert model.Settings().duration_s == 3.8
    assert settings.duration_s == pytest.approx(26.55)
    assert settings.motor_angle(3.75) == pytest.approx(0.)
    assert settings.motor_angle(4.1) == pytest.approx(settings.motor_angle(.85))
    assert settings.motor_angle(26.55) == pytest.approx(0.)
    assert model.phase(3.79, cycles=8) == "Hold"
    assert model.phase(3.8, cycles=8) == "Feed"
    assert model.phase(5.05, cycles=8) == "Pause"
    assert model.phase(5.65, cycles=8) == "Reverse"


def test_all_requested_cycles_must_pass_and_finish():
    first = phase_record()
    repeat = first["t"] > .55
    recorded = {key: np.concatenate((value, value[repeat])) for key, value in first.items()}
    recorded["t"][len(first["t"]):] += 3.25
    assert analysis.measure(recorded, cycles=2)["task_success"]

    late_failure = dict(recorded, reel_wrenches=recorded["reel_wrenches"].copy())
    late_failure["reel_wrenches"][recorded["t"] > 3.8] = 0.
    result = analysis.measure(late_failure, cycles=2)
    assert result["evaluation_eligible"] and not result["task_success"]
    assert [cycle["task_success"] for cycle in result["cycles"]] == [True, False]
    assert not result["cycles"][1]["checks"]["both_reels_loaded"]

    truncated = {key: value[recorded["t"] < 6.9] for key, value in recorded.items()}
    result = analysis.measure(truncated, cycles=2)
    assert not result["evaluation_eligible"] and not result["task_success"]
    assert [cycle["evaluation_eligible"] for cycle in result["cycles"]] == [True, False]

    reverse_end = int(np.flatnonzero(np.isclose(recorded["t"], 6.9))[0])
    rejected = {key: value[:reverse_end + 1] for key, value in recorded.items()}
    assert analysis.measure(rejected, cycles=2)["task_success"]
    result = analysis.measure(rejected, cycles=2, rejected_sample_index=reverse_end)
    assert [cycle["evaluation_eligible"] for cycle in result["cycles"]] == [True, False]
    assert not result["task_success"]


@pytest.mark.parametrize("args", [
    ["--speed-m-s", "nan"], ["--speed-m-s", "-1"],
    ["--duration-s", "inf"], ["--duration-s", "0.011"],
    ["--exchange-hz", "0"], ["--rod-hz", "7"],
    ["--cycles", "0"], ["--cycles", "9"],
    ["--cycles", "2", "--duration-s", "7.1"],
])
def test_cli_rejects_invalid_settings_before_creating_output(tmp_path, args):
    out = tmp_path / "trial"
    with pytest.raises(SystemExit) as exc:
        run.main(["--out", str(out), *args])
    assert exc.value.code == 2
    assert not out.exists()


def test_incomplete_trial_is_bound_to_preserved_trajectory(tmp_path, monkeypatch):
    def failing_backend(settings, frames, report):
        assert not json.loads((out / "report.json").read_text())["complete"]
        raise RuntimeError("intentional backend failure")

    monkeypatch.setattr(run, "simulate", failing_backend)
    out = tmp_path / "trial"
    with pytest.raises(RuntimeError, match="intentional backend failure"):
        run.execute(out, model.Settings())
    report_before = (out / "report.json").read_bytes()
    trajectory_before = (out / "trajectory.npz").read_bytes()
    report = json.loads(report_before)
    assert not report["complete"]
    assert not report["metrics"]["task_success"]
    assert "intentional backend failure" in report["error"]
    assert report["trajectory_sha256"] == analysis.sha256(out / "trajectory.npz")
    with np.load(out / "trajectory.npz") as trajectory:
        assert trajectory["t"].tolist() == [0.0]
        assert np.isfinite(trajectory["positions"]).all()
    with pytest.raises(FileExistsError):
        run.execute(out, model.Settings())
    assert (out / "report.json").read_bytes() == report_before
    assert (out / "trajectory.npz").read_bytes() == trajectory_before


@pytest.mark.parametrize("failure", ["warning", "error", "strain", "nonfinite"])
def test_incomplete_trial_marks_a_retained_rejected_sample(tmp_path, monkeypatch, failure):
    # Inject a specific backend failure after one valid exchange. The actual
    # simulation loop and file writer decide which frames to keep and label.
    class FakeRod:
        last_warning = last_error = 0
        step = 0

        def __init__(self, positions, *args, **kwargs):
            self.points = positions.copy()

        def ignore(self, *args, **kwargs):
            pass

        set_masses = set_gravity = set_contact = enable_contact_forces = ignore
        add_mesh = set_mesh_poses_all = set_velocities = ignore

        def step_free(self, dt, substeps):
            self.step += 1
            if self.step != 2:
                return
            if failure == "warning":
                self.last_warning = 7
            elif failure == "error":
                self.last_error = 1
            elif failure == "strain":
                self.points[0, len(self.points[0]) // 2, 2] += 0.1
            else:
                self.points[0, 0, 2] = np.nan

        def inv_masses(self):
            return np.ones(self.points.shape[:2])

        def positions(self):
            return self.points

        def velocities(self):
            return np.zeros_like(self.points)

        def mesh_wrenches(self):
            return np.zeros((1, 2, 6))

        contact_forces = velocities

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        __version__="test", version=SimpleNamespace(cuda="test"),
        cuda=SimpleNamespace(is_available=lambda: True, get_device_name=lambda _: "test",
                             synchronize=lambda: None),
        as_tensor=lambda value: SimpleNamespace(cpu=lambda: SimpleNamespace(numpy=lambda: value)),
    ))
    grid = SimpleNamespace(values=np.zeros(1), as_args=lambda: ())
    monkeypatch.setitem(sys.modules, "ropesim", SimpleNamespace(
        _ropesim=SimpleNamespace(__file__=__file__), RopeBatch=SimpleNamespace(from_positions=FakeRod),
        sdf=SimpleNamespace(from_function=lambda *args: grid),
    ))

    out = tmp_path / "trial"
    with pytest.raises(RuntimeError):
        run.execute(out, model.Settings())
    report, frames = analysis.load_recording(out)
    assert not report["complete"]
    assert not report["metrics"]["task_success"]
    assert np.isfinite(frames["positions"]).all()
    if failure == "nonfinite":
        assert report["sample_count"] == len(frames["t"]) == 2
        assert "rejected_sample_index" not in report
        assert "rejected_sample_time_s" not in report
    else:
        assert report["sample_count"] == len(frames["t"]) == 3
        assert report["rejected_sample_index"] == 2
        assert report["rejected_sample_time_s"] == frames["t"][2] == pytest.approx(2 / 240)
