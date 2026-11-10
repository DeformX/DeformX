"""Contracts for the portable beginner CLIs and the physical goal task."""

import json
import os
import subprocess
import sys

import numpy as np
import pytest


def cli(module, *args):
    return subprocess.run(
        [sys.executable, "-m", f"demo.basics.{module}", *map(str, args)],
        capture_output=True, text=True, env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
    )


@pytest.mark.parametrize("module", ["run", "task", "plot"])
def test_help_needs_only_the_standard_library(module):
    code = """
import runpy, sys

class BlockSim:
    def find_spec(self, name, *args):
        if name.split('.')[0] in {'torch', 'ropesim', 'isaacsim', 'omni', 'numpy', 'matplotlib'}:
            raise AssertionError(f'help tried to import {name}')

sys.meta_path.insert(0, BlockSim())
sys.argv = [sys.argv[1], '--help']
runpy.run_module(sys.argv[0], run_name='__main__')
"""
    result = subprocess.run(
        [sys.executable, "-c", code, f"demo.basics.{module}"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout


@pytest.mark.parametrize(
    "module,args,phrase",
    [
        ("run", ("--case", "hanging", "--seconds", "nan"), "seconds"),
        ("run", ("--case", "hanging", "--substeps", "0"), "substeps"),
        ("run", ("--case", "hanging", "--contact", "off"), "contact"),
        ("task", ("--max-steps", "0"), "max_steps"),
        ("task", ("--tolerance-m", "0.2"), "tolerance"),
        ("task", ("--seed", "-1"), "seed"),
    ],
)
def test_rejects_bad_cli_values_before_output(tmp_path, module, args, phrase):
    out = tmp_path / "new"
    result = cli(module, *args, "--out", out)
    assert result.returncode == 2, result.stderr
    assert phrase in result.stderr and "CUDA is required" not in result.stderr
    assert not out.exists()


@pytest.mark.parametrize("module,args", [("run", ("--case", "hanging")), ("task", ())])
def test_existing_output_is_preserved(tmp_path, module, args):
    marker = tmp_path / "report.json"
    marker.write_text('{"keep": true}')
    result = cli(module, *args, "--out", tmp_path)
    assert result.returncode != 0 and "already exists" in result.stderr
    assert json.loads(marker.read_text()) == {"keep": True}
    assert list(tmp_path.iterdir()) == [marker]


def test_missing_cuda_names_dependency_and_leaves_output_fresh(tmp_path):
    out = tmp_path / "new"
    result = cli("run", "--case", "hanging", "--out", out)
    assert result.returncode != 0
    assert "CUDA is required" in result.stderr
    assert "Traceback" not in result.stderr
    assert not out.exists()


@pytest.fixture
def task_type():
    from demo.basics.task import TipTargetTask

    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("Task simulation requires CUDA")
    pytest.importorskip("ropesim")
    return TipTargetTask


def test_reset_repeats_seed_and_discards_previous_state(task_type):
    task = task_type()
    obs, info = task.reset(seed=41)
    assert obs["positions"].shape == obs["velocities"].shape
    assert obs["positions"].shape[-1] == 3
    assert info["step"] == 0 and not info["success"]
    moved, _, _, _, _ = task.step([1, 0])
    assert moved["positions"][0, 0] > obs["positions"][0, 0]
    repeated, _ = task.reset(seed=41)
    for key in ("positions", "velocities", "target"):
        np.testing.assert_array_equal(obs[key], repeated[key])
    different, _ = task.reset(seed=42)
    assert np.linalg.norm(different["target"] - obs["target"]) > 0.005


def test_invalid_action_does_not_advance_or_clip(task_type):
    task = task_type(max_steps=1)
    task.reset(seed=0)
    for action in ([0], [np.nan, 0], [1.1, 0], [1.0, 1.0]):
        with pytest.raises(ValueError, match="action"):
            task.step(action)
    _, _, terminated, truncated, info = task.step([0, 0])
    assert info["step"] == 1
    assert not terminated and truncated
    with pytest.raises(RuntimeError, match="reset"):
        task.step([0, 0])


def test_scoring_tracks_measured_tip_and_zero_action_cannot_succeed(task_type):
    task = task_type(max_steps=12, tolerance_m=0.012)
    obs, first = task.reset(seed=0)
    assert first["error_m"] > 0.05
    for index in range(12):
        obs, reward, terminated, truncated, info = task.step([0, 0])
        measured = np.linalg.norm(obs["positions"][-1] - obs["target"])
        assert measured == pytest.approx(info["error_m"])
        assert reward == pytest.approx(-measured)
        assert measured > 0.012
        assert not info["success"] and not terminated
        assert truncated == (index == 11)


def test_feedback_solves_by_moving_the_free_tip(task_type):
    from demo.basics.task import feedback_action

    task = task_type(tolerance_m=0.012)
    obs, first = task.reset(seed=0)
    root, tip = obs["positions"][[0, -1]].copy()
    while True:
        obs, _, terminated, truncated, info = task.step(feedback_action(obs))
        if terminated or truncated:
            break
    assert terminated and info["success"] and not truncated
    assert info["error_m"] <= 0.012 < first["error_m"]
    assert np.linalg.norm(obs["positions"][-1] - tip) > 0.04
    assert np.linalg.norm(obs["positions"][0] - root) > 0.01


def test_materials_steps_both_rods_and_records_different_tip_deflections(task_type, tmp_path):
    from demo.basics.run import main

    out = tmp_path / "materials"
    assert main(["--case", "materials", "--seconds", "0.5", "--out", str(out)]) == 0
    report = json.loads((out / "report.json").read_text())
    assert report["complete"]
    with np.load(out / "trajectory.npz", allow_pickle=False) as data:
        p, t = data["positions"], data["t"]
        assert p.shape == (61, 2, 33, 3)
        assert np.isfinite(p).all()
        assert t[-1] == pytest.approx(0.5)
        np.testing.assert_array_equal(p[-1, :, 0], p[0, :, 0])
        drop = p[0, :, -1, 2] - p[-1, :, -1, 2]
        assert drop[0] > drop[1] + 0.015


@pytest.fixture
def legacy_backend(monkeypatch):
    from types import ModuleType, SimpleNamespace

    class RopeBatch:
        @staticmethod
        def from_positions(*args, **kwargs):
            """from_positions(positions: numpy.ndarray, pinned: object = None) -> RopeBatch"""
            raise AssertionError("legacy constructor should not be called")

    torch = ModuleType("torch")
    torch.cuda = SimpleNamespace(is_available=lambda: True)
    ropesim = ModuleType("ropesim")
    ropesim.RopeBatch = RopeBatch
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "ropesim", ropesim)


@pytest.mark.parametrize("module,args", [("run", ["--case", "hanging"]), ("task", [])])
def test_legacy_backend_cli_explains_rebuild_before_creating_output(legacy_backend, tmp_path, capsys, module, args):
    from importlib import import_module

    out = tmp_path / "new"
    with pytest.raises(SystemExit) as exc:
        import_module(f"demo.basics.{module}").main([*args, "--out", str(out)])
    assert exc.value.code == 2
    error = capsys.readouterr().err
    assert "deterministic" in error
    assert "python -m pip install -e Cosserat-Rod-Solver-CUDA" in error
    assert not out.exists()


def test_legacy_backend_task_api_explains_rebuild(legacy_backend):
    from demo.basics.task import TipTargetTask

    with pytest.raises(RuntimeError, match=r"pip install -e Cosserat-Rod-Solver-CUDA"):
        TipTargetTask().reset(seed=0)


@pytest.mark.parametrize(
    "case,settings,complete,required,forbidden",
    [
        ("self_contact", {"contact_enabled": True}, True,
         ["Self-contact on", "Barrier onset"], ["Rod diameter reference"]),
        ("self_contact", {"contact_enabled": False}, False,
         ["Self-contact off (control)", "Incomplete recording", "Rod diameter reference"],
         ["Barrier onset"]),
        ("obstacle", {"contact_enabled": False}, True,
         ["Obstacle contact off (control)", "Sphere (reference)"], ["Incomplete recording"]),
        ("tip_target", {"policy": "feedback"}, True, ["Feedback policy"], ["Incomplete recording"]),
        ("tip_target", {"policy": "zero"}, True, ["Zero-action policy (control)"], ["Feedback policy"]),
    ],
)
def test_standalone_plot_labels_reflect_recording(tmp_path, case, settings, complete, required, forbidden):
    import hashlib
    from demo.basics.plot import main

    pytest.importorskip("matplotlib")
    p = np.array([[[[0, 0, 0.02], [0.01, 0.01, 0.02], [0.02, 0.01, 0.02]]],
                  [[[0, 0, 0.02], [0.01, 0.01, 0.015], [0.02, 0.015, 0.015]]]])
    trajectory = tmp_path / "trajectory.npz"
    np.savez(trajectory, t=[0, 0.01], positions=p, nonlocal_distance_m=[0.012, 0.009],
             mesh_wrench=np.zeros((2, 6)), target=[0.03, 0.02, 0.02], error_m=[0.03, 0.02])
    (tmp_path / "report.json").write_text(json.dumps(dict(
        complete=complete, trajectory_sha256=hashlib.sha256(trajectory.read_bytes()).hexdigest(),
        settings=dict(case=case, pinned_vertices=[[0]], radius_m=0.005,
                      obstacle=dict(radius=0.01, pos=[0.01, 0, 0]), tolerance_m=0.012, **settings),
    )))
    out = tmp_path / "plot.svg"
    assert main([str(tmp_path), "--out", str(out)]) == 0
    svg = out.read_text()
    for label in required:
        assert label in svg
    for label in forbidden:
        assert label not in svg
