"""CPU contract tests for the portable cascade and its artifact reader."""

import json
import subprocess
import sys

import numpy as np
import pytest


def test_entry_point_help_without_site_packages():
    for module in ("run", "audit", "render"):
        result = subprocess.run(
            [sys.executable, "-E", "-S", "-m", f"demo.cable_cascade.{module}", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        assert "usage:" in result.stdout


@pytest.mark.parametrize(
    ("option", "value"),
    [("--count", "0"), ("--count", "49"), ("--seed", "-1"),
     ("--seconds", "nan"), ("--seconds", ".01"), ("--substeps", "0"),
     ("--iterations", "-1")],
)
def test_invalid_options_fail_before_importing_cuda(tmp_path, option, value):
    out = tmp_path / "uncreated"
    result = subprocess.run(
        [sys.executable, "-E", "-S", "-m", "demo.cable_cascade.run",
         "--out", str(out), option, value],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert option in result.stderr and "error:" in result.stderr
    assert not out.exists()


def test_seeded_reference_has_one_staggered_stack():
    from demo.cable_cascade.model import initial_positions

    positions = initial_positions(48, 0)
    assert positions.shape == (48, 33, 3)
    assert positions.dtype == np.float32
    np.testing.assert_allclose(positions[:3, 16], [
        [-.3, -.195, .65], [-.3, -.195, .695], [-.18, -.195, .74],
    ])
    np.testing.assert_allclose(positions[-1, 16], [.3, .195, 2.765])
    np.testing.assert_array_equal(initial_positions(3, 0), positions[:3])
    assert not np.array_equal(initial_positions(3, 1), positions[:3])


def test_clearance_distinguishes_rod_surface_and_analytic_fixtures():
    from demo.cable_cascade.audit import measure_clearance
    from demo.cable_cascade.model import RADIUS, SPHERE_CENTER, SPHERE_RADIUS

    center_z = SPHERE_CENTER[2]
    positions = np.array([
        [[-.5, 0, center_z], [.5, 0, center_z]],
        [[-.5, .01, center_z], [.5, .01, center_z]],
        [[-.5, 1., RADIUS], [.5, 1., RADIUS]],
    ])
    metrics = measure_clearance(positions)
    assert metrics["min_cross_rod_centerline_distance_m"] == pytest.approx(.01)
    assert metrics["min_cross_rod_surface_gap_m"] == pytest.approx(.01 - 2 * RADIUS)
    assert metrics["min_sphere_surface_gap_m"] == pytest.approx(-SPHERE_RADIUS - RADIUS)
    assert metrics["min_floor_surface_gap_m"] == pytest.approx(0.)


def test_audit_rejects_hash_matched_wrong_frame_clock(tmp_path):
    from demo.cable_cascade.audit import load_trajectory
    from demo.cable_cascade.model import initial_positions, sha

    initial = initial_positions(1, 0)
    trajectory = tmp_path / "trajectory.npz"
    np.savez(trajectory, t=[0., .05], positions=[initial, initial],
             velocities=np.zeros((2, 1, 33, 3), np.float32),
             rest_lengths=np.linalg.norm(np.diff(initial, axis=1), axis=-1))
    (tmp_path / "report.json").write_text(json.dumps({
        "complete": True,
        "settings": {"count": 1, "seed": 0, "seconds": 1 / 30},
        "frames": 2,
        "trajectory_sha256": sha(trajectory),
    }))
    with pytest.raises(ValueError, match="cadence"):
        load_trajectory(tmp_path)


def test_existing_output_is_preserved(tmp_path):
    sentinel = tmp_path / "keep.txt"
    sentinel.write_text("existing data")
    result = subprocess.run(
        [sys.executable, "-E", "-S", "-m", "demo.cable_cascade.run",
         "--out", str(tmp_path)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "fresh directory" in result.stderr
    assert sentinel.read_text() == "existing data"


def test_camera_fits_a_short_run_at_portrait_resolution():
    from demo.cable_cascade._native import camera_path
    from demo.cable_cascade.model import initial_positions

    initial = initial_positions(48, 0)
    # A short recording may end before the camera's usual landing window.
    poses = np.stack([initial, initial - np.array([0, 0, .05])])
    eyes, targets, max_ndc = camera_path(poses, np.array([0., 1 / 30]), 480, 640)
    assert eyes.shape == targets.shape == (2, 3)
    assert np.isfinite(eyes).all() and np.isfinite(targets).all()
    assert (max_ndc < .99).all()
