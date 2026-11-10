"""Check the rack entry points with Python's site packages disabled."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]


def cli(module, *args, runtime=None):
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    if runtime is not None:
        env["PYTHONPATH"] = str(runtime)
    return subprocess.run(
        [sys.executable, "-S", "-m", f"demo.rack_insertion.{module}", *map(str, args)],
        cwd=ROOT, env=env, text=True, capture_output=True, timeout=10,
    )


@pytest.mark.parametrize("module, option", [
    ("assets", "--out"), ("prepare", "--sources"), ("run", "--profile"),
    ("audit", "--data"), ("render", "--data"),
])
def test_help_without_runtime(module, option):
    result = cli(module, "--help")
    assert result.returncode == 0, result.stderr
    assert option in result.stdout


def test_missing_profile_fails_before_simulator_start(tmp_path):
    output = tmp_path / "run"
    result = cli("run", "--profile", tmp_path / "missing.json", "--out", output)
    assert result.returncode == 2
    assert "demo.rack_insertion.prepare" in result.stderr
    assert not output.exists()


def test_invalid_profile_fails_before_simulator_start(tmp_path):
    profile = tmp_path / "profile.json"
    profile.write_text('{"kind": "unrelated"}')
    result = cli("run", "--profile", profile, "--out", tmp_path / "run")
    assert result.returncode == 2
    assert "Unsupported rack insertion profile" in result.stderr


def test_changed_transitive_payload_is_rejected_before_simulation(tmp_path):
    asset = tmp_path / "asset"
    asset.write_bytes(b"unchanged direct reference")
    payload = tmp_path / "payload.usd"
    payload.write_bytes(b"altered payload")
    direct = {"path": asset.name, "sha256": hashlib.sha256(asset.read_bytes()).hexdigest()}
    profile = tmp_path / "profile.json"
    profile.write_text(json.dumps({
        "kind": "deformx-rack-insertion-v1",
        "assets": {name: direct for name in (
            "connector", "plug", "latch", "socket", "panel", "robot", "table", "termination_visual",
        )},
        "dependencies": [
            {"path": payload.name, "sha256": hashlib.sha256(b"original payload").hexdigest()},
        ],
    }))
    output = tmp_path / "run"
    result = cli("run", "--profile", profile, "--out", output)
    assert result.returncode == 2
    assert f"USD dependency checksum mismatch: {payload}" in result.stderr
    assert not output.exists()


def test_incomplete_downloads_fail_before_preparation(tmp_path):
    result = cli("prepare", "--sources", tmp_path, "--out", tmp_path / "assets")
    assert result.returncode == 2
    assert "demo.rack_insertion.assets" in result.stderr
    assert not (tmp_path / "assets").exists()


def test_renderer_rejects_missing_fresh_result_before_simulator_start(tmp_path):
    result = cli("render", "--data", tmp_path / "missing", "--out", tmp_path / "video")
    assert result.returncode == 2
    assert "report.json" in result.stderr
    assert not (tmp_path / "video").exists()


def test_kit_starts_before_rod_import_and_cannot_mask_incomplete_physics(tmp_path):
    source = tmp_path / "asset"
    source.write_bytes(b"local test asset")
    direct = {"path": source.name, "sha256": hashlib.sha256(source.read_bytes()).hexdigest()}
    panel = tmp_path / "panel.json"
    panel.write_text(json.dumps({
        "assets": {"local": direct}, "usd_composition_dependencies": [direct],
        "socket_position_m": [0, 0, 0], "table_position_m": [0, 0, 0],
        "collision_groups": [{"path": "Panel", "spacing_m": .001, "thickness_m": 0.}],
        "decorative_roots": [],
    }))
    assets = {name: direct for name in ("connector", "plug", "latch", "socket", "robot", "table", "termination_visual")}
    assets["panel"] = {"path": panel.name, "sha256": hashlib.sha256(panel.read_bytes()).hexdigest()}
    profile = tmp_path / "profile.json"
    profile.write_text(json.dumps({"kind": "deformx-rack-insertion-v1", "assets": assets, "dependencies": [direct]}))

    # Simulate Kit's external fast exit, which bypasses the Python traceback and return code.
    runtime = tmp_path / "runtime"
    (runtime / "isaaclab").mkdir(parents=True)
    (runtime / "isaaclab/__init__.py").touch()
    (runtime / "isaacsim.py").touch()
    (runtime / "isaaclab/app.py").write_text(
        "import os, sys\n"
        "class AppLauncher:\n"
        "    def __init__(self, config):\n"
        "        assert 'ropesim' not in sys.modules, 'rod package loaded before Kit'\n"
        "        print('Kit received control before the rod import', flush=True)\n"
        "        os._exit(0)\n"
    )
    (runtime / "ropesim").mkdir()
    (runtime / "ropesim/__init__.py").write_text("class RopeBatch:\n    from_positions = None\n")
    (runtime / "ropesim/isaac_bridge.py").write_text("bake_body = None\n")
    (runtime / "torch.py").write_text("class cuda:\n    is_available = staticmethod(lambda: True)\n")

    output = tmp_path / "run"
    result = cli("run", "--profile", profile, "--out", output, runtime=runtime)
    assert result.returncode == 1
    assert "Kit received control before the rod import" in result.stdout
    assert "Physical simulation did not complete" in result.stderr
    assert json.loads((output / "report.json").read_text())["complete"] is False
