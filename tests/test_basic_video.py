import hashlib
import json
import subprocess
import sys

import numpy as np
import pytest

from demo.basics import render


def test_each_view_uses_recorded_samples_and_keeps_terminal_pose():
    times = np.arange(13) / 120
    whole, detail = render.replay_segments(times)
    assert whole["source_indices"] == [0, 4, 8, 12]
    assert whole["speed"] == 1
    assert detail["source_indices"][:8] == [0, 0, 1, 1, 2, 2, 3, 3]
    assert detail["source_indices"][-1] == 12
    assert detail["speed"] == 0.125


def test_preflight_rejects_recording_replaced_after_report(tmp_path):
    p = tmp_path / "trajectory.npz"
    np.savez(p, t=np.arange(3) / 120, positions=np.zeros((3, 1, 2, 3), dtype="f"))
    (tmp_path / "report.json").write_text(json.dumps(dict(
        complete=True, settings=dict(case="hanging"),
        trajectory_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
    )))
    with p.open("ab") as stream:
        stream.write(b"unexpected change")
    with pytest.raises(ValueError, match="hash"):
        render.load_source("hanging", tmp_path)


def test_output_guard_preserves_existing_capture(tmp_path):
    existing = tmp_path / "video.mp4"
    existing.write_bytes(b"accepted")
    result = subprocess.run(
        [sys.executable, "-m", "demo.basics.render", "--case", "hanging",
         "--data", str(tmp_path), "--assets", str(tmp_path), "--out", str(tmp_path)],
        capture_output=True, text=True,
    )
    assert result.returncode == 2
    assert "already exists" in result.stderr
    assert existing.read_bytes() == b"accepted"
