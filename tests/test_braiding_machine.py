import hashlib
import json

import numpy as np
import pytest


def test_alternating_carriers_exchange_while_the_third_stays_parked():
    from demo.braiding_machine.model import PRESETS, carriers_at

    cfg = PRESETS["showcase"]
    t0 = cfg.settle_s
    t1 = t0 + cfg.exchange_s
    at_mid, at_end, second_mid = carriers_at(
        np.array([t0 + cfg.motion_s / 2, t1, t1 + cfg.motion_s / 2]), cfg
    )
    assert at_mid[0, 0] == pytest.approx(at_mid[1, 0])
    assert at_mid[0, 1] < 0 < at_mid[1, 1]
    assert at_mid[2, 0] > 0 and at_mid[2, 1] == 0
    np.testing.assert_allclose(at_end[:, 0], [0, -0.107, 0.107], atol=1e-7)
    assert second_mid[2, 1] < 0 < second_mid[0, 1]
    np.testing.assert_array_equal(second_mid[1], at_end[1])
    held = carriers_at(np.array([0., t1]), cfg, drive=False)
    np.testing.assert_array_equal(held[0], held[1])


def braid_fixture(second_depth=-0.02):
    # In ascending z: red crosses left -> center in front of green, then
    # blue crosses right -> center. The stored direction is carrier -> clamp.
    return np.array([
        [[-1, 0, 0], [-.5, -.02, 1], [0, 0, 2], [.5, -second_depth, 3], [1, 0, 4]],
        [[0, 0, 0], [-.5, .02, 1], [-1, 0, 2], [-1, 0, 3], [-1, 0, 4]],
        [[1, 0, 0], [1, 0, 1], [1, 0, 2], [.5, second_depth, 3], [0, 0, 4]],
    ])[:, ::-1].copy()


def test_braid_word_reads_measured_depth_at_exact_knot():
    from demo.braiding_machine.analyze import braid_order

    crossings, reason = braid_order(braid_fixture())
    assert reason is None
    assert [c["generator"] for c in crossings] == [-1, 2]
    assert [c["pair"] for c in crossings] == [[0, 1], [0, 2]]
    assert [c["z_m"] for c in crossings] == [1., 3.]
    wrong, reason = braid_order(braid_fixture(.02))
    assert reason is None
    assert [c["generator"] for c in wrong] == [-1, -2]


def test_braid_order_rejects_ambiguous_projection_and_backtracking():
    from demo.braiding_machine.analyze import braid_order

    assert braid_order(braid_fixture(0.))[1] == "unresolved crossing depth"
    p = braid_fixture()
    p[0, 1, 2] = 4.1
    assert braid_order(p)[1] == "nonmonotone strand height"
    p = braid_fixture()
    p[0, 1:4, 0] = p[2, 1:4, 0]
    assert braid_order(p)[1] is not None


def test_exact_segment_clearance_and_pair_participation_are_separate_from_strain():
    from demo.braiding_machine.analyze import geometry

    p = np.array([
        [[-1, 0, 0], [1, 0, 0], [2, 0, 0]],
        [[0, -1, .023], [0, 1, .023], [0, 2, .023]],
        [[4, -1, 0], [4, 1, 0], [4, 2, 0]],
    ])
    rest = np.ones((3, 2))
    result = geometry(p[None], rest, barrier_limit=.024)
    assert result["max_abs_segment_strain"] == pytest.approx(1.)
    assert result["min_interstrand_centerline_m"] == pytest.approx(.023)
    assert result["pairs_in_barrier"] == {"0-1": 1, "0-2": 0, "1-2": 0}
    assert result["min_nonlocal_self_centerline_m"] is None


def test_binding_rejects_tampered_and_shifted_source(tmp_path):
    from dataclasses import asdict
    from demo.braiding_machine.analyze import read_run
    from demo.braiding_machine.model import PRESETS, initial_positions, carriers_at

    cfg = PRESETS["short"]
    t = np.arange(cfg.steps + 1) / cfg.step_hz
    roots = carriers_at(t, cfg, drive=False)
    p = np.repeat(initial_positions(cfg)[None], len(t), axis=0)
    path = tmp_path / "trajectory.npz"
    report_path = tmp_path / "report.json"

    def save(times, positions=p, carriers=roots):
        np.savez(path, t=times, positions=positions, carriers=carriers)
        report_path.write_text(json.dumps({
            "complete": True, "settings": asdict(cfg), "drive": False,
            "trajectory_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }))

    save(t)
    np.testing.assert_array_equal(read_run(tmp_path)[1]["positions"], p)
    with path.open("ab") as file:
        file.write(b"tampered")
    with pytest.raises(ValueError, match="hash"):
        read_run(tmp_path)
    save(t + 0.1)
    with pytest.raises(ValueError, match="time"):
        read_run(tmp_path)
    translated = p.copy()
    translated[:, 0, 0, 1] += .001
    save(t, positions=translated)
    with pytest.raises(ValueError, match="endpoints"):
        read_run(tmp_path)
    translated_roots = roots.copy()
    translated_roots[:, 0, 1] += .001
    save(t, positions=translated, carriers=translated_roots)
    with pytest.raises(ValueError, match="endpoint program"):
        read_run(tmp_path)


def test_nonlocal_self_clearance_excludes_next_two_segments():
    from demo.braiding_machine.analyze import geometry

    p = np.zeros((3, 5, 3))
    p[..., 0] = [0., .03, .06, .09, .12]
    p[..., 1] = np.array([0., .2, .4])[:, None]
    result = geometry(p[None], np.full((3, 4), .03), barrier_limit=.024)
    assert result["min_nonlocal_self_centerline_m"] == pytest.approx(.06)


def test_bad_or_reused_outputs_are_rejected_before_loading_cuda(tmp_path):
    from demo.braiding_machine.run import main

    marker = tmp_path / "retain.txt"
    marker.write_text("untouched")
    with pytest.raises(SystemExit) as error:
        main(["--out", str(tmp_path)])
    assert error.value.code == 2 and marker.read_text() == "untouched"
    with pytest.raises(SystemExit) as error:
        main(["--out", str(tmp_path / "new"), "--preset", "missing"])
    assert error.value.code == 2 and not (tmp_path / "new").exists()


def test_replay_uses_exact_steady_source_samples_without_speed_change():
    from demo.braiding_machine.render import replay_indices

    t = np.arange(55) / 60
    np.testing.assert_array_equal(replay_indices(t, .3, 30), np.arange(18, 54, 2))
    with pytest.raises(ValueError, match="aligned"):
        replay_indices(t, .31, 30)
    with pytest.raises(ValueError, match="aligned"):
        replay_indices(t, .3, 24)


def test_long_preset_spreads_five_exchanges_across_the_native_replay():
    from demo.braiding_machine.model import PRESETS, carriers_at
    from demo.braiding_machine.render import FPS, replay_indices

    assert "long" in PRESETS
    cfg = PRESETS["long"]
    base = PRESETS["showcase"]
    t = np.arange(cfg.steps + 1) / cfg.step_hz
    assert cfg.exchanges == base.exchanges == 5
    assert cfg.motion_s > 4 * base.motion_s
    assert len(replay_indices(t, cfg.settle_s, FPS)) == 24 * FPS

    exchanges = np.arange(cfg.exchanges)
    long_mid = cfg.settle_s + exchanges * cfg.exchange_s + cfg.motion_s / 2
    base_mid = base.settle_s + exchanges * base.exchange_s + base.motion_s / 2
    np.testing.assert_allclose(carriers_at(long_mid, cfg), carriers_at(base_mid, base), atol=1e-7)


def test_tall_fixture_changes_carrier_height_but_keeps_fixed_grips():
    from dataclasses import replace

    from demo.braiding_machine.model import PRESETS, carriers_at, fixtures, initial_positions

    cfg = replace(PRESETS["showcase"], carrier_z_m=2.95)
    initial = initial_positions(cfg)
    roots = carriers_at(np.array([0., cfg.settle_s + cfg.motion_s / 2]), cfg)
    np.testing.assert_array_equal(roots[0], initial[:, 0])
    np.testing.assert_allclose(roots[..., 2], 2.95)
    np.testing.assert_allclose(initial[:, -1], fixtures())
    assert np.all(np.diff(initial[..., 2], axis=-1) < 0)


def test_process_adds_normal_cadence_exchanges_with_smooth_starts_and_stops():
    from demo.braiding_machine.model import PRESETS, carriers_at
    from demo.braiding_machine.render import FPS, replay_indices

    assert "process" in PRESETS
    cfg = PRESETS["process"]
    base = PRESETS["showcase"]
    assert cfg.exchanges == 20
    assert cfg.exchange_s == base.exchange_s and cfg.motion_s == base.motion_s
    t = np.arange(cfg.steps + 1) / cfg.step_hz
    assert len(replay_indices(t, cfg.settle_s, FPS)) == 21 * FPS

    half = cfg.settle_s + cfg.motion_s / 2
    np.testing.assert_allclose(carriers_at(half, cfg)[:, :2], carriers_at(half, base)[:, :2])
    epsilon = .001
    for boundary in (cfg.settle_s, cfg.settle_s + cfg.motion_s):
        samples = carriers_at(boundary + np.array([-epsilon, 0., epsilon]), cfg)
        acceleration = np.diff(samples[..., 1], n=2, axis=0) / epsilon**2
        assert np.max(np.abs(acceleration)) < .03
