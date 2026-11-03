# Wire-end calibration

Replays a recorded UR5e whip trajectory in Isaac Sim with the Cosserat rod on the end
effector, exports the simulated wire-end path, and compares it with the wire end measured by
motion capture. It is how the rod's density and Young's modulus were fitted for the paper.

Run everything from the repository root (setup: [demo README](../../README.md#running)).

## Inputs

Neither input is tracked; put your own under `calibration/data/` or pass paths explicitly.

| File | Format |
|---|---|
| `data/trajectory/*.csv` | Joint trajectory: header `t, shoulder_pan, shoulder_lift, elbow, wrist_1, wrist_2, wrist_3`, radians, one row per control step |
| `data/reference/*.csv` | Motion-capture wire-end path: header with `time` and `X, Y, Z` columns (millimetres by default, see `--ref_pos_scale`) |
| `config/*.json` | CoSim engine parameters (`replay_whip_traj_wire_end_calibration_engine_cfg.json` is the fitted set; `rho700_E220000.json` an example override) |

## Scripts

```bash
# replay + export the simulated wire-end path; compare with the reference if --ref_csv is given
$ISAAC_PYTHON demo/rope_swing_rl/calibration/scripts/replay_whip_traj_wire_end_calibration.py --headless \
  --traj_csv demo/rope_swing_rl/calibration/data/trajectory/whip_traj_high.csv \
  --engine_cfg demo/rope_swing_rl/calibration/config/replay_whip_traj_wire_end_calibration_engine_cfg.json \
  --out_csv outputs/raw/whip_wire_end_positions.csv --out_plot outputs/plots/whip_wire_end_trajectory.png \
  [--ref_csv data/reference/whipping_high_1_001_stacked_transformed.csv --compare_out_csv outputs/comparison/cmp.csv] \
  [--make_video --out_video outputs/videos/whip.mp4] [--max_frames N]

# compare an exported wire-end CSV with a reference over a time window (no Isaac Sim)
python demo/rope_swing_rl/calibration/scripts/compare_trajectory_range.py \
  --sim_csv outputs/raw/whip_wire_end_positions.csv --ref_csv data/reference/<mocap>.csv --t_start 0 --t_end 3

# sweep density x Young's modulus: runs the replay per combination and tabulates the error
$ISAAC_PYTHON demo/rope_swing_rl/calibration/scripts/sweep_density_youngs.py \
  --densities 600,700,800 --youngs 1.5e5,2.2e5,3e5 --traj_csv <traj.csv> --ref_csv <mocap.csv> --output_dir outputs/sweep

# replay a trajectory exported as .npz (keys joint_positions|positions, times) instead of a CSV
$ISAAC_PYTHON demo/rope_swing_rl/calibration/scripts/replay_exported_trajectory_isaac45.py --headless --traj_npz <traj.npz>
```

`run_replay_whip_medium.sh` and `run_compare.sh` wrap the replay and comparison with the
paper's file names (`--help` shows their options); they expect the `data/` layout above.

Outputs go under `calibration/outputs/` by default: `raw/` (exported wire-end CSV),
`comparison/` (aligned sim-vs-reference CSV with errors), `plots/`, `videos/`.
