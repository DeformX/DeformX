# Wire-End Calibration Bundle

Organized files for whip trajectory replay, wire-end calibration, comparison, and visualization.

## Layout
- scripts/: replay and helper scripts
- config/: CoSim/engine parameter config
- data/trajectory/: input replay trajectory CSV
- data/reference/: reference trajectory CSV files
- outputs/raw/: exported raw wire-end CSV
- outputs/comparison/: aligned sim-vs-reference CSV and error CSV
- outputs/plots/: generated figures
- outputs/videos/: generated 3D videos

## Main command (example)
$ISAAC_PYTHON scripts/replay_whip_traj_wire_end_calibration.py \
  --headless \
  --traj_csv data/trajectory/whip_traj_high.csv \
  --ref_csv "data/reference/whipping_high_1_001_stacked_transformed (2).csv" \
  --engine_cfg config/replay_whip_traj_wire_end_calibration_engine_cfg.json \
  --compare_out_csv outputs/comparison/whip_reference_comparison.csv \
  --out_plot outputs/plots/whip_wire_end_trajectory_cmp.png \
  --make_video \
  --out_video outputs/videos/whip_wire_end_trajectory_cmp.mp4
