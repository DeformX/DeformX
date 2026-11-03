# Rope-swing RL demo

Reinforcement learning on a UR5e that swings a wire, with the wire simulated by the Cosserat
rod engine (`PyElastica-Mesh`) inside Isaac Sim. Hydra configures everything; PPO is a small
pure-PyTorch implementation (`RL/algos/ppo.py`).

Run all commands **from this directory** with Isaac Sim's Python
(environment setup: [root README](../../README.md#environment-setup)):

```bash
cd demo/rope_swing_rl
export ISAAC_PYTHON=python              # or /path/to/isaacsim/python.sh
export OMNI_KIT_ACCEPT_EULA=YES
$ISAAC_PYTHON -m pip install hydra-core omegaconf      # once; wandb is optional
```

`RL/__init__.py` puts `PyElastica-Mesh` on the import path, so no `PYTHONPATH` is needed here.

## Layout

- `conf/config.yaml` — training, logging, checkpoint and eval-export settings
- `conf/task/*.yaml` — one file per task; `conf/algo/ppo.yaml` — PPO hyperparameters
- `RL/train.py`, `RL/eval.py` — training, and offline evaluation with trajectory export
- `RL/envs/` — the environments; `RL/algos/ppo.py` — PPO
- `tools/` — replay, ranking and motion-capture evaluation helpers
- `calibration/` — wire-end calibration against motion capture ([README](calibration/README.md))

## Tasks

| `task=` | What |
|---|---|
| `wire_swing` | UR5e swings a Cosserat-rod wire to a target (the default) |
| `wire_swing_hit_apple` | the swing task with an apple-shaped target |
| `wire_swing_bj` | the swing task with a PhysX ball-joint chain instead of the rod (baseline) |
| `wire_twist` | the swing environment with a shorter, coarser-timestep configuration |
| `franka_reach` | Franka reaching, no wire; the smallest PPO smoke test |

Defaults from `conf/config.yaml`: task `wire_swing`, algo `ppo`, device `cuda`,
`total_steps` 200000, `render` false.

## Training

```bash
# the default task
$ISAAC_PYTHON -m RL.train task=wire_swing algo=ppo render=false

# ball-joint baseline, with the Isaac Sim viewport open
$ISAAC_PYTHON -m RL.train task=wire_swing_bj algo=ppo render=true

# a short run to check the setup
$ISAAC_PYTHON -m RL.train task=wire_swing algo=ppo render=false total_steps=300 log_every=100
```

Common overrides (any config key works the same way):

```bash
$ISAAC_PYTHON -m RL.train task=wire_swing_bj algo=ppo render=false \
  total_steps=50000 log_every=1000 checkpoint_every=5000 algo.rollout_len=64 task.num_envs=8
```

- `render=true|false` — open the Isaac Sim viewport
- `total_steps`, `task.num_envs`, `algo.rollout_len`, `checkpoint_every`
- `resume_checkpoint=/abs/path/to/checkpoints/ppo_step_50000.pt` — resume
- `task.wire_usd=/abs/path/wire.usdc` — wire asset (default: `DEFORMX_WIRE_USD`, else the bundled yellow wire)

Outputs of a run:

- `checkpoints/` — periodic checkpoints and `ppo_final.pt`
- `data/success_trajectories/` — successful episodes: `*.csv` joint commands with header
  `t, shoulder_pan, shoulder_lift, elbow, wrist_1, wrist_2, wrist_3`, `*_actions.npz`, and the
  run's merged/task/algo configs. Disable with `export_success_actions=false save_configs_to_success_dir=false`
- `outputs/` — Hydra's per-run logs
- `wandb/` — when enabled

## Evaluation and trajectory export

```bash
$ISAAC_PYTHON -m RL.eval task=wire_swing algo=ppo eval.checkpoint=/abs/path/to/ppo_final.pt eval.episodes=5
```

Writes to `eval_exports/` (`eval.export_dir`): `eval_<mode>_ep<NNN>.csv` (the joint trajectory),
`eval_<mode>_ep<NNN>_actions.npz`, `eval_episode_summary.csv` and `eval_summary.json`.
`eval.export_mode` picks the exported episode (`first_success | best_success | best_return |
best_min_dist | latest`); `eval.resample_dt=0.002` resamples the CSV to a real-arm control period.

## Replay and tools

```bash
# replay a joint CSV with the rod on the arm: tip-trace CSV, YZ plot and joint plots next to the CSV
$ISAAC_PYTHON tools/replay_wire_traj.py /abs/path/traj.csv [--headless] [--use-ground-contact]

# the same with the ball-joint wire (records a camera video)
$ISAAC_PYTHON tools/replay_wire_traj_bj.py /abs/path/traj.csv [--headless] [--task-config conf/task/wire_swing_bj.yaml]

# evaluate a checkpoint, export the chosen episode and replay it in one go
$ISAAC_PYTHON tools/replay_checkpoint_wire_traj.py --checkpoint /abs/path/ppo_final.pt --task wire_swing [--skip-replay] [--headless]

# replay with Gaussian noise or a delay on the commanded joints (robustness checks)
$ISAAC_PYTHON tools/replay_wire_traj_random_noise.py /abs/path/traj.csv --noise-std 0.01 [--delay-joint-index 2 --delay-seconds 0.05]

# rank exported episodes by joint speed (no Isaac Sim)
python tools/rank_the_fast_traj.py data/success_trajectories --top-k 10

# real-world motion-capture CSVs: nearest point to a target, per file or as a batch summary (no Isaac Sim)
python tools/eval_demo3_aligned.py <mocap.csv or dir> --target X Y Z
python tools/eval_demo3_aligned_batch.py <mocap dir> --target X Y Z --outdir plots
```

## Weights & Biases

Disabled by default so a fresh clone trains without an account:

```bash
$ISAAC_PYTHON -m RL.train task=wire_swing algo=ppo wandb.enabled=true wandb.project=deformx-rl
$ISAAC_PYTHON -m RL.train task=wire_swing algo=ppo wandb.enabled=true wandb.mode=offline     # no login
```

## Notes

- Tasks default to `device=cuda`; `device=cpu` works but Isaac Sim still needs a GPU.
- The wire USD and the UR5e USD are resolved by `demo/common/paths.py`
  (`DEFORMX_WIRE_USD`, `ISAAC_ASSETS_ROOT`; the robot falls back to NVIDIA's public asset mirror).
