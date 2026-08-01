# RL_Demo

A minimal reinforcement-learning demo for DeformX: Hydra for configuration, Isaac Sim for
the runtime. Training lives in `RL/train.py`; offline evaluation and trajectory export live
in `RL/eval.py`.

## Prerequisites

These environments must run under **Isaac Sim's Python**, not a plain system or Conda
interpreter. Two Isaac Sim install layouts are supported:

```bash
# Launcher install -> use Isaac Sim's bundled python.sh
export ISAAC_PYTHON=/path/to/isaac-sim/python.sh

# pip install (Isaac Sim 4.5 / 5.x) -> Isaac Sim runs under the active interpreter
export ISAAC_PYTHON=python
```

Install the extra dependencies into that interpreter:

```bash
$ISAAC_PYTHON -m pip install hydra-core omegaconf

# optional: Weights & Biases logging
$ISAAC_PYTHON -m pip install wandb
```

All commands below are run from the `RL_Demo/` directory.

## Layout

- `conf/config.yaml` — global training, logging, checkpoint, and eval-export settings.
- `conf/task/*.yaml` — per-task configuration.
- `conf/algo/ppo.yaml` — PPO hyperparameters.
- `RL/train.py` — training entry point: checkpoints, success-trajectory export, wandb.
- `RL/eval.py` — offline evaluation entry point.
- `tools/` — trajectory replay, ranking, and other helpers.

## Available tasks

- `franka_reach` — Franka reaching task; the smallest useful PPO smoke test.
- `wire_swing` — Cosserat-wire swinging task.
- `wire_swing_bj` — PhysX ball-joint chain wire swinging task.
- `wire_swing_hit_apple` — swing-to-strike task with an apple target.
- `wire_twist` — wire twisting task.

Defaults from `conf/config.yaml`:

- task: `wire_swing`
- algorithm: `ppo`
- device: `cuda`
- total training steps: `200000`

## Training

```bash
# smallest example
$ISAAC_PYTHON -m RL.train task=franka_reach algo=ppo render=false

# Cosserat wire swing
$ISAAC_PYTHON -m RL.train task=wire_swing algo=ppo render=false

# ball-joint wire swing
$ISAAC_PYTHON -m RL.train task=wire_swing_bj algo=ppo render=false

# with the Isaac Sim viewport open
$ISAAC_PYTHON -m RL.train task=wire_swing_bj algo=ppo render=true
```

### Common overrides

```bash
$ISAAC_PYTHON -m RL.train \
  task=wire_swing_bj \
  algo=ppo \
  render=false \
  total_steps=50000 \
  log_every=1000 \
  checkpoint_every=5000 \
  algo.rollout_len=64 \
  task.num_envs=8
```

- `render=true|false` — open the Isaac Sim viewport.
- `total_steps=<int>` — total training steps.
- `task.num_envs=<int>` — number of parallel environments.
- `algo.rollout_len=<int>` — PPO rollout length.
- `checkpoint_every=<int>` — checkpoint interval.
- `resume_checkpoint=/abs/path/to/ppo_step_x.pt` — resume from a checkpoint.

### Resuming

```bash
$ISAAC_PYTHON -m RL.train \
  task=wire_swing_bj \
  algo=ppo \
  render=false \
  resume_checkpoint=/abs/path/to/checkpoints/ppo_step_50000.pt
```

## Training outputs

By default a run produces:

- `checkpoints/` — periodic checkpoints plus a final one.
- `data/wire_swing_bj_2/success_traj_csv_high_new_2/` — exported successful episodes.
- `wandb/` — wandb logs, when enabled.

`train.py` enables these by default:

- `save_checkpoints: true`
- `save_final_checkpoint: true`
- `export_success_actions: true`
- `save_configs_to_success_dir: true`

An exported successful episode typically contains:

- `*_actions.npz` — the policy's action sequence.
- `*.csv` — joint command trajectory, with header
  `t, shoulder_pan, shoulder_lift, elbow, wrist_1, wrist_2, wrist_3`.
- `train_config_*/` — the merged/task/algo configs captured at the start of that run.

To skip success-trajectory export:

```bash
$ISAAC_PYTHON -m RL.train \
  task=wire_swing_bj \
  export_success_actions=false \
  save_configs_to_success_dir=false
```

## Weights & Biases logging

wandb is **disabled by default** so a fresh clone trains without an account. Enable it with:

```bash
$ISAAC_PYTHON -m RL.train \
  task=wire_swing_bj \
  algo=ppo \
  render=false \
  wandb.enabled=true \
  wandb.project=deformx-rl
```

Offline mode (logs locally, no login required):

```bash
$ISAAC_PYTHON -m RL.train \
  task=wire_swing_bj \
  algo=ppo \
  render=false \
  wandb.enabled=true \
  wandb.mode=offline
```

## Replay

Given a joint-trajectory CSV exported by training, replay it with
`tools/replay_wire_traj.py`:

```bash
$ISAAC_PYTHON RL_Demo/tools/replay_wire_traj.py /abs/path/to/trajectory.csv
```

The script reads these CSV columns:

- `t`
- `shoulder_pan`
- `shoulder_lift`
- `elbow`
- `wrist_1`
- `wrist_2`
- `wrist_3`

It replays the joint trajectory in Isaac Sim and writes the replay results — typically a
tip-trace CSV, a YZ trajectory plot, and per-joint arm plots.

## Notes

- Most tasks default to `cuda`. Pass `device=cpu` if no GPU is available, though Isaac Sim
  environments are still best run on a GPU machine.
- `wire_swing` and `wire_swing_hit_apple` resolve `wire_usd` through the `DEFORMX_WIRE_USD`
  environment variable, defaulting to a repo-relative path (see `deformx_paths.py` at the
  repository root). Override it with that variable, or per run with
  `task.wire_usd=/abs/path/to/wire.usdc`.
- `wire_swing_bj` is the task the default success-export path in `conf/config.yaml`
  corresponds to.
