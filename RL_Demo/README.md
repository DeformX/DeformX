# RL_Demo

Minimal Hydra + (env, algo, train) RL scaffold based on **Isaac Sim**.

## Run

> ⚠️ **Important**:  
> This project **must be run using Isaac Sim’s bundled Python**,  
> **not** system Python or Conda.

From the `RL_Demo/` directory:

```bash
# use Isaac Sim's python
/path/to/isaac-sim/python.sh -m pip install hydra-core omegaconf
# optional: wandb logging
/path/to/isaac-sim/python.sh -m pip install wandb

# run training (headless by default)
 /path/to/isaac-sim/python.sh -m RL.train task=franka_reach algo=ppo render=false

# run with GUI rendering
 /path/to/isaac-sim/python.sh -m RL.train task=franka_reach algo=ppo render=true

# swing-wire task (CoSimEngine + SkeletonRodDriver)
 /path/to/isaac-sim/python.sh -m RL.train task=wire_swing algo=ppo render=false

# swing-wire task with ball-joint wire attachment
 /path/to/isaac-sim/python.sh -m RL.train task=wire_swing_bj algo=ppo render=false
```

## Optional Weights & Biases Logging

Enable W&B with Hydra overrides:

```bash
/path/to/isaac-sim/python.sh -m RL.train \
  task=wire_swing algo=ppo render=false \
  wandb.enabled=true \
  wandb.project=cosseratx-rl
```

Offline logging (sync later):

```bash
/path/to/isaac-sim/python.sh -m RL.train \
  task=wire_swing algo=ppo render=false \
  wandb.enabled=true \
  wandb.mode=offline
```

## Evaluate Checkpoint + Export Arm Motion

`RL.eval` evaluates a PPO checkpoint and exports a replay/deploy trajectory CSV with the same schema as:
`t,shoulder_pan,shoulder_lift,elbow,wrist_1,wrist_2,wrist_3`

```bash
# evaluate checkpoint and export best successful episode
/path/to/isaac-sim/python.sh -m RL.eval \
  task=wire_swing algo=ppo render=false \
  eval.checkpoint=/abs/path/to/ppo_step_50000.pt \
  eval.episodes=10 \
  eval.export_mode=best_success \
  eval.export_dir=eval_exports \
  eval.export_prefix=wire_swing_ckpt50000

# export with 2ms sampling for real-arm replay loop
/path/to/isaac-sim/python.sh -m RL.eval \
  task=wire_swing algo=ppo render=false \
  eval.checkpoint=/abs/path/to/ppo_step_50000.pt \
  eval.episodes=10 \
  eval.resample_dt=0.002
```

Useful flags:
- `eval.deterministic=true|false`: use actor mean action (deterministic) or stochastic sampling.
- `eval.export_mode=first_success|best_success|best_return|best_min_dist|latest`
- `eval.export_joint_csv=true|false`
- `eval.export_actions_npz=true|false`
