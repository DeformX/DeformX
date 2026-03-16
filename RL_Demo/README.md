# RL_Demo

基于 Hydra 组织配置、基于 Isaac Sim 运行环境的最小强化学习 demo。当前训练入口在 `RL/train.py`，评估与轨迹导出入口在 `RL/eval.py`。

## 运行前提

必须使用 Isaac Sim 自带的 Python 运行，不能直接用系统 Python 或 Conda Python。

在 `RL_Demo/` 目录下执行：

```bash
/path/to/isaac-sim/python.sh -m pip install hydra-core omegaconf

# 可选：启用 wandb 日志
/path/to/isaac-sim/python.sh -m pip install wandb
```

## 目录说明

- `conf/config.yaml`：全局训练、日志、checkpoint、eval 导出配置。
- `conf/task/*.yaml`：任务配置。
- `conf/algo/ppo.yaml`：PPO 超参数。
- `RL/train.py`：训练入口，支持 checkpoint、成功轨迹导出、wandb。
- `RL/eval.py`：底层离线评估入口。
- `tools/`：轨迹回放、排序等辅助脚本。

## 当前可用任务

- `franka_reach`：Franka 到点任务，适合作为最小 PPO 验证样例。
- `wire_swing`：Cosserat wire 摆动任务。
- `wire_swing_bj`：PhysX ball-joint 链式 wire 摆动任务。
- `wire_swing_hit_apple`：带苹果目标物的击打任务。
- `wire_twist`：wire 扭转环境。

默认配置来自 `conf/config.yaml`：

- 默认任务：`task=wire_swing`
- 默认算法：`algo=ppo`
- 默认设备：`device=cuda`
- 默认训练步数：`total_steps=200000`

## 训练

### 基本命令

```bash
# 最小示例
/path/to/isaac-sim/python.sh -m RL.train task=franka_reach algo=ppo render=false

# Cosserat wire 摆动
/path/to/isaac-sim/python.sh -m RL.train task=wire_swing algo=ppo render=false

# Ball-joint wire 摆动
/path/to/isaac-sim/python.sh -m RL.train task=wire_swing_bj algo=ppo render=false

# GUI 渲染
/path/to/isaac-sim/python.sh -m RL.train task=wire_swing_bj algo=ppo render=true
```

### 常用覆盖参数

```bash
/path/to/isaac-sim/python.sh -m RL.train \
  task=wire_swing_bj \
  algo=ppo \
  render=false \
  total_steps=50000 \
  log_every=1000 \
  checkpoint_every=5000 \
  algo.rollout_len=64 \
  task.num_envs=8
```

常用参数：

- `render=true|false`：是否打开 Isaac Sim 渲染。
- `total_steps=<int>`：总训练步数。
- `task.num_envs=<int>`：并行环境数。
- `algo.rollout_len=<int>`：PPO rollout 长度。
- `checkpoint_every=<int>`：checkpoint 保存频率。
- `resume_checkpoint=/abs/path/to/ppo_step_x.pt`：恢复训练。

### 恢复训练

```bash
/path/to/isaac-sim/python.sh -m RL.train \
  task=wire_swing_bj \
  algo=ppo \
  render=false \
  resume_checkpoint=/abs/path/to/checkpoints/ppo_step_50000.pt
```

## 训练输出

默认会生成以下内容：

- `checkpoints/`：按 `checkpoint_every` 保存的模型，以及最终 checkpoint。
- `data/wire_swing_bj_2/success_traj_csv_high_new_2/`：成功 episode 导出目录。
- `wandb/`：wandb 日志目录。

`train.py` 里默认开启了以下功能：

- `save_checkpoints: true`
- `save_final_checkpoint: true`
- `export_success_actions: true`
- `save_configs_to_success_dir: true`

成功 episode 导出通常包含：

- `*_actions.npz`：策略动作序列。
- `*.csv`：关节命令轨迹，表头为 `t, shoulder_pan, shoulder_lift, elbow, wrist_1, wrist_2, wrist_3`。
- `train_config_*/`：启动该次训练时保存的 merged/task/algo 配置。

如果你不想导出成功轨迹，可以覆盖：

```bash
/path/to/isaac-sim/python.sh -m RL.train \
  task=wire_swing_bj \
  export_success_actions=false \
  save_configs_to_success_dir=false
```

## wandb 日志

```bash
/path/to/isaac-sim/python.sh -m RL.train \
  task=wire_swing_bj \
  algo=ppo \
  render=false \
  wandb.enabled=true \
  wandb.project=cosseratx-rl
```

离线模式：

```bash
/path/to/isaac-sim/python.sh -m RL.train \
  task=wire_swing_bj \
  algo=ppo \
  render=false \
  wandb.enabled=true \
  wandb.mode=offline
```

## 回放

如果你已经有训练导出的关节轨迹 CSV，直接用 `tools/replay_wire_traj.py`：

```bash
/path/to/isaac-sim/python.sh RL_Demo/tools/replay_wire_traj.py \
  /abs/path/to/trajectory.csv
```

这个脚本会读取 CSV 表头中的：

- `t`
- `shoulder_pan`
- `shoulder_lift`
- `elbow`
- `wrist_1`
- `wrist_2`
- `wrist_3`

然后在 Isaac Sim 中回放关节轨迹，并生成 replay 结果文件。

## 回放输出

使用 `tools/replay_wire_traj.py` 时，通常会生成 replay 阶段的 tip trace CSV、YZ 轨迹图、机械臂关节图。

## 备注

- 大多数任务默认使用 `cuda`，如果机器没有可用 GPU，可以显式传 `device=cpu`，但 Isaac Sim 相关环境通常仍建议在 GPU 环境运行。
- `wire_swing` 和 `wire_swing_hit_apple` 配置里引用了固定的 `wire_usd` 绝对路径；如果本机资源位置不同，需要先改配置。
- `wire_swing_bj` 目前是 `conf/config.yaml` 中成功轨迹默认导出路径对应的主要任务。
