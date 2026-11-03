"""Closed-loop PPO training for the RopeSwing goal task (rsl_rl).

    python demo/rope_swing_rl_gpu/train.py --headless --num_envs 512 --max_iterations 1500

Logs + checkpoints land in logs/rsl_rl/rope_swing/<timestamp>/ (TensorBoard).
"""
import argparse
import os
import sys
from datetime import datetime

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=None,
                    help="default: 512 headless, 16 with GUI")
parser.add_argument("--max_iterations", type=int, default=1500)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--run_name", type=str, default=None)
parser.add_argument("--checkpoint", type=str, default=None, help="resume from")
parser.add_argument("--realtime", action="store_true",
                    help="pace env steps to wall-clock 60 Hz so the viewport "
                         "shows real-time motion (training slows to ~60 "
                         "steps/s per env batch - debugging only)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# GUI-aware sizing: a 512-env scene takes minutes to construct and render —
# with a window open the app cannot pump UI events during construction and the
# OS flags Isaac Sim as "not responding" (it isn't hung, but unusable). Small
# scenes build in seconds and render every step.
GUI = not args.headless
if args.num_envs is None:
    args.num_envs = 16 if GUI else 512
if GUI and args.num_envs > 64:
    print(f"[train][WARNING] {args.num_envs} envs with the GUI will appear "
          "frozen during scene construction and render very slowly — "
          "use --headless for big runs.")
if GUI:
    print("[train] GUI mode: the window may briefly report 'not responding' "
          "while the scene is constructed — it recovers once training starts.")

# the rsl_rl wrapper ships inside the vendored isaaclab source tree, which is
# not on sys.path by default in the pip install
import isaaclab  # noqa: E402
_src = os.path.join(os.path.dirname(isaaclab.__file__), "source", "isaaclab_rl")
if _src not in sys.path:
    sys.path.insert(0, _src)

import time  # noqa: E402

import torch  # noqa: E402
from isaaclab_rl.rsl_rl import (RslRlOnPolicyRunnerCfg,  # noqa: E402
                                RslRlPpoActorCriticCfg,
                                RslRlPpoAlgorithmCfg, RslRlVecEnvWrapper)
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # this demo's ropesim_tasks
from ropesim_tasks.swing_env import RopeSwingEnv, RopeSwingEnvCfg  # noqa: E402


class RealtimeVecEnvWrapper(RslRlVecEnvWrapper):
    """Paces each env step to wall-clock physics dt (watchable training)."""

    def step(self, actions):
        t0 = time.perf_counter()
        ret = super().step(actions)
        dt = self.unwrapped.physics_dt
        time.sleep(max(0.0, dt - (time.perf_counter() - t0)))
        return ret


def main():
    env_cfg = RopeSwingEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.seed = args.seed
    env_cfg.debug_vis = GUI  # goal/rope/tube markers + tip trails in viewport
    env = RopeSwingEnv(env_cfg)
    env = (RealtimeVecEnvWrapper if args.realtime else RslRlVecEnvWrapper)(env)

    agent_cfg = RslRlOnPolicyRunnerCfg(
        seed=args.seed,
        device=env.unwrapped.device,
        num_steps_per_env=24,
        max_iterations=args.max_iterations,
        # NOTE: the deprecated top-level empirical_normalization flag is
        # SILENTLY IGNORED by rsl_rl 3.0 when the per-net fields serialize to
        # {} (dataclasses.MISSING -> {} defeats the deprecation shim, review
        # finding) - set the per-net flags explicitly.
        empirical_normalization=False,
        policy=RslRlPpoActorCriticCfg(
            init_noise_std=1.0,
            actor_obs_normalization=True,
            critic_obs_normalization=True,
            actor_hidden_dims=[256, 128, 64],
            critic_hidden_dims=[256, 128, 64],
            activation="elu",
        ),
        algorithm=RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.006,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=1.0e-3,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
        ),
        save_interval=50,
        experiment_name="rope_swing",
        logger="tensorboard",
    )

    run_name = args.run_name or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name, run_name)
    os.makedirs(log_dir, exist_ok=True)
    print(f"[train] logging to {log_dir}")

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir,
                            device=agent_cfg.device)
    if args.checkpoint:
        runner.load(args.checkpoint)
    runner.learn(num_learning_iterations=agent_cfg.max_iterations,
                 init_at_random_ep_len=True)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
