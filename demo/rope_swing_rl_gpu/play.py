"""Evaluate / watch / export rollouts of a trained RopeSwing checkpoint.

Headless eval (success rate and closest approach to the target):
    python demo/rope_swing_rl_gpu/play.py --headless --num_envs 256 --episodes 4 \
        --checkpoint logs/rsl_rl/rope_swing/<run>/model_<iter>.pt

Watch in the Isaac Sim GUI (target sphere, rope+stick cylinders,
tip trails; --realtime paces to 60 Hz so swings are watchable):
    python demo/rope_swing_rl_gpu/play.py --num_envs 4 --realtime --checkpoint <pt>

Export env-0 rollouts (actions, joint targets, rope trajectory, distances):
    python demo/rope_swing_rl_gpu/play.py --headless --episodes 8 --export out.npz \
        --checkpoint <pt>
"""
import argparse
import os
import sys
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=None,
                    help="default: 256 headless, 4 with GUI")
parser.add_argument("--episodes", type=int, default=4,
                    help="episode batches to run")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--realtime", action="store_true",
                    help="pace stepping to wall-clock 60 Hz")
parser.add_argument("--export", type=str, default=None,
                    help="save env-0 rollouts (npz): actions, joint targets, "
                         "rope trajectory, distances, success flags")
parser.add_argument("--debug_vis", action="store_true",
                    help="force markers on (auto-enabled with GUI)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

GUI = not args.headless
if args.num_envs is None:
    args.num_envs = 4 if GUI else 256

import isaaclab  # noqa: E402
_src = os.path.join(os.path.dirname(isaaclab.__file__), "source", "isaaclab_rl")
if _src not in sys.path:
    sys.path.insert(0, _src)

import torch  # noqa: E402
from isaaclab_rl.rsl_rl import (RslRlOnPolicyRunnerCfg,  # noqa: E402
                                RslRlPpoActorCriticCfg,
                                RslRlPpoAlgorithmCfg, RslRlVecEnvWrapper)
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # this demo's ropesim_tasks
from ropesim_tasks.swing_env import RopeSwingEnv, RopeSwingEnvCfg  # noqa: E402


def make_runner_cfg(device):
    """Same construction as train.py so the checkpoint loads (net shape and
    the rsl_rl 3.0 schema, e.g. obs_groups, come from the cfg class)."""
    return RslRlOnPolicyRunnerCfg(
        seed=0,
        device=device,
        num_steps_per_env=24,
        max_iterations=1,
        empirical_normalization=False,
        policy=RslRlPpoActorCriticCfg(
            init_noise_std=1.0,
            actor_obs_normalization=True,   # must mirror train.py exactly
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
        save_interval=1000,
        experiment_name="rope_swing_eval",
        logger="tensorboard",
    )


def main():
    env_cfg = RopeSwingEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.debug_vis = GUI or args.debug_vis
    env = RopeSwingEnv(env_cfg)
    wrapped = RslRlVecEnvWrapper(env)

    runner = OnPolicyRunner(wrapped, make_runner_cfg(env.unwrapped.device).to_dict(),
                            log_dir=None, device=env.unwrapped.device)
    runner.load(args.checkpoint)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # Aggregate per-EPISODE metrics at reset boundaries: the env logs each
    # finishing batch's success/min-dist into extras["log"] inside _reset_idx
    # (reading buffers after a fixed step count races the auto-reset, which
    # clears them in the same step they would be read).
    steps_total = args.episodes * env.unwrapped.max_episode_length   # at most; episodes may end early
    u = env.unwrapped
    episodes, hits, dist_wsum = 0, 0.0, 0.0
    # env-0 rollout recording: one row per step with the episode index, since episodes end
    # on success or timeout at different lengths. Success and min-distance are recomputed
    # offline from the exported trajectory with the env's own criterion (Y-Z distance).
    rec = {"episode": [], "actions": [], "targets": [], "rope": [], "dist": []} if args.export else None
    ep0 = 0
    obs = wrapped.get_observations()
    dt_wall = u.physics_dt
    for k in range(steps_total):
        t0 = time.perf_counter()
        with torch.no_grad():
            actions = policy(obs)
        if rec is not None:
            rec["actions"].append(actions[0].detach().cpu().numpy().copy())
        obs, _, _, extras = wrapped.step(actions)
        if rec is not None:
            rec["episode"].append(ep0)
            rec["targets"].append(u._targets[0].detach().cpu().numpy().copy())
            rec["rope"].append(u._rope_pos[0].detach().cpu().numpy().copy())
            rec["dist"].append(float(u._dist[0].item()))
            if bool(u.reset_buf[0].item()):
                ep0 += 1
        n_reset = int(u.reset_buf.sum().item())
        log = extras.get("log") or {}
        if n_reset and "Metrics/success_rate" in log:
            episodes += n_reset
            hits += log["Metrics/success_rate"] * n_reset
            dist_wsum += log["Metrics/min_dist"] * n_reset
        if args.realtime:
            time.sleep(max(0.0, dt_wall - (time.perf_counter() - t0)))
    if episodes:
        print(f"\n  {episodes} episodes: success {100*hits/episodes:.1f}%, "
              f"mean min Y-Z distance {dist_wsum/episodes:.3f} m")
    else:
        print("\n  no episodes finished - increase --episodes")
    if rec is not None:
        import numpy as np

        ep = np.array(rec["episode"])
        dist = np.array(rec["dist"])
        n_ep = int(ep.max()) + 1 if ep.size else 0
        min_dist = np.array([dist[ep == e].min() for e in range(n_ep)])
        success = min_dist < env_cfg.success_thresh
        np.savez(args.export,
                 episode=ep,
                 actions=np.array(rec["actions"]),
                 joint_targets=np.array(rec["targets"]),
                 rope=np.array(rec["rope"]),
                 dist=dist,
                 target=np.asarray(env_cfg.target_local),
                 success=success, min_dist=min_dist, dt=u.physics_dt)
        print(f"  exported {n_ep} env-0 episodes -> {args.export} "
              f"(success {int(success.sum())}/{n_ep}, min Y-Z distance {min_dist.mean():.3f} m)")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
