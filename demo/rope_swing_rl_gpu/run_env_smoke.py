"""Headless smoke test of the RopeSwing DirectRLEnv scaffold.

Verifies the full wiring: Isaac Sim boots, the UR5 streams from S3, the rope
batch couples to the EE, observations are finite, roots track the arm, and
vectorized resets restore the ropes.

    python demo/rope_swing_rl_gpu/run_env_smoke.py --headless --num_envs 16 --steps 180
"""
import argparse
import os
import sys
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--steps", type=int, default=180)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# isaaclab imports must come after app launch
import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # this demo's ropesim_tasks
from ropesim_tasks.swing_env import RopeSwingEnv, RopeSwingEnvCfg  # noqa: E402


def main():
    cfg = RopeSwingEnvCfg()
    cfg.scene.num_envs = args.num_envs
    env = RopeSwingEnv(cfg)
    obs, _ = env.reset()
    assert torch.isfinite(obs["policy"]).all(), "non-finite obs after reset"

    N = env.num_envs
    t0 = time.perf_counter()
    for k in range(args.steps):
        # the three active joints (shoulder lift, elbow, wrist 1), actions in [0, 1]
        phase = torch.tensor(k * env.physics_dt * 2.0 * torch.pi * 0.5)
        actions = 0.5 * (1.0 + torch.sin(phase)) * torch.ones(N, env.cfg.action_space, device=env.device)
        obs, rew, terminated, truncated, info = env.step(actions)
        assert torch.isfinite(obs["policy"]).all(), f"non-finite obs at step {k}"

        # PhysX joint speed is capped at the UR5e specification (velocity_limit_sim)
        jv = env.robot.data.joint_vel[:, env._joint_ids].abs().max().item()
        vcap = float(np.pi)
        assert jv <= vcap * 1.05 + 1e-3, (
            f"joint vel {jv:.3f} exceeds the cap {vcap:.3f} rad/s at step {k}")

        if k % 60 == 0:
            # rope root must track the EE + rigid tube transform
            root_pos, _, _ = env._compute_root_state()
            rope_root = env._rope_pos[:, 0, :]
            err = torch.linalg.vector_norm(rope_root - root_pos, dim=-1).max()
            tip = env._rope_pos[:, -1, :] - env.scene.env_origins
            print(f"  step {k:4d}: root tracking err {err:.2e} m | "
                  f"tip[0] = {tip[0].tolist()}")
            assert err < 1e-3, "rope root lost the end-effector"
    wall = time.perf_counter() - t0
    print(f"\n  {args.steps} steps x {N} envs in {wall:.2f} s "
          f"-> {args.steps/wall:.1f} env-steps/s/env, "
          f"{(args.steps*env.physics_dt)/wall:.2f}x realtime")

    # vectorized rope reset restores the construction state
    before = env._rope_pos.clone()
    env._reset_idx(torch.arange(N, device=env.device))
    after = env._rope_pos
    moved = torch.linalg.vector_norm(after - before, dim=-1).max()
    init_err = torch.linalg.vector_norm(
        after[:, 0, :] - env._init_root_pos, dim=-1).max()
    print(f"  reset: ropes moved {moved:.3f} m back; "
          f"root-vs-init err {init_err:.2e} m")
    assert init_err < 1e-5
    assert torch.isfinite(after).all()

    # the reset shape: the rope hangs from the stick tip and lies on the floor, settled
    # (floor contact on), with even segment lengths and nothing below the floor
    seg = (after[:, 1:, :] - after[:, :-1, :]).norm(dim=-1)
    zmin = after[:, :, 2].min().item()
    print(f"  reset shape: seg len {seg.mean():.4f}+-{seg.std():.4f} m (target {env.cfg.rope_seg_len}) | "
          f"lowest vertex z = {zmin:.4f} m (rope radius {env.cfg.rope_radius})")
    seg_dev = (seg - env.cfg.rope_seg_len).abs().max().item()
    # the segment at the floor fold stays a little compressed under the contact penalty
    assert seg_dev < 0.25 * env.cfg.rope_seg_len, f"segment length off by {seg_dev:.4f} m at reset"
    assert zmin > 0.0, "rope below the floor at reset"
    print(f"  target (env frame) = {env.cfg.target_local}, tip-target Y-Z distance at reset = "
          f"{env._dist.mean():.3f} m, tip speed after reset {env._tip_vel.norm(dim=-1).max():.2e} m/s")

    env.close()
    print("\nSMOKE OK")


if __name__ == "__main__":
    main()
    simulation_app.close()
