"""The rope-swing environment with and without arm contact: --contact 1 adds the UR5e's seven link
collision meshes as kinematic colliders for the rope (ropesim.isaac_bridge) on top of the floor the
environment already has, --contact 0 is the plain RL environment. Both run the same scripted joint motion and record rope and link poses.
One Isaac Sim instance per process, so run it twice to compare.

  python demo/rope_swing_rl_gpu/swing_contact_compare.py --headless --contact 1 --out swing_contact1.npz
  python demo/rope_swing_rl_gpu/swing_contact_compare.py --headless --contact 0 --out swing_contact0.npz
"""
import argparse
import os
import sys
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--contact", type=int, default=1)
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--seconds", type=float, default=8.0)
parser.add_argument("--out", default="swing.npz")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # this demo's ropesim_tasks

import numpy as np  # noqa: E402
import torch  # noqa: E402

import ropesim  # noqa: E402
import ropesim_tasks  # noqa: E402
from ropesim.isaac_bridge import ColliderSpec, RopeColliders  # noqa: E402
from ropesim_tasks.swing_env import RopeSwingEnv, RopeSwingEnvCfg  # noqa: E402

print("ropesim from", ropesim.__file__, "| ropesim_tasks from", ropesim_tasks.__file__, flush=True)
LINKS = ["base_link", "shoulder_link", "upper_arm_link", "forearm_link", "wrist_1_link", "wrist_2_link", "wrist_3_link"]


class ContactSwingEnv(RopeSwingEnv):
    """RopeSwingEnv + the bridge: link poses in before every rope step, teleport on reset."""

    def __init__(self, cfg, contact: bool, **kw):
        self.contact = contact
        self.bridge = None
        self._pending_reset = None
        super().__init__(cfg, **kw)

    def _step_rope(self):
        if self.contact:
            if self.bridge is None:   # the env already has the floor plane and the surface material
                self.bridge = RopeColliders(self.rope, self.scene,
                                            [ColliderSpec("robot", bodies=LINKS, dynamic=False, spacing=0.005, margin=0.03, round=0.01)])
            self.bridge.sync_poses(self._pending_reset)
            self._pending_reset = None
        super()._step_rope()

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        self._pending_reset = env_ids if torch.is_tensor(env_ids) else self.robot._ALL_INDICES


def main():
    cfg = RopeSwingEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.episode_length_s = 1e6                      # no timeouts inside the recording
    cfg.terminate_on_success = False
    cfg.too_far_thresh = 1e6
    env = ContactSwingEnv(cfg, contact=bool(args.contact))
    env.reset()
    N, V = env.num_envs, env.cfg.rope_num_verts
    ids = [env.robot.body_names.index(n) for n in LINKS]
    frames = int(args.seconds * 60)
    rec = dict(t=[], P=[], link_pos=[], link_quat=[])
    t_wall = []
    for f in range(frames):
        t = f / 60
        # scripted swing on the three active joints (shoulder lift, elbow, wrist 1), actions in
        # [0, 1]: the env integrates them into joint targets (rate-limited, limit-clamped)
        a = torch.zeros((N, env.cfg.action_space), device=env.device)
        a[:, 0] = 0.5 * (1.0 + np.sin(2 * np.pi * t / 2.0))
        a[:, 1] = 0.5 * (1.0 + np.sin(2 * np.pi * t / 2.0 + 0.6))
        a[:, 2] = 0.5 * (1.0 + np.cos(2 * np.pi * t / 3.0))
        t0 = time.perf_counter()
        env.step(a)
        torch.cuda.synchronize(); t_wall.append(time.perf_counter() - t0)
        d = env.robot.data
        rec["t"].append((f + 1) / 60)
        rec["P"].append(env._rope_pos.cpu().numpy().copy())
        rec["link_pos"].append(d.body_link_pos_w[:, ids].cpu().numpy().copy())
        rec["link_quat"].append(d.body_link_quat_w[:, ids].cpu().numpy().copy())
        if f % 60 == 0:
            P = rec["P"][-1]
            print(f"t={t:5.2f} tip z={P[0, -1, 2]:+.3f} rope zmin={P[:, :, 2].min():+.3f} step {t_wall[-1]*1e3:.1f} ms", flush=True)
    out = {k: np.asarray(v) for k, v in rec.items()}
    out["link_names"] = np.asarray(LINKS)
    out["ms_per_step"] = float(np.median(t_wall[10:]) * 1e3)
    out["num_envs"] = N
    out["rope_len"] = (V - 1) * env.cfg.rope_seg_len
    out["contact"] = int(args.contact)
    if env.bridge is not None:
        out["link_V"] = np.asarray([g[0] for g in env.bridge.geometry], dtype=object)
        out["link_F"] = np.asarray([g[1] for g in env.bridge.geometry], dtype=object)
    np.savez(args.out, **out)
    print("saved", args.out, "median step", out["ms_per_step"], "ms", flush=True)
    sys.stdout.flush()
    os._exit(0)     # Kit's headless teardown can hang; everything is on disk


if __name__ == "__main__":
    main()
