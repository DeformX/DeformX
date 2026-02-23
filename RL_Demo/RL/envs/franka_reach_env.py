"""
FrankaReachEnv (Isaac Sim) — vectorized (ArticulationView/RigidPrimView)

Design goals:
- Env owns SimulationApp lifecycle (simple for demo)
- Batch stepping for many envs
- obs: [ee_pos_rel(3), box_pos_rel(3)]  -> obs_dim=6
- act: 7-DoF joint velocity commands     -> act_dim=7

Note:
- This is a demo env. For serious training, you'll likely:
  - move SimulationApp ownership outside env
  - add proper episode tracking / logging
"""

from __future__ import annotations

import numpy as np

# -------------------------
# Isaac Sim App singleton
# -------------------------
_SIM_APP = None


def _ensure_sim_app(headless: bool):
    global _SIM_APP
    if _SIM_APP is None:
        from isaacsim import SimulationApp  # must be imported before omni/torch sometimes
        _SIM_APP = SimulationApp({"headless": bool(headless)})
    return _SIM_APP


class FrankaReachEnv:
    def __init__(self, cfg, *, headless: bool = True):
        """
        cfg is cfg.task.franka_reach (omegaconf DictConfig)
        """
        self.cfg = cfg
        self.headless = bool(headless)

        # Start Isaac Sim
        self.simulation_app = _ensure_sim_app(self.headless)

        # Imports AFTER SimulationApp
        import torch  # noqa: F401
        from omni.isaac.core import World
        from omni.isaac.franka import Franka
        from omni.isaac.core.objects import DynamicCuboid
        from omni.isaac.core.prims import RigidPrimView
        from omni.isaac.core.articulations import ArticulationView

        # Basic params
        self.num_envs = int(cfg.num_envs)
        self.env_spacing = float(getattr(cfg, "env_spacing", 2.0))
        self.phys_dt = float(getattr(cfg, "phys_dt", 1.0 / 60.0))
        self.max_steps = int(getattr(cfg, "max_steps", 300))

        self.success_thresh = float(getattr(cfg, "success_thresh", 0.08))
        self.reward_clip_min = float(getattr(cfg, "reward_clip_min", 0.0))
        self.reward_clip_max = float(getattr(cfg, "reward_clip_max", 5.0))

        self.max_joint_vel = float(getattr(cfg, "max_joint_vel", 1.5))

        self.obs_dim = 6
        self.act_dim = 7

        # Create World
        self.world = World(stage_units_in_meters=1.0, physics_dt=self.phys_dt, rendering_dt=self.phys_dt)
        self.world.scene.add_default_ground_plane()

        # Spawn env copies
        for i in range(self.num_envs):
            root = f"/World/Env_{i}"
            offset = np.array([i * self.env_spacing, 0.0, 0.0], dtype=np.float32)

            self.world.scene.add(
                Franka(
                    prim_path=f"{root}/Franka",
                    name=f"franka_{i}",
                    position=offset,
                )
            )

            self.world.scene.add(
                DynamicCuboid(
                    prim_path=f"{root}/Box",
                    name=f"box_{i}",
                    position=offset + np.array([0.5, 0.0, 0.05], dtype=np.float32),
                    scale=np.array([0.05, 0.05, 0.05], dtype=np.float32),
                    color=np.array([1.0, 0.0, 0.0], dtype=np.float32),
                    mass=0.5,
                )
            )

        # Views (vectorized handles)
        self.franka_view = ArticulationView(
            prim_paths_expr="/World/Env_*/Franka",
            name="franka_view",
        )
        self.world.scene.add(self.franka_view)

        self.box_view = RigidPrimView(
            prim_paths_expr="/World/Env_*/Box",
            name="box_view",
        )
        self.world.scene.add(self.box_view)

        self.ee_view = RigidPrimView(
            prim_paths_expr="/World/Env_*/Franka/panda_link8",
            name="ee_view",
        )
        self.world.scene.add(self.ee_view)

        # Reset once to initialize PhysX handles
        self.world.reset()

        self.num_dof = int(self.franka_view.num_dof)

        # PD gains (demo uses kp=0, kd>0 to allow velocity driving)
        kp = float(getattr(cfg, "kp", 0.0))
        kd = float(getattr(cfg, "kd", 40.0))
        kps = np.full((self.num_envs, self.num_dof), kp, dtype=np.float32)
        kds = np.full((self.num_envs, self.num_dof), kd, dtype=np.float32)
        self.franka_view.set_gains(kps=kps, kds=kds)

        self.step_count = np.zeros((self.num_envs,), dtype=np.int32)

        # warmup
        for _ in range(10):
            self.world.step(render=not self.headless)

        # initial reset
        self.reset()

    def close(self):
        # Close SimulationApp if this env owns it (demo behavior)
        # If you want multiple env instances, manage app outside instead.
        global _SIM_APP
        if _SIM_APP is not None:
            _SIM_APP.close()
            _SIM_APP = None

    def reset(self, ids=None):
        if ids is None:
            ids = np.arange(self.num_envs, dtype=np.int32)
        ids = np.asarray(ids, dtype=np.int32)
        if ids.size == 0:
            return self.obs()

        # Reset step counters
        self.step_count[ids] = 0

        # Randomize box positions
        rand_pos = np.zeros((len(ids), 3), dtype=np.float32)
        rand_pos[:, 0] = ids.astype(np.float32) * self.env_spacing + np.random.uniform(0.4, 0.6, size=len(ids)).astype(np.float32)
        rand_pos[:, 1] = np.random.uniform(-0.3, 0.3, size=len(ids)).astype(np.float32)
        rand_pos[:, 2] = 0.05
        self.box_view.set_world_poses(positions=rand_pos, indices=ids)

        # Reset joints to a home pose + noise
        home = np.array([0, -0.78, 0, -2.35, 0, 1.57, 0.78, 0, 0], dtype=np.float32)
        q = np.tile(home, (len(ids), 1))
        q += np.random.uniform(-0.1, 0.1, size=(len(ids), self.num_dof)).astype(np.float32)
        qd = np.zeros((len(ids), self.num_dof), dtype=np.float32)

        self.franka_view.set_joint_positions(q, indices=ids)
        self.franka_view.set_joint_velocities(qd, indices=ids)

        # Apply one step to commit reset
        self.world.step(render=False)

        return self.obs()

    def obs(self):
        ee_pos, _ = self.ee_view.get_world_poses()
        box_pos, _ = self.box_view.get_world_poses()

        env_offsets = np.zeros((self.num_envs, 3), dtype=np.float32)
        env_offsets[:, 0] = np.arange(self.num_envs, dtype=np.float32) * self.env_spacing

        rel_ee = ee_pos - env_offsets
        rel_box = box_pos - env_offsets

        obs = np.hstack([rel_ee, rel_box]).astype(np.float32)
        return obs

    def _reward_done_info(self, obs):
        # obs: (N,6)
        dist = np.linalg.norm(obs[:, 0:3] - obs[:, 3:6], axis=1).astype(np.float32)

        rew = 1.0 / (dist + 0.1)
        rew = np.clip(rew, self.reward_clip_min, self.reward_clip_max)

        success = dist < self.success_thresh
        rew = rew + success.astype(np.float32) * 2.0

        timeout = self.step_count >= self.max_steps
        done = np.logical_or(success, timeout)

        info = {
            "dist_mean": float(dist.mean()),
            "success_rate": float(success.mean()),
            "success_mask": success.copy(),
        }
        return rew.astype(np.float32), done.astype(np.bool_), info

    def step(self, actions: np.ndarray):
        """
        actions: (N, 7) in [-1, 1]
        Returns:
          obs, reward, done, info
        """
        actions = np.asarray(actions, dtype=np.float32)
        if actions.shape != (self.num_envs, self.act_dim):
            raise ValueError(f"Expected actions shape {(self.num_envs, self.act_dim)}, got {actions.shape}")

        actions = np.clip(actions, -1.0, 1.0)

        # build full dof velocity command (9 dof, last 2 gripper = 0)
        qd_cmd = np.zeros((self.num_envs, self.num_dof), dtype=np.float32)
        qd_cmd[:, :7] = actions * self.max_joint_vel

        self.franka_view.set_joint_velocities(qd_cmd)

        # step physics
        self.world.step(render=not self.headless)
        self.step_count += 1

        obs = self.obs()
        rew, done, info = self._reward_done_info(obs)

        # auto reset done envs
        reset_ids = np.where(done)[0].astype(np.int32)
        if reset_ids.size > 0:
            self.reset(reset_ids)

        return obs, rew, done, info
