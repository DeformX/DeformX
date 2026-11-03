"""RopeSwing DirectRLEnv on the GPU rod solver: the DeformX rope-swing task of
demo/rope_swing_rl with the wire simulated by ropesim instead of PyElastica.

Rig and task match the CPU version: a UR5e mounted at z = 1.7 m and rotated -90 deg about X
(the DeformX wall mount), a rigid 0.65 m stick from the wrist, a 1 m / 12.7 mm wire hanging
from its tip, and a fixed target in front of the arm. The policy drives three joints
(shoulder lift, elbow, wrist 1) with positive position-target deltas; the reward is the CPU
task's (proximity, progress, new-best, staged bonuses, tip velocity toward the target,
wrist-1 velocity, time / action / smoothness penalties, success bonus), distances are taken
in the Y-Z plane, and the episode ends on success, on timeout, or when the tip strays too far.

What differs comes from the solver, not the task:
- control at 60 Hz (the CPU task acts every 2 ms physics step); the joint delta per step is
  scaled by the control period, so the commanded joint speed is the same;
- the wire's bending stiffness is the CPU wire's (E = 2e5 Pa on a 6.35 mm radius: EI =
  2.55e-4 N m^2); its axial stiffness is EA = 3000 N rather than the 25 N the same E implies,
  which the CPU model cannot avoid (one modulus for both) and this solver can;
- dissipation is the solver's own: linear drag 0.2 /s, no internal damping, 100 substeps x
  8 iterations per frame;
- at the DeformX home pose the stick tip is ~0.4 m above the floor and the rope is 1 m long,
  so rod-plane contact with the floor is on, with the CPU task's ground friction. The reset
  shape is the rope settled on the floor, computed once at construction with a straight rest
  shape (from_positions(straight_rest=True)).

Wiring: PhysX at 60 Hz, decimation 1; one-way coupling with second-order root driving (after
each PhysX step the rope root follows a quadratic from the measured end-effector point
velocity to the frame-end pose); rope state lives only in the solver, observations are
zero-copy views.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply, quat_mul

import ropesim

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
from demo.common import paths as deformx_paths  # noqa: E402

# The UR5e, as in demo/rope_swing_rl: a local Isaac assets tree or NVIDIA's public mirror.
UR5E_USD = os.environ.get("ROPE_UR5E_USD") or deformx_paths.resolve_ur_robot_usd(prefer="ur5e")

JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]

# the DeformX wire (demo/rope_swing_rl/conf/task/wire_swing.yaml)
WIRE_RADIUS = 0.00635          # [m]
WIRE_DENSITY = 700.0           # [kg/m^3]
WIRE_YOUNGS = 2.0e5            # [Pa]; sets EI here, EA is the solver's (see module docstring)


@configclass
class RopeSwingEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 1
    episode_length_s = 1.4           # CPU task: max_steps 700 x phys_dt 0.002
    debug_vis = False                # target sphere, rope + stick cylinders, tip trails
    action_space = 3                 # position-target deltas of the active joints
    observation_space = 18           # jp(3) jv(3) tip(3) tip_vel(3) stick_tip(3) target(3)
    state_space = 0

    # control (CPU task): three active joints, deltas >= 0, 90 deg/s times a delta scale
    active_joints = (1, 2, 3)        # shoulder_lift, elbow, wrist_1
    positive_only_actions = True
    joint_vel_limits = (1.5708, 1.5708, 1.5708, 1.5708, 1.5708, 1.5708)   # [rad/s]
    joint_delta_scale = 3.0
    joint_pos_lower = (-6.28, -6.28, -6.28, -6.28, -6.28, -6.28)
    joint_pos_upper = (6.28, -1.92, -0.44, 6.28, 6.28, 6.28)

    sim: SimulationCfg = SimulationCfg(dt=1.0 / 60.0, render_interval=1)
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=512, env_spacing=8.0, replicate_physics=True)

    robot: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(usd_path=UR5E_USD, activate_contact_sensors=False),
        init_state=ArticulationCfg.InitialStateCfg(
            # DeformX wall mount: base at z = 1.7 m, -90 deg about X (the arm points along +Y)
            pos=(0.0, 0.0, 1.7),
            rot=(0.70710678, -0.70710678, 0.0, 0.0),   # wxyz
            joint_pos={                                # the DeformX home pose
                "shoulder_pan_joint": -1.5935,
                "shoulder_lift_joint": -2.2756,
                "elbow_joint": -1.1913,
                "wrist_1_joint": -0.3241,
                "wrist_2_joint": 1.5707,
                "wrist_3_joint": 3.1415,
            },
        ),
        actuators={
            # stiff position control like the real arm; PhysX joint speed capped at the UR5e
            # specification (180 deg/s)
            "arm": ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=1500.0, damping=150.0,
                                       velocity_limit_sim=float(np.pi)),
        },
    )

    ee_body_name = "wrist_3_link"
    tube_offset = (0.0, 0.0, 0.65)   # the rigid stick: rope root in the end-effector frame
    tube_dir_local = (0.0, 0.0, 1.0)

    # rope (units: rho [kg/m], EA [N], EI [N m^2])
    rope_num_verts = 21              # 20 segments x 5 cm = 1.0 m
    rope_seg_len = 0.05
    rope_radius = WIRE_RADIUS
    rope_density = float(WIRE_DENSITY * np.pi * WIRE_RADIUS**2)          # 0.0887 kg/m
    rope_ea = 3000.0
    rope_ei = float(WIRE_YOUNGS * np.pi * WIRE_RADIUS**4 / 4.0)          # 2.55e-4 N m^2
    rope_substeps = 100
    rope_num_itr = 8
    rope_drag = 0.2                  # [1/s] the solver's aerodynamic knob
    rope_damping = 0.0               # keep 0 (solver-internal damping brakes absolute velocity)

    # floor contact (CPU task: use_ground_contact, static mu 1.0, kinetic mu 0.5)
    floor_contact = True
    floor_mu_s = 1.0
    floor_mu_k = 0.5
    contact_v_slip = 1e-3
    contact_kappa = 2.0
    settle_seconds = 1.0             # settle the folded rope on the floor at construction

    # task (CPU task values)
    target_local = (0.0, 2.0, 1.5)   # [m] in the env frame (floor level under the base)
    success_thresh = 0.05
    touch_thresh = 0.03
    too_far_thresh = 4.0
    terminate_on_success = True
    reward_clip = (-10.0, 200.0)
    w_proximity = 1.0
    w_progress = 10.0
    w_new_best = 30.0
    w_tip_velocity = 1.0
    w_joint3_velocity = 0.5
    joint3_index = 3                 # wrist_1
    joint3_velocity_abs = False
    w_time_penalty = 0.05
    action_penalty = 0.05
    smooth_penalty = 0.02
    success_bonus = 50.0
    touch_bonus = 50.0
    bonus_thresholds = (0.5, 0.2, 0.1, 0.05)
    bonus_values = (1.0, 3.0, 5.0, 10.0)


class RopeSwingEnv(DirectRLEnv):
    cfg: RopeSwingEnvCfg

    def __init__(self, cfg: RopeSwingEnvCfg, render_mode: str | None = None, **kwargs):
        self._visualizer = None  # before super(): hooks may be probed early
        super().__init__(cfg, render_mode, **kwargs)
        dev = self.device
        N = self.num_envs

        self._joint_ids, joint_names = self.robot.find_joints(JOINT_NAMES, preserve_order=True)
        if len(self._joint_ids) != 6:
            raise RuntimeError(f"expected the 6 UR joints, found {joint_names} (available: {self.robot.joint_names})")
        ee_ids, _ = self.robot.find_bodies(self.cfg.ee_body_name)
        self._ee_idx = ee_ids[0] if ee_ids else self.robot.num_bodies - 1
        self._use_link_api = hasattr(self.robot.data, "body_link_lin_vel_w")

        # control: the active joints move by action x max_delta per step and stay inside the
        # CPU task's limits; with positive-only actions they also never go below home
        act = list(self.cfg.active_joints)
        self._active = torch.tensor([self._joint_ids[i] for i in act], device=dev, dtype=torch.long)
        self._default_joint_pos = self.robot.data.default_joint_pos.clone()
        self._targets = self._default_joint_pos.clone()
        self.actions = torch.zeros(N, self.cfg.action_space, device=dev)
        self._prev_actions = torch.zeros_like(self.actions)
        vel = torch.tensor(self.cfg.joint_vel_limits, device=dev)
        self._max_delta = (vel * self.physics_dt * self.cfg.joint_delta_scale)[act]          # [3]
        lo = torch.tensor(self.cfg.joint_pos_lower, device=dev)[act].expand(N, -1)
        hi = torch.tensor(self.cfg.joint_pos_upper, device=dev)[act].expand(N, -1)
        default_active = self._default_joint_pos[:, self._active]
        self._active_lo = torch.maximum(lo, default_active) if self.cfg.positive_only_actions else lo.clone()
        self._active_hi = hi.clone()
        self._joint3 = self._joint_ids[self.cfg.joint3_index]

        # the rigid stick: rope root in the end-effector frame
        self._tube_offset = torch.tensor(self.cfg.tube_offset, device=dev).expand(N, 3)
        x = np.array([1.0, 0.0, 0.0])
        d = np.asarray(self.cfg.tube_dir_local, dtype=np.float64)
        d = d / np.linalg.norm(d)
        axis = np.cross(x, d)
        s = np.linalg.norm(axis)
        if s < 1e-8:
            q_align = np.array([1.0, 0.0, 0.0, 0.0]) if d[0] > 0 else np.array([0.0, 0.0, 0.0, 1.0])
        else:
            axis = axis / s
            ang = float(np.arctan2(s, float(np.dot(x, d))))
            q_align = np.concatenate([[np.cos(ang / 2)], np.sin(ang / 2) * axis])
        self._q_align = torch.tensor(q_align, dtype=torch.float32, device=dev).expand(N, 4)

        # settle the arm at the home pose before building the rope (at __init__ time body
        # poses still reflect the USD spawn pose, not the configured joints)
        jp = self.robot.data.default_joint_pos.clone()
        self.robot.write_joint_state_to_sim(jp, torch.zeros_like(jp))
        self.robot.set_joint_position_target(jp)
        self.scene.write_data_to_sim()
        for _ in range(24):
            self.sim.step(render=False)
            self.scene.update(dt=self.physics_dt)

        # the rope: hanging from the stick tip, folded onto the floor and settled there; that
        # settled shape is the construction state, i.e. what every reset restores
        root_pos, _, root_q = self._compute_root_state()
        rest = self._settled_shape(root_pos)
        pin = np.zeros((N, self.cfg.rope_num_verts), bool)
        pin[:, 0] = True
        self.rope = ropesim.RopeBatch.from_positions(
            rest, self.cfg.rope_density, self.cfg.rope_ea, self.cfg.rope_ei,
            self.physics_dt / self.cfg.rope_substeps, self.cfg.rope_num_itr, pinned=pin, straight_rest=True)
        self._configure_rope(self.rope)
        self._rope_pos = torch.as_tensor(self.rope.positions())      # zero-copy views
        self._rope_vel = torch.as_tensor(self.rope.velocities())
        self._init_root_pos = root_pos.clone()
        self._init_root_q = root_q.clone()
        self._prev_root_pos = root_pos.clone()
        self._prev_root_q = root_q.clone()
        self._prev_root_vel = torch.zeros_like(root_pos)

        # task state
        self._target_w = self.scene.env_origins + torch.tensor(self.cfg.target_local, device=dev)
        tip0 = self._rope_pos[:, -1, :].clone()
        self._prev_tip = tip0
        self._tip_vel = torch.zeros(N, 3, device=dev)
        self._dist = self._yz_dist(tip0)
        self._prev_dist = self._dist.clone()
        self._min_dist = self._dist.clone()
        self._success = torch.zeros(N, dtype=torch.bool, device=dev)
        self._success_buf = torch.zeros(N, dtype=torch.bool, device=dev)
        self._died = torch.zeros(N, dtype=torch.bool, device=dev)

        if self.cfg.debug_vis:
            self.set_debug_vis(True)

    # ------------------------------------------------------------------ rope
    def _configure_rope(self, rope):
        rope.set_gravity([0.0, 0.0, -9.81])
        rope.drag = self.cfg.rope_drag
        rope.damping = self.cfg.rope_damping
        if self.cfg.floor_contact:
            rope.set_planes(np.array([[0.0, 0.0, 1.0, 0.0]], np.float32), radius=self.cfg.rope_radius,
                            mu_s=self.cfg.floor_mu_s, mu_k=self.cfg.floor_mu_k,
                            v_slip=self.cfg.contact_v_slip, kappa=self.cfg.contact_kappa)

    def _settled_shape(self, root_pos: torch.Tensor) -> np.ndarray:
        """[N, V, 3] float32: the rope hanging from the stick tip. Without floor contact it hangs
        straight down. With it, the part that would be below the floor is folded along +Y on
        the floor and the shape is settled for cfg.settle_seconds in a throwaway solver."""
        cfg = self.cfg
        N, V, L = self.num_envs, cfg.rope_num_verts, cfg.rope_seg_len
        root = root_pos.detach().cpu().numpy().astype(np.float64)
        s = np.arange(V) * L
        pos = np.empty((N, V, 3), np.float64)
        pos[:, :, 0] = root[:, None, 0]
        pos[:, :, 1] = root[:, None, 1]
        pos[:, :, 2] = root[:, None, 2] - s[None, :]
        if not cfg.floor_contact:
            return pos.astype(np.float32)
        z0 = cfg.rope_radius + 0.002                                   # resting height of the centreline
        drop = np.clip(root[:, 2] - z0, 0.0, None)                     # vertical run per env
        on_floor = s[None, :] > drop[:, None]
        pos[:, :, 1] = np.where(on_floor, root[:, None, 1] + (s[None, :] - drop[:, None]), root[:, None, 1])
        pos[:, :, 2] = np.where(on_floor, z0, root[:, None, 2] - s[None, :])
        pos[:, :, 0] += 1e-3 * s[None, :]                              # a hair of tilt: no exactly vertical column
        pos = pos.astype(np.float32)
        if cfg.settle_seconds <= 0:
            return pos
        pin = np.zeros((N, V), bool)
        pin[:, 0] = True
        tmp = ropesim.RopeBatch.from_positions(pos, cfg.rope_density, cfg.rope_ea, cfg.rope_ei,
                                               self.physics_dt / cfg.rope_substeps, max(cfg.rope_num_itr, 16),
                                               pinned=pin, straight_rest=True)       # extra iterations: a converged fold
        self._configure_rope(tmp)
        tmp.drag = 3.0                                                  # heavy air drag: settle fast
        zero = torch.zeros_like(root_pos)
        for _ in range(int(round(cfg.settle_seconds / self.physics_dt))):
            tmp.step(root_pos, zero, dt=self.physics_dt, substeps=cfg.rope_substeps)
        settled = torch.as_tensor(tmp.positions()).cpu().numpy().astype(np.float32)
        del tmp
        return settled

    def _ee_state(self):
        d = self.robot.data
        if self._use_link_api:
            return (d.body_link_pos_w[:, self._ee_idx], d.body_link_quat_w[:, self._ee_idx],
                    d.body_link_lin_vel_w[:, self._ee_idx], d.body_link_ang_vel_w[:, self._ee_idx])
        return (d.body_pos_w[:, self._ee_idx], d.body_quat_w[:, self._ee_idx],
                d.body_lin_vel_w[:, self._ee_idx], d.body_ang_vel_w[:, self._ee_idx])

    def _compute_root_state(self):
        """World root pos [N,3], root-point velocity [N,3] (end-effector velocity plus the
        omega x r term of the rigid stick) and the root frame quat (x,y,z,w) [N,4]."""
        ee_pos, ee_quat, ee_lin, ee_ang = self._ee_state()
        r = quat_apply(ee_quat, self._tube_offset)
        root_pos = ee_pos + r
        root_vel = ee_lin + torch.cross(ee_ang, r, dim=-1)
        root_q = quat_mul(ee_quat, self._q_align)
        root_q_xyzw = torch.cat([root_q[:, 1:4], root_q[:, 0:1]], dim=-1)
        return root_pos.contiguous(), root_vel.contiguous(), root_q_xyzw.contiguous()

    def _yz_dist(self, tip):
        d = (tip - self._target_w)[:, 1:3].norm(dim=-1)
        return torch.nan_to_num(d, nan=1.0e3, posinf=1.0e3, neginf=1.0e3)

    def _step_rope(self):
        """Drive every rope root over the PhysX step that just ran, then refresh the tip metrics."""
        root_pos, root_vel, root_q = self._compute_root_state()
        self.rope.step(self._prev_root_pos, self._prev_root_vel, dt=self.physics_dt,
                       substeps=self.cfg.rope_substeps, root_pos_end=root_pos)
        self._prev_root_pos, self._prev_root_vel, self._prev_root_q = root_pos, root_vel, root_q
        tip = self._rope_pos[:, -1, :]
        self._tip_vel = torch.nan_to_num((tip - self._prev_tip) / self.physics_dt, nan=0.0, posinf=0.0, neginf=0.0)
        self._prev_tip = tip.clone()
        self._dist = self._yz_dist(tip)
        if self._visualizer is not None:
            self._visualizer.record_tip()

    # ------------------------------------------------------------------ mdp
    def _pre_physics_step(self, actions: torch.Tensor):
        self.extras.pop("log", None)   # DirectRLEnv never clears extras; rsl_rl re-appends stale dicts
        a = torch.nan_to_num(actions, nan=0.0, posinf=1.0, neginf=-1.0)
        self.actions = torch.clamp(a, 0.0 if self.cfg.positive_only_actions else -1.0, 1.0)

    def _apply_action(self):
        nxt = self._targets[:, self._active] + self.actions * self._max_delta
        self._targets[:, self._active] = torch.minimum(torch.maximum(nxt, self._active_lo), self._active_hi)
        self.robot.set_joint_position_target(self._targets)

    def _get_dones(self):
        self._step_rope()                                               # poses are fresh after physics
        self._died = ~torch.isfinite(self._rope_pos[:, -1, :]).all(dim=-1)
        self._success = self._dist < self.cfg.success_thresh
        self._success_buf |= self._success
        terminated = self._died | (self._dist > self.cfg.too_far_thresh)
        if self.cfg.terminate_on_success:
            terminated = terminated | self._success
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, time_out

    def _get_rewards(self):
        cfg = self.cfg
        dist = self._dist
        tip = self._rope_pos[:, -1, :]
        to_target = self._target_w - tip
        to_target = to_target.clone()
        to_target[:, 0] = 0.0                                           # the task lives in the Y-Z plane
        to_target_dir = to_target / dist.clamp(min=1e-6).unsqueeze(-1)
        tip_speed_toward = (self._tip_vel * to_target_dir).sum(dim=-1)

        old_min = self._min_dist
        r_proximity = cfg.w_proximity * torch.exp(-dist)
        r_progress = cfg.w_progress * (self._prev_dist - dist)
        improvement = (old_min - dist).clamp(min=0.0)
        r_new_best = cfg.w_new_best * (dist < old_min).float() * improvement
        r_bonus = torch.zeros_like(dist)
        for thr, val in zip(cfg.bonus_thresholds, cfg.bonus_values):
            r_bonus = r_bonus + (dist < thr).float() * val
        r_touch = (dist < cfg.touch_thresh).float() * cfg.touch_bonus
        r_tip_velocity = cfg.w_tip_velocity * tip_speed_toward
        jv3 = torch.nan_to_num(self.robot.data.joint_vel[:, self._joint3], nan=0.0, posinf=0.0, neginf=0.0)
        r_joint3 = cfg.w_joint3_velocity * (jv3.abs() if cfg.joint3_velocity_abs else jv3)
        r_time = -cfg.w_time_penalty * torch.ones_like(dist)
        r_action = -cfg.action_penalty * self.actions.square().sum(dim=-1)
        r_smooth = -cfg.smooth_penalty * (self.actions - self._prev_actions).square().sum(dim=-1)
        r_success = self._success.float() * cfg.success_bonus
        reward = (r_proximity + r_progress + r_new_best + r_bonus + r_touch + r_tip_velocity
                  + r_joint3 + r_time + r_action + r_smooth + r_success)
        reward = torch.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0)
        reward = torch.clamp(reward, cfg.reward_clip[0], cfg.reward_clip[1])
        reward = torch.where(self._died, torch.zeros_like(reward), reward)   # a NaN would poison the batch

        self._prev_actions = self.actions.clone()
        self._prev_dist = dist.clone()
        self._min_dist = torch.minimum(old_min, dist)
        return reward

    def _get_observations(self):
        jp = self.robot.data.joint_pos[:, self._active]
        jv = self.robot.data.joint_vel[:, self._active]
        origins = self.scene.env_origins
        tip_local = self._rope_pos[:, -1, :] - origins
        stick_tip_local = self._prev_root_pos - origins
        target_local = self._target_w - origins
        obs = torch.cat([jp, jv, tip_local, self._tip_vel, stick_tip_local, target_local], dim=-1)
        obs = torch.nan_to_num(obs, nan=0.0, posinf=1.0e3, neginf=-1.0e3).clamp(-1.0e3, 1.0e3)
        return {"policy": obs}

    def _reset_idx(self, env_ids):
        if env_ids is None:   # tensor[None] would add a dimension instead of erroring
            env_ids = self.robot._ALL_INDICES
        finite_min = self._min_dist[env_ids]
        finite_min = finite_min[torch.isfinite(finite_min)]
        self.extras["log"] = {
            "Metrics/success_rate": self._success_buf[env_ids].float().mean().item(),
            "Metrics/min_dist": finite_min.mean().item() if finite_min.numel() else 0.0,
        }
        super()._reset_idx(env_ids)

        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        self.robot.write_joint_state_to_sim(joint_pos, torch.zeros_like(joint_pos), env_ids=env_ids)
        self._targets[env_ids] = joint_pos
        self.robot.set_joint_position_target(self._targets[env_ids], env_ids=env_ids)

        ids = env_ids.detach().cpu().numpy().astype(np.int32)
        self.rope.reset_envs(ids)                                       # the settled shape, at rest
        self._prev_root_pos[env_ids] = self._init_root_pos[env_ids]
        self._prev_root_q[env_ids] = self._init_root_q[env_ids]
        self._prev_root_vel[env_ids] = 0.0
        tip0 = self._rope_pos[env_ids, -1, :]
        self._prev_tip[env_ids] = tip0
        self._tip_vel[env_ids] = 0.0
        d0 = self._yz_dist(self._rope_pos[:, -1, :])[env_ids]
        self._dist[env_ids] = d0
        self._prev_dist[env_ids] = d0
        self._min_dist[env_ids] = d0
        self._success[env_ids] = False
        self._success_buf[env_ids] = False
        self._prev_actions[env_ids] = 0.0
        self.actions[env_ids] = 0.0
        if self._visualizer is not None:
            self._visualizer.reset_trail(env_ids)

    # ------------------------------------------------------------- debug vis
    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if self._visualizer is None:
                from .visualization import RopeSwingVisualizer
                self._visualizer = RopeSwingVisualizer(self)
            self._visualizer.set_visibility(True)
        elif self._visualizer is not None:
            self._visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # Kit fires post-update events on any app update (clicks, timeline stop, teardown);
        # visualization must never take the app down
        if self._visualizer is None or getattr(self, "scene", None) is None:
            return
        try:
            self._visualizer.update()
        except Exception as e:  # noqa: BLE001
            if not getattr(self, "_vis_error_reported", False):
                self._vis_error_reported = True
                print(f"[RopeSwing] debug-vis update failed (suppressed): {e!r}")

    def close(self):
        self._visualizer = None
        super().close()

    # ------------------------------------------------------------------ scene
    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot)
        ground = sim_utils.GroundPlaneCfg()
        ground.func("/World/ground", ground)
        light = sim_utils.DomeLightCfg(intensity=2000.0)
        light.func("/World/light", light)
        self.scene.clone_environments(copy_from_source=False)
        self.scene.articulations["robot"] = self.robot
