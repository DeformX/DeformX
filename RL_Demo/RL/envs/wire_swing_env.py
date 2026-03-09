"""
WireSwingEnv (Isaac Sim + CoSimEngine + SkeletonRodDriver)

Task:
- Control selected UR5e joints to swing a rope tip to a fixed target.
- Robot physics runs in Isaac Sim PhysX.
- Rope physics runs in co_sim.CoSimEngine (PyElastica).
- Rope skeleton visual can be driven by SkeletonRodDriver.
"""

from __future__ import annotations

import os
import sys
import inspect
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
PYELASTICA_MESH_ROOT = REPO_ROOT / "PyElastica-Mesh"
if PYELASTICA_MESH_ROOT.is_dir() and str(PYELASTICA_MESH_ROOT) not in sys.path:
    sys.path.insert(0, str(PYELASTICA_MESH_ROOT))

from co_sim.engine import CoSimEngine
from co_sim.models import CoSimConfig, FrameState


_SIM_APP = None


def _ensure_sim_app(headless: bool):
    global _SIM_APP
    if _SIM_APP is None:
        try:
            from isaacsim import SimulationApp
        except ImportError:
            from omni.isaac.kit import SimulationApp
        app_cfg = {"headless": bool(headless)}
        if not bool(headless):
            app_cfg.update({"physics_gpu": 0, "active_gpu": 0})
        _SIM_APP = SimulationApp(app_cfg)
    return _SIM_APP


def _orthonormalize(R: np.ndarray) -> np.ndarray:
    u, _, vt = np.linalg.svd(R)
    Rn = u @ vt
    if np.linalg.det(Rn) < 0.0:
        u[:, -1] *= -1.0
        Rn = u @ vt
    return Rn


def _rotation_matrix_to_rotvec(R: np.ndarray) -> np.ndarray:
    R = _orthonormalize(R)
    tr = np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0)
    angle = float(np.arccos(tr))
    if angle < 1.0e-9:
        return np.zeros(3, dtype=np.float64)
    s = np.sin(angle)
    if abs(s) < 1.0e-9:
        return np.zeros(3, dtype=np.float64)
    axis = np.array(
        [
            (R[2, 1] - R[1, 2]) / (2.0 * s),
            (R[0, 2] - R[2, 0]) / (2.0 * s),
            (R[1, 0] - R[0, 1]) / (2.0 * s),
        ],
        dtype=np.float64,
    )
    return axis * angle


def _resolve_ur5e_usd_path() -> str:
    local_candidates = [
        "/home/robot/isaacsim_assets/Assets/Isaac/4.5/Isaac/Robots/UniversalRobots/ur5/ur5.usd",
        "/home/robot/isaacsim_assets/Assets/Isaac/5.1/Isaac/Robots/UniversalRobots/ur5/ur5.usd",
        "/home/robot/isaacsim_assets/Assets/Isaac/4.5/Isaac/Robots/UniversalRobots/ur5e/ur5e.usd",
        "/home/robot/isaacsim_assets/Assets/Isaac/5.1/Isaac/Robots/UniversalRobots/ur5e/ur5e.usd",
        "/home/robot/isaacsim/Assets/Isaac/4.5/Isaac/Robots/UniversalRobots/ur5e/ur5e.usd",
    ]
    for path in local_candidates:
        if os.path.exists(path):
            return path
    return (
        "https://omniverse-content-production.s3-us-west-2.amazonaws.com/"
        "Assets/Isaac/4.5/Isaac/Robots/UniversalRobots/ur5/ur5.usd"
    )


class WireSwingEnv:
    def __init__(self, cfg, *, headless: bool = True):
        self.cfg = cfg
        self.headless = bool(headless)
        self.simulation_app = _ensure_sim_app(self.headless)

        # Imports after SimulationApp is up.
        global Usd, UsdGeom, UsdPhysics, Gf, PhysxSchema, UsdLux
        global World, add_reference_to_stage, ArticulationView, omni
        from pxr import Usd, UsdGeom, UsdPhysics, Gf, PhysxSchema, UsdLux
        from omni.isaac.core import World
        from omni.isaac.core.utils.stage import add_reference_to_stage
        from omni.isaac.core.articulations import ArticulationView
        import omni
        import omni.usd

        # Core env params
        self.num_envs = int(getattr(cfg, "num_envs", 8))
        self.env_spacing = float(getattr(cfg, "env_spacing", 3.0))
        self.phys_dt = float(getattr(cfg, "phys_dt", 1.0 / 60.0))
        self.num_substeps = int(getattr(cfg, "num_substeps", 2))
        self.max_steps = int(getattr(cfg, "max_steps", 300))
        self.init_warmup_steps = int(getattr(cfg, "init_warmup_steps", 20))

        # Robot/task params
        self.num_robot_dofs = int(getattr(cfg, "num_robot_dofs", 6))
        self.joint_names = [
            "shoulder_pan",
            "shoulder_lift",
            "elbow",
            "wrist_1",
            "wrist_2",
            "wrist_3",
        ][: self.num_robot_dofs]
        self.end_effector_link = str(getattr(cfg, "end_effector_link", "wrist_3_link"))
        self.active_joints = np.asarray(getattr(cfg, "active_joints", [1, 2, 3]), dtype=np.int64)
        self.act_dim = int(self.active_joints.size)
        if self.act_dim <= 0:
            raise ValueError("active_joints must contain at least one joint index.")
        if np.any(self.active_joints < 0) or np.any(self.active_joints >= self.num_robot_dofs):
            raise ValueError(
                f"active_joints must be in [0, {self.num_robot_dofs - 1}], got {self.active_joints.tolist()}"
            )
        self.obs_dim = int(self.act_dim * 2 + 3 + 3 + 3 + 3)

        self.default_joint_positions = np.asarray(
            getattr(
                cfg,
                "default_joint_positions",
                [-1.5935, -2.2756, -1.1913, -0.3241, 1.7070, 3.0234],
            ),
            dtype=np.float32,
        )
        if self.default_joint_positions.shape != (self.num_robot_dofs,):
            raise ValueError(
                "default_joint_positions must match num_robot_dofs. "
                f"Got {self.default_joint_positions.shape}, expected ({self.num_robot_dofs},)."
            )

        self.robot_offset = np.asarray(getattr(cfg, "robot_offset", [0.0, 0.0, 1.7]), dtype=np.float64)
        self.robot_orient_xyz_deg = np.asarray(
            getattr(cfg, "robot_orient_xyz_deg", [-90.0, 0.0, 0.0]), dtype=np.float64
        )
        self.target_local = np.asarray(getattr(cfg, "target_local", [0.0, 1.7, 2.5]), dtype=np.float64)

        self.stick_length = float(getattr(cfg, "stick_length", 0.65))
        self.stick_radius = float(getattr(cfg, "stick_radius", 0.010))

        # Co-sim rope config
        self.wire_base_length = float(getattr(cfg, "wire_base_length", 1.0))
        self.wire_n_elem = int(getattr(cfg, "wire_n_elem", 20))
        self.wire_base_radius = float(getattr(cfg, "wire_base_radius", 0.006))
        self.py_dt = float(getattr(cfg, "py_dt", 2.0e-5))
        self.wire_density = float(getattr(cfg, "wire_density", 500.0))
        self.wire_youngs_modulus = float(getattr(cfg, "wire_youngs_modulus", 1.0e5))
        self.wire_shear_modulus_ratio = float(getattr(cfg, "wire_shear_modulus_ratio", 1.5))
        self.wire_damping_constant = float(getattr(cfg, "wire_damping_constant", 0.05))
        self.joint_k = float(getattr(cfg, "joint_k", 500.0))
        self.joint_nu = float(getattr(cfg, "joint_nu", 20.0))
        self.joint_kt = float(getattr(cfg, "joint_kt", 10.0))
        self.joint_nut = float(getattr(cfg, "joint_nut", 0.0))
        self.joint_model = str(getattr(cfg, "joint_model", "fixed")).strip().lower()
        self.use_ground_contact = bool(getattr(cfg, "use_ground_contact", False))
        self.ground_z = float(getattr(cfg, "ground_z", 0.0))
        self.ground_contact_k = float(getattr(cfg, "ground_contact_k", 1.0e4))
        self.ground_contact_nu = float(getattr(cfg, "ground_contact_nu", 5.0))
        self.ground_static_mu = np.asarray(getattr(cfg, "ground_static_mu", [1.0, 1.0, 1.0]), dtype=np.float64)
        self.ground_kinetic_mu = np.asarray(getattr(cfg, "ground_kinetic_mu", [0.5, 0.5, 0.5]), dtype=np.float64)
        self.ground_slip_velocity_tol = float(getattr(cfg, "ground_slip_velocity_tol", 1.0e-6))
        self.settle_time = float(getattr(cfg, "settle_time", 1.0))
        raw_initial_wire_theta = getattr(cfg, "initial_wire_theta", None)
        if isinstance(raw_initial_wire_theta, str):
            token = raw_initial_wire_theta.strip().lower()
            if token in ("", "none", "null", "~"):
                self.initial_wire_theta = None
            else:
                self.initial_wire_theta = float(raw_initial_wire_theta)
        elif raw_initial_wire_theta is None:
            self.initial_wire_theta = None
        else:
            self.initial_wire_theta = float(raw_initial_wire_theta)
        self.axial_stretch_stiffening = float(getattr(cfg, "axial_stretch_stiffening", 1.0))
        self._warned_unsupported_joint_model = False

        # Action and limits
        self.joint_pos_limits = np.asarray(
            getattr(cfg, "joint_pos_limits", [6.28, 6.28, 6.28, 6.28, 6.28, 6.28]),
            dtype=np.float32,
        )
        lower_cfg = getattr(cfg, "joint_pos_lower_limits", None)
        upper_cfg = getattr(cfg, "joint_pos_upper_limits", None)
        if lower_cfg is None and upper_cfg is None:
            self.joint_pos_lower_limits = -self.joint_pos_limits.copy()
            self.joint_pos_upper_limits = self.joint_pos_limits.copy()
        else:
            if lower_cfg is None or upper_cfg is None:
                raise ValueError(
                    "Set both joint_pos_lower_limits and joint_pos_upper_limits, or neither."
                )
            self.joint_pos_lower_limits = np.asarray(lower_cfg, dtype=np.float32)
            self.joint_pos_upper_limits = np.asarray(upper_cfg, dtype=np.float32)
        self.joint_vel_limits = np.asarray(
            getattr(cfg, "joint_vel_limits", [1.5708, 1.5708, 1.5708, 1.5708, 1.5708, 1.5708]),
            dtype=np.float32,
        )
        if self.joint_pos_limits.shape != (self.num_robot_dofs,):
            raise ValueError(
                "joint_pos_limits must match num_robot_dofs. "
                f"Got {self.joint_pos_limits.shape}, expected ({self.num_robot_dofs},)."
            )
        if self.joint_pos_lower_limits.shape != (self.num_robot_dofs,):
            raise ValueError(
                "joint_pos_lower_limits must match num_robot_dofs. "
                f"Got {self.joint_pos_lower_limits.shape}, expected ({self.num_robot_dofs},)."
            )
        if self.joint_pos_upper_limits.shape != (self.num_robot_dofs,):
            raise ValueError(
                "joint_pos_upper_limits must match num_robot_dofs. "
                f"Got {self.joint_pos_upper_limits.shape}, expected ({self.num_robot_dofs},)."
            )
        if np.any(self.joint_pos_lower_limits > self.joint_pos_upper_limits):
            raise ValueError("joint_pos_lower_limits must be <= joint_pos_upper_limits for all joints.")
        if np.any(self.default_joint_positions < self.joint_pos_lower_limits) or np.any(
            self.default_joint_positions > self.joint_pos_upper_limits
        ):
            raise ValueError("default_joint_positions must lie within [joint_pos_lower_limits, joint_pos_upper_limits].")
        if self.joint_vel_limits.shape != (self.num_robot_dofs,):
            raise ValueError(
                "joint_vel_limits must match num_robot_dofs. "
                f"Got {self.joint_vel_limits.shape}, expected ({self.num_robot_dofs},)."
            )
        self.control_dt = float(self.phys_dt * self.num_substeps)
        delta_scale = float(getattr(cfg, "joint_delta_scale", 1.0))
        self.joint_max_delta = self.joint_vel_limits * self.control_dt * delta_scale
        self.positive_only_active_joints = bool(getattr(cfg, "positive_only_active_joints", False))

        # Reward/termination
        self.success_thresh = float(getattr(cfg, "success_thresh", 0.3))
        self.touch_thresh = float(getattr(cfg, "touch_thresh", 0.03))
        self.too_far_thresh = float(getattr(cfg, "too_far_thresh", 4.0))
        self.reward_clip_min = float(getattr(cfg, "reward_clip_min", -10.0))
        self.reward_clip_max = float(getattr(cfg, "reward_clip_max", 200.0))
        self.action_penalty = float(getattr(cfg, "action_penalty", 0.05))
        self.smooth_penalty = float(getattr(cfg, "smooth_penalty", 0.02))
        self.w_proximity = float(getattr(cfg, "w_proximity", 1.0))
        self.w_progress = float(getattr(cfg, "w_progress", 10.0))
        self.w_new_best = float(getattr(cfg, "w_new_best", 30.0))
        self.w_tip_velocity_toward_target = float(
            getattr(cfg, "w_tip_velocity_toward_target", 0.0)
        )
        self.w_joint3_velocity = float(getattr(cfg, "w_joint3_velocity", 0.0))
        self.joint3_reward_index = int(getattr(cfg, "joint3_reward_index", 3))
        self.joint3_velocity_abs = bool(getattr(cfg, "joint3_velocity_abs", True))
        self.w_time_penalty = float(getattr(cfg, "w_time_penalty", 0.0))
        self.success_bonus = float(getattr(cfg, "success_bonus", 0.0))
        self.bonus_thresh_1 = float(getattr(cfg, "bonus_thresh_1", 0.5))
        self.bonus_thresh_2 = float(getattr(cfg, "bonus_thresh_2", 0.2))
        self.bonus_thresh_3 = float(getattr(cfg, "bonus_thresh_3", 0.1))
        self.bonus_thresh_4 = float(getattr(cfg, "bonus_thresh_4", 0.05))
        self.bonus_value_1 = float(getattr(cfg, "bonus_value_1", 1.0))
        self.bonus_value_2 = float(getattr(cfg, "bonus_value_2", 3.0))
        self.bonus_value_3 = float(getattr(cfg, "bonus_value_3", 5.0))
        self.bonus_value_4 = float(getattr(cfg, "bonus_value_4", 10.0))
        self.touch_bonus = float(getattr(cfg, "touch_bonus", 50.0))

        self.swing_done_enabled = bool(getattr(cfg, "swing_done_enabled", True))
        self.swing_grace_steps = int(getattr(cfg, "swing_grace_steps", 60))
        self.swing_fallback_steps = int(getattr(cfg, "swing_fallback_steps", 45))
        self.terminate_on_success = bool(getattr(cfg, "terminate_on_success", False))
        if self.joint3_reward_index < 0 or self.joint3_reward_index >= self.num_robot_dofs:
            raise ValueError(
                f"joint3_reward_index must be in [0, {self.num_robot_dofs - 1}], "
                f"got {self.joint3_reward_index}."
            )

        # Optional rope visual: skeleton driver or debug spheres.
        self.enable_wire_visual = bool(getattr(cfg, "enable_wire_visual", not self.headless))
        self.max_visual_envs = int(getattr(cfg, "max_visual_envs", 1))
        mode_default = "skeleton" if self.enable_wire_visual else "none"
        self.wire_visual_mode = str(getattr(cfg, "wire_visual_mode", mode_default)).strip().lower()
        if self.wire_visual_mode in ("off", "false"):
            self.wire_visual_mode = "none"
        if self.wire_visual_mode not in ("none", "skeleton", "debug_spheres"):
            print(f"[WireSwingEnv] Unknown wire_visual_mode '{self.wire_visual_mode}', fallback to 'none'.")
            self.wire_visual_mode = "none"
        if not self.enable_wire_visual:
            self.wire_visual_mode = "none"
        self.wire_usd = str(
            getattr(
                cfg,
                "wire_usd",
                "/home/robot/Workspace/Siemens_Cable_Simulator/usd/wire_usdc/wire_usdc/wire_yellow_s20_r0.005_l1.usdc",
            )
        )
        if self.wire_visual_mode == "skeleton" and not os.path.isfile(self.wire_usd):
            print(f"[WireSwingEnv] wire_usd not found, disable visual driver: {self.wire_usd}")
            self.wire_visual_mode = "none"
        self.max_visual_nodes = int(getattr(cfg, "max_visual_nodes", self.wire_n_elem + 1))
        self.wire_debug_sphere_radius = float(getattr(cfg, "wire_debug_sphere_radius", 0.015))
        self.wire_debug_sphere_color = np.asarray(
            getattr(cfg, "wire_debug_sphere_color", [1.0, 0.0, 0.0]), dtype=np.float64
        )
        if self.wire_debug_sphere_color.shape != (3,):
            raise ValueError("wire_debug_sphere_color must be length-3 RGB values.")
        if self.wire_debug_sphere_radius <= 0.0:
            raise ValueError("wire_debug_sphere_radius must be > 0.")
        self.skeleton_driver_cls = None
        if self.wire_visual_mode == "skeleton":
            SkeletonRodDriver = None
            try:
                from tools.rod_skel_driver_sim import SkeletonRodDriver
            except Exception:
                try:
                    from rod_skel_driver_sim import SkeletonRodDriver
                except Exception as exc:
                    print(f"[WireSwingEnv] Failed to import SkeletonRodDriver, disable visual driver: {exc}")
                    self.wire_visual_mode = "none"
                    SkeletonRodDriver = None
            self.skeleton_driver_cls = SkeletonRodDriver
        print(f"[WireSwingEnv] wire_visual_mode={self.wire_visual_mode}")

        self.episode_steps = np.zeros((self.num_envs,), dtype=np.int32)
        self.best_approach_step = np.zeros((self.num_envs,), dtype=np.int32)
        self.prev_actions = np.zeros((self.num_envs, self.act_dim), dtype=np.float32)
        self.prev_dist = np.full((self.num_envs,), 1.0e9, dtype=np.float32)
        self.min_dist = np.full((self.num_envs,), 1.0e9, dtype=np.float32)
        self.commanded_q = np.tile(
            self.default_joint_positions.reshape(1, self.num_robot_dofs),
            (self.num_envs, 1),
        ).astype(np.float32)

        self.tip_world = np.zeros((self.num_envs, 3), dtype=np.float64)
        self.prev_tip_world = np.zeros((self.num_envs, 3), dtype=np.float64)
        self.tip_vel_world = np.zeros((self.num_envs, 3), dtype=np.float64)
        self.ee_tip_world = np.zeros((self.num_envs, 3), dtype=np.float64)

        self.engines = [None for _ in range(self.num_envs)]
        self.kin_state = [None for _ in range(self.num_envs)]
        self.wire_drivers = [None for _ in range(self.num_envs)]
        self.wire_debug_marker_ops = [[] for _ in range(self.num_envs)]
        self._wire_update_warned = np.zeros((self.num_envs,), dtype=np.bool_)
        self.last_good_rod_pos = [None for _ in range(self.num_envs)]
        self.last_good_rod_dir = [None for _ in range(self.num_envs)]
        self.ee_link_paths: list[str] = []
        self.stick_tip_paths: list[str] = []

        self._build_scene()
        self.reset()

    def close(self):
        global _SIM_APP
        if _SIM_APP is not None:
            _SIM_APP.close()
            _SIM_APP = None

    def _build_scene(self):
        self.world = World(
            stage_units_in_meters=1.0,
            physics_dt=self.phys_dt,
            rendering_dt=self.phys_dt,
        )
        self.stage = omni.usd.get_context().get_stage()
        self.world.scene.add_default_ground_plane()

        physics_scene = UsdPhysics.Scene.Define(self.stage, "/physicsScene")
        physics_scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
        physics_scene.CreateGravityMagnitudeAttr().Set(9.81)
        PhysxSchema.PhysxSceneAPI.Apply(
            self.stage.GetPrimAtPath("/physicsScene")
        ).CreateEnableGPUDynamicsAttr().Set(True)

        dome = UsdLux.DomeLight.Define(self.stage, "/World/DomeLight")
        dome.CreateIntensityAttr().Set(0.0)
        dome.CreateColorAttr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
        distant = UsdLux.DistantLight.Define(self.stage, "/World/DistantLight")
        distant.CreateIntensityAttr().Set(2500.0)
        distant.CreateColorAttr().Set(Gf.Vec3f(1.0, 1.0, 1.0))

        ur5e_usd_path = str(getattr(self.cfg, "ur5e_usd_path", _resolve_ur5e_usd_path()))
        self.env_origins = np.zeros((self.num_envs, 3), dtype=np.float64)

        for i in range(self.num_envs):
            root = f"/World/Env_{i}"
            origin = np.array([i * self.env_spacing, 0.0, 0.0], dtype=np.float64)
            self.env_origins[i] = origin
            UsdGeom.Xform.Define(self.stage, root)

            self._create_target(root, origin + self.target_local)

            robot_path = f"{root}/UR5e"
            add_reference_to_stage(usd_path=ur5e_usd_path, prim_path=robot_path)

            robot_xf = UsdGeom.Xformable(self.stage.GetPrimAtPath(robot_path))
            robot_xf.ClearXformOpOrder()
            robot_xf.AddTranslateOp().Set(Gf.Vec3d(*(origin + self.robot_offset)))
            robot_xf.AddRotateXYZOp().Set(Gf.Vec3f(*self.robot_orient_xyz_deg))

            self._set_ur5e_default_joint_targets(robot_path)

            ee_link_path = self._find_link_path(robot_path, self.end_effector_link)
            if ee_link_path is None:
                raise RuntimeError(f"Cannot find link '{self.end_effector_link}' under {robot_path}")
            self.ee_link_paths.append(ee_link_path)
            tip_path = self._create_long_stick_with_tip(ee_link_path)
            self.stick_tip_paths.append(tip_path)

        self.world.reset()
        self.robot_view = ArticulationView("/World/Env_*/UR5e", name="ur5e_view")
        self.world.scene.add(self.robot_view)
        self.world.reset()
        self.robot_view.initialize()
        self.num_dof = int(self.robot_view.num_dof)

        all_ids = np.arange(self.num_envs, dtype=np.int32)
        jp = np.zeros((self.num_envs, self.num_dof), dtype=np.float32)
        jp[:, : self.num_robot_dofs] = self.default_joint_positions
        self.robot_view.set_joint_positions(jp, indices=all_ids)
        self.robot_view.set_joint_velocities(np.zeros_like(jp), indices=all_ids)
        self.robot_view.set_joint_position_targets(jp, indices=all_ids)

        for _ in range(max(0, self.init_warmup_steps)):
            self.world.step(render=not self.headless)

        self._init_wire_visuals()
        self.target_world = self.env_origins + self.target_local.reshape(1, 3)

    def _init_wire_visuals(self):
        if self.wire_visual_mode == "skeleton":
            self._init_wire_visual_drivers()
        elif self.wire_visual_mode == "debug_spheres":
            self._init_wire_debug_markers()

    def _init_wire_visual_drivers(self):
        if self.wire_visual_mode != "skeleton" or self.skeleton_driver_cls is None:
            return

        for i in range(min(self.num_envs, self.max_visual_envs)):
            if i == 0:
                skel_root = "/World/PyElasticaWire"
            else:
                skel_root = f"/World/PyElasticaWire_{i}"
            try:
                driver = self.skeleton_driver_cls(self.stage, skeleton_path=skel_root)
                driver.load_asset(self.wire_usd)
                self.wire_drivers[i] = driver
            except Exception as exc:
                self.wire_drivers[i] = None
                print(f"[WireSwingEnv] Failed to initialize wire driver for env {i}: {exc}")

    def _init_wire_debug_markers(self):
        if self.wire_visual_mode != "debug_spheres":
            return
        n_markers = int(max(1, min(self.max_visual_nodes, self.wire_n_elem + 1)))
        color = Gf.Vec3f(
            float(self.wire_debug_sphere_color[0]),
            float(self.wire_debug_sphere_color[1]),
            float(self.wire_debug_sphere_color[2]),
        )
        for env_idx in range(min(self.num_envs, self.max_visual_envs)):
            root = f"/World/Env_{env_idx}/WireDebug"
            UsdGeom.Xform.Define(self.stage, root)
            marker_ops = []
            for marker_idx in range(n_markers):
                marker_path = f"{root}/Node_{marker_idx:03d}"
                sphere = UsdGeom.Sphere.Define(self.stage, marker_path)
                sphere.GetRadiusAttr().Set(float(self.wire_debug_sphere_radius))
                sphere.GetDisplayColorAttr().Set([color])
                xf = UsdGeom.Xformable(sphere.GetPrim())
                xf.ClearXformOpOrder()
                marker_ops.append(xf.AddTranslateOp())
            self.wire_debug_marker_ops[env_idx] = marker_ops

    def _update_wire_driver(self, env_idx: int, rod_pos: np.ndarray, rod_dir: np.ndarray, *, strict: bool):
        driver = self.wire_drivers[env_idx]
        if driver is None:
            return
        try:
            driver.update_skeleton(rod_pos, rod_dir, time_code=None)
        except Exception as exc:
            if strict:
                raise RuntimeError(f"Wire driver update failed for env {env_idx}: {exc}") from exc
            if not bool(self._wire_update_warned[env_idx]):
                self._wire_update_warned[env_idx] = True
                print(f"[WireSwingEnv] wire update skipped for env {env_idx}: {exc}")

    def _update_wire_debug_markers(self, env_idx: int, rod_pos: np.ndarray):
        marker_ops = self.wire_debug_marker_ops[env_idx]
        if len(marker_ops) == 0:
            return
        n_nodes = int(rod_pos.shape[1])
        if n_nodes <= 0:
            return
        node_ids = np.linspace(0, n_nodes - 1, num=len(marker_ops), dtype=np.int64)
        for marker_idx, node_idx in enumerate(node_ids.tolist()):
            p = np.asarray(rod_pos[:, node_idx], dtype=np.float64).reshape(3)
            if not np.all(np.isfinite(p)):
                continue
            marker_ops[marker_idx].Set(Gf.Vec3d(float(p[0]), float(p[1]), float(p[2])))

    def _update_wire_visual(self, env_idx: int, rod_pos: np.ndarray, rod_dir: np.ndarray, *, strict: bool):
        if self.wire_visual_mode == "skeleton":
            self._update_wire_driver(env_idx, rod_pos, rod_dir, strict=strict)
        elif self.wire_visual_mode == "debug_spheres":
            self._update_wire_debug_markers(env_idx, rod_pos)

    def _set_ur5e_default_joint_targets(self, robot_path: str):
        joint_names = [
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint",
        ]
        robot_prim = self.stage.GetPrimAtPath(robot_path)
        for i, jname in enumerate(joint_names):
            for prim in Usd.PrimRange(robot_prim):
                if prim.GetName() != jname:
                    continue
                drive = UsdPhysics.DriveAPI.Get(prim, "angular")
                if drive:
                    drive.GetTargetPositionAttr().Set(float(np.degrees(self.default_joint_positions[i])))
                break

    def _find_link_path(self, robot_path: str, link_name: str) -> str | None:
        for prim in Usd.PrimRange(self.stage.GetPrimAtPath(robot_path)):
            if prim.GetName() == link_name:
                return prim.GetPath().pathString
        return None

    def _create_long_stick_with_tip(self, ee_link_path: str) -> str:
        stick_path = f"{ee_link_path}/LongStickVisual"
        cap = UsdGeom.Capsule.Define(self.stage, stick_path)
        cap.CreateHeightAttr(float(self.stick_length))
        cap.CreateRadiusAttr(float(self.stick_radius))
        cap.CreateAxisAttr("Z")
        cap.CreateDisplayColorAttr().Set([Gf.Vec3f(0.55, 0.55, 0.6)])
        stick_xf = UsdGeom.Xformable(cap.GetPrim())
        stick_xf.ClearXformOpOrder()
        stick_xf.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, float(self.stick_length * 0.5)))

        tip_path = f"{ee_link_path}/LongStickTip"
        tip = UsdGeom.Xform.Define(self.stage, tip_path)
        tip_xf = UsdGeom.Xformable(tip.GetPrim())
        tip_xf.ClearXformOpOrder()
        tip_xf.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, float(self.stick_length)))
        return tip_path

    def _create_target(self, env_root: str, local_pos: np.ndarray):
        sphere = UsdGeom.Sphere.Define(self.stage, f"{env_root}/Target")
        sphere.GetRadiusAttr().Set(0.03)
        sphere.GetDisplayColorAttr().Set([Gf.Vec3f(0.0, 1.0, 0.0)])
        xf = UsdGeom.Xformable(sphere.GetPrim())
        xf.ClearXformOpOrder()
        xf.AddTranslateOp().Set(
            Gf.Vec3d(float(local_pos[0]), float(local_pos[1]), float(local_pos[2]))
        )

    def _get_prim_world_pose(self, prim_path: str) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
        prim = self.stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            return None, None
        world_tf = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        t = world_tf.ExtractTranslation()
        R = np.array(world_tf.ExtractRotationMatrix(), dtype=np.float64)
        p = np.array([float(t[0]), float(t[1]), float(t[2])], dtype=np.float64)
        return p, _orthonormalize(R)

    def _build_wire_engine(self, frame_initial_state: FrameState) -> CoSimEngine:
        cfg_kwargs = dict(
            base_length=float(self.wire_base_length),
            n_elem=int(self.wire_n_elem),
            base_radius=float(self.wire_base_radius),
            py_dt=float(self.py_dt),
            isaac_dt=float(self.phys_dt),
            final_time=1.0e9,
            render=False,
            joint_k=float(self.joint_k),
            joint_nu=float(self.joint_nu),
            joint_kt=float(self.joint_kt),
            joint_nut=float(self.joint_nut),
            joint_model=str(self.joint_model),
            use_ground_contact=bool(self.use_ground_contact),
            ground_z=float(self.ground_z),
            ground_contact_k=float(self.ground_contact_k),
            ground_contact_nu=float(self.ground_contact_nu),
            ground_static_mu=np.asarray(self.ground_static_mu, dtype=np.float64),
            ground_kinetic_mu=np.asarray(self.ground_kinetic_mu, dtype=np.float64),
            ground_slip_velocity_tol=float(self.ground_slip_velocity_tol),
            settle_time=float(self.settle_time),
            axial_stretch_stiffening=float(self.axial_stretch_stiffening),
            density=float(self.wire_density),
            youngs_modulus=float(self.wire_youngs_modulus),
            shear_modulus_ratio=float(self.wire_shear_modulus_ratio),
            damping_constant=float(self.wire_damping_constant),
            rod_direction=np.asarray(frame_initial_state.director[2], dtype=np.float64),
            rod_normal=np.asarray(frame_initial_state.director[0], dtype=np.float64),
            frame_initial_position=np.asarray(frame_initial_state.position, dtype=np.float64),
            frame_initial_director=np.asarray(frame_initial_state.director, dtype=np.float64),
            frame_initial_velocity=np.asarray(frame_initial_state.velocity, dtype=np.float64),
            frame_initial_acceleration=np.asarray(frame_initial_state.acceleration, dtype=np.float64),
            frame_initial_omega=np.asarray(frame_initial_state.omega, dtype=np.float64),
            frame_initial_alpha=np.asarray(frame_initial_state.alpha, dtype=np.float64),
            rod_start=np.asarray(frame_initial_state.position, dtype=np.float64),
        )
        if self.initial_wire_theta is not None:
            cfg_kwargs["initial_wire_theta"] = float(self.initial_wire_theta)
        accepted_keys = set(inspect.signature(CoSimConfig).parameters.keys())
        dropped_keys = [k for k in cfg_kwargs.keys() if k not in accepted_keys]
        if "joint_model" in dropped_keys and self.joint_model != "fixed" and not self._warned_unsupported_joint_model:
            print(
                f"[WireSwingEnv] CoSimConfig has no 'joint_model'; requested '{self.joint_model}' is ignored."
            )
            self._warned_unsupported_joint_model = True
        cfg = CoSimConfig(**{k: v for k, v in cfg_kwargs.items() if k in accepted_keys})
        return CoSimEngine(config=cfg, frame_initial_state=frame_initial_state)

    def _safe_wire_snapshot(self, env_idx: int, snap) -> tuple[np.ndarray, np.ndarray]:
        rod_pos = np.asarray(snap.rod_position, dtype=np.float64)
        rod_dir = np.asarray(snap.rod_director, dtype=np.float64)
        finite = np.all(np.isfinite(rod_pos)) and np.all(np.isfinite(rod_dir))
        if finite:
            self.last_good_rod_pos[env_idx] = rod_pos.copy()
            self.last_good_rod_dir[env_idx] = rod_dir.copy()
            return rod_pos, rod_dir

        fallback_pos = self.last_good_rod_pos[env_idx]
        fallback_dir = self.last_good_rod_dir[env_idx]
        if fallback_pos is not None and fallback_dir is not None:
            return fallback_pos, fallback_dir

        rod_pos = np.nan_to_num(rod_pos, nan=0.0, posinf=0.0, neginf=0.0)
        rod_dir = np.nan_to_num(rod_dir, nan=0.0, posinf=0.0, neginf=0.0)
        return rod_pos, rod_dir

    def _make_frame_state_from_pose(
        self,
        position: np.ndarray,
        director: np.ndarray,
        prev_kin: dict[str, np.ndarray],
        dt: float,
    ) -> tuple[FrameState, dict[str, np.ndarray]]:
        pos = np.asarray(position, dtype=np.float64)
        R = _orthonormalize(np.asarray(director, dtype=np.float64))
        vel = (pos - prev_kin["position"]) / dt
        acc = (vel - prev_kin["velocity"]) / dt
        R_delta = R @ prev_kin["director"].T
        omega = _rotation_matrix_to_rotvec(R_delta) / dt
        alpha = (omega - prev_kin["omega"]) / dt

        state = FrameState(
            position=pos,
            director=R,
            velocity=vel,
            acceleration=acc,
            omega=omega,
            alpha=alpha,
        )
        new_kin = {
            "position": pos,
            "director": R,
            "velocity": vel,
            "omega": omega,
        }
        return state, new_kin

    def _rebuild_cosim(self, ids: np.ndarray):
        for idx in ids.tolist():
            tip_pos, tip_dir = self._get_prim_world_pose(self.stick_tip_paths[idx])
            if tip_pos is None:
                raise RuntimeError(f"Cannot get stick-tip pose for env {idx}: {self.stick_tip_paths[idx]}")

            frame_init = FrameState(
                position=tip_pos,
                director=tip_dir,
                velocity=np.zeros(3, dtype=np.float64),
                acceleration=np.zeros(3, dtype=np.float64),
                omega=np.zeros(3, dtype=np.float64),
                alpha=np.zeros(3, dtype=np.float64),
            )
            engine = self._build_wire_engine(frame_init)
            self.engines[idx] = engine
            self.kin_state[idx] = {
                "position": np.asarray(frame_init.position, dtype=np.float64),
                "director": np.asarray(frame_init.director, dtype=np.float64),
                "velocity": np.asarray(frame_init.velocity, dtype=np.float64),
                "omega": np.asarray(frame_init.omega, dtype=np.float64),
            }

            snap = engine.snapshot()
            rod_pos, rod_dir = self._safe_wire_snapshot(idx, snap)
            tip = np.asarray(rod_pos[:, -1], dtype=np.float64)
            if not np.all(np.isfinite(tip)):
                tip = tip_pos.copy()
            self.tip_world[idx] = tip
            self.prev_tip_world[idx] = tip
            self.tip_vel_world[idx] = 0.0
            self.ee_tip_world[idx] = tip_pos

            self._update_wire_visual(idx, rod_pos, rod_dir, strict=True)

    def _step_cosim(self, duration: float):
        for i in range(self.num_envs):
            tip_pos, tip_dir = self._get_prim_world_pose(self.stick_tip_paths[i])
            if tip_pos is None:
                continue

            frame_cmd, kin = self._make_frame_state_from_pose(
                tip_pos,
                tip_dir,
                self.kin_state[i],
                duration,
            )
            self.kin_state[i] = kin
            self.ee_tip_world[i] = tip_pos

            engine = self.engines[i]
            if engine is None:
                continue
            engine.update_frame_state(frame_cmd, duration=duration)
            snap = engine.snapshot()
            rod_pos, rod_dir = self._safe_wire_snapshot(i, snap)
            tip = np.asarray(rod_pos[:, -1], dtype=np.float64)
            if not np.all(np.isfinite(tip)):
                tip = self.prev_tip_world[i].copy()
            self.tip_world[i] = tip

            self._update_wire_visual(i, rod_pos, rod_dir, strict=False)

    def reset(self, ids=None) -> np.ndarray:
        if ids is None:
            ids = np.arange(self.num_envs, dtype=np.int32)
        ids = np.asarray(ids, dtype=np.int32)
        if ids.size == 0:
            return self.obs()

        self.episode_steps[ids] = 0
        self.best_approach_step[ids] = 0
        self.prev_actions[ids] = 0.0
        self.prev_dist[ids] = 1.0e9
        self.min_dist[ids] = 1.0e9
        self.tip_vel_world[ids] = 0.0

        n = ids.size
        jp = np.zeros((n, self.num_dof), dtype=np.float32)
        jp[:, : self.num_robot_dofs] = self.default_joint_positions
        self.robot_view.set_joint_positions(jp, indices=ids)
        self.robot_view.set_joint_velocities(np.zeros_like(jp), indices=ids)
        self.robot_view.set_joint_position_targets(jp, indices=ids)
        self.commanded_q[ids] = self.default_joint_positions.reshape(1, -1)

        self.world.step(render=not self.headless)
        self._rebuild_cosim(ids)

        for _ in range(500):
            self.world.step(render=True)

        dist = np.linalg.norm(self.tip_world[ids] - self.target_world[ids], axis=1).astype(np.float32)
        dist = np.nan_to_num(dist, nan=1.0e3, posinf=1.0e3, neginf=1.0e3)
        self.prev_dist[ids] = dist
        self.min_dist[ids] = dist

        return self.obs()

    def obs(self) -> np.ndarray:
        all_jp = self.robot_view.get_joint_positions()[:, : self.num_robot_dofs].astype(np.float32)
        all_jv = self.robot_view.get_joint_velocities()[:, : self.num_robot_dofs].astype(np.float32)

        jp = all_jp[:, self.active_joints]
        jv = all_jv[:, self.active_joints]

        tip_local = (self.tip_world - self.env_origins).astype(np.float32)
        tip_vel = self.tip_vel_world.astype(np.float32)
        ee_local = (self.ee_tip_world - self.env_origins).astype(np.float32)
        target_local = np.repeat(self.target_local.reshape(1, 3), self.num_envs, axis=0).astype(np.float32)

        obs = np.hstack([jp, jv, tip_local, tip_vel, ee_local, target_local]).astype(np.float32)
        obs = np.nan_to_num(obs, nan=0.0, posinf=1.0e3, neginf=-1.0e3)
        return np.clip(obs, -1.0e3, 1.0e3)

    def _reward_done_info(self, actions: np.ndarray):
        # Task is evaluated on the Y-Z plane: ignore X-axis displacement.
        tip_world = np.nan_to_num(self.tip_world, nan=0.0, posinf=1.0e3, neginf=-1.0e3)
        target_world = np.nan_to_num(self.target_world, nan=0.0, posinf=1.0e3, neginf=-1.0e3)
        tip_vel_world = np.nan_to_num(self.tip_vel_world, nan=0.0, posinf=0.0, neginf=0.0)

        delta_yz = (tip_world - target_world)[:, 1:3]
        dist = np.linalg.norm(delta_yz, axis=1).astype(np.float32)
        safe_dist = np.maximum(dist, 1.0e-6)
        to_target_vec = (target_world - tip_world).copy()
        to_target_vec[:, 0] = 0.0
        to_target_dir = to_target_vec / safe_dist.reshape(-1, 1)
        tip_speed_toward_target = np.sum(tip_vel_world * to_target_dir, axis=1).astype(np.float32)

        old_min = self.min_dist.copy()
        r_proximity = self.w_proximity * np.exp(-dist)
        r_progress = self.w_progress * (self.prev_dist - dist)

        improvement = np.clip(old_min - dist, a_min=0.0, a_max=None)
        improved = (dist < old_min).astype(np.float32)
        r_new_best = self.w_new_best * improved * improvement

        r_bonus = (
            (dist < self.bonus_thresh_1).astype(np.float32) * self.bonus_value_1
            + (dist < self.bonus_thresh_2).astype(np.float32) * self.bonus_value_2
            + (dist < self.bonus_thresh_3).astype(np.float32) * self.bonus_value_3
            + (dist < self.bonus_thresh_4).astype(np.float32) * self.bonus_value_4
        )
        r_touch = (dist < self.touch_thresh).astype(np.float32) * self.touch_bonus
        r_tip_velocity = self.w_tip_velocity_toward_target * tip_speed_toward_target
        all_jv = self.robot_view.get_joint_velocities()[:, : self.num_robot_dofs].astype(np.float32)
        all_jv = np.nan_to_num(all_jv, nan=0.0, posinf=0.0, neginf=0.0)
        joint3_vel = all_jv[:, self.joint3_reward_index]
        joint3_signal = np.abs(joint3_vel) if self.joint3_velocity_abs else joint3_vel
        r_joint3_velocity = self.w_joint3_velocity * joint3_signal
        r_time = -self.w_time_penalty * np.ones_like(dist, dtype=np.float32)
        r_action = -self.action_penalty * np.sum(actions**2, axis=1)
        r_smooth = -self.smooth_penalty * np.sum((actions - self.prev_actions) ** 2, axis=1)

        success = dist < self.success_thresh
        r_success = success.astype(np.float32) * self.success_bonus

        rewards = (
            r_proximity
            + r_progress
            + r_new_best
            + r_bonus
            + r_touch
            + r_tip_velocity
            + r_joint3_velocity
            + r_time
            + r_action
            + r_smooth
            + r_success
        )
        rewards = np.nan_to_num(rewards, nan=0.0, posinf=0.0, neginf=0.0)
        rewards = np.clip(rewards, self.reward_clip_min, self.reward_clip_max).astype(np.float32)

        self.prev_dist = np.nan_to_num(dist, nan=self.too_far_thresh + 1.0, posinf=self.too_far_thresh + 1.0, neginf=0.0)
        self.min_dist = np.minimum(self.min_dist, self.prev_dist)
        improved_mask = dist < old_min
        self.best_approach_step[improved_mask] = self.episode_steps[improved_mask]
        self.episode_steps += 1

        timeout = self.episode_steps >= self.max_steps
        too_far = dist > self.too_far_thresh

        if self.swing_done_enabled:
            steps_since_best = self.episode_steps - self.best_approach_step
            swing_done = (self.episode_steps > self.swing_grace_steps) & (
                steps_since_best > self.swing_fallback_steps
            )
        else:
            swing_done = np.zeros((self.num_envs,), dtype=np.bool_)

        done = timeout | too_far | swing_done
        if self.terminate_on_success:
            done = done | success

        info = {
            "dist": dist.copy(),
            "dist_mean": float(np.mean(dist)),
            "min_dist_mean": float(np.mean(self.min_dist)),
            "success_rate": float(np.mean(success.astype(np.float32))),
            "success_mask": success.copy(),
            "timeout_rate": float(np.mean(timeout.astype(np.float32))),
            "swing_done_rate": float(np.mean(swing_done.astype(np.float32))),
            "r_proximity_mean": float(np.mean(r_proximity)),
            "r_progress_mean": float(np.mean(r_progress)),
            "r_new_best_mean": float(np.mean(r_new_best)),
            "r_bonus_mean": float(np.mean(r_bonus)),
            "r_touch_mean": float(np.mean(r_touch)),
            "r_tip_velocity_mean": float(np.mean(r_tip_velocity)),
            "r_joint3_velocity_mean": float(np.mean(r_joint3_velocity)),
            "joint3_vel_mean": float(np.mean(joint3_vel)),
            "r_action_mean": float(np.mean(r_action)),
            "r_smooth_mean": float(np.mean(r_smooth)),
            "r_success_mean": float(np.mean(r_success)),
        }
        return rewards, done.astype(np.bool_), info

    def step(self, actions: np.ndarray):
        actions = np.asarray(actions, dtype=np.float32)
        if actions.shape != (self.num_envs, self.act_dim):
            raise ValueError(f"Expected actions shape {(self.num_envs, self.act_dim)}, got {actions.shape}")
        actions = np.nan_to_num(actions, nan=0.0, posinf=1.0, neginf=-1.0)
        actions = np.clip(actions, -1.0, 1.0)
        if self.positive_only_active_joints:
            actions = np.clip(actions, 0.0, 1.0)

        q = self.robot_view.get_joint_positions()[:, : self.num_robot_dofs].astype(np.float32)
        q = np.nan_to_num(q, nan=0.0, posinf=0.0, neginf=0.0)
        targets = np.tile(self.default_joint_positions.reshape(1, -1), (self.num_envs, 1))

        if self.positive_only_active_joints:
            active_now = self.commanded_q[:, self.active_joints]
        else:
            active_now = q[:, self.active_joints]
        active_delta = actions * self.joint_max_delta[self.active_joints]
        active_next = active_now + active_delta

        lo = self.joint_pos_lower_limits[self.active_joints]
        hi = self.joint_pos_upper_limits[self.active_joints]
        if self.positive_only_active_joints:
            lo = np.maximum(lo, self.default_joint_positions[self.active_joints])
        active_next = np.clip(active_next, lo, hi)
        targets[:, self.active_joints] = active_next

        full = np.zeros((self.num_envs, self.num_dof), dtype=np.float32)
        full[:, : self.num_robot_dofs] = targets
        self.robot_view.set_joint_position_targets(full)
        commanded_q = full[:, : self.num_robot_dofs].copy()
        self.commanded_q = commanded_q.copy()

        for _ in range(self.num_substeps):
            self.world.step(render=not self.headless)
            self._step_cosim(self.phys_dt)

        dt = max(self.control_dt, 1.0e-8)
        self.tip_vel_world = (self.tip_world - self.prev_tip_world) / dt
        self.tip_vel_world = np.nan_to_num(self.tip_vel_world, nan=0.0, posinf=0.0, neginf=0.0)
        self.prev_tip_world = self.tip_world.copy()

        rewards, done, info = self._reward_done_info(actions)
        info["commanded_joint_positions"] = commanded_q
        info["control_dt"] = float(self.control_dt)
        info["joint_names"] = list(self.joint_names)
        self.prev_actions = actions.copy()

        reset_ids = np.where(done)[0].astype(np.int32)
        if reset_ids.size > 0:
            self.reset(reset_ids)

        return self.obs(), rewards, done, info
