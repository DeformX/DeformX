from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

import RL.envs.wire_swing_env as _base
from RL.envs.wire_swing_env import WireSwingEnv


@dataclass
class _BjWireState:
    root_path: str
    link_paths: list[str]
    link_view: object | None
    first_link_path: str
    last_link_path: str


def _safe_normalize(v: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    arr = np.asarray(v, dtype=np.float64).reshape(3)
    n = float(np.linalg.norm(arr))
    if not np.isfinite(n) or n < 1.0e-12:
        return np.asarray(fallback, dtype=np.float64).reshape(3).copy()
    return arr / n


def _quat_wxyz_from_z_axis(target_dir: np.ndarray) -> np.ndarray:
    """Quaternion (wxyz) that rotates +Z to target_dir."""
    z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    t = _safe_normalize(target_dir, np.array([1.0, 0.0, 0.0], dtype=np.float64))
    dot = float(np.clip(np.dot(z_axis, t), -1.0, 1.0))

    if dot > 1.0 - 1.0e-9:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    if dot < -1.0 + 1.0e-9:
        # 180deg rotation about +Y maps +Z to -Z.
        return np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float64)

    axis = np.cross(z_axis, t)
    quat = np.array([1.0 + dot, axis[0], axis[1], axis[2]], dtype=np.float64)
    qn = float(np.linalg.norm(quat))
    if qn < 1.0e-12 or not np.isfinite(qn):
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    quat /= qn
    return quat


def _quat_mul_wxyz(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product in wxyz order."""
    aw, ax, ay, az = [float(v) for v in np.asarray(a, dtype=np.float64).reshape(4)]
    bw, bx, by, bz = [float(v) for v in np.asarray(b, dtype=np.float64).reshape(4)]
    out = np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dtype=np.float64,
    )
    n = float(np.linalg.norm(out))
    if not np.isfinite(n) or n < 1.0e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return out / n


class WireSwingBallJointEnv(WireSwingEnv):
    """
    Ball-joint baseline variant of WireSwingEnv:
    - Reuses robot/task/reward/logging interfaces from WireSwingEnv.
    - Replaces the PyElastica backend with a PhysX chain of capsule links connected
      by spherical joints.

    Convention in this version (IMPORTANT):
    - The stick tip pins link_000 LOCAL -Z endpoint to the tip position.
      => center = tip_pos + first_dir * half
    - Spherical joints connect link_i LOCAL +Z to link_{i+1} LOCAL -Z:
      localPos0 = +half  (on link_i)
      localPos1 = -half  (on link_{i+1})
    - Therefore the chain extends outward along +first_dir (stick local +Z).
    """

    def __init__(self, cfg, *, headless: bool = True):
        self.sync_reset_ratio = float(getattr(cfg, "sync_reset_ratio", 1.0))
        self.pending_reset = np.zeros((int(getattr(cfg, "num_envs", 1)),), dtype=np.bool_)

        self._bj_joint_damping = float(getattr(cfg, "bj_joint_damping", 5.0))
        self._bj_joint_stiffness = float(getattr(cfg, "bj_joint_stiffness", 0.0))
        self._bj_joint_drive_type = str(getattr(cfg, "bj_joint_drive_type", "force")).strip().lower()
        if self._bj_joint_drive_type not in ("force", "acceleration"):
            print(
                f"[WireSwingBallJointEnv] unsupported bj_joint_drive_type='{self._bj_joint_drive_type}', "
                "fallback to 'force'."
            )
            self._bj_joint_drive_type = "force"
        self._bj_link_linear_damping = float(getattr(cfg, "bj_link_linear_damping", 0.6))
        self._bj_link_angular_damping = float(getattr(cfg, "bj_link_angular_damping", 1.2))
        self._bj_mat_static_friction = float(getattr(cfg, "bj_mat_static_friction", 0.6))
        self._bj_mat_dynamic_friction = float(getattr(cfg, "bj_mat_dynamic_friction", 0.5))
        self._bj_mat_restitution = float(getattr(cfg, "bj_mat_restitution", 0.0))
        self._bj_collision_rest_offset = float(getattr(cfg, "bj_collision_rest_offset", 0.0))
        self._bj_collision_contact_offset = float(getattr(cfg, "bj_collision_contact_offset", 0.002))
        self._bj_link_density = float(getattr(cfg, "bj_link_density", getattr(cfg, "wire_density", 700.0)))

        self._bj_wire_states: list[_BjWireState] = []
        self._bj_ready = False
        self._bj_link_half = 0.0
        self._bj_link_len = 0.0
        self._bj_num_links = 0

        super().__init__(cfg, headless=headless)
        if self.pending_reset.shape[0] != self.num_envs:
            self.pending_reset = np.zeros((self.num_envs,), dtype=np.bool_)
        self._enable_gravity()

    def _enable_gravity(self):
        """Enable PhysX gravity; compatible with Isaac Sim versions where set_gravity expects a scalar."""
        ctx = self.world.get_physics_context()

        from pxr import UsdGeom
        mpu = float(UsdGeom.GetStageMetersPerUnit(self.stage) or 1.0)
        print(f"[Gravity] stage metersPerUnit = {mpu}")

        # Scale gravity for stage units
        g = -9.81 / max(mpu, 1.0e-9)

        # Different Isaac Sim versions expose different signatures:
        # - some accept a scalar (gravity magnitude along -Z)
        # - some accept a 3D vector
        # We'll try both.
        try:
            # Try vector form first (if supported)
            ctx.set_gravity(np.array([0.0, 0.0, g], dtype=np.float32))
        except Exception as e_vec:
            try:
                # Fallback: scalar form (most common in isaacsim.core.api)
                ctx.set_gravity(float(g))
            except Exception as e_s:
                raise RuntimeError(f"Failed to set gravity via physics_context. vec_err={e_vec}, scalar_err={e_s}")

        # Verify
        try:
            print(f"[Gravity] get_gravity() -> {ctx.get_gravity()}")
        except Exception:
            print("[Gravity] gravity set (get_gravity() not available).")    # -------------------------
    # Visuals
    # -------------------------
    def _init_wire_visuals(self):
        if self.wire_visual_mode in ("skeleton", "debug_spheres"):
            print(
                "[WireSwingBallJointEnv] disabling BJ wire debug spheres "
                f"(requested wire_visual_mode={self.wire_visual_mode})."
            )
            self.wire_visual_mode = "none"
        super()._init_wire_visuals()

    def _init_wire_debug_markers(self):
        if self.wire_visual_mode != "debug_spheres":
            return

        n_links = int(max(1, self.wire_n_elem))
        n_joint_markers = int(max(1, min(self.max_visual_nodes, max(1, n_links - 1))))
        color = _base.Gf.Vec3f(
            float(self.wire_debug_sphere_color[0]),
            float(self.wire_debug_sphere_color[1]),
            float(self.wire_debug_sphere_color[2]),
        )
        for env_idx in range(min(self.num_envs, self.max_visual_envs)):
            root = f"/World/Env_{env_idx}/WireDebug"
            _base.UsdGeom.Xform.Define(self.stage, root)
            marker_ops = []
            for marker_idx in range(n_joint_markers):
                marker_path = f"{root}/Joint_{marker_idx:03d}"
                sphere = _base.UsdGeom.Sphere.Define(self.stage, marker_path)
                sphere.GetRadiusAttr().Set(float(self.wire_debug_sphere_radius))
                sphere.GetDisplayColorAttr().Set([color])
                xf = _base.UsdGeom.Xformable(sphere.GetPrim())
                xf.ClearXformOpOrder()
                marker_ops.append(xf.AddTranslateOp())
            self.wire_debug_marker_ops[env_idx] = marker_ops

    def _update_wire_debug_markers(self, env_idx: int, rod_pos: np.ndarray):
        marker_ops = self.wire_debug_marker_ops[env_idx]
        if len(marker_ops) == 0:
            return
        n_nodes = int(rod_pos.shape[1])
        if n_nodes <= 2:
            return
        # internal joint markers only
        joint_nodes = np.asarray(rod_pos[:, 1:-1], dtype=np.float64).T
        if joint_nodes.shape[0] <= 0:
            return
        joint_ids = np.linspace(0, joint_nodes.shape[0] - 1, num=len(marker_ops), dtype=np.int64)
        for marker_idx, joint_idx in enumerate(joint_ids.tolist()):
            p = joint_nodes[joint_idx]
            if not np.all(np.isfinite(p)):
                continue
            marker_ops[marker_idx].Set(_base.Gf.Vec3d(float(p[0]), float(p[1]), float(p[2])))

    # -------------------------
    # USD helpers
    # -------------------------
    def _set_orient_op_wxyz(self, prim_path: str, quat_wxyz: np.ndarray):
        prim = self.stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            raise RuntimeError(f"Invalid prim for orient op: {prim_path}")

        xf = _base.UsdGeom.Xformable(prim)
        translate_op = None
        orient_op = None
        for op in xf.GetOrderedXformOps():
            if op.GetOpType() == _base.UsdGeom.XformOp.TypeTranslate and translate_op is None:
                translate_op = op
            if op.GetOpType() == _base.UsdGeom.XformOp.TypeOrient and orient_op is None:
                orient_op = op

        if orient_op is None:
            orient_op = xf.AddOrientOp()
        if translate_op is not None:
            xf.SetXformOpOrder([translate_op, orient_op], resetXformStack=False)

        w, x, y, z = [float(v) for v in np.asarray(quat_wxyz, dtype=np.float64).reshape(4)]
        if orient_op.GetPrecision() == _base.UsdGeom.XformOp.PrecisionDouble:
            orient_op.Set(_base.Gf.Quatd(w, _base.Gf.Vec3d(x, y, z)))
        else:
            orient_op.Set(_base.Gf.Quatf(w, _base.Gf.Vec3f(x, y, z)))

    def _set_translate_op(self, prim_path: str, pos_xyz: np.ndarray):
        prim = self.stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            raise RuntimeError(f"Invalid prim for translate op: {prim_path}")

        xf = _base.UsdGeom.Xformable(prim)
        translate_op = None
        for op in xf.GetOrderedXformOps():
            if op.GetOpType() == _base.UsdGeom.XformOp.TypeTranslate:
                translate_op = op
                break
        if translate_op is None:
            translate_op = xf.AddTranslateOp()

        x, y, z = [float(v) for v in np.asarray(pos_xyz, dtype=np.float64).reshape(3)]
        if translate_op.GetPrecision() == _base.UsdGeom.XformOp.PrecisionDouble:
            translate_op.Set(_base.Gf.Vec3d(x, y, z))
        else:
            translate_op.Set(_base.Gf.Vec3f(x, y, z))

    def _set_kinematic(self, prim_path: str):
        prim = self.stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            raise RuntimeError(f"Invalid prim for kinematic setup: {prim_path}")
        rb = _base.UsdPhysics.RigidBodyAPI(prim)
        if not rb:
            rb = _base.UsdPhysics.RigidBodyAPI.Apply(prim)
        rb.CreateKinematicEnabledAttr().Set(True)

    # -------------------------
    # Chain dir helpers
    # -------------------------
    def _resolve_chain_direction(self, tip_rot: np.ndarray) -> np.ndarray:
        # optional override by angle
        if self.initial_wire_theta is not None:
            theta = float(self.initial_wire_theta)
            chain_dir = np.array([math.cos(theta), math.sin(theta), 0.0], dtype=np.float64)
            return _safe_normalize(chain_dir, np.array([1.0, 0.0, 0.0], dtype=np.float64))

        # default: use stick local +Z in world
        chain_dir = np.asarray(tip_rot, dtype=np.float64) @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
        return _safe_normalize(chain_dir, np.array([1.0, 0.0, 0.0], dtype=np.float64))

    # -------------------------
    # Build BJ assets
    # -------------------------
    def _create_bj_wire_for_env(self, env_idx: int) -> _BjWireState:
        from pxr import UsdShade
        from omni.physx.scripts import physicsUtils
        from omni.isaac.core.prims import RigidPrimView

        root_path = f"/World/Env_{env_idx}/BallJointWire"
        _base.UsdGeom.Scope.Define(self.stage, root_path)

        mat_path = f"{root_path}/WireMaterial"
        UsdShade.Material.Define(self.stage, mat_path)
        mat = _base.UsdPhysics.MaterialAPI.Apply(self.stage.GetPrimAtPath(mat_path))
        mat.CreateStaticFrictionAttr().Set(float(self._bj_mat_static_friction))
        mat.CreateDynamicFrictionAttr().Set(float(self._bj_mat_dynamic_friction))
        mat.CreateRestitutionAttr().Set(float(self._bj_mat_restitution))

        origin = self.env_origins[env_idx]
        x0 = float(origin[0])
        y0 = float(origin[1])
        z0 = float(self.wire_base_radius)

        link_paths: list[str] = []
        for i in range(self._bj_num_links):
            path = f"{root_path}/link_{i:03d}"
            cap = _base.UsdGeom.Capsule.Define(self.stage, path)
            cap.CreateHeightAttr(float(self._bj_link_len))
            cap.CreateRadiusAttr(float(self.wire_base_radius))
            cap.CreateAxisAttr("Z")
            cap.CreateDisplayColorAttr().Set([_base.Gf.Vec3f(0.92, 0.48, 0.12)])
            _base.UsdGeom.Xformable(cap.GetPrim()).AddTranslateOp().Set(
                _base.Gf.Vec3d(x0 - (i + 0.5) * self._bj_link_len, y0, z0)
            )

            prim = cap.GetPrim()
            _base.UsdPhysics.RigidBodyAPI.Apply(prim)
            _base.UsdPhysics.CollisionAPI.Apply(prim)
            _base.UsdPhysics.MassAPI.Apply(prim).CreateDensityAttr().Set(float(self._bj_link_density))

            rb = _base.PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
            rb.CreateLinearDampingAttr().Set(float(self._bj_link_linear_damping))
            rb.CreateAngularDampingAttr().Set(float(self._bj_link_angular_damping))
            col = _base.PhysxSchema.PhysxCollisionAPI.Apply(prim)
            col.CreateRestOffsetAttr().Set(float(self._bj_collision_rest_offset))
            col.CreateContactOffsetAttr().Set(float(self._bj_collision_contact_offset))

            physicsUtils.add_physics_material_to_prim(self.stage, prim, mat_path)
            link_paths.append(path)

        # joints: link_i +Z  <->  link_{i+1} -Z
        for i in range(self._bj_num_links - 1):
            joint_path = f"{root_path}/joint_{i:03d}"
            joint = _base.UsdPhysics.SphericalJoint.Define(self.stage, joint_path)
            joint.GetBody0Rel().SetTargets([link_paths[i]])
            joint.GetBody1Rel().SetTargets([link_paths[i + 1]])
            joint.CreateLocalPos0Attr().Set(_base.Gf.Vec3f(0.0, 0.0, float(self._bj_link_half)))    # +Z on link_i
            joint.CreateLocalPos1Attr().Set(_base.Gf.Vec3f(0.0, 0.0, float(-self._bj_link_half)))   # -Z on link_{i+1}
            joint.CreateLocalRot0Attr().Set(_base.Gf.Quatf(1.0))
            joint.CreateLocalRot1Attr().Set(_base.Gf.Quatf(1.0))

            drive = _base.UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "angular")
            drive.CreateTypeAttr(str(self._bj_joint_drive_type))
            drive.CreateDampingAttr(float(self._bj_joint_damping))
            drive.CreateStiffnessAttr(float(self._bj_joint_stiffness))

        first_link_path = link_paths[0]
        self._set_kinematic(first_link_path)

        view = None
        try:
            view = RigidPrimView(f"{root_path}/link_*", name=f"bj_link_view_{env_idx}")
            view.initialize()
        except Exception as exc:
            view = None
            print(f"[WireSwingBallJointEnv] Failed to initialize BJ RigidPrimView env={env_idx}: {exc}")

        return _BjWireState(
            root_path=root_path,
            link_paths=link_paths,
            link_view=view,
            first_link_path=first_link_path,
            last_link_path=link_paths[-1],
        )

    def _ensure_bj_wire_assets(self):
        if self._bj_ready:
            return

        self._bj_num_links = int(max(1, self.wire_n_elem))
        self._bj_link_half = float(self.wire_base_length) / (2.0 * float(self._bj_num_links))
        self._bj_link_len = 2.0 * self._bj_link_half

        self._bj_wire_states = []
        for env_idx in range(self.num_envs):
            self._bj_wire_states.append(self._create_bj_wire_for_env(env_idx))

        self._bj_ready = True

    # -------------------------
    # Reset pose + anchor
    # -------------------------
    def _set_link_velocities_zero(self, env_idx: int):
        wire = self._bj_wire_states[env_idx]
        if wire.link_view is not None:
            try:
                if self._bj_num_links > 1:
                    dyn_ids = np.arange(1, self._bj_num_links, dtype=np.int64)
                    v = np.zeros((dyn_ids.size, 6), dtype=np.float32)
                    wire.link_view.set_velocities(v, indices=dyn_ids)
                return
            except Exception:
                pass

        for i, pth in enumerate(wire.link_paths):
            if i == 0:
                continue  # kinematic anchor
            prim = self.stage.GetPrimAtPath(pth)
            if not prim.IsValid():
                continue
            rb = _base.UsdPhysics.RigidBodyAPI(prim)
            if not rb:
                rb = _base.UsdPhysics.RigidBodyAPI.Apply(prim)
            rb.GetVelocityAttr().Set(_base.Gf.Vec3f(0.0, 0.0, 0.0))
            rb.GetAngularVelocityAttr().Set(_base.Gf.Vec3f(0.0, 0.0, 0.0))

    def _set_wire_chain_pose(self, env_idx: int, first_node: np.ndarray, chain_dir: np.ndarray):
        """
        Initialize chain such that it extends outward along chain_dir,
        while being consistent with the pin -Z convention at the stick tip.
        """
        wire = self._bj_wire_states[env_idx]
        chain_dir = _safe_normalize(chain_dir, np.array([1.0, 0.0, 0.0], dtype=np.float64))

        # Align capsule local +Z with chain_dir
        q_wxyz = _quat_wxyz_from_z_axis(chain_dir).astype(np.float64)

        centers = np.zeros((self._bj_num_links, 3), dtype=np.float64)
        for i in range(self._bj_num_links):
            centers[i] = np.asarray(first_node, dtype=np.float64) + chain_dir * ((i + 0.5) * self._bj_link_len)

        if wire.link_view is not None:
            try:
                quats = np.repeat(q_wxyz.reshape(1, 4), repeats=self._bj_num_links, axis=0).astype(np.float32)
                wire.link_view.set_world_poses(positions=centers.astype(np.float32), orientations=quats)
            except Exception:
                pass

        for i, link_path in enumerate(wire.link_paths):
            self._set_orient_op_wxyz(link_path, q_wxyz)
            self._set_translate_op(link_path, centers[i])

    def _anchor_first_link_to_tip(self, env_idx: int):
        """
        Pin link_000 LOCAL -Z endpoint to stick tip position.
        Also ensure link_000 LOCAL +Z aligns with stick tip local +Z direction.
        """
        tip_pos, tip_rot = self._get_prim_world_pose(self.stick_tip_paths[env_idx])
        if tip_pos is None:
            return

        wire = self._bj_wire_states[env_idx]

        # stick local +Z in world
        first_dir = _safe_normalize(
            np.asarray(tip_rot, dtype=np.float64) @ np.array([0.0, 0.0, 1.0], dtype=np.float64),
            np.array([0.0, 0.0, 1.0], dtype=np.float64),
        )

        # Make link_000 local +Z align with first_dir (robust)
        q_wxyz = _quat_wxyz_from_z_axis(first_dir)

        # Optional: if you still need extra twist around Z for visuals only, do it here:
        # q_wxyz = _quat_mul_wxyz(q_wxyz, np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64))

        # Pin -Z endpoint at tip: center - first_dir*half == tip_pos
        first_center = np.asarray(tip_pos, dtype=np.float64) + first_dir * self._bj_link_half

        if wire.link_view is not None:
            try:
                wire.link_view.set_world_poses(
                    positions=first_center.reshape(1, 3).astype(np.float32),
                    orientations=q_wxyz.reshape(1, 4).astype(np.float32),
                    indices=np.asarray([0], dtype=np.int32),
                )
            except Exception:
                pass

        self._set_orient_op_wxyz(wire.first_link_path, q_wxyz)
        self._set_translate_op(wire.first_link_path, first_center)

        self.ee_tip_world[env_idx] = np.asarray(tip_pos, dtype=np.float64).copy()

    # -------------------------
    # Snapshot
    # -------------------------
    def _get_wire_nodes_world(self, env_idx: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Return nodes consistent with pin -Z convention:
        nodes[0] = link_000 -Z end (pinned at tip)
        then follow connected ends outward via +Z ends.
        """
        wire = self._bj_wire_states[env_idx]
        plus_end = np.zeros((self._bj_num_links, 3), dtype=np.float64)
        minus_end = np.zeros((self._bj_num_links, 3), dtype=np.float64)
        dirs = np.zeros((self._bj_num_links, 3), dtype=np.float64)
        z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)

        for i, pth in enumerate(wire.link_paths):
            prim = self.stage.GetPrimAtPath(pth)
            if not prim.IsValid():
                continue
            tf = _base.UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(_base.Usd.TimeCode.Default())
            t = tf.ExtractTranslation()
            R = np.array(tf.ExtractRotationMatrix(), dtype=np.float64)
            c = np.array([float(t[0]), float(t[1]), float(t[2])], dtype=np.float64)
            z_dir = _safe_normalize(R @ z_axis, np.array([1.0, 0.0, 0.0], dtype=np.float64))
            dirs[i] = z_dir
            plus_end[i] = c + z_dir * self._bj_link_half
            minus_end[i] = c - z_dir * self._bj_link_half

        nodes = np.zeros((self._bj_num_links + 1, 3), dtype=np.float64)
        nodes[0] = minus_end[0]  # pinned end (tip)
        for i in range(1, self._bj_num_links):
            nodes[i] = plus_end[i - 1]
        nodes[self._bj_num_links] = plus_end[-1]
        return nodes, dirs

    def _get_bj_snapshot_like(self, env_idx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        nodes, dirs = self._get_wire_nodes_world(env_idx)
        rod_pos = nodes.T.astype(np.float64)  # (3, n_nodes)

        n = self._bj_num_links
        rod_dir = np.zeros((3, 3, n), dtype=np.float64)
        x_hint = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        y_hint = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        for i in range(n):
            z_dir = _safe_normalize(dirs[i], np.array([1.0, 0.0, 0.0], dtype=np.float64))
            x_dir = np.cross(y_hint, z_dir)
            if np.linalg.norm(x_dir) < 1.0e-8:
                x_dir = np.cross(z_dir, x_hint)
            x_dir = _safe_normalize(x_dir, x_hint)
            y_dir = _safe_normalize(np.cross(z_dir, x_dir), y_hint)
            rod_dir[:, :, i] = np.stack([x_dir, y_dir, z_dir], axis=0)

        tip = np.asarray(nodes[-1], dtype=np.float64)
        return rod_pos, rod_dir, tip

    # -------------------------
    # "cosim" interface
    # -------------------------
    def _rebuild_cosim(self, ids: np.ndarray):
        self._ensure_bj_wire_assets()

        for idx in ids.tolist():
            tip_pos, tip_rot = self._get_prim_world_pose(self.stick_tip_paths[idx])
            if tip_pos is None:
                raise RuntimeError(f"Cannot get stick-tip pose for env {idx}: {self.stick_tip_paths[idx]}")

            stick_dir = _safe_normalize(
                np.asarray(tip_rot, dtype=np.float64) @ np.array([0.0, 0.0, 1.0], dtype=np.float64),
                np.array([0.0, 0.0, 1.0], dtype=np.float64),
            )

            chain_dir = stick_dir  # outward along stick +Z
            self._set_wire_chain_pose(idx, first_node=np.asarray(tip_pos, dtype=np.float64), chain_dir=chain_dir)
            self._set_link_velocities_zero(idx)
            self._anchor_first_link_to_tip(idx)

            rod_pos, rod_dir, tip = self._get_bj_snapshot_like(idx)
            if not np.all(np.isfinite(tip)):
                tip = np.asarray(tip_pos, dtype=np.float64).copy()
            self.tip_world[idx] = tip
            self.prev_tip_world[idx] = tip
            self.tip_vel_world[idx] = 0.0
            self.ee_tip_world[idx] = np.asarray(tip_pos, dtype=np.float64).copy()
            self._update_wire_visual(idx, rod_pos, rod_dir, strict=True)

    def _step_cosim(self, duration: float):
        _ = duration
        if not self._bj_ready:
            return

        for i in range(self.num_envs):
            tip_pos, _ = self._get_prim_world_pose(self.stick_tip_paths[i])
            if tip_pos is None:
                continue
            self.ee_tip_world[i] = np.asarray(tip_pos, dtype=np.float64)

            rod_pos, rod_dir, tip = self._get_bj_snapshot_like(i)
            if not np.all(np.isfinite(tip)):
                tip = self.prev_tip_world[i].copy()
            self.tip_world[i] = tip
            self._update_wire_visual(i, rod_pos, rod_dir, strict=False)

    def reset(self, ids=None) -> np.ndarray:
        if ids is None:
            ids = np.arange(self.num_envs, dtype=np.int32)
        ids = np.asarray(ids, dtype=np.int32)
        if ids.size > 0:
            self.pending_reset[ids] = False
        return super().reset(ids)

    def _apply_sync_resets(self, rewards: np.ndarray, done: np.ndarray):
        rewards = np.asarray(rewards, dtype=np.float32).reshape(self.num_envs)
        done = np.asarray(done, dtype=np.bool_).reshape(self.num_envs)

        self.pending_reset |= done
        pending_mask = self.pending_reset.copy()
        rewards[pending_mask] = 0.0
        done_out = pending_mask.copy()

        pending_count = int(np.sum(pending_mask))
        pending_ratio = float(pending_count / max(self.num_envs, 1))
        trigger_reset = False
        if pending_count > 0:
            trigger_reset = (pending_count == self.num_envs) or (pending_ratio >= float(self.sync_reset_ratio))

        if trigger_reset:
            reset_ids = np.where(pending_mask)[0].astype(np.int32)
            self.reset(reset_ids)
            self.pending_reset[reset_ids] = False

        return rewards, done_out, pending_count, pending_ratio, trigger_reset

    # -------------------------
    # Step (same as your version)
    # -------------------------
    def step(self, actions: np.ndarray):
        actions = np.asarray(actions, dtype=np.float32)
        if actions.shape != (self.num_envs, self.act_dim):
            raise ValueError(f"Expected actions shape {(self.num_envs, self.act_dim)}, got {actions.shape}")
        actions = np.nan_to_num(actions, nan=0.0, posinf=1.0, neginf=-1.0)
        actions = np.clip(actions, -1.0, 1.0)
        if self.positive_only_active_joints:
            actions = np.clip(actions, 0.0, 1.0)
        frozen_mask = self.pending_reset.copy()
        if np.any(frozen_mask):
            actions[frozen_mask] = 0.0

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
            for env_idx in range(self.num_envs):
                self._anchor_first_link_to_tip(env_idx)
            self.world.step(render=False)
            for env_idx in range(self.num_envs):
                self._anchor_first_link_to_tip(env_idx)
            self._step_cosim(self.phys_dt)
        if not self.headless:
            self.world.render()

        dt = max(self.control_dt, 1.0e-8)
        self.tip_vel_world = (self.tip_world - self.prev_tip_world) / dt
        self.tip_vel_world = np.nan_to_num(self.tip_vel_world, nan=0.0, posinf=0.0, neginf=0.0)
        self.prev_tip_world = self.tip_world.copy()

        rewards, done, info = self._reward_done_info(actions)
        rewards, done, pending_count, pending_ratio, sync_reset_triggered = self._apply_sync_resets(rewards, done)
        info["commanded_joint_positions"] = commanded_q
        info["control_dt"] = float(self.control_dt)
        info["joint_names"] = list(self.joint_names)
        info["pending_reset_count"] = pending_count
        info["pending_reset_ratio"] = pending_ratio
        info["sync_reset_triggered"] = bool(sync_reset_triggered)
        self.prev_actions = actions.copy()

        return self.obs(), rewards, done, info
