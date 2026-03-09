from __future__ import annotations

import numpy as np

import RL.envs.wire_swing_env as _base
from RL.envs.wire_swing_env import WireSwingEnv


class WireSwingHitAppleEnv(WireSwingEnv):
    """Wire swing task with PyElastica wire and an apple-shaped visual target."""

    @staticmethod
    def _set_translate(prim, pos_xyz: np.ndarray) -> None:
        xf = _base.UsdGeom.Xformable(prim)
        xf.ClearXformOpOrder()
        xf.AddTranslateOp().Set(
            _base.Gf.Vec3d(float(pos_xyz[0]), float(pos_xyz[1]), float(pos_xyz[2]))
        )

    def _create_target(self, env_root: str, local_pos: np.ndarray):
        center = np.asarray(local_pos, dtype=np.float64).reshape(3)

        apple_radius = float(getattr(self.cfg, "apple_radius", 0.05))
        apple_color = np.asarray(getattr(self.cfg, "apple_color", [0.85, 0.08, 0.08]), dtype=np.float32)

        stem_radius = float(getattr(self.cfg, "apple_stem_radius", 0.004))
        stem_height = float(getattr(self.cfg, "apple_stem_height", 0.04))
        stem_color = np.asarray(getattr(self.cfg, "apple_stem_color", [0.40, 0.24, 0.10]), dtype=np.float32)

        leaf_radius = float(getattr(self.cfg, "apple_leaf_radius", 0.007))
        leaf_length = float(getattr(self.cfg, "apple_leaf_length", 0.035))
        leaf_color = np.asarray(getattr(self.cfg, "apple_leaf_color", [0.08, 0.55, 0.18]), dtype=np.float32)

        target_root = f"{env_root}/Target"
        _base.UsdGeom.Xform.Define(self.stage, target_root)

        body = _base.UsdGeom.Sphere.Define(self.stage, f"{target_root}/Body")
        body.GetRadiusAttr().Set(float(apple_radius))
        body.GetDisplayColorAttr().Set(
            [
                _base.Gf.Vec3f(
                    float(apple_color[0]),
                    float(apple_color[1]),
                    float(apple_color[2]),
                )
            ]
        )
        self._set_translate(body.GetPrim(), center)

        stem = _base.UsdGeom.Cylinder.Define(self.stage, f"{target_root}/Stem")
        stem.CreateAxisAttr("Z")
        stem.GetRadiusAttr().Set(float(stem_radius))
        stem.GetHeightAttr().Set(float(stem_height))
        stem.GetDisplayColorAttr().Set(
            [
                _base.Gf.Vec3f(
                    float(stem_color[0]),
                    float(stem_color[1]),
                    float(stem_color[2]),
                )
            ]
        )
        stem_pos = center + np.array([0.0, 0.0, apple_radius + 0.5 * stem_height], dtype=np.float64)
        self._set_translate(stem.GetPrim(), stem_pos)

        leaf = _base.UsdGeom.Capsule.Define(self.stage, f"{target_root}/Leaf")
        leaf.CreateAxisAttr("X")
        leaf.GetRadiusAttr().Set(float(leaf_radius))
        leaf.GetHeightAttr().Set(float(leaf_length))
        leaf.GetDisplayColorAttr().Set(
            [
                _base.Gf.Vec3f(
                    float(leaf_color[0]),
                    float(leaf_color[1]),
                    float(leaf_color[2]),
                )
            ]
        )
        leaf_xf = _base.UsdGeom.Xformable(leaf.GetPrim())
        leaf_xf.ClearXformOpOrder()
        leaf_pos = center + np.array(
            [apple_radius * 0.25, 0.0, apple_radius + stem_height * 0.75],
            dtype=np.float64,
        )
        leaf_xf.AddTranslateOp().Set(
            _base.Gf.Vec3d(float(leaf_pos[0]), float(leaf_pos[1]), float(leaf_pos[2]))
        )
        leaf_xf.AddRotateXYZOp().Set(_base.Gf.Vec3f(0.0, 20.0, 35.0))
