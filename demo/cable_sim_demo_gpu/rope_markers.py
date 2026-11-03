"""Isaac Lab viewport markers for ropesim ropes: one cylinder per segment, drawn in one batched
PointInstancer call per frame, plus a box marker for kinematic obstacles. Used by the GPU demos
in this folder; the RL task has its own richer visualizer."""
from __future__ import annotations

import torch

import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg


def quat_from_to(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    """Batched quaternion (w, x, y, z) rotating unit vectors src -> dst, both [n, 3]."""
    h = src + dst
    l = h.norm(dim=-1, keepdim=True)
    h_ok = h / l.clamp(min=1e-8)
    w = (src * h_ok).sum(-1, keepdim=True)
    xyz = torch.cross(src, h_ok, dim=-1)
    q = torch.cat([w, xyz], dim=-1)
    anti = l.squeeze(-1) < 1e-6                      # antiparallel: 180 deg about any axis perpendicular to src
    if anti.any():
        s = src[anti]
        e = torch.zeros_like(s)
        e[:, 0] = 1.0
        p = torch.cross(s, e, dim=-1)
        bad = p.norm(dim=-1) < 1e-6
        if bad.any():
            e2 = torch.zeros_like(s)
            e2[:, 1] = 1.0
            p[bad] = torch.cross(s[bad], e2[bad], dim=-1)
        p = p / p.norm(dim=-1, keepdim=True)
        q[anti] = torch.cat([torch.zeros_like(p[:, :1]), p], dim=-1)
    return q


class RopeMarkers:
    """Cylinders along every rope segment. update() takes positions [N, V, 3] (torch, any device)."""

    def __init__(self, radius: float = 0.00635, color=(1.0, 0.55, 0.1), prim_path: str = "/Visuals/Ropes"):
        cfg = VisualizationMarkersCfg(
            prim_path=prim_path,
            markers={
                "segment": sim_utils.CylinderCfg(
                    radius=radius, height=1.0, axis="Z",
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color)),
            },
        )
        self.markers = VisualizationMarkers(cfg)

    def update(self, positions) -> None:
        P = torch.as_tensor(positions)
        a = P[:, :-1].reshape(-1, 3)
        b = P[:, 1:].reshape(-1, 3)
        d = b - a
        L = d.norm(dim=-1, keepdim=True)
        z = torch.zeros_like(d)
        z[:, 2] = 1.0
        q = quat_from_to(z, d / L.clamp(min=1e-8))
        s = torch.ones_like(d)
        s[:, 2] = L[:, 0]
        self.markers.visualize(translations=(a + b) / 2, orientations=q, scales=s)


class BoxMarker:
    """One cuboid, posed with update(pos [3], quat_wxyz [4])."""

    def __init__(self, size, color=(0.3, 0.5, 0.9), prim_path: str = "/Visuals/Box"):
        cfg = VisualizationMarkersCfg(
            prim_path=prim_path,
            markers={"box": sim_utils.CuboidCfg(size=tuple(float(s) for s in size),
                                                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color))},
        )
        self.markers = VisualizationMarkers(cfg)

    def update(self, pos, quat_wxyz) -> None:
        self.markers.visualize(translations=torch.as_tensor(pos, dtype=torch.float32).reshape(1, 3),
                               orientations=torch.as_tensor(quat_wxyz, dtype=torch.float32).reshape(1, 4))
