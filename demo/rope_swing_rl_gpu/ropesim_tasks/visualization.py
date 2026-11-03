"""Viewport visualization for the RopeSwing task.

Everything persistent/shaded uses Isaac Lab VisualizationMarkers (USD
PointInstancer — one batched visualize() per element per frame):
  - target       : sphere of the success radius; yellow while pending, green once hit
  - rope         : orange cylinder per segment (N envs x 32 segments)
  - plastic tube : thicker dark-gray cylinder, EE -> rope root
The rope TIP TRAIL uses the immediate-mode debug-draw interface (fading
polyline of the last ~2 s), which is unavailable headless — guarded.

Instantiated lazily via the env's debug_vis hooks; zero cost when off.
GUI mode is intended for small env counts (<= 16).
"""
from __future__ import annotations

import torch

import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg


def quat_from_to(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    """Batched quaternion (w,x,y,z) rotating unit vectors src -> dst [n,3]."""
    h = src + dst
    l = h.norm(dim=-1, keepdim=True)
    h_ok = h / l.clamp(min=1e-8)
    w = (src * h_ok).sum(-1, keepdim=True)
    xyz = torch.cross(src, h_ok, dim=-1)
    q = torch.cat([w, xyz], dim=-1)
    # antiparallel: 180 deg about any axis perpendicular to src
    anti = l.squeeze(-1) < 1e-6
    if anti.any():
        s = src[anti]
        e = torch.zeros_like(s)
        e[:, 0] = 1.0
        p = torch.cross(s, e, dim=-1)
        bad = p.norm(dim=-1) < 1e-6
        if bad.any():
            e2 = torch.zeros_like(s[bad])
            e2[:, 1] = 1.0
            p[bad] = torch.cross(s[bad], e2, dim=-1)
        p = p / p.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        q[anti] = torch.cat([torch.zeros_like(p[:, :1]), p], dim=-1)
    return q


class RopeSwingVisualizer:
    ROPE_RADIUS = 0.008   # visual radius [m]
    TUBE_RADIUS = 0.016
    TRAIL_LEN = 120       # frames of tip history (~2 s)
    TRAIL_MAX_ENVS = 8    # trails only for the first K envs (clutter)

    def __init__(self, env):
        self.env = env
        dev = env.device
        n = env.num_envs

        # cached so the render callback NEVER routes through env properties
        # (env.num_envs -> env.scene raises during construction/teardown app
        # updates, e.g. UI clicks - the crash-on-click bug)
        self.n = env.num_envs
        self.dev = env.device

        # solid sphere = EXACTLY the final success radius (5 cm). The active
        # curriculum radius is logged to TensorBoard, not drawn.
        self.goal_markers = VisualizationMarkers(VisualizationMarkersCfg(
            prim_path="/Visuals/goal_spheres",
            markers={
                "pending": sim_utils.SphereCfg(
                    radius=1.0,
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(1.0, 0.8, 0.1), opacity=0.8)),
                "hit": sim_utils.SphereCfg(
                    radius=1.0,
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.1, 1.0, 0.25), opacity=0.9)),
            }))
        self.rope_markers = VisualizationMarkers(VisualizationMarkersCfg(
            prim_path="/Visuals/rope",
            markers={"seg": sim_utils.CylinderCfg(
                radius=1.0, height=1.0, axis="Z",
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.95, 0.45, 0.10)))}))
        self.tube_markers = VisualizationMarkers(VisualizationMarkersCfg(
            prim_path="/Visuals/tube",
            markers={"tube": sim_utils.CylinderCfg(
                radius=1.0, height=1.0, axis="Z",
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.25, 0.25, 0.32)))}))

        # tip trail ring buffer (GPU; drained to CPU per render)
        self.trail_envs = min(n, self.TRAIL_MAX_ENVS)
        self._trail = torch.zeros(self.trail_envs, self.TRAIL_LEN, 3,
                                  device=dev)
        self._trail_i = 0
        self._trail_init = False

        # immediate-mode line drawing (GUI only; absent headless)
        self._draw = None
        try:
            from isaacsim.core.utils.extensions import enable_extension
            enable_extension("isaacsim.util.debug_draw")
            from isaacsim.util.debug_draw import _debug_draw
            self._draw = _debug_draw.acquire_debug_draw_interface()
        except Exception:
            self._draw = None  # headless / extension unavailable

    # ------------------------------------------------------------- lifecycle
    def set_visibility(self, visible: bool):
        for m in (self.goal_markers,
                  self.rope_markers, self.tube_markers):
            m.set_visibility(visible)
        if not visible and self._draw is not None:
            self._draw.clear_lines()

    def record_tip(self):
        """Called once per physics step (from _step_rope)."""
        tip = self.env._rope_pos[: self.trail_envs, -1, :]
        if not self._trail_init:
            self._trail[:] = tip.unsqueeze(1)
            self._trail_init = True
        self._trail[:, self._trail_i, :] = tip
        self._trail_i = (self._trail_i + 1) % self.TRAIL_LEN

    def reset_trail(self, env_ids: torch.Tensor):
        """Collapse resetting envs' trails to the fresh tip (no teleport line)."""
        if not self._trail_init:
            return
        m = env_ids[env_ids < self.trail_envs]
        if m.numel():
            self._trail[m] = self.env._rope_pos[m, -1, :].unsqueeze(1)

    # ------------------------------------------------------------- rendering
    def update(self):
        """Called per render frame (post-update event)."""
        env = self.env
        n = self.n
        dev = self.dev

        # target spheres: solid = the success radius (5 cm); green once hit this episode
        scales = torch.full((n, 3), env.cfg.success_thresh, device=dev)
        self.goal_markers.visualize(translations=env._target_w,
                                    scales=scales,
                                    marker_indices=env._success_buf.long())
        # rope segments as cylinders
        p = env._rope_pos
        a, b = p[:, :-1, :], p[:, 1:, :]
        seg = (b - a).reshape(-1, 3)
        length = seg.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        direction = seg / length
        z = torch.zeros_like(direction)
        z[:, 2] = 1.0
        qr = quat_from_to(z, direction)
        rr = torch.full_like(length, self.ROPE_RADIUS)
        self.rope_markers.visualize(
            translations=((a + b) * 0.5).reshape(-1, 3),
            orientations=qr,
            scales=torch.cat([rr, rr, length], dim=-1))
        # tube: EE -> rope root
        ee_pos, _, _, _ = env._ee_state()
        root = env._prev_root_pos
        tseg = root - ee_pos
        tlen = tseg.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        tq = quat_from_to(z[:n], tseg / tlen)
        tr = torch.full_like(tlen, self.TUBE_RADIUS)
        self.tube_markers.visualize(
            translations=(ee_pos + root) * 0.5,
            orientations=tq,
            scales=torch.cat([tr, tr, tlen], dim=-1))
        # tip trails (fading polylines)
        if self._draw is not None and self._trail_init:
            self._draw.clear_lines()
            pts = torch.roll(self._trail, shifts=-self._trail_i, dims=1)
            pts = pts.cpu().numpy()
            starts, ends, colors, sizes = [], [], [], []
            h = self.TRAIL_LEN
            for e in range(self.trail_envs):
                for i in range(h - 1):
                    a_ = pts[e, i]
                    b_ = pts[e, i + 1]
                    starts.append((float(a_[0]), float(a_[1]), float(a_[2])))
                    ends.append((float(b_[0]), float(b_[1]), float(b_[2])))
                    alpha = 0.85 * (i + 1) / h
                    colors.append((1.0, 1.0, 1.0, alpha))
                    sizes.append(2.0)
            if starts:
                self._draw.draw_lines(starts, ends, colors, sizes)
