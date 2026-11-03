"""Two-way rope <-> rigid-body co-simulation: the CUDA rope solver (ropesim) and PhysX exchange poses and
wrenches every frame through ropesim.isaac_bridge. Three scenes:

  box:    a rope_A draped over a 0.5 kg box on the ground settles (the wrench on the box = the weight it
          carries, the box stays put), then the held end is pulled at 0.1 m/s: rope-box friction (mu 0.8)
          beats box-ground friction (0.05) and the box is dragged along.
  pulley: a rope laid over a 0.3 kg cylinder lying on the ground; pulling the held end rolls the cylinder. Run it with
          --exchange 10: a freely rolling body is held only by the rope's friction (~400 N/m, ~1 N s/m), and a wrench
          delayed by one 60 Hz frame is negative damping for that mode (4.9 Hz, growing +1.2/s); exchanging ten times
          per frame (PhysX at 600 Hz, the rope's substeps unchanged) removes it.
  freebox: zero gravity, a 0.5 kg box floating in space is rammed head-on by the rope's tip (held end driven
          at 0.3 m/s): the box's momentum and angular momentum change must equal the integrated wrench the
          bridge handed to PhysX (the two-way path, with nothing else touching the box). The push is a stiff
          axial one, i.e. the regime where the one-frame explicit exchange injects energy (a 69 N spike sends
          the 0.5 kg box off at 2.6 m/s); the book-keeping is exact all the same. A sideways sweep was tried
          and never reaches the box (a free rope pulled by one end trails behind it in zero gravity).

Without --headless the Isaac Lab viewport shows the rope and the body. With --out it writes an .npz
(rope positions, body poses, wrenches, PhysX body velocities).

  python demo/cable_sim_demo_gpu/rope_box_pulley.py --scene box
  python demo/cable_sim_demo_gpu/rope_box_pulley.py --headless --scene pulley --exchange 10 --out pulley.npz
"""
import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--scene", default="box", choices=["box", "pulley", "freebox"])
parser.add_argument("--out", default="", help="optional .npz with the recording")
parser.add_argument("--seconds", type=float, default=6.0)
parser.add_argument("--exchange", type=int, default=1, help="wrench/pose exchanges per 1/60 s frame: PhysX runs at 60*K Hz, the rope SUB/K substeps per exchange. "
                    "A light body held only by rope friction (the cylinder) is unstable at K = 1 (one-frame lag = negative damping); K = 10 is quiet.")
parser.add_argument("--init", default="auto", choices=["auto", "arc", "tent"], help="pulley: start the rope wrapped around the cylinder (arc) or laid over its top (tent); auto = tent")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab.sim import SimulationCfg, SimulationContext  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402

import ropesim  # noqa: E402
from ropesim.isaac_bridge import ColliderSpec, RopeColliders  # noqa: E402
from rope_markers import RopeMarkers  # noqa: E402

ROPE_A = dict(rho=0.11714786175836406, ea=11472.95064075322, ei=0.0044797351719720385,
              drag_quad=0.010167854691459678, drag=0.02558028824431089, seg_len=0.05071785753493182)
R, NV, SUB, DT = 0.00635, 41, 100, 1 / 60
SEG = ROPE_A["seg_len"]
MU_RB = (0.8, 0.7)      # rope-body friction
MU_GROUND = 0.05        # body-ground friction (PhysX)

slippery = sim_utils.RigidBodyMaterialCfg(static_friction=MU_GROUND, dynamic_friction=MU_GROUND, restitution=0.0)


@configclass
class BoxSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg(physics_material=slippery))
    light = AssetBaseCfg(prim_path="/World/light", spawn=sim_utils.DomeLightCfg(intensity=2000.0))
    body = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Body",
        spawn=sim_utils.CuboidCfg(size=(0.4, 0.4, 0.2), rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                                  mass_props=sim_utils.MassPropertiesCfg(mass=0.5),
                                  collision_props=sim_utils.CollisionPropertiesCfg(), physics_material=slippery),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.0, 0.1)))


@configclass
class FreeBoxSceneCfg(InteractiveSceneCfg):
    light = AssetBaseCfg(prim_path="/World/light", spawn=sim_utils.DomeLightCfg(intensity=2000.0))
    body = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Body",
        spawn=sim_utils.CuboidCfg(size=(0.4, 0.4, 0.2), rigid_props=sim_utils.RigidBodyPropertiesCfg(linear_damping=0.0, angular_damping=0.0),
                                  mass_props=sim_utils.MassPropertiesCfg(mass=0.5),
                                  collision_props=sim_utils.CollisionPropertiesCfg(), physics_material=slippery),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.3, 0.0, 1.0)))


@configclass
class PulleySceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg(
        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0)))
    light = AssetBaseCfg(prim_path="/World/light", spawn=sim_utils.DomeLightCfg(intensity=2000.0))
    body = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Body",
        spawn=sim_utils.CylinderCfg(radius=0.05, height=0.3, axis="Y", rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                                    mass_props=sim_utils.MassPropertiesCfg(mass=0.3),
                                    collision_props=sim_utils.CollisionPropertiesCfg(),
                                    physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0)),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.2, 0.0, 0.05)))


def main():
    free = args.scene == "freebox"
    K = args.exchange
    if SUB % K: raise SystemExit("--exchange must divide the rope's substep count")
    sim = SimulationContext(SimulationCfg(dt=DT / K, render_interval=K, device="cuda", gravity=(0.0, 0.0, 0.0) if free else (0.0, 0.0, -9.81)))
    sim.set_camera_view([2.0, 2.0, 1.5], [0.3, 0.0, 0.1])
    cfg = {"box": BoxSceneCfg, "pulley": PulleySceneCfg, "freebox": FreeBoxSceneCfg}[args.scene](num_envs=1, env_spacing=3.0)
    scene = InteractiveScene(cfg)
    sim.reset()
    for _ in range(30 * K):       # let PhysX settle the body on the ground
        scene.write_data_to_sim(); sim.step(); scene.update(DT / K)

    # the rope (2 m): draped over the body along x, vertex 0 held on the ground at the left end, the rest
    # over the body and down onto the ground on the right (a polyline resampled at the segment length)
    zg = R + 0.002
    body = scene["body"]
    bx, by, bz = body.data.root_link_pos_w[0].cpu().numpy()          # where PhysX actually settled the body
    if args.scene == "box":
        x0, x1, top = bx - 0.2, bx + 0.2, bz + 0.1
        path = [(x0 - 0.7, 0, zg), (x0 - R, 0, zg), (x0 - R, 0, top + R), (x1 + R, 0, top + R), (x1 + R, 0, zg), (x1 + 2.0, 0, zg)]
    elif free:                                                        # straight along x, tip 30 cm short of the box's -x face, 5 cm below centre
        path = [(bx - 0.2 - 0.3 - (NV - 1) * SEG, 0, bz - 0.05), (bx - 0.2 - 0.3 + 0.01, 0, bz - 0.05)]
    elif args.init == "arc":                                          # wrapped 180 deg around the cylinder at r + R
        rc, xc = 0.05, bx
        arc = [(xc + (rc + R) * np.cos(a), 0, bz + (rc + R) * np.sin(a)) for a in np.linspace(np.pi, 0, 19)]
        path = [(xc - 1.0, 0, zg), (xc - rc - R - 0.05, 0, zg)] + arc + [(xc + rc + R + 0.05, 0, zg), (xc + 2.0, 0, zg)]
    else:                                                             # tent: straight slopes from the ground to 3 mm above the top
        rc, xc = 0.05, bx
        top = bz + rc + R + 0.003
        path = [(xc - 1.0, 0, zg), (xc - 0.35, 0, zg), (xc, 0, top), (xc + 0.35, 0, zg), (xc + 2.0, 0, zg)]
    print(f"body at ({bx:.3f}, {by:.3f}, {bz:.3f})", flush=True)
    pts = np.asarray(path, np.float64)
    seglen = np.linalg.norm(np.diff(pts, axis=0), axis=1); s = np.r_[0, np.cumsum(seglen)]
    want = np.arange(NV) * SEG
    pos = np.stack([np.interp(want, s, pts[:, k]) for k in range(3)], -1)[None].astype(np.float32)
    pin = np.zeros((1, NV), bool); pin[0, 0] = True
    rope = ropesim.RopeBatch.from_positions(pos, ROPE_A["rho"], ROPE_A["ea"], ROPE_A["ei"], DT / SUB, 16, pinned=pin)
    rope.damping = 0.0; rope.drag = ROPE_A["drag"]; rope.drag_quad = ROPE_A["drag_quad"]
    rope.set_gravity(np.array([0, 0, 0.0 if free else -9.81]))
    rope.set_contact(radius=R, mu_s=0.5, mu_k=0.4, v_slip=1e-3, kappa=2.0)
    if not free:
        rope.set_planes(np.array([[0, 0, 1.0, 0.0]], np.float32), radius=R, mu_s=0.5, mu_k=0.4)   # the ground, for the rope
    bridge = RopeColliders(rope, scene, [ColliderSpec("body", dynamic=True, mu_s=MU_RB[0], mu_k=MU_RB[1], spacing=0.005,
                                                      margin=0.03, round=0.0)])

    markers = None if args.headless else RopeMarkers(radius=R)
    root = torch.tensor(pos[:, 0], device="cuda"); zero = torch.zeros((1, 3), device="cuda")
    pull_v, pull_dir = (0.3 if free else -0.1), torch.tensor([[1.0, 0.0, 0.0]], device="cuda")   # freebox: the held end drives the tip into the box; else pulled away
    T_SETTLE, frames = (0.5 if free else 2.5), int(args.seconds / DT)
    rec = dict(t=[], P=[], body_pos=[], body_quat=[], body_vel=[], body_angvel=[], W=[], root=[])
    for f in range(frames):
        t0, t1 = f * DT, (f + 1) * DT
        d0, d1 = pull_v * max(0.0, t0 - T_SETTLE), pull_v * max(0.0, t1 - T_SETTLE)
        p0 = root + d0 * pull_dir; p1 = root + d1 * pull_dir
        v = (p1 - p0) / DT
        # --- K exchanges per frame: PhysX step (with the last exchange's rope wrench), then the rope over the same interval
        Wsum = 0.0
        for k in range(K):
            pa = p0 + (p1 - p0) * (k / K); pb = p0 + (p1 - p0) * ((k + 1) / K)
            scene.write_data_to_sim(); sim.step(); scene.update(DT / K)
            bridge.sync_poses()
            rope.step(pa, v, DT / K, SUB // K, root_pos_end=pb)
            bridge.apply_wrenches()
            Wsum = Wsum + bridge.wrenches()[0, 0]
        P = torch.as_tensor(rope.positions())
        if markers is not None:
            markers.update(P)
        rec["t"].append(t1); rec["P"].append(P.cpu().numpy()[0].copy())
        rec["body_pos"].append(body.data.root_link_pos_w[0].cpu().numpy().copy())
        rec["body_quat"].append(body.data.root_link_quat_w[0].cpu().numpy().copy())      # wxyz
        rec["body_vel"].append(body.data.root_lin_vel_w[0].cpu().numpy().copy())
        rec["body_angvel"].append(body.data.root_ang_vel_w[0].cpu().numpy().copy())
        rec["W"].append((Wsum / K).cpu().numpy().copy())
        rec["root"].append(p1[0].cpu().numpy().copy())
        if f % 60 == 0:
            W = rec["W"][-1]; bp = rec["body_pos"][-1]; bv = rec["body_vel"][-1]
            print(f"t={t1:5.2f}  wrench F=({W[0]:+.3f},{W[1]:+.3f},{W[2]:+.3f}) N  body x={bp[0]:.4f} z={bp[2]:.4f} vx={bv[0]:+.4f}  "
                  f"rope zmin={rec['P'][-1][:, 2].min():.4f} xmean={rec['P'][-1][:, 0].mean():.3f}", flush=True)
    if args.out:
        out = {k: np.asarray(v) for k, v in rec.items()}
        out["scene"] = args.scene; out["mass"] = 0.3 if args.scene == "pulley" else 0.5
        out["exchange"] = K
        out["inertia"] = body.root_physx_view.get_inertias()[0].cpu().numpy().copy()
        np.savez(args.out, **out)
        print("saved", args.out, flush=True)
    sys.stdout.flush()
    os._exit(0)     # Kit's headless teardown can hang; everything is on disk


if __name__ == "__main__":
    main()
