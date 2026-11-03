"""Hundreds of ropes dropped onto a box and the floor, piling up with rope-rope contact.

All ropes live in one CUDA Cosserat solver (`ropesim.RopeBatch`): free ropes (no pinned
vertex) stepped with step_free(), contact with a ground plane and a signed-distance box, and
rope-rope contact between every pair. Isaac Lab draws the scene. Prints the throughput.

  python demo/cable_sim_demo_gpu/rope_pile.py                             # viewport, 128 ropes
  python demo/cable_sim_demo_gpu/rope_pile.py --headless --num_ropes 512 --seconds 4 --out pile.npz
"""
import argparse
import os
import sys
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Rope pile (ropesim + Isaac Lab).")
parser.add_argument("--num_ropes", type=int, default=128)
parser.add_argument("--seconds", type=float, default=6.0, help="simulated time to run")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--out", default="", help="optional .npz: rope positions every 6th frame")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.sim import SimulationCfg, SimulationContext  # noqa: E402

import ropesim  # noqa: E402
from ropesim import sdf  # noqa: E402
from rope_markers import BoxMarker, RopeMarkers  # noqa: E402

RHO, EA, EI, R = 0.117, 11473.0, 4.48e-3, 0.00635      # the DeformX wire
NV, SEG, SUB, ITR, DT = 13, 0.05, 100, 8, 1 / 60       # 0.6 m ropes
HALF = np.array([0.25, 0.25, 0.15])
BOX_POS = np.array([0.0, 0.0, 0.15], np.float32)       # resting on the floor


def main():
    N = args.num_ropes
    rng = np.random.default_rng(args.seed)
    sim = SimulationContext(SimulationCfg(dt=DT, device="cuda"))
    sim.set_camera_view([2.2, 2.2, 1.8], [0.0, 0.0, 0.3])
    sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
    sim_utils.DomeLightCfg(intensity=2000.0).func("/World/light", sim_utils.DomeLightCfg(intensity=2000.0))
    sim.reset()

    # straight ropes at random positions and headings above the box
    centre = np.c_[rng.uniform(-0.45, 0.45, N), rng.uniform(-0.45, 0.45, N), rng.uniform(0.8, 2.4, N)]
    yaw = rng.uniform(0, 2 * np.pi, N)
    tilt = rng.uniform(-0.3, 0.3, N)
    d = np.c_[np.cos(yaw) * np.cos(tilt), np.sin(yaw) * np.cos(tilt), np.sin(tilt)]
    s = (np.arange(NV) - (NV - 1) / 2) * SEG
    pos = (centre[:, None, :] + s[None, :, None] * d[:, None, :]).astype(np.float32)
    rope = ropesim.RopeBatch.from_positions(pos, RHO, EA, EI, DT / SUB, ITR, pinned=np.zeros((N, NV), bool))
    rope.damping = 0.0
    rope.drag = 0.1
    rope.set_gravity(np.array([0.0, 0.0, -9.81]))
    rope.set_contact(radius=R, mu_s=0.5, mu_k=0.4, v_slip=1e-3, kappa=2.0)
    rope.set_planes(np.array([[0, 0, 1.0, 0.0]], np.float32), radius=R, mu_s=0.5, mu_k=0.4)
    rope.set_rope_contact(True, 0.9 * R, 0.2 * R, 0.025, 0.4, 1e7, 1.2, N)
    grid = sdf.from_function(sdf.sdf_box(HALF, round=0.02), -HALF - 0.06, HALF + 0.06, 0.01)
    rope.add_mesh(*grid.as_args(), pos=BOX_POS, mu_s=0.6, mu_k=0.5)

    markers = None
    if not args.headless:
        markers = RopeMarkers(radius=R)
        BoxMarker(size=2 * HALF).update(BOX_POS, torch.tensor([1.0, 0.0, 0.0, 0.0]))
    frames = int(args.seconds / DT)
    rec, t_wall = [], []
    for f in range(frames):
        t0 = time.perf_counter()
        rope.step_free(DT, SUB)
        torch.cuda.synchronize()
        t_wall.append(time.perf_counter() - t0)
        P = torch.as_tensor(rope.positions())
        if args.out and f % 6 == 0:
            rec.append(P.cpu().numpy().copy())
        if markers is not None:
            markers.update(P)
            sim.step()
        if f % 60 == 0:
            z = P[:, :, 2]
            print(f"t={(f + 1) * DT:5.2f}  highest vertex {z.max().item():.2f} m  lowest {z.min().item():.4f} m  "
                  f"solver {t_wall[-1] * 1e3:.1f} ms/frame", flush=True)
    ms = float(np.median(t_wall[10:]) * 1e3)
    print(f"{N} ropes x {NV} vertices with rope-rope contact: {ms:.1f} ms per 60 Hz frame ({1e3 / ms / 60:.2f}x realtime)",
          flush=True)
    if args.out:
        np.savez(args.out, P=np.asarray(rec), dt=6 * DT, seg_len=SEG, radius=R, box_half=HALF, box_pos=BOX_POS)
        print("saved", args.out, flush=True)
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
