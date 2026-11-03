"""A curtain of hanging ropes swept by a moving box.

The ropes are simulated by the CUDA Cosserat solver (`ropesim`): rod-mesh contact against a
signed-distance box that moves every frame (friction relative to the moving surface),
rope-rope contact between all ropes, and a ground plane. Isaac Lab draws the scene. Prints the
force the ropes exert on the box and the solver throughput.

  python demo/cable_sim_demo_gpu/rope_curtain.py                      # viewport
  python demo/cable_sim_demo_gpu/rope_curtain.py --headless --seconds 4 --out curtain.npz
"""
import argparse
import os
import sys
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Rope curtain swept by a box (ropesim + Isaac Lab).")
parser.add_argument("--num_ropes", type=int, default=48, help="ropes in the curtain (3 cm apart)")
parser.add_argument("--seconds", type=float, default=10.0, help="simulated time to run")
parser.add_argument("--box_speed", type=float, default=0.5, help="box sweep speed [m/s]")
parser.add_argument("--out", default="", help="optional .npz: rope positions every frame")
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

# the DeformX wire: 12.7 mm braided nylon, 5 cm segments (same material as the RL task)
RHO, EA, EI, R = 0.117, 11473.0, 4.48e-3, 0.00635
NV, SEG, SUB, ITR, DT = 21, 0.05, 100, 8, 1 / 60
HALF = np.array([0.15, 0.15, 0.15])          # the box
BOX_Z, X0, X1 = 0.75, -0.9, 0.9              # sweeps back and forth between X0 and X1 at BOX_Z


def main():
    N = args.num_ropes
    sim = SimulationContext(SimulationCfg(dt=DT, device="cuda"))
    sim.set_camera_view([2.5, 2.5, 1.6], [0.0, 0.0, 0.7])
    sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
    sim_utils.DomeLightCfg(intensity=2000.0).func("/World/light", sim_utils.DomeLightCfg(intensity=2000.0))
    sim.reset()

    # ropes hang from pinned roots at z = 1.3, 3 cm apart along y
    y = (np.arange(N) - (N - 1) / 2) * 0.03
    pos = np.zeros((N, NV, 3), np.float32)
    pos[:, :, 1] = y[:, None]
    pos[:, :, 2] = 1.3 - np.arange(NV)[None] * SEG
    pin = np.zeros((N, NV), bool)
    pin[:, 0] = True
    rope = ropesim.RopeBatch.from_positions(pos, RHO, EA, EI, DT / SUB, ITR, pinned=pin)
    rope.damping = 0.0
    rope.drag = 0.1
    rope.set_gravity(np.array([0.0, 0.0, -9.81]))
    rope.set_contact(radius=R, mu_s=0.5, mu_k=0.4, v_slip=1e-3, kappa=2.0)
    rope.set_planes(np.array([[0, 0, 1.0, 0.0]], np.float32), radius=R, mu_s=0.5, mu_k=0.4)                       # the floor
    rope.set_rope_contact(True, 0.9 * R, 0.2 * R, 0.025, 0.4, 1e7, 1.2, N)          # all ropes in one group

    # the box: a signed-distance grid with rounded edges, one pose shared by every rope
    grid = sdf.from_function(sdf.sdf_box(HALF, round=0.02), -HALF - 0.06, HALF + 0.06, 0.01)
    box_pos = torch.tensor([[X0, 0.0, BOX_Z]], device="cuda").repeat(N, 1)
    box_quat = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device="cuda").repeat(N, 1)   # x, y, z, w
    mesh = rope.add_mesh(*grid.as_args(), pos=box_pos.cpu().numpy(), quat=box_quat.cpu().numpy(), mu_s=0.6, mu_k=0.5)
    rope.enable_contact_forces(True)

    markers = None
    if not args.headless:
        markers = RopeMarkers(radius=R)
        box_marker = BoxMarker(size=2 * HALF)
    root = torch.tensor(pos[:, 0], device="cuda")
    zero = torch.zeros((N, 3), device="cuda")
    frames = int(args.seconds / DT)
    period = 2 * (X1 - X0) / args.box_speed
    rec, t_wall = [], []
    for f in range(frames):
        t = (f + 1) * DT
        phase = (t % period) / period                                  # triangle wave X0 -> X1 -> X0
        x = X0 + (X1 - X0) * (2 * phase if phase < 0.5 else 2 - 2 * phase)
        box_pos[:, 0] = x
        t0 = time.perf_counter()
        rope.set_mesh_poses(mesh, box_pos, box_quat)                   # pose at the END of this frame
        rope.step(root, zero, DT, SUB)
        torch.cuda.synchronize()
        t_wall.append(time.perf_counter() - t0)
        P = torch.as_tensor(rope.positions())
        if args.out:
            rec.append(P.cpu().numpy().copy())
        if markers is not None:
            markers.update(P)
            box_marker.update(box_pos[0], torch.tensor([1.0, 0.0, 0.0, 0.0]))
            sim.step()
        if f % 60 == 0:
            F = torch.as_tensor(rope.mesh_wrenches_view())[:, 0, :3].sum(0).cpu().numpy()
            print(f"t={t:5.2f}  box x={x:+.2f}  force on box=({F[0]:+.2f},{F[1]:+.2f},{F[2]:+.2f}) N  "
                  f"rope zmin={P[:, :, 2].min().item():.3f}  solver {t_wall[-1] * 1e3:.1f} ms/frame", flush=True)
    ms = float(np.median(t_wall[10:]) * 1e3)
    print(f"{N} ropes x {NV} vertices, {SUB}x{ITR} substeps/iterations per frame: {ms:.1f} ms per 60 Hz frame "
          f"({1e3 / ms / 60:.1f}x realtime)", flush=True)
    if args.out:
        np.savez(args.out, P=np.asarray(rec), dt=DT, seg_len=SEG, radius=R, box_half=HALF, box_z=BOX_Z)
        print("saved", args.out, flush=True)
    sys.stdout.flush()
    os._exit(0)     # Kit's headless teardown can hang; everything is on disk


if __name__ == "__main__":
    main()
