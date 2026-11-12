# How the co-simulation works

DeformX does not put cables inside the rigid-body engine. A dedicated Cosserat rod solver owns the cables, NVIDIA PhysX (through Isaac Sim or Isaac Lab) owns the robots and rigid objects, and the two exchange poses and forces every step. This page describes that exchange for the GPU solver, `ropesim`. The CPU engine in `PyElastica-Mesh` follows the same division with a multi-rate scheme, which the [paper](https://arxiv.org/abs/2606.22116) describes.

## Who owns what

| | Owner | State |
|---|---|---|
| Robots, grippers, plugs, boxes, the environment | PhysX | joint and rigid-body poses, velocities, contacts between rigid bodies |
| Cables, ropes, strings | `ropesim` | vertex positions, velocities and material frames of every rod |
| Contact between a cable and a body | `ropesim` | the body appears to the rod solver as a signed-distance grid |

A rod is a chain of vertices with stretch stiffness `EA` [N], bending stiffness `EI` [N·m²] and linear density [kg/m]. Many rods live in one batch and are advanced together on the GPU. The [solver notes](https://github.com/DeformX/Cosserat-Rod-Solver-CUDA/blob/96b8514c7c55c724f1abba1fbf6ed6642b7db72b/docs/SOLVER.md) give the model, the contact laws and their constants.

## One exchange step

Each control frame runs in this order:

1. **PhysX steps** the robots and rigid bodies, with the cable forces from the previous frame applied as external forces.
2. **Poses go to the rod solver.** The world pose of every body that a cable can touch is written to the solver's collider slots.
3. **The rod solver steps** its substeps. A cable end that is held by a robot follows that end effector as a driven root; a free cable is stepped without a root.
4. **Forces go back to PhysX.** The wrench the cables exerted on each dynamic body during the frame is handed to PhysX for its next step.

`ropesim.isaac_bridge.RopeColliders` packages steps 2 and 4 as `sync_poses()` and `apply_wrenches()`. [`demo/cable_sim_demo_gpu/rope_box_pulley.py`](../demo/cable_sim_demo_gpu/rope_box_pulley.py) is a compact example of the whole loop.

## Bodies as colliders

A body's USD collision geometry is baked once into a signed-distance grid in the body's own frame. During the run only its pose changes.

- Closed meshes are baked as solids. Open meshes are welded first, and otherwise baked as a two-sided shell with a chosen thickness.
- Grid spacing sets how much detail the cable feels. The rack example uses 0.5 mm for the patch panel, 1 mm for the socket, 2 mm for the gripper fingers and 4–8 mm for large or distant bodies.
- A batch holds up to 64 mesh colliders.
- **Kinematic** bodies push the cable and receive nothing back. Use this for robot links whose motion the cable should not disturb.
- **Dynamic** bodies also receive the cable's wrench, so a cable can drag a box or load a plug.

## Attaching a cable to a body

- **Driven root.** One end of the rod follows a pose, for example the end effector. This is how the rope-swing task holds its rope.
- **Compliant termination.** A free rod end and a rigid body are joined by a spring and damper, and the equal and opposite forces go to the rod and to PhysX. The rack example joins the cable to its plug this way, with 200 N/m and 0.125 N·s/m, so the plug stays a freely dynamic PhysX body that is held only by finger contact.

## Stability of the exchange

The exchange is explicit: PhysX sees the cable force one frame late. That is harmless for kinematic links and for bodies resting on the ground. It can become unstable for a light body that is held only by the cable, because a delayed spring force acts as negative damping. Two remedies are used in the examples:

- **Exchange more often.** The rack insertion and cable extraction examples exchange at 1,200 Hz. `rope_box_pulley.py --exchange K` sub-cycles the exchange K times per 60 Hz frame.
- **Keep the stiff link inside one solver.** A stiff axial push on a free body would need an implicit coupling, which is not implemented.

The [solver notes, section 6](https://github.com/DeformX/Cosserat-Rod-Solver-CUDA/blob/96b8514c7c55c724f1abba1fbf6ed6642b7db72b/docs/SOLVER.md) give the stability condition and measured cases.

## Rates used by the shipped examples

| Example | Exchange or record rate | Rod substeps per step | Solver iterations |
|---|---|---|---|
| Rack insertion | 1,200 Hz exchange with PhysX | 4 | 96 |
| Cable extraction | 1,200 Hz exchange with PhysX | see its README | 24 |
| Humanoid guitar | 480 Hz recorded states, prescribed robot motion | 200 | 24 |
| Rope-swing RL, GPU | 60 Hz control | 100 | 8 |

## What to expect for speed

The GPU solver is fast when many rods advance together. It is not fast per rod.

- **Many environments.** 1,024 ropes of 33 vertices advance at 406 frames/s on an RTX 4090, solver only, and the Isaac Lab rope-swing task reaches about 55,000 environment steps per second at 512 environments. These figures are from the [solver README](https://github.com/DeformX/Cosserat-Rod-Solver-CUDA/blob/96b8514c7c55c724f1abba1fbf6ed6642b7db72b/README.md#performance-measured).
- **One detailed scene.** A single contact-rich scene is stepped serially at a high exchange rate, and each step has a fixed cost for PhysX, the collider queries and the data exchange. The recorded rack insertion simulated 20.07 s in 329.91 s of loop time. Expect contact-rich single-environment examples to run well below real time.
- **Rod-rod contact** is brute force within a contact group, so scenes where every rod can touch every other rod cost more as the group grows.

## Conventions

- Lengths are in metres, time in seconds, forces in N, torques in N·m.
- `ropesim` uses quaternions in `xyzw` order. Isaac Lab uses `wxyz`. Reorder when passing poses between them.
- Keep the solver's internal `damping` at 0 and use `drag` for dissipation, as the solver notes explain.
- Start Kit before importing `ropesim` in an Isaac script, so that Kit loads its own Warp.
