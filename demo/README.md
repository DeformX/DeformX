# demo

Standalone Isaac Sim demos built on the Cosserat rod engine (`PyElastica-Mesh`), plus the
rope-swing reinforcement-learning demo.

| Folder | What |
|---|---|
| `cable_sim_demo/interactive/` | Drive a rod from a ball you move in the viewport; the rigid-capsule baseline; both side by side; a wire render |
| `cable_sim_demo/trajectory_replay/` | A UR5e scene, and replay of a joint-trajectory CSV with the rod or a ball-joint wire on the arm |
| `cable_sim_demo/ball_joint_wire/` | A PhysX ball-joint wire that follows an endpoint trajectory, and plots of the result |
| `cable_sim_demo/bunny_wire/` | A wire draped over the bunny, rendered from a precomputed rod trajectory |
| `rope_swing_rl/` | RL environments, PPO training and evaluation, replay tools and wire-end calibration: see [its README](rope_swing_rl/README.md) |
| `cable_sim_demo_gpu/` | **GPU solver** (Isaac Lab): a rope curtain swept by a box, a pile of hundreds of ropes, two-way coupling with PhysX bodies: see [its README](cable_sim_demo_gpu/README.md) |
| `rope_swing_rl_gpu/` | **GPU solver** (Isaac Lab + rsl_rl): the rope-swing RL task with 512 environments in one solver, plus the contact variant: see [its README](rope_swing_rl_gpu/README.md) |
| `common/` | Shared code: `paths.py` (wire and robot USD resolution) and `rod_skel_driver.py` (`SkeletonRodDriver`) |

The `*_gpu` demos need the `Cosserat-Rod-Solver-CUDA` submodule built and Isaac Lab installed
(see [The GPU rod solver](../README.md#the-gpu-rod-solver) in the root README); the others use
the CPU engine in `PyElastica-Mesh` and plain Isaac Sim.

## Standalone CUDA examples

[Videos](SHOWCASE.md) · [Installation and full catalog](EXAMPLES.md)

```bash
python -m demo.examples --list
python -m demo.examples hanging --out runs/hanging
python -m demo.examples braid --preset process --out runs/braid
```

The eleven standalone entries simulate without Isaac Sim. [Basics and a tip-target task](basics/README.md), [three-strand braiding](braiding_machine/README.md), [traction reels](winch_tension/README.md) and the [moving-guide benchmark](contact_arena/README.md) each document their options, controls and optional Isaac replay. The [humanoid guitar](humanoid_guitar/README.md) includes model downloads and Blender preparation; [dense cable cascade](cable_cascade/README.md) requires no external models. [Rack insertion](rack_insertion/README.md) adds contact-driven grasping and insertion in Isaac Lab, with asset preparation, launch and audit commands.

## Robot cable extraction (CUDA + Isaac Lab)

The [cable extraction task](cable_extraction/README.md) uses a Panda's finger contact to draw one strand out of an intertwined five-cable bundle. It includes source, Blender pull-tab generation, a measured-trajectory audit and an open-gripper control. It requires Isaac Lab and the external Panda/table assets, separately from the standalone CUDA examples.

## Running

The Isaac Sim scripts below run **from the repository root** with Isaac Sim's Python and the rod engine
on the path (setup in the [root README](../README.md#environment-setup)):

```bash
export ISAAC_PYTHON=python              # or /path/to/isaacsim/python.sh
export OMNI_KIT_ACCEPT_EULA=YES
export PYTHONPATH=PyElastica-Mesh
```

The interactive demos open the Isaac Sim viewport. Every script also takes `--headless`,
and most take `--max_steps N` to stop after N steps, which is how to smoke-test them on a
machine without a display. The wire asset defaults to the bundled
`dataset_generator/assets/wire/wire_yellow_s20_r0.005_l1_smooth.usdc` (`--wire_usd` or
`DEFORMX_WIRE_USD` to change it); the UR5e comes from your local Isaac assets or, failing
that, is streamed from NVIDIA's public mirror.

## Interactive

```bash
# Cosserat rod hanging from a ball: drag /World/Ball with the viewport gizmo while playing
$ISAAC_PYTHON demo/cable_sim_demo/interactive/move_wire_with_ball.py

# the same rig with a PhysX rigid-capsule chain instead of the rod
$ISAAC_PYTHON demo/cable_sim_demo/interactive/move_with_ball_rigid.py --num_links 20

# both wires on one ball, for a side-by-side comparison
$ISAAC_PYTHON demo/cable_sim_demo/interactive/two_rods_one_ball.py

# play a rod trajectory on the skinned wire mesh (synthetic two-frame trajectory if --npz_path is omitted)
$ISAAC_PYTHON demo/cable_sim_demo/interactive/render_wire.py [--npz_path traj.npz]
```

Add `--headless --max_steps 60` to any of them for a no-display check.

## Trajectory replay (UR5e)

A trajectory is a CSV with columns `t, shoulder_pan, shoulder_lift, elbow, wrist_1, wrist_2,
wrist_3` (radians), as exported by the RL demo's `RL.eval`.

```bash
# empty UR5e scene with a camera; a starting point for your own layouts
$ISAAC_PYTHON demo/cable_sim_demo/trajectory_replay/render_env.py [--headless]

# replay with the Cosserat rod on the end effector; writes a tip-trace CSV, a YZ plot and joint plots next to the CSV
$ISAAC_PYTHON demo/cable_sim_demo/trajectory_replay/replay_wire_traj.py /abs/path/traj.csv [--headless] [--use-ground-contact]

# replay with the ball-joint wire from the RL demo and record a camera video (<csv>_bj_camera.mp4)
$ISAAC_PYTHON demo/cable_sim_demo/trajectory_replay/replay_wire_traj_bj.py /abs/path/traj.csv [--headless] [--task-config demo/rope_swing_rl/conf/task/wire_swing_bj.yaml]
```

## Ball-joint wire

Input: an `.npz` with `pos` of shape `(T, 3)` (the moving endpoint, metres) and optionally
`time` `(T,)`. Each run writes `<name>_wire_joint_positions.npz` next to the input, which
`plot_bj_wire.py` turns into figures or a video.

```bash
$ISAAC_PYTHON demo/cable_sim_demo/ball_joint_wire/visualize_bj_wire.py --traj_npz endpoint.npz [--headless --max_steps 500]
$ISAAC_PYTHON demo/cable_sim_demo/ball_joint_wire/visualize_bj_wire_envcfg.py --traj_npz endpoint.npz      # same, with the RL task's wire parameters
$ISAAC_PYTHON demo/cable_sim_demo/ball_joint_wire/visualize_bj_wire_two_init_states.py --traj_npz endpoint.npz
python demo/cable_sim_demo/ball_joint_wire/plot_bj_wire.py --npz endpoint_wire_joint_positions.npz [--video --output out.mp4]
```

The trajectory used in the paper is not included; any `(T, 3)` endpoint path works.

## Bunny wire

`render_bunny_with_wire.py` expects three files in `demo/cable_sim_demo/bunny_wire/data/`
that are not tracked: `demo_1_b.usdc` (the stage), `yellow_rope_n100.usdc` (the wire) and
the rod trajectory `bunny_fixed_waypoint_rod_4view_n100_E200000_d0p4_sd0p04_cf5_state_1.npz`.
With them in place:

```bash
$ISAAC_PYTHON demo/cable_sim_demo/bunny_wire/render_bunny_with_wire.py [--scene_mode stage|empty] [--headless --exit_immediately]
```

## Shared code

`demo/common/paths.py` resolves the wire USD (`DEFORMX_WIRE_USD`) and the UR robot USD
(`ISAAC_ASSETS_ROOT`, then Isaac Sim's own assets root, then the public Omniverse mirror).
`demo/common/rod_skel_driver.py` is the `SkeletonRodDriver` that skins rod node positions
onto a wire USD skeleton; the RL environment, the replay tools and the interactive demos
all use it.
