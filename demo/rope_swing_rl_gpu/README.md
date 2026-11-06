# Rope-swing RL demo on the GPU solver (Isaac Lab)

The rope-swing task of [`demo/rope_swing_rl`](../rope_swing_rl/README.md) with the wire simulated by
the CUDA Cosserat solver ([`Cosserat-Rod-Solver-CUDA`](../../Cosserat-Rod-Solver-CUDA), module
`ropesim`) inside Isaac Lab. Same rig, same task, same reward; all environments share one
rope solver, so 512 to 2048 of them train in parallel.

| | CPU demo (`rope_swing_rl`) | this demo |
|---|---|---|
| Robot, mount, home pose | UR5e at z = 1.7 m, −90° about X; DeformX home | same |
| Stick, wire | rigid 0.65 m stick; 1 m wire, 20 segments, 12.7 mm, 700 kg/m³ | same |
| Wire attachment | `FixedJoint` at the stick tip (k 1500 N/m, kt 10 N·m/rad: rigid next to the wire's EI/l of 5e-3 N·m/rad) | root clamped in position and orientation, the first segment along the stick (`fix_root_orientation`) |
| Target, success | (0, 2.0, 1.5) m in the env frame; tip within 5 cm in the Y-Z plane | same |
| Actions | shoulder lift, elbow, wrist 1; positive position-target deltas, 90°/s × 3 | same, per 60 Hz control step |
| Observation (18) | active joint pos/vel, tip, tip velocity, stick tip, target | same |
| Reward, termination | proximity, progress, new-best, staged bonuses, tip velocity toward the target, wrist-1 velocity, penalties, success bonus; ends on success / 1.4 s / too far | same |
| Wire stiffness | one Young's modulus (2e5 Pa): EI 2.6e-4 N·m², EA 25 N | EI 2.6e-4 N·m², EA 3000 N (axial stiffness is independent here) |
| Dissipation | PyElastica damping constant 0.1 | solver drag 0.2 /s, no internal damping, 100 substeps × 8 iterations |
| Floor | ground contact in the engine, μ 1.0 / 0.5 | rod-plane contact with the same friction; the rope settles on the floor at the home pose |
| Framework | Isaac Sim + Hydra, own PPO, 500 Hz | Isaac Lab `DirectRLEnv` + rsl_rl PPO, 60 Hz |

Requirements on top of the root setup: Isaac Lab 2.x (tested with 2.3.2 on Isaac Sim 5.1) in
the Isaac Sim environment and the solver built in it (`pip install -e Cosserat-Rod-Solver-CUDA`).
Run from the repository root.

```bash
# wiring check: Isaac Sim boots, the UR5e loads, ropes rest on the floor and follow the stick, resets work
python demo/rope_swing_rl_gpu/run_env_smoke.py --headless --num_envs 16 --steps 180

# train headless (512 envs by default); logs and checkpoints in logs/rsl_rl/rope_swing/<run>/
python demo/rope_swing_rl_gpu/train.py --headless --num_envs 1024 --max_iterations 3000

# watch training in the viewport (16 envs, markers on; --realtime paces to wall clock)
python demo/rope_swing_rl_gpu/train.py --realtime

# evaluate a checkpoint (success rate, closest approach), watch it, or export env-0 rollouts
python demo/rope_swing_rl_gpu/play.py --headless --num_envs 256 --episodes 5 --checkpoint logs/rsl_rl/rope_swing/<run>/model_<n>.pt
python demo/rope_swing_rl_gpu/play.py --num_envs 4 --realtime --checkpoint <model.pt>
python demo/rope_swing_rl_gpu/play.py --headless --episodes 8 --export out.npz --checkpoint <model.pt>

# the same environment with the UR5e's link meshes as rope colliders (ropesim.isaac_bridge)
python demo/rope_swing_rl_gpu/swing_contact_compare.py --headless --contact 1 --out swing_contact1.npz
python demo/rope_swing_rl_gpu/swing_contact_compare.py --headless --contact 0 --out swing_contact0.npz
```

`ropesim_tasks/swing_env.py` holds the environment; `RopeSwingEnvCfg` has every parameter
(target, limits, wire, solver, floor friction, reward weights) and documents the few places
where the solver, not the task, dictates a value. `visualization.py` draws the target sphere,
the rope and the stick in the viewport.

The exported `.npz` holds one row per step of each finished env-0 episode (`episode`, `actions`,
`joint_targets`, `rope` positions `[V, 3]`, `dist`; the last row is the terminal step, not the
reset) plus per-episode `success` and `min_dist`; the joint targets are the 60 Hz sequence a real
arm would execute open-loop.

Notes: the reset shape is the rope settled on the floor, computed once at construction
(`from_positions(..., straight_rest=True)`, so the folded rope still wants to be straight).
Checkpoints trained before the root was clamped (`fix_root_orientation = False`, a free pivot)
saw different rope dynamics; retrain them, or set the flag to False to evaluate them as trained.
One Isaac Sim instance per process; set `PYTHONUNBUFFERED=1` when redirecting output, since
Kit drops buffered stdout at exit. The UR5e USD comes from a local Isaac assets tree or
NVIDIA's public mirror (`ROPE_UR5E_USD` overrides it).
