# cable_sim_demo_gpu

Isaac Lab demos of the GPU Cosserat rod solver
([`Cosserat-Rod-Solver-CUDA`](../../Cosserat-Rod-Solver-CUDA), Python module `ropesim`):
many ropes in one solver, contact with planes, meshes and other ropes, and two-way coupling
with PhysX bodies. Without `--headless` each demo opens the Isaac Lab viewport and draws the
ropes; with it, it prints what it measures and can save an `.npz`.

Requirements on top of the root setup: Isaac Lab 2.x in the Isaac Sim environment and the
solver built in it (`pip install -e Cosserat-Rod-Solver-CUDA`, see the
[root README](../../README.md#the-gpu-rod-solver)). Run from the repository root.

| Script | What it shows |
|---|---|
| `rope_box_pulley.py` | Two-way coupling through `ropesim.isaac_bridge`: a rope draped over a PhysX box is pulled and drags the box (`--scene box`), rolls a cylinder (`--scene pulley`), or rams a floating box in zero gravity (`--scene freebox`); the wrench PhysX receives is printed |
| `rope_markers.py` | The viewport markers (cylinder per segment) the demos share |

```bash
python demo/cable_sim_demo_gpu/rope_box_pulley.py --scene box
python demo/cable_sim_demo_gpu/rope_box_pulley.py --headless --scene pulley --exchange 10 --out pulley.npz
```

Isaac Lab's own flags apply (`--headless`, `--device`, ...). `rope_box_pulley.py` runs at several times real time. The saved `.npz` files hold rope
positions per frame (`P` `[T, N, V, 3]`) plus the scene parameters. The ropes use the DeformX
wire's calibrated parameters (12.7 mm braided nylon: ρ 0.117 kg/m, EA 11.5 kN, EI 4.5e-3 N·m²).
