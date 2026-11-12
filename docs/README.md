# DeformX documentation

Start with the [project README](../README.md) for what DeformX is and a first run. This folder holds the guides that do not fit there.

| If you want to | Read |
|---|---|
| Install the environments and fix a failing setup | [INSTALL.md](INSTALL.md) |
| Understand how the rod solver and Isaac Sim/PhysX are coupled | [COSIMULATION.md](COSIMULATION.md) |
| See which checks ran on the GPU release, and their results | [GPU_VALIDATION.md](GPU_VALIDATION.md) |

## Where the rest lives

| Topic | Location |
|---|---|
| Running the standalone GPU examples | [demo/EXAMPLES.md](../demo/EXAMPLES.md) |
| Videos, credits and provenance of the showcase | [demo/SHOWCASE.md](../demo/SHOWCASE.md) |
| CPU-engine Isaac Sim demos | [demo/README.md](../demo/README.md) |
| WireSeg-36k dataset generation | [dataset_generator/README.md](../dataset_generator/README.md) |
| Rope-swing RL, CPU engine | [demo/rope_swing_rl/README.md](../demo/rope_swing_rl/README.md) |
| Rope-swing RL, GPU solver | [demo/rope_swing_rl_gpu/README.md](../demo/rope_swing_rl_gpu/README.md) |
| GPU solver API, physics model, contact laws and constants | [Cosserat-Rod-Solver-CUDA README](https://github.com/DeformX/Cosserat-Rod-Solver-CUDA/blob/96b8514c7c55c724f1abba1fbf6ed6642b7db72b/README.md) and [solver notes](https://github.com/DeformX/Cosserat-Rod-Solver-CUDA/blob/96b8514c7c55c724f1abba1fbf6ed6642b7db72b/docs/SOLVER.md) |

Each example folder has its own README with the exact commands, assets and recorded results:

| Example | Folder |
|---|---|
| Hanging rod, driven endpoints, obstacle, self-contact, materials, tip target | [demo/basics](../demo/basics/README.md) |
| Three-strand braiding | [demo/braiding_machine](../demo/braiding_machine/README.md) |
| Traction reels | [demo/winch_tension](../demo/winch_tension/README.md) |
| Batched contact arena | [demo/contact_arena](../demo/contact_arena/README.md) |
| Dense cable cascade | [demo/cable_cascade](../demo/cable_cascade/README.md) |
| Humanoid guitar strumming | [demo/humanoid_guitar](../demo/humanoid_guitar/README.md) |
| Robot rack insertion | [demo/rack_insertion](../demo/rack_insertion/README.md) |
| Robot cable extraction | [demo/cable_extraction](../demo/cable_extraction/README.md) |
| Isaac Lab demos on the GPU solver | [demo/cable_sim_demo_gpu](../demo/cable_sim_demo_gpu/README.md) |

## Citing

BibTeX for DeformX and for Point It, Strike It is in the [README](../README.md#citation).
