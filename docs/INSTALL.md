# Installation and troubleshooting

DeformX uses up to three Python environments. Most setup problems come from running something in the wrong one, so pick the row that matches what you want to run.

| You want to run | Environment | Setup |
|---|---|---|
| Standalone GPU examples: basics, braider, reels, contact arena, cable cascade, guitar simulation | Standalone CUDA environment | [demo/EXAMPLES.md](../demo/EXAMPLES.md#installation) |
| Isaac Lab examples on the GPU solver: rack insertion, cable extraction, `cable_sim_demo_gpu`, `rope_swing_rl_gpu` | Isaac Sim Python with Isaac Lab and the GPU solver | [README: Isaac Sim runtime](../README.md#4-isaac-sim-runtime-environment) and [README: the GPU rod solver](../README.md#the-gpu-rod-solver) |
| CPU-engine demos, CPU rope-swing RL, dataset generation | Isaac Sim Python with `PyElastica-Mesh` on `PYTHONPATH` | [README: Environment Setup](../README.md#environment-setup) |
| Dataset organization, plots, contact sheets | `uv` virtualenv, CPU only | [README: Utility environment](../README.md#3-utility-environment-uv-cpu-only) |
| Native video replay of a saved run | Isaac Sim Python, kept separate from the standalone CUDA packages | each example's README |

Clone with `git clone --recursive`. Both rod engines are public submodules, and a plain clone leaves their folders empty.

## Configuration that was validated

The GPU release was built and checked with the versions below. Other combinations may work; these are the ones with recorded results in [GPU_VALIDATION.md](GPU_VALIDATION.md).

| Component | Version |
|---|---|
| GPU | NVIDIA RTX 5090 |
| CUDA toolkit | 12.8 |
| Python | 3.11 |
| PyTorch | 2.7.0+cu128 |
| NumPy | 1.26.0 |
| GCC | 11.4 |
| CMake | 3.31 (3.24 or newer is required) |
| GLM | 0.9.9.8 |
| Isaac Sim | 5.1.0 |
| Isaac Lab | 2.3.2.post1 |
| Blender, for example assets | 4.5 LTS |

Extra packages by example:

| Example | Also needs |
|---|---|
| Guitar simulation | SciPy 1.15.3, Trimesh 4.5.1, Warp 1.14.0; Blender 4.5.9 LTS to build the props; internet access on the first asset download |
| Guitar rendering | Isaac Sim 5.1.0 with Pillow, SciPy and Trimesh; FFmpeg with `libx264` |
| Rack insertion | Isaac Lab including `isaaclab_assets`; Isaac Sim's bundled Warp 1.8.2; a separate Python 3.11 environment with `usd-core==25.5.1` for asset preparation; Blender 4.5; access to NVIDIA's Isaac asset server or a local Isaac 5.1 asset root |
| Backend mesh-baking tests | `warp-lang`, SciPy, Trimesh |

## Check an installation

```bash
# standalone CUDA environment: which build of the solver is imported, and which GPU it sees
python -c "import ropesim, torch; print(ropesim._ropesim.__file__); print(torch.cuda.get_device_name())"
python -m demo.examples --list
python -m demo.examples hanging --out runs/hanging

# Isaac Sim environment, from the repository root: the CPU rod engine imports
PYTHONPATH=PyElastica-Mesh $ISAAC_PYTHON -c "from co_sim.engine import CoSimEngine; print('rod engine OK')"
```

## Troubleshooting

**The submodule folders are empty.** Run `git submodule update --init --recursive`.

**The solver build cannot find `nvcc`.** A PyTorch wheel does not install the CUDA compiler. Install a CUDA toolkit that matches your PyTorch build (PyTorch `cu128` goes with CUDA 12.8), set `CUDACXX` to its `nvcc`, and add its `lib64` to `LD_LIBRARY_PATH` before importing `ropesim`.

**The build fails in GLM headers.** Use GLM 0.9.9.8 and point `GLM_INCLUDE_DIR` (or `-DGLM_INCLUDE_DIR=...` in `CMAKE_ARGS`) at the folder that contains `glm/glm.hpp`.

**The built extension does not run on another GPU.** The build targets the GPU that is visible at build time. Pass the architectures you need, for example `CMAKE_ARGS="-DCMAKE_CUDA_ARCHITECTURES=89;120"`.

**An example behaves like an older solver.** Print `ropesim._ropesim.__file__`. If it points outside the checkout you just built, another installed copy is being imported first. The examples use the deterministic stepping API of the pinned backend revision and should not be run against an older one.

**Isaac Sim breaks after installing packages.** Keep `numpy` below 2.0 in the Isaac Sim interpreter: Isaac Sim pins `numpy==1.26.0` and `numba` needs `numpy<1.27`. Do not install `opencv-python` there, because Isaac Sim ships `opencv-python-headless` and a second copy shadows it. Keep the standalone CUDA packages in their own environment.

**Headless runs stop at a licence prompt.** Set `OMNI_KIT_ACCEPT_EULA=YES`. The first launch also bootstraps Kit once, which takes minutes and several GB of cache.

**Rack insertion fails at start-up after importing Warp.** Kit has to start before `ropesim` is imported, so that Kit loads its own compatible Warp. The shipped launcher does this; a custom script that imports standalone Warp or `ropesim` first will fail.

**Rack asset preparation breaks Isaac Sim's USD.** Run the preparation step in its own Python 3.11 environment. `usd-core` installed next to Isaac Sim shadows the USD bindings that Isaac Sim ships.

**Rack insertion cannot load its environment.** It loads `Isaac/Environments/Grid/default_environment.usd` from the Isaac 5.1 assets. Give the runtime access to NVIDIA's asset server, or configure a local asset root as described in the [rack README](../demo/rack_insertion/README.md).

**Rendering fails on a headless Linux host.** On hosts whose NVIDIA Vulkan driver needs an X session, set `DISPLAY` and `XAUTHORITY` for that session before rendering.

**An example refuses to start because the output folder exists.** Choose a new output directory for every run. The examples do not overwrite earlier results.

**A GPU test was skipped.** GPU checks need a visible device, and tests that need optional packages skip when those are missing. A skipped test is not evidence that the behaviour was simulated.
