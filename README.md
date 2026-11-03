# DeformX: Deformable Linear Objects Simulation

### [🌐 Project Page](https://deformx.github.io/) | [📄 Paper](https://arxiv.org/abs/2606.22116) | [🤗 WireSeg-36k Dataset](https://huggingface.co/datasets/DeformX/WireSeg-36K) | [🤗 Render Assets](https://huggingface.co/datasets/DeformX/DeformX-Assets) | [🎥 Video](https://www.youtube.com/watch?v=CQvS8lQYSS8)

DeformX is a co-simulation framework for deformable linear objects (DLOs) such as wires, cables, and ropes. It couples a dedicated **Cosserat rod physics engine** with **NVIDIA Isaac Sim** to deliver DLO simulation that is **physically faithful** and **visually realistic** at the same time, while remaining directly compatible with scalable synthetic data generation and robot-learning pipelines.

> **Note**
> The **GPU/CUDA Stable Cosserat Rods solver** is here: the `Cosserat-Rod-Solver-CUDA` submodule (Python module `ropesim`), with Isaac Lab demos and the rope-swing RL task under `demo/`. See [The GPU rod solver](#the-gpu-rod-solver).

## 🔥 Updates

* **\[Nov 2026\]** We released the **GPU/CUDA Stable Cosserat Rods solver** ([Cosserat-Rod-Solver-CUDA](https://github.com/DeformX/Cosserat-Rod-Solver-CUDA)): thousands of ropes in one solver, contact with planes, meshes and other ropes, and two-way coupling with PhysX in Isaac Lab, with the rope-swing RL task and new demos in this repository.
* **\[Jun 2026\]** We open-sourced **DeformX**, the co-simulation framework integrating a dedicated Cosserat rod engine with NVIDIA Isaac Sim, together with the dataset-generation and RL tooling in this repository.
* **\[Jun 2026\]** We released **[WireSeg-36k](https://huggingface.co/datasets/DeformX/WireSeg-36K)** — a synthetic wire instance segmentation dataset (36k RGB images, depth maps, and per-wire instance masks) — together with its generation pipeline and the [render assets](https://huggingface.co/datasets/DeformX/DeformX-Assets) needed to reproduce it.

## Why DeformX?

Existing DLO simulators rarely satisfy three requirements at once: visual realism for perception, physical fidelity under gravity/contact/manipulation, and compatibility with robot-learning frameworks. Procedural tools (e.g. Bézier curves or connected cylinders in Blender) look plausible but lack physically grounded deformation; rigid-link or generic soft-body physics oversimplify the bending, twisting, and shear mechanics of slender elastic structures. DeformX is, to the best of our knowledge, the first framework to unify all three.

Crucially, because DLOs are visualized as meshes skinned to the rod, DeformX also supports **direct import of real DLO CAD assets** — unlike procedural or primitive-based pipelines that cannot use reusable CAD geometry. This further improves visual realism and lets you build diverse, photorealistic cables from existing CAD models.

### Key Advantages

* **Physically faithful Cosserat rod dynamics.** A dedicated Cosserat rod engine explicitly captures stretching, shearing, bending, and twisting of 1D continua — reproducing characteristic behaviors such as gravity-induced equilibria, curvature propagation, and torsion–bending coupling.
* **Interpretable material parameters, minimal tuning.** Dynamics are driven by physically meaningful properties (Young's modulus, shear modulus) rather than hand-tuned joint stiffness/damping, giving a clear link between simulation and real materials.
* **Free-form mesh contact.** DLOs interact with arbitrary free-form meshes via a penalty-based contact formulation, accelerated by a Bounding Volume Hierarchy (BVH) with AABB broad-phase pruning and robust repulsion handling for watertight meshes.
* **CAD-quality visualization via mesh skinning.** Discrete rod deformations are skinned onto smooth tubular meshes, so you can import real DLO CAD assets and render them photorealistically in Isaac Sim while staying fully consistent with the underlying rod dynamics.
* **Stable multi-rate co-simulation.** A multi-rate scheme advances fine-grained DLO dynamics within each Isaac Sim step and feeds back integrated contact wrenches/impulses, enabling stable bidirectional coupling across disparate time scales.
* **Built for robot learning.** The same modular interface powers interactive authoring, headless execution, large-scale dataset generation, and closed-loop RL training/evaluation.

## Repository Layout

```text
dataset_generator/          WireSeg-36k generation: drop/hang and datacenter renderers, the tracked wire USDs, Blender and queue tools
PyElastica-Mesh/            the CPU Cosserat rod engine (git submodule)
Cosserat-Rod-Solver-CUDA/   the GPU Cosserat rod solver, module `ropesim` (git submodule)
demo/
  cable_sim_demo/           Isaac Sim cable demos on the CPU engine: interactive, ball-joint wire, trajectory replay, bunny
  cable_sim_demo_gpu/       Isaac Lab demos on the GPU solver: rope curtain, rope pile, two-way coupling with PhysX bodies
  rope_swing_rl/            rope-swing RL (CPU engine, Isaac Sim + Hydra PPO) and wire-end calibration
```

Each folder has a README with every command:
[dataset_generator](dataset_generator/README.md) ·
[demo](demo/README.md) ·
[demo/cable_sim_demo_gpu](demo/cable_sim_demo_gpu/README.md) ·
[demo/rope_swing_rl](demo/rope_swing_rl/README.md) ·
[demo/rope_swing_rl/calibration](demo/rope_swing_rl/calibration/README.md) ·
[Cosserat-Rod-Solver-CUDA](Cosserat-Rod-Solver-CUDA/README.md).

Created on demand and ignored by git: `asset_wireseg36k/` (render assets), `output/`,
`outputs/`, `logs/`, `checkpoints/`, `data/`, `eval_exports/`, `wandb/`.

## Environment Setup

DeformX uses **two Python environments**. Using the right one is the most common source of confusion:

| Environment | Use it for | Python |
|---|---|---|
| **`uv` virtualenv** | CPU utilities: dataset organization, segmentation QA, plots, contact sheets | system Python 3.10–3.12 |
| **Isaac Sim Python** | Anything importing Isaac Sim / Omniverse / the rod engine: rendering, dataset generation, demos, RL | Isaac Sim's own interpreter |

### 1. Prerequisites

| Component | Requirement |
|---|---|
| OS | Linux (tested on Ubuntu 22.04) |
| Python | 3.10–3.12 for the `uv` utilities; 3.11 for a pip-installed Isaac Sim |
| NVIDIA Isaac Sim | 4.5 or 5.x (the code uses the `omni.isaac.*` API, which 5.x keeps as deprecated extensions enabled by its default experience) |
| GPU | NVIDIA GPU with a recent driver (required by Isaac Sim) |
| [`uv`](https://github.com/astral-sh/uv) | package manager for the utility env (`python3 -m pip install uv` if missing) |

### 2. Clone with submodules

Both rod engines are submodules (`PyElastica-Mesh`, `Cosserat-Rod-Solver-CUDA`); a plain `git clone` leaves them empty.

```bash
git clone --recursive https://github.com/DeformX/DeformX.git
cd DeformX
# already cloned without --recursive?
git submodule update --init --recursive
```

### 3. Utility environment (`uv`, CPU only)

```bash
uv sync
source .venv/bin/activate
```

### 4. Isaac Sim runtime environment

Two install layouts are supported; `ISAAC_PYTHON` names the interpreter either way.

```bash
# (a) pip install (Isaac Sim 4.5 / 5.x, recommended)
conda create -n deformx python=3.11 -y && conda activate deformx
pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com
export ISAAC_PYTHON=python

# (b) launcher install
export ISAAC_PYTHON=/path/to/isaacsim/python.sh
```

Then install the extra dependencies into that interpreter:

```bash
# rod engine (PyElastica-Mesh builds on PyElastica). Install these FIRST: numba pins numpy<1.27, which Isaac Sim requires.
$ISAAC_PYTHON -m pip install numba scipy tqdm matplotlib open3d
# generators, demos and RL
$ISAAC_PYTHON -m pip install hydra-core omegaconf pillow pyyaml "numpy<2"
```

> **Note**
> Keep `numpy` below 2.0: Isaac Sim pins `numpy==1.26.0` and `numba` requires `numpy<1.27`.
> Do **not** install `opencv-python` here; Isaac Sim ships `opencv-python-headless` and a second copy shadows it.

The engine is added to the path with `PYTHONPATH=PyElastica-Mesh` (the RL demo does it
itself). Set `OMNI_KIT_ACCEPT_EULA=YES` for headless and batch runs; otherwise the first
launch asks you to accept the Omniverse EULA. The first launch also bootstraps Kit once
(minutes, several GB of cache).

### 5. Quick sanity check

```bash
# utility env: path resolution (prints repo-relative defaults)
python -c "from dataset_generator import paths as d; print(d.REPO_ROOT, d.ASSET_ROOT)"

# Isaac env, from the repo root: the rod engine imports
PYTHONPATH=PyElastica-Mesh $ISAAC_PYTHON -c "from co_sim.engine import CoSimEngine; print('rod engine OK')"

# Isaac env: a demo without a display (60 physics steps, then exit)
PYTHONPATH=PyElastica-Mesh OMNI_KIT_ACCEPT_EULA=YES $ISAAC_PYTHON demo/cable_sim_demo/interactive/move_wire_with_ball.py --headless --max_steps 60

# Isaac env: render one dataset frame (needs the assets, see dataset_generator/README.md)
PYTHONPATH=PyElastica-Mesh OMNI_KIT_ACCEPT_EULA=YES $ISAAC_PYTHON dataset_generator/drop_hang/cli.py --variant 3 --frame 0 --do_seg --do_depth
```

## Configuration (paths and environment variables)

DeformX contains **no machine-specific hardcoded paths**. Every location is resolved by a
small helper module (`dataset_generator/paths.py` for the generators, `demo/common/paths.py`
for the demos) with a repository-relative default, so the code runs right after `git clone`.
Override only if your assets or data live elsewhere:

| Variable | Read by | Purpose | Default |
|---|---|---|---|
| `ISAAC_PYTHON` | both | Isaac Sim's interpreter: `python.sh`, or `python` for a pip install | `isaacsim/python.sh` |
| `DEFORMX_ROOT` | both | repository root | auto-detected |
| `DEFORMX_ASSET_ROOT` | generators | the render asset bundle | `<repo>/asset_wireseg36k` |
| `DEFORMX_DATA_ROOT` | generators | local working data (npz, renders) | `<repo>/data` |
| `DEFORMX_OUTPUT_ROOT` | generators | generated outputs | `<repo>/output` |
| `DEFORMX_HDR` | generators | HDR dome light for the datacenter animation renderer | `<asset_root>/hdr/dome.hdr` |
| `DEFORMX_WIRE_USD` | demos | wire USD used by the demos and RL | `<repo>/dataset_generator/assets/wire/wire_yellow_s20_r0.005_l1_smooth.usdc` |
| `ISAAC_ASSETS_ROOT` | demos | local Isaac `Assets/Isaac` tree for robot USDs; otherwise Isaac Sim's own assets root, then the public Omniverse mirror | unset |

Hydra configs read the same variables (e.g. `wire_usd: ${oc.env:DEFORMX_WIRE_USD,null}`) and
accept per-run overrides such as `task.wire_usd=/abs/path/to/wire.usdc`.

## Dataset Generation

The [WireSeg-36k](https://huggingface.co/datasets/DeformX/WireSeg-36K) pipeline: rod
trajectories are simulated once with the engine, then skinned onto wire meshes and rendered
in Isaac Sim with randomized cameras, lights, backgrounds and colors. Assets come from
[DeformX/DeformX-Assets](https://huggingface.co/datasets/DeformX/DeformX-Assets); the
datacenter scenes additionally need NVIDIA's data hall content pack.

```bash
PYTHONPATH=PyElastica-Mesh OMNI_KIT_ACCEPT_EULA=YES $ISAAC_PYTHON dataset_generator/drop_hang/cli.py \
  --variant 3 --frame_start 0 --frame_end 199 --do_seg --do_depth --out_dir output/wireseg36k/hang_n2
$ISAAC_PYTHON dataset_generator/datacenter/render_wireseg36k_datacenter.py --dry_run
```

Everything else (asset download, scene variants, the datacenter passes, post-processing,
wire asset generation in Blender) is in [dataset_generator/README.md](dataset_generator/README.md).

## The GPU rod solver

`Cosserat-Rod-Solver-CUDA/` is the Stable Cosserat Rods solver (Hsu et al., SIGGRAPH 2025)
wrapped as the Python module `ropesim`: all ropes of all environments in one GPU solver,
kinematic root driving for a robot end effector, contact with planes, signed-distance meshes
and other ropes, and `ropesim.isaac_bridge`, which turns PhysX bodies into rope colliders and
hands the rope's wrench back to PhysX. It is GPL-3.0 (its vendored solver is) and has its own
[README](Cosserat-Rod-Solver-CUDA/README.md) and [solver notes](Cosserat-Rod-Solver-CUDA/docs/SOLVER.md).

Build it into the Isaac Sim environment (a CUDA toolkit matching your torch build, CMake,
Ninja and glm are needed):

```bash
export CUDACXX=/usr/local/cuda-12.8/bin/nvcc             # torch cu128 <-> CUDA 12.8
$ISAAC_PYTHON -m pip install -e Cosserat-Rod-Solver-CUDA  # about a minute
$ISAAC_PYTHON -m pytest Cosserat-Rod-Solver-CUDA/tests    # 124 physics tests, optional
```

The GPU demos and the GPU RL task run on **Isaac Lab** (2.3.x on Isaac Sim 5.1):
`$ISAAC_PYTHON -m pip install "isaaclab[all]==2.3.2.post1" --extra-index-url https://pypi.nvidia.com`.

```bash
python demo/cable_sim_demo_gpu/rope_curtain.py                     # 48 ropes swept by a box, viewport
python demo/cable_sim_demo_gpu/rope_box_pulley.py --scene pulley   # two-way coupling with a PhysX cylinder
```

## Demos

All demos run from the repository root with Isaac Sim's Python and take `--headless`
(plus `--max_steps N` where there is a free-running loop). Commands and inputs for each are in
[demo/README.md](demo/README.md); training, evaluation, replay and calibration are in
[demo/rope_swing_rl/README.md](demo/rope_swing_rl/README.md).

```bash
PYTHONPATH=PyElastica-Mesh $ISAAC_PYTHON demo/cable_sim_demo/interactive/move_wire_with_ball.py      # drag the ball in the viewport
cd demo/rope_swing_rl && $ISAAC_PYTHON -m RL.train task=wire_swing algo=ppo render=false total_steps=1000
```

## Citation

If you find DeformX useful, please consider citing our work:

```bibtex
@inproceedings{deformx,
  title     = {DeformX: A Versatile Co-Simulation Framework for Deformable Linear Objects},
  author    = {Yang, Yi and Fei, Xiang and Wang, Lehong and Li, Chenhao and Dai, Zilin and Kou, Henry and Li, Lu and Choset, Howie},
  booktitle = {IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  year      = {2026}
}
```

## License

MIT (see `LICENSE`) for this repository. The submodules carry their own licenses:
`PyElastica-Mesh` (see its LICENSE) and `Cosserat-Rod-Solver-CUDA` (GPL-3.0, it vendors the
YarnBall solver). The NVIDIA datacenter content pack and the Texturelabs ground textures are
not covered and are not redistributed here.
