# DeformX: Deformable Linear Objects Simulation

### [🌐 Project Page](https://deformx.github.io/)

DeformX is a co-simulation framework for deformable linear objects (DLOs) such as wires, cables, and ropes. It couples a dedicated **Cosserat rod physics engine** with **NVIDIA Isaac Sim** to deliver DLO simulation that is **physically faithful** and **visually realistic** at the same time, while remaining directly compatible with scalable synthetic data generation and robot-learning pipelines.

> **Note**
> We are about to release a major upgrade built on a **Stable Cosserat Rods solver** with full **GPU/CUDA** acceleration, delivering large speedups over the current CPU engine. If you are interested, please star ⭐ this repo to stay tuned.

## 🔥 Updates

* **\[Coming Soon\]** We will release a **GPU/CUDA-accelerated Stable Cosserat Rods solver**. Built on a split position–rotation optimization scheme with a closed-form Gauss–Seidel quasi-static orientation update, it stays stable under large stiffness parameters and large time steps while supporting GPU parallelization. This **massively accelerates** simulation and helps eliminate the time-scale discrepancy between Isaac Sim and the rod engine. ⭐ Star this repo to be notified.
* **\[Jun 2026\]** We open-sourced **DeformX**, the co-simulation framework integrating a dedicated Cosserat rod engine with NVIDIA Isaac Sim, together with the dataset-generation and RL tooling in this repository.
* **\[Jun 2026\]** We released the **WireSeg-32k** generation pipeline — a synthetic wire instance segmentation dataset (32k RGB images, depth maps, and per-wire instance masks).

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
Dataset_generator/                  Drop/hang wire dataset generation, cleanup, organization, QA utilities
Dataset_generator_datacenter/       Datacenter-specific rendering scripts and wire skeleton driver
RL_Demo/                            Isaac Sim RL environments, PPO training, evaluation, replay tools
visualization_scripts/              Standalone Isaac Sim visualization and replay scripts
scripts/                            Queue/helper scripts
PyElastica-Mesh/                    Pinned submodule for PyElastica-Mesh
asset_wireseg32k/                   External dataset/render assets, ignored by git
usd/wire/                           Small tracked wire USDs for RL/replay visualization
output/                             Generated local outputs, ignored by git
logs/                               Local run logs, ignored by git
```

## Tracked vs External Data

Tracked in git:

```text
Python source code
Hydra configs
small RL initial states/configs
small wire USD color set for RL/replay visualization
submodule pointer for PyElastica-Mesh
```

Not tracked in git:

```text
asset_wireseg32k/
asset/
output/
outputs/
logs/
wandb/
checkpoints/
__pycache__/
*.pyc
```

The ignored asset/data folders must be restored separately before dataset rendering or full experiments.
RL and replay smoke tests can use the small bundled wire assets under `usd/wire/`.

## Environment Setup

DeformX uses **two separate Python environments** — using the right one for each task is the most common source of confusion:

| Environment | Use it for | Python |
|---|---|---|
| **`uv` virtualenv** | Lightweight CPU utilities: dataset organization, segmentation QA, plotting/contact sheets | system Python 3.10–3.12 |
| **Isaac Sim Python** (`python.sh`) | Anything importing Isaac Sim / Omniverse / the Cosserat rod engine: rendering, dataset generation, RL training & eval, visualization | bundled with Isaac Sim |

### 1. Prerequisites

| Component | Requirement |
|---|---|
| OS | Linux (tested on Ubuntu 22.04) |
| Python | 3.10 – 3.12 (for the `uv` utilities) |
| NVIDIA Isaac Sim | 4.5 or 5.x, with its bundled `python.sh` |
| GPU | NVIDIA GPU with a recent CUDA driver (required by Isaac Sim) |
| [`uv`](https://github.com/astral-sh/uv) | package manager for the utility env (`python3 -m pip install uv` if missing) |

Isaac Sim is **not** a pip package — download it from NVIDIA and note the path to its `python.sh` launcher.

### 2. Clone the repository (with submodules)

The Cosserat rod engine lives in the **`PyElastica-Mesh` submodule**, so you must clone recursively. A plain `git clone` leaves that directory empty and nothing will run.

```bash
git clone --recursive git@github.com:DeformX/DeformX.git
cd DeformX
```

Already cloned without `--recursive`? Pull the submodule in:

```bash
git submodule update --init --recursive
```

### 3. Utility environment (`uv`, CPU-only)

For dataset organization, QA, and plotting scripts that don't need Isaac Sim:

```bash
uv sync                          # core utilities
uv sync --extra rl --extra dev   # optional: RL + dev extras (torch, wandb, pytest, ruff)
source .venv/bin/activate
```

### 4. Isaac Sim runtime environment

Rendering, dataset generation, RL, and visualization must run with **Isaac Sim's Python**, not the `uv` venv. Point `ISAAC_PYTHON` at the launcher and install the extra deps into it. Isaac Sim/Omniverse modules (`isaacsim`, `omni`, `pxr`, `carb`) ship with Isaac Sim and are not pip-installable.

```bash
export ISAAC_PYTHON=/path/to/isaacsim/python.sh

# App-level dependencies used by the generators and RL:
$ISAAC_PYTHON -m pip install hydra-core omegaconf pillow opencv-python pyyaml wandb

# Cosserat rod engine dependencies (PyElastica-Mesh builds on PyElastica):
$ISAAC_PYTHON -m pip install numba scipy tqdm matplotlib open3d
```

The `PyElastica-Mesh` engine is added to the Python path automatically by the scripts, so it needs no separate install — only the dependencies above.

### 5. Restore assets and configure paths

Large assets and datasets are not tracked in git (see [Tracked vs External Data](#tracked-vs-external-data)) and must be restored into `asset_wireseg32k/` before rendering or training. Every path is overridable via environment variables — see [Configuration](#configuration-paths--environment-variables).

### Quick sanity check

```bash
# Utility env: confirm path resolution works (prints repo-relative defaults)
python -c "import deformx_paths as d; print(d.REPO_ROOT, d.ASSET_ROOT)"

# Isaac env (run from the repo root): confirm the rod engine imports
PYTHONPATH=PyElastica-Mesh $ISAAC_PYTHON -c "from co_sim.engine import CoSimEngine; print('rod engine OK')"
```

## Configuration (Paths & Environment Variables)

DeformX contains **no machine-specific hardcoded paths**. Every location is
resolved by the helper module `deformx_paths.py` at the repository root, which
falls back to sensible **repo-relative defaults** so the code runs right after
`git clone`. Override any of the following environment variables only if your
assets/data live elsewhere:

| Variable | Purpose | Default |
|---|---|---|
| `ISAAC_PYTHON` | Path to Isaac Sim's `python.sh` launcher | `isaacsim/python.sh` |
| `ISAAC_ASSETS_ROOT` | Local Isaac `Assets/Isaac` tree (for robot USDs). If unset, DeformX queries Isaac Sim's own assets root and finally falls back to the public Omniverse S3 mirror. | unset |
| `DEFORMX_ROOT` | Repository root | auto-detected |
| `DEFORMX_ASSET_ROOT` | Large render/simulation assets (USDs, textures, trajectories) | `<repo>/asset_wireseg32k` |
| `DEFORMX_DATA_ROOT` | Local working data (npz, renders) | `<repo>/data` |
| `DEFORMX_OUTPUT_ROOT` | Generated outputs | `<repo>/output` |
| `DEFORMX_WIRE_USD` | Wire USD used by RL/visualization demos | `<repo>/usd/wire/wire_yellow_s20_r0.005_l1_smooth.usdc` |
| `DEFORMX_HDR` | HDR dome-light image for datacenter renders | `<asset_root>/hdr/dome.hdr` |

Example:

```bash
export ISAAC_PYTHON=/opt/isaacsim/python.sh
export DEFORMX_ASSET_ROOT=/data/deformx/assets
$ISAAC_PYTHON Dataset_generator/cli.py --frame 0 --do_seg
```

Hydra-based RL configs read the same variables via interpolation (e.g.
`wire_usd: ${oc.env:DEFORMX_WIRE_USD,null}`), and you can always override per-run
on the command line, e.g. `task.wire_usd=/abs/path/to/wire.usdc`.

Bundled wire USD variants for RL/replay visualization live in:

```text
usd/wire/
```

They are intentionally small and tracked so the RL and visualization demos do not
depend on private local asset paths. Large scene assets and generated datasets
remain external.

## Generating Wire Assets

Wire assets are generated procedurally in Blender: a Python script builds each
wire as a smooth cylindrical mesh rigged to a bone chain, then exports it to USD
(`.usdc`). To produce more wires, open Blender's **Scripting** tab, load
`scripts/batch_generate_wire.py`, and click **Run Script** — it generates and
exports one `.usdc` per color/radius combination, saving them next to your
`.blend` file (or wherever `WIRE_EXPORT_DIR` points). The companion
`scripts/make_wire.py` previews a single wire in-scene without exporting.

## Asset Layout

Expected high-level asset layout:

```text
asset_wireseg32k/
  background/
  ground/
  usd/
  wires_traj_data/
  datacenter/
    data_grid_clean/
      dgrid_c1_ns04.npz
      dgrid_c1_ns08.npz
      ...
    data center camera 1.usdc
    data center camera 2.usdc
    data center camera 3.usdc
    data center camera 4.usdc
    data center lights.usdc
    data center cable_*.usdc
    Datacenter_NVD@10012/
```

The datacenter scene uses `Datacenter_NVD@10012/.../DataHall_Full_01.usd` plus the camera/light/cable USDs under `asset_wireseg32k/datacenter/`.

## Drop/Hang Dataset Generation

Run with Isaac Sim Python:

```bash
ISAAC_PYTHON=/path/to/isaacsim/python.sh
$ISAAC_PYTHON Dataset_generator/cli.py \
  --frame_start 0 \
  --frame_end 99 \
  --do_seg \
  --do_depth \
  --seed 42 \
  --num_variants 1 \
  --accum_steps 80 \
  --accum_subframes 16
```

The generator uses paths from `Dataset_generator/config.py`, including `asset_wireseg32k/usd`, `asset_wireseg32k/wires_traj_data`, `asset_wireseg32k/background`, and `asset_wireseg32k/ground`.

After rendering, organize raw Replicator output:

```bash
python Dataset_generator/organize_after_run.py \
  --root output/<raw_run>/<config_or_seed> \
  --out output/<organized_run>/<config_name> \
  --copy
```

For drop/hang data, plane segmentation cleanup is handled by `Dataset_generator/delete_plane.py` through `organize_after_run.py`.

## Datacenter Dataset Generation

Datacenter rendering uses `Dataset_generator_datacenter/scripts/render_wireseg32k_datacenter.py`.

Smoke test:

```bash
ISAAC_PYTHON=/path/to/isaacsim/python.sh
$ISAAC_PYTHON Dataset_generator_datacenter/scripts/render_wireseg32k_datacenter.py \
  --asset_root asset_wireseg32k/datacenter \
  --output_root output/wireseg32k/datacenter_smoke \
  --config dgrid_c1_ns04 \
  --traj_index 0 \
  --seed_index 0 \
  --camera_num 1 \
  --reuse_stage \
  --do_seg \
  --do_depth \
  --lights_on_per_frame 8 \
  --light_intensity_min 800 \
  --light_intensity_max 2000 \
  --dome_intensity 700 \
  --render_mode RayTracedLighting \
  --accum_steps 80 \
  --accum_subframes 16
```

Formal-style render command:

```bash
ISAAC_PYTHON=/path/to/isaacsim/python.sh
$ISAAC_PYTHON Dataset_generator_datacenter/scripts/render_wireseg32k_datacenter.py \
  --asset_root asset_wireseg32k/datacenter \
  --output_root output/wireseg32k/datacenter_formal \
  --camera_num 5 \
  --reuse_stage \
  --do_seg \
  --do_depth \
  --lights_on_per_frame 8 \
  --light_intensity_min 800 \
  --light_intensity_max 2000 \
  --dome_intensity 700 \
  --render_mode RayTracedLighting \
  --accum_steps 80 \
  --accum_subframes 16 \
  --wire_color_mode same \
  --seed_start 0 \
  --max_seeds 6
```

For mixed-color variants:

```bash
$ISAAC_PYTHON Dataset_generator_datacenter/scripts/render_wireseg32k_datacenter.py \
  --asset_root asset_wireseg32k/datacenter \
  --output_root output/wireseg32k/datacenter_formal \
  --camera_num 5 \
  --reuse_stage \
  --do_seg \
  --do_depth \
  --lights_on_per_frame 8 \
  --light_intensity_min 800 \
  --light_intensity_max 2000 \
  --dome_intensity 700 \
  --render_mode RayTracedLighting \
  --accum_steps 80 \
  --accum_subframes 16 \
  --wire_color_mode mixed \
  --seed_start 6 \
  --max_seeds 4
```

## Final Organized Dataset Structure

Expected final structure:

```text
wireseg32k_organized/
  drop_n2/
    index.jsonl
    rgb/rgb_000000.png
    seg/seg_000000.png
    seg/seg_000000_mapping.json
    depth/dep_000000.npy
  drop_n4/
  drop_n8/
  hang_n2/
  hang_n4/
  hang_n8/
  datacenter_formal/
    dgrid_c1_ns04/
      index.jsonl
      rgb/
      seg/
      depth/
    dgrid_c1_ns08/
    dgrid_c1_ns16/
    ...
```

Current target counts:

```text
drop_n2/drop_n4/drop_n8:       5000 each
hang_n2/hang_n4/hang_n8:       5000 each
datacenter_formal/dgrid_*:      500 each
```

Total target size:

```text
36,000 RGB images
36,000 segmentation PNGs
36,000 segmentation mapping JSONs
36,000 depth NPY files
36,000 index rows
```

## RL Demo

See `RL_Demo/README.md` for training, evaluation, checkpoint, and replay commands.

Minimal example:

```bash
cd RL_Demo
$ISAAC_PYTHON -m RL.train task=wire_swing_bj algo=ppo render=false total_steps=1000
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
