# DeformX: Deformable Linear Objects Simulation

### [🌐 Project Page](https://deformx.github.io/) | [📄 Paper](https://arxiv.org/abs/2606.22116) | [🤗 WireSeg-36K Dataset](https://huggingface.co/datasets/DeformX/WireSeg-36K) | [🤗 Render Assets](https://huggingface.co/datasets/DeformX/DeformX-Assets) | [🎥 Video](https://www.youtube.com/watch?v=CQvS8lQYSS8)

DeformX is a co-simulation framework for deformable linear objects (DLOs) such as wires, cables, and ropes. It couples a dedicated **Cosserat rod physics engine** with **NVIDIA Isaac Sim** to deliver DLO simulation that is **physically faithful** and **visually realistic** at the same time, while remaining directly compatible with scalable synthetic data generation and robot-learning pipelines.

> **Note**
> We are about to release a major upgrade built on a **Stable Cosserat Rods solver** with full **GPU/CUDA** acceleration, delivering large speedups over the current CPU engine. If you are interested, please star ⭐ this repo to stay tuned.

## 🔥 Updates

* **\[Coming Soon\]** We will release a **GPU/CUDA-accelerated Stable Cosserat Rods solver**. Built on a split position–rotation optimization scheme with a closed-form Gauss–Seidel quasi-static orientation update, it stays stable under large stiffness parameters and large time steps while supporting GPU parallelization. This **massively accelerates** simulation and helps eliminate the time-scale discrepancy between Isaac Sim and the rod engine. ⭐ Star this repo to be notified.
* **\[Jun 2026\]** We open-sourced **DeformX**, the co-simulation framework integrating a dedicated Cosserat rod engine with NVIDIA Isaac Sim, together with the dataset-generation and RL tooling in this repository.
* **\[Jun 2026\]** We released **[WireSeg-36K](https://huggingface.co/datasets/DeformX/WireSeg-36K)** — a synthetic wire instance segmentation dataset (36k RGB images, depth maps, and per-wire instance masks) — together with its generation pipeline and the [render assets](https://huggingface.co/datasets/DeformX/DeformX-Assets) needed to reproduce it.

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
asset_wireseg36k/                   External dataset/render assets, ignored by git
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
asset_wireseg36k/
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
| **Isaac Sim Python** | Anything importing Isaac Sim / Omniverse / the Cosserat rod engine: rendering, dataset generation, RL training & eval, visualization | Isaac Sim's own interpreter |

### 1. Prerequisites

| Component | Requirement |
|---|---|
| OS | Linux (tested on Ubuntu 22.04) |
| Python | 3.10 – 3.12 (for the `uv` utilities); 3.11 for a pip-installed Isaac Sim |
| NVIDIA Isaac Sim | 4.5 or 5.x |
| GPU | NVIDIA GPU with a recent CUDA driver (required by Isaac Sim) |
| [`uv`](https://github.com/astral-sh/uv) | package manager for the utility env (`python3 -m pip install uv` if missing) |

### 2. Clone the repository (with submodules)

The Cosserat rod engine lives in the **`PyElastica-Mesh` submodule**, so you must clone recursively. A plain `git clone` leaves that directory empty and nothing will run.

```bash
git clone --recursive https://github.com/DeformX/DeformX.git
cd DeformX
```

Already cloned without `--recursive`? Pull the submodule in:

```bash
git submodule update --init --recursive
```

### 3. Utility environment (`uv`, CPU-only)

For the helper scripts that don't need Isaac Sim — segmentation QA, contact sheets, the
render queue driver, and the Blender wire generators:

```bash
uv sync
source .venv/bin/activate
```

### 4. Isaac Sim runtime environment

Rendering, dataset generation, RL, and visualization must run with **Isaac Sim's Python**, not the `uv` venv. Isaac Sim/Omniverse modules (`isaacsim`, `omni`, `pxr`, `carb`) come from Isaac Sim itself. There are two supported install layouts:

**(a) pip install (Isaac Sim 4.5 / 5.x, recommended).** Isaac Sim runs under a normal Python 3.11 environment, so `ISAAC_PYTHON` is just that interpreter:

```bash
conda create -n deformx python=3.11 -y && conda activate deformx
pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com
export ISAAC_PYTHON=python
```

**(b) Launcher install.** Download Isaac Sim from NVIDIA and use its bundled launcher:

```bash
export ISAAC_PYTHON=/path/to/isaacsim/python.sh
```

Then install the extra dependencies into whichever interpreter you chose:

```bash
# Cosserat rod engine dependencies (PyElastica-Mesh builds on PyElastica).
# Install these FIRST: numba pins numpy<1.27, matching what Isaac Sim requires.
$ISAAC_PYTHON -m pip install numba scipy tqdm matplotlib open3d

# App-level dependencies used by the generators and RL:
$ISAAC_PYTHON -m pip install hydra-core omegaconf pillow pyyaml wandb "numpy<2"
```

> **Note**
> Keep `numpy` below 2.0. Isaac Sim pins `numpy==1.26.0` and `numba` requires `numpy<1.27`; installing a package that pulls numpy 2.x will break `import isaacsim`. For the same reason, do **not** install `opencv-python` here — Isaac Sim already ships `opencv-python-headless`, and having both shadows Isaac's copy.

The `PyElastica-Mesh` engine is added to the Python path automatically by the scripts, so it needs no separate install — only the dependencies above.

First run only: Isaac Sim asks you to accept the Omniverse EULA. Set `OMNI_KIT_ACCEPT_EULA=YES` for headless/batch runs.

### 5. Download render assets

The scene USDs, wire meshes, rod trajectories, and environment textures used for dataset rendering are not tracked in git. Get them from the companion HuggingFace repo:

```bash
pip install huggingface_hub
hf download DeformX/DeformX-Assets --repo-type dataset --local-dir ./asset_wireseg36k
```

That is the full ~2.1 GB set. Most of it is HDRIs and datacenter trajectories; for a ~200 MB subset that still runs the drop/hang generator end to end:

```bash
hf download DeformX/DeformX-Assets --repo-type dataset --local-dir ./asset_wireseg36k \
  --include "usd/*" --include "wires_traj_data/*" --include "ground/*" \
  --include "scripts/*" --include "*.md" --include "*.json" \
  --include "background/boiler_room_4k.hdr" --include "background/machine_shop_01_4k.hdr" \
  --include "background/empty_workshop_4k.hdr" --include "background/small_workshop_4k.hdr"
```

`--include` takes one pattern per flag; passing several patterns to a single `--include` makes the CLI treat all but the first as positional filenames and silently skip them.

See [Asset Layout](#asset-layout) for what the bundle contains and what it deliberately leaves out. Every path is overridable via environment variables — see [Configuration](#configuration-paths--environment-variables).

The RL and visualization demos need **no** external assets; they run from the wire USDs bundled in `usd/wire/`.

### Quick sanity check

```bash
# Utility env: confirm path resolution works (prints repo-relative defaults)
python -c "import deformx_paths as d; print(d.REPO_ROOT, d.ASSET_ROOT)"

# Isaac env (run from the repo root): confirm the rod engine imports
PYTHONPATH=PyElastica-Mesh $ISAAC_PYTHON -c "from co_sim.engine import CoSimEngine; print('rod engine OK')"

# Isaac env: render a single frame (needs the assets from step 5)
OMNI_KIT_ACCEPT_EULA=YES PYTHONPATH=PyElastica-Mesh \
  $ISAAC_PYTHON Dataset_generator/cli.py --frame 0 --do_seg --do_depth
```

## Configuration (Paths & Environment Variables)

DeformX contains **no machine-specific hardcoded paths**. Every location is
resolved by the helper module `deformx_paths.py` at the repository root, which
falls back to sensible **repo-relative defaults** so the code runs right after
`git clone`. Override any of the following environment variables only if your
assets/data live elsewhere:

| Variable | Purpose | Default |
|---|---|---|
| `ISAAC_PYTHON` | Isaac Sim's interpreter: its `python.sh` launcher, or just `python` for a pip install | `isaacsim/python.sh` |
| `ISAAC_ASSETS_ROOT` | Local Isaac `Assets/Isaac` tree (for robot USDs). If unset, DeformX queries Isaac Sim's own assets root and finally falls back to the public Omniverse S3 mirror. | unset |
| `DEFORMX_ROOT` | Repository root | auto-detected |
| `DEFORMX_ASSET_ROOT` | Large render/simulation assets (USDs, textures, trajectories) | `<repo>/asset_wireseg36k` |
| `DEFORMX_DATA_ROOT` | Local working data (npz, renders) | `<repo>/data` |
| `DEFORMX_OUTPUT_ROOT` | Generated outputs | `<repo>/output` |
| `DEFORMX_WIRE_USD` | Wire USD used by RL/visualization demos | `<repo>/usd/wire/wire_yellow_s20_r0.005_l1_smooth.usdc` |
| `DEFORMX_HDR` | HDR dome-light image for datacenter renders | `<asset_root>/hdr/dome.hdr` |

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
asset_wireseg36k/
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
    Datacenter_NVD@10012/          # NVIDIA content pack, obtained separately
```

Everything except `Datacenter_NVD@10012/` is published at
[**DeformX/DeformX-Assets**](https://huggingface.co/datasets/DeformX/DeformX-Assets).

**What the published bundle contains.** All scene USDs, the 72 wire meshes (12 colors × 3 radii × 2 finishes), all six drop/hang rod trajectories, the datacenter camera/light/cable USDs, **all 12** datacenter trajectory configs, **all 47** HDRI environment maps, and 18 ground textures. This is everything the **drop/hang generator** needs, and everything the datacenter renderer needs except the NVIDIA scene below.

**What it does not, and why.**

* **`Datacenter_NVD@10012/`** (~9.6 GB) is an NVIDIA Omniverse content pack covered by the [NVIDIA Omniverse License Agreement](https://docs.omniverse.nvidia.com/platform/latest/common/NVIDIA_Omniverse_License_Agreement.html), so we cannot redistribute it — but NVIDIA mirrors it publicly and you can fetch it yourself. See [Getting the NVIDIA datacenter scene](#getting-the-nvidia-datacenter-scene) below. Only the *datacenter* renders need it.
* **35 of the 53 ground textures** are from [Texturelabs](https://texturelabs.org), whose license does not permit redistributing the raw files. The 18 [Pexels](https://pexels.com) textures are included; download the Texturelabs ones directly to reproduce the paper's exact randomization.

Both the HDRI and ground-texture samplers randomize over whatever files are present, so a partial set still renders — you just get less appearance diversity.

### Getting the NVIDIA datacenter scene

The datacenter scene lives on NVIDIA's public Omniverse content bucket — no account, no Omniverse Launcher. The bucket permits anonymous access, so `aws s3 sync --no-sign-request` works:

```bash
# This directory name is what the renderer expects by default.
NVD='asset_wireseg36k/datacenter/Datacenter_NVD@10012'

aws s3 sync --no-sign-request \
  s3://omniverse-content-production/Assets/DigitalTwin/Assets/Datacenter/ \
  "$NVD/Assets/DigitalTwin/Assets/Datacenter/"          # ~0.30 GB

aws s3 sync --no-sign-request \
  s3://omniverse-content-production/Assets/DigitalTwin/Materials/ \
  "$NVD/Assets/DigitalTwin/Materials/"                  # ~9.98 GB
```

Land it at that exact path and `--datahall_usd` can be omitted — it already defaults to
`<asset_root>/datacenter/Datacenter_NVD@10012/.../Data_Hall/DataHall_Full_01.usd`. Anywhere
else, pass the stage explicitly:

```bash
--datahall_usd /abs/path/to/.../Facilities/Stages/Data_Hall/DataHall_Full_01.usd
```

Individual files are also reachable over plain HTTPS under
`https://omniverse-content-production.s3.us-west-2.amazonaws.com/Assets/DigitalTwin/`.

> **Note**
> Sync only the two prefixes above. All of `Assets/DigitalTwin/` is ~34 GB, most of which the datacenter scene does not use.
>
> The published renders used the internal `Datacenter_NVD@10012` snapshot. The public mirror is the current tree and carries a superset of the materials, so it is not guaranteed byte-for-byte identical to that pinned version — geometry and the data hall stages match, but exact material parity is not promised. Objects on the mirror are stamped 2024-04-17.

This content remains under the NVIDIA Omniverse License Agreement; it is not covered by this repository's MIT license.

The datacenter scene uses `Datacenter_NVD@10012/.../DataHall_Full_01.usd` plus the camera/light/cable USDs under `asset_wireseg36k/datacenter/`.

## Drop/Hang Dataset Generation

Run with Isaac Sim Python:

```bash
export ISAAC_PYTHON=python          # or /path/to/isaacsim/python.sh
export OMNI_KIT_ACCEPT_EULA=YES
PYTHONPATH=PyElastica-Mesh $ISAAC_PYTHON Dataset_generator/cli.py \
  --frame_start 0 \
  --frame_end 99 \
  --do_seg \
  --do_depth \
  --seed 42 \
  --num_variants 1 \
  --accum_steps 80 \
  --accum_subframes 16
```

Pick which scene to render with `--variant`, which indexes
`GeneratorConfig.scene_variant_specs`:

| `--variant` | Scene | Trajectory | Wires | Frames |
|---|---|---|---|---|
| 0 | `rod_drop_multi_2_plane.usdc` | `drop_n2_100.npz` | 2 | 0–99 |
| 1 | `rod_drop_multi_4_plane.usdc` | `drop_n4_100.npz` | 4 | 0–99 |
| 2 | `rod_drop_multi_8_plane.usdc` | `drop_n8_100.npz` | 8 | 0–99 |
| 3 | `rod_hang_flying.usdc` | `hang_n2_100.npz` | 2 | 0–199 |
| 4 | `rod_hang_flying.usdc` | `hang_n4_100.npz` | 4 | 0–199 |
| 5 | `rod_hang_flying.usdc` | `hang_n8_100.npz` | 8 | 0–199 |

Without `--variant`, frame numbers index a single **concatenated** range spanning all six
variants in order (0–99 is variant 0, 100–199 is variant 1, and so on) — which is easy to
trip over, so prefer `--variant` when you want one specific scene.

`--out_dir` overrides the output directory and `--cams_sample_per_frame` the number of
cameras sampled per frame, so a full sweep needs no edits to `config.py`:

```bash
PYTHONPATH=PyElastica-Mesh $ISAAC_PYTHON Dataset_generator/cli.py \
  --variant 3 --out_dir output/wireseg36k/hang_n2 --num_variants 5 --do_seg --do_depth
```

All three hang configurations can be queued back to back with:

```bash
python scripts/run_hang_wireseg36k_queue.py            # add --dry_run to preview
```

Remaining defaults come from `Dataset_generator/config.py`, which resolves everything under
`asset_wireseg36k/` (`usd`, `wires_traj_data`, `background`, `ground`) via `DEFORMX_ASSET_ROOT`.

Renders land as `<out_dir>/seed_XXX/frame_XXXXXX/{rgb,seg,depth}/`. `Dataset_generator/`
also contains `organize_after_run.py` (flattens that tree into `rgb/seg/depth` plus an
`index.jsonl`), `delete_plane.py` (drops the ground plane from segmentation masks), and
`make_review_contact_sheets.py` (QA contact sheets) — run any of them with `--help`.

## Datacenter Dataset Generation

Datacenter rendering uses `Dataset_generator_datacenter/scripts/render_wireseg36k_datacenter.py`.

It needs the NVIDIA data hall scene — see [Getting the NVIDIA datacenter scene](#getting-the-nvidia-datacenter-scene). Add `--dry_run` to list what would be rendered without starting Isaac Sim.

```bash
export ISAAC_PYTHON=python          # or /path/to/isaacsim/python.sh
export OMNI_KIT_ACCEPT_EULA=YES
$ISAAC_PYTHON Dataset_generator_datacenter/scripts/render_wireseg36k_datacenter.py \
  --asset_root asset_wireseg36k/datacenter \
  --output_root output/wireseg36k/datacenter \
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

Vary these flags for the other passes:

| Goal | Flags to change |
|---|---|
| Quick smoke test (one image) | `--config dgrid_c1_ns04 --traj_index 0 --seed_index 0 --camera_num 1` |
| Same-color pass (as above) | `--wire_color_mode same --seed_start 0 --max_seeds 6` |
| Mixed-color pass | `--wire_color_mode mixed --seed_start 6 --max_seeds 4` |

The published dataset is the same-color and mixed-color passes combined. Run
`--help` for the full flag list.

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
