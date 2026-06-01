# CosseratX

CosseratX is a research workspace for cable/wire simulation, Isaac Sim rendering, dataset generation, and reinforcement-learning experiments around Cosserat-style and ball-joint wire models.

The repository contains code only. Large USD assets, generated datasets, logs, checkpoints, and review images are intentionally ignored by git.

## Repository Layout

```text
Dataset_generator/                  Drop/hang wire dataset generation, cleanup, organization, QA utilities
Dataset_generator_datacenter/       Datacenter-specific rendering scripts and wire skeleton driver
RL_Demo/                            Isaac Sim RL environments, PPO training, evaluation, replay tools
visualization_scripts/              Standalone Isaac Sim visualization and replay scripts
scripts/                            Queue/helper scripts
PyElastica-Mesh/                    Pinned submodule for PyElastica-Mesh
asset_wireseg32k/                   External dataset/render assets, ignored by git
output/                             Generated local outputs, ignored by git
logs/                               Local run logs, ignored by git
```

## Tracked vs External Data

Tracked in git:

```text
Python source code
Hydra configs
small RL initial states/configs
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

The ignored asset/data folders must be restored separately before rendering or training.

## Environment Setup

### Non-Isaac Utilities

Use `uv` for normal Python utilities such as dataset organization, segmentation percentage QA, and contact-sheet scripts.

```bash
cd /path/to/CosseratX
uv sync
source .venv/bin/activate
```

Optional RL/development dependencies:

```bash
uv sync --extra rl --extra dev
```

If `uv` is not installed:

```bash
python3 -m pip install uv
```

### Isaac Sim Runtime

Rendering and RL simulation must run with Isaac Sim Python, not the normal `uv` virtual environment.

Example:

```bash
ISAAC_PYTHON=/path/to/isaacsim/python.sh
$ISAAC_PYTHON -m pip install hydra-core omegaconf pillow opencv-python pyyaml wandb
```

Isaac Sim/Omniverse modules such as `isaacsim`, `omni`, `pxr`, and `carb` are provided by Isaac Sim and are not listed as normal PyPI dependencies.

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

## Datacenter Organization and Filtering

Organize raw datacenter Replicator output:

```bash
python Dataset_generator_datacenter/scripts/organize_wireseg32k_datacenter.py \
  --root output/wireseg32k/datacenter_formal \
  --out output/wireseg32k/datacenter_formal_organized
```

Survey segmentation percentage:

```bash
python Dataset_generator/segment_percentage_survey.py \
  --root output/wireseg32k/datacenter_formal_organized \
  --out output/wireseg32k/review/segment_percentage_datacenter \
  --threshold 2.0 \
  --all-mapped
```

Filter and merge original datacenter data with refill data, keeping only samples with `wire_percent > 2.0` and capping each config at 500 samples:

```bash
python Dataset_generator/filter_reorganize_by_segmentation.py \
  --source /path/to/base/wireseg32k_organized/datacenter_formal \
  --source /path/to/refill/datacenter_refill_organized \
  --out /path/to/final/wireseg32k_organized/datacenter_formal \
  --threshold 2.0 \
  --target-per-config 500 \
  --selection source_order \
  --copy-mode hardlink
```

The filter writes a new organized dataset and does not modify the input sources.

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
/path/to/isaacsim/python.sh -m RL.train task=wire_swing_bj algo=ppo render=false total_steps=1000
```

## Release Checklist

Before tagging a release:

```bash
git status --short
python3 - <<'PY'
from pathlib import Path
roots = ['Dataset_generator', 'Dataset_generator_datacenter', 'RL_Demo', 'scripts', 'visualization_scripts']
for root in roots:
    for path in Path(root).rglob('*.py'):
        if '__pycache__' in path.parts:
            continue
        compile(path.read_text(encoding='utf-8'), str(path), 'exec')
print('syntax_ok')
PY
```

Recommended checks:

```bash
python Dataset_generator/segment_percentage_survey.py --help
python Dataset_generator/filter_reorganize_by_segmentation.py --help
python Dataset_generator_datacenter/scripts/organize_wireseg32k_datacenter.py --help
```

Do not commit generated datasets, raw renders, logs, checkpoints, or local asset folders.
