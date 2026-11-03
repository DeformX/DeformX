# dataset_generator

Synthetic wire datasets rendered in Isaac Sim with the Cosserat rod engine
(`PyElastica-Mesh`). This is the pipeline behind
[WireSeg-36k](https://huggingface.co/datasets/DeformX/WireSeg-36K): rod trajectories are
simulated once, then skinned onto wire meshes and rendered with randomized cameras, lights,
backgrounds and wire colors.

| Folder / file | What |
|---|---|
| `drop_hang/` | Drop and hang scenes: `cli.py` renders RGB / instance masks / depth from precomputed rod trajectories; plus cleanup, organization and QA tools |
| `datacenter/` | Datacenter scenes: `render_wireseg36k_datacenter.py` (the dataset renderer), two animation renderers, and image utilities |
| `assets/wire/` | 13 small tracked wire USDs (12 colors + texture). The demos load these; the generators use the published asset bundle |
| `tools/` | Blender wire generators, the hang-scene render queue, a depth-map viewer |
| `rod_skel_driver.py` | `SkeletonRodDriver`: writes rod node positions into a wire USD skeleton each frame |
| `paths.py` | `DEFORMX_*` path resolution shared by the scripts here |

All Isaac Sim commands below run **from the repository root** with Isaac Sim's Python and
the rod engine on the path:

```bash
export ISAAC_PYTHON=python              # or /path/to/isaacsim/python.sh
export OMNI_KIT_ACCEPT_EULA=YES
export PYTHONPATH=PyElastica-Mesh
```

Environment setup (Isaac Sim, the `uv` utility env, dependencies) is in the
[root README](../README.md#environment-setup).

## 1. Assets

Scene USDs, wire meshes, rod trajectories and environment textures are not tracked in git.
Get them from the companion HuggingFace repo
[DeformX/DeformX-Assets](https://huggingface.co/datasets/DeformX/DeformX-Assets):

```bash
pip install huggingface_hub
hf download DeformX/DeformX-Assets --repo-type dataset --local-dir ./asset_wireseg36k
```

That is the full ~2.1 GB set. A ~200 MB subset still runs the drop/hang generator end to end:

```bash
hf download DeformX/DeformX-Assets --repo-type dataset --local-dir ./asset_wireseg36k \
  --include "usd/*" --include "wires_traj_data/*" --include "ground/*" \
  --include "scripts/*" --include "*.md" --include "*.json" \
  --include "background/boiler_room_4k.hdr" --include "background/machine_shop_01_4k.hdr" \
  --include "background/empty_workshop_4k.hdr" --include "background/small_workshop_4k.hdr"
```

`--include` takes one pattern per flag. Put the bundle at `<repo>/asset_wireseg36k` or point
`DEFORMX_ASSET_ROOT` at it.

Layout of the bundle:

```text
asset_wireseg36k/
  background/            47 HDRI environment maps
  ground/                18 ground textures (Pexels)
  usd/                   drop/hang scene USDs + usd/wire_usdc/ (72 wire meshes: 12 colors x 3 radii x 2 finishes)
  wires_traj_data/       the six drop/hang rod trajectories (drop_n{2,4,8}_100.npz, hang_n{2,4,8}_100.npz)
  datacenter/
    data_grid_clean/     12 datacenter trajectory bundles (dgrid_c<camera>_ns<wires>.npz)
    data center camera {1..4}.usdc, data center lights.usdc, data center cable_*.usdc
    Datacenter_NVD@10012/    NVIDIA content pack, obtained separately (below)
```

Not in the bundle, and why:

* **`Datacenter_NVD@10012/`** (~10 GB) is an NVIDIA Omniverse content pack covered by the
  [NVIDIA Omniverse License Agreement](https://docs.omniverse.nvidia.com/platform/latest/common/NVIDIA_Omniverse_License_Agreement.html),
  so it cannot be redistributed. Only the datacenter renders need it. NVIDIA mirrors it
  publicly and the bucket allows anonymous access:

  ```bash
  NVD='asset_wireseg36k/datacenter/Datacenter_NVD@10012'     # the path the renderer expects
  aws s3 sync --no-sign-request \
    s3://omniverse-content-production/Assets/DigitalTwin/Assets/Datacenter/ \
    "$NVD/Assets/DigitalTwin/Assets/Datacenter/"            # ~0.30 GB
  aws s3 sync --no-sign-request \
    s3://omniverse-content-production/Assets/DigitalTwin/Materials/ \
    "$NVD/Assets/DigitalTwin/Materials/"                    # ~9.98 GB
  ```

  Sync only these two prefixes (all of `Assets/DigitalTwin/` is ~34 GB). Anywhere else, pass
  the stage explicitly with `--datahall_usd .../Facilities/Stages/Data_Hall/DataHall_Full_01.usd`.
  The mirror is the current tree, a superset of the pinned snapshot the paper used; geometry
  and stages match, exact material parity is not promised.
* **35 of the 53 ground textures** are from [Texturelabs](https://texturelabs.org), whose
  license does not permit redistribution. Download them yourself to reproduce the paper's
  exact randomization; the samplers randomize over whatever files are present.

## 2. Drop / hang dataset

```bash
$ISAAC_PYTHON dataset_generator/drop_hang/cli.py \
  --frame_start 0 --frame_end 99 --do_seg --do_depth --seed 42 \
  --num_variants 1 --accum_steps 80 --accum_subframes 16
```

Pick the scene with `--variant` (indexes `GeneratorConfig.scene_variant_specs` in `drop_hang/config.py`):

| `--variant` | Scene | Trajectory | Wires | Frames |
|---|---|---|---|---|
| 0 | `rod_drop_multi_2_plane.usdc` | `drop_n2_100.npz` | 2 | 0–99 |
| 1 | `rod_drop_multi_4_plane.usdc` | `drop_n4_100.npz` | 4 | 0–99 |
| 2 | `rod_drop_multi_8_plane.usdc` | `drop_n8_100.npz` | 8 | 0–99 |
| 3 | `rod_hang_flying.usdc` | `hang_n2_100.npz` | 2 | 0–199 |
| 4 | `rod_hang_flying.usdc` | `hang_n4_100.npz` | 4 | 0–199 |
| 5 | `rod_hang_flying.usdc` | `hang_n8_100.npz` | 8 | 0–199 |

Without `--variant`, frame numbers index one concatenated range over all six variants
(0–99 is variant 0, 100–199 is variant 1, and so on), so prefer `--variant`.
`--out_dir` sets the output directory and `--cams_sample_per_frame` the cameras sampled per
frame. A single frame is a good smoke test:

```bash
$ISAAC_PYTHON dataset_generator/drop_hang/cli.py --variant 3 --frame 0 --do_seg --do_depth --out_dir output/smoke
```

Renders land as `<out_dir>/seed_XXX/frame_XXXXXX/{rgb,seg,depth}/`. The three hang
configurations can be queued back to back (each run is a separate `cli.py` subprocess,
logs in `logs/dataset_queue/`):

```bash
python dataset_generator/tools/run_hang_wireseg36k_queue.py            # --dry_run prints the commands
```

Post-processing (plain Python, `uv` env is enough):

| Command | What |
|---|---|
| `python dataset_generator/drop_hang/organize_after_run.py --root <out_dir> [--out DIR] [--copy]` | Flattens `seed_*/frame_*` into `rgb/ seg/ depth/` plus `index.jsonl`. Reads the per-camera `Replicator*` folders that only exist with `--cams_sample_per_frame 2` or more; a single-camera run exits with `No samples found` |
| `python dataset_generator/drop_hang/delete_plane.py <seg_dir>` | Removes the ground plane from instance masks |
| `python dataset_generator/drop_hang/make_review_contact_sheets.py --seed_dir <seed_XXX> --out_dir <dir>` | QA contact sheets |
| `python dataset_generator/tools/visualize_depth.py <depth.npy>` | Shows one depth map |

## 3. Datacenter dataset

The dataset renderer needs the NVIDIA data hall scene (section 1). `--dry_run` validates
config selection and counts renders without starting Isaac Sim:

```bash
$ISAAC_PYTHON dataset_generator/datacenter/render_wireseg36k_datacenter.py --dry_run
$ISAAC_PYTHON dataset_generator/datacenter/render_wireseg36k_datacenter.py \
  --asset_root asset_wireseg36k/datacenter \
  --output_root output/wireseg36k/datacenter \
  --camera_num 5 --reuse_stage --do_seg --do_depth \
  --lights_on_per_frame 8 --light_intensity_min 800 --light_intensity_max 2000 --dome_intensity 700 \
  --render_mode RayTracedLighting --accum_steps 80 --accum_subframes 16 \
  --wire_color_mode same --seed_start 0 --max_seeds 6
```

| Pass | Flags |
|---|---|
| Same-color pass | `--wire_color_mode same --seed_start 0 --max_seeds 6` |
| Mixed-color pass | `--wire_color_mode mixed --seed_start 6 --max_seeds 4` |

The published dataset is both passes combined. `--help` lists every flag (trajectory and
seed selection, cameras per frame, lights, resolution).

Animation renderers (one cable set, one fixed camera, frames to disk):

```bash
# cables animated inside the data hall; RGB + depth + instance seg per frame
$ISAAC_PYTHON dataset_generator/datacenter/render_datacenter.py --headless \
  --npz asset_wireseg36k/datacenter/data_grid_clean/dgrid_c1_ns04.npz --seed_index 0 \
  --cable_usd "asset_wireseg36k/datacenter/data center cable_blue.usdc" \
  --out output/datacenter_animation          # --datahall_usd defaults to the NVIDIA scene; skipped if missing

# cables only (no scene), RGB frames + mp4 (needs ffmpeg unless --skip_video)
$ISAAC_PYTHON dataset_generator/datacenter/render_easy_cable_animation.py --headless \
  --npz asset_wireseg36k/datacenter/data_grid_clean/dgrid_c1_ns04.npz --seed_index 0 \
  --cable_usd "asset_wireseg36k/datacenter/data center cable_blue.usdc" --out output/easy_cable_animation
```

Both read `pos` `(T, N, 3, K)`, `director` `(T, N, 3, 3, K-1)` and `time` `(T,)`; the
published `dgrid_*.npz` bundles carry a leading seed axis, selected with `--seed_index`.

Image utilities (plain Python):

```bash
python dataset_generator/datacenter/images_to_video.py -i output/datacenter_animation/rgb -o output/anim.mp4 --fps 30
python dataset_generator/datacenter/recolor_seg.py --input_dir <instance_seg> --output_dir <colorful> [--bg_mode black]
```

## 4. Generating wire assets (Blender)

The wire USDs are built procedurally in Blender: each wire is a smooth cylindrical mesh
rigged to a bone chain and exported as `.usdc`. Open Blender's **Scripting** tab, load
`dataset_generator/tools/batch_generate_wire.py` and click **Run Script**. It exports one
`.usdc` per color/radius combination next to your `.blend` file (or to `WIRE_EXPORT_DIR`).
`dataset_generator/tools/make_wire.py` previews a single wire in-scene without exporting.

## Paths and environment variables

Resolved by `dataset_generator/paths.py`; every default is repository-relative.

| Variable | Purpose | Default |
|---|---|---|
| `DEFORMX_ROOT` | Repository root | auto-detected |
| `DEFORMX_ASSET_ROOT` | The asset bundle | `<repo>/asset_wireseg36k` |
| `DEFORMX_DATA_ROOT` | Local working data | `<repo>/data` |
| `DEFORMX_OUTPUT_ROOT` | Generated outputs | `<repo>/output` |
| `DEFORMX_HDR` | HDR dome light for the datacenter animation renderer | `<asset_root>/hdr/dome.hdr` |
| `ISAAC_PYTHON` | Isaac Sim's interpreter | `isaacsim/python.sh` |
