# Run the examples

The standalone scenes simulate with NumPy, CUDA-enabled PyTorch and `ropesim`; Isaac Sim is used for native video replay. Guitar also needs collision-baking dependencies and prepared assets. Robot rack insertion runs inside Isaac Lab. Each scene README gives its setup and rendering commands.

## Installation

Use Linux, an NVIDIA GPU and driver, Python 3.11, a C++17 compiler with OpenMP, CMake ≥ 3.24, Ninja and a CUDA toolkit compatible with PyTorch. The release candidate was built with CUDA 12.8, PyTorch 2.7.0+cu128, GCC 11.4 and GLM 0.9.9.8 on an RTX 5090. Use the pinned GLM version for this configuration.

```bash
git clone https://github.com/DeformX/DeformX.git
cd DeformX
git submodule update --init --recursive Cosserat-Rod-Solver-CUDA

python3.11 -m venv .venv-gpu
source .venv-gpu/bin/activate
python -m pip install --upgrade pip
python -m pip install numpy==1.26.0 cmake ninja
python -m pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu128

# Header-only build dependency, kept outside tracked source.
git clone --depth 1 --branch 0.9.9.8 https://github.com/g-truc/glm.git .deps/glm
CUDA_TOOLKIT_ROOT=/usr/local/cuda-12.8  # adjust to your CUDA toolkit
export CUDACXX="$CUDA_TOOLKIT_ROOT/bin/nvcc"
export LD_LIBRARY_PATH="$CUDA_TOOLKIT_ROOT/lib64:$CUDA_TOOLKIT_ROOT/targets/x86_64-linux/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CMAKE_ARGS="-DGLM_INCLUDE_DIR=$PWD/.deps/glm"
python -m pip install ./Cosserat-Rod-Solver-CUDA

python -c "import ropesim; print(ropesim._ropesim.__file__); import torch; print(torch.cuda.get_device_name())"
python -m demo.examples --list
python -m demo.examples hanging --out runs/hanging
```

The build targets the visible GPU by default. For an explicit architecture, add `-DCMAKE_CUDA_ARCHITECTURES=120` for RTX 5090, or the architecture appropriate to your GPU, to `CMAKE_ARGS`. A PyTorch wheel does not install the `nvcc` build tool. See the [backend installation notes](https://github.com/DeformX/Cosserat-Rod-Solver-CUDA/blob/96b8514c7c55c724f1abba1fbf6ed6642b7db72b/README.md#install) and [official PyTorch version instructions](https://pytorch.org/get-started/previous-versions/#v270).

During the release preview the CUDA submodule requires repository access. Both the parent and backend commits must be published before an anonymous recursive clone can work. Do not substitute an older backend: these examples use the deterministic stepping API in the pinned revision.

## Choose a scene

Run from the repository root and choose a fresh output directory each time.

| Name | Command | Try changing |
|---|---|---|
| Hanging rod | `python -m demo.examples hanging --out runs/hanging` | gravity and stiffness in the scene config |
| Driven endpoints | `python -m demo.examples endpoints --out runs/endpoints` | support motion |
| Obstacle | `python -m demo.examples obstacle --out runs/obstacle` | `--contact off` |
| Self-contact | `python -m demo.examples self-contact --out runs/self-contact` | `--contact off` |
| Materials | `python -m demo.examples materials --out runs/materials` | bending stiffness |
| Free-tip target | `python -m demo.examples tip-target --out runs/tip-target` | `--policy zero`, `--seed 3`, or your own policy |
| Braider | `python -m demo.examples braid --out runs/braid` | `--preset process`, `--no-contact` |
| Traction reels | `python -m demo.examples winch --out runs/reels` | `--cycles 8`, `--mode frictionless` |
| Contact arena | `python -m demo.examples contact-arena --worlds 64 --seconds 2 --out runs/arena` | batch size, contact and substeps |
| [Guitar](humanoid_guitar/README.md) | `python -m demo.examples guitar --assets runs/guitar-assets --out runs/guitar` | prepare the G1/guitar assets first; `--substeps 300` for refinement |
| [Cable cascade](cable_cascade/README.md) | `python -m demo.examples cascade --out runs/cascade` | `--count 4`, `--seconds 2`, `--seed 3` |

```bash
python -m demo.examples --info braid
python -m demo.examples braid --help
```

Options are example-specific. These runners already run without a viewport; they do not share the Isaac Lab `--headless` interface.

## Robot rack insertion (Isaac Lab)

[Download and prepare the robot, rack, panel and connector](rack_insertion/README.md#reproduce), then run in the configured Isaac Lab environment:

```bash
python -m demo.examples rack-insertion --profile runs/rack-assets/profile.json --out runs/rack-trial
python -m demo.rack_insertion.audit --data runs/rack-trial --out runs/rack-audit.json
```

This performs contact-driven grasping, insertion and release. The [cable extraction task](cable_extraction/README.md) is another Isaac Lab example with its own launcher.

## Read the outputs

The basic scenes, braider, reels, arena and cascade write `report.json` and `trajectory.npz`; the benchmark also writes timing arrays. Guitar and rack insertion use their own multi-stage recording layouts, documented in their READMEs. Positions and time are in metres and seconds, forces in N, torques in N·m. The report distinguishes numerical completion from a task or contact criterion. A control or short smoke run can finish without satisfying the full demonstration's criteria.

The target task provides `reset(seed=...)` and `step(action)`. Its tip is advanced by the rod solver; replace the policy to try your controller. [Task API and recording schema →](basics/README.md)

The longer examples include controls and optional resolution comparisons. See [braiding](braiding_machine/README.md), [reels](winch_tension/README.md) and [contact arena](contact_arena/README.md). Their parameters are illustrative, not calibrated for a particular real cable.

## Render the same run

Individual READMEs show Blender asset generation and Isaac Sim replay commands. Use your saved trajectory; rendering does not advance or replace its physics. Install Pillow, FFmpeg/libx264 and DejaVu Sans in that environment. Keep standalone CUDA packages separate from Isaac Sim's pinned packages. [Isaac setup →](../README.md#environment-setup)

## Checks

```bash
python -m pip install pytest matplotlib
python -m pytest tests
# Optional backend mesh-baking/geometry checks used in the recorded full suite:
python -m pip install warp-lang scipy trimesh
python -m pytest Cosserat-Rod-Solver-CUDA/tests
```

GPU checks require a visible device. Catalog, CLI validation, recording contracts and geometry helpers also have tests without a GPU. A skipped GPU test is not evidence of simulated behavior. The [candidate validation record](../docs/GPU_VALIDATION.md) states which checks actually ran.
