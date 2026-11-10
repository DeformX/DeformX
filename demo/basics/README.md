# DeformX basics

Run a free tip under gravity, move a support, compare contact on and off, or replace the controller in a small target task.

| Example | Video | What to try |
|---|---|---|
| `hanging` | [Hanging rod](media/hanging.mp4) | Watch the tilted rod settle from a fixed root. |
| `endpoints` | [Driven endpoints](media/endpoints.mp4) | Move the left support while the right one stays fixed. |
| `obstacle` | [Sphere obstacle](media/obstacle.mp4) | Turn off sphere contact and compare clearance. |
| `self-contact` | [Self-contact](media/self-contact.mp4) | Turn off rod self-contact while keeping the floor active. |
| `materials` | [Bending stiffness](media/materials.mp4) | Compare cantilevers that differ only in bending stiffness. |
| `tip-target` | [Tip target](media/tip-target.mp4) | Replace the feedback policy that steers the free tip. |

The reference videos show the whole scene at 1×, then replay the same recorded states in close-up at ⅛×.

## Run

Use the [CUDA example setup](../EXAMPLES.md). From the repository root, choose a fresh output directory for each run. NumPy, CUDA-enabled PyTorch and the built `ropesim` extension are sufficient for simulation.

```bash
python -m demo.examples hanging --out runs/hanging
python -m demo.examples endpoints --out runs/endpoints
python -m demo.examples obstacle --out runs/obstacle
python -m demo.examples self-contact --out runs/self-contact
python -m demo.examples materials --out runs/materials
python -m demo.examples tip-target --out runs/tip-target

python -m demo.examples obstacle --contact off --out runs/obstacle-off
python -m demo.examples self-contact --contact off --out runs/self-contact-off
python -m demo.examples tip-target --policy zero --out runs/tip-zero
python -m pip install matplotlib  # optional recording plots
python -m demo.basics.plot runs/tip-target
```

Use `--help` to inspect options for an example. The five scene runners accept `--seconds`, `--substeps` and `--iterations`. The outer step is 1/120 s. [Scene settings](scenes.py) use metres, seconds, kg/m for linear density, newtons for axial stiffness and N·m² for bending stiffness.

## Replace the policy

```python
from demo.basics.task import TipTargetTask, feedback_action

task = TipTargetTask()
obs, info = task.reset(seed=0)
while True:
    action = feedback_action(obs)
    obs, reward, terminated, truncated, info = task.step(action)
    if terminated or truncated:
        break
print(info["success"], info["error_m"])
```

The observation contains rod positions, velocities and the target position. A finite XY action with norm at most one controls the support velocity up to 0.20 m/s. The rod solver advances the free tip. Reward is negative tip-to-target distance in metres. Entering the tolerance terminates the episode, and reaching the step budget truncates it. Reset before stepping again.

## Read and replay a recording

The runners save `report.json` and `trajectory.npz`. The report separates numerical `complete` from `metrics.demonstrated` or `metrics.success`. A zero-action run can complete by truncation. The trajectory stores seconds in `t` and positions and velocities in `[frame, rope, vertex, xyz]` order. The contact cases add gap, wrench or nonlocal distance measurements. The target task adds `target`, `error_m`, `actions` and `rewards`. Matplotlib plots the saved arrays on CPU.

For optional replay, install Blender with USD export, and use Isaac Sim Python with Pillow, FFmpeg and DejaVu Sans available. On Ubuntu the font is in `fonts-dejavu-core`.

```bash
blender -b --python-exit-code 1 --python demo/basics/blender_props.py -- --out runs/basic-props
# Run with the Isaac Sim Python interpreter.
python -m demo.basics.render --case tip-target --data runs/tip-target \
  --assets runs/basic-props --out runs/tip-target-video
```

The Blender script generates the visual props. The renderer reads saved states while the physics clock stays fixed. Contact measurements describe the selected numerical model at saved samples. The materials and forces are illustrative.
