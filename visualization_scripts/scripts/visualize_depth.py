import os
import sys

import numpy as np
import matplotlib.pyplot as plt

# Depth .npy to visualize: pass as the first CLI arg or set DEFORMX_DEPTH_NPY.
_depth_npy = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("DEFORMX_DEPTH_NPY")
if not _depth_npy:
    raise SystemExit(
        "Usage: python visualize_depth.py <depth.npy>  (or set DEFORMX_DEPTH_NPY)"
    )

d = np.load(_depth_npy)   # usually (H, W) or (H, W, 1)
d = np.squeeze(d).astype(np.float32, copy=False)

valid = np.isfinite(d)  # exclude inf/nan
dv = d[valid]

lo, hi = np.percentile(dv, [0, 100])   # try [2, 98] to clip outliers
d_clip = np.clip(d, lo, hi)

d_norm = (d_clip - lo) / (hi - lo + 1e-12)

# Optional: make invalid regions stand out instead of hiding detail in black.
d_norm[~valid] = 1.0  # render invalid depth as white (0.0 or a mask also work)

plt.imshow(d_norm, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
plt.axis("off")
plt.imsave("depth_visualization.png", d_norm, cmap="gray", vmin=0, vmax=1)
plt.show()
