#!/usr/bin/env python3
"""
Convert NPZ from:
    pos:      (T, N, 3, K)
    director: (T, N, 3, 3, D)
    time:     (T,)

To the format expected by the original test_drop_multi_diff_rods.py:
    positions: (T, W, 3, N)   ← same shape as pos, just renamed
    time:      (T,)           ← unchanged
"""

import numpy as np
import sys
import os

# ============ CONFIG ============
INPUT_NPZ  = "/home/robot/Workspace/CosseratX/Dataset_generator_datacenter/data/npz_file/easy.npz"
OUTPUT_NPZ = "/home/robot/Workspace/CosseratX/Dataset_generator_datacenter/data/npz_file/easy_converted.npz"
# ================================

data = np.load(INPUT_NPZ)

print("Input NPZ keys:", data.files)
for k in data.files:
    print(f"  {k}: shape={data[k].shape}, dtype={data[k].dtype}")

pos      = data["pos"]       # (T, N, 3, K)
time_arr = data["time"]      # (T,)

# pos shape (T, N, 3, K) is already identical to positions (T, W, 3, N_nodes)
# Just rename the key
positions = pos

print(f"\nConverted:")
print(f"  positions: {positions.shape}  (was pos)")
print(f"  time:      {time_arr.shape}")

np.savez(OUTPUT_NPZ, positions=positions, time=time_arr)
print(f"\nSaved to: {OUTPUT_NPZ}")
