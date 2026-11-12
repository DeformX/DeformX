"""One continuous camera path for the rack insertion replay.

Reads the recorded 30 Hz trajectory and writes camera.npz with an eye and a target per frame.
The camera stays on one side of the plug for the whole trial and only changes its distance and
height, so there are no cuts: medium shot of the approach, push in for the grasp, pull back while
the arm carries the cable to the rack, push in to the port for the insertion, ease back at the end.
"""
import sys
import numpy as np

tr = np.load(sys.argv[1], allow_pickle=False)
t, plug, tcp = tr['t'], tr['plug_pos'].astype(float), tr['tcp'].astype(float)

# time [s]: (eye offset from the focus point [m], weight of the hand in the focus point)
KEYS = [
    (0.0,  (0.66, -0.92, 0.40), 0.45),   # approach: hand above the plug, both in frame
    (2.6,  (0.30, -0.42, 0.20), 0.25),   # hand arrives
    (3.4,  (0.20, -0.29, 0.13), 0.00),   # fingers close: detail of the grasp
    (5.2,  (0.24, -0.35, 0.16), 0.00),   # lift and hold
    (7.4,  (0.44, -0.66, 0.27), 0.15),   # carry: arm, trailing cable and the rack ahead
    (9.4,  (0.36, -0.56, 0.20), 0.10),   # swing toward the panel
    (11.2, (0.21, -0.31, 0.10), 0.00),   # arrive at the port
    (13.0, (0.150, -0.220, 0.065), 0.00),  # insertion close-up, port numbers readable
    (16.8, (0.165, -0.215, 0.070), 0.00),  # slow drift while the plug seats
    (18.6, (0.25, -0.38, 0.13), 0.00),   # fingers open
    (20.1, (0.36, -0.62, 0.14), 0.00),   # hand withdraws, plug stays seated
]


def smooth(x, seconds):
    """Gaussian low-pass along time with the end values held."""
    sigma = seconds * 30
    half = int(4 * sigma)
    kernel = np.exp(-0.5 * (np.arange(-half, half + 1) / sigma) ** 2)
    kernel /= kernel.sum()
    x = np.asarray(x, float).reshape(len(x), -1)
    padded = np.pad(x, ((half, half), (0, 0)), mode='edge')
    return np.stack([np.convolve(padded[:, k], kernel, mode='valid') for k in range(x.shape[1])], axis=1)


DROP = {18.6: 0.03, 20.1: 0.09}   # lower the aim at the end to show the hanging loop
kt = np.array([k[0] for k in KEYS])
linear = lambda values: np.stack([np.interp(t, kt, values[:, k]) for k in range(values.shape[1])], axis=1)
# straight segments between the keys, then a low-pass: no corner and no stop at a key
offset = smooth(linear(np.array([k[1] for k in KEYS])), 0.6)
weight = smooth(linear(np.array([[k[2]] for k in KEYS])), 0.6)
focus = smooth(plug + weight * (tcp - plug), 0.35)
focus = focus - np.c_[0 * t, 0 * t, smooth(np.interp(t, kt, [DROP.get(x, 0.0) for x in kt])[:, None], 0.6)[:, 0]]
eye = focus + offset
target = focus
np.savez(sys.argv[2], eye=eye, target=target, t=t)

speed = np.linalg.norm(np.diff(eye, axis=0), axis=1) * 30
accel = np.linalg.norm(np.diff(eye, 2, axis=0), axis=1) * 900
dist = np.linalg.norm(eye - target, axis=1)
print(f'frames {len(t)}; eye speed max {speed.max():.3f} m/s; accel max {accel.max():.3f} m/s^2; '
      f'distance {dist.min():.3f}..{dist.max():.3f} m')
for i in range(0, len(t), 60):
    print(f't={t[i]:5.2f} {tr["phase"][i]:9s} eye={np.round(eye[i], 3)} target={np.round(target[i], 3)} dist={dist[i]:.2f}')
