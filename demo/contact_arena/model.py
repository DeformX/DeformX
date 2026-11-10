"""SI scene and diagnostics, independent of CUDA."""
import numpy as np
from demo.batch_arena.model import diagnostics, timing

DT = 1 / 120
VERTICES = 65
RADIUS = .012
GUIDE_RADIUS = .18
EIS = np.array([.001, .002, .004, .008], np.float32)
COLORS = [[.08,.85,.75],[.15,.52,1.],[1.,.56,.12],[.94,.18,.42]]


def build(count):
    if count < 1:
        raise ValueError('positive world count required')
    u = np.linspace(0, 1, VERTICES)
    p = np.zeros((count, VERTICES, 3), np.float32)
    p[:,:,0] = 1.2*u-.6
    p[:,:,2] = .65 + .28*np.sin(np.pi*u)
    pins = np.zeros((count, VERTICES), bool); pins[:,[0,-1]] = True
    return dict(positions=p, pins=pins, family=np.arange(count)%4,
                phase=(np.arange(count)//4%8)*np.pi/4)


def guide_at(scene, t):
    ramp = np.clip(t, 0, 1); ramp = ramp*ramp*(3-2*ramp)
    out = np.zeros((len(scene['family']), 3), np.float32)
    out[:,0] = .16*ramp*np.sin(2*np.pi*.35*t+scene['phase'])
    out[:,2] = .36
    return out


def clearance(positions, guides):
    d = positions-guides[:,None,:]
    return np.linalg.norm(d[:,:,[0,2]],axis=-1)-GUIDE_RADIUS-RADIUS


def contact_summary(counts, steps):
    counts = np.asarray(counts)
    return dict(active_worlds=int(np.count_nonzero(counts)), total_worlds=len(counts),
                every_world_contacted=bool(np.all(counts>0)),
                minimum_contact_fraction=float(counts.min()/steps) if steps else 0.,
                maximum_contact_fraction=float(counts.max()/steps) if steps else 0.)
