"""Wall-clock performance statistics, separate from presentation frame rate."""

import numpy as np


def summarize_timing(samples_ms, dt, worlds=1, warmup=15):
    values = np.asarray(samples_ms, dtype=float)
    if (
        values.ndim != 1
        or len(values) <= warmup
        or warmup < 0
        or not np.isfinite(values).all()
        or (values <= 0).any()
        or not np.isfinite(dt)
        or dt <= 0
        or worlds < 1
    ):
        raise ValueError(
            "Positive finite timings and at least one post-warmup sample required"
        )
    a = values[warmup:]
    mean = float(a.mean())
    return dict(
        samples=len(a),
        warmup_frames=warmup,
        mean_ms=mean,
        median_ms=float(np.median(a)),
        p95_ms=float(np.percentile(a, 95)),
        per_world_rtf=float(dt * 1000 / mean),
        world_steps_per_second=float(worlds * 1000 / mean),
        worlds=int(worlds),
        control_dt_s=float(dt),
        scope="drive commands + CUDA solve + synchronization; excludes diagnostics, recording and rendering",
    )
