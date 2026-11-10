"""Move a cable's support in XY to bring the simulated free tip to a target."""

import argparse
import math
from numbers import Integral
from pathlib import Path
import time

from .run import finish, prepare, report_header, write_report


class TipTargetTask:
    """Small goal task with no RL framework.

    `reset(seed)` returns (observation, info). `step(action)` takes an XY
    vector of norm <= 1, interpreted as support velocity times max_speed_mps.
    The tip is unpinned. Reward is the negative XYZ error in meters; success
    ends the episode, otherwise the step budget truncates it.
    """

    dt_s = 1 / 60
    max_speed_mps = 0.20
    substeps = 32
    iterations = 12
    min_target_radius_m = 0.06
    max_target_radius_m = 0.09

    def __init__(self, *, max_steps=160, tolerance_m=0.012):
        if type(max_steps) is not int or not 1 <= max_steps <= 3600:
            raise ValueError("max_steps must be an integer in [1, 3600]")
        if not math.isfinite(tolerance_m) or not 0 < tolerance_m < self.min_target_radius_m:
            raise ValueError("tolerance_m must be finite, positive and below the 0.06 m minimum target offset")
        import numpy as np
        from .scenes import line, scene_from_positions

        self.max_steps = max_steps
        self.tolerance_m = tolerance_m
        self._rng = np.random.default_rng()
        self._rod = None
        self._done = True
        self._scene = scene_from_positions(
            line([0, 0, 0.70], [0, 0, 0.34], n=25),
            max_steps * self.dt_s, clamp=True, ei=(0.006,), drag=3.0,
        )

    def _reset(self, seed, cuda):
        import numpy as np
        from .scenes import Rod

        self._done = True
        self._rod = Rod(self._scene, *cuda, dt=self.dt_s,
                        substeps=self.substeps, iterations=self.iterations)
        self._root = self._scene.positions[:, 0].copy()
        self._step = 0
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        angle = self._rng.uniform(-np.pi, np.pi)
        radius = self._rng.uniform(self.min_target_radius_m, self.max_target_radius_m)
        self._target = self._scene.positions[0, -1] + np.array(
            [radius * np.cos(angle), radius * np.sin(angle), 0.0], np.float32,
        )
        obs, info = self._observe()
        self._done = False
        return obs, info

    def reset(self, seed=None):
        """Rebuild the physical state; the same explicit seed repeats the target."""
        from .scenes import load_cuda

        validate_seed(seed)
        return self._reset(seed, load_cuda())

    def _observe(self):
        import numpy as np

        p, v, _ = self._rod.sample()
        obs = dict(positions=p[0], velocities=v[0], target=self._target.copy())
        error = float(np.linalg.norm(obs["positions"][-1] - obs["target"]))
        return obs, dict(step=self._step, time_s=self._step * self.dt_s,
                         error_m=error, success=error <= self.tolerance_m)

    def step(self, action):
        import numpy as np

        if self._done:
            raise RuntimeError("call reset() before stepping a new episode")
        try:
            action = np.asarray(action)
            if np.iscomplexobj(action):
                raise ValueError("complex action")
            action = action.astype(float)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("action must be a finite XY vector of norm <= 1") from exc
        if action.shape != (2,) or not np.isfinite(action).all() or np.linalg.norm(action) > 1 + 1e-8:
            raise ValueError("action must be a finite XY vector of norm <= 1")
        velocity = self.max_speed_mps * np.array([[action[0], action[1], 0.0]])
        end = self._root + velocity * self.dt_s
        try:
            self._rod.advance(self._root, velocity, end)
            self._step += 1
            obs, info = self._observe()
        except RuntimeError:
            self._done = True
            raise
        self._root = end
        terminated = info["success"]
        truncated = self._step >= self.max_steps and not terminated
        self._done = terminated or truncated
        return obs, -info["error_m"], terminated, truncated, info


def validate_seed(seed):
    if seed is not None and (
        not isinstance(seed, Integral) or isinstance(seed, bool) or seed < 0
    ):
        raise ValueError("seed must be a nonnegative integer or None")


def feedback_action(observation):
    """Proportional tip-error feedback with tip-velocity damping (a baseline)."""
    import numpy as np

    delta = observation["target"][:2] - observation["positions"][-1, :2]
    action = 9.0 * delta - 1.5 * observation["velocities"][-1, :2]
    action = action.astype(float)
    return action / max(1.0, np.linalg.norm(action))


def run_task(args, task, torch, rs):
    import numpy as np

    settings = dict(
        case="tip_target", policy=args.policy, seed=args.seed, max_steps=task.max_steps,
        tolerance_m=task.tolerance_m, dt_s=task.dt_s, substeps=task.substeps,
        iterations=task.iterations, max_speed_m_per_s=task.max_speed_mps,
        target_radius_range_m=[task.min_target_radius_m, task.max_target_radius_m],
        **task._scene.settings(),
    )
    report = report_header(torch, rs, settings)
    report["limitations"] = [
        "Only XY support velocity is commanded. The target lies at the nominal initial tip height.",
        "Success is sampled at control times; one in-tolerance frame terminates the episode.",
        "One seed and a feedback example are not a robustness or force-accuracy study.",
    ]
    write_report(args.out, report)
    try:
        obs, info = task._reset(args.seed, (torch, rs))
    except (RuntimeError, ValueError) as exc:
        report["failure"] = str(exc)
        write_report(args.out, report)
        print(report["failure"])
        return 1
    times, positions, velocities = [0.0], [obs["positions"]], [obs["velocities"]]
    errors, actions, rewards, step_ms = [info["error_m"]], [], [], []
    terminated = truncated = False
    begin = time.perf_counter()
    while not (terminated or truncated):
        action = feedback_action(obs) if args.policy == "feedback" else np.zeros(2)
        tick = time.perf_counter()
        try:
            obs, reward, terminated, truncated, info = task.step(action)
        except (RuntimeError, ValueError) as exc:
            report["failure"] = str(exc)
            break
        step_ms.append((time.perf_counter() - tick) * 1e3)
        times.append(info["time_s"])
        positions.append(obs["positions"])
        velocities.append(obs["velocities"])
        errors.append(info["error_m"])
        actions.append(action)
        rewards.append(reward)

    report["complete"] = terminated or truncated
    report["metrics"] = dict(
        success=terminated, terminated=terminated, truncated=truncated,
        initial_error_m=errors[0], final_error_m=errors[-1], min_error_m=min(errors),
        target_m=obs["target"].tolist(), steps=len(actions), simulated_seconds=times[-1],
    )
    report["timing"] = dict(
        scope="policy and step loop, including synchronization and readback; excludes setup and I/O",
        step_scope="action validation, transfers, solver and measured observation; no warmup excluded",
        loop_wall_s=time.perf_counter() - begin,
        mean_step_ms=float(np.mean(step_ms)) if step_ms else None,
        p95_step_ms=float(np.percentile(step_ms, 95)) if step_ms else None,
    )
    return finish(args.out, report, t=times, positions=np.asarray(positions)[:, None],
                  velocities=np.asarray(velocities)[:, None], target=obs["target"],
                  error_m=errors, actions=np.asarray(actions).reshape(-1, 2), rewards=rewards,
                  step_ms=step_ms)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="new directory for the episode")
    parser.add_argument("--seed", type=int, default=0, help="target RNG seed (default: 0)")
    parser.add_argument("--policy", choices=("feedback", "zero"), default="feedback")
    parser.add_argument("--max-steps", type=int, default=160)
    parser.add_argument("--tolerance-m", type=float, default=0.012)
    args = parser.parse_args(argv)
    try:
        validate_seed(args.seed)
        task = TipTargetTask(max_steps=args.max_steps, tolerance_m=args.tolerance_m)
    except ImportError as exc:
        parser.error(f"install NumPy to run the task ({exc})")
    except (TypeError, ValueError) as exc:
        parser.error(str(exc))
    return run_task(args, task, *prepare(parser, args.out))


if __name__ == "__main__":
    raise SystemExit(main())
