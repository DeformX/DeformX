from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
for _extra in (_REPO_ROOT / "RL_Demo", _REPO_ROOT / "PyElastica-Mesh"):
    if _extra.is_dir() and str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))
import deformx_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize full swing trajectory on the wire rig (no downsampling)."
    )
    parser.add_argument("--headless", action="store_true", help="Run Isaac Sim in headless mode.")
    parser.add_argument("--max_steps", type=int, default=0, help="If > 0, run this many sim-app update steps then exit.")
    parser.add_argument(
        "--stage_usd",
        type=str,
        default=deformx_paths.wire_usd(),
        help="USD stage containing the wire skeleton.",
    )
    parser.add_argument(
        "--npz_path",
        type=str,
        default="",
        help="Input trajectory npz file. If omitted, a tiny synthetic trajectory is used.",
    )
    parser.add_argument(
        "--skeleton_path",
        type=str,
        default="/root/Armature_003/Armature_004",
        help="Skeleton prim path in stage.",
    )
    return parser.parse_args()


ARGS = parse_args()

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": ARGS.headless})

from pxr import Usd, UsdLux, UsdSkel  # noqa: E402
import omni.usd  # noqa: E402
import omni.timeline  # noqa: E402

from RL_Demo.tools.rod_skel_driver_sim import SkeletonRodDriver  # noqa: E402


TARGET_INTENSITY = 300.0
TARGET_EXPOSURE = 0.0
ASSUME_CHAIN_PARENT = True


def print_npz_shapes(data: np.lib.npyio.NpzFile) -> None:
    print("NPZ keys:", data.files)
    for k in data.files:
        arr = data[k]
        print(f"{k:24s} shape={arr.shape}, dtype={arr.dtype}")


def orthonormalize(R: np.ndarray) -> np.ndarray:
    U, _, Vt = np.linalg.svd(R)
    Rn = U @ Vt
    if np.linalg.det(Rn) < 0:
        U[:, -1] *= -1
        Rn = U @ Vt
    return Rn


def build_rotation_from_tangent(t: np.ndarray, prev_n: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Same logic as animate_skel_seq_replay_copy.py.
    Columns in returned R are [normal, tangent, binormal].
    """
    t = t / (np.linalg.norm(t) + 1e-12)

    if prev_n is None:
        up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        if abs(np.dot(t, up)) > 0.95:
            up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        n = np.cross(up, t)
        n = n / (np.linalg.norm(n) + 1e-12)
    else:
        n = prev_n - t * np.dot(prev_n, t)
        if np.linalg.norm(n) < 1e-6:
            up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
            n = np.cross(up, t)
        n = n / (np.linalg.norm(n) + 1e-12)

    b = np.cross(t, n)
    R = np.stack([n, t, b], axis=1)
    return orthonormalize(R), n


def build_director_per_node(pos_3xn: np.ndarray) -> np.ndarray:
    """
    Build one director frame per node so it can drive a node-aligned skeleton rig.
    Output shape: (3, 3, N_nodes), with director rows:
      row 0 = binormal, row 1 = normal, row 2 = tangent
    """
    n_nodes = pos_3xn.shape[1]
    if n_nodes < 2:
        raise ValueError("Need at least 2 nodes to build tangents.")

    dirs = np.zeros((3, 3, n_nodes), dtype=np.float64)
    prev_n = None
    prev_t = None

    for i in range(n_nodes):
        if i < n_nodes - 1:
            tangent = pos_3xn[:, i + 1] - pos_3xn[:, i]
        else:
            tangent = pos_3xn[:, -1] - pos_3xn[:, -2]

        tnorm = np.linalg.norm(tangent)
        if tnorm < 1e-12:
            tangent = prev_t if prev_t is not None else np.array([1.0, 0.0, 0.0], dtype=np.float64)
        else:
            tangent = tangent / tnorm

        Rw, prev_n = build_rotation_from_tangent(tangent, prev_n)
        prev_t = tangent

        dirs[0, :, i] = Rw[:, 2]  # binormal
        dirs[1, :, i] = Rw[:, 0]  # normal
        dirs[2, :, i] = Rw[:, 1]  # tangent

    return dirs


def load_or_make_trajectory(npz_path: str, rig_joints: int) -> tuple[np.ndarray, np.ndarray | None]:
    if npz_path:
        with np.load(npz_path) as data:
            print_npz_shapes(data)
            if "position" not in data or "time" not in data:
                raise RuntimeError("Expected npz keys: 'position' and 'time'.")
            source_pos = data["position"]  # (T, 3, N)
            source_dir = data["director"] if "director" in data else None
        return source_pos, source_dir

    # Smoke-test fallback: drive the rig with two simple frames so the script can
    # validate skeleton loading without requiring private trajectory data.
    x = np.linspace(0.0, 1.0, rig_joints, dtype=np.float64)
    source_pos = np.zeros((2, 3, rig_joints), dtype=np.float64)
    source_pos[:, 0, :] = x[None, :]
    source_pos[1, 2, :] = 0.03 * np.sin(np.linspace(0.0, np.pi, rig_joints))
    print(f"[Info] No --npz_path provided; using synthetic trajectory with {rig_joints} joints.")
    return source_pos, None


def main() -> None:
    ctx = omni.usd.get_context()
    ctx.open_stage(ARGS.stage_usd)
    for _ in range(60):
        simulation_app.update()

    stage = ctx.get_stage()
    if stage is None:
        raise RuntimeError("Stage failed to open.")

    for p in stage.Traverse():
        if p.IsA(UsdLux.DomeLight):
            dome = UsdLux.DomeLight(p)
            dome.CreateIntensityAttr(TARGET_INTENSITY)
            dome.CreateExposureAttr(TARGET_EXPOSURE)

    skeleton_prim = stage.GetPrimAtPath(ARGS.skeleton_path)
    if not skeleton_prim.IsValid():
        # Fallback: auto-find first UsdSkel.Skeleton in stage.
        found = None
        for p in stage.Traverse():
            if p.IsA(UsdSkel.Skeleton):
                found = p
                break
        if found is None:
            raise RuntimeError(
                f"Skeleton prim not found: {ARGS.skeleton_path}, and no UsdSkel.Skeleton found in stage."
            )
        ARGS.skeleton_path = found.GetPath().pathString
        skeleton_prim = found
        print(f"[Info] Using auto-detected skeleton path: {ARGS.skeleton_path}")

    driver = SkeletonRodDriver(stage, ARGS.skeleton_path, assume_chain=ASSUME_CHAIN_PARENT)
    driver.skel_prim = skeleton_prim
    driver.skeleton_path = ARGS.skeleton_path
    driver._setup_animation()

    rig_joints = driver.num_joints
    source_pos, source_dir = load_or_make_trajectory(ARGS.npz_path, rig_joints)
    T, _, source_nodes = source_pos.shape
    if source_nodes < 2:
        raise RuntimeError(f"Invalid source node count: {source_nodes}")

    source_elems = source_nodes - 1
    has_src_dir = source_dir is not None and source_dir.ndim == 4 and source_dir.shape[-1] == source_elems

    if rig_joints == source_nodes:
        mode = "node-mode"
    elif rig_joints == source_elems:
        mode = "element-mode"
    else:
        raise RuntimeError(
            f"Rig joint count ({rig_joints}) does not match source nodes ({source_nodes}) "
            f"or source elements ({source_elems})."
        )

    print(f"Source nodes: {source_nodes}, source elements: {source_elems}, rig_joints: {rig_joints}, frames: {T}")
    print(f"Driving mode: {mode}")

    for frame in range(T):
        tc = Usd.TimeCode(frame)
        if mode == "node-mode":
            p = source_pos[frame].astype(np.float64)  # (3, source_nodes)
            if has_src_dir and source_dir.shape[-1] == rig_joints:
                d = source_dir[frame].astype(np.float64)
            else:
                d = build_director_per_node(p)
        else:
            # Use element centers for positions so joint count matches source elements (40).
            p0 = source_pos[frame, :, :-1]
            p1 = source_pos[frame, :, 1:]
            p = (0.5 * (p0 + p1)).astype(np.float64)  # (3, source_elems)
            if has_src_dir:
                d = source_dir[frame].astype(np.float64)  # (3,3,source_elems)
            else:
                d = build_director_per_node(p)

        driver.update_skeleton(p, d, tc)


    fps = 100.0

    try:
        stage.SetTimeCodesPerSecond(fps)
        stage.SetFramesPerSecond(fps)
    except Exception:
        pass

    timeline = omni.timeline.get_timeline_interface()
    timeline.set_start_time(0.0)
    timeline.set_end_time((T - 1) / fps if T > 1 else 1.0 / fps)
    timeline.set_current_time(0.0)
    timeline.play()

    print(f"Animation authored. fps={fps:.3f}")
    if ARGS.max_steps > 0:
        for _ in range(ARGS.max_steps):
            simulation_app.update()
    else:
        while simulation_app.is_running():
            simulation_app.update()

    simulation_app.close()


if __name__ == "__main__":
    main()
