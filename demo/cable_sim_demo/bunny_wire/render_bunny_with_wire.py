from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import numpy.testing  # Keep numpy.testing bound before Kit mutates import paths.

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))


STAGE_USD = str(THIS_DIR / "data" / "demo_1_b.usdc")
WIRE_USD = str(THIS_DIR / "data" / "yellow_rope_n100.usdc")
NPZ_PATH = str(
    THIS_DIR / "data" / "bunny_fixed_waypoint_rod_4view_n100_E200000_d0p4_sd0p04_cf5_state_1.npz"
)
WIRE_ROOT_PATH = "/World/YellowRopeN100"
BUNNY_PATH = "/World/Bunny"

TARGET_INTENSITY = 300.0
TARGET_EXPOSURE = 0.0
R_ALIGN_BUNNY = np.eye(3, dtype=np.float64)
USE_TEMPORAL_DIRECTOR_CONTINUITY = True
DEBUG_NODE_STRIDE = 1
DEBUG_NODE_RADIUS = 0.004
DEBUG_NODE_ROOT = "/World/DebugRodNodes"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--physics_gpu", type=int, default=0)
    parser.add_argument(
        "--exit_immediately",
        action="store_true",
        help="Exit right after authoring animation instead of keeping the app running.",
    )
    parser.add_argument(
        "--debug_node_stride",
        type=int,
        default=DEBUG_NODE_STRIDE,
        help="Place one red debug sphere every N rod nodes (>=1). Default 1 = every node.",
    )
    parser.add_argument(
        "--debug_node_radius",
        type=float,
        default=DEBUG_NODE_RADIUS,
        help="Radius (meters) for red debug spheres.",
    )
    parser.add_argument(
        "--disable_debug_nodes",
        action="store_true",
        help="Disable red debug node spheres.",
    )
    parser.add_argument(
        "--scene_mode",
        type=str,
        choices=["empty", "stage"],
        default="empty",
        help="`empty`: create a new empty scene and visualize only wire. `stage`: open demo stage.",
    )
    return parser.parse_known_args()[0]


def create_simulation_app(headless: bool, physics_gpu: int):
    try:
        from isaacsim import SimulationApp
    except ImportError:
        from omni.isaac.kit import SimulationApp

    return SimulationApp(
        {
            "headless": bool(headless),
            "physics_gpu": int(physics_gpu),
            "active_gpu": int(physics_gpu),
        }
    )


def print_npz_shapes(data: np.lib.npyio.NpzFile) -> None:
    print("NPZ keys:", data.files)
    for key in data.files:
        arr = data[key]
        print(f"{key:24s} shape={arr.shape}, dtype={arr.dtype}")


def orthonormalize(R: np.ndarray) -> np.ndarray:
    U, _, Vt = np.linalg.svd(R)
    Rn = U @ Vt
    if np.linalg.det(Rn) < 0.0:
        U[:, -1] *= -1.0
        Rn = U @ Vt
    return Rn


def mat3_to_quatd(R: np.ndarray) -> Gf.Quatd:
    R = R.astype(np.float64)
    trace = R[0, 0] + R[1, 1] + R[2, 2]

    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    else:
        if (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
            s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s

    norm = np.sqrt(w * w + x * x + y * y + z * z) + 1.0e-12
    return Gf.Quatd(float(w / norm), float(x / norm), float(y / norm), float(z / norm))


def _transport_frame_from_tangent(t: np.ndarray, prev_n: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    t = t / (np.linalg.norm(t) + 1.0e-12)
    if prev_n is None:
        up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        if abs(np.dot(t, up)) > 0.95:
            up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        n = np.cross(up, t)
        n = n / (np.linalg.norm(n) + 1.0e-12)
    else:
        n = prev_n - t * np.dot(prev_n, t)
        if np.linalg.norm(n) < 1.0e-8:
            up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
            if abs(np.dot(t, up)) > 0.95:
                up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
            n = np.cross(up, t)
        n = n / (np.linalg.norm(n) + 1.0e-12)

    b = np.cross(t, n)
    b = b / (np.linalg.norm(b) + 1.0e-12)
    R = orthonormalize(np.stack([n, t, b], axis=1))
    return R, R[:, 0]


def build_directors_from_nodes(
    rod_nodes: np.ndarray, prev_normals: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build rod directors with shape (3, 3, N_elems) from node positions (3, N_nodes).
    Mapping matches SkeletonRodDriver.update_skeleton():
      D[2] -> tangent column, D[1] -> normal column, D[0] -> binormal column.
    """
    if rod_nodes.ndim != 2 or rod_nodes.shape[0] != 3 or rod_nodes.shape[1] < 2:
        raise ValueError(f"rod_nodes must be (3, N>=2), got {rod_nodes.shape}")

    n_elems = rod_nodes.shape[1] - 1
    rod_dir = np.zeros((3, 3, n_elems), dtype=np.float64)
    out_normals = np.zeros((3, n_elems), dtype=np.float64)
    prev_n = None

    for i in range(n_elems):
        p0 = rod_nodes[:, i]
        p1 = rod_nodes[:, i + 1]
        tangent = p1 - p0
        tangent = tangent / (np.linalg.norm(tangent) + 1.0e-12)

        R, prev_n = _transport_frame_from_tangent(tangent, prev_n)

        # Keep normal/binormal sign temporally consistent across frames.
        if prev_normals is not None and prev_normals.shape == (3, n_elems):
            if float(np.dot(R[:, 0], prev_normals[:, i])) < 0.0:
                R[:, 0] *= -1.0
                R[:, 2] *= -1.0

        D = np.zeros((3, 3), dtype=np.float64)
        D[2, :] = R[:, 1]  # tangent
        D[1, :] = R[:, 0]  # normal
        D[0, :] = R[:, 2]  # binormal
        rod_dir[:, :, i] = D
        out_normals[:, i] = R[:, 0]

    return rod_dir, out_normals


def canonicalize_rod_directors(
    rod_director: np.ndarray, n_frames: int, n_elems: int
) -> np.ndarray:
    """
    Convert rod directors to shape (T, 3, 3, N_elems).
    """
    if rod_director.ndim != 4:
        raise RuntimeError(
            f"rod_director must be 4D, got shape {rod_director.shape}"
        )
    if rod_director.shape[0] != n_frames:
        raise RuntimeError(
            f"rod_director time length mismatch: expected {n_frames}, got {rod_director.shape[0]}"
        )

    if rod_director.shape == (n_frames, 3, 3, n_elems):
        out = rod_director
    elif rod_director.shape == (n_frames, n_elems, 3, 3):
        out = np.transpose(rod_director, (0, 2, 3, 1))
    else:
        raise RuntimeError(
            "Unsupported rod_director shape. "
            f"Expected (T,3,3,N) or (T,N,3,3), got {rod_director.shape}"
        )

    if not np.all(np.isfinite(out)):
        raise RuntimeError("rod_director contains NaN/Inf values")
    return np.asarray(out, dtype=np.float64)


def create_debug_node_markers(
    stage, n_nodes: int, node_stride: int, node_radius: float
) -> list[tuple[int, object]]:
    node_stride = max(1, int(node_stride))
    node_radius = float(max(1.0e-4, node_radius))

    UsdGeom.Scope.Define(stage, DEBUG_NODE_ROOT)

    indices = list(range(0, n_nodes, node_stride))
    if indices[-1] != n_nodes - 1:
        indices.append(n_nodes - 1)

    marker_ops: list[tuple[int, object]] = []
    for idx in indices:
        marker_path = f"{DEBUG_NODE_ROOT}/node_{idx:03d}"
        sphere = UsdGeom.Sphere.Define(stage, marker_path)
        sphere.CreateRadiusAttr(node_radius)
        sphere.CreateDisplayColorAttr().Set([Gf.Vec3f(1.0, 0.0, 0.0)])
        xformable = UsdGeom.Xformable(sphere.GetPrim())
        xformable.ClearXformOpOrder()
        marker_ops.append((idx, xformable.AddTranslateOp()))
    print(f"Debug red node markers: {len(marker_ops)} (stride={node_stride})")
    return marker_ops


def update_debug_node_markers(marker_ops, rod_nodes: np.ndarray, tc) -> None:
    for idx, op in marker_ops:
        p = rod_nodes[:, idx]
        op.Set(Gf.Vec3f(float(p[0]), float(p[1]), float(p[2])), tc)


def get_or_create_bunny_ops(stage):
    prim = stage.GetPrimAtPath(BUNNY_PATH)
    if not prim.IsValid():
        prim = UsdGeom.Xform.Define(stage, BUNNY_PATH).GetPrim()

    xformable = UsdGeom.Xformable(prim)
    translate_op = None
    orient_op = None
    for op in xformable.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate and translate_op is None:
            translate_op = op
        elif op.GetOpType() == UsdGeom.XformOp.TypeOrient and orient_op is None:
            orient_op = op

    if translate_op is None or orient_op is None:
        xformable.ClearXformOpOrder()
        translate_op = xformable.AddTranslateOp()
        orient_op = xformable.AddOrientOp()

    return translate_op, orient_op


def _setup_stage(args):
    ctx = omni.usd.get_context()
    if args.scene_mode == "empty":
        ctx.new_stage()
        stage = ctx.get_stage()
        if stage is None:
            raise RuntimeError("Failed to create empty stage.")
        world = stage.DefinePrim("/World", "Xform")
        if not world.IsValid():
            raise RuntimeError("Failed to define /World in empty stage.")
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        dome = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
        dome.CreateIntensityAttr(TARGET_INTENSITY)
        dome.CreateExposureAttr(TARGET_EXPOSURE)
        print("Scene mode: empty (wire only).")
        return stage

    ctx.open_stage(STAGE_USD)
    stage = ctx.get_stage()
    if stage is None:
        raise RuntimeError(f"Stage failed to open: {STAGE_USD}")
    for prim in stage.Traverse():
        if prim.IsA(UsdLux.DomeLight):
            dome = UsdLux.DomeLight(prim)
            dome.CreateIntensityAttr(TARGET_INTENSITY)
            dome.CreateExposureAttr(TARGET_EXPOSURE)
    print(f"Scene mode: stage ({STAGE_USD}).")
    return stage


def main(args) -> None:
    from rod_skel_driver import SkeletonRodDriver

    stage = _setup_stage(args)

    if not Path(WIRE_USD).is_file():
        raise FileNotFoundError(f"Wire USD not found: {WIRE_USD}")
    if not Path(NPZ_PATH).is_file():
        raise FileNotFoundError(f"NPZ not found: {NPZ_PATH}")

    wire_driver = SkeletonRodDriver(stage, skeleton_path=WIRE_ROOT_PATH, assume_chain=True)
    wire_driver.load_asset(WIRE_USD)

    translate_op = None
    orient_op = None
    if args.scene_mode == "stage":
        translate_op, orient_op = get_or_create_bunny_ops(stage)

    data = np.load(NPZ_PATH)
    print_npz_shapes(data)

    rod_pos = np.asarray(data["rod_position"], dtype=np.float64)    # (T, 3, N_nodes)
    time_arr = np.asarray(data["time"], dtype=np.float64) if "time" in data.files else None
    mesh_pos = (
        np.asarray(data["mesh_position"], dtype=np.float64)
        if "mesh_position" in data.files
        else None
    )
    mesh_dir = (
        np.asarray(data["mesh_director"], dtype=np.float64)
        if "mesh_director" in data.files
        else None
    )

    if rod_pos.ndim != 3 or rod_pos.shape[1] != 3:
        raise RuntimeError(f"rod_position must be (T,3,N), got {rod_pos.shape}")

    if time_arr is None:
        time_arr = np.arange(rod_pos.shape[0], dtype=np.float64)
    elif time_arr.shape[0] != rod_pos.shape[0]:
        raise RuntimeError("time length mismatch with rod_position")

    if args.scene_mode == "stage":
        if mesh_pos is None or mesh_dir is None:
            raise RuntimeError("Stage mode requires mesh_position and mesh_director in NPZ.")
        if mesh_pos.shape[0] != rod_pos.shape[0] or mesh_dir.shape[0] != rod_pos.shape[0]:
            raise RuntimeError("mesh_position/mesh_director time length mismatch with rod_position")

    n_frames = rod_pos.shape[0]
    n_elems = rod_pos.shape[2] - 1
    n_nodes = rod_pos.shape[2]

    if "rod_director" not in data.files:
        raise RuntimeError("NPZ missing required key `rod_director` for wire orientation")
    rod_dir = canonicalize_rod_directors(
        np.asarray(data["rod_director"], dtype=np.float64), n_frames=n_frames, n_elems=n_elems
    )

    if wire_driver.num_joints != n_elems:
        raise RuntimeError(
            f"Wire skeleton joints ({wire_driver.num_joints}) != rod elements ({n_elems})."
        )

    if n_frames < 2:
        raise RuntimeError("Need at least 2 frames in NPZ")
    dt = float(np.median(np.diff(time_arr)))
    fps = 1.0 / max(dt, 1.0e-9)

    start_frame = 0
    end_frame = n_frames - 1
    print(f"Authoring frames [{start_frame}, {end_frame}] at fps={fps:.3f}")

    marker_ops = []
    if not args.disable_debug_nodes:
        marker_ops = create_debug_node_markers(
            stage=stage,
            n_nodes=n_nodes,
            node_stride=args.debug_node_stride,
            node_radius=args.debug_node_radius,
        )

    for frame in range(start_frame, end_frame + 1):
        tc = Usd.TimeCode(frame)

        nodes = rod_pos[frame]
        directors = rod_dir[frame]
        wire_driver.update_skeleton(nodes, directors, tc)

        if translate_op is not None and orient_op is not None and mesh_pos is not None and mesh_dir is not None:
            bp = mesh_pos[frame]
            translate_op.Set(Gf.Vec3f(float(bp[0]), float(bp[1]), float(bp[2])), tc)

            Rb = orthonormalize(mesh_dir[frame])
            Rb = R_ALIGN_BUNNY @ Rb
            qb = mat3_to_quatd(Rb)
            orient_op.Set(Gf.Quatf(qb), tc)

        if marker_ops:
            update_debug_node_markers(marker_ops, nodes, tc)

    try:
        stage.SetTimeCodesPerSecond(fps)
        stage.SetFramesPerSecond(fps)
    except Exception:
        pass

    timeline = omni.timeline.get_timeline_interface()
    timeline.set_start_time(start_frame / fps)
    timeline.set_end_time(end_frame / fps)
    timeline.set_current_time(0.0)
    timeline.play()

    print("Animation authored and playing.")


if __name__ == "__main__":
    args = parse_args()
    simulation_app = create_simulation_app(args.headless, args.physics_gpu)
    try:
        import omni.timeline
        import omni.usd
        from pxr import Gf, Usd, UsdGeom, UsdLux

        globals().update(
            {
                "omni": omni,
                "Gf": Gf,
                "Usd": Usd,
                "UsdGeom": UsdGeom,
                "UsdLux": UsdLux,
            }
        )

        main(args)

        if not args.exit_immediately and not args.headless:
            while simulation_app.is_running():
                simulation_app.update()
    finally:
        simulation_app.close()
