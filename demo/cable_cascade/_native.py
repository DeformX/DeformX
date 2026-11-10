"""Isaac Sim capture worker supervised by render.py."""

import argparse
import hashlib
from importlib.metadata import version
from itertools import product
from pathlib import Path
import traceback

from .audit import load_trajectory
from .model import DT, FPS, RADIUS, SPHERE_CENTER, SPHERE_COLOR, SPHERE_RADIUS, palette, sha, write_json

FOCAL_LENGTH = 22.
HORIZONTAL_APERTURE = 20.955


def camera_path(positions, times, width, height):
    import numpy as np

    back = np.array([.34, -.89, .37])
    back /= np.linalg.norm(back)
    right = np.cross([0, 0, 1], back)
    right /= np.linalg.norm(right)
    basis = np.column_stack([right, np.cross(back, right), back])
    sphere = np.array(list(product((-SPHERE_RADIUS, SPHERE_RADIUS), repeat=3))) + SPHERE_CENTER
    cloud = np.concatenate([
        positions.reshape(len(times), -1, 3),
        np.broadcast_to(sphere, (len(times), len(sphere), 3)),
    ], axis=1) @ basis
    tangent = HORIZONTAL_APERTURE / (2 * FOCAL_LENGTH) * np.array([1., height / width])

    def fit(selection, fill):
        points = cloud[selection].reshape(-1, 3)
        center = (points.min(axis=0) + points.max(axis=0)) / 2
        rel = points - center
        distance = rel[:, 2] + np.max((np.abs(rel[:, :2]) + .014) / (tangent * fill), axis=1)
        return center, float(distance.max())

    opening, wide = fit(times <= .56, .93)
    landing, close = fit(times >= min(.72, times[-1]), .91)
    blend = np.clip((times - .22) / (.88 - .22), 0., 1.)
    blend = blend**2 * (3. - 2. * blend)
    centers = (1. - blend[:, None]) * opening + blend[:, None] * landing
    distances = (1. - blend) * wide + blend * close
    rel = cloud - centers[:, None, :]
    needed = rel[..., 2] + np.max((np.abs(rel[..., :2]) + RADIUS) / (.98 * tangent), axis=-1)
    distances = np.maximum(distances, needed.max(axis=1))
    depth = distances[:, None] - rel[..., 2]
    max_ndc = np.max((np.abs(rel[..., :2]) + RADIUS) / (depth[..., None] * tangent), axis=(0, 1))
    targets = centers @ basis.T
    eyes = targets + distances[:, None] * back
    return eyes, targets, max_ndc


def capture(args, run, times, positions, report):
    import numpy as np
    from PIL import Image
    from isaacsim.core.api import World
    from isaacsim.core.utils.stage import get_current_stage
    from isaacsim.sensors.camera import Camera
    import omni.replicator.core as rep
    import omni.timeline
    import omni.usd
    from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdShade, Vt

    world = World(physics_dt=DT, rendering_dt=1 / FPS, stage_units_in_meters=1., backend="numpy", device="cpu")
    camera = None
    try:
        stage = get_current_stage()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        materials = {}

        def bind(obj, color, metal=0., roughness=.32, emission=0.):
            key = (tuple(color), metal, roughness, emission)
            if key not in materials:
                path = f"/World/Materials/M{len(materials)}"
                material = UsdShade.Material.Define(stage, path)
                shader = UsdShade.Shader.Define(stage, path + "/Shader")
                shader.CreateIdAttr("UsdPreviewSurface")
                shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
                shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
                shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metal)
                shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(
                    Gf.Vec3f(*(c * emission for c in color))
                )
                material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
                materials[key] = material
            UsdShade.MaterialBindingAPI.Apply(obj.GetPrim()).Bind(materials[key])

        nr, nv = positions.shape[1:3]
        colors = palette(nr)

        def instancer(path, shape, elements_per_rod):
            instances = UsdGeom.PointInstancer.Define(stage, path)
            targets = []
            for i, color in enumerate(colors):
                child = f"{path}/P{i}"
                proto = shape.Define(stage, child)
                proto.CreateRadiusAttr(RADIUS)
                if shape is UsdGeom.Cylinder:
                    proto.CreateHeightAttr(1.)
                    proto.CreateAxisAttr(UsdGeom.Tokens.z)
                bind(proto, color, emission=.35)
                targets.append(Sdf.Path(child))
            instances.CreatePrototypesRel().SetTargets(targets)
            instances.CreateProtoIndicesAttr().Set(np.repeat(np.arange(nr), elements_per_rod).tolist())
            return instances

        def set_xyz(instances, xyz):
            instances.CreatePositionsAttr().Set(Vt.Vec3fArray.FromNumpy(np.ascontiguousarray(xyz, np.float32)))

        cables = instancer("/World/Cable", UsdGeom.Cylinder, nv - 1)
        nodes = instancer("/World/Nodes", UsdGeom.Sphere, nv)

        def set_rods(p):
            a = p[:, :-1].reshape(-1, 3)
            b = p[:, 1:].reshape(-1, 3)
            delta = b - a
            lengths = np.linalg.norm(delta, axis=-1)
            scales = np.ones_like(delta)
            scales[:, 2] = lengths
            cables.CreateScalesAttr().Set(Vt.Vec3fArray.FromNumpy(scales))
            orientations = []
            for direction in delta / lengths[:, None]:
                q = Gf.Rotation(Gf.Vec3d(0, 0, 1), Gf.Vec3d(*map(float, direction))).GetQuat()
                orientations.append(Gf.Quath(q.GetReal(), Gf.Vec3h(*q.GetImaginary())))
            cables.CreateOrientationsAttr().Set(Vt.QuathArray(orientations))
            set_xyz(cables, (a + b) / 2)
            set_xyz(nodes, p.reshape(-1, 3))

        floor = UsdGeom.Cube.Define(stage, "/World/Floor")
        floor.CreateSizeAttr(1.)
        xf = UsdGeom.Xformable(floor.GetPrim())
        xf.AddTranslateOp().Set(Gf.Vec3d(0, 0, -.08))
        xf.AddScaleOp().Set(Gf.Vec3f(100, 100, .16))
        bind(floor, (.018, .026, .043), roughness=.95)
        sphere = UsdGeom.Sphere.Define(stage, "/World/Sphere")
        sphere.CreateRadiusAttr(SPHERE_RADIUS)
        UsdGeom.Xformable(sphere.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(*SPHERE_CENTER))
        bind(sphere, SPHERE_COLOR, metal=.5)

        dome = UsdLux.DomeLight.Define(stage, "/World/Light/Dome")
        dome.CreateIntensityAttr(300)
        dome.CreateColorAttr(Gf.Vec3f(.73, .83, 1.))
        key = UsdLux.DistantLight.Define(stage, "/World/Light/Key")
        key.CreateIntensityAttr(1600)
        key.CreateAngleAttr(8)
        UsdGeom.Xformable(key.GetPrim()).AddRotateXYZOp().Set(Gf.Vec3f(-40, -25, 15))
        fill = UsdLux.DistantLight.Define(stage, "/World/Light/Fill")
        fill.CreateIntensityAttr(700)
        fill.CreateColorAttr(Gf.Vec3f(.3, .75, 1.))
        UsdGeom.Xformable(fill.GetPrim()).AddRotateXYZOp().Set(Gf.Vec3f(20, 50, -60))

        camera = Camera(prim_path="/World/Camera", frequency=-1, resolution=(args.width, args.height),
                        annotator_device="cpu")
        camera.prim.GetAttribute("focalLength").Set(FOCAL_LENGTH)
        camera.prim.GetAttribute("horizontalAperture").Set(HORIZONTAL_APERTURE)
        camera.prim.GetAttribute("verticalAperture").Set(HORIZONTAL_APERTURE * args.height / args.width)
        camera.prim.GetAttribute("clippingRange").Set(Gf.Vec2f(.01, 150.))

        def look(eye, target):
            view = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*target), Gf.Vec3d(0, 0, 1))
            q = view.GetInverse().ExtractRotationQuat()
            camera.set_world_pose(
                position=np.array(eye), orientation=np.array([q.GetReal(), *q.GetImaginary()]), camera_axes="usd"
            )

        eyes, targets, max_ndc = camera_path(positions, times, args.width, args.height)
        look(eyes[0], targets[0])
        set_rods(positions[0])
        world.reset()
        camera.initialize(attach_rgb_annotator=True)
        rep.orchestrator.set_capture_on_play(False)
        timeline = omni.timeline.get_timeline_interface()
        reset_history = getattr(omni.usd.get_context(), "reset_renderer_accumulation", None)
        max_clock_error = 0.

        def tick():
            nonlocal max_clock_error
            before = float(world.current_time), float(timeline.get_current_time()), world.current_time_step_index
            rep.orchestrator.step(rt_subframes=4, pause_timeline=False, delta_time=0.)
            after = float(world.current_time), float(timeline.get_current_time()), world.current_time_step_index
            error = max(abs(after[0] - before[0]), abs(after[1] - before[1]))
            max_clock_error = max(max_clock_error, error)
            if error or after[2] != before[2]:
                raise RuntimeError("native capture advanced a simulation clock")

        (args.out / "frames").mkdir()
        rows = []
        for i, (t, p, eye, target) in enumerate(zip(times, positions, eyes, targets, strict=True)):
            look(eye, target)
            set_rods(p)
            if reset_history is not None:
                reset_history()
            if i == 0 or reset_history is None:
                for _ in range(16):
                    tick()
            tick()
            rgba = np.asarray(camera.get_rgba(device="cpu"))
            if rgba.shape != (args.height, args.width, 4) or rgba.dtype != np.uint8:
                raise RuntimeError(f"invalid native frame {i}")
            rgb = rgba[:, :, :3]
            if rgb.std() < 2:
                raise RuntimeError(f"empty or uniform native frame {i}")
            relative = f"frames/{i:06d}.png"
            image_path = args.out / relative
            Image.fromarray(rgb).save(image_path, compress_level=1)
            rows.append(dict(
                file=relative, trajectory_index=i, source_time_s=float(t),
                state_sha256=hashlib.sha256(p.tobytes()).hexdigest(), png_sha256=sha(image_path),
            ))
            if i % 60 == 0:
                print(f"Capture {i + 1}/{len(times)}", flush=True)

        write_json(args.out / "frame-map.json", rows)
        report.update(
            frames=len(rows), trajectory_sha256=run["trajectory_sha256"],
            visual_radius_m=RADIUS, renderer="RaytracedLighting", anti_aliasing="FXAA", render_subframes=4,
            temporal_history_reset=reset_history is not None, clock_delta_max_s=max_clock_error,
            camera=dict(
                first_eye_m=eyes[0].tolist(), first_target_m=targets[0].tolist(),
                final_eye_m=eyes[-1].tolist(), final_target_m=targets[-1].tolist(),
                focal_length_mm=FOCAL_LENGTH, max_absolute_ndc_xy=max_ndc.tolist(),
            ),
        )
    finally:
        if camera is not None:
            camera.pause()
            camera.destroy()
        world.stop()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    args = parser.parse_args()
    path = args.out / "native-report.json"
    report = {"complete": False, "app_close_invoked": False}
    app = None
    try:
        run, times, positions = load_trajectory(args.data)
        from isaacsim import SimulationApp

        app = SimulationApp(dict(
            headless=True, hide_ui=True, active_gpu=0, physics_gpu=0, multi_gpu=False,
            max_gpu_count=1, limit_cpu_threads=8, width=args.width, height=args.height,
            renderer="RaytracedLighting", anti_aliasing=2, sync_loads=True,
            fast_shutdown=True, enable_crashreporter=False,
        ))
        report["isaacsim"] = version("isaacsim")
        capture(args, run, times, positions, report)
        report["complete"] = True
    except Exception:
        report["error"] = traceback.format_exc()
        print(report["error"], flush=True)
    finally:
        if app is not None:
            report["app_close_invoked"] = True
            write_json(path, report)
            app.close(wait_for_replicator=True, skip_cleanup=False)
        else:
            write_json(path, report)
    return 0 if report["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
