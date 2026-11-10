"""Isaac rendering worker using the zero-time capture pattern from obstacle_course."""
from pathlib import Path
import socket
import time
import traceback

import numpy as np

from demo.showcase.simulate import sha, write
from .model import CARRIER_Z_M, STRAND_COLORS, STRAND_NAMES
from .render import FPS, RESOLUTION, frame_hash


def annotate(rgb, t, cfg, progress, strain, view):
    from PIL import Image, ImageDraw, ImageFont

    image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(image, "RGBA")
    def text(x, y, label, size, color, face="DejaVuSans.ttf"):
        try:
            font = ImageFont.truetype(face, size)
        except OSError:
            font = ImageFont.load_default()
        draw.text((x, y), label, font=font, fill=color)

    left = 48
    mono = "DejaVuSansMono.ttf"
    bold = "DejaVuSans-Bold.ttf"
    white = (244, 247, 248)
    subdued = (178, 199, 207)
    mint = (130, 233, 213)
    draw.rectangle((0, 0, 432, 720), fill=(5, 13, 21, 215))
    draw.line((432, 0, 432, 720), fill=(147, 185, 201, 68), width=1)
    draw.line((left, 45, left + 41, 45), fill=mint, width=4)
    text(left, 64, "DEFORMX   /   MOTION STUDIES", 15, mint, mono)
    text(left - 3, 114, "THREE-STRAND", 37, white, bold)
    text(left - 3, 161, "BRAIDER", 50, white, bold)
    text(left, 240, "Alternating carrier exchanges", 19, subdued)
    text(left, 268, "CUDA rod–rod contact", 19, subdued)
    draw.line((left, 316, 377, 316), fill=(147, 185, 201, 72), width=1)
    text(left, 339, "VERIFIED SPATIAL CROSSINGS", 15, mint, mono)
    text(left - 2, 368, f"{progress:02d}", 53, white, mono)
    text(left + 79, 395, f"/ {cfg.exchanges:02d}", 22, subdued, mono)
    bar_pitch = min(43, (377 - left) // cfg.exchanges)
    bar_width = max(1, bar_pitch - 9)
    for i in range(cfg.exchanges):
        x = left + 2 + i * bar_pitch
        color = mint if i < progress else (88, 113, 126)
        draw.rounded_rectangle((x, 448, x + bar_width, 451), 2, fill=color)
    text(left, 479, f"{100 * strain:4.1f}%  max segment strain", 15, subdued, mono)
    for j, name in enumerate(STRAND_NAMES):
        x = left + j * 111
        color = ((254, 127, 111), (108, 229, 188), (249, 191, 80))[j]
        draw.ellipse((x, 548, x + 8, 556), fill=color)
        text(x + 15, 542, name, 14, subdued, mono)
    draw.line((left, 598, 377, 598), fill=(147, 185, 201, 72), width=1)
    text(left, 621, "PRESCRIBED CARRIERS", 13, subdued, mono)
    text(left, 645, "DYNAMIC STRAND INTERIORS", 13, subdued, mono)
    draw.rounded_rectangle((1060, 38, 1241, 101), 7, fill=(8, 19, 30, 199))
    captions = {
        "whole": "01   WHOLE MACHINE",
        "closeup": "02   BRAID CLOSEUP",
        "carriers": "02   CARRIER DETAIL",
        "braid": "03   FORMED BRAID",
    }
    text(1072, 49, captions[view], 14, mint, mono)
    text(1072, 74, f"SIM {t:04.2f} s   /   1.00×", 14, white, mono)
    return image


def capture(args, run, data, cfg, metrics, indices):
    out = args.out
    (out / "frames").mkdir()
    manifest = args.assets / "manifest.json"
    report_file = out / "native_report.json"
    report = dict(
        complete=False, errors=[], app_close_invoked=False, frames=[], capture_clock_delta_max_s=0.,
        max_render_position_error_m=0., amplitude_scale=1., physics_stepping=False, renderer="Isaac Sim RaytracedLighting",
        host=socket.gethostname(), trajectory_sha256=run["trajectory_sha256"], resolution=list(RESOLUTION),
        material_visual_radius_m=cfg.barrier_radius_m, asset_manifest_sha256=sha(manifest),
        asset_sha256={p.name: sha(p) for p in sorted(args.assets.glob("*.usdc"))},
        source_sha256={p.name: sha(p) for p in Path(__file__).parent.glob("render*.py")},
    )
    write(report_file, report)
    app = world = camera = None
    start = time.perf_counter()
    try:
        from isaacsim import SimulationApp
        width, height = RESOLUTION
        app = SimulationApp(dict(headless=True, hide_ui=True, active_gpu=0, physics_gpu=0, multi_gpu=False,
                                 max_gpu_count=1, limit_cpu_threads=8, width=width, height=height,
                                 renderer="RaytracedLighting", anti_aliasing=2, sync_loads=True,
                                 fast_shutdown=True, enable_crashreporter=False))
        from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdShade, Vt
        from isaacsim.core.api import World
        from isaacsim.core.utils.stage import get_current_stage
        from isaacsim.sensors.camera import Camera
        import omni.replicator.core as rep
        import omni.timeline
        import omni.usd

        world = World(physics_dt=1 / cfg.step_hz, rendering_dt=1 / FPS,
                      stage_units_in_meters=1., backend="numpy", device="cpu")
        stage = get_current_stage()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        stage.DefinePrim("/World/Hardware", "Xform")
        def reference(dest, file, root):
            source = str((args.assets / file).resolve())
            part = Usd.Stage.Open(source)
            if UsdGeom.GetStageMetersPerUnit(part) != 1. or not part.GetPrimAtPath(root):
                raise ValueError(f"asset units or root path mismatch for {file}")
            prim = stage.DefinePrim(dest, "Xform")
            prim.GetReferences().AddReference(source, root)
            if not prim.GetChildren():
                raise ValueError(f"reference did not compose for {file}")
            return UsdGeom.Xformable(prim)

        reference("/World/Hardware/Frame", "frame.usdc", "/Frame")
        moves, curves = [], []
        for i, rgb in enumerate(STRAND_COLORS):
            carrier = reference(f"/World/Hardware/Carrier_{i}", f"carrier_{i}.usdc", f"/Carrier_{i}")
            moves.append(carrier.AddTranslateOp(opSuffix="MeasuredEndpoint"))
            curve = UsdGeom.BasisCurves.Define(stage, f"/World/Strand_{i}")
            curve.CreateTypeAttr("linear")
            curve.CreateWrapAttr("nonperiodic")
            curve.CreateCurveVertexCountsAttr([cfg.vertices])
            curve.CreateWidthsAttr([2 * cfg.barrier_radius_m])
            curve.SetWidthsInterpolation("constant")
            curves.append(curve.CreatePointsAttr())
            material = UsdShade.Material.Define(stage, f"/World/Looks/Strand_{i}")
            shader = UsdShade.Shader.Define(stage, f"/World/Looks/Strand_{i}/PBR")
            shader.CreateIdAttr("UsdPreviewSurface")
            shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgb))
            shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(.27)
            shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*(c * .17 for c in rgb)))
            material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
            UsdShade.MaterialBindingAPI.Apply(curve.GetPrim()).Bind(material)

        dome = UsdLux.DomeLight.Define(stage, "/World/Lights/StudioAmbient")
        dome.CreateIntensityAttr(495)
        dome.CreateColorAttr(Gf.Vec3f(.75, .84, 1.))
        for name, rotation, intensity, color in [
            ("SoftKey", (-37, -24, 17), 2300, (1., .92, .80)),
            ("BlueFill", (44, 52, -73), 700, (.36, .67, 1.)),
            ("CoolRim", (118, -15, 30), 1150, (.62, .85, 1.)),
        ]:
            light = UsdLux.DistantLight.Define(stage, f"/World/Lights/{name}")
            light.CreateAngleAttr(15)
            light.CreateIntensityAttr(intensity)
            light.CreateColorAttr(Gf.Vec3f(*color))
            UsdGeom.Xformable(light.GetPrim()).AddRotateXYZOp().Set(Gf.Vec3f(*rotation))

        camera = Camera(prim_path="/World/Camera", frequency=-1, resolution=RESOLUTION, annotator_device="cpu")
        camera.prim.GetAttribute("focalLength").Set(33.)
        camera.prim.GetAttribute("clippingRange").Set(Gf.Vec2f(.01, 150.))

        extra = cfg.carrier_z_m - CARRIER_Z_M
        whole_eye = np.array([.41, -4.03 - 2.9 * extra, 1.66 + .65 * extra])
        whole_target = np.array([-.39 - .23 * extra, 0., .604 + .50 * extra])
        close_eye = np.array([-.16 - .26 * extra, -1.73 - 2.60 * extra, .92 + .70 * extra])
        close_target = np.array([-.16 - .26 * extra, 0., .764 + .625 * extra])
        carrier_eye = np.array([.12, -1.85, cfg.carrier_z_m + .015])
        carrier_target = np.array([-.19, 0., cfg.carrier_z_m - .17])

        def aim(t):
            phase = (t - cfg.settle_s) / cfg.exchange_s
            if extra > 0:
                if phase < 2:
                    eye, target, view = whole_eye, whole_target, "whole"
                elif phase < 7:
                    eye, target, view = carrier_eye, carrier_target, "carriers"
                else:
                    eye, target, view = close_eye, close_target, "braid"
            else:
                u = float(np.clip((t - cfg.settle_s - 1.75 * cfg.exchange_s) / 1.35, 0., 1.))
                u = u * u * (3. - 2. * u)
                eye = (1. - u) * whole_eye + u * close_eye
                target = (1. - u) * whole_target + u * close_target
                view = "closeup" if u > .6 else "whole"
            look = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*target), Gf.Vec3d(0, 0, 1)).GetInverse()
            q = look.ExtractRotationQuat()
            camera.set_world_pose(position=eye, orientation=np.array([q.GetReal(), *q.GetImaginary()]), camera_axes="usd")
            return view

        aim(float(data["t"][indices[0]]))
        world.reset()
        camera.initialize(attach_rgb_annotator=True)
        rep.orchestrator.set_capture_on_play(False)
        timeline = omni.timeline.get_timeline_interface()
        reset = getattr(omni.usd.get_context(), "reset_renderer_accumulation", None)

        def tick():
            before = float(world.current_time), float(timeline.get_current_time()), world.current_time_step_index
            rep.orchestrator.step(rt_subframes=2, pause_timeline=False, delta_time=0.)
            after = float(world.current_time), float(timeline.get_current_time()), world.current_time_step_index
            delta = max(abs(after[0] - before[0]), abs(after[1] - before[1]))
            report["capture_clock_delta_max_s"] = max(report["capture_clock_delta_max_s"], delta)
            if before != after:
                raise RuntimeError("capture advanced a simulation clock")

        for frame, j in enumerate(indices):
            p, roots, t = data["positions"][j], data["carriers"][j], float(data["t"][j])
            for attr, move, rod, root in zip(curves, moves, p, roots):
                attr.Set(Vt.Vec3fArray.FromNumpy(np.ascontiguousarray(rod, dtype=np.float32)))
                move.Set(Gf.Vec3d(*map(float, root)))
                error = max(float(np.max(abs(np.asarray(attr.Get()) - rod))),
                            float(np.max(abs(np.asarray(move.Get()) - root))))
                report["max_render_position_error_m"] = max(report["max_render_position_error_m"], error)
            view = aim(t)
            if reset:
                reset()
            for _ in range(14 if frame == 0 or args.preview else 2):
                tick()
            rgb = np.asarray(camera.get_rgba(device="cpu"))[:, :, :3]
            if rgb.shape != (height, width, 3) or rgb.dtype != np.uint8 or rgb.std() < 2:
                raise RuntimeError("invalid native RGB capture")
            checkpoints = [row for row in metrics["checkpoints"][:-1] if row["index"] <= j]
            strain = float(np.max(np.abs(np.linalg.norm(np.diff(p, axis=1), axis=-1)
                                         / (np.linalg.norm(np.diff(data["positions"][0], axis=1), axis=-1)) - 1)))
            image = annotate(rgb, t, cfg, len(checkpoints), strain, view)
            png = out / "frames" / f"{frame:06d}.png"
            image.save(png, compress_level=1)
            report["frames"].append(dict(index=int(j), t_s=t, image_sha256=sha(png),
                                         state_sha256=frame_hash(p, roots), view=view))
            if frame % 30 == 0:
                print(f"CAPTURE {frame + 1}/{len(indices)}", flush=True)
                write(report_file, report)
        stage.GetRootLayer().Export(str(out / "scene.usda"))
        report.update(complete=True, wall_s=time.perf_counter() - start, accumulation_reset=reset is not None)
    except Exception:
        report["errors"].append(traceback.format_exc())
        print(report["errors"][-1], flush=True)
    finally:
        if camera is not None:
            camera.pause()
            camera.destroy()
        if world is not None:
            world.stop()
        if app is not None:
            report["app_close_invoked"] = True
        write(report_file, report)
        if app is not None:
            app.close(wait_for_replicator=True, skip_cleanup=False)
    return int(not report["complete"])
