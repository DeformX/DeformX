"""Capture a 1x native Isaac Sim replay from recorded reel, cable and tether states."""
import argparse
import json
from pathlib import Path
import subprocess
import time
import traceback

from . import analysis, model

RESOLUTION = (1280, 720)
FPS = 30
COMPONENTS = ("chassis", "drive", "guide", "tether", "ferrule")


def encode(out, expected_frames):
    video = out / "winch_tension.mp4"
    encode_command = ["ffmpeg", "-v", "error", "-nostdin", "-framerate", str(FPS),
                      "-i", str(out / "frames" / "%06d.png"), "-c:v", "libx264", "-crf", "17",
                      "-preset", "medium", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(video)]
    subprocess.run(encode_command, check=True, timeout=180)
    decoded = subprocess.run(["ffmpeg", "-v", "error", "-xerror", "-nostdin", "-i", str(video),
                              "-map", "0:v:0", "-f", "null", "-"], capture_output=True, check=True, timeout=180)
    if decoded.stderr:
        raise RuntimeError(decoded.stderr.decode())
    stream_info = subprocess.check_output(
        ["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,avg_frame_rate,nb_read_frames", "-of", "json", str(video)],
        timeout=60,
    )
    stream = json.loads(stream_info)["streams"][0]
    if (stream["avg_frame_rate"] != f"{FPS}/1" or int(stream["nb_read_frames"]) != expected_frames
            or (stream["width"], stream["height"]) != RESOLUTION):
        raise RuntimeError(f"Video stream does not match the capture: {stream}")
    return dict(name=video.name, sha256=analysis.sha256(video), full_decode_ok=True, **stream)


def preflight(data, assets):
    source, frames = analysis.load_recording(data)
    if not source["complete"]:
        raise ValueError("Replay requires a numerically complete recording")
    manifest = json.loads((assets / "manifest.json").read_text())
    names = [name + ".usdc" for name in COMPONENTS]
    verified = {name: analysis.sha256(assets / name) for name in names}
    if verified != {name: manifest["asset_sha256"][name] for name in names}:
        raise ValueError("Blender asset hashes do not match their manifest")
    return source, frames, verified


def capture(args, source, frames, report, persist):
    from isaacsim import SimulationApp

    started = time.perf_counter()
    width, height = RESOLUTION
    app = SimulationApp(dict(headless=True, hide_ui=True, active_gpu=0, physics_gpu=0,
                             multi_gpu=False, max_gpu_count=1, limit_cpu_threads=8,
                             width=width, height=height, renderer="RaytracedLighting",
                             anti_aliasing=2, sync_loads=True, fast_shutdown=True,
                             enable_crashreporter=False))
    world = camera = None
    try:
        import numpy as np
        from PIL import Image, ImageDraw, ImageFont
        from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdShade, Vt
        from isaacsim.core.api import World
        from isaacsim.core.utils.stage import get_current_stage
        from isaacsim.sensors.camera import Camera
        import omni.replicator.core as rep
        import omni.timeline
        import omni.usd

        settings = source["settings"]
        cycles = settings.get("cycles", 1)
        times = frames["t"]
        exchange_hz = settings["exchange_hz"]
        stride = exchange_hz // FPS
        if stride * FPS != exchange_hz:
            raise ValueError("Exchange rate is incompatible with 30 Hz replay")
        first = round(model.FEED_START_S * exchange_hz) if times[-1] > model.FEED_END_S and not args.include_settle else 0
        indices = np.arange(first, len(times), stride)
        if len(indices) < 2 or not np.allclose(np.diff(times[indices]), 1 / FPS, atol=1e-10, rtol=0):
            raise ValueError("Frame timestamps are incompatible with 1x capture")
        report.update(source_start_s=float(times[indices[0]]), source_last_s=float(times[indices[-1]]),
                      maximum_source_interval_error_s=float(np.max(np.abs(np.diff(times[indices]) - 1 / FPS))),
                      frame_count=int(len(indices)), scheduled_cycles=cycles, capture_frames=[])
        (args.out / "frames").mkdir()
        world = World(physics_dt=1 / 120, rendering_dt=1 / FPS,
                      stage_units_in_meters=1., backend="numpy", device="cpu")
        stage = get_current_stage()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)

        def reference(path, component):
            root = UsdGeom.Xform.Define(stage, path)
            root.GetPrim().GetReferences().AddReference(str((args.assets / f"{component}.usdc").resolve()), "/Asset")
            if not list(root.GetPrim().GetChildren()):
                raise RuntimeError(f"Blender reference has no geometry: {component}")
            return root

        reference("/World/Chassis", "chassis")
        reels = []
        for component, center in zip(["drive", "guide"], settings["reel_centers_m"], strict=True):
            xform = reference("/World/" + component.title(), component)
            xform.AddTranslateOp().Set(Gf.Vec3d(*center))
            reels.append(xform.AddRotateYOp())
        springs = []
        ferrules = []
        for i in range(2):
            coil = reference(f"/World/Spring{i}", "tether")
            springs.append((coil.AddTranslateOp(), coil.AddOrientOp(), coil.AddScaleOp()))
            terminal = reference(f"/World/Terminal{i}", "ferrule")
            ferrules.append((terminal.AddTranslateOp(), terminal.AddOrientOp()))

        def material(name, color, roughness, emission=0.):
            mat = UsdShade.Material.Define(stage, f"/World/Looks/{name}")
            shader = UsdShade.Shader.Define(stage, f"/World/Looks/{name}/Surface")
            shader.CreateIdAttr("UsdPreviewSurface")
            shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
            shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
            shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*(c * emission for c in color)))
            mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
            return mat

        colors = [material("Cable", (.79, .16, .028), .46, .12),
                  material("IndexWrap", (.96, .86, .47), .32, .17)]
        radius = settings["rod_radius_m"]
        def cable(path, primitive):
            instancer = UsdGeom.PointInstancer.Define(stage, path)
            targets = []
            for j, mat in enumerate(colors):
                prototype = primitive.Define(stage, path + f"/P{j}")
                prototype.CreateRadiusAttr(radius)
                if primitive is UsdGeom.Cylinder:
                    prototype.CreateHeightAttr(1.)
                    prototype.CreateAxisAttr("Z")
                UsdShade.MaterialBindingAPI.Apply(prototype.GetPrim()).Bind(mat)
                targets.append(prototype.GetPath())
            instancer.CreatePrototypesRel().SetTargets(targets)
            return instancer

        segments = cable("/World/Cable", UsdGeom.Cylinder)
        nodes = cable("/World/CableJoins", UsdGeom.Sphere)
        node_count = frames["positions"].shape[1]
        middle = node_count // 2
        node_palette = np.zeros(node_count, dtype=np.int32)
        node_palette[middle - 3:middle + 4] = 1
        nodes.CreateProtoIndicesAttr().Set(node_palette.tolist())
        segments.CreateProtoIndicesAttr().Set((node_palette[:-1] & node_palette[1:]).tolist())

        floor = UsdGeom.Cube.Define(stage, "/World/StudioFloor")
        floor.CreateSizeAttr(1.)
        floor.AddTranslateOp().Set(Gf.Vec3d(0., 0., -.106))
        floor.AddScaleOp().Set(Gf.Vec3d(200., 200., .025))
        UsdShade.MaterialBindingAPI.Apply(floor.GetPrim()).Bind(material("Floor", (.078, .103, .129), .67))
        dome = UsdLux.DomeLight.Define(stage, "/World/Lights/Dome")
        dome.CreateIntensityAttr(410.)
        dome.CreateColorAttr(Gf.Vec3f(.79, .88, 1.))
        for name, rotation, intensity, color in [
            ("Softbox", (-24., -29., -17.), 2050., (1., .92, .82)),
            ("Rim", (35., 30., 136.), 1220., (.59, .76, 1.)),
            ("FrontFill", (76., 0., 12.), 1350., (1., .96, .89)),
        ]:
            light = UsdLux.DistantLight.Define(stage, f"/World/Lights/{name}")
            light.CreateIntensityAttr(intensity)
            light.CreateColorAttr(Gf.Vec3f(*color))
            light.CreateAngleAttr(12.)
            UsdGeom.Xformable(light.GetPrim()).AddRotateXYZOp().Set(Gf.Vec3f(*rotation))

        camera = Camera(prim_path="/World/Camera", frequency=-1, resolution=RESOLUTION, annotator_device="cpu")
        camera.prim.GetAttribute("clippingRange").Set(Gf.Vec2f(.02, 300.))
        shots = {
            "whole": ((.155, -1.53, .90), (0., .017, .196), 24.),
            "right_reel": ((.47, -.91, .68), (.20, 0., .32), 29.),
        }

        def look(shot):
            eye, target, focal = shots[shot]
            camera.prim.GetAttribute("focalLength").Set(focal)
            matrix = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*target), Gf.Vec3d(0, 0, 1)).GetInverse()
            q = matrix.ExtractRotationQuat()
            camera.set_world_pose(position=np.asarray(eye), orientation=np.array([q.GetReal(), *q.GetImaginary()]),
                                  camera_axes="usd")

        look("whole")

        anchors = frames["tether_anchors_m"]
        def update(index):
            p = frames["positions"][index]
            segment_dir = np.diff(p, axis=0)
            lengths = np.linalg.norm(segment_dir, axis=1)
            midpoints = (p[:-1] + p[1:]) / 2.
            nodes.CreatePositionsAttr().Set(Vt.Vec3fArray.FromNumpy(np.ascontiguousarray(p, np.float32)))
            segments.CreatePositionsAttr().Set(Vt.Vec3fArray.FromNumpy(np.ascontiguousarray(midpoints, np.float32)))
            scales = np.ones_like(segment_dir)
            scales[:, 2] = lengths
            segments.CreateScalesAttr().Set(Vt.Vec3fArray.FromNumpy(scales))
            orientations = []
            for direction in segment_dir / lengths[:, None]:
                rot = Gf.Rotation(Gf.Vec3d(0, 0, 1), Gf.Vec3d(*map(float, direction))).GetQuat()
                orientations.append(Gf.Quath(rot.GetReal(), Gf.Vec3h(*rot.GetImaginary())))
            segments.CreateOrientationsAttr().Set(Vt.QuathArray(orientations))
            for angle, rotate in zip(frames["reel_angle_rad"][index], reels, strict=True):
                rotate.Set(float(np.degrees(angle)))
            for end, anchor, spring, ferrule in zip(p[[0, -1]], anchors, springs, ferrules, strict=True):
                delta = end - anchor
                length = float(np.linalg.norm(delta))
                orient = Gf.Rotation(Gf.Vec3d(0, 0, 1), Gf.Vec3d(*map(float, delta / length))).GetQuat()
                quat = Gf.Quatf(orient.GetReal(), Gf.Vec3f(*orient.GetImaginary()))
                spring[0].Set(Gf.Vec3d(*map(float, anchor)))
                spring[1].Set(quat)
                spring[2].Set(Gf.Vec3f(1., 1., length))
                ferrule[0].Set(Gf.Vec3d(*map(float, end)))
                ferrule[1].Set(quat)

        update(int(indices[0]))
        world.reset()
        camera.initialize(attach_rgb_annotator=True)
        rep.orchestrator.set_capture_on_play(False)
        timeline = omni.timeline.get_timeline_interface()
        reset_history = getattr(omni.usd.get_context(), "reset_renderer_accumulation", None)
        report["renderer_history_reset"] = reset_history is not None
        report["maximum_capture_clock_delta_s"] = 0.
        report["physics_steps_during_capture"] = 0

        def tick():
            world_t, timeline_t = float(world.current_time), float(timeline.get_current_time())
            world_k = int(world.current_time_step_index)
            rep.orchestrator.step(rt_subframes=4, pause_timeline=False, delta_time=0.)
            delta = max(abs(float(world.current_time) - world_t), abs(float(timeline.get_current_time()) - timeline_t))
            step_delta = abs(int(world.current_time_step_index) - world_k)
            report["maximum_capture_clock_delta_s"] = max(report["maximum_capture_clock_delta_s"], delta)
            report["physics_steps_during_capture"] += step_delta
            if delta > 1e-9 or step_delta:
                raise RuntimeError("Isaac capture advanced the simulation clock")

        def font(name, size):
            try:
                return ImageFont.truetype(f"DejaVuSans{name}.ttf", size)
            except OSError:
                return ImageFont.load_default(size=size)

        title = font("-Bold", 32)
        medium = font("-Bold", 17)
        mono = font("Mono", 15)
        small = font("", 15)
        base = (15, 24, 35)
        white = (241, 244, 245)
        gray = (180, 195, 206)
        amber = (255, 154, 75)
        blue = (102, 211, 232)
        reference_times = np.arange(cycles) * model.CYCLE_PERIOD_S + model.FEED_START_S
        reference_times[0] = times[first]
        begin_x = np.interp(reference_times, times, frames["positions"][:, middle, 0])
        begin_angle = np.interp(reference_times, times, frames["reel_angle_rad"][:, 1])

        def annotate(rgba, index, cycle, phase, shot):
            t = float(times[index])
            image = Image.fromarray(rgba[:, :, :3])
            draw = ImageDraw.Draw(image)
            draw.rectangle((0, 0, width, 82), fill=base)
            draw.text((34, 12), "TRACTION REELS", fill=white, font=title)
            view = "RIGHT REEL DETAIL" if shot == "right_reel" else "WHOLE RIG"
            subtitle = f"TENSIONED CABLE  /  {view}" if cycles > 1 else "TENSIONED CABLE  /  TWO-WAY TRANSPORT"
            draw.text((36, 54), subtitle, fill=gray, font=mono)
            labels = {"Settle": "00  SETTLE", "Feed": "01  FEED  →", "Pause": "02  PAUSE",
                      "Reverse": "03  ←  REVERSE", "Hold": "04  HOLD"}
            label = labels[phase]
            if settings["mode"] != "driven":
                label = settings["mode"].upper() + " CONTROL"
            if cycles > 1:
                label = f"CYCLE {cycle + 1:02d}/{cycles:02d}    {label}"
            draw.text((width - 34, 20), label, fill=amber if phase == "Feed" else blue,
                      font=medium, anchor="ra")
            draw.text((width - 34, 54), f"SOURCE  {t:04.2f} s    1.00×", fill=gray, font=mono, anchor="ra")

            draw.rectangle((0, 618, width, height), fill=base)
            x = frames["positions"][index, middle, 0] - begin_x[cycle]
            angle = np.degrees(frames["reel_angle_rad"][index, 1] - begin_angle[cycle])
            torque = frames["reel_wrenches"][index, 1, 4] * 1000.
            draw.text((34, 637), "LEFT / PRESCRIBED MOTOR", fill=amber, font=medium)
            draw.text((488, 637), "MATERIAL MARKER", fill=white, font=medium)
            draw.text((886, 637), "RIGHT / PASSIVE AXLE", fill=blue, font=medium)
            draw.text((34, 663), "Surface speed is programmed", fill=gray, font=small)
            draw.text((488, 663), f"Travel {x * 1000:+.0f} mm", fill=gray, font=mono)
            draw.text((886, 663), f"{angle:+.0f}°   {torque:+.1f} mN·m*", fill=gray, font=mono)
            draw.text((width - 34, 691), "*Native contact-law torque", fill=gray, font=small, anchor="ra")
            local_t = t - cycle * model.CYCLE_PERIOD_S
            progress = np.clip((local_t - model.FEED_START_S) / (model.REVERSE_END_S - model.FEED_START_S), 0., 1.)
            draw.rectangle((34, 694, 775, 698), fill=(63, 82, 101))
            x0 = 34 + 741 * progress
            draw.rectangle((34, 694, x0, 698), fill=blue)
            for split in [model.FEED_END_S, model.REVERSE_START_S]:
                marker = 34 + 741 * (split - model.FEED_START_S) / (model.REVERSE_END_S - model.FEED_START_S)
                draw.rectangle((marker, 689, marker + 1, 703), fill=gray)
            return image

        last_shot = "whole"
        for frame_number, index in enumerate(indices):
            index = int(index)
            t = float(times[index])
            cycle = model.cycle_index(t, cycles)
            phase = model.phase(t, cycles)
            shot = "right_reel" if cycle in (2, 4, 6) and cycle < cycles - 1 else "whole"
            cut = shot != last_shot
            if cut:
                look(shot)
                last_shot = shot
            update(index)
            if reset_history:
                reset_history()
            for _ in range(12 if frame_number == 0 else (6 if cut else 2)):
                tick()
            rgba = np.asarray(camera.get_rgba(device="cpu"))
            if rgba.shape != (height, width, 4) or rgba.dtype != np.uint8 or rgba[:, :, :3].std() < 2:
                raise RuntimeError(f"Invalid camera image at source index {index}")
            image = annotate(rgba, index, cycle, phase, shot)
            image.save(args.out / "frames" / f"{frame_number:06d}.png", compress_level=1)
            report["capture_frames"].append(dict(
                frame=frame_number, trajectory_index=index, t_s=t, cycle=cycle + 1, phase=phase, shot=shot,
                reel_angle_rad=frames["reel_angle_rad"][index].tolist(),
                tracked_material_position_m=frames["positions"][index, middle].tolist(),
            ))
            if frame_number % 30 == 0:
                print(f"Captured {frame_number + 1}/{len(indices)}", flush=True)
                persist()
        if not stage.GetRootLayer().Export(str(args.out / "scene.usda")):
            raise RuntimeError("Could not export the native scene")
        report["video"] = encode(args.out, len(indices))
        report["complete"] = True
    except BaseException:
        report["error"] = traceback.format_exc()
        raise
    finally:
        try:
            if camera is not None:
                camera.pause()
                camera.destroy()
            if world is not None:
                world.stop()
        except BaseException:
            report["complete"] = False
            report["error"] = traceback.format_exc()
            raise
        finally:
            report["capture_and_encode_wall_s"] = time.perf_counter() - started
            report["app_close_invoked"] = True
            persist()
            app.close(wait_for_replicator=True, skip_cleanup=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path, help="Folder from the winch numerical runner")
    parser.add_argument("--assets", required=True, type=Path, help="Folder from Blender authoring")
    parser.add_argument("--out", required=True, type=Path, help="Fresh capture directory")
    parser.add_argument("--include-settle", action="store_true", help="Show the initial settle interval")
    args = parser.parse_args(argv)
    try:
        source, frames, assets = preflight(args.data, args.assets)
        args.out.mkdir(parents=True, exist_ok=False)
    except (OSError, ValueError, KeyError) as exc:
        parser.error(str(exc))
    report = dict(complete=False, source_sha256=analysis.sha256(__file__),
                  trajectory_sha256=source["trajectory_sha256"], asset_sha256=assets,
                  render_resolution=list(RESOLUTION), fps=FPS, amplitude_scale=1., playback_speed=1.,
                  rendering="Native Isaac Sim USD geometry with Pillow annotation; no physics during replay",
                  app_close_invoked=False)

    def persist():
        (args.out / "native_report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    persist()
    try:
        capture(args, source, frames, report, persist)
    except BaseException:
        report["complete"] = False
        report["error"] = traceback.format_exc()
        print(report["error"], flush=True)
        persist()
    return 0 if report["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
