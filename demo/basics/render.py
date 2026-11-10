"""Native Isaac replay of recorded basics cases, with a normal and a slow view."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import traceback

from .run import sha256

FPS = 30
RESOLUTION = (1280, 720)
VIEWS = (("whole", 1.0), ("detail", 0.125))
PROPS = ("fixed-grip", "driven-grip", "sphere", "floor", "goal")
# Title, what to watch, color key, wide and close camera directions, poster source fraction.
CASES = {
    "hanging": ("Hanging rod", "Gravity swings the free tip beneath a fixed pin.",
                ("Fixed support",), ((.23, -1, .18), (.09, -1, .06)), .24),
    "endpoints": ("Driven endpoints", "The left grip feeds slack sideways. The right grip stays put.",
                  ("Driven support", "Fixed support"), ((.45, -1, .46), (.41, -1, .36)), .63),
    "obstacle": ("Sphere obstacle", "The driven grip feeds slack and the rod meets the sphere.",
                 ("Driven support", "Fixed support"), ((.27, -1, .30), (.10, -1, .18)), .58),
    "self-contact": ("Self-contact", "An open coil settles against the floor and its own turns.",
                     ("Fixed support",), ((.63, -1, .85), (.45, -1, .52)), .33),
    "materials": ("Bending stiffness", "Two rods bend under gravity. Only their bending stiffness differs.",
                  ("More flexible", "Stiffer"), ((.91, -1, .66), (1.05, -1, .50)), .55),
    "tip-target": ("Tip target", "The controller moves the upper grip to guide the free tip into the goal.",
                   ("Driven support", "Goal"), ((.55, -1, .21), (.70, -1, .15)), .80),
}
LEGEND = {
    "Driven support": "#ffc16b", "Fixed support": "#b1c6d9",
    "More flexible": "#ffc16b", "Stiffer": "#55dbff", "Goal": "#61efa6",
}


def replay_segments(t):
    """Hold the latest saved sample at each presentation tick; include the final pose."""
    import numpy as np

    t = np.asarray(t, dtype=float)
    if t.ndim != 1 or len(t) < 2 or not np.isfinite(t).all() or np.any(np.diff(t) <= 0):
        raise ValueError("Expected at least two finite, increasing source timestamps")
    segments = []
    start = 0
    for view, speed in VIEWS:
        ticks = math.ceil((t[-1] - t[0]) * FPS / speed - 1e-9) + 1
        intended = np.minimum(t[0] + np.arange(ticks) * speed / FPS, t[-1])
        indices = np.searchsorted(t, intended + 1e-10, side="right") - 1
        segments.append(dict(view=view, speed=speed, first_video_frame=start,
                             end_video_frame_exclusive=start + ticks,
                             start_video_s=start / FPS, end_video_s=(start + ticks) / FPS,
                             source_indices=indices.tolist(),
                             source_times_s=t[indices].tolist()))
        start += ticks
    return segments


def load_source(case, data):
    import numpy as np

    trajectory = data / "trajectory.npz"
    source_path = data / "report.json"
    source = json.loads(source_path.read_text())
    digest = sha256(trajectory)
    if digest != source["trajectory_sha256"]:
        raise ValueError(f"Trajectory hash does not match {source_path}")
    settings = source["settings"]
    if source["complete"] is not True or settings["case"].replace("_", "-") != case:
        raise ValueError(f"Expected a complete {case} recording")
    if case in ("obstacle", "self-contact") and settings["contact_enabled"] is not True:
        raise ValueError(f"{case} replay requires contact enabled")
    if case == "tip-target" and not source["metrics"]["success"]:
        raise ValueError("Tip target replay requires a successful episode")
    with np.load(trajectory, allow_pickle=False) as stored:
        t, p = stored["t"], stored["positions"]
        target = stored["target"] if case == "tip-target" else None
    if (p.shape != (len(t), settings["ropes"], settings["vertices_per_rope"], 3)
            or p.dtype != np.float32 or not np.isfinite(p).all()
            or np.any(np.linalg.norm(np.diff(p, axis=2), axis=-1) == 0)
            or (target is not None and (target.shape != (3,) or not np.isfinite(target).all()))):
        raise ValueError("Invalid recorded geometry")
    return dict(settings=settings, times=t, positions=p, target=target, segments=replay_segments(t),
                trajectory_sha256=digest, run_report_sha256=sha256(source_path))


def props_hashes(assets):
    expected = json.loads((assets / "manifest.json").read_text())["asset_sha256"]
    observed = {f"{name}.usdc": sha256(assets / f"{name}.usdc") for name in PROPS}
    if expected != observed:
        raise ValueError("Blender props do not match their manifest")
    return observed


def selected_indices(case, segment, preview):
    indices = segment["source_indices"]
    if preview:
        return sorted({indices[0], indices[round((len(indices)-1)*CASES[case][-1])], indices[-1]})
    return list(dict.fromkeys(indices))


def positions_hash(positions):
    return hashlib.sha256(positions.tobytes()).hexdigest()


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def native(a, recordings):
    from isaacsim import SimulationApp

    app = SimulationApp(dict(headless=True, hide_ui=True, active_gpu=0, physics_gpu=0,
                             multi_gpu=False, max_gpu_count=1, limit_cpu_threads=8,
                             width=RESOLUTION[0], height=RESOLUTION[1], renderer="RaytracedLighting",
                             anti_aliasing=2, sync_loads=True, fast_shutdown=True,
                             enable_crashreporter=False))
    camera = world = None
    report = dict(complete=False, app_close_invoked=False, renderer="Isaac Sim RaytracedLighting",
                  asset_sha256=props_hashes(a.assets), renderer_sha256=sha256(__file__), cases={})
    path = a.out / "native-report.json"
    try:
        import numpy as np
        import omni.replicator.core as rep
        import omni.timeline
        import omni.usd
        from isaacsim.core.api import World
        from isaacsim.core.utils.stage import get_current_stage
        from isaacsim.sensors.camera import Camera
        from pxr import Gf, UsdGeom, UsdLux

        world = World(physics_dt=1/120, rendering_dt=1/FPS, stage_units_in_meters=1.,
                      backend="numpy", device="cpu")
        stage = get_current_stage()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        dome = UsdLux.DomeLight.Define(stage, "/World/Dome")
        dome.CreateIntensityAttr(410)
        dome.CreateColorAttr(Gf.Vec3f(.79, .88, 1.))
        for name, rotation, intensity, color in [
            ("Key", (-24., -29., -17.), 2050., (1., .92, .82)),
            ("Rim", (35., 30., 136.), 1220., (.59, .76, 1.)),
            ("Fill", (76., 0., 12.), 1350., (1., .96, .89)),
        ]:
            light = UsdLux.DistantLight.Define(stage, f"/World/{name}")
            light.CreateIntensityAttr(intensity)
            light.CreateColorAttr(Gf.Vec3f(*color))
            light.CreateAngleAttr(12.)
            UsdGeom.Xformable(light.GetPrim()).AddRotateXYZOp().Set(Gf.Vec3f(*rotation))
        usd_camera = UsdGeom.Camera.Define(stage, "/World/Camera")
        usd_camera.CreateHorizontalApertureAttr(36.)
        usd_camera.CreateVerticalApertureAttr(36.*9/16)
        camera = Camera(prim_path="/World/Camera", frequency=-1, resolution=RESOLUTION,
                        annotator_device="cpu")
        for name, value in [("focalLength", 35.), ("clippingRange", Gf.Vec2f(.005, 10.))]:
            camera.prim.GetAttribute(name).Set(value)
        camera.set_world_pose(position=np.array([0., -2., .5]))
        world.reset()
        camera.initialize(attach_rgb_annotator=True)
        rep.orchestrator.set_capture_on_play(False)
        timeline = omni.timeline.get_timeline_interface()
        reset_history = getattr(omni.usd.get_context(), "reset_renderer_accumulation", None)
        report["renderer_history_reset"] = reset_history is not None

        def get_clock():
            return float(world.current_time), float(timeline.get_current_time()), world.current_time_step_index

        for case, source in recordings.items():
            dest = a.out / case
            dest.mkdir()
            audit = capture_case(a, case, source, dest, stage, camera, rep, get_clock, reset_history)
            report["cases"][case] = audit
            write_json(path, report)
            stage.RemovePrim(case_path(case))
        report["complete"] = True
    except BaseException:
        report["error"] = traceback.format_exc()
        traceback.print_exc()
        raise
    finally:
        write_json(path, report)
        try:
            if camera is not None:
                camera.pause()
                camera.destroy()
            if world is not None:
                world.stop()
        finally:
            report["app_close_invoked"] = True
            write_json(path, report)
            app.close(wait_for_replicator=True, skip_cleanup=False)


def case_path(case):
    return "/World/" + case.replace("-", "_")


def capture_case(a, case, source, dest, stage, camera, rep, get_clock, reset_history):
    import numpy as np
    from PIL import Image
    from pxr import Gf, Sdf, UsdGeom, UsdShade, Vt

    root = case_path(case)
    UsdGeom.Xform.Define(stage, root)
    settings, positions, target = (source[k] for k in ("settings", "positions", "target"))

    def prop(component, name, position, scale=1.):
        obj = UsdGeom.Xform.Define(stage, f"{root}/{name}")
        obj.GetPrim().GetReferences().AddReference(str(a.assets / f"{component}.usdc"), "/Asset")
        if not list(obj.GetPrim().GetChildren()):
            raise RuntimeError(f"Missing Blender geometry for {component}")
        translate = obj.AddTranslateOp()
        translate.Set(Gf.Vec3d(*map(float, position)))
        if scale != 1.:
            obj.AddScaleOp().Set(Gf.Vec3f(scale))
        return translate

    extent_points = [positions.reshape(-1, 3)]
    if settings["floor"]:
        prop("floor", "Floor", (0., 0., 0.))
    if obstacle := settings["obstacle"]:
        center, radius = np.asarray(obstacle["pos"]), obstacle["radius"]
        prop("sphere", "Obstacle", center, radius)
        extent_points += [center + radius*np.eye(3), center - radius*np.eye(3)]
    if target is not None:
        radius = settings["tolerance_m"]
        prop("goal", "Goal", target, radius)
        extent_points += [target + radius*np.eye(3), target - radius*np.eye(3)]

    grips = []
    for rope, pins in enumerate(settings["pinned_vertices"]):
        for node in pins:
            moves = bool(np.any(positions[:, rope, node] != positions[0, rope, node]))
            # Position constraints with floating-point jitter are still fixed supports.
            moves = moves and bool(np.max(np.ptp(positions[:, rope, node], axis=0)) > 1e-5)
            component = "driven-grip" if moves else "fixed-grip"
            grips.append((rope, node, prop(component, f"Grip{rope}_{node}", positions[0, rope, node])))

    palette = [(1., .38, .085), (.045, .65, .90)] if case == "materials" else [(.98, .28, .055)]
    materials = []
    for index, color in enumerate(palette):
        material = UsdShade.Material.Define(stage, f"{root}/RodMaterial{index}")
        shader = UsdShade.Shader.Define(stage, f"{root}/RodMaterial{index}/Surface")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(.34)
        shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*(c*.22 for c in color)))
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        materials.append(material)

    def cable(name, primitive, count):
        obj = UsdGeom.PointInstancer.Define(stage, f"{root}/{name}")
        targets = []
        for i, material in enumerate(materials):
            proto = primitive.Define(stage, f"{root}/{name}/P{i}")
            proto.CreateRadiusAttr(settings["radius_m"])
            if primitive is UsdGeom.Cylinder:
                proto.CreateAxisAttr("Z")
                proto.CreateHeightAttr(1.)
            UsdShade.MaterialBindingAPI.Apply(proto.GetPrim()).Bind(material)
            targets.append(proto.GetPath())
        obj.CreatePrototypesRel().SetTargets(targets)
        obj.CreateProtoIndicesAttr(np.repeat(np.arange(positions.shape[1]), count).tolist())
        return obj

    nodes = cable("RodNodes", UsdGeom.Sphere, positions.shape[2])
    segments = cable("RodSegments", UsdGeom.Cylinder, positions.shape[2]-1)
    node_pos = nodes.CreatePositionsAttr()
    seg_pos, seg_scale, seg_rot = (segments.CreatePositionsAttr(), segments.CreateScalesAttr(),
                                   segments.CreateOrientationsAttr())

    def look(direction, detail):
        points = np.concatenate(extent_points)
        center = (points.min(0) + points.max(0)) * .5
        back = np.asarray(direction) / np.linalg.norm(direction)
        right = np.cross([0., 0., 1.], back)
        right /= np.linalg.norm(right)
        up = np.cross(back, right)
        projected = (points - center) @ np.array([right, up, back]).T
        x, y, z = projected.T
        # Include margin for grip props and the picture's caption bands.
        frame = 1.14 if detail else 1.43
        pad = .023 if case != "self-contact" else .012
        dist = max(np.max(z + frame*(abs(x) + pad)/(18/35)),
                   np.max(z + (frame+.13)*(abs(y) + pad)/(18/35*9/16)))
        eye = center + back*dist
        matrix = Gf.Matrix4d().SetLookAt(
            Gf.Vec3d(*map(float, eye)), Gf.Vec3d(*map(float, center)), Gf.Vec3d(0, 0, 1)).GetInverse()
        q = matrix.ExtractRotationQuat()
        camera.set_world_pose(position=eye, orientation=np.array([q.GetReal(), *q.GetImaginary()]),
                              camera_axes="usd")

    audit = dict(trajectory_sha256=source["trajectory_sha256"], run_report_sha256=source["run_report_sha256"],
                 maximum_node_position_error_m=0., maximum_grip_position_error_m=0.,
                 maximum_capture_clock_delta_s=0., physics_steps_during_capture=0,
                 visual_radius_m=settings["radius_m"], frames=[])
    for shot, segment in enumerate(source["segments"]):
        view = segment["view"]
        look(CASES[case][-2][shot], shot == 1)
        for n, index in enumerate(selected_indices(case, segment, a.preview)):
            points = positions[index]
            flat = points.reshape(-1, 3)
            direction = np.diff(points, axis=1).reshape(-1, 3)
            lengths = np.linalg.norm(direction, axis=1)
            mid = ((points[:, :-1] + points[:, 1:]) * .5).reshape(-1, 3)
            scales = np.ones_like(direction)
            scales[:, 2] = lengths
            node_pos.Set(Vt.Vec3fArray.FromNumpy(flat))
            seg_pos.Set(Vt.Vec3fArray.FromNumpy(mid))
            seg_scale.Set(Vt.Vec3fArray.FromNumpy(scales))
            quats = []
            for d in direction / lengths[:, None]:
                q = Gf.Rotation(Gf.Vec3d(0, 0, 1), Gf.Vec3d(*map(float, d))).GetQuat()
                quats.append(Gf.Quath(q.GetReal(), Gf.Vec3h(*q.GetImaginary())))
            seg_rot.Set(Vt.QuathArray(quats))
            for rope, vertex, grip in grips:
                grip.Set(Gf.Vec3d(*map(float, points[rope, vertex])))
                error = float(np.max(np.abs(np.asarray(grip.Get()) - points[rope, vertex])))
                audit["maximum_grip_position_error_m"] = max(audit["maximum_grip_position_error_m"], error)
            readback = np.asarray(node_pos.Get())
            node_error = float(np.max(np.abs(readback - flat)))
            audit["maximum_node_position_error_m"] = max(audit["maximum_node_position_error_m"], node_error)
            if node_error or audit["maximum_grip_position_error_m"]:
                raise RuntimeError("USD rod or grip position differs from its stored sample")
            if reset_history:
                reset_history()
            before = get_clock()
            rep.orchestrator.step(rt_subframes=8 if n == 0 else 2, pause_timeline=False, delta_time=0.)
            after = get_clock()
            if before != after:
                raise RuntimeError(f"Capture advanced the Isaac clock: {before} to {after}")
            rgba = np.asarray(camera.get_rgba(device="cpu"))
            if rgba.shape != (*RESOLUTION[::-1], 4) or rgba.dtype != np.uint8 or rgba[:, :, :3].std() < 2:
                raise RuntimeError("Invalid Isaac image")
            name = f"{view}-{index:06d}.png"
            file = dest / name
            Image.fromarray(rgba[:, :, :3]).save(file, compress_level=1)
            audit["frames"].append(dict(view=view, source_index=index,
                                        source_time_s=float(source["times"][index]), file=name,
                                        usd_nodes_sha256=positions_hash(readback), image_sha256=sha256(file)))
            if n % 40 == 0:
                print(f"CAPTURE {case} {view} source {index}", flush=True)
    return audit


def validate_capture(out, case, source, audit, preview):
    if (audit["trajectory_sha256"] != source["trajectory_sha256"]
            or audit["run_report_sha256"] != source["run_report_sha256"]):
        raise ValueError(f"{case}: recording changed during capture")
    for key in ("maximum_node_position_error_m", "maximum_grip_position_error_m",
                "maximum_capture_clock_delta_s", "physics_steps_during_capture"):
        if audit[key] != 0:
            raise ValueError(f"{case}: invalid capture check: {key}")
    expected = [(segment["view"], i) for segment in source["segments"]
                for i in selected_indices(case, segment, preview)]
    if [(f["view"], f["source_index"]) for f in audit["frames"]] != expected:
        raise ValueError(f"{case}: unexpected native frame map")
    for frame in audit["frames"]:
        i = frame["source_index"]
        if (frame["source_time_s"] != source["times"][i]
                or frame["usd_nodes_sha256"] != positions_hash(source["positions"][i])
                or frame["image_sha256"] != sha256(out / case / frame["file"])):
            raise ValueError(f"{case}: mismatched frame {i}")
    return {(f["view"], f["source_index"]): out / case / f["file"] for f in audit["frames"]}


def compose(out, case, source, paths, preview):
    from PIL import Image, ImageDraw, ImageFont

    def font(size, suffix=""):
        return ImageFont.truetype(f"DejaVuSans{suffix}.ttf", size)

    title, description, legend, _, poster_at = CASES[case]
    title_font, copy_font, ui_font = font(30, "-Bold"), font(17), font(15, "-Bold")
    accent = "#79e5ed"
    base = "#101a29"
    width, height = RESOLUTION

    def edit(view, i):
        with Image.open(paths[view, i]) as raw:
            im = raw.convert("RGB")
        d = ImageDraw.Draw(im)
        d.rectangle((0, 0, width, 81), fill=base)
        d.rectangle((0, height-55, width, height), fill=base)
        d.text((31, 10), title, font=title_font, fill="#f2f6fb")
        d.text((33, 50), description, font=copy_font, fill="#c1cfdf")
        number = list(CASES).index(case) + 1
        d.text((width-31, 25), f"DEFORMX BASICS / {number:02}", font=ui_font, fill=accent, anchor="ra")
        d.text((33, height-36), "1×  NORMAL SPEED" if view == "whole" else "⅛×  SLOW REPLAY",
               font=ui_font, fill=accent)
        x = width - 32
        for label in reversed(legend):
            d.text((x, height-36), label, font=copy_font, fill="#dce5f0", anchor="ra")
            x -= d.textlength(label, font=copy_font) + 12
            d.ellipse((x-9, height-32, x, height-23), fill=LEGEND[label])
            x -= 35
        return im

    _, detail = source["segments"]
    indices = detail["source_indices"]
    poster_index = indices[round((len(indices)-1)*poster_at)]
    poster = out / f"{case}-preview.jpg"
    edit("detail", poster_index).save(poster, quality=92, optimize=True)
    thumbnail = dict(file=poster.name, sha256=sha256(poster))
    if preview:
        return None, thumbnail

    video = out / f"{case}.mp4"
    expected_count = detail["end_video_frame_exclusive"]
    args = ["ffmpeg", "-v", "error", "-n", "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{width}x{height}", "-framerate", str(FPS), "-i", "-", "-an",
            "-c:v", "libx264", "-threads", "4", "-crf", "18", "-pix_fmt", "yuv420p",
            # Match the movie and track timescales to the 30 fps presentation clock.
            "-video_track_timescale", "30000", "-movie_timescale", "30000",
            "-movflags", "+faststart", str(video)]
    with subprocess.Popen(args, stdin=subprocess.PIPE) as encoder:
        for segment in source["segments"]:
            previous = None
            for index in segment["source_indices"]:
                if previous != index:
                    pixels = edit(segment["view"], index).tobytes()
                    previous = index
                encoder.stdin.write(pixels)
        encoder.stdin.close()
        if encoder.wait(timeout=120):
            raise RuntimeError(f"{case}: video encode failed")
    subprocess.run(["ffmpeg", "-v", "error", "-xerror", "-err_detect", "explode",
                    "-i", str(video), "-map", "0:v:0", "-f", "null", "-"],
                   check=True, timeout=120)
    probe = json.loads(subprocess.check_output(
        ["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
         "-show_entries", "stream=codec_name,pix_fmt,width,height,avg_frame_rate,nb_read_frames,duration",
         "-of", "json", str(video)], timeout=60))["streams"][0]
    duration = expected_count / FPS
    if (int(probe["nb_read_frames"]) != expected_count or (probe["width"], probe["height"]) != RESOLUTION
            or probe["avg_frame_rate"] != f"{FPS}/1" or probe["codec_name"] != "h264"
            or probe["pix_fmt"] != "yuv420p" or abs(float(probe["duration"]) - duration) > 1e-5):
        raise ValueError(f"{case}: encoded stream disagrees with replay: {probe}")
    return dict(file=video.name, sha256=sha256(video), frame_count=expected_count,
                duration_s=duration, full_decode_ok=True), thumbnail


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=(*CASES, "all"), required=True)
    parser.add_argument("--data", type=Path, required=True,
                        help="Recording directory, or the directory of six recordings with --case all")
    parser.add_argument("--assets", type=Path, required=True, help="Output of blender_props.py")
    parser.add_argument("--out", type=Path, required=True, help="New capture directory")
    parser.add_argument("--preview", action="store_true", help="Capture only three stills per view")
    parser.add_argument("--native", action="store_true", help=argparse.SUPPRESS)
    a = parser.parse_args(argv)
    if a.native:
        if not a.out.is_dir() or {p.name for p in a.out.iterdir()} != {"native.log"}:
            parser.error("native output is not a freshly reserved capture directory")
    elif a.out.exists() or a.out.is_symlink():
        parser.error(f"output already exists: {a.out}")
    a.assets = a.assets.resolve()
    data = ({case: a.data / ("self_contact" if case == "self-contact" else case) for case in CASES}
            if a.case == "all" else {a.case: a.data})
    try:
        sources = {case: load_source(case, path) for case, path in data.items()}
        assets = props_hashes(a.assets)
        if not a.native:
            a.out.mkdir(parents=True)
    except (OSError, ValueError, KeyError) as exc:
        parser.error(str(exc))
    if a.native:
        native(a, sources)
        return 0
    cmd = [sys.executable, "-m", "demo.basics.render", "--native", "--case", a.case,
           "--data", str(a.data.resolve()), "--assets", str(a.assets), "--out", str(a.out.resolve())]
    if a.preview:
        cmd.append("--preview")
    with (a.out / "native.log").open("x") as log:
        proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, timeout=1800)
    if proc.returncode:
        raise RuntimeError(f"Native Isaac exited {proc.returncode}; see {a.out / 'native.log'}")
    native_path = a.out / "native-report.json"
    result = json.loads(native_path.read_text())
    if not result["complete"] or not result["app_close_invoked"] or result["asset_sha256"] != assets:
        raise ValueError(f"Native capture or shutdown incomplete; see {native_path}")
    manifest = dict(schema_version=1, fps=FPS, resolution=RESOLUTION, native_exit_code=proc.returncode,
                    native_report_sha256=sha256(native_path), renderer_sha256=result["renderer_sha256"],
                    asset_sha256=assets, cases={})
    for case, source in sources.items():
        audit = result["cases"][case]
        paths = validate_capture(a.out, case, source, audit, a.preview)
        video, thumbnail = compose(a.out, case, source, paths, a.preview)
        manifest["cases"][case] = dict(
            video=video, thumbnail=thumbnail, segments=source["segments"],
            source=dict(recording_dir=data[case].name, trajectory_sha256=source["trajectory_sha256"],
                        run_report_sha256=source["run_report_sha256"],
                        start_s=float(source["times"][0]), end_s=float(source["times"][-1])),
            native_frame_count=len(audit["frames"]),
            **{key: audit[key] for key in ("maximum_node_position_error_m", "maximum_grip_position_error_m",
                                          "maximum_capture_clock_delta_s", "physics_steps_during_capture",
                                          "visual_radius_m")},
        )
        print(f"{case}: {'preview' if a.preview else video}", flush=True)
    write_json(a.out / "manifest.json", manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
