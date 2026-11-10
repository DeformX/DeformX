#!/usr/bin/env python3
"""Replay the existing braid with a descent, a five-second hold on the crossings, and a pullback.

Activate Isaac Sim Python and put FFmpeg and FFprobe on PATH first. Use the
recorded renderer source tree, nominal run, Blender assets and original
native capture. --lock accepts the GPU lock shared by jobs on the host.
"""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys


RENDER_SHA256 = {
    "render.py": "73841737b2d6a084e0b53f14792c796f6ae34183b0c12ca7e484f9f610340870",
    "render_native.py": "a9c5beb3ba949db411777f721ed317591322e7544033ea9e6a1c93e275c08c87",
}
TRAJECTORY_SHA256 = "71b2623c0ba8e44d5c854b63a73bf2c9446d80fe7bf5517aabb378bc0ecc3ce2"
ASSET_MANIFEST_SHA256 = "0383b89508bac06861a4ccecf77d9c6f2b5fad46f4ccaf574d65289bbf0718ff"
ORIGINAL_ASSET_MANIFEST_SHA256 = "b5d38ee1b6313e876c6ca25b25fed7115a2dfc4caaa9ffd3170bd8f4f45efd16"
ORIGINAL_VIDEO_SHA256 = "ea2025caad05641bb9d968236f8d5b8b019aca8701f5848916601202389d4cc8"
STUDIO_DOME = '        dome = UsdLux.DomeLight.Define(stage, "/World/Lights/StudioAmbient")'
STUDIO_INTENSITY = 600
STUDIO_SELECTION = '''        hardware = stage.GetPrimAtPath("/World/Hardware")
        authored_domes = [prim for prim in Usd.PrimRange(hardware) if prim.IsA(UsdLux.DomeLight)]
        expected_domes = [f"/World/Hardware/{name}/env_light"
                          for name in ("Frame", "Carrier_0", "Carrier_1", "Carrier_2")]
        if {str(prim.GetPath()) for prim in authored_domes} != set(expected_domes):
            raise ValueError("unexpected Blender environment lights in the braider assets")
        # Keep the explicit studio dome selected, matching the previous published replay.
        for prim in authored_domes:
            prim.SetActive(False)
        report["studio_environment"] = {
            "selected_dome": "/World/Lights/StudioAmbient",
            "inactive_blender_domes": expected_domes,
            "studio_ambient_intensity": STUDIO_INTENSITY,
        }
'''.replace("STUDIO_INTENSITY", str(STUDIO_INTENSITY))


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_hash(path, expected):
    if sha256(path) != expected:
        raise ValueError(f"Unexpected source bytes for {path.name}")


def replace_once(source, before, after):
    if source.count(before) != 1:
        raise ValueError(f"Renderer source does not match expected section {before[:45]!r}")
    return source.replace(before, after)


def clean_native_source(source):
    # Render RGB directly, with zero-time capture and recorded pose checks unchanged.
    start = source.index("def annotate(")
    end = source.index("def capture(", start)
    source = source[:start] + source[end:]
    source = replace_once(source, "import numpy as np\n", "import numpy as np\nfrom PIL import Image\n")
    source = replace_once(
        source,
        '            ("BlueFill", (44, 52, -73), 700, (.36, .67, 1.)),',
        '''            ("BlueFill", (44, 52, -73), 700, (.36, .67, 1.)),
            ("FrontFill", (90, 0, 0), 1100, (.78, .88, 1.)),''',
    )
    source = replace_once(
        source, '        camera.prim.GetAttribute("focalLength").Set(33.)',
        '''        camera.prim.GetAttribute("focalLength").Set(33.)
        camera.prim.GetAttribute("horizontalAperture").Set(20.955)
        camera.prim.GetAttribute("verticalAperture").Set(20.955 * height / width)''',
    )
    source = replace_once(
        source,
        '        reference("/World/Hardware/Frame", "frame.usdc", "/Frame")',
        '''        reference("/World/Hardware/Frame", "frame.usdc", "/Frame")
        frame_root = stage.GetPrimAtPath("/World/Hardware/Frame")
        lettering = [prim for prim in Usd.PrimRange(frame_root) if prim.GetName() == "FrontLogo"]
        if len(lettering) != 1:
            raise ValueError("expected exactly one authored lettering prim in the original fixture")
        UsdGeom.Imageable(lettering[0]).MakeInvisible()
        report["hidden_visual_lettering_prim"] = str(lettering[0].GetPath())
        report["image_overlay"] = False''',
    )
    source = replace_once(source, STUDIO_DOME, STUDIO_SELECTION + STUDIO_DOME)
    source = replace_once(source, "dome.CreateIntensityAttr(495)",
                          f"dome.CreateIntensityAttr({STUDIO_INTENSITY})")
    start = source.index("        extra = cfg.carrier_z_m")
    end = source.index("\n        def aim", start)
    source = source[:start] + '''        detail_target = np.array([0., 0., cfg.carrier_z_m + .03])
        detail_eye = detail_target + np.array([.24, -1.65, .18])
        braid_target = np.array([0., 0., 2.45])
        braid_eye = braid_target + (detail_eye - detail_target)
        # Fit the complete authored machine, excluding the studio environment.
        bounds = []
        bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
        for prim in Usd.PrimRange(frame_root):
            names = str(prim.GetPath()).split("/")
            if not prim.IsA(UsdGeom.Mesh) or any(
                name in ("StudioFloor", "StudioBackdrop", "FrontLogo") for name in names
            ):
                continue
            box = bbox_cache.ComputeWorldBound(prim).ComputeAlignedRange()
            if not box.IsEmpty():
                bounds.extend((list(box.GetMin()), list(box.GetMax())))
        if not bounds:
            raise ValueError("no physical machine meshes found for whole-view framing")
        low, high = np.min(bounds, axis=0), np.max(bounds, axis=0)
        corners = np.array([[x, y, z] for x in (low[0], high[0])
                            for y in (low[1], high[1]) for z in (low[2], high[2])])
        whole_target = (low + high) / 2.
        focal = float(camera.prim.GetAttribute("focalLength").Get())
        aperture = np.array([float(camera.prim.GetAttribute(name).Get())
                             for name in ("horizontalAperture", "verticalAperture")])

        def project_bounds(eye, target):
            view = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*target), Gf.Vec3d(0, 0, 1))
            q = np.array([list(view.Transform(Gf.Vec3d(*point))) for point in corners])
            if np.any(q[:, 2] >= 0.):
                raise ValueError("machine bound is behind the camera")
            return (2. * focal * q[:, :2] / -q[:, 2, None]) / aperture

        distance = 6.0
        for _ in range(100):
            whole_eye = whole_target + distance * np.array([.07, -1., .085])
            ndc = project_bounds(whole_eye, whole_target)
            if float(np.max(np.abs(ndc))) <= .95:
                break
            distance *= 1.01
        else:
            raise ValueError("complete machine did not fit inside the framing margin")
        report["camera_framing"] = {
            "native_focal_length_mm": focal,
            "native_aperture_mm": aperture.tolist(),
            "detail": {"subject": "three moving top carriers", "eye_m": detail_eye.tolist(), "target_m": detail_target.tolist()},
            "whole": {"eye_m": whole_eye.tolist(), "target_m": whole_target.tolist()},
            "machine_bounds_m": [low.tolist(), high.tolist()],
            "whole_machine_ndc_bounds": [ndc.min(axis=0).tolist(), ndc.max(axis=0).tolist()],
            "max_abs_whole_machine_ndc": float(np.max(np.abs(ndc))),
            "full_machine_ndc_limit": .95,
            "descent_start_playback_s": 1.,
            "descent_end_playback_s": 4.,
            "braid_detail": {"eye_m": braid_eye.tolist(), "target_m": braid_target.tolist()},
            "detail_hold_playback_s": [4., 9.],
            "pullback_start_playback_s": 9.,
            "pullback_end_playback_s": 13.,
            "transition": "quintic smootherstep; one continuous camera shot",
            "up_vector": [0., 0., 1.],
            "front_fill": {"intensity": 1100, "rgb": [.78, .88, 1.], "rotation_xyz_deg": [90, 0, 0]},
        }
''' + source[end:]
    start = source.index("            if extra > 0:")
    end = source.index("            look =", start)
    source = source[:start] + '''            up = Gf.Vec3d(0, 0, 1)
            def ease(value):
                u = float(np.clip(value, 0., 1.))
                return u * u * u * (u * (u * 6. - 15.) + 10.)
            if phase < 9.:
                blend = ease((phase - 1.) / 3.)
                eye = (1. - blend) * detail_eye + blend * braid_eye
                target = (1. - blend) * detail_target + blend * braid_target
                view = "carriers" if phase < 1. else ("descend" if phase < 4. else "braid_detail")
            else:
                blend = ease((phase - 9.) / 4.)
                eye = (1. - blend) * braid_eye + blend * whole_eye
                target = (1. - blend) * braid_target + blend * whole_target
                view = "whole" if phase >= 13. else "pullback"
            report["last_camera_eye_m"] = eye.tolist()
            report["last_camera_target_m"] = target.tolist()
''' + source[end:]
    source = replace_once(source, "phase = (t - cfg.settle_s) / cfg.exchange_s",
                          "phase = round((t - cfg.settle_s) / cfg.exchange_s, 12)")
    start = source.index("            checkpoints =")
    end = source.index("            png =", start)
    source = source[:start] + "            image = Image.fromarray(rgb)\n" + source[end:]
    source = replace_once(source,
        'state_sha256=frame_hash(p, roots), view=view))',
        'state_sha256=frame_hash(p, roots), view=view, camera_eye_m=report["last_camera_eye_m"], camera_target_m=report["last_camera_target_m"]))')
    return source


def prepare_source(source_root, destination, preview):
    demo = source_root / "demo"
    original = demo / "braiding_machine"
    for name, expected in RENDER_SHA256.items():
        require_hash(original / name, expected)
    target = destination / "demo"
    target.mkdir(parents=True)
    shutil.copy2(demo / "__init__.py", target)
    for name in ("braiding_machine", "showcase"):
        package = target / name
        package.mkdir()
        for path in (demo / name).iterdir():
            if path.is_file() and (path.suffix == ".py" or path.name == "criteria.json"):
                shutil.copy2(path, package)
    worker = target / "braiding_machine/render_native.py"
    worker.write_text(clean_native_source(worker.read_text()))
    if preview:
        driver = worker.with_name("render.py")
        driver.write_text(replace_once(
            driver.read_text(),
            "indices = indices[np.linspace(0, len(indices) - 1, 5).astype(int)]",
            "indices = indices[[0, 45, 90, 120, 195, 269, 300, 345, 390, 629]]",
        ))
    return worker


def verify_and_collect(options, worker, capture, old_native):
    new_native_path = capture / "native_report.json"
    new_native = json.loads(new_native_path.read_text())
    verified = json.loads((capture / "verification.json").read_text())
    keys = ("index", "t_s", "state_sha256")
    old_timeline = [[frame[key] for key in keys] for frame in old_native["frames"]]
    new_timeline = [[frame[key] for key in keys] for frame in new_native["frames"]]
    if len(old_timeline) != 630 or new_timeline != old_timeline:
        raise ValueError("clean capture does not match all original state and source-time samples")
    if new_native["image_overlay"] or new_native["physics_stepping"]:
        raise ValueError("clean capture used an overlay or advanced physics")
    if new_native["source_sha256"]["render_native.py"] != sha256(worker):
        raise ValueError("unexpected native worker executed")
    if new_native["asset_manifest_sha256"] != ASSET_MANIFEST_SHA256:
        raise ValueError("capture did not use the verified wider Blender frame")
    framing = new_native["camera_framing"]
    view_counts = {"carriers": 30, "descend": 90, "braid_detail": 150, "pullback": 120, "whole": 240}
    expected_views = [view for view, count in view_counts.items() for _ in range(count)]
    if [frame["view"] for frame in new_native["frames"]] != expected_views:
        raise ValueError("camera phases do not match the intended 21-second playback schedule")
    if framing["max_abs_whole_machine_ndc"] > .95:
        raise ValueError("complete machine must fit inside the image for the final eight seconds")
    if verified["native_exit"] != 0 or verified["playback_speed"] != 1.0 or verified["capture_clock_delta_max_s"] != 0:
        raise ValueError("capture did not replay at 1x with an unchanged simulation clock and normal renderer exit")
    release = options.out / "delivery"
    release.mkdir()
    poster = release / "preview-process-clean.jpg"
    shutil.copy2(capture / "preview.jpg", poster)
    result = {
        "original": {
            "trajectory_sha256": TRAJECTORY_SHA256,
            "video_sha256": ORIGINAL_VIDEO_SHA256,
            "native_report_sha256": sha256(options.original_native / "native_report.json"),
            "renderer_source_sha256": RENDER_SHA256,
            "fixture_manifest_sha256": ORIGINAL_ASSET_MANIFEST_SHA256,
        },
        "clean_capture": {
            "matches_all_original_timestamps_and_saved_states": True,
            "frames": len(new_timeline),
            "source_first_s": new_timeline[0][1],
            "source_last_s": new_timeline[-1][1],
            "playback_speed": verified["playback_speed"],
            "capture_clock_delta_max_s": verified["capture_clock_delta_max_s"],
            "max_render_position_error_m": verified["max_render_position_error_m"],
            "physics_stepping": False,
            "image_overlay": False,
            "hidden_visual_lettering_prim": new_native["hidden_visual_lettering_prim"],
            "camera_framing": new_native["camera_framing"],
            "studio_environment": new_native["studio_environment"],
            "complete_machine_framing_verified": True,
            "whole_view_frames": view_counts["whole"],
            "camera_view_frame_counts": view_counts,
            "fixture_manifest_sha256": new_native["asset_manifest_sha256"],
            "fixture_asset_sha256": new_native["asset_sha256"],
            "frame_presentation": json.loads((options.assets / "manifest.json").read_text())["presentation"],
            "native_report_sha256": sha256(new_native_path),
            "native_worker_sha256": sha256(worker),
            "recipe_sha256": sha256(Path(__file__)),
        },
        "poster": {
            "path": poster.name,
            "source_frame": len(new_timeline) - 1,
            "source_time_s": new_timeline[-1][1],
            "sha256": sha256(poster),
        },
    }
    if not options.preview:
        video_info = verified["video"]
        if not video_info["full_decode"] or (
            video_info["frames"], video_info["fps"], video_info["duration_s"]
        ) != (630, 30, 21.):
            raise ValueError("unexpected decoded video length or cadence")
        video = release / "braiding-process-clean.mp4"
        shutil.copy2(capture / "braiding-machine.mp4", video)
        require_hash(video, video_info["sha256"])
        result["video"] = {"path": video.name, "bytes": video.stat().st_size, **video_info}
    (release / "braiding-process-clean-provenance.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-root", "data", "assets", "original-native", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--lock", type=Path)
    options = parser.parse_args()
    for name in ("source_root", "data", "assets", "original_native", "out", "lock"):
        value = getattr(options, name)
        if value is not None:
            setattr(options, name, value.resolve())
    require_hash(options.data / "trajectory.npz", TRAJECTORY_SHA256)
    require_hash(options.assets / "manifest.json", ASSET_MANIFEST_SHA256)
    presentation = json.loads((options.assets / "manifest.json").read_text())["presentation"]
    if (presentation["original_manifest_sha256"] != ORIGINAL_ASSET_MANIFEST_SHA256
            or not presentation["protected_geometry_unchanged"] or not presentation["carrier_assets_unchanged"]):
        raise ValueError("expected the audited frame-only Blender revision")
    require_hash(options.original_native / "braiding-machine.mp4", ORIGINAL_VIDEO_SHA256)
    old_native = json.loads((options.original_native / "native_report.json").read_text())
    if old_native["source_sha256"] != RENDER_SHA256 or old_native["trajectory_sha256"] != TRAJECTORY_SHA256:
        raise ValueError("original native capture does not bind the expected renderer and trajectory")
    options.out.mkdir(parents=True)
    tree = options.out / "source"
    worker = prepare_source(options.source_root, tree, options.preview)
    capture = options.out / "capture"
    command = [
        sys.executable, "-m", "demo.braiding_machine.render",
        "--data", str(options.data), "--assets", str(options.assets), "--out", str(capture),
    ]
    if options.preview:
        command.append("--preview")
    if options.lock:
        command[:0] = ["flock", "-o", "-w", "1800", str(options.lock)]
    subprocess.run(command, cwd=tree, stdin=subprocess.DEVNULL, check=True)
    if options.preview:
        print(capture)
    else:
        verify_and_collect(options, worker, capture, old_native)


if __name__ == "__main__":
    main()
