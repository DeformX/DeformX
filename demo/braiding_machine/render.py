"""Render a verified 1x Isaac replay of the recorded braiding machine."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from demo.showcase.simulate import sha, write
from .analyze import CRITERIA, diagnostics, endpoint_error, gates, read_run
from .model import CARRIER_Z_M, Settings

FPS = 30
RESOLUTION = (1280, 720)


def replay_indices(t, start_s, fps):
    """Select a uniform, exact source grid after conditioning, without interpolation."""
    stride = (1 / fps) / (t[1] - t[0])
    i = int(np.argmin(abs(t - start_s)))
    if not np.isclose(t[i], start_s, atol=1e-10, rtol=0) or not np.isclose(stride, round(stride)):
        raise ValueError("source timestamps and playback cadence must be aligned")
    j = np.arange(i, len(t) - 1, round(stride))
    if not len(j) or not np.allclose(t[j], start_s + np.arange(len(j)) / fps, atol=1e-10, rtol=0):
        raise ValueError("source timestamps and playback cadence must be aligned")
    return j


def frame_hash(positions, carriers):
    return hashlib.sha256(positions.tobytes() + carriers.tobytes()).hexdigest()


def preflight(data_path, assets_path, preview):
    run, data = read_run(data_path)
    cfg = Settings(**run["settings"])
    if not run.get("task_success") or run.get("criteria_sha256") != sha(CRITERIA):
        raise ValueError("the simulation must pass the frozen braid gates")
    metrics = diagnostics(data["t"], data["positions"], cfg)
    metrics["max_endpoint_error_m"] = endpoint_error(data["positions"], data["carriers"])
    if not all(gates(metrics, cfg, True, run["drive"], run["contact"]).values()):
        raise ValueError("the saved states do not pass the frozen braid gates")
    manifest = json.loads((assets_path / "manifest.json").read_text())
    usd_names = ("frame.usdc", *(f"carrier_{i}.usdc" for i in range(3)))
    if (manifest["units"] != "meters"
            or manifest.get("carrier_z_m", CARRIER_Z_M) != cfg.carrier_z_m
            or manifest["model_source_sha256"] != run["source_sha256"]["model.py"]
            or not all(sha(assets_path / name) == manifest["files"][name] for name in usd_names)):
        raise ValueError("Blender fixture assets do not match the run or manifest")
    indices = replay_indices(data["t"], cfg.settle_s, FPS)
    if preview:
        indices = indices[np.linspace(0, len(indices) - 1, 5).astype(int)]
    return run, data, cfg, metrics, indices


def check_native(out, run, data, indices):
    report = json.loads((out / "native_report.json").read_text())
    if (
        not report["complete"] or report["errors"] or not report["app_close_invoked"]
        or report["capture_clock_delta_max_s"] != 0
        or report["max_render_position_error_m"] != 0
        or report["trajectory_sha256"] != run["trajectory_sha256"]
        or report["amplitude_scale"] != 1.
        or len(report["frames"]) != len(indices)
    ):
        raise RuntimeError("native capture failed verification")
    for frame, (row, j) in enumerate(zip(report["frames"], indices)):
        path = out / "frames" / f"{frame:06d}.png"
        if (row["index"] != j or row["t_s"] != data["t"][j] or row["image_sha256"] != sha(path)
                or row["state_sha256"] != frame_hash(data["positions"][j], data["carriers"][j])):
            raise RuntimeError(f"frame {frame} does not match the verified source")
    return report


def encode(out, source_time):
    video = out / "braiding-machine.mp4"
    subprocess.run([
        "ffmpeg", "-v", "error", "-framerate", str(FPS), "-i", str(out / "frames" / "%06d.png"),
        "-an", "-c:v", "libx264", "-preset", "medium", "-crf", "17", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", str(video),
    ], check=True, stdin=subprocess.DEVNULL)
    subprocess.run(["ffmpeg", "-v", "error", "-xerror", "-err_detect", "explode",
                    "-i", str(video), "-f", "null", "-"], check=True, stdin=subprocess.DEVNULL)
    probe = subprocess.check_output([
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_streams", "-show_frames",
        "-show_entries", "stream=width,height,r_frame_rate,duration:frame=best_effort_timestamp_time",
        "-of", "json", str(video),
    ], text=True)
    decoded = json.loads(probe)
    pts = np.array([float(f["best_effort_timestamp_time"]) for f in decoded["frames"]])
    stream, = decoded["streams"]
    if (len(pts) != len(source_time) or (stream["width"], stream["height"]) != RESOLUTION
            or stream["r_frame_rate"] != f"{FPS}/1"):
        raise RuntimeError("encoded video metadata does not match the replay")
    time_error = float(np.max(np.abs(pts - (source_time - source_time[0]))))
    if time_error > 1e-6:
        raise RuntimeError("decoded video PTS do not match the source clock")
    return dict(full_decode=True, frames=len(pts), fps=FPS, duration_s=float(stream["duration"]),
                width=stream["width"], height=stream["height"], relative_pts_error_s=time_error,
                sha256=sha(video))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="completed run directory")
    parser.add_argument("--assets", type=Path, required=True, help="Blender-generated asset directory")
    parser.add_argument("--out", type=Path, required=True, help="fresh render directory")
    parser.add_argument("--preview", action="store_true", help="capture five source frames and skip video encoding")
    parser.add_argument("--native-child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    run, data, cfg, metrics, indices = preflight(args.data, args.assets, args.preview)
    if args.native_child:
        from .render_native import capture
        return capture(args, run, data, cfg, metrics, indices)
    try:
        args.out.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        parser.error("output already exists; choose a fresh directory")
    cmd = [
        sys.executable, "-m", "demo.braiding_machine.render", "--native-child",
        "--data", str(args.data), "--assets", str(args.assets), "--out", str(args.out),
    ]
    if args.preview:
        cmd.append("--preview")
    with (args.out / "native.log").open("w") as log:
        child = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT,
                               stdin=subprocess.DEVNULL, timeout=3600)
    if child.returncode:
        raise RuntimeError(f"Isaac exited with code {child.returncode}. See {args.out / 'native.log'}")
    native = check_native(args.out, run, data, indices)
    verified = dict(
        complete=True, native_exit=child.returncode, source_frames=len(indices), playback_speed=1.,
        trajectory_sha256=run["trajectory_sha256"], capture_clock_delta_max_s=native["capture_clock_delta_max_s"],
        max_render_position_error_m=native["max_render_position_error_m"],
        native_report_sha256=sha(args.out / "native_report.json"),
    )
    from PIL import Image
    poster = len(indices) - 1
    with Image.open(args.out / "frames" / f"{poster:06d}.png") as image:
        image.save(args.out / "preview.jpg", quality=96)
    if not args.preview:
        verified["video"] = encode(args.out, data["t"][indices])
    write(args.out / "verification.json", verified)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
