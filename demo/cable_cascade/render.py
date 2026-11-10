"""Render a completed cascade recording with the native Isaac Sim camera."""

import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

from .audit import load_trajectory
from .model import FPS, RADIUS, sha, write_json


def even_size(text):
    value = int(text)
    if value < 2 or value % 2:
        raise argparse.ArgumentTypeError("must be a positive even pixel count")
    return value


def video_command(*argv, **kwargs):
    return subprocess.run(argv, stdin=subprocess.DEVNULL, check=True, **kwargs)


def verify_video(video, size, frames):
    video_command(
        "ffmpeg", "-nostdin", "-v", "error", "-xerror", "-err_detect", "explode",
        "-threads", "2", "-i", str(video), "-f", "null", "-",
    )
    decoded = json.loads(video_command(
        "ffprobe", "-v", "error", "-show_streams", "-show_frames",
        "-show_entries", "stream=codec_type,codec_name,width,height,avg_frame_rate,time_base,duration:frame=pts",
        "-of", "json", str(video), capture_output=True, text=True,
    ).stdout)
    stream, = decoded["streams"]
    if (
        stream["codec_type"] != "video" or stream["codec_name"] != "h264"
        or stream["avg_frame_rate"] != f"{FPS}/1"
        or (stream["width"], stream["height"]) != size
    ):
        raise ValueError("unexpected video codec, frame rate, or resolution")
    ticks = Fraction(stream["time_base"])
    pts = [row["pts"] * ticks for row in decoded["frames"]]
    if pts != [Fraction(i, FPS) for i in range(frames)]:
        raise ValueError("video presentation timestamps disagree with the recording")
    return float(stream["duration"])


def render(args):
    for executable in ("ffmpeg", "ffprobe"):
        if not shutil.which(executable):
            raise RuntimeError(f"install FFmpeg with libx264 and make {executable} available on PATH")

    run, times, positions = load_trajectory(args.data)
    out = args.out
    out.mkdir(parents=True)
    report = {"complete": False, "source_trajectory_sha256": run["trajectory_sha256"]}
    report_path = out / "render-report.json"
    try:
        with (out / "native.log").open("w") as log:
            child = subprocess.run(
                [sys.executable, "-u", "-m", "demo.cable_cascade._native",
                 "--data", str(args.data), "--out", str(out),
                 "--width", str(args.width), "--height", str(args.height)],
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
            )
        report["native_child_exit_code"] = child.returncode
        if child.returncode != 0:
            raise RuntimeError(f"native capture exited {child.returncode}; see {out / 'native.log'}")
        native_path = out / "native-report.json"
        native = json.loads(native_path.read_text())
        if (
            native.get("complete") is not True or native.get("app_close_invoked") is not True
            or native["clock_delta_max_s"] != 0. or native["frames"] != len(times)
            or native["visual_radius_m"] != RADIUS
            or native["trajectory_sha256"] != run["trajectory_sha256"]
        ):
            raise ValueError("native capture report is incomplete or inconsistent")
        frame_map = out / "frame-map.json"
        rows = json.loads(frame_map.read_text())
        if len(rows) != len(times):
            raise ValueError("native frame count disagrees with the recording")
        for i, (row, t, p) in enumerate(zip(rows, times, positions, strict=True)):
            path = f"frames/{i:06d}.png"
            if (
                row["trajectory_index"] != i or row["file"] != path
                or row["source_time_s"] != float(t)
                or row["state_sha256"] != hashlib.sha256(p.tobytes()).hexdigest()
                or row["png_sha256"] != sha(out / path)
            ):
                raise ValueError(f"native frame {i} does not match the recording or its hash")
        video = out / "cable-cascade.mp4"
        video_command(
            "ffmpeg", "-nostdin", "-v", "error", "-n", "-framerate", str(FPS),
            "-i", str(out / "frames/%06d.png"), "-an", "-c:v", "libx264", "-threads", "6",
            "-preset", "medium", "-crf", "17", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(video),
        )
        duration = verify_video(video, (args.width, args.height), len(times))
        poster_frame = min(27, len(times) - 1)
        poster = out / "preview.jpg"
        video_command(
            "ffmpeg", "-nostdin", "-v", "error", "-n", "-i", str(out / rows[poster_frame]["file"]),
            "-frames:v", "1", "-q:v", "2", "-update", "1", str(poster),
        )
        report.update(
            complete=True, frames=len(times), fps=FPS, resolution=[args.width, args.height],
            simulation_interval_s=[float(times[0]), float(times[-1])],
            encoded_duration_s=duration, exact_pts_verified=True, full_decode=True,
            native_report_sha256=sha(native_path), frame_map_sha256=sha(frame_map),
            video_sha256=sha(video), poster_frame=poster_frame, poster_sha256=sha(poster),
            source_sha256={name: sha(Path(__file__).with_name(name)) for name in ("render.py", "_native.py")},
        )
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        write_json(report_path, report)
    print(f"Rendered {len(times)} frames to {video}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="output directory from cascade run")
    parser.add_argument("--out", type=Path, required=True, help="fresh render directory")
    parser.add_argument("--width", type=even_size, default=1280)
    parser.add_argument("--height", type=even_size, default=720)
    args = parser.parse_args(argv)
    if args.out.exists():
        parser.error("--out must be a fresh directory")
    try:
        render(args)
    except (OSError, ImportError, RuntimeError, TypeError, ValueError, KeyError, subprocess.SubprocessError) as error:
        parser.exit(1, f"Cascade rendering failed: {error}\nSee demo/cable_cascade/README.md for Isaac Sim setup.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
