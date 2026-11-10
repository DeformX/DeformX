"""Capture native Isaac frames from a rack insertion run and verify an MP4 export."""

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

from .assets import sha
from .audit import verify_recording


def verify_trajectory(path, expected_frames):
    import numpy as np

    with np.load(path, allow_pickle=False) as trajectory:
        times = trajectory["t"]
        if (times.shape != (expected_frames,) or expected_frames < 2
                or not np.allclose(np.diff(times), 1 / 30, atol=1e-8, rtol=0)):
            raise ValueError("Native replay requires a uniform finite 30 Hz trajectory")
        for name in ("robot_q", "body_pos", "body_quat", "plug_pos", "plug_quat",
                     "latch_pos", "latch_quat", "cable"):
            values = trajectory[name]
            if len(values) != expected_frames or not np.isfinite(values).all():
                raise ValueError(f"Invalid replay states in {name}")
        for name in ("body_quat", "plug_quat", "latch_quat"):
            if not np.allclose(np.linalg.norm(trajectory[name], axis=-1), 1, atol=1e-4):
                raise ValueError(f"Invalid replay quaternions in {name}")
        names = trajectory["body_names"].tolist()
        if len(names) != len(set(names)):
            raise ValueError("Duplicate robot body names in the replay trajectory")


def encode(folder, records, width, height):
    video = folder / "rack-insertion.mp4"
    subprocess.run([
        "ffmpeg", "-nostdin", "-v", "error", "-n", "-framerate", "30",
        "-i", str(folder / "frames/%06d.png"), "-frames:v", str(len(records)),
        "-an", "-c:v", "libx264", "-threads", "8", "-preset", "medium", "-crf", "18",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(video),
    ], check=True)
    subprocess.run([
        "ffmpeg", "-nostdin", "-v", "error", "-xerror", "-err_detect", "explode", "-threads", "2",
        "-i", str(video), "-f", "null", "-",
    ], check=True)
    metadata = json.loads(subprocess.run([
        "ffprobe", "-v", "error", "-show_streams", "-show_frames",
        "-show_entries",
        "stream=codec_type,codec_name,width,height,r_frame_rate,pix_fmt:frame=best_effort_timestamp_time",
        "-of", "json", str(video),
    ], check=True, capture_output=True, text=True).stdout)
    stream, = metadata["streams"]
    start = records[0]["actual_t_s"]
    errors = [
        abs(float(frame["best_effort_timestamp_time"]) - (row["actual_t_s"] - start))
        for frame, row in zip(metadata["frames"], records, strict=True)
    ]
    if (max(errors) > 1e-6 or stream["r_frame_rate"] != "30/1"
            or stream["codec_name"] != "h264" or stream["pix_fmt"] != "yuv420p"
            or (stream["width"], stream["height"]) != (width, height)):
        raise ValueError("Encoded video does not match the native replay clock or dimensions")
    return {
        "file": video.name, "sha256": sha(video), "bytes": video.stat().st_size,
        "codec": "h264", "width": width, "height": height, "frames": len(records),
        "fps": 30, "duration_s": len(records) / 30, "full_decode": True,
        "max_relative_pts_error_s": max(errors),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Simulation output directory")
    parser.add_argument("--out", type=Path, required=True, help="New native replay output directory")
    parser.add_argument("--preview", action="store_true", help="Capture stills at key phases")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    args = parser.parse_args()
    try:
        run = verify_recording(args.data.resolve())
        if any(size <= 0 or size % 2 for size in (args.width, args.height)):
            raise ValueError("Use positive even image dimensions")
        if not args.preview and not all(shutil.which(tool) for tool in ("ffmpeg", "ffprobe")):
            raise ValueError("Install ffmpeg and ffprobe or use --preview for stills")
        if args.out.exists():
            raise ValueError(f"Choose a new render output directory: {args.out}")
        verify_trajectory(args.data / "trajectory.npz", run["recorded_frames"])
    except (OSError, ValueError, KeyError, ImportError) as exc:
        parser.error(str(exc))
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=False)
    command = [sys.executable]
    if sys.flags.no_site:
        command.append("-S")
    command.extend([
        "-u", "-m", "demo.rack_insertion.render_native", "--data", str(args.data.resolve()),
        "--out", str(args.out), "--width", str(args.width), "--height", str(args.height),
    ])
    if args.preview:
        command.append("--preview")
    with (args.out / "native.log").open("w") as log:
        child = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
    if child.returncode:
        raise RuntimeError(f"Native capture exited {child.returncode}. Inspect {args.out / 'native.log'}")
    report = json.loads((args.out / "native_report.json").read_text())
    records = json.loads((args.out / "frames.json").read_text())
    if (not report["complete"] or report["errors"] or not report["app_close_invoked"]
            or not report["physics_replay_disabled"] or report["capture_clock_delta_max_s"] != 0.
            or report["trajectory_sha256"] != run["trajectory_sha256"]
            or report["frames"] != len(records) or not records):
        raise ValueError("Native capture did not satisfy replay checks")
    for row in records:
        if sha(args.out / row["file"]) != row["sha256"]:
            raise ValueError(f"Native frame checksum mismatch for {row['file']}")
    report.update(native_exit=child.returncode, frames_verified=len(records))
    if not args.preview:
        if [row["trajectory_index"] for row in records] != list(range(run["recorded_frames"])):
            raise ValueError("Capture did not replay the complete source trajectory")
        report["video"] = encode(args.out, records, args.width, args.height)
    (args.out / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Verified {len(records)} native frames in {args.out}")


if __name__ == "__main__":
    main()
