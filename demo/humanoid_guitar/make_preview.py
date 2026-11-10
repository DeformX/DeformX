"""Make an eight-second close-up → pullback → overview preview on the render host.

python demo/humanoid_guitar/make_preview.py --out /path/to/fresh/preview
Uses the packaged recording; does not run physics or interpolate motion.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

SOURCE_SHA256 = "6776de6c8623cd4c9cae3d9aefb6b7d8d870ff8328a55112e1db10f6867723b5"
SOURCE = "demo/humanoid_guitar/media/humanoid-guitar-fullbody-long.mp4"
# The recording's first 184 frames run at 0.25x. Keep every second frame
# for a 0.5x opening, followed by 148 frames of normal-speed performance.
SOURCE_FRAMES = list(range(0, 184, 2)) + list(range(184, 332))


def command(*args):
    return subprocess.check_output([str(x) for x in args], stderr=subprocess.PIPE)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def frame_hashes(path, filters=None):
    args = ["ffmpeg", "-nostdin", "-v", "error", "-threads", "2", "-i", path]
    if filters:
        args += ["-vf", filters]
    output = command(*args, "-an", "-f", "framemd5", "-").decode()
    return [line.rsplit(",", 1)[1].strip() for line in output.splitlines()
            if line and not line.startswith("#")]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--out", type=Path, required=True)
    a = parser.parse_args()
    source = a.root / SOURCE
    assert sha(source) == SOURCE_SHA256
    a.out.mkdir(parents=True, exist_ok=False)
    video = a.out / "humanoid-guitar-preview.mp4"
    filters = ("select='lt(n,184)*not(mod(n,2))+gte(n,184)*lt(n,332)',"
               "setpts=N/(30*TB)")
    with tempfile.TemporaryDirectory(dir=a.out) as tmp:
        master = Path(tmp) / "preview.mkv"
        command("ffmpeg", "-nostdin", "-v", "error", "-n", "-threads", 2,
                "-i", source, "-vf", filters, "-an", "-map_metadata", -1,
                "-c:v", "ffv1", "-threads", 2, master)
        hashes = frame_hashes(source)
        assert len(hashes) == 478
        assert frame_hashes(master) == [hashes[i] for i in SOURCE_FRAMES]
        command("ffmpeg", "-nostdin", "-v", "error", "-n", "-threads", 2,
                "-i", master, "-an", "-map_metadata", -1, "-c:v", "libx264",
                "-preset", "medium", "-crf", 17, "-threads", 2,
                "-pix_fmt", "yuv420p", "-movflags", "+faststart", video)
    command("ffmpeg", "-nostdin", "-v", "error", "-xerror", "-err_detect",
            "explode", "-i", video, "-f", "null", "-")
    probe = json.loads(command("ffprobe", "-v", "error", "-count_frames",
        "-show_streams", "-show_format", "-of", "json", video))
    stream, = probe["streams"]
    assert (stream["width"], stream["height"], stream["avg_frame_rate"],
            int(stream["nb_read_frames"])) == (1280, 720, "30/1", 240)
    assert abs(float(probe["format"]["duration"]) - 8) < .001
    segments = [
        dict(chapter="closeup", output_frames=[0, 60], source_frames=[0, 120],
             source_stride=2, simulation_time_multiplier=.5),
        dict(chapter="pullback", output_frames=[60, 92], source_frames=[120, 184],
             source_stride=2, simulation_time_multiplier=.5),
        dict(chapter="overview", output_frames=[92, 240], source_frames=[184, 332],
             source_stride=1, simulation_time_multiplier=1),
    ]
    evidence = a.root / "demo/media/clean-replay-provenance.json"
    report = dict(source=SOURCE, source_sha256=SOURCE_SHA256,
        recording_evidence=str(evidence.relative_to(a.root)),
        recording_evidence_sha256=sha(evidence),
        recipe_sha256=sha(Path(__file__)), video=video.name, video_sha256=sha(video),
        encoding_host="ps", duration_s=8, frames=240, fps=30, dimensions=[1280, 720],
        segments=segments, frame_ranges="half-open", output_to_source_frame=SOURCE_FRAMES,
        all_lossless_master_frames_match_source=True, full_decode=True,
        physics_recomputed=False, motion_interpolation=False, text_overlays=False)
    (a.out / "presentation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("video_sha256", "frames", "duration_s", "segments")}))


if __name__ == "__main__":
    main()
