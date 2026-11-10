#!/usr/bin/env python3
"""Render DeformX's silent two-column, three-row showcase using CPU FFmpeg.

From the repository root, with FFmpeg (FFV1, libx264 and GIF) and FFprobe:

    python demo/media/render_gpu_showcase.py --out /path/to/fresh/outputs

Pass --source-root when the demo media have been copied to another tree.
"""

import argparse
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile


ROWS, COLUMNS = 3, 2
WIDTH, HEIGHT = 1280, 1080
GIF_WIDTH, GIF_HEIGHT = 736, 621
VIDEO_WIDTH, VIDEO_HEIGHT = 640, 360
SOURCE_FPS, GIF_FPS, DURATION_S = 30, 10, 21
PALETTE_COLORS = 128


@dataclass(frozen=True)
class Clip:
    title: str
    path: str
    sha256: str
    start_s: float
    evidence: str
    source_normal_speed_frame: int | None = None
    source_fps: int = 30
    recorded_playback_multiplier: float = 1
    opening_recorded_playback_multiplier: float = .25
    duration_s: float = 8
    hold_last: bool = False


CLIPS = (
    Clip("Humanoid guitar", "demo/humanoid_guitar/media/humanoid-guitar-preview.mp4",
         "9c96e1ec6fb0d87865564ae4593a93a6a1472b727cecebb66dc3512ef1a2bdbe", 0,
         "demo/humanoid_guitar/presentation.json", source_normal_speed_frame=92,
         opening_recorded_playback_multiplier=.5),
    Clip("Three-strand braider", "demo/braiding_machine/media/braiding-process-hold.mp4",
         "854cb279d1eadfc3258bbb631a6db5d470208269bf650683ecf13d0a31a8541d", 0,
         "demo/braiding_machine/presentation.json", duration_s=21),
    Clip("Robot cable extraction", "demo/cable_extraction/media/cable-extraction.mp4",
         "849f864a2029608e542e170fab6b011a27c6ff4d3c37097e02e33d45d84ad8e9", 0,
         "demo/cable_extraction/validation.json", source_fps=60,
         recorded_playback_multiplier=2, duration_s=18.7, hold_last=True),
    Clip("Robot rack insertion", "demo/rack_insertion/media/rack-insertion.mp4",
         "731127811c33ddae443a54d0d1b26587c7ddd58bf627a01b36a35be1b369663f", 6,
         "demo/rack_insertion/evidence/clean-replay.json"),
    Clip("Traction reels", "demo/winch_tension/media/winch-tension-long.mp4",
         "fb264788078b369f9adab8a2af37535e08db792c1a0caa0f1b090ee0f32270f0", 6,
         "demo/media/clean-replay-provenance.json"),
    Clip("Contact arena", "demo/contact_arena/media/contact-arena-pullback.mp4",
         "5660e2a332fded66037628b30d91ced0aa28a2d6fa28111879b9a35f59588f19", 0,
         "demo/contact_arena/presentation.json"),
)


def panel_coordinates():
    return [[(i % COLUMNS) * VIDEO_WIDTH, (i // COLUMNS) * VIDEO_HEIGHT]
            for i in range(len(CLIPS))]


def command(*args, cwd=None):
    result = subprocess.run(args, cwd=cwd, stdin=subprocess.DEVNULL,
                            capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f"{args[0]} exited with {result.returncode}\n{result.stderr}")
    return result.stdout


def ffmpeg(*args, cwd=None):
    command("ffmpeg", "-nostdin", "-hide_banner", "-v", "error", "-n", *args, cwd=cwd)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def probe(path, *, count_frames=False):
    options = ["-count_frames"] if count_frames else []
    return json.loads(command(
        "ffprobe", "-v", "error", *options, "-show_streams", "-show_format",
        "-of", "json", path,
    ))


def stream_of(probe_data):
    streams = probe_data["streams"]
    if len(streams) != 1 or streams[0]["codec_type"] != "video":
        raise ValueError("Expected exactly one video stream")
    return streams[0]


def verify_source(root, clip):
    path = root / clip.path
    if sha256(path) != clip.sha256:
        raise ValueError(f"Source hash changed for {clip.path}")
    metadata = probe(path)
    stream = stream_of(metadata)
    if (
        (stream["width"], stream["height"]) != (1280, 720)
        or stream.get("sample_aspect_ratio", "N/A") not in ("N/A", "1:1")
        or stream.get("side_data_list")
        or stream["pix_fmt"] != "yuv420p"
    ):
        raise ValueError(f"Unexpected source geometry or color format for {clip.path}")
    if Fraction(stream["avg_frame_rate"]) != clip.source_fps:
        raise ValueError(f"Unexpected frame rate in {clip.path}")
    packets = json.loads(command(
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_packets",
        "-show_entries", "packet=pts", "-of", "json", path,
    ))["packets"]
    time_base = Fraction(stream["time_base"])
    pts = sorted(int(packet["pts"]) * time_base for packet in packets)
    first_frame = Fraction(str(clip.start_s)) * clip.source_fps
    if first_frame.denominator != 1:
        raise ValueError("Clip start must align to a source frame")
    if len(pts) < first_frame + clip.duration_s * clip.source_fps or any(
        timestamp != Fraction(i, clip.source_fps) for i, timestamp in enumerate(pts)
    ):
        raise ValueError(f"Expected continuous zero-based {clip.source_fps} fps timestamps in {clip.path}")
    return {
        "title": clip.title,
        "path": clip.path,
        "sha256": clip.sha256,
        "codec": stream["codec_name"],
        "dimensions": [stream["width"], stream["height"]],
        "sample_aspect_ratio": stream.get("sample_aspect_ratio", "unspecified"),
        "color_metadata": {name: stream.get(name, "unspecified") for name in
                           ("color_range", "color_space", "color_transfer", "color_primaries")},
        "frame_rate": stream["avg_frame_rate"],
        "frame_count": len(pts),
        "container_duration_s": float(metadata["format"]["duration"]),
        "source_in_s": clip.start_s,
        "source_out_s_exclusive": clip.start_s + clip.duration_s,
        "source_in_frame": int(first_frame),
        "source_out_frame_exclusive": int(first_frame) + round(clip.duration_s * clip.source_fps),
        "composition_playback_multiplier": 1,
        "repeat_period_s": clip.duration_s if clip.duration_s < DURATION_S and not clip.hold_last else None,
        "end_hold_s": DURATION_S - clip.duration_s if clip.hold_last else 0,
        "composition_duration_s": DURATION_S,
        "recording_evidence": clip.evidence,
        "recording_evidence_sha256": sha256(root / clip.evidence),
    }


def composition_graph():
    panels = []
    for index, clip in enumerate(CLIPS):
        first = int(Fraction(str(clip.start_s)) * clip.source_fps)
        last = first + round(clip.duration_s * clip.source_fps)
        filters = [
            f"trim=start_frame={first}:end_frame={last}",
            "setpts=PTS-STARTPTS",
        ]
        if clip.hold_last and clip.source_fps != SOURCE_FPS:
            if clip.source_fps % SOURCE_FPS:
                raise ValueError("Endpoint-preserving sampling requires an integer frame-rate ratio")
            stride = clip.source_fps // SOURCE_FPS
            count = last - first
            filters.extend([
                f"select='lt(n,{count-stride})*not(mod(n,{stride}))+eq(n,{count-1})'",
                f"setpts=N/({SOURCE_FPS}*TB)",
                f"fps={SOURCE_FPS}",
            ])
        else:
            filters.append(f"fps={SOURCE_FPS}")
        filters.extend([
            f"scale={VIDEO_WIDTH}:{VIDEO_HEIGHT}:flags=lanczos",
            "setsar=1",
            "format=yuv444p",
        ])
        if clip.hold_last:
            filters.extend([
                f"tpad=stop_mode=clone:stop={round((DURATION_S - clip.duration_s) * SOURCE_FPS)}",
                f"trim=end_frame={DURATION_S * SOURCE_FPS}",
            ])
        elif clip.duration_s < DURATION_S:
            filters.extend([
                f"loop=loop=-1:size={round(clip.duration_s * SOURCE_FPS)}:start=0",
                f"trim=end_frame={DURATION_S * SOURCE_FPS}",
                f"setpts=N/({SOURCE_FPS}*TB)",
            ])
        panels.append(f"[{index}:v:0]" + ",".join(filters) + f"[p{index}]")
    layout = "|".join(f"{x}_{y}" for x, y in panel_coordinates())
    inputs = "".join(f"[p{i}]" for i in range(len(CLIPS)))
    panels.append(f"{inputs}xstack=inputs={len(CLIPS)}:layout={layout}[out]")
    return ";".join(panels)


def verify_braider_panel(root, master):
    """Check every uncompressed braider panel against the complete source."""
    def hashes(path, filters):
        output = command("ffmpeg", "-nostdin", "-v", "error", "-threads", "2",
                         "-i", path, "-vf", filters, "-an", "-f", "framemd5", "-")
        return [row.rsplit(",", 1)[1].strip() for row in output.splitlines()
                if row and not row.startswith("#")]
    index = next(i for i, clip in enumerate(CLIPS) if clip.title == "Three-strand braider")
    clip = CLIPS[index]
    assert clip.start_s == 0 and clip.duration_s == DURATION_S
    expected = hashes(root / clip.path,
                      f"fps={SOURCE_FPS},scale={VIDEO_WIDTH}:{VIDEO_HEIGHT}:flags=lanczos,"
                      "setsar=1,format=yuv444p")
    x, y = panel_coordinates()[index]
    actual = hashes(master, f"crop={VIDEO_WIDTH}:{VIDEO_HEIGHT}:{x}:{y}")
    if len(expected) != DURATION_S * SOURCE_FPS or actual != expected:
        raise ValueError("The composite must preserve every frame of the full braider recording")
    return {"source": clip.path, "source_sha256": clip.sha256,
            "source_frames_half_open": [0, len(expected)], "matched_master_frames": len(actual),
            "full_source_in_order": True, "playback_multiplier": 1,
            "scaled_frame_hashes_sha256": hashlib.sha256("\n".join(actual).encode()).hexdigest()}


def encode(root, work):
    master = work / "composite.mkv"
    mp4 = work / "gpu-showcase.mp4"
    gif = work / "gpu-showcase.gif"
    palette = work / "palette.png"
    source_args = [arg for clip in CLIPS for arg in ("-threads", "2", "-i", root / clip.path)]
    ffmpeg(*source_args, "-filter_complex_threads", "2", "-filter_complex", composition_graph(),
           "-map", "[out]", "-map_metadata", "-1", "-an", "-c:v", "ffv1",
           "-threads", "2", master, cwd=work)
    braider_check = verify_braider_panel(root, master)
    extraction_check = verify_extraction_panel(root, master)
    ffmpeg("-threads", "2", "-i", master, "-map", "0:v:0", "-map_metadata", "-1",
           "-an", "-c:v", "libx264", "-preset", "medium", "-crf", "17", "-threads", "2",
           "-pix_fmt", "yuv420p", "-movflags", "+faststart", mp4)

    # Resample the presentation cadence without changing elapsed time.
    gif_picture = (f"fps={GIF_FPS},"
                   f"scale={GIF_WIDTH}:{GIF_HEIGHT}:flags=lanczos")
    ffmpeg("-threads", "2", "-i", master, "-filter_threads", "2", "-vf",
           f"{gif_picture},palettegen=max_colors={PALETTE_COLORS}:stats_mode=full",
           "-frames:v", "1", "-update", "1", palette)
    ffmpeg("-threads", "2", "-i", master, "-i", palette, "-filter_complex_threads", "2",
           "-filter_complex",
           f"[0:v:0]{gif_picture}[pic];[pic][1:v:0]paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle[out]",
           "-map", "[out]", "-map_metadata", "-1", "-an", "-loop", "0",
           "-final_delay", str(round(100 / GIF_FPS)), gif)
    return mp4, gif, braider_check, extraction_check


def verify_extraction_panel(root, master):
    """Verify the entire accelerated extraction, followed only by its last frame."""
    def hashes(path, filters):
        output = command("ffmpeg", "-nostdin", "-v", "error", "-threads", "2",
                         "-i", path, "-vf", filters, "-an", "-f", "framemd5", "-")
        return [row.rsplit(",", 1)[1].strip() for row in output.splitlines()
                if row and not row.startswith("#")]
    index = next(i for i, clip in enumerate(CLIPS) if clip.title == "Robot cable extraction")
    clip = CLIPS[index]
    # Hash every original frame independently, then select the explicit frame map.
    original = hashes(root / clip.path,
                      f"scale={VIDEO_WIDTH}:{VIDEO_HEIGHT}:flags=lanczos,setsar=1,format=yuv444p")
    stride = clip.source_fps // SOURCE_FPS
    indices = [*range(0, len(original) - stride, stride), len(original) - 1]
    expected = [original[i] for i in indices]
    source_frames = round(clip.duration_s * SOURCE_FPS)
    if clip.start_s != 0 or len(expected) != source_frames or not clip.hold_last:
        raise ValueError("Extraction must start at the beginning and include its entire recording")
    held_frames = DURATION_S * SOURCE_FPS - len(expected)
    expected += [expected[-1]] * held_frames
    x, y = panel_coordinates()[index]
    actual = hashes(master, f"crop={VIDEO_WIDTH}:{VIDEO_HEIGHT}:{x}:{y}")
    if actual != expected:
        raise ValueError("Extraction panel differs from full source plus final-frame hold")
    if actual[0] != original[0] or actual[-1] != original[-1]:
        raise ValueError("Extraction must preserve the original first and terminal frames")
    return {"source": clip.path, "source_sha256": clip.sha256,
            "source_start_s": 0, "source_end_s": clip.duration_s,
            "matched_sampled_source_frames": source_frames, "held_frames": held_frames,
            "source_frame_indices": indices,
            "original_first_and_last_frames_preserved": True,
            "terminal_source_frame_md5": original[-1],
            "recorded_playback_multiplier": clip.recorded_playback_multiplier,
            "full_source_timeline_in_order": True,
            "scaled_frame_hashes_sha256": hashlib.sha256("\n".join(actual).encode()).hexdigest()}


def verify_export(path, expected_codec, fps, dimensions):
    metadata = probe(path, count_frames=True)
    stream = stream_of(metadata)
    frames = int(stream["nb_read_frames"])
    duration = float(metadata["format"]["duration"])
    if (stream["codec_name"], stream["width"], stream["height"], frames) != (
        expected_codec, *dimensions, DURATION_S * fps
    ):
        raise ValueError(f"Unexpected video stream in {path.name}")
    if abs(duration - DURATION_S) > 0.01:
        raise ValueError(f"Unexpected duration in {path.name}: {duration:.6f} s")
    packets = json.loads(command(
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_packets",
        "-show_entries", "packet=pts", "-of", "json", path,
    ))["packets"]
    ticks = sorted(int(packet["pts"]) for packet in packets)
    time_base = Fraction(stream["time_base"])
    max_pts_error = max(abs(tick * time_base - Fraction(i, fps)) for i, tick in enumerate(ticks))
    tolerance = Fraction(1, 100) if expected_codec == "gif" else Fraction(1, 1_000_000)
    if len(ticks) != frames or max_pts_error > tolerance:
        raise ValueError(f"Unexpected frame cadence in {path.name}")
    ffmpeg("-xerror", "-err_detect", "explode", "-threads", "2", "-i", path,
           "-map", "0:v:0", "-f", "null", "-")
    return {
        "sha256": sha256(path), "bytes": path.stat().st_size,
        "duration_s": duration, "dimensions": list(dimensions), "frames": frames,
        "codec": expected_codec, "nominal_fps": fps,
        "probe_avg_frame_rate": stream["avg_frame_rate"],
        "max_presentation_timestamp_error_s": float(max_pts_error),
        "complete_decode": True, "audio_streams": 0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--out", type=Path, required=True)
    options = parser.parse_args()
    root, out = options.source_root.resolve(), options.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    for name in ("gpu-showcase.mp4", "gpu-showcase.gif", "gpu-showcase-provenance.json"):
        if (out / name).exists():
            raise FileExistsError(out / name)

    inputs = [verify_source(root, clip) for clip in CLIPS]
    if len(CLIPS) != ROWS * COLUMNS:
        raise ValueError("The showcase requires exactly six clips")
    for record, clip in zip(inputs, CLIPS):
        length = round(clip.duration_s * SOURCE_FPS)
        change_s = (Fraction(clip.source_normal_speed_frame or 0, clip.source_fps)
                    - Fraction(str(clip.start_s)))
        change = min(length, max(0, int(change_s * SOURCE_FPS)))
        segments = []
        if change:
            segments.append({"composition_start_frame": 0,
                             "composition_end_frame_exclusive": change,
                             "simulation_time_multiplier": clip.opening_recorded_playback_multiplier})
        if change < length:
            segments.append({"composition_start_frame": change,
                             "composition_end_frame_exclusive": length,
                             "simulation_time_multiplier": clip.recorded_playback_multiplier})
        repeated_segments = []
        offsets = [0] if clip.hold_last else range(0, DURATION_S * SOURCE_FPS, length)
        for offset in offsets:
            for segment in segments:
                first = offset + segment["composition_start_frame"]
                last = min(DURATION_S * SOURCE_FPS, offset + segment["composition_end_frame_exclusive"])
                if first < last:
                    repeated_segments.append(dict(segment, composition_start_frame=first,
                                                  composition_end_frame_exclusive=last))
        if clip.hold_last and length < DURATION_S * SOURCE_FPS:
            repeated_segments.append({"composition_start_frame": length,
                                      "composition_end_frame_exclusive": DURATION_S * SOURCE_FPS,
                                      "simulation_time_multiplier": 0,
                                      "kind": "hold_final_frame"})
        record["underlying_recorded_playback"] = repeated_segments
    report = {
        "trajectory_sources": "Per-scene recording evidence; simulation revisions may differ",
        "toolchain": {
            "ffmpeg": command("ffmpeg", "-version").splitlines()[0],
            "ffprobe": command("ffprobe", "-version").splitlines()[0],
            "renderer_sha256": sha256(Path(__file__)),
        },
        "source_clips_in_reading_order": inputs,
        "composition": {
            "duration_s": DURATION_S,
            "dimensions": [WIDTH, HEIGHT],
            "fit": "uncropped uniform scale to square-pixel 640x360 panels; no text or HUD",
            "rows": ROWS, "columns": COLUMNS,
            "panel_coordinates": panel_coordinates(),
            "text_overlays": False,
            "base_fps": SOURCE_FPS,
            "gif_sampling": f"{GIF_FPS} fps sampling at the original playback rate, uniformly downscaled",
            "gif_palette_colors": PALETTE_COLORS,
            "repeat_policy": "Full 21-second braider at 1x; full 18.7-second extraction at 2x with a 2.3-second final hold; other four 8-second excerpts repeat at their recorded rates.",
            "audio": "silent sources, silent composition",
        },
    }

    with tempfile.TemporaryDirectory(prefix="gpu-showcase-", dir=out) as temp:
        mp4, gif, braider_check, extraction_check = encode(root, Path(temp))
        report["full_braider_frame_check"] = braider_check
        report["full_extraction_frame_check"] = extraction_check
        report["exports"] = {
            mp4.name: verify_export(mp4, "h264", SOURCE_FPS, (WIDTH, HEIGHT)),
            gif.name: verify_export(gif, "gif", GIF_FPS, (GIF_WIDTH, GIF_HEIGHT)),
        }
        for export in (mp4, gif):
            export.rename(out / export.name)
    provenance = out / "gpu-showcase-provenance.json"
    provenance.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report["exports"], indent=2))


if __name__ == "__main__":
    main()
