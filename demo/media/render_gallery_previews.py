"""Encode small, silent animated README previews from packaged simulation videos.

Run on the render host with FFmpeg and FFprobe installed. Full videos remain
linked from the README; trims preserve the original recorded playback rate.
"""
import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

CLIPS = {
    "rack_insertion": ("media/rack-insertion.mp4", 0, None),
    "cable_cascade": ("media/cable-cascade.mp4", 0, None),
    "cable_extraction": ("media/cable-extraction.mp4", 0, None),
    "braiding_machine": ("media/braiding-process-hold.mp4", 0, None),
    "winch_tension": ("media/winch-tension-long.mp4", 4, 12),
    "contact_arena": ("media/contact-arena-pullback.mp4", 0, None),
}


def command(*args):
    return subprocess.check_output([str(x) for x in args], stderr=subprocess.PIPE)


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cases", nargs="+", choices=tuple(CLIPS), default=list(CLIPS))
    a = parser.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    result = {"renderer_sha256": sha(Path(__file__)), "dimensions": [416, 234],
              "nominal_fps": 10, "caption_or_hud_added": False,
              "cropped": False, "playback_multiplier": 1, "cases": {}}
    for case in a.cases:
        name, start, end = CLIPS[case]
        src = a.root / "demo" / case / name
        probe = json.loads(command("ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", src))
        stream, = probe["streams"]
        if stream["codec_type"] != "video" or Fraction(stream["avg_frame_rate"]) not in (30, 60):
            raise ValueError("Expected one silent 30 or 60 fps video")
        end = float(stream["duration"]) if end is None else end
        if end > float(stream["duration"]) + .001 or end <= start:
            raise ValueError("Preview trim outside the source")
        trim = f"trim=start={start}:end={end},setpts=PTS-STARTPTS,fps=10,scale=416:234:flags=lanczos,setsar=1"
        gif = a.out / (case + ".gif")
        with tempfile.TemporaryDirectory(dir=a.out) as tmp:
            palette = Path(tmp) / "palette.png"
            command("ffmpeg", "-nostdin", "-v", "error", "-threads", 2, "-i", src,
                    "-filter_threads", 2, "-vf", trim + ",palettegen=max_colors=128:stats_mode=full",
                    "-frames:v", 1, "-update", 1, palette)
            command("ffmpeg", "-nostdin", "-v", "error", "-threads", 2, "-i", src, "-i", palette,
                    "-filter_complex_threads", 2, "-filter_complex",
                    f"[0:v]{trim}[v];[v][1:v]paletteuse=dither=bayer:bayer_scale=4[out]",
                    "-map", "[out]", "-an", "-loop", 0, gif)
        command("ffmpeg", "-nostdin", "-v", "error", "-xerror", "-err_detect", "explode",
                "-i", gif, "-f", "null", "-")
        q = json.loads(command("ffprobe", "-v", "error", "-count_frames", "-show_streams", "-show_format", "-of", "json", gif))
        output, = q["streams"]
        duration = float(q["format"]["duration"])
        assert (output["width"], output["height"]) == (416, 234)
        assert abs(duration - (end-start)) <= .11
        assert int(output["nb_read_frames"]) == round(duration*10)
        result["cases"][case] = {"source": str(src.relative_to(a.root)),
            "source_sha256": sha(src), "source_trim_s": [start, end],
            "source_fps": stream["avg_frame_rate"],
            "video": gif.name, "sha256": sha(gif), "bytes": gif.stat().st_size,
            "frames": int(output["nb_read_frames"]), "duration_s": duration, "full_decode": True}
        if case == 'cable_extraction':
            presentation = a.root / 'demo' / case / 'presentation.json'
            if presentation.exists():
                evidence = json.loads(presentation.read_text())
                if evidence['video_sha256'] == sha(src):
                    result['cases'][case]['recorded_playback_multiplier'] = evidence['playback_multiplier']
                    result['cases'][case]['presentation_evidence_sha256'] = sha(presentation)
                elif evidence['source_video_sha256'] == sha(src):
                    result['cases'][case]['recorded_playback_multiplier'] = 1
                else:
                    raise ValueError('Extraction source does not match presentation evidence')
        print(case, result["cases"][case], flush=True)
    (a.out/"provenance.json").write_text(json.dumps(result,indent=2)+"\n")


if __name__ == "__main__":
    main()
