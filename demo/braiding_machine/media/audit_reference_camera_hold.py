#!/usr/bin/env python3
"""Verify the braid-formation close-up against the original replay and previous hold on ps."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image

from replay_reference_camera_hold import STUDIO_INTENSITY


SEQUENCE = {
    "carriers": (0, 1),
    "descend": (1, 4),
    "braid_detail": (4, 9),
    "pullback": (9, 13),
    "whole": (13, 21),
}
WIDTH, HEIGHT, FPS = 1280, 720, 30
REGIONS = {
    "opening_backdrop": (0, 30, (40, 360, 280, 600)),
    "braid_detail_backdrop": (120, 270, (40, 150, 280, 550)),
    "whole_backdrop": (390, 630, (40, 70, 280, 230)),
}
MATCHED_RANGES = {"unchanged_opening": (0, 30), "whole_machine_after_pullback": (390, 630)}
DETAIL_RAISE_M = .70
LUMA = [.2126, .7152, .0722]
REVIEW = (0, 90, 119, 120, 160, 210, 255, 269, 300, 345, 390, 629)
DETAIL_REVIEW = (120, 210, 269)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    return json.loads(path.read_text())


def matches_at(index):
    return next((key for key, (start, stop) in MATCHED_RANGES.items() if start <= index < stop), None)


def compare(old, new):
    diff = new.astype(np.int16) - old
    return (float(np.abs(diff).mean()), float(abs(diff.mean(axis=(0, 1)) @ LUMA)))


def patch_means(rgb, index, values):
    for key, (start, stop, (x0, y0, x1, y1)) in REGIONS.items():
        if start <= index < stop:
            values[key].append(float(rgb[y0:y1, x0:x1].mean(axis=(0, 1)) @ LUMA))


def native_rgb(path):
    with Image.open(path) as image:
        assert image.size == (WIDTH, HEIGHT)
        return np.asarray(image.convert("RGB"))


def decode(path):
    command = ["ffmpeg", "-nostdin", "-v", "error", "-xerror", "-err_detect", "explode",
               "-threads", "2", "-i", str(path), "-threads", "2", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    with subprocess.Popen(command, stdout=subprocess.PIPE) as process:
        frame_bytes = WIDTH * HEIGHT * 3
        for index in range(FPS * 21):
            raw = process.stdout.read(frame_bytes)
            assert len(raw) == frame_bytes, (path, index)
            yield np.frombuffer(raw, np.uint8).reshape(HEIGHT, WIDTH, 3)
        assert process.stdout.read(1) == b"" and process.wait() == 0, path


def patch_metrics(values):
    result = {}
    for key, samples in values.items():
        start, stop, box = REGIONS[key]
        assert len(samples) == stop - start
        delta = np.abs(np.diff(samples))
        assert delta.max() < 1.0, (key, delta.max())
        result[key] = {"frames": len(samples), "backdrop_box_xyxy": box, "playback_frames": [start, stop],
                       "adjacent_mean_luma_p95": float(np.percentile(delta, 95)),
                       "adjacent_mean_luma_max": float(delta.max())}
    return result


def reference_metrics(values):
    result = {}
    for key, samples in values.items():
        start, stop = MATCHED_RANGES[key]
        assert len(samples) == stop - start
        rgb, luma = np.array(samples).T
        assert rgb.max() < 4.0 and luma.max() < 2.0, (key, rgb.max(), luma.max())
        result[key] = {"frames": len(samples), "playback_frames": [start, stop],
                       "mean_abs_rgb_delta_p95": float(np.percentile(rgb, 95)),
                       "mean_abs_rgb_delta_max": float(rgb.max()),
                       "abs_mean_luma_shift_p95": float(np.percentile(luma, 95)),
                       "abs_mean_luma_shift_max": float(luma.max())}
    return result


def smootherstep(value):
    clipped = np.clip(value, 0., 1.)
    return clipped**3 * (clipped * (6. * clipped - 15.) + 10.)


def check_cameras(report, previous):
    frames = report["frames"]
    previous_frames = previous["frames"]
    views = [view for view, (a, b) in SEQUENCE.items() for _ in range(FPS * (b - a))]
    assert [row["view"] for row in frames] == [row["view"] for row in previous_frames] == views
    assert len(frames) == len(previous_frames) == FPS * 21
    for new, old in zip(frames, previous_frames, strict=True):
        for key in ("index", "t_s", "state_sha256"):
            assert new[key] == old[key], (key, new["t_s"])
    play_s = np.arange(len(frames)) / FPS
    descent = smootherstep((play_s - 1.) / 3.)
    pullback = 1. - smootherstep((play_s - 9.) / 4.)
    translation = np.zeros((len(frames), 3))
    translation[:, 2] = DETAIL_RAISE_M * descent * pullback
    for key in ("camera_eye_m", "camera_target_m"):
        points = np.array([row[key] for row in frames])
        prior_points = np.array([row[key] for row in previous_frames])
        np.testing.assert_allclose(points - prior_points, translation, rtol=0, atol=1e-12)
        for start, stop in MATCHED_RANGES.values():
            assert np.array_equal(points[start:stop], prior_points[start:stop]), key
    eye = np.array([row["camera_eye_m"] for row in frames])
    target = np.array([row["camera_target_m"] for row in frames])
    assert np.max(np.abs((eye - target)[:270] - (eye - target)[0])) < 1e-12
    assert np.all(np.diff(target[30:121, 2]) <= 1e-12) and target[120, 2] == 2.45
    assert np.array_equal(eye[120:270], np.broadcast_to(eye[120], (150, 3)))
    assert np.array_equal(target[120:270], np.broadcast_to(target[120], (150, 3)))
    steps = np.linalg.norm(np.diff(eye, axis=0), axis=1)
    assert steps.max() < .16
    framing = report["camera_framing"]
    previous_framing = previous["camera_framing"]
    assert framing.keys() == previous_framing.keys()
    for key in framing.keys() - {"braid_detail"}:
        assert framing[key] == previous_framing[key], key
    for key in ("eye_m", "target_m"):
        np.testing.assert_allclose(
            np.asarray(framing["braid_detail"][key]) - previous_framing["braid_detail"][key],
            [0., 0., DETAIL_RAISE_M], rtol=0, atol=1e-12,
        )
    assert framing["max_abs_whole_machine_ndc"] <= .95
    return {
        "sequence_s": {("downward_traverse" if key == "descend" else "whole_machine" if key == "whole" else key): value
                       for key, value in SEQUENCE.items()},
        "frame_counts": {key: FPS * (b - a) for key, (a, b) in SEQUENCE.items()},
        "close_target_z_m": 2.45,
        "close_eye_and_target_translation_m": [0., 0., DETAIL_RAISE_M],
        "constant_eye_offset_during_descent": True,
        "max_adjacent_eye_step_m": float(steps.max()),
        "complete_machine_max_abs_ndc": framing["max_abs_whole_machine_ndc"],
        "opening_camera_poses_match_previous": True,
        "all_camera_poses_match_expected_vertical_translation": True,
        "whole_machine_framing_matches_previous": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--previous", type=Path, required=True)
    parser.add_argument("--original-native", type=Path, required=True)
    args = parser.parse_args()
    cap = args.out / "capture"
    delivery = args.out / "delivery"
    report_path = cap / "native_report.json"
    report, verify = load(report_path), load(cap / "verification.json")
    previous_path = args.previous / "capture/native_report.json"
    previous, original = load(previous_path), load(args.original_native / "native_report.json")
    previous_video_path = args.previous / "delivery/braiding-process-clean.mp4"
    previous_verified = load(args.previous / "capture/verification.json")
    video_path = delivery / "braiding-process-clean.mp4"
    frames = report["frames"]
    assert report["host"] == "ps" and len(frames) == 630 and report["complete"] and not report["errors"]
    assert report["app_close_invoked"] and verify["native_exit"] == 0
    assert report["capture_clock_delta_max_s"] == report["max_render_position_error_m"] == 0
    assert not report["physics_stepping"] and not report["image_overlay"]
    assert report["amplitude_scale"] == verify["playback_speed"] == 1.0
    assert all(report[key] == previous[key] for key in ("trajectory_sha256", "asset_manifest_sha256",
               "asset_sha256", "renderer", "resolution", "material_visual_radius_m"))
    timeline = [(row["index"], row["t_s"], row["state_sha256"]) for row in frames]
    assert timeline == [(row["index"], row["t_s"], row["state_sha256"]) for row in original["frames"]]
    camera = check_cameras(report, previous)
    assert verify["video"]["full_decode"] and verify["video"]["sha256"] == sha(video_path)
    assert (verify["video"]["frames"], verify["video"]["fps"], verify["video"]["duration_s"]) == (630, 30, 21)
    assert previous_verified["video"]["sha256"] == sha(previous_video_path)
    assert verify["native_report_sha256"] == sha(report_path)

    # The exact worker comparison allows only the detail target height to change.
    worker_path = args.out / "source/demo/braiding_machine/render_native.py"
    previous_worker_path = args.previous / "source/demo/braiding_machine/render_native.py"
    prior_worker = previous_worker_path.read_text()
    before = "braid_target = np.array([0., 0., 1.75])"
    after = "braid_target = np.array([0., 0., 2.45])"
    assert prior_worker.count(before) == 1
    assert worker_path.read_text() == prior_worker.replace(before, after)
    assert report["studio_environment"] == previous["studio_environment"] == {
        "selected_dome": "/World/Lights/StudioAmbient",
        "inactive_blender_domes": [f"/World/Hardware/{name}/env_light"
                                  for name in ("Frame", "Carrier_0", "Carrier_1", "Carrier_2")],
        "studio_ambient_intensity": STUDIO_INTENSITY,
    }
    assert report["source_sha256"]["render_native.py"] == sha(worker_path)
    assert previous["source_sha256"]["render_native.py"] == sha(previous_worker_path)

    patch = {domain: {key: [] for key in REGIONS} for domain in ("native", "encoded")}
    references = {domain: {key: [] for key in MATCHED_RANGES} for domain in patch}
    for index, row in enumerate(frames):
        filename = f"{index:06d}.png"
        path = cap / "frames" / filename
        assert sha(path) == row["image_sha256"], filename
        matched = matches_at(index)
        if matched or any(start <= index < stop for start, stop, _ in REGIONS.values()):
            rgb = native_rgb(path)
            patch_means(rgb, index, patch["native"])
            if matched:
                previous_frame = args.previous / "capture/frames" / filename
                assert sha(previous_frame) == previous["frames"][index]["image_sha256"], filename
                references["native"][matched].append(compare(native_rgb(previous_frame), rgb))

    montage = Image.new("RGB", (1280, 540))
    detail_montage = Image.new("RGB", (1280, 1080))
    for index, (old, new) in enumerate(zip(decode(previous_video_path), decode(video_path), strict=True)):
        patch_means(new, index, patch["encoded"])
        matched = matches_at(index)
        if matched:
            references["encoded"][matched].append(compare(old, new))
        if index in REVIEW:
            tile = REVIEW.index(index)
            montage.paste(Image.fromarray(new).resize((320, 180), Image.Resampling.LANCZOS),
                          ((tile % 4) * 320, (tile // 4) * 180))
        if index in DETAIL_REVIEW:
            row = DETAIL_REVIEW.index(index)
            for column, frame in enumerate((old, new)):
                detail_montage.paste(Image.fromarray(frame).resize((640, 360), Image.Resampling.LANCZOS),
                                     (column * 640, row * 360))
    montage.save(delivery / "encoded-review.jpg", quality=96)
    detail_montage.save(delivery / "encoded-detail-before-after.jpg", quality=96)
    result = {
        "decision": "Raise the five-second detail eye and target by 0.70 m to frame the forming braid beneath the carriers.",
        "scope": "All 630 saved state/time tuples and native image hashes checked; both complete encoded streams decoded. "
                 "Frame comparisons use identical source times and camera poses; brightness stability measures use fixed backdrop patches.",
        "host": report["host"],
        "physics_stepping": False,
        "image_overlay": False,
        "playback_speed": verify["playback_speed"],
        "native_capture": str(cap),
        "native_exit": verify["native_exit"],
        "app_close_invoked": report["app_close_invoked"],
        "capture_clock_delta_max_s": report["capture_clock_delta_max_s"],
        "max_render_position_error_m": report["max_render_position_error_m"],
        "saved_source_states_and_times_verified": len(timeline),
        "native_frames_hash_verified": len(frames),
        "native_report_sha256": sha(report_path),
        "renderer_sha256": sha(worker_path),
        "measurement_script_sha256": sha(Path(__file__)),
        "video": verify["video"],
        "poster_sha256": sha(delivery / "preview-process-clean.jpg"),
        "camera": camera,
        "fixed_patch_luma_metrics": {domain: patch_metrics(values) for domain, values in patch.items()},
        "matched_previous_appearance": {
            "native_report_sha256": sha(previous_path),
            "video_sha256": sha(previous_video_path),
            "metrics": {domain: reference_metrics(values) for domain, values in references.items()},
        },
        "encoded_review_playback_indices": REVIEW,
        "encoded_detail_before_after_indices": DETAIL_REVIEW,
        "studio_environment": report["studio_environment"],
        "renderer_settings": "The worker changes only the detail target height. The selected StudioAmbient dome, "
                             "its intensity, the disabled Blender HDR domes, distant lights, materials and antialiasing match the previous replay.",
    }
    (delivery / "render-stability.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
