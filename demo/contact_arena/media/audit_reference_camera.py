"""Independent frame-by-frame QA for the DeformX contact-arena pullback.

Run on ps after sourcing /home/ps/yy_workspace/activate_render.sh.
"""
import argparse
import hashlib
import itertools
import json
import re
import subprocess
from pathlib import Path

import cv2
import numpy as np

BASE = Path('/home/ps/yy_workspace/runs')
SOURCE = BASE / 'long-showcases-20261108/arena-showcase-1024-24s'
ARCHIVED = BASE / 'long-showcases-20261108/arena-video-24s/native_report.json'
ARCHIVED_ENVIRONMENT_SHA = 'c4c6bc7a125abb414d9336a2b03c698f9db650c7917b69068beaedb569cb2291'
WIDTH, HEIGHT, FPS, COUNT = 1280, 720, 30, 720
BANDS = {'near': (0, 45), 'pullback': (45, 165), 'whole': (165, 720)}
BGR_TO_LUMA = np.array([.114, .587, .299], dtype=np.float32)


def require(ok, explanation):
    if not ok:
        raise RuntimeError(explanation)


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(path.read_text())


def audit_appearance_sources(capture, report, archived):
    source_dir = capture.parent / 'source/demo/contact_arena'
    require({name: sha(source_dir / name) for name in report['source_sha256']} == report['source_sha256'],
            'A staged source file differs from its recorded hash')
    stage = capture / 'scene.usda'
    with stage.open() as handle:
        asset_paths = {path for line in handle for path in re.findall(r'@([^@]+/(?:guide|cell)\.usda)@', line)}
    evidence = dict(all_staged_source_hashes_verified=True, referenced_asset_hashes_verified=bool(asset_paths))
    if asset_paths:
        resolved = {(stage.parent / path).resolve() for path in asset_paths}
        roots = {path.parent for path in resolved}
        require(len(resolved) == 2 and len(roots) == 1, 'Inconsistent referenced plate and guide assets')
        root, = roots
        require({name: sha(root / name) for name in report['asset_sha256']} == report['asset_sha256'],
                'A referenced asset or its metadata differs from the archive')
        textures = set()
        for path in resolved:
            with path.open() as handle:
                for line in handle:
                    if 'inputs:texture:file' in line:
                        textures.update((path.parent / texture).resolve() for texture in re.findall(r'@([^@]+)@', line))
        require(len(textures) == 1, 'Expected both referenced asset domes to share the original environment HDR')
        texture, = textures
        digest = sha(texture)
        require(digest == ARCHIVED_ENVIRONMENT_SHA, 'Nested environment texture differs from the archive')
        evidence.update(nested_environment_hdr=str(texture), nested_environment_hdr_sha256=digest)
    baseline = capture / 'appearance-baseline.png'
    if 'archived_pose_rgb_mae' in report:
        archive_frame = ARCHIVED.parent / '000000.png'
        require(sha(archive_frame) == archived['frames'][0]['sha256'], 'Archived appearance frame hash differs')
        images = [cv2.imread(str(path), cv2.IMREAD_COLOR) for path in (baseline, archive_frame)]
        require(all(image is not None and image.shape == (HEIGHT, WIDTH, 3) for image in images),
                'Unable to decode appearance gate image or reference')
        mae = np.abs(images[0].astype(np.float32) - images[1]).mean(dtype=np.float64)
        require(abs(mae - report['archived_pose_rgb_mae']) <= 1e-8, 'Appearance gate report differs from images')
        require(mae <= 5, f'Archived-pose appearance exceeds the 5-code-value RGB MAE gate: {mae:.4f}')
        evidence.update(archived_pose_rgb_mae=rounded(mae), appearance_baseline_sha256=sha(baseline))
    return evidence


def rounded(value):
    return round(float(value), 5)


def summary(values):
    values = np.asarray(values, dtype=float)
    require(values.size and np.isfinite(values).all(), 'Empty or invalid measurement')
    return {name: rounded(np.quantile(values, level)) for name, level in
            [('median', .50), ('p95', .95), ('p99', .99), ('max', 1)]}


def grid_offsets():
    ids = np.arange(64)
    return np.c_[(ids % 8 - 3.5) * 1.4, (ids // 8 - 3.5) * .72, np.zeros(64)]


def patch_geometry():
    patch_offsets = np.array([[-.4, -.24, -.0075], [.4, -.24, -.0075],
                              [-.4, .24, -.0075], [.4, .24, -.0075]])
    centers = (grid_offsets()[:, None, :] + patch_offsets).reshape(-1, 3)
    step = np.linspace(-.028, .028, 7)
    grid = np.stack([*np.meshgrid(step, step), np.zeros((7, 7))], axis=-1).reshape(-1, 3)
    return centers[:, None, :] + grid[None, :, :], np.repeat(np.arange(64), 4)


def project(points, eye, target, profile):
    forward = target - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0, 0, 1])
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    rays = points - eye
    depth = rays @ forward
    aperture = np.array([profile['horizontal_aperture_mm'], profile['vertical_aperture_mm']])
    ndc = 2 * profile['focal_length_mm'] * (rays @ np.stack((right, down)).T)
    ndc /= depth[..., None] * aperture
    return ndc, depth


def sample_positions(grid, eye, target, profile):
    ndc, depth = project(grid, eye, target, profile)
    px = ndc * (np.array([WIDTH, HEIGHT]) / 2) + (np.array([WIDTH, HEIGHT]) - 1) / 2
    px = px.astype(np.float32)
    # Require an in-frame bilinear footprint and a footprint with at least two pixels on the short axis.
    in_view = ((px >= 2) & (px <= np.array([WIDTH - 3, HEIGHT - 3]))).all(axis=(1, 2))
    footprint = np.stack((px[:, 27] - px[:, 21], px[:, 45] - px[:, 3]), axis=-1)
    fit = (depth > .01).all(axis=1) & in_view & (np.linalg.svd(footprint, compute_uv=False)[:, -1] >= 2)
    px[~fit] = -1
    return px, fit


def audit_camera(report, positions, guides, *, complete=True):
    profile = report['camera_profile']
    require(np.allclose([profile['focal_length_mm'], profile['horizontal_aperture_mm'],
                         profile['vertical_aperture_mm']], [23, 20.955, 11.7871875], rtol=0, atol=1e-6),
            'Lens intrinsics differ from the expected camera')
    require(profile['near_hold_s'] == [0, 1.5] and profile['pullback_s'] == [1.5, 5.5]
            and profile['whole_hold_s'] == [5.5, 24], 'Camera time ranges differ')
    require(profile['transition'] == 'quintic smootherstep', 'Expected quintic smootherstep')
    rows = report['frames']
    eye, target = (np.array([row[key] for row in rows], dtype=float)
                   for key in ('camera_eye_m', 'camera_target_m'))
    frame_idx = np.array([row['trajectory_index'] for row in rows])
    phase = np.select([frame_idx < 45, frame_idx < 165], ['near', 'pullback'], default='whole')
    require(np.array_equal(phase, [row['camera_phase'] for row in rows]), 'Camera phase labels differ')
    u = np.clip((frame_idx[:, None] - 45) / 120, 0, 1)
    blend = u**3 * (10 - 15 * u + 6 * u**2)
    near_eye, whole_eye, near_target, whole_target = (np.array(profile[key], dtype=float)
        for key in ('near_eye_m', 'whole_eye_m', 'near_target_m', 'whole_target_m'))
    eye_error = np.abs(eye - (near_eye + blend * (whole_eye - near_eye))).max()
    target_error = np.abs(target - (near_target + blend * (whole_target - near_target))).max()
    require(max(eye_error, target_error) < 1e-10, 'Recorded eye or target departs from the continuous planned path')
    distances = np.linalg.norm(target - eye, axis=-1)
    require(distances[-1] > distances[0], 'Expected pullback from near to far camera')
    require(np.all(np.diff(distances) >= -1e-10), 'Camera distance reverses during the pullback')
    offset = grid_offsets()
    radius = report['visual_radius_m']
    low = np.minimum(offset.min(axis=0) - [.675, .325, .0725],
                     (positions.min(axis=(0, 2)) + offset).min(axis=0) - radius)
    high = np.maximum(offset.max(axis=0) + [.675, .325, .67],
                      (positions.max(axis=(0, 2)) + offset).max(axis=0) + radius)
    require(np.allclose([low, high], profile['bounds_m'], rtol=0, atol=1e-8),
            'Bounds do not match independently calculated plate and cable envelope')
    corners = np.array(list(itertools.product(*zip(low, high))))
    max_abs_ndc = []
    for e, t, label in zip(eye, target, phase):
        if label == 'whole':
            ndc, depth = project(corners, e, t, profile)
            require((depth > .01).all() and (depth < 150).all(), 'Whole-grid box outside clipping planes')
            max_abs_ndc.append(np.abs(ndc).max())
    require(max_abs_ndc, 'No sampled whole-grid camera pose')
    max_ndc = max(max_abs_ndc)
    require(profile['whole_ndc_limit'] <= .92 and max_ndc <= profile['whole_ndc_limit'],
            'Whole-grid bounds exceed camera framing limit')
    require(abs(max_ndc - profile['max_abs_whole_bounds_ndc']) < 1e-8,
            'Independent whole-grid reprojection does not match capture report')
    step = np.linalg.norm(np.diff(eye, axis=0), axis=-1)
    forward = (target - eye) / distances[:, None]
    turn = np.rad2deg(np.arctan2(np.linalg.norm(np.cross(forward[:-1], forward[1:]), axis=-1),
                                np.einsum('ij,ij->i', forward[:-1], forward[1:])))
    metrics = dict(lens_mm=profile['focal_length_mm'], aperture_mm=[profile['horizontal_aperture_mm'],
                   profile['vertical_aperture_mm']], phase_lengths_frames={name: int((phase == name).sum())
                       for name in BANDS}, transition='quintic smootherstep: zero endpoint velocity and acceleration',
                   max_eye_path_error_m=float(eye_error), max_target_path_error_m=float(target_error),
                   camera_distance_range_m=[rounded(distances[0]), rounded(distances[-1])],
                   max_eye_step_m=rounded(step.max()), max_camera_turn_deg=rounded(turn.max()),
                   box_m=[low.tolist(), high.tolist()], whole_grid_box_max_abs_ndc=rounded(max_ndc),
                   ndc_limit=profile['whole_ndc_limit'], minimum_frame_edge_margin_pct=rounded((1 - max_ndc) * 50),
                   whole_frames_bounds_checked=len(max_abs_ndc),
                   source_guide_center_extents_m=np.stack(((guides + offset).min(axis=(0, 1)),
                                                           (guides + offset).max(axis=(0, 1)))).tolist())
    if complete:
        metrics['eye_steps_m_at_transition_edges'] = {f'{i}->{i + 1}': float(step[i])
                                                      for i in [44, 45, 164, 165]}
    return metrics, eye, target


def probe_video(path):
    command = ['ffprobe', '-v', 'error', '-show_streams', '-show_format', '-show_packets',
               '-show_entries', 'stream=codec_type,codec_name,width,height,pix_fmt,r_frame_rate,avg_frame_rate,'
                                'time_base:packet=pts,duration:format=duration', '-of', 'json', str(path)]
    data = json.loads(subprocess.check_output(command))
    require(len(data['streams']) == 1, 'Expected one video stream and no additional streams')
    stream, = data['streams']
    require(stream['codec_type'] == 'video' and (stream['width'], stream['height']) == (WIDTH, HEIGHT),
            'Encoded video stream dimensions differ')
    require(stream['avg_frame_rate'] == '30/1' and stream['r_frame_rate'] == '30/1', 'Encoded frame rate differs')
    require(abs(float(data['format']['duration']) - 24) < 1e-4, 'Encoded duration differs')
    num, den = map(int, stream['time_base'].split('/'))
    pts = sorted(int(packet['pts']) for packet in data['packets'])
    require(len(pts) == COUNT and all(p * num * FPS == i * den for i, p in enumerate(pts)),
            'Encoded presentation times are not exactly 720 equally spaced 30 fps samples from zero')
    require(all(int(packet['duration']) * num * FPS == den for packet in data['packets']),
            'At least one encoded packet duration differs from one frame at 30 fps')
    return dict(codec=stream['codec_name'], dimensions=[WIDTH, HEIGHT], fps=FPS, duration_s=24,
                frames=COUNT, pix_fmt=stream['pix_fmt'], all_presentation_times_verified=True)


def track_video(capture, video, report, eyes, targets, positions, guides, times):
    profile = report['camera_profile']
    grid, plates = patch_geometry()
    raw = np.empty((COUNT, len(plates), 49), dtype=np.float32)
    encoded = np.empty_like(raw)
    visible = np.empty(raw.shape[:2], dtype=bool)
    frame_mae = np.empty(COUNT)
    command = ['ffmpeg', '-nostdin', '-v', 'error', '-xerror', '-err_detect', 'explode', '-threads', '2',
               '-i', str(video), '-an', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-threads', '2', 'pipe:1']
    with subprocess.Popen(command, stdout=subprocess.PIPE) as process:
        try:
            for j, row in enumerate(report['frames']):
                path = capture / f'{j:06d}.png'
                require(row['file'] == path.name and row['sha256'] == sha(path), f'PNG hash or filename differs at {j}')
                require(row['actual_t_s'] == times[j], f'Exact archived timestamp differs at {j}')
                state = hashlib.sha256(positions[j].tobytes() + guides[j].tobytes()).hexdigest()
                require(row['state_sha256'] == state, f'Archived state hash differs at {j}')
                native = cv2.imread(str(path), cv2.IMREAD_COLOR)
                require(native is not None and native.shape == (HEIGHT, WIDTH, 3), f'PNG decode or size failed at {j}')
                buffer = process.stdout.read(HEIGHT * WIDTH * 3)
                require(len(buffer) == HEIGHT * WIDTH * 3, f'Video decode incomplete at {j}')
                decoded = np.frombuffer(buffer, dtype=np.uint8).reshape(HEIGHT, WIDTH, 3)
                px, visible[j] = sample_positions(grid, eyes[j], targets[j], profile)
                for image, sink in [(native, raw), (decoded, encoded)]:
                    sink[j] = cv2.remap(image, px[:, :, 0], px[:, :, 1], cv2.INTER_LINEAR) @ BGR_TO_LUMA
                raster_diff = (native[4::8, 4::8].astype(np.float32) - decoded[4::8, 4::8]) @ BGR_TO_LUMA
                frame_mae[j] = np.abs(raster_diff).mean()
            require(not process.stdout.read(1) and process.wait() == 0, 'Video decode had extra frames or errors')
        except BaseException:
            process.kill()
            raise
    require(frame_mae.max() < 5, 'At least one decoded frame departs materially from its paired PNG')
    return raw, encoded, visible, plates, frame_mae


def pick_mask(valid, plates, focus):
    # Keep the near phase focused on the three physically closest rows, even if farther plates fit.
    return valid & (plates < 24) if focus == 'near' else valid


def metric_rows(field, visibility, plates, frame_numbers, focus, label):
    # Collapse each 49-point patch to its median, then one signed vote per distinct plate.
    patch = np.median(field, axis=-1)
    plate_values = []
    counts, patch_counts = [], []
    candidates = []
    all_patches, all_samples = [], []
    for values, patches, valid, frame in zip(field, patch, visibility, frame_numbers):
        valid = pick_mask(valid, plates, focus)
        seen = np.unique(plates[valid])
        require(len(seen) >= 3, f'{label}: fewer than three plates at frame {frame}')
        counts.append(len(seen))
        patch_counts.append(int(valid.sum()))
        plate_values.append(np.median([np.median(patches[valid & (plates == i)]) for i in seen]))
        all_patches.extend(patches[valid])
        all_samples.append(values[valid].ravel())
        selected = np.flatnonzero(valid)[np.argmax(np.abs(patches[valid]))]
        candidates.append((int(frame), int(selected), rounded(patches[selected])))
    ordered = sorted(zip(frame_numbers, plate_values), key=lambda item: abs(item[1]), reverse=True)
    return dict(min_visible_plates=min(counts), max_visible_plates=max(counts), min_visible_patches=min(patch_counts),
                max_visible_patches=max(patch_counts), coherent_abs=summary(np.abs(plate_values)),
                coherent_signed_median=rounded(np.median(plate_values)),
                local_patch_abs=summary(np.abs(all_patches)), local_sample_abs=summary(np.abs(np.concatenate(all_samples))),
                largest_coherent=[dict(frame=int(f), signed_luma=rounded(v)) for f, v in ordered[:3]],
                largest_local=[dict(frame=f, plate=int(plates[k]), patch=k % 4, signed_luma=v)
                               for f, k, v in sorted(candidates, key=lambda item: abs(item[2]), reverse=True)[:3]])


def lighting(values, visible, plates):
    adjacent = np.diff(values, axis=0)
    adjacent_visible = visible[:-1] & visible[1:]
    residual = values[1:-1] - (values[:-2] + values[2:]) * .5
    residual_visible = visible[1:-1] & visible[:-2] & visible[2:]
    out = {}
    for name, (a, b) in BANDS.items():
        out[name] = dict(frames=[a, b - 1],
            adjacent=metric_rows(adjacent[a:b - 1], adjacent_visible[a:b - 1], plates, np.arange(a + 1, b), name, name),
            temporal_residual=metric_rows(residual[a:b - 2], residual_visible[a:b - 2], plates, np.arange(a + 1, b - 1), name, name))
    for i, label in [(45, 'near_to_pullback'), (165, 'pullback_to_whole')]:
        focus = 'near' if i == 45 else 'whole'
        out[label] = dict(frames=[i - 1, i],
            adjacent=metric_rows(adjacent[i - 1:i], adjacent_visible[i - 1:i], plates, [i], focus, label),
            temporal_residual=metric_rows(residual[i - 2:i], residual_visible[i - 2:i], plates, [i - 1, i], focus, label))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--video', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    report_path = args.capture / 'native_report.json'
    report = read_json(report_path)
    archived = read_json(ARCHIVED)
    run_report = read_json(SOURCE / 'report.json')
    native_exit = read_json(args.capture.parent / 'native-exit.json')['exit_code']
    capture_check = read_json(args.capture.parent / 'capture-check.json')
    require(native_exit == 0 and capture_check['native_exit'] == 0 and report['complete'] and not report['errors']
            and report['app_close_invoked'], 'Native capture did not complete cleanly')
    require(not report['physics_stepping'] and report['capture_clock_delta_max_s'] == 0
            and capture_check['physics_recomputed'] is False, 'Physics stepping/clock advance recorded')
    require(report['amplitude_scale'] == 1 and not report['image_overlay']
            and report['max_render_position_error_m'] <= 1e-7, 'Replay geometry or presentation was modified')
    require(not report['render_settings']['automatic_exposure'], 'Automatic exposure is enabled')
    require(report['displayed_worlds'] == 64 and report['simulated_worlds'] == 1024
            and run_report['settings']['worlds'] == 1024 and run_report['displayed_worlds'] == 64
            and run_report['complete'] and run_report['contact_qualified'], 'Run qualification or world counts differ')
    require(report['trajectory_sha256'] == archived['trajectory_sha256'] == run_report['trajectory_sha256']
            == sha(SOURCE / 'trajectory.npz') and report['run_report_sha256'] == archived['run_report_sha256']
            == sha(SOURCE / 'report.json'), 'Original trajectory or run-report hash differs')
    require(report['asset_sha256'] == archived['asset_sha256'], 'Appearance asset hash metadata differs')
    require({k: v for k, v in report['source_sha256'].items() if k != 'render.py'} ==
            {k: v for k, v in archived['source_sha256'].items() if k != 'render.py'}, 'Non-renderer source hashes differ')
    require(capture_check['native_report_sha256'] == sha(report_path), 'Parent capture report hash differs')
    worker = args.capture.parent / 'source/demo/contact_arena/render.py'
    require(capture_check['worker_sha256'] == report['source_sha256']['render.py'] == sha(worker), 'Renderer worker hash differs')
    appearance = audit_appearance_sources(args.capture, report, archived)
    require([row['trajectory_index'] for row in report['frames']] == list(range(COUNT)),
            'Expected exactly one image for each source state, in original order')
    with np.load(SOURCE / 'trajectory.npz') as data:
        positions, guides, times = (data[key] for key in ('positions', 'guides', 't'))
    require(positions.shape == (720, 64, 65, 3) and guides.shape == (720, 64, 3)
            and times.shape == (720,) and np.max(np.abs(np.diff(times) - 1 / FPS)) < 1e-9,
            'Unexpected archived source data shape or cadence')
    camera, eyes, targets = audit_camera(report, positions, guides)
    video_info = probe_video(args.video)
    raw, encoded, visible, plates, raster_error = track_video(args.capture, args.video, report, eyes, targets,
                                                              positions, guides, times)
    video_info.update(sha256=sha(args.video), full_decode=True, paired_raster_frame_mae=summary(raster_error),
                      paired_visible_patch_luma_abs_error=summary(np.abs(raw[visible] - encoded[visible])))
    info = dict(host='ps', native_capture=str(args.capture), video_path=str(args.video), video=video_info,
                native_exit=native_exit, native_report_sha256=sha(report_path), capture_clock_delta_max_s=0,
                displayed_worlds=64, simulated_worlds=1024, all_png_and_source_state_hashes_verified=True,
                all_source_times_verified=True, non_renderer_source_hashes_match_archive=True,
                asset_hash_metadata_matches_archive=True, trajectory_sha256=report['trajectory_sha256'],
                run_report_sha256=report['run_report_sha256'], renderer_source_sha256=report['source_sha256']['render.py'],
                capture_render_settings=report['render_settings'], appearance=appearance,
                actual_source_time_range_s=[float(times[0]), float(times[-1])], camera=camera,
                units="8-bit luma code values, Y' = 0.299 R' + 0.587 G' + 0.114 B'.",
                method='Track 7×7 samples in fixed 5.6 cm square floor-plate patches via each recorded camera pose. '
                       'Require the same projected patch to fit in every adjacent pair or residual triple. '
                       'Near uses the first three rows; other phases use all qualifying plates. '
                       'Compute the signed median luma change per patch and take one median vote per plate. '
                       'Local temporal residual is the center sample minus the average of its neighbors. '
                       'This reduces smooth illumination drift but retains sharp local shadows/reflections.',
                scope='These measurements cover only projected floor-plate patches. They do not prove flicker absence '
                      'over the full image, and patches can include transient occlusions, shadow edges or reflections.',
                patch_offsets_m=[[-.4, -.24], [.4, -.24], [-.4, .24], [.4, .24]],
                native=lighting(raw, visible, plates), encoded=lighting(encoded, visible, plates))
    args.out.write_text(json.dumps(info, indent=2, allow_nan=False) + '\n')
    print(json.dumps({key: info[key] for key in ('native_exit', 'video', 'camera')}, indent=2))
    for source in ('native', 'encoded'):
        for phase, values in info[source].items():
            adjacent, residual = (values[key] for key in ('adjacent', 'temporal_residual'))
            print(source, phase, 'visible plates', [adjacent['min_visible_plates'], adjacent['max_visible_plates']],
                  'coherent adjacent p95/max', [adjacent['coherent_abs'][k] for k in ('p95', 'max')],
                  'local patch residual p95/max', [residual['local_patch_abs'][k] for k in ('p95', 'max')],
                  'largest coherent adjacent', adjacent['largest_coherent'][0],
                  'largest local residual', residual['largest_local'][0])
    print(f'Full QA: {args.out}')


if __name__ == '__main__':
    main()
