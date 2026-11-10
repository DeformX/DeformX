"""Verify native guitar capture, encode it with FFmpeg, and decode every frame."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from . import contact_fullbody_framing as framing
from . import contact_loop_pullback as pullback
from .contact_integrity import sha
from .portable_audit import validate_capture


def verify_capture(out):
    report = json.loads((out / 'native_report.json').read_text())
    frames = json.loads((out / 'frames.json').read_text())
    camera_file = out / 'camera_profile.json'
    if (report.get('complete') is not True or report['errors'] or not report['app_close_invoked']
            or report['frames'] != report['planned_frames'] or len(frames) != report['frames']
            or report['capture_clock_delta_max_s'] > 1e-9 or report['replay_max_matrix_error'] > 1e-5
            or report['physics_steps_during_capture'] != 0 or report['amplitude_scale'] != 1.
            or report['loop_edit'] != pullback.FULLBODY_EDIT or report['record_hz'] != 480):
        raise ValueError('Native replay did not complete with the recorded geometry and clock held fixed')
    if (sha(camera_file) != report['camera_profile_sha256']
            or sha(Path(__file__).with_name('contact_render_native.py')) != report['source_sha256']
            or sha(pullback.__file__) != report['loop_replay_source_sha256']
            or sha(framing.__file__) != report['fullbody_framing_source_sha256']):
        raise ValueError('Camera or capture source changed before encoding')
    case = Path(report['case_path'])
    proof = validate_capture(case, Path(report['audit_path']))
    if any(report.get(key) != value for key, value in proof.items()):
        raise ValueError('Measured single-case audit changed before encoding')
    case_report = json.loads((case / 'report.json').read_text())
    if any(report.get(key) != case_report[key] for key in ('profile_sha256', 'motion_sha256', 'trajectory_sha256')):
        raise ValueError('Capture does not bind the audited profile, robot motion and string trajectory')
    if len(list((out / 'frames').glob('*.png'))) != len(frames):
        raise ValueError('Native PNG frame count changed')
    for i, row in enumerate(frames):
        if row['file'] != f'frames/{i:06d}.png' or row['png_sha256'] != sha(out / row['file']):
            raise ValueError(f'Native PNG changed at frame {i}')
    profile = json.loads((case / 'profile.json').read_text())
    camera_proof = pullback.verify_pullback_edit(
        profile['contact_motion'], frames, report['loop_replay'], json.loads(camera_file.read_text()),
        profile['guitar']['placement'], pullback.FULLBODY_EDIT,
    )
    framing_proof = framing.verify_fullbody_framing(frames, report['frame_resolution_px'])
    if camera_proof != report['pullback_camera_proof'] or framing_proof != report['fullbody_framing_proof']:
        raise ValueError('Native camera or full-body framing proof changed')
    return report


def encode(out, capture_exit_code=None):
    report = verify_capture(out)
    video = out / 'humanoid-guitar-fullbody-long.mp4'
    subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-n', '-framerate', '30',
                    '-i', str(out / 'frames/%06d.png'), '-c:v', 'libx264', '-threads', '8',
                    '-crf', '18', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(video)], check=True)
    probe = json.loads(subprocess.check_output([
        'ffprobe', '-v', 'error', '-select_streams', 'v:0', '-count_frames', '-show_entries',
        'stream=codec_name,width,height,pix_fmt,avg_frame_rate,nb_read_frames:format=duration',
        '-of', 'json', str(video),
    ], text=True))
    stream = probe['streams'][0]
    if (int(stream['nb_read_frames']) != report['frames'] or stream['avg_frame_rate'] != '30/1'
            or [stream['width'], stream['height']] != report['frame_resolution_px']):
        raise ValueError('Encoded frame count, cadence or resolution differs from native capture')
    subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-xerror', '-i', str(video), '-f', 'null', '-'], check=True)
    summary = dict(complete=True, full_decode_success=True, frames=report['frames'],
                   acceptance_scope=report['acceptance_scope'], video_sha256=sha(video),
                   native_report_sha256=sha(out / 'native_report.json'), frame_map_sha256=sha(out / 'frames.json'),
                   trajectory_sha256=report['trajectory_sha256'], native_capture_exit_code=capture_exit_code, probe=probe)
    (out / 'video_report.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(video)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', type=Path)
    parser.add_argument('--audit', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--encode-existing', action='store_true')
    args = parser.parse_args()
    capture_exit_code = None
    if not args.encode_existing:
        if args.case is None or args.audit is None:
            parser.error('--case and --audit are required for native capture')
        cmd = [sys.executable, *(['-S'] if sys.flags.no_site else []), '-m', 'demo.humanoid_guitar.contact_render_native',
               '--case', str(args.case), '--audit', str(args.audit), '--out', str(args.out)]
        capture_exit_code = subprocess.run(cmd, check=True).returncode
    encode(args.out, capture_exit_code)


if __name__ == '__main__':
    main()
