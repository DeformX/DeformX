"""Make a 2× presentation from a verified 1× capture without re-encoding frames."""
import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import subprocess


def command(*args):
    return subprocess.check_output([str(x) for x in args], stderr=subprocess.PIPE)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def probe(path):
    return json.loads(command('ffprobe', '-v', 'error', '-count_frames',
                             '-show_streams', '-show_format', '-of', 'json', path))


def packets(path):
    return json.loads(command('ffprobe', '-v', 'error', '-select_streams', 'v:0',
                             '-show_packets', '-show_data_hash', 'sha256',
                             '-show_entries', 'packet=pts,dts,duration,data_hash',
                             '-of', 'json', path))['packets']


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source', type=Path, required=True)
    ap.add_argument('--capture', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    capture = json.loads(a.capture.read_text())
    if not capture['full_decode_passed'] or sha(a.source) != capture['video_sha256']:
        raise ValueError('Source must match a verified native capture')
    original = probe(a.source)
    before, = original['streams']
    if before['codec_type'] != 'video' or Fraction(before['avg_frame_rate']) != 30:
        raise ValueError('Expected one silent 30 fps native video')
    a.out.mkdir(parents=True, exist_ok=False)
    video = a.out / 'cable-extraction.mp4'
    command('ffmpeg', '-nostdin', '-v', 'error', '-n',
            '-i', a.source, '-map', '0:v:0', '-c', 'copy', '-r', '60', '-an',
            '-bsf:v', 'setts=pts=PTS/2:dts=DTS/2:duration=DURATION/2',
            '-map_metadata', '-1', '-movflags', '+faststart', video)
    command('ffmpeg', '-nostdin', '-v', 'error', '-xerror', '-err_detect',
            'explode', '-i', video, '-f', 'null', '-')
    result = probe(video)
    after, = result['streams']
    for key in ('codec_name', 'width', 'height', 'pix_fmt', 'nb_read_frames'):
        if before[key] != after[key]:
            raise ValueError(f'Changed source frames: {key}')
    if Fraction(after['avg_frame_rate']) != 60:
        raise ValueError('Expected exact 2×, 60 fps presentation')
    if abs(float(after['duration']) * 2 - float(before['duration'])) > 1e-6:
        raise ValueError('Duration was not halved')
    old_packets, new_packets = packets(a.source), packets(video)
    if len(old_packets) != len(new_packets):
        raise ValueError('Packet count changed')
    old_tb, new_tb = Fraction(before['time_base']), Fraction(after['time_base'])
    for old, new in zip(old_packets, new_packets):
        if old['data_hash'] != new['data_hash']:
            raise ValueError('Encoded frame payload changed')
        for key in ('pts', 'dts', 'duration'):
            if int(old[key]) * old_tb / 2 != int(new[key]) * new_tb:
                raise ValueError(f'Incorrect {key} retiming')
    proof = dict(source_video_sha256=sha(a.source), source_capture_sha256=sha(a.capture),
                 source_duration_s=float(before['duration']),
                 video=video.name, video_sha256=sha(video),
                 duration_s=float(after['duration']), frames=int(after['nb_read_frames']),
                 fps=60, playback_multiplier=2,
                 dimensions=[after['width'], after['height']],
                 encoded_frame_payloads_identical=True, all_timestamps_halved=True,
                 frame_dropping=False, frame_interpolation=False, full_decode=True,
                 physics_recomputed=False, text_overlays=False,
                 retime_source_sha256=sha(Path(__file__)),
                 ffmpeg=command('ffmpeg', '-version').decode().splitlines()[0])
    (a.out / 'presentation.json').write_text(json.dumps(proof, indent=2)+'\n')
    print(json.dumps(proof, indent=2))


if __name__ == '__main__':
    main()
