"""Encode and independently verify a complete native capture."""
import argparse
import json
from pathlib import Path
import subprocess

from .model import FPS
from .run import sha


def ffmpeg(*args):
    subprocess.run(['ffmpeg','-hide_banner','-loglevel','error',*map(str,args)],check=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--capture',type=Path,required=True)
    ap.add_argument('--native-exit',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--poster-frame',type=int,required=True)
    a = ap.parse_args()
    native = json.loads((a.capture/'native_report.json').read_text())
    frames = json.loads((a.capture/'frames.json').read_text())
    if (a.native_exit.read_text().strip() != '0' or not native['complete'] or native['errors']
            or native.get('capture_mode') != 'full' or not native.get('accepted_audit')
            or native.get('source_frames') != len(frames) or not native.get('audit_sha256')
            or native['overlay'] or native['rate'] != 1. or native['frames'] != len(frames)
            or native['max_capture_clock_delta_s'] > 1e-9 or native['max_replay_matrix_error'] > 1e-5):
        raise RuntimeError('Invalid full native capture')
    for i,row in enumerate(frames):
        if (row['file'] != f'frames/{i:06d}.png' or row['trajectory_frame'] != i or
                abs(row['t_s']-(i+1)/FPS) > 1e-9 or sha(a.capture/row['file']) != row['sha256']):
            raise RuntimeError(f'Frame or clock mismatch at {i}')
    if not 0 <= a.poster_frame < len(frames):
        raise ValueError('Poster frame must be inside the verified capture')
    a.out.mkdir(parents=True)
    movie = a.out/'cable-extraction.mp4'
    poster = a.out/'preview.jpg'
    ffmpeg('-framerate',FPS,'-i',a.capture/'frames/%06d.png','-an','-c:v','libx264',
           '-preset','slow','-crf',18,'-pix_fmt','yuv420p','-movflags','+faststart',movie)
    ffmpeg('-i',a.capture/frames[a.poster_frame]['file'],'-frames:v',1,'-q:v',3,'-update',1,poster)
    ffmpeg('-xerror','-i',movie,'-f','null','-')
    probe = subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-count_frames',
        '-show_entries','stream=codec_name,width,height,pix_fmt,avg_frame_rate,duration,nb_read_frames',
        '-of','json',str(movie)],text=True)
    video = json.loads(probe)['streams'][0]
    if (int(video['nb_read_frames']) != len(frames) or video['avg_frame_rate'] != f'{FPS}/1' or
            abs(float(video['duration'])-len(frames)/FPS) > 1e-5):
        raise RuntimeError('Encoded playback does not match native capture')
    proof = dict(native=native, native_report_sha256=sha(a.capture/'native_report.json'),
                encode_source_sha256=sha(__file__),
                verified_frames_manifest_sha256=sha(a.capture/'frames.json'),
                full_decode_passed=True, video=video, video_sha256=sha(movie),
                poster_frame=a.poster_frame, poster_simulated_t_s=frames[a.poster_frame]['t_s'],
                poster_source_frame_sha256=frames[a.poster_frame]['sha256'], poster_sha256=sha(poster))
    (a.out/'capture.json').write_text(json.dumps(proof,indent=2,allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
