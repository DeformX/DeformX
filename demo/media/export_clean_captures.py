#!/usr/bin/env python3
"""Encode saved Isaac RGB captures without adding titles or a HUD.

Run on the rendering host with Pillow, FFmpeg and FFprobe installed:
  python demo/media/export_clean_captures.py --basics /path/to/basic-capture \
      --arena /path/to/arena-capture --out /path/to/new-output

Capture folders and their original JSON manifests must be retained separately.
Playback frame maps are preserved; no simulation, cropping or retiming is done.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def encode(paths, video, poster, poster_frame):
    from PIL import Image
    args=['ffmpeg','-nostdin','-v','error','-n','-f','rawvideo','-pix_fmt','rgb24',
          '-s','1280x720','-framerate','30','-i','-','-an','-c:v','libx264',
          '-threads','2','-crf','18','-pix_fmt','yuv420p','-video_track_timescale','30000',
          '-movie_timescale','30000','-movflags','+faststart',str(video)]
    with subprocess.Popen(args,stdin=subprocess.PIPE) as process:
        last=None
        for i,path in enumerate(paths):
            if path!=last:
                with Image.open(path) as im:
                    im=im.convert('RGB')
                    assert im.size==(1280,720)
                    pixels=im.tobytes()
                last=path
            if i==poster_frame:
                with Image.open(path) as im: im.convert('RGB').save(poster,quality=95)
            process.stdin.write(pixels)
        process.stdin.close()
        if process.wait()!=0: raise RuntimeError('FFmpeg encode failed')
    subprocess.run(['ffmpeg','-v','error','-xerror','-err_detect','explode','-i',str(video),
                    '-f','null','-'],check=True)
    info=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames',
        '-show_streams','-show_format','-of','json',str(video)]))
    s=info['streams'][0]
    assert len(info['streams'])==1 and s['codec_name']=='h264'
    assert int(s['nb_read_frames'])==len(paths) and s['avg_frame_rate']=='30/1'
    assert (s['width'],s['height'])==(1280,720)
    assert abs(float(s['duration'])-len(paths)/30)<1e-5
    return dict(video=video.name,sha256=sha(video),poster=poster.name,poster_sha256=sha(poster),
                frames=len(paths),fps=30,duration_s=len(paths)/30,full_decode=True,
                added_text=False,cropped=False,amplitude_scale=1)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--basics',type=Path,required=True)
    p.add_argument('--arena',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    manifest=json.loads((a.basics/'manifest.json').read_text())
    native=json.loads((a.basics/'native-report.json').read_text())
    assert native['complete'] and native['app_close_invoked']
    report=dict(script_sha256=sha(Path(__file__)),basics_manifest_sha256=sha(a.basics/'manifest.json'),
                basics_native_report_sha256=sha(a.basics/'native-report.json'),cases={})
    for name,case in manifest['cases'].items():
        audit=native['cases'][name];mapping={}
        for row in audit['frames']:
            file=a.basics/name/row['file']
            assert sha(file)==row['image_sha256']
            mapping[row['view'],row['source_index']]=file
        paths=[mapping[segment['view'],index] for segment in case['segments'] for index in segment['source_indices']]
        target=a.out/name;target.mkdir()
        result=encode(paths,target/(name+'.mp4'),target/(name+'-preview.jpg'),len(paths)//2)
        result['trajectory_sha256']=audit['trajectory_sha256']
        result['playback_segments']=[{k:s[k] for k in ['view','speed','first_video_frame','end_video_frame_exclusive']} for s in case['segments']]
        report['cases'][name]=result
        print(name,result['frames'],flush=True)
    arena=json.loads((a.arena/'native_report.json').read_text())
    assert arena['complete'] and not arena['errors']
    paths=[]
    for row in arena['frames']:
        file=a.arena/row['file'];assert sha(file)==row['sha256'];paths.append(file)
    target=a.out/'contact-arena';target.mkdir()
    result=encode(paths,target/'contact-arena-long.mp4',target/'long-preview.jpg',len(paths)//2)
    result.update(native_report_sha256=sha(a.arena/'native_report.json'),trajectory_sha256=arena['trajectory_sha256'],
                  displayed_worlds=arena['displayed_worlds'],simulated_worlds=arena['simulated_worlds'],playback_speed=1)
    report['cases']['contact-arena']=result
    (a.out/'clean-captures-provenance.json').write_text(json.dumps(report,indent=2)+'\n')
    print('contact-arena',len(paths),flush=True)


if __name__=='__main__': main()
