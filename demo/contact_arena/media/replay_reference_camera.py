"""Reframe the archived contact arena from two front cells to all 64 cells.

Run on ps with Isaac Sim Python. Original numerical sources and assets are
copied and hash-checked; only the native camera and capture evidence change.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

RENDER_SHA = 'b304c101414ae082a0c9225540173eba84aae886e6c26a70e334c6341bf1f75c'
TRAJECTORY_SHA = 'aae257f82615bcbcbe780e2a8f0b576a40a322628a3465f23d27afcfd62f5c9a'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def write(p, value):
    p.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def patch(source, reference_frame):
    def replace(old, new):
        nonlocal source
        assert source.count(old) == 1, old[:100]
        source = source.replace(old, new)
    replace("        key=UsdLux.DistantLight.Define", """        # Every Blender cell/guide already carries the same dark HDR dome.
        # Explicitly exclude the competing untextured dome to retain archived appearance.
        UsdGeom.Imageable(dome.GetPrim()).MakeInvisible()
        report['environment_selection']=dict(disabled_dome='/World/Dome',
            selected_environment='Original Blender asset color_121212.hdr; distant key unchanged')
        key=UsdLux.DistantLight.Define""")
    camera_setup = """        import carb.settings
        import hashlib
        render_settings = carb.settings.get_settings()
        camera.prim.GetAttribute('horizontalAperture').Set(20.955)
        camera.prim.GetAttribute('verticalAperture').Set(20.955 * 720 / 1280)
        near_target = np.array([0., -2.52, .40])
        near_eye = np.array([.35, -5.3, 1.9])
        # Include the entire recorded cable envelope, plates, pedestals and guides.
        cable_low = (P.min(axis=(0,2)) + offset).min(axis=0) - RADIUS
        cable_high = (P.max(axis=(0,2)) + offset).max(axis=0) + RADIUS
        low = np.minimum(offset.min(axis=0) + [-.675, -.325, -.0725], cable_low)
        high = np.maximum(offset.max(axis=0) + [.675, .325, .67], cable_high)
        corners = np.array([[x,y,z] for x in (low[0],high[0])
            for y in (low[1],high[1]) for z in (low[2],high[2])])
        whole_target = (low+high)/2
        aperture = np.array([float(camera.prim.GetAttribute(name).Get())
            for name in ('horizontalAperture','verticalAperture')])
        def projected(eye, target):
            matrix=Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye),Gf.Vec3d(*target),Gf.Vec3d(0,0,1))
            q=np.array([list(matrix.Transform(Gf.Vec3d(*p))) for p in corners])
            if np.any(q[:,2]>=0): raise ValueError('Arena bounds behind camera')
            return 2*23*q[:,:2]/(-q[:,2,None]*aperture)
        distance=7.
        for _ in range(100):
            whole_eye=whole_target+distance*np.array([0.,-1.,.9])
            ndc=projected(whole_eye,whole_target)
            if abs(ndc).max()<=.92: break
            distance*=1.025
        else: raise ValueError('Could not frame complete arena')
        report['camera_profile']=dict(near_eye_m=near_eye.tolist(),near_target_m=near_target.tolist(),
            whole_eye_m=whole_eye.tolist(),whole_target_m=whole_target.tolist(),
            near_hold_s=[0,1.5],pullback_s=[1.5,5.5],whole_hold_s=[5.5,24],
            bounds_m=[low.tolist(),high.tolist()],max_abs_whole_bounds_ndc=float(abs(ndc).max()),
            whole_ndc_limit=.92,focal_length_mm=23,horizontal_aperture_mm=float(aperture[0]),
            vertical_aperture_mm=float(aperture[1]),transition='quintic smootherstep')
        report['max_render_position_error_m']=0.
        report['image_overlay']=False
        report['physics_stepping']=False
"""
    replace("indices=sorted(set([0,len(ts)//2,len(ts)-1])) if a.preview else range(len(ts))",
            "indices=[0,45,90,135,165,719] if a.preview else range(len(ts))")
    start=source.index('            u=float(np.clip(')
    end=source.index('            if reset:', start)
    first_frame_setup = '''            if frame==0:
                # Preserve original initialization before fitting/moving the camera.
                look([0,-10,11],[0,0,.3])
                if reset: reset()
                for _ in range(12): tick()
                baseline=np.asarray(camera.get_rgba(device='cpu'))[:,:,:3]
                Image.fromarray(baseline).save(a.out/'appearance-baseline.png')
                with Image.open(__ARENA_REFERENCE_FRAME__) as reference:
                    difference=float(np.abs(baseline.astype(float)-np.asarray(reference.convert('RGB'))).mean())
                report['archived_pose_rgb_mae']=difference
                if difference>5: raise RuntimeError(f'Archived appearance mismatch: RGB MAE {difference}')
''' + ''.join('        '+line+'\n' for line in camera_setup.splitlines())
    source=source[:start]+first_frame_setup+'''            elapsed=j/30.
            u=float(np.clip((elapsed-1.5)/4.,0,1)); u=u*u*u*(u*(u*6-15)+10)
            eye=(1-u)*near_eye+u*whole_eye
            camera_target=(1-u)*near_target+u*whole_target
            look(eye,camera_target)
            actual_nodes=np.asarray(nodes.GetPositionsAttr().Get())
            actual_rods=np.asarray(rods.GetPositionsAttr().Get())
            actual_guides=np.array([list(op.Get()) for op in movers])
            error=max(float(abs(actual_nodes-p.reshape(-1,3).astype(np.float32)).max()),
                float(abs(actual_rods-(start+.5*d).astype(np.float32)).max()),
                float(abs(actual_guides-(guides[j]+offset)).max()))
            if error>1e-7: raise ValueError('Recorded geometry changed during replay')
            report['max_render_position_error_m']=max(report['max_render_position_error_m'],error)
            state_hash=hashlib.sha256(P[j].tobytes()+guides[j].tobytes()).hexdigest()
''' + source[end:]
    source=source.replace('__ARENA_REFERENCE_FRAME__',repr(str(reference_frame)))
    replace("            rgb=np.asarray(camera.get_rgba(device='cpu'))[:,:,:3]", """            aa=render_settings.get('/rtx/post/aa/op')
            auto_exposure=bool(render_settings.get('/rtx/post/histogram/enabled'))
            if auto_exposure: raise ValueError('Automatic exposure must remain disabled')
            report['render_settings']=dict(anti_aliasing_mode=aa,automatic_exposure=auto_exposure,
                lighting='Original Blender HDR environment selected explicitly; original distant key and materials unchanged')
            rgb=np.asarray(camera.get_rgba(device='cpu'))[:,:,:3]""")
    replace("actual_t_s=float(ts[j]),sha256=sha(path)))", """actual_t_s=float(ts[j]),sha256=sha(path),
                state_sha256=state_hash,camera_eye_m=eye.tolist(),camera_target_m=camera_target.tolist(),
                camera_phase='near' if elapsed<1.5 else ('pullback' if elapsed<5.5 else 'whole')))""")
    return source


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('source','data','assets','original','out'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--preview',action='store_true');p.add_argument('--lock',type=Path)
    a=p.parse_args()
    assert sha(a.source/'demo/contact_arena/render.py')==RENDER_SHA
    assert sha(a.data/'trajectory.npz')==TRAJECTORY_SHA
    original=json.loads((a.original/'native_report.json').read_text())
    for name,digest in original['asset_sha256'].items():assert sha(a.assets/name)==digest
    for name,digest in original['source_sha256'].items():assert sha(a.source/'demo/contact_arena'/name)==digest
    a.out.mkdir(parents=True,exist_ok=False)
    src=a.out/'source';(src/'demo').mkdir(parents=True)
    shutil.copyfile(a.source/'demo/__init__.py',src/'demo/__init__.py')
    for package in ('contact_arena','batch_arena'):
        (src/'demo'/package).mkdir()
        for f in (a.source/'demo'/package).iterdir():
            if f.is_file() and f.suffix in ('.py','.json'):shutil.copyfile(f,src/'demo'/package/f.name)
    worker=src/'demo/contact_arena/render.py';worker.write_text(patch(worker.read_text(),a.original/'000000.png'))
    capture=a.out/'capture';capture.mkdir()
    command=[sys.executable,'-m','demo.contact_arena.render','--native','--data',str(a.data),
             '--assets',str(a.assets),'--out',str(capture)]
    if a.preview:command.append('--preview')
    if a.lock:command=['flock','-o',str(a.lock),*command]
    with (capture/'native.log').open('w') as log:
        result=subprocess.run(command,cwd=src,stdout=log,stderr=subprocess.STDOUT,timeout=2400)
    write(a.out/'native-exit.json',dict(exit_code=result.returncode))
    if result.returncode:raise RuntimeError('Native worker failed; see capture/native.log')
    report=json.loads((capture/'native_report.json').read_text())
    assert report['complete'] and not report['errors'] and report['app_close_invoked']
    assert report['capture_clock_delta_max_s']==0 and report['max_render_position_error_m']<=1e-7
    assert report['trajectory_sha256']==original['trajectory_sha256']
    assert report['run_report_sha256']==original['run_report_sha256']
    assert report['asset_sha256']==original['asset_sha256']
    import numpy as np
    from PIL import Image
    data=np.load(a.data/'trajectory.npz');indices=[0,45,90,135,165,719] if a.preview else list(range(720))
    assert [f['trajectory_index'] for f in report['frames']]==indices
    for f in report['frames']:
        j=f['trajectory_index'];assert f['actual_t_s']==float(data['t'][j])
        assert f['state_sha256']==hashlib.sha256(data['positions'][j].tobytes()+data['guides'][j].tobytes()).hexdigest()
        assert f['sha256']==sha(capture/f['file'])
        with Image.open(capture/f['file']) as im:assert im.size==(1280,720)
    write(a.out/'capture-check.json',dict(native_exit=0,images=len(indices),all_source_states_and_times_match=True,
        all_image_hashes_match=True,physics_recomputed=False,recipe_sha256=sha(Path(__file__)),
        worker_sha256=sha(worker),native_report_sha256=sha(capture/'native_report.json')))
    if not a.preview:
        delivery=a.out/'delivery';delivery.mkdir()
        video=delivery/'contact-arena-pullback.mp4'
        subprocess.run(['ffmpeg','-nostdin','-v','error','-n','-threads','2','-framerate','30','-i',str(capture/'%06d.png'),
            '-an','-c:v','libx264','-preset','medium','-crf','17','-threads','2','-pix_fmt','yuv420p','-movflags','+faststart',str(video)],check=True)
        subprocess.run(['ffmpeg','-nostdin','-v','error','-xerror','-err_detect','explode','-i',str(video),'-f','null','-'],check=True)
        metadata=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-show_streams','-show_format','-of','json',str(video)]))
        stream,=metadata['streams'];assert int(stream['nb_read_frames'])==720 and stream['avg_frame_rate']=='30/1'
        assert abs(float(metadata['format']['duration'])-24)<.001
        with Image.open(capture/'000719.png') as im:im.save(delivery/'long-preview.jpg',quality=95)
        write(delivery/'presentation.json',dict(video=video.name,video_sha256=sha(video),frames=720,fps=30,duration_s=24,
            dimensions=[1280,720],full_decode=True,host='ps',playback_multiplier=1,physics_recomputed=False,
            image_overlay=False,trajectory_sha256=TRAJECTORY_SHA,run_report_sha256=report['run_report_sha256'],
            native_report_sha256=sha(capture/'native_report.json'),recipe_sha256=sha(Path(__file__)),
            original_native_report_sha256=sha(a.original/'native_report.json'),original_renderer_sha256=RENDER_SHA,
            native_worker_sha256=sha(worker),asset_sha256=report['asset_sha256'],camera=report['camera_profile'],
            capture_check=json.loads((a.out/'capture-check.json').read_text()),render_settings=report['render_settings'],
            archived_pose_rgb_mae=report['archived_pose_rgb_mae'],environment_selection=report['environment_selection'],
            poster_sha256=sha(delivery/'long-preview.jpg')))
    print(json.dumps(dict(preview=a.preview,frames=len(indices),camera=report['camera_profile']),indent=2))


if __name__=='__main__':main()
