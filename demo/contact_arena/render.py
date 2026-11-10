"""Native Isaac replay of the contact arena; no physics is stepped during capture."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import traceback
from .run import sha, write


def native(a):
    report=dict(complete=False,errors=[],frames=[],app_close_invoked=False,
                capture_clock_delta_max_s=0.,amplitude_scale=1.,renderer='Isaac Sim RaytracedLighting',
                trajectory_sha256=sha(a.data/'trajectory.npz'),run_report_sha256=sha(a.data/'report.json'),
                source_sha256={p.name:sha(p) for p in Path(__file__).parent.glob('*.py')},
                asset_sha256={p.name:sha(p) for p in a.assets.glob('*') if p.is_file()})
    target=a.out/'native_report.json'
    # Reserve the worker's evidence atomically, including direct --native calls.
    with target.open('x') as stream:
        json.dump(report,stream,indent=2,allow_nan=False); stream.write('\n')
    app=camera=world=None
    try:
        import numpy as np
        from .model import COLORS, RADIUS
        source=json.loads((a.data/'report.json').read_text())
        if not source['complete'] or not source['contact_qualified'] or source['trajectory_sha256']!=report['trajectory_sha256']:
            raise ValueError('Run is incomplete, unqualified or changed')
        for name in ('run.py','model.py'):
            if source['source_sha256'][name]!=sha(Path(__file__).with_name(name)):
                raise ValueError('Simulation source changed since recording: '+name)
        data=np.load(a.data/'trajectory.npz'); P=data['positions']; ts=data['t']; guides=data['guides']; family=data['family']
        if not np.isfinite(P).all() or not np.allclose(np.diff(ts),1/30,atol=1e-9): raise ValueError('Invalid recording')
        n=P.shape[1]; side=int(np.ceil(np.sqrt(n))); ids=np.arange(n)
        offset=np.c_[(ids%side-(side-1)/2)*1.4,(ids//side-(side-1)/2)*.72,np.zeros(n)]
        from isaacsim import SimulationApp
        app=SimulationApp(dict(headless=True,hide_ui=True,active_gpu=0,physics_gpu=0,multi_gpu=False,max_gpu_count=1,
            limit_cpu_threads=8,width=1280,height=720,renderer='RaytracedLighting',anti_aliasing=2,sync_loads=True,fast_shutdown=True,enable_crashreporter=False))
        from pxr import UsdGeom,UsdLux,UsdShade,Gf,Sdf,Vt
        from isaacsim.core.api import World
        from isaacsim.core.utils.stage import get_current_stage
        from isaacsim.sensors.camera import Camera
        import omni.replicator.core as rep
        import omni.timeline,omni.usd
        from PIL import Image
        world=World(physics_dt=1/120,rendering_dt=1/30,stage_units_in_meters=1.,backend='numpy',device='cpu')
        stage=get_current_stage(); UsdGeom.SetStageUpAxis(stage,UsdGeom.Tokens.z)
        movers=[]
        for i in range(n):
            cell=UsdGeom.Xform.Define(stage,f'/World/Cell{i}'); cell.GetPrim().GetReferences().AddReference(str(a.assets/'cell.usda'))
            cell.AddTranslateOp().Set(Gf.Vec3d(*offset[i]))
            guide=UsdGeom.Xform.Define(stage,f'/World/Guide{i}'); guide.GetPrim().GetReferences().AddReference(str(a.assets/'guide.usda'))
            movers.append(guide.AddTranslateOp())
        def instances(path,kind,radius):
            obj=UsdGeom.PointInstancer.Define(stage,path); targets=[]
            for i,color in enumerate(COLORS):
                child=path+'/P'+str(i)
                proto=UsdGeom.Cylinder.Define(stage,child) if kind=='cylinder' else UsdGeom.Sphere.Define(stage,child)
                proto.CreateRadiusAttr(radius)
                if kind=='cylinder': proto.CreateHeightAttr(1.); proto.CreateAxisAttr('Z')
                mat=UsdShade.Material.Define(stage,child+'/Material'); shader=UsdShade.Shader.Define(stage,child+'/Material/S')
                shader.CreateIdAttr('UsdPreviewSurface'); shader.CreateInput('diffuseColor',Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
                shader.CreateInput('roughness',Sdf.ValueTypeNames.Float).Set(.28)
                mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(),'surface'); UsdShade.MaterialBindingAPI.Apply(proto.GetPrim()).Bind(mat)
                targets.append(Sdf.Path(child))
            obj.CreatePrototypesRel().SetTargets(targets); return obj
        rods=instances('/World/Cables','cylinder',RADIUS); nodes=instances('/World/Nodes','sphere',RADIUS)
        rods.CreateProtoIndicesAttr().Set(np.repeat(family,P.shape[2]-1).tolist()); nodes.CreateProtoIndicesAttr().Set(np.repeat(family,P.shape[2]).tolist())
        dome=UsdLux.DomeLight.Define(stage,'/World/Dome'); dome.CreateIntensityAttr(650); dome.CreateColorAttr(Gf.Vec3f(.76,.86,1))
        key=UsdLux.DistantLight.Define(stage,'/World/Key'); key.CreateIntensityAttr(2200); key.CreateAngleAttr(12)
        UsdGeom.Xformable(key.GetPrim()).AddRotateXYZOp().Set(Gf.Vec3f(-45,-20,20))
        camera=Camera(prim_path='/World/Camera',frequency=-1,resolution=(1280,720),annotator_device='cpu')
        camera.prim.GetAttribute('focalLength').Set(23.); camera.prim.GetAttribute('clippingRange').Set(Gf.Vec2f(.01,150))
        def look(eye,target):
            m=Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye),Gf.Vec3d(*target),Gf.Vec3d(0,0,1)).GetInverse(); q=m.ExtractRotationQuat()
            camera.set_world_pose(position=np.array(eye),orientation=np.array([q.GetReal(),*q.GetImaginary()]),camera_axes='usd')
        look([0,-10,11],[0,0,.3]); world.reset(); camera.initialize(attach_rgb_annotator=True); rep.orchestrator.set_capture_on_play(False)
        timeline=omni.timeline.get_timeline_interface(); reset=getattr(omni.usd.get_context(),'reset_renderer_accumulation',None)
        def tick():
            before=(float(world.current_time),float(timeline.get_current_time()),world.current_time_step_index)
            rep.orchestrator.step(rt_subframes=2,pause_timeline=False,delta_time=0.)
            delta=max(abs(float(world.current_time)-before[0]),abs(float(timeline.get_current_time())-before[1]))
            report['capture_clock_delta_max_s']=max(report['capture_clock_delta_max_s'],delta)
            if delta>1e-9 or world.current_time_step_index!=before[2]: raise RuntimeError('Capture clock advanced')
        indices=sorted(set([0,len(ts)//2,len(ts)-1])) if a.preview else range(len(ts))
        for frame,j in enumerate(indices):
            p=P[j]+offset[:,None,:]; start=p[:,:-1].reshape(-1,3); d=np.diff(p,axis=1).reshape(-1,3); length=np.linalg.norm(d,axis=-1)
            rods.CreatePositionsAttr().Set(Vt.Vec3fArray.FromNumpy((start+.5*d).astype(np.float32)))
            scale=np.ones_like(d); scale[:,2]=length; rods.CreateScalesAttr().Set(Vt.Vec3fArray.FromNumpy(scale.astype(np.float32)))
            orientations=[]
            for direction,l in zip(d,length):
                q=Gf.Rotation(Gf.Vec3d(0,0,1),Gf.Vec3d(*map(float,direction/max(l,1e-10)))).GetQuat()
                orientations.append(Gf.Quath(q.GetReal(),Gf.Vec3h(*q.GetImaginary())))
            rods.CreateOrientationsAttr().Set(Vt.QuathArray(orientations))
            nodes.CreatePositionsAttr().Set(Vt.Vec3fArray.FromNumpy(p.reshape(-1,3).astype(np.float32)))
            for i,op in enumerate(movers): op.Set(Gf.Vec3d(*map(float,guides[j,i]+offset[i])))
            u=float(np.clip((j/max(len(ts)-1,1)-.3)/.5,0,1)); u=u*u*(3-2*u)
            look((1-u)*np.array([0,-10,11])+u*np.array([-.2,-4.7,3.6]),(1-u)*np.array([0,0,.3])+u*np.array([0,-1.8,.45]))
            if reset: reset()
            for _ in range(12 if frame==0 or a.preview else 2): tick()
            rgb=np.asarray(camera.get_rgba(device='cpu'))[:,:,:3]
            if rgb.shape!=(720,1280,3) or rgb.dtype!=np.uint8 or rgb.std()<2: raise RuntimeError('Invalid native image')
            path=a.out/f'{frame:06d}.png'; Image.fromarray(rgb).save(path,compress_level=1)
            report['frames'].append(dict(file=path.name,trajectory_index=int(j),actual_t_s=float(ts[j]),sha256=sha(path)))
            if frame%30==0: print('CAPTURE',frame,len(indices),flush=True); write(target,report)
        report.update(complete=True,displayed_worlds=n,simulated_worlds=source['settings']['worlds'],visual_radius_m=RADIUS,
                      layout='Independent environment-local states translated to grid; no deformation or time scaling')
        stage.GetRootLayer().Export(str(a.out/'scene.usda')); write(target,report)
    except BaseException:
        report['errors'].append(traceback.format_exc()); write(target,report)
    finally:
        if camera is not None: camera.pause(); camera.destroy()
        if world is not None: world.stop()
        if app is not None:
            report['app_close_invoked']=True; write(target,report); app.close(wait_for_replicator=True,skip_cleanup=False)
    return int(bool(report['errors']))


def compose(a,r):
    from PIL import Image,ImageDraw,ImageFont
    source=json.loads((a.data/'report.json').read_text()); tm=source['timing']; n=source['settings']['worlds']
    edit=a.out/'edit'; edit.mkdir()
    font=lambda n:ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',n)
    for i,item in enumerate(r['frames']):
        im=Image.open(a.out/item['file']).convert('RGB'); d=ImageDraw.Draw(im)
        d.rectangle((0,0,1280,112),fill='#091322'); d.rectangle((0,627,1280,720),fill='#091322')
        d.text((34,20),'DEFORMX / CONTACT ARENA',font=font(30),fill='#43dec4')
        d.text((36,65),f'{n:,} independent worlds / {r["displayed_worlds"]} displayed / moving guides + friction',font=font(20),fill='#edf5ff')
        d.text((36,641),f'{tm["median_ms"]:.2f} ms median  |  {tm["aggregate_world_steps_per_s"]:,.0f} world-steps/s  |  RTF {tm["per_world_rtf"]:.2f}x',font=font(22),fill='#edf5ff')
        d.text((36,677),f'All worlds contacted / timing excludes rendering / recorded t={item["actual_t_s"]:.2f} s / 1x replay',font=font(17),fill='#a5bdd3')
        im.save(edit/f'{i:06d}.png',compress_level=1)
    Image.open(edit/f'{len(r["frames"])//2:06d}.png').save(a.out/'preview.jpg',quality=95)
    if not a.preview:
        video=a.out/'contact-arena.mp4'
        subprocess.run(['ffmpeg','-v','error','-framerate','30','-i',str(edit/'%06d.png'),'-c:v','libx264','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(video)],check=True)
        subprocess.run(['ffmpeg','-v','error','-xerror','-err_detect','explode','-i',str(video),'-f','null','-'],check=True)
        write(a.out/'video_report.json',dict(complete=True,full_decode=True,frames=len(r['frames']),fps=30,
            duration_s=len(r['frames'])/30,amplitude_scale=1,sha256=sha(video),native_report_sha256=sha(a.out/'native_report.json')))


def validate_capture(data,out,preview=False):
    """Bind every image to its recorded index/time; usable without Isaac."""
    import numpy as np
    r=json.loads((out/'native_report.json').read_text())
    source=json.loads((data/'report.json').read_text())
    with np.load(data/'trajectory.npz') as trajectory: ts=trajectory['t']
    expected=sorted(set([0,len(ts)//2,len(ts)-1])) if preview else list(range(len(ts)))
    if not r['complete'] or r['errors'] or not r['app_close_invoked'] or r['capture_clock_delta_max_s']!=0 or r['amplitude_scale']!=1:
        raise ValueError('Invalid native capture state')
    if not source['complete'] or not source['contact_qualified'] or source['trajectory_sha256']!=sha(data/'trajectory.npz'):
        raise ValueError('Invalid simulation evidence')
    if r['trajectory_sha256']!=sha(data/'trajectory.npz') or r['run_report_sha256']!=sha(data/'report.json'):
        raise ValueError('Changed capture source')
    if [f['trajectory_index'] for f in r['frames']]!=expected: raise ValueError('Capture frame map mismatch')
    for f in r['frames']:
        if abs(f['actual_t_s']-float(ts[f['trajectory_index']]))>1e-10: raise ValueError('Capture timestamp mismatch')
        if sha(out/f['file'])!=f['sha256']: raise ValueError('Capture frame hash mismatch')
    return r


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('data','assets','out'): p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--preview',action='store_true'); p.add_argument('--native',action='store_true'); a=p.parse_args(argv)
    if a.native:
        if not a.out.is_dir() or any(f.name!='native.log' for f in a.out.iterdir()):
            p.error('native output already exists or was not reserved as a fresh directory')
        return native(a)
    if a.out.exists(): p.error('output already exists')
    a.out.mkdir(parents=True)
    cmd=[sys.executable,'-m','demo.contact_arena.render','--native','--data',str(a.data),'--assets',str(a.assets),'--out',str(a.out)]
    if a.preview: cmd.append('--preview')
    with (a.out/'native.log').open('w') as log: result=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=3600)
    if result.returncode:
        raise RuntimeError('Native capture failed; see native_report.json')
    r=validate_capture(a.data,a.out,a.preview)
    compose(a,r); return 0


if __name__=='__main__': raise SystemExit(main())
