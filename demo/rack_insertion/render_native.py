"""Replay measured robot/connector/cable state in native Isaac; no physics step."""
import argparse
import json
from pathlib import Path
import socket
import traceback


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--width',type=int,default=1280);ap.add_argument('--height',type=int,default=720)
    ap.add_argument('--preview',action='store_true');args=ap.parse_args()
    from .assets import sha
    from .audit import verify_recording
    run=verify_recording(args.data)
    report=dict(complete=False,errors=[],app_close_invoked=False,host=socket.gethostname(),
                trajectory_sha256=run['trajectory_sha256'],source_sha256={p.name:sha(p) for p in Path(__file__).parent.glob('render*.py')},renderer='RaytracedLighting',capture_clock_delta_max_s=0.,replay_max_position_error_m=0.)
    report['usd_composition_integrity']='verified local dependency snapshot'
    rp=args.out/'native_report.json'
    write=lambda:rp.write_text(json.dumps(report,indent=2)+'\n')
    write();app=None;camera=None;world=None
    try:
        from isaacsim import SimulationApp
        app=SimulationApp(dict(headless=True,hide_ui=True,active_gpu=0,physics_gpu=0,multi_gpu=False,max_gpu_count=1,
            limit_cpu_threads=8,width=args.width,height=args.height,renderer='RaytracedLighting',anti_aliasing=2,
            sync_loads=True,fast_shutdown=True,enable_crashreporter=False))
        import numpy as np
        from PIL import Image
        from pxr import Usd,UsdGeom,UsdPhysics,UsdShade,UsdLux,Gf,Sdf,Vt
        from isaacsim.core.api import World
        from isaacsim.core.utils.stage import get_current_stage
        from isaacsim.sensors.camera import Camera
        import carb.settings
        import omni.replicator.core as rep
        import omni.timeline
        import omni.usd
        world=World(physics_dt=1/120,rendering_dt=1/30,stage_units_in_meters=1.,backend='numpy',device='cpu')
        stage=get_current_stage();stage.GetRootLayer().subLayerPaths.append(str((args.data/'scene.usda').resolve()))
        joint_paths=[]
        for prim in Usd.PrimRange.Stage(stage):
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):UsdPhysics.RigidBodyAPI(prim).GetRigidBodyEnabledAttr().Set(False)
            if prim.HasAPI(UsdPhysics.CollisionAPI):UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Set(False)
            if prim.IsA(UsdPhysics.Joint):joint_paths.append(prim.GetPath())
        for path in joint_paths:stage.GetPrimAtPath(path).SetActive(False)
        report['physics_replay_disabled']=True
        # Render-only fallback for external MDL resources absent from the source rack package.
        # Keep imported mesh/physics geometry intact; only replace shading.
        rack=stage.GetPrimAtPath('/World/envs/env_0/Rack')
        original_materials=[p.GetPath() for p in Usd.PrimRange(rack) if p.IsA(UsdShade.Material)]
        for path in original_materials:stage.GetPrimAtPath(path).SetActive(False)
        rack_material=UsdShade.Material.Define(stage,'/World/CaptureMaterials/RackGraphite')
        rack_shader=UsdShade.Shader.Define(stage,'/World/CaptureMaterials/RackGraphite/Shader')
        rack_shader.CreateIdAttr('UsdPreviewSurface')
        rack_shader.CreateInput('diffuseColor',Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(.12,.15,.19))
        rack_shader.CreateInput('metallic',Sdf.ValueTypeNames.Float).Set(.65)
        rack_shader.CreateInput('roughness',Sdf.ValueTypeNames.Float).Set(.35)
        rack_material.CreateSurfaceOutput().ConnectToSource(rack_shader.ConnectableAPI(),'surface')
        UsdShade.MaterialBindingAPI.Apply(rack).Bind(rack_material,bindingStrength=UsdShade.Tokens.strongerThanDescendants)
        report['rack_render_material']='uniform graphite UsdPreviewSurface; source MDL texture dependencies unavailable; mesh/physics unchanged'
        tr=np.load(args.data/'trajectory.npz',allow_pickle=False)
        report['playback_rate']=1.0;report['simulation_loop_wall_s']=run.get('simulation_loop_wall_s')
        report['simulated_duration_s']=float(tr['t'][-1])
        report['realtime_factor']=float(tr['t'][-1])/run['simulation_loop_wall_s'] if run.get('simulation_loop_wall_s') else None
        panel_profile=run.get('rack_panel')
        report["image_overlay"] = False
        if panel_profile:
            jack_material=UsdShade.Material.Define(stage,'/World/CaptureMaterials/SocketCharcoal')
            jack_shader=UsdShade.Shader.Define(stage,'/World/CaptureMaterials/SocketCharcoal/Shader')
            jack_shader.CreateIdAttr('UsdPreviewSurface')
            jack_shader.CreateInput('diffuseColor',Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(.022,.028,.037))
            jack_shader.CreateInput('roughness',Sdf.ValueTypeNames.Float).Set(.36)
            jack_material.CreateSurfaceOutput().ConnectToSource(jack_shader.ConnectableAPI(),'surface')
            UsdShade.MaterialBindingAPI.Apply(stage.GetPrimAtPath('/World/envs/env_0/Socket')).Bind(jack_material,bindingStrength=UsdShade.Tokens.strongerThanDescendants)
            report['active_socket_render_material']='charcoal matching neighbouring jack bodies; unchanged original cavity mesh'
        report['camera_variant']='rack-panel' if panel_profile else 'legacy-contact'
        prefix='/World/envs/env_0'
        def replay_ops(path):
            xf=UsdGeom.Xformable(stage.GetPrimAtPath(path));xf.ClearXformOpOrder();xf.SetResetXformStack(True)
            return xf.AddTranslateOp(opSuffix='recorded'),xf.AddOrientOp(UsdGeom.XformOp.PrecisionDouble,opSuffix='recorded')
        body_ops=[replay_ops(prefix+'/Robot/'+n) for n in tr['body_names'].tolist()]
        plug_ops=replay_ops(prefix+'/Plug');latch_ops=replay_ops(prefix+'/Latch')
        curve=UsdGeom.BasisCurves.Define(stage,'/World/RecordedCable');curve.CreateTypeAttr('linear');curve.CreateWrapAttr('nonperiodic')
        curve.CreateWidthsAttr([.004]);curve.SetWidthsInterpolation('constant')
        material=UsdShade.Material.Define(stage,'/World/CableMaterial');shader=UsdShade.Shader.Define(stage,'/World/CableMaterial/Shader')
        shader.CreateIdAttr('UsdPreviewSurface');shader.CreateInput('diffuseColor',Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(.025,.6,.9))
        shader.CreateInput('roughness',Sdf.ValueTypeNames.Float).Set(.3);material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(),'surface')
        UsdShade.MaterialBindingAPI.Apply(curve.GetPrim()).Bind(material)
        for path in (prefix+'/Plug', prefix+'/Latch'):
            UsdShade.MaterialBindingAPI.Apply(stage.GetPrimAtPath(path)).Bind(
                material,bindingStrength=UsdShade.Tokens.strongerThanDescendants)
        report['cable_and_plug_render_material']=dict(
            base_color_rgb=[.025,.6,.9],roughness=.3,
            scope='Cable, plug, latch and termination visual; render-only, unchanged physics')
        termination_ops=None
        if (run.get('cable') or {}).get('termination_model')=='compliant':
            visual=Path(run['cable']['termination_visual_usd'])
            if not visual.exists():raise FileNotFoundError('Blender termination visual required: '+str(visual))
            term=stage.DefinePrim('/World/TerminationVisual','Xform');term.GetReferences().AddReference(str(visual))
            UsdShade.MaterialBindingAPI.Apply(term).Bind(
                material,bindingStrength=UsdShade.Tokens.strongerThanDescendants)
            xf=UsdGeom.Xformable(term);termination_ops=(xf.AddTranslateOp(),xf.AddOrientOp(UsdGeom.XformOp.PrecisionDouble),xf.AddScaleOp())
            report['termination_visual_sha256']=sha(visual)
        light=UsdLux.DistantLight.Define(stage,'/World/CaptureKey');light.CreateIntensityAttr(900);light.CreateAngleAttr(8)
        UsdGeom.Xformable(light).AddRotateXYZOp().Set(Gf.Vec3f(-45,-30,25))
        dome=UsdLux.DomeLight.Define(stage,'/World/CaptureDome');dome.CreateIntensityAttr(900);dome.CreateColorAttr(Gf.Vec3f(.94,.97,1.))
        def soft_light(path,eye,target,width,height,intensity):
            ob=UsdLux.RectLight.Define(stage,path);ob.CreateWidthAttr(width);ob.CreateHeightAttr(height);ob.CreateIntensityAttr(intensity)
            ob.CreateColorAttr(Gf.Vec3f(1.,.98,.95))
            matrix=Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye),Gf.Vec3d(*target),Gf.Vec3d(0,0,1)).GetInverse()
            UsdGeom.Xformable(ob).AddTransformOp().Set(matrix)
        soft_light('/World/CaptureFront',[1.1,-1.5,2.5],[.3,0,1.25],1.8,1.2,2500)
        soft_light('/World/CaptureFill',[-.9,-.6,2.0],[.3,.1,1.25],1.6,1.2,1800)
        report['lighting']='neutral dome900; front area2500/fill1800; rim distant900 (render-only)'
        camera=Camera(prim_path='/World/CaptureCamera',frequency=-1,resolution=(args.width,args.height),annotator_device='cpu')
        camera.prim.GetAttribute('focalLength').Set(26.)
        camera.prim.GetAttribute('clippingRange').Set(Gf.Vec2f(.005,100.))
        def look(eye,target):
            q=Gf.Matrix4d().SetLookAt(Gf.Vec3d(*map(float,eye)),Gf.Vec3d(*map(float,target)),Gf.Vec3d(0,0,1)).GetInverse().ExtractRotationQuat()
            camera.set_world_pose(position=np.asarray(eye),orientation=np.r_[q.GetReal(),q.GetImaginary()],camera_axes='usd')
        look([2.1,-2.8,2.35],[.3,.20,1.35])
        world.reset();camera.initialize(attach_rgb_annotator=True);rep.orchestrator.set_capture_on_play(False)
        settings=carb.settings.get_settings()
        timeline=omni.timeline.get_timeline_interface()
        reset_history=getattr(omni.usd.get_context(),'reset_renderer_accumulation',None)
        def tick():
            before=(float(world.current_time),float(timeline.get_current_time()),int(world.current_time_step_index))
            rep.orchestrator.step(rt_subframes=4,pause_timeline=False,delta_time=0.)
            after=(float(world.current_time),float(timeline.get_current_time()),int(world.current_time_step_index))
            error=max(abs(before[0]-after[0]),abs(before[1]-after[1]));report['capture_clock_delta_max_s']=max(report['capture_clock_delta_max_s'],error)
            if error>1e-9 or before[2]!=after[2]:raise RuntimeError('Capture advanced physics clock')
        def set_pose(ops,p,q):
            q=np.asarray(q,dtype=np.float64);q=q/np.linalg.norm(q)
            ops[0].Set(Gf.Vec3d(*map(float,p)));ops[1].Set(Gf.Quatd(float(q[0]),Gf.Vec3d(*map(float,q[1:]))))
        dest=args.out/'frames';dest.mkdir(exist_ok=True)
        if args.preview:
            selected=[0]
            for label in (('hold','route','align','insert','place') if panel_profile else ('hold','route','insert','place')):
                found=np.flatnonzero(tr['phase']==label)
                if len(found):selected.append(int(found[len(found)//2]))
            selected.append(len(tr['t'])-1);indices=np.asarray(sorted(set(selected)))
        else:indices=np.arange(len(tr['t']))
        records=[]
        previous_index=None
        previous_shot=None
        history_reset_frames=[]
        warmup_ticks=36
        for frame,j in enumerate(indices):
            for bi,ops in enumerate(body_ops):set_pose(ops,tr['body_pos'][j,bi],tr['body_quat'][j,bi])
            set_pose(plug_ops,tr['plug_pos'][j],tr['plug_quat'][j]);set_pose(latch_ops,tr['latch_pos'][j],tr['latch_quat'][j])
            if tr['cable'].shape[1]>0:
                curve.CreateCurveVertexCountsAttr([tr['cable'].shape[1]]);curve.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(tr['cable'][j].astype(np.float32)))
            gap=0.
            if termination_ops is not None:
                q=tr['plug_quat'][j].astype(np.float64);q=q/np.linalg.norm(q);v=np.asarray(run['connector']['cable_exit_offset'])
                exit_point=tr['plug_pos'][j]+v+2*np.cross(q[1:],np.cross(q[1:],v)+q[0]*v)
                end=tr['cable'][j,0];delta=end-exit_point;gap=float(np.linalg.norm(delta))
                termination_ops[0].Set(Gf.Vec3d(*map(float,(exit_point+end)/2)))
                rotation=Gf.Rotation(Gf.Vec3d(0,0,1),Gf.Vec3d(*map(float,delta/max(gap,1e-12)))).GetQuat() if gap>1e-12 else Gf.Quatd(1)
                termination_ops[1].Set(rotation);termination_ops[2].Set(Gf.Vec3f(1,1,gap))
            phase=str(tr['phase'][j])
            shot='wide'
            if phase in ('close','lift','hold','align','insert','place','release') and (frame>0 or not args.preview):
                target=tr['plug_pos'][j]
                offset=np.array([.20,-.22,.07]) if run['case']=='insert' and phase in ('align','insert','release') else np.array([.16,-.25,.14])
                shot='plug-closeup' if run['case']=='insert' and phase in ('align','insert','release') else 'plug-hold'
                look(target+offset,target)
            else:look([2.1,-2.8,2.35],[.3,.20,1.35])
            if panel_profile:
                center=np.asarray(panel_profile['panel_center_m'])
                if phase=='align':
                    shot='panel-align'
                    look(center+[.34,-.70,.20],center)
                elif phase=='insert':
                    shot='panel-insert'
                    look(tr['plug_pos'][j]+[.15,-.22,.065],tr['plug_pos'][j])
                elif phase=='release':
                    release=np.flatnonzero(tr['phase']=='release')
                    if tr['t'][j]-tr['t'][release[0]]>=1.5:
                        shot='panel-withdrawal'
                        look(center+[.30,-.74,.19],center)
            pose_cache=UsdGeom.XformCache()
            checks=[(prefix+'/Robot/'+name,tr['body_pos'][j,bi],tr['body_quat'][j,bi]) for bi,name in enumerate(tr['body_names'].tolist())]+[(prefix+'/Plug',tr['plug_pos'][j],tr['plug_quat'][j]),(prefix+'/Latch',tr['latch_pos'][j],tr['latch_quat'][j])]
            for path,position,quaternion in checks:
                quaternion=np.asarray(quaternion,dtype=np.float64);quaternion=quaternion/np.linalg.norm(quaternion)
                expected=Gf.Matrix4d(1).SetRotate(Gf.Quatd(float(quaternion[0]),Gf.Vec3d(*map(float,quaternion[1:]))))
                expected.SetTranslateOnly(Gf.Vec3d(*map(float,position)))
                error=float(np.max(np.abs(np.asarray(pose_cache.GetLocalToWorldTransform(stage.GetPrimAtPath(path)))-np.asarray(expected))))
                report['replay_max_matrix_error']=max(report.get('replay_max_matrix_error',0.),error)
                if error>1e-5:raise RuntimeError('Recorded pose replay mismatch: '+path)
            actual=np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(stage.GetPrimAtPath(prefix+'/Plug')))[3,:3]
            report['replay_max_position_error_m']=max(report['replay_max_position_error_m'],float(np.max(np.abs(actual-tr['plug_pos'][j]))))
            if previous_index is None:
                # Initialize the archived RTX scene before changing only the AA setting.
                if reset_history is not None:
                    reset_history()
                for _ in range(12):
                    tick()
                tick()
                settings.set_int('/rtx/post/aa/op',4)
            if previous_index is None or j!=previous_index+1 or shot!=previous_shot:
                tick()  # Submit the newly authored shot before warming its history.
                if reset_history is not None:
                    reset_history()
                for _ in range(warmup_ticks):
                    tick()
                history_reset_frames.append(int(j))
            tick()
            previous_index,previous_shot=j,shot
            aa=int(settings.get('/rtx/post/aa/op'))
            exposure=bool(settings.get('/rtx/post/histogram/enabled'))
            if aa!=4 or exposure:
                raise RuntimeError(f'Rack replay requires DLAA and fixed exposure, got {aa=} {exposure=}')
            rgba=np.asarray(camera.get_rgba(device='cpu'))
            if rgba.shape!=(args.height,args.width,4) or rgba.dtype!=np.uint8 or rgba[:,:,:3].std()<2:raise RuntimeError('Invalid native image')
            image=Image.fromarray(rgba[:,:,:3])
            file=dest/f'{frame:06d}.png';image.save(file,compress_level=1)
            records.append(dict(file=str(file.relative_to(args.out)),trajectory_index=int(j),actual_t_s=float(tr['t'][j]),termination_gap_m=gap,sha256=sha(file)))
            if frame%30==0:print('CAPTURE',frame,'/',len(indices),flush=True)
        (args.out/'frames.json').write_text(json.dumps(records,indent=2)+'\n')
        report['quaternion_conversion']='raw float32 recordings preserved; normalize in float64 for USD rotation conversion'
        report.update(complete=True,frames=len(records),antialiasing_mode=aa,histogram_enabled=exposure,
                      renderer_history_reset_source_frames=history_reset_frames,
                      renderer_history_reset_available=reset_history is not None,
                      discarded_warmup_ticks_per_cut=warmup_ticks);write()
    except BaseException:
        report['errors'].append(traceback.format_exc());print(traceback.format_exc(),flush=True);write()
        raise
    finally:
        if camera is not None:camera.pause();camera.destroy()
        if world is not None:world.stop()
        if app is not None:
            report['app_close_invoked']=True;write();app.close(skip_cleanup=False)


if __name__=='__main__':main()
