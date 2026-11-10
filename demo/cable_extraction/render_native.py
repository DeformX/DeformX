"""Replay the measured workcell in native Isaac at 30 fps, without physics or HUD."""
import argparse
import hashlib
import json
from pathlib import Path
import socket
import traceback

import numpy as np

from .model import COLORS, FPS
from .run import sha


def verify_recording(data, preview, audit_path):
    run_report_path = data/'report.json'
    run = json.loads(run_report_path.read_text())
    if not run['complete'] or run['errors'] or (not preview and not run['metrics']['passed']):
        raise RuntimeError('Native playback requires a complete, accepted recording')
    if not preview:
        if audit_path is None:
            raise ValueError('Full capture requires --audit from the measured recording')
        audit = json.loads(audit_path.read_text())
        if (not audit['accepted'] or audit['trajectory_sha256'] != run['trajectory_sha256']
                or audit['run_report_sha256'] != sha(run_report_path)
                or audit['model_source_sha256'] != run['source_sha256']['model.py']):
            raise RuntimeError('Audit did not accept this trajectory')
    for file, key in [('scene.usda','scene_sha256'),('trajectory.npz','trajectory_sha256')]:
        if sha(data/file) != run[key]:
            raise RuntimeError(f'Changed recorded input {file}')
    for filename, digest in run['usd_layers'].items():
        if sha(filename) != digest:
            raise RuntimeError(f'Changed original USD dependency {filename}')
    with np.load(data/'trajectory.npz',allow_pickle=False) as archive:
        tr = {k:archive[k] for k in archive.files}
    expected = np.arange(1, len(tr['t'])+1) / FPS
    if len(expected) != run['frames'] or not np.allclose(tr['t'],expected,atol=1e-9,rtol=0):
        raise RuntimeError('Recorded clock is not uniform 30 Hz')
    return run, tr


def capture(args, run, tr, report):
    from PIL import Image
    from pxr import Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade, Gf, Sdf, Vt
    from isaacsim.core.api import World
    from isaacsim.core.utils.stage import get_current_stage
    from isaacsim.sensors.camera import Camera
    import carb.settings
    import omni.replicator.core as rep
    import omni.timeline
    import omni.usd

    world = World(physics_dt=1/120, rendering_dt=1/FPS, stage_units_in_meters=1., backend='numpy', device='cpu')
    stage = get_current_stage()
    stage.GetRootLayer().subLayerPaths.append(str((args.data/'scene.usda').resolve()))
    joints = []
    for prim in Usd.PrimRange.Stage(stage):
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            UsdPhysics.RigidBodyAPI(prim).GetRigidBodyEnabledAttr().Set(False)
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Set(False)
        if prim.IsA(UsdPhysics.Joint):
            joints.append(prim.GetPath())
    for path in joints:
        stage.GetPrimAtPath(path).SetActive(False)
    prefix = '/World/envs/env_0'
    base = stage.GetPrimAtPath(prefix+'/Robot/panda_link0')
    box_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],useExtentsHint=False)
    box = box_cache.ComputeWorldBound(base).ComputeAlignedRange()
    base_bounds = np.asarray([tuple(box.GetMin()),tuple(box.GetMax())])
    table_bounds = np.asarray(run['table_collision_bounds_m'])
    support_margin = float(min((base_bounds[0,:2]-table_bounds[0,:2]).min(),
                               (table_bounds[1,:2]-base_bounds[1,:2]).min()))
    if not np.isfinite(base_bounds).all() or box.IsEmpty() or support_margin < .01:
        raise RuntimeError('Imported Panda base is outside the verified tabletop footprint')
    report['panda_base_world_bounds_m'] = base_bounds.tolist()
    report['panda_base_table_xy_support_margin_m'] = support_margin
    imported_light = stage.GetPrimAtPath(prefix+"/PullTab/env_light")
    if imported_light.IsValid():
        imported_light.SetActive(False)
    pose_paths = [prefix+'/Robot/'+name for name in tr['body_names'].tolist()] + [prefix+'/PullTab']
    pose_ops = []
    for path in pose_paths:
        xf = UsdGeom.Xformable(stage.GetPrimAtPath(path))
        xf.ClearXformOpOrder()
        xf.SetResetXformStack(True)
        pose_ops.append((xf.AddTranslateOp(opSuffix='recorded'),
                         xf.AddOrientOp(UsdGeom.XformOp.PrecisionDouble,opSuffix='recorded')))

    def material(name, color):
        path = '/World/CaptureMaterials/'+name
        mat = UsdShade.Material.Define(stage,path)
        shader = UsdShade.Shader.Define(stage,path+'/Preview')
        shader.CreateIdAttr('UsdPreviewSurface')
        shader.CreateInput('diffuseColor',Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
        shader.CreateInput('roughness',Sdf.ValueTypeNames.Float).Set(.36)
        mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(),'surface')
        return mat

    curves = []
    radius = run['cable']['radius_m']
    if not np.isfinite(radius) or radius <= 0 or len(COLORS) != tr['cable'].shape[1]:
        raise RuntimeError('Recorded cable geometry does not match the replay scene')
    for i, color in enumerate(COLORS):
        curve = UsdGeom.BasisCurves.Define(stage,f'/World/Cables/Cable{i}')
        curve.CreateTypeAttr('linear')
        curve.CreateWrapAttr('nonperiodic')
        curve.CreateWidthsAttr([2*radius])
        curve.SetWidthsInterpolation('constant')
        curve.CreateCurveVertexCountsAttr([tr['cable'].shape[2] + int(i == 0)])
        mat = material(f'Jacket{i}',color)
        UsdShade.MaterialBindingAPI.Apply(curve.GetPrim()).Bind(mat)
        if i == 0:
            UsdShade.MaterialBindingAPI.Apply(stage.GetPrimAtPath(prefix+'/PullTab')).Bind(
                mat,bindingStrength=UsdShade.Tokens.strongerThanDescendants)
        curves.append(curve.CreatePointsAttr())

    dome = UsdLux.DomeLight.Define(stage,'/World/CaptureDome')
    dome.CreateIntensityAttr(740)
    dome.CreateColorAttr(Gf.Vec3f(.92,.96,1.))
    key = UsdLux.DistantLight.Define(stage,'/World/CaptureKey')
    key.CreateIntensityAttr(1200)
    key.CreateAngleAttr(15)
    UsdGeom.Xformable(key).AddRotateXYZOp().Set(Gf.Vec3f(-27,-31,18))
    light = UsdLux.RectLight.Define(stage,'/World/CaptureFill')
    light.CreateWidthAttr(1.9)
    light.CreateHeightAttr(1.25)
    light.CreateIntensityAttr(2100)
    light.CreateColorAttr(Gf.Vec3f(1.,.97,.92))
    fill = Gf.Matrix4d().SetLookAt(Gf.Vec3d(1.2,-1.25,2.7),Gf.Vec3d(.3,0,1.1),Gf.Vec3d(0,0,1)).GetInverse()
    UsdGeom.Xformable(light).AddTransformOp().Set(fill)

    camera = Camera(prim_path='/World/CaptureCamera',frequency=-1,
                    resolution=(args.width,args.height),annotator_device='cpu')
    focal_attr = camera.prim.GetAttribute('focalLength')
    camera.prim.GetAttribute('clippingRange').Set(Gf.Vec2f(.01,100.))
    eye, target = np.asarray(args.eye), np.asarray(args.target)
    end_eye = eye + [.17,-.15,.03] if args.tail_pullback else eye
    end_target = target + [0.,-.15,.015] if args.tail_pullback else target
    end_focal = args.focal-.9 if args.tail_pullback else args.focal
    pullback_start, pullback_end = 28.,34.5

    def set_camera(t):
        u = float(np.clip((t-pullback_start)/(pullback_end-pullback_start),0.,1.))
        s = u*u*(3.-2.*u)
        p = eye + s*(end_eye-eye)
        aim = target + s*(end_target-target)
        rotation = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*p),Gf.Vec3d(*aim),Gf.Vec3d(0,0,1)).GetInverse().ExtractRotationQuat()
        focal_attr.Set(args.focal + s*(end_focal-args.focal))
        camera.set_world_pose(position=p,orientation=np.r_[rotation.GetReal(),rotation.GetImaginary()],camera_axes='usd')

    set_camera(0.)
    world.reset()
    settings = carb.settings.get_settings()
    disabled_effects = ('/rtx/reflections/enabled','/rtx/indirectDiffuse/enabled',
                        '/rtx/post/motionblur/enabled')
    for key in disabled_effects:
        settings.set_bool(key,False)
    settings.set_bool('/rtx/post/histogram/enabled',False)
    camera.initialize(attach_rgb_annotator=True)
    rep.orchestrator.set_capture_on_play(False)
    rep.settings.set_render_rtx_realtime(antialiasing='DLAA')
    timeline = omni.timeline.get_timeline_interface()
    reset_history = getattr(omni.usd.get_context(),'reset_renderer_accumulation',None)
    light_prims = [prim for prim in Usd.PrimRange.Stage(stage) if prim.HasAPI(UsdLux.LightAPI)]
    settings_seen = set()
    light_states = set()
    history_reset_frames = []
    warmup_ticks = 36

    def tick(subframes=4):
        clock = lambda: (float(world.current_time), float(timeline.get_current_time()), int(world.current_time_step_index))
        before = clock()
        rep.orchestrator.step(rt_subframes=subframes,pause_timeline=False,delta_time=0.)
        after = clock()
        error = max(abs(before[i]-after[i]) for i in range(2))
        report['max_capture_clock_delta_s'] = max(error,report['max_capture_clock_delta_s'])
        if error > 1e-9 or before[2] != after[2]:
            raise RuntimeError('Native RGB capture advanced the clock')

    frames = args.out/'frames'
    frames.mkdir()
    if args.preview:
        selected = [0, len(tr['t'])-1]
        selected.extend(int(np.argmin(tr['cable'][:,0,:,axis].min(axis=1))) for axis in (1,2))
        for phase, fractions in (('descend',(1.,)), ('close',(1.,)), ('lift',(.7,)),
                                 ('pull',(.2,.45,.65,.85,1.)), ('hold',(.4,))):
            samples = np.flatnonzero(tr['phase']==phase)
            selected.extend(int(samples[round((len(samples)-1)*f)]) for f in fractions if len(samples))
        indices = sorted(set(selected))
    else:
        indices = list(range(len(tr['t'])))
    frames_record = []
    previous_index = None
    max_curve_error = 0.
    try:
        for frame, i in enumerate(indices):
            set_camera(tr['t'][i])
            positions = np.vstack([tr['robot_pos'][i],tr['tab_pos'][i]])
            rotations = np.vstack([tr['robot_quat'][i],tr['tab_quat'][i]]).astype(np.float64)
            rotations /= np.linalg.norm(rotations,axis=-1,keepdims=True)
            expected = []
            for ops, p, q in zip(pose_ops,positions,rotations):
                usd_q = Gf.Quatd(float(q[0]),Gf.Vec3d(*q[1:]))
                ops[0].Set(Gf.Vec3d(*map(float,p)))
                ops[1].Set(usd_q)
                matrix = Gf.Matrix4d().SetRotate(usd_q)
                matrix.SetTranslateOnly(Gf.Vec3d(*map(float,p)))
                expected.append(np.asarray(matrix))
            expected_curves = []
            for k, attr in enumerate(curves):
                points = tr['cable'][i,k]
                if k == 0:
                    points = np.vstack([tr['tab_exit'][i],points])
                points = points.astype(np.float32)
                expected_curves.append(points)
                attr.Set(Vt.Vec3fArray.FromNumpy(points))
            tick()  # synchronize authored state to Hydra before capture
            if previous_index is None or i != previous_index+1:
                if reset_history is not None:
                    reset_history()
                settings.set_bool('/rtx-transient/post/dlss/forceParamReset',True)
                tick(8)
                # Discard convergence frames at the first state and preview cuts.
                # A reset after this warm-up would make the first output cold again.
                for _ in range(warmup_ticks):
                    tick()
                history_reset_frames.append(i)
            tick(8)
            previous_index = i
            cache = UsdGeom.XformCache()
            for path, matrix in zip(pose_paths,expected):
                replay = np.asarray(cache.GetLocalToWorldTransform(stage.GetPrimAtPath(path)))
                error = float(np.max(np.abs(replay-matrix)))
                report['max_replay_matrix_error'] = max(report['max_replay_matrix_error'],error)
                if error > 1e-5:
                    raise RuntimeError(f'Replay pose mismatch at {path}')
            for attr, points in zip(curves,expected_curves):
                error = float(np.max(np.abs(np.asarray(attr.Get(),dtype=np.float32)-points)))
                max_curve_error = max(max_curve_error,error)
                if error != 0.:
                    raise RuntimeError(f'Replay curve mismatch at {attr.GetPath()}')
            mode = int(settings.get('/rtx/post/aa/op'))
            exposure = bool(settings.get('/rtx/post/histogram/enabled'))
            settings_seen.add((mode,exposure))
            if mode != 4 or exposure:
                raise RuntimeError(f'Replay requires DLAA and fixed exposure, got {mode=} {exposure=}')
            light_data = [(str(prim.GetPath()),
                           np.asarray(cache.GetLocalToWorldTransform(prim)).tolist(),
                           [(a.GetName(),str(a.Get())) for a in prim.GetAttributes()
                            if a.GetName().startswith('inputs:')]) for prim in light_prims]
            light_states.add(hashlib.sha256(json.dumps(light_data).encode()).hexdigest())
            if len(light_states) != 1:
                raise RuntimeError('Replay lighting changed across frames')
            rgba = np.asarray(camera.get_rgba(device='cpu'))
            if rgba.shape != (args.height,args.width,4) or rgba.dtype != np.uint8 or rgba[:,:,:3].std() < 2:
                raise RuntimeError('Invalid native camera image')
            file = frames/f'{frame:06d}.png'
            Image.fromarray(rgba[:,:,:3]).save(file,compress_level=1)
            camera_matrix = np.asarray(cache.GetLocalToWorldTransform(camera.prim),dtype='<f8')
            frames_record.append(dict(file=str(file.relative_to(args.out)),trajectory_frame=i,
                                      t_s=float(tr['t'][i]),sha256=sha(file),
                                      camera_matrix_sha256=hashlib.sha256(camera_matrix.tobytes()).hexdigest(),
                                      camera_focal_length=float(focal_attr.Get())))
            if frame % FPS == 0 or args.preview:
                print(f'CAPTURE {frame}/{len(indices)} t={tr["t"][i]:.3f}',flush=True)
    finally:
        camera.pause()
        camera.destroy()
        world.stop()
    (args.out/'frames.json').write_text(json.dumps(frames_record,indent=2)+'\n')
    report.update(complete=True, frames=len(frames_record), fps=FPS, rate=1., renderer='RaytracedLighting',
                  source_trajectory_sha256=run['trajectory_sha256'], physics_enabled=False,
                  recorded_cable_radius_m=radius,
                  overlay=False, camera_eye_m=eye.tolist(), camera_target_m=target.tolist(),
                  camera_focal_length=args.focal,
                  rtx_disabled_effects={key:bool(settings.get(key)) for key in disabled_effects},
                  antialiasing_mode=4, renderer_runtime_states=sorted(settings_seen),
                  renderer_history_reset='First sample and nonconsecutive preview cuts only, followed by discarded warm-up',
                  renderer_history_reset_source_frames=history_reset_frames,
                  discarded_warmup_ticks_per_cut=warmup_ticks,
                  max_replay_curve_error_m=max_curve_error,
                  authored_light_states_sha256=sorted(light_states),
                  tail_pullback=dict(timing_s=[pullback_start,pullback_end],eye_m=end_eye.tolist(),
                                     target_m=end_target.tolist(),focal_length=end_focal) if args.tail_pullback else None)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--width',type=int,default=1280)
    ap.add_argument('--height',type=int,default=720)
    ap.add_argument('--preview',action='store_true')
    ap.add_argument('--audit',type=Path)
    ap.add_argument('--eye',type=float,nargs=3,default=(1.88,.26,2.55))
    ap.add_argument('--target',type=float,nargs=3,default=(.34,.23,1.23))
    ap.add_argument('--focal',type=float,default=20.2)
    ap.add_argument('--tail-pullback',action='store_true')
    args = ap.parse_args()
    if not np.isfinite([*args.eye,*args.target,args.focal]).all() or args.focal <= 1.:
        ap.error('Camera parameters must be finite with focal length greater than 1')
    run, tr = verify_recording(args.data,args.preview,args.audit)
    args.out.mkdir(parents=True)
    report = dict(host=socket.gethostname(), complete=False, errors=[],
                  capture_mode='preview' if args.preview else 'full',
                  source_frames=len(tr['t']), accepted_audit=not args.preview,
                  max_capture_clock_delta_s=0., max_replay_matrix_error=0.,
                  render_source_sha256=sha(__file__), audit_sha256=sha(args.audit) if args.audit else None)
    app = None
    try:
        from isaacsim import SimulationApp
        app = SimulationApp(dict(headless=True,hide_ui=True,active_gpu=0,physics_gpu=0,multi_gpu=False,
            max_gpu_count=1,limit_cpu_threads=8,width=args.width,height=args.height,
            renderer='RaytracedLighting',anti_aliasing=2,sync_loads=True,fast_shutdown=True,enable_crashreporter=False))
        capture(args,run,tr,report)
    except Exception:
        report['errors'].append(traceback.format_exc())
        print(report['errors'][-1],flush=True)
        raise
    finally:
        report['app_close_invoked'] = app is not None
        (args.out/'native_report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        if app is not None:
            app.close(skip_cleanup=False)


if __name__ == '__main__':
    main()
