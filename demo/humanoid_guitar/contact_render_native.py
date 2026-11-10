"""Render the measured 24-stroke trajectory using the calibrated native camera."""

import argparse
import json
import socket
import traceback
from pathlib import Path

from . import contact_fullbody_framing as framing
from . import contact_loop_pullback as pullback
from .contact_integrity import sha, validate
from .portable_audit import validate_capture


def capture(case, out, profile, run, camera_profile, report):
    import numpy as np
    import omni.replicator.core as rep
    import omni.timeline
    import omni.usd
    from isaacsim.core.api import World
    from isaacsim.core.utils.stage import get_current_stage
    from isaacsim.sensors.camera import Camera
    from PIL import Image
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdShade, Vt

    motion_profile = profile['contact_motion']
    hz = motion_profile['record_hz']
    width, height = report['frame_resolution_px']
    with np.load(case / 'motion.npz', allow_pickle=False) as archive:
        motion = {key: archive[key] for key in ('t', 'phase', 'link_names', 'link_transforms', 'pick_transforms')}
    with np.load(case / 'trajectory.npz', allow_pickle=False) as archive:
        times, positions, loads, poses = (archive[key] for key in ('t', 'positions', 'mesh_wrenches', 'collider_transforms'))
    if not np.array_equal(times, motion['t']):
        raise ValueError('Saved robot and string sample times disagree')
    placement = np.asarray(profile['guitar']['placement'])
    schedule = pullback.build_pullback_edit(motion_profile, pullback.FULLBODY_EDIT)
    spec = pullback.pullback_spec(motion_profile, pullback.FULLBODY_EDIT)
    shots = pullback.build_pullback_camera_path(schedule, spec, camera_profile, placement)
    report.update(planned_frames=len(schedule), loop_replay=spec, loop_edit=pullback.FULLBODY_EDIT,
                  loop_replay_source_sha256=sha(pullback.__file__),
                  fullbody_framing_source_sha256=sha(framing.__file__))

    world = World(physics_dt=1/hz, rendering_dt=1/30, stage_units_in_meters=1., backend='numpy', device='cpu')
    stage = get_current_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    camera = None
    try:
        for label in ('Robot', 'Guitar', 'Pick', 'Stool', 'Floor'):
            prim = stage.DefinePrim('/World/' + label, 'Xform')
            prim.GetReferences().AddReference(profile[label.lower()]['path'], '/root')
        for prim in list(stage.Traverse()):
            if prim.HasAPI(UsdLux.LightAPI):
                prim.SetActive(False)

        def xform(prim):
            xf = UsdGeom.Xformable(prim)
            xf.ClearXformOpOrder()
            xf.SetResetXformStack(True)
            return xf.AddTransformOp(UsdGeom.XformOp.PrecisionDouble)

        def set_matrix(op, matrix):
            op.Set(Gf.Matrix4d(*np.asarray(matrix, dtype=float).T.flatten().tolist()))

        guitar = stage.GetPrimAtPath('/World/Guitar')
        pick = stage.GetPrimAtPath('/World/Pick')
        set_matrix(xform(guitar), placement)
        pick_op = xform(pick)
        links = {name: stage.GetPrimAtPath('/World/Robot/Robot/' + name) for name in motion['link_names'].tolist()}
        if any(not prim.IsValid() for prim in links.values()):
            raise ValueError('Missing imported link in robot USD')
        link_ops = [xform(prim) for prim in links.values()]
        subjects = {name: stage.GetPrimAtPath(path) for name, path in framing.SUBJECT_PATHS.items()}
        if any(not prim.IsValid() for prim in subjects.values()):
            raise ValueError('Missing full-body framing subject')

        curves = UsdGeom.BasisCurves.Define(stage, '/World/MeasuredStrings')
        curves.CreateTypeAttr('linear')
        curves.CreateWrapAttr('nonperiodic')
        curves.CreateCurveVertexCountsAttr([positions.shape[2]] * positions.shape[1])
        curves.CreateWidthsAttr([2 * run['material']['radius_m']])
        curves.SetWidthsInterpolation('constant')
        points = curves.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(positions[0].reshape(-1, 3).astype(np.float32)))
        mat = UsdShade.Material.Define(stage, '/World/StringMaterial')
        shader = UsdShade.Shader.Define(stage, '/World/StringMaterial/Shader')
        shader.CreateIdAttr('UsdPreviewSurface')
        shader.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(.74, .80, .86))
        shader.CreateInput('metallic', Sdf.ValueTypeNames.Float).Set(.8)
        shader.CreateInput('roughness', Sdf.ValueTypeNames.Float).Set(.23)
        mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), 'surface')
        UsdShade.MaterialBindingAPI.Apply(curves.GetPrim()).Bind(mat)

        def light(name, eye, target, color, intensity, size):
            value = UsdLux.RectLight.Define(stage, '/World/' + name)
            value.CreateWidthAttr(size)
            value.CreateHeightAttr(size)
            value.CreateIntensityAttr(intensity)
            value.CreateColorAttr(Gf.Vec3f(*color))
            look = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*target), Gf.Vec3d(0, 0, 1)).GetInverse()
            UsdGeom.Xformable(value).AddTransformOp().Set(look)
            return value

        dome = UsdLux.DomeLight.Define(stage, '/World/Dome')
        dome.CreateIntensityAttr(200)
        dome.CreateColorAttr(Gf.Vec3f(.35, .43, .55))
        light('WarmKey', [2., -1.6, 2.4], [.2, 0, .9], [1., .82, .63], 2200, 1.7)
        light('CoolRim', [-.7, 1.2, 1.9], [.1, 0, .9], [.38, .65, 1.], 2800, 1.3)
        light('GuitarFill', [1.4, .7, 1.2], [.35, .1, .8], [.75, .85, 1.], 900, 1.)
        fill = camera_profile['fill']
        fill_eye = (placement @ np.r_[fill['position_guitar_local_m'], 1])[:3]
        fill_target = (placement @ np.r_[camera_profile['target_guitar_local_m'], 1])[:3]
        pick_fill = light('RightMacroNeutralFill', fill_eye.tolist(), fill_target.tolist(), fill['color'], 0., fill['size_m'])

        camera = Camera(prim_path='/World/Camera', frequency=-1, resolution=(width, height), annotator_device='cpu')
        focal = camera.prim.GetAttribute('focalLength')
        focal.Set(38.)
        camera.prim.GetAttribute('clippingRange').Set(Gf.Vec2f(.001, 100.))

        def look(eye, target):
            matrix = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*map(float, eye)), Gf.Vec3d(*map(float, target)), Gf.Vec3d(0, 0, 1))
            quat = matrix.GetInverse().ExtractRotationQuat()
            camera.set_world_pose(position=np.asarray(eye), orientation=np.r_[quat.GetReal(), quat.GetImaginary()], camera_axes='usd')

        look([3.05, -2.65, 1.7], [.18, .1, .64])
        world.reset()
        camera.initialize(attach_rgb_annotator=True)
        rep.orchestrator.set_capture_on_play(False)
        timeline = omni.timeline.get_timeline_interface()
        reset_accumulation = getattr(omni.usd.get_context(), 'reset_renderer_accumulation', None)

        def tick():
            before = world.current_time, timeline.get_current_time(), world.current_time_step_index
            rep.orchestrator.step(rt_subframes=4, pause_timeline=False, delta_time=0.)
            after = world.current_time, timeline.get_current_time(), world.current_time_step_index
            error = float(max(abs(before[0] - after[0]), abs(before[1] - after[1])))
            report['capture_clock_delta_max_s'] = max(report['capture_clock_delta_max_s'], error)
            if error > 1e-9 or before[2] != after[2]:
                raise RuntimeError('Capture advanced the physics clock')

        manifest = run['mesh_manifest']
        indices = {mesh['id']: mesh['index'] for mesh in manifest}
        hand_meshes = [mesh for mesh in manifest if mesh['role'] == 'finger']
        face = placement[:3, 2]
        selected = motion_profile['selected_string_index']
        finger_load = loads[:, selected, indices['L_index_intermediate'], :3] @ face
        fret_load = -(loads[:, selected, indices['fret_05'], :3] @ face)
        pick_load = np.linalg.norm(loads[:, :, indices['pick'], :3], axis=-1)
        dest = out / 'frames'
        dest.mkdir()
        records = []
        for frame, (row, shot) in enumerate(zip(schedule, shots)):
            j = row['trajectory_index']
            for op, matrix in zip(link_ops, motion['link_transforms'][j]):
                set_matrix(op, matrix)
            set_matrix(pick_op, motion['pick_transforms'][j])
            points.Set(Vt.Vec3fArray.FromNumpy(positions[j].reshape(-1, 3).astype(np.float32)))
            look(shot['eye_world_m'], shot['target_world_m'])
            focal.Set(shot['focal_length_mm'])
            pick_fill.GetIntensityAttr().Set(shot['macro_fill_intensity'])
            if reset_accumulation:
                reset_accumulation()
            if frame == 0:
                for _ in range(12):
                    tick()
            tick()
            rgba = np.asarray(camera.get_rgba(device='cpu'))
            if rgba.shape != (height, width, 4) or rgba[:, :, :3].std() < 2:
                raise RuntimeError('Invalid native camera frame')
            file = dest / f'{frame:06d}.png'
            Image.fromarray(rgba[:, :, :3]).save(file, compress_level=1)
            cache = UsdGeom.XformCache()
            actual = lambda prim: np.asarray(cache.GetLocalToWorldTransform(prim)).T
            errors = [np.max(abs(actual(pick) - poses[j, indices['pick']])), np.max(abs(actual(guitar) - placement))]
            for mesh in hand_meshes:
                actual_shell = actual(links[mesh['link']]) @ np.asarray(mesh['link_from_mesh'])
                errors.append(np.max(abs(actual_shell - poses[j, mesh['index']])))
            error = float(max(errors))
            report['replay_max_matrix_error'] = max(report['replay_max_matrix_error'], error)
            if error > 1e-5:
                raise RuntimeError('Visible instrument, pick or finger transform differs from its collider')
            record = dict(row, video_frame=frame, video_time_s=frame / 30, simulation_time_s=float(times[j]),
                          file=str(file.relative_to(out)), png_sha256=sha(file), phase=str(motion['phase'][j]),
                          distal_finger_normal_N=float(finger_load[j]), fret5_opposing_normal_N=float(fret_load[j]),
                          pick_load_N_per_string=pick_load[j].tolist(), camera=dict(
                              world_from_camera=actual(camera.prim).tolist(),
                              focal_length_mm=float(focal.Get()),
                              macro_fill_intensity=float(pick_fill.GetIntensityAttr().Get())))
            if row['chapter'] == 'overview':
                bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ['default', 'render'])
                corners = {}
                for name, prim in subjects.items():
                    bound = bbox_cache.ComputeWorldBound(prim)
                    box, transform = bound.GetRange(), bound.GetMatrix()
                    if box.IsEmpty():
                        raise ValueError(f'Empty full-body framing bound for {name}')
                    corners[name] = [list(transform.Transform(box.GetCorner(i))) for i in range(8)]
                record['framing'] = dict(intrinsics_px=camera.get_intrinsics_matrix(device='cpu').tolist(),
                                         clip_near_far_m=list(camera.get_clipping_range()),
                                         world_bound_corners_m=corners)
            records.append(record)
            if frame % 30 == 0:
                print(f'CAPTURE {frame} / {len(schedule)}', flush=True)
        report['pullback_camera_proof'] = pullback.verify_pullback_edit(
            motion_profile, records, spec, camera_profile, placement, pullback.FULLBODY_EDIT)
        report['fullbody_framing_proof'] = framing.verify_fullbody_framing(records, report['frame_resolution_px'])
        (out / 'frames.json').write_text(json.dumps(records, indent=2, allow_nan=False) + '\n')
        report.update(complete=True, frames=len(records), amplitude_scale=1., physics_steps_during_capture=0)
    finally:
        try:
            if camera is not None:
                camera.pause()
                camera.destroy()
        finally:
            world.stop()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', type=Path, required=True)
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    run = json.loads((args.case / 'report.json').read_text())
    profile = json.loads((args.case / 'profile.json').read_text())
    validate(profile)
    proof = validate_capture(args.case, args.audit)
    args.out.mkdir(parents=True, exist_ok=False)
    camera_file = args.out / 'camera_profile.json'
    camera_file.write_bytes(Path(__file__).with_name('contact_loop_fullbody_camera.json').read_bytes())
    camera_profile = json.loads(camera_file.read_text())
    report = dict(complete=False, host=socket.gethostname(), errors=[], renderer='RaytracedLighting',
                  capture_clock_delta_max_s=0., replay_max_matrix_error=0., app_close_invoked=False,
                  case_path=str(args.case.resolve()), trajectory_sha256=run['trajectory_sha256'],
                  motion_sha256=run['motion_sha256'], profile_sha256=run['profile_sha256'],
                  source_sha256=sha(__file__), camera_profile_sha256=sha(camera_file),
                  frame_resolution_px=[1280, 720], record_hz=480, **proof)
    report_file = args.out / 'native_report.json'

    def write():
        report_file.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')

    write()
    app = None
    try:
        from isaacsim import SimulationApp

        app = SimulationApp(dict(headless=True, hide_ui=True, active_gpu=0, physics_gpu=0, multi_gpu=False,
                                 max_gpu_count=1, limit_cpu_threads=8, width=1280, height=720,
                                 renderer='RaytracedLighting', anti_aliasing=2, sync_loads=True,
                                 fast_shutdown=True, enable_crashreporter=False))
        capture(args.case, args.out, profile, run, camera_profile, report)
    except BaseException:
        report['complete'] = False
        report['errors'].append(traceback.format_exc())
        raise
    finally:
        if app is not None:
            report['app_close_invoked'] = True
        write()
        if app is not None:
            app.close(skip_cleanup=False)


if __name__ == '__main__':
    main()
