"""Joint-actuated, contact-driven task runner. Run with CUDA DeformX and Isaac Sim."""
import argparse
import hashlib
from importlib.util import find_spec
import json
from pathlib import Path
import socket
import subprocess
import sys
import time
import traceback


def main():
    from .profile import load_profile

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True, help="Profile from demo.rack_insertion.prepare")
    parser.add_argument("--out", type=Path, required=True, help="New simulation output directory")
    parser.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.out = args.out.resolve()
    try:
        args.profile_data = load_profile(args.profile)
        if args.out.exists():
            raise ValueError(f"Choose a new output directory: {args.out}")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(f"{exc}. Prepare the assets with python -m demo.rack_insertion.prepare --help")
    missing = [name for name in ("isaaclab", "isaacsim", "ropesim", "torch") if find_spec(name) is None]
    if missing:
        parser.error("Activate the documented Isaac Sim, Isaac Lab and CUDA DeformX runtime. "
                     "Missing packages: " + ", ".join(missing))
    if not args._worker:
        return supervise(args.profile.resolve(), args.out)
    try:
        from isaaclab.app import AppLauncher
    except ImportError as exc:
        parser.error(f"Activate the documented Isaac Sim, Isaac Lab and CUDA DeformX runtime: {exc}")

    # Physical parameters of the recorded nominal trial.
    vars(args).update(case="insert", seconds=28., exchange_hz=1200, rigid_pilot=False,
                      release_mode="place", settle_seconds=1.5, cable_initialization="draped",
                      rod_iterations=96, termination="compliant", termination_stiffness=200.,
                      termination_damping=.03, max_termination_gap=.005, pickup_height=.08,
                      max_joint_target_error=.10, open_gripper=False, offset_x=-.05,
                      offset_y=0., initial_plug_position=None)
    args.out.mkdir(parents=True, exist_ok=False)
    frozen_dir=args.out/'source';frozen_dir.mkdir()
    for source in Path(__file__).parent.glob('*.py'):
        (frozen_dir/source.name).write_bytes(source.read_bytes())
    # Resolve all late relative imports from this immutable per-run copy.
    sys.modules[__package__].__path__=[str(frozen_dir.resolve())]
    report = dict(source_snapshot=str(frozen_dir.resolve()),requested_case=args.case,trajectory_schema_version=2,complete=False, finite=True, host=socket.gethostname(), case=args.case,
                  errors=[], rigid_pilot=args.rigid_pilot, open_gripper_control=args.open_gripper,
                  gripper_attachment_constraints=0, post_reset_object_pose_writes=0,
                  exchange_hz=args.exchange_hz,
                  portable_profile=args.profile_data,
                  source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in frozen_dir.glob('*.py')})
    def save_report():
        (args.out/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    save_report()
    app = None
    try:
        app = AppLauncher({"headless": True, "device": "cuda:0", "fast_shutdown": True}).app
        run_task(args, report)
    except Exception:
        report['errors'].append(traceback.format_exc())
        if (report.get('cable') or {}).get('finite') is False:report['finite']=False
        print(traceback.format_exc(), flush=True)
        raise
    finally:
        if app is not None:
            report['app_close_invoked'] = True
        save_report()
        if app is not None:
            from isaaclab.sim import SimulationContext
            SimulationContext.clear_instance()
            import faulthandler
            print('APP_CLOSE_BEGIN fast_shutdown=True wait_for_replicator=False',flush=True)
            faulthandler.dump_traceback_later(30,repeat=True)
            app.close(wait_for_replicator=False,skip_cleanup=False)
            faulthandler.cancel_dump_traceback_later()
            print('APP_CLOSE_RETURNED',flush=True)


def supervise(profile, output):
    # Kit fast shutdown can exit 0 while unwinding a Python exception.
    # The parent also verifies the worker's report before returning success.
    command = [sys.executable, *(["-S"] if sys.flags.no_site else []),
               "-u", "-m", __spec__.name, "--_worker", "--profile", str(profile), "--out", str(output)]
    child = subprocess.run(command)
    if child.returncode:
        raise SystemExit(child.returncode if child.returncode > 0 else 128 - child.returncode)
    report_path = output / "report.json"
    report = json.loads(report_path.read_text()) if report_path.is_file() else {}
    if (not report.get("complete") or not report.get("finite") or report.get("errors")
            or not report.get("app_close_invoked")):
        raise SystemExit(f"Physical simulation did not complete. See {report_path} and the simulation log.")


def run_task(args, report):
    import numpy as np
    import torch
    # Start Kit before importing the rod bridge, which imports Warp.
    import ropesim
    if not torch.cuda.is_available() or not hasattr(ropesim.RopeBatch, "from_positions"):
        raise RuntimeError("Rack insertion requires a CUDA GPU and the release DeformX backend")
    import warp
    report["warp_runtime"] = {"version": warp.__version__, "file": warp.__file__}
    from pxr import Usd, UsdGeom, UsdPhysics, PhysxSchema, Gf, Sdf
    import isaaclab.sim as su
    from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
    from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
    from isaaclab.sensors import ContactSensorCfg
    from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
    from isaaclab.utils import configclass
    from isaaclab.utils.math import quat_apply, quat_apply_inverse, quat_mul, quat_conjugate, skew_symmetric_matrix
    from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG
    from isaacsim.core.utils.stage import get_current_stage
    from .contracts import score_grasp, score_insertion, score_acquisition, score_controlled_release

    setup_wall_start=time.perf_counter()
    from .panel import load_panel_manifest

    assets = args.profile_data["assets"]
    manifest = json.loads(Path(assets["connector"]["path"]).read_text())
    for part in ("plug", "latch", "socket"):
        manifest[part] = assets[part]["path"]
    panel = load_panel_manifest(assets["panel"]["path"])
    args.panel_profile = panel
    args.termination_visual_usd = assets["termination_visual"]["path"]
    report["rack_panel"] = panel
    material = su.RigidBodyMaterialCfg(static_friction=0.9, dynamic_friction=0.7, restitution=0.0)
    dt = 1.0 / args.exchange_hz
    plug_position = np.asarray(args.initial_plug_position,dtype=float) if args.initial_plug_position is not None else np.array([0.45,-0.15,1.15 if args.case=='passive' else 1.007+args.pickup_height])
    plug_position += [args.offset_x,args.offset_y,0]
    report['initial_plug_position_m']=plug_position.tolist()
    report['placement_offset_m']=[args.offset_x,args.offset_y]
    report['controller_feedback_assumption']='privileged simulator TCP/plug poses and pair contact sensors; known initial placement, no perception/noise model'
    grasp_offset = np.array(manifest['grasp_offset'])
    grip_point = plug_position + grasp_offset
    latch_position = plug_position + np.array(manifest['latch_pivot_in_plug'])
    socket_position = tuple(panel['socket_position_m']) if panel else (0.40, 0.32, 1.25)

    robot_cfg = FRANKA_PANDA_CFG.copy()
    robot_cfg.prim_path = '{ENV_REGEX_NS}/Robot'
    robot_cfg.spawn.usd_path = assets["robot"]["path"]
    robot_cfg.spawn.activate_contact_sensors = True
    robot_cfg.spawn.rigid_props.disable_gravity = False
    robot_cfg.spawn.articulation_props.solver_position_iteration_count = 32
    robot_cfg.spawn.articulation_props.solver_velocity_iteration_count = 4
    robot_cfg.init_state.pos = tuple(panel.get('robot_position_m',[0.,0.,1.003])) if panel else (0.0, 0.0, 1.003)
    for name in ('panda_shoulder','panda_forearm'):
        robot_cfg.actuators[name].stiffness = 1000.0
        robot_cfg.actuators[name].damping = 80.0
    robot_cfg.actuators['panda_hand'].effort_limit_sim = 10.0
    robot_cfg.actuators['panda_hand'].stiffness = 2000.0
    robot_cfg.actuators['panda_hand'].damping = 20.0
    robot_cfg.spawn.collision_props = su.CollisionPropertiesCfg(contact_offset=0.00025, rest_offset=0.0)

    rigid = su.RigidBodyPropertiesCfg(disable_gravity=False, linear_damping=0.0,
               angular_damping=0.0, max_depenetration_velocity=0.5,
               solver_position_iteration_count=32, solver_velocity_iteration_count=4)

    @configclass
    class SceneCfg(InteractiveSceneCfg):
        ground = AssetBaseCfg(prim_path='/World/Ground', spawn=su.GroundPlaneCfg())
        light = AssetBaseCfg(prim_path='/World/Light', spawn=su.DomeLightCfg(intensity=1800))
        robot = robot_cfg
        table = AssetBaseCfg(prim_path='{ENV_REGEX_NS}/Table', spawn=su.UsdFileCfg(
            usd_path=assets["table"]["path"]),
            init_state=AssetBaseCfg.InitialStateCfg(pos=tuple(panel['table_position_m']) if panel else (0,0,1.003)))
        rack = AssetBaseCfg(prim_path='{ENV_REGEX_NS}/Rack', spawn=su.UsdFileCfg(
            usd_path=panel["assets"]["rack_overlay"]["path"]),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(0.5,1.10,0), rot=(0.70710678,0,0,0.70710678)))
        fixtures = AssetBaseCfg(prim_path='{ENV_REGEX_NS}/Fixtures', spawn=su.UsdFileCfg(
            usd_path=panel["assets"]["fixtures"]["path"]))
        plug = RigidObjectCfg(prim_path='{ENV_REGEX_NS}/Plug', spawn=su.UsdFileCfg(
            usd_path=manifest['plug'], rigid_props=rigid,
            activate_contact_sensors=True), init_state=RigidObjectCfg.InitialStateCfg(pos=tuple(plug_position)))
        latch = RigidObjectCfg(prim_path='{ENV_REGEX_NS}/Latch', spawn=su.UsdFileCfg(
            usd_path=manifest['latch'], rigid_props=rigid, activate_contact_sensors=bool(panel)),
            init_state=RigidObjectCfg.InitialStateCfg(pos=tuple(latch_position)))
        socket = AssetBaseCfg(prim_path='{ENV_REGEX_NS}/Socket', spawn=su.UsdFileCfg(
            usd_path=manifest['socket']),
            init_state=AssetBaseCfg.InitialStateCfg(pos=socket_position))
        left = ContactSensorCfg(prim_path='{ENV_REGEX_NS}/Robot/panda_leftfinger',
            filter_prim_paths_expr=['{ENV_REGEX_NS}/Plug'], update_period=0.0, history_length=0,max_contact_data_count_per_prim=128)
        socket_contact = ContactSensorCfg(prim_path='{ENV_REGEX_NS}/Plug',
            filter_prim_paths_expr=['{ENV_REGEX_NS}/Socket'], update_period=0.0, history_length=0,max_contact_data_count_per_prim=128)
        right = ContactSensorCfg(prim_path='{ENV_REGEX_NS}/Robot/panda_rightfinger',
            filter_prim_paths_expr=['{ENV_REGEX_NS}/Plug'], update_period=0.0, history_length=0,max_contact_data_count_per_prim=128)

    sim = su.SimulationContext(su.SimulationCfg(dt=dt, render_interval=args.exchange_hz//30,
                                               device='cuda:0', use_fabric=True))
    scene_cfg = SceneCfg(num_envs=1, env_spacing=4.0)
    panel_sensor_names = []
    if panel:
        for body in [f'Robot/panda_link{i}' for i in range(8)] + ['Robot/panda_hand','Robot/panda_leftfinger','Robot/panda_rightfinger','Plug','Latch']:
            name = 'fixture_' + body.replace('/', '_')
            setattr(scene_cfg, name, ContactSensorCfg(prim_path='{ENV_REGEX_NS}/' + body,
                filter_prim_paths_expr=['{ENV_REGEX_NS}/Fixtures'],update_period=0.,history_length=0,max_contact_data_count_per_prim=256))
            panel_sensor_names.append(name)
    scene = InteractiveScene(scene_cfg)
    stage = get_current_stage()
    prefix = '/World/envs/env_0'
    # A fixed kinematic actor gives GPU contact filters a rigid-body target;
    # its imported triangle cavity is unchanged and its pose is never commanded.
    socket_body=UsdPhysics.RigidBodyAPI.Apply(stage.GetPrimAtPath(prefix+'/Socket'))
    socket_body.CreateRigidBodyEnabledAttr(True)
    socket_body.CreateKinematicEnabledAttr(True)
    material.func('/World/ContactMaterial', material)
    for label in ('Plug', 'Latch', 'Socket'):
        su.bind_physics_material(prefix+'/'+label, '/World/ContactMaterial')
    # Every actual rack/fixture mesh participates in rigid-body collision.
    for label in ('Rack','Fixtures','Robot'):
        parent = stage.GetPrimAtPath(prefix+'/'+label)
        # Changing instanceability invalidates descendant proxy handles; retain paths only.
        paths = [p.GetPath() for p in Usd.PrimRange(parent, Usd.TraverseInstanceProxies()) if p.IsInstance()]
        for path in paths:
            p = stage.GetPrimAtPath(path)
            if p.IsValid() and p.IsInstance():
                p.SetInstanceable(False)
        if label == 'Robot':
            su.bind_physics_material(prefix+'/'+label, '/World/ContactMaterial')
            continue
        if label == 'Fixtures' and panel:
            from .panel import fixture_mesh_selection
            selected = fixture_mesh_selection(stage, prefix, panel)
            for prim, physical in selected:
                if physical:
                    UsdPhysics.CollisionAPI.Apply(prim)
                    UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr('none')
                elif prim.HasAPI(UsdPhysics.CollisionAPI):
                    prim.RemoveAPI(UsdPhysics.CollisionAPI)
            actor = UsdPhysics.RigidBodyAPI.Apply(parent)
            actor.CreateRigidBodyEnabledAttr(True); actor.CreateKinematicEnabledAttr(True)
            su.bind_physics_material(prefix+'/Fixtures', '/World/ContactMaterial')
            report['fixture_collision_selection'] = [dict(path=str(p.GetPath()),physical=flag) for p,flag in selected]
            continue
        for p in Usd.PrimRange(parent):
            if p.IsA(UsdGeom.Mesh):
                UsdPhysics.CollisionAPI.Apply(p)
                UsdPhysics.MeshCollisionAPI.Apply(p).CreateApproximationAttr('none')
    for p in Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies()):
        if p.HasAPI(UsdPhysics.CollisionAPI) and not p.IsInstanceProxy():
            api = PhysxSchema.PhysxCollisionAPI.Apply(p)
            api.CreateContactOffsetAttr(0.00025)
            api.CreateRestOffsetAttr(0.0)
        if p.GetPath().HasPrefix(prefix+'/Plug') or p.GetPath().HasPrefix(prefix+'/Latch'):
            if p.IsA(UsdGeom.Mesh):
                sdf = PhysxSchema.PhysxSDFMeshCollisionAPI.Apply(p)
                sdf.CreateSdfResolutionAttr(256)
                sdf.CreateSdfSubgridResolutionAttr(6)
    report['finger_colliders'] = [dict(path=str(p.GetPath()),contact_offset_m=PhysxSchema.PhysxCollisionAPI(p).GetContactOffsetAttr().Get(),rest_offset_m=PhysxSchema.PhysxCollisionAPI(p).GetRestOffsetAttr().Get())
        for p in Usd.PrimRange.Stage(stage) if p.HasAPI(UsdPhysics.CollisionAPI) and 'finger' in str(p.GetPath())]
    hinge = UsdPhysics.RevoluteJoint.Define(stage, prefix+'/LatchHinge')
    hinge.CreateBody0Rel().SetTargets([Sdf.Path(prefix+'/Plug')])
    hinge.CreateBody1Rel().SetTargets([Sdf.Path(prefix+'/Latch')])
    hinge.CreateLocalPos0Attr(Gf.Vec3f(*manifest['latch_pivot_in_plug']))
    hinge.CreateLocalPos1Attr(Gf.Vec3f(0))
    hinge.CreateAxisAttr('X')
    hinge.CreateLowerLimitAttr(-17.18873385)
    hinge.CreateUpperLimitAttr(11.4591559)
    hinge.CreateCollisionEnabledAttr(False)
    drive = UsdPhysics.DriveAPI.Apply(hinge.GetPrim(), 'angular')
    drive.CreateTypeAttr('force')
    drive.CreateStiffnessAttr(0.15)
    drive.CreateDampingAttr(0.03)
    drive.CreateTargetPositionAttr(0.0)
    import faulthandler
    print('Scene authored; resetting physics', flush=True)
    faulthandler.dump_traceback_later(60, repeat=True)
    initialization_joint=None
    if args.settle_seconds>0:
        initialization_joint=UsdPhysics.FixedJoint.Define(stage,prefix+'/InitializationPlugSupport')
        initialization_joint.CreateBody1Rel().SetTargets([Sdf.Path(prefix+'/Plug')])
        initialization_joint.CreateLocalPos0Attr(Gf.Vec3f(*map(float,plug_position)))
        initialization_joint.CreateLocalPos1Attr(Gf.Vec3f(0))
        initialization_joint.CreateJointEnabledAttr(True)
        initialization_joint.GetPrim().CreateAttribute('physics:excludeFromArticulation',Sdf.ValueTypeNames.Bool).Set(True)
    sim.reset()
    faulthandler.cancel_dump_traceback_later()
    print('Physics reset complete', flush=True)
    robot, plug, latch = scene['robot'], scene['plug'], scene['latch']
    # Asset initialization is explicit; a SimulationContext reset does not write IsaacLab defaults.
    robot.write_joint_state_to_sim(robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone())
    robot.set_joint_position_target(robot.data.default_joint_pos.clone())
    scene.reset()
    sim.forward()
    scene.update(dt)
    if panel:
        from ropesim.isaac_bridge import body_geometry
        def collider_world_bounds(path):
            vertices,_ = body_geometry(stage,path)
            matrix=np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(stage.GetPrimAtPath(path)))
            points=(np.c_[vertices,np.ones(len(vertices))]@matrix)[:,:3]
            return np.array([points.min(0),points.max(0)])
        table_bounds=collider_world_bounds(prefix+'/Table')
        base_bounds=collider_world_bounds(prefix+'/Robot/panda_link0')
        if np.any(base_bounds[0,:2]<table_bounds[0,:2]+.01) or np.any(base_bounds[1,:2]>table_bounds[1,:2]-.01):
            raise ValueError('Robot base footprint outside actual tabletop collider')
        report['mounted_layout_audit']=dict(table_collider_bounds_m=table_bounds.tolist(),robot_base_collider_bounds_m=base_bounds.tolist(),
            robot_base_position_m=list(robot_cfg.init_state.pos),robot_joint_soft_limits_rad=robot.data.soft_joint_pos_limits.detach().cpu().numpy()[0].tolist(),
            base_xy_clearance_min_m=float(np.min(np.r_[base_bounds[0,:2]-table_bounds[0,:2],table_bounds[1,:2]-base_bounds[1,:2]])))
    hand_id = robot.body_names.index('panda_hand')
    arm_ids = [robot.joint_names.index('panda_joint'+str(i)) for i in range(1,8)]
    fingers = [robot.joint_names.index('panda_finger_joint'+str(i)) for i in (1,2)]
    controller = DifferentialIKController(DifferentialIKControllerCfg(command_type='pose',
                           use_relative_mode=False, ik_method='dls'), num_envs=1, device=sim.device)
    tcp_offset = torch.tensor([[0,0,0.1034]], device=sim.device)
    target_quat = torch.tensor([[0,2**-0.5,2**-0.5,0]], device=sim.device)
    cpu = lambda t: t.detach().cpu().numpy().copy()
    def tcp():
        q = robot.data.body_link_quat_w[:,hand_id]
        return robot.data.body_link_pos_w[:,hand_id]+quat_apply(q,tcp_offset), q
    start_tcp = cpu(tcp()[0])[0]
    coupler = None
    if not args.rigid_pilot:
        from .coupling import CableCoupling
        coupler = CableCoupling(scene, stage, manifest, args, dt)
    report['cable']=None if coupler is None else coupler.info
    if args.settle_seconds>0:
        if coupler is None:raise ValueError('Pre-settle is for cable initialization only')
        print('INIT_SETTLE_BEGIN',args.settle_seconds,flush=True)
        settling={k:[] for k in ('t','plug_pos','plug_quat','cable','reaction','energy')}
        settle_start=time.perf_counter()
        try:
            for j in range(round(args.settle_seconds/dt)):
                coupler.current_phase='initialization'
                robot.set_joint_effort_target(robot.root_physx_view.get_gravity_compensation_forces())
                coupler.before_physics();scene.write_data_to_sim();sim.step(render=False);scene.update(dt);coupler.after_physics()
                if j%(args.exchange_hz//30)==args.exchange_hz//30-1:
                    values=dict(t=(j+1)*dt,plug_pos=cpu(plug.data.root_link_pos_w)[0],plug_quat=cpu(plug.data.root_link_quat_w)[0],cable=coupler.positions(),reaction=coupler.reaction.copy(),energy=coupler.energy())
                    for key,value in values.items():settling[key].append(value)
        finally:
            np.savez_compressed(args.out/'settling.npz',**{k:np.asarray(v) for k,v in settling.items()})
            report['initialization']=dict(seconds=args.settle_seconds,wall_s=time.perf_counter()-settle_start,frames=len(settling['t']),temporary_world_joint=str(initialization_joint.GetPrim().GetPath()),natural_rest_shape='straight',gap_limit_unchanged_m=args.max_termination_gap,complete=len(settling['t'])>=round(args.settle_seconds*30),settling_sha256=hashlib.sha256((args.out/'settling.npz').read_bytes()).hexdigest())
        print('INIT_SETTLE_COMPLETE; disabling temporary support',flush=True)
        initialization_joint.GetJointEnabledAttr().Set(False)
        sim.forward()
        report['initialization']['support_joint_disabled_before_trial']=not initialization_joint.GetJointEnabledAttr().Get()
        report['initialization']['final_reaction_N']=coupler.reaction.tolist()
        report['initialization']['final_horizontal_reaction_N']=float(np.linalg.norm(coupler.reaction[:2]))
    report['setup_wall_s']=time.perf_counter()-setup_wall_start
    report['robot'] = dict(path=robot_cfg.spawn.usd_path, gravity_compensation=True,
                          arm_effort_limits_Nm=[87]*4+[12]*3, finger_effort_limit_N=10,
                          arm_stiffness_Nm_rad=1000,arm_damping_Nms_rad=80,max_joint_target_error_rad=args.max_joint_target_error)
    report['connector'] = manifest
    report['rigid_collision_meshes'] = sum(p.HasAPI(UsdPhysics.CollisionAPI) for p in Usd.PrimRange.Stage(stage,Usd.TraverseInstanceProxies()))
    report['cable'] = None if coupler is None else coupler.info
    joints=[]
    forbidden=[]
    for prim in Usd.PrimRange.Stage(stage):
        if prim.IsA(UsdPhysics.Joint):
            joint=UsdPhysics.Joint(prim)
            targets=[str(x) for x in joint.GetBody0Rel().GetTargets()+joint.GetBody1Rel().GetTargets()]
            joints.append(dict(path=str(prim.GetPath()),bodies=targets,enabled=joint.GetJointEnabledAttr().Get()))
            if any('/Plug' in x for x in targets) and str(prim.GetPath()) != prefix+'/LatchHinge' and joint.GetJointEnabledAttr().Get():
                forbidden.append(str(prim.GetPath()))
    report['joint_audit']=joints
    report['gripper_attachment_constraints']=len(forbidden)
    if forbidden:
        raise RuntimeError('Unexpected plug attachment joint: '+str(forbidden))
    stage.GetRootLayer().Export(str(args.out/'scene.usda'))
    report['scene_sha256']=hashlib.sha256((args.out/'scene.usda').read_bytes()).hexdigest()
    report['physics_scene_authored'] = [dict(path=str(p.GetPath()),gpu_dynamics=PhysxSchema.PhysxSceneAPI(p).GetEnableGPUDynamicsAttr().Get(),broadphase=PhysxSchema.PhysxSceneAPI(p).GetBroadphaseTypeAttr().Get()) for p in Usd.PrimRange.Stage(stage) if p.IsA(UsdPhysics.Scene)]
    rec = {k:[] for k in ('t','phase','robot_q','body_pos','body_quat','plug_pos','plug_quat','plug_vel',
                          'latch_pos','latch_quat','tcp','target','left_force','right_force','cable','reaction','socket_force','depth','lateral_error','axis_error','left_force_vector','right_force_vector','min_contact_separation','energy','applied_force','applied_torque','plug_com','latch_com','latch_vel','support_force','cable_vel','cable_quat','fixture_pair_force','fixture_pair_min_separation','seating_preload_m','seating_contact_dwell_s')}
    sim_times = []
    decimation = max(1, round(args.exchange_hz/240))
    capture = args.exchange_hz//30
    from .controller import ContactController
    seated = np.asarray(socket_position)+np.asarray(manifest['seated_plug_minus_socket'])+grasp_offset
    task_controller = ContactController(grip_point,start_tcp,args.case,seated,manifest['seated_depth_m'],release_mode=args.release_mode,
        seating_preload=dict(limit_m=.0005,speed_m_s=.00025,depth_margin_m=.00075,contact_dwell_s=.1) if panel else None)
    def measured():
        ee_pos,ee_quat=tcp()
        state=cpu(plug.data.root_state_w)[0]
        left=float(torch.linalg.vector_norm(scene['left'].data.force_matrix_w).item())
        right=float(torch.linalg.vector_norm(scene['right'].data.force_matrix_w).item())
        socket_force=float(torch.linalg.vector_norm(scene['socket_contact'].data.force_matrix_w).item())
        axis=quat_apply(plug.data.root_link_quat_w,torch.tensor([[0,1,0]],device=sim.device,dtype=torch.float32))
        tip=plug.data.root_link_pos_w+quat_apply(plug.data.root_link_quat_w,torch.tensor([[0,manifest['plug_tip_y'],0]],device=sim.device,dtype=torch.float32))
        depth=float(cpu(tip)[0,1]-socket_position[1]-manifest['socket_entry_y'])
        seated_pos=np.asarray(socket_position)+manifest['seated_plug_minus_socket']
        return dict(tcp=cpu(ee_pos)[0],angle=float(2*np.arccos(np.clip(abs(float((ee_quat*target_quat).sum().item())),-1,1))),
                    left=left,right=right,width=float(cpu(robot.data.joint_pos)[0,fingers].sum()),z=float(state[2]),
                    depth=depth,lateral=float(np.linalg.norm((state[:3]-seated_pos)[[0,2]])),
                    axis=float(np.arccos(np.clip(cpu(axis)[0,1],-1,1))),socket=socket_force)
    grasp_adapted=False
    final_target = start_tcp
    report['trial_start_physics_time_s']=float(sim.current_time)
    print('TRIAL_BEGIN',report.get('initialization',{}),flush=True)
    loop_wall_start=time.perf_counter()
    try:
        for i in range(round(args.seconds/dt)):
            t = i*dt
            if i % decimation == 0:
                final_target, opening, phase = task_controller.command(t,measured()) if args.case!='passive' else (start_tcp,.04,'passive')
                if phase=='route' and not grasp_adapted:
                    held_pos,held_q=tcp()
                    local_position=quat_apply_inverse(held_q,plug.data.root_link_pos_w-held_pos)
                    local_quat=quat_mul(quat_conjugate(held_q),plug.data.root_link_quat_w)
                    target_quat=quat_conjugate(local_quat)
                    actual_offset=-cpu(quat_apply(target_quat,local_position))[0]
                    task_controller.adapt_grasp(actual_offset-grasp_offset)
                    report['measured_grasp_frame']=dict(plug_in_tcp_position=cpu(local_position)[0].tolist(),plug_in_tcp_quat=cpu(local_quat)[0].tolist(),target_hand_quat=cpu(target_quat)[0].tolist())
                    grasp_adapted=True
                if args.open_gripper:
                    opening = .04
                goal = torch.cat((torch.tensor(final_target,dtype=torch.float32,device=sim.device)[None],target_quat),dim=-1)
                controller.set_command(goal)
                ee_pos, ee_q = tcp()
                jac = robot.root_physx_view.get_jacobians()[:,hand_id-1,:,arm_ids].clone()
                offset_w = quat_apply(ee_q,tcp_offset)
                jac[:,:3,:] -= skew_symmetric_matrix(offset_w) @ jac[:,3:,:]
                q = robot.data.joint_pos[:,arm_ids]
                goal_q = controller.compute(ee_pos,ee_q,jac,q) if args.case!='passive' else robot.data.default_joint_pos[:,arm_ids]
                goal_q = q + (goal_q-q).clamp(-args.max_joint_target_error,args.max_joint_target_error)
                limits = robot.data.soft_joint_pos_limits[:,arm_ids]
                goal_q = torch.maximum(torch.minimum(goal_q,limits[...,1]),limits[...,0])
                robot.set_joint_position_target(goal_q,joint_ids=arm_ids)
                robot.set_joint_position_target(torch.full((1,2),opening,device=sim.device),joint_ids=fingers)
            robot.set_joint_effort_target(robot.root_physx_view.get_gravity_compensation_forces())
            if coupler is not None:
                coupler.current_phase=phase
                coupler.before_physics()
            scene.write_data_to_sim()
            wall = time.perf_counter()
            sim.step(render=False)
            scene.update(dt)
            if coupler is not None:
                coupler.after_physics()
            if i % capture == capture-1:
                torch.cuda.synchronize()
                sim_times.append((time.perf_counter()-wall)*1000)
                state = cpu(plug.data.root_state_w)[0]
                if not np.isfinite(state).all():
                    report['finite'] = False
                    raise RuntimeError('Nonfinite connector state')
                if np.linalg.norm(state[7:10]) > 10:
                    report['speed_boundary_exceeded']=True
                    raise RuntimeError('Excessive connector velocity exceeds10m/s')
                left = float(torch.linalg.vector_norm(scene['left'].data.force_matrix_w).item())
                right = float(torch.linalg.vector_norm(scene['right'].data.force_matrix_w).item())
                socket_force = float(torch.linalg.vector_norm(scene['socket_contact'].data.force_matrix_w).item())
                axis = quat_apply(plug.data.root_link_quat_w,torch.tensor([[0,1,0]],device=sim.device,dtype=torch.float32))
                tip = plug.data.root_link_pos_w+quat_apply(plug.data.root_link_quat_w,torch.tensor([[0,manifest['plug_tip_y'],0]],device=sim.device,dtype=torch.float32))
                depth = float(cpu(tip)[0,1] - socket_position[1] - manifest['socket_entry_y'])
                seated_pos = np.asarray(socket_position)+manifest['seated_plug_minus_socket']
                lateral_error = float(np.linalg.norm((state[:3]-seated_pos)[[0,2]]))
                axis_error = float(np.arccos(np.clip(cpu(axis)[0,1],-1,1)))
                min_separation=0.
                fixture_forces, fixture_separations = [], []
                for sensor_name in ['left','right','socket_contact'] + panel_sensor_names:
                    _,_,_,separation,count,start = scene[sensor_name].contact_physx_view.get_contact_data(dt=dt)
                    pair_min = 0.
                    for n,a in zip(cpu(count).ravel(),cpu(start).ravel()):
                        if n:
                            pair_min=min(pair_min,float(cpu(separation)[int(a):int(a+n)].min()))
                    min_separation=min(min_separation,pair_min)
                    if sensor_name in panel_sensor_names:
                        fixture_forces.append(float(torch.linalg.vector_norm(scene[sensor_name].data.force_matrix_w).item()))
                        fixture_separations.append(pair_min)
                if min_separation < -.001:
                    report['contact_boundary_failure']=dict(t_s=(i+1)*dt,minimum_separation_m=min_separation,fixture_pair_min_separation_m=fixture_separations,sensor_names=panel_sensor_names)
                    raise RuntimeError(f'Contact penetration exceeds1mm: {min_separation}')
                values = dict(t=(i+1)*dt, phase=phase, robot_q=cpu(robot.data.joint_pos)[0],
                    body_pos=cpu(robot.data.body_link_pos_w)[0], body_quat=cpu(robot.data.body_link_quat_w)[0],
                    plug_pos=state[:3], plug_quat=state[3:7], plug_vel=state[7:13],
                    latch_pos=cpu(latch.data.root_link_pos_w)[0], latch_quat=cpu(latch.data.root_link_quat_w)[0],
                    tcp=cpu(tcp()[0])[0], target=np.array(final_target), left_force=left, right_force=right,
                    seating_preload_m=task_controller.preload_peak_m,seating_contact_dwell_s=task_controller.seating_observation.get('contact_dwell_s',0.),
                    fixture_pair_force=np.asarray(fixture_forces),fixture_pair_min_separation=np.asarray(fixture_separations),
                    cable=np.empty((0,3)) if coupler is None else coupler.positions(),
                    cable_vel=np.empty((0,3)) if coupler is None else coupler.velocities(),
                    cable_quat=np.empty((0,4)) if coupler is None else coupler.quaternions(),
                    reaction=np.zeros(3) if coupler is None else coupler.reaction.copy(),
                    socket_force=socket_force,depth=depth,lateral_error=lateral_error,axis_error=axis_error,min_contact_separation=min_separation,
                    left_force_vector=cpu(scene['left'].data.force_matrix_w).reshape(-1,3).sum(0),
                    right_force_vector=cpu(scene['right'].data.force_matrix_w).reshape(-1,3).sum(0),
                    energy=np.zeros(8) if coupler is None else coupler.energy(),
                    applied_force=np.zeros(3) if coupler is None else coupler.applied_force.copy(),
                    applied_torque=np.zeros(3) if coupler is None else coupler.applied_torque.copy(),
                    plug_com=cpu(plug.data.root_com_pos_w)[0],latch_com=cpu(latch.data.root_com_pos_w)[0],
                    latch_vel=cpu(latch.data.root_com_state_w)[0,7:13],
                    support_force=max(0.,float(scene['socket_contact'].data.net_forces_w[0,0,2].item())))
                for name,value in values.items():
                    data=np.asarray(value)
                    if data.dtype.kind in 'fi' and not np.isfinite(data).all():
                        report['finite']=False
                        raise RuntimeError('Nonfinite recorded state: '+name)
                for name,value in values.items():
                    rec[name].append(value)
                if len(rec['t']) % 30 == 0:
                    print(json.dumps(dict(t=round((i+1)*dt,3),phase=phase,plug=state[:3].tolist(),
                        tcp=values['tcp'].tolist(), fingers=cpu(robot.data.joint_pos)[0,fingers].tolist(),
                        contact_N=[left,right])),flush=True)
                    (args.out/'progress.json').write_text(json.dumps(dict(t_s=float((i+1)*dt),phase=phase,controller_stage=task_controller.stage,failure=task_controller.failure,plug_position=state[:3].tolist(),finger_forces_N=[left,right]),indent=2)+'\n')
                if task_controller.stage == 'release' and (i+1)*dt-task_controller.since >= 3.0:
                    break
    finally:
        report['simulation_loop_wall_s']=time.perf_counter()-loop_wall_start
        np.savez_compressed(args.out/'trajectory.npz', **{k:np.asarray(v) for k,v in rec.items()},
                            body_names=np.asarray(robot.body_names), joint_names=np.asarray(robot.joint_names))
        report['recorded_frames'] = len(rec['t'])
        if panel and rec['t']:
            report['fixture_contact_audit']=dict(sample_hz=30,sensors=panel_sensor_names,target=prefix+'/Fixtures',
                peak_pair_force_N=np.asarray(rec['fixture_pair_force']).max(0).tolist(),
                minimum_pair_separation_m=np.asarray(rec['fixture_pair_min_separation']).min(0).tolist(),
                scope='All robot links, plug and latch against every physical fixture; sampled, not swept certification')
        report['controller'] = dict(stage=task_controller.stage,failure=task_controller.failure,failures=task_controller.failures,transitions=task_controller.events,seating_preload_profile=task_controller.seating_preload,seating_observation=task_controller.seating_observation,seating_events=task_controller.seating_events)
        if coupler is not None:
            report['cable'] = coupler.info
        if len(rec['t'])>1:
            report['partial_stage_scores']=True
            report['acquisition']=score_acquisition(rec['t'],np.asarray(rec['plug_pos'])[:,2],rec['left_force'],rec['right_force'],rec['phase'])
            report['grasp']=score_grasp(rec['t'],np.asarray(rec['plug_pos'])[:,2],rec['left_force'],rec['right_force'],rec['phase']) if args.case=='grasp' else dict(passed=None,scope='standalone release-drop test not applicable')
            report['insertion']=score_insertion(rec['t'],rec['depth'],rec['lateral_error'],rec['axis_error'],rec['socket_force'],np.maximum(rec['left_force'],rec['right_force']),rec['phase'],manifest['seated_depth_m'],tcp_z=np.asarray(rec['tcp'])[:,2])
            report['insertion']['attempted']=any(event['to_stage']=='insert' for event in task_controller.events)
            report['routing']=dict(attempted=any(event['to_stage']=='route1' for event in task_controller.events))
    report['partial_stage_scores']=False
    report['acquisition'] = score_acquisition(rec['t'],np.asarray(rec['plug_pos'])[:,2],rec['left_force'],rec['right_force'],rec['phase'])
    report['grasp'] = score_grasp(rec['t'],np.asarray(rec['plug_pos'])[:,2],rec['left_force'],rec['right_force'],rec['phase']) if args.case=='grasp' else dict(passed=None,scope='standalone free-drop release test is not applicable; see acquisition and insertion')
    report['insertion'] = score_insertion(rec['t'],rec['depth'],rec['lateral_error'],rec['axis_error'],rec['socket_force'],np.maximum(rec['left_force'],rec['right_force']),rec['phase'],manifest['seated_depth_m'],tcp_z=np.asarray(rec['tcp'])[:,2])
    report['routing'] = dict(attempted=any(event['to_stage']=='route1' for event in task_controller.events),mesh_phase_peak_force_N=None if coupler is None else coupler.info.get('mesh_phase_peak_force_N'),mesh_phase_contact_exchange_counts=None if coupler is None else coupler.info.get('mesh_phase_contact_exchange_counts'))
    report['insertion']['attempted']=any(event['to_stage']=='insert' for event in task_controller.events)
    report['free_drop_release']=report['grasp'].copy() if args.case=='grasp' and args.release_mode=='drop' else dict(passed=None,attempted=False)
    report['controlled_release']=score_controlled_release(rec['t'],np.asarray(rec['plug_pos'])[:,2],np.asarray(rec['tcp'])[:,2],rec['left_force'],rec['right_force'],rec['support_force'],rec['phase']) if args.release_mode=='place' and np.any(np.asarray(rec['phase'])=='place') else dict(passed=None,attempted=False)
    if args.case=='grasp' and args.release_mode=='place':
        report['grasp']=dict(passed=bool(report['acquisition']['passed'] and report['controlled_release']['passed']),mode='contact_lift_hold_controlled_place_release',acquisition=report['acquisition'],release=report['controlled_release'])
    report['minimum_sampled_contact_separation_m'] = float(np.min(rec['min_contact_separation']))
    report['complete'] = True
    report['recorded_frames'] = len(rec['t'])
    report['sampled_exchange_ms'] = dict(median=float(np.median(sim_times)), scope='sampled sim.step+scene.update+rod.after_physics; excludes before_physics/controller; not a throughput benchmark')
    report['trajectory_sha256'] = hashlib.sha256((args.out/'trajectory.npz').read_bytes()).hexdigest()
    report['actual_object_initial_pose'] = rec['plug_pos'][0].tolist()
    print(json.dumps(report['grasp']),flush=True)
    print('SIM_STOP_BEGIN',flush=True)
    # Public cleanup unsubscribes IsaacLab's keep-rendering-on-STOP callback.
    # Otherwise its while-not-playing render loop intentionally blocks standalone shutdown.
    su.SimulationContext.clear_instance()
    sim.stop()
    print('SIM_STOP_RETURNED',flush=True)


if __name__ == '__main__':
    main()
