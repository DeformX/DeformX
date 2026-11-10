"""Simulate a joint-actuated Panda drawing one cable out of an overlapping pile."""
import argparse
import faulthandler
import hashlib
import json
import math
from pathlib import Path
import shutil
import socket
import time
import traceback

from .model import END_TIME, FPS


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def execute(args, report):
    import numpy as np
    import torch
    from pxr import Usd, UsdGeom, UsdPhysics, PhysxSchema
    import isaaclab.sim as su
    from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
    from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
    from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
    from isaaclab.sensors import ContactSensorCfg
    from isaaclab.utils import configclass
    from isaaclab.utils.math import quat_apply, skew_symmetric_matrix
    from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG
    from isaacsim.core.utils.stage import get_current_stage
    from ropesim.isaac_bridge import body_geometry

    from .coupling import BundleCoupling, summarize
    from .model import PULL_TAB_MASS, PULL_TAB_SIZE, PULL_TAB_XY, ROBOT_POS, TABLE_POS, command, score

    dt = 1. / args.exchange_hz
    stride = args.exchange_hz // FPS
    cpu = lambda tensor: tensor.detach().cpu().numpy().copy()
    panda = FRANKA_PANDA_CFG.copy()
    panda.prim_path = '{ENV_REGEX_NS}/Robot'
    panda.spawn.usd_path = str(args.robot_usd.resolve())
    panda.spawn.activate_contact_sensors = True
    panda.spawn.rigid_props.disable_gravity = False
    panda.spawn.articulation_props.solver_position_iteration_count = 32
    panda.spawn.articulation_props.solver_velocity_iteration_count = 4
    panda.init_state.pos = ROBOT_POS
    for name in ('panda_shoulder','panda_forearm'):
        panda.actuators[name].stiffness = 1000.
        panda.actuators[name].damping = 80.
    panda.actuators['panda_hand'].effort_limit_sim = 10.
    panda.actuators['panda_hand'].stiffness = 2000.
    panda.actuators['panda_hand'].damping = 20.

    @configclass
    class SceneCfg(InteractiveSceneCfg):
        ground = AssetBaseCfg(prim_path='/World/Ground', spawn=su.GroundPlaneCfg())
        robot = panda
        table = AssetBaseCfg(prim_path='{ENV_REGEX_NS}/Table',
            spawn=su.UsdFileCfg(usd_path=str(args.table_usd.resolve())),
            init_state=AssetBaseCfg.InitialStateCfg(pos=TABLE_POS))
        tab = RigidObjectCfg(prim_path='{ENV_REGEX_NS}/PullTab',
            spawn=su.UsdFileCfg(usd_path=str(args.tab_usd.resolve())),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(*PULL_TAB_XY, 1.+PULL_TAB_SIZE[2]/2+.0002)))
        left = ContactSensorCfg(prim_path='{ENV_REGEX_NS}/Robot/panda_leftfinger',
            filter_prim_paths_expr=['{ENV_REGEX_NS}/PullTab'], max_contact_data_count_per_prim=128)
        right = ContactSensorCfg(prim_path='{ENV_REGEX_NS}/Robot/panda_rightfinger',
            filter_prim_paths_expr=['{ENV_REGEX_NS}/PullTab'], max_contact_data_count_per_prim=128)

    sim = su.SimulationContext(su.SimulationCfg(dt=dt, render_interval=stride,
                                               device='cuda:0', use_fabric=True))
    scene = InteractiveScene(SceneCfg(num_envs=1, env_spacing=4.))
    stage = get_current_stage()
    prefix = '/World/envs/env_0'
    tab_prim = stage.GetPrimAtPath(prefix+'/PullTab')
    # A generic rigid convex tab, using the Blender mesh. Contact sensor mass and
    # rigidity are authored on the referenced root before the PhysX reset.
    su.define_rigid_body_properties(str(tab_prim.GetPath()),
        su.RigidBodyPropertiesCfg(disable_gravity=False, linear_damping=.05,
            angular_damping=.10, max_depenetration_velocity=.5,
            solver_position_iteration_count=32, solver_velocity_iteration_count=4))
    su.define_mass_properties(str(tab_prim.GetPath()), su.MassPropertiesCfg(mass=PULL_TAB_MASS))
    collider_paths = []
    for prim in Usd.PrimRange(tab_prim):
        if prim.IsA(UsdGeom.Mesh):
            UsdPhysics.CollisionAPI.Apply(prim)
            UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr('convexHull')
            collider_paths.append(str(prim.GetPath()))
    if len(collider_paths) != 1:
        raise RuntimeError(f'Expected one Blender pull tab mesh, found {collider_paths}')
    tab_material = su.RigidBodyMaterialCfg(static_friction=.95, dynamic_friction=.80, restitution=0.)
    tab_material.func('/World/PullTabContactMaterial', tab_material)
    su.bind_physics_material(prefix+'/PullTab', '/World/PullTabContactMaterial')
    robot_prim = stage.GetPrimAtPath(prefix+'/Robot')
    for path in [p.GetPath() for p in Usd.PrimRange(robot_prim, Usd.TraverseInstanceProxies()) if p.IsInstance()]:
        stage.GetPrimAtPath(path).SetInstanceable(False)
    for prim in Usd.PrimRange.Stage(stage):
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            api = PhysxSchema.PhysxCollisionAPI.Apply(prim)
            api.CreateContactOffsetAttr(.00025)
            api.CreateRestOffsetAttr(0.)
        if prim.IsA(UsdPhysics.Joint):
            joint = UsdPhysics.Joint(prim)
            targets = joint.GetBody0Rel().GetTargets() + joint.GetBody1Rel().GetTargets()
            if any(str(target).startswith(prefix+'/PullTab') for target in targets):
                raise RuntimeError(f'Unexpected constraint on tab: {prim.GetPath()}')
    print('Resetting scene', flush=True)
    faulthandler.dump_traceback_later(60, repeat=True)
    try:
        sim.reset()
    finally:
        faulthandler.cancel_dump_traceback_later()
    print('Physics reset complete', flush=True)
    torch.cuda.set_stream(torch.cuda.default_stream())
    robot, tab = scene['robot'], scene['tab']
    robot.write_joint_state_to_sim(robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone())
    robot.set_joint_position_target(robot.data.default_joint_pos.clone())
    scene.reset()
    sim.forward()
    scene.update(dt)
    table_prim = stage.GetPrimAtPath(prefix+'/Table')
    verts, _ = body_geometry(stage, str(table_prim.GetPath()))
    matrix = np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(table_prim))
    verts_w = (np.c_[verts,np.ones(len(verts))] @ matrix)[:,:3]
    table_z = float(verts_w[:,2].max())
    report['table_collision_bounds_m'] = [verts_w.min(axis=0).tolist(), verts_w.max(axis=0).tolist()]
    report['tab'] = dict(mass_kg=float(tab.root_physx_view.get_masses()[0,0]), colliders=collider_paths,
                        dimensions_m=PULL_TAB_SIZE, friction_static_dynamic=[.95,.80])
    bridge = BundleCoupling(scene, stage, args, table_z)
    if np.any(bridge.initial[:,:,:2] < verts_w[:,:2].min(axis=0)+.01) or np.any(
            bridge.initial[:,:,:2] > verts_w[:,:2].max(axis=0)-.01):
        raise RuntimeError('Initial cable pile exceeds imported table collision bounds')
    report['usd_layers'] = {str(Path(layer.realPath).resolve()): sha(layer.realPath)
                            for layer in stage.GetUsedLayers() if layer.realPath and Path(layer.realPath).is_file()}
    scene_path = args.out/'scene.usda'
    stage.GetRootLayer().Export(str(scene_path))
    report['scene_sha256'] = sha(scene_path)
    np.save(args.out/'initial_cables.npy', bridge.initial)
    report['initial_cables_sha256'] = sha(args.out/'initial_cables.npy')

    hand = robot.body_names.index('panda_hand')
    arm = [robot.joint_names.index(f'panda_joint{i}') for i in range(1,8)]
    fingers = [robot.joint_names.index(f'panda_finger_joint{i}') for i in (1,2)]
    ik = DifferentialIKController(DifferentialIKControllerCfg(command_type='pose',
                           use_relative_mode=False, ik_method='dls'), num_envs=1, device=sim.device)
    tcp_offset = torch.tensor([[0, 0, .1034]], device=sim.device)
    goal_quat = torch.tensor([[0, 2**-.5, 2**-.5, 0]], device=sim.device)
    limits = robot.data.soft_joint_pos_limits[:,arm]

    def tcp():
        q = robot.data.body_link_quat_w[:,hand]
        return robot.data.body_link_pos_w[:,hand]+quat_apply(q,tcp_offset), q

    def advance():
        robot.set_joint_effort_target(robot.root_physx_view.get_gravity_compensation_forces())
        bridge.before()
        scene.write_data_to_sim()
        sim.step(render=False)
        scene.update(dt)
        return bridge.after()

    def pair_force():
        return np.array([float(torch.linalg.vector_norm(scene[name].data.force_matrix_w))
                         for name in ('left','right')])

    records = []
    started = time.perf_counter()
    settle_start = started
    print('Settling', args.settle, flush=True)
    try:
        for j in range(round(args.settle/dt)):
            gap = advance()
            if (j+1) % args.exchange_hz == 0:
                print('Settled seconds', (j+1)*dt, flush=True)
        p, v, _ = bridge.snapshot()
        report['settling'] = dict(simulated_s=args.settle, wall_s=time.perf_counter()-settle_start,
            maximum_final_vertex_speed_m_s=float(np.linalg.norm(v,axis=-1).max()),
            rms_final_vertex_speed_m_s=float(np.sqrt(np.mean(np.sum(v*v,axis=-1)))),
            final_tab_position_m=cpu(tab.data.root_link_pos_w)[0].tolist(),
            final_tab_speed_m_s=float(torch.linalg.vector_norm(tab.data.root_com_lin_vel_w)))
        home = cpu(tcp()[0])[0]
        grip = cpu(tab.data.root_link_pos_w)[0] + [0,0,args.grip_z]
        report['command_home_m'] = home.tolist()
        report['command_grip_m'] = grip.tolist()
        started = time.perf_counter()
        for step in range(round(args.seconds/dt)):
            if step % (args.exchange_hz // 120) == 0:
                target, opening, phase = command(step*dt, home, grip)
                if args.open_gripper:
                    opening = .04
                ik.set_command(torch.cat((torch.tensor(target[None],dtype=torch.float32,device=sim.device),goal_quat), dim=-1))
                ee, q = tcp()
                jac = robot.root_physx_view.get_jacobians()[:,hand-1,:,arm].clone()
                jac[:,:3,:] -= skew_symmetric_matrix(quat_apply(q,tcp_offset)) @ jac[:,3:,:]
                actual_joints = robot.data.joint_pos[:,arm]
                joint_target = actual_joints + (ik.compute(ee,q,jac,actual_joints)-actual_joints).clamp(-.10,.10)
                joint_target = torch.maximum(torch.minimum(joint_target,limits[...,1]),limits[...,0])
                robot.set_joint_position_target(joint_target,joint_ids=arm)
                robot.set_joint_position_target(torch.full((1,2),opening,device=sim.device),joint_ids=fingers)
            gap = advance()
            if (step+1) % stride:
                continue
            p, v, q = bridge.snapshot()
            state = cpu(tab.data.root_link_state_w)[0]
            separation = 0.
            for name in ('left','right'):
                _,_,_,sep,count,start = scene[name].contact_physx_view.get_contact_data(dt=dt)
                for n, k in zip(cpu(count).ravel(),cpu(start).ravel()):
                    if n:
                        separation = min(separation,float(cpu(sep)[k:k+n].min()))
            values = dict(t=(step+1)*dt, phase=phase, cable=p, cable_vel=v, cable_quat=q,
                tab_pos=state[:3], tab_quat=state[3:7], tab_vel=state[7:],
                tab_exit=cpu(bridge.exit())[0], termination_gap=gap,
                tab_force_from_cable=-bridge.endpoint_force.copy(), finger_force=pair_force(),
                minimum_finger_contact_separation_m=separation, mesh_wrenches=bridge.last_wrenches,
                robot_pos=cpu(robot.data.body_link_pos_w)[0], robot_quat=cpu(robot.data.body_link_quat_w)[0],
                robot_joints=cpu(robot.data.joint_pos)[0], tcp=cpu(tcp()[0])[0], tcp_target=target)
            if any(not np.isfinite(v).all() for v in values.values() if not isinstance(v,str)):
                raise RuntimeError('Nonfinite measured state')
            if np.linalg.norm(state[7:10]) > 5. or separation < -.001:
                raise RuntimeError(f'Rigid contact/velocity bound at {values["t"]:.3f}: {separation}')
            records.append(values)
            if len(records) % FPS == 0:
                status = {k:np.asarray(values[k]).tolist() for k in
                          ('t','phase','tab_pos','tcp','tcp_target','finger_force','termination_gap')}
                status['target_free_tip_m'] = p[0,-1].tolist()
                print(json.dumps(status), flush=True)
    finally:
        report['cable'] = summarize(bridge)
        report['simulation_loop_wall_s'] = time.perf_counter()-started
        report['frames'] = len(records)
        if records:
            arrays = {k:np.asarray([row[k] for row in records]) for k in records[0]}
            record = args.out/'trajectory.npz'
            np.savez_compressed(record, **arrays, body_names=np.asarray(robot.body_names),
                                joint_names=np.asarray(robot.joint_names))
            report['trajectory_sha256'] = sha(record)
            report['minimum_sampled_finger_separation_m'] = float(arrays['minimum_finger_contact_separation_m'].min())
    report['metrics'] = score(arrays)
    report['complete'] = True
    print('SCORE', json.dumps(report['metrics']), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ('out','cache','robot-usd','table-usd','tab-usd'):
        ap.add_argument('--'+name,type=Path,required=True)
    ap.add_argument('--seconds',type=float,default=END_TIME)
    ap.add_argument('--settle',type=float,default=2.5)
    ap.add_argument('--exchange-hz',type=int,default=1200)
    ap.add_argument('--iterations',type=int,default=24)
    ap.add_argument('--grip-z',type=float,default=.005)
    ap.add_argument('--open-gripper',action='store_true')
    ap.add_argument('--no-rod-contact',action='store_true')
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(ap)
    args = ap.parse_args()
    if (args.exchange_hz <= 0 or args.exchange_hz % 120 or 4800 % args.exchange_hz or args.iterations < 1
            or not math.isfinite(args.seconds) or args.seconds < 1/FPS or not math.isfinite(args.settle)
            or args.settle < 0 or not math.isfinite(args.grip_z)):
        ap.error('Use a positive exchange rate divisible by 120 and dividing 4800, positive iterations and finite durations')
    args.out.mkdir(parents=True,exist_ok=False)
    snapshot = args.out/'source'
    snapshot.mkdir()
    for file in Path(__file__).parent.glob('*.py'):
        shutil.copyfile(file,snapshot/file.name)
    report = dict(host=socket.gethostname(), complete=False, frame_rate=FPS, errors=[],
                  attachment_joints=0, post_reset_tab_pose_writes=0,
                  settings={key:str(value) if isinstance(value,Path) else value for key,value in vars(args).items()},
                  source_sha256={p.name:sha(p) for p in snapshot.glob('*.py')},
                  asset_sha256={name:sha(path) for name,path in dict(robot=args.robot_usd,table=args.table_usd,tab=args.tab_usd).items()})
    app = None
    try:
        app = AppLauncher(args).app
        execute(args, report)
    except Exception:
        report['errors'].append(traceback.format_exc())
        print(report['errors'][-1], flush=True)
        raise
    finally:
        report['app_close_invoked'] = app is not None
        (args.out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        if app is not None:
            from isaaclab.sim import SimulationContext
            SimulationContext.clear_instance()
            app.close(wait_for_replicator=False,skip_cleanup=False)


if __name__ == '__main__':
    main()
