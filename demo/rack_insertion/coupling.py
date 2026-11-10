"""Explicit point-termination and mesh-contact exchange, with no grasp binding."""
import hashlib
from pathlib import Path
import numpy as np


def vertex_masses(points, density):
    points = np.asarray(points, dtype=np.float64)
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=-1)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 3 or not np.isfinite(points).all() or np.any(lengths <= 0) or not np.isfinite(density) or density <= 0:
        raise ValueError('finite noncoincident vertices and positive linear density required')
    return .5 * density * (np.r_[lengths, 0] + np.r_[0, lengths])


def initial_cable_points(root,mode,table_height=1.):
    root=np.asarray(root,dtype=float)
    if root.shape!=(3,) or not np.isfinite(root).all() or mode not in ('level','draped'):
        raise ValueError('finite cable exit and level/draped initialization required')
    u=np.linspace(0,1,49)
    points=root+np.c_[-.32*np.sin(u*np.pi/2),-.56*u,np.zeros_like(u)]
    if mode=='draped':
        points[:,:2]=root[:2]+np.c_[-.60*u*u,-.22*u*(1-u)]
        distance=np.linalg.norm(points[:,:2]-root[:2],axis=1)
        points[:,2]=table_height+.002+(root[2]-table_height-.002)*np.exp(-distance/.065)
    return points


def exchange_substeps(hz):
    if hz<=0 or 4800%hz:
        raise ValueError('exchange Hz must divide4800 to preserve internal rod timestep')
    return 4800//hz


def termination_force(cable_position,cable_velocity,exit_position,exit_velocity,stiffness,damping):
    values=[np.asarray(v,dtype=float) for v in (cable_position,cable_velocity,exit_position,exit_velocity)]
    if any(v.shape!=(3,) or not np.isfinite(v).all() for v in values) or not np.isfinite(stiffness) or not np.isfinite(damping) or stiffness<=0 or damping<0:
        raise ValueError('finite world3-vectors, positive stiffness and nonnegative damping required')
    displacement=values[2]-values[0];relative_velocity=values[3]-values[1]
    length=float(np.linalg.norm(displacement))
    direction=displacement/length if length>1e-12 else np.zeros(3)
    radial_velocity=float(relative_velocity@direction)
    force=stiffness*displacement+damping*radial_velocity*direction
    return force,.5*stiffness*float(displacement@displacement),damping*radial_velocity**2


class CableCoupling:
    def __init__(self, scene, stage, manifest, args, dt):
        import torch
        import ropesim as rs
        from ropesim.isaac_bridge import body_geometry, bake_body, ColliderSpec
        from isaaclab.utils.math import quat_apply
        from pxr import UsdGeom
        self.torch, self.scene, self.dt, self.quat_apply = torch, scene, dt, quat_apply
        self.substeps=exchange_substeps(args.exchange_hz)
        self.compliant=args.termination=='compliant'
        self.termination_k=args.termination_stiffness
        self.termination_c=args.termination_damping
        self.gap_limit=args.max_termination_gap
        self.dissipated=0.;self.peak_gap=0.;self.peak_damper_power=0.;self.damper_power=0.
        torch.cuda.set_stream(torch.cuda.default_stream())
        self.plug, self.robot = scene['plug'], scene['robot']
        self.passive=args.case=='passive'
        self.rigid_bodies=(self.plug,scene['latch'])
        self.rigid_masses=[float(obj.root_physx_view.get_masses()[0,0].item()) for obj in self.rigid_bodies]
        from collections import deque
        self.failure_trace=deque(maxlen=256);self.step_count=0;self.out=args.out
        self.freeflight_samples=0;self.max_freeflight_rigid_residual=0.;self.max_freeflight_total_residual=0.
        self.offset = torch.tensor([manifest['cable_exit_offset']], device=scene.device)
        root = self.attachment().detach().cpu().numpy()[0]
        # A slack, unpinned tail initially lies behind the plug on the imported tabletop.
        u = np.linspace(0, 1, 49)
        table_v,_=body_geometry(stage,'/World/envs/env_0/Table')
        table_matrix=np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(stage.GetPrimAtPath('/World/envs/env_0/Table')))
        table_world=(np.c_[table_v,np.ones(len(table_v))]@table_matrix)[:,:3]
        table_bounds=np.array([table_world.min(0),table_world.max(0)])
        points = initial_cable_points(root,args.cable_initialization,float(table_bounds[1,2]))
        if args.cable_initialization=='draped' and (np.any(points[:,:2]<table_bounds[0,:2]+.02) or np.any(points[:,:2]>table_bounds[1,:2]-.02)):
            raise ValueError('Draped initial cable does not fit inside actual imported table footprint')
        self.rest_lengths = np.linalg.norm(np.diff(points, axis=0), axis=-1)
        density, radius = .025, .002
        self.masses = vertex_masses(points, density)
        self.first_free=0 if self.compliant else 1
        self.free_mass = float(self.masses[self.first_free:].sum())
        pins = np.zeros((1,len(u)), bool); pins[:,0] = not self.compliant
        self.rope = rs.RopeBatch.from_positions(points[None].astype(np.float32), density, 800.0, .001,
                       1/4800, args.rod_iterations, pinned=pins, fix_root_orientation=False, straight_rest=True)
        expected_mass=self.masses.copy()
        if not self.compliant:expected_mass[0]=0.
        inv=np.asarray(self.rope.inv_masses(),dtype=np.float64)[0]
        self.masses=np.divide(1.,inv,out=np.zeros_like(inv),where=inv>0)
        self.native_mass_relative_error=float(np.max(np.abs(self.masses-expected_mass)/np.maximum(expected_mass,1e-15)))
        if self.native_mass_relative_error>1e-3:raise RuntimeError('Native cable masses disagree with half-segment construction')
        self.free_mass=float(self.masses[self.first_free:].sum())
        self.rope.damping = 0.; self.rope.drag = 0.; self.rope.drag_quad = 0.; self.rope.drag_quad_t = 0.
        self.rope.set_gravity(np.array([0,0,-9.81], np.float32))
        self.rope.set_contact(radius, .45, .35)
        self.rope.enable_contact_forces(True)
        self.entries, profiles = [], []
        cache = args.out/'collider-cache'
        cache.mkdir(parents=True, exist_ok=True)
        def add(path, asset=None, body=None, spacing=.006, thickness=0.):
            V,F = body_geometry(stage,path)
            actual_thickness=thickness if thickness else (0. if rs.sdf.is_closed(F) else 2*spacing)
            spec = ColliderSpec(asset or 'static', spacing=spacing, margin=.02, thickness=thickness)
            key = hashlib.sha256(V.tobytes()+F.tobytes()+repr(spec).encode()).hexdigest()
            file = cache/(key+'.npz')
            if file.exists():
                with np.load(file) as data:
                    values,origin,step = data['values'],data['origin'],float(data['spacing'])
            else:
                print('Baking cable collider:',path,'triangles',len(F),'spacing',spacing,'shell',actual_thickness,flush=True)
                grid = bake_body(V,F,spec,path)
                values,origin,step = grid.values,grid.origin,grid.spacing
                np.savez_compressed(file,values=values,origin=origin,spacing=step)
            aperture_probe=None
            if getattr(args,'panel_profile',None) and path.endswith('/PatchPanelPhysical'):
                if actual_thickness!=0.:
                    raise RuntimeError('Panel plate/carriers must remain closed; automatic shell could fill aperture')
                config=args.panel_profile;cx,cy,cz=config['socket_position_m'];face=config['panel_center_m'][1]
                world=np.array([[cx,face+.001,cz],[cx+.009,face+.001,cz],[cx,face+.001,config['panel_center_m'][2]],[cx+.010,face+.004,cz],[cx,face+.004,cz]])
                matrix=np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(stage.GetPrimAtPath(path)))
                frame=matrix.copy();frame[:3,:3]/=np.linalg.norm(frame[:3,:3],axis=1,keepdims=True)
                local=(np.c_[world,np.ones(len(world))]@np.linalg.inv(frame))[:,:3]
                phi,inside=rs.sdf.SdfGrid(values,np.asarray(origin,np.float32),step).sample(local)
                if not inside.all() or phi[0]<=.002 or phi[1]>=0. or phi[2]<=.002 or phi[3]>=0. or phi[4]<=.002:
                    raise RuntimeError('Native panel SDF aperture/web probe failed: '+str(phi))
                aperture_probe=dict(world_points_m=world.tolist(),phi_m=phi.tolist(),labels=['socket_axis_open','interport_web_solid','slot_center_open','carrier_side_solid','carrier_inner_void'],method='actual native trilinear grid interpolation')
            if asset:
                obj=scene[asset]
                if body is None:
                    pos,quat=obj.data.root_link_pos_w,obj.data.root_link_quat_w
                else:
                    pos,quat=obj.data.body_link_pos_w[:,body],obj.data.body_link_quat_w[:,body]
                pos=pos.detach().cpu().numpy()[0]; quat=quat.detach().cpu().numpy()[0,[1,2,3,0]]
            else:
                xf=np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(stage.GetPrimAtPath(path)))
                pos=xf[3,:3].astype(np.float32)
                from pxr import Gf
                q=Gf.Transform(UsdGeom.XformCache().GetLocalToWorldTransform(stage.GetPrimAtPath(path))).GetRotation().GetQuat()
                quat=np.r_[q.GetImaginary(),q.GetReal()].astype(np.float32)
            index=self.rope.add_mesh(values,np.asarray(origin,np.float32),step,pos=np.asarray(pos,np.float32),quat=np.asarray(quat,np.float32))
            self.entries.append(dict(asset=asset,body=body,path=path,pos=pos,quat=quat))
            profiles.append(dict(index=index,path=path,spacing=step,shape=list(values.shape),thickness_m=actual_thickness,
                                 geometry_sha256=key,triangles=len(F),dynamic=bool(asset),closed=bool(rs.sdf.is_closed(F)),aperture_grid_probe=aperture_probe))
            print('Cable collider:',path,values.shape,flush=True)
        for bi,name in enumerate(self.robot.body_names):
            path=self.robot.root_physx_view.link_paths[0][bi]
            try:
                add(path,'robot',bi,spacing=.002 if 'finger' in name else .006)
            except ValueError as e:
                if 'no collision geometry' not in str(e): raise
        prefix='/World/envs/env_0'
        for label,step in [('Table',.008),('Rack',.004)]:
            add(prefix+'/'+label,spacing=step,thickness=.008 if label=='Rack' else 0.)
        from pxr import Usd,UsdGeom
        if getattr(args,'panel_profile',None):
            for group in args.panel_profile['collision_groups']:
                add(prefix+'/Fixtures/'+group['path'],spacing=group['spacing_m'],thickness=group['thickness_m'])
        else:
            for prim in Usd.PrimRange(stage.GetPrimAtPath(prefix+'/Fixtures')):
                if prim.IsA(UsdGeom.Mesh):add(str(prim.GetPath()),spacing=.003)
        add(prefix+'/Socket',spacing=.001)
        if len(self.entries)>32:raise RuntimeError('Exceeded native32mesh limit')
        self.last_root = self.attachment().clone()
        self.reaction=np.zeros(3)
        self.robot_force=torch.zeros((1,len(self.robot.body_names),3),device=scene.device)
        self.robot_torque=torch.zeros_like(self.robot_force)
        self.max_strain=0.; self.max_reaction=0.; self.max_balance_residual=0.; self.mesh_peak=np.zeros(len(self.entries));self.current_phase='initialization';self.phase_peaks={};self.phase_counts={}
        self.info=dict(extension=str(rs._ropesim.__file__),extension_sha256=hashlib.sha256(Path(rs._ropesim.__file__).read_bytes()).hexdigest(),
            vertices=len(u),initialization=args.cable_initialization,table_world_bounds_m=table_bounds.tolist(),natural_rest_shape='straight',quaternion_order='xyzw',rest_length_m=float(self.rest_lengths.sum()),initial_positions_sha256=hashlib.sha256(points.astype(np.float32).tobytes()).hexdigest(),native_mass_relative_error=self.native_mass_relative_error,actual_rigid_masses_kg=self.rigid_masses,linear_density_kg_m=density,free_mass_kg=self.free_mass,zero_mass_attachment_vertex=None if self.compliant else 0,
            termination_model=args.termination,termination_damper_model='central radial; zero damping at coincident endpoints',termination_visual_usd=args.termination_visual_usd,termination_k_N_m=self.termination_k if self.compliant else None,
            termination_c_Ns_m=self.termination_c if self.compliant else None,max_termination_gap_m=self.gap_limit if self.compliant else None,
            pinned_vertices=int(pins.sum()),
            EA_N=800.,EI_Nm2=.001,radius_m=radius,substeps=self.substeps,internal_h_s=dt/self.substeps,iterations=args.rod_iterations,damping=0.,drag=0.,
            permanent_point_attachments=1,far_end_free=True,root_tangent_free=True,colliders=profiles,
            energy_columns=['cable_translational_kinetic_J','rigid_kinetic_J','gravity_J','cable_axial_proxy_J','partial_energy_sum_J','termination_spring_J','termination_dissipated_J','termination_damper_power_W'],
            energy_omissions='rod rotational/bending/shear/contact energies and latch spring; partial sum is not total energy conservation',
            reaction_method=('paired finite spring/damper impulse at dynamic cable endpoint and plug exit; no kinematic cable vertices' if self.compliant else 'free-vertex linear momentum minus measured surface and gravity impulse; explicit one exchange delay'))

    def attachment(self):
        return self.plug.data.root_link_pos_w+self.quat_apply(self.plug.data.root_link_quat_w,self.offset)

    def positions(self):
        return self.torch.as_tensor(self.rope.positions()).detach().cpu().numpy()[0].copy()

    def velocities(self):
        return self.torch.as_tensor(self.rope.velocities()).detach().cpu().numpy()[0].copy()

    def quaternions(self):
        return self.torch.as_tensor(self.rope.quaternions()).detach().cpu().numpy()[0].copy()

    def energy(self):
        """Diagnostic components, not an exact total: omit rod rotational/bending/contact energies."""
        from isaaclab.utils.math import quat_apply_inverse
        torch=self.torch
        p=self.positions();v=torch.as_tensor(self.rope.velocities()).detach().cpu().numpy()[0]
        cable_k=.5*float(np.sum(self.masses[self.first_free:,None]*v[self.first_free:]**2))
        gravity=9.81*float(np.sum(self.masses[self.first_free:]*p[self.first_free:,2]))
        lengths=np.linalg.norm(np.diff(p,axis=0),axis=-1)
        axial=.5*800.*float(np.sum((lengths-self.rest_lengths)**2/self.rest_lengths))
        rigid_k=0.
        for obj in (self.plug,self.scene['latch']):
            m=float(obj.root_physx_view.get_masses()[0,0].item())
            velocity=obj.data.root_com_lin_vel_w
            omega=quat_apply_inverse(obj.data.root_com_quat_w,obj.data.root_com_ang_vel_w)
            inertia=obj.root_physx_view.get_inertias()[0].reshape(3,3).to(omega.device)
            rigid_k+=.5*m*float((velocity**2).sum().item())+.5*float((omega[0]@(inertia@omega[0])).item())
            gravity+=9.81*m*float(obj.data.root_com_pos_w[0,2].item())
        # Components are stored separately so omitted energy cannot masquerade as conservation.
        gap=float(np.linalg.norm(self.attachment().detach().cpu().numpy()[0]-p[0]))
        spring=.5*self.termination_k*gap*gap if self.compliant else 0.
        return np.array([cable_k,rigid_k,gravity,axial,cable_k+rigid_k+gravity+axial+spring,spring,self.dissipated,self.damper_power])

    def before_physics(self):
        torch=self.torch
        self.rigid_momentum0=sum(m*obj.data.root_com_lin_vel_w[0].detach().cpu().numpy() for m,obj in zip(self.rigid_masses,self.rigid_bodies))
        self.v_before=torch.as_tensor(self.rope.velocities()).detach().cpu().numpy()[0].copy()
        if self.compliant:
            p=self.positions()
            exit_position=self.attachment()
            exit_velocity=self.plug.data.root_com_lin_vel_w+torch.cross(self.plug.data.root_com_ang_vel_w,exit_position-self.plug.data.root_com_pos_w,dim=-1)
            force,spring,power=termination_force(p[0],self.v_before[0],exit_position.detach().cpu().numpy()[0],exit_velocity.detach().cpu().numpy()[0],self.termination_k,self.termination_c)
            self.reaction=-force
            self.damper_power=power;self.dissipated+=power*self.dt
            self.peak_damper_power=max(self.peak_damper_power,power)
            velocity=self.v_before.copy();velocity[0]+=force*self.dt/self.masses[0]
            self.rope.set_velocities(velocity[None].astype(np.float32))
        f=torch.tensor(self.reaction,dtype=torch.float32,device=self.scene.device)[None]
        lever=self.attachment()-self.plug.data.root_com_pos_w
        torque=torch.cross(lever,f,dim=-1)
        self.applied_force=f[0].detach().cpu().numpy().copy();self.applied_torque=torque[0].detach().cpu().numpy().copy()
        self.plug.permanent_wrench_composer.set_forces_and_torques(f[:,None],torque[:,None],is_global=True)
        self.robot.permanent_wrench_composer.set_forces_and_torques(self.robot_force,self.robot_torque,is_global=True)

    def after_physics(self):
        from .contracts import reaction_from_momentum
        torch=self.torch
        new_root=self.attachment().clone()
        v0=self.v_before
        pos=[]; quat=[]
        for e in self.entries:
            if e['asset']:
                obj=self.scene[e['asset']]; bi=e['body']
                p,q=(obj.data.root_link_pos_w,obj.data.root_link_quat_w) if bi is None else (obj.data.body_link_pos_w[:,bi],obj.data.body_link_quat_w[:,bi])
                pos.append(p[0]);quat.append(q[0,[1,2,3,0]])
            else:
                pos.append(torch.tensor(e['pos'],dtype=torch.float32,device=self.scene.device))
                quat.append(torch.tensor(e['quat'],dtype=torch.float32,device=self.scene.device))
        self.rope.set_mesh_poses_all(torch.stack(pos)[None],torch.stack(quat)[None])
        if self.compliant:
            self.rope.step_free(self.dt,self.substeps)
        else:
            self.rope.step(self.last_root,(new_root-self.last_root)/self.dt,self.dt,self.substeps,root_pos_end=new_root)
        torch.cuda.synchronize()
        if self.rope.last_error or self.rope.last_warning:
            raise RuntimeError(f'Cable solver flag: error={self.rope.last_error}, warning={self.rope.last_warning}')
        p=self.positions()
        v1=torch.as_tensor(self.rope.velocities()).detach().cpu().numpy()[0].copy()
        q1=self.quaternions()
        contact=self.rope.contact_forces()[0,self.first_free:].sum(axis=0,dtype=np.float64)
        momentum0=(v0[self.first_free:]*self.masses[self.first_free:,None]).sum(0)
        momentum1=(v1[self.first_free:]*self.masses[self.first_free:,None]).sum(0)
        if not self.compliant:
            self.reaction=reaction_from_momentum(momentum0,momentum1,self.free_mass,np.array([0,0,-9.81]),contact,self.dt)
        W=self.rope.mesh_wrenches()[0]
        magnitude=np.linalg.norm(W[:,:3],axis=-1)
        self.mesh_peak=np.maximum(self.mesh_peak,magnitude)
        phase=self.current_phase
        self.phase_peaks[phase]=np.maximum(self.phase_peaks.get(phase,np.zeros(len(self.entries))),magnitude)
        self.phase_counts[phase]=self.phase_counts.get(phase,np.zeros(len(self.entries),dtype=int))+(magnitude>.02)
        self.robot_force.zero_();self.robot_torque.zero_()
        for j,e in enumerate(self.entries):
            if e['asset']=='robot':
                bi=e['body'];F=torch.tensor(W[j,:3],device=self.scene.device)
                T=torch.tensor(W[j,3:],device=self.scene.device)
                lever=self.robot.data.body_com_pos_w[0,bi]-self.robot.data.body_link_pos_w[0,bi]
                self.robot_force[0,bi]=F
                self.robot_torque[0,bi]=T-torch.cross(lever,F,dim=-1)
        strain=float(np.max(np.abs(np.linalg.norm(np.diff(p,axis=0),axis=-1)/self.rest_lengths-1)))
        self.max_strain=max(self.max_strain,strain)
        self.max_reaction=max(self.max_reaction,float(np.linalg.norm(self.reaction)))
        self.last_root=new_root
        gap=float(np.linalg.norm(new_root.detach().cpu().numpy()[0]-p[0]))
        self.peak_gap=max(self.peak_gap,gap)
        self.step_count+=1
        exit_velocity=self.plug.data.root_com_lin_vel_w+torch.cross(self.plug.data.root_com_ang_vel_w,new_root-self.plug.data.root_com_pos_w,dim=-1)
        self.failure_trace.append(dict(t=self.step_count*self.dt,gap=gap,cable_root=p[0].copy(),cable_velocity=v1[0].copy(),exit_position=new_root.detach().cpu().numpy()[0].copy(),exit_velocity=exit_velocity.detach().cpu().numpy()[0].copy(),force=self.applied_force.copy(),phase=self.current_phase))
        self.info.update(max_strain=self.max_strain,max_reaction_N=self.max_reaction,peak_termination_gap_m=self.peak_gap,finite=bool(np.isfinite(p).all() and np.isfinite(v1).all() and np.isfinite(q1).all() and np.isfinite(self.reaction).all()))
        if self.compliant and gap>self.gap_limit:
            np.savez_compressed(self.out/'coupling_failure.npz',**{key:np.asarray([row[key] for row in self.failure_trace]) for key in self.failure_trace[-1]})
            raise RuntimeError(f'Physical termination gap exceeds{self.gap_limit}m: {gap}')
        if not np.isfinite(p).all() or not np.isfinite(v1).all() or not np.isfinite(q1).all() or strain>.25 or np.linalg.norm(self.reaction)>50:
            raise RuntimeError(f'Cable numerical boundary: strain={strain}, reaction={self.reaction}')
        # This identity diagnoses accounting/sign errors only; step sensitivity is independent evidence.
        residual=momentum1-momentum0-self.dt*(self.free_mass*np.array([0,0,-9.81])+contact-self.reaction)
        if self.compliant and self.passive and np.min(p[:,2])>1.03 and float(self.plug.data.root_com_pos_w[0,2].item())>1.03 and np.max(np.abs(W))<1e-4:
            rigid_momentum1=sum(m*obj.data.root_com_lin_vel_w[0].detach().cpu().numpy() for m,obj in zip(self.rigid_masses,self.rigid_bodies))
            rigid_residual=rigid_momentum1-self.rigid_momentum0-self.dt*(sum(self.rigid_masses)*np.array([0,0,-9.81])+self.applied_force)
            self.freeflight_samples+=1
            self.max_freeflight_rigid_residual=max(self.max_freeflight_rigid_residual,float(np.linalg.norm(rigid_residual)))
            self.max_freeflight_total_residual=max(self.max_freeflight_total_residual,float(np.linalg.norm(rigid_residual+residual)))
        self.max_balance_residual=max(self.max_balance_residual,float(np.linalg.norm(residual)))
        self.info.update(max_strain=self.max_strain,max_reaction_N=self.max_reaction,
            momentum_residual_Ns=self.max_balance_residual,
            residual_scope='independent rod momentum balance for compliant model; accounting identity for rigid model',
            peak_termination_gap_m=self.peak_gap,termination_dissipated_J=self.dissipated,peak_termination_damper_power_W=self.peak_damper_power,
            mesh_peak_force_N=self.mesh_peak.tolist(),mesh_phase_peak_force_N={k:v.tolist() for k,v in self.phase_peaks.items()},
            mesh_phase_contact_exchange_counts={k:v.tolist() for k,v in self.phase_counts.items()},contact_count_threshold_N=.02,
            surface_force_readout='native final-iterate force law, not measured integrated projection impulse',freeflight_samples=self.freeflight_samples,
            max_freeflight_rigid_impulse_residual_Ns=self.max_freeflight_rigid_residual,
            max_freeflight_combined_impulse_residual_Ns=self.max_freeflight_total_residual)
