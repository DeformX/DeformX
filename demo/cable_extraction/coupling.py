"""Explicit, force-driven exchange between five DeformX rods and PhysX."""
import hashlib
from pathlib import Path

import numpy as np

from .model import DENSITY, EA, EI, EXIT, RADIUS, initial_positions


SPRING_K = 220.
DAMPER_C = .025
GAP_LIMIT = .008
INTERNAL_HZ = 4800


class BundleCoupling:
    def __init__(self, scene, stage, args, table_z):
        import ropesim as rs
        from ropesim.isaac_bridge import ColliderSpec, bake_body, body_geometry
        import torch
        from isaaclab.utils.math import quat_apply

        self.torch = torch
        self.quat_apply = quat_apply
        self.robot, self.tab = scene['robot'], scene['tab']
        self.device = scene.device
        self.dt = 1. / args.exchange_hz
        self.substeps = INTERNAL_HZ // args.exchange_hz
        self.exit_offset = torch.tensor([EXIT], device=self.device)
        self.initial = initial_positions(self.tab.data.root_link_pos_w[0].cpu().numpy(), table_z)
        self.rest_lengths = np.linalg.norm(np.diff(self.initial, axis=1), axis=-1)
        self.rope = rs.RopeBatch.from_positions(self.initial, DENSITY, EA, EI, 1./INTERNAL_HZ,
                    args.iterations, pinned=np.zeros(self.initial.shape[:2], bool),
                    straight_rest=False, deterministic=True)
        inverse_mass = np.asarray(self.rope.inv_masses())
        if np.any(inverse_mass <= 0):
            raise RuntimeError('All vertices must remain dynamic')
        self.endpoint_inv_mass = float(inverse_mass[0, 0])
        self.rope.set_gravity(np.array([0, 0, -9.81], np.float32))
        self.rope.drag = .10
        self.rope.damping = 0.
        self.rope.set_contact(RADIUS, .48, .36)
        self.rope.enable_contact_forces(True)
        self.rope.set_rope_contact(not args.no_rod_contact, .9*RADIUS, .2*RADIUS, .025,
                                   .32, 1e4, 1.25, len(self.initial))
        self.colliders = []
        ids = []
        cache = args.cache.resolve()
        cache.mkdir(parents=True, exist_ok=True)
        names = self.robot.body_names

        def register(path, pose, quat, spacing, asset):
            verts, tris = body_geometry(stage, path)
            spec = ColliderSpec(asset, spacing=spacing, margin=.02)
            key = hashlib.sha256(verts.tobytes()+tris.tobytes()+repr(spec).encode()).hexdigest()
            file = cache/(key+'.npz')
            if not file.exists():
                print('Baking', path, spacing, len(tris), flush=True)
                g = bake_body(verts, tris, spec, path)
                np.savez_compressed(file, values=g.values, origin=g.origin, spacing=g.spacing)
            with np.load(file) as baked:
                values = baked['values']
                self.rope.add_mesh(values, baked['origin'], float(baked['spacing']),
                                   pos=pose, quat=quat)
            self.colliders.append(dict(path=path, spacing_m=spacing, dimensions=list(values.shape),
                                       triangle_count=len(tris), mesh_key=key, closed=rs.sdf.is_closed(tris)))

        for i, name in enumerate(names):
            path = self.robot.root_physx_view.link_paths[0][i]
            try:
                register(path, self.robot.data.body_link_pos_w[0,i].cpu().numpy(),
                         self.robot.data.body_link_quat_w[0,i,[1,2,3,0]].cpu().numpy(),
                         .002 if 'finger' in name else .006, 'robot')
            except ValueError as error:
                if 'no collision geometry' not in str(error):
                    raise
            else:
                ids.append(i)
        from .model import TABLE_POS
        register('/World/envs/env_0/Table', TABLE_POS, [0,0,0,1], .008, 'static')
        self.robot_ids = ids
        self.table_pos = torch.tensor([TABLE_POS], device=self.device, dtype=torch.float32)
        self.table_quat = torch.tensor([[0,0,0,1]], device=self.device, dtype=torch.float32)
        self.count = len(self.initial)
        self.max_gap = self.max_strain = self.max_force = 0.
        self.mesh_peak = np.zeros((self.count, len(self.colliders)))
        self.endpoint_force = np.zeros(3)
        self.last_wrenches = np.zeros((self.count, len(self.colliders), 6))
        extension = Path(rs._ropesim.__file__)
        self.settings = dict(extension_sha256=hashlib.sha256(extension.read_bytes()).hexdigest(),
            extension_file=extension.name, internal_hz=INTERNAL_HZ, substeps=self.substeps,
            iterations=args.iterations, all_vertices_free=True, cables=self.count,
            vertices_per_cable=self.initial.shape[1], natural_rest_shape='precurved initial geometry',
            rest_lengths_m=self.rest_lengths.sum(axis=1).tolist(), linear_density_kg_m=DENSITY,
            EA_N=EA, EI_Nm2=EI, radius_m=RADIUS, gravity_m_s2=[0,0,-9.81],
            linear_drag=.10, material_damping=0., mesh_mu_static_kinetic=[.48,.36],
            rope_rope=dict(enabled=not args.no_rod_contact, same_workcell_group=self.count,
                radius_m=.9*RADIUS, barrier_m=.2*RADIUS, k_collision=.025,
                mu=.32, k_friction=1e4, detection_scaler=1.25),
            termination=dict(stiffness_N_m=SPRING_K, damping_N_s_m=DAMPER_C,
                model='paired central spring and radial damping at tab exit and free rod vertex',
                max_allowed_gap_m=GAP_LIMIT),
            colliders=self.colliders, mesh_loads='solver surface force estimates, not integrated impulses')

    def snapshot(self):
        return [self.torch.as_tensor(f()).detach().cpu().numpy().copy()
                for f in (self.rope.positions, self.rope.velocities, self.rope.quaternions)]

    def exit(self):
        data = self.tab.data
        return data.root_link_pos_w + self.quat_apply(data.root_link_quat_w, self.exit_offset)

    def before(self):
        torch = self.torch
        data = self.tab.data
        exit_pos = self.exit()
        arm = exit_pos - data.root_com_pos_w
        exit_vel = data.root_com_lin_vel_w + torch.cross(data.root_com_ang_vel_w, arm, dim=-1)
        p = torch.as_tensor(self.rope.positions())[0,0]
        v = torch.as_tensor(self.rope.velocities()).cpu().numpy().copy()
        delta = exit_pos - p
        direction = delta / torch.linalg.vector_norm(delta, dim=-1, keepdim=True).clamp_min(1e-10)
        v_endpoint = torch.as_tensor(v[0,0], device=self.device)
        speed = ((exit_vel - v_endpoint)*direction).sum(dim=-1, keepdim=True)
        force = SPRING_K*delta + DAMPER_C*speed*direction
        self.endpoint_force = force[0].cpu().numpy().copy()
        v[0,0] += self.endpoint_force * (self.dt*self.endpoint_inv_mass)
        self.rope.set_velocities(v)
        self.tab.permanent_wrench_composer.set_forces_and_torques(
            -force[:,None,:], torch.cross(arm, -force, dim=-1)[:,None,:], is_global=True)

    def after(self):
        torch = self.torch
        data = self.robot.data
        pos = torch.cat((data.body_link_pos_w[0,self.robot_ids], self.table_pos))
        q = torch.cat((data.body_link_quat_w[0,self.robot_ids][:,[1,2,3,0]], self.table_quat))
        self.rope.set_mesh_poses_all(pos.expand(self.count,-1,-1).contiguous(),
                                     q.expand(self.count,-1,-1).contiguous())
        self.rope.step_free(self.dt, self.substeps)
        torch.cuda.synchronize()
        if self.rope.last_error or self.rope.last_warning:
            raise RuntimeError(f'DeformX flags error={self.rope.last_error} warning={self.rope.last_warning}')
        wrench = torch.as_tensor(self.rope.mesh_wrenches_view())[:,:len(self.colliders)]
        self.last_wrenches = wrench.cpu().numpy().copy()
        self.mesh_peak = np.maximum(self.mesh_peak, np.linalg.norm(self.last_wrenches[:,:,:3],axis=-1))
        total = wrench[:,:-1,:].sum(dim=0)
        lever = data.body_com_pos_w[0,self.robot_ids] - pos[:-1]
        torque = total[:,3:] - torch.cross(lever, total[:,:3], dim=-1)
        self.robot.permanent_wrench_composer.set_forces_and_torques(
            total[None,:,:3], torque[None], body_ids=self.robot_ids, is_global=True)
        p = torch.as_tensor(self.rope.positions()).cpu().numpy()
        v = torch.as_tensor(self.rope.velocities()).cpu().numpy()
        q = torch.as_tensor(self.rope.quaternions()).cpu().numpy()
        if not all(np.isfinite(a).all() for a in (p, v, q, self.last_wrenches, self.endpoint_force)):
            raise RuntimeError('Nonfinite coupled rod state')
        gap = float(np.linalg.norm(p[0,0]-self.exit()[0].cpu().numpy()))
        strain = float(np.max(np.abs(np.linalg.norm(np.diff(p,axis=1),axis=-1)/self.rest_lengths-1)))
        force = float(np.linalg.norm(self.endpoint_force))
        self.max_gap = max(gap, self.max_gap)
        self.max_strain = max(strain, self.max_strain)
        self.max_force = max(force, self.max_force)
        if gap > GAP_LIMIT or strain > .20 or force > 10.:
            raise RuntimeError(f'Coupling bound exceeded, gap={gap:.6f} strain={strain:.5f} force={force:.5f}')
        return gap


def summarize(coupler):
    return coupler.settings | dict(max_termination_gap_m=coupler.max_gap,
        max_segment_strain=coupler.max_strain, max_termination_force_N=coupler.max_force,
        mesh_force_peak_by_cable_and_collider_N=coupler.mesh_peak.tolist())
