"""Dynamic position-pinned strings against exact fixed guitar and recorded finger/pick meshes."""
import json,time,socket
from pathlib import Path
import numpy as np
from .contact_integrity import full_hands,spatial_scale,specs,grid,sha,validate
from .contact_wide_path import is_wide
from .contact_strum_path import is_strum
from .contact_loop_strum_path import is_loop_strum


MATERIAL=dict(density_kg_m=.005,EA_N=600.,EI_Nm2=1e-7,radius_m=.00055,prestress_strain=.01,drag_per_s=.05)

def pick_sampling(profile):
    if is_loop_strum(profile):
        return profile["contact_motion"]["loop_strum"]
    if is_strum(profile):
        return profile["contact_motion"]["strum_pick"]
    if is_wide(profile):
        return profile["contact_motion"]["wide_pick"]
    return None


def samples(profile,corridor_spacing=.0015):
    nut=np.asarray(profile["guitar"]["nut_endpoints_local_m"]);bridge=np.asarray(profile["guitar"]["bridge_endpoints_local_m"])
    fy=profile["contact_motion"]["fret_y_local_m"];lo=fy-.012;hi=fy+.039
    base=np.linspace(float(bridge[:,1].mean()),float(nut[:,1].mean()),97)
    base=base[(base<lo-.001)|(base>hi+.001)]
    corridor=fy+np.arange(-round(.012/corridor_spacing),round(.039/corridor_spacing)+1)*corridor_spacing
    pad_y=profile["contact_motion"]["pad_y_local_m"]
    if full_hands(profile):
        # Preserve the real crown and pad stations without making a near-zero
        # rest segment when the pad lies almost exactly on a regular sample.
        corridor=corridor[abs(corridor-pad_y)>=.5*corridor_spacing]
    pick_nodes=[]
    pick=pick_sampling(profile)
    if pick is not None:
        station=pick["station_y_m"]
        count=round(pick["pick_sampling_halfwidth_m"]/corridor_spacing)
        pick_nodes=station+np.arange(-count,count+1)*corridor_spacing
        base=base[(base<pick_nodes[0]-.5*corridor_spacing)|(base>pick_nodes[-1]+.5*corridor_spacing)]
    y=np.unique(np.r_[base,corridor,fy,pad_y,pick_nodes])[::-1]
    # Remove sub-micron duplicates introduced by floating arithmetic.
    y=y[np.r_[True,np.abs(np.diff(y))>1e-6]]
    u=(nut[0,1]-y)/(nut[0,1]-bridge[0,1]);u[0]=0.;u[-1]=1.
    final=nut[:,None]+u[None,:,None]*(bridge-nut)[:,None]
    return final.astype(np.float32),u.astype(np.float64)


def left_control_mesh(profile,mesh):
    return mesh["role"]=="finger" and (not full_hands(profile) or mesh["side"]=="left")


def transforms(profile,motion,index,entries,finger_enabled=True):
    G=np.asarray(profile["guitar"]["placement"]);poses=[];fi=0
    for m in entries:
        if m["role"]=="instrument":pose=G.copy()
        elif m["role"]=="finger":
            pose=motion["finger_transforms"][index,fi].copy();fi+=1
            if not finger_enabled and left_control_mesh(profile,m):pose[:3,3]+=[100,0,0]
        else:pose=motion["pick_transforms"][index].copy()
        poses.append(pose)
    return np.asarray(poses)


def gaps(points,poses,grids,radius=.00055):
    output=[]
    for pose,g in zip(poses,grids):
        local=(points-pose[:3,3])@pose[:3,:3];flat=local.reshape(-1,3)
        phi,inside=g.sample(flat);phi=np.where(inside,phi-radius,np.inf)
        output.append(phi.reshape(points.shape[:2]).min(-1))
    return np.stack(output,axis=-1)


def execute(profile,motion,out,cache,substeps=200,iterations=24,finger_enabled=True,corridor_spacing=.0015,grid_scale=1.):
    import torch,ropesim as rs
    from scipy.spatial.transform import Rotation
    validate(profile);entries=specs(profile)
    material={**MATERIAL,**profile.get("contact_material",{})}
    if set(material)!=set(MATERIAL) or not all(np.isfinite(v) and v>0 for v in material.values()):raise ValueError("Invalid immutable string material profile")
    grids=[];bindings=[]
    for m in entries:
        g,binding=grid(profile,m,cache,spatial_scale(profile,m,grid_scale),allow_bake=False);grids.append(g);bindings.append(binding)
    torch.cuda.set_stream(torch.cuda.default_stream())
    times=motion["t"];dt=float(times[1]-times[0]);local,u=samples(profile,corridor_spacing);G=np.asarray(profile["guitar"]["placement"])
    final=local@G[:3,:3].T+G[:3,3];nut=final[:,0].copy();bridge=final[:,-1].copy()
    root_initial=nut+(bridge-nut)*(material["prestress_strain"]/(1+material["prestress_strain"]))
    initial=root_initial[:,None]+u[None,:,None]*(bridge-root_initial)[:,None]
    initial=initial.astype(np.float32);pins=np.zeros(initial.shape[:2],bool);pins[:,[0,-1]]=True
    rest=np.linalg.norm(np.diff(initial,axis=1),axis=-1)
    deterministic=full_hands(profile)
    sweep_options={"deterministic":True} if deterministic else {}
    b=rs.RopeBatch.from_positions(initial,material["density_kg_m"],material["EA_N"],material["EI_Nm2"],dt/substeps,iterations,pinned=pins,fix_root_orientation=False,straight_rest=True,**sweep_options)
    b.damping=0.;b.drag=20.;b.set_gravity(np.array([0,0,-9.81],np.float32));b.set_contact(material["radius_m"],.25,.20);b.enable_contact_forces(True)
    first=transforms(profile,motion,0,entries)
    mesh_manifest=[]
    for i,(m,g,binding,pose) in enumerate(zip(entries,grids,bindings,first)):
        quat=Rotation.from_matrix(pose[:3,:3]).as_quat().astype(np.float32)
        actual=b.add_mesh(*g.as_args(),pos=pose[:3,3].astype(np.float32),quat=quat,mu_s=float(m["friction"]),mu_k=.8*float(m["friction"]))
        if actual!=i:raise RuntimeError("Backend collider index changed")
        mesh_manifest.append(dict(m,index=actual,cache=binding))
    dev=lambda x:torch.as_tensor(np.ascontiguousarray(x,np.float32),device="cuda")
    read=lambda f:torch.as_tensor(f()).cpu().numpy().copy()
    def smooth(x):x=float(np.clip(x,0,1));return x*x*(3-2*x)
    root_at=lambda t:root_initial+(nut-root_initial)*smooth(t/.7)
    start=time.perf_counter()
    for i in range(round(2./dt)):
        r0=root_at(i*dt);r1=root_at((i+1)*dt)
        b.step(dev(r0),dev((r1-r0)/dt),dt,substeps,root_pos_end=dev(r1))
    torch.cuda.synchronize();settle_wall=time.perf_counter()-start
    baseline=read(b.positions);baseline_v=read(b.velocities);baseline_q=read(b.quaternions)
    tension=material["EA_N"]*(np.linalg.norm(np.diff(baseline,axis=1),axis=-1)/rest-1)
    chord=bridge-nut;axis=chord/np.linalg.norm(chord,axis=-1,keepdims=True)
    offset=baseline-nut[:,None];transverse=offset-(offset*axis[:,None]).sum(-1)[...,None]*axis[:,None]
    sag=np.linalg.norm(transverse,axis=-1).max(1)
    native_inv_mass=np.asarray(b.inv_masses(),dtype=np.float64);masses=np.divide(1.,native_inv_mass,out=np.zeros_like(native_inv_mass),where=native_inv_mass>0)
    baseline_force=np.array(b.contact_forces(),copy=True);baseline_wrench=np.array(b.mesh_wrenches(),copy=True)
    if not finger_enabled:
        parked=transforms(profile,motion,0,entries,False)
        for mi,(m,pose) in enumerate(zip(entries,parked)):
            if left_control_mesh(profile,m):b.set_mesh_poses(mi,pose[:3,3].astype(np.float32),Rotation.from_matrix(pose[:3,:3]).as_quat().astype(np.float32),teleport=True)
    b.drag=material["drag_per_s"]
    ps=[];vs=[];qs=[];forces=[];wrenches=[];poses=[];vertex_gaps=[];latencies=[];record_times=[];flags=[];maxstrain=0.;maxpin=0.;finite=True
    start=time.perf_counter()
    for i,t in enumerate(times):
        actual_poses=transforms(profile,motion,i,entries,finger_enabled)
        if i:
            for mi,(m,pose) in enumerate(zip(entries,actual_poses)):
                if m["role"]!="instrument":b.set_mesh_poses(mi,pose[:3,3].astype(np.float32),Rotation.from_matrix(pose[:3,:3]).as_quat().astype(np.float32))
            torch.cuda.synchronize();tick=time.perf_counter()
            b.step(dev(nut),dev(np.zeros_like(nut)),dt,substeps,root_pos_end=dev(nut))
            torch.cuda.synchronize();latencies.append(time.perf_counter()-tick)
        p=read(b.positions);v=read(b.velocities)
        q=read(b.quaternions) if deterministic else None
        finite=bool(np.isfinite(p).all() and np.isfinite(v).all() and (q is None or np.isfinite(q).all()))
        if not finite:flags.append(dict(t=float(t),nonfinite=True));break
        strain=float(abs(np.linalg.norm(np.diff(p,axis=1),axis=-1)/rest-1).max());maxstrain=max(maxstrain,strain)
        maxpin=max(maxpin,float(np.linalg.norm(p[:,0]-nut,axis=-1).max()),float(np.linalg.norm(p[:,-1]-bridge,axis=-1).max()))
        if b.last_error or b.last_warning:flags.append(dict(t=float(t),error=int(b.last_error),warning=int(b.last_warning)))
        ps.append(p);vs.append(v);forces.append(np.array(b.contact_forces(),copy=True));wrenches.append(np.array(b.mesh_wrenches(),copy=True));poses.append(actual_poses);record_times.append(t)
        if q is not None:qs.append(q)
        vertex_gaps.append(gaps(p,actual_poses,grids))
        if flags or strain>.05:break
        if i and i%240==0:print(f"CONTACT t={t:.2f}s strain={maxstrain:.5f}",flush=True)
    elapsed=time.perf_counter()-start
    ps=np.asarray(ps);vs=np.asarray(vs);forces=np.asarray(forces);wrenches=np.asarray(wrenches)
    report=dict(complete=finite and not flags and len(record_times)==len(times) and maxstrain<=.05 and maxpin<=50e-6,host=socket.gethostname(),finger_enabled=finger_enabled,instrument_enabled=True,material=material,settings=dict(substeps=substeps,iterations=iterations,dt_s=dt,record_hz=1/dt,corridor_spacing_m=corridor_spacing,grid_scale=grid_scale,vertices_per_string=len(u),strings=6,pinned_vertices=int(pins.sum()),only_endpoint_positions_pinned=True,fix_root_orientation=False,settling_physical_s=2.,main_physical_s=float(record_times[-1]),requested_main_physical_s=float(times[-1])),
        metrics=dict(finite=finite,flags=flags,max_segment_strain=maxstrain,max_endpoint_error_m=maxpin,settled_max_velocity_m_s=float(np.linalg.norm(baseline_v,axis=-1).max()),initial_contact_force_N=float(np.linalg.norm(baseline_force,axis=-1).max()),settled_tension_proxy_N_per_string_percentiles=np.percentile(tension,[0,5,50,95,100],axis=1).T.tolist(),settled_transverse_sag_m_per_string=sag.tolist()),mesh_manifest=mesh_manifest,
        reaction_definition="mesh_wrenches[string,actual_mesh_index,:] is outer-step-averaged force/torque BY each string ON each mesh; origin is the mesh frame origin. No sum across strings for fret/finger attribution.",
        main_loop_wall_s=elapsed,settling_wall_s=settle_wall,mean_step_ms=float(np.mean(latencies))*1000 if latencies else None,extension_sha256=sha(rs._ropesim.__file__),extension=str(rs._ropesim.__file__),assumptions="Fixed instrument and pelvis, prescribed robot joints, independent strings; genuine finger/pick/instrument surface contact; no interior bindings, no humanoid balance or acoustic calibration.")
    if deterministic:
        report["settings"]["deterministic_sweeps"]=True
        report["settings"]["pad_node_replaces_nearby_corridor_node"]=True
        report["metrics"]["min_rest_segment_length_m"]=float(rest.min())
        pick=pick_sampling(profile)
        if pick is not None:
            report["settings"]["pick_sampling_halfwidth_m"]=pick["pick_sampling_halfwidth_m"]
        report["finger_control_scope"]="all left-hand meshes parked; right hand, pick and instrument remain active"
    out=Path(out)
    rotation_history={"quaternions":qs} if deterministic else {}
    np.savez_compressed(out/"trajectory.npz",t=record_times,positions=ps,velocities=vs,surface_forces=forces,mesh_wrenches=wrenches,collider_transforms=poses,vertex_gaps=vertex_gaps,baseline=baseline,baseline_velocities=baseline_v,baseline_quaternions=baseline_q,baseline_surface_forces=baseline_force,baseline_mesh_wrenches=baseline_wrench,rest_lengths=rest,settled_tension_proxy_N=tension,vertex_masses=masses,pinned=pins,sample_u=u,pick_transforms=motion["pick_transforms"][:len(record_times)],finger_transforms=motion["finger_transforms"][:len(record_times)],phase=motion["phase"][:len(record_times)],**rotation_history)
    report["trajectory_sha256"]=sha(out/"trajectory.npz")
    return report

