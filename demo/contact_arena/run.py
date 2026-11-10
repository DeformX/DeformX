"""Cable/roller contact benchmark. Units: metres, seconds, kilograms, newtons."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time
import traceback


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def execute(a):
    criteria_path=Path(__file__).with_name('criteria.json')
    criteria=json.loads(criteria_path.read_text())
    report = dict(complete=False, settings=vars(a)|{'out':str(a.out)}, metrics={},
                  host=os.uname().nodename,
                  timing_scope='synchronized guide-pose update + solver step; excludes diagnostics, recording, input upload and rendering',
                  limitations=['Independent worlds; no inter-world rope contact.',
                    'Position-pinned ends, free rotation; guide motion prescribed.',
                    'Numerical/contact checks do not establish physical force accuracy.',
                    'Visual panels are decoration, not additional colliders.'],
                  source_sha256={p.name:sha(p) for p in Path(__file__).parent.glob('*.py')},
                  criteria=criteria, criteria_sha256=sha(criteria_path))
    write(a.out/'report.json', report)
    times=[]; states=[]; guide_states=[]; forces=[]; sample_ms=[]; completed=0
    try:
        import numpy as np
        import torch
        import ropesim as rs
        from ropesim import sdf
        from .model import build, guide_at, clearance, contact_summary, diagnostics, timing, DT, VERTICES, EIS, RADIUS
        torch.cuda.set_stream(torch.cuda.default_stream())
        report.update(gpu_name=torch.cuda.get_device_name(), backend_sha256=sha(rs._ropesim.__file__))
        report['settings'].update(dt_s=DT, vertices_per_world=VERTICES, radius_m=RADIUS,
                    EI_Nm2=EIS.tolist(), EA_N=3000., density_kg_m=.117, gravity_m_s2=[0,0,-9.81])
        s=build(a.worlds); initial=s['positions']; display=min(a.worlds,64)
        b=rs.RopeBatch.from_positions(initial,.117,3000.,float(EIS[0]),DT/a.substeps,
                  a.iterations,pinned=s['pins'],fix_root_orientation=False,straight_rest=True,deterministic=True)
        b.set_bend_stiffnesses(np.repeat(EIS[s['family'],None],VERTICES-1,axis=1))
        b.drag=.15; b.damping=0.; b.set_gravity(np.array([0,0,-9.81],np.float32))
        b.set_rope_contact(False); b.set_contact(RADIUS,.4,.3); b.enable_contact_forces(True)
        grid=sdf.from_function(sdf.sdf_cylinder(.18,axis=1),[-.25,-.25,-.25],[.25,.25,.25],.004)
        if a.contact=='on': b.add_mesh(*grid.as_args(),pos=guide_at(s,0),quat=np.tile(np.array([0,0,0,1],np.float32),(a.worlds,1)))
        settle=round(a.settle/DT); frames=max(round(a.seconds/DT),a.measure_steps)
        guides=np.stack([guide_at(s,max(0,(k-settle)*DT)) for k in range(settle+frames+1)])
        gd=torch.as_tensor(guides,device='cuda'); roots=torch.as_tensor(initial[:,0].copy(),device='cuda')
        vel=torch.zeros_like(roots); quat=torch.zeros((a.worlds,4),device='cuda'); quat[:,3]=1
        counts=np.zeros(a.worlds,np.int64); peak=np.zeros(a.worlds); maxstrain=pinerror=0.; min_gap=1e9; flags=[]
        write(a.out/'report.json',report)
        for k in range(settle+frames):
            torch.cuda.synchronize(); start=time.perf_counter_ns()
            if a.contact=='on': b.set_mesh_poses(0,gd[k+1],quat)
            b.step(roots,vel,DT,a.substeps,root_pos_end=roots)
            torch.cuda.synchronize(); elapsed=(time.perf_counter_ns()-start)*1e-6
            completed=k+1
            p,v,q=[torch.as_tensor(f()).cpu().numpy().copy() for f in (b.positions,b.velocities,b.quaternions)]
            d=diagnostics(p,v,q,initial,initial[:,0])
            if not d['finite']: raise RuntimeError('nonfinite state in one or more worlds')
            maxstrain=max(maxstrain,d['max_strain']); pinerror=max(pinerror,d['max_pin_error_m'])
            if b.last_error or b.last_warning: flags.append(dict(step=k,error=int(b.last_error),warning=int(b.last_warning)))
            if flags or maxstrain>.25 or pinerror>5e-5: raise RuntimeError('numerical limit exceeded')
            wrench=np.asarray(b.mesh_wrenches())[:,0,:].copy() if a.contact=='on' else np.zeros((a.worlds,6))
            if not np.isfinite(wrench).all(): raise RuntimeError('nonfinite guide wrench')
            if k>=settle:
                sample_ms.append(elapsed); magnitude=np.linalg.norm(wrench[:,:3],axis=-1)
                counts+=magnitude>1e-4; peak=np.maximum(peak,magnitude)
                # Sample segments as well as vertices to expose coarse contact leakage.
                samples=np.concatenate([p, .5*(p[:,:-1]+p[:,1:])],axis=1)
                min_gap=min(min_gap,float(clearance(samples,guides[k+1]).min()))
                if (k-settle)%4==0:
                    states.append(p[:display]); times.append((k+1)*DT)
                    guide_states.append(guides[k+1,:display]); forces.append(wrench[:display])
            if k%120==0: print('STEP',k,'strain',maxstrain,flush=True)
        metrics=dict(max_segment_strain=maxstrain,max_pin_error_m=pinerror,minimum_sampled_surface_gap_m=min_gap,
             guide_peak_force_N_range=[float(peak.min()),float(peak.max())],contact=contact_summary(counts,frames),flags=flags)
        report.update(complete=True,metrics=metrics,timing=timing(sample_ms,a.worlds,DT),
            contact_qualified=a.contact=='on' and metrics['contact']['every_world_contacted']
                and metrics['contact']['minimum_contact_fraction']>=criteria['minimum_contact_fraction']
                and min_gap>=criteria['minimum_sampled_surface_gap_m']
                and maxstrain<=criteria['max_segment_strain'] and pinerror<=criteria['max_pin_error_m'],
            displayed_worlds=display,completed_steps=completed,hidden_settling_s=a.settle)
        np.save(a.out/'world_contact_counts.npy',counts)
    except BaseException as error:
        report.update(error=dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc()),completed_steps=completed)
        raise
    finally:
        if states:
            import numpy as np
            np.savez_compressed(a.out/'trajectory.npz',t=times,positions=states,guides=guide_states,
                                guide_wrenches=forces,family=s['family'][:display])
            report['trajectory_sha256']=sha(a.out/'trajectory.npz')
        if sample_ms:
            import numpy as np
            np.save(a.out/'timing_ms.npy',sample_ms)
        write(a.out/'report.json',report)
    return report


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--worlds',type=int,default=64)
    p.add_argument('--seconds',type=float,default=4.)
    p.add_argument('--settle',type=float,default=2.)
    p.add_argument('--measure-steps',type=int,default=120)
    p.add_argument('--substeps',type=int,default=80)
    p.add_argument('--iterations',type=int,default=12)
    p.add_argument('--contact',choices=['on','off'],default='on')
    a=p.parse_args(argv)
    if a.out.exists(): p.error('output already exists; choose a new directory')
    if min(a.worlds,a.substeps,a.iterations)<1 or a.measure_steps<2: p.error('positive counts and at least 2 measured steps required')
    if any(not math.isfinite(x) or x<0 for x in (a.seconds,a.settle)): p.error('finite nonnegative durations required')
    a.out.mkdir(parents=True)
    try: execute(a)
    except ImportError as e: p.exit(1,f'Requires a CUDA-capable DeformX installation (NumPy, PyTorch, ropesim): {e}\n')
    return 0


if __name__=='__main__': raise SystemExit(main())
