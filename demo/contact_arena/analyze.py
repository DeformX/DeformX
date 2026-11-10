"""Verify saved runs and compare contact, refinement and scaling on recorded data."""
import argparse
import json
from pathlib import Path
from .run import sha, write


def matching_workload(reference,other,allowed=()):
    """Reject comparisons that silently change material, geometry or runtime."""
    omit={'out',*allowed}
    left={k:v for k,v in reference['settings'].items() if k not in omit}
    right={k:v for k,v in other['settings'].items() if k not in omit}
    if left!=right: raise ValueError('Comparison workload/settings differ')
    for key in ('backend_sha256','criteria_sha256','gpu_name','host'):
        if reference[key]!=other[key]: raise ValueError('Comparison '+key+' differs')
    for name in ('run.py','model.py'):
        if reference['source_sha256'][name]!=other['source_sha256'][name]: raise ValueError('Comparison simulation source differs')


def load(path):
    import numpy as np
    report=json.loads((path/'report.json').read_text())
    if not report['complete'] or report['trajectory_sha256']!=sha(path/'trajectory.npz'):
        raise ValueError('Incomplete or modified trajectory: '+str(path))
    data=np.load(path/'trajectory.npz')
    if not all(np.isfinite(data[k]).all() for k in data.files): raise ValueError('nonfinite recording')
    from .model import contact_summary
    counts=np.load(path/'world_contact_counts.npy')
    steps=len(np.load(path/'timing_ms.npy'))
    if counts.shape!=(report['settings']['worlds'],) or np.any(counts<0) or np.any(counts>steps) or contact_summary(counts,steps)!=report['metrics']['contact']:
        raise ValueError('Per-world contact evidence mismatch')
    if report['settings']['contact']=='off' and (np.any(counts) or np.any(data['guide_wrenches'])):
        raise ValueError('Contact-off recording has guide contact evidence')
    return report,data


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('nominal','control','refined','out'): p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--benchmarks',type=Path,nargs='+',required=True)
    p.add_argument('--off-benchmarks',type=Path,nargs='+',required=True); a=p.parse_args(argv)
    if a.out.exists(): p.error('output already exists')
    import numpy as np
    from .model import timing
    runs=[load(x) for x in (a.nominal,a.control,a.refined)]
    r,d=runs[0]; criteria=r['criteria']
    matching_workload(r,runs[1][0],('contact',))
    matching_workload(r,runs[2][0],('substeps',))
    if r['settings']['contact']!='on' or runs[1][0]['settings']['contact']!='off':
        raise ValueError('Need a contact-on nominal and contact-off control')
    if runs[2][0]['settings']['substeps']<=r['settings']['substeps']:
        raise ValueError('Refinement must use more substeps')
    for other,arrays in runs[1:]:
        if not np.array_equal(d['t'],arrays['t']) or d['positions'].shape!=arrays['positions'].shape:
            raise ValueError('Comparison needs equal clocks and layouts')
        if r['criteria_sha256']!=other['criteria_sha256']: raise ValueError('Different frozen criteria')
    control_rms=float(np.sqrt(np.mean(np.sum((d['positions']-runs[1][1]['positions'])**2,axis=-1))))
    refined_rms=float(np.sqrt(np.mean(np.sum((d['positions']-runs[2][1]['positions'])**2,axis=-1))))
    grouped={}; off_grouped={}; evidence=[]
    all_benchmarks=a.benchmarks+a.off_benchmarks
    if len({path.resolve() for path in all_benchmarks})!=len(all_benchmarks):
        raise ValueError('Each benchmark repetition needs a distinct recording')
    for path in all_benchmarks:
        report,_=load(path)
        off=path in a.off_benchmarks
        matching_workload(r,report,('worlds','contact') if off else ('worlds',))
        if (not off and not report['contact_qualified']) or (off and report['settings']['contact']!='off'):
            raise ValueError('Unexpected contact workload')
        n=report['settings']['worlds']; target=off_grouped if off else grouped
        target.setdefault(n,[]).append(np.load(path/'timing_ms.npy'))
        evidence.append(dict(run_name=path.name,contact=report['settings']['contact'],worlds=n,
            report_sha256=sha(path/'report.json'),trajectory_sha256=report['trajectory_sha256'],
            timing_sha256=sha(path/'timing_ms.npy'),world_counts_sha256=sha(path/'world_contact_counts.npy'),metrics=report['metrics']))
    scaling=[dict(worlds=n,repetitions=len(samples),**timing(np.concatenate(samples),n)) for n,samples in sorted(grouped.items())]
    off_scaling=[dict(worlds=n,repetitions=len(samples),**timing(np.concatenate(samples),n)) for n,samples in sorted(off_grouped.items())]
    checks=dict(nominal_contact=r['contact_qualified'],refined_contact=runs[2][0]['contact_qualified'],
        contact_control_changes_geometry=control_rms>=criteria['minimum_control_position_rms_m'],
        refinement_position_difference=refined_rms<=criteria['max_refinement_position_rms_m'],
        repeated_benchmarks=all(row['repetitions']>=2 for row in scaling+off_scaling),
        matched_contact_counts=grouped.keys()==off_grouped.keys())
    summary=dict(complete=all(checks.values()),checks=checks,criteria=criteria,criteria_sha256=r['criteria_sha256'],
        analysis_source_sha256=sha(__file__),
        control_position_rms_m=control_rms,refinement_position_rms_m=refined_rms,scaling=scaling,contact_off_scaling=off_scaling,
        position_rms_definition='sqrt(mean(sum_xyz((p_a-p_b)^2))) over stored times, displayed worlds and vertices',
        shared_diagnostics_sha256=sha(Path(__file__).parents[1]/'batch_arena/model.py'),
        nominal_metrics=r['metrics'],refined_metrics=runs[2][0]['metrics'],benchmark_evidence=evidence,
        comparison_report_sha256={x.name:sha(x/'report.json') for x in (a.nominal,a.control,a.refined)},
        limits='Saved-state geometry and demo contact only; force convergence and experimental accuracy not established.')
    a.out.mkdir(parents=True); write(a.out/'summary.json',summary)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(13,3.8),layout='constrained')
    for (_,data),label,color in zip(runs,['Contact on','Contact off','160 substeps'],['#139a92','#cc6776','#405a9c']):
        pos=data['positions'][-1,0]; axes[0].plot(pos[:,0],pos[:,2],label=label,color=color)
        force=np.linalg.norm(data['guide_wrenches'][:,0,:3],axis=-1)
        axes[1].plot(data['t'],force,label=label,color=color)
    axes[0].set(xlabel='x (m)',ylabel='z (m)',title='Final cable / world 0'); axes[0].legend(fontsize=8)
    axes[1].set(xlabel='Source time (s)',ylabel='Guide force magnitude (N)',title='Solver reaction / world 0')
    axes[2].plot([x['worlds'] for x in scaling],[x['median_ms'] for x in scaling],'o-',color='#139a92',label='Contact on')
    axes[2].plot([x['worlds'] for x in off_scaling],[x['median_ms'] for x in off_scaling],'o--',color='#cc6776',label='Contact off')
    axes[2].legend(fontsize=8)
    axes[2].set(xscale='log',xlabel='Independent worlds',ylabel='Synchronized call (ms)',title='Contact workload / two repeats')
    for axis in axes: axis.grid(alpha=.2)
    fig.savefig(a.out/'comparison.png',dpi=160); plt.close(fig)
    print(json.dumps(summary,indent=2))
    return 0 if summary['complete'] else 1


if __name__=='__main__': raise SystemExit(main())
