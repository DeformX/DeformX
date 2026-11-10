"""Immutable startup snapshot and fail-closed execution of contact-v2 cases."""
import argparse,json,shutil,traceback
from pathlib import Path
import numpy as np
from .contact_integrity import full_hands,sha,validate


def main():
    a=argparse.ArgumentParser();a.add_argument("--motion",type=Path,required=True);a.add_argument("--out",type=Path,required=True);a.add_argument("--cache",type=Path,required=True);a.add_argument("--substeps",type=int,default=200);a.add_argument("--iterations",type=int,default=24);a.add_argument("--finger-off",action="store_true");a.add_argument("--corridor-spacing",type=float,default=.0015);a.add_argument("--critical-grid-scale",type=float,default=1.);a.add_argument("--criteria",type=Path);v=a.parse_args()
    v.out.mkdir(parents=True,exist_ok=False);source=v.out/"source";source.mkdir()
    hashes={}
    for file in [*Path(__file__).parent.glob("*.py"),Path(__file__).with_name("contact_parameters.json")]:hashes[file.name]=sha(file);shutil.copy2(file,source/file.name)
    execution=dict(complete=False,source_snapshot_timing="before CUDA initialization",source_sha256=hashes,argv={k:str(x) if isinstance(x,Path) else x for k,x in vars(v).items()})
    write=lambda:(v.out/"execution.json").write_text(json.dumps(execution,indent=2)+"\n")
    write()
    try:
        for file in ("profile.json","motion.npz"):shutil.copy2(v.motion/file,v.out/file)
        profile=json.loads((v.out/"profile.json").read_text());integrity=validate(profile)
        if full_hands(profile):
            plan=v.motion/"pose_plan.json"
            if sha(plan)!=profile["contact_motion"]["pose_plan_sha256"]:
                raise ValueError("Full-hand pose plan does not match the prepared motion")
            shutil.copy2(plan,v.out/"pose_plan.json")
        if not profile["contact_motion"]["probe"] and v.criteria is None:raise ValueError("Final runs require frozen acceptance criteria")
        if v.criteria is not None:shutil.copy2(v.criteria,v.out/"criteria.json")
        (v.out/"asset_integrity.json").write_text(json.dumps(integrity,indent=2)+"\n")
        with np.load(v.out/"motion.npz",allow_pickle=False) as stored_motion:
            motion=dict(stored_motion)
        from .contact_physics import execute
        report=execute(profile,motion,v.out,v.cache,v.substeps,v.iterations,not v.finger_off,v.corridor_spacing,v.critical_grid_scale)
        report.update(criteria_sha256=sha(v.out/"criteria.json") if v.criteria else None,profile_sha256=sha(v.out/"profile.json"),motion_sha256=sha(v.out/"motion.npz"),source_sha256=hashes,source_snapshot_timing=execution["source_snapshot_timing"])
        (v.out/"report.json").write_text(json.dumps(report,indent=2)+"\n")
        if not report["complete"]:raise RuntimeError("Contact simulation failed; preserved report and partial trajectory")
        execution["complete"]=True;write();print(json.dumps(report["metrics"],indent=2))
    except BaseException:execution["error"]=traceback.format_exc();write();raise
if __name__=="__main__":main()

