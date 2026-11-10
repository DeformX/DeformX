"""Exact-mesh and cache contracts for the contact-v2 instrument/finger scene."""
import hashlib,json
from pathlib import Path
import numpy as np

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def full_hands(profile):
    return profile.get("contact_schema_version", 1) >= 2


def mesh_limit(profile):
    return 64 if full_hands(profile) else 32


def spatial_scale(profile, mesh, critical_scale):
    if full_hands(profile):
        critical=mesh["id"] in ("fret_05","L_index_intermediate")
    else:
        critical=mesh["role"]=="finger" or mesh["id"]=="fret_05"
    return critical_scale if critical else 1.


def specs(profile):
    entries=[]
    for mesh in profile["guitar"]["contact_meshes"]:
        entries.append(dict(mesh,role="instrument",pose_source="guitar"))
    for mesh in profile["robot"]["contact_meshes"]:
        entries.append(dict(mesh,role="finger",pose_source=mesh["link"]))
    pick=profile["pick"]
    entries.append(dict(id="pick",role="pick",pose_source="pick",mesh_path=pick["collision_mesh_npz"],obj_path=pick["mesh_path"],sha256=sha(pick["collision_mesh_npz"]),obj_sha256=pick.get("collision_mesh_sha256",sha(pick["mesh_path"])),spacing_m=.00035,margin_m=.012,friction=.2))
    if len(entries)>mesh_limit(profile):raise ValueError("Native backend collider limit exceeded")
    if full_hands(profile):
        expected={
            f"{side}_{link}" for side in ("L","R")
            for link in ("hand_base_link","thumb_proximal_base","thumb_proximal",
                         "thumb_intermediate","thumb_distal",
                         "index_proximal","index_intermediate","middle_proximal","middle_intermediate",
                         "ring_proximal","ring_intermediate","pinky_proximal","pinky_intermediate")
        }
        hands=profile["robot"]["contact_meshes"]
        if {m["link"] for m in hands}!=expected or any(
            m["side"]!=("left" if m["link"].startswith("L_") else "right") for m in hands
        ):
            raise ValueError("Full-hand manifest must cover all 26 visible links on their recorded side")
    if len({m["id"] for m in entries})!=len(entries):raise ValueError("Duplicate collider id")
    return entries


def is_closed(faces):
    # Pure NumPy here: importing ropesim.sdf would import system Warp before
    # Isaac Kit installs its own compatible Warp in the renderer.
    edges=np.concatenate((faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]))
    _,counts=np.unique(np.sort(edges,axis=1),axis=0,return_counts=True)
    return bool(len(counts) and np.all(counts==2))


def validate(profile):
    import trimesh
    dependencies={}
    for label in ("robot","guitar","pick","stool","floor"):
        asset=profile[label]
        for file in asset["files"]:
            if sha(file["path"])!=file["sha256"]:raise ValueError("Imported dependency changed: "+file["path"])
            dependencies[file["path"]]=file["sha256"]
        if asset["path"] not in dependencies:raise ValueError("Root USD absent from dependency manifest")
    if full_hands(profile):
        robot=profile["robot"]
        coverage=robot["hand_coverage"]
        if sha(coverage["path"])!=coverage["sha256"]:
            raise ValueError("Full-hand coverage inventory changed")
        verified=json.loads(Path(coverage["path"]).read_text())["colliders"]
        fields=("id","link","side","shell_index","shell_count","mesh_path","obj_path",
                "sha256","obj_sha256","visual_prim_paths","frame","link_from_mesh","function")
        projection=lambda meshes:sorted(tuple(m.get(k) for k in fields) for m in meshes)
        if projection(verified)!=projection(robot["contact_meshes"]):
            raise ValueError("Full-hand collider manifest does not match verified render-mesh coverage")
    result=[]
    for m in specs(profile):
        if sha(m["mesh_path"])!=m["sha256"] or sha(m["obj_path"])!=m["obj_sha256"]:raise ValueError("Contact mesh checksum mismatch: "+m["id"])
        d=np.load(m["mesh_path"],allow_pickle=False);v=d["vertices"];f=d["faces"]
        if not np.isfinite(v).all() or v.ndim!=2 or v.shape[1]!=3 or f.ndim!=2 or f.shape[1]!=3 or f.min()<0 or f.max()>=len(v):raise ValueError("Invalid triangle mesh: "+m["id"])
        if not is_closed(f):raise ValueError("Signed mesh is not welded/closed: "+m["id"])
        obj=trimesh.load(m["obj_path"],force="mesh",process=False)
        if v.shape!=obj.vertices.shape or not np.array_equal(f,obj.faces) or not np.allclose(v,obj.vertices,rtol=0,atol=1e-7):raise ValueError("Rendered OBJ/collision NPZ triangle mismatch: "+m["id"])
        result.append(dict(id=m["id"],role=m["role"],mesh_sha256=m["sha256"],obj_sha256=m["obj_sha256"],triangles=len(f),closed=True,exact_shape_match_tolerance_m=1e-7,max_OBJ_NPZ_vertex_difference_m=float(np.max(abs(v-obj.vertices)))))
    return dict(dependencies=dependencies,meshes=result,mesh_count=len(result),backend_mesh_limit=mesh_limit(profile))


def cache_path(directory,mesh,scale=1.):
    spacing=float(mesh["spacing_m"])*scale
    key=hashlib.sha256(json.dumps(dict(mesh=mesh["sha256"],spacing=spacing,margin=mesh["margin_m"],thickness=0.,round=0.),sort_keys=True).encode()).hexdigest()[:24]
    return Path(directory)/(mesh["id"]+"-"+key+".npz")


def grid(profile,mesh,directory,scale=1.,allow_bake=False):
    from ropesim import sdf
    file=cache_path(directory,mesh,scale);meta=file.with_suffix(".json")
    expected=dict(mesh_sha256=mesh["sha256"],obj_sha256=mesh["obj_sha256"],spacing_m=float(mesh["spacing_m"])*scale,margin_m=mesh["margin_m"],thickness_m=0.,round_m=0.,baker_source_sha256=sha(sdf.__file__))
    if file.exists():
        record=json.loads(meta.read_text())
        if any(record.get(k)!=v for k,v in expected.items()) or sha(file)!=record["cache_sha256"]:raise ValueError("Stale mesh SDF cache: "+mesh["id"])
        d=np.load(file,allow_pickle=False);g=sdf.SdfGrid(d["values"],d["origin"],float(d["spacing"]))
        if not np.isfinite(g.values).all():raise ValueError("Nonfinite SDF cache")
        return g,record
    if not allow_bake:raise FileNotFoundError("Missing frozen SDF cache: "+str(file))
    d=np.load(mesh["mesh_path"]);g=sdf.bake_mesh(d["vertices"],d["faces"],expected["spacing_m"],expected["margin_m"])
    file.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(file,values=g.values,origin=g.origin,spacing=g.spacing)
    record=dict(expected,cache_sha256=sha(file),path=str(file.resolve()),shape=list(g.values.shape))
    meta.write_text(json.dumps(record,indent=2)+"\n")
    return g,record


def main():
    import argparse
    a=argparse.ArgumentParser();a.add_argument("--profile",type=Path,required=True);a.add_argument("--cache",type=Path);a.add_argument("--bake",action="store_true");a.add_argument("--out",type=Path,required=True);a.add_argument("--critical-scale",type=float,default=1.);v=a.parse_args()
    p=json.loads(v.profile.read_text());report=validate(p)
    if v.cache:
        report["caches"]={m["id"]:grid(p,m,v.cache,spatial_scale(p,m,v.critical_scale),allow_bake=v.bake)[1] for m in specs(p)}
    v.out.write_text(json.dumps(report,indent=2)+"\n");print(json.dumps(report,indent=2))
if __name__=="__main__":main()

