"""Recorded joint-limit/mimic-valid right plucks and true left-index pad press."""
import argparse,copy,json,hashlib
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from .motion import Kinematics,solve_targets


def ease(x):
    x=float(np.clip(x,0,1));return x*x*(3-2*x)


PARAMETERS=Path(__file__).with_name("contact_parameters.json")


def prepare(asset_profile,out,probe=False,angle=55.,press_z=-.0112,preset="tight_75",stroke="profile"):
    parameters=json.loads(PARAMETERS.read_text());settings=parameters[preset];stroke_cfg=settings["pick"] if stroke=="profile" else parameters["soft_6"]["pick"]
    hz=int(settings["record_hz"]);x0=stroke_cfg["start_offset_m"];travel=stroke_cfg["travel_m"];low=stroke_cfg["engagement_lift_m"];high=stroke_cfg["clear_lift_m"]
    p=copy.deepcopy(asset_profile);k=p["robot"]["kinematics"];p["contact_material"]=settings["material"]
    a=np.deg2rad(12.);R=np.array([[0,0,1],[np.sin(a),np.cos(a),0],[-np.cos(a),np.sin(a),0.]])
    G=np.eye(4);G[:3,:3]=R;G[:3,3]=[.30,-.04,.82];p["guitar"]["placement"]=G.tolist()
    k["preset"]["waist_pitch_joint"]=.40
    k["preset"].update({"R_thumb_proximal_yaw_joint":.2885803,"R_thumb_proximal_pitch_joint":.3807029,"R_index_proximal_joint":1.23085795,"R_middle_proximal_joint":1.,"R_ring_proximal_joint":1.,"R_pinky_proximal_joint":1.,"L_middle_proximal_joint":1.3,"L_ring_proximal_joint":1.3,"L_pinky_proximal_joint":1.3})
    # The hand-mounted visible pick retains its measured v1 thumb/index pinch.
    attach=np.eye(4);attach[:3,:3]=Rotation.from_rotvec(-np.pi/2*R[:,0]).as_matrix().T@R;attach[:3,3]=[.186369,.034169,.035020]
    p["pick"]["attachment_transform"]=attach.tolist()
    finger=next(m for m in p["robot"]["contact_meshes"] if m["id"]=="L_index_intermediate")
    fret=next(m for m in p["guitar"]["contact_meshes"] if m["id"]=="fret_05")
    fret_y=float(np.mean(np.asarray(fret["bounds"])[:,1]))
    pad_y=fret_y+.015
    nut=np.asarray(p["guitar"]["nut_endpoints_local_m"]);bridge=np.asarray(p["guitar"]["bridge_endpoints_local_m"])
    def line(y,index=0):
        u=(y-bridge[index,1])/(nut[index,1]-bridge[index,1]);return bridge[index]+u*(nut[index]-bridge[index])
    pad_open=line(pad_y);pick_open=line(.05)
    duration=2. if probe else 6.5;tc=np.arange(round(duration*120)+1)/120
    def left_target(t):
        if probe:
            progress=ease((t-.25)/.45) if t<1.4 else 1-ease((t-1.4)/.45)
            phase="OPEN" if t<.25 else ("PRESS" if t<.70 else ("HOLD" if t<1.4 else ("RELEASE" if t<1.85 else "OPEN")))
        else:
            progress=ease((t-1.6)/.6) if t<4.5 else 1-ease((t-4.5)/.6)
            phase="OPEN" if t<1.6 else ("PRESS" if t<2.2 else ("HOLD" if t<4.5 else ("RELEASE" if t<5.1 else "OPEN")))
        point=pad_open.copy();point[2]=(pad_open[2]+.012)*(1-progress)+press_z*progress
        return (G@np.r_[point,1])[:3],phase
    def pick_target(t):
        point=pick_open.copy();point[0]+=x0;point[2]+=high
        if not probe:
            for start in (.65,3.15,5.65):
                u=t-start
                if 0<=u<.03:point[2]=pick_open[2]+high+(low-high)*ease(u/.03)
                elif .03<=u<.09:point[0]=pick_open[0]+x0+travel*ease((u-.03)/.06);point[2]=pick_open[2]+low
                elif .09<=u<.12:point[0]=pick_open[0]+x0+travel;point[2]=pick_open[2]+low+(high-low)*ease((u-.09)/.03)
                elif .12<=u<.20:point[0]=pick_open[0]+x0+travel*(1-ease((u-.12)/.08))
        return (G@np.r_[point,1])[:3]
    right=copy.deepcopy(k);right_names=[n for n in right["control_joints"] if n.startswith("right_")]
    right["control_joints"]=right_names;right["tcp_link"]="right_wrist_yaw_link";right["tcp_transform"]=attach.tolist()
    right["seed"]=[-.77173789,-.51949241,.56113222,.83847953,1.03578818,-.74639063,1.56337852]
    rt=solve_targets(right,[pick_target(t) for t in tc],np.repeat(R[None],len(tc),0),tolerance=.001)
    left=copy.deepcopy(k);left_names=[n.replace("right_","left_") for n in right_names]+["L_index_proximal_joint"]
    left["control_joints"]=left_names;left["tcp_link"]="L_index_intermediate"
    tool=np.eye(4);tool[:3,3]=finger["pad_point_link_local_m"];left["tcp_transform"]=tool.tolist()
    left["seed"]=[-3.03744643,.73346391,2.12489041,.43670659,-1.85998167,.58286300,-.91371609,.64283106]
    left_rotation=R@Rotation.from_euler("z",angle,degrees=True).as_matrix()
    lt=solve_targets(left,[left_target(t)[0] for t in tc],np.repeat(left_rotation[None],len(tc),0),tolerance=.0005)
    names=right_names+left_names;k["control_joints"]=names;k["seed"]=np.r_[rt["joints"][0],lt["joints"][0]].tolist();k["tcp_link"]="right_wrist_yaw_link";k["tcp_transform"]=attach.tolist()
    times=np.arange(round(duration*hz)+1)/hz;keyq=np.c_[rt["joints"],lt["joints"]];q=np.stack([np.interp(times,tc,keyq[:,j]) for j in range(len(names))],axis=1)
    kin=Kinematics(k);link_names=list(kin.forward(q[0])[0]);links=[];picks=[];pads=[];finger_poses=[]
    for value in q:
        frames,pick=kin.forward(value);links.append([frames[n] for n in link_names]);picks.append(pick);pads.append((frames[finger["link"]]@tool)[:3,3])
        finger_poses.append([frames[m["link"]]@np.asarray(m["link_from_mesh"]) for m in p["robot"]["contact_meshes"]])
    p["contact_motion"]=dict(version=2,probe=probe,duration_s=duration,record_hz=hz,IK_keyframe_hz=120,parameter_preset=preset,parameter_profile_sha256=hashlib.sha256(PARAMETERS.read_bytes()).hexdigest(),pick_stroke=stroke_cfg,selected_string_index=0,fret_id="fret_05",fret_y_local_m=fret_y,pad_y_local_m=pad_y,pad_target_z_local_m=press_z,left_pad_angle_deg=angle,position_pin_indices=[0,-1],endpoint_orientation_free=True,
        max_left_keyframe_position_error_m=float(lt["ik_error"].max()),max_left_keyframe_orientation_error_rad=float(lt["orientation_error"].max()),max_right_keyframe_position_error_m=float(rt["ik_error"].max()),max_right_keyframe_orientation_error_rad=float(rt["orientation_error"].max()))
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    import trimesh
    mesh=trimesh.load(p["pick"]["mesh_path"],force="mesh",process=False);file=out/"pick_collision.npz";np.savez_compressed(file,vertices=np.asarray(mesh.vertices,np.float32),faces=np.asarray(mesh.faces,np.int32));p["pick"]["collision_mesh_npz"]=str(file.resolve())
    (out/"profile.json").write_text(json.dumps(p,indent=2)+"\n")
    np.savez_compressed(out/"motion.npz",t=times,joints=q,joint_names=np.asarray(names),link_names=np.asarray(link_names),link_transforms=links,pick_transforms=picks,finger_transforms=finger_poses,pad_world=pads,phase=np.asarray([left_target(t)[1] for t in times]))
    print(json.dumps(p["contact_motion"],indent=2))


def same_stroke_probe(source,out,seconds=2.8):
    """Keep the soft run's exact FK samples and cadence; change only material EA."""
    source=Path(source);out=Path(out);out.mkdir(parents=True,exist_ok=False)
    p=json.loads((source/"profile.json").read_text());m=np.load(source/"motion.npz",allow_pickle=False);hz=p["contact_motion"]["record_hz"];n=round(seconds*hz)+1
    settings=json.loads(PARAMETERS.read_text())["tight_75"];p["contact_material"]=settings["material"]
    p["contact_motion"].update(probe=True,duration_s=float(m["t"][n-1]),parameter_preset="tight_75_exact_soft_stroke_probe",parameter_profile_sha256=hashlib.sha256(PARAMETERS.read_bytes()).hexdigest(),same_stroke_source_motion_sha256=hashlib.sha256((source/"motion.npz").read_bytes()).hexdigest())
    (out/"profile.json").write_text(json.dumps(p,indent=2)+"\n")
    arrays={k:(v[:n] if v.ndim and v.shape[0]==len(m["t"]) else v) for k,v in m.items()}
    np.savez_compressed(out/"motion.npz",**arrays)


def main():
    a=argparse.ArgumentParser();a.add_argument("--profile",type=Path);a.add_argument("--same-stroke-source",type=Path);a.add_argument("--out",type=Path,required=True);a.add_argument("--probe",action="store_true");a.add_argument("--angle",type=float,default=55.);a.add_argument("--press-z",type=float,default=-.0112);a.add_argument("--preset",choices=("soft_6","tight_75"),default="tight_75");a.add_argument("--stroke",choices=("profile","soft"),default="profile");v=a.parse_args()
    if v.same_stroke_source:same_stroke_probe(v.same_stroke_source,v.out)
    elif v.profile:prepare(json.loads(v.profile.read_text()),v.out,v.probe,v.angle,v.press_z,v.preset,v.stroke)
    else:a.error("--profile or --same-stroke-source required")
if __name__=="__main__":main()
