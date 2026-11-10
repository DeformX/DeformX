"""Record the seated underside fret pose and shallow extended-tip pick stroke."""
import argparse
import copy
import json
from pathlib import Path

import numpy as np

from .contact_integrity import sha, validate
from .contact_loop_strum_path import LOOP_PRESET, LOOP_PRESETS, loop_contract, loop_strum_path
from .contact_motion import PARAMETERS, ease
from .contact_strum_path import STRUM_DURATION, strum_path
from .contact_wide_path import WIDE_HOLD_START, WIDE_PRESS_START, WIDE_STARTS, wide_path
from .motion import Kinematics, solve_targets


def press_phase(time, probe, held_pick_probe=False):
    if held_pick_probe:
        start,held,release,clear=.25,.70,2.00,2.45
    else:
        start,held,release,clear=(.25,.70,1.40,1.85) if probe else (1.60,2.20,4.50,5.10)
    if time < start:
        return 0., "OPEN"
    if time < held:
        return ease((time - start) / (held - start)), "PRESS"
    if time < release:
        return 1., "HOLD"
    if time < clear:
        return 1. - ease((time - release) / (clear - release)), "RELEASE"
    return 0., "OPEN"


def pick_motion(time, engage, travel, clearance, starts):
    point = engage + [0, 0, clearance]
    for start in starts:
        u = time - start
        if 0 <= u < .03:
            return engage + [0, 0, clearance * (1. - ease(u / .03))]
        if .03 <= u < .09:
            return engage + [travel * ease((u - .03) / .06), 0, 0]
        if .09 <= u < .12:
            return engage + [travel, 0, clearance * ease((u - .09) / .03)]
        if .12 <= u < .20:
            return point + [travel * (1. - ease((u - .12) / .08)), 0, 0]
    return point


def prepare(asset_path, plan_path, out, probe=False, pick_probe=False, held_pick_probe=False, pick_preset="single", loop_probe=False):
    p = json.loads(asset_path.read_text())
    plan = json.loads(plan_path.read_text())
    parameters = json.loads(PARAMETERS.read_text())["tight_75"]
    if p["contact_schema_version"] != 2 or plan["asset_profile_sha256"] != sha(asset_path):
        raise ValueError("Full-hand pose plan must match the exact v3 asset manifest")
    p["contact_material"] = parameters["material"]
    p["guitar"]["placement"] = plan["guitar_placement"]
    p["pick"]["attachment_transform"] = plan["pick_attachment_transform"]
    G = np.asarray(plan["guitar_placement"])
    k = p["robot"]["kinematics"]
    k["preset"].update(plan["preset"])
    right, left = plan["right"], plan["left"]
    selected = int(left["string_index"])
    reciprocal = pick_preset in LOOP_PRESETS
    held_sixth = pick_preset == "strum5" or reciprocal
    if not 0 <= selected < 6 or (not held_sixth and selected != right["string_index"]):
        raise ValueError("Pick and fretting pad must target the same playable string")
    if held_sixth and selected != 5:
        raise ValueError("The five-string stroke excludes the held sixth string")
    if loop_probe and pick_preset != LOOP_PRESET:
        raise ValueError("The reciprocal probe requires the reciprocal preset")
    right_names = [name for name in k["control_joints"] if name.startswith("right_")]
    left_names = left["joint_names"]
    right_model = dict(copy.deepcopy(k), control_joints=right_names, seed=right["seed"],
                       tcp_link="right_wrist_yaw_link", tcp_transform=plan["pick_attachment_transform"])
    hz = int(parameters["record_hz"])
    short_press=probe or held_pick_probe
    duration = 1.1 if pick_probe else (2.5 if held_pick_probe else (2. if probe else 6.5))
    key_time = np.arange(round(duration * 120) + 1) / 120
    travel = float(right["stroke_x_m"])
    clearance = float(right["clearance_m"])
    if abs(abs(travel) - parameters["pick"]["travel_m"]) > 1e-10 or not 0 < clearance <= .015:
        raise ValueError("Full-hand trajectory must retain the shallow physical pick stroke")
    engage = np.asarray(right["engaged_center_guitar_local_m"])
    starts=(.95,) if held_pick_probe else (() if probe else (.65,3.15,5.65))
    wide=None
    strum=None
    loop=None
    if held_sixth:
        if probe or pick_probe or held_pick_probe:
            raise ValueError("Use the same short complete strum for feasibility and production")
        config=right["wide"]
        if reciprocal:
            loop_version, _, duration, _ = loop_contract(pick_preset, loop_probe)
        else:
            duration = STRUM_DURATION
        key_time=np.arange(round(duration*120)+1)/120
        if reciprocal:
            target,rotation,loop=loop_strum_path(p["guitar"],config,right.get("strum5",{}),loop_probe,pick_preset)
        else:
            target,rotation,strum=strum_path(p["guitar"],config,right.get("strum5",{}))
        pick_targets=np.asarray([target(t) for t in key_time])
        orientation=G[:3,:3]@rotation
        right_model["seed"]=config["seed"]
    elif pick_preset=="wide":
        if pick_probe:
            raise ValueError("The wide preset uses --held-pick-probe for one physical sweep")
        config=right["wide"]
        duration=2.6 if held_pick_probe else (2. if probe else 6.5)
        key_time=np.arange(round(duration*120)+1)/120
        starts=WIDE_STARTS[:1] if held_pick_probe else (() if probe else WIDE_STARTS)
        target,rotation,wide=wide_path(p["guitar"],config,starts)
        pick_targets=np.asarray([target(t) for t in key_time])
        orientation=G[:3,:3]@rotation
        right_model["seed"]=config["seed"]
    else:
        pick_targets = np.asarray([pick_motion(t, engage, travel, clearance, starts) for t in key_time])
        orientation = G[:3, :3] @ np.asarray(right["rotation_guitar_local"])
    rt = solve_targets(right_model, pick_targets @ G[:3, :3].T + G[:3, 3],
                       np.repeat(orientation[None], len(key_time), 0), tolerance=.0003)
    k.update(control_joints=right_names + left_names, tcp_link=right_model["tcp_link"],
             tcp_transform=right_model["tcp_transform"])
    progress_axis = np.asarray(left["progress"])
    left_key = np.asarray(left["joint_keyframes"])
    if len(left_key) != len(progress_axis) or progress_axis[0] != 0 or progress_axis[-1] != 1 or not np.all(np.diff(progress_axis) > 0):
        raise ValueError("Left motion requires ordered open-to-held joint waypoints")
    times = np.arange(round(duration * hz) + 1) / hz
    if wide is None and not held_sixth:
        states=[press_phase(t,short_press,held_pick_probe) for t in times]
    else:
        states=[(ease((t-WIDE_PRESS_START)/(WIDE_HOLD_START-WIDE_PRESS_START)),
                 "OPEN" if t<WIDE_PRESS_START else ("PRESS" if t<WIDE_HOLD_START else "HOLD")) for t in times]
    qright = np.stack([np.interp(times, key_time, rt["joints"][:, j]) for j in range(len(right_names))], axis=1)
    qleft = np.stack([np.interp([s[0] for s in states], progress_axis, left_key[:, j])
                      for j in range(len(left_names))], axis=1)
    joints = np.c_[qright, qleft]
    k["seed"] = joints[0].tolist()
    kin = Kinematics(k)
    link_names = list(kin.forward(joints[0])[0])
    links, picks, pads, fingers = [], [], [], []
    hand_meshes = p["robot"]["contact_meshes"]
    distal = next(mesh for mesh in hand_meshes if mesh["id"] == "L_index_intermediate")
    point = np.asarray(left["fingertip_point_link_local_m"])
    for q in joints:
        frames, pick = kin.forward(q)
        pad_pose = frames[distal["link"]]
        pads.append(point @ pad_pose[:3, :3].T + pad_pose[:3, 3])
        links.append([frames[name] for name in link_names])
        picks.append(pick)
        fingers.append([frames[mesh["link"]] @ np.asarray(mesh["link_from_mesh"]) for mesh in hand_meshes])
    fret = next(m for m in p["guitar"]["contact_meshes"] if m["id"] == "fret_05")
    hold_pad = (np.asarray(pads)[np.argmax([s[0] for s in states])] - G[:3, 3]) @ G[:3, :3]
    p["contact_motion"] = dict(
        version=3, probe=short_press or pick_probe, duration_s=duration, record_hz=hz, IK_keyframe_hz=120,
        parameter_preset="tight_75", parameter_profile_sha256=sha(PARAMETERS),
        pose_plan_sha256=sha(plan_path), asset_profile_sha256=sha(asset_path),
        selected_string_index=selected, fret_id="fret_05", fret_y_local_m=float(np.mean(np.asarray(fret["bounds"])[:, 1])),
        pad_y_local_m=float(hold_pad[1]), pad_target_z_local_m=float(hold_pad[2]),
        fingertip_point_link_local_m=point.tolist(), left_posture="underneath neck, thumb behind neck",
        pick_stroke=dict(travel_m=abs(travel), signed_transverse_travel_m=travel, clear_lift_m=clearance,
                         engage_center_guitar_local_m=engage.tolist(), tip_reference_local_m=right["tip_reference_local_m"]),
        position_pin_indices=[0, -1], endpoint_orientation_free=True,
        max_right_keyframe_position_error_m=float(rt["ik_error"].max()),
        max_right_keyframe_orientation_error_rad=float(rt["orientation_error"].max()))
    if held_pick_probe:
        p["contact_motion"]["held_pick_probe_s"]=starts[0]
    if wide is not None:
        p["contact_motion"].update(version=4,pick_preset="wide",wide_pick=wide)
    if strum is not None:
        p["contact_motion"].update(version=5,pick_preset="strum5",strum_pick=strum)
    if loop is not None:
        p["contact_motion"].update(version=loop_version,pick_preset=pick_preset,loop_strum=loop,probe=loop_probe)
    validate(p)
    out.mkdir(parents=True, exist_ok=False)
    (out / "profile.json").write_text(json.dumps(p, indent=2) + "\n")
    (out / "pose_plan.json").write_bytes(plan_path.read_bytes())
    np.savez_compressed(out / "motion.npz", t=times, joints=joints, joint_names=np.asarray(k["control_joints"]),
                        link_names=np.asarray(link_names), link_transforms=links, pick_transforms=picks,
                        finger_transforms=fingers, pad_world=pads, phase=np.asarray([s[1] for s in states]))
    print(json.dumps(p["contact_motion"], indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--pose-plan", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pick-preset",choices=("single","wide","strum5",*LOOP_PRESETS),default="single")
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument("--probe", action="store_true")
    mode.add_argument("--pick-probe", action="store_true")
    mode.add_argument("--held-pick-probe", action="store_true")
    mode.add_argument("--loop-probe", action="store_true")
    args = parser.parse_args()
    prepare(args.profile, args.pose_plan, args.out, args.probe, args.pick_probe, args.held_pick_probe,args.pick_preset,args.loop_probe)


if __name__ == "__main__":
    main()
