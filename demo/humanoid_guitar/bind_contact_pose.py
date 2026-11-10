"""Bind the recorded full-hand pose to an equivalent, freshly prepared asset set."""

import argparse
import copy
import json
from pathlib import Path

from .contact_integrity import full_hands, sha, validate


def select(mapping, keys):
    return {key: copy.deepcopy(mapping[key]) for key in keys}


def asset_contract(profile):
    """Exclude installation paths and nondeterministic USD serialization only.

    Geometry arrays, local frames, units, joint definitions and visual prim
    associations must match. The prepared asset validator remains responsible
    for proving visible USD/collision equivalence.
    """
    if not full_hands(profile):
        raise ValueError("Pose template requires complete-hand assets")
    frame = ("meters_per_unit", "up_axis", "default_prim")
    mesh_fields = ("id", "sha256", "obj_sha256", "frame", "bounds",
                   "visual_prim_paths", "spacing_m", "margin_m", "friction")
    hand_fields = ("link", "link_from_mesh", "side", "function", "shell_index", "shell_count")
    robot = select(profile["robot"], frame + ("kinematics", "link_prim_paths"))
    robot["contact_meshes"] = [select(item, mesh_fields + hand_fields)
                               for item in profile["robot"]["contact_meshes"]]
    robot["arm_clearance_meshes"] = [select(item, ("id", "sha256", "frame", "link", "link_from_mesh", "bounds"))
                                     for item in profile["robot"]["arm_clearance_meshes"]]
    guitar = select(profile["guitar"], frame + (
        "nut_endpoints_local_m", "bridge_endpoints_local_m", "face_normal_local",
        "transverse_axis_local", "neck_axis_local", "source_to_local"))
    guitar["contact_meshes"] = [select(item, mesh_fields) for item in profile["guitar"]["contact_meshes"]]
    pick = select(profile["pick"], frame + (
        "collision_npz_sha256", "collision_mesh_sha256", "attachment_link",
        "bounds", "visual_prim_paths", "tip_extension_m", "tip_axis_local", "loaded_side_axis_local"))
    return dict(schema=1, contact_schema_version=profile["contact_schema_version"],
                robot=robot, guitar=guitar, pick=pick)


def export(profile_path, plan_path, out):
    profile = json.loads(profile_path.read_text())
    plan = json.loads(plan_path.read_text())
    if plan["asset_profile_sha256"] != sha(profile_path):
        raise ValueError("Original pose plan does not bind the supplied asset profile")
    validate(profile)
    template = dict(schema="deformx_full_hand_pose_template_v1",
                    source_pose_sha256=sha(plan_path), asset_contract=asset_contract(profile), pose=plan)
    write_new(out, template)


def bind(profile_path, template_path, out):
    profile = json.loads(profile_path.read_text())
    template = json.loads(template_path.read_text())
    if template.get("schema") != "deformx_full_hand_pose_template_v1":
        raise ValueError("Unsupported full-hand pose template")
    validate(profile)
    if asset_contract(profile) != template["asset_contract"]:
        raise ValueError("Prepared geometry or kinematics differs from the recorded pose contract")
    plan = copy.deepcopy(template["pose"])
    plan["asset_profile_sha256"] = sha(profile_path)
    plan["portable_template_sha256"] = sha(template_path)
    write_new(out, plan)


def write_new(path, value):
    with path.open("x") as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")
    print(sha(path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--export-plan", type=Path, help="Author a portable template from its original verified assets")
    source.add_argument("--template", type=Path, help="Bind a supplied template to equivalent local assets")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.export_plan:
        export(args.profile, args.export_plan, args.out)
    else:
        bind(args.profile, args.template, args.out)


if __name__ == "__main__":
    main()
