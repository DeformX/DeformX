"""Bind the checked-in numerical limits to a fresh local tight motion profile."""
import argparse,json
from pathlib import Path
from .contact_integrity import full_hands,sha
from .contact_strum_path import is_strum,strum_criteria
from .contact_loop_strum_path import is_loop_strum,loop_criteria
from .contact_wide_path import is_wide,wide_criteria


def main():
    p=argparse.ArgumentParser();p.add_argument("--profile",type=Path,required=True);p.add_argument("--template",type=Path,required=True);p.add_argument("--out",type=Path,required=True);a=p.parse_args()
    profile=json.loads(a.profile.read_text());criteria=json.loads(a.template.read_text());motion=profile["contact_motion"]
    params=Path(__file__).with_name("contact_parameters.json");expected=json.loads(params.read_text())["tight_75"]
    stroke=motion["pick_stroke"]
    loop=is_loop_strum(profile)
    if loop:
        if not full_hands(profile):
            raise ValueError("The reciprocal preset requires complete hand collision geometry")
        loop_criteria(criteria,motion)
    elif is_strum(profile):
        if not full_hands(profile):
            raise ValueError("The five-string preset requires complete hand collision geometry")
        strum_criteria(criteria,motion)
    elif full_hands(profile):
        wide=is_wide(profile)
        travel=motion["wide_pick"]["signed_transverse_travel_m"] if wide else stroke["signed_transverse_travel_m"]
        if motion["version"] not in (3,4) or abs(abs(travel)-expected["pick"]["travel_m"])>1e-10:
            raise ValueError("Full-hand motion does not retain the shallow physical stroke")
        criteria["scope"]="full_hand_v3_full_batch"
        criteria["spatial_probe_hold_window_s"]=[.90,1.30]
        criteria["max_non_selected_pick_load_N"]=.0001
        criteria["method"]="480 Hz around press, release and all pick windows, 60 Hz otherwise. Exact triangle hand/string audit at every recorded frame, full instrument geometry checks, all-six recorded repeatability, and a short fret-only spatial check refining the pad and fret fields."
        if wide:
            wide_criteria(criteria,motion)
    elif stroke!=expected["pick"]:
        raise ValueError("Tight profile does not retain the archived shallow stroke")
    if (motion["probe"] and not loop) or motion["parameter_preset"]!="tight_75" or motion["record_hz"]!=expected["record_hz"] or profile["contact_material"]!=expected["material"] or motion["parameter_profile_sha256"]!=sha(params):raise ValueError("This frozen template applies to the 75N/480Hz shallow-stroke profile")
    criteria.update(tight_main_profile_sha256=sha(a.profile),criteria_template_sha256=sha(a.template),frozen_before_final_gpu_runs=True)
    with a.out.open("x") as f:f.write(json.dumps(criteria,indent=2)+"\n")
    print(sha(a.out))
if __name__=="__main__":main()

