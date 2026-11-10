"""One continuous five-string downstroke with a raised return."""
import numpy as np

from .contact_motion import ease
from .contact_wide_path import WIDE_HOLD_START, WIDE_PRESS_START, WIDE_VISIBLE_START


STRUM_SCOPE = "full_hand_v5_held_strum5"
STRUM_ORDER = (4, 3, 2, 1, 0)
STRUM_START = 1.25
STRUM_END = 1.70
STRUM_APPROACH = 1.15
STRUM_LIFT_END = 1.80
STRUM_RETURN_END = 2.30
STRUM_DURATION = 2.60


def is_strum(profile):
    motion = profile.get("contact_motion", {})
    mode = motion.get("pick_preset", "single")
    version = motion.get("version")
    if (mode == "strum5") != (version == 5):
        raise ValueError("Inconsistent five-string strum preset")
    return mode == "strum5"


def strum_path(guitar, config, tuning):
    station = float(config["station_y_m"])
    lift = float(config["clear_lift_m"])
    depth = float(tuning.get("tip_z_offset_m", config["tip_z_offset_m"]))
    if not .025 <= station <= .080 or not .010 <= lift <= .022 or not -.002 <= depth <= .001:
        raise ValueError("Invalid continuous strum station, lift, or engagement depth")
    bridge = np.asarray(guitar["bridge_endpoints_local_m"], dtype=float)
    nut = np.asarray(guitar["nut_endpoints_local_m"], dtype=float)
    chord = bridge + ((station - bridge[:, 1]) / (nut[:, 1] - bridge[:, 1]))[:, None] * (nut - bridge)
    if not np.isfinite(chord).all() or not np.all(np.diff(chord[:, 0]) > 0):
        raise ValueError("Strum requires six playable strings ordered by transverse position")
    rotation = np.asarray(config["right_orientation"], dtype=float)
    pivot = np.asarray(config["tip_reference_local_m"], dtype=float)
    if (rotation.shape != (3, 3) or pivot.shape != (3,) or not np.isfinite(pivot).all()
            or not np.allclose(rotation.T @ rotation, np.eye(3), rtol=0, atol=1e-7)
            or np.linalg.det(rotation) < 0):
        raise ValueError("Invalid strum pick rotation or tip reference")
    z = float(chord[:5, 2].mean() + depth)
    first = np.array([chord[4, 0] + .003, station, z]) - rotation @ pivot
    last = np.array([chord[0, 0] - .003, station, z]) - rotation @ pivot
    raised = np.array([0., 0., lift])
    travel = last - first

    def target(t):
        if t < STRUM_START:
            return first + (1 - ease((t - STRUM_APPROACH) / (STRUM_START - STRUM_APPROACH))) * raised
        if t < STRUM_END:
            return first + ease((t - STRUM_START) / (STRUM_END - STRUM_START)) * travel
        if t < STRUM_LIFT_END:
            return last + ease((t - STRUM_END) / (STRUM_LIFT_END - STRUM_END)) * raised
        return last - ease((t - STRUM_LIFT_END) / (STRUM_RETURN_END - STRUM_LIFT_END)) * travel + raised

    record = dict(visible_start_s=WIDE_VISIBLE_START, press_start_s=WIDE_PRESS_START,
                  hold_start_s=WIDE_HOLD_START, approach_start_s=STRUM_APPROACH,
                  stroke_window_s=[STRUM_START, STRUM_END], lift_end_s=STRUM_LIFT_END,
                  return_end_s=STRUM_RETURN_END, target_string_order=list(STRUM_ORDER),
                  station_y_m=station, clear_lift_m=lift, tip_z_offset_m=depth,
                  target_tip_range_m=float(-travel[0]), rotation_guitar_local=rotation.tolist(),
                  tip_reference_local_m=pivot.tolist(), pick_sampling_halfwidth_m=.014)
    return target, rotation, record


def strum_criteria(criteria, motion):
    pick = motion["strum_pick"]
    if (motion["duration_s"] != STRUM_DURATION or motion["selected_string_index"] != 5
            or pick["target_string_order"] != list(STRUM_ORDER)
            or pick["stroke_window_s"] != [STRUM_START, STRUM_END]
            or (pick["press_start_s"], pick["hold_start_s"], pick["visible_start_s"],
                pick["approach_start_s"], pick["lift_end_s"], pick["return_end_s"]) != (
                WIDE_PRESS_START, WIDE_HOLD_START, WIDE_VISIBLE_START, STRUM_APPROACH,
                STRUM_LIFT_END, STRUM_RETURN_END)):
        raise ValueError("Strum schedule differs from the frozen single continuous stroke")
    for key in ("release_window_s", "open_pluck_excursion_range_m",
                "post_pluck_energy_windows_s", "max_pressed_to_off_nut_bridge_energy_ratio"):
        criteria.pop(key, None)
    criteria.update(
        scope=STRUM_SCOPE, strum_pick=pick,
        hold_windows_s=[[WIDE_VISIBLE_START, STRUM_DURATION], [STRUM_START, STRUM_END]],
        pick_windows_s=[[STRUM_APPROACH, STRUM_RETURN_END]],
        max_non_selected_pick_load_N=.0001, strum_tip_range_m=[.043, .050],
        max_tip_reverse_step_m=.00002, max_tip_stroke_y_span_m=.00010,
        max_tip_stroke_z_span_m=.00010, min_actual_return_lift_m=.008,
        max_support_dropout_s=.025, max_precapture_rms_velocity_m_s=.05,
        max_held_palm_neck_gap_m=.020, max_held_rear_thumb_neck_gap_m=.002,
        picked_excursion_range_m=[.0002,.0025],
        max_temporal_pick_impulse_relative_change=.30,
        max_spatial_pick_impulse_relative_change=.30,
        method="One downstroke across strings 5 through 1 at fixed normal height. Contacts may overlap on adjacent strings. The left hand holds string 6 without picking it. Compare temporal and spatial refinements over the identical complete stroke; nominal is the coarse spatial case.")
