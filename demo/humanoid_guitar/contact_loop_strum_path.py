"""Alternating down/up strokes at one continuous engagement depth."""
import numpy as np

from .contact_motion import ease
from .contact_strum_path import STRUM_END, STRUM_ORDER, STRUM_START, strum_path

LOOP_SCOPE = "full_hand_v6_loop_strum5"
LOOP_LONG_SCOPE = "full_hand_v7_loop_strum5_long"
LOOP_PRESET = "strum5-loop"
LOOP_LONG_PRESET = "strum5-loop-long"
LOOP_PRESETS = (LOOP_PRESET, LOOP_LONG_PRESET)
LOOP_DURATION = 5.25
LOOP_PROBE_DURATION = 2.60
STROKE_DURATION = .45
STROKE_COUNT = 8
LONG_STROKE_COUNT = 24


def loop_contract(preset, probe=False):
    """Return version, stroke count, duration and scope for a named schedule."""
    if preset == LOOP_PRESET:
        return (6, 2, LOOP_PROBE_DURATION, LOOP_SCOPE) if probe else (6, STROKE_COUNT, LOOP_DURATION, LOOP_SCOPE)
    if preset == LOOP_LONG_PRESET and not probe:
        duration = round(LOOP_DURATION + (LONG_STROKE_COUNT - STROKE_COUNT) * STROKE_DURATION, 8)
        return 7, LONG_STROKE_COUNT, duration, LOOP_LONG_SCOPE
    raise ValueError("Unknown or unsupported reciprocal strum schedule")


def is_loop_strum(profile):
    motion = profile.get("contact_motion", {})
    preset = motion.get("pick_preset")
    version = motion.get("version")
    if preset in LOOP_PRESETS:
        if version != loop_contract(preset, motion.get("probe", False))[0]:
            raise ValueError("Inconsistent reciprocal strum preset")
        return True
    if version in (6, 7):
        raise ValueError("Inconsistent reciprocal strum preset")
    return False


def strokes(count):
    return [dict(window_s=[round(STRUM_START+i*STROKE_DURATION, 8),
                           round(STRUM_START+(i+1)*STROKE_DURATION, 8)],
                 target_string_order=list(STRUM_ORDER if i % 2 == 0 else reversed(STRUM_ORDER)))
            for i in range(count)]


def loop_strum_path(guitar, config, tuning, probe, preset=LOOP_PRESET):
    single, rotation, record = strum_path(guitar, config, tuning)
    start, end = single(STRUM_START), single(STRUM_END)
    _, count, _, _ = loop_contract(preset, probe)
    sweep = strokes(count)
    last = sweep[-1]["window_s"][1]
    lift_end = round(last+.10, 8)
    raised = np.array([0., 0., record["clear_lift_m"]])

    def target(t):
        if t < STRUM_START:
            return single(t)
        for i, row in enumerate(sweep):
            a,b = row["window_s"]
            if t < b:
                fraction = ease((t-a)/(b-a))
                return start + (end-start)*(fraction if i % 2 == 0 else 1-fraction)
        return start + ease((t-last)/(lift_end-last))*raised

    del record["stroke_window_s"],record["target_string_order"]
    record.update(strokes=sweep,lift_end_s=lift_end,return_end_s=lift_end)
    return target,rotation,record


def loop_criteria(criteria, motion):
    loop = motion["loop_strum"]
    version, count, duration, scope = loop_contract(motion["pick_preset"], motion["probe"])
    expected = strokes(count)
    end = expected[-1]["window_s"][1]
    if (motion["version"] != version or motion["duration_s"] != duration
            or motion["selected_string_index"] != 5 or loop["strokes"] != expected
            or (scope == LOOP_LONG_SCOPE and loop["tip_z_offset_m"] != -.0005)
            or (loop["press_start_s"],loop["hold_start_s"],loop["visible_start_s"],
                loop["approach_start_s"],loop["lift_end_s"],loop["return_end_s"]) !=
                (.25,.85,1.15,1.15,round(end+.10,8),round(end+.10,8))):
        raise ValueError("Invalid reciprocal strum schedule")
    for name in ("release_window_s","open_pluck_excursion_range_m","post_pluck_energy_windows_s",
                 "max_pressed_to_off_nut_bridge_energy_ratio","strum_pick","wide_pick"):
        criteria.pop(name, None)
    criteria.update(
        scope=scope,loop_strum=loop,
        hold_windows_s=[[1.15,motion["duration_s"]],*[row["window_s"] for row in expected]],
        pick_windows_s=[[1.15,loop["lift_end_s"]]],max_non_selected_pick_load_N=.0001,
        strum_tip_range_m=[.043,.050],max_tip_reverse_step_m=.00002,
        max_tip_stroke_y_span_m=.00010,max_tip_stroke_z_span_m=.00010,
        min_actual_return_lift_m=.008,max_support_dropout_s=.025,max_precapture_rms_velocity_m_s=.05,
        max_held_palm_neck_gap_m=.020,max_held_rear_thumb_neck_gap_m=.002,
        picked_excursion_range_m=[.0002,.0025],max_temporal_pick_impulse_relative_change=.30,
        max_spatial_pick_impulse_relative_change=.30,
        method="Continuous alternating strokes, fixed y/z throughout all turnarounds. Five ordered impulses per stroke, zero excluded sixth and visible fingerboard reaction. Nominal and both refinements compare the same full sequence.")
