"""Six shallow contacts connected by raised transfers and a raised return."""
import numpy as np

from .contact_motion import ease


WIDE_STARTS = (1.20, 2.90, 4.60)
WIDE_VISIBLE_START = 1.15
WIDE_PRESS_START = .25
WIDE_HOLD_START = .85
EVENT_DURATION = .15
CONTACT_DURATION = .12
RETURN_DURATION = .40


def wide_path(guitar, config, starts):
    station = float(config["station_y_m"])
    travel = float(config["stroke_x_m"])
    clearance = float(config["clear_lift_m"])
    offset = float(config["start_offset_m"])
    depth = float(config["tip_z_offset_m"])
    if not .025 <= station <= .080 or abs(abs(travel) - .0022) > 1e-10 or not .010 <= clearance <= .022:
        raise ValueError("Wide sweep requires shallow engagements and raised transfers near the bridge")
    if np.sign(offset) != -np.sign(travel) or not 0 < abs(offset) < abs(travel) or abs(depth) > .002:
        raise ValueError("Wide pick must start on the approaching side of the chord")
    nut = np.asarray(guitar["nut_endpoints_local_m"], dtype=float)
    bridge = np.asarray(guitar["bridge_endpoints_local_m"], dtype=float)
    fraction = (station - bridge[:, 1]) / (nut[:, 1] - bridge[:, 1])
    chord = bridge + fraction[:, None] * (nut - bridge)
    offsets=np.asarray(config.get("string_target_offsets_m",np.zeros((6,3))),dtype=float)
    if offsets.shape!=(6,3) or not np.isfinite(offsets).all() or np.max(abs(offsets))>.0015 or np.any(offsets[:,1]!=0):
        raise ValueError("Per-string target offsets must be small transverse/normal corrections")
    order = list(range(6) if travel > 0 else range(5, -1, -1))
    r = np.asarray(config["right_orientation"], dtype=float)
    pivot = np.asarray(config["tip_reference_local_m"], dtype=float)
    if r.shape != (3, 3) or not np.allclose(r.T @ r, np.eye(3), rtol=0, atol=1e-7) or np.linalg.det(r) < 0:
        raise ValueError("Wide pick orientation must be a proper rotation")
    center = (chord+offsets)[order] + [offset, 0, depth] - r @ pivot
    high = np.array([0., 0., clearance])
    step = np.array([travel, 0., 0.])
    sweep_duration = EVENT_DURATION * len(order)
    default = center[0] + high

    def target(t):
        for start in starts:
            u = t - start
            if not 0 <= u < sweep_duration + RETURN_DURATION:
                continue
            if u >= sweep_duration:
                amount = ease((u - sweep_duration) / RETURN_DURATION)
                return (1 - amount) * (center[-1] + step + high) + amount * default
            j = min(int(u / EVENT_DURATION), len(center) - 1)
            x = u - j * EVENT_DURATION
            current = center[j]
            if x < .03:
                return current + high * (1 - ease(x / .03))
            if x < .09:
                return current + step * ease((x - .03) / .06)
            if x < CONTACT_DURATION:
                return current + step + high * ease((x - .09) / .03)
            following = center[j + 1] if j + 1 < len(center) else current + step
            amount = ease((x - CONTACT_DURATION) / (EVENT_DURATION - CONTACT_DURATION))
            return (1 - amount) * (current + step) + amount * following + high
        return default

    sweeps = []
    for start in starts:
        events = [
            dict(string_index=string, window_s=[start + i * EVENT_DURATION, start + i * EVENT_DURATION + CONTACT_DURATION])
            for i, string in enumerate(order)
        ]
        sweeps.append(dict(start_s=start, end_s=start + sweep_duration, return_end_s=start + sweep_duration + RETURN_DURATION, events=events))
    record = dict(visible_start_s=WIDE_VISIBLE_START, press_start_s=WIDE_PRESS_START, hold_start_s=WIDE_HOLD_START,
                  station_y_m=station, clear_lift_m=clearance, signed_transverse_travel_m=travel,
                  target_tip_range_m=float(np.ptp(chord[:, 0]) + abs(travel)), sweeps=sweeps,
                  rotation_guitar_local=r.tolist(), tip_reference_local_m=pivot.tolist(),
                  string_target_offsets_m=offsets.tolist(), pick_sampling_halfwidth_m=.014)
    return target, r, record


WIDE_SCOPE = "full_hand_v4_held_wide"


def is_wide(profile):
    motion = profile.get("contact_motion", {})
    mode = motion.get("pick_preset", "single")
    if mode == "wide" and motion.get("version") == 4:
        return True
    if mode != "single" or motion.get("version") == 4:
        raise ValueError("Unknown or inconsistent guitar pick preset")
    return False


def wide_criteria(criteria, motion):
    """Declare the held-only windows and bounds before the production runs."""
    wide = motion["wide_pick"]
    sweeps = wide["sweeps"]
    if len(sweeps) != 3 or any(len(s["events"]) != 6 for s in sweeps):
        raise ValueError("A final wide case requires three complete six-string sweeps")
    duration=float(motion["duration_s"])
    boundaries=(wide["press_start_s"],wide["hold_start_s"],wide["visible_start_s"])
    if boundaries!=(WIDE_PRESS_START,WIDE_HOLD_START,WIDE_VISIBLE_START) or not np.isfinite(duration):
        raise ValueError("Invalid held-only press and visible interval")
    travel=wide["signed_transverse_travel_m"]
    order=list(range(6) if travel>0 else range(5,-1,-1))
    selected = motion["selected_string_index"]
    energy = []
    holds = [[wide["visible_start_s"], motion["duration_s"]]]
    events = []
    prior=wide["visible_start_s"]
    for start,s in zip(WIDE_STARTS,sweeps):
        end=start+6*EVENT_DURATION
        returned=end+RETURN_DURATION
        if (s["start_s"],s["end_s"],s["return_end_s"])!=(start,end,returned) or not prior<=start<end<returned<=duration:
            raise ValueError("Wide sweep or return falls outside the frozen visible interval")
        for i,e in enumerate(s["events"]):
            a=start+i*EVENT_DURATION
            if e["string_index"]!=order[i] or e["window_s"]!=[a,a+CONTACT_DURATION]:
                raise ValueError("Wide event windows must be ordered, disjoint, and inside their sweep")
        prior=returned
        events.extend(e["window_s"] for e in s["events"])
        holds.append([s["start_s"], s["end_s"]])
        holds.append([s["end_s"], s["return_end_s"]])
        selected_event = next(e for e in s["events"] if e["string_index"] == selected)
        t = selected_event["window_s"][1]
        energy.append([t + .02, t + .35])
    for key in ("release_window_s", "open_pluck_excursion_range_m"):
        criteria.pop(key, None)
    criteria.update(scope=WIDE_SCOPE, wide_pick=wide, hold_windows_s=holds,
                    pick_windows_s=events, post_pluck_energy_windows_s=energy,
                    spatial_probe_hold_window_s=[.95, 1.30], max_non_selected_pick_load_N=.0001,
                    spatial_probe_mode="one_sweep_held",
                    wide_tip_range_m=[.045, .060], min_actual_transit_lift_m=.008,
                    max_support_dropout_s=.025, max_precapture_rms_velocity_m_s=.05,
                    picked_excursion_range_m=[.0002,.0025],
                    max_spatial_pick_impulse_relative_change=.30,
                    method="Held-only wide picking with offscreen press. All event loads use recorded mesh wrenches. The minimum support fraction and longest consecutive outage apply to the whole visible hold. Exact geometry covers all recorded hand/string states. The spatial pair each repeats one actual six-string sweep while refining unpinned pick and fret sampling.")
