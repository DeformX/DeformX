"""Check one recorded 24-stroke case; this is not a convergence study."""

import argparse
import json
from pathlib import Path

from .contact_integrity import grid, sha, specs, validate
from .contact_loop_strum_path import LOOP_LONG_SCOPE, loop_criteria


SCOPE = 'single_case_24_stroke'
GATES = frozenset({
    'finite_unflagged_full_record', 'only_endpoints_pinned', 'fixed_held_left_joints',
    'opposing_pad_and_fret_support', 'five_ordered_pick_contacts_per_stroke',
    'measured_transverse_excursion', 'pick_released_at_each_turnaround',
    'no_pick_on_held_string_or_outside_strokes', 'no_right_hand_or_fingerboard_contact',
})
BOUND_FILES = {
    'profile_sha256': 'profile.json', 'motion_sha256': 'motion.npz',
    'trajectory_sha256': 'trajectory.npz', 'criteria_sha256': 'criteria.json',
}


def binding(case):
    return {key: sha(case / name) for key, name in BOUND_FILES.items()}


def check(case):
    import numpy as np

    run = json.loads((case / 'report.json').read_text())
    profile = json.loads((case / 'profile.json').read_text())
    criteria = json.loads((case / 'criteria.json').read_text())
    hashes = binding(case)
    if any(run.get(key) != digest for key, digest in hashes.items()):
        raise ValueError('Case report does not match its profile, motion, trajectory and frozen criteria')
    validate(profile)
    motion = profile['contact_motion']
    schedule = motion['loop_strum']
    strokes = schedule['strokes']
    if (criteria['scope'] != LOOP_LONG_SCOPE or len(strokes) != 24
            or schedule != criteria['loop_strum'] or motion['selected_string_index'] != 5
            or motion['record_hz'] != 480 or motion['probe']):
        raise ValueError('Expected the frozen 24-stroke, 480 Hz case with string 6 held')
    expected_criteria = dict(criteria)
    loop_criteria(expected_criteria, motion)
    if expected_criteria != criteria:
        raise ValueError('The recorded schedule or frozen strumming thresholds changed')

    manifest = run['mesh_manifest']
    ids = {mesh['id']: mesh['index'] for mesh in manifest}
    if ids != {mesh['id']: i for i, mesh in enumerate(specs(profile))}:
        raise ValueError('Recorded mesh/wrench indices do not match the exact contact meshes')
    hz = motion['record_hz']
    with np.load(case / 'trajectory.npz', allow_pickle=False) as saved:
        fields = ('positions', 'velocities', 'quaternions', 'mesh_wrenches', 'collider_transforms', 'surface_forces')
        data = {key: saved[key] for key in (*fields, 't', 'phase', 'pinned', 'vertex_masses',
                                            'pick_transforms', 'finger_transforms', 'baseline')}
    with np.load(case / 'motion.npz', allow_pickle=False) as fk:
        if not all(np.array_equal(data[key], fk[key]) for key in ('t', 'phase', 'pick_transforms', 'finger_transforms')):
            raise ValueError('Saved trajectory and robot motion disagree')
        left_joints = np.char.startswith(fk['joint_names'], 'left_') | np.char.startswith(fk['joint_names'], 'L_')
        joints = fk['joints'][:, left_joints]
        pads = fk['pad_world']
    t, poses, xyz = data['t'], data['collider_transforms'], data['positions']
    if not np.array_equal(t, np.arange(round(motion['duration_s'] * hz) + 1) / hz):
        raise ValueError('Full 480 Hz trajectory is missing or has a wrong sample time')
    if not all(np.isfinite(data[key]).all() for key in fields):
        raise ValueError('Nonfinite saved positions, velocities, forces or transforms')
    expected_phase = np.where(t < schedule['press_start_s'], 'OPEN',
                              np.where(t < schedule['hold_start_s'], 'PRESS', 'HOLD'))
    if not np.array_equal(data['phase'], expected_phase):
        raise ValueError('Recorded press and hold phases disagree with the frozen schedule')
    placement = np.asarray(profile['guitar']['placement'])
    finger = [mesh['index'] for mesh in manifest if mesh['role'] == 'finger']
    static = [mesh['index'] for mesh in manifest if mesh['role'] == 'instrument']
    if not (np.all(poses[:, static] == placement) and np.array_equal(poses[:, finger], data['finger_transforms'])
            and np.array_equal(poses[:, ids['pick']], data['pick_transforms'])):
        raise ValueError('Recorded collider transforms disagree with the fixed guitar and prescribed FK')

    forces = data['mesh_wrenches'][..., :3]
    pick = np.linalg.norm(forces[:, :, ids['pick']], axis=-1)
    right = [mesh['index'] for mesh in manifest if mesh.get('side') == 'right']
    visible = t >= schedule['visible_start_s']
    force_min = criteria['min_same_string_opposing_normal_N']
    quiet = criteria['max_non_selected_pick_load_N']
    held = 5
    fret, distal = ids['fret_05'], ids['L_index_intermediate']
    face = placement[:3, 2]
    fret_load = -(forces[:, held, fret] @ face)
    pad_load = forces[:, held, distal] @ face
    radius = run['material']['radius_m']
    cache = case.parent / 'cache'
    grids = []
    for index in (fret, distal):
        mesh = manifest[index]
        g, record = grid(profile, mesh, cache, run['settings']['grid_scale'], allow_bake=False)
        if record != mesh['cache']:
            raise ValueError(f'Case SDF binding changed for {mesh["id"]}')
        grids.append(g)
    points = xyz[:, held]
    local = (points - placement[:3, 3]) @ placement[:3, :3]
    phi, inside = grids[0].sample(local.reshape(-1, 3))
    gap = np.where(inside, phi - radius, np.inf).reshape(local.shape[:2])
    nearest = np.argmin(gap, axis=1)
    rows = np.arange(len(t))
    crown = local[rows, nearest]
    bounds = np.asarray(manifest[fret]['bounds'])
    tangent = local[rows, np.minimum(nearest + 1, local.shape[1] - 1)] - local[rows, np.maximum(nearest - 1, 0)]
    crown_contact = ((gap[rows, nearest] <= criteria['max_fret_native_gap_m'])
                     & (crown[:, 1] >= bounds[0, 1] - .0001) & (crown[:, 1] <= bounds[1, 1] + .0001)
                     & (np.abs(crown[:, 0]) <= np.max(np.abs(bounds[:, 0])) + .0001)
                     & (np.abs(tangent[:, 1]) >= criteria['min_tangent_abs_longitudinal']
                        * np.linalg.norm(tangent, axis=-1)))
    distal_poses = poses[:, distal]
    local_pad = (points - distal_poses[:, None, :3, 3]) @ distal_poses[:, :3, :3]
    phi, inside = grids[1].sample(local_pad.reshape(-1, 3))
    pad_bounds = np.asarray(manifest[distal]['bounds'])
    pad_radius = (pad_bounds[1, 0] - pad_bounds[0, 0]) / 2
    pad_contact = np.any((inside & (phi <= radius + .00015)).reshape(points.shape[:2])
                         & (np.linalg.norm(points - pads[:, None], axis=-1) <= pad_radius), axis=1)
    supported = (fret_load > force_min) & (pad_load > force_min) & crown_contact & pad_contact
    # Length of the longest contiguous loss of opposing support, in saved samples.
    boundaries = np.flatnonzero(np.diff(np.r_[False, ~supported[visible], False]))
    outage = int(np.max(boundaries[1::2] - boundaries[::2], initial=0))

    baseline = data['baseline']
    axis = baseline[:, -1] - baseline[:, 0]
    axis /= np.linalg.norm(axis, axis=-1, keepdims=True)
    station = (baseline - placement[:3, 3]) @ placement[:3, 1]
    events, fractions = [], []
    for index, stroke in enumerate(strokes):
        start, end = stroke['window_s']
        mask = (t >= start) & ((t <= end) if index == len(strokes) - 1 else (t < end))
        fractions.append(float(supported[mask].mean()))
        channels = []
        for string in stroke['target_string_order']:
            loaded = np.flatnonzero(mask & (pick[:, string] > quiet))
            excursion = first_time = None
            if len(loaded):
                first_time = float(t[loaded[0]])
                delta = xyz[loaded[0]:loaded[-1] + round(.05 * hz) + 1, string] - baseline[string]
                transverse = delta - (delta @ axis[string])[..., None] * axis[string]
                excursion = float(np.linalg.norm(transverse[:, (station[string] > .015) & (station[string] < .25)], axis=-1).max())
            channels.append(dict(string=string + 1, first_load_s=first_time,
                                 impulse_Ns=float(pick[mask, string].sum() / hz),
                                 peak_load_N=float(pick[mask, string].max()), transverse_excursion_m=excursion))
        events.append(dict(window_s=[start, end], channels=channels))

    turn_indices = sorted({round(bound * hz) for row in strokes for bound in row['window_s']})
    engaged = (t >= strokes[0]['window_s'][0]) & (t <= strokes[-1]['window_s'][1])
    excluded_peak = float(pick[:, held].max())
    outside_peak = float(pick[~engaged].max())
    turn_peak = float(pick[turn_indices].max())
    right_peak = float(np.linalg.norm(forces[:, :, right], axis=-1).max())
    board_peak = float(np.linalg.norm(forces[visible, :, ids['neck_fingerboard']], axis=-1).max())
    pin_mask = np.zeros(data['pinned'].shape, dtype=bool)
    pin_mask[:, [0, -1]] = True
    metrics = run['metrics']
    min_excursion, max_excursion = criteria['picked_excursion_range_m']
    flat_channels = [channel for stroke in events for channel in stroke['channels']]
    loaded_count = sum(channel['first_load_s'] is not None for channel in flat_channels)
    excursions = [channel['transverse_excursion_m'] for channel in flat_channels if channel['transverse_excursion_m'] is not None]
    gates = {
        'finite_unflagged_full_record': run['complete'] is True and metrics['finite'] is True and not metrics['flags']
        and run['finger_enabled'] is True and metrics['max_segment_strain'] <= criteria['max_segment_strain']
        and metrics['max_endpoint_error_m'] <= criteria['max_endpoint_error_m'],
        'only_endpoints_pinned': np.array_equal(data['pinned'], pin_mask)
        and np.array_equal(data['vertex_masses'] == 0, pin_mask),
        'fixed_held_left_joints': joints.shape[1] > 0 and bool(np.all(joints[visible] == joints[visible][0])),
        'opposing_pad_and_fret_support': min(float(supported[visible].mean()), *fractions) >= criteria['min_hold_opposing_fraction']
        and outage / hz <= criteria['max_support_dropout_s'],
        'five_ordered_pick_contacts_per_stroke': len(flat_channels) == 120 and loaded_count == 120
        and all(channel['impulse_Ns'] >= criteria['min_pick_impulse_Ns'] for channel in flat_channels)
        and all(all(a['first_load_s'] < b['first_load_s'] for a, b in zip(row['channels'], row['channels'][1:]))
                for row in events),
        'measured_transverse_excursion': loaded_count == 120 and all(min_excursion <= value <= max_excursion for value in excursions),
        'pick_released_at_each_turnaround': turn_peak <= quiet,
        'no_pick_on_held_string_or_outside_strokes': max(excluded_peak, outside_peak) <= quiet,
        'no_right_hand_or_fingerboard_contact': right_peak <= quiet and board_peak <= force_min,
    }
    assert gates.keys() == GATES
    failed = [gate for gate, passed in gates.items() if not passed]
    return dict(complete=not failed, acceptance_scope=SCOPE, gates=gates, failed_required_gates=failed,
                audit_source_sha256=sha(__file__), case_report_sha256=sha(case / 'report.json'),
                **hashes, stroke_count=len(strokes), saved_states=len(t), recorded_seconds=float(t[-1]),
                loaded_stroke_string_channels=loaded_count, strokes=events,
                visible_supported_states=int(supported[visible].sum()), visible_states=int(visible.sum()),
                longest_visible_support_dropout_s=outage / hz, minimum_per_stroke_support_fraction=min(fractions),
                median_visible_fret_load_N=float(np.median(fret_load[visible])),
                transverse_excursion_range_m=[min(excursions), max(excursions)] if excursions else None,
                max_excluded_pick_load_N=excluded_peak, max_outside_stroke_pick_load_N=outside_peak,
                max_turnaround_pick_load_N=turn_peak, max_right_hand_load_N=right_peak,
                max_visible_fingerboard_load_N=board_peak,
                validation_scope='Single case; no left-disabled control, temporal/spatial comparisons or dense penetration audit.')


def validate_capture(case, audit_path):
    audit = json.loads(audit_path.read_text())
    if (audit.get('acceptance_scope') != SCOPE or audit.get('complete') is not True
            or audit.get('failed_required_gates') != [] or audit.get('stroke_count') != 24
            or set(audit.get('gates', {})) != GATES or not all(v is True for v in audit['gates'].values())
            or audit.get('audit_source_sha256') != sha(__file__)
            or audit.get('case_report_sha256') != sha(case / 'report.json')
            or any(audit.get(key) != digest for key, digest in binding(case).items())):
        raise ValueError('Replay requires a passing single-case audit of these exact recorded states; rerun audit')
    return dict(acceptance_scope=SCOPE, single_case_audit_complete=True,
                acceptance_audit_sha256=sha(audit_path), audit_path=str(audit_path.resolve()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    report = check(args.case)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({key: value for key, value in report.items() if key != 'strokes'}, indent=2))
    if not report['complete']:
        parser.exit(1, f'Single-case audit failed; see {args.out}\n')


if __name__ == '__main__':
    main()
