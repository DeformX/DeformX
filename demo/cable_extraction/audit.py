"""Geometric audit of a measured trajectory, independent of controller goals."""
import argparse
from itertools import combinations
import json
from pathlib import Path

import numpy as np

from .model import FPS, PULL_TAB_SIZE, RADIUS, crossing_summary, minimum_segment_distance, score
from .run import sha


def audit(record, report, initialized):
    cables = record['cable']
    phase = record['phase']
    # Audit the full pair set, including decoy–decoy crossings. Segment-centerline
    # distance is geometric evidence at 30 Hz, not a native force readout.
    pair_clearance = {
        f'{i}-{j}': min(minimum_segment_distance(frame[i],frame[j]) for frame in cables)
        for i,j in combinations(range(cables.shape[1]),2)
    }
    min_pair = min(pair_clearance.values())
    tab_c = record['tab_pos'][:,None,None,:]
    q = record['tab_quat'].astype(np.float64)
    q /= np.linalg.norm(q,axis=1,keepdims=True)
    v, w = -q[:,None,None,1:], q[:,None,None,:1]
    # Quarter segments, transformed to the tab frame. The box encloses the
    # actual rounded collider and subtracts the rod radius and sample bound.
    neighbors = cables[:,1:].astype(np.float64)
    segment_len = np.linalg.norm(np.diff(neighbors,axis=2),axis=-1)
    points = [neighbors]
    for s in (.25,.5,.75):
        points.append((1.-s)*neighbors[:,:,:-1] + s*neighbors[:,:,1:])
    p = np.concatenate(points,axis=2) - tab_c
    local = p + 2.*np.cross(v,np.cross(v,p)+w*p)
    d = np.abs(local) - np.asarray(PULL_TAB_SIZE)/2
    box_distance = np.linalg.norm(np.maximum(d,0),axis=-1)+np.minimum(np.max(d,axis=-1),0)
    conservative_tab_gap = float(box_distance.min() - RADIUS - .125*segment_len.max())
    bounds = np.asarray(report['table_collision_bounds_m'])
    def footprint_margin(points):
        xy = points[:,:,:2].reshape(-1,2)
        return float(min((xy-bounds[0,:2]).min(), (bounds[1,:2]-xy).min()))

    initial_footprint_margin = footprint_margin(initialized)
    settled_footprint_margin = footprint_margin(cables[0])
    table_xy = (cables[:,:,:,:2] >= bounds[0,:2]) & (cables[:,:,:,:2] <= bounds[1,:2])
    table_gap = float(np.min(cables[:,:,:,2][table_xy.all(axis=-1)]-bounds[1,2]-RADIUS))
    initial_rms_speed = float(np.sqrt(np.mean(np.sum(record['cable_vel'][0]**2,axis=-1))))
    peak_bend_stretch = report['cable']['max_segment_strain']
    lengths = np.linalg.norm(np.diff(initialized,axis=1),axis=-1).sum(axis=1)
    if not np.allclose(lengths, report['cable']['rest_lengths_m'], rtol=1e-6, atol=1e-8):
        raise RuntimeError('Reported rest lengths disagree with the hashed initialization')
    edges = np.diff(cables[0],axis=1)
    edge_lengths = np.linalg.norm(edges,axis=-1)
    tangents = edges/edge_lengths[:,:,None]
    cosine = np.sum(tangents[:,:-1]*tangents[:,1:],axis=-1)
    angles = np.degrees(np.arccos(np.clip(cosine,-1.,1.)))
    overlaps = crossing_summary(cables[0])
    enough_overlap = (overlaps['stacked_crossings'] >= 60
        and overlaps['target_over'] + overlaps['target_under'] >= 20
        and overlaps['target_over'] >= 5 and overlaps['target_under'] >= 2)
    layout_passed = (cables.shape[2] >= 193 and 1.10 <= lengths[0] <= 1.40
                     and np.all((lengths[1:] >= 1.50) & (lengths[1:] <= 2.0)) and enough_overlap)
    settled = phase == 'ready'
    ready_motion = float(np.linalg.norm(cables[settled]-cables[0],axis=-1).max())
    held = (record['finger_force'][-FPS:] > .10).all(axis=1)
    numerical_geometry_passed = (min_pair >= 1.8*RADIUS-.00025 and table_gap >= -.002
        and conservative_tab_gap > .003 and peak_bend_stretch < .025 and initial_rms_speed < .015
        and min(initial_footprint_margin,settled_footprint_margin) >= .010)
    return dict(minimum_rod_pair_centerline_m=min_pair, min_pair_centerlines_m=pair_clearance,
                minimum_rod_surface_gap_m=min_pair-2*RADIUS,
                minimum_gap_above_rope_barrier_wall_m=min_pair-1.8*RADIUS,
                sampled_rod_table_surface_gap_m=table_gap,
                conservative_sampled_neighbor_rod_to_tab_gap_m=conservative_tab_gap,
                initial_vertex_speed_rms_m_s=initial_rms_speed,
                maximum_ready_drift_m=ready_motion, held_final_second_fraction=float(held.mean()),
                max_segment_strain=peak_bend_stretch, vertices_per_cable=int(cables.shape[2]),
                rest_lengths_m=lengths.tolist(), initial_settled_overlap=overlaps,
                initial_segment_mean_m=edge_lengths.mean(axis=1).tolist(),
                initial_segment_max_m=edge_lengths.max(axis=1).tolist(),
                initial_segment_turn_mean_deg=angles.mean(axis=1).tolist(),
                initial_segment_turn_max_deg=angles.max(axis=1).tolist(),
                initial_cable_table_xy_margin_m=initial_footprint_margin,
                settled_cable_table_xy_margin_m=settled_footprint_margin,
                long_dense_layout_passed=bool(layout_passed),
                numerical_geometry_passed=bool(numerical_geometry_passed),
                geometry_passed=bool(numerical_geometry_passed and held.mean() > .90 and layout_passed))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    args = ap.parse_args()
    report_path = args.data/'report.json'
    report = json.loads(report_path.read_text())
    model_sha = sha(Path(__file__).with_name('model.py'))
    if model_sha != report['source_sha256']['model.py']:
        raise RuntimeError('Audit model does not match the recorded physical source')
    path = args.data/'trajectory.npz'
    initialized = args.data/'initial_cables.npy'
    if sha(path) != report['trajectory_sha256'] or sha(initialized) != report['initial_cables_sha256']:
        raise RuntimeError('Changed trajectory or initial cable array')
    with np.load(path,allow_pickle=False) as rec:
        result = dict(trajectory_sha256=sha(path), initial_cables_sha256=sha(initialized),
                      run_report_sha256=sha(report_path), model_source_sha256=model_sha,
                      audit_source_sha256=sha(__file__),
                      task=score(rec), geometry=audit(rec,report,np.load(initialized,allow_pickle=False)))
    result['accepted'] = bool(report['complete'] and not report['errors'] and
                              result['task']['passed'] and result['geometry']['geometry_passed'])
    with args.out.open('x') as stream:
        json.dump(result,stream,indent=2,allow_nan=False)
        stream.write('\n')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
