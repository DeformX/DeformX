"""Scene geometry, the scripted joint-space goal schedule and geometric scoring."""
from itertools import combinations

import numpy as np


RADIUS = .0033
VERTICES = 193
CABLE_COUNT = 5
DENSITY = .040
EA = 900.
EI = .0010
EXIT = (0., .023, 0.)
PULL_TAB_SIZE = (.018, .046, .024)
PULL_TAB_MASS = .018
PILE_SHIFT_Y = .640
PULL_TAB_XY = (.38, -.075 + PILE_SHIFT_Y)
TABLE_POS = (.12, .34, 1.003)
ROBOT_POS = (0., .025, 1.003)
COLORS = ((.015, .57, .87), (.84, .29, .075), (.52, .22, .64), (.14, .54, .32), (.44, .48, .53))
FPS = 30
END_TIME = 37.4

# Each open spiral is offset and rippled differently. The target begins in
# front of the bundle; its body crosses three strands above and one below.
COILS = (
    ((.342, .242), (.134, .085), (.95, 1.05), -1.9, -1.53, .002, .3),
    ((.295, .215), (.080, .185), (.85, 1.00), 2.1, 2.25, .008, 1.0),
    ((.353, .237), (.075, .168), (1.00, 1.00), -1.1, -2.30, .010, 1.7),
    ((.385, .180), (.070, .162), (.83, 1.15), 2.7, 2.42, .007, .8),
    ((.343, .213), (.080, .172), (.91, 1.05), -.7, -2.15, .009, 2.8),
)


def coil(spec):
    center, radii, scale, start, turns, ripple, phase = spec
    u = np.linspace(0., 1., 1001)
    a = start + turns * (2*np.pi) * u
    r = np.interp(u, [0., 1.], radii) + ripple*np.sin(3*a + phase)*np.sin(np.pi*u)
    return np.asarray(center) + [0., PILE_SHIFT_Y] + np.asarray(scale) * (
        r[:,None] * np.column_stack([np.cos(a), np.sin(a)]))


def resample_curve(polyline):
    """Space actual solver vertices by arc length along the initialized route."""
    s = np.r_[0., np.cumsum(np.linalg.norm(np.diff(polyline, axis=0), axis=1))]
    samples = np.linspace(0., s[-1], VERTICES)
    return np.column_stack([np.interp(samples, s, v) for v in polyline.T])


def initial_positions(tab_pos, table_z):
    points = []
    exit_pos = np.asarray(tab_pos) + EXIT
    for i, (spec, layer) in enumerate(zip(COILS, (3, 0, 2, 1, 4))):
        polyline = coil(spec)
        if i == 0:
            # Cubic lead-in joins the coil with its own entry tangent.
            start, end = exit_pos[:2], polyline[0]
            tangent = polyline[1] - end
            tangent /= np.linalg.norm(tangent)
            a, b = start + [0., .080], end - .070*tangent
            u = np.linspace(0., 1., 161, endpoint=False)[:, None]
            v = 1-u
            lead = v**3*start + 3*v*v*u*a + 3*v*u*u*b + u**3*end
            polyline = np.vstack([lead, polyline])
        xy = resample_curve(polyline)
        z = np.full(VERTICES, table_z + RADIUS + .0008 + layer * .0074)
        if i == 0:
            z[:10] = np.linspace(exit_pos[2], z[9], 10)
        points.append(np.column_stack([xy, z]))
    return np.asarray(points, dtype=np.float32)


def crossing_summary(cables, maximum_height_gap=.040):
    """Count proper projected inter-cable segment crossings and their Z order.

    The result describes overlap in a frame. It does not infer a knot type.
    """
    rows = []
    for i, j in combinations(range(len(cables)), 2):
        a, b = cables[i,:-1], cables[j,:-1]
        u, v = np.diff(cables[i],axis=0), np.diff(cables[j],axis=0)
        w = b[None,:,:] - a[:,None,:]
        det = u[:,None,0]*v[None,:,1] - u[:,None,1]*v[None,:,0]
        nonparallel = np.abs(det) > 1e-12
        t = np.divide(w[:,:,0]*v[None,:,1]-w[:,:,1]*v[None,:,0], det,
                      out=np.zeros_like(det), where=nonparallel)
        s = np.divide(w[:,:,0]*u[:,None,1]-w[:,:,1]*u[:,None,0], det,
                      out=np.zeros_like(det), where=nonparallel)
        proper = nonparallel & (t > 0) & (t < 1) & (s > 0) & (s < 1)
        dz = (a[:,None,2] + t*u[:,None,2] - b[None,:,2] - s*v[None,:,2])[proper]
        near = dz[np.abs(dz) <= maximum_height_gap]
        rows.append(dict(pair=[i,j], projected=int(len(dz)), stacked=int(len(near)),
                         first_over=int((near>0).sum()), first_under=int((near<0).sum())))
    target = [row for row in rows if row['pair'][0] == 0]
    return dict(projected_crossings=sum(row['projected'] for row in rows),
        stacked_crossings=sum(row['stacked'] for row in rows),
        target_projected=sum(row['projected'] for row in target),
        target_over=sum(row['first_over'] for row in target),
        target_under=sum(row['first_under'] for row in target),
        maximum_height_gap_m=maximum_height_gap, pairs=rows)


def command(t, home, grip):
    """Generate Cartesian IK goals and finger targets from the known pickup pose."""
    approach = grip + [0, 0, .13]
    lifted = grip + [0, .025, .110]
    swept = np.array([.455, .085, 1.205])
    drawn = np.array([.430, -.455, 1.420])
    clear = np.array([.360, -.525, 1.605])
    swept_clear = np.array([.180, -.550, 1.655])
    stages = (
        (0., 1., home, home, .04, .04, 'ready'),
        (1., 3.3, home, approach, .04, .04, 'approach'),
        (3.3, 5.0, approach, grip, .04, .04, 'descend'),
        (5.0, 6.2, grip, grip, .04, .002, 'close'),
        (6.2, 8.4, grip, lifted, .002, .002, 'lift'),
        (8.4, 15.5, lifted, swept, .002, .002, 'pull'),
        (15.5, 25.6, swept, drawn, .002, .002, 'pull'),
        (25.6, 30.6, drawn, clear, .002, .002, 'pull'),
        (30.6, 34.6, clear, swept_clear, .002, .002, 'pull'),
        (34.6, END_TIME, swept_clear, swept_clear, .002, .002, 'hold'),
    )
    for start, end, p, q, opening0, opening1, phase in stages:
        if t < end or phase == 'hold':
            u = np.clip((t - start) / (end - start), 0., 1.)
            ease = u*u*(3. - 2.*u)
            return (1. - ease)*p + ease*q, (1. - ease)*opening0 + ease*opening1, phase


def closest_segments(p, q):
    """All segment-to-segment distances, including endpoint-projection cases."""
    p0, p1 = p[..., :-1, None, :], p[..., 1:, None, :]
    q0, q1 = q[..., None, :-1, :], q[..., None, 1:, :]
    return segment_pair_distances(p1-p0, q1-q0, p0-q0)


def segment_pair_distances(u, v, w):
    dot = lambda a, b: np.sum(a*b, axis=-1)
    a, b, c, d, e = dot(u,u), dot(u,v), dot(v,v), dot(u,w), dot(v,w)
    det = a*c - b*b
    with np.errstate(divide='ignore', invalid='ignore'):
        s = np.divide(b*e-c*d, det, out=np.zeros_like(det), where=det > 1e-20)
        t = np.divide(a*e-b*d, det, out=np.zeros_like(det), where=det > 1e-20)
        internal = (det > 1e-20) & (0 <= s) & (s <= 1) & (0 <= t) & (t <= 1)
        squared = np.where(internal, dot(w+s[...,None]*u-t[...,None]*v, w+s[...,None]*u-t[...,None]*v), np.inf)
        for s, t in [(0., np.clip(e/c,0.,1.)), (1., np.clip((e+b)/c,0.,1.)),
                     (np.clip(-d/a,0.,1.),0.), (np.clip((b-d)/a,0.,1.),1.)]:
            delta = w + np.asarray(s)[..., None]*u - np.asarray(t)[..., None]*v
            squared = np.minimum(squared, dot(delta,delta))
    return np.sqrt(squared)


def minimum_segment_distance(p, q):
    """Exact minimum via a conservative midpoint bound on candidate pairs."""
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    u, v = np.diff(p, axis=0), np.diff(q, axis=0)
    a, b = .5*(p[:-1]+p[1:]), .5*(q[:-1]+q[1:])
    dx = a[:,None,0] - b[None,:,0]
    dy = a[:,None,1] - b[None,:,1]
    dz = a[:,None,2] - b[None,:,2]
    distance2 = dx*dx + dy*dy + dz*dz
    # The closest midpoint pair gives a valid upper bound. Any segment pair
    # farther than that bound plus both half lengths cannot improve on it.
    reach = .5*(np.linalg.norm(u,axis=-1)[:,None] + np.linalg.norm(v,axis=-1)[None,:])
    limit = np.sqrt(distance2.min()) + reach + 1e-12
    i, j = np.nonzero(distance2 <= limit**2)
    return float(segment_pair_distances(u[i],v[j],p[i]-q[j]).min())


def score(record):
    cables, phase = record['cable'], record['phase']
    contacts = (record['finger_force'] > .10).all(axis=1)
    distance = np.asarray([[minimum_segment_distance(frame[0],strand) for strand in frame[1:]] for frame in cables])
    displacement = np.linalg.norm(cables[:, 1:] - cables[0, 1:], axis=-1).max(axis=-1)
    started_pull = np.flatnonzero(phase == 'lift')
    finished = np.flatnonzero(phase == 'hold')
    pull = (phase == 'lift') | (phase == 'pull')
    contact = distance < 2 * RADIUS + .00025
    tab_distance = np.linalg.norm(record['tab_pos'] - record['tab_pos'][0], axis=1)
    tip_displacement = float(np.linalg.norm(cables[-1,0,-1]-cables[0,0,-1]))
    centroid_displacement = float(np.linalg.norm((cables[-1,0]-cables[0,0]).mean(axis=0)))
    result = dict(bilateral_grasp_samples=int(np.count_nonzero(contacts & (phase=='close'))),
        bilateral_pull_fraction=float(contacts[pull].mean()) if np.any(pull) else 0.,
        tab_displacement_m=float(tab_distance[-1]), minimum_target_neighbor_centerline_m=float(distance.min()),
        target_free_tip_displacement_m=tip_displacement, target_centerline_centroid_displacement_m=centroid_displacement,
        final_target_neighbor_centerline_m=float(distance[-1].min()),
        near_contact_pairs_per_frame_peak=int(contact.sum(axis=1).max()),
        near_contact_samples_during_pull=int(contact[pull].any(axis=1).sum()),
        max_neighbor_vertex_displacements_m=displacement.max(axis=0).tolist())
    if len(started_pull) and len(finished):
        moving = np.linalg.norm(cables[:,1:] - cables[started_pull[0],1:], axis=-1).max(axis=-1)
        result['neighbor_displacement_during_pull_m'] = moving[pull].max(axis=0).tolist()
        result['neighbor_displacement_while_target_near_m'] = float(np.where(contact & pull[:,None], moving, 0.).max())
        result['final_hold_clearance_min_m'] = float(distance[finished[-FPS:]].min())
        result['passed'] = bool(result['bilateral_grasp_samples'] >= 5 and result['bilateral_pull_fraction'] >= .90
            and tab_distance[-1] >= .35 and tip_displacement >= .12
            and len(finished) >= FPS and result['final_hold_clearance_min_m'] >= .035
            and result['near_contact_samples_during_pull'] >= 5
            and result['neighbor_displacement_while_target_near_m'] >= .003)
    else:
        result['passed'] = False
    return result
