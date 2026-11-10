"""Physical bookkeeping and acceptance criteria, independent of rendering."""

import numpy as np


def reaction_from_momentum(p0, p1, mass, gravity, contact_force, dt):
    """Force BY the cable ON its sole point attachment, in world coordinates.

    p0/p1 are total FREE-vertex momenta, mass excludes the pinned vertex, and
    contact_force must be a true frame-mean force ON all free vertices. Native
    final-iterate contact-law readout does not generally satisfy that premise;
    its use in the retained ideal-coupling diagnostic is not force validation.
    Native drag/damping and every other unaccounted external force must be zero.
    A point attachment transmits this force, but no independently prescribed moment.
    """
    if not np.isfinite(dt) or dt <= 0 or not np.isfinite(mass) or mass <= 0:
        raise ValueError("dt and free cable mass must be finite and positive")
    vectors = [np.asarray(v, dtype=np.float64) for v in (p0, p1, gravity, contact_force)]
    if any(v.shape != (3,) or not np.isfinite(v).all() for v in vectors):
        raise ValueError("momentum, gravity and force must be finite world-space 3-vectors")
    before, after, g, external = vectors
    return mass * g + external - (after - before) / dt


def score_grasp(times, z, contact_left, contact_right, phase):
    """Audit lift/hold/release from measured plug height and finger/plug forces.

    Inputs are a single trial, in seconds/metres/newtons. Contact arrays contain
    force magnitudes for the PLUG-filtered pair, not a finger's total scene load.
    A grasp is only demonstrated after a later open-finger release also succeeds.
    """
    t, height, left, right = [np.asarray(v, dtype=float) for v in
                              (times, z, contact_left, contact_right)]
    labels = np.asarray(phase)
    if (t.ndim != 1 or len(t) < 2 or any(v.shape != t.shape for v in (height, left, right, labels))
            or any(not np.isfinite(v).all() for v in (t, height, left, right))
            or not np.all(np.diff(t) > 0) or np.any(left < 0) or np.any(right < 0)):
        raise ValueError("expected aligned finite samples, increasing time and force magnitudes")
    baseline = height[0]
    lifted = height - baseline >= 0.08
    bilateral = (left >= 0.05) & (right >= 0.05)
    held = (labels == "hold") & lifted & bilateral
    max_interval = 2 * np.median(np.diff(t))
    duration = best = 0.0
    for i in range(1, len(t)):
        duration = duration + t[i] - t[i - 1] if held[i] and held[i - 1] and t[i] - t[i - 1] <= max_interval else 0.0
        best = max(best, duration)
    release = np.flatnonzero(labels == "release")
    drop = float(height[release[0]] - np.min(height[release])) if len(release) > 1 else 0.0
    released = bool(len(release) > 1 and drop >= 0.05
                    and left[release[-1]] < 0.05 and right[release[-1]] < 0.05)
    return dict(passed=bool(best >= 0.5 - 1e-9 and released),
                lift_m=float(np.max(height) - baseline), held_s=float(best),
                release_drop_m=drop, released=released,
                bilateral_samples=int(np.count_nonzero(bilateral)))


def score_insertion(times, depth, lateral_error, axis_error, socket_force, finger_force, phase, seated_depth, *, tcp_z=None):
    """Require seating and retention after measured hand clearance.

    TCP must withdraw at least 50 mm in world +Z from its release-entry position,
    the controller's withdrawal direction and the controlled-release clearance.
    Seating, socket contact, open fingers and that clearance must then hold for
    at least 0.5 s through the final sample. Command targets are not evidence.
    Post-open retention remains separately reported. This does not establish
    electrical connection or calibrated latch force.
    """
    t,d,l,a,s,f=[np.asarray(x,dtype=float) for x in (times,depth,lateral_error,axis_error,socket_force,finger_force)]
    labels=np.asarray(phase)
    tcp=np.asarray(tcp_z,dtype=float)
    if t.ndim!=1 or len(t)<2 or any(x.shape!=t.shape for x in (d,l,a,s,f,labels,tcp)) or any(not np.isfinite(x).all() for x in (t,d,l,a,s,f,tcp)) or not np.all(np.diff(t)>0) or not np.isfinite(seated_depth) or seated_depth<=0:
        raise ValueError('aligned finite state, measured TCP z and positive seated depth required')
    release=np.flatnonzero(labels=='release')
    retained=(np.abs(d-seated_depth)<=.001)&(l<=.001)&(a<=.08)&(s>=.01)&(f<.05)
    tail=release[-1] if len(release) else None
    duration=withdrawn_duration=retreat=final_withdrawal=0.
    if tail is not None:
        clearance=tcp-tcp[release[0]]
        retreat=float(np.max(clearance[release]))
        final_withdrawal=float(clearance[tail])
        for independent in (False,True):
            valid=retained & (labels=='release')
            if independent:valid=valid & (clearance>=.05)
            elapsed=0.
            if tail==len(t)-1 and valid[tail]:
                first=tail
                while first>0 and valid[first-1]:first-=1
                elapsed=float(t[tail]-t[first])
            if independent:withdrawn_duration=elapsed
            else:duration=elapsed
    return dict(passed=bool(withdrawn_duration>=.5-1e-9),retained_after_release_s=duration,
                retained_after_withdrawal_s=withdrawn_duration,actual_retreat_m=retreat,
                final_withdrawal_m=final_withdrawal,withdrawal_threshold_m=.05,
                withdrawal_direction='world +Z from release-entry measured TCP',
                final_depth_m=float(d[-1]),seated_depth_m=float(seated_depth),
                final_lateral_error_m=float(l[-1]),final_axis_error_rad=float(a[-1]),
                final_socket_force_N=float(s[-1]),final_finger_force_N=float(f[-1]))


def score_acquisition(times,z,contact_left,contact_right,phase):
    """Contact acquisition/lift/hold; insertion retention is scored separately."""
    score=score_grasp(times,z,contact_left,contact_right,phase)
    return dict(passed=bool(score['held_s']>=.5-1e-9),lift_m=score['lift_m'],held_s=score['held_s'],bilateral_samples=score['bilateral_samples'])



def score_controlled_release(times,z,tcp_z,contact_left,contact_right,support_force_z,phase):
    """Controlled place/open/retreat; this is deliberately separate from free-drop stress."""
    t,h,tcp,left,right,support=[np.asarray(x,dtype=float) for x in (times,z,tcp_z,contact_left,contact_right,support_force_z)]
    labels=np.asarray(phase)
    if len(t)<2 or any(x.shape!=t.shape for x in (h,tcp,left,right,support,labels)) or any(not np.isfinite(x).all() for x in (t,h,tcp,left,right,support)) or not np.all(np.diff(t)>0):
        raise ValueError('aligned finite measured release states required')
    release=np.flatnonzero(labels=='release')
    good=(labels=='release')&(np.abs(h-h[0])<.01)&(left<.05)&(right<.05)&(support>.05)
    held=0.
    if len(release) and good[-1]:
        first=len(t)-1
        while first>0 and good[first-1]:first-=1
        held=float(t[-1]-t[first])
    retreat=float(np.max(tcp[release])-tcp[release[0]]) if len(release) else 0.
    return dict(passed=bool(np.any(labels=='place') and held>=.5-1e-9 and retreat>=.05),mode='controlled_place_open_retreat',
        supported_after_open_s=held,actual_retreat_m=retreat,final_support_normal_N=float(support[-1]),
        final_height_error_m=float(h[-1]-h[0]),free_drop_stress_validated=False)
