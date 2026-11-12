"""Patch cords already installed between the patch panel and the unit two rack units below it.

Every cord is one rod in the same solver batch as the cable the robot handles, so cords and cable
touch each other through the solver's rod-rod contact. Both ends of a cord are held where its
plugs sit in their ports; the two vertices nearest each end are fixed so that the cord leaves the
plug along the plug axis, like a moulded boot.
"""
import numpy as np

PITCH_M = 0.018            # port spacing of the 24-port panel
UNIT_DROP_M = 0.0889       # two rack units
ACTIVE_PORT = 7
# Ports 5, 6, 8 and 9 stay empty: the gripper fingers need that room around port 7.
PORTS = (1, 2, 3, 4, 10, 11, 12, 13, 15, 16, 17, 18, 20, 21)
# Typical patch-cord colours, RGB.
COLOURS = {'navy': (0.05, 0.13, 0.36), 'grey': (0.52, 0.55, 0.58), 'amber': (0.80, 0.50, 0.07)}
# Blocks of one colour, the way a rack is patched by function.
PORT_COLOURS = ('navy',) * 4 + ('grey',) * 4 + ('navy',) * 4 + ('amber',) * 2


def port_offset_x(port):
    return (port - ACTIVE_PORT) * PITCH_M


def seated_plug_position(socket_position, manifest, port, lower=False):
    """Plug origin when seated in a port of the panel, or of the unit below it."""
    position = np.asarray(socket_position, float) + np.asarray(manifest['seated_plug_minus_socket'], float)
    return position + [port_offset_x(port), 0., -UNIT_DROP_M if lower else 0.]


def loop(top, bottom, length, vertices):
    """A hanging loop from top to bottom that leaves both ends horizontally toward -y."""
    top, bottom = np.asarray(top, float), np.asarray(bottom, float)
    theta = np.linspace(0, np.pi, 4001)
    centre, half = (top + bottom) / 2, (top[2] - bottom[2]) / 2

    def curve(reach):
        return np.c_[np.interp(theta, [0, np.pi], [top[0], bottom[0]]),
                     centre[1] - reach * np.sin(theta), centre[2] + half * np.cos(theta)]

    def arc(reach):
        return np.linalg.norm(np.diff(curve(reach), axis=0), axis=1).sum()

    low, high = 0.0, 2.0
    if not arc(low) < length < arc(high):
        raise ValueError('cord length does not fit between its two ports')
    for _ in range(60):
        mid = (low + high) / 2
        low, high = (mid, high) if arc(mid) < length else (low, mid)
    dense = curve((low + high) / 2)
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(dense, axis=0), axis=1))]
    return np.stack([np.interp(np.linspace(0, s[-1], vertices), s, dense[:, k]) for k in range(3)], axis=1)


def cords(socket_position, manifest, vertices):
    """Initial vertices [C, V, 3], fixed-vertex mask [C, V] and a description of every cord."""
    exit_offset = np.asarray(manifest['cable_exit_offset'], float)
    rng = np.random.default_rng(7)
    points, description = [], []
    for port, colour in zip(PORTS, PORT_COLOURS):
        upper = seated_plug_position(socket_position, manifest, port)
        lower = seated_plug_position(socket_position, manifest, port, lower=True)
        length = float(rng.uniform(0.20, 0.26))
        points.append(loop(upper + exit_offset, lower + exit_offset, length, vertices))
        description.append(dict(port=port, colour=colour, colour_rgb=COLOURS[colour], length_m=length,
                                upper_plug_position_m=upper.tolist(), lower_plug_position_m=lower.tolist()))
    points = np.asarray(points)
    fixed = np.zeros(points.shape[:2], bool)
    fixed[:, [0, 1, -2, -1]] = True
    return points, fixed, description


def new_cord_route(root, far_exit, vertices, tray_top, turn_radius=0.05, return_x=0.30, advance_y=0.345):
    """Initial shape of the cord the robot installs, whose far plug already sits in the lower unit.

    From the plug on the tray the cord runs back, turns on the tray toward the robot side, comes
    forward beside the plug (outside the reach of the open fingers) and climbs to the far port.
    """
    root, far_exit = np.asarray(root, float), np.asarray(far_exit, float)
    level = tray_top + 0.002
    # Straight out of the plug for the length of its boot, then down onto the tray.
    boot_end = root - [0, 0.014, 0]
    back = np.concatenate([np.linspace(root, boot_end, 8), np.linspace(boot_end, [root[0], root[1] - 0.025, level], 14)[1:]])
    centre = np.array([root[0] - turn_radius, root[1] - 0.025])
    angle = np.linspace(0, -np.pi, 80)
    turn = np.c_[centre[0] + turn_radius * np.cos(angle), centre[1] + turn_radius * np.sin(angle), np.full(80, level)]
    turn[:, 0] = np.interp(np.linspace(0, 1, 80), [0, 1], [1, (root[0] - return_x) / (2 * turn_radius)]) * (turn[:, 0] - root[0]) + root[0]
    start = np.array([return_x, turn[-1, 1], level])
    forward = np.linspace(start, [return_x, advance_y, level], 60)
    t = np.linspace(0, 1, 200)[:, None]
    p0, p3 = forward[-1], far_exit
    p1, p2 = p0 + [0, 0.06, 0], p3 - [0, 0.06, 0]
    climb = (1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * p1 + 3 * (1 - t) * t ** 2 * p2 + t ** 3 * p3
    dense = np.concatenate([back, turn[1:], forward[1:], climb[1:]])
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(dense, axis=0), axis=1))]
    points = np.stack([np.interp(np.linspace(0, s[-1], vertices), s, dense[:, k]) for k in range(3)], axis=1)
    return points, float(s[-1])
