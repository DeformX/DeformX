"""Dummy Isaac-side command generation helpers."""

from __future__ import annotations

import numpy as np

from .models import FrameState


def sine_frame_state(command_time: float, amp: float = 0.1, freq: float = 1.0) -> FrameState:
    """Sine translation in world-Y with fixed orientation and zero angular motion."""
    omega = 2.0 * np.pi * freq
    pos = np.array([0.0, amp * np.sin(omega * command_time), 0.0])
    vel = np.array([0.0, amp * omega * np.cos(omega * command_time), 0.0])
    acc = np.array([0.0, -amp * omega * omega * np.sin(omega * command_time), 0.0])
    default_director = np.array(
        [
            [0.0, 0.0, 1.0],
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
        ]
    )
    return FrameState(
        position=pos,
        director=default_director,
        velocity=vel,
        acceleration=acc,
        omega=np.zeros(3),
        alpha=np.zeros(3),
    )
