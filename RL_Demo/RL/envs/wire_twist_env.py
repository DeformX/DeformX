"""Backward-compatible alias for the swing-wire task env."""

from __future__ import annotations

from RL.envs.wire_swing_env import WireSwingEnv


class WireTwistEnv(WireSwingEnv):
    pass
