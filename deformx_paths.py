"""Centralized, environment-configurable path resolution for DeformX.

No path in this project is hardcoded to a specific machine. Every location can
be overridden with an environment variable and otherwise falls back to a
sensible repository-relative default, so the codebase is portable for
open-source use (it works right after ``git clone`` without editing sources).

Environment variables (all optional):

  DEFORMX_ROOT          Repository root. Default: auto-detected from this file.
  DEFORMX_ASSET_ROOT    Root of large render/simulation assets (USDs, textures,
                        trajectories). Default: ``<repo>/asset_wireseg36k``.
  DEFORMX_DATA_ROOT     Root of local working data (npz files, renders, ...).
                        Default: ``<repo>/data``.
  DEFORMX_OUTPUT_ROOT   Root of generated outputs. Default: ``<repo>/output``.
  DEFORMX_WIRE_USD      Path to the wire USD asset used by RL/visualization.
                        Default: ``<repo>/usd/wire/
                        wire_yellow_s20_r0.005_l1_smooth.usdc``.
  DEFORMX_HDR           Path to an HDR dome-light image for datacenter renders.
                        Default: ``<asset_root>/hdr/dome.hdr``.
  ISAAC_PYTHON          Path to Isaac Sim's ``python.sh`` launcher.
                        Default: ``isaacsim/python.sh`` (resolved on PATH/cwd).
  ISAAC_ASSETS_ROOT     Root of a local Isaac ``Assets/Isaac`` tree (used to
                        locate bundled robot USDs). Optional; if unset we ask
                        Isaac Sim for its assets root and finally fall back to
                        the public Omniverse S3 mirror.
"""

from __future__ import annotations

import os
from pathlib import Path


def _resolve_repo_root() -> Path:
    env = os.environ.get("DEFORMX_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    return Path(__file__).resolve().parent


REPO_ROOT = _resolve_repo_root()


def env_path(var: str, *default_parts: object) -> Path:
    """Return ``$var`` as a Path if set, else ``REPO_ROOT`` joined with defaults."""
    val = os.environ.get(var)
    if val:
        return Path(val).expanduser()
    return REPO_ROOT.joinpath(*[str(p) for p in default_parts])


ASSET_ROOT = env_path("DEFORMX_ASSET_ROOT", "asset_wireseg36k")
DATA_ROOT = env_path("DEFORMX_DATA_ROOT", "data")
OUTPUT_ROOT = env_path("DEFORMX_OUTPUT_ROOT", "output")
DEFAULT_WIRE_USD = REPO_ROOT / "usd" / "wire" / "wire_yellow_s20_r0.005_l1_smooth.usdc"

#: Path to Isaac Sim's ``python.sh`` launcher (used by docs/subprocess callers).
ISAAC_PYTHON = os.environ.get("ISAAC_PYTHON", "isaacsim/python.sh")


def wire_usd() -> str:
    """Path to the wire USD asset used by the RL and visualization demos."""
    val = os.environ.get("DEFORMX_WIRE_USD")
    if val:
        return str(Path(val).expanduser())
    return str(DEFAULT_WIRE_USD)


def hdr_path() -> str:
    """Path to an HDR dome-light image used by datacenter renders."""
    val = os.environ.get("DEFORMX_HDR")
    if val:
        return str(Path(val).expanduser())
    return str(ASSET_ROOT / "hdr" / "dome.hdr")


# Robot USD locations relative to an Isaac "Assets/Isaac" tree, ur5e preferred.
_UR_ASSET_RELPATHS = (
    "4.5/Isaac/Robots/UniversalRobots/ur5e/ur5e.usd",
    "5.1/Isaac/Robots/UniversalRobots/ur5e/ur5e.usd",
    "4.5/Isaac/Robots/UniversalRobots/ur5/ur5.usd",
    "5.1/Isaac/Robots/UniversalRobots/ur5/ur5.usd",
)

_S3_ASSETS_ROOT = (
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac"
)


def isaac_assets_roots() -> list[str]:
    """Candidate roots of the Isaac ``Assets/Isaac`` tree, most-preferred first."""
    roots: list[str] = []
    env = os.environ.get("ISAAC_ASSETS_ROOT")
    if env:
        roots.append(env.rstrip("/"))
    # Best-effort: ask Isaac Sim itself where its assets live (API name varies
    # across Isaac versions, so try the known import paths and ignore failures).
    for module_name, attr in (
        ("isaacsim.storage.native", "get_assets_root_path"),
        ("omni.isaac.nucleus", "get_assets_root_path"),
        ("omni.isaac.core.utils.nucleus", "get_assets_root_path"),
    ):
        try:
            module = __import__(module_name, fromlist=[attr])
            resolver = getattr(module, attr)
            resolved = resolver()
            if resolved:
                roots.append(resolved.rstrip("/") + "/Isaac")
        except Exception:
            continue
    return roots


def resolve_ur_robot_usd(prefer: str = "ur5e") -> str:
    """Resolve a UR robot USD path portably.

    Order: ``ISAAC_ASSETS_ROOT`` / Isaac Sim's assets root (local files) → the
    public Omniverse S3 mirror (streamable, always available).
    """
    rels = sorted(_UR_ASSET_RELPATHS, key=lambda p: prefer not in p)
    for root in isaac_assets_roots():
        for rel in rels:
            candidate = f"{root}/{rel}"
            if os.path.exists(candidate):
                return candidate
    return f"{_S3_ASSETS_ROOT}/{rels[0]}"
