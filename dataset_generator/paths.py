"""Path resolution for the dataset generators (drop/hang and datacenter).

No path in this project is hardcoded to a specific machine. Every location can
be overridden with an environment variable and otherwise falls back to a
repository-relative default, so the code runs right after ``git clone``.

Environment variables (all optional):

  DEFORMX_ROOT          Repository root. Default: auto-detected from this file.
  DEFORMX_ASSET_ROOT    Root of large render/simulation assets (USDs, textures,
                        trajectories). Default: ``<repo>/asset_wireseg36k``.
  DEFORMX_DATA_ROOT     Root of local working data (npz files, renders, ...).
                        Default: ``<repo>/data``.
  DEFORMX_OUTPUT_ROOT   Root of generated outputs. Default: ``<repo>/output``.
  DEFORMX_HDR           Path to an HDR dome-light image for datacenter renders.
                        Default: ``<asset_root>/hdr/dome.hdr``.
  ISAAC_PYTHON          Isaac Sim's interpreter: its ``python.sh`` launcher, or
                        just ``python`` for a pip install.
                        Default: ``isaacsim/python.sh``.

The demos resolve their own paths in ``demo/common/paths.py``.
"""

from __future__ import annotations

import os
from pathlib import Path


def _resolve_repo_root() -> Path:
    env = os.environ.get("DEFORMX_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    return Path(__file__).resolve().parents[1]


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

#: Isaac Sim's interpreter (used by docs/subprocess callers).
ISAAC_PYTHON = os.environ.get("ISAAC_PYTHON", "isaacsim/python.sh")


def hdr_path() -> str:
    """Path to an HDR dome-light image used by datacenter renders."""
    val = os.environ.get("DEFORMX_HDR")
    if val:
        return str(Path(val).expanduser())
    return str(ASSET_ROOT / "hdr" / "dome.hdr")
