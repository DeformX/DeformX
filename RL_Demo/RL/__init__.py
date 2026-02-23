from __future__ import annotations

import sys
from pathlib import Path


def _prepend_if_exists(path: Path):
    if path.is_dir():
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
PYELASTICA_MESH_ROOT = WORKSPACE_ROOT / "PyElastica-Mesh"

_prepend_if_exists(PYELASTICA_MESH_ROOT)
