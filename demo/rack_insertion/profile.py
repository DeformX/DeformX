"""Validate a generated rack profile and its file hashes before starting Isaac."""

import json
from pathlib import Path

from .assets import sha
from .panel import load_panel_manifest, verify_usd_dependencies


PROFILE_KIND = "deformx-rack-insertion-v1"
REQUIRED_ASSETS = {
    "connector", "plug", "latch", "socket", "panel", "robot", "table", "termination_visual",
}


def load_profile(path):
    path = Path(path).resolve()
    profile = json.loads(path.read_text())
    if not isinstance(profile, dict) or profile.get("kind") != PROFILE_KIND:
        raise ValueError("Unsupported rack insertion profile")
    assets = profile["assets"]
    if REQUIRED_ASSETS != assets.keys():
        raise ValueError("Rack insertion profile is missing required assets or has unknown asset labels")
    verify_usd_dependencies(list(assets.values()), base=path.parent)
    verify_usd_dependencies(profile["dependencies"], base=path.parent)
    load_panel_manifest(assets["panel"]["path"])
    profile["path"] = str(path)
    profile["sha256"] = sha(path)
    return profile
