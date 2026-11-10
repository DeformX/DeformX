"""Widen the Blender presentation frame without moving rope grips or carrier paths.

Generate the original hardware with blender_assets.py first, then run in Blender:

    blender -b --factory-startup --python-exit-code 1 \
      --python demo/braiding_machine/widen_frame.py -- \
      --assets runs/braid-hardware --out runs/braid-hardware-wide --width 2.1

Only display geometry changes. The saved trajectory, grip locations, carrier
assets and model-source binding remain unchanged.
"""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import socket
import sys

import bpy
from mathutils import Vector


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def snapshot(obj):
    return {
        "matrix": [float(x) for row in obj.matrix_world for x in row],
        "vertices": [[float(x) for x in v.co] for v in obj.data.vertices],
    }


def bounds():
    points = []
    depsgraph = bpy.context.evaluated_depsgraph_get()
    for obj in bpy.context.scene.objects:
        if obj.type != "MESH" or obj.name in ("StudioFloor", "StudioBackdrop", "FrontLogo"):
            continue
        evaluated = obj.evaluated_get(depsgraph)
        points.extend(evaluated.matrix_world @ Vector(corner) for corner in evaluated.bound_box)
    return [[min(p[i] for p in points) for i in range(3)],
            [max(p[i] for p in points) for i in range(3)]]


def resize(obj, x=1., y=1.):
    # dimensions is evaluated lazily; consecutive component assignments can
    # reset the previous axis before the dependency graph updates.
    obj.scale.x *= x
    obj.scale.y *= y
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--width", type=float, default=2.1, help="footing width in metres")
    args = parser.parse_args(argv)
    source, out = args.assets.resolve(), args.out.resolve()
    if out.exists():
        parser.error("choose a fresh output directory")
    manifest_path = source / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["units"] != "meters" or manifest["meters_per_unit"] != 1.:
        raise ValueError("expected original SI Blender hardware")
    for name, digest in manifest["files"].items():
        if Path(name).name != name or sha(source / name) != digest:
            raise ValueError(f"source asset hash mismatch: {name}")

    bpy.ops.wm.open_mainfile(filepath=str(source / "frame.blend"))
    objects = bpy.context.scene.objects
    footing = objects.get("Footing")
    if footing is None or not .98 < footing.dimensions.x < 1.00:
        raise ValueError("expected the original 0.99 m wide frame")
    if not footing.dimensions.x < args.width <= 3.:
        raise ValueError("width must exceed the original frame width and be at most 3 m")
    original_bounds = bounds()
    protected = {obj.name: snapshot(obj) for obj in objects
                 if obj.type == "MESH" and obj.name.startswith(
                     ("FixedGrip", "Anchor", "CarrierLane", "DriveCassette"))}
    if not protected:
        raise ValueError("missing protected grip and carrier-path geometry")
    factor = args.width / float(footing.dimensions.x)
    depth_factor = .80 / float(footing.dimensions.y)
    front_shift = -.5 * (.80 - float(footing.dimensions.y))
    horizontal = {"OverheadGantry", "GantryTop", "TopBumper", "TopSignalStrip"}
    changed = []
    for obj in list(objects):
        name = obj.name
        if name in ("Footing", "RecessedDeck", "DarkWorkSurface"):
            resize(obj, factor, depth_factor)
        elif name in horizontal or name == "FrontIdentificationPlate":
            resize(obj, factor)
            if name == "FrontIdentificationPlate":
                obj.location.y += front_shift
        elif name == "FrontStatusWindow":
            obj.location.x *= factor
            obj.location.y += front_shift
        elif name == "FrontLogo":
            obj.location.y += front_shift
        elif name.startswith(("Tower", "EtchedIndex")):
            side = -1 if obj.location.x < 0 else 1
            obj.location.x = side * .35 * factor + (obj.location.x - side * .35) * 1.35
            obj.location.y = .20 + (obj.location.y - .20) * 1.20
            resize(obj, 1.35, 1.20)
        elif name.startswith("DeckBolt"):
            obj.location.x *= factor
        else:
            continue
        changed.append(name)
    bpy.context.view_layer.update()
    if any(snapshot(objects[name]) != data for name, data in protected.items()):
        raise ValueError("frame styling modified a grip or carrier-path object")
    final_bounds = bounds()
    if abs(final_bounds[1][2] - original_bounds[1][2]) > 1e-6:
        raise ValueError("frame styling modified the machine height")
    width = final_bounds[1][0] - final_bounds[0][0]
    height = final_bounds[1][2] - final_bounds[0][2]
    if abs(width - args.width) > 1e-5:
        raise ValueError("final frame width does not match the requested width")

    out.mkdir(parents=True)
    for name in manifest["files"]:
        if name not in ("frame.blend", "frame.usdc"):
            shutil.copy2(source / name, out / name)
    if (source / "textures").is_dir():
        shutil.copytree(source / "textures", out / "textures")
    bpy.ops.object.select_all(action="DESELECT")
    bpy.ops.wm.save_as_mainfile(filepath=str(out / "frame.blend"))
    bpy.ops.wm.usd_export(filepath=str(out / "frame.usdc"), root_prim_path="/Frame",
                         selected_objects_only=False, export_materials=True)
    manifest["presentation"] = {
        "style": "wide portal frame", "width_m": width, "height_m": height,
        "height_to_width": height / width, "bounds_m": final_bounds,
        "original_bounds_m": original_bounds, "original_manifest_sha256": sha(manifest_path),
        "original_author_source_sha256": manifest["author_source_sha256"],
        "protected_geometry_unchanged": True, "protected_objects": sorted(protected),
        "changed_display_objects": sorted(changed), "carrier_assets_unchanged": True,
    }
    manifest["host"] = socket.gethostname()
    manifest["blender"] = bpy.app.version_string
    manifest["author_source_sha256"] = sha(Path(__file__))
    manifest["files"] = {name: sha(out / name) for name in manifest["files"]}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"width_m": width, "height_m": height, "height_to_width": height / width,
                      "protected_geometry_unchanged": True, "manifest_sha256": sha(out / "manifest.json")}, indent=2))


if __name__ == "__main__":
    main(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else [])
