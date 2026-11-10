"""Create presentation props in Blender. No simulation geometry is replaced."""
import argparse
import hashlib
import json
from math import pi
from pathlib import Path
import sys

import bpy


def material(name, rgb, roughness, metallic=0, emission=0):
    mat = bpy.data.materials.new(name)
    mat.diffuse_color = (*rgb, 1)
    mat.use_nodes = True
    shader = mat.node_tree.nodes["Principled BSDF"]
    for key, value in {
        "Base Color": (*rgb, 1), "Roughness": roughness, "Metallic": metallic,
        "Emission Color": (*rgb, 1), "Emission Strength": emission,
    }.items():
        shader.inputs[key].default_value = value
    return mat


def surface(mat, smooth=False, bevel=0):
    obj = bpy.context.object
    obj.data.materials.append(mat)
    for face in obj.data.polygons:
        face.use_smooth = smooth
    if bevel:
        mod = obj.modifiers.new("Machined edge", "BEVEL")
        mod.width, mod.segments = bevel, 3
        bpy.ops.object.modifier_apply(modifier=mod.name)


def grip(accent, body):
    for y, radius, length, mat in [
        (.013, .017, .038, body), (-.007, .0175, .005, accent), (.033, .015, .007, body),
    ]:
        bpy.ops.mesh.primitive_cylinder_add(vertices=48, radius=radius, depth=length,
                                            location=(0, y, 0), rotation=(pi/2, 0, 0))
        surface(mat, smooth=True, bevel=.0012)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    out = parser.parse_args(sys.argv[sys.argv.index("--")+1:]).out
    out.mkdir(parents=True)
    scene = bpy.context.scene
    scene.unit_settings.system = "METRIC"
    body = material("Grip body", (.065, .10, .15), .39, .55)
    nickel = material("Fixed support", (.60, .72, .80), .28, .60, .08)
    amber = material("Driven support", (.85, .46, .10), .32, .32, .12)
    teal = material("Sphere", (.06, .26, .31), .32, .28)
    slate = material("Floor", (.067, .104, .139), .62, .12)
    mint = material("Goal marker", (.14, .95, .57), .35, .10, .55)

    def sphere():
        bpy.ops.mesh.primitive_uv_sphere_add(segments=96, ring_count=64, radius=1)
        surface(teal, smooth=True)

    def floor():
        bpy.ops.mesh.primitive_cube_add(size=1, location=(0, 0, -.008))
        bpy.context.object.dimensions = (1.2, 1.2, .016)
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
        surface(slate, bevel=.003)

    def goal():
        for rotation in [(0, 0, 0), (pi/2, 0, 0), (0, pi/2, 0)]:
            bpy.ops.mesh.primitive_torus_add(major_radius=1, minor_radius=.055,
                                            major_segments=48, minor_segments=10, rotation=rotation)
            surface(mint, smooth=True)

    for name, make in [
        ("fixed-grip", lambda: grip(nickel, body)), ("driven-grip", lambda: grip(amber, body)),
        ("sphere", sphere), ("floor", floor), ("goal", goal),
    ]:
        bpy.ops.object.select_all(action="SELECT")
        bpy.ops.object.delete(use_global=False)
        make()
        path = out / (name + ".usdc")
        bpy.ops.wm.usd_export(filepath=str(path), export_materials=True,
                              export_textures=False, root_prim_path="/Asset")
    (out / "manifest.json").write_text(json.dumps({
        "generator": "Blender", "units": "metres",
        "asset_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in sorted(out.glob("*.usdc"))},
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
