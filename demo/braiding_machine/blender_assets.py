"""Author original SI hardware in Blender and export static and carrier USD assets."""
import argparse
import hashlib
import json
from pathlib import Path
import socket
import sys

import bpy
from math import cos, pi, sin

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from demo.braiding_machine.model import CARRIER_DEPTH_M, CARRIER_Z_M, GRIP_Z_M, PRESETS, SLOT_PITCH_M, STRAND_COLORS


def material(name, color, metal=0., roughness=.34, emission=0.):
    m = bpy.data.materials.new(name)
    m.diffuse_color = (*color, 1.)
    m.use_nodes = True
    bs = m.node_tree.nodes.get("Principled BSDF")
    bs.inputs["Base Color"].default_value = (*color, 1.)
    bs.inputs["Metallic"].default_value = metal
    bs.inputs["Roughness"].default_value = roughness
    bs.inputs["Emission Color"].default_value = (*color, 1.)
    bs.inputs["Emission Strength"].default_value = emission
    return m


def finish(name, mat, bevel=0., smooth=False):
    obj = bpy.context.object
    obj.name = name
    obj.data.materials.append(mat)
    if smooth:
        for face in obj.data.polygons:
            face.use_smooth = len(face.vertices) == 4
    if bevel:
        mod = obj.modifiers.new("Cut edge", "BEVEL")
        mod.width, mod.segments = bevel, 3
        mod.harden_normals = True
        obj.modifiers.new("Face normals", "WEIGHTED_NORMAL")
    return obj


def box(name, at, size, mat, bevel=.006):
    bpy.ops.mesh.primitive_cube_add(size=1., location=at)
    bpy.context.object.dimensions = size
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    return finish(name, mat, bevel)


def cylinder(name, at, radius, depth, mat, bevel=.002, rotation=(0., 0., 0.)):
    bpy.ops.mesh.primitive_cylinder_add(vertices=64, radius=radius, depth=depth,
                                       location=at, rotation=rotation)
    return finish(name, mat, bevel, smooth=True)


def torus(name, at, radius, tube, mat):
    bpy.ops.mesh.primitive_torus_add(major_segments=64, minor_segments=12, location=at,
                                    major_radius=radius, minor_radius=tube)
    obj = finish(name, mat)
    for face in obj.data.polygons:
        face.use_smooth = True
    return obj


def lane(name, middle_x, z, mat):
    curve = bpy.data.curves.new(name, "CURVE")
    curve.dimensions = "3D"
    curve.bevel_depth = .0026
    curve.bevel_resolution = 3
    spline = curve.splines.new("POLY")
    spline.points.add(95)
    for k, point in enumerate(spline.points):
        theta = 2 * pi * k / len(spline.points)
        point.co = (middle_x + SLOT_PITCH_M / 2 * cos(theta), CARRIER_DEPTH_M * sin(theta), z, 1.)
    spline.use_cyclic_u = True
    obj = bpy.data.objects.new(name, curve)
    bpy.context.collection.objects.link(obj)
    obj.data.materials.append(mat)
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.convert(target="MESH")


def lettering(name, text, at, size, mat):
    bpy.ops.object.text_add(location=at, rotation=(pi / 2, 0., 0.))
    obj = bpy.context.object
    obj.name = name
    obj.data.body = text
    obj.data.size = size
    obj.data.extrude = .0001
    obj.data.materials.append(mat)
    bpy.ops.object.convert(target="MESH")


def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)


def save(out, stem, root):
    bpy.ops.wm.save_as_mainfile(filepath=str(out / f"{stem}.blend"))
    bpy.ops.wm.usd_export(filepath=str(out / f"{stem}.usdc"), root_prim_path=f"/{root}",
                          selected_objects_only=False, export_materials=True)


def frame(out, palette, carrier_z_m):
    extra = carrier_z_m - CARRIER_Z_M
    shell, steel, edge, brass, cyan, white, ground = palette
    box("StudioFloor", (0, 0, -.046), (20, 20, .07), ground, 0.)
    box("StudioBackdrop", (0, 1.15, 2.), (12, .06, 4.), ground, 0.)
    box("Footing", (0, .04, .035), (.99, .62, .075), shell, .029)
    box("RecessedDeck", (0, .04, .087), (.92, .55, .053), edge, .019)
    box("DarkWorkSurface", (0, .04, .121), (.86, .49, .024), shell, .013)
    box("FrontIdentificationPlate", (0, -.275, .058), (.76, .017, .058), steel, .008)
    lettering("FrontLogo", "TRI / 03     ·     STRAND LAB", (-.31, -.285, .043), .029, white)
    box("FrontStatusWindow", (.32, -.285, .059), (.072, .005, .012), cyan, .002)
    marks = (.23, .34, .45, .56, .67, .78, .89, 1.)
    marks += tuple(1. + .11 * i for i in range(1, 1 + int(extra / .11)))
    for side in (-1, 1):
        x = side * .35
        box(f"TowerShoe_{side}", (x, .20, .155), (.134, .14, .04), steel)
        box(f"Tower_{side}", (x, .20, .665 + extra / 2), (.09, .10, 1.02 + extra), shell, .016)
        box(f"TowerFace_{side}", (x, .145, .66 + extra / 2), (.054, .012, .93 + extra), steel)
        box(f"TowerTrack_{side}", (x + side * .019, .135, .67 + extra / 2), (.011, .004, .84 + extra), edge, .001)
        for z in marks:
            box(f"EtchedIndex_{side}_{z}", (x - side * .008, .1375, z), (.018, .002, .002), brass, 0.)
        for y in (-.13, .22):
            cylinder(f"DeckBolt_{side}_{y}", (side * .40, y, .141), .017, .010, steel)
    box("OverheadGantry", (0, .129, 1.208 + extra), (.85, .30, .095), shell, .023)
    box("GantryTop", (0, .099, 1.257 + extra), (.75, .21, .022), edge, .009)
    box("TopBumper", (0, -.036, 1.218 + extra), (.76, .015, .043), steel, .007)
    box("TopSignalStrip", (0, -.046, 1.223 + extra), (.42, .003, .005), cyan, .001)
    box("DriveCassette", (0, 0., 1.187 + extra), (.34, .20, .041), steel, .017)
    for side in (-1, 1):
        lane(f"CarrierLane_{side}", side * SLOT_PITCH_M / 2, 1.162 + extra, brass)
    box("AnchorPlinth", (0, .024, .169), (.36, .176, .087), steel, .018)
    box("AnchorTray", (0, .024, .214), (.337, .148, .018), edge)
    for i, col in enumerate(STRAND_COLORS):
        x = (i - 1) * SLOT_PITCH_M
        color = material(f"Jaw paint {i}", col, .2)
        cylinder(f"FixedGripStem_{i}", (x, 0, .297), .018, .156, steel)
        cylinder(f"FixedGripHousing_{i}", (x, 0, GRIP_Z_M - .031), .030, .062, shell)
        cylinder(f"FixedGripCollar_{i}", (x, 0, GRIP_Z_M - .015), .029, .015, color)
        torus(f"FixedGripFerrule_{i}", (x, 0, GRIP_Z_M - .001), .014, .003, brass)
    save(out, "frame", "Frame")


def carrier(out, i, palette):
    shell, steel, edge, brass, _, _, _ = palette
    color = material(f"Spool lacquer {i}", STRAND_COLORS[i], .24)
    cylinder("MountPin", (0, 0, .095), .012, .044, steel)
    cylinder("UpperFlange", (0, 0, .075), .035, .012, edge)
    cylinder("Spool", (0, 0, .051), .027, .04, color, .005)
    for z in (.036, .043, .05, .057, .064):
        torus(f"SpoolGroove_{z}", (0, 0, z), .027, .0007, color)
    cylinder("LowerFlange", (0, 0, .027), .035, .011, edge)
    cylinder("CableEyeHousing", (0, 0, .014), .021, .018, shell)
    torus("CableEye", (0, 0, .004), .014, .003, brass)
    save(out, f"carrier_{i}", f"Carrier_{i}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="fresh asset directory")
    parser.add_argument("--preset", choices=PRESETS, default="showcase",
                        help="match the carrier height of the simulation preset")
    args = parser.parse_args(argv)
    carrier_z_m = PRESETS[args.preset].carrier_z_m
    args.out.mkdir(parents=True, exist_ok=False)
    clear_scene()
    bpy.context.scene.unit_settings.system = "METRIC"
    bpy.context.scene.unit_settings.scale_length = 1.
    palette = (
        material("Powder-coated charcoal", (.045, .064, .092), .27),
        material("Blue anodized tooling", (.15, .24, .31), .6),
        material("Brushed graphite", (.075, .12, .16), .55),
        material("Brass accents", (.65, .39, .125), .7, .25),
        material("Cyan status indicator", (.018, .42, .5), .1, .26, .22),
        material("Identification lettering", (.72, .85, .9), .1),
        material("Studio cyclorama", (.023, .035, .06), 0., .8),
    )
    frame(args.out, palette, carrier_z_m)
    for i in range(3):
        clear_scene()
        carrier(args.out, i, palette)
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    model = Path(__file__).with_name("model.py")
    files = {p.name: sha(p) for p in sorted(args.out.iterdir()) if p.suffix in (".blend", ".usdc")}
    manifest = dict(
        units="meters", meters_per_unit=1., carrier_z_m=carrier_z_m, author="Original procedural fixture for DeformX",
        host=socket.gethostname(), blender=bpy.app.version_string,
        license="Project LICENSE. Original geometry and Blender built-in font.",
        scope="Hardware geometry for visualization only. No fixture collision.",
        carrier_origin="Each moving carrier's root transform is the solver endpoint.",
        author_source_sha256=sha(Path(__file__)), model_source_sha256=sha(model), files=files,
    )
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else [])
