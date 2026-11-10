"""Author original meter-scale traction reel assembly assets in Blender."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

# Blender launches scripts with its own import root.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from demo.winch_tension import model

DRIVE = (.70, .19, .038)
GUIDE = (.025, .39, .53)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path, help="Fresh asset directory")
    args = parser.parse_args(argv)

    import bpy
    from mathutils import Vector

    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    bpy.context.scene.unit_settings.system = "METRIC"
    bpy.context.scene.unit_settings.scale_length = 1.

    def material(name, color, roughness, metal=0.):
        mat = bpy.data.materials.new(name)
        mat.diffuse_color = (*color, 1.)
        mat.use_nodes = True
        shader = mat.node_tree.nodes.get("Principled BSDF")
        shader.inputs["Base Color"].default_value = (*color, 1.)
        shader.inputs["Roughness"].default_value = roughness
        shader.inputs["Metallic"].default_value = metal
        return mat

    steel = material("Brushed nickel", (.39, .47, .53), .32, .63)
    pale = material("Powder coated chassis", (.32, .38, .43), .36, .22)
    dark = material("Graphite housing", (.027, .041, .057), .41, .14)
    rubber = material("Traction sleeve", (.046, .064, .072), .65)
    white = material("Index enamel", (.87, .90, .85), .3)
    drive = material("Motor amber", DRIVE, .34, .20)
    guide = material("Bearing blue", GUIDE, .34, .20)

    def part(name, mat, bevel=0.):
        obj = bpy.context.object
        obj.name = name
        obj.data.materials.append(mat)
        if bevel:
            rounded = obj.modifiers.new("Machined radius", "BEVEL")
            rounded.width, rounded.segments = bevel, 3
            normal = obj.modifiers.new("Weighted corners", "WEIGHTED_NORMAL")
            normal.keep_sharp = True
        return obj

    def box(name, size, pos, mat, bevel=.003):
        bpy.ops.mesh.primitive_cube_add(size=1., location=pos)
        bpy.context.object.dimensions = size
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
        return part(name, mat, bevel)

    def cylinder(name, radius, width, pos, mat, *, axis="Y", sides=96, bevel=.001):
        rot = (math.pi / 2., 0., 0.) if axis == "Y" else (0., 0., 0.)
        bpy.ops.mesh.primitive_cylinder_add(vertices=sides, radius=radius, depth=width, location=pos, rotation=rot)
        return part(name, mat, bevel)

    def export(name, previous):
        objects = set(bpy.context.scene.objects) - previous
        bpy.ops.object.select_all(action="DESELECT")
        for obj in objects:
            obj.select_set(True)
        bpy.ops.wm.usd_export(filepath=str(out / f"{name}.usdc"), selected_objects_only=True,
                              root_prim_path="/Asset", export_materials=True)
        return objects

    chassis_prev = set(bpy.context.scene.objects)
    box("Base plate", (.96, .39, .048), (0., .059, -.034), pale, .014)
    box("Recessed fascia", (.889, .008, .028), (0., -.139, -.033), dark)
    for x in [-.397, .397]:
        box("Isolation foot", (.11, .29, .036), (x, .059, -.074), dark, .012)
        for y in [-.071, .199]:
            cylinder("Base socket screw", .007, .006, (x, y, -.009), steel, axis="Z", sides=12)
    for i in range(8):
        box("Fascia cooling slot", (.009, .002, .011), (.239 + i * .016, -.1435, -.034), pale, .0008)
    for i, mat in enumerate([drive, guide, white]):
        box("Fascia indicator", (.011, .003, .007), (-.333 + i * .019, -.144, -.034), mat, .001)
    for i, (x, y, z) in enumerate(model.REEL_CENTERS_M):
        mat = [drive, guide][i]
        box("Bearing tower shoe", (.154, .138, .028), (x, .130, .006), dark, .009)
        box("Bearing tower", (.084, .081, z), (x, .132, z / 2.), dark, .008)
        box("Reinforcing plate", (.055, .010, .235), (x, .088, .169), pale, .006)
        cylinder("Fixed bearing", .067, .062, (x, .113, z), steel, bevel=.003)
        cylinder("Motor or idler casing", .057, .059, (x, .169, z), mat, bevel=.003)
        cylinder("Rear bearing end cap", .048, .006, (x, .201, z), dark)
        for j in range(6):
            xbolt = x + .056 * math.cos(j * math.pi / 3)
            zbolt = z + .056 * math.sin(j * math.pi / 3)
            cylinder("Bearing socket", .004, .004, (xbolt, .080, zbolt), dark, sides=10)
        anchor_x = model.tether_anchors(model.initial_positions())[i, 0]
        cylinder("Tether anchor", .026, .012, (anchor_x, 0., -.006), mat, axis="Z", bevel=.002)
        cylinder("Fixed eye", .013, .012, (anchor_x, 0., -.000), steel, axis="Z")
    export("chassis", chassis_prev)

    rotors = []
    for i, (name, accent) in enumerate((("drive", drive), ("guide", guide))):
        existing = set(bpy.context.scene.objects)
        w = model.BARREL_HALF_WIDTH_M
        r = model.REEL_RADIUS_M
        cylinder("Physical barrel sleeve", r, 2 * w, (0., 0., 0.), rubber, sides=160, bevel=0.)
        cylinder("Axle journal", .019, .175, (0., .012, 0.), steel)
        for side in [-1, 1]:
            cylinder("Rolled rim", r + .005, .007, (0., side * (w + .004), 0.), steel, bevel=.0015)
            cylinder("Anodized cheek", r - .005, .007, (0., side * (w + .008), 0.), accent)
            cylinder("Sunken hub plate", .042, .005, (0., side * (w + .013), 0.), dark)
            cylinder("Hub cap", .024, .006, (0., side * (w + .017), 0.), steel)
        for j in range(8):
            angle = j * math.pi / 4
            c, s = math.cos(angle), math.sin(angle)
            cylinder("Fastener", .0035, .003, (.034 * s, -.066, .034 * c), steel, sides=10)
            lug = box("Radial recess", (.014, .004, .023), (.061 * s, -.062, .061 * c), dark, .0025)
            lug.rotation_euler[1] = angle
        box("Single phase witness", (.013, .006, .039), (0., -.066, .060), white, .002)
        if i == 1:
            box("Bearing marker", (.009, .006, .017), (.017, -.066, .070), white, .001)
        rotors.append(export(name, existing))

    curve = bpy.data.curves.new("Tensioner coil profile", "CURVE")
    curve.dimensions = "3D"
    curve.bevel_depth, curve.bevel_resolution = .0015, 3
    samples = 161
    spline = curve.splines.new("POLY")
    spline.points.add(samples - 1)
    for j, point in enumerate(spline.points):
        u = j / (samples - 1)
        radius = .009 * min(1., 14 * u, 14 * (1 - u))
        point.co = (radius * math.cos(16 * math.pi * u), radius * math.sin(16 * math.pi * u), u, 1.)
    coil = bpy.data.objects.new("Tension spring", curve)
    bpy.context.scene.collection.objects.link(coil)
    coil.data.materials.append(steel)
    bpy.ops.object.select_all(action="DESELECT")
    coil.select_set(True)
    bpy.context.view_layer.objects.active = coil
    bpy.ops.object.convert(target="MESH")
    spring = bpy.context.object
    cylinder("Terminal ferrule", .009, .024, (0., 0., 0.), steel, axis="Z")
    ferrule = bpy.context.object
    bpy.ops.object.select_all(action="DESELECT")
    spring.select_set(True)
    bpy.ops.wm.usd_export(filepath=str(out / "tether.usdc"), selected_objects_only=True,
                          root_prim_path="/Asset", export_materials=True)
    bpy.ops.object.select_all(action="DESELECT")
    ferrule.select_set(True)
    bpy.ops.wm.usd_export(filepath=str(out / "ferrule.usdc"), selected_objects_only=True,
                          root_prim_path="/Asset", export_materials=True)

    # Source blend opens as a static assembly at the starting dimensions.
    for rotor, center in zip(rotors, model.REEL_CENTERS_M, strict=True):
        for obj in rotor:
            obj.location += Vector(center)
    positions = model.initial_positions()
    anchors = model.tether_anchors(positions)
    for index, (terminal, anchor) in enumerate(zip(positions[[0, -1]], anchors, strict=True)):
        if index == 0:
            second_spring, second_ferrule = spring.copy(), ferrule.copy()
            for obj in [second_spring, second_ferrule]:
                bpy.context.scene.collection.objects.link(obj)
        s, f = (spring, ferrule) if index == 0 else (second_spring, second_ferrule)
        s.location = tuple(anchor)
        s.scale.z = float(terminal[2] - anchor[2])
        f.location = tuple(terminal)
    bpy.ops.wm.save_as_mainfile(filepath=str(out / "traction_reels.blend"))
    manifest = dict(units="meters, Z up", original_geometry=True, blender=bpy.app.version_string,
                    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    asset_sha256={f.name: hashlib.sha256(f.read_bytes()).hexdigest()
                                  for f in sorted(out.iterdir()) if f.suffix in {".usdc", ".blend"}},
                    physical="Only the two finite barrel SDFs contact the cable. Chassis, ferrules, coils, hub trim and rims are illustrations.")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else None)
