import bpy
import bmesh
from mathutils import Vector, Matrix
from pathlib import Path
import os
import re

# ============================================================
# Batch-generate wires (mesh + armature), smooth weights for ALL groups,
# and export to USD (.usdc):
#   wire_{color}_s20_r{SEGMENT_RADIUS}_l1.usdc
#
# Total length = SEGMENT_TOTAL_LENGTH * CHAIN_COUNT = 1.0 (=> l1)
# ============================================================

# -----------------------------
# Global geometry settings (your values)
# -----------------------------
SEGMENT_TOTAL_LENGTH = 0.05
CYLINDER_SIDES = 12
VERTICAL_CUTS = 10
AXIS_WORLD = Vector((0, 0, 1))

CHAIN_COUNT = 20
MERGE_DISTANCE = 1e-6
AUTO_WEIGHT = True

WEIGHT_SMOOTH_FACTOR = 2.0
WEIGHT_SMOOTH_REPEAT = 100

# Radii
SEGMENT_RADII = [0.001, 0.003, 0.005]

# Material look
MATERIAL_ROUGHNESS = 0.75  # 0 = glossy, 1 = matte
# MATERIAL_ROUGHNESS = 0.05  # 0 = glossy, 1 = matte

# Colors (name -> hex)
COLOR_HEX = {
    "Blue": "#3a7ca5ff",
    "Dark Blue": "#1D2A60FF",
    "Yellow": "#e7d516",
    "Orange": "#df7224",
    "Red": "#9A1515",
    # "Pink": "#ff85c8",
    "Purple": "#49294fff",
    # "Green": "#53b753",
    "Dark Green": "#2C5132",
    "White": "#cfcfcf",
    "Grey": "#a0a0a0",
    "Dark Grey": "#5d5d5d",
    "Black": "#000000",
    "Brown": "#674030",
}

# Where to export. Defaults to a folder next to the .blend file (or Blender's
# temp dir if the file is unsaved). Override by setting the WIRE_EXPORT_DIR
# environment variable to an absolute path.
EXPORT_DIR = Path(
    os.environ.get(
        "WIRE_EXPORT_DIR",
        bpy.path.abspath("//") if bpy.data.filepath else bpy.app.tempdir,
    )
)
EXPORT_DIR.mkdir(parents=True, exist_ok=True)

COLLECTION_NAME = "WireGenBatch"


# ============================================================
# Helpers
# ============================================================
def ensure_collection(name: str) -> bpy.types.Collection:
    col = bpy.data.collections.get(name)
    if col is None:
        col = bpy.data.collections.new(name)
        bpy.context.scene.collection.children.link(col)
    return col


def deselect_all():
    for o in bpy.context.selected_objects:
        o.select_set(False)


def select_only(obj: bpy.types.Object):
    deselect_all()
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj


def apply_scale(obj: bpy.types.Object):
    select_only(obj)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)


def duplicate_object(obj: bpy.types.Object) -> bpy.types.Object:
    c = obj.copy()
    c.data = obj.data.copy()
    return c


def keep_object_only_in_collection(obj: bpy.types.Object, target_col: bpy.types.Collection):
    if obj.name not in target_col.objects:
        target_col.objects.link(obj)
    for c in list(obj.users_collection):
        if c != target_col:
            c.objects.unlink(obj)


def hex_to_rgba(hex_color: str):
    h = hex_color.lstrip("#")
    if len(h) not in (6, 8):
        raise ValueError("HEX color must be #RRGGBB or #RRGGBBAA")
    r = int(h[0:2], 16) / 255.0
    g = int(h[2:4], 16) / 255.0
    b = int(h[4:6], 16) / 255.0
    a = int(h[6:8], 16) / 255.0 if len(h) == 8 else 1.0
    return (r, g, b, a)


def create_material_from_hex(name: str, hex_color: str):
    mat = bpy.data.materials.get(name)
    if mat is None:
        mat = bpy.data.materials.new(name)
    mat.use_nodes = True

    nt = mat.node_tree
    nodes = nt.nodes
    links = nt.links
    nodes.clear()

    out = nodes.new("ShaderNodeOutputMaterial")
    out.location = (300, 0)

    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.location = (0, 0)
    links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])

    bsdf.inputs["Base Color"].default_value = hex_to_rgba(hex_color)
    bsdf.inputs["Roughness"].default_value = MATERIAL_ROUGHNESS
    return mat


def create_material_from_image(name: str, image_path: str):
    mat = bpy.data.materials.get(name)
    if mat is None:
        mat = bpy.data.materials.new(name)
    mat.use_nodes = True

    nt = mat.node_tree
    nodes = nt.nodes
    links = nt.links
    nodes.clear()

    out = nodes.new("ShaderNodeOutputMaterial")
    out.location = (300, 0)

    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.location = (0, 0)
    links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])

    tex = nodes.new("ShaderNodeTexImage")
    tex.location = (-300, 0)
    img = bpy.data.images.load(image_path, check_existing=True)
    tex.image = img
    links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
    return mat


def auto_uv_cylinder_project(obj: bpy.types.Object):
    select_only(obj)
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.uv.cylinder_project(direction="VIEW_ON_EQUATOR", align="POLAR_ZX", radius=1.0)
    bpy.ops.object.mode_set(mode="OBJECT")


def cap_open_ends_with_bmesh(obj: bpy.types.Object):
    select_only(obj)
    bpy.ops.object.mode_set(mode="EDIT")
    me = obj.data
    bm = bmesh.from_edit_mesh(me)

    boundary_edges = [e for e in bm.edges if e.is_boundary]
    if boundary_edges:
        bmesh.ops.holes_fill(bm, edges=boundary_edges, sides=0)

    bmesh.update_edit_mesh(me)
    bpy.ops.object.mode_set(mode="OBJECT")


def smooth_weights_all_groups(mesh_obj: bpy.types.Object, factor: float, repeat: int):
    """
    Smooth weights for ALL vertex groups.
    Tries the "ALL" mode call you wrote, falls back to per-group loop.
    """
    select_only(mesh_obj)

    # Ensure weight paint context
    bpy.ops.paint.weight_paint_toggle()
    try:
        # Some Blender builds accept group_select_mode="ALL"
        bpy.ops.object.vertex_group_smooth(
            group_select_mode="ALL",
            factor=factor,
            repeat=repeat,
        )
    except Exception:
        # Reliable fallback: smooth each group
        for vg in mesh_obj.vertex_groups:
            mesh_obj.vertex_groups.active = vg
            bpy.ops.object.vertex_group_smooth(factor=factor, repeat=repeat)

    bpy.ops.paint.weight_paint_toggle()


def export_usdc(filepath: Path, objects: list[bpy.types.Object]):
    deselect_all()
    for o in objects:
        o.select_set(True)
    bpy.context.view_layer.objects.active = objects[0]

    # Requires Blender USD exporter enabled (usually built-in).
    bpy.ops.wm.usd_export(
        filepath=str(filepath),
        selected_objects_only=True,
        export_materials=True,
    )


def slug(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def cleanup_objects(objs: list[bpy.types.Object]):
    # Remove objects and their data blocks when possible
    for o in objs:
        if o and o.name in bpy.data.objects:
            bpy.data.objects.remove(o, do_unlink=True)


# ============================================================
# Core: build a single wire (mesh + armature)
# ============================================================
def build_wire(
    *,
    segment_length: float,
    segment_radius: float,
    cylinder_sides: int,
    vertical_cuts: int,
    axis_world: Vector,
    chain_count: int,
    merge_distance: float,
    material: bpy.types.Material,
    do_uv: bool,
    collection: bpy.types.Collection,
    name_prefix: str,
):
    """
    Returns: (joined_mesh_obj, chain_arm_obj, temp_base_objs_to_delete)
    """
    axis_n = axis_world.normalized()
    z_axis = Vector((0, 0, 1))
    step = segment_length
    L = segment_length

    # ----- Base segment mesh (open cylinder), geometry shifted to [0,L] in local Z -----
    mesh_data = bpy.data.meshes.new(f"{name_prefix}_seg_mesh_data")
    base_mesh = bpy.data.objects.new(f"{name_prefix}_seg_mesh", mesh_data)
    collection.objects.link(base_mesh)

    base_mesh.location = (0.0, 0.0, 0.0)
    base_mesh.rotation_euler = (0.0, 0.0, 0.0)

    bm = bmesh.new()
    bmesh.ops.create_cone(
        bm,
        segments=cylinder_sides,
        radius1=segment_radius,
        radius2=segment_radius,
        depth=L,
        cap_ends=False,
    )
    bmesh.ops.translate(bm, verts=bm.verts, vec=Vector((0.0, 0.0, 0.5 * L)))

    # Subdivide vertical side edges only
    eps = 1e-9
    vertical_edges = []
    for e in bm.edges:
        v1, v2 = e.verts
        dx = abs(v1.co.x - v2.co.x)
        dy = abs(v1.co.y - v2.co.y)
        dz = abs(v1.co.z - v2.co.z)
        if dx < eps and dy < eps and dz > eps:
            vertical_edges.append(e)

    if vertical_cuts > 0 and vertical_edges:
        bmesh.ops.subdivide_edges(bm, edges=vertical_edges, cuts=vertical_cuts, use_grid_fill=True)

    bm.to_mesh(mesh_data)
    bm.free()

    select_only(base_mesh)
    bpy.ops.object.shade_smooth()

    # Rotate so local +Z aligns with AXIS_WORLD
    if (axis_n - z_axis).length > 1e-8:
        rot = z_axis.rotation_difference(axis_n)
        base_mesh.rotation_mode = "QUATERNION"
        base_mesh.rotation_quaternion = rot

    apply_scale(base_mesh)

    # Apply material
    if base_mesh.data.materials:
        base_mesh.data.materials[0] = material
    else:
        base_mesh.data.materials.append(material)

    # ----- Duplicate segments -> join -> weld -> cap ends -----
    mesh_copies = []
    m0 = duplicate_object(base_mesh)
    m0.name = f"{name_prefix}_seg0"
    collection.objects.link(m0)
    mesh_copies.append(m0)

    for i in range(1, chain_count):
        m = duplicate_object(base_mesh)
        m.name = f"{name_prefix}_seg{i}"
        collection.objects.link(m)
        m.matrix_world.translation += axis_n * (step * i)
        mesh_copies.append(m)

    deselect_all()
    for m in mesh_copies:
        m.select_set(True)
    bpy.context.view_layer.objects.active = mesh_copies[0]
    bpy.ops.object.join()
    joined_mesh = bpy.context.view_layer.objects.active
    joined_mesh.name = f"{name_prefix}_mesh"

    # Weld seams
    select_only(joined_mesh)
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.remove_doubles(threshold=merge_distance)
    bpy.ops.object.mode_set(mode="OBJECT")

    # Cap only the two outer ends
    cap_open_ends_with_bmesh(joined_mesh)

    keep_object_only_in_collection(joined_mesh, collection)

    # Ensure material on final mesh
    if joined_mesh.data.materials:
        joined_mesh.data.materials[0] = material
    else:
        joined_mesh.data.materials.append(material)

    # UV unwrap if needed
    if do_uv:
        auto_uv_cylinder_project(joined_mesh)

    # Remove any old vertex groups / armature modifiers (rebinding later)
    for vg in list(joined_mesh.vertex_groups):
        joined_mesh.vertex_groups.remove(vg)
    for mod in list(joined_mesh.modifiers):
        if mod.type == "ARMATURE":
            joined_mesh.modifiers.remove(mod)

    # ----- Chain armature with exactly chain_count bones, rotation only -----
    chain_arm_data = bpy.data.armatures.new(f"{name_prefix}_arm_data")
    chain_arm = bpy.data.objects.new(f"{name_prefix}_arm", chain_arm_data)
    collection.objects.link(chain_arm)

    chain_arm.matrix_world = Matrix.Translation((0, 0, 0)) @ base_mesh.matrix_world.to_3x3().to_4x4()
    apply_scale(chain_arm)

    select_only(chain_arm)
    bpy.ops.object.mode_set(mode="EDIT")
    ceb = chain_arm.data.edit_bones

    prev = None
    for i in range(chain_count):
        b = ceb.new(f"bone_{i}")
        if prev is None:
            b.head = Vector((0, 0, 0))
        else:
            b.head = prev.tail
            b.parent = prev
            b.use_connect = True
        b.tail = b.head + Vector((0, 0, step))
        prev = b

    bpy.ops.object.mode_set(mode="OBJECT")
    keep_object_only_in_collection(chain_arm, collection)

    # ----- Bind -----
    if AUTO_WEIGHT:
        deselect_all()
        joined_mesh.select_set(True)
        chain_arm.select_set(True)
        bpy.context.view_layer.objects.active = chain_arm
        bpy.ops.object.parent_set(type="ARMATURE_AUTO")

    return joined_mesh, chain_arm, [base_mesh]


# ============================================================
# Batch loop: colors x radii
# ============================================================
def main():
    col = ensure_collection(COLLECTION_NAME)

    # Keep batch outputs in that collection only (optional)
    # (We won't unlink existing objects; only ensure our generated objects are in col.)

    total_length = SEGMENT_TOTAL_LENGTH * CHAIN_COUNT  # should be 1.0
    l_tag = f"l{total_length:g}"  # "l1" for 1.0

    for color_name, color_hex in COLOR_HEX.items():
        color_slug = slug(color_name)

        # create material per color (shared across radii)
        # mat_name = f"{MATERIAL_NAME}_{color_slug}"
        mat_name = f"mat_{color_slug}"
        mat = create_material_from_hex(mat_name, color_hex)

        for r in SEGMENT_RADII:
            name_prefix = f"wire_{color_slug}_s{CHAIN_COUNT}_r{r:g}_{l_tag}"

            joined_mesh, chain_arm, temp_objs = build_wire(
                segment_length=SEGMENT_TOTAL_LENGTH,
                segment_radius=r,
                cylinder_sides=CYLINDER_SIDES,
                vertical_cuts=VERTICAL_CUTS,
                axis_world=AXIS_WORLD,
                chain_count=CHAIN_COUNT,
                merge_distance=MERGE_DISTANCE,
                material=mat,
                do_uv=True,  # always unwrap; safe even for solid colors
                collection=col,
                name_prefix=name_prefix,
            )

            # Smooth weights for ALL groups
            smooth_weights_all_groups(joined_mesh, WEIGHT_SMOOTH_FACTOR, WEIGHT_SMOOTH_REPEAT)

            # Export
            out_path = EXPORT_DIR / f"wire_{color_slug}_s{CHAIN_COUNT}_r{r:g}_{l_tag}_rough.usdc"
            export_usdc(out_path, [joined_mesh, chain_arm])

            # Cleanup temporary base objects (keep the final mesh+armature, or delete them too if you want)
            cleanup_objects(temp_objs)

            # OPTIONAL: If you want to delete the wire objects after export (headless batch), uncomment:
            # cleanup_objects([joined_mesh, chain_arm])

            print(f"Exported: {out_path}")

    print("All exports done.")


main()
