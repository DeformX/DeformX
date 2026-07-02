import bpy
import bmesh
from mathutils import Vector, Matrix

# ============================================================
# FINAL SCRIPT
#   - Generates a base "segment": open cylinder (NO caps) + 2-bone armature (root/tip)
#   - Duplicates segment mesh CHAIN_COUNT times -> join -> weld -> cap ONLY ends
#   - Builds ONE chain armature with exactly CHAIN_COUNT bones
#   - Root is at world origin; extends forward along +AXIS_WORLD
#   - Parents with Automatic Weights
#   - Applies global weight smoothing:
#       bpy.ops.paint.weight_paint_toggle()
#       bpy.ops.object.vertex_group_smooth(factor=1, repeat=50)
#   - Material can be set by HEX color (#RRGGBB or #RRGGBBAA) OR image path
#   - Auto UV unwrap (cylinder project) when using image texture
#   - Avoids "skeleton moved twice" by shifting GEOMETRY to [0, L] (not object location)
# ============================================================

# -----------------------------
# USER SETTINGS
# -----------------------------
# Segment geometry
SEGMENT_TOTAL_LENGTH = 0.05
SEGMENT_RADIUS = 0.001
CYLINDER_SIDES = 12
VERTICAL_CUTS = 10
AXIS_WORLD = Vector((0, 0, 1))  # chain direction in WORLD

# Chain
CHAIN_COUNT = 20                  # number of mesh segments AND number of bones
MERGE_DISTANCE = 1e-6
AUTO_WEIGHT = True

# Naming / collection
COLLECTION_NAME = "ChainGen"
BASE_MESH_NAME = "Cylinder.001"
BASE_ARM_NAME = "Armature.001"

# Material input (choose one)
MATERIAL_NAME = "ChainMaterial"
MATERIAL_HEX = "#4fa3ff"         # "#RRGGBB" or "#RRGGBBAA"
MATERIAL_IMAGE_PATH = ""         # absolute path or "//relative.png"; if non-empty, overrides HEX

# UV unwrap for texture
DO_AUTO_UV_UNWRAP = True

# Cleanup base objects (segment-only objects) after chain created
DELETE_BASE_ARM_AFTER = True
DELETE_BASE_MESH_AFTER = True

# If spacing is off, override step manually (0 means use SEGMENT_TOTAL_LENGTH)
MANUAL_STEP = 0.0

# Weight smoothing
DO_WEIGHT_SMOOTH = True
WEIGHT_SMOOTH_FACTOR = 2.0
WEIGHT_SMOOTH_REPEAT = 100


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

def create_chain_material(name: str, hex_color: str, image_path: str):
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

    if image_path:
        tex = nodes.new("ShaderNodeTexImage")
        tex.location = (-300, 0)
        img = bpy.data.images.load(image_path, check_existing=True)
        tex.image = img
        links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
    else:
        bsdf.inputs["Base Color"].default_value = hex_to_rgba(hex_color)

    return mat

def auto_uv_cylinder_project(obj: bpy.types.Object):
    # Cylinder projection is good for rope/cable-like meshes.
    select_only(obj)
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.uv.cylinder_project(direction='VIEW_ON_EQUATOR', align='POLAR_ZX', radius=1.0)
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


# ============================================================
# MAIN
# ============================================================
chain_col = ensure_collection(COLLECTION_NAME)
axis_n = AXIS_WORLD.normalized()
z_axis = Vector((0, 0, 1))

step = MANUAL_STEP if MANUAL_STEP > 0 else SEGMENT_TOTAL_LENGTH
L = SEGMENT_TOTAL_LENGTH

# ------------------------------------------------------------
# 1) Base segment mesh: open cylinder, GEOMETRY shifted to [0, L] in local Z
# ------------------------------------------------------------
mesh_data = bpy.data.meshes.new(f"{BASE_MESH_NAME}_DATA")
base_mesh = bpy.data.objects.new(BASE_MESH_NAME, mesh_data)
chain_col.objects.link(base_mesh)

# Force clean transform (no translation)
base_mesh.location = (0.0, 0.0, 0.0)
base_mesh.rotation_euler = (0.0, 0.0, 0.0)

bm = bmesh.new()

# Centered cylinder first ([-L/2, +L/2]) along local Z
bmesh.ops.create_cone(
    bm,
    segments=CYLINDER_SIDES,
    radius1=SEGMENT_RADIUS,
    radius2=SEGMENT_RADIUS,
    depth=L,
    cap_ends=False,  # IMPORTANT: no caps
)

# Shift GEOMETRY forward so it spans [0, L] in local Z
bmesh.ops.translate(bm, verts=bm.verts, vec=Vector((0.0, 0.0, 0.5 * L)))

# Subdivide vertical edges only
eps = 1e-9
vertical_edges = []
for e in bm.edges:
    v1, v2 = e.verts
    dx = abs(v1.co.x - v2.co.x)
    dy = abs(v1.co.y - v2.co.y)
    dz = abs(v1.co.z - v2.co.z)
    if dx < eps and dy < eps and dz > eps:
        vertical_edges.append(e)

if VERTICAL_CUTS > 0 and vertical_edges:
    bmesh.ops.subdivide_edges(
        bm,
        edges=vertical_edges,
        cuts=VERTICAL_CUTS,
        use_grid_fill=True,
    )

bm.to_mesh(mesh_data)
bm.free()

select_only(base_mesh)
bpy.ops.object.shade_smooth()

# Rotate so local +Z aligns with AXIS_WORLD (no translation involved)
if (axis_n - z_axis).length > 1e-8:
    rot = z_axis.rotation_difference(axis_n)
    base_mesh.rotation_mode = 'QUATERNION'
    base_mesh.rotation_quaternion = rot

apply_scale(base_mesh)

# Create material (hex or image)
mat = create_chain_material(MATERIAL_NAME, MATERIAL_HEX, MATERIAL_IMAGE_PATH)

# Apply to base mesh (preview)
if base_mesh.data.materials:
    base_mesh.data.materials[0] = mat
else:
    base_mesh.data.materials.append(mat)

# ------------------------------------------------------------
# 2) Base armature with 2 bones (root/tip), rotation only, at origin
# ------------------------------------------------------------
arm_data = bpy.data.armatures.new(f"{BASE_ARM_NAME}_DATA")
base_arm = bpy.data.objects.new(BASE_ARM_NAME, arm_data)
chain_col.objects.link(base_arm)

# Copy rotation only; keep translation at origin
base_arm.matrix_world = Matrix.Translation((0, 0, 0)) @ base_mesh.matrix_world.to_3x3().to_4x4()
apply_scale(base_arm)

select_only(base_arm)
bpy.ops.object.mode_set(mode="EDIT")
eb = base_arm.data.edit_bones
for b in list(eb):
    eb.remove(b)

root = eb.new("root")
root.head = Vector((0, 0, 0))
root.tail = Vector((0, 0, 0.5 * L))

tip = eb.new("tip")
tip.head = root.tail
tip.tail = Vector((0, 0, L))
tip.parent = root
tip.use_connect = True

bpy.ops.object.mode_set(mode="OBJECT")

# ------------------------------------------------------------
# 3) Duplicate segments -> join -> weld -> cap ONLY ends
# ------------------------------------------------------------
mesh_copies = []
m0 = duplicate_object(base_mesh)
m0.name = f"{BASE_MESH_NAME}_seg0"
chain_col.objects.link(m0)
mesh_copies.append(m0)

for i in range(1, CHAIN_COUNT):
    m = duplicate_object(base_mesh)
    m.name = f"{BASE_MESH_NAME}_seg{i}"
    chain_col.objects.link(m)
    m.matrix_world.translation += axis_n * (step * i)
    mesh_copies.append(m)

deselect_all()
for m in mesh_copies:
    m.select_set(True)
bpy.context.view_layer.objects.active = mesh_copies[0]
bpy.ops.object.join()
joined_mesh = bpy.context.view_layer.objects.active
joined_mesh.name = f"{BASE_MESH_NAME}_CHAIN"

# Weld seams
select_only(joined_mesh)
bpy.ops.object.mode_set(mode="EDIT")
bpy.ops.mesh.select_all(action="SELECT")
bpy.ops.mesh.remove_doubles(threshold=MERGE_DISTANCE)
bpy.ops.object.mode_set(mode="OBJECT")

# Cap the two outer ends
cap_open_ends_with_bmesh(joined_mesh)

# Keep only in ChainGen
keep_object_only_in_collection(joined_mesh, chain_col)

# Apply material to final mesh
if joined_mesh.data.materials:
    joined_mesh.data.materials[0] = mat
else:
    joined_mesh.data.materials.append(mat)

# Auto UV unwrap if using texture
if MATERIAL_IMAGE_PATH and DO_AUTO_UV_UNWRAP:
    auto_uv_cylinder_project(joined_mesh)

# Remove old weights/modifiers
for vg in list(joined_mesh.vertex_groups):
    joined_mesh.vertex_groups.remove(vg)
for mod in list(joined_mesh.modifiers):
    if mod.type == "ARMATURE":
        joined_mesh.modifiers.remove(mod)

# ------------------------------------------------------------
# 4) Chain armature with EXACTLY CHAIN_COUNT bones, at origin, rotation only
# ------------------------------------------------------------
chain_arm_data = bpy.data.armatures.new(f"{BASE_ARM_NAME}_CHAIN_DATA")
chain_arm = bpy.data.objects.new(f"{BASE_ARM_NAME}_CHAIN", chain_arm_data)
chain_col.objects.link(chain_arm)

chain_arm.matrix_world = Matrix.Translation((0, 0, 0)) @ base_mesh.matrix_world.to_3x3().to_4x4()
apply_scale(chain_arm)

select_only(chain_arm)
bpy.ops.object.mode_set(mode="EDIT")
ceb = chain_arm.data.edit_bones

prev = None
for i in range(CHAIN_COUNT):
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
keep_object_only_in_collection(chain_arm, chain_col)

# ------------------------------------------------------------
# 5) Parent with Automatic Weights
# ------------------------------------------------------------
if AUTO_WEIGHT:
    deselect_all()
    joined_mesh.select_set(True)
    chain_arm.select_set(True)
    bpy.context.view_layer.objects.active = chain_arm
    bpy.ops.object.parent_set(type="ARMATURE_AUTO")

# ------------------------------------------------------------
# 6) Weight smoothing (requested ops)
# ------------------------------------------------------------
if DO_WEIGHT_SMOOTH:
    select_only(joined_mesh)
    bpy.ops.paint.weight_paint_toggle()
    bpy.ops.object.vertex_group_smooth(group_select_mode="ALL", factor=WEIGHT_SMOOTH_FACTOR, repeat=WEIGHT_SMOOTH_REPEAT)
    bpy.ops.paint.weight_paint_toggle()

# ------------------------------------------------------------
# 7) Cleanup base objects
# ------------------------------------------------------------
if DELETE_BASE_ARM_AFTER:
    bpy.data.objects.remove(base_arm, do_unlink=True)

if DELETE_BASE_MESH_AFTER:
    bpy.data.objects.remove(base_mesh, do_unlink=True)

print("Done.")
print("Final mesh:", joined_mesh.name)
print("Final armature:", chain_arm.name, "bones:", CHAIN_COUNT, "segments:", CHAIN_COUNT)
print("Material:", ("IMAGE" if MATERIAL_IMAGE_PATH else "HEX"), MATERIAL_IMAGE_PATH or MATERIAL_HEX)
