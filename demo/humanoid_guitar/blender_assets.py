"""Blender-only mesh import, inspection, conversion and simple prop authoring."""
import argparse
import json
import math
import sys
from pathlib import Path
import bpy
from mathutils import Matrix, Vector


def clear():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)


def bounds(obj):
    points = [obj.matrix_world @ Vector(v) for v in obj.bound_box]
    return [[min(p[i] for p in points) for i in range(3)], [max(p[i] for p in points) for i in range(3)]]


def material(name, rgba, metallic=0., roughness=.4):
    mat = bpy.data.materials.new(name)
    mat.diffuse_color = rgba
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get('Principled BSDF')
    bsdf.inputs['Base Color'].default_value = rgba
    bsdf.inputs['Metallic'].default_value = metallic
    bsdf.inputs['Roughness'].default_value = roughness
    return mat


def select(objects):
    bpy.ops.object.select_all(action='DESELECT')
    for obj in objects:
        obj.select_set(True)


def export(path, objects):
    select(objects)
    bpy.ops.wm.usd_export(filepath=str(path), selected_objects_only=True,
                         export_materials=True, generate_preview_surface=True,
                         export_textures=True, relative_paths=True)


def import_mesh(path):
    before = set(bpy.data.objects)
    suffix = Path(path).suffix.lower()
    if suffix == '.stl':
        bpy.ops.wm.stl_import(filepath=str(path))
    elif suffix == '.obj':
        bpy.ops.wm.obj_import(filepath=str(path))
    elif suffix in ('.glb', '.gltf'):
        bpy.ops.import_scene.gltf(filepath=str(path))
    elif suffix == '.dae':
        bpy.ops.wm.collada_import(filepath=str(path))
    else:
        raise ValueError('Unsupported original mesh '+str(path))
    return list(set(bpy.data.objects)-before)


def empty(name, parent=None, matrix=None):
    obj = bpy.data.objects.new(name, None)
    bpy.context.collection.objects.link(obj)
    obj.parent = parent
    if matrix is not None:
        obj.matrix_local = Matrix(matrix)
    return obj


def inspect_guitar(spec):
    clear()
    objects = import_mesh(spec['guitar_glb'])
    result = []
    for obj in objects:
        if obj.type != 'MESH':
            continue
        result.append(dict(name=obj.name, mesh=obj.data.name, vertices=len(obj.data.vertices),
                           polygons=len(obj.data.polygons), bounds=bounds(obj),
                           materials=[m.name if m else None for m in obj.data.materials],
                           matrix=[list(row) for row in obj.matrix_world]))
    out = Path(spec['output'])
    (out/'guitar_inspect.json').write_text(json.dumps(result, indent=2)+'\n')
    bpy.ops.wm.save_as_mainfile(filepath=str(out/'guitar_original.blend'))
    print(json.dumps(result, indent=2))
    return objects


def build(spec, objects):
    out = Path(spec['output'])
    # glTF importer has already converted the source to Blender meter/Z-up.
    # Normalize with a proper rigid rotation, origin at bridge saddle center.
    normalize = Matrix(((1,0,0,0),(0,0,1,-.129),(0,-1,0,-.0455),(0,0,0,1)))
    strings = sorted([o for o in objects if o.name.startswith('String ')], key=lambda o:o.name)
    if len(strings) != 6:
        raise ValueError('Expected the original six named string meshes')
    nut, bridge, radii, removed = [], [], [], []
    for obj in strings:
        # Recover four cross-sections from true mesh coordinates. glTF has seam
        # duplicates, so use bounding center rather than biased vertex average.
        points = [obj.matrix_world @ v.co for v in obj.data.vertices]
        groups = []
        for i in sorted(range(len(points)), key=lambda i:points[i].z):
            if not groups or points[i].z-points[groups[-1][-1]].z > .001:
                groups.append([])
            groups[-1].append(i)
        if len(groups) != 4:
            raise ValueError('Unexpected source string topology '+obj.name)
        centers = [Vector([(min(points[i][k] for i in group)+max(points[i][k] for i in group))/2 for k in range(3)]) for group in groups]
        b, n = centers[1], centers[2]
        bridge.append(list(normalize@b)); nut.append(list(normalize@n))
        radii.append((max(points[i].x for i in groups[1])-min(points[i].x for i in groups[1]))/2)
        # Retain source tail and tuner spans. Remove only quads/triangles joining
        # the saddle cross-section to the nut cross-section, replaced by solver.
        middle = set(groups[1]+groups[2])
        faces = []
        count = 0
        for poly in obj.data.polygons:
            ids = list(poly.vertices)
            playable = all(i in middle for i in ids) and any(i in groups[1] for i in ids) and any(i in groups[2] for i in ids)
            if playable:
                count += 1
            else:
                faces.append(ids)
        mesh = bpy.data.meshes.new(obj.name+'_nonplayable_source')
        mesh.from_pydata([v.co[:] for v in obj.data.vertices], [], faces)
        for mat in obj.data.materials:
            mesh.materials.append(mat)
        obj.data = mesh
        removed.append(dict(name=obj.name, removed_playable_faces=count, bridge_source=list(b), nut_source=list(n)))
        if count == 0:
            raise ValueError('No playable source faces removed')
    root = empty('Guitar')
    for obj in objects:
        if obj.parent is None:
            world = obj.matrix_world.copy()
            obj.parent = root
            obj.matrix_world = normalize @ world
    export(out/'guitar.usdc', [root]+objects)
    bpy.ops.wm.save_as_mainfile(filepath=str(out/'guitar.blend'))
    allbounds = [bounds(o) for o in objects if o.type=='MESH']
    guitar = dict(path=str(out/'guitar.usdc'), default_prim='/root', meters_per_unit=1, up_axis='Z',
                  placement=Matrix.Identity(4)[:], nut_endpoints_local_m=nut, bridge_endpoints_local_m=bridge,
                  source_string_radius_m=radii, face_normal_local=[0,0,1], transverse_axis_local=[1,0,0],
                  neck_axis_local=[0,1,0], source_to_local=[list(r) for r in normalize],
                  bounds=[[min(b[0][k] for b in allbounds) for k in range(3)], [max(b[1][k] for b in allbounds) for k in range(3)]],
                  suppressed_playable_geometry=removed)
    clear()
    robot_root = empty('Robot')
    k = spec['kinematics']
    # Export flat, independently addressable link Xforms. FK computes world link
    # transforms from actual URDF; visuals remain strictly link-local.
    frames = {n:Matrix(m) for n,m in k['roots'].items()}
    pending = k['joints'].copy()
    while pending:
        ready = [j for j in pending if j['parent'] in frames]
        if not ready:
            raise ValueError('Unresolved URDF tree')
        for j in ready:
            frames[j['child']] = frames[j['parent']] @ Matrix(j['parent_frame'])
            pending.remove(j)
    link_paths = {}
    for name, visuals in spec['links'].items():
        link = empty(name, robot_root, frames[name])
        link_paths[name] = '/root/Robot/'+name
        for v in visuals:
            imported = import_mesh(v['path'])
            for index, obj in enumerate(imported):
                if obj.type != 'MESH':
                    continue
                # STL imported matrix is identity; preserve importer transform
                # for other mesh formats before applying the URDF visual frame.
                imported_world = obj.matrix_world.copy()
                obj.name = 'visual_'+str(v['index'])+'_'+str(index)
                obj.parent = link
                obj.matrix_local = Matrix(v['transform']) @ Matrix.Diagonal(Vector(v['scale']).to_4d()) @ imported_world
                if v['rgba'] is not None:
                    obj.data.materials.clear()
                    obj.data.materials.append(material(name+'_original_material', tuple(v['rgba']), .35, .32))
    export(out/'robot.usdc', list(bpy.data.objects))
    bpy.ops.wm.save_as_mainfile(filepath=str(out/'robot.blend'))
    # Seated setup is an articulated prescribed pose, not a balance simulation.
    k['placement'][2][3] = .68
    k['preset'].update({side+'_hip_pitch_joint':-1.5 for side in ('left','right')})
    k['preset'].update({side+'_knee_joint':1.5 for side in ('left','right')})
    k['preset'].update(left_hip_roll_joint=.08, right_hip_roll_joint=-.08)
    robot = dict(path=str(out/'robot.usdc'), default_prim='/root', meters_per_unit=1, up_axis='Z', kinematics=k,
                 link_prim_paths=link_paths, joint_source='Pinned official Unitree URDF; no inferred dimensions')
    clear()
    pick_mat = material('Amber_plectrum',(1.,.27,.025,1), .12, .25)
    bpy.ops.mesh.primitive_cube_add(size=1)
    pick_obj = bpy.context.object
    pick_obj.name = 'Pick'
    pick_obj.dimensions = (.005,.016,.008)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    bevel = pick_obj.modifiers.new('Rounded_contact_edges','BEVEL')
    bevel.width=.0023; bevel.segments=6
    bpy.context.view_layer.objects.active=pick_obj
    bpy.ops.object.modifier_apply(modifier=bevel.name)
    pick_obj.data.materials.append(pick_mat)
    for poly in pick_obj.data.polygons: poly.use_smooth=True
    export(out/'pick.usdc',[pick_obj])
    # OBJ exports the exact same evaluated closed surface for SDF/contact bake.
    select([pick_obj])
    bpy.ops.wm.obj_export(filepath=str(out/'pick.obj'), export_selected_objects=True,
                         forward_axis='Y', up_axis='Z', export_materials=False)
    bpy.ops.wm.save_as_mainfile(filepath=str(out/'pick.blend'))
    pick = dict(path=str(out/'pick.usdc'), default_prim='/root', mesh_path=str(out/'pick.obj'),
                meters_per_unit=1, up_axis='Z', dimensions_m=[.005,.016,.008], bevel_radius_m=.0023,
                attachment_link=k['tcp_link'], attachment_transform=[list(r) for r in Matrix.Identity(4)],
                geometry_note='Blender authored rounded closed plectrum; OBJ and USD use the identical applied-bevel mesh; dimensions in guitar local axes')
    clear()
    seatmat=material('Satin_charcoal',(.026,.034,.044,1),.12,.48)
    legmat=material('Brushed_steel',(.13,.15,.17,1),.7,.3)
    bpy.ops.mesh.primitive_cylinder_add(vertices=64, radius=.19, depth=.065, location=(-.035,0,.5275))
    seat=bpy.context.object; seat.name='Seat'; seat.data.materials.append(seatmat)
    bevel=seat.modifiers.new('Upholstery_edge','BEVEL'); bevel.width=.018; bevel.segments=4
    bpy.context.view_layer.objects.active=seat; bpy.ops.object.modifier_apply(modifier=bevel.name)
    for x in (-.14,.07):
        for y in (-.12,.12):
            bpy.ops.mesh.primitive_cylinder_add(vertices=24,radius=.014,depth=.49,location=(x,y,.245))
            bpy.context.object.name='Stool_leg'; bpy.context.object.data.materials.append(legmat)
    export(out/'stool.usdc',list(bpy.data.objects))
    bpy.ops.wm.save_as_mainfile(filepath=str(out/'stool.blend'))
    stool=dict(path=str(out/'stool.usdc'),meters_per_unit=1,up_axis='Z',seat_top_m=.56,
               note='Original simple support stool authored in Blender; pose placement must be checked against pelvis bounds')
    clear()
    bpy.ops.mesh.primitive_plane_add(size=200)
    floor=bpy.context.object; floor.name='Floor'; floor.data.materials.append(material('Warm_studio_floor',(.19,.18,.16,1),0,.68))
    export(out/'floor.usdc',[floor])
    result=dict(schema_version=1, robot=robot,guitar=guitar,pick=pick,stool=stool,
                floor=dict(path=str(out/'floor.usdc'),meters_per_unit=1,up_axis='Z'),
                task='Fixed-pelvis seated humanoid with prescribed joint kinematics and solver-deformed guitar strings; no dynamic balance or grasp-force claim')
    # mathutils slices contain vectors; serialize simple lists only.
    result['guitar']['placement']=[list(r) for r in Matrix.Identity(4)]
    (out/'asset_profile.json').write_text(json.dumps(result,indent=2)+'\n')
    print('ASSET_PROFILE',out/'asset_profile.json')
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--spec', type=Path, required=True)
    p.add_argument('--inspect-only', action='store_true')
    a = p.parse_args(sys.argv[sys.argv.index('--')+1:])
    spec = json.loads(a.spec.read_text())
    objects = inspect_guitar(spec)
    if a.inspect_only:
        return
    build(spec, objects)


if __name__ == '__main__':
    main()
