"""Extend the visible rounded plectrum on the exposed side of its original grip."""
import argparse
import json
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix
from mathutils.kdtree import KDTree

if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).parent))

from blender_contact_assets import mesh_data, triangulate_visible, write_contact_arrays


def build_extended_pick(prepared, profile, out, extension_m=.010):
    bpy.ops.wm.open_mainfile(filepath=str(prepared/'pick.blend'))
    objects = [obj for obj in bpy.context.scene.objects if obj.type == 'MESH']
    if len(objects) != 1 or objects[0].name != 'Pick':
        raise ValueError('Expected the single original visible Pick mesh')
    obj = objects[0]
    if obj.matrix_world != Matrix.Identity(4):
        raise ValueError('Original pick must use the verified identity tool frame')
    old_vertices, old_faces, old_topology = mesh_data(obj)
    if not old_topology['watertight']:
        raise ValueError('Original plectrum is open')
    source = profile['pick']
    original_obj = Path(source['mesh_path'])
    lines = original_obj.read_text().splitlines()
    bound_vertices = [tuple(float(value) for value in line.split()[1:4])
                      for line in lines if line.startswith('v ')]
    if len(bound_vertices) != len(old_vertices):
        raise ValueError('Original .blend/OBJ pick vertex count differs')
    tree = KDTree(len(bound_vertices))
    for index, vertex in enumerate(bound_vertices):
        tree.insert(vertex, index)
    tree.balance()
    # Blender's original OBJ exporter rounded coordinates to six decimals.
    if max(tree.find(vertex)[2] for vertex in old_vertices) > 1e-6:
        raise ValueError('Original .blend plectrum geometry differs from verified OBJ')

    mesh = obj.data
    index_map = [tree.find(vertex.co)[1] for vertex in mesh.vertices]
    source_faces = sorted(tuple(sorted(index_map[index] for index in face.vertices)) for face in mesh.polygons)
    bound_faces = sorted(tuple(sorted(int(token.split('/')[0])-1 for token in line.split()[1:]))
                         for line in lines if line.startswith('f '))
    if source_faces != bound_faces:
        raise ValueError('Original .blend plectrum topology differs from verified OBJ')
    old_local = np.asarray([list(vertex.co) for vertex in mesh.vertices])
    start_z = -.0015
    tip_z = float(old_local[:, 2].min())
    extension = float(extension_m)
    if not np.isfinite(extension) or extension <= 0:
        raise ValueError('Exposed pick extension must be positive and finite')
    half_x = float(old_local[:, 0].max())
    if abs(tip_z+.004) > 1e-7 or abs(half_x-.0025) > 1e-7:
        raise ValueError('Unexpected original plectrum frame or size')
    for vertex in mesh.vertices:
        x, y, z = vertex.co
        if z >= start_z:
            continue
        blend = min(1., max(0., (start_z-z)/(start_z-tip_z)))
        vertex.co = (half_x+(x-half_x)*(1-.35*blend), y*(1-.28*blend), z-extension*blend)
    mesh.update()
    triangulate_visible(obj)
    vertices, faces, topology = mesh_data(obj)
    if not topology['watertight'] or topology['volume_mesh_m3'] <= 0:
        raise ValueError('Extended plectrum must remain one closed oriented solid')

    item = write_contact_arrays(vertices, faces, [dict(name=obj.name, topology=topology)],
                                'pick', out, 'pick_local', .00035, .012, [obj.name], ['/root/Pick'])
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.wm.usd_export(filepath=str(out/'pick.usdc'), selected_objects_only=True,
                         export_materials=True, generate_preview_surface=True,
                         export_textures=True, relative_paths=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(out/'pick.blend'))
    after = np.asarray([list(vertex.co) for vertex in mesh.vertices])
    retained = old_local[:, 2] >= start_z
    if not np.array_equal(old_local[retained], after[retained]):
        raise ValueError('Visible grip moved during plectrum extension')
    report = dict(
        extension_local_minus_z_m=extension, unchanged_grip_at_and_above_local_z_m=start_z,
        original_bounds_m=[np.min(old_vertices, axis=0).tolist(), np.max(old_vertices, axis=0).tolist()],
        bounds_m=item['bounds'], original_grip_vertices_unchanged=int(retained.sum()),
        original_triangles=len(old_faces), triangles=len(faces), watertight=True,
        loaded_side_axis_local=[1, 0, 0], tip_axis_local=[0, 0, -1],
        materials=[material.name for material in mesh.materials],
        design='Deform the original rounded exposed end into a single continuous tapered blade.')
    (out/'plectrum_geometry.json').write_text(json.dumps(report, indent=2)+'\n')
    source.update(path=str(out/'pick.usdc'), mesh_path=item['obj_path'],
                  collision_mesh_npz=item['mesh_path'], collision_mesh_sha256=item['obj_sha256'],
                  collision_npz_sha256=item['sha256'], visual_prim_paths=item['visual_prim_paths'],
                  bounds=item['bounds'], tip_extension_m=extension, tip_axis_local=[0, 0, -1],
                  dimensions_m=np.subtract(item['bounds'][1],item['bounds'][0]).tolist(),
                  loaded_side_axis_local=[1, 0, 0], geometry_report=str(out/'plectrum_geometry.json'),
                  geometry_note='Continuous tapered extension of the Blender-authored exposed pick end. The verified grip and tool frame are preserved.')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--profile', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(sys.argv[sys.argv.index('--')+1:])
    profile = json.loads(args.profile.read_text())
    original = json.loads((args.prepared/'asset_profile.json').read_text())
    old_pick = profile['pick']
    profile['pick'] = original['pick']
    report = build_extended_pick(args.prepared, profile, args.out, extension_m=.016)
    new_pick = profile['pick']
    for key in ('attachment_link', 'attachment_transform', 'default_prim', 'meters_per_unit', 'up_axis'):
        new_pick[key] = old_pick[key]
    with np.load(old_pick['collision_mesh_npz'], allow_pickle=False) as old, \
         np.load(new_pick['collision_mesh_npz'], allow_pickle=False) as new:
        grip = old['vertices'][:, 2] >= report['unchanged_grip_at_and_above_local_z_m']
        if not np.array_equal(old['vertices'][grip], new['vertices'][grip]):
            raise ValueError('Wide plectrum changed the verified grip vertices')

        def grip_triangles(faces):
            return sorted(map(tuple, np.sort(faces[grip[faces].all(1)], axis=1)))

        triangles = grip_triangles(old['faces'])
        if triangles != grip_triangles(new['faces']):
            raise ValueError('Wide plectrum changed the verified grip triangles')
        report.update(v3_grip_vertices_identical=int(grip.sum()),
                      v3_grip_triangles_identical=len(triangles),
                      additional_tip_extension_m=float(old['vertices'][:, 2].min()-new['vertices'][:, 2].min()))
    (args.out/'plectrum_geometry.json').write_text(json.dumps(report, indent=2)+'\n')
    (args.out/'contact_profile.json').write_text(json.dumps(profile, indent=2)+'\n')


if __name__ == '__main__':
    main()
