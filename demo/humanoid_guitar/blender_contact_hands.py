"""Export the official hand meshes as closed, visible link-local contact solids."""
import json
import re

import bmesh
import bpy
import numpy as np
from mathutils.bvhtree import BVHTree

from blender_contact_assets import mesh_data, write_contact_arrays


def require_solid(mesh, label):
    bad_edges = sum(not edge.is_manifold for edge in mesh.edges)
    bad_vertices = sum(not vertex.is_manifold for vertex in mesh.verts)
    if bad_edges or bad_vertices:
        raise ValueError(f'{label}: {bad_edges} nonmanifold edges, {bad_vertices} nonmanifold vertices')


def repair_source_mesh(obj, link):
    """Repair only the known CAD seam defects, on the actual render mesh."""
    mesh = bmesh.new()
    mesh.from_mesh(obj.data)
    bmesh.ops.remove_doubles(mesh, verts=list(mesh.verts), dist=1e-7)
    mesh.normal_update()
    old_tree = BVHTree.FromBMesh(mesh)
    old_points = [vertex.co.copy() for vertex in mesh.verts]
    old_samples = old_points + [face.calc_center_median() for face in mesh.faces]
    old_count = len(mesh.faces)
    volume_source = mesh.copy()
    bmesh.ops.recalc_face_normals(volume_source, faces=list(volume_source.faces))
    old_volume = volume_source.calc_volume(signed=True)
    volume_source.free()
    report = {'link': link, 'source_triangles': old_count, 'source_signed_volume_m3': old_volume}

    if link == 'L_hand_base_link':
        flap = [face for face in mesh.faces
                if sum(edge.is_boundary for edge in face.edges) == 2
                and any(len(edge.link_faces) > 2 for edge in face.edges)]
        if len(flap) != 1:
            raise ValueError('Expected the single pinned left-palm CAD flap')
        report.update(method='Remove one unsupported coplanar nonmanifold flap',
                      removed_flap_area_m2=flap[0].calc_area(),
                      removed_flap_vertices_m=[list(vertex.co) for vertex in flap[0].verts])
        bmesh.ops.delete(mesh, geom=flap, context='FACES')
    else:
        tolerance = {'L_pinky_intermediate': 15e-6, 'R_pinky_intermediate': 20e-6}[link]
        boundary_vertices = [vertex for vertex in mesh.verts
                             if any(edge.is_boundary for edge in vertex.link_edges)]
        bmesh.ops.remove_doubles(mesh, verts=boundary_vertices, dist=tolerance)
        seam = [edge for edge in mesh.edges if edge.is_boundary]
        report.update(method='Weld boundary vertices on the source CAD seam',
                      boundary_weld_tolerance_m=tolerance, cap_area_m2=0.)
        if link == 'R_pinky_intermediate':
            if len(seam) != 4:
                raise ValueError('Expected one four-edge right-pinky slit')
            caps = bmesh.ops.holes_fill(mesh, edges=seam, sides=4)['faces']
            if len(caps) != 1 or len(caps[0].verts) != 4:
                raise ValueError('Unexpected right-pinky seam cap')
            report['method'] += ' and close one four-edge slit'
            report['cap_area_m2'] = caps[0].calc_area()

    bmesh.ops.triangulate(mesh, faces=list(mesh.faces))
    bmesh.ops.recalc_face_normals(mesh, faces=list(mesh.faces))
    require_solid(mesh, link)
    repaired_tree = BVHTree.FromBMesh(mesh)
    new_points = [vertex.co.copy() for vertex in mesh.verts]
    new_samples = new_points + [face.calc_center_median() for face in mesh.faces]
    new_volume = mesh.calc_volume(signed=True)
    report.update(
        repaired_triangles=len(mesh.faces), repaired_signed_volume_m3=new_volume,
        signed_volume_change_m3=new_volume-old_volume,
        max_source_sample_to_repaired_m=max(repaired_tree.find_nearest(p)[3] for p in old_samples),
        max_repaired_sample_to_source_m=max(old_tree.find_nearest(p)[3] for p in new_samples),
        max_source_vertex_to_repaired_surface_m=max(repaired_tree.find_nearest(p)[3] for p in old_points),
        max_repaired_vertex_to_source_surface_m=max(old_tree.find_nearest(p)[3] for p in new_points),
    )
    limit = .00033 if link == 'L_hand_base_link' else report['boundary_weld_tolerance_m']
    if max(report['max_source_sample_to_repaired_m'], report['max_repaired_sample_to_source_m']) > limit:
        raise ValueError(f'CAD seam repair exceeds measured scope on {link}')
    mesh.to_mesh(obj.data)
    mesh.free()
    obj.data.update()
    return report


def connected_shells(vertices, faces):
    """Return deterministic face-connected shells with independent vertex indices."""
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int32)
    adjacency = [[] for _ in vertices]
    for index, face in enumerate(faces):
        for vertex in face:
            adjacency[vertex].append(index)
    unseen = np.ones(len(faces), dtype=bool)
    shells = []
    for first in range(len(faces)):
        if not unseen[first]:
            continue
        unseen[first] = False
        stack = [first]
        members = []
        while stack:
            face_index = stack.pop()
            members.append(face_index)
            for vertex in faces[face_index]:
                for neighbor in adjacency[vertex]:
                    if unseen[neighbor]:
                        unseen[neighbor] = False
                        stack.append(neighbor)
        member_faces = faces[sorted(members)]
        keep, local = np.unique(member_faces, return_inverse=True)
        shells.append((vertices[keep], local.reshape(-1, 3)))
    return sorted(shells, key=lambda shell: (-len(shell[1]), tuple(shell[0].min(axis=0))))


def shell_topology(vertices, faces):
    edges = np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
    _, counts = np.unique(np.sort(edges, axis=1), axis=0, return_counts=True)
    triangles = vertices[faces]
    signed_volume = float(np.einsum('ij,ij->i', triangles[:, 0], np.cross(triangles[:, 1], triangles[:, 2])).sum()/6)
    return dict(vertices=len(vertices), triangles=len(faces), nonmanifold_edges=int(np.count_nonzero(counts != 2)),
                boundary_edges=int(np.count_nonzero(counts == 1)), volume_mesh_m3=signed_volume,
                watertight=bool(len(counts) and np.all(counts == 2)))


def original_visuals(profile, link):
    root = bpy.data.objects[link]
    objects = sorted((obj for obj in root.children if obj.type == 'MESH'), key=lambda obj: obj.name)
    if len(objects) != 1:
        raise ValueError(f'Expected one direct source visual on {link}, found {len(objects)}')
    obj = objects[0]
    transform = root.matrix_world.inverted() @ obj.matrix_world
    path = profile['robot']['link_prim_paths'][link] + '/' + obj.name.replace('.', '_')
    return obj, transform, path


def build_hand_profile(args):
    out = args.out
    profile = json.loads(args.extend_hands_from.read_text())
    bpy.ops.wm.open_mainfile(filepath=str(args.prepared/'robot.blend'))
    old_contacts = {entry['id']: entry for entry in profile['robot']['contact_meshes']}
    hands = sorted(link for link in profile['robot']['link_prim_paths'] if link.startswith(('L_', 'R_')))
    arm = sorted(link for link in profile['robot']['link_prim_paths']
                 if re.fullmatch(r'(left|right)_(elbow|wrist_(roll|pitch|yaw))_link', link))
    if len(hands) != 26 or len(arm) != 8:
        raise ValueError(f'Unexpected G1 link inventory: {len(hands)} hand, {len(arm)} wrist/forearm')

    repairs = []
    inventory = []
    contacts = []
    for link in hands:
        obj, transform, path = original_visuals(profile, link)
        if link in {'L_hand_base_link', 'L_pinky_intermediate', 'R_pinky_intermediate'}:
            repairs.append(repair_source_mesh(obj, link))
        vertices, faces, topology = mesh_data(obj, transform)
        if not topology['watertight']:
            raise ValueError('Unclosed hand visual '+link)
        shells = connected_shells(vertices, faces)
        expected_shells = {'L_hand_base_link': 8, 'R_hand_base_link': 9}.get(link, 1)
        if len(shells) != expected_shells:
            raise ValueError(f'{link}: expected {expected_shells} source shells, got {len(shells)}')
        ids = []
        for index, (shell_vertices, shell_faces) in enumerate(shells):
            label = link if index == 0 else f'{link}_shell_{index+1:02d}'
            solid = shell_topology(shell_vertices, shell_faces)
            if not solid['watertight'] or solid['volume_mesh_m3'] <= 0:
                raise ValueError(f'{label}: invalid closed solid {solid}')
            entry = write_contact_arrays(shell_vertices, shell_faces, [dict(name=obj.name, topology=solid)],
                                         label, out, 'link_local', .00035, .004, [obj.name], [path])
            entry.update(link=link, link_from_mesh=np.eye(4).tolist(),
                         side='left' if link.startswith('L_') else 'right',
                         function='fretting_pad' if link == 'L_index_intermediate' else 'hand',
                         shell_index=index, shell_count=len(shells))
            if entry['function'] == 'fretting_pad':
                for key in ('pad_point_link_local_m', 'pad_outward_normal_link_local'):
                    entry[key] = old_contacts[link][key]
            contacts.append(entry)
            ids.append(label)
        inventory.append(dict(link=link, visual_prim_path=path, collider_ids=ids,
                              link_from_visual=[list(row) for row in transform],
                              materials=[material.name for material in obj.data.materials],
                              triangles=len(faces), shell_count=len(shells)))

    arm_meshes = []
    audit_dir = out/'arm_clearance'
    audit_dir.mkdir()
    for link in arm:
        obj, transform, path = original_visuals(profile, link)
        vertices, faces, topology = mesh_data(obj, transform)
        entry = write_contact_arrays(vertices, faces, [dict(name=obj.name, topology=topology)],
                                     link, audit_dir, 'link_local', .00035, .004, [obj.name], [path])
        entry.update(link=link, link_from_mesh=np.eye(4).tolist(),
                     side='left' if link.startswith('left') else 'right',
                     function='wrist_forearm_clearance')
        arm_meshes.append(entry)

    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.wm.usd_export(filepath=str(out/'robot.usdc'), selected_objects_only=True,
                         export_materials=True, generate_preview_surface=True,
                         export_textures=True, relative_paths=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(out/'robot.blend'))
    profile['robot'].update(path=str(out/'robot.usdc'), contact_meshes=contacts,
                            arm_clearance_meshes=arm_meshes)
    from blender_contact_pick import build_extended_pick
    build_extended_pick(args.prepared, profile, out)
    profile['contact_schema_version'] = 2
    report = dict(hand_links=len(hands), hand_colliders=len(contacts), arm_clearance_links=len(arm),
                  instrument_colliders=len(profile['guitar']['contact_meshes']), pick_colliders=1,
                  active_mesh_count=len(contacts)+len(profile['guitar']['contact_meshes'])+1,
                  inventory=inventory, repairs=repairs,
                  colliders=[{key: item[key] for key in ('id', 'link', 'frame', 'link_from_mesh', 'side', 'function',
                             'shell_index', 'shell_count', 'mesh_path', 'sha256', 'obj_path', 'obj_sha256', 'visual_prim_paths')}
                             for item in contacts])
    (out/'hand_coverage.json').write_text(json.dumps(report, indent=2)+'\n')
    (out/'contact_profile.json').write_text(json.dumps(profile, indent=2)+'\n')
    print('COMPLETE_HAND_PROFILE', out/'contact_profile.json', report['active_mesh_count'])
