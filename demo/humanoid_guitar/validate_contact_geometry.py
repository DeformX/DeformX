"""Audit the visible USD and exact collision triangles in their shared frames."""

import argparse
import hashlib
import json
from collections import defaultdict
from itertools import permutations
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from pxr import Usd, UsdGeom, UsdShade, UsdUtils


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def visual(stage, paths, link=None):
    points, triangles, mesh_paths = [], [], set()
    cache = UsdGeom.XformCache()
    inverse = np.eye(4)
    if link:
        inverse = np.linalg.inv(np.asarray(cache.GetLocalToWorldTransform(stage.GetPrimAtPath(link))).T)
    for path in paths:
        prim = stage.GetPrimAtPath(path)
        if not prim:
            raise ValueError('Missing visual prim '+path)
        for child in Usd.PrimRange(prim):
            if not child.IsA(UsdGeom.Mesh):
                continue
            child_path = str(child.GetPath())
            if child_path in mesh_paths:
                raise ValueError('Repeated visible mesh '+child_path)
            mesh_paths.add(child_path)
            mesh = UsdGeom.Mesh(child)
            vertices = np.asarray(mesh.GetPointsAttr().Get(), dtype=float)
            counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get())
            faces = np.asarray(mesh.GetFaceVertexIndicesAttr().Get())
            if not np.all(counts == 3):
                raise ValueError('Expected visible triangles on '+child_path)
            transform = inverse @ np.asarray(cache.GetLocalToWorldTransform(child)).T
            world = vertices @ transform[:3, :3].T + transform[:3, 3]
            offset = len(points)
            points.extend(world)
            triangles.extend(faces.reshape(-1, 3)+offset)
    if not mesh_paths:
        raise ValueError(f'No visible triangles at {paths}')
    return np.asarray(points), np.asarray(triangles), mesh_paths


def compare_triangles(vertices, faces, collision_vertices, collision_faces):
    corners = np.asarray(vertices, dtype=np.float64)[faces]
    order = np.lexsort((corners[:, :, 2], corners[:, :, 1], corners[:, :, 0]), axis=1)
    keys = np.take_along_axis(corners, order[:, :, None], axis=1).reshape(-1, 9)
    unique, visible_ids = np.unique(keys, axis=0, return_inverse=True)
    ordered = unique.reshape(-1, 3, 3)
    # Compare all corner orders. Near-equal coordinates can swap sorting order
    # after Blender's float transform even though the physical triangle agrees.
    variants = np.concatenate([ordered[:, perm].reshape(-1, 9) for perm in permutations(range(3))])
    measured = np.asarray(collision_vertices, dtype=np.float64)[collision_faces].reshape(-1, 9)
    errors, candidate_ids = cKDTree(variants).query(measured)
    collision_ids = candidate_ids % len(unique)
    multiplicity = np.array_equal(np.bincount(visible_ids, minlength=len(unique)),
                                  np.bincount(collision_ids, minlength=len(unique)))
    return float(errors.max()), multiplicity


def distance(a, b):
    return float(max(cKDTree(a).query(b)[0].max(), cKDTree(b).query(a)[0].max()))


def load_collision(item):
    if sha(item['mesh_path']) != item['sha256'] or sha(item['obj_path']) != item['obj_sha256']:
        raise ValueError('Collision mesh checksum mismatch: '+item['id'])
    with np.load(item['mesh_path'], allow_pickle=False) as mesh:
        vertices, faces = mesh['vertices'], mesh['faces']
    if not np.isfinite(vertices).all() or faces.min() < 0 or faces.max() >= len(vertices):
        raise ValueError('Invalid collision vertices or indices: '+item['id'])
    lines = Path(item['obj_path']).read_text().splitlines()
    obj_v = np.array([[float(value) for value in line.split()[1:]] for line in lines if line.startswith('v ')])
    obj_f = np.array([[int(index)-1 for index in line.split()[1:]] for line in lines if line.startswith('f ')])
    if not np.array_equal(faces, obj_f) or not np.allclose(vertices, obj_v, atol=1e-7, rtol=0):
        raise ValueError('Collision OBJ and NPZ differ: '+item['id'])
    edges = np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
    _, counts = np.unique(np.sort(edges, axis=1), axis=0, return_counts=True)
    closed = bool(np.all(counts == 2))
    if closed != item['watertight']:
        raise ValueError('Incorrect closure metadata: '+item['id'])
    return vertices, faces


def compare_group(stage, entries, link=None):
    paths = sorted({path for item in entries for path in item['visual_prim_paths']})
    vertices, faces, mesh_paths = visual(stage, paths, link)
    buffers, indices = [], []
    offset = 0
    for item in entries:
        local_vertices, local_faces = load_collision(item)
        buffers.append(local_vertices)
        indices.append(local_faces+offset)
        offset += len(local_vertices)
    collision_vertices = np.concatenate(buffers)
    collision_faces = np.concatenate(indices)
    vertex_error = distance(vertices, collision_vertices)
    centroid_error = distance(vertices[faces].mean(1), collision_vertices[collision_faces].mean(1))
    triangle_error, multiplicity = compare_triangles(vertices, faces, collision_vertices, collision_faces)
    label = entries[0]['link'] if link else entries[0]['id']
    if len(faces) != len(collision_faces) or max(vertex_error, centroid_error, triangle_error) >= 1e-7:
        raise ValueError(f'Visible/collision geometry mismatch {label}: {vertex_error}, {centroid_error}, {triangle_error}')
    if not multiplicity:
        raise ValueError('Missing or duplicated collision triangles: '+label)
    if link:
        expected = {str(prim.GetPath()) for prim in Usd.PrimRange(stage.GetPrimAtPath(link))
                    if prim.IsA(UsdGeom.Mesh)}
        if expected != mesh_paths:
            raise ValueError(f'Uncovered visible mesh on {label}: {sorted(expected-mesh_paths)}')
    return dict(id=label, max_vertex_difference_m=vertex_error,
                max_triangle_centroid_difference_m=centroid_error, max_triangle_vertex_difference_m=triangle_error,
                visible_triangles=len(faces), collision_triangles=len(collision_faces),
                watertight=all(item['watertight'] for item in entries),
                triangle_multiplicity_matches=True,
                collider_ids=[item['id'] for item in entries], visual_mesh_paths=sorted(mesh_paths))


def dependencies(profile, label):
    path = profile[label]['path']
    layers, other, missing = UsdUtils.ComputeAllDependencies(path)
    paths = [layer.identifier for layer in layers] + list(other)
    manifest = {entry['path']: entry['sha256'] for entry in profile[label]['files']}
    if missing or any(item not in manifest or sha(item) != manifest[item] for item in paths):
        raise ValueError(f'Unbound USD dependencies on {label}: {missing}')
    return paths


def material_signature(prim):
    material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
    if not material:
        return ()
    return tuple((str(shader.GetPath()), tuple(sorted((attr.GetName(), str(attr.Get()))
                 for attr in shader.GetAttributes()))) for shader in Usd.PrimRange(material.GetPrim()))


def compare_visual_bindings(new_stage, original_path, repaired_mesh_paths):
    original = Usd.Stage.Open(original_path)
    old_meshes = {str(prim.GetPath()): prim for prim in original.Traverse() if prim.IsA(UsdGeom.Mesh)}
    new_meshes = {str(prim.GetPath()): prim for prim in new_stage.Traverse() if prim.IsA(UsdGeom.Mesh)}
    if set(old_meshes) != set(new_meshes):
        raise ValueError('Exported render mesh paths differ from the original asset')
    original_cache, new_cache = UsdGeom.XformCache(), UsdGeom.XformCache()
    for path, old in old_meshes.items():
        new = new_meshes[path]
        if not np.allclose(np.asarray(original_cache.GetLocalToWorldTransform(old)),
                           np.asarray(new_cache.GetLocalToWorldTransform(new)), atol=1e-8, rtol=0):
            raise ValueError('Visible transform changed: '+path)
        if material_signature(old) != material_signature(new):
            raise ValueError('Visible material changed: '+path)
        if path not in repaired_mesh_paths:
            old_geom, new_geom = UsdGeom.Mesh(old), UsdGeom.Mesh(new)
            for accessor in ('GetPointsAttr', 'GetFaceVertexCountsAttr', 'GetFaceVertexIndicesAttr'):
                if not np.array_equal(np.asarray(getattr(old_geom, accessor)().Get()),
                                      np.asarray(getattr(new_geom, accessor)().Get())):
                    raise ValueError('Unrepaired visible mesh changed: '+path)
    return dict(visible_mesh_count=len(old_meshes), transforms_equal=True, material_bindings_equal=True,
                repaired_mesh_paths=sorted(repaired_mesh_paths))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', type=Path, required=True)
    args = parser.parse_args()
    profile = json.loads(args.profile.read_text())
    stages = {label: Usd.Stage.Open(profile[label]['path']) for label in ('guitar', 'robot', 'pick')}
    report = [compare_group(stages['guitar'], [entry]) for entry in profile['guitar']['contact_meshes']]
    grouped = defaultdict(list)
    for entry in profile['robot']['contact_meshes']:
        if entry['frame'] != 'link_local' or not np.array_equal(entry['link_from_mesh'], np.eye(4)):
            raise ValueError('Expected exact link-local hand geometry: '+entry['id'])
        grouped[entry['link']].append(entry)
    links = profile['robot']['link_prim_paths']
    hand_report = [compare_group(stages['robot'], entries, links[link]) for link, entries in grouped.items()]
    report.extend(hand_report)
    resolved = {label: dependencies(profile, label) for label in stages}
    result = dict(visible_collision_identity=report, usd_missing_dependencies=[],
                  usd_resolved_dependencies=resolved['guitar'], all_usd_dependencies=resolved,
                  baseline_unchanged=profile['contact_asset_provenance']['baseline_unchanged'])

    if profile['contact_schema_version'] >= 2:
        expected = {link for link in links if link.startswith(('L_', 'R_'))}
        if set(grouped) != expected or len(expected) != 26:
            raise ValueError(f'Full hand collision scope differs: {sorted(expected^set(grouped))}')
        binding = profile['robot']['hand_coverage']
        if sha(binding['path']) != binding['sha256']:
            raise ValueError('Hand coverage manifest checksum mismatch')
        coverage = json.loads(Path(binding['path']).read_text())
        registered = profile['robot']['contact_meshes']
        by_id = {item['id']: item for item in registered}
        if len(by_id) != len(registered) or set(by_id) != {item['id'] for item in coverage['colliders']}:
            raise ValueError('Registered hand collider IDs differ from the coverage manifest')
        if len(registered) != coverage['hand_colliders'] or len(registered) != len(coverage['colliders']):
            raise ValueError('Registered hand collider count differs from the coverage manifest')
        for item in coverage['colliders']:
            if any(by_id[item['id']].get(key) != value for key, value in item.items()):
                raise ValueError('Registered collider differs from the coverage manifest: '+item['id'])
        inventory = {entry['link']: entry for entry in coverage['inventory']}
        if set(inventory) != expected or len(coverage['inventory']) != len(inventory):
            raise ValueError('Coverage manifest link inventory differs')
        for row in hand_report:
            item = inventory[row['id']]
            if item['collider_ids'] != row['collider_ids'] or item['triangles'] != row['visible_triangles']:
                raise ValueError('Visible link coverage differs from the manifest: '+row['id'])
            if item['shell_count'] != len(row['collider_ids']):
                raise ValueError('Visible link shell count differs from the manifest: '+row['id'])
        for link, entries in grouped.items():
            if {item['shell_index'] for item in entries} != set(range(len(entries))):
                raise ValueError('Incomplete closed-shell indices: '+link)
            if any(item['shell_count'] != len(entries) or not item['watertight'] for item in entries):
                raise ValueError('Inconsistent closed-shell inventory: '+link)
            if any(item['side'] != ('left' if link.startswith('L_') else 'right') for item in entries):
                raise ValueError('Incorrect hand side: '+link)
        pads = [entry for entries in grouped.values() for entry in entries if entry['function'] == 'fretting_pad']
        if len(pads) != 1 or pads[0]['id'] != 'L_index_intermediate':
            raise ValueError('Expected the measured left index fretting pad')
        pick = profile['pick']
        report.append(compare_group(stages['pick'], [dict(id='pick', mesh_path=pick['collision_mesh_npz'],
                     obj_path=pick['mesh_path'], sha256=pick['collision_npz_sha256'],
                     obj_sha256=pick['collision_mesh_sha256'], watertight=True, visual_prim_paths=pick['visual_prim_paths'])]))
        result['arm_clearance_geometry'] = [compare_group(stages['robot'], [entry], links[entry['link']])
                                            for entry in profile['robot']['arm_clearance_meshes']]
        originals = profile['contact_asset_provenance']['full_hand_baseline_assets']
        repaired = {path for item in hand_report
                    if item['id'] in {'L_hand_base_link', 'L_pinky_intermediate', 'R_pinky_intermediate'}
                    for path in item['visual_mesh_paths']}
        result['robot_visual_binding'] = compare_visual_bindings(stages['robot'], originals['robot'], repaired)
        result['pick_visual_binding'] = compare_visual_bindings(stages['pick'], originals['pick'],
                                                                 set(report[-1]['visual_mesh_paths']))
        result['complete_hand_links'] = sorted(expected)
        result['hand_collider_count'] = sum(map(len, grouped.values()))
        result['hand_coverage_sha256'] = binding['sha256']
    (args.profile.resolve().parent/'contact_geometry_validation.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
