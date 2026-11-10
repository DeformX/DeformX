"""Adapt the pinned MIT 3MF panel and author mounted-interface details in Blender."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import socket
import sys
import xml.etree.ElementTree as ET
import zipfile

import bpy
from mathutils import Matrix

PANEL_SHA256 = '87d961554221eb1cb5d496a41c54a11cd57a169023030bc26c1bfa82e0f1b2f2'
LICENSE_SHA256 = 'cada501eae9fad746f1df474519a7ab6b7d478fc979a643ca97306938b700adf'


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source', type=Path, required=True)
    ap.add_argument('--license', type=Path, required=True)
    ap.add_argument('--socket-usd', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args(sys.argv[sys.argv.index('--') + 1:])
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    if sha(args.source) != PANEL_SHA256 or sha(args.license) != LICENSE_SHA256:
        raise ValueError('Expected pinned MIT panel and license bytes')
    args.out.mkdir(parents=True, exist_ok=False)
    bpy.ops.object.select_all(action='SELECT'); bpy.ops.object.delete(use_global=False)
    bpy.context.scene.unit_settings.system = 'METRIC'
    bpy.context.scene.unit_settings.scale_length = 1.
    def material(name, color, metallic=0.):
        m = bpy.data.materials.new(name); m.diffuse_color = (*color, 1)
        m.use_nodes = True; p = m.node_tree.nodes.get('Principled BSDF')
        p.inputs['Base Color'].default_value = (*color, 1)
        p.inputs['Metallic'].default_value = metallic; p.inputs['Roughness'].default_value = .36
        return m
    graphite = material('Panel graphite', (.025, .033, .045), .65)
    plastic = material('Keystone charcoal', (.022, .028, .037))
    steel = material('Fastener steel', (.35, .39, .43), .85)
    white = material('Port numbering', (.83, .87, .9))
    cyan = material('Selected port label', (.03, .57, .69))
    black = material('Screw recess', (.006, .008, .011))
    physical, decorative, neighbours = [], [], []
    def mesh(name, vertices, faces, mat):
        data = bpy.data.meshes.new(name); data.from_pydata(vertices, [], faces); data.update()
        ob = bpy.data.objects.new(name, data); bpy.context.collection.objects.link(ob)
        ob.data.materials.append(mat); return ob
    def cube(name, loc, dims, mat, objects):
        bpy.ops.mesh.primitive_cube_add(size=1, location=loc); ob = bpy.context.object
        ob.name = name; ob.dimensions = dims
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
        ob.data.materials.append(mat); objects.append(ob); return ob
    # 3MF millimeter unit and build transform are consumed explicitly. The pinned
    # single build item has no transform; reject silently ignored placement data.
    with zipfile.ZipFile(args.source) as archive:
        root = ET.fromstring(archive.read('3D/3dmodel.model'))
    ns = {'m': root.tag.split('}')[0][1:]}
    items = root.findall('m:build/m:item', ns)
    if root.get('unit') != 'millimeter' or len(items) != 1 or items[0].get('transform'):
        raise ValueError('Unrecognized 3MF units/build placement')
    obj = root.find("m:resources/m:object[@id='" + items[0].get('objectid') + "']", ns)
    vertices = [[float(v.get(k)) * .001 for k in 'xyz'] for v in obj.findall('m:mesh/m:vertices/m:vertex', ns)]
    faces = [[int(f.get(k)) for k in ('v1', 'v2', 'v3')] for f in obj.findall('m:mesh/m:triangles/m:triangle', ns)]
    panel = mesh('PanelFace', vertices, faces, graphite)
    # Remove all rear snap retainers; preserve the sourced face/holes/ears. A
    # closed exact intersection gives a 2 mm plate with real open apertures.
    clip = cube('Temporary trimming solid', (0, 0, .00050015), (1., .2, .0030003), graphite, [])
    bpy.context.view_layer.objects.active = panel
    mod = panel.modifiers.new('Rear retainer adaptation', 'BOOLEAN')
    mod.operation = 'INTERSECT'; mod.solver = 'EXACT'; mod.object = clip
    bpy.ops.object.modifier_apply(modifier=mod.name); bpy.data.objects.remove(clip, do_unlink=True)
    face_y, center_x, center_z = .537, .5, 1.25
    # Swap the raw depth/height axes and reverse winding under this reflection.
    for v in panel.data.vertices:
        x, y, z = v.co; v.co = (center_x + x, face_y + z, center_z + y)
    bpy.context.view_layer.objects.active = panel
    bpy.ops.object.mode_set(mode='EDIT'); bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.normals_make_consistent(inside=False); bpy.ops.object.mode_set(mode='OBJECT')
    physical.append(panel)
    # Original socket mesh is reused for 23 neighbours. The active socket is a
    # separate unchanged PhysX actor supplied by prepare.py, not this visual copy.
    before = set(bpy.data.objects)
    bpy.ops.wm.usd_import(filepath=str(args.socket_usd))
    imported = [o for o in bpy.data.objects if o not in before]
    candidates = [o for o in imported if o.type == 'MESH']
    if len(candidates) != 1: raise ValueError('Expected one prepared socket mesh')
    template = candidates[0].data.copy()
    for ob in imported: bpy.data.objects.remove(ob, do_unlink=True)
    template.materials.clear(); template.materials.append(plastic)
    socket_origin_y = face_y + .0022 + .01275
    socket_z = center_z - .0025  # asymmetric Newton latch envelope within native opening
    centers = [center_x + (-207 + 18 * i) * .001 for i in range(24)]
    for i, x in enumerate(centers):
        # Carrier ring clears the original 16.7 x 12.7 mm housing by 0.15 mm;
        # rings attach to the plate beyond the aperture, with no lower shelf.
        ring_vertices = []
        for y in (face_y + .0019, face_y + .006):
            for hx, hz in ((.0105, .0095), (.0085, .0065)):
                ring_vertices.extend([(x-hx,y,socket_z-hz),(x+hx,y,socket_z-hz),
                                      (x+hx,y,socket_z+hz),(x-hx,y,socket_z+hz)])
        ring_faces = []
        for j in range(4):
            k = (j+1)%4
            ring_faces += [(j,k,k+8,j+8),(j+4,j+12,k+12,k+4),
                           (j,j+4,k+4,k),(j+8,k+8,k+12,j+12)]
        physical.append(mesh(f'Carrier{i+1:02d}', ring_vertices, [tuple(reversed(f)) for f in ring_faces], plastic))
        if i != 6:
            ob = bpy.data.objects.new(f'Socket{i + 1:02d}', template)
            bpy.context.collection.objects.link(ob); ob.location = (x, socket_origin_y, socket_z); neighbours.append(ob)
        bpy.ops.object.text_add(location=(x, face_y - .0003, center_z + .014))
        text = bpy.context.object; text.name = f'PortNumber{i + 1:02d}'
        text.rotation_euler = (math.pi / 2, 0, 0)
        text.data.body = f'{i + 1:02d}'; text.data.align_x = 'CENTER'; text.data.size = .0038
        text.data.extrude = .00002; text.data.materials.append(cyan if i == 6 else white)
        bpy.ops.object.convert(target='MESH'); decorative.append(bpy.context.object)
    # Four visible mounting screws coincide with the imported panel's 1U ears.
    for x in (center_x - .2325, center_x + .2325):
        for z in (center_z - .015875, center_z + .015875):
            bpy.ops.mesh.primitive_cylinder_add(vertices=32, radius=.0042, depth=.003, location=(x, face_y - .0015, z), rotation=(math.pi/2, 0, 0))
            ob = bpy.context.object; ob.name = 'MountingScrew'; ob.data.materials.append(steel); decorative.append(ob)
            for dims in ((.004, .0002, .0007), (.0007, .0002, .004)):
                cube('ScrewCross', (x, face_y - .0031, z), dims, black, decorative)
    # Keep the validated pickup fixture/guides; the old shelf/back/foot are absent.
    fixture_objects = []
    for i, (x, y) in enumerate(((.30, .04), (.55, .08))):
        bpy.ops.mesh.primitive_cylinder_add(vertices=48, radius=.0225, depth=.09, location=(x, y, 1.045))
        ob = bpy.context.object; ob.name = 'CableGuide' + str(i); ob.data.materials.append(cyan); fixture_objects.append(ob)
    cube('PickupPedestal', (.4, -.176, 1.04), (.006, .033, .08), cyan, fixture_objects)
    # Explicit Blender triangulation preserves boolean apertures: the generic
    # runtime bridge otherwise uses a triangle fan, unsuitable for concave n-gons.
    for ob in physical + neighbours + fixture_objects:
        if ob.data.users > 1: ob.data = ob.data.copy()
        bpy.context.view_layer.objects.active = ob
        mod = ob.modifiers.new('Export real aperture triangles', 'TRIANGULATE')
        bpy.ops.object.modifier_apply(modifier=mod.name)
    def join(objects, name):
        bpy.ops.object.select_all(action='DESELECT')
        for ob in objects: ob.select_set(True)
        bpy.context.view_layer.objects.active = objects[0]; bpy.ops.object.join()
        ob = bpy.context.object; ob.name = name; return ob
    panel = join(physical, 'PatchPanelPhysical')
    neighbour_mesh = join(neighbours, 'NeighbourSocketsPhysical')
    for ob in [panel, neighbour_mesh] + fixture_objects: ob['deformx_collision'] = True
    for ob in decorative: ob['deformx_collision'] = False
    bpy.ops.wm.save_as_mainfile(filepath=str(args.out / 'patch-panel.blend'))
    bpy.ops.wm.usd_export(filepath=str(args.out / 'fixtures.usdc'), selected_objects_only=False, export_custom_properties=True)
    report = dict(authoring='Blender ' + bpy.app.version_string, host=socket.gethostname(),
        source=dict(url='https://github.com/tomasf/keystone-patch-panel', commit='33b2e66203d57256e38e85a160be145cd03f3a86', sha256=sha(args.source), license='MIT', copyright='2024 Tomas Franzén', license_sha256=sha(args.license)),
        adaptation='Trim rear snap retainers to 2 mm front plate; native 24 apertures/ears retained; original rear carrier rings and noncollision numbers/screws added',
        script_sha256=sha(Path(__file__)), socket_source_sha256=sha(args.socket_usd),
        port_count=24, pitch_m=.018, active_port=7, panel_dimensions_m=[.4832, .0439, .002],
        panel_front_y_m=face_y, socket_mouth_recess_m=.0022,
        socket_position_m=[centers[6], socket_origin_y, socket_z], socket_vertical_datum_from_aperture_m=-.0025, front_plate_thickness_m=.0020003,
        physical_objects=['PatchPanelPhysical', 'NeighbourSocketsPhysical', 'CableGuide0', 'CableGuide1', 'PickupPedestal'],
        decorative_objects=[ob.name for ob in decorative],
        decorative_reason='Port numbers and screw heads away from aperture/contact sweep; no support/cavity surfaces excluded')
    (args.out / 'authoring_manifest.json').write_text(json.dumps(report, indent=2) + '\n')
    (args.out / 'LICENSE.panel-MIT.txt').write_bytes(args.license.read_bytes())
    (args.out / 'generator.py').write_bytes(Path(__file__).read_bytes())


if __name__ == '__main__':
    main()
