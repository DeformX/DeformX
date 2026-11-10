"""Prepare exact visible guitar and finger contact meshes in Blender, on ps.

Original prepared assets are read-only inputs. Contact-v2 outputs use a fresh
folder, corrected visible source geometry, and identical collision triangles.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path
import bpy
import bmesh
import numpy as np
from mathutils import Matrix, Vector


def mesh_data(obj, matrix=None):
    mesh = obj.data.copy()
    bm = bmesh.new(); bm.from_mesh(mesh)
    bmesh.ops.remove_doubles(bm, verts=list(bm.verts), dist=1e-7)
    bmesh.ops.recalc_face_normals(bm, faces=list(bm.faces))
    bmesh.ops.triangulate(bm, faces=list(bm.faces))
    bm.to_mesh(mesh); bm.free(); mesh.update()
    mat = obj.matrix_world if matrix is None else matrix
    vertices = [list(mat@v.co) for v in mesh.vertices]
    triangles = [list(p.vertices) for p in mesh.polygons]
    bm = bmesh.new(); bm.from_mesh(mesh)
    bad = sum(not e.is_manifold for e in bm.edges)
    boundary = sum(e.is_boundary for e in bm.edges)
    volume = bm.calc_volume(signed=True)
    bm.free(); bpy.data.meshes.remove(mesh)
    return vertices, triangles, dict(vertices=len(vertices), triangles=len(triangles), nonmanifold_edges=bad,
                                     boundary_edges=boundary, volume_mesh_m3=volume, watertight=bad==0)


def bounds(vertices):
    return [[float(min(p[k] for p in vertices)) for k in range(3)], [float(max(p[k] for p in vertices)) for k in range(3)]]


def audit(out):
    objects=[]
    for obj in bpy.data.objects:
        if obj.type!='MESH': continue
        verts,tris,topology=mesh_data(obj)
        objects.append(dict(name=obj.name,bounds=bounds(verts),topology=topology,
                            materials=[m.name if m else None for m in obj.data.materials]))
    (out/'geometry_audit.json').write_text(json.dumps(objects,indent=2)+'\n')
    for obj in objects:
        if any(t in obj['name'].lower() for t in ('fret','nut','bridge','pickup','body','neck')):
            print(json.dumps(obj))


def weld(obj):
    bm=bmesh.new();bm.from_mesh(obj.data)
    bmesh.ops.remove_doubles(bm,verts=list(bm.verts),dist=1e-7)
    bmesh.ops.recalc_face_normals(bm,faces=list(bm.faces));bm.to_mesh(obj.data);bm.free()


def triangulate_visible(obj):
    bm=bmesh.new();bm.from_mesh(obj.data)
    bmesh.ops.triangulate(bm,faces=list(bm.faces));bm.to_mesh(obj.data);bm.free()


def bake_world(obj):
    mat=obj.matrix_world.copy();obj.parent=None;obj.data.transform(mat);obj.matrix_world=Matrix.Identity(4);weld(obj)


def boolean(obj,other,operation):
    bpy.context.view_layer.objects.active=obj
    mod=obj.modifiers.new('Exact_visible_contact_'+operation,'BOOLEAN');mod.operation=operation;mod.solver='EXACT';mod.object=other
    if operation=='UNION':mod.material_mode='TRANSFER'
    bpy.ops.object.modifier_apply(modifier=mod.name)
    bpy.data.objects.remove(other,do_unlink=True)


def union(names,label):
    objects=[bpy.data.objects[n] for n in names]
    for obj in objects:bake_world(obj)
    obj=objects[0]
    for other in objects[1:]:boolean(obj,other,'UNION')
    obj.name=label;weld(obj)
    obj['contact_source_objects']=json.dumps(names)
    return obj


def grooves(obj,endpoints,label):
    bake_world(obj)
    bb=bounds([list(v.co) for v in obj.data.vertices]);yc=(bb[0][1]+bb[1][1])/2
    radius=.0008
    for point in endpoints:
        x,y,z=point
        bpy.ops.mesh.primitive_cylinder_add(vertices=48,radius=radius,depth=bb[1][1]-bb[0][1]+.008,
                                          rotation=(np.pi/2,0,0),location=(x,yc,z))
        boolean(obj,bpy.context.object,'DIFFERENCE')
        height=bb[1][2]+.004-z
        bpy.ops.mesh.primitive_cube_add(size=1,location=(x,yc,z+height/2))
        cutter=bpy.context.object;cutter.dimensions=(2*radius,bb[1][1]-bb[0][1]+.008,height)
        bpy.ops.object.transform_apply(location=False,rotation=False,scale=True)
        boolean(obj,cutter,'DIFFERENCE')
    weld(obj)
    return dict(component=label,groove_radius_m=radius,width_m=2*radius,
                centerline_endpoints_local_m=endpoints,method='Visible exact Boolean open U slots, round-bottomed; identical collision boundary')


def export_contact(objects,label,out,frame,spacing,margin):
    vertices=[];faces=[];parts=[]
    for obj in objects:
        verts,tris,topology=mesh_data(obj,Matrix.Identity(4) if frame=='link_local' else None)
        if not topology['watertight']:raise ValueError('Open contact mesh '+obj.name+': '+str(topology))
        offset=len(vertices);vertices.extend(verts);faces.extend([[i+offset for i in tri] for tri in tris]);parts.append(dict(name=obj.name,topology=topology))
    return write_contact_arrays(vertices,faces,parts,label,out,frame,spacing,margin,
                                [name for o in objects for name in json.loads(o.get('contact_source_objects',json.dumps([o.name])))],
                                ['/root/Guitar/'+o.name.replace(' ','_').replace('.','_') for o in objects])


def write_contact_arrays(vertices,faces,parts,label,out,frame,spacing,margin,source_names,visual_paths):
    vertices=np.asarray(vertices,np.float32);faces=np.asarray(faces,np.int32)
    path=out/(label+'.npz');np.savez_compressed(path,vertices=vertices,faces=faces,triangles=faces)
    objpath=out/(label+'.obj')
    with objpath.open('w') as f:
        for v in vertices:f.write('v '+' '.join(format(float(x),'.9g') for x in v)+'\n')
        for tri in faces:f.write('f '+' '.join(str(int(x)+1) for x in tri)+'\n')
    return dict(id=label,mesh_path=str(path),collision_mesh_npz=str(path),obj_path=str(objpath),frame=frame,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),obj_sha256=hashlib.sha256(objpath.read_bytes()).hexdigest(),
                bounds=bounds(vertices),watertight=all(part['topology']['watertight'] for part in parts),parts=parts,
                spacing_m=spacing,margin_m=margin,friction=.25,source_object_names=source_names,visual_prim_paths=visual_paths)


def build(a):
    out=a.out;profile=json.loads((a.prepared/'asset_profile.json').read_text());g=profile['guitar']
    corrections=[]
    moved=[o for o in bpy.data.objects if o.name.startswith('Pickup cover') or o.name.startswith('Pickup pole piece')]
    for obj in moved:obj.matrix_world.translation.z-=.005
    corrections.append(dict(objects=[o.name for o in moved],translation_local_m=[0,0,-.005],
                            reason='Measured source strings intersected pole/cover tops; lower actual pickup assembly, preserve flush mounting rings and original string action'))
    slots=[grooves(bpy.data.objects['Nut'],g['nut_endpoints_local_m'],'Nut'),
           grooves(bpy.data.objects['Bridge saddle'],g['bridge_endpoints_local_m'],'Bridge saddle')]
    necknames=['Tapered maple neck','Individually fretted fingerboard','Seated neck heel','Seated fingerboard extension bed']
    neck=union(necknames,'Contact_neck_fingerboard')
    bridge_names=['Bridge base','Bridge saddle','Seated string anchor bar']+[f'Bridge string anchor {i:02}' for i in range(1,7)]
    bridge=union(bridge_names,'Contact_bridge')
    pickups=[]
    for index in range(2):
        suffix='' if index==0 else '.001'
        names=['Pickup cover'+suffix,'Pickup mounting ring'+suffix]
        names += ['Pickup pole piece'+('' if i==0 else f'.{i:03}') for i in range(index*6,(index+1)*6)]
        pickups.append(union(names,'Contact_pickup_'+str(index)))
    groups=[([bpy.data.objects['Original carved solid instrument body']],'guitar_body',.0015,.008),
            ([neck],'neck_fingerboard',.0007,.004),([bridge],'bridge',.00025,.003),
            ([bpy.data.objects['Nut']],'nut',.0002,.003)]
    groups += [([o],'pickup_'+str(i),.00035,.004) for i,o in enumerate(pickups)]
    frets=sorted([o for o in bpy.data.objects if o.name.startswith('Fret ')],key=lambda o:o.name)
    groups.append(([o for o in frets if o.name=='Fret 05'],'fret_05',.0002,.003))
    groups.append(([o for o in frets if o.name!='Fret 05'],'other_frets',.00025,.003))
    # Bake and triangulate the visible contact objects themselves, so USD and
    # collision exports share triangle positions rather than different ngon
    # triangulations of an otherwise similar surface.
    for objects,_,_,_ in groups:
        for obj in objects:bake_world(obj);triangulate_visible(obj)
    metadata=[]
    for obj in frets:
        verts,_,_=mesh_data(obj);bb=bounds(verts);y=(bb[0][1]+bb[1][1])/2
        metadata.append(dict(number=int(obj.name.split()[-1]),name=obj.name,y_local_m=y,crown_z_local_m=bb[1][2],bounds=bb))
    contacts=[export_contact(objects,label,out,'guitar_local',spacing,margin) for objects,label,spacing,margin in groups]
    root=bpy.data.objects['Guitar']
    for obj in bpy.data.objects:
        if obj.type=='MESH' and obj.parent is None:
            world=obj.matrix_world.copy();obj.parent=root;obj.matrix_world=world
    material_counts={obj.name:[dict(material=mat.name if mat else None,triangles=sum(p.material_index==i for p in obj.data.polygons))
                              for i,mat in enumerate(obj.data.materials)] for obj in [neck]+pickups}
    (out/'material_validation.json').write_text(json.dumps(material_counts,indent=2)+'\n')
    # All collision triangle boundaries are the same evaluated visible meshes.
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.wm.usd_export(filepath=str(out/'guitar.usdc'),selected_objects_only=True,export_materials=True,
                         generate_preview_surface=True,export_textures=True,relative_paths=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(out/'guitar.blend'))
    g.update(path=str(out/'guitar.usdc'),default_prim='/root',contact_meshes=contacts,frets=metadata,
             corrections=corrections,anchor_slots=slots,collision_scope='Body, complete neck/fingerboard, all22 frets, both pickups including mounting rings, nut, bridge/anchors. Headstock tuning mechanisms outside playable span are not colliders.')
    profile['contact_schema_version']=1
    # Export original robot finger meshes strictly in their URDF link frame.
    bpy.ops.wm.open_mainfile(filepath=str(a.prepared/'robot.blend'))
    fingers=[]
    for link in ('L_index_intermediate','L_index_proximal'):
        objects=[o for o in bpy.data.objects if o.type=='MESH' and o.parent and o.parent.name==link]
        for obj in objects:obj.data.transform(obj.matrix_local);obj.parent=None;obj.matrix_world=Matrix.Identity(4)
        item=export_contact(objects,link,out,'link_local',.00035,.004)
        item.update(link=link,link_from_mesh=[list(r) for r in Matrix.Identity(4)],
                    visual_prim_paths=[profile['robot']['link_prim_paths'][link]+'/'+o.name.replace('.','_') for o in objects])
        if link=='L_index_intermediate':item.update(pad_point_link_local_m=[-.0010578642,-.0335902013,-.0105199264],pad_outward_normal_link_local=[.0387258828,-.0467939526,-.9981536269])
        fingers.append(item)
    profile['robot']['contact_meshes']=fingers
    (out/'contact_profile.json').write_text(json.dumps(profile,indent=2)+'\n')
    print('CONTACT_PROFILE',out/'contact_profile.json')


def main():
    p=argparse.ArgumentParser();p.add_argument('--prepared',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--audit-only',action='store_true')
    p.add_argument('--extend-hands-from',type=Path)
    a=p.parse_args(sys.argv[sys.argv.index('--')+1:]);a.out=a.out.resolve()
    if a.out.exists() and any(a.out.iterdir()): raise ValueError('Expected fresh empty contact output directory')
    a.out.mkdir(parents=True,exist_ok=True)
    sys.path.insert(0,str(Path(__file__).parent))
    from blender_contact_binding import verify_baseline
    profile=json.loads((a.prepared/'asset_profile.json').read_text())
    binding=verify_baseline(a.prepared,profile)
    (a.out/'baseline_geometry_binding.json').write_text(json.dumps(binding,indent=2)+'\n')
    if a.extend_hands_from:
        from blender_contact_hands import build_hand_profile
        build_hand_profile(a)
        return
    bpy.ops.wm.open_mainfile(filepath=str(a.prepared/'guitar.blend'))
    audit(a.out)
    if not a.audit_only:build(a)


if __name__=='__main__': main()
