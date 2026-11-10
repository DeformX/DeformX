"""Bind consumed .blend mesh geometry to verified USD per guitar object/robot link."""
import re
from pathlib import Path
import bpy
import numpy as np
from mathutils import Matrix, Vector
from mathutils.kdtree import KDTree


def clean(name):return re.sub(r'[^a-zA-Z0-9_]','_',name)


def snapshot(kind,links,imported=False):
    groups={}
    for obj in bpy.data.objects:
        if obj.type!='MESH':continue
        if kind=='robot':
            parent=obj
            while parent and parent.name not in links:parent=parent.parent
            if parent is None:raise ValueError('Robot visual has no named URDF link: '+obj.name)
            key=parent.name;mat=parent.matrix_world.inverted()@obj.matrix_world
        else:
            name=obj.parent.name if imported and obj.parent else obj.name
            # USD identifiers contain no dots. Blender adds .001 when an
            # imported Mesh and its Xform share a prim basename; strip only
            # that importer collision suffix, not source _001 identifiers.
            if imported:name=re.sub(r'\.\d{3}$','',name)
            key=clean(name);root=bpy.data.objects.get('Guitar')
            mat=(root.matrix_world.inverted() if root else Matrix.Identity(4))@obj.matrix_world
        obj.data.calc_loop_triangles()
        v=np.array([list(mat@point.co) for point in obj.data.vertices],float)
        tri=np.array([list(t.vertices) for t in obj.data.loop_triangles],int)
        points,centers=groups.setdefault(key,([],[]));points.extend(v);centers.extend(v[tri].mean(1))
    return groups


def nearest_error(a,b):
    tree=KDTree(len(b))
    for i,p in enumerate(b):tree.insert(Vector(p),i)
    tree.balance()
    return max((tree.find(Vector(p))[2] for p in a),default=0.)


def verify_baseline(prepared,profile):
    reports=[];links=set(profile['robot']['kinematics']['roots'])|{j['child'] for j in profile['robot']['kinematics']['joints']}
    for kind in ('guitar','robot'):
        bpy.ops.wm.open_mainfile(filepath=str(prepared/(kind+'.blend')))
        if kind == 'guitar':
            root = bpy.data.objects.get('Guitar')
            if root is None or root.matrix_world != Matrix.Identity(4):
                raise ValueError('Baseline Guitar root must have an identity world transform')
        before=snapshot(kind,links)
        bpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)
        bpy.ops.wm.usd_import(filepath=profile[kind]['path'],import_materials=False,import_cameras=False,import_lights=False,merge_parent_xform=False)
        after=snapshot(kind,links,imported=True)
        if set(before)!=set(after):raise ValueError('Baseline .blend/USD mesh assignments differ '+kind+': '+str(set(before)^set(after)))
        for name,(bv,bt) in before.items():
            av,at=after[name]
            if len(bt)!=len(at):raise ValueError('Baseline .blend/USD triangle count differs '+kind+'/'+name)
            vertex_error=max(nearest_error(bv,av),nearest_error(av,bv));triangle_error=max(nearest_error(bt,at),nearest_error(at,bt))
            if max(vertex_error,triangle_error)>1e-6:raise ValueError('Baseline .blend/USD geometry differs '+kind+'/'+name+': '+str((vertex_error,triangle_error)))
            reports.append(dict(asset=kind,object_or_link=name,triangles=len(bt),maximum_vertex_difference_m=vertex_error,maximum_triangle_centroid_difference_m=triangle_error))
    return dict(tolerance_m=1e-6,robot_frame='URDF link-local, grouped by actual parent link',guitar_frame='Guitar-local, grouped by named object',comparisons=reports)
