"""Original visual cell and roller, authored with Blender. Units: metres."""
import argparse
import json
import hashlib
from pathlib import Path
import sys
import bpy

p=argparse.ArgumentParser(); p.add_argument('--out',type=Path,required=True)
a=p.parse_args(sys.argv[sys.argv.index('--')+1:])
if a.out.exists(): p.error('output already exists')
a.out.mkdir(parents=True)
bpy.ops.object.select_all(action='SELECT'); bpy.ops.object.delete(use_global=False)

def material(name,color,metal=0.):
    m=bpy.data.materials.new(name); m.diffuse_color=(*color,1); m.use_nodes=True
    bs=m.node_tree.nodes.get('Principled BSDF'); bs.inputs['Base Color'].default_value=(*color,1)
    bs.inputs['Metallic'].default_value=metal; bs.inputs['Roughness'].default_value=.3
    return m

dark=material('Graphite',(.025,.04,.065),.55)
metal=material('Satin steel',(.28,.36,.43),.8)
rubber=material('Guide tread',(.075,.09,.11),.2)

def box(name,pos,size,mat):
    bpy.ops.mesh.primitive_cube_add(size=1,location=pos)
    o=bpy.context.object; o.name=name; o.dimensions=size
    bpy.ops.object.transform_apply(location=False,rotation=False,scale=True)
    bevel=o.modifiers.new('Edge','BEVEL'); bevel.width=.015; bevel.segments=3
    bpy.ops.object.modifier_apply(modifier=bevel.name); o.data.materials.append(mat)

box('Cell',(0,0,-.04),(1.35,.65,.065),dark)
for x in (-.6,.6): box('Grip pedestal',(x,0,.30),(.045,.1,.6),metal)
bpy.ops.wm.usd_export(filepath=str(a.out/'cell.usda'),export_materials=True)
bpy.ops.object.select_all(action='SELECT'); bpy.ops.object.delete(use_global=False)
bpy.ops.mesh.primitive_cylinder_add(vertices=96,radius=.18,depth=.32,rotation=(1.5707963267948966,0,0))
o=bpy.context.object; o.name='Guide'; o.data.materials.append(rubber)
bpy.ops.object.transform_apply(location=False,rotation=True,scale=True)
for f in o.data.polygons: f.use_smooth=True
for y in (-.165,.165):
    bpy.ops.mesh.primitive_cylinder_add(vertices=64,radius=.125,depth=.012,location=(0,y,0),rotation=(1.5707963267948966,0,0))
    bpy.context.object.data.materials.append(metal)
bpy.ops.wm.usd_export(filepath=str(a.out/'guide.usda'),export_materials=True)
bpy.ops.wm.save_as_mainfile(filepath=str(a.out/'guide.blend'))
(a.out/'manifest.json').write_text(json.dumps(dict(author='DeformX',source='Original Blender primitives',units='metres',
    collision='Analytic infinite Y-axis cylinder, radius 0.18 m; cable constrained to y=0. Visual caps are decoration.',
    sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in a.out.iterdir() if p.is_file()}),indent=2)+'\n')
