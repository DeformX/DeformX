"""Blender CPU audit of unloaded full-span string/visible-collider clearances."""
import argparse,sys,bpy,json,numpy as np
from pathlib import Path
from mathutils import Vector
from mathutils.bvhtree import BVHTree
parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--profile',type=Path,required=True)
args=parser.parse_args(sys.argv[sys.argv.index('--')+1:]);p=args.profile.resolve().parent
a=json.load(open(args.profile));radius=.00055;reports=[]
for item in a['guitar']['contact_meshes']:
 n=np.load(item['mesh_path']);v=n['vertices'];f=n['faces'];tree=BVHTree.FromPolygons([Vector(x) for x in v],f,all_triangles=True)
 minimum=1;crossings=[]
 for string,(b,nut) in enumerate(zip(a['guitar']['bridge_endpoints_local_m'],a['guitar']['nut_endpoints_local_m'])):
  b=np.array(b);nut=np.array(nut);delta=nut-b;length=np.linalg.norm(delta)
  hit=tree.ray_cast(Vector(b+delta*1e-7),Vector(delta/length),length*(1-2e-7))
  if hit[0] is not None:crossings.append(dict(string=string+1,point=list(hit[0])))
  for t in np.linspace(0,1,1201):
   point=b+t*delta;nearest=tree.find_nearest(Vector(point))
   if nearest[0] is not None:minimum=min(minimum,nearest[3]-radius)
 reports.append(dict(id=item['id'],minimum_surface_clearance_m=minimum,centerline_triangle_crossings=crossings))
print(json.dumps(reports,indent=2));(p/'static_clearance_validation.json').write_text(json.dumps(dict(radius_m=radius,samples_per_string=1201,unloaded_straight_endpoints=True,groups=reports),indent=2)+'\n')
