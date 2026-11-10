"""Author a render-only massless termination element in Blender on ps."""
import argparse
import hashlib
import json
from pathlib import Path
import socket
import sys
import bpy


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args(sys.argv[sys.argv.index('--')+1:])
    args.out.mkdir(parents=True,exist_ok=False)
    bpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)
    bpy.context.scene.unit_settings.system='METRIC';bpy.context.scene.unit_settings.scale_length=1.
    bpy.ops.mesh.primitive_cylinder_add(vertices=24,radius=.0014,depth=1.,location=(0,0,0))
    obj=bpy.context.object;obj.name='MasslessCompliantTermination'
    mat=bpy.data.materials.new('Assumed termination amber');mat.diffuse_color=(1.,.46,.025,1.)
    obj.data.materials.append(mat)
    bpy.ops.wm.save_as_mainfile(filepath=str(args.out/'termination.blend'))
    bpy.ops.wm.usd_export(filepath=str(args.out/'termination.usdc'),selected_objects_only=False)
    report=dict(host=socket.gethostname(),authoring='Blender '+bpy.app.version_string,
        nominal_length_m=1.,radius_m=.0014,axis='Z',kind='render-only massless constitutive element; no commercial hardware claim',
        animation='scaled between actual recorded plug exit and free cable endpoint; poses are never offset to hide compliance',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (args.out/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
