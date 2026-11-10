"""Author the generic molded cable pull tab in Blender 4.5."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import bpy

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import COLORS, EXIT, PULL_TAB_SIZE


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args(sys.argv[sys.argv.index('--') + 1:])
    args.out.mkdir(parents=True)
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    bpy.context.scene.unit_settings.system = 'METRIC'
    bpy.ops.mesh.primitive_cube_add(size=1.)
    tab = bpy.context.object
    tab.name = 'MoldedPullTab'
    tab.dimensions = PULL_TAB_SIZE
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    rounded = tab.modifiers.new('Molded corners', 'BEVEL')
    rounded.width, rounded.segments = .002, 4
    bpy.ops.object.modifier_apply(modifier=rounded.name)
    material = bpy.data.materials.new('Blue molded rubber')
    material.diffuse_color = (*COLORS[0], 1.)
    tab.data.materials.append(material)
    bpy.ops.wm.save_as_mainfile(filepath=str(args.out/'pull-tab.blend'))
    bpy.ops.wm.usd_export(filepath=str(args.out/'pull-tab.usdc'),
                         export_materials=True, export_textures=False)
    report = dict(blender=bpy.app.version_string, dimensions_m=PULL_TAB_SIZE, exit_offset_m=EXIT,
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  usd_sha256=hashlib.sha256((args.out/'pull-tab.usdc').read_bytes()).hexdigest())
    (args.out/'manifest.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
