"""Prepare the pinned humanoid and guitar assets with Blender.

No robot dimensions are synthesized. Every visual transform, joint origin, axis,
and limit is read from the pinned original URDF. See README.md for provenance.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


URDF_PIN = 'unitree/robots/g1_description/g1_29dof_rev_1_0_with_inspire_hand_DFQ.urdf'
GUITAR_PIN = 'guitar/solid-body-electric-guitar.glb'


def verify_pinned(path, relative, pins):
    """Verify a known source path; never accept an arbitrary same-named mesh."""
    entry = pins.get(relative)
    if entry is None:
        raise ValueError('Source is absent from pinned asset manifest: '+relative)
    path = Path(path)
    if not path.is_file() or path.stat().st_size != entry['bytes'] or sha256(path) != entry['sha256']:
        raise ValueError('Pinned source checksum/size mismatch: '+str(path))


def require_outputs(out, inspect_only=False):
    names = ['guitar_inspect.json', 'guitar_original.blend']
    if not inspect_only:
        names += ['asset_profile.json', 'pick.obj']
        names += [name+'.usdc' for name in ('robot', 'guitar', 'pick', 'stool', 'floor')]
        names += [name+'.blend' for name in ('robot', 'guitar', 'pick', 'stool')]
    for name in names:
        path = out/name
        if path.is_symlink() or not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError('Blender did not create required fresh output: '+str(path))


def parse_urdf(path):
    import numpy as np
    tree = ET.parse(path).getroot()
    def numbers(value, fallback):
        return [float(x) for x in value.split()] if value else fallback
    def origin(element):
        out = np.eye(4)
        if element is None:
            return out.tolist()
        x, y, z = numbers(element.get('rpy'), [0, 0, 0])
        cx, cy, cz, sx, sy, sz = np.cos(x), np.cos(y), np.cos(z), np.sin(x), np.sin(y), np.sin(z)
        out[:3, :3] = [[cz*cy, cz*sy*sx-sz*cx, cz*sy*cx+sz*sx],
                        [sz*cy, sz*sy*sx+cz*cx, sz*sy*cx-cz*sx], [-sy, cy*sx, cy*cx]]
        out[:3, 3] = numbers(element.get('xyz'), [0, 0, 0])
        return out.tolist()
    materials = {x.get('name'): numbers(x.find('color').get('rgba'), [0.7]*3+[1])
                  for x in tree.findall('material') if x.find('color') is not None}
    links = {}
    for link in tree.findall('link'):
        visuals = []
        for index, v in enumerate(link.findall('visual')):
            mesh = v.find('geometry/mesh')
            if mesh is None:
                raise ValueError('Non-mesh URDF visual requires explicit Blender support: '+link.get('name'))
            name = mesh.get('filename')
            if name.startswith('package://'):
                name = name[len('package://'):]
                parts = Path(name).parts
                if parts[0] == path.parent.name:
                    name = str(Path(*parts[1:]))
            source = (path.parent / name).resolve()
            if not source.exists():
                raise FileNotFoundError(source)
            mat = v.find('material')
            color = None
            if mat is not None:
                color = numbers(mat.find('color').get('rgba'), [0.7]*3+[1]) if mat.find('color') is not None else materials.get(mat.get('name'))
            visuals.append(dict(index=index, path=str(source), transform=origin(v.find('origin')),
                                scale=numbers(mesh.get('scale'), [1, 1, 1]), rgba=color))
        links[link.get('name')] = visuals
    joints, children = [], set()
    for j in tree.findall('joint'):
        kind = j.get('type')
        if kind not in ('revolute', 'continuous', 'prismatic', 'fixed'):
            raise ValueError('Unsupported URDF joint '+kind)
        limit = j.find('limit')
        lower = float(limit.get('lower', '-3.141592653589793')) if limit is not None else -3.141592653589793
        upper = float(limit.get('upper', '3.141592653589793')) if limit is not None else 3.141592653589793
        child = j.find('child').get('link')
        data = dict(name=j.get('name'), parent=j.find('parent').get('link'), child=child,
                    kind='revolute' if kind == 'continuous' else kind,
                    parent_frame=origin(j.find('origin')), child_frame=np.eye(4).tolist(),
                    axis=numbers(j.find('axis').get('xyz'), [1, 0, 0]) if j.find('axis') is not None else [1, 0, 0],
                    lower=lower, upper=upper)
        mimic = j.find('mimic')
        if mimic is not None:
            data['mimic'] = dict(joint=mimic.get('joint'), multiplier=float(mimic.get('multiplier', 1)), offset=float(mimic.get('offset', 0)))
        joints.append(data)
        children.add(child)
    roots = {n: np.eye(4).tolist() for n in links if n not in children}
    controls = [j['name'] for j in joints if j['name'].startswith('right_') and j['kind'] != 'fixed' and any(t in j['name'] for t in ('shoulder', 'elbow', 'wrist'))]
    tcp_candidates = [n for n in links if n in ('right_wrist_yaw_link', 'right_hand_base_link', 'right_hand_palm_link')]
    tcp = 'right_wrist_yaw_link' if 'right_wrist_yaw_link' in links else tcp_candidates[-1]
    model = dict(roots=roots, joints=joints, control_joints=controls, preset={}, placement=np.eye(4).tolist(),
                 tcp_link=tcp, tcp_transform=np.eye(4).tolist(), seed=[0]*len(controls))
    return dict(links=links, kinematics=model)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--urdf', type=Path, required=True)
    p.add_argument('--guitar', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--blender', type=Path, required=True)
    p.add_argument('--inspect-only', action='store_true')
    a = p.parse_args()
    # Refuse stale profiles before any import or Blender launch. A failed run
    # must be inspected/removed explicitly or retried in a new directory.
    a.out = a.out.resolve()
    if a.out.exists() and (not a.out.is_dir() or any(a.out.iterdir())):
        p.error('Output directory must be absent or empty: '+str(a.out))
    pins = {x['path']: x for x in json.loads(Path(__file__).with_name('asset_sources.json').read_text())['files']}
    a.urdf, a.guitar = a.urdf.resolve(), a.guitar.resolve()
    if a.urdf.name != Path(URDF_PIN).name or a.guitar.name != Path(GUITAR_PIN).name:
        p.error('Expected the pinned Unitree URDF and original guitar filenames')
    verify_pinned(a.urdf, URDF_PIN, pins)
    verify_pinned(a.guitar, GUITAR_PIN, pins)
    source = parse_urdf(a.urdf)
    sources = sorted({a.urdf, a.guitar} | {Path(v['path']) for values in source['links'].values() for v in values})
    for mesh in sources:
        if mesh in (a.urdf, a.guitar):
            continue
        # Accept both the fetch directory layout and a relocated complete
        # g1_description package. Mesh hierarchy must match the pinned package.
        try:
            relative = mesh.relative_to(a.urdf.parent)
        except ValueError as exc:
            raise ValueError('Mesh escapes the verified URDF package: '+str(mesh)) from exc
        verify_pinned(mesh, (Path(URDF_PIN).parent/relative).as_posix(), pins)
    a.out.mkdir(parents=True, exist_ok=True)
    source['urdf'] = str(a.urdf.resolve())
    source['guitar_glb'] = str(a.guitar.resolve())
    source['output'] = str(a.out.resolve())
    path = a.out / 'import_spec.json'
    path.write_text(json.dumps(source, indent=2)+'\n')
    cmd = [str(a.blender), '-b', '--factory-startup', '--python-exit-code', '1', '--python', str(Path(__file__).with_name('blender_assets.py')), '--', '--spec', str(path.resolve())]
    if a.inspect_only:
        cmd.append('--inspect-only')
    subprocess.run(cmd, check=True)
    require_outputs(a.out, a.inspect_only)
    if a.inspect_only:
        return
    profile = json.loads((a.out/'asset_profile.json').read_text())
    for key in ('robot', 'guitar', 'pick', 'stool', 'floor'):
        if Path(profile[key]['path']).resolve() != a.out/(key+'.usdc'):
            raise RuntimeError('Blender profile references an unexpected output: '+key)
    if Path(profile['pick']['mesh_path']).resolve() != a.out/'pick.obj':
        raise RuntimeError('Blender profile references an unexpected pick collision mesh')
    profile['sources'] = [dict(path=str(x), sha256=sha256(x), size_bytes=x.stat().st_size) for x in sources]
    for key in ('robot', 'guitar', 'pick', 'stool', 'floor'):
        item = profile[key]
        path = Path(item['path'])
        # Blender can export the inherited world HDR with every USD. Include
        # all exported images conservatively, rather than silently omitting a
        # transitive texture when this script is run without the USD Python API.
        item['files'] = [dict(path=str(f), sha256=sha256(f))
                         for f in [path] + sorted((a.out/'textures').rglob('*')) if f.is_file()]
        if key == 'pick':
            mesh = Path(item['mesh_path'])
            item['collision_mesh_sha256'] = sha256(mesh)
            item['files'].append(dict(path=str(mesh), sha256=sha256(mesh)))
    (a.out/'asset_profile.json').write_text(json.dumps(profile, indent=2, allow_nan=False)+'\n')
    print(a.out/'asset_profile.json')


if __name__ == '__main__':
    main()
