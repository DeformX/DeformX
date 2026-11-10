"""Prepare fresh contact assets from verified Blender and USD sources."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


ASSETS = ('robot', 'guitar', 'pick', 'stool', 'floor')


def verify_files(profile):
    dependencies = {}
    records = list(profile['sources'])
    for kind in ASSETS:
        asset = profile[kind]
        files = {entry['path']: entry['sha256'] for entry in asset['files']}
        if asset['path'] not in files:
            raise ValueError('Baseline USD path is absent from its hash manifest: '+asset['path'])
        records.extend(asset['files'])
    for entry in records:
        if dependencies.setdefault(entry['path'], entry['sha256']) != entry['sha256']:
            raise ValueError('Conflicting checksums for source asset: '+entry['path'])
    for path, digest in dependencies.items():
        if sha(path) != digest:
            raise ValueError('Baseline hash mismatch: '+path)
    return dependencies


def verify_frozen_contact(path, baseline):
    profile = json.loads(path.read_text())
    for kind in ('robot', 'pick'):
        if profile[kind]['path'] != baseline[kind]['path']:
            raise ValueError('Frozen contact profile does not use the verified original '+kind)
    for key in ('link_prim_paths', 'kinematics'):
        if profile['robot'][key] != baseline['robot'][key]:
            raise ValueError('Frozen contact profile robot frame changed: '+key)
    if profile['sources'] != baseline['sources'] or profile['pick']['mesh_path'] != baseline['pick']['mesh_path']:
        raise ValueError('Frozen contact profile source assets differ')
    files = verify_files(profile)
    for mesh in profile['guitar']['contact_meshes'] + profile['robot']['contact_meshes']:
        for key, checksum in (('mesh_path', 'sha256'), ('obj_path', 'obj_sha256')):
            path_to_mesh = mesh[key]
            if sha(path_to_mesh) != mesh[checksum]:
                raise ValueError('Frozen contact mesh checksum mismatch: '+path_to_mesh)
            files[path_to_mesh] = mesh[checksum]
    files[str(path)] = sha(path)
    return files


def verify_wide_pick_base(path, original):
    profile = json.loads(path.read_text())
    if (profile.get('contact_schema_version') != 2 or 'contact_motion' in profile
            or profile['pick'].get('tip_extension_m') != .010 or profile['sources'] != original['sources']):
        raise ValueError('Wide pick preparation requires the original verified full-hand v3 asset profile')
    baseline = profile['contact_asset_provenance']['full_hand_baseline_assets']
    if any(baseline[kind] != original[kind]['path'] for kind in ('robot', 'pick')):
        raise ValueError('Wide pick v3 profile uses a different original asset source')
    files = verify_files(profile)
    hand = profile['robot']
    entries = profile['guitar']['contact_meshes'] + hand['contact_meshes'] + hand['arm_clearance_meshes']
    for entry in entries:
        files[entry['mesh_path']] = entry['sha256']
        files[entry['obj_path']] = entry['obj_sha256']
    coverage = hand['hand_coverage']
    files[coverage['path']] = coverage['sha256']
    pick = profile['pick']
    files[pick['collision_mesh_npz']] = pick['collision_npz_sha256']
    files[str(path)] = sha(path)
    for filename, digest in files.items():
        if sha(filename) != digest:
            raise ValueError('Wide pick input checksum mismatch: '+filename)
    return files


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepared',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--blender',type=Path,required=True)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument('--audit-only',action='store_true')
    mode.add_argument('--extend-hands-from',type=Path,help='Verified contact-v2 profile for full hand and tapered pick preparation')
    mode.add_argument('--wide-pick-from',type=Path,help='Verified full-hand v3 profile for the 24 mm wide-picking plectrum')
    a=p.parse_args();a.prepared=a.prepared.resolve();a.out=a.out.resolve()
    if a.out.exists() and (not a.out.is_dir() or any(a.out.iterdir())):p.error('Output must be absent or empty')
    profile=json.loads((a.prepared/'asset_profile.json').read_text())
    verify_files(profile)
    pins={x['sha256'] for x in json.loads(Path(__file__).with_name('asset_sources.json').read_text())['files']}
    if any(x['sha256'] not in pins for x in profile['sources']):raise ValueError('Unpinned source in baseline')
    before={str(x):sha(x) for x in a.prepared.rglob('*') if x.is_file()}
    frozen = {}
    if a.extend_hands_from:
        a.extend_hands_from = a.extend_hands_from.resolve()
        if a.out.is_relative_to(a.prepared) or a.out.is_relative_to(a.extend_hands_from.parent):
            p.error('Full hand output must be separate from both input directories')
        frozen = verify_frozen_contact(a.extend_hands_from, profile)
        original_assets = {kind: profile[kind]['path'] for kind in ('robot', 'pick')}
    if a.wide_pick_from:
        a.wide_pick_from = a.wide_pick_from.resolve()
        if a.out.is_relative_to(a.prepared) or a.out.is_relative_to(a.wide_pick_from.parent):
            p.error('Wide pick output must be separate from both input directories')
        frozen = verify_wide_pick_base(a.wide_pick_from, profile)
    a.out.mkdir(parents=True,exist_ok=True)
    cmd=[str(a.blender),'-b','--factory-startup','--python-exit-code','1','--python',str(Path(__file__).with_name('blender_contact_assets.py')),'--','--prepared',str(a.prepared),'--out',str(a.out)]
    if a.audit_only:cmd.append('--audit-only')
    if a.extend_hands_from:cmd.extend(['--extend-hands-from',str(a.extend_hands_from)])
    if a.wide_pick_from:
        cmd = [str(a.blender), '-b', '--factory-startup', '--python-exit-code', '1', '--python',
               str(Path(__file__).with_name('blender_contact_pick.py')), '--', '--prepared', str(a.prepared),
               '--profile', str(a.wide_pick_from), '--out', str(a.out)]
    subprocess.run(cmd,check=True)
    if before!={str(x):sha(x) for x in a.prepared.rglob('*') if x.is_file()}:raise RuntimeError('Baseline changed during preparation')
    if any(sha(path) != digest for path,digest in frozen.items()):raise RuntimeError('Frozen contact inputs changed during preparation')
    if a.audit_only:return
    dest=a.out/'contact_profile.json';profile=json.loads(dest.read_text())
    if a.wide_pick_from:
        paths = [a.out/name for name in ('pick.usdc', 'pick.obj', 'pick.npz', 'pick.blend', 'plectrum_geometry.json')]
        if any(path.is_symlink() or not path.is_file() or path.stat().st_size == 0 for path in paths):
            raise RuntimeError('Missing fresh wide pick output')
        source = json.loads(a.wide_pick_from.read_text())
        if any(profile[kind] != source[kind] for kind in ('robot', 'guitar', 'stool', 'floor')):
            raise RuntimeError('Wide pick preparation changed another asset binding')
        textures = [path for path in sorted((a.out/'textures').rglob('*')) if path.is_file()]
        profile['pick']['files'] = [dict(path=str(path), sha256=sha(path)) for path in paths[:2]+textures]
        proof = json.loads((a.out/'plectrum_geometry.json').read_text())
        proof.update(verified_v3_profile=str(a.wide_pick_from), verified_v3_profile_sha256=frozen[str(a.wide_pick_from)],
                     all_other_asset_bindings_identical=True, input_files_unchanged=len(frozen))
        profile['contact_asset_provenance']['wide_pick'] = proof
        (a.out/'wide_pick_provenance.json').write_text(json.dumps(proof, indent=2)+'\n')
        dest.write_text(json.dumps(profile, indent=2)+'\n')
        print(dest)
        return
    if a.extend_hands_from:
        required = ['robot.usdc', 'robot.blend', 'pick.usdc', 'pick.blend', 'pick.obj', 'pick.npz',
                    'baseline_geometry_binding.json', 'hand_coverage.json', 'plectrum_geometry.json']
        for entry in profile['robot']['contact_meshes'] + profile['robot']['arm_clearance_meshes']:
            required.extend(str(Path(entry[key]).relative_to(a.out)) for key in ('mesh_path', 'obj_path'))
        for name in required:
            output = a.out/name
            if output.is_symlink() or not output.is_file() or output.stat().st_size == 0:
                raise RuntimeError('Missing fresh output: '+str(output))
        textures = [path for path in sorted((a.out/'textures').rglob('*')) if path.is_file()]
        for kind in ('robot', 'pick'):
            paths = [a.out/(kind+'.usdc')] + textures
            if kind == 'pick':
                paths.append(a.out/'pick.obj')
            profile[kind]['files'] = [dict(path=str(path),sha256=sha(path)) for path in paths]
        profile['contact_asset_provenance'].update(
            full_hand_baseline_assets=original_assets, full_hand_baseline_sha256=before,
            frozen_contact_input_sha256=frozen, frozen_contact_inputs_unchanged=True)
        coverage = a.out/'hand_coverage.json'
        profile['robot']['hand_coverage'] = dict(path=str(coverage), sha256=sha(coverage))
        dest.write_text(json.dumps(profile,indent=2)+'\n')
        print(dest)
        return
    required=['guitar.usdc','guitar.blend','geometry_audit.json','contact_profile.json',
              'baseline_geometry_binding.json','material_validation.json']
    for item in profile['guitar']['contact_meshes']+profile['robot']['contact_meshes']:
        required.extend([Path(item['mesh_path']).name,Path(item['obj_path']).name])
    for name in required:
        f=a.out/name
        if f.is_symlink() or not f.is_file() or f.stat().st_size==0:raise RuntimeError('Missing fresh output: '+str(f))
    profile['guitar']['files']=[dict(path=str(f),sha256=sha(f)) for f in [a.out/'guitar.usdc']+sorted((a.out/'textures').rglob('*')) if f.is_file()]
    profile['contact_asset_provenance']=dict(baseline_files_sha256=before,baseline_unchanged=True,
       correction='Both pickup covers/poles lowered 5mm; original neck/action/endpoints retained; visible exact Boolean nut/saddle slots',
       original_source_manifest=str(Path(__file__).with_name('asset_sources.json').resolve()))
    dest.write_text(json.dumps(profile,indent=2)+'\n');print(dest)


if __name__=='__main__':main()
