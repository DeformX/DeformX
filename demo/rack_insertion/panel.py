"""Rack service overlay and explicit physical fixture selection."""
import json
import math
from pathlib import Path, PurePosixPath


from .assets import sha


def snapshot_usd_dependencies(paths):
    """Resolve all local USD composition layers, including nested payloads."""
    from pxr import UsdUtils
    resolved = {}
    for path in paths:
        layers, _, unresolved = UsdUtils.ComputeAllDependencies(str(Path(path).resolve()))
        missing = [x for x in unresolved if Path(x).suffix.lower() in ('.usd','.usda','.usdc','.usdz')]
        if missing: raise ValueError('Unresolved USD composition dependency: '+str(missing))
        for layer in layers:
            file=Path(layer.realPath or layer.identifier).resolve()
            if not file.is_file(): raise ValueError('Nonlocal USD dependency cannot be snapshotted: '+str(file))
            resolved[str(file)]=sha(file)
    return [dict(path=path,sha256=digest) for path,digest in sorted(resolved.items())]


def verify_usd_dependencies(records, base=None):
    if not records: raise ValueError('Recorded USD dependency snapshot required')
    for record in records:
        file=Path(record['path'])
        if not file.is_absolute(): file=Path(base or '.')/file
        if sha(file)!=record['sha256']:
            raise ValueError('USD dependency checksum mismatch: '+str(file))
        record['path']=str(file.resolve())


def load_panel_manifest(path):
    path = Path(path).resolve()
    value = json.loads(path.read_text())
    for label, asset in value['assets'].items():
        file = Path(asset['path'])
        file = file if file.is_absolute() else path.parent / file
        if sha(file) != asset['sha256']:
            raise ValueError('Panel asset hash mismatch: ' + label)
        asset['path'] = str(file.resolve())
    verify_usd_dependencies(value.get('usd_composition_dependencies'),base=path.parent)
    for name in ('socket_position_m', 'table_position_m') + (('robot_position_m',) if 'robot_position_m' in value else ()):
        v = value[name]
        if len(v) != 3 or not all(math.isfinite(x) for x in v):
            raise ValueError('finite scene positions required')
    groups = value['collision_groups']
    if not groups: raise ValueError('physical collision groups required')
    roots = [g['path'] for g in groups] + value['decorative_roots']
    for root in roots:
        p = PurePosixPath(root)
        if p.is_absolute() or '..' in p.parts or root in ('', '.'):
            raise ValueError('relative named fixture roots required')
    for i, a in enumerate(roots):
        if any(a == b or a.startswith(b + '/') or b.startswith(a + '/') for b in roots[i+1:]):
            raise ValueError('physical/decorative selections overlap')
    for group in groups:
        h, shell = group['spacing_m'], group['thickness_m']
        if not math.isfinite(h) or h <= 0 or not math.isfinite(shell) or shell < 0 or (shell and shell < 2*h):
            raise ValueError('positive grid spacing and resolved shell required')
    value['manifest_path'] = str(path)
    value['manifest_sha256'] = sha(path)
    return value


def fixture_mesh_selection(stage, prefix, profile):
    """Every mesh must be explicitly physical or decorative, in both engines."""
    from pxr import Usd, UsdGeom
    physical = [g['path'] for g in profile['collision_groups']]
    decorative = profile['decorative_roots']
    selected = []
    for prim in Usd.PrimRange(stage.GetPrimAtPath(prefix + '/Fixtures')):
        if not prim.IsA(UsdGeom.Mesh): continue
        relative = str(prim.GetPath())[len(prefix + '/Fixtures/'):]
        match = lambda root: relative == root or relative.startswith(root + '/')
        p, d = any(map(match, physical)), any(map(match, decorative))
        if p == d: raise ValueError('unclassified/ambiguous fixture mesh: ' + relative)
        selected.append((prim, p))
    return selected


def prepare_profile(rack, panel, out):
    """Non-destructive rack service overlay; original commercial assets untouched."""
    from pxr import Usd, UsdGeom, Gf
    import numpy as np
    rack, panel, out = Path(rack).resolve(), Path(panel).resolve(), Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    authoring = json.loads((panel / 'authoring_manifest.json').read_text())
    overlay = out / 'rack-service-overlay.usda'
    stage = Usd.Stage.CreateNew(str(overlay))
    root = UsdGeom.Xform.Define(stage, '/Rack').GetPrim()
    root.GetReferences().AddReference(str(rack)); stage.SetDefaultPrim(root)
    UsdGeom.SetStageMetersPerUnit(stage, 1.); UsdGeom.SetStageUpAxis(stage, 'Z')
    paths = [p.GetPath() for p in Usd.PrimRange(root, Usd.TraverseInstanceProxies()) if p.IsInstance()]
    for path in paths:
        prim = stage.GetPrimAtPath(path)
        if prim.IsValid() and prim.IsInstance(): prim.SetInstanceable(False)
    assembly = '/Rack/external/tn__VR3307VRCS_ASM_1_wN/'
    removed = []
    for suffix in ('DOOR_SPLIT_PERFORATE_WELD_LH_AS', 'DOOR_SPLIT_PERFORATE_WELD_RH_AS', 'CYBERACK_SWING_HANDLE__DIRAK__4'):
        prim = stage.GetPrimAtPath(assembly + suffix)
        if not prim.IsValid(): raise ValueError('Expected pinned front door/handle: ' + suffix)
        prim.SetActive(False); removed.append(str(prim.GetPath()))
    rails = []
    world_rotation = np.array([[0., 1., 0.], [-1., 0., 0.], [0., 0., 1.]])
    shift = np.array([0., -.2547, 0.])
    for suffix in ('ASSY_PMA_600W_48U_0001_ASM_1', 'tn__ASS_Y_PMA_600W_48U_RH_ESERIES__1_pl0'):
        prim = stage.GetPrimAtPath(assembly + 'tn__VR3307VRCS_1_ASM_1_tQ/' + suffix)
        if not prim.IsValid(): raise ValueError('Expected pinned adjustable front rail: ' + suffix)
        matrix = np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(prim))
        local = shift @ np.linalg.inv(matrix[:3,:3] @ world_rotation)
        UsdGeom.Xformable(prim).AddTranslateOp(opSuffix='serviceAdjustment').Set(Gf.Vec3d(*map(float, local)))
        rails.append(dict(path=str(prim.GetPath()), world_shift_m=shift.tolist()))
    stage.GetRootLayer().Save()
    fixtures = panel / 'fixtures.usdc'
    fs = Usd.Stage.Open(str(fixtures)); decorative = []
    for prim in fs.GetDefaultPrim().GetChildren():
        if prim.GetAttribute('userProperties:deformx_collision').Get() is False:
            decorative.append(prim.GetName())
    groups = [dict(path=name, spacing_m=.0005 if name=='PatchPanelPhysical' else .001 if name in ('NeighbourSocketsPhysical','PickupPedestal') else .003,
                   thickness_m=.002 if name=='NeighbourSocketsPhysical' else 0.)
              for name in authoring['physical_objects']]
    profile = dict(kind='rack-mounted-patch-panel', assets={
        'rack_overlay':dict(path=str(overlay),sha256=sha(overlay)),
        'rack_source':dict(path=str(rack),sha256=sha(rack)),
        'fixtures':dict(path=str(fixtures),sha256=sha(fixtures)),
        'authoring_manifest':dict(path=str(panel/'authoring_manifest.json'),sha256=sha(panel/'authoring_manifest.json'))},
        socket_position_m=authoring['socket_position_m'], table_position_m=[0.,-.4,1.003], robot_position_m=[0.,.15,1.003],
        collision_groups=groups, decorative_roots=decorative,
        overlay=dict(removed_front_service_parts=removed, moved_original_front_rails=rails),
        socket_mouth_recess_m=authoring['socket_mouth_recess_m'], port_count=24, active_port=7,
        camera_variant='rack-panel', panel_center_m=[.5,.537,1.25],
        source=authoring['source'], original_active_cavity_unchanged=True,
        old_adapters_present=False, generator_sha256=sha(Path(__file__)))
    for label,file,expected in (
            ('panel_license',panel/'LICENSE.panel-MIT.txt',authoring['source']['license_sha256']),
            ('panel_generator',panel/'generator.py',authoring['script_sha256'])):
        if sha(file)!=expected: raise ValueError('Authored panel source/license mismatch: '+label)
        profile['assets'][label]=dict(path=str(file),sha256=expected)
    profile['usd_composition_dependencies']=snapshot_usd_dependencies([overlay,fixtures])
    path = out / 'manifest.json'; path.write_text(json.dumps(profile, indent=2)+'\n')
    load_panel_manifest(path)
    return profile
