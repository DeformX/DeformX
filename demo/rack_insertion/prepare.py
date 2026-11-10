"""Prepare the pinned connector, mounted panel, robot and table for rack insertion."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess

from .assets import sha, verify_sources
from .panel import prepare_profile, snapshot_usd_dependencies
from .profile import PROFILE_KIND

RJ45_SHA256 = '50c95bcfb63544777f9148d548aac6f16b62f65cacbaaa9316453d579de4b4fa'


def prepare_connector(source, out):
    import numpy as np
    from pxr import Usd, UsdGeom, UsdPhysics, UsdUtils, Sdf
    source, out = Path(source).resolve(), Path(out).resolve()
    digest = sha(source)
    if digest != RJ45_SHA256:
        raise ValueError('Unexpected RJ45 asset bytes; inspect and update the pinned profile deliberately')
    stage = Usd.Stage.Open(str(source))
    if UsdGeom.GetStageMetersPerUnit(stage) != 1 or UsdGeom.GetStageUpAxis(stage) != 'Z':
        raise ValueError('Expected pinned meter/Z-up connector asset')
    _, _, missing = UsdUtils.ComputeAllDependencies(str(source))
    if missing:
        raise ValueError('Unresolved connector dependencies: ' + str(missing))
    out.mkdir(parents=True, exist_ok=False)
    cache = UsdGeom.XformCache()
    info, origins, bounds = {}, {}, {}
    for name, mass in [('Plug', 0.012), ('Latch', 0.0003), ('Socket', None)]:
        prim = stage.GetPrimAtPath('/World/' + name)
        matrix = np.asarray(cache.GetLocalToWorldTransform(prim))
        if not np.allclose(matrix[:3, :3], np.eye(3)):
            raise ValueError('Pinned profile expects translation-only source mesh transforms')
        origins[name] = matrix[3, :3]
        points = np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get())
        bounds[name] = (points.min(0), points.max(0))
        target = Usd.Stage.CreateNew(str(out / (name.lower() + '.usda')))
        UsdGeom.SetStageMetersPerUnit(target, 1.0)
        UsdGeom.SetStageUpAxis(target, 'Z')
        root = UsdGeom.Xform.Define(target, '/' + name).GetPrim()
        target.SetDefaultPrim(root)
        path = root.GetPath().AppendChild('Geometry')
        Sdf.CopySpec(stage.GetRootLayer(), prim.GetPath(), target.GetRootLayer(), path)
        geometry = target.GetPrimAtPath(path)
        UsdGeom.Xformable(geometry).ClearXformOpOrder()
        UsdPhysics.CollisionAPI.Apply(geometry)
        UsdPhysics.MeshCollisionAPI.Apply(geometry).CreateApproximationAttr('sdf' if mass else 'none')
        if mass is not None:
            body = UsdPhysics.RigidBodyAPI.Apply(root)
            body.CreateKinematicEnabledAttr(False)
            body.CreateRigidBodyEnabledAttr(True)
            UsdPhysics.MassAPI.Apply(root).CreateMassAttr(mass)
        target.GetRootLayer().Save()
        info[name.lower()] = str(out / (name.lower() + '.usda'))
    info.update(source=str(source), source_sha256=digest, units='m',
        insertion_axis=[0.0, 1.0, 0.0],
        seated_plug_minus_socket=(origins['Plug'] - origins['Socket']).tolist(),
        latch_pivot_in_plug=(origins['Latch'] - origins['Plug']).tolist(),
        grasp_offset=[0.0, -0.026, 0.0],
        cable_exit_offset=[0.0, float(bounds['Plug'][0][1]), 0.0006787498],
        socket_entry_y=float(bounds['Socket'][0][1]),
        plug_tip_y=float(bounds['Plug'][1][1]),
        seated_depth_m=float(origins['Plug'][1] + bounds['Plug'][1][1] - origins['Socket'][1] - bounds['Socket'][0][1]),
        assumed_mass_kg={'plug':0.012, 'latch':0.0003},
        source_commit='f1aa3296cb984e9a341f92f5a335b0d807632cc9',
        source_license='Apache-2.0; Newton Developers',
        source_url='https://github.com/newton-physics/newton/blob/f1aa3296cb984e9a341f92f5a335b0d807632cc9/newton/examples/assets/rj45_plug.usd')
    (out / 'manifest.json').write_text(json.dumps(info, indent=2) + '\n')
    return info


def prepare(sources, output, blender, source_records):
    output.mkdir(parents=True, exist_ok=False)
    connector = output / "connector"
    prepare_connector(sources / "newton/rj45_plug.usd", connector)
    shutil.copyfile(Path(__file__).parent / "licenses/NEWTON-Apache-2.0.txt", connector / "LICENSE.txt")
    blender_steps = (
        ("blender_patch_panel.py", [
            "--source", sources / "patch-panel/patch-panel-19inch.3mf",
            "--license", sources / "patch-panel/LICENSE",
            "--socket-usd", connector / "socket.usda",
            "--out", output / "panel",
        ]),
        ("blender_termination.py", ["--out", output / "termination"]),
    )
    for script, arguments in blender_steps:
        subprocess.run([
            blender, "--background", "--factory-startup", "--python-exit-code", "1",
            "--python", Path(__file__).with_name(script), "--", *arguments,
        ], check=True)
    prepare_profile(sources / "VR3307/simready_usd/VR3307_main.usd",
                    output / "panel", output / "mounted")
    isaac = sources / "isaac-4.5/Isaac"
    files = {
        "connector": connector / "manifest.json",
        "plug": connector / "plug.usda",
        "latch": connector / "latch.usda",
        "socket": connector / "socket.usda",
        "panel": output / "mounted/manifest.json",
        "robot": isaac / "IsaacLab/Robots/FrankaEmika/panda_instanceable.usd",
        "table": isaac / "Props/Mounts/SeattleLabTable/table_instanceable.usd",
        "termination_visual": output / "termination/termination.usdc",
    }
    pinned = {str((sources / name).resolve()): record["sha256"]
              for name, record in source_records.items()}
    usd_layers = snapshot_usd_dependencies([files["robot"], files["table"], files["termination_visual"]])
    pinned.update({record["path"]: record["sha256"] for record in usd_layers})
    profile = {
        "kind": PROFILE_KIND,
        "assets": {label: {"path": str(path), "sha256": sha(path)}
                   for label, path in files.items()},
        "dependencies": [{"path": path, "sha256": digest} for path, digest in sorted(pinned.items())],
    }
    path = output / "profile.json"
    path.write_text(json.dumps(profile, indent=2) + "\n")
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, required=True, help="Pinned files from demo.rack_insertion.assets")
    parser.add_argument("--out", type=Path, required=True, help="New prepared asset directory")
    parser.add_argument("--blender", default="blender", help="Path to Blender 4.5")
    args = parser.parse_args()
    try:
        source_records = verify_sources(args.sources.resolve())
        if args.out.exists():
            raise ValueError(f"Choose a new asset output directory: {args.out}")
        from pxr import Usd  # Verify the standalone authoring runtime before creating output.
        import numpy
        blender = shutil.which(args.blender)
        if blender is None:
            raise ValueError("Blender executable not found. Pass --blender /path/to/blender.")
        path = prepare(args.sources.resolve(), args.out.resolve(), blender, source_records)
    except ImportError as exc:
        parser.error(f"Use Python 3.11 with usd-core and numpy for asset preparation: {exc}")
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        parser.error(str(exc))
    print(f"Prepared rack insertion profile at {path}")


if __name__ == "__main__":
    main()
