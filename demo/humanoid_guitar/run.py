"""Prepare, simulate, check or render the 24-stroke humanoid guitar example."""

import argparse
import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path


PACKAGE = Path(__file__).resolve().parent
URDF = 'unitree/robots/g1_description/g1_29dof_rev_1_0_with_inspire_hand_DFQ.urdf'
GUITAR = 'guitar/solid-body-electric-guitar.glb'
PROFILE = Path('contact-v4/contact_profile.json')


def command(module, *args, gpu_lock=None):
    launch = [sys.executable, *(['-S'] if sys.flags.no_site else []),
              '-m', f'demo.humanoid_guitar.{module}', *map(str, args)]
    if gpu_lock:
        launch = ['flock', '-o', str(gpu_lock), *launch]
    subprocess.run(launch, check=True)


def prepare(assets, blender):
    sources = assets / 'sources'
    command('fetch_assets', '--out', sources)
    command('prepare_assets', '--urdf', sources / URDF, '--guitar', sources / GUITAR,
            '--out', assets / 'prepared', '--blender', blender)
    for dest, base in [('contact-v2', None), ('contact-v3', 'contact-v2'), ('contact-v4', 'contact-v3')]:
        args = ['--prepared', assets / 'prepared', '--out', assets / dest, '--blender', blender]
        if base:
            flag = '--extend-hands-from' if dest == 'contact-v3' else '--wide-pick-from'
            args.extend((flag, assets / base / 'contact_profile.json'))
        command('prepare_contact_assets', *args)


def simulate(assets, out, substeps, gpu_lock):
    out.mkdir(parents=True, exist_ok=False)
    command('bind_contact_pose', '--profile', assets / PROFILE,
            '--template', PACKAGE / 'pose_template.json', '--out', out / 'pose.json')
    command('contact_full_motion', '--profile', assets / PROFILE, '--pose-plan', out / 'pose.json',
            '--pick-preset', 'strum5-loop-long', '--out', out / 'motion')
    command('contact_freeze', '--profile', out / 'motion/profile.json',
            '--template', PACKAGE / 'criteria_template.json', '--out', out / 'criteria.json')
    command('contact_integrity', '--profile', out / 'motion/profile.json', '--cache', out / 'cache',
            '--bake', '--out', out / 'cache-report.json', gpu_lock=gpu_lock)
    command('contact_run', '--motion', out / 'motion', '--cache', out / 'cache',
            '--criteria', out / 'criteria.json', '--substeps', substeps, '--iterations', 24,
            '--out', out / 'case', gpu_lock=gpu_lock)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__, epilog=(
        'Commands: prepare (download and build assets), run (default), audit, render. '
        'Use COMMAND --help for command options.'))
    action = argv.pop(0) if argv and argv[0] in {'prepare', 'run', 'audit', 'render'} else 'run'
    if action == 'prepare':
        parser.add_argument('--assets', required=True, type=Path, help='asset root to download and build')
        parser.add_argument('--blender', default='blender', help='Blender 4.5 executable')
    elif action == 'run':
        parser.add_argument('--assets', required=True, type=Path, help='asset root built by prepare')
        parser.add_argument('--out', required=True, type=Path, help='new simulation output directory')
        parser.add_argument('--substeps', type=int, choices=(200, 300), default=200,
                            help='solver substeps per 480 Hz state (default: 200)')
    else:
        parser.add_argument('--out', required=True, type=Path, help='completed simulation output directory')
    if action in {'run', 'render'}:
        parser.add_argument('--gpu-lock', type=Path, help='optional flock file shared by GPU jobs on this host')
    args = parser.parse_args(argv)
    gpu_lock = getattr(args, 'gpu_lock', None)
    if gpu_lock and not shutil.which('flock'):
        parser.error('flock is required with --gpu-lock')
    if action == 'prepare':
        blender = shutil.which(args.blender)
        if blender is None:
            parser.error('Blender 4.5 executable not found; provide --blender /path/to/blender')
        assets = args.assets.resolve()
        for name in ('prepared', 'contact-v2', 'contact-v3', 'contact-v4'):
            dest = assets / name
            if dest.exists() and (not dest.is_dir() or any(dest.iterdir())):
                parser.error(f'Asset preparation output already exists: {dest}; use a new --assets directory')
        prepare(assets, blender)
    elif action == 'run':
        assets, out = args.assets.resolve(), args.out.resolve()
        if not (assets / PROFILE).is_file():
            parser.error(f'Missing {assets / PROFILE}; run prepare --assets {assets} first')
        if out.exists():
            parser.error(f'Output already exists: {out}; use a new --out directory')
        needed = ('numpy', 'scipy', 'torch', 'trimesh', 'warp', 'ropesim')
        missing = [name for name in needed if importlib.util.find_spec(name) is None]
        if missing:
            parser.error('Missing simulation dependencies: ' + ', '.join(missing) + '; see the guitar README')
        try:
            from ropesim import RopeBatch
            from torch import cuda
        except (ImportError, OSError) as exc:
            parser.error(f'Could not load the pinned CUDA backend; see the CUDA example installation: {exc}')
        from_positions = getattr(RopeBatch, 'from_positions', None)
        add_mesh = getattr(RopeBatch, 'add_mesh', None)
        if ('deterministic' not in (getattr(from_positions, '__doc__', None) or '')
                or '64 meshes' not in (getattr(add_mesh, '__doc__', None) or '')):
            parser.error('This scene requires the pinned ropesim backend with deterministic sweeps and 64 mesh slots; see the guitar README')
        if not cuda.is_available():
            parser.error('No CUDA GPU is available to PyTorch; check the NVIDIA driver, GPU visibility and CUDA PyTorch installation')
        simulate(assets, out, args.substeps, gpu_lock)
    else:
        out = args.out.resolve()
        if not (out / 'case/report.json').is_file():
            parser.error(f'Missing {out / "case/report.json"}; complete a simulation with run --out {out} first')
        if action == 'audit':
            command('portable_audit', '--case', out / 'case', '--out', out / 'audit.json')
        else:
            if not (out / 'audit.json').is_file():
                parser.error(f'Missing {out / "audit.json"}; run audit --out {out} first')
            if (out / 'native').exists():
                parser.error(f'Capture output already exists: {out / "native"}; move it before rendering again')
            render_packages = ('isaacsim', 'numpy', 'scipy', 'trimesh', 'PIL')
            missing = [name for name in render_packages if importlib.util.find_spec(name) is None]
            missing.extend(name for name in ('ffmpeg', 'ffprobe') if not shutil.which(name))
            if missing:
                parser.error('Missing render dependencies: ' + ', '.join(missing) + '; see the guitar README')
            command('contact_render', '--case', out / 'case', '--audit', out / 'audit.json',
                    '--out', out / 'native', gpu_lock=gpu_lock)


if __name__ == '__main__':
    main()
