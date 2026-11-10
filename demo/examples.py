"""Discover and launch DeformX examples without importing a simulation runtime."""
import argparse
import subprocess
import sys

EXAMPLES = {
    'hanging': ('beginner', 'Gravity and a fixed root', 'demo.basics.run', ['--case','hanging']),
    'endpoints': ('beginner', 'Move cable endpoints', 'demo.basics.run', ['--case','endpoints']),
    'obstacle': ('beginner', 'Cable against an obstacle', 'demo.basics.run', ['--case','obstacle']),
    'self-contact': ('beginner', 'Self-contact on/off comparison', 'demo.basics.run', ['--case','self_contact']),
    'materials': ('beginner', 'Compare bending stiffnesses', 'demo.basics.run', ['--case','materials']),
    'tip-target': ('task', 'reset/step task with a simulated free-tip goal', 'demo.basics.task', []),
    'braid': ('showcase', 'Three dynamic strands and alternating carriers', 'demo.braiding_machine.run', []),
    'winch': ('showcase', 'Frictional cable transport and a passive axle', 'demo.winch_tension.run', []),
    'contact-arena': ('benchmark', 'Independent worlds with moving guide contact', 'demo.contact_arena.run', []),
    'guitar': ('showcase', 'Repeated pick contact with five guitar strings', 'demo.humanoid_guitar.run', []),
    'rack-insertion': ('task', 'Contact grasp, connector insertion and release', 'demo.rack_insertion.run', []),
    'cascade': ('showcase', '48 interacting cables falling over an obstacle', 'demo.cable_cascade.run', []),
}

RUNTIMES = {
    'guitar': 'CUDA-capable DeformX, SciPy, Trimesh, Warp and prepared G1/guitar assets; Blender for preparation and Isaac Sim for rendering.',
    'rack-insertion': 'Isaac Lab, Isaac Sim and CUDA-capable DeformX; prepared robot, rack and connector assets.',
}
INPUTS = {
    'guitar': '--assets runs/guitar-assets ',
    'rack-insertion': '--profile runs/rack-assets/profile.json ',
}


def main(argv=None):
    argv=list(sys.argv[1:] if argv is None else argv)
    # Parse catalog switches only before the example name; runner --help remains useful.
    if argv and argv[0] in EXAMPLES:
        _,_,module,extra=EXAMPLES[argv[0]]
        flags=['-S'] if sys.flags.no_site else []
        return subprocess.call([sys.executable,*flags,'-m',module,*extra,*argv[1:]])
    parser=argparse.ArgumentParser(description=__doc__,epilog='Run: python -m demo.examples NAME --help')
    parser.add_argument('--list',action='store_true')
    parser.add_argument('--info',choices=list(EXAMPLES))
    if argv and not argv[0].startswith('-'): parser.error('Unknown example: '+argv[0])
    args=parser.parse_args(argv)
    if args.info:
        level,title,module,extra=EXAMPLES[args.info]
        doc='/'.join(module.split('.')[:-1])+'/README.md'
        runtime=RUNTIMES.get(args.info, 'CUDA-capable DeformX (NumPy, PyTorch, ropesim); Isaac Sim only for optional native rendering.')
        inputs=INPUTS.get(args.info, '')
        print(f'{args.info} / {level}\n{title}\nModule: {module}\nDocumentation: {doc}\nRuntime: {runtime}\nUsage: python -m demo.examples {args.info} {inputs}--out runs/{args.info}\nOptions: python -m demo.examples {args.info} --help')
    else:
        for name,(level,title,_,_) in EXAMPLES.items(): print(f'{name:16} {level:10} {title}')
    return 0


if __name__=='__main__': raise SystemExit(main())
