#!/usr/bin/env python3
"""Queue the three hang-scene renders (hang_n2 / hang_n4 / hang_n8) back to back.

Each run is a separate ``Dataset_generator/cli.py`` subprocess, configured purely
through CLI flags -- this script never edits tracked source. Logs land in
``logs/dataset_queue/<name>.log``.

Usage:
    export ISAAC_PYTHON=/path/to/isaacsim/python.sh   # or a pip-install Isaac python
    python scripts/run_hang_wireseg36k_queue.py

    python scripts/run_hang_wireseg36k_queue.py --dry_run    # print commands only
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT_ROOT = REPO / "output" / "wireseg36k"
LOG_ROOT = REPO / "logs" / "dataset_queue"

# (name, variant index into GeneratorConfig.scene_variant_specs)
#   0=drop_n2  1=drop_n4  2=drop_n8  3=hang_n2  4=hang_n4  5=hang_n8
RUNS = [
    ("hang_n2", 3),
    ("hang_n4", 4),
    ("hang_n8", 5),
]

SEED_VARIANTS = 5
FRAME_START = 0
FRAME_END = 199
CAMS_SAMPLE_PER_FRAME = 5


def isaac_python() -> str:
    """Interpreter used to run the generator.

    ``ISAAC_PYTHON`` should point at Isaac Sim's ``python.sh`` for a launcher-style
    install. For a pip install (``pip install isaacsim[all]``) Isaac Sim runs under
    the active interpreter, so that is the default.
    """
    return os.environ.get("ISAAC_PYTHON") or sys.executable


def build_cmd(name: str, variant: int, python: str) -> list[str]:
    return [
        python,
        "Dataset_generator/cli.py",
        "--variant", str(variant),
        "--out_dir", str(OUT_ROOT / name),
        "--cams_sample_per_frame", str(CAMS_SAMPLE_PER_FRAME),
        "--frame_start", str(FRAME_START),
        "--frame_end", str(FRAME_END),
        "--num_variants", str(SEED_VARIANTS),
        "--do_seg",
        "--do_depth",
    ]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--dry_run", action="store_true", help="Print the commands without running them.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    python = isaac_python()

    if args.dry_run:
        for name, variant in RUNS:
            print(f"[{name}] " + " ".join(build_cmd(name, variant, python)))
        return 0

    if not Path(python).exists():
        print(
            f"[ERROR] Isaac Sim interpreter not found: {python}\n"
            "        Set ISAAC_PYTHON to Isaac Sim's python.sh, or run this script with "
            "the Python environment Isaac Sim is pip-installed into.",
            file=sys.stderr,
        )
        return 1

    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    for name, variant in RUNS:
        cmd = build_cmd(name, variant, python)
        log_path = LOG_ROOT / f"{name}.log"
        with log_path.open("w") as log:
            log.write(f"[RUN] {name} variant={variant} out={OUT_ROOT / name}\n")
            log.write("[CMD] " + " ".join(cmd) + "\n")
            log.flush()
            ret = subprocess.run(cmd, cwd=REPO, stdout=log, stderr=subprocess.STDOUT).returncode
        if ret != 0:
            print(f"[FAIL] {name} exited with {ret}. See {log_path}", flush=True)
            return ret
        print(f"[DONE] {name}. See {log_path}", flush=True)

    print("[DONE] all hang wireseg36k runs", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
