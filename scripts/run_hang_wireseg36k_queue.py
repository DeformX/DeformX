#!/usr/bin/env python3
from __future__ import annotations

import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CFG = REPO / "Dataset_generator" / "config.py"
PYTHON = REPO / ".venv-isaac45" / "bin" / "python"
ASSET = REPO / "asset_wireseg32k"
OUT_ROOT = REPO / "output" / "wireseg32k"
LOG_ROOT = REPO / "logs" / "dataset_queue"
SEED_VARIANTS = 5

RUNS = [
    ("hang_n2", 2, ASSET / "wires_traj_data" / "hang_n2_100.npz"),
    ("hang_n4", 4, ASSET / "wires_traj_data" / "hang_n4_100.npz"),
    ("hang_n8", 8, ASSET / "wires_traj_data" / "hang_n8_100.npz"),
]


def replace_line(text: str, prefix: str, newline: str) -> str:
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if line.strip().startswith(prefix):
            lines[i] = newline
            return "\n".join(lines) + "\n"
    raise RuntimeError(f"Missing config line starting with {prefix!r}")


def patch_config(name: str, num_wires: int, npz_path: Path) -> None:
    text = CFG.read_text()
    text = replace_line(text, "scene_usd:", f'    scene_usd: str = "{ASSET / "usd" / "rod_hang_flying.usdc"}"')
    text = replace_line(text, "npz_path:", f'    npz_path: str = "{npz_path}"')
    text = replace_line(text, "table_texture_dir:", f'    table_texture_dir: str = "{ASSET / "ground"}"')
    text = replace_line(text, "wire_asset_dir:", f'    wire_asset_dir: str = "{ASSET / "usd" / "wire_usdc" / "wire_usdc"}"')
    text = replace_line(text, "num_wires:", f"    num_wires: int = {num_wires}")
    text = replace_line(text, "hdr_path:", f'    hdr_path: str = "{ASSET / "background" / "boiler_room_4k.hdr"}"')
    text = replace_line(text, "hdr_dir:", f'    hdr_dir: str = "{ASSET / "background"}"')
    text = replace_line(text, "cams_sample_per_frame:", "    cams_sample_per_frame: int = 5")
    text = replace_line(text, "capture_out_dir:", f'    capture_out_dir: str = "{OUT_ROOT / name}"')
    CFG.write_text(text)


def main() -> int:
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    for name, num_wires, npz_path in RUNS:
        patch_config(name, num_wires, npz_path)
        log_path = LOG_ROOT / f"{name}.log"
        cmd = [
            str(PYTHON),
            "Dataset_generator/cli.py",
            "--frame_start",
            "0",
            "--frame_end",
            "199",
            "--num_variants",
            str(SEED_VARIANTS),
            "--do_seg",
            "--do_depth",
        ]
        with log_path.open("w") as log:
            log.write(f"[RUN] {name} num_wires={num_wires} npz={npz_path} out={OUT_ROOT / name}\n")
            log.write("[CMD] " + " ".join(cmd) + "\n")
            log.flush()
            ret = subprocess.run(cmd, cwd=REPO, stdout=log, stderr=subprocess.STDOUT).returncode
        if ret != 0:
            print(f"[FAIL] {name} exited with {ret}. See {log_path}", flush=True)
            return ret
        print(f"[DONE] {name}. See {log_path}", flush=True)

    print("[DONE] all hang wireseg32k runs", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
