"""Download and verify the pinned upstream assets for rack insertion."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
from tempfile import NamedTemporaryFile
from urllib.request import Request, urlopen


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def sources():
    return json.loads(Path(__file__).with_name("asset_sources.json").read_text())


def verify_sources(root):
    files = sources()
    for name, record in files.items():
        file = root / name
        if not file.is_file() or sha(file) != record["sha256"]:
            raise ValueError(
                f"Missing or changed source asset {file}. "
                f"Run python -m demo.rack_insertion.assets --out {root} "
                "or copy in the exact files listed in asset_sources.json."
            )
    return files


def download(root, name, record):
    target = root / name
    if target.is_file():
        if sha(target) != record["sha256"]:
            raise ValueError(f"Checksum mismatch in {target}. Move it aside and retry.")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        request = Request(record["url"], headers={"User-Agent": "DeformX rack-insertion assets"})
        with NamedTemporaryFile(dir=target.parent, delete=False) as stream:
            temporary = Path(stream.name)
            with urlopen(request, timeout=90) as response:
                shutil.copyfileobj(response, stream)
        if sha(temporary) != record["sha256"]:
            raise ValueError(f"Downloaded SHA-256 mismatch for {name}")
        temporary.replace(target)
    except (OSError, ValueError) as exc:
        raise ValueError(f"Could not fetch {record['url']} into {target}: {exc}") from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="Local source asset directory")
    args = parser.parse_args()
    files = sources()
    errors = []
    with ThreadPoolExecutor(max_workers=6) as pool:
        for future in [pool.submit(download, args.out, name, item) for name, item in files.items()]:
            try:
                future.result()
            except ValueError as exc:
                errors.append(str(exc))
    if errors:
        parser.error("\n".join(errors))
    print(f"Verified {len(files)} pinned upstream files under {args.out.resolve()}")


if __name__ == "__main__":
    main()
