"""Download pinned humanoid and guitar source assets with exact SHA-256 checks."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath
from urllib.request import Request, urlopen
import argparse
import hashlib
import json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    spec = json.loads(Path(__file__).with_name('asset_sources.json').read_text())
    root = args.out.resolve()
    root.mkdir(parents=True, exist_ok=True)

    def fetch(item):
        rel = PurePosixPath(item['path'])
        if rel.is_absolute() or '..' in rel.parts:
            raise ValueError('Unsafe asset path: ' + str(rel))
        destination = root / rel
        if not destination.resolve().is_relative_to(root):
            raise ValueError('Asset path resolves outside output root: ' + str(rel))
        if destination.exists():
            data = destination.read_bytes()
        else:
            with urlopen(Request(item['url'], headers={'User-Agent':'DeformX-demo-assets/1.0'}), timeout=90) as response:
                data = response.read()
        digest = hashlib.sha256(data).hexdigest()
        if digest != item['sha256']:
            raise ValueError('Asset hash mismatch: ' + str(rel))
        if not destination.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(data)
        return dict(item, resolved_path=str(destination), bytes=len(data))

    with ThreadPoolExecutor(max_workers=6) as pool:
        files = list(pool.map(fetch, spec['files']))
    result = dict(spec, files=files)
    (root/'verified-sources.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'files':len(files), 'output':str(root), 'verified':True}))


if __name__ == '__main__':
    main()
