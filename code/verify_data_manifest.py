from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--manifest', type=Path,
                        default=Path(__file__).with_name('data_manifest.json'))
    args = parser.parse_args(argv)
    root = args.root.resolve()
    manifest = json.loads(args.manifest.read_text(encoding='utf-8'))

    for item in manifest['raw_archives']:
        path = root / 'data' / 'raw' / 'cgc2006' / 'zips' / item['filename']
        actual = sha256(path)
        if actual != item['sha256']:
            raise SystemExit(f"FAIL HASH RAW {path}: {actual}")
        print(f"PASS HASH RAW: {item['filename']}")

    raw_dir = root / 'data' / 'raw' / 'cgc2006' / 'data'
    for item in manifest['raw_required']:
        path = raw_dir / item['filename']
        if path.stat().st_size != item['size_bytes']:
            raise SystemExit(f"FAIL SIZE RAW {path}: {path.stat().st_size}")
        print(f"PASS SIZE RAW: {item['filename']}")

    for item in manifest['processed']:
        candidates = [
            root / 'data' / 'processed' / 'cgc2006' / item['filename'],
            root / 'data' / 'processed' / item['filename'],
        ]
        path = next((p for p in candidates if p.is_file()), None)
        if path is None:
            raise SystemExit(f"FAIL MISSING PROCESSED: {item['filename']}")
        actual_hash = sha256(path)
        actual_shape = list(pd.read_csv(path).shape)
        if actual_hash != item['sha256']:
            raise SystemExit(f"FAIL HASH PROCESSED {path}: {actual_hash}")
        if actual_shape != item['shape']:
            raise SystemExit(f"FAIL SHAPE PROCESSED {path}: {actual_shape}")
        print(f"PASS PROCESSED: {item['filename']} shape={actual_shape}")

    print("DATA MANIFEST: PASS")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
