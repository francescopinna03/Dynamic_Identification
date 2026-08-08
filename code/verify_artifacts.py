from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    manifest_path = args.manifest or root / "artifact_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    failures = []
    for item in manifest["artifacts"]:
        path = root / item["path"]
        if not path.is_file():
            failures.append(f"missing {item['path']}")
            continue
        if path.stat().st_size != item["size_bytes"]:
            failures.append(f"size {item['path']}")
        if digest(path) != item["sha256"]:
            failures.append(f"sha256 {item['path']}")
    if failures:
        raise SystemExit("artifact verification failed: " + ", ".join(failures))
    print(f"ARTIFACT MANIFEST: PASS ({len(manifest['artifacts'])} files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
