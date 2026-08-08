from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import stat
import sys
import zipfile
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen


DATA_URL = "https://econweb.ucsd.edu/~vcrawfor/december06_data_20041028.zip"
WEB_URL  = "https://econweb.ucsd.edu/~vcrawfor/december06_app_20041028.zip"

UA = "Mozilla/5.0 (compatible; CGC2006-Acquisition/1.0; academic research)"


def download(url: str, dest: Path, chunk_size: int = 1 << 16) -> None:
    print(f"  Download: {url}")
    print(f"  -> {dest}")
    req = Request(url, headers={"User-Agent": UA})
    with urlopen(req, timeout=60) as resp:
        total = resp.headers.get("Content-Length")
        total = int(total) if total is not None else None
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "wb") as f:
            n_read = 0
            while True:
                buf = resp.read(chunk_size)
                if not buf:
                    break
                f.write(buf)
                n_read += len(buf)
                if total:
                    pct = 100.0 * n_read / total
                    print(f"\r    {n_read/1024:.1f} KiB "
                          f"/ {total/1024:.1f} KiB ({pct:.1f}%)",
                          end="", flush=True)
                else:
                    print(f"\r    {n_read/1024:.1f} KiB", end="", flush=True)
    print()


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for buf in iter(lambda: f.read(1 << 20), b""):
            h.update(buf)
    return h.hexdigest()


def make_readonly(path: Path) -> None:
    for root, dirs, files in os.walk(path):
        for d in dirs:
            full = Path(root) / d
            full.chmod(full.stat().st_mode | stat.S_IRUSR | stat.S_IRGRP
                       | stat.S_IROTH)
        for f in files:
            full = Path(root) / f
            mode = full.stat().st_mode
            new_mode = mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH)
            full.chmod(new_mode)


def extract_zip(zip_path: Path, dest: Path) -> list[str]:
    print(f"  Extract: {zip_path.name} -> {dest}")
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        for name in zf.namelist():
            if name.startswith("/") or ".." in Path(name).parts:
                raise RuntimeError(f"Suspicious path in archive: {name!r}")
        zf.extractall(dest)
        names = zf.namelist()
    print(f"    {len(names)} entries extracted")
    return names


def setup_dirs(root: Path) -> dict[str, Path]:
    paths = {
        "raw_zips":  root / "data" / "raw" / "cgc2006" / "zips",
        "raw_data":  root / "data" / "raw" / "cgc2006" / "data",
        "raw_web":   root / "data" / "raw" / "cgc2006" / "web",
        "interim":   root / "data" / "interim" / "cgc2006",
        "processed": root / "data" / "processed" / "cgc2006",
        "code":      root / "code" / "cgc2006",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths


def inventory(root: Path, out: Path) -> None:
    lines = []
    for base in sorted(root.iterdir()):
        if not base.is_dir():
            continue
        lines.append(f"\n=== {base.name}/ ===")
        for sub in sorted(base.rglob("*")):
            if sub.is_file():
                size_kb = sub.stat().st_size / 1024
                rel = sub.relative_to(root)
                lines.append(f"  {size_kb:>10.1f} KiB  {rel}")
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n  Inventory written to: {out}")
    print("\n".join(lines[:60]))
    if len(lines) > 60:
        print(f"  ... ({len(lines) - 60} additional lines in {out.name})")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd(),
                        help="Project root (default: cwd)")
    parser.add_argument("--force-redownload", action="store_true",
                        help="Download again even if the archives exist")
    args = parser.parse_args(argv)

    root = args.root.resolve()
    print(f"Root: {root}\n")

    paths = setup_dirs(root)
    print("Directory structure created:")
    for k, p in paths.items():
        print(f"  {k:<10}  {p}")
    print()

    print("[1/4] Download ZIP archives")
    archives = []
    for url, label in [(DATA_URL, "data_appendix"), (WEB_URL, "web_appendix")]:
        fname = Path(urlparse(url).path).name
        dest  = paths["raw_zips"] / fname
        if dest.exists() and not args.force_redownload:
            print(f"  Skip (already exists): {dest.name}")
        else:
            try:
                download(url, dest)
            except Exception as e:
                print(f"  Download error for {url}: {e}")
                print("  Check the connection and try the link manually in a browser.")
                return 2
        archives.append((label, dest))

    print("\n[2/4] Compute SHA-256 hashes")
    checks = []
    for label, path in archives:
        h = sha256_of(path)
        size_kb = path.stat().st_size / 1024
        line = f"{h}  {path.name}  ({size_kb:.1f} KiB, {label})"
        print(f"  {line}")
        checks.append(line)
    (paths["raw_zips"].parent / "CHECKSUMS.txt").write_text(
        "\n".join(checks) + "\n", encoding="utf-8"
    )

    print("\n[3/4] Extract archives")
    for label, zip_path in archives:
        dest = paths["raw_data"] if label == "data_appendix" else paths["raw_web"]
        try:
            extract_zip(zip_path, dest)
        except zipfile.BadZipFile as e:
            print(f"  Error: {zip_path.name} is not a valid ZIP archive: {e}")
            return 3

    print("\n  Set raw/cgc2006/ to read-only...")
    make_readonly(paths["raw_zips"].parent)

    print("\n[4/4] Inventory")
    inventory(paths["raw_zips"].parent,
              out=paths["raw_zips"].parent / "INVENTORY.txt")

    print("\n" + "=" * 60)
    print("Acquisition completed.")
    print("=" * 60)
    print(f"Raw zips:   {paths['raw_zips']}")
    print(f"Data dir:   {paths['raw_data']}")
    print(f"Web dir:    {paths['raw_web']}")
    print(f"Checksums:  {paths['raw_zips'].parent / 'CHECKSUMS.txt'}")
    print(f"Inventory:  {paths['raw_zips'].parent / 'INVENTORY.txt'}")
    print()
    print("Next step: inspect the extracted files and run preprocessing.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
