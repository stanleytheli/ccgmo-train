#!/usr/bin/env python3
"""Remove old local runs and legacy model caches after an explicit dry run."""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
DATA_ROOT = Path(
    os.environ.get("AUDIT_DATA_ROOT", "/data/jiang/vennemdp/audit")
).expanduser()
PROTECTED_DATA_ROOT = Path("/data/jiang").resolve()


def directory_size(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(
        entry.stat().st_size
        for entry in path.rglob("*")
        if entry.is_file() or entry.is_symlink()
    )


def human_size(size: int) -> str:
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    value = float(size)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} TiB"


def unique_paths(paths: list[Path]) -> list[Path]:
    result = []
    seen = set()
    for path in paths:
        resolved = path.expanduser().resolve()
        if resolved in seen or not resolved.exists():
            continue
        seen.add(resolved)
        result.append(resolved)
    return result


def is_under(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Actually delete the displayed paths. Without this flag, dry-run only.",
    )
    parser.add_argument(
        "--include-data-root",
        action="store_true",
        help="Also delete future runs and caches under AUDIT_DATA_ROOT.",
    )
    args = parser.parse_args()

    legacy_cache = os.environ.get("TRANSFORMERS_CACHE")
    targets = [
        REPO_ROOT / "runs",
        Path.home() / ".cache" / "huggingface",
        Path.home() / ".cache" / "torch",
        Path.home() / ".cache" / "torch_extensions",
        Path.home() / ".triton",
    ]
    if legacy_cache:
        targets.append(Path(legacy_cache))
    if args.include_data_root:
        targets.append(DATA_ROOT)
    targets = unique_paths(targets)
    protected = [
        path
        for path in targets
        if is_under(path, PROTECTED_DATA_ROOT)
        and not (args.include_data_root and path == DATA_ROOT.resolve())
    ]
    targets = [path for path in targets if path not in protected]

    for path in protected:
        print(f"Protected data path, not selected: {path}")

    if not targets:
        print("No matching run or cache directories exist.")
        return

    total = 0
    print("Directories selected for deletion:")
    for path in targets:
        size = directory_size(path)
        total += size
        print(f"  {human_size(size):>10}  {path}")
    print(f"Total apparent size: {human_size(total)}")

    if not args.yes:
        print("\nDry run only. Add --yes to delete these directories.")
        return

    for path in targets:
        print(f"Deleting {path}")
        shutil.rmtree(path)
    print("Cleanup complete.")


if __name__ == "__main__":
    main()
