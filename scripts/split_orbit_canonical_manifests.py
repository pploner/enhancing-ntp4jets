#!/usr/bin/env python3
"""Create deterministic, disjoint canonical train/validation and test manifests.

The canonical tokenizer configs consume ``<process>_train_val.txt`` and
``<process>_test.txt``.  This script derives those files from the raw
``<process>.txt`` manifests generated for the production_final dataset.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from gabbro.data.orbit_taxonomy import FIVE_CLASS_GROUPS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        default=PROJECT_ROOT / "manifests" / "production_final",
        help="Directory containing raw <process>.txt manifests.",
    )
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace existing generated split manifests.",
    )
    return parser.parse_args()


def read_paths(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return [
        line.strip()
        for line in path.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def split_paths(paths: list[str], process: str, seed: int, test_fraction: float):
    if len(paths) < 2:
        raise ValueError(f"{process} needs at least two parquet files, got {len(paths)}")
    ordered = sorted(
        paths,
        key=lambda value: hashlib.sha256(
            f"{seed}\0{process}\0{value}".encode()
        ).hexdigest(),
    )
    test_count = min(max(round(len(ordered) * test_fraction), 1), len(ordered) - 1)
    return ordered[test_count:], ordered[:test_count]


def write_manifest(path: Path, paths: list[str], force: bool) -> None:
    if path.exists() and not force:
        raise FileExistsError(f"Refusing to overwrite {path}; pass --force to replace it")
    path.write_text("\n".join(paths) + "\n")


def main() -> None:
    args = parse_args()
    if not 0 < args.test_fraction < 1:
        raise ValueError("--test-fraction must be strictly between zero and one")
    if not args.manifest_dir.is_dir():
        raise NotADirectoryError(args.manifest_dir)

    processes = [process for group in FIVE_CLASS_GROUPS.values() for process in group]
    for process in processes:
        try:
            paths = read_paths(args.manifest_dir / f"{process}.txt")
        except FileNotFoundError:
            print(f"{process}: SKIPPED, no raw manifest found (empty or missing EOS directory)")
            continue
        try:
            train_val, test = split_paths(paths, process, args.seed, args.test_fraction)
        except ValueError as exc:
            print(f"{process}: SKIPPED, {exc}")
            continue
        write_manifest(args.manifest_dir / f"{process}_train_val.txt", train_val, args.force)
        write_manifest(args.manifest_dir / f"{process}_test.txt", test, args.force)
        print(f"{process}: {len(train_val)} train_val files, {len(test)} test files")


if __name__ == "__main__":
    main()
