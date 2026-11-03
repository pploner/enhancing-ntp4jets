"""Generate per-class parquet manifests from an EOS (or local) directory tree.

Usage
-----
python scripts/make_eos_manifests.py \
    --eos-root /eos/project/f/foundational-model-dataset/samples/production_final \
    --out-dir manifests/

Each immediate subdirectory of ``--eos-root`` is treated as one process class.
One ``.txt`` manifest (one absolute path per line) is written to ``--out-dir``
for every class found.  The manifest format is understood by the existing
``read_path_manifest`` loader in ``gabbro/data/orbit_common.py``.
"""

import argparse
import sys
from pathlib import Path


def collect_parquet_files(directory: Path) -> list[Path]:
    files = sorted(p for p in directory.rglob("*.parquet") if p.is_file())
    return files


def make_manifests(eos_root: Path, out_dir: Path) -> None:
    if not eos_root.is_dir():
        sys.exit(f"ERROR: --eos-root does not exist or is not a directory: {eos_root}")

    class_dirs = sorted(p for p in eos_root.iterdir() if p.is_dir())
    if not class_dirs:
        sys.exit(f"ERROR: No subdirectories found under {eos_root}")

    out_dir.mkdir(parents=True, exist_ok=True)

    for class_dir in class_dirs:
        class_name = class_dir.name
        files = collect_parquet_files(class_dir)
        if not files:
            print(f"  WARNING: no .parquet files found under {class_dir}, skipping")
            continue

        manifest_path = out_dir / f"{class_name}.txt"
        manifest_path.write_text("\n".join(str(f.resolve()) for f in files) + "\n")
        print(f"  {class_name}: {len(files)} files -> {manifest_path}")

    print(f"\nDone. {len(class_dirs)} class(es) written to {out_dir}/")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--eos-root",
        required=True,
        type=Path,
        help="Root directory whose immediate subdirectories are process classes.",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        type=Path,
        help="Directory where the .txt manifests will be written.",
    )
    args = parser.parse_args()
    make_manifests(args.eos_root, args.out_dir)


if __name__ == "__main__":
    main()
