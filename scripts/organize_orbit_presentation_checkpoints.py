#!/usr/bin/env python3
"""Index the exact model runs used by the legacy presentation plots."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


DEFAULT_PRESENTATION_DIR = Path(
    "/eos/user/y/yelberke/enhancing_ntp4jets_runs/presentation_plots"
)
DEFAULT_OUTPUT_DIR = Path(
    "/eos/user/y/yelberke/enhancing_ntp4jets_runs/writeup_plots/"
    "presentation_checkpoints"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--presentation-dir", type=Path, default=DEFAULT_PRESENTATION_DIR
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def read_records(path: Path) -> list[dict]:
    records = json.loads(path.read_text())
    if not isinstance(records, list):
        raise TypeError(f"Expected a list in {path}")
    return records


def slug(value: object) -> str:
    result = re.sub(r"[^a-z0-9]+", "_", str(value).lower()).strip("_")
    return result or "unknown"


def run_config(run_dir: Path) -> Path:
    for relative in ("config_resolved.yaml", ".hydra/config.yaml", "config.yaml"):
        candidate = run_dir / relative
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"No run config below {run_dir}")


def safe_symlink(link: Path, target: Path) -> None:
    if not target.exists():
        raise FileNotFoundError(target)
    target = target.resolve()
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.is_symlink() and link.resolve() == target:
        return
    if link.exists() or link.is_symlink():
        raise FileExistsError(f"Refusing to replace {link}")
    link.symlink_to(target)


def add_sources(
    selected: dict[str, dict],
    records: list[dict],
    *,
    training_sample: str,
    selection: str,
    primary_manifest: Path,
    family_filter: str | None = None,
) -> None:
    for record in records:
        if family_filter is not None and record.get("plot_family") != family_filter:
            continue
        run_dir = str(Path(record["run_dir"]).resolve())
        selected.setdefault(
            run_dir,
            {
                "record": record,
                "training_sample": training_sample,
                "selection": selection,
                "primary_manifest": primary_manifest,
            },
        )


def all_manifest_uses(root: Path) -> dict[str, list[str]]:
    uses: dict[str, set[str]] = defaultdict(set)
    for path in sorted(root.rglob("manifest.json")):
        try:
            records = read_records(path)
        except (json.JSONDecodeError, TypeError):
            continue
        relative = str(path.relative_to(root))
        for record in records:
            if record.get("run_dir"):
                uses[str(Path(record["run_dir"]).resolve())].add(relative)
    return {run_dir: sorted(paths) for run_dir, paths in uses.items()}


def main() -> None:
    args = parse_args()
    presentation_dir = args.presentation_dir.resolve()
    output_dir = args.output_dir

    tt_manifest = (
        presentation_dir
        / "canonical_tt_golden/golden/full/training_like/manifest.json"
    )
    full500_manifest = (
        presentation_dir
        / "canonical_tt_full500_golden/training_like/manifest.json"
    )
    comparison_manifest = presentation_dir / "tt_vs_mixture_vq_ste/tt/manifest.json"
    for path in (tt_manifest, full500_manifest, comparison_manifest):
        if not path.is_file():
            raise FileNotFoundError(path)

    selected: dict[str, dict] = {}
    add_sources(
        selected,
        read_records(tt_manifest),
        training_sample="ttbar",
        selection="puppiw_filtered_128",
        primary_manifest=tt_manifest,
    )
    add_sources(
        selected,
        read_records(full500_manifest),
        training_sample="ttbar",
        selection="unfiltered_500",
        primary_manifest=full500_manifest,
    )
    comparison_records = read_records(comparison_manifest)
    add_sources(
        selected,
        comparison_records,
        training_sample="sm_mixture",
        selection="puppiw_filtered_128",
        primary_manifest=comparison_manifest,
        family_filter="mixture-trained VQ-STE",
    )
    # Seven tt comparison checkpoints are shared with the golden scan. The repaired
    # 256-code source differs, so setdefault adds only that extra source run.
    add_sources(
        selected,
        comparison_records,
        training_sample="ttbar",
        selection="puppiw_filtered_128",
        primary_manifest=comparison_manifest,
        family_filter="tt-trained VQ-STE",
    )

    uses = all_manifest_uses(presentation_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    entries: list[dict] = []
    occupied: set[Path] = set()

    ordered = sorted(
        selected.items(),
        key=lambda item: (
            item[1]["training_sample"],
            item[1]["selection"],
            str(item[1]["record"].get("plot_family")),
            int(item[1]["record"].get("total_codebook_size") or 0),
            item[0],
        ),
    )
    for run_dir_text, source in ordered:
        record = source["record"]
        run_dir = Path(run_dir_text)
        family = slug(record.get("plot_family") or "unknown")
        codes = int(record.get("total_codebook_size") or 0)
        relative_dir = (
            Path(source["training_sample"])
            / source["selection"]
            / family
            / str(codes)
        )
        if relative_dir in occupied:
            relative_dir = relative_dir.with_name(
                f"{relative_dir.name}__{slug(record.get('label') or run_dir.name)}"
            )
        occupied.add(relative_dir)
        model_dir = output_dir / relative_dir

        config = run_config(run_dir)
        checkpoint = run_dir / "checkpoints/best.ckpt"
        safe_symlink(model_dir / "config.yaml", config)
        safe_symlink(model_dir / "source_run", run_dir)
        if checkpoint.is_file():
            safe_symlink(model_dir / "best.ckpt", checkpoint)
            state = "checkpoint"
            artifact = str((model_dir / "best.ckpt").relative_to(output_dir))
            artifact_target = str(checkpoint.resolve())
        else:
            state = "configuration_only"
            artifact = None
            artifact_target = None

        entries.append(
            {
                "path": str(relative_dir),
                "training_sample": source["training_sample"],
                "selection": source["selection"],
                "plot_family": record.get("plot_family"),
                "label": record.get("label"),
                "total_codebook_size": codes,
                "state": state,
                "checkpoint": artifact,
                "checkpoint_target": artifact_target,
                "config": str((model_dir / "config.yaml").relative_to(output_dir)),
                "config_target": str(config),
                "source_run": str(run_dir),
                "primary_manifest": str(
                    source["primary_manifest"].relative_to(presentation_dir)
                ),
                "used_by_manifests": uses.get(run_dir_text, []),
            }
        )

    states = Counter(entry["state"] for entry in entries)
    groups = Counter(
        (entry["training_sample"], entry["selection"]) for entry in entries
    )
    manifest = {
        "format": "symlink-index",
        "source_collection": str(presentation_dir),
        "note": (
            "Exact source models used by the legacy presentation_plots collection. "
            "Links do not duplicate checkpoint bytes. Historical FAISS runs did not "
            "persist a fitted checkpoint and are retained as configuration-only entries."
        ),
        "counts": {
            "source_runs": len(entries),
            "checkpoints": states["checkpoint"],
            "configuration_only": states["configuration_only"],
        },
        "models": entries,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    lines = [
        "# Presentation golden-plot checkpoints",
        "",
        "This is a symlink index of the exact historical model runs used by the",
        "plots under `presentation_plots/`. No checkpoint bytes are duplicated.",
        "",
        f"- Unique source runs: {len(entries)}",
        f"- Neural checkpoints (`best.ckpt`): {states['checkpoint']}",
        f"- Historical FAISS configuration-only entries: {states['configuration_only']}",
        "",
        "## Organization",
        "",
    ]
    for (training, selection), count in sorted(groups.items()):
        lines.append(f"- `{training}/{selection}`: {count} source runs")
    lines += [
        "",
        "Each model directory contains `config.yaml` and a `source_run` symlink.",
        "Neural models additionally contain `best.ckpt`. Consult `manifest.json` for",
        "the original label and every presentation manifest which used the model.",
        "",
        "The FAISS entries are included for provenance, but their historical run",
        "directories contain no persisted fitted codebook or Lightning checkpoint.",
    ]
    (output_dir / "README.md").write_text("\n".join(lines) + "\n")

    print(f"Indexed {len(entries)} presentation source runs in {output_dir}")
    print(f"  checkpoints: {states['checkpoint']}")
    print(f"  configuration-only: {states['configuration_only']}")


if __name__ == "__main__":
    main()
