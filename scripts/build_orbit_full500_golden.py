"""Build presentation-ready canonical-tt full-500 plots from existing tests.

The source records come from the established full-vs-PUPPI comparison. Only
full_top500 entries are used; absent tests are reported rather than silently
replaced with filtered or differently trained models.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.collect_orbit_multirun import (
    _add_multirun_metric_aliases,
    _collect_record,
    _load_config,
    _save_figures,
    _select,
    _write_outputs,
)
from scripts.compare_orbit_tt_selections import configuration_key
from gabbro.plotting.orbit import multirun_color


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-records", type=Path, required=True)
    parser.add_argument("--filtered-golden-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    selections = json.loads(args.selection_records.read_text())
    full = [row for row in selections if row.get("selection") == "full_top500"]
    if not full:
        raise ValueError("Selection records contain no full_top500 runs")

    filtered = json.loads(args.filtered_golden_manifest.read_text())
    families = {}
    for row in filtered:
        key = configuration_key(str(row.get("label", "")))
        if key:
            families[key] = row.get("plot_family") or row["label"]

    configs = [row["configuration"] for row in full]
    duplicates = [key for key, count in Counter(configs).items() if count != 1]
    if duplicates:
        raise ValueError(f"Duplicate full-500 configurations: {duplicates}")

    for suite, group, output_name in (
        ("training_like", "all", "training_like"),
        ("tt_vs_gghbb", "ggHbb", "ggHbb"),
    ):
        records = []
        histogram_runs = []
        missing = []
        for row in full:
            key = row["configuration"]
            run_dir = Path(row["run_dir"])
            cfg = _load_config(run_dir / ".hydra" / "config.yaml")
            if (
                _select(cfg, "data.sequence_type") != "particle_full"
                or int(_select(cfg, "data.max_sequence_length", 0)) != 500
            ):
                raise ValueError(f"Not a full-500 run: {run_dir}")
            family = families.get(key)
            if family is None:
                raise ValueError(f"Golden-set family is unknown for {key}")
            record, histograms = _collect_record(
                run_dir,
                "test",
                group,
                suite,
                label_override=key,
                family_override=family,
            )
            if histograms is None or record["metrics_path"] is None:
                missing.append(key)
                continue
            records.append(record)
            histogram_runs.append(
                {
                    "label": key,
                    "plot_color": multirun_color(family),
                    "histograms": histograms,
                }
            )

        if not records:
            raise ValueError(f"No complete test artifacts for {suite}/{group}")
        _add_multirun_metric_aliases(records)
        destination = args.output_dir / output_name
        _write_outputs(records, destination)
        _save_figures(
            records,
            histogram_runs,
            destination,
            original_bits_per_input_particle=43.0,
            compression_reference_bits=40.0,
            show_titles=False,
        )
        print(
            f"{destination}: plotted {len(records)} full-500 runs; "
            f"missing artifacts for {missing}",
            flush=True,
        )


if __name__ == "__main__":
    main()
