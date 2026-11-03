#!/usr/bin/env python
"""Render one model's cached downstream physics plots without rerunning inference."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gabbro.plotting.writeup.summary import (
    plot_higgs,
    plot_jet_residuals,
    plot_met,
    plot_pid_residuals,
    plot_tau32,
    save_figure,
    validate_downstream_run,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = validate_downstream_run(args.run_dir.resolve())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    products: list[Path] = []

    scalar_plots = (
        (
            "higgs_njets_and_mass",
            args.cache_dir / "higgs/higgs_candidates.csv",
            plot_higgs,
        ),
        (
            "tau32_njets_and_shape",
            args.cache_dir / "tau32/tt_hadronic_ak8_pt450_tau32_values.npz",
            plot_tau32,
        ),
        (
            "missing_transverse_momentum",
            args.cache_dir / "met/sm_mixture_met_histograms.npz",
            plot_met,
        ),
    )
    for name, cache, renderer in scalar_plots:
        if not cache.is_file():
            raise FileNotFoundError(cache)
        figure, values = renderer(cache)
        products.extend(save_figure(figure, args.output_dir / name))
        values_path = args.output_dir / f"{name}_values.npz"
        np.savez_compressed(
            values_path,
            **{key: value for key, value in values.items() if isinstance(value, np.ndarray)},
        )
        products.append(values_path)

    jet_cache = args.cache_dir / "jets/sm_mixture_jet_residuals.npz"
    if not jet_cache.is_file():
        raise FileNotFoundError(jet_cache)
    copied_jets = args.output_dir / "jet_residual_values.npz"
    shutil.copy2(jet_cache, copied_jets)
    products.append(copied_jets)
    for prefix, name in (("all", "jet_residuals_all"), ("leading5", "jet_residuals_leading5")):
        figure, _ = plot_jet_residuals(jet_cache, prefix)
        products.extend(save_figure(figure, args.output_dir / name))

    pid_cache = args.cache_dir / "sm_pid/pid_conditional_pull_values.npz"
    if not pid_cache.is_file():
        raise FileNotFoundError(pid_cache)
    products.extend(
        save_figure(plot_pid_residuals(pid_cache), args.output_dir / "pid_residuals")
    )
    copied_pid = args.output_dir / "pid_residual_values.npz"
    shutil.copy2(pid_cache, copied_pid)
    products.append(copied_pid)

    provenance = {
        **model,
        "test_events_per_plot": 10000,
        "cache_dir": str(args.cache_dir.resolve()),
    }
    provenance_path = args.output_dir / "provenance.json"
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")
    products.append(provenance_path)
    readme = [
        "# SM-trained VQ-STE downstream plots",
        "",
        "All figures use 10,000 events and are provided as PDF and 300-dpi PNG.",
        "The exact training run is recorded in `provenance.json`.",
        "",
    ]
    readme.extend(f"- [{path.name}]({path.name})" for path in products)
    (args.output_dir / "README.md").write_text("\n".join(readme) + "\n")
    print(f"Wrote {len(products)} products to {args.output_dir}")


if __name__ == "__main__":
    main()
