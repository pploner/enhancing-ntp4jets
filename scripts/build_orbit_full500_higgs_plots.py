"""Build full-500 canonical-tt Higgs presentation panels from candidate caches."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_orbit_presentation_plots import (
    HIGGS_MASS_EXTENDED_PLOT_BINS,
    HIGGS_MASS_LOG_PLOT_BINS,
    load_higgs_records,
    plot_mass_response,
    plot_mu_sigma,
    plot_near_histogram,
    plot_near_observable_histogram,
    plot_single_mass_histogram,
    plot_single_observable_histogram,
    select_plot_families,
    write_higgs_summary,
)
from scripts.compare_orbit_tt_selections import configuration_key


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-manifest", type=Path, required=True)
    parser.add_argument("--near-template", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-replicas", type=int, default=50)
    args = parser.parse_args()

    full = json.loads(args.full_manifest.read_text())
    near_template = json.loads(args.near_template.read_text())
    full_by_key = {row["label"]: row for row in full}
    near = []
    for template in near_template:
        key = configuration_key(template["label"])
        if key not in full_by_key:
            raise ValueError(f"Near-4096 configuration absent from full-500 manifest: {key}")
        near.append({**full_by_key[key], "plot_family": template["plot_family"]})

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    records, reference_values, reference_fit = load_higgs_records(
        full, args.cache_dir, args.bootstrap_replicas
    )
    plot_mu_sigma(records, reference_fit, output / "higgs_mass_mu_vs_sigma.png")
    plot_mass_response(
        records,
        reference_fit,
        output / "higgs_mass_response_vs_compression_ratio.png",
        40.0,
    )
    faiss_vq = select_plot_families(records, {"FAISS k-means", "VQ STE"})
    plot_mu_sigma(
        faiss_vq, reference_fit, output / "higgs_mass_mu_vs_sigma_faiss_vq_ste.png"
    )
    plot_mass_response(
        faiss_vq,
        reference_fit,
        output / "higgs_mass_response_vs_compression_ratio_faiss_vq_ste.png",
        40.0,
    )
    plot_near_histogram(
        near, full, args.cache_dir, reference_values, args.bootstrap_replicas,
        output / "higgs_mass_near_4096.png",
    )
    plot_near_histogram(
        near, full, args.cache_dir, reference_values, args.bootstrap_replicas,
        output / "higgs_mass_near_4096_extended_500.png",
        bins=HIGGS_MASS_EXTENDED_PLOT_BINS,
    )
    plot_near_histogram(
        near, full, args.cache_dir, reference_values, args.bootstrap_replicas,
        output / "higgs_mass_near_4096_log_1000.png",
        bins=HIGGS_MASS_LOG_PLOT_BINS,
        log_y=True,
    )
    for observable, xlabel in (
        ("leading_particle_pt", r"Leading particle $p_{\mathrm{T}}$ [GeV]"),
        ("n_jets", "Number of AK4 jets"),
    ):
        plot_near_observable_histogram(
            near, full, args.cache_dir, observable, xlabel,
            output / f"{observable}_near_4096.png",
        )

    vq_index = next(
        index for index, row in enumerate(full) if row["label"] == "vq_ste_codes_4096"
    )
    vq_cache = args.cache_dir / f"{vq_index:03d}" / "higgs_candidates.csv"
    individual = output.parent / "single_models" / "vq_ste_4096_tt_full500"
    plot_single_mass_histogram(
        vq_cache, args.bootstrap_replicas, individual / "higgs_mass_resolved.png"
    )
    plot_single_mass_histogram(
        vq_cache, args.bootstrap_replicas,
        individual / "higgs_mass_resolved_extended_500.png",
        bins=HIGGS_MASS_EXTENDED_PLOT_BINS,
    )
    plot_single_mass_histogram(
        vq_cache, args.bootstrap_replicas,
        individual / "higgs_mass_resolved_log_1000.png",
        bins=HIGGS_MASS_LOG_PLOT_BINS,
        log_y=True,
    )
    for observable, xlabel in (
        ("leading_particle_pt", r"Leading particle $p_{\mathrm{T}}$ [GeV]"),
        ("n_jets", "Number of AK4 jets"),
    ):
        plot_single_observable_histogram(
            vq_cache, observable, xlabel, output / f"{observable}_vq_ste_4096.png"
        )
    write_higgs_summary(records, reference_fit, output)
    print(f"Wrote full-500 Higgs presentation panels for {len(records)} models to {output}")


if __name__ == "__main__":
    main()
