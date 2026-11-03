#!/usr/bin/env python
"""Measure plain and tokenized ORBIT payload sizes for one trained model."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import lzma
import os
import shlex
import subprocess
import sys
from collections import OrderedDict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VQTORCH_ROOT = PROJECT_ROOT / "vqtorch"
for import_root in (VQTORCH_ROOT, PROJECT_ROOT):
    value = str(import_root)
    if value in sys.path:
        sys.path.remove(value)
    sys.path.insert(0, value)

import numpy as np
import torch
from omegaconf import OmegaConf

from gabbro.data.orbit_parquet import OrbitParquetDataset
from gabbro.utils.orbit_binary import (
    bits_for_size,
    write_packed_ragged_binary,
)
from gabbro.utils.orbit_firmware import (
    ANGLE_LSB,
    ETA_BITS,
    PHI_BITS,
    PID_BITS,
    PT_BITS,
    PT_LSB_GEV,
    PUPPI_COMMON_BITS,
    quantize_puppi_common,
)
from scripts.export_orbit_event_binaries import _collect_class, _load_model, _quantizer_shape


TT_PROCESSES = OrderedDict(
    [
        ("tt_hadronic", "tt0123j_5f_ckm_LO_MLM_hadronic_test.txt"),
        ("tt_leptonic", "tt0123j_5f_ckm_LO_MLM_leptonic_test.txt"),
        ("tt_semileptonic", "tt0123j_5f_ckm_LO_MLM_semiLeptonic_test.txt"),
    ]
)
MIXTURE_GROUPS = OrderedDict(
    [
        (
            "QCD",
            OrderedDict(
                [
                    ("QCD_HT50tobb", "QCD_HT50tobb_test.txt"),
                    ("QCD_HT50toInf", "QCD_HT50toInf_test.txt"),
                ]
            ),
        ),
        ("tt", TT_PROCESSES),
        (
            "VJets",
            OrderedDict(
                [
                    ("WJetsToLNu", "WJetsToLNu_13TeV-madgraphMLM-pythia8_test.txt"),
                    ("WJetsToQQ", "WJetsToQQ_13TeV-madgraphMLM-pythia8_test.txt"),
                    ("DYJetsToLL", "DYJetsToLL_13TeV-madgraphMLM-pythia8_test.txt"),
                    ("ZJetsTobb", "ZJetsTobb_13TeV-madgraphMLM-pythia8_test.txt"),
                    ("ZJetsTocc", "ZJetsTocc_13TeV-madgraphMLM-pythia8_test.txt"),
                    ("ZJetsToQQ", "ZJetsToQQ_13TeV-madgraphMLM-pythia8_test.txt"),
                    ("ZJetsTovv", "ZJetsTovv_13TeV-madgraphMLM-pythia8_test.txt"),
                ]
            ),
        ),
        (
            "VV",
            OrderedDict(
                (name, f"{name}_test.txt")
                for name in (
                    "WW_hadronic",
                    "WW_leptonic",
                    "WW_semileptonic",
                    "WZ_hadronic",
                    "WZ_leptonic",
                    "WZ_semileptonic",
                    "ZZ_hadronic",
                    "ZZ_leptonic",
                    "ZZ_semileptonic",
                )
            ),
        ),
    ]
)
SAMPLE_LABELS = OrderedDict(
    [
        ("minbias", "Minimum bias"),
        ("gghbb", r"$\mathrm{H}\to\mathrm{b\bar{b}}$"),
        ("tt", r"$\mathrm{t\bar{t}}$"),
        ("mixture", "Standard mixture"),
    ]
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--manifest-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--events-per-sample", type=int, default=1000)
    parser.add_argument(
        "--samples",
        nargs="+",
        choices=SAMPLE_LABELS,
        default=["minbias", "gghbb", "tt"],
    )
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-label")
    parser.add_argument("--cmssw-base", type=Path, default=os.environ.get("CMSSW_BASE"))
    parser.add_argument("--skip-containers", action="store_true")
    parser.add_argument("--keep-existing-containers", action="store_true")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


def balanced_allocation(total: int, names) -> OrderedDict[str, int]:
    names = list(names)
    if total < 0 or not names:
        raise ValueError("A non-negative total and at least one name are required")
    quotient, remainder = divmod(total, len(names))
    return OrderedDict(
        (name, quotient + (index < remainder)) for index, name in enumerate(names)
    )


def sample_processes(sample: str, events: int) -> OrderedDict[str, tuple[str, int]]:
    if sample == "minbias":
        return OrderedDict([("minbias", ("minbias.txt", events))])
    if sample == "gghbb":
        return OrderedDict([("ggHbb", ("ggHbb_test.txt", events))])
    if sample == "tt":
        counts = balanced_allocation(events, TT_PROCESSES)
        return OrderedDict((name, (TT_PROCESSES[name], counts[name])) for name in counts)
    if sample != "mixture":
        raise ValueError(f"Unknown sample {sample!r}")
    group_counts = balanced_allocation(events, MIXTURE_GROUPS)
    result = OrderedDict()
    for group, processes in MIXTURE_GROUPS.items():
        counts = balanced_allocation(group_counts[group], processes)
        for process, count in counts.items():
            result[process] = (processes[process], count)
    return result


def build_dataset(cfg, manifest: Path, events: int, batch_size: int):
    if not manifest.is_file():
        raise FileNotFoundError(f"Missing storage benchmark manifest: {manifest}")
    pid_cfg = OmegaConf.to_container(cfg.get("pid") or {}, resolve=True)
    pid_cfg["enabled"] = True
    pid_cfg.setdefault("num_classes", 8)
    return OrbitParquetDataset(
        str(manifest),
        sequence_type=str(cfg.data.get("sequence_type", "particle")),
        max_sequence_length=cfg.data.get("max_sequence_length"),
        batch_size=batch_size,
        shuffle_row_groups=False,
        mask_column=cfg.data.get("mask_column"),
        mask_min_value=cfg.data.get("mask_min_value"),
        min_pt=cfg.data.get("min_pt"),
        event_filter_sequence_type=None,
        event_filter_min_pt=None,
        max_events=events,
        return_raw_features=True,
        pid_cfg=pid_cfg,
        include_energy=bool(cfg.data.get("include_energy", False)),
        energy_shift=float(cfg.data.get("energy_shift", 2.5)),
    )


def collect_sample(
    model,
    cfg,
    manifest_root,
    process_spec,
    batch_size,
    device,
    num_codes,
    num_quantizers,
):
    records = []
    allocation = {}
    for process, (manifest_name, event_count) in process_spec.items():
        if event_count == 0:
            continue
        dataset = build_dataset(cfg, manifest_root / manifest_name, event_count, batch_size)
        originals, pids, tokens = _collect_class(
            model,
            dataset,
            device,
            event_count,
            num_codes,
            progress_description=f"Exporting {process}",
            num_quantizers=num_quantizers,
        )
        if len(pids) != len(originals):
            raise RuntimeError(f"PID data are required for firmware packing ({process})")
        records.extend(zip(originals, pids, tokens))
        allocation[process] = {"manifest": manifest_name, "events": event_count}
    return records, allocation


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def compress_lzma(source: Path) -> Path:
    target = source.with_suffix(source.suffix + ".xz")
    target.write_bytes(lzma.compress(source.read_bytes(), format=lzma.FORMAT_XZ, preset=9))
    return target


def run_cmssw(
    cmssw_base: Path,
    input_path: Path,
    output_path: Path,
    representation: str,
    output_kind: str,
    events: int,
):
    cfg_path = PROJECT_ROOT / "cmssw/OrbitCompression/StorageBenchmark/python/orbitStorage_cfg.py"
    command = " ".join(
        shlex.quote(value)
        for value in (
            "cmsRun",
            str(cfg_path),
            f"inputFiles={input_path}",
            f"outputFile={output_path}",
            f"representation={representation}",
            f"outputKind={output_kind}",
            f"eventCount={events}",
        )
    )
    shell = (
        "source /cvmfs/cms.cern.ch/cmsset_default.sh && "
        f"cd {shlex.quote(str(cmssw_base / 'src'))} && "
        'eval "$(scram runtime -sh)" && '
        + command
    )
    subprocess.run(["bash", "-c", shell], check=True)


def compressed_branch_bytes(path: Path, output_kind: str, representation: str) -> tuple[int, list[str]]:
    import uproot

    with uproot.open(path) as root_file:
        tree = root_file["Events"]
        if output_kind == "nano":
            prefix = "Puppi" if representation == "plain" else "Token"
            selected = [name for name in tree.keys() if name == f"n{prefix}" or name.startswith(f"{prefix}_")]
        else:
            selected = [name for name in tree.keys() if "orbitPayload" in name]
        if not selected:
            raise RuntimeError(f"No {representation} payload branches found in {path}")
        total = sum(int(tree[name].compressed_bytes) for name in selected)
    return total, selected


def ensure_container(
    *,
    cmssw_base: Path,
    input_path: Path,
    output_path: Path,
    representation: str,
    output_kind: str,
    events: int,
    keep_existing: bool,
) -> dict:
    signature_path = output_path.with_suffix(output_path.suffix + ".json")
    expected = {
        "input_sha256": sha256(input_path),
        "representation": representation,
        "output_kind": output_kind,
        "events": events,
        "product_schema": "native_edm_vectors_nano_columns_v3",
        "compression": {"algorithm": "LZMA", "level": 9},
    }
    if keep_existing and output_path.is_file():
        if not signature_path.is_file():
            raise RuntimeError(f"Cannot validate existing container without {signature_path}")
        observed = json.loads(signature_path.read_text())
        if observed != expected:
            raise RuntimeError(
                f"Existing container signature does not match current payload: {output_path}"
            )
    else:
        run_cmssw(
            cmssw_base,
            input_path,
            output_path,
            representation,
            output_kind,
            events,
        )
        signature_path.write_text(json.dumps(expected, indent=2, sort_keys=True) + "\n")
    return expected


def format_row(label, representation, mean_n, measurements, reference):
    row = {"sample": label, "representation": representation, "mean_objects": mean_n}
    for key, value in measurements.items():
        row[f"{key}_bytes_per_event"] = value
        row[f"{key}_percent"] = 100.0 * value / reference
    return row


def latex_table(rows, model_label):
    lines = [
        r"\begin{table}[ht]",
        r"  \centering",
        r"  \begin{tabular}{l r r@{\ }l r@{\ }l r@{\ }l r@{\ }l}",
        r"    \toprule",
        r"     & {$\langle N\rangle$/evt} & \multicolumn{2}{c}{Raw} "
        r"& \multicolumn{2}{c}{Standalone} & \multicolumn{2}{c}{EDM} "
        r"& \multicolumn{2}{c}{nanoAOD} \\",
        r"    \cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}\cmidrule(lr){9-10}",
        r"    Sample & & B & (\%) & B & (\%) & B & (\%) & B & (\%) \\",
        r"    \midrule",
    ]
    previous = None
    for row in rows:
        if previous is not None and row["sample"] != previous:
            lines.append(r"    \addlinespace")
        values = []
        for key in ("raw", "standalone", "edm", "nano"):
            value = row.get(f"{key}_bytes_per_event")
            percent = row.get(f"{key}_percent")
            values.append("-- & --" if value is None else f"{value:.1f} & ({percent:.1f})")
        lines.append(
            f"    {row['sample']}, {row['representation']} & {row['mean_objects']:.1f} & "
            + " & ".join(values)
            + r" \\"
        )
        previous = row["sample"]
    lines.extend(
        [
            r"    \bottomrule",
            r"  \end{tabular}",
            r"  \caption{Average storage cost per event for the plain and tokenized "
            r"representations. The plain Raw payload uses the 40-bit common PUPPI layout "
            r"($p_{\mathrm{T}}$: 14 bits, $\eta$: 12, $\phi$: 11, PID: 3); the tokenized "
            r"Raw payload contains only densely packed VQ indices. Standalone uses LZMA "
            r"level 9. EDM uses native integer vectors and nanoAOD uses aligned flat "
            r"columns: 16-bit pT/$\eta$/$\phi$ codes and an 8-bit PID code for plain "
            r"candidates, and 16-bit token IDs for codebooks of up to 65536 entries. "
            r"They are written through \texttt{PoolOutputModule} and "
            r"\texttt{NanoAODOutputModule}, respectively. Parentheses give percentages of the "
            r"corresponding sample's plain Raw size. Model: "
            + model_label.replace("_", r"\_")
            + ".}",
            r"  \label{tab:compression}",
            r"\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    if args.events_per_sample < 1:
        raise ValueError("--events-per-sample must be positive")
    run_dir = args.run_dir.resolve()
    config_path = run_dir / "config_resolved.yaml"
    checkpoint = run_dir / "checkpoints/best.ckpt"
    if not config_path.is_file() or not checkpoint.is_file():
        raise FileNotFoundError(f"Expected config_resolved.yaml and checkpoints/best.ckpt below {run_dir}")
    cfg = OmegaConf.load(config_path)
    num_codes, num_quantizers = _quantizer_shape(cfg)
    token_bits = bits_for_size(num_codes)
    configured_length = cfg.data.get("max_sequence_length")
    default_length = 500 if cfg.data.get("sequence_type") == "particle_full" else 128
    effective_length = int(configured_length or default_length)
    batch_size = args.batch_size or (16 if effective_length > 128 else 256)
    device = torch.device(args.device)
    model = _load_model(cfg, checkpoint, device)
    output_dir = args.output_dir.resolve()
    binary_dir = output_dir / "payloads"
    container_dir = output_dir / "containers"
    binary_dir.mkdir(parents=True, exist_ok=True)
    container_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    manifest = {
        "format": "orbit-storage-benchmark",
        "version": 1,
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint),
        "manifest_root": str(args.manifest_root.resolve()),
        "events_per_sample": args.events_per_sample,
        "num_codes": num_codes,
        "num_quantizers": num_quantizers,
        "token_bits": token_bits,
        "plain_layout": {
            "bits_per_candidate": PUPPI_COMMON_BITS,
            "rounding": "nearest",
            "overflow": "saturate",
            "fields": {
                "pt": {"bits": PT_BITS, "offset": 0, "lsb_gev": PT_LSB_GEV},
                "eta": {"bits": ETA_BITS, "offset": PT_BITS, "lsb": ANGLE_LSB},
                "phi": {
                    "bits": PHI_BITS,
                    "offset": PT_BITS + ETA_BITS,
                    "lsb": ANGLE_LSB,
                },
                "pid": {
                    "bits": PID_BITS,
                    "offset": PT_BITS + ETA_BITS + PHI_BITS,
                },
            },
        },
        "standalone_compression": {"format": "xz", "algorithm": "LZMA", "level": 9},
        "cmssw_base": None if args.skip_containers else str(args.cmssw_base),
        "selection": "checkpoint particle mask/order/cap; event filters disabled",
        "samples": {},
    }
    rows = []
    for sample in args.samples:
        spec = sample_processes(sample, args.events_per_sample)
        records, allocation = collect_sample(
            model,
            cfg,
            args.manifest_root.resolve(),
            spec,
            batch_size,
            device,
            num_codes,
            num_quantizers,
        )
        rng.shuffle(records)
        originals, pids, tokens = map(list, zip(*records))
        plain_words = [quantize_puppi_common(features, pid) for features, pid in zip(originals, pids)]
        plain_path = binary_dir / f"{sample}_plain_puppi40.bin"
        token_path = binary_dir / f"{sample}_tokens_packed.bin"
        common_metadata = {"sample": sample, "checkpoint": str(checkpoint)}
        write_packed_ragged_binary(
            plain_path,
            plain_words,
            bits_per_value=PUPPI_COMMON_BITS,
            metadata={
                **common_metadata,
                "kind": "puppi_common_40bit",
                "fields": [
                    {"name": "pt", "bits": PT_BITS, "offset": 0, "lsb_gev": PT_LSB_GEV},
                    {"name": "eta", "bits": ETA_BITS, "offset": PT_BITS, "lsb": ANGLE_LSB},
                    {
                        "name": "phi",
                        "bits": PHI_BITS,
                        "offset": PT_BITS + ETA_BITS,
                        "lsb": ANGLE_LSB,
                    },
                    {
                        "name": "pid",
                        "bits": PID_BITS,
                        "offset": PT_BITS + ETA_BITS + PHI_BITS,
                    },
                ],
                "rounding": "nearest",
                "overflow": "saturate",
            },
        )
        write_packed_ragged_binary(
            token_path,
            tokens,
            bits_per_value=token_bits,
            metadata={
                **common_metadata,
                "kind": "vq_token_ids",
                "num_codes": num_codes,
                "num_quantizers": num_quantizers,
                "token_order": "latent-major, then residual stage",
            },
        )
        plain_xz = compress_lzma(plain_path)
        token_xz = compress_lzma(token_path)
        event_count = len(records)
        sample_metrics = {
            "plain": {
                "raw": plain_path.stat().st_size / event_count,
                "standalone": plain_xz.stat().st_size / event_count,
            },
            "tokenized": {
                "raw": token_path.stat().st_size / event_count,
                "standalone": token_xz.stat().st_size / event_count,
            },
        }
        container_meta = {}
        if not args.skip_containers:
            if args.cmssw_base is None:
                raise ValueError("--cmssw-base or CMSSW_BASE is required unless --skip-containers is used")
            for representation, input_path in (("plain", plain_path), ("tokenized", token_path)):
                container_meta[representation] = {}
                for output_kind in ("edm", "nano"):
                    output_path = container_dir / f"{sample}_{representation}_{output_kind}.root"
                    signature = ensure_container(
                        cmssw_base=args.cmssw_base.resolve(),
                        input_path=input_path,
                        output_path=output_path,
                        representation=representation,
                        output_kind=output_kind,
                        events=event_count,
                        keep_existing=args.keep_existing_containers,
                    )
                    byte_count, branches = compressed_branch_bytes(
                        output_path, output_kind, representation
                    )
                    sample_metrics[representation][output_kind] = byte_count / event_count
                    container_meta[representation][output_kind] = {
                        "path": str(output_path), "bytes": output_path.stat().st_size,
                        "payload_compressed_bytes": byte_count, "branches": branches,
                        "signature": signature,
                    }
        reference = sample_metrics["plain"]["raw"]
        mean_particles = sum(map(len, plain_words)) / event_count
        mean_tokens = sum(map(len, tokens)) / event_count
        label = SAMPLE_LABELS[sample]
        rows.append(format_row(label, "plain", mean_particles, sample_metrics["plain"], reference))
        rows.append(format_row(label, "tokenized", mean_tokens, sample_metrics["tokenized"], reference))
        manifest["samples"][sample] = {
            "allocation": allocation,
            "mean_particles": mean_particles,
            "mean_tokens": mean_tokens,
            "plain": {
                "path": str(plain_path),
                "bytes": plain_path.stat().st_size,
                "sha256": sha256(plain_path),
                "xz_path": str(plain_xz),
                "xz_bytes": plain_xz.stat().st_size,
                "xz_sha256": sha256(plain_xz),
            },
            "tokens": {
                "path": str(token_path),
                "bytes": token_path.stat().st_size,
                "sha256": sha256(token_path),
                "xz_path": str(token_xz),
                "xz_bytes": token_xz.stat().st_size,
                "xz_sha256": sha256(token_xz),
            },
            "metrics": sample_metrics,
            "containers": container_meta,
        }

    model_label = args.model_label or run_dir.name
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "compression_measurements.json").write_text(
        json.dumps({**manifest, "rows": rows}, indent=2, sort_keys=True) + "\n"
    )
    fieldnames = [
        "sample", "representation", "mean_objects",
        *[
            f"{kind}_{suffix}"
            for kind in ("raw", "standalone", "edm", "nano")
            for suffix in ("bytes_per_event", "percent")
        ],
    ]
    with (output_dir / "compression_measurements.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "compression_table.tex").write_text(latex_table(rows, model_label))
    print(f"Wrote storage benchmark to {output_dir}")


if __name__ == "__main__":
    main()
