#!/usr/bin/env python3
"""Tabulate the hardware PID composition of a deterministic parquet sample."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import shlex
import struct
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow
import pyarrow.parquet as pq


PID_ROWS = (
    (0, "neutral_hadron", r"$h^0$", "neutral hadron"),
    (1, "photon", r"$\gamma$", "photon"),
    (2, "negative_hadron", r"$h^-$", "negative hadron"),
    (3, "positive_hadron", r"$h^+$", "positive hadron"),
    (4, "electron", r"$e^-$", "electron"),
    (5, "positron", r"$e^+$", "positron"),
    (6, "muon", r"$\mu^-$", "muon"),
    (7, "antimuon", r"$\mu^+$", "antimuon"),
)
AGGREGATES = (
    ("hadrons_total", "Hadrons (total)", (0, 2, 3)),
    ("leptons_total", "Leptons (total)", (4, 5, 6, 7)),
)
PDG_COLUMN = "L1T_PUPPIPart_PID"
CHARGE_COLUMN = "L1T_PUPPIPart_Charge"
PUPPIW_COLUMN = "L1T_PUPPIPart_PuppiW"
PT_COLUMN = "L1T_PUPPIPart_PT"
COLUMNS = (PDG_COLUMN, CHARGE_COLUMN, PUPPIW_COLUMN, PT_COLUMN)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--events", type=int, default=1_000)
    parser.add_argument("--puppiw-min", type=float, default=0.05)
    parser.add_argument("--max-particles", type=int, default=128)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def manifest_files(path: Path) -> list[Path]:
    files = []
    for line in path.read_text().splitlines():
        value = line.strip()
        if not value or value.startswith("#"):
            continue
        candidate = Path(value).expanduser()
        files.append(candidate if candidate.is_absolute() else path.parent / candidate)
    if not files:
        raise ValueError(f"Manifest contains no files: {path}")
    return files


def pid_classes(pdg_id: np.ndarray, charge: np.ndarray) -> np.ndarray:
    """Match gabbro.data.orbit_parquet.map_pdg_charge_to_pid_class."""
    classes = np.where(charge < 0, 2, np.where(charge > 0, 3, 0))
    classes = np.where(np.abs(pdg_id) == 22, 1, classes)
    classes = np.where(pdg_id == 11, 4, classes)
    classes = np.where(pdg_id == -11, 5, classes)
    classes = np.where(pdg_id == 13, 6, classes)
    return np.where(pdg_id == -13, 7, classes).astype(np.int8)


def git_metadata(repo: Path) -> dict:
    def run(*args: str) -> str:
        return subprocess.run(
            args,
            cwd=repo,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
        ).stdout.strip()

    return {
        "commit": run("git", "rev-parse", "HEAD"),
        "dirty": bool(run("git", "status", "--porcelain")),
    }


def table_rows(counts: Counter, total: int) -> list[dict]:
    rows = []
    for value, key, latex_label, description in PID_ROWS:
        count = int(counts[value])
        rows.append(
            {
                "key": key,
                "pid_value": value,
                "constituent_pid_values": [value],
                "latex_label": latex_label,
                "description": description,
                "count": count,
                "percentage": 100.0 * count / total,
            }
        )
    for key, label, values in AGGREGATES:
        count = sum(int(counts[value]) for value in values)
        rows.append(
            {
                "key": key,
                "pid_value": None,
                "latex_label": label,
                "description": "+".join(str(value) for value in values),
                "constituent_pid_values": list(values),
                "count": count,
                "percentage": 100.0 * count / total,
            }
        )
    return rows


def write_latex(
    path: Path,
    rows: list[dict],
    events: int,
    particles: int,
    puppiw_min: float,
    max_particles: int,
) -> None:
    lines = [
        r"\begin{table}[ht]",
        r"  \centering",
        r"  \begin{tabular}{l r r r}",
        r"    \toprule",
        r"    PID category & Hardware value & Count & Fraction [\%] \\",
        r"    \midrule",
    ]
    for index, row in enumerate(rows):
        if index == len(PID_ROWS):
            lines.append(r"    \addlinespace")
        value = "--" if row["pid_value"] is None else str(row["pid_value"])
        lines.append(
            f"    {row['latex_label']} & {value} & {row['count']:,} & "
            f"{row['percentage']:.4f} \\\\"
        )
    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        (
            r"  \caption{PID composition of the first "
            rf"{events:,} events in the ordered leptonic-$ZZ$ test manifest. "
            r"Candidates satisfy $p_T>0$ and "
            rf"$w_{{\mathrm{{PUPPI}}}}>{puppiw_min:g}$; the first "
            f"{max_particles} surviving candidates in stored order are retained per event. "
            r"Percentages "
            f"use all {particles:,} retained candidates as their denominator. Hadrons are "
            r"$h^0+h^-+h^+$ and leptons are $e^-+e^++\mu^-+\mu^+$.}"
        ),
        r"  \label{tab:zz-leptonic-pid-composition}",
        r"\end{table}",
        "",
    ]
    path.write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    if args.events < 1 or args.max_particles < 1:
        raise ValueError("--events and --max-particles must be positive")
    manifest = args.manifest.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    files = manifest_files(manifest)
    repo = Path(__file__).resolve().parents[1]

    counts: Counter = Counter()
    event_rows = []
    source_files = []
    sample_digest = hashlib.sha256()
    sampled_events = 0
    columns = list(COLUMNS)
    for source_path in files:
        parquet = pq.ParquetFile(source_path)
        missing = set(columns) - set(parquet.schema_arrow.names)
        if missing:
            raise ValueError(f"{source_path} is missing columns: {sorted(missing)}")
        stat = source_path.stat()
        source_record = {
            "path": str(source_path),
            "size_bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "parquet_rows": parquet.metadata.num_rows,
            "sampled_rows": [],
        }
        source_row = 0
        sampled_start = sampled_events
        for batch in parquet.iter_batches(columns=columns, batch_size=2_048):
            arrays = [batch.column(name).to_pylist() for name in columns]
            for pdg_values, charge_values, puppiw_values, pt_values in zip(*arrays):
                pdg = np.asarray(pdg_values, dtype=np.int64)
                charge = np.asarray(charge_values, dtype=np.float64)
                puppiw = np.asarray(puppiw_values, dtype=np.float64)
                pt = np.asarray(pt_values, dtype=np.float64)
                classes = pid_classes(pdg, charge)
                physical = np.isfinite(pt) & (pt > 0.0)
                selected_mask = physical & np.isfinite(puppiw) & (puppiw > args.puppiw_min)
                selected = classes[selected_mask]
                selected_pretruncate = len(selected)
                selected = selected[: args.max_particles]
                event_counts = np.bincount(selected, minlength=8)
                counts.update(selected.tolist())

                sample_digest.update(str(source_path).encode())
                sample_digest.update(struct.pack("<qq", source_row, len(selected)))
                sample_digest.update(selected.tobytes())
                event_rows.append(
                    {
                        "event_index": sampled_events,
                        "source_file": str(source_path),
                        "source_row": source_row,
                        "raw_candidates": len(pt),
                        "physical_candidates": int(np.count_nonzero(physical)),
                        "selected_before_truncation": selected_pretruncate,
                        "selected_candidates": len(selected),
                        **{f"pid_{value}_count": int(event_counts[value]) for value in range(8)},
                    }
                )
                sampled_events += 1
                source_row += 1
                if sampled_events == args.events:
                    break
            if sampled_events == args.events:
                break
        if sampled_events > sampled_start:
            source_record["sampled_rows"] = [0, sampled_events - sampled_start - 1]
            source_record["sampled_event_count"] = sampled_events - sampled_start
            source_files.append(source_record)
        if sampled_events == args.events:
            break
    if sampled_events != args.events:
        raise RuntimeError(f"Requested {args.events} events, found {sampled_events}")

    total = sum(counts.values())
    if total == 0:
        raise RuntimeError("The selection retained no candidates")
    rows = table_rows(counts, total)
    audit_csv = output_dir / "sampled_events.csv"
    with audit_csv.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(event_rows[0]))
        writer.writeheader()
        writer.writerows(event_rows)
    with (output_dir / "pid_composition.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_latex(
        output_dir / "pid_composition.tex",
        rows,
        sampled_events,
        total,
        args.puppiw_min,
        args.max_particles,
    )

    command = [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]]
    (output_dir / "reproduce_command.txt").write_text(
        " ".join(shlex.quote(value) for value in command) + "\n"
    )
    mapping_source = repo / "gabbro/data/orbit_parquet.py"
    audit_paths = {
        "event_counts": audit_csv,
        "table_csv": output_dir / "pid_composition.csv",
        "latex": output_dir / "pid_composition.tex",
        "reproduce_command": output_dir / "reproduce_command.txt",
    }
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "sample_definition": "deterministic prefix of the ordered manifest",
        "manifest": {"path": str(manifest), "sha256": sha256(manifest)},
        "source_files_used": source_files,
        "events": sampled_events,
        "selection": {
            "physical_candidate": f"finite {PT_COLUMN} and {PT_COLUMN} > 0",
            "mask": f"finite {PUPPIW_COLUMN} and {PUPPIW_COLUMN} > {args.puppiw_min}",
            "truncation": f"first {args.max_particles} selected candidates in stored order",
            "denominator": "all retained candidates across the sampled events",
        },
        "columns": columns,
        "pid_mapping": {
            "implementation": "gabbro.data.orbit_parquet.map_pdg_charge_to_pid_class",
            "source_path": str(mapping_source),
            "source_sha256": sha256(mapping_source),
            "classes": [
                {"value": value, "key": key, "description": description}
                for value, key, _, description in PID_ROWS
            ],
        },
        "retained_particles": total,
        "mean_retained_particles_per_event": total / sampled_events,
        "sample_fingerprint_sha256": sample_digest.hexdigest(),
        "rows": rows,
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pyarrow": pyarrow.__version__,
            "git": git_metadata(repo),
            "script": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256(Path(__file__).resolve()),
            },
        },
        "audit_files": {
            key: {"path": path.name, "sha256": sha256(path)}
            for key, path in audit_paths.items()
        },
    }
    (output_dir / "provenance.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"events": sampled_events, "particles": total, "rows": rows}, indent=2))
    print(f"Wrote reproducible PID table to {output_dir}")


if __name__ == "__main__":
    main()
