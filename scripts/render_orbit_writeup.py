#!/usr/bin/env python
"""Render the complete ORBIT writeup plot collection from durable caches.

Commands:
  summary           Core golden, ablation, and single-model downstream plots.
  model-comparison  Joint eight-model 5M comparison plots.
  training-scale    Matched 200k-versus-5M VQ-STE comparison plots.
  all               Run all configured renderers in dependency order.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Callable

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gabbro.plotting.writeup import model_comparison, summary, training_scale


COMMANDS: dict[str, tuple[str, Callable[[list[str] | None], None]]] = {
    "summary": (
        "Render the core golden, ablation, and single-model downstream collection.",
        summary.main,
    ),
    "model-comparison": (
        "Render the joint eight-model 5M comparison collection.",
        model_comparison.main,
    ),
    "training-scale": (
        "Render one matched 200k-versus-5M VQ-STE comparison.",
        training_scale.main,
    ),
}


def _root_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        nargs="?",
        choices=(*COMMANDS, "all"),
        help="Plot collection to render.",
    )
    return parser


def _expand_environment(value: Any) -> Any:
    if isinstance(value, str):
        expanded = os.path.expanduser(os.path.expandvars(value))
        if "$" in expanded:
            raise ValueError(f"Unresolved environment variable in {value!r}")
        return expanded
    if isinstance(value, list):
        return [_expand_environment(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand_environment(item) for key, item in value.items()}
    return value


def config_to_argv(config: dict[str, Any]) -> list[str]:
    """Convert a YAML command mapping to its ordinary command-line arguments."""
    if not isinstance(config, dict):
        raise TypeError(f"Expected a command mapping, received {type(config).__name__}")
    argv: list[str] = []
    for raw_key, raw_value in config.items():
        option = "--" + str(raw_key).replace("_", "-")
        value = _expand_environment(raw_value)
        if value is None or value is False:
            continue
        if value is True:
            argv.append(option)
            continue
        values = value if isinstance(value, list) else [value]
        for item in values:
            argv.append(option)
            if isinstance(item, (list, tuple)):
                argv.extend(str(part) for part in item)
            else:
                argv.append(str(item))
    return argv


def run_all(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(
        prog=f"{Path(sys.argv[0]).name} all",
        description="Render every collection declared in one writeup YAML file.",
    )
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(argv)
    payload = _expand_environment(yaml.safe_load(args.config.read_text()))
    if not isinstance(payload, dict):
        raise TypeError("The writeup render config must contain a mapping")

    required = {"training_scale", "summary", "model_comparison"}
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"Writeup render config is missing sections: {missing}")

    scale_jobs = payload["training_scale"]
    if isinstance(scale_jobs, dict):
        scale_jobs = [scale_jobs]
    if not isinstance(scale_jobs, list) or not scale_jobs:
        raise ValueError("training_scale must contain at least one command mapping")

    scale_argv = [config_to_argv(job) for job in scale_jobs]
    summary_argv = config_to_argv(payload["summary"])
    comparison_argv = config_to_argv(payload["model_comparison"])

    # Parse every section before creating outputs, so missing required options
    # fail as a configuration error rather than after a partial render.
    for job_argv in scale_argv:
        training_scale.parse_args(job_argv)
    summary.parse_args(summary_argv)
    model_comparison.parse_args(comparison_argv)

    # The later collections index or copy the matched training-scale products.
    for index, job_argv in enumerate(scale_argv, start=1):
        print(f"[writeup] training-scale {index}/{len(scale_jobs)}", flush=True)
        training_scale.main(job_argv)
    print("[writeup] summary", flush=True)
    summary.main(summary_argv)
    print("[writeup] model-comparison", flush=True)
    model_comparison.main(comparison_argv)


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in {"-h", "--help"}:
        parser = _root_parser()
        parser.print_help()
        if not argv:
            parser.exit(2)
        return

    command, command_argv = argv[0], argv[1:]
    if command == "all":
        run_all(command_argv)
        return
    if command not in COMMANDS:
        _root_parser().error(f"unknown command: {command}")
    COMMANDS[command][1](command_argv)


if __name__ == "__main__":
    main()

