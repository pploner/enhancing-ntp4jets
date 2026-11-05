#!/usr/bin/env python3
"""Fit standardisation constants for the ORBIT kinematic preprocessing.

Streams the *training* split of an experiment's data config with the identity
transform (``data.eta_scale=1``, ``data.pt_shift=0``) and accumulates
per-particle statistics over real (masked-in) particles, which is what the
reconstruction loss averages over. It reports

* ``eta_scale = std(eta)``      -> transformed eta has unit variance,
* ``pt_shift  = mean(log pT)``  -> transformed log pT has zero mean,

plus the per-feature variance under the default and the fitted transform. With
an unweighted MSE, those variances are each feature's share of a trivial
"predict the mean" reconstruction loss, i.e. a direct view of the implicit
loss weights. Only the training split is used, so no val/test information
leaks into the preprocessing.

Example (TT ROOT, full training split)::

    uv run --locked python scripts/fit_orbit_kinematic_stats.py \\
        --experiment orbit_root_tt_vq4096
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import hydra  # noqa: E402
import numpy as np  # noqa: E402
from hydra import compose, initialize_config_dir  # noqa: E402
from hydra.core.global_hydra import GlobalHydra  # noqa: E402
from hydra.core.hydra_config import HydraConfig  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from gabbro.data.orbit_transforms import KinematicTransform  # noqa: E402

FEATURES = ("eta", "cos_phi", "sin_phi", "log_pt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--experiment", default="orbit_root_tt_vq4096")
    parser.add_argument(
        "--events",
        type=int,
        default=None,
        help="Training events to read (default: the config's data.train_event_budget).",
    )
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Extra Hydra override, repeatable (e.g. --override data.split_seed=1).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="JSON output (default: $LOG_DIR/kinematic_stats/<experiment>.json).",
    )
    return parser.parse_args()


def compose_data(experiment: str, overrides: list[str]):
    if not OmegaConf.has_resolver("nodename_bigram"):
        OmegaConf.register_new_resolver("nodename_bigram", lambda: "kinematicstats")
    if not OmegaConf.has_resolver("eval"):
        OmegaConf.register_new_resolver("eval", eval)
    GlobalHydra.instance().clear()
    with initialize_config_dir(version_base="1.3", config_dir=str(REPO_ROOT / "configs")):
        cfg = compose(
            config_name="train.yaml",
            overrides=[f"experiment={experiment}", *overrides],
            return_hydra_config=True,
        )
        HydraConfig.instance().set_config(cfg)
        OmegaConf.resolve(cfg.data)
    return cfg.data


class RunningStats:
    """Exact float64 sums; fine for O(1e8) particles."""

    def __init__(self, n_features: int):
        self.count = 0
        self.sum = np.zeros(n_features, dtype=np.float64)
        self.sum_sq = np.zeros(n_features, dtype=np.float64)
        self.min = np.full(n_features, np.inf)
        self.max = np.full(n_features, -np.inf)

    def update(self, values: np.ndarray) -> None:
        if not len(values):
            return
        values = values.astype(np.float64)
        self.count += len(values)
        self.sum += values.sum(axis=0)
        self.sum_sq += np.square(values).sum(axis=0)
        self.min = np.minimum(self.min, values.min(axis=0))
        self.max = np.maximum(self.max, values.max(axis=0))

    @property
    def mean(self) -> np.ndarray:
        return self.sum / self.count

    @property
    def std(self) -> np.ndarray:
        return np.sqrt(np.maximum(self.sum_sq / self.count - np.square(self.mean), 0.0))


def transformed_variances(std: np.ndarray, transform: KinematicTransform) -> dict[str, float]:
    # Shifts do not change variances; only eta is rescaled.
    variances = np.square(std)
    variances[0] /= transform.eta_scale**2
    total = float(variances.sum())
    return {
        name: {"variance": float(var), "fraction": float(var / total)}
        for name, var in zip(FEATURES, variances)
    } | {"total": total}


def main() -> None:
    args = parse_args()
    os.environ.setdefault("LOG_DIR", str(REPO_ROOT / "logs"))
    overrides = [
        "data.eta_scale=1.0",
        "data.pt_shift=0.0",
        f"data.num_workers={args.num_workers}",
        *args.override,
    ]
    if args.events is not None:
        if args.events < 1:
            raise ValueError("--events must be positive")
        overrides.append(f"data.train_event_budget={args.events}")
    data_cfg = compose_data(args.experiment, overrides)
    if "pin_memory" in data_cfg:
        data_cfg.pin_memory = False  # only throughput; stats are unaffected

    datamodule = hydra.utils.instantiate(data_cfg)
    identity = KinematicTransform.from_data_config(datamodule.hparams)
    if (identity.eta_scale, identity.pt_shift) != (1.0, 0.0):
        raise RuntimeError(f"Expected the identity transform, got {identity}")

    stats = RunningStats(len(FEATURES))
    events = 0
    started = time.perf_counter()
    for batch in datamodule.train_dataloader():
        mask = batch["part_mask"].numpy().astype(bool)
        stats.update(batch["part_features"].numpy()[..., :4][mask])
        events += int(mask.shape[0])
    elapsed = time.perf_counter() - started
    if stats.count == 0:
        raise RuntimeError("No particles were read from the training split")

    mean, std = stats.mean, stats.std
    fitted = KinematicTransform(eta_scale=float(std[0]), pt_shift=float(mean[3]))
    default = KinematicTransform()
    result = {
        "experiment": args.experiment,
        "overrides": overrides,
        "split": "train",
        "split_seed": int(data_cfg.split_seed),
        "events": events,
        "particles": int(stats.count),
        "seconds": round(elapsed, 1),
        "per_particle_stats": {
            name: {
                "mean": float(mean[i]),
                "std": float(std[i]),
                "min": float(stats.min[i]),
                "max": float(stats.max[i]),
            }
            for i, name in enumerate(FEATURES)
        },
        "fitted": fitted.as_kwargs(),
        "transformed_variance": {
            "default": transformed_variances(std, default),
            "fitted": transformed_variances(std, fitted),
        },
    }
    output = args.output or (
        Path(os.environ["LOG_DIR"]) / "kinematic_stats" / f"{args.experiment}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")

    print(f"{events} events, {stats.count} particles from the train split in {elapsed:.0f} s")
    for i, name in enumerate(FEATURES):
        print(
            f"  {name:8s} mean {mean[i]:+.4f}  std {std[i]:.4f}  "
            f"range [{stats.min[i]:+.3f}, {stats.max[i]:+.3f}]"
        )
    if abs(mean[0]) > 0.1 * std[0]:
        print(f"  note: mean(eta) = {mean[0]:+.3f} is not ~0; there is no eta shift knob.")
    for label in ("default", "fitted"):
        shares = result["transformed_variance"][label]
        fractions = ", ".join(f"{n} {shares[n]['fraction']:.0%}" for n in FEATURES)
        print(f"  variance share ({label}): {fractions}")
    print(
        f"Fitted override: data.eta_scale={fitted.eta_scale:.6g} data.pt_shift={fitted.pt_shift:.6g}"
    )
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
