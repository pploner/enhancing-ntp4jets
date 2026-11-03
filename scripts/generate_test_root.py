"""Generate a small synthetic Phase-2 L1 NanoAOD-style ROOT file for smoke tests.

Mimics the layout of the real ``merged.root`` ntuples: one ``Events`` tree with
``n<Collection>`` counters and ``<Collection>_<variable>`` branches, float
``pdgId``/``charge``, and PUPPI candidates that are deliberately *not*
pT-sorted (as in the real files).
"""

import argparse
import pathlib

import awkward as ak
import numpy as np
import uproot


def generate_root_events(n_events: int = 1000, seed: int = 0) -> dict[str, ak.Array]:
    rng = np.random.default_rng(seed)
    pid_values = np.asarray([130, 22, -211, 211, 11, -11, 13, -13], dtype=np.float32)
    charge_values = np.asarray([0, 0, -1, 1, -1, 1, -1, 1], dtype=np.float32)

    # A few events exceed the 128-particle default to exercise truncation.
    counts = rng.integers(5, 150, n_events)
    total = int(counts.sum())
    pid_indices = rng.integers(0, len(pid_values), total)
    candidates = ak.unflatten(
        ak.zip(
            {
                "pt": (1.0 + rng.exponential(5.0, total)).astype(np.float32),
                "eta": rng.uniform(-4.0, 4.0, total).astype(np.float32),
                "phi": rng.uniform(-np.pi, np.pi, total).astype(np.float32),
                "puppiWeight": rng.uniform(0.04, 1.0, total).astype(np.float32),
                "pdgId": pid_values[pid_indices],
                "charge": charge_values[pid_indices],
            }
        ),
        counts,
    )

    jet_counts = rng.integers(0, 10, n_events)
    jet_total = int(jet_counts.sum())
    jets = ak.unflatten(
        ak.zip(
            {
                "pt": rng.uniform(10.0, 200.0, jet_total).astype(np.float32),
                "eta": rng.uniform(-3.0, 3.0, jet_total).astype(np.float32),
                "phi": rng.uniform(-np.pi, np.pi, jet_total).astype(np.float32),
            }
        ),
        jet_counts,
    )
    return {
        "event": np.arange(n_events, dtype=np.uint64),
        "L1PuppiCands": candidates,
        "L1PuppiJets": jets,
    }


def generate_root(output_path: pathlib.Path, n_events: int = 1000, seed: int = 0) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    events = generate_root_events(n_events=n_events, seed=seed)
    branch_types = {
        name: values.type.content if isinstance(values, ak.Array) else values.dtype
        for name, values in events.items()
    }
    with uproot.recreate(output_path) as root_file:
        # Explicit mktree: plain assignment writes an RNTuple in recent uproot,
        # while the real ntuples are TTrees with n<Collection> counters.
        root_file.mktree("Events", branch_types).extend(events)
    print(f"Wrote {n_events} events to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=pathlib.Path)
    parser.add_argument("--n-events", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    generate_root(args.output, n_events=args.n_events, seed=args.seed)
