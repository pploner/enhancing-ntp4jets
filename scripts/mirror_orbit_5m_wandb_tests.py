"""Mirror the six externally run 5M campaign test matrices from W&B.

This downloads only files actually uploaded to W&B. In particular, local
``saved_histograms/*.npz`` files are not available from these runs.
"""

from __future__ import annotations

import argparse
import concurrent.futures
from pathlib import Path, PurePosixPath
import time

import wandb


PROJECT = "ORBIT-scouting/orbit-tokenizer"
SUITES = ("ttbar", "sm_mixture", "gghbb", "minbias")
RUNS = {
    "orbit_campaign_full500_tt_vq4096_5m": ("v0pvksvj", "xgw1x8tx", "hxbnjku4", "3qq2npgi"),
    "orbit_campaign_full500_sm_vq4096_5m": ("k1ys4qni", "yo9spf60", "5ahohtj4", "fblsuqkd"),
    "orbit_campaign_full500_sm_pidweighted_vq4096_5m": ("pk9khw9a", "v9jr9zcb", "be540ao2", "5s643ttu"),
    "orbit_campaign_puppi128_tt_vq4096_5m": ("tyciyoy7", "63bzl5f5", "nmv2k671", "uorvipix"),
    "orbit_campaign_puppi128_sm_vq4096_5m": ("7pmijlrh", "hn8j9ykw", "pxcmuw75", "7ttq0ym7"),
    "orbit_campaign_puppi128_sm_pidweighted_vq4096_5m": ("hzqx5xvp", "i10kxwl9", "gzby7s4z", "w5ppiwl7"),
}


def download_file(run_file, root: Path) -> str:
    relative = PurePosixPath(run_file.name)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Unsafe W&B filename: {run_file.name!r}")
    target = root.joinpath(*relative.parts)
    if target.exists():
        if target.is_file() and target.stat().st_size == run_file.size:
            return "already present"
        raise FileExistsError(f"Refusing to replace an existing file: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(3):
        try:
            run_file.download(root=root, replace=False)
            if not target.is_file() or target.stat().st_size != run_file.size:
                raise OSError(f"Incomplete W&B download: {target}")
            return "downloaded"
        except Exception:
            if attempt == 2 or target.exists():
                raise
            time.sleep(2 * (attempt + 1))
    raise AssertionError("unreachable")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")

    api = wandb.Api(timeout=60)
    totals = {"downloaded": 0, "already present": 0}
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        for model, run_ids in RUNS.items():
            for suite, run_id in zip(SUITES, run_ids, strict=True):
                run = api.run(f"{PROJECT}/{run_id}")
                expected_name = f"{model}_{suite}"
                if run.name != expected_name or run.state != "finished":
                    raise RuntimeError(
                        f"Unexpected run {run_id}: name={run.name!r}, state={run.state!r}; "
                        f"expected finished {expected_name!r}"
                    )
                # W&B lists artifact manifests as run files, but their payloads
                # live in the artifact store and are not the uploaded run files.
                files = [file for file in run.files() if not file.name.startswith("artifact/")]
                root = args.output_dir / model / f"{suite}_{run_id}"
                print(f"{run.url} -> {root} ({len(files)} files)", flush=True)
                if args.dry_run:
                    continue
                futures = [pool.submit(download_file, file, root) for file in files]
                for future in concurrent.futures.as_completed(futures):
                    totals[future.result()] += 1
    if args.dry_run:
        print("Dry run: no files downloaded.")
    else:
        print(f"Finished: {totals['downloaded']} downloaded, {totals['already present']} already present.")


if __name__ == "__main__":
    main()
