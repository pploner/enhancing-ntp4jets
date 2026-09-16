# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

**ORBIT Tokenizer** (package `enhancing-ntp4jets`, importable as `gabbro`): VQ-VAE tokenization of absolute-coordinate particle/jet sequences stored in parquet files, for CMS L1 Trigger firmware-compression research. The focus is on classification-fidelity vs. firmware-storage-cost tradeoffs (ggHbb/QCD/tt processes).

The repo merges three ancestor codebases: a Hydra/Lightning training pipeline, an event-level jet tokenization use case, and a parquet loader/split-quantizer architecture. This is why an older **legacy JetClass pipeline** (`gabbro/data/iterable_dataset_jetclass.py`, `configs/experiment/example_experiment_*`, `scripts/create_tokenized_jetclass_files.py`) coexists with the active **ORBIT parquet pipeline** — treat them as separate workflows; the two should not be mixed without deliberate review.

This is a fork of `phi-5454/enhancing-ntp4jets` (`origin` = user's fork, `upstream` = original).

## Commands

Environment is managed with `uv` (Python 3.12).

- Setup: `uv sync --locked`. Set `LOG_DIR` and `MPLCONFIGDIR` env vars, or copy `.env.example` to `.env` (auto-loaded via pyrootutils). `GABBRO_ENV_FILE` can point to an external env file for W&B/Comet credentials (useful on Condor nodes where `.env` isn't available).
- Train (main entrypoint `gabbro/train.py`, Hydra-composed):
  ```
  uv run --locked python gabbro/train.py experiment=orbit_parquet_smoke model=model_vqvae_transformer
  uv run --locked python gabbro/train.py experiment=orbit_parquet_smoke model=model_vqvae_transformer_split
  ```
  Any config value can be overridden on the command line (standard Hydra overrides).
- End-to-end smoke test: `scripts/smoke_test_orbit_parquet.sh` — runs the `orbit_parquet_smoke` and `orbit_jet_parquet_smoke` experiments against a small local parquet fixture (`PARQUET_FILE` env var, defaults to `../test_data/...`).
- Tests (pytest, config in `pyproject.toml`):
  - `uv run pytest` — full suite
  - `uv run pytest tests/test_orbit_binary.py` — single file
  - `uv run pytest tests/test_orbit_binary.py::test_name` — single test
  - `uv run pytest -m "not slow"` — skip slow-marked tests
  - `--doctest-modules` is enabled by default, so doctests in docstrings anywhere in the package are also collected.
- Lint/format: `uv run ruff check --fix`, `uv run ruff format`, or `pre-commit run --all-files` for the complete pipeline (ruff import-sort+format, yamlfmt, mdformat on Markdown, codespell, nbQA black/isort/flake8 on notebooks, nbstripout). There is no CI in this repo — these are developer-invoked only.
- Hyperparameter sweeps via Optuna (separate from Hydra's own sweeper, SQLite-backed and resumable): `scripts/run_optuna_study.py`, `scripts/report_optuna_study.py`.
- Aggregate results across a Hydra multirun / Condor scan: `scripts/collect_orbit_multirun.py`.
- Firmware storage benchmarking (requires an external CMSSW checkout): `scripts/setup_orbit_storage_cmssw.sh <path/to/CMSSW_X_Y_Z>`, then `scripts/benchmark_orbit_storage.py`.

## Architecture

- **Hydra composition**: `configs/train.yaml` is the top-level config; `configs/experiment/*.yaml` composes data + model + trainer + callbacks per run. `gabbro/train.py` is the single entrypoint. Notable behavior:
  - Loads `GABBRO_ENV_FILE` via a small custom KEY=VALUE parser *before* Hydra composition — it must not be passed as a Hydra override, since config values resolve against it via `${oc.env:...}`.
  - Registers custom OmegaConf resolvers `nodename_bigram` (human-readable run IDs like `<nodename>_<bigram>`, seeded from `SLURM_JOB_ID`/`JOB_ID` if present) and `eval`.
  - Supports `continue_from_checkpoint`, `load_weights_from` (with an optional feature-expansion warm-start path for checkpoints with differing feature-dict sizes), and a standalone `ckpt_path_for_evaluation` mode that reloads the original training config and can substitute `evaluation_data_config`/`evaluation_test_suites` for cross-domain testing.
- **Data pipeline**: `gabbro.data.orbit_parquet.OrbitParquetDataModule` reads parquet files/directories/manifests (`.txt/.list/.lst`), with deterministic file-level train/val split (`data.split_seed`, `data.train_fraction`). Model input contract: `part_features [batch, seq, 4]` (scaled eta, cos φ, sin φ, transformed pT), `part_mask [batch, seq]`, `jet_type_labels [batch]`.
- **Model architecture**: `gabbro/models/vqvae.py` (Lightning module), `gabbro/models/transformer.py` (backbone), `gabbro/models/quantizers.py` (single VQ, and split quantizer: encoder latent → Φ → per-branch quantizers, each FSQ or VQ → Ψ → decoder latent). `configs/model/` has ~90 yaml files; most are systematic FSQ/VQ codebook-size and morphology sweep variants (`model_vqvae_transformer_split_fsq_mu_*_fsq_alpha_*`, etc.) generated for Condor scan grids. Add new sweep points by following this naming convention rather than reorganizing the directory.
- **"Canonical" pattern**: `orbit_canonical_tt`, `orbit_canonical_qcd_tt_vjets_vv` data configs and the matching `condor/*_canonical*.sub` files are the standardized, blessed physics-process mixtures used for cross-model comparison. The project follows an explicitly *additive* design philosophy throughout (canonical configs, downstream benchmarks, multirun collector `--family` flags) — extend these families rather than modifying or replacing them, to preserve reproducibility of prior scans.
- **Callbacks/plotting**: `gabbro/callbacks/orbit_plotting_callback.py` + `gabbro/plotting/orbit.py` produce reconstruction/residual/codebook-usage plots and FastJet-backed physics diagnostics (pT resolution, MET, jet mass, tau32) during training. Run outputs land under `${LOG_DIR}/<project>/runs/<timestamp>_<id>/{checkpoints,plots,saved_histograms,saved_metrics,wandb,csv}/`.
- **Downstream physics-fidelity benchmarks**: paired original/decoded event classifiers, plus mass-fidelity scripts (`scripts/evaluate_orbit_higgs_mass.py`, `scripts/evaluate_orbit_z_mumu_mass.py`) — these evaluate tokenization fidelity against physics observables, not just reconstruction loss.
- **Binary export & firmware storage**: `scripts/export_orbit_event_binaries.py` produces firmware-aligned bit-packed event representations; `scripts/benchmark_orbit_storage.py` compares storage cost against EDM/NanoAOD CMSSW output, via a companion C++ plugin at `cmssw/OrbitCompression/StorageBenchmark/` — built inside an external CMSSW checkout, not part of this repo's own Python build/test flow.
- **HTCondor jobs**: `condor/` has ~90 `.sub`/`.dag` files, launched via `scripts/condor_run_training.sh` using `conda run` (not `uv`). Site-specific vars (`PROJECT_DIR`, `OUTPUT_DIR`, `CONDA_ENV`, `ORBIT_MANIFEST_DIR`, `GABBRO_ENV_FILE`) are edited at the top of each `.sub` file.
- **`vqtorch/`**: vendored third-party VQ library, installed as a local `uv` path dependency (`[tool.uv.sources]` in `pyproject.toml`) — treat as third-party code, not project code.
- **`.project-root`**: marker file required by `pyrootutils.setup_root()` for resolving `PROJECT_ROOT` and `configs/paths/default.yaml` regardless of invocation directory — do not delete it.
- **`docker/`**: alternate conda/pip container environment; its `wandb` pin is notably older than the one in `pyproject.toml` — known drift, not something to silently reconcile by upgrading/downgrading one to match the other.

## Conventions

- Commit messages follow a Conventional-Commits-like style: `type(scope): imperative summary` (e.g. `feat(vq): add configurable rotation-trick quantization`, `fix(vq): restore torch import`, `docs: record canonical evaluation and storage workflows`).
