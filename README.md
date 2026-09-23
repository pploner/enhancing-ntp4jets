# ORBIT TokenizerThis repository combines three related codebases:- the Hydra and Lightning training pipeline from [`enhancing-ntp4jets`](https://github.com/uhh-pd-ml/enhancing-ntp4jets/tree/main);
- the event-level jet tokenization use case from the [L1T tokenizer repo](https://github.com/philiw/vq-tokenizer-l1t);
- the parquet loader, split-quantizer architecture, and selected plotting
  conventions used by the current ORBIT tokenization workflow.

The current focus is VQ-VAE tokenization of absolute-coordinate particle or jet
sequences stored in parquet files. The original JetClass and downstream
pre-training code remains available, but it is not the primary workflow described
below.

## Quick Start

### Install with UV

Install [UV](https://docs.astral.sh/uv/) and create the local environment:

```bash
uv sync --locked
```

Set a log directory before launching Hydra jobs:

```bash
export LOG_DIR="$PWD/outputs"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/enhancing-mplconfig}"
```

Alternatively, copy `.env.example` to `.env` and set the values there. The
training entrypoint loads the repository-root `.env` automatically through
`pyrootutils`. The file is ignored by Git and is the appropriate place for
`WANDB_API_KEY`, `WANDB_ENTITY`, and `COMET_API_TOKEN`.

You can also point a run at an explicit env file, which is useful on batch
systems or when keeping W&B credentials outside the repo:

```bash
GABBRO_ENV_FILE=/path/to/wandb.env \
uv run --locked python gabbro/train.py experiment=orbit_parquet_smoke logger=wandb.yaml
```

`GABBRO_ENV_FILE` is loaded before Hydra composes the config, so it can provide
`LOG_DIR`, `MPLCONFIGDIR`, `WANDB_API_KEY`, `WANDB_ENTITY`, and
`COMET_API_TOKEN`. Values in the explicit env file override values already
loaded from the repository `.env`. Do not pass this as a Hydra override such as
`env_file=...`; that would be too late for `${oc.env:LOG_DIR}`.

`MPLCONFIGDIR` is useful on systems where the home directory is read-only. For
online W&B logging, authenticate once with `uv run wandb login` or provide the
usual W&B environment variables. The W&B project can be selected with
`logger.wandb.project=...`; explicit Hydra logger settings take precedence over
environment defaults.

### Arrange parquet files

The ORBIT parquet loader accepts parquet files, directories, or text manifests.
A manifest is a `.txt`, `.list`, or `.lst` file containing one
parquet path per line, with blank lines and `#` comments ignored:

```text
/path/to/sample_000.parquet
/path/to/sample_001.parquet
# /path/to/temporarily_disabled.parquet
```

Relative paths inside a manifest are resolved relative to the manifest file.
Direct directories remain supported. Use a separate manifest or directory for
the test set:

```text
/path/to/dataset/
├── train_val/
│   ├── sample_000.parquet
│   ├── sample_001.parquet
│   └── ...
└── test/
    ├── sample_100.parquet
    └── ...
```

The loader deterministically shuffles and splits the files under `train_val/`
using `data.split_seed` and `data.train_fraction`. The training dataset then
reshuffles parquet row groups between iterations. Validation and test row groups
are not shuffled. Automatic train/validation splitting requires at least two
parquet files.

Manifest inputs are the preferred production shape:

```bash
'data.parquet_files_train_val=[/path/to/pq_files_train_val.txt]' \
'data.parquet_files_test=[/path/to/pq_files_test.txt]'
```

Explicit parquet lists remain available for fixed splits and smoke tests:

```bash
data.parquet_files_train='[/path/train.parquet]' \
data.parquet_files_val='[/path/val.parquet]' \
data.parquet_files_test='[/path/test.parquet]'
```

### Multi-class mode

When data lives in separate per-class directories (e.g. on EOS), use
`parquet_files_train_val_per_class` instead of `parquet_files_train_val`. Pass
a mapping of class name to manifest (or directory). Class names are sorted
case-insensitively and assigned deterministic integer labels (0, 1, …). Each
class is split independently into train/val using the same `data.split_seed`
and `data.train_fraction`. The resulting `class_to_label` mapping is saved in
the run's `hparams` for reproducibility.

Each class value may also be a structured spec:

```yaml
data:
  parquet_files_train_val_per_class:
    ggHbb:
      paths: /path/to/ggHbb_train_val.txt
      eval_sequence_type: jet_ak8
      eval_min_pt: 250
      weight: 1.0
      max_train_events: 100000
      max_val_events: 100000
    minbias:
      paths: /path/to/minbias_train_val.txt
      eval_sequence_type: jet_ak4
      weight: 1.0
      max_train_events: 100000
      max_val_events: 100000
  parquet_files_test_per_class:
    ggHbb:
      paths: /path/to/ggHbb_test.txt
      eval_sequence_type: jet_ak8
      eval_min_pt: 250
      max_test_events: 10000
    minbias:
      paths: /path/to/minbias_test.txt
      eval_sequence_type: jet_ak4
      max_test_events: 10000
```

Global `data.sequence_type` selects the model input collection; for the
particle pipeline, leave it as `particle`. `eval_sequence_type` and
`eval_min_pt` select whole validation/test events using a class-specific jet
collection before particle padding. `weight` controls train-time class sampling
when multiple class datasets are interleaved. Event caps are applied after
object cuts and validation/test event filters.

The three event caps also define split-specific class composition. Set a cap
to `0` to keep a class in the shared label mapping while disabling it for that
split. For example, a class with `max_train_events: 0` and
`max_val_events: 0` can still be evaluated with a positive
`max_test_events`. Unequal positive caps provide different class ratios for
training, validation, and testing. The `weight` field controls train-time
interleaving order; the event caps control how many examples each split uses.

`parquet_files_train_val_per_class` is mutually exclusive with
`parquet_files_train_val` and explicit `parquet_files_train`/`parquet_files_val`.
Use `parquet_files_test_per_class` for class-labelled test plots and metrics;
do not combine it with `parquet_files_test`.

**Step 1 — generate manifests.** The EOS dataset contains many process classes
as subdirectories. Run `make_eos_manifests.py` once to write one `.txt` manifest
per subdirectory:

```bash
python scripts/make_eos_manifests.py \
  --eos-root /eos/project/f/foundational-model-dataset/samples/production_final \
  --out-dir /path/to/manifests/
```

**Step 2 — select the classes you want.** The manifest directory will contain a
`.txt` file for every class in the EOS tree, but you only load the ones you
explicitly list in `parquet_files_train_val_per_class`. The rest are ignored.
The production experiment config `orbit_jet_puppi_ak8_production` already
selects four classes via `$ORBIT_MANIFEST_DIR`:

```yaml
# configs/experiment/orbit_jet_puppi_ak8_production.yaml (excerpt)
data:
  parquet_files_train_val_per_class:
    ggHbb:         ${oc.env:ORBIT_MANIFEST_DIR}/ggHbb.txt
    QCD_HT50toInf: ${oc.env:ORBIT_MANIFEST_DIR}/QCD_HT50toInf.txt
    VBFHbb:        ${oc.env:ORBIT_MANIFEST_DIR}/VBFHbb.txt
    minbias:       ${oc.env:ORBIT_MANIFEST_DIR}/minbias.txt
```

To use a different class selection, create a new experiment config with the
desired subset. Then set the manifest directory and launch:

```bash
export ORBIT_MANIFEST_DIR=/path/to/manifests/
uv run --locked python gabbro/train.py experiment=orbit_jet_puppi_ak8_production
```

A two-class ggHbb/minbias variant is also available for jobs that should only
require those two manifests:

```bash
export ORBIT_MANIFEST_DIR=/path/to/manifests/
uv run --locked python gabbro/train.py experiment=orbit_jet_puppi_ak8_ggHbb_minbias
```

For the local ggHbb/minbias split used by the Condor scan, the manifest
directory should contain `ggHbb_train_val.txt`, `ggHbb_test.txt`,
`minbias_train_val.txt`, and `minbias_test.txt`. That experiment config trains on particles, validates/tests
ggHbb events with an AK8 jet above 250 GeV, and validates/tests minbias events
with at least one AK4 jet.

### Run a smoke test

The checked-in smoke script runs both particle and PUPPI AK8 jet modes against a
single parquet fixture:

```bash
PARQUET_FILE=/path/to/example.parquet \
OUTPUT_DIR=/tmp/enhancing-smoke \
./scripts/smoke_test_orbit_parquet.sh
```

### Train with the single VQ quantizer

Particle PID is enabled by default. Raw `L1T_PUPPIPart_PID` and
`L1T_PUPPIPart_Charge` values are mapped to eight categories (neutral hadron,
photon, negative/positive hadron, electron/positron, and muon/antimuon). PID is
provided to the encoder as a one-hot vector and reconstructed with masked
cross-entropy alongside the four-feature kinematic loss. Disable the complete
PID path for legacy comparisons with:

```bash
pid.enabled=false
```

The PID loss coefficient is configurable through `pid.loss_weight` and defaults
to `1.0`. Jet-sequence experiments disable PID because their elements are jets,
not particles.

The following is a runnable starting point for particle tokenization:

```bash
LOG_DIR="$PWD/outputs" uv run --locked python gabbro/train.py \
  experiment=orbit_parquet_smoke \
  model=model_vqvae_transformer \
  logger=wandb.yaml \
  task_name=orbit_particle_vq \
  logger.wandb.name=orbit_particle_vq \
  logger.wandb.project=orbit-tokenizer \
  'data.parquet_files_train_val=[/path/to/pq_files_train_val.txt]' \
  'data.parquet_files_test=[/path/to/pq_files_test.txt]' \
  trainer.max_epochs=30 \
  trainer.limit_train_batches=null \
  trainer.limit_val_batches=null \
  trainer.enable_checkpointing=true
```

### Train with the split quantizer

The split model applies:

```text
encoder latent -> Phi -> branch quantizers -> Psi -> decoder latent
```

`Phi` and `Psi` use NormFormer stacks. Each branch may use FSQ or VQ. The
checked-in split config uses FSQ for both `mu` and `alpha`:

```bash
LOG_DIR="$PWD/outputs" uv run --locked python gabbro/train.py \
  experiment=orbit_parquet_smoke \
  model=model_vqvae_transformer_split \
  logger=wandb.yaml \
  task_name=orbit_particle_split_fsq \
  logger.wandb.name=orbit_particle_split_fsq \
  logger.wandb.project=orbit-tokenizer \
  'data.parquet_files_train_val=[/path/to/pq_files_train_val.txt]' \
  'data.parquet_files_test=[/path/to/pq_files_test.txt]' \
  trainer.max_epochs=30 \
  trainer.limit_train_batches=null \
  trainer.limit_val_batches=null \
  trainer.enable_checkpointing=true
```

Split-quantizer branch losses can be weighted with
`model.model_kwargs.split_quantizer_cfg.branch_loss_weights`. Missing branches
default to `1.0`. For split quantizers, these branch weights are the only
quantizer-loss scale factors; the global `model.model_kwargs.alpha` is reserved
for the single-quantizer path and is not applied again.

```text
single quantizer:
  total_loss = reconstruction_loss + model.model_kwargs.alpha * quantizer_loss

split quantizer:
  quantizer_loss = sum(branch_loss[branch] * branch_loss_weights[branch])
  total_loss = reconstruction_loss + quantizer_loss
```

All loss terms are mask-aware. Reconstruction losses are averaged over valid
particles only. Single-VQ quantizer loss is recomputed from `vq_out["z"]` and
`vq_out["z_q"]` and averaged over `part_mask`. Split-quantizer branch losses are
computed per token, zero padded before branch quantization and before `Psi`, and
then reduced as:

```text
branch_loss = sum(per_token_branch_loss * part_mask) / sum(part_mask)
```

This also covers the "single FSQ branch" scan configs, which are split
quantizers with `branch_order: ["mu"]`; only the active `mu` branch contributes.
For one-branch split quantizers, `bypass_single_branch_projection: true` skips
`Phi` and `Psi` entirely and sends the model latent directly through the active
branch quantizer. This is only valid when the active branch dimension matches
`model.model_kwargs.latent_dim`; for example, an FSQ branch with levels
`[21, 21, 21]` must use `latent_dim: 3`.
The pure three-dimensional FSQ codebook scans enable this bypass so they match
the direct encoder-quantizer-decoder structure of the single-VQ scans.
The FSQ-related Condor scan configs currently use `mu: 0.25` and `alpha: 1.0`
for branch loss weights. In mu-only FSQ runs this gives
`0.25 * loss_quantizer_mu`, while the inactive `alpha` branch does not
contribute. The same branch weights are used by the split VQ-mu/FSQ-alpha scan.

FSQ loss-space caveat: the implementation must not compare the continuous
latent directly to an integer packed code such as `[0, num_codes)`. That would
make the auxiliary loss scale with token IDs and produce a pathological loss
landscape, especially for large scalar levels. The FSQ branch loss is computed
in scalar-quantizer space instead:

```text
default:
  f(z) = tanh(z)
  z_hat = round(tanh(z) * half_width) / half_width
  loss = ||f(z) - z_hat||^2

with fsq_loss_use_half_width=true:
  f(z) = tanh(z) * half_width
  z_hat = round(tanh(z) * half_width)
  loss = ||f(z) - z_hat||^2
```

The default is the normalized, subtler bounded-space loss. The half-width form
can be enabled through
`model.model_kwargs.split_quantizer_cfg.fsq_loss_use_half_width=true` when a
larger directional signal in scalar-bin space is desired. This auxiliary loss is
not intrinsic to FSQ and should be treated as an experiment knob, not as part of
token ID reconstruction.

Validation and test logs include both raw and weighted branch terms, for example
`val_metrics/loss_quantizer_mu`, `val_metrics/loss_quantizer_mu_weighted`,
`val_metrics/loss_quantizer_alpha`, and
`val_metrics/loss_quantizer_alpha_weighted`.

### Shorten the transmitted latent sequence

`model.model_kwargs.latent_sequence_compression` supports two experimental
modes. `learned_cross_attention` uses learned compressor and expander queries.
`direct_prefix_masking` instead retains the first
`ceil(number_of_particles * ratio)` encoder latents in each event. It assumes
the valid input particles form a prefix ordered by descending pT.

In direct-prefix mode, only the retained prefix is passed to the quantizer and
used for its losses, codebook updates, utilization metrics, and exported token
mask. Dropped valid positions are zero before quantization and receive learned,
position-specific mask embeddings before the decoder. The decoder and
reconstruction loss still use the original full particle mask, so the output
sequence retains its original length. Supported controls are `ratio` in
`(0, 1]`, `min_tokens`, and `rounding: ceil`.

The ratio-one direct-prefix mode retains every valid latent and does not use the
mask embeddings, making it numerically equivalent to the uncompressed direct
path for the same initialization. The initial half-length comparison can be
submitted with:

```bash
condor_submit condor/orbit_fsq_prefix_masking_ablation.sub
```

To use the L1T-style event representation, replace
`experiment=orbit_parquet_smoke` with `experiment=orbit_jet_parquet_smoke`.
Each sequence element is then one absolute-coordinate PUPPI AK8 jet.

The current ORBIT experiment configs are smoke-ready integration starting
points. Before a long production run, review model size, sequence length, batch
size, checkpoint policy, and train/validation limits.

### Optimizer and LR Schedule

The VQ and split-quantizer tokenizer configs use AdamW with weight decay `0.01`
and the Transformer one-cycle schedule:

```text
epochs 0-4:   linearly increase 0.0005 -> 0.001
epochs 4-24:  linearly decrease 0.001 -> 0.0005
epochs 24-30: linearly decrease 0.0005 -> 0.0003
```

The schedule is implemented with
`gabbro.schedulers.lr_scheduler.OneCycleCooldown` and stepped once per epoch.

### Progress bars and short runs

Lightning progress bars are disabled by default with
`trainer.enable_progress_bar=false` to keep batch-system logs compact. Enable the
default tqdm-style Lightning progress bar with:

```bash
trainer.enable_progress_bar=true
```

The usual Lightning batch limits are supported and are the preferred way to run
short tests:

```bash
trainer.max_epochs=1 \
trainer.limit_train_batches=10 \
trainer.limit_val_batches=2 \
trainer.limit_test_batches=2
```

Use integer limits for ORBIT parquet runs. The dataloader is iterable and
does not advertise a fixed epoch length, so fractional limits such as
`trainer.limit_train_batches=0.1` are not the robust choice.

Validation/test plotting has a separate retention limit. The model still
computes validation loss for the batches selected by `trainer.limit_val_batches`,
but by default it only keeps the first validation batch for reconstruction plots
and future FastJet-style physics diagnostics:

```bash
model.max_validation_plot_batches=1
model.max_test_plot_batches=null
```

Set `model.max_validation_plot_batches=10` if you want validation plots over ten
validation batches, or `model.max_test_plot_batches=20` to keep test plotting
bounded. `null` means keep all processed batches for that stage.

The ORBIT plotting callback skips Lightning's validation sanity check. The
sanity loop can still compute validation loss unless you disable it with
`trainer.num_sanity_val_steps=0`.

For multi-class ORBIT runs, validation and test plots are saved/logged for both
the combined sample and each class label. In W&B the image keys are grouped as
`val/all/...`, `val/<class>/...`, `test/all/...`, and `test/<class>/...`.
Combined histogram and metric artifacts keep the legacy names
`<stage>_orbit_*`; per-class artifacts are named
`<stage>_<class>_orbit_*`.
Particle-mode physics plots also include reclustered jet-count comparisons
(`N_reco - N_orig`). For mixed evaluation filters, per-class plots use the
class radius inferred from `eval_sequence_type`, and combined plots use that
class-specific radius event by event.

For an even smaller end-to-end check, use Lightning's fast dev mode:

```bash
trainer.fast_dev_run=true
```

If you want to skip the initial validation sanity loop during a quick run, add:

```bash
trainer.num_sanity_val_steps=0
```

## Data Contract

`gabbro.data.orbit_parquet.OrbitParquetDataModule` supports:

| `sequence_type` | Input prefix      | Default length |
| --------------- | ----------------- | -------------: |
| `particle`      | `L1T_PUPPIPart`   |            128 |
| `particle_full` | `L1T_PUPPIPart`   |            500 |
| `jet_ak4`       | `L1T_JetAK4`      |             14 |
| `jet_ak8`       | `L1T_JetAK8`      |              7 |
| `jet_puppi_ak4` | `L1T_JetPuppiAK4` |             14 |
| `jet_puppi_ak8` | `L1T_JetPuppiAK8` |              7 |

All modes emit the same model input contract:

```text
part_features:   [batch, sequence, 4]
part_mask:       [batch, sequence]
jet_type_labels: [batch]
```

The four features are scaled `eta`, `cos(phi)`, `sin(phi)`, and transformed
`pT`. Coordinates remain absolute: `eta` and `phi` are not made relative to a
jet axis.

`particle` keeps the established `PuppiW > 0.05` selection. The additive
`particle_full` mode applies no PUPPI-weight cut and retains the first 500
candidates in the stored descending-$p_T$ order. `PuppiW` is not an input or
reconstruction target, so full-event four-vectors use the stored unweighted
candidate $p_T$.

## Repository Structure

The main training entrypoint is `gabbro/train.py`. It composes Hydra configs,
instantiates the datamodule, model, callbacks, loggers, and Lightning trainer,
then runs train/validation/test according to the config.

Pipeline responsibilities are split as follows:

| Pipeline task                       | Main files                                                                                                             |
| ----------------------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| Hydra experiment composition        | `configs/train.yaml`, `configs/experiment/*.yaml`                                                                      |
| Data source and batch schema        | `configs/data/*.yaml`, `gabbro/data/orbit_parquet.py`, `gabbro/data/iterable_dataset_jetclass.py`                      |
| Feature preprocessing definitions   | `configs/feature_dict/*.yaml`, `gabbro/utils/arrays.py`                                                                |
| Model architecture                  | `configs/model/*.yaml`, `gabbro/models/vqvae.py`, `gabbro/models/transformer.py`                                       |
| Single and split quantizers         | `gabbro/models/quantizers.py`, `configs/model/model_vqvae_transformer*.yaml`                                           |
| Lightning training/evaluation logic | `gabbro/models/vqvae.py`, `gabbro/models/lightning_models.py`, `gabbro/models/backbone_multihead.py`                   |
| Trainer, accelerator, and run paths | `configs/trainer/*.yaml`, `configs/hydra/default.yaml`, `configs/paths/default.yaml`                                   |
| Logging backends                    | `configs/logger/*.yaml`, `gabbro/utils/utils.py`                                                                       |
| Per-run plotting callbacks          | `configs/callbacks/*.yaml`, `gabbro/callbacks/orbit_plotting_callback.py`, `gabbro/callbacks/tokenization_callback.py` |
| Plot rendering utilities            | `gabbro/plotting/orbit.py`, `gabbro/plotting/feature_plotting.py`, `gabbro/plotting/utils.py`                          |
| Token export/reconstruction helpers | `gabbro/models/vqvae.py`, `gabbro/data/data_tokenization.py`, `scripts/create_tokenized_jetclass_files.py`             |
| Multirun post-processing            | `scripts/collect_orbit_multirun.py`, `gabbro/plotting/orbit.py`                                                        |
| Standalone checks and utilities     | `scripts/smoke_test_orbit_parquet.sh`, `scripts/evaluate_token_quality.py`, `scripts/filter_sparse_orbit_parquet.py`   |
| Environment and container setup     | `pyproject.toml`, `uv.lock`, `.env.example`, `docker/`                                                                 |

The ORBIT parquet path is configured through `orbit_parquet_smoke` or
`orbit_jet_parquet_smoke`. The original enhancing JetClass path is configured
through `example_experiment_tokenization_transformer` and
`example_experiment_backbone_and_head`.

## Tracking Runs

### Output directories

Hydra writes each normal run under:

```text
${LOG_DIR}/${project_name}/runs/<timestamp>_<generated-id>/
```

Relevant files include:

```text
config.yaml
config_resolved.yaml
checkpoints/
plots/
saved_histograms/
saved_metrics/
wandb/
csv/
```

The ORBIT plotting callback saves reconstruction histograms, residual plots,
codebook-usage plots, and the single-run physical-coordinate paper plots under
`plots/`. For particle runs, the paper plot set includes FastJet-backed jet
`pT` resolution, missing transverse energy, jet mass, and `tau32` residuals.
Attention-weight plots are intentionally not produced. Compressed histogram
arrays are saved under `saved_histograms/`, and compact metrics JSON files land
under `saved_metrics/`. The same plot images are logged to W&B or Comet when
those loggers are active.

For multi-class ORBIT runs, the callback writes separate `all`, `ggHbb`, and
`minbias` artifacts whenever those classes are present in the selected split.
Each class metrics file includes `loss_reco`, `loss_reco_l1`, `loss_reco_l2`,
their per-value reductions, `metrics/active_codes_total`, and
`metrics/utilization_total`. Validation artifacts intentionally use the
configured bounded plotting sample; test artifacts use the full configured
test sample by default. Select a class during comparison collection with, for
example, `--stage test --group ggHbb` or `--stage test --group minbias`.

The same files include sparse empirical token-rate measurements under
`metrics/entropy/` and `metrics/rate/`. The primary values are marginal entropy
in bits per latent token, marginal bits per event, and marginal bits per input
particle. The latter two multiply token entropy by the observed latent length,
so reduced-sequence models receive the corresponding rate reduction. Split
quantizers additionally report branch entropy, FSQ scalar-dimension entropy,
and total correlation. All calculations use the latent `code_mask`; they do
not require the optional codebook histogram.

These values estimate a static independent-token entropy coder. They do not
include sequence-conditional probabilities, coder headers, metadata, or an
actual compressed byte stream. The logged sample event/token counts make the
bounded validation estimate explicit; use test entropy for model comparison.

### Binary event export

`scripts/export_orbit_event_binaries.py` writes matched, length-prefixed binary
files for storage-size studies. For each requested test class it writes the
raw `L1T_PUPPIPart_PT`, `L1T_PUPPIPart_Eta`, and `L1T_PUPPIPart_Phi` values
directly from parquet, plus the valid quantized token IDs. No model feature
transformation is applied to the original values, and their source floating
precision is retained. The checkpoint-defined particle mask, event selection,
particle order, and sequence cap are applied to both representations (128 for
legacy particle mode and 500 for full-event mode). Token IDs use
the narrowest unsigned integer type that can represent the configured
codebook. Padding is omitted, while a uint32 event length preserves each event
boundary. Every file embeds a JSON schema; the accompanying `manifest.json`
records byte sizes, hashes, checkpoint, event selection, particle counts, and
token counts.

For VQ-STE scan `957029`, process 4 is the 2048-code run. Export 1000 test
events from each class with:

```bash
RUN_ROOT=../enhancing_ntp4jets_runs/orbit_vq_ste_scan_957029/orbit-particle-ggHbb-minbias/runs
RUN_DIR=$(for config in "$RUN_ROOT"/*/config_resolved.yaml; do
  grep -q '^task_name: orbit_particle_ggHbb_minbias_vq_ste_codes_2048$' "$config" \
    && dirname "$config"
done)

conda run -p /eos/home-y/yelberke/conda_condor_orbit_env \
  python scripts/export_orbit_event_binaries.py \
  --run-dir "$RUN_DIR" \
  --output-dir ../enhancing_ntp4jets_runs/orbit_binary_export_957029_4 \
  --events-per-class 1000 \
  --device cuda
```

The original files contain `[PT, Eta, Phi]` exactly as read from parquet. The
2048-code token files use little-endian uint16 IDs.

### Firmware-aware storage table

`scripts/benchmark_orbit_storage.py` generates a per-checkpoint LaTeX storage
table for minimum bias, ggHbb, and inclusive tt by default. The optional
`mixture` sample adds the canonical QCD/tt/VJets/VV mixture. The standalone
plain representation contains the common PUPPI payload only: 14-bit pT (0.25
GeV LSB), 12-bit signed eta and 11-bit signed phi (both pi/720 LSB), and the
3-bit hardware PID class. These fields are rounded to the nearest code,
saturated, and densely packed into 40 bits per candidate. VQ indices are also
densely packed, using `ceil(log2(num_codes))` bits per token. PID is predicted
by the decoder and is therefore not stored beside the tokens.

The EDM and NanoAOD measurements use the same compact leaf types. Plain
candidates have aligned `uint16` pT, eta, and phi code columns plus a `uint8`
PID column; unused high bits are zero. Token IDs use `uint16` for codebooks of
up to 65536 entries. EDM stores these as native integer vector products through
`PoolOutputModule`, while nanoAOD uses `nanoaod::FlatTable` and
`NanoAODOutputModule`. All compressed outputs use LZMA level 9. Reported EDM and NanoAOD
payload sizes sum only the corresponding compressed physics branches; total
file sizes and the exact branch list remain available in the JSON manifest.

Build the small companion plugin once in any compatible CMSSW area:

```bash
./scripts/setup_orbit_storage_cmssw.sh /path/to/CMSSW_X_Y_Z
```

Then generate the three-sample table for one trained model:

```bash
conda run --no-capture-output \
  -p /eos/home-y/yelberke/conda_condor_orbit_env \
  python scripts/benchmark_orbit_storage.py \
  --run-dir /path/to/orbit/run \
  --manifest-root /eos/home-y/yelberke/enhancing-ntp4jets/manifests/production_final \
  --cmssw-base /path/to/CMSSW_X_Y_Z \
  --output-dir /path/to/storage_benchmark \
  --events-per-sample 1000 \
  --device cuda
```

The output directory contains `compression_table.tex`, a flat CSV,
`compression_measurements.json`, the packed streams and their `.xz` versions,
and the EDM/NanoAOD ROOT files. Use `--skip-containers` for a quick Raw and
Standalone-only pass, or `--keep-existing-containers` to resume a partially
completed container run.

### GPU FAISS k-means baseline

`model_faiss_kmeans_baseline` provides a test-only classical baseline that
fits one centroid dictionary to valid transformed training particles and then
reconstructs each test particle with its nearest centroid. It uses the same
`eta/3`, `cos(phi)`, `sin(phi)`, and `log(pT)-1.8` representation and the same
masked L2 objective as the neural tokenizer. No extra normalization or
projection onto the phi unit circle is applied.

The fitting sample is capped at one million particles. Per-class particle
quotas follow the configured train-time class weights, so the default 1:1
ggHbb/minbias setup contributes 500k particles from each class rather than
implicitly favoring the class with larger event multiplicity. The fitted
centroids and metadata are saved under `artifacts/` and uploaded as a W&B
`codebook` artifact. All normal test plots, class-specific reconstruction
metrics, utilization, entropy, and effective-rate metrics are reused.

GPU-enabled FAISS is intentionally an optional environment dependency. Verify
the exact Condor environment before submitting:

```bash
conda run -p /eos/home-y/yelberke/conda_condor_orbit_env \
  python -c 'import faiss; print(faiss.__version__, faiss.get_num_gpus())'
```

The final value must be at least one. The Condor jobs repeat this check and
fail before reading data if GPU FAISS is unavailable; they never silently use
CPU clustering.

```bash
condor_submit condor/orbit_faiss_kmeans_smoke.sub
condor_submit condor/orbit_faiss_kmeans_scan.sub
```

The full scan matches the VQ grid from 128 through 16384 centroids and appears
as the `FAISS k-means` family in multirun scalar plots.

### Supported loggers

Select a logger with a Hydra override:

```bash
logger=wandb.yaml
logger=csv.yaml
logger=comet.yaml
logger=many_loggers.yaml
```

W&B run names default to `task_name` and can be overridden explicitly:

```bash
task_name=orbit_vq_v1 logger.wandb.name=orbit_vq_v1
```

For an offline W&B run:

```bash
logger=wandb.yaml logger.wandb.offline=true
```

Offline payloads land under the run directory in `wandb/offline-run-*` and can
later be uploaded with `uv run wandb sync <offline-run-directory>`.

Comet requires `COMET_API_TOKEN`. CSV logging requires no external service.
MLflow is not currently configured in this repository.

### Hydra multiruns

Basic parameter sweeps use Hydra's `-m` mode. Each job gets its own output
directory under `${LOG_DIR}/${project_name}/multiruns/`:

```bash
LOG_DIR="$PWD/outputs" uv run --locked python gabbro/train.py -m \
  experiment=orbit_parquet_smoke \
  model=model_vqvae_transformer \
  logger=wandb.yaml \
  task_name=orbit_vq_scan \
  'logger.wandb.name=orbit_vq_scan_codes_${model.model_kwargs.vq_kwargs.num_codes}_seed_${seed}' \
  logger.wandb.group=orbit_vq_scan \
  'data.parquet_files_train_val=[/path/to/pq_files_train_val.txt]' \
  'data.parquet_files_test=[/path/to/pq_files_test.txt]' \
  model.model_kwargs.vq_kwargs.num_codes=256,512,1024 \
  seed=42,43 \
  trainer.max_epochs=30 \
  trainer.limit_train_batches=null \
  trainer.limit_val_batches=null
```

Each job writes its own plots, compressed histograms, and compact metrics JSON.
At training start, the ORBIT plotting callback also samples at most 2,048
training events and writes `plots/train_start_input_particle_multiplicity.png`.
The figure contains the joint distribution and class-wise overlays of the
post-selection model-input particle count. Configure this with
`callbacks.orbit_plotting_callback.train_particle_count_max_events`, or disable
it with `callbacks.orbit_plotting_callback.include_train_particle_count_histogram=false`.
After the sweep finishes, aggregate the jobs with:

```bash
uv run --locked python scripts/collect_orbit_multirun.py \
  --multirun-dir "${LOG_DIR}/orbit-smoke/multiruns/<timestamp>"
```

Use `--stage test` to select test artifacts instead of validation artifacts. By
default, the script creates `comparisons/<stage>/all/` inside the multirun
directory. Use `--group <class>` to collect class-specific artifacts, for
example `--stage test --group ggHbb` or `--stage test --group minbias`.
Canonical runs have multiple named test suites; select one explicitly, for
example `--stage test --suite training_like --group all`.
To compare arbitrary run output directories instead of a Hydra multirun
directory, pass them explicitly:

```bash
uv run --locked python scripts/collect_orbit_multirun.py \
  --run-dir /path/to/run_a /path/to/run_b /path/to/run_c \
  --stage test \
  --group all \
  --output-dir /path/to/comparison
```

To set display labels explicitly, use repeatable `--run` entries. The label is
optional; unlabeled entries fall back to the W&B name, then `task_name`, then the
directory name:

```bash
uv run --locked python scripts/collect_orbit_multirun.py \
  --run /path/to/run_a "FSQ 20x3" \
  --run /path/to/run_b "VQ rotation 4096" \
  --run /path/to/run_c \
  --stage test \
  --group all \
  --output-dir /path/to/comparison
```

To connect points as a family in scalar plots, use repeatable `--family`
entries. Each family gets a consistent color and its points are connected by a
line ordered by total codebook size:

```bash
uv run --locked python scripts/collect_orbit_multirun.py \
  --family FSQ /path/to/fsq_15x3 /path/to/fsq_20x3 /path/to/fsq_21x3 \
  --family VQ /path/to/vq_4096 /path/to/vq_8192 \
  --stage test \
  --group all \
  --output-dir /path/to/comparison
```

If a run contains several evaluations with the same suite/group names, select
the intended artifact directory explicitly with `--artifact-family`. Supply
each run together with the corresponding evaluation directory:

```bash
uv run --locked python scripts/collect_orbit_multirun.py \
  --artifact-family "Trained on tt" \
    /path/to/tt_run /path/to/tt_run/evaluation/mixture_test \
  --artifact-family "Trained on SM mixture" \
    /path/to/mixture_run /path/to/mixture_run/evaluation/best.ckpt \
  --stage test --suite training_like --group all \
  --output-dir /path/to/comparison
```

When `WANDB_API_KEY` is available—either in the shell, repository `.env`, or
the same `GABBRO_ENV_FILE` used by training—the collector automatically creates
its own W&B comparison run. It uploads the plots plus `manifest.json` and
`summary.csv`; the W&B project defaults to the compared runs' project. Override
the comparison identity when needed:

```bash
uv run --locked python scripts/collect_orbit_multirun.py \
  --family FSQ /path/to/fsq_15x3 /path/to/fsq_20x3 \
  --family VQ /path/to/vq_4096 /path/to/vq_8192 \
  --stage test \
  --group all \
  --output-dir /path/to/comparison \
  --wandb-project orbit-tokenizer \
  --wandb-name fsq_vs_vq_test_comparison \
  --wandb-group orbit_multirun_comparisons
```

Use `--no-wandb` for a local-only comparison.

Use `--no-titles` to render presentation-ready copies without axes or panel
titles. Axis labels, legends, CMS labels, and reference lines are retained.

The particle rate-distortion plot includes a vertical reference at 43
bits/input particle, representing the original particle encoding. To also show
the reconstruction error of a continuous autoencoder as a horizontal reference,
pass its MSE explicitly:

```bash
uv run --locked python scripts/collect_orbit_multirun.py \
  --multirun-dir "${LOG_DIR}/orbit-smoke/multiruns/<timestamp>" \
  --continuous-autoencoder-mse 0.00123
```

The value must use the same reconstruction-MSE definition and data split as the
plotted runs. Override the source-size reference with
`--original-bits-per-input-particle` if the input representation changes.

An additional `compression_ratio_40bit_vs_reco_mse.png` view normalizes the
marginal compressed rate per input particle to the 40-bit common PUPPI payload.
Thus, an x value of 0.25 means that the compressed representation uses 25% of
that payload size (a factor-four reduction), and the original payload is marked
at 1. Override the denominator with `--compression-reference-bits` when needed.

Each explicit run entry must point at a single Hydra run directory containing
`.hydra/config.yaml`. The collector first looks for `saved_histograms/` and
`saved_metrics/` directly under that run directory, then falls back to
`evaluation/best.ckpt/` and other `evaluation/*/` artifact directories. If
`--output-dir` is omitted in explicit-run mode, outputs are written under
`./orbit_run_comparison/<stage>/<group>/`.
The output directory contains:

```text
manifest.json
summary.csv
combined_reconstruction_features.png
combined_reconstruction_residuals.png
codebook_size_vs_mse_total.png
codebook_size_vs_utilization_total.png
codebook_size_vs_val_loss.png
codebook_size_vs_marginal_entropy_bits_per_token.png
codebook_size_vs_normalized_entropy.png
codebook_size_vs_marginal_bits_per_event.png
codebook_size_vs_marginal_bits_per_input_particle.png
marginal_bits_per_input_particle_vs_reco_mse.png
compression_ratio_40bit_vs_reco_mse.png
marginal_bits_per_event_vs_reco_mse.png
```

The collector is adapted from the Phaedra prototype's multirun aggregation logic, but it does
not launch training jobs itself. Hydra remains responsible for sweep expansion
and execution. The collector reads each job's Hydra config to infer single-VQ or
split-quantizer codebook metadata, then consumes the local
`saved_histograms/*.npz` and `saved_metrics/*.json` artifacts. It also falls back
to Lightning CSV metrics for older runs.

### Native Optuna studies

Optuna studies use the native runner rather than Hydra's sweeper plugin. A
study owns a resumable SQLite database, one output directory per trial, and a
report bundle with CSV/JSON, PNG, and interactive HTML plots:

```bash
CONDA_ENV="${CONDA_ENV:-../conda_condor_orbit_env}"
conda run --no-capture-output -p "${CONDA_ENV}" python scripts/run_optuna_study.py \
  --study-root "${LOG_DIR}/optuna/fsq_mu_16x3_lr" \
  experiment=orbit_jet_puppi_ak8_ggHbb_minbias \
  hparams_search=orbit_optuna_fsq_mu_16x3_lr \
  logger=wandb.yaml \
  data.num_workers=4
```

Re-run the same command to resume until the configuration's `target_trials`
count is reached. Regenerate a report without training:

```bash
CONDA_ENV="${CONDA_ENV:-../conda_condor_orbit_env}"
conda run --no-capture-output -p "${CONDA_ENV}" python scripts/report_optuna_study.py \
  "${LOG_DIR}/optuna/fsq_mu_16x3_lr"
```

The report directory contains `trials.csv`, `best_trial.json`, static PNG
figures, and an `index.html` linking to interactive Optuna plots. Promote the
selected parameters into a normal experiment config before using the multirun
collector for comparisons between studies.

HTCondor jobs use this same environment through `scripts/condor_run_training.sh`.
Before submitting a native study, ensure `../conda_condor_orbit_env` contains
the locked Optuna and Plotly versions; the wrapper checks both imports before
starting a trial.

### Canonical ORBIT datasets

The directly runnable full-statistics eight-model comparison, including the
standalone-DY SM mixture, is documented in
[`docs/orbit_full_statistics_campaign.md`](docs/orbit_full_statistics_campaign.md).

Two additive experiment configs define the standard process mixtures without
replacing any existing experiment:

- `orbit_canonical_tt` balances the hadronic, leptonic, and semileptonic tt
  processes.
- `orbit_canonical_qcd_tt_vjets_vv` first balances QCD, tt, VJets, and VV, then
  balances the processes within each group. VJets includes W/Z+jets and
  `DYJetsToLL`; VV includes WW/WZ/ZZ hadronic, leptonic, and semileptonic
  samples (including `WW_semileptonic`). Photon, gamma+V, triboson, and Higgs
  samples are intentionally excluded from these four training groups.

Both configs use fixed budgets of 200,000 training and 200,000 validation
events. The loader splits every `<process>_train_val.txt` manifest
deterministically at the file level, then applies the group/process quotas to
each split. Tests are read only from separate `<process>_test.txt` manifests.
Each named test suite contains 20,000 events:

- `training_like` follows the balanced training mixture.
- `tt_vs_gghbb` contains 10,000 tt events, balanced over the three tt decay
  modes, and 10,000 `ggHbb` events.

Create the required disjoint split manifests next to the raw production
manifests before submitting a canonical run:

```bash
python scripts/split_orbit_canonical_manifests.py \
  --manifest-dir /eos/home-y/yelberke/enhancing-ntp4jets/manifests/production_final
```

The original submit files remain available. Matching additive files end in
`_canonical.sub` and default to the four-group experiment. For example:

```bash
condor_submit condor/orbit_vq_ste_scan.sub
condor_submit condor/orbit_vq_ste_scan_canonical.sub
condor_submit -append 'CANONICAL_EXPERIMENT=orbit_canonical_tt' \
  condor/orbit_vq_ste_scan_canonical.sub
```

All canonical submit files share site settings in
`condor/orbit_canonical_common.sub`. Their run, W&B, and log names include the
canonical experiment name, so the two mixtures can be submitted side by side.


### Downstream physics-fidelity benchmarks

The additive downstream tools compare paired events before and after a tokenizer encode/decode path. Prepare manifests and export the five-class data with:

```bash
python scripts/prepare_orbit_downstream_manifests.py --canonical-manifest-dir /path/to/canonical_manifests --output-dir /path/to/downstream_manifests
python scripts/export_orbit_downstream_events.py --run-dir /path/to/tokenizer_run --manifest-dir /path/to/downstream_manifests --output-dir /path/to/paired_events --device cuda
```

For an older tokenizer checkpoint without PID inputs, pass `--allow-no-pid`.
This writes a kinematics-only paired dataset; both original and decoded events
receive empty PID channels, so their classifier comparison remains matched.

The export contains 200k training, 200k validation, and 20k test events balanced over QCD, tt, VJets, VV, and ggHbb, then over subprocesses. Run the five-seed original/decoded Transformer matrix with:

```bash
PYTHON_BIN=python ./scripts/run_orbit_classifier_matrix.sh /path/to/paired_events /path/to/classifier_results
```

For a small end-to-end Condor check of the `957029_5` checkpoint, submit:

```bash
mkdir -p /eos/user/y/yelberke/enhancing_ntp4jets_runs/condor_logs
condor_submit condor/orbit_downstream_classifier_957029_5_smoke.sub
```

It exports 10k balanced events for each split, trains one epoch on the
original representation, and tests the paired original and decoded versions
of the same 10k test events.

Evaluate truth-independent resolved Higgs mass fidelity from the two leading
anti-$k_t$ $R=0.4$ jets with:

```bash
python scripts/evaluate_orbit_higgs_mass.py --candidate-mode leading_pt --run-dir /path/to/tokenizer_run --gghbb-test-manifest /path/to/downstream_manifests/ggHbb_test.txt --output-dir /path/to/higgs_mass_results --device cuda
```

To compare several checkpoints, use the same labelled `--run RUN_DIR LABEL`
form as the multirun collector. The output contains each run's individual
artifacts under `runs/<label>/`, plus decoded outline overlays and a combined
metrics JSON at the output root:

```bash
python scripts/evaluate_orbit_higgs_mass.py --candidate-mode leading_pt \
  --run /path/to/vq_run "VQ (1024)" \
  --run /path/to/fsq_run "FSQ" \
  --gghbb-test-manifest /path/to/downstream_manifests/ggHbb_test.txt \
  --output-dir /path/to/higgs_mass_comparison --device cuda
```

Both mass evaluators sync their PNG plots to W&B by default (project
`orbit-tokenizer`). They also upload an `orbit-downstream-evaluation` artifact
containing every metrics JSON and compact histogram `.npz` input: `bins`, the
original spectrum, and each decoded spectrum with its label. Use
`--wandb-name`, `--wandb-group`, and `--wandb-entity` to name the benchmark run,
or `--no-wandb` to retain only local outputs. To place a single-model Higgs
plot and its fitted peak metrics directly on the originating tokenizer run,
pass that training run's W&B ID with `--wandb-run-id`. The appended keys live
under `downstream/higgs_mass/`; resuming the W&B run does not resume or modify
model training.

The canonical-tt presentation suite combines title-free copies of all four
golden-child multirun collections with a 46-model Higgs peak comparison and a
six-model near-4096 Higgs histogram. It evaluates each tokenizer once on 10k
ggHbb events, caches the candidate masses, and runs the presentation collector
only after every cache job succeeds:

```bash
mkdir -p /eos/user/y/yelberke/enhancing_ntp4jets_runs/condor_logs
condor_submit_dag condor/orbit_presentation_plots.dag
```

The resulting collection is written below
`presentation_plots/canonical_tt_golden/` and uploaded as the single W&B run
`canonical_tt_presentation_plots`. The Higgs scatter uses the fitted
double-sided-Crystal-Ball peak mean and width on its axes, with the inclusive
original reconstruction marked as the common reference. A complementary mass
response plot shows the fitted decoded-to-original peak ratio,
$\mu_{\mathrm{decoded}}/\mu_{\mathrm{original}}$, against the marginal
compressed rate normalized to the 40-bit common particle payload.

For PID-enabled tokenizers, evaluate the dimuon response on leptonic ZZ events
using the leading reconstructed muon and antimuon in each representation. The
ZZ manifest below is the default and may be omitted:

```bash
python scripts/evaluate_orbit_z_mumu_mass.py --candidate-mode leading_pt --run-dir /path/to/tokenizer_run --z-test-manifest /path/to/downstream_manifests/ZZ_leptonic_test.txt --output-dir /path/to/z_mumu_mass_results --device cuda
```

The Z evaluator accepts the same repeatable `--run RUN_DIR LABEL` interface;
it writes `z_mumu_mass_multirun.png` and
`z_mumu_mass_multirun_metrics.json` alongside per-run outputs.

Canonical PID-enabled test runs now log one PID-conditioned residual figure per
test suite and save compact JSON/NPZ inputs. Four-feature checkpoints show
$\Delta\eta$, wrapped $\Delta\phi$, and
$\log(p_T^\mathrm{reco}/p_T^\mathrm{orig})$; checkpoints with the optional
energy feature add $\log(E^\mathrm{reco}/E^\mathrm{orig})$. To run only this
diagnostic for an existing checkpoint, use:

```bash
python scripts/evaluate_orbit_pid_pulls.py \
  --run-dir /path/to/tokenizer_run \
  --test-manifest /path/to/downstream_manifests/ZZ_leptonic_test.txt \
  --output-dir /path/to/pid_pull_results \
  --events 10000 --device cuda --wandb
```

The distributions are conditioned on the original PID at each sequence
position. Since the decoder does not predict a per-particle uncertainty, these
are reconstruction residuals rather than uncertainty-normalized statistical
pulls.
Existing output can be uploaded without repeating inference using
`--output-dir /path/to/pid_pull_results --wandb --upload-only`.

The focused warm-started VQ-STE 4096 pilot adds $\log E-2.5$ and applies
inverse-frequency PID weights only to the reconstruction loss. It trains for
at most five epochs on 20k events and tests on 2k events per canonical suite:

```bash
mkdir -p /eos/user/y/yelberke/enhancing_ntp4jets_runs/condor_logs
condor_submit condor/orbit_tt_pid_balanced_loge_pilot.sub
```

Both evaluators also accept `--candidate-mode hungarian`. This anchors the
candidate on the original leading objects and Hungarian-matches compatible
decoded objects by $\Delta R$; Z matches retain the reconstructed muon charge
classes. Neither mode reads generator truth. Post-processing is disabled by
default. Add `--max-abs-eta 2.5` and/or `--max-match-dr 0.2` independently, or
use `--apply-current-cuts` to enable all cuts applicable to the selected mode.
The Higgs AK4 clustering threshold remains
30 GeV in every configuration and is not a post-processing cut.

The previous generator-matched benchmark remains available as
`--candidate-mode truth`; use `--apply-current-cuts` with it to reproduce the
old resolved acceptance. Truth mode also retains the legacy boosted AK8
observable. Both mass benchmarks save and log empirical means and standard
deviations and include them in plot legends alongside the fitted peak
diagnostics.

To submit all four truth-free comparisons (Higgs/Z crossed with leading-$p_T$/
Hungarian), each with 10k events and no acceptance or matching cuts, use:

```bash
mkdir -p /eos/user/y/yelberke/enhancing_ntp4jets_runs/condor_logs
condor_submit \
  -append 'RUN_SPECS=--run /path/to/run_a \"Run A\" --run /path/to/run_b \"Run B\"' \
  condor/orbit_mass_candidate_modes_multirun.sub
```

The backslashes are required by HTCondor when labels contain spaces. Labels
such as `FSQ`, `VQ_STE`, and `VQ_rotation` can be passed without quotes.

The checkpoint configuration selects 128-particle or 500-particle export and
Higgs evaluation automatically. For full-event canonical scans and the
matching classifier matrix, use:

```bash
./scripts/submit_orbit_canonical_tt_full_event.sh
./scripts/submit_orbit_canonical_qcd_tt_vjets_vv_full_event.sh
PYTHON_BIN=python ./scripts/run_orbit_classifier_full_event_matrix.sh /path/to/paired_events /path/to/classifier_results
python scripts/visualize_orbit_event.py /path/to/events.parquet 0 --mode full-event --output event_full.png
```

Full-event tokenizer and classifier training use batches of 16 with 16-step
gradient accumulation. Their submit wrappers request 32 GB of host memory and
the `nextweek` job flavour; the legacy submit wrappers are unchanged.

Exports record stable event IDs plus checkpoint and manifest hashes, and refuse train/test overlap or non-empty output directories.

### HTCondor jobs

The repository includes starter HTCondor submit files for this repo's Hydra
entrypoint. Condor workers are launched through `conda run`, so `uv` is not
required on the cluster worker nodes:

```text
condor/orbit_jet_production_smoke.sub  # one tiny GPU smoke job
condor/orbit_wandb_logging_smoke.sub   # 10-batch ggHbb/minbias W&B logging smoke
condor/orbit_vq_codebook_scan.sub      # one GPU job per VQ codebook size
condor/orbit_continuous_autoencoder_baseline.sub # no-quantization reconstruction ceiling
condor/orbit_vq_rotation_scan.sub  # rotation-trick VQ diagnostic scan without k-means init
condor/orbit_vq_rotation_512_guardrail_scan.sub # 512-code rotation-trick guardrail scan
condor/orbit_faiss_kmeans_smoke.sub # small GPU FAISS fit/test integration check
condor/orbit_faiss_kmeans_scan.sub  # GPU k-means baseline on the full VQ size grid
condor/orbit_fsq_codebook_scan.sub     # one GPU job per FSQ split-quantizer setting
condor/orbit_fsq_l1_codebook_scan.sub  # FSQ split-quantizer scan with L1 reconstruction loss
condor/orbit_split_vq_mu_fsq_alpha_l1_scan.sub # STE VQ-mu/FSQ-alpha scan with L1 reconstruction loss
condor/orbit_fsq_noaux_2epoch_scan.sub # 2-epoch FSQ diagnostic with auxiliary loss disabled
condor/orbit_fsq_l1_noaux_2epoch_scan.sub # 2-epoch L1 FSQ diagnostic with auxiliary loss disabled
condor/orbit_split_vq_mu_fsq_alpha_l1_noaux_2epoch_scan.sub # 2-epoch mixed VQ/FSQ diagnostic with FSQ alpha loss disabled
scripts/condor_run_training.sh         # shared Condor executable
```

Before submitting, edit the site-specific variables at the top of the `.sub`
file:

```text
PROJECT_DIR
OUTPUT_DIR
CONDA_ENV
ORBIT_MANIFEST_DIR
GABBRO_ENV_FILE
```

`PROJECT_DIR` should point to this checkout on the shared filesystem.
`CONDA_ENV` should point to the conda environment visible on the worker node;
prefer the canonical path shown by `python -c 'import sys; print(sys.prefix)'`.
`ORBIT_MANIFEST_DIR` should contain the per-class manifests from
`scripts/make_eos_manifests.py`. `GABBRO_ENV_FILE` may provide W&B credentials
and other environment variables as described above. Create the Condor log
directory once before submission:

```bash
mkdir -p /path/to/output/condor_logs
condor_submit condor/orbit_wandb_logging_smoke.sub
```

The wrapper sets `LOG_DIR` to `OUTPUT_DIR/SUITE_ID`, keeps W&B and Matplotlib
state inside the run directory, then launches:

```bash
conda run --no-capture-output -p "$CONDA_ENV" python gabbro/train.py ...
```

Additional Hydra overrides can be appended to the `arguments` line in the
submit file.

### Multirun plotting

The multirun comparison collector uses the CMS style and histogram rendering
from `mplhep`. Install it in the shared Condor environment once before running
`multirun_suite.sh` or `multirun_suite_selected_splits.sh`:

```bash
conda install -p /eos/home-y/yelberke/conda_condor_orbit_env -c conda-forge mplhep
```

## What Changed

### From enhancing-ntp4jets

The original Hydra/Lightning structure remains the base. Existing JetClass
loading, tokenization scripts, Comet/CSV logging, and downstream backbone
workflows remain in the tree.

Additions and adjustments:

- UV project metadata and lockfile for local development without Docker;
- W&B as an explicit logger option with predictable run names;
- the schema-neutral ORBIT plotting callback;
- the split quantizer abstraction and split-token reconstruction;
- the L1T-compatible Docker pin `tables==3.10.1`.

### From the L1T tokenizer repo

The event-level jet use case is preserved as a separate loader mode rather than
replacing particle loading:

- `jet_puppi_ak8` represents each event as a variable-length sequence of PUPPI
  AK8 jets;
- particle and jet modes emit the same batch keys and feature width;
- the main architectural difference seen by the model is sequence length;
- absolute `phi` is represented with `cos(phi)` and `sin(phi)`.

Sparse-event parquet filtering is intentionally left as an optional boxed-out
script until profiling shows that it is needed.

### From the Phaedra Prototype

The imported parquet-tokenization additions are:

- parquet streaming by row group;
- text manifests containing parquet paths;
- deterministic shuffled file splitting for train/validation and a separate
  test directory;
- row-group reshuffling for training to avoid long contiguous data periods;
- absolute-coordinate preprocessing;
- a `Phi -> branches -> Psi` split quantizer with FSQ/VQ branches;
- storage of combined `part_token_id` plus explicit `part_token_<branch>`
  fields;
- reconstruction from explicit branch tokens or packed combined IDs;
- selected reconstruction, residual, codebook-usage, and attention plotting
  helpers.

Attention plotting helpers are present, but they are not wired into the
callback until the VQ-VAE exposes attention tensors during validation and test.

## Rotation Trick Extension

The rotation-trick VQ gradient estimator from the Phaedra prototype has not been ported yet.
The natural implementation point is `gabbro/models/quantizers.py`:

1. Add a rotation-trick autograd function equivalent to the prototype's
   `_RotationTrick`.
2. Extend `VQBranch` with a `gradient_estimator` option such as `"ste"` or
   `"rotation_trick"`.
3. Pass that option through `split_quantizer_cfg.vq_kwargs`.
4. Add the same option to the single VectorQuant path if rotation-trick support
   is also required without `Phi` and `Psi`.
5. Preserve the existing normalized output contract so token export,
   reconstruction, plotting, and `part_token_id` compatibility continue to
   work.

The plotting helpers already reserve a `vq_rotation` family for
future comparison plots.

## Legacy Workflows

The original JetClass tokenizer, token export, joint pre-training, and
classification paths remain available. Their starting points are:

```text
configs/experiment/example_experiment_tokenization_transformer.yaml
configs/experiment/example_experiment_backbone_and_head.yaml
scripts/create_tokenized_jetclass_files.py
```

These paths retain assumptions from the original enhancing repository and
should be reviewed separately before mixing them with the ORBIT parquet runs.
