# Full-statistics ORBIT campaign

The large canonical comparison is defined by eight ordinary Hydra experiment
configs under `configs/experiment/`. There is no campaign preparation or job
bundle: select one experiment config and run the standard training entrypoint.

## Data and evaluation contract

- Reference sample: all 80,268,065 canonical ttbar events.
- Split exposure: 51,338,575 train, 12,856,486 validation, and 16,073,004 test.
- Default representation: leading 500 particles, without a PuppiW cut.
- Default tokenizer: VQ-STE with 4096 codes.
- Evaluation is split into four independent jobs: `ttbar`, `sm_mixture`,
  `gghbb`, and `minbias`. A failed or preempted suite can therefore be rerun
  without repeating training or any other suite.
- The SM sample is balanced first across five top-level classes: QCD, ttbar,
  V+jets, VV, and standalone DY. It is then balanced evenly across subprocesses
  within each class. `DYJetsToLL` is not included in V+jets.
- ggHbb contains only 2,098,560 held-out events, so all of those events are
  used rather than oversampling them to the ttbar test exposure.
- The existing batch settings are retained. Full-500 VQ runs use physical
  batch 16 with gradient accumulation 16; filtered-128 VQ runs use batch 256;
  no campaign-specific batch-size increase is applied.
- Neural runs see the complete 51,338,575-event training split each epoch, run
  for at most 30 epochs, and stop early after four validation epochs without
  improvement. Validation physics plots are produced every five epochs.
- Test metrics stream over the full suites, while plotting data are bounded to
  roughly 10,000 representative events per suite to avoid retaining millions
  of padded events in RAM.
- PID-weighted variants apply inverse-frequency weights only to the kinematic
  reconstruction loss. They do **not** add log-energy as an input or target.
  The selection-specific weights come from deterministic 100,000-event SM
  samples; raw counts, seeds, and manifest hashes are recorded in
  `campaigns/orbit_full_statistics_pid_weights.json`.
- **FAISS k-means uses a different fitting dataset.** Rather than iterating over
  the complete 51,338,575-event training split for multiple epochs, it fits
  once on a fixed 10-million-particle reservoir: 2 million selected particles
  from each of QCD, ttbar, V+jets, VV, and DY. Its saved centroid dictionary is
  then reused by the four test jobs. It has no learned encoder/decoder or
  backpropagation; encoding is nearest-centroid lookup and decoding returns the
  selected centroid.

The ttbar-only runs use every physically available event. Their explicit
per-process quotas preserve the tiny natural imbalance among hadronic,
leptonic, and semileptonic decays rather than dropping or duplicating events.

## Implemented model matrix

1. `orbit_campaign_full500_tt_vq4096`: VQ-STE, 4096 codes, leading-500
   unfiltered particles, trained on ttbar.
2. `orbit_campaign_full500_sm_vq4096`: VQ-STE, 4096 codes, leading-500
   unfiltered particles, trained on the SM mixture.
3. `orbit_campaign_full500_sm_pidweighted_vq4096`: VQ-STE, 4096 codes,
   PID-frequency-weighted kinematic loss, leading-500 unfiltered particles,
   trained on the SM mixture.
4. `orbit_campaign_puppi128_sm_vq4096`: VQ-STE, 4096 codes, PuppiW > 0.05 and
   leading 128 particles, trained on the SM mixture.
5. `orbit_campaign_puppi128_tt_vq4096`: VQ-STE, 4096 codes, PuppiW > 0.05 and
   leading 128 particles, trained on ttbar.
6. `orbit_campaign_puppi128_sm_pidweighted_vq4096`: VQ-STE, 4096 codes,
   PuppiW > 0.05 and leading 128 particles, with the PID-frequency-weighted
   loss, trained on the SM mixture.
7. `orbit_campaign_full500_sm_vq8192`: VQ-STE, 8192 codes, leading-500
   unfiltered particles, trained on the SM mixture.
8. `orbit_campaign_full500_sm_faiss4096`: FAISS k-means, 4096 fixed centroids,
   leading-500 unfiltered particles, fitted once on the 10-million-particle
   balanced SM reservoir described above.

## Training and fitting

Activate an environment containing this repository's dependencies and make
the manifests visible:

```bash
export ORBIT_MANIFEST_DIR=/path/to/manifests/production_final
export LOG_DIR=/path/to/output

python gabbro/train.py \
  experiment=orbit_campaign_full500_tt_vq4096 \
  logger=wandb.yaml
```

For model configurations 1--7, this command trains and validates only. It does
not start the large test pass. Replace the experiment name with any of the
seven neural configurations above.

The FAISS configuration is its fitting job:

```bash
python gabbro/train.py \
  experiment=orbit_campaign_full500_sm_faiss4096 \
  logger=wandb.yaml
```

It collects the balanced 10-million-particle fitting reservoir, fits the
centroids, saves `faiss_kmeans_centroids.npz`, and processes only one test batch
so Lightning runs the fitting hook. Locate the reusable dictionary with:

```bash
find /path/to/faiss-fit-run -name faiss_kmeans_centroids.npz
```

## Independent test jobs

The four evaluation configs are:

- `evaluation=ttbar`
- `evaluation=sm_mixture`
- `evaluation=gghbb`
- `evaluation=minbias`

Run a neural checkpoint on one suite with:

```bash
python gabbro/train.py \
  experiment=orbit_campaign_full500_tt_vq4096 \
  evaluation=ttbar \
  ckpt_path_for_evaluation=/path/to/training-run/checkpoints/best.ckpt \
  logger=wandb.yaml
```

Change only `evaluation=...` to create the other three independent jobs. The
experiment selector remains explicit so the command documents which of the
eight model definitions produced the checkpoint; the checkpoint's saved model
configuration is authoritative when it is loaded. Results are written below
the training run as `evaluation/ttbar`, `evaluation/sm_mixture`,
`evaluation/gghbb`, or `evaluation/minbias`, and each job keeps its logger.

For FAISS, point each test job at the dictionary produced by the fitting job:

```bash
python gabbro/train.py \
  experiment=orbit_campaign_full500_sm_faiss4096 \
  evaluation=ttbar \
  model.codebook_path=/path/to/faiss_kmeans_centroids.npz \
  logger=wandb.yaml
```

This loads the existing centroids and does not recollect the reservoir or rerun
k-means. These are ordinary Hydra configurations rather than a preparation or
job-generation layer, so Condor and Slurm wrappers can execute the same commands
directly and schedule training/fitting separately from each test suite.
