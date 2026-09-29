# VQ-STE 4096 DY-augmented SM learning-rate sweeps

These directly runnable Hydra experiment configs scan the constant AdamW
learning rate independently for codebook-loss weights `alpha = 1`, `5`, and
`10`. They inherit the established full-500 VQ-STE 4096 setup and change only
the scan identifiers, alpha, learning rate, and controlled event budgets.

Every point uses:

- the five equally weighted top-level training groups `QCD`, `tt`, `VJets`,
  `VV`, and standalone `DY`, with equal subprocess weighting inside a group;
- the leading 500 particles without a PUPPIW threshold;
- 200,000 fixed training events per epoch and 20,000 fixed validation events;
- at most 30 epochs, with early stopping after four unimproved validation
  epochs;
- VQ-STE with 4096 codes and a constant learning rate (`scheduler: null`);
- training and validation only. Run the named test suites after selecting a
  preferred alpha/LR point.

## Config selectors

| Alpha | Learning rate | Experiment selector |
|---:|---:|---|
| 1 | `1e-7` | `vq_ste_4096_sm_dy_lr_sweeps/alpha_1/lr_1e-7` |
| 1 | `1e-6` | `vq_ste_4096_sm_dy_lr_sweeps/alpha_1/lr_1e-6` |
| 1 | `1e-5` | `vq_ste_4096_sm_dy_lr_sweeps/alpha_1/lr_1e-5` |
| 1 | `1e-4` | `vq_ste_4096_sm_dy_lr_sweeps/alpha_1/lr_1e-4` |
| 1 | `1e-3` | `vq_ste_4096_sm_dy_lr_sweeps/alpha_1/lr_1e-3` |
| 5 | `1e-7` | `vq_ste_4096_sm_dy_lr_sweeps/alpha_5/lr_1e-7` |
| 5 | `1e-6` | `vq_ste_4096_sm_dy_lr_sweeps/alpha_5/lr_1e-6` |
| 5 | `1e-5` | `vq_ste_4096_sm_dy_lr_sweeps/alpha_5/lr_1e-5` |
| 5 | `1e-4` | `vq_ste_4096_sm_dy_lr_sweeps/alpha_5/lr_1e-4` |
| 5 | `1e-3` | `vq_ste_4096_sm_dy_lr_sweeps/alpha_5/lr_1e-3` |
| 10 | `1e-7` | `vq_ste_4096_sm_dy_lr_sweeps/alpha_10/lr_1e-7` |
| 10 | `1e-6` | `vq_ste_4096_sm_dy_lr_sweeps/alpha_10/lr_1e-6` |
| 10 | `1e-5` | `vq_ste_4096_sm_dy_lr_sweeps/alpha_10/lr_1e-5` |
| 10 | `1e-4` | `vq_ste_4096_sm_dy_lr_sweeps/alpha_10/lr_1e-4` |
| 10 | `1e-3` | `vq_ste_4096_sm_dy_lr_sweeps/alpha_10/lr_1e-3` |

## Running one point

Set the production-manifest directory and choose one selector:

```bash
export ORBIT_MANIFEST_DIR=/path/to/manifests/production_final
EXPERIMENT=vq_ste_4096_sm_dy_lr_sweeps/alpha_5/lr_1e-5
python gabbro/train.py experiment="${EXPERIMENT}"
```

Add `logger=wandb.yaml` only when the execution site is configured for W&B.

For Slurm, pass the same selector through the site's normal environment setup:

```bash
sbatch --wrap="python gabbro/train.py experiment=${EXPERIMENT}"
```

For HTCondor, expose an `EXPERIMENT` macro in the site submit description and
forward it as `experiment=$(EXPERIMENT)`. For example, its queue block can be:

```text
queue EXPERIMENT from (
  vq_ste_4096_sm_dy_lr_sweeps/alpha_1/lr_1e-7
  vq_ste_4096_sm_dy_lr_sweeps/alpha_1/lr_1e-6
)
```

The experiment YAMLs intentionally contain no batch-system paths or W&B
credentials, so the same selectors can be used on another cluster unchanged.
