# ORBIT writeup plots

`scripts/render_orbit_writeup.py` is the single cache-only entrypoint for the
LaTeX-ready plot collection. It never launches inference. Model evaluation and
rendering remain separate so labels, bins, layouts, and styles can be changed
without repeating GPU work.

Its commands are named for the products they create:

- `summary`: golden, ablation, and single-model downstream products;
- `model-comparison`: the joint eight-model 5M comparison;
- `training-scale`: one matched 200k-versus-5M comparison; and
- `all`: every collection declared in one YAML file.

Every figure is written as PDF and 300-dpi PNG without a main title. The output
also contains CSV/JSON/LaTeX tables, numerical NPZ caches, a linked README, and
model provenance in Markdown, LaTeX, CSV, and JSON.

## Expensive caches

The exact SM downstream model must be the unweighted, full-500,
SM-mixture-trained VQ-STE checkpoint with 4096 codes (W&B training run
`kit9ge98`). Stage its `best.ckpt` and resolved config together in one run
directory before running the following commands.

The established Higgs and tau32 evaluators create two of the caches:

```bash
python scripts/evaluate_orbit_higgs_mass.py \
  --candidate-mode leading_pt \
  --run-dir "$SM_VQ4096_RUN" \
  --gghbb-test-manifest "$ORBIT_MANIFEST_DIR/ggHbb_test.txt" \
  --output-dir "$CACHE_DIR/higgs" \
  --events 10000 --device cuda

python scripts/evaluate_orbit_tau32.py \
  --run-dir "$SM_VQ4096_RUN" \
  --test-manifest "$ORBIT_MANIFEST_DIR/tt0123j_5f_ckm_LO_MLM_hadronic_test.txt" \
  --output-dir "$CACHE_DIR/tau32" \
  --events 10000 --device cuda
```

Jet and PID caches accept repeated top-level classes. Comma-separated manifests
inside each class are balanced within that class; the classes themselves are
balanced against one another.

```bash
SM_CLASSES=(
  --test-class "QCD=$ORBIT_MANIFEST_DIR/QCD_HT50tobb_test.txt,$ORBIT_MANIFEST_DIR/QCD_HT50toInf_test.txt"
  --test-class "tt=$ORBIT_MANIFEST_DIR/tt0123j_5f_ckm_LO_MLM_hadronic_test.txt,$ORBIT_MANIFEST_DIR/tt0123j_5f_ckm_LO_MLM_leptonic_test.txt,$ORBIT_MANIFEST_DIR/tt0123j_5f_ckm_LO_MLM_semiLeptonic_test.txt"
  --test-class "VJets=$ORBIT_MANIFEST_DIR/WJetsToLNu_13TeV-madgraphMLM-pythia8_test.txt,$ORBIT_MANIFEST_DIR/WJetsToQQ_13TeV-madgraphMLM-pythia8_test.txt,$ORBIT_MANIFEST_DIR/ZJetsTobb_13TeV-madgraphMLM-pythia8_test.txt,$ORBIT_MANIFEST_DIR/ZJetsTocc_13TeV-madgraphMLM-pythia8_test.txt,$ORBIT_MANIFEST_DIR/ZJetsToQQ_13TeV-madgraphMLM-pythia8_test.txt,$ORBIT_MANIFEST_DIR/ZJetsTovv_13TeV-madgraphMLM-pythia8_test.txt"
  --test-class "VV=$ORBIT_MANIFEST_DIR/WW_hadronic_test.txt,$ORBIT_MANIFEST_DIR/WW_leptonic_test.txt,$ORBIT_MANIFEST_DIR/WW_semileptonic_test.txt,$ORBIT_MANIFEST_DIR/WZ_hadronic_test.txt,$ORBIT_MANIFEST_DIR/WZ_leptonic_test.txt,$ORBIT_MANIFEST_DIR/WZ_semileptonic_test.txt,$ORBIT_MANIFEST_DIR/ZZ_hadronic_test.txt,$ORBIT_MANIFEST_DIR/ZZ_leptonic_test.txt,$ORBIT_MANIFEST_DIR/ZZ_semileptonic_test.txt"
  --test-class "DY=$ORBIT_MANIFEST_DIR/DYJetsToLL_13TeV-madgraphMLM-pythia8_test.txt"
)

python scripts/evaluate_orbit_jet_residuals.py \
  --run-dir "$SM_VQ4096_RUN" "${SM_CLASSES[@]}" \
  --output-dir "$CACHE_DIR/jets" --events 10000 --device cuda

python scripts/evaluate_orbit_pid_pulls.py \
  --run-dir "$SM_VQ4096_RUN" "${SM_CLASSES[@]}" \
  --output-dir "$CACHE_DIR/sm_pid" --events 10000 --device cuda

# MET uses the dedicated 8192-code SM-mixture model. This paired cache retains
# decoded-minus-original residuals event by event.
python scripts/evaluate_orbit_met.py \
  --run-dir "$SM_VQ8192_RUN" "${SM_CLASSES[@]}" \
  --output-dir "$CACHE_DIR/met" --events 10000 --device cuda
```

For the 200k-event ttbar VQ-STE 4096 PID panels, run the PID evaluator on its
exact golden checkpoint and the established ttbar test manifest. The resulting
NPZ supplies both residuals and the confusion matrix.

The joint 5M model comparison reads one raw 8-by-8 PID confusion matrix per
model and test suite. The full-500 matrices are retained from the original full
test evaluations. The filtered 128-particle matrices can be reproduced without
running the slower jet and residual evaluations:

```bash
python scripts/evaluate_orbit_pid_confusion.py \
  --registry campaigns/orbit_campaign_5m_plot_models.yaml \
  --manifest-dir "$ORBIT_MANIFEST_DIR" \
  --output-dir "$LOG_DIR/writeup_cache/campaign_5m_comparison/pid_confusion_sources" \
  --model-id puppi128_tt_vq4096 \
  --model-id puppi128_sm_vq4096 \
  --model-id puppi128_sm_pidweighted_vq4096 \
  --suite ttbar --suite sm_mixture --suite gghbb --suite minbias \
  --events 10000 --batch-size 256 --device cuda
```

The comparison renderer writes row-normalized, paper-style comparisons to
`$LOG_DIR/writeup_plots/model_comparisons_5m/pid_confusion/{full500,puppi128}`.
Every PNG/PDF has a matching NPZ containing both the integer counts and plotted
fractions, plus a JSON sidecar with its exact source files and event count.

## Rendering

```bash
python scripts/render_orbit_writeup.py summary \
  --golden-manifest "$GOLDEN_MANIFEST" \
  --campaign-summary-root "$CAMPAIGN_WANDB_MIRROR" \
  --campaign-summary-root "$CAMPAIGN_7_8_RUN_ROOT" \
  --downstream-run-dir "$SM_VQ4096_RUN" \
  --golden-pid-confusion "$GOLDEN_PID_CACHE" \
  --golden-pid-residuals "$GOLDEN_PID_CACHE" \
  --higgs-cache "$CACHE_DIR/higgs/higgs_candidates.csv" \
  --tau32-cache "$CACHE_DIR/tau32/tt_hadronic_ak8_pt450_tau32_values.npz" \
  --met-cache "$CACHE_DIR/met/sm_mixture_met_histograms.npz" \
  --met-run-dir "$SM_VQ8192_RUN" \
  --jet-residual-cache "$CACHE_DIR/jets/sm_mixture_jet_residuals.npz" \
  --sm-pid-residuals "$CACHE_DIR/sm_pid/pid_conditional_pull_values.npz" \
  --output-dir "$WRITEUP_DIR"
```

By default, missing inputs make the renderer fail after it writes the status
README. During plot development, `--allow-incomplete` renders every available
product and records missing inputs under **Pending inputs**.

## Full-500 training-scale comparison

The particle-residual and rate--distortion comparison uses the 5M-event,
ttbar-trained VQ-STE model at 4096 codes. Its lower-statistics reference is the
ttbar **full-event** sweep (200k training events, leading 500
particles, no PUPPIW cut), not the filtered 128-particle golden sweep. The
renderer rejects non-ttbar training samples as well as mismatched selections,
exposures, and codebook sizes.

```bash
python scripts/render_orbit_writeup.py training-scale \
  --canonical-manifest "$LOG_DIR/comparisons/canonical_tt_full500_vq_ste_training_like_all/manifest.json" \
  --campaign-residual-cache "$LOG_DIR/plotting_ckpts/full500-tt-vq4096-evaluation/ttbar/saved_histograms/test_ttbar_all_orbit_hists_step_0.npz" \
  --campaign-4096-metrics "$LOG_DIR/plotting_ckpts/full500-tt-vq4096-evaluation/ttbar/saved_metrics/test_ttbar_all_orbit_metrics_step_0.json" \
  --campaign-4096-config "$LOG_DIR/orbit_campaign_5m_wandb_test_imports/orbit_campaign_full500_tt_vq4096_5m/ttbar_v0pvksvj/config.yaml" \
  --output-dir "$LOG_DIR/writeup_plots/comparisons/ttbar_full500_5m_vs_200k"
```

The 200k-reference 8192-code test artifact is currently missing, so its reference
curve contains the other seven codebook sizes. No SM-trained point is inserted
to fill that gap. This omission is recorded in `comparison_values.json` rather
than being interpolated.

## Render every writeup product

The complete renderer runs training-scale comparisons first because the summary
and joint comparison collections index or copy those products. Copy the example
configuration, export or replace the paths it references, and run:

```bash
cp campaigns/orbit_writeup_render.example.yaml /tmp/orbit_writeup_render.yaml
python scripts/render_orbit_writeup.py all \
  --config /tmp/orbit_writeup_render.yaml
```

The YAML uses the same option names as the focused subcommands, with underscores
in place of dashes. Undefined environment variables are rejected before any
rendering starts.
