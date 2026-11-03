# Joint 5M model comparison

The workflow is restartable: each model writes an independent NPZ and JSON
under `/eos/user/y/yelberke/enhancing_ntp4jets_runs/writeup_cache/campaign_5m_comparison`.
Completed pairs are skipped on retry. DAGMan rescue files preserve unfinished
nodes if the submission is interrupted.

Submit from the repository root:

```bash
mkdir -p /eos/user/y/yelberke/enhancing_ntp4jets_runs/condor_logs
module load lxbatch/eossubmit
condor_submit_dag condor/orbit_campaign_5m_comparison.dag
```

The EOSSubmit module is required because the executable, logs, caches, and
plot products all live on EOS.

Inspect progress:

```bash
module load lxbatch/eossubmit
condor_q -dag
find /eos/user/y/yelberke/enhancing_ntp4jets_runs/writeup_cache/campaign_5m_comparison \
  -type f -name '*.json' -print | sort
```

If DAGMan creates a rescue file, resume with:

```bash
module load lxbatch/eossubmit
condor_submit_dag condor/orbit_campaign_5m_comparison.dag.rescue001
```

The final renderer writes
`/eos/user/y/yelberke/enhancing_ntp4jets_runs/writeup_plots/model_comparisons_5m`.
AK4 residual jets use minimum-total-delta-R Hungarian assignment with no
maximum-delta-R rejection.
