#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
OUTPUT_ROOT=${OUTPUT_ROOT:-/eos/user/y/yelberke/enhancing_ntp4jets_runs/presentation_plots/canonical_tt_full500_golden}
CONDA_ENV=${CONDA_ENV:-/eos/home-y/yelberke/conda_condor_orbit_env}
GGHBB_MANIFEST=${GGHBB_MANIFEST:-${PROJECT_DIR}/manifests/production_final/ggHbb_test.txt}
MANIFEST=${OUTPUT_ROOT}/training_like/manifest.json
NEAR_TEMPLATE=/eos/user/y/yelberke/enhancing_ntp4jets_runs/presentation_plots/canonical_tt_golden/golden/near_4096/training_like/manifest.json
CACHE_DIR=${OUTPUT_ROOT}/higgs_cache
LOG_DIR=${OUTPUT_ROOT}/interactive_logs
PARALLEL_EVALUATIONS=${PARALLEL_EVALUATIONS:-4}

if ! [[ ${PARALLEL_EVALUATIONS} =~ ^[1-9][0-9]*$ ]]; then
  echo "PARALLEL_EVALUATIONS must be a positive integer" >&2
  exit 2
fi

mkdir -p -- "${CACHE_DIR}" "${LOG_DIR}"
cd -- "${PROJECT_DIR}"

if [[ -n ${WAIT_FOR_PID:-} ]]; then
  echo "[$(date -Is)] Waiting for existing evaluation PID ${WAIT_FOR_PID}"
  while kill -0 "${WAIT_FOR_PID}" 2>/dev/null; do
    sleep 20
  done
fi

count=$("${CONDA_ENV}/bin/python" -c 'import json, sys; print(len(json.load(open(sys.argv[1]))))' "${MANIFEST}")
evaluate_index() {
  local index=$1
  local padded cache run_dir
  printf -v padded '%03d' "${index}"
  cache=${CACHE_DIR}/${padded}/higgs_candidates.csv
  if [[ -s ${cache} ]] && [[ $(wc -l < "${cache}") -eq 10001 ]]; then
    echo "[$(date -Is)] ${padded}: cached; skipping"
    return 0
  fi
  run_dir=$("${CONDA_ENV}/bin/python" -c 'import json, sys; print(json.load(open(sys.argv[1]))[int(sys.argv[2])]["run_dir"])' "${MANIFEST}" "${index}")
  echo "[$(date -Is)] ${padded}: evaluating ${run_dir}"
  if ORBIT_MANIFEST_DIR=$(dirname -- "${GGHBB_MANIFEST}") \
      conda run --no-capture-output -p "${CONDA_ENV}" \
      python scripts/evaluate_orbit_higgs_mass.py \
        --run-dir "${run_dir}" \
        --gghbb-test-manifest "${GGHBB_MANIFEST}" \
        --output-dir "${CACHE_DIR}/${padded}" \
        --events 10000 \
        --batch-size 4 \
        --bootstrap-replicas 0 \
        --candidate-mode leading_pt \
        --device cuda \
        --no-wandb > "${LOG_DIR}/${padded}.out" 2> "${LOG_DIR}/${padded}.err"; then
    echo "[$(date -Is)] ${padded}: completed"
  else
    echo "[$(date -Is)] ${padded}: failed; see ${LOG_DIR}/${padded}.err" >&2
    return 1
  fi
}

failures=0
pids=()
for ((index = 0; index < count; index++)); do
  evaluate_index "${index}" &
  pids+=("$!")
  if ((${#pids[@]} >= PARALLEL_EVALUATIONS)); then
    for pid in "${pids[@]}"; do
      if ! wait "${pid}"; then
        failures=$((failures + 1))
      fi
    done
    pids=()
  fi
done
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    failures=$((failures + 1))
  fi
done

if ((failures > 0)); then
  echo "${failures} evaluations failed; rerun this script after resolving them." >&2
  exit 1
fi

"${CONDA_ENV}/bin/python" scripts/build_orbit_full500_higgs_plots.py \
  --full-manifest "${MANIFEST}" \
  --near-template "${NEAR_TEMPLATE}" \
  --cache-dir "${CACHE_DIR}" \
  --output-dir "${OUTPUT_ROOT}/higgs" \
  --bootstrap-replicas 50
