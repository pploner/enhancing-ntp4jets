#!/usr/bin/env bash
set -euo pipefail

SELECTION=${1:?selection required}
SUITE=${2:?suite required}
PROJECT_DIR=${3:?project directory required}
CACHE_DIR=${4:?cache directory required}
MANIFEST_DIR=${5:?manifest directory required}
CONDA_ENV=${CONDA_ENV:-/eos/home-y/yelberke/conda_condor_orbit_env}

cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}/vqtorch:${PROJECT_DIR}:${PYTHONPATH:-}"
exec conda run --no-capture-output -p "${CONDA_ENV}" \
  python scripts/evaluate_orbit_5m_plot_cache.py \
  --registry campaigns/orbit_campaign_5m_plot_models.yaml \
  --selection "${SELECTION}" --suite "${SUITE}" \
  --manifest-dir "${MANIFEST_DIR}" --output-dir "${CACHE_DIR}" \
  --events 10000 --batch-size 16 --device cuda
