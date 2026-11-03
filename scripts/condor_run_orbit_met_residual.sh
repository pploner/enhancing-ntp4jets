#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=${1:?project directory is required}
RUN_DIR=${2:?checkpoint run directory is required}
OUTPUT_DIR=${3:?output directory is required}
MANIFEST_DIR=${4:?manifest directory is required}
CONDA_ENV=${CONDA_ENV:-/eos/home-y/yelberke/conda_condor_orbit_env}
DEVICE=${DEVICE:-cuda}
BATCH_SIZE=${BATCH_SIZE:-16}

cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}/vqtorch:${PROJECT_DIR}:${PYTHONPATH:-}"

conda run --no-capture-output -p "${CONDA_ENV}" \
  python scripts/evaluate_orbit_met.py \
  --run-dir "${RUN_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --events 10000 \
  --batch-size "${BATCH_SIZE}" \
  --device "${DEVICE}" \
  --test-class "QCD=${MANIFEST_DIR}/QCD_HT50tobb_test.txt,${MANIFEST_DIR}/QCD_HT50toInf_test.txt" \
  --test-class "tt=${MANIFEST_DIR}/tt0123j_5f_ckm_LO_MLM_hadronic_test.txt,${MANIFEST_DIR}/tt0123j_5f_ckm_LO_MLM_leptonic_test.txt,${MANIFEST_DIR}/tt0123j_5f_ckm_LO_MLM_semiLeptonic_test.txt" \
  --test-class "VJets=${MANIFEST_DIR}/WJetsToLNu_13TeV-madgraphMLM-pythia8_test.txt,${MANIFEST_DIR}/WJetsToQQ_13TeV-madgraphMLM-pythia8_test.txt,${MANIFEST_DIR}/ZJetsTobb_13TeV-madgraphMLM-pythia8_test.txt,${MANIFEST_DIR}/ZJetsTocc_13TeV-madgraphMLM-pythia8_test.txt,${MANIFEST_DIR}/ZJetsToQQ_13TeV-madgraphMLM-pythia8_test.txt,${MANIFEST_DIR}/ZJetsTovv_13TeV-madgraphMLM-pythia8_test.txt" \
  --test-class "VV=${MANIFEST_DIR}/WW_hadronic_test.txt,${MANIFEST_DIR}/WW_leptonic_test.txt,${MANIFEST_DIR}/WW_semileptonic_test.txt,${MANIFEST_DIR}/WZ_hadronic_test.txt,${MANIFEST_DIR}/WZ_leptonic_test.txt,${MANIFEST_DIR}/WZ_semileptonic_test.txt,${MANIFEST_DIR}/ZZ_hadronic_test.txt,${MANIFEST_DIR}/ZZ_leptonic_test.txt,${MANIFEST_DIR}/ZZ_semileptonic_test.txt" \
  --test-class "DY=${MANIFEST_DIR}/DYJetsToLL_13TeV-madgraphMLM-pythia8_test.txt"
