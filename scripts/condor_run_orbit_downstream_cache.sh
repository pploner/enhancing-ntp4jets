#!/usr/bin/env bash
set -euo pipefail

MODE=${1:?evaluation mode is required}
PROJECT_DIR=${2:?project directory is required}
RUN_DIR=${3:?checkpoint run directory is required}
CACHE_DIR=${4:?cache directory is required}
MANIFEST_DIR=${5:?manifest directory is required}
CONDA_ENV=${CONDA_ENV:-/eos/home-y/yelberke/conda_condor_orbit_env}
DEVICE=${DEVICE:-cuda}
BATCH_SIZE=${BATCH_SIZE:-16}

cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}/vqtorch:${PROJECT_DIR}:${PYTHONPATH:-}"

SM_CLASSES=(
  --test-class "QCD=${MANIFEST_DIR}/QCD_HT50tobb_test.txt,${MANIFEST_DIR}/QCD_HT50toInf_test.txt"
  --test-class "tt=${MANIFEST_DIR}/tt0123j_5f_ckm_LO_MLM_hadronic_test.txt,${MANIFEST_DIR}/tt0123j_5f_ckm_LO_MLM_leptonic_test.txt,${MANIFEST_DIR}/tt0123j_5f_ckm_LO_MLM_semiLeptonic_test.txt"
  --test-class "VJets=${MANIFEST_DIR}/WJetsToLNu_13TeV-madgraphMLM-pythia8_test.txt,${MANIFEST_DIR}/WJetsToQQ_13TeV-madgraphMLM-pythia8_test.txt,${MANIFEST_DIR}/ZJetsTobb_13TeV-madgraphMLM-pythia8_test.txt,${MANIFEST_DIR}/ZJetsTocc_13TeV-madgraphMLM-pythia8_test.txt,${MANIFEST_DIR}/ZJetsToQQ_13TeV-madgraphMLM-pythia8_test.txt,${MANIFEST_DIR}/ZJetsTovv_13TeV-madgraphMLM-pythia8_test.txt"
  --test-class "VV=${MANIFEST_DIR}/WW_hadronic_test.txt,${MANIFEST_DIR}/WW_leptonic_test.txt,${MANIFEST_DIR}/WW_semileptonic_test.txt,${MANIFEST_DIR}/WZ_hadronic_test.txt,${MANIFEST_DIR}/WZ_leptonic_test.txt,${MANIFEST_DIR}/WZ_semileptonic_test.txt,${MANIFEST_DIR}/ZZ_hadronic_test.txt,${MANIFEST_DIR}/ZZ_leptonic_test.txt,${MANIFEST_DIR}/ZZ_semileptonic_test.txt"
  --test-class "DY=${MANIFEST_DIR}/DYJetsToLL_13TeV-madgraphMLM-pythia8_test.txt"
)

TT_CLASSES=(
  --test-class "tt=${MANIFEST_DIR}/tt0123j_5f_ckm_LO_MLM_hadronic_test.txt,${MANIFEST_DIR}/tt0123j_5f_ckm_LO_MLM_leptonic_test.txt,${MANIFEST_DIR}/tt0123j_5f_ckm_LO_MLM_semiLeptonic_test.txt"
)

case "${MODE}" in
  higgs)
    conda run --no-capture-output -p "${CONDA_ENV}" \
      python scripts/evaluate_orbit_higgs_mass.py \
      --run-dir "${RUN_DIR}" \
      --gghbb-test-manifest "${MANIFEST_DIR}/ggHbb_test.txt" \
      --output-dir "${CACHE_DIR}/higgs" \
      --events 10000 --batch-size "${BATCH_SIZE}" --bootstrap-replicas 0 \
      --candidate-mode leading_pt --device "${DEVICE}" --no-wandb
    ;;
  tau32)
    conda run --no-capture-output -p "${CONDA_ENV}" \
      python scripts/evaluate_orbit_tau32.py \
      --run-dir "${RUN_DIR}" \
      --test-manifest "${MANIFEST_DIR}/tt0123j_5f_ckm_LO_MLM_hadronic_test.txt" \
      --output-dir "${CACHE_DIR}/tau32" \
      --events 10000 --batch-size "${BATCH_SIZE}" --device "${DEVICE}"
    ;;
  jets)
    conda run --no-capture-output -p "${CONDA_ENV}" \
      python scripts/evaluate_orbit_jet_residuals.py \
      --run-dir "${RUN_DIR}" "${SM_CLASSES[@]}" \
      --output-dir "${CACHE_DIR}/jets" \
      --events 10000 --batch-size "${BATCH_SIZE}" --device "${DEVICE}"
    ;;
  pid)
    conda run --no-capture-output -p "${CONDA_ENV}" \
      python scripts/evaluate_orbit_pid_pulls.py \
      --run-dir "${RUN_DIR}" "${SM_CLASSES[@]}" \
      --output-dir "${CACHE_DIR}/sm_pid" \
      --events 10000 --batch-size "${BATCH_SIZE}" --device "${DEVICE}"
    ;;
  met)
    conda run --no-capture-output -p "${CONDA_ENV}" \
      python scripts/evaluate_orbit_met.py \
      --run-dir "${RUN_DIR}" "${SM_CLASSES[@]}" \
      --output-dir "${CACHE_DIR}/met" \
      --events 10000 --batch-size "${BATCH_SIZE}" --device "${DEVICE}"
    ;;
  particles)
    conda run --no-capture-output -p "${CONDA_ENV}" \
      python scripts/evaluate_orbit_particle_residuals.py \
      --run-dir "${RUN_DIR}" "${TT_CLASSES[@]}" \
      --output-dir "${CACHE_DIR}/ttbar_particles" \
      --events 10000 --batch-size "${BATCH_SIZE}" --device "${DEVICE}"
    ;;
  *)
    echo "Unknown evaluation mode: ${MODE}" >&2
    exit 2
    ;;
esac
