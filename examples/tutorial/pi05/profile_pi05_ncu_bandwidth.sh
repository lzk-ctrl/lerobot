#!/usr/bin/env bash

set -eo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

MODEL_ID="${MODEL_ID:-/home/lzk/pi05_base}"
DEVICE="${DEVICE:-cuda:1}"
STEPS="${STEPS:-10}"
NUM_ACTION_SAMPLES="${NUM_ACTION_SAMPLES:-1}"
OBSERVATION_JSON="${OBSERVATION_JSON:-/home/lzk/lerobot/observation_with_image.json}"
OUTPUT_BASE="${OUTPUT_BASE:-${REPO_ROOT}/tmp/pi05_ncu_denoise_bandwidth}"
WARMUP_RUNS="${WARMUP_RUNS:-1}"
PROFILE_RUNS="${PROFILE_RUNS:-1}"
NVTX_INCLUDE="${NVTX_INCLUDE:-pi05.model.denoise_step/}"
NCU_SECTIONS="${NCU_SECTIONS:-SpeedOfLight,MemoryWorkloadAnalysis}"
NCU_METRICS="${NCU_METRICS:-}"
NCU_LAUNCH_COUNT="${NCU_LAUNCH_COUNT:-}"
NCU_LAUNCH_SKIP="${NCU_LAUNCH_SKIP:-}"

mkdir -p "${REPO_ROOT}/tmp/ncu_home" "$(dirname "${OUTPUT_BASE}")"

export HOME="${REPO_ROOT}/tmp/ncu_home"

NCU_ARGS=(
  --target-processes all
  --nvtx
  --nvtx-include "${NVTX_INCLUDE}"
  --print-details all
  --force-overwrite
  --export "${OUTPUT_BASE}"
)

if [[ -n "${NCU_METRICS}" ]]; then
  NCU_ARGS+=(--metrics "${NCU_METRICS}")
else
  IFS=',' read -r -a SECTION_LIST <<< "${NCU_SECTIONS}"
  for section in "${SECTION_LIST[@]}"; do
    NCU_ARGS+=(--section "${section}")
  done
fi

if [[ -n "${NCU_LAUNCH_COUNT}" ]]; then
  NCU_ARGS+=(--launch-count "${NCU_LAUNCH_COUNT}")
fi
if [[ -n "${NCU_LAUNCH_SKIP}" ]]; then
  NCU_ARGS+=(--launch-skip "${NCU_LAUNCH_SKIP}")
fi

exec ncu \
  "${NCU_ARGS[@]}" \
  python "${REPO_ROOT}/examples/tutorial/pi05/profile_pi05_inference.py" \
    --model-id "${MODEL_ID}" \
    --device "${DEVICE}" \
    --observation-json "${OBSERVATION_JSON}" \
    --num-inference-steps "${STEPS}" \
    --num-action-samples "${NUM_ACTION_SAMPLES}" \
    --warmup-runs "${WARMUP_RUNS}" \
    --profile-runs "${PROFILE_RUNS}"
