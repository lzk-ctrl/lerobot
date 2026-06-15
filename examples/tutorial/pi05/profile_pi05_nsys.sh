#!/usr/bin/env bash

set -eo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

MODEL_ID="${MODEL_ID:-/home/lzk/pi05_base}"
DEVICE="${DEVICE:-cuda:1}"
STEPS="${STEPS:-10}"
NUM_ACTION_SAMPLES="${NUM_ACTION_SAMPLES:-4}"
ACTIONS_PER_CHUNK="${ACTIONS_PER_CHUNK:-1}"
OBSERVATION_JSON="${OBSERVATION_JSON:-/home/lzk/lerobot/observation_with_image.json}"
OUTPUT_BASE="${OUTPUT_BASE:-${REPO_ROOT}/tmp/pi05_nsys_n${NUM_ACTION_SAMPLES}}"
WARMUP_RUNS="${WARMUP_RUNS:-3}"
PROFILE_RUNS="${PROFILE_RUNS:-1}"
NSYS_TRACE="${NSYS_TRACE:-cuda,nvtx,osrt,cublas,cudnn}"
GPU_METRICS_FREQUENCY="${GPU_METRICS_FREQUENCY:-10000}"
GPU_METRICS_DEVICE="${GPU_METRICS_DEVICE:-}"

mkdir -p "$(dirname "${OUTPUT_BASE}")"

if [[ -z "${GPU_METRICS_DEVICE}" && "${DEVICE}" =~ ^cuda:([0-9]+)$ ]]; then
  GPU_METRICS_DEVICE="${BASH_REMATCH[1]}"
fi

NSYS_ARGS=(
  profile
  --trace "${NSYS_TRACE}"
  --cuda-memory-usage=true
  --force-overwrite=true
  -o "${OUTPUT_BASE}"
)

if [[ -n "${GPU_METRICS_DEVICE}" ]]; then
  NSYS_ARGS+=(--gpu-metrics-device "${GPU_METRICS_DEVICE}")
  NSYS_ARGS+=(--gpu-metrics-frequency "${GPU_METRICS_FREQUENCY}")
fi

exec nsys \
  "${NSYS_ARGS[@]}" \
  python "${REPO_ROOT}/examples/tutorial/pi05/profile_pi05_inference.py" \
    --model-id "${MODEL_ID}" \
    --device "${DEVICE}" \
    --observation-json "${OBSERVATION_JSON}" \
    --num-inference-steps "${STEPS}" \
    --num-action-samples "${NUM_ACTION_SAMPLES}" \
    --actions-per-chunk "${ACTIONS_PER_CHUNK}" \
    --warmup-runs "${WARMUP_RUNS}" \
    --profile-runs "${PROFILE_RUNS}"
