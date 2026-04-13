#!/usr/bin/env bash
# Step2
# set -euo pipefail
set -x

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# vLLM loaded model
BASE_MODEL="${BASE_MODEL:-/hpc2hdd/home/yliu167/models_trained/qwen3-1_7b-backward-merged}"

# tokenizer loaded from the original base
TOKENIZER_MODEL="${TOKENIZER_MODEL:-/hpc2hdd/home/yliu167/models/Qwen3-1___7B}"

ADAPTER="${ADAPTER:-}"
VLLM_DTYPE="${VLLM_DTYPE:-auto}"

EXTRA=(--dtype "${VLLM_DTYPE}" --tokenizer-model "${TOKENIZER_MODEL}")
if [[ -n "${ADAPTER}" ]]; then
  EXTRA+=(--adapter "${ADAPTER}")
fi
if [[ -n "${TP_SIZE:-}" ]]; then
  EXTRA+=(--tensor-parallel-size "${TP_SIZE}")
fi

python step2/generate_step2_instructions.py \
  --input "${ROOT}/step2/lima_step2_sample150.jsonl" \
  --output "${ROOT}/step2/lima_step2_augmented.jsonl" \
  --model "${BASE_MODEL}" \
  "${EXTRA[@]}" \
  --print-examples 5

echo "Step2 output: ${ROOT}/step2/lima_step2_augmented.jsonl (contains generated_instruction + lima_completion)"
