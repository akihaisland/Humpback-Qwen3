#!/usr/bin/env bash
# Step4：LLaMA-Factory 正向指令微调（数据集来自 Step3）
set -euo pipefail
set -x
export PYTHONUNBUFFERED=1

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# Run in the same way as train_step1.sh: in the LlamaFactory root directory
LLAMAFACTORY_ROOT="${LLAMAFACTORY_ROOT:-/hpc2hdd/home/yliu167/LlamaFactory}"
cd "${LLAMAFACTORY_ROOT}"

MODEL_PATH="${MODEL_PATH:-/hpc2hdd/home/yliu167/models/Qwen3-1___7B}"
STEP4_YAML="${STEP4_YAML:-${ROOT}/step4/train_qwen3_step4_forward_lora.yaml}"

DATASET="${DATASET:-lima_curated_step3_sft}"

FORCE_TORCHRUN="${FORCE_TORCHRUN:-1}" llamafactory-cli train "${STEP4_YAML}" \
  model_name_or_path="${MODEL_PATH}" \
  dataset="${DATASET}" \
  "$@"
