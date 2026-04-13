#!/usr/bin/env bash
# 将 Step4 正向 LoRA 与 Qwen3-1.7B 基座合并，便于上传 HF 或统一部署推理。
set -euo pipefail
set -x

LLAMAFACTORY_ROOT="${LLAMAFACTORY_ROOT:-/hpc2hdd/home/yliu167/LlamaFactory}"
BASE_MODEL="${BASE_MODEL:-/hpc2hdd/home/yliu167/models/Qwen3-1___7B}"
ADAPTER="${ADAPTER:-/hpc2hdd/home/yliu167/LlamaFactory/saves/qwen3-1.7b-step4-forward-lora}"
EXPORT_DIR="${EXPORT_DIR:-/hpc2hdd/home/yliu167/LlamaFactory/saves/qwen3-1.7b-step4-forward-merged}"
TEMPLATE="${TEMPLATE:-qwen3}"
EXPORT_DEVICE="${EXPORT_DEVICE:-cpu}"

cd "${LLAMAFACTORY_ROOT}"

llamafactory-cli export examples/merge_lora/qwen3_lora_sft.yaml \
  model_name_or_path="${BASE_MODEL}" \
  adapter_name_or_path="${ADAPTER}" \
  export_dir="${EXPORT_DIR}" \
  template="${TEMPLATE}" \
  export_device="${EXPORT_DEVICE}"

echo "合并完成: ${EXPORT_DIR}"
