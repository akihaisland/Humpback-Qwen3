#!/usr/bin/env bash
# 训练完成后打印 5 条示例回复（可改 MODEL / ADAPTER）
set -euo pipefail
set -x

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# 未 merge：基座 + LoRA
MODEL="${MODEL:-/hpc2hdd/home/yliu167/models/Qwen3-1___7B}"
ADAPTER="${ADAPTER:-/hpc2hdd/home/yliu167/LlamaFactory/saves/qwen3-1.7b-step4-forward-lora}"
TOKENIZER_MODEL="${TOKENIZER_MODEL:-/hpc2hdd/home/yliu167/models/Qwen3-1___7B}"
CURATED="${CURATED:-${ROOT}/step3/lima_curated_for_sft_sharegpt.jsonl}"

EXTRA=(--tokenizer-model "${TOKENIZER_MODEL}" --from-jsonl "${CURATED}" --n 5)
if [[ -n "${USE_MERGED_ONLY:-}" ]]; then
  MODEL="${MERGED_MODEL:-/hpc2hdd/home/yliu167/models_trained/qwen3-1.7b-step4-forward-merged}"
  ADAPTER=""
  EXTRA=(--tokenizer-model "${TOKENIZER_MODEL}" --from-jsonl "${CURATED}" --n 5)
fi

CMD=(python "${ROOT}/step4/print_example_responses.py" --model "${MODEL}" "${EXTRA[@]}")
if [[ -n "${ADAPTER}" ]]; then
  CMD+=(--adapter "${ADAPTER}")
fi
"${CMD[@]}"
