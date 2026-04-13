#!/usr/bin/env bash
# Step3：自筛选打分 + 导出 SFT 子集（在作业仓库根目录执行）
set -euo pipefail
set -x

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# Step2 增广结果（含 generated_instruction + lima_completion）
INPUT="${INPUT:-${ROOT}/step2/lima_step2_augmented.jsonl}"

# 评分模型：作业要求 Qwen3-1.7B（可用 HF 基座或你本地路径）
MODEL="${MODEL:-/hpc2hdd/home/yliu167/models/Qwen3-1___7B}"
TOKENIZER_MODEL="${TOKENIZER_MODEL:-/hpc2hdd/home/yliu167/models/Qwen3-1___7B}"
# 若 M0 为 LoRA：export ADAPTER=/path/to/m0_lora
ADAPTER="${ADAPTER:-}"

EXTRA=(--tokenizer-model "${TOKENIZER_MODEL}")
if [[ -n "${ADAPTER}" ]]; then
  EXTRA+=(--adapter "${ADAPTER}")
fi

python step3/score_curation_vllm.py \
  --input "${INPUT}" \
  --output-scored "${ROOT}/step3/lima_curated_scored.jsonl" \
  --output-sft "${ROOT}/step3/lima_curated_for_sft_sharegpt.jsonl" \
  --model "${MODEL}" \
  "${EXTRA[@]}" \
  --min-score "${MIN_SCORE:-4}" \
  --print-extremes 5

echo "Step3 完成: scored=${ROOT}/step3/lima_curated_scored.jsonl  sft=${ROOT}/step3/lima_curated_for_sft_sharegpt.jsonl"
echo "上传到 HF: 设置 HF_TOKEN 后运行 python step3/push_hf_curated.py --jsonl ... --repo-id <你的用户名/数据集名>"
