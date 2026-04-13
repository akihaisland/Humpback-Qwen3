#!/usr/bin/env bash
# 使用 LLaMA-Factory 将 Step1 反向 LoRA 与基座权重合并为完整模型（便于 vLLM 等直接加载）。
# 在服务器上于 LlamaFactory 仓库根目录执行（或本脚本 cd 到该目录）。
#
# 参考：examples/merge_lora/qwen3_lora_sft.yaml
# 用法：
#   chmod +x step1/merge_backward_lora.sh
#   ./step1/merge_backward_lora.sh
# 或覆盖环境变量：
#   BASE_MODEL=/path/to/Qwen3 ADAPTER=/path/to/lora EXPORT_DIR=/path/to/out ./step1/merge_backward_lora.sh

set -euo pipefail
set -x

# LlamaFactory 安装根目录（含 examples/、llamafactory-cli）
LLAMAFACTORY_ROOT="${LLAMAFACTORY_ROOT:-/hpc2hdd/home/yliu167/LlamaFactory}"

# 基座模型（与训练 Step1 时一致）
BASE_MODEL="${BASE_MODEL:-/hpc2hdd/home/yliu167/models/Qwen3-1___7B}"

# Step1 训练输出目录（内含 adapter 与 checkpoint）
ADAPTER="${ADAPTER:-/hpc2hdd/home/yliu167/LlamaFactory/saves/qwen3-1.7b-backward-lora}"

# 合并后权重保存目录（勿与 ADAPTER 相同）
EXPORT_DIR="${EXPORT_DIR:-/hpc2hdd/home/yliu167/models_trained/qwen3-1_7b-backward-merged}"

# merge 时 tokenizer/chat 模板；与训练时 qwen3 一致即可，若导出报错可改为 qwen3_nothink
TEMPLATE="${TEMPLATE:-qwen3}"

# 官方示例为 cpu，合并更稳；有 GPU 且显存充足可改为 auto
EXPORT_DEVICE="${EXPORT_DEVICE:-cpu}"

cd "${LLAMAFACTORY_ROOT}"

llamafactory-cli export examples/merge_lora/qwen3_lora_sft.yaml \
  model_name_or_path="${BASE_MODEL}" \
  adapter_name_or_path="${ADAPTER}" \
  export_dir="${EXPORT_DIR}" \
  template="${TEMPLATE}" \
  export_device="${EXPORT_DEVICE}"

echo "合并完成，输出目录: ${EXPORT_DIR}"
