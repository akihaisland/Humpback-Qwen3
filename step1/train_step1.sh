#!/bin/bash
set -euo pipefail
set -x
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export SWANLAB_MODE=offline
export SWANLAB_LOG_DIR=/hpc2hdd/home/yliu167/projects/dsaa6000q/swanlab_log/a3-step1
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NPROC_PER_NODE=4

cd /hpc2hdd/home/yliu167/LlamaFactory

MODEL_PATH=/hpc2hdd/home/yliu167/models/Qwen3-1___7B
STEP1_YAML=/hpc2hdd/home/yliu167/projects/dsaa6000q/step1/train_qwen3_backward_lora.yaml

FORCE_TORCHRUN=1 llamafactory-cli train "${STEP1_YAML}" \
  model_name_or_path=${MODEL_PATH} \
  dataset="openassistant_backward_step1" \
  use_swanlab=true \
  swanlab_project=DSAA6000Q \
  swanlab_run_name=Qwen3-1_7B-A3-Step1 \
  num_train_epochs=1 \
  max_samples=20000