#!/usr/bin/env python3
"""
将 Step4 训练产出的 LoRA 适配器目录上传到 Hugging Face Model Hub（体积小，无需传合并全量）。

使用方需在本地/推理端加载与训练一致的基座模型，再挂载本仓库中的 LoRA（与 print 脚本里
「基座 + --adapter」用法一致）。

默认路径与 step4/merge_step4_forward_lora.sh 中的 ADAPTER 一致；若仍要上传合并后的全量，
请显式传入: --folder <merge 输出目录>。

需已登录: `huggingface-cli login` 或环境变量 HF_TOKEN。

用法示例:
  python step4/push_hf_step4_model.py --repo-id yourname/qwen3-1.7b-step4-forward-lora

  python step4/push_hf_step4_model.py --folder /path/to/lora --repo-id yourname/xxx --private

环境变量（可选）:
  STEP4_LORA_DIR           默认 --folder（LoRA 保存目录）
  HF_STEP4_REPO_ID         默认 --repo-id（可与命令行二选一，命令行优先）
  HF_STEP4_BASE_MODEL_ID   默认 --hf-base-model（写入 README 的合法 HF 模型 id）

说明:
  训练目录里的 README.md 常把 base_model 写成本地路径，Hub 会校验失败。
  本脚本上传时会忽略原 README.md，并单独上传一份带合法 base_model 的简短说明。
"""

from __future__ import annotations

import argparse
import inspect
import io
import os
import sys
from pathlib import Path


# 与 merge_step4_forward_lora.sh 中 ADAPTER 默认一致
_DEFAULT_LORA = "/hpc2hdd/home/yliu167/LlamaFactory/saves/qwen3-1.7b-step4-forward-lora"


def _default_folder() -> Path:
    return Path(os.environ.get("STEP4_LORA_DIR", _DEFAULT_LORA))


def _default_repo_id() -> str | None:
    v = os.environ.get("HF_STEP4_REPO_ID", "").strip()
    return v or None


def _default_hf_base_model() -> str:
    return os.environ.get("HF_STEP4_BASE_MODEL_ID", "Qwen/Qwen3-1.7B").strip() or "Qwen/Qwen3-1.7B"


def _model_card_readme(base_model_id: str) -> str:
    # Hub 仅接受 hf.co/models 上的 model id，不可使用本地路径
    return f"""---
base_model: {base_model_id}
library_name: peft
pipeline_tag: text-generation
tags:
  - lora
---

# Step4 LoRA（正向 SFT）

本仓库为 **PEFT / LoRA** 权重，请在推理或继续训练时加载与 Step4 一致的基座:

**`{base_model_id}`**
"""


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--folder",
        type=Path,
        default=None,
        help=f"本地 LoRA 目录（默认: 环境变量 STEP4_LORA_DIR 或 {_DEFAULT_LORA}）",
    )
    p.add_argument(
        "--repo-id",
        type=str,
        default=None,
        help="HF 模型库 id，如 username/repo（默认: 环境变量 HF_STEP4_REPO_ID）",
    )
    p.add_argument("--private", action="store_true", help="创建为私有库（若库已存在则忽略）")
    p.add_argument(
        "--commit-message",
        type=str,
        default="Upload Step4 forward SFT LoRA (Qwen3-1.7B)",
    )
    p.add_argument(
        "--token",
        type=str,
        default=None,
        help="HF token（默认读环境变量 HUGGING_FACE_HUB_TOKEN / HF_TOKEN）",
    )
    p.add_argument(
        "--hf-base-model",
        type=str,
        default=None,
        help=(
            "写入 README 的 Hugging Face 基座模型 id（默认: 环境变量 HF_STEP4_BASE_MODEL_ID 或 "
            "Qwen/Qwen3-1.7B）。若你训练用的是 Base 权重，可改为 Qwen/Qwen3-1.7B-Base"
        ),
    )
    args = p.parse_args()

    folder = args.folder if args.folder is not None else _default_folder()
    repo_id = (args.repo_id or "").strip() or _default_repo_id()
    if not repo_id:
        raise SystemExit(
            "请指定目标仓库: --repo-id yourname/repo-name\n"
            "或设置环境变量: export HF_STEP4_REPO_ID=yourname/repo-name"
        )

    folder = folder.expanduser().resolve()
    if not folder.is_dir():
        raise SystemExit(
            f"找不到目录: {folder}\n"
            "请先完成 Step4 训练产出 LoRA，或用 --folder / STEP4_LORA_DIR 指定保存目录。"
        )

    try:
        from huggingface_hub import HfApi
    except ImportError as e:
        raise SystemExit("请安装: pip install huggingface_hub\n" + str(e)) from e

    token = args.token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    hf_base = (args.hf_base_model or "").strip() or _default_hf_base_model()

    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="model", private=args.private, exist_ok=True)

    print(
        f"[info] 上传 LoRA 目录 -> {folder}\n"
        f"[info] 目标: https://huggingface.co/{repo_id}\n"
        f"[info] README base_model -> {hf_base}（忽略本地训练目录中的 README.md，避免 Hub YAML 校验失败）\n"
        f"[hint] 推理时加载基座 {hf_base} + 本仓库 LoRA",
        file=sys.stderr,
    )
    # 不上传 LLaMA-Factory 生成的 README.md（其中 base_model 常为本地路径，会触发 validate-yaml 400）
    upload_kw: dict = dict(
        folder_path=str(folder),
        repo_id=repo_id,
        repo_type="model",
        commit_message=args.commit_message,
        token=token,
    )
    if "ignore_patterns" in inspect.signature(api.upload_folder).parameters:
        upload_kw["ignore_patterns"] = ["README.md"]
    else:
        print(
            "[warn] 当前 huggingface_hub 不支持 upload_folder(ignore_patterns=…)；"
            "若仍因 README.md 报错请升级: pip install -U huggingface_hub",
            file=sys.stderr,
        )
    api.upload_folder(**upload_kw)

    readme_buf = io.BytesIO(_model_card_readme(hf_base).encode("utf-8"))
    readme_buf.seek(0)
    api.upload_file(
        path_or_fileobj=readme_buf,
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="model",
        commit_message="Add Hub-valid README (base_model from --hf-base-model)",
        token=token,
    )
    print(f"[done] https://huggingface.co/{repo_id}", file=sys.stderr)


if __name__ == "__main__":
    main()
