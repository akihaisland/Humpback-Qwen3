#!/usr/bin/env python3
"""
将 step3 产出的 ShareGPT jsonl 推到 Hugging Face Hub（作业 readme 要求粘贴数据集 URL）。

需已登录: `huggingface-cli login` 或环境变量 HF_TOKEN。

用法:
  python step3/push_hf_curated.py \\
    --jsonl step3/lima_curated_for_sft_sharegpt.jsonl \\
    --repo-id yourname/lima-qwen3-curated-step3 \\
    --private
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--jsonl", type=Path, required=True)
    p.add_argument("--repo-id", type=str, required=True)
    p.add_argument("--private", action="store_true", help="创建为私有库")
    p.add_argument("--max-rows", type=int, default=None, help="调试用：最多上传多少条")
    args = p.parse_args()

    if not args.jsonl.is_file():
        raise SystemExit(f"找不到文件: {args.jsonl}")

    try:
        from datasets import Dataset
    except ImportError as e:
        raise SystemExit("请安装: pip install datasets huggingface_hub\n" + str(e)) from e

    rows: list[dict] = []
    with args.jsonl.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            if args.max_rows is not None and i >= args.max_rows:
                break
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    if not rows:
        raise SystemExit("jsonl 为空")

    ds = Dataset.from_list(rows)
    print(f"[info] 上传 {len(rows)} 条 -> {args.repo_id} …", file=sys.stderr)
    ds.push_to_hub(args.repo_id, private=args.private)
    print(f"[done] https://huggingface.co/datasets/{args.repo_id}", file=sys.stderr)


if __name__ == "__main__":
    main()
