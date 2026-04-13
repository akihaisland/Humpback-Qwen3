#!/usr/bin/env python3
"""
Step2：用 Step1 反向模型，对 LIMA completion 生成指令 x'，构成 (x', y_LIMA) 自增广对。

使用 **vLLM** 做批量推理（较 Transformers.generate 吞吐更高）。仍需 transformers 的
tokenizer 以调用 apply_chat_template，与训练侧 chat 格式对齐。

依赖：pip install vllm transformers
（vLLM 通常仅支持 Linux + CUDA；无 GPU 时请改用其它推理方式。）

若 --model 为「合并后的目录」，请务必另设 --tokenizer-model 指向原始 Qwen3 基座（含完整
tokenizer 文件）。合并导出目录常损坏 Fast 配置且缺少慢分词器所需文件，仅改 use_fast 往往无效。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from transformers import AutoTokenizer

try:
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
except ImportError as e:  # pragma: no cover
    raise SystemExit(
        "未安装 vLLM 或导入失败。请先安装：pip install vllm\n"
        f"原始错误: {e}"
    ) from e


# 与 scripts/prepare_step1_sharegpt.py 中 en 模板一致（{completion} 占位）
BACKWARD_USER_PROMPT_EN = (
    "You are training for instruction backtranslation: predict the user instruction "
    "that would elicit the following assistant response.\n\n"
    "Assistant response:\n{completion}\n\n"
    "User instruction (write only the instruction, no preamble):"
)

# Qwen3 推理块：`` ... ``（用字节拼接，避免编辑器吞掉特殊 token）
_TB = bytes([0x3C, 0x7C, 0x74, 0x68, 0x69, 0x6E, 0x6B, 0x7C, 0x3E]).decode("ascii")
_TE = bytes([0x3C, 0x7C, 0x2F, 0x74, 0x68, 0x69, 0x6E, 0x6B, 0x7C, 0x3E]).decode("ascii")
_THINK_PIPE_BLOCK = re.compile(
    re.escape(_TB) + r"[\s\S]*?" + re.escape(_TE),
    re.DOTALL | re.IGNORECASE,
)


def strip_think_and_special(text: str) -> str:
    """去掉推理块与 chat 模板残留。"""
    t = _THINK_PIPE_BLOCK.sub("", text)
    for end_mark in ("</think>",):
        if end_mark in t:
            t = t.split(end_mark, 1)[-1]
    t = re.sub(r"<\|im_start\|>assistant\s*", "", t, flags=re.IGNORECASE)
    t = re.sub(r"<\|im_end\|>", "", t, flags=re.IGNORECASE)
    return t.strip()


def load_tokenizer(pretrained_dir: str, *, use_fast: bool = True):
    """
    合并目录 tokenizer 常不兼容：先 Fast，再 Slow；仍失败则提示改用基座 --tokenizer-model。
    """
    kwargs = {"trust_remote_code": True, "use_fast": use_fast}
    try:
        return AutoTokenizer.from_pretrained(pretrained_dir, **kwargs)
    except (AttributeError, TypeError, ValueError, OSError) as e:
        if use_fast:
            print(
                f"[warn] Tokenizer(use_fast=True) 失败: {e}\n"
                f"[warn] 改用 use_fast=False 重试（路径: {pretrained_dir}）",
                file=sys.stderr,
            )
            return load_tokenizer(pretrained_dir, use_fast=False)
        raise SystemExit(
            "无法从当前目录加载 tokenizer。合并导出目录常与 transformers 不兼容，且可能缺少\n"
            "慢分词器所需的 vocab.json。\n\n"
            "请增加参数: --tokenizer-model <原始 Qwen3 基座目录>\n"
            "（与 vLLM 的 --model 合并权重路径分开；基座应与训练 Step1 时一致。）\n\n"
            f"已尝试路径: {pretrained_dir}\n"
            f"最后一次错误: {type(e).__name__}: {e}"
        ) from e


def build_chat_input(tokenizer, user_text: str) -> str:
    messages = [{"role": "user", "content": user_text}]
    kwargs: dict = {"tokenize": False, "add_generation_prompt": True}
    try:
        return tokenizer.apply_chat_template(messages, **kwargs, enable_thinking=False)  # type: ignore[call-arg]
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--input",
        type=Path,
        default=Path("step2/lima_step2_sample150.jsonl"),
        help="prepare_lima_step2.py 输出",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("step2/lima_step2_augmented.jsonl"),
        help="带 generated_instruction 的输出",
    )
    p.add_argument("--model", type=Path, required=True, help="HF 格式的基座模型目录（或已 merge 的完整权重），供 vLLM 加载")
    p.add_argument(
        "--tokenizer-model",
        type=Path,
        default=None,
        help="强烈建议：原始 Qwen3 基座目录。会同时用于 (1) apply_chat_template (2) vLLM 的 tokenizer=…；"
        "合并目录仅作 --model 时若不设此项，vLLM 仍会从合并目录加载 tokenizer 并报错。",
    )
    p.add_argument("--adapter", type=Path, default=None, help="可选：Step1 LoRA adapter 目录（vLLM 动态加载）")
    p.add_argument(
        "--dtype",
        choices=("auto", "half", "bfloat16", "float16"),
        default="auto",
        help="vLLM 权重精度；GPU 不支持 bf16 时可试 half/float16",
    )
    p.add_argument("--tensor-parallel-size", type=int, default=1, help="张量并行 GPU 数")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.90, help="vLLM 显存占用上限比例")
    p.add_argument("--max-model-len", type=int, default=12288, help="vLLM 最大上下文长度")
    p.add_argument("--max-lora-rank", type=int, default=64, help="需 >= Step1 训练时的 lora_rank")
    p.add_argument("--batch-size", type=int, default=16, help="每批送入 vLLM 的样本数")
    p.add_argument("--print-examples", type=int, default=5, help="打印几条生成指令示例（0 关闭）")
    p.add_argument("--max-new-tokens", type=int, default=384)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.9)
    args = p.parse_args()

    if not args.input.is_file():
        raise SystemExit(f"找不到输入: {args.input}")

    model_dir = str(args.model)
    tok_dir = str(args.tokenizer_model) if args.tokenizer_model is not None else model_dir
    if args.tokenizer_model is None and "merged" in model_dir.lower():
        raise SystemExit(
            "当前 --model 路径名包含 merged，但未指定 --tokenizer-model。\n"
            "vLLM 在初始化时也会从模型目录加载 tokenizer，合并目录会触发与 transformers 相同的错误。\n\n"
            "请增加：--tokenizer-model <原始 Qwen3 基座目录>\n"
            "（与训练 Step1 使用的基座一致；run_step2.sh 里已默认设置 TOKENIZER_MODEL。）"
        )

    tokenizer = load_tokenizer(tok_dir)
    if tok_dir != model_dir:
        print(
            f"[info] tokenizer 与 vLLM 均使用目录: {tok_dir}\n"
            f"[info] 仅权重自: {model_dir}",
            file=sys.stderr,
        )

    rows: list[dict] = []
    with args.input.open(encoding="utf-8") as fin:
        for idx, line in enumerate(fin):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            comp = row.get("lima_completion") or row.get("completion")
            if not isinstance(comp, str) or not comp.strip():
                print(f"[warn] line {idx + 1}: 缺少 lima_completion，跳过", file=sys.stderr)
                continue
            row["_completion_for_prompt"] = comp.strip()
            rows.append(row)

    if not rows:
        raise SystemExit("没有可推理的样本")

    llm_kw: dict = {
        "model": model_dir,
        # 关键：vLLM 内部也会加载 tokenizer；必须与上面 load_tokenizer 使用同一基座路径，
        # 否则仍会从合并目录读 tokenizer 并报 extra_special_tokens 错误。
        "tokenizer": tok_dir,
        "trust_remote_code": True,
        "tensor_parallel_size": args.tensor_parallel_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_model_len": args.max_model_len,
        "dtype": args.dtype,
    }
    lora_request: LoRARequest | None = None
    if args.adapter is not None:
        llm_kw["enable_lora"] = True
        llm_kw["max_lora_rank"] = args.max_lora_rank
        llm_kw["max_loras"] = 4
        lora_request = LoRARequest("backward_step2", 1, str(args.adapter.resolve()))

    print("[info] 初始化 vLLM …", file=sys.stderr)
    llm = LLM(**llm_kw)

    if args.temperature <= 0:
        sampling = SamplingParams(
            max_tokens=args.max_new_tokens,
            temperature=0.0,
            top_p=1.0,
        )
    else:
        sampling = SamplingParams(
            max_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    printed = 0
    bs = max(1, args.batch_size)

    with args.output.open("w", encoding="utf-8") as fout:
        for start in range(0, len(rows), bs):
            batch = rows[start : start + bs]
            prompts = [
                build_chat_input(
                    tokenizer,
                    BACKWARD_USER_PROMPT_EN.format(completion=r["_completion_for_prompt"]),
                )
                for r in batch
            ]
            gen_kw: dict = {}
            if lora_request is not None:
                gen_kw["lora_request"] = lora_request
            outputs = llm.generate(prompts, sampling, **gen_kw)
            for row, out in zip(batch, outputs):
                raw = out.outputs[0].text
                gen = strip_think_and_special(raw)
                row.pop("_completion_for_prompt", None)
                row["generated_instruction"] = gen
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")

                if args.print_examples > 0 and printed < args.print_examples:
                    printed += 1
                    comp = row.get("lima_completion") or ""
                    print("=" * 72, file=sys.stderr)
                    print(
                        f"[示例 {printed}/{args.print_examples}] generated_instruction:\n{gen}\n",
                        file=sys.stderr,
                    )
                    print(
                        f"[同一示例] lima_completion（前 400 字）:\n{comp[:400]}{'…' if len(comp) > 400 else ''}\n",
                        file=sys.stderr,
                    )
                    if row.get("lima_instruction"):
                        li = row["lima_instruction"]
                        print(
                            f"[对照] 原始 LIMA instruction:\n{li[:400]}{'…' if len(li) > 400 else ''}\n",
                            file=sys.stderr,
                        )

    print(f"完成，写出 -> {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
