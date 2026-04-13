#!/usr/bin/env python3
"""Print 5 example responses after finetuning on Step3 data."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

try:
    from peft import PeftModel
except ImportError:
    PeftModel = None  # type: ignore[misc, assignment]

_TB = bytes([0x3C, 0x7C, 0x74, 0x68, 0x69, 0x6E, 0x6B, 0x7C, 0x3E]).decode("ascii")
_TE = bytes([0x3C, 0x7C, 0x2F, 0x74, 0x68, 0x69, 0x6E, 0x6B, 0x7C, 0x3E]).decode("ascii")
_THINK = re.compile(re.escape(_TB) + r"[\s\S]*?" + re.escape(_TE), re.DOTALL | re.IGNORECASE)


def load_tokenizer(path: str, use_fast: bool = True):
    kw = {"trust_remote_code": True, "use_fast": use_fast}
    try:
        return AutoTokenizer.from_pretrained(path, **kw)
    except (AttributeError, TypeError, OSError):
        if use_fast:
            return load_tokenizer(path, use_fast=False)
        raise


def strip_gen(text: str) -> str:
    t = _THINK.sub("", text)
    t = re.sub(r"<\|im_start\|>assistant\s*", "", t, flags=re.IGNORECASE)
    t = re.sub(r"<\|im_end\|>", "", t, flags=re.IGNORECASE)
    return t.strip()


def build_prompt(tokenizer, user: str) -> str:
    messages = [{"role": "user", "content": user.strip()}]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False  # type: ignore[call-arg]
        )
    except TypeError:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


DEFAULT_PROMPTS = [
    "What is the capital of France? Answer in one sentence.",
    "Explain what LoRA is in fine-tuning, in about 3 sentences.",
    "Write a short polite email declining a meeting due to a schedule conflict.",
    "List three best practices for writing clear user instructions for an LLM.",
    "What is the difference between precision and recall in binary classification?",
]


def collect_prompts(args: argparse.Namespace) -> list[str]:
    if args.from_jsonl is not None:
        prompts: list[str] = []
        with args.from_jsonl.open(encoding="utf-8") as f:
            for line in f:
                if len(prompts) >= args.n:
                    break
                row = json.loads(line)
                conv = row.get("conversations") or row.get("messages")
                if not isinstance(conv, list) or not conv:
                    continue
                first = conv[0]
                role = first.get("from") or first.get("role")
                val = first.get("value") or first.get("content")
                if val and (role in (None, "human", "Human", "user", "User")):
                    prompts.append(str(val).strip())
        if len(prompts) < args.n:
            print(
                f"[warn] 从 jsonl 只取到 {len(prompts)} 条，用默认问题补足。",
                file=sys.stderr,
            )
            for p in DEFAULT_PROMPTS:
                if len(prompts) >= args.n:
                    break
                if p not in prompts:
                    prompts.append(p)
        return prompts[: args.n]
    return DEFAULT_PROMPTS[: args.n]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True, help="基座或已 merge 的模型目录")
    p.add_argument("--adapter", type=Path, default=None, help="Step4 LoRA 输出目录（未 merge 时）")
    p.add_argument("--tokenizer-model", type=Path, default=None, help="合并目录时常需指向原始基座 tokenizer")
    p.add_argument("--from-jsonl", type=Path, default=None, help="从 ShareGPT jsonl 取前 n 条 user 作为问题")
    p.add_argument("--n", type=int, default=5, help="示例条数（作业为 5）")
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--bf16", action="store_true")
    args = p.parse_args()

    model_path = str(args.model)
    tok_path = str(args.tokenizer_model) if args.tokenizer_model else model_path
    tokenizer = load_tokenizer(tok_path)

    dtype = torch.bfloat16 if args.bf16 and torch.cuda.is_available() else (
        torch.float16 if torch.cuda.is_available() else torch.float32
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map="auto",
    )
    if args.adapter is not None:
        if PeftModel is None:
            raise SystemExit("需要 peft: pip install peft")
        model = PeftModel.from_pretrained(model, str(args.adapter))
    model.eval()

    gen_cfg = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        do_sample=args.temperature > 0,
        temperature=args.temperature if args.temperature > 0 else None,
        top_p=args.top_p if args.temperature > 0 else None,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    prompts = collect_prompts(args)
    print("\n======== Step4：{} 条示例回复 ========\n".format(len(prompts)))

    for i, user_q in enumerate(prompts, 1):
        text = build_prompt(tokenizer, user_q)
        dev = next(model.parameters()).device
        inputs = tokenizer(text, return_tensors="pt").to(dev)
        with torch.inference_mode():
            out = model.generate(**inputs, generation_config=gen_cfg)
        new_tok = out[0, inputs["input_ids"].shape[1] :]
        reply = strip_gen(tokenizer.decode(new_tok, skip_special_tokens=False))
        print(f"--- 示例 {i} ---")
        print(f"[用户]\n{user_q}\n")
        print(f"[模型]\n{reply}\n")


if __name__ == "__main__":
    main()
