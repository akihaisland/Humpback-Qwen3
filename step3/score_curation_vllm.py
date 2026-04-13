#!/usr/bin/env python3
"""
Step3：自筛选（self-curation）——用 Qwen3-1.7B + vLLM 对 (instruction, response) 打分 1–5。

- 评分提示：论文附录 **Table 19**（readme 中 “Table 1” 指数据统计；§2.3 指向 Table 19 为打分模板）。
- Few-shot：见 paper_curation_prompt.FEW_SHOT_BLOCK。
- 输入：Step2 的 augmented jsonl（默认用 generated_instruction + lima_completion）。
- 输出：逐条带 curation_score / curation_raw；另导出 ShareGPT 供 Step4；打印 5 条高分与 5 条低分样例。

依赖：pip install vllm transformers
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# 允许 `python step3/score_curation_vllm.py` 从仓库根目录运行
_STEP3_DIR = str(Path(__file__).resolve().parent)
if _STEP3_DIR not in sys.path:
    sys.path.insert(0, _STEP3_DIR)

from paper_curation_prompt import build_user_prompt
from transformers import AutoTokenizer

try:
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
except ImportError as e:  # pragma: no cover
    raise SystemExit(f"请安装 vllm: pip install vllm\n{e}") from e

# 与 step2 一致：Qwen3 think 块清理
_TB = bytes([0x3C, 0x7C, 0x74, 0x68, 0x69, 0x6E, 0x6B, 0x7C, 0x3E]).decode("ascii")
_TE = bytes([0x3C, 0x7C, 0x2F, 0x74, 0x68, 0x69, 0x6E, 0x6B, 0x7C, 0x3E]).decode("ascii")
_THINK_PIPE_BLOCK = re.compile(
    re.escape(_TB) + r"[\s\S]*?" + re.escape(_TE),
    re.DOTALL | re.IGNORECASE,
)


def strip_think_and_special(text: str) -> str:
    t = _THINK_PIPE_BLOCK.sub("", text)
    for end_mark in ("</think>",):
        if end_mark in t:
            t = t.split(end_mark, 1)[-1]
    t = re.sub(r"<\|im_start\|>assistant\s*", "", t, flags=re.IGNORECASE)
    t = re.sub(r"<\|im_end\|>", "", t, flags=re.IGNORECASE)
    return t.strip()


def load_tokenizer(pretrained_dir: str, *, use_fast: bool = True):
    kwargs = {"trust_remote_code": True, "use_fast": use_fast}
    try:
        return AutoTokenizer.from_pretrained(pretrained_dir, **kwargs)
    except (AttributeError, TypeError, ValueError, OSError) as e:
        if use_fast:
            print(
                f"[warn] Tokenizer(use_fast=True) 失败: {e}；改用 use_fast=False（{pretrained_dir}）",
                file=sys.stderr,
            )
            return load_tokenizer(pretrained_dir, use_fast=False)
        raise SystemExit(
            "无法加载 tokenizer。合并目录请使用 --tokenizer-model 指向原始 Qwen3 基座。\n"
            f"路径: {pretrained_dir}\n错误: {e}"
        ) from e


def build_chat_input(tokenizer, user_text: str) -> str:
    messages = [{"role": "user", "content": user_text}]
    kwargs: dict = {"tokenize": False, "add_generation_prompt": True}
    try:
        return tokenizer.apply_chat_template(messages, **kwargs, enable_thinking=False)  # type: ignore[call-arg]
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)


def parse_score(text: str) -> int | None:
    t = strip_think_and_special(text)
    m = re.search(r"Score:\s*([1-5])\s*$", t, flags=re.IGNORECASE | re.MULTILINE)
    if m:
        return int(m.group(1))
    m = re.search(r"Score:\s*([1-5])\b", t, flags=re.IGNORECASE)
    if m:
        return int(m.group(1))
    return None


def pair_from_row(row: dict) -> tuple[str, str] | None:
    inst = row.get("generated_instruction") or row.get("lima_instruction")
    resp = row.get("lima_completion") or row.get("completion")
    if not isinstance(inst, str) or not isinstance(resp, str):
        return None
    inst, resp = inst.strip(), resp.strip()
    if not inst or not resp:
        return None
    return inst, resp


def to_sharegpt_record(instruction: str, response: str) -> dict:
    return {
        "conversations": [
            {"from": "human", "value": instruction},
            {"from": "gpt", "value": response},
        ]
    }


def print_extremes(rows: list[dict], k: int) -> None:
    scored = [r for r in rows if isinstance(r.get("curation_score"), int)]
    scored.sort(key=lambda r: (r["curation_score"], r.get("_line_idx", 0)))
    if not scored:
        print("[Step3] 没有成功解析出分数的样本，跳过高低分样例打印。", file=sys.stderr)
        return
    kk = min(k, len(scored))
    lows = scored[:kk]
    highs = scored[-kk:][::-1]
    print("\n" + "=" * 72 + "\n[Step3] 低分样例（前 {} 条，分数升序）\n".format(k), file=sys.stderr)
    for r in lows:
        _print_one(r, file=sys.stderr)
    print("\n" + "=" * 72 + "\n[Step3] 高分样例（前 {} 条，分数降序）\n".format(k), file=sys.stderr)
    for r in highs:
        _print_one(r, file=sys.stderr)


def _print_one(r: dict, *, file) -> None:
    print(f"--- score={r.get('curation_score')} line={r.get('_line_idx')} ---", file=file)
    ins = str(r.get("curation_instruction", ""))
    print(f"instruction:\n{ins[:600]}{'…' if len(ins) > 600 else ''}\n", file=file)
    resp = str(r.get("curation_response", ""))
    print(f"response:\n{resp[:500]}{'…' if len(resp) > 500 else ''}\n", file=file)
    if r.get("curation_raw"):
        raw = str(r["curation_raw"])
        print(f"raw (末 400 字): …{raw[-400:]}\n", file=file)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True, help="Step2 augmented jsonl")
    p.add_argument("--output-scored", type=Path, default=Path("step3/lima_curated_scored.jsonl"))
    p.add_argument(
        "--output-sft",
        type=Path,
        default=Path("step3/lima_curated_for_sft_sharegpt.jsonl"),
        help=">= min-score 的 ShareGPT，供 Step4",
    )
    p.add_argument("--model", type=Path, required=True, help="评分用 Qwen3-1.7B（HF 或合并目录）")
    p.add_argument("--tokenizer-model", type=Path, default=None, help="合并权重时必指向原始基座 tokenizer")
    p.add_argument("--adapter", type=Path, default=None, help="若 M0 为 LoRA，则传 adapter 目录")
    p.add_argument("--dtype", default="auto", choices=("auto", "half", "bfloat16", "float16"))
    p.add_argument("--tensor-parallel-size", type=int, default=1)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    p.add_argument("--max-model-len", type=int, default=16384)
    p.add_argument("--max-lora-rank", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--min-score", type=int, default=4, help="写入 output-sft 的最低分（论文常用 >=4）")
    p.add_argument("--print-extremes", type=int, default=5, help="打印几条最低分 / 几条最高分样例")
    args = p.parse_args()

    if not args.input.is_file():
        raise SystemExit(f"找不到输入: {args.input}")

    model_dir = str(args.model)
    tok_dir = str(args.tokenizer_model) if args.tokenizer_model is not None else model_dir
    if args.tokenizer_model is None and "merged" in model_dir.lower():
        raise SystemExit("合并 --model 时必须指定 --tokenizer-model 为原始 Qwen3 基座目录。")

    tokenizer = load_tokenizer(tok_dir)
    if tok_dir != model_dir:
        print(f"[info] tokenizer={tok_dir} | weights={model_dir}", file=sys.stderr)

    rows: list[dict] = []
    with args.input.open(encoding="utf-8") as fin:
        for idx, line in enumerate(fin):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            pair = pair_from_row(row)
            if pair is None:
                print(f"[warn] line {idx+1}: 无法取 (instruction, response)，跳过", file=sys.stderr)
                continue
            inst, resp = pair
            row["_line_idx"] = idx + 1
            row["curation_instruction"] = inst
            row["curation_response"] = resp
            rows.append(row)

    if not rows:
        raise SystemExit("没有可打分样本")

    llm_kw: dict = {
        "model": model_dir,
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
        lora_request = LoRARequest("curation_m0", 1, str(args.adapter.resolve()))

    print("[info] 初始化 vLLM（自筛选打分）…", file=sys.stderr)
    llm = LLM(**llm_kw)
    sampling = SamplingParams(
        max_tokens=args.max_new_tokens,
        temperature=max(0.0, args.temperature),
        top_p=args.top_p if args.temperature > 0 else 1.0,
    )

    args.output_scored.parent.mkdir(parents=True, exist_ok=True)
    args.output_sft.parent.mkdir(parents=True, exist_ok=True)

    bs = max(1, args.batch_size)
    scored_rows: list[dict] = []

    with args.output_scored.open("w", encoding="utf-8") as fscored, args.output_sft.open(
        "w", encoding="utf-8"
    ) as fsft:
        for start in range(0, len(rows), bs):
            batch = rows[start : start + bs]
            user_texts = [build_user_prompt(r["curation_instruction"], r["curation_response"]) for r in batch]
            prompts = [build_chat_input(tokenizer, ut) for ut in user_texts]
            gen_kw: dict = {}
            if lora_request is not None:
                gen_kw["lora_request"] = lora_request
            outs = llm.generate(prompts, sampling, **gen_kw)
            for row, out in zip(batch, outs):
                raw = strip_think_and_special(out.outputs[0].text)
                score = parse_score(raw)
                row["curation_raw"] = raw
                row["curation_score"] = score
                scored_rows.append(row)
                fscored.write(json.dumps(row, ensure_ascii=False) + "\n")
                if score is not None and score >= args.min_score:
                    rec = to_sharegpt_record(row["curation_instruction"], row["curation_response"])
                    fsft.write(json.dumps(rec, ensure_ascii=False) + "\n")

    if args.print_extremes > 0:
        print_extremes(scored_rows, args.print_extremes)

    n_sft = sum(1 for r in scored_rows if isinstance(r.get("curation_score"), int) and r["curation_score"] >= args.min_score)
    print(
        f"[done] scored={len(scored_rows)} -> {args.output_scored}\n"
        f"[done] min_score>={args.min_score} -> {n_sft} 条写入 {args.output_sft}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
