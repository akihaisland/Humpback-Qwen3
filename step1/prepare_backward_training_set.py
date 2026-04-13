import json
import re
import sys
from pathlib import Path
from typing import Iterable


PAIR_RE = re.compile(
    r"###\s*Human:\s*(.*?)\s*###\s*Assistant:\s*(.*?)(?=\s*###\s*Human:|\s*$)",
    re.DOTALL | re.IGNORECASE,
)

PROMPTS = {
    "en": (
        "You are training for instruction backtranslation: predict the user instruction "
        "that would elicit the following assistant response.\n\n"
        "Assistant response:\n{y}\n\n"
        "User instruction (write only the instruction, no preamble):"
    ),
    "zh": (
        "你在做指令反译（instruction backtranslation）：根据下面的助手回答，写出最可能对应的那条"
        "用户指令或问题。\n\n"
        "助手回答：\n{y}\n\n"
        "用户指令（只写指令本身，不要解释）："
    ),
}


def iter_pairs_from_text(text: str) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for m in PAIR_RE.finditer(text or ""):
        x, y = m.group(1).strip(), m.group(2).strip()
        if x and y:
            pairs.append((x, y))
    return pairs


def build_sharegpt_record(
    instruction_x: str,
    response_y: str,
    prompt_lang: str,
) -> dict:
    tmpl = PROMPTS.get(prompt_lang, PROMPTS["en"])
    user_block = tmpl.format(y=response_y)
    return {
        "conversations": [
            {"from": "human", "value": user_block},
            {"from": "gpt", "value": instruction_x},
        ]
    }


def process_file(
    input_path: Path,
    mode: str,
    prompt_lang: str,
    min_chars: int,
    max_samples: int | None,
) -> Iterable[dict]:
    n = 0
    with input_path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[warn] line {line_no}: skip invalid json: {e}", file=sys.stderr)
                continue
            text = obj.get("text")
            if not isinstance(text, str) or not text.strip():
                print(f"[warn] line {line_no}: missing/empty text", file=sys.stderr)
                continue

            pairs = iter_pairs_from_text(text)
            if not pairs:
                print(f"[warn] line {line_no}: no Human/Assistant pairs parsed", file=sys.stderr)
                continue

            chosen = [pairs[0]] if mode == "first_turn" else pairs
            for x, y in chosen:
                if len(x) < min_chars or len(y) < min_chars:
                    continue
                yield build_sharegpt_record(x, y, prompt_lang)
                n += 1
                if max_samples is not None and n >= max_samples:
                    return


def main(input_path: Path, output_path: Path) -> None:
    prompt_lang = "en"
    mode = "first_turn"
    min_chars = 8
    max_samples = None
    # p = argparse.ArgumentParser(description=__doc__)
    # p.add_argument(
    #     "--mode",
    #     choices=("first_turn", "all_turns"),
    #     default="first_turn",
    #     help="first_turn：每条原始样本只取第一轮；all_turns：展开多轮为多条样本",
    # )
    # p.add_argument(
    #     "--prompt-lang",
    #     choices=("en", "zh"),
    #     default="en",
    #     help="human 侧任务提示语言（标签侧保持原始指令语言）",
    # )
    # p.add_argument("--min-chars", type=int, default=8, help="过滤过短的 x 或 y")
    # p.add_argument("--max-samples", type=int, default=None, help="调试用：最多写出多少条")
    # args = p.parse_args()

    if not input_path.is_file():
        raise SystemExit(f"input not found: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", encoding="utf-8") as out:
        for rec in process_file(
            input_path,
            mode=mode,
            prompt_lang=prompt_lang,
            min_chars=min_chars,
            max_samples=max_samples,
        ):
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            count += 1

    print(f"wrote {count} samples -> {output_path}")


if __name__ == "__main__":
    former_dataset = Path("datasets/openassistant-guanaco/openassistant_best_replies_train.jsonl")
    output_dataset = Path("datasets/openassistant-guanaco/openassistant_backward_step1_sharegpt.jsonl")
    main(former_dataset, output_dataset)
