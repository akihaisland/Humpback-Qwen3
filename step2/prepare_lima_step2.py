import json
import random
import sys
from pathlib import Path


def is_single_turn_lima_row(obj: dict) -> tuple[str, str] | None:
    conv = obj.get("conversations")
    if not isinstance(conv, list) or len(conv) != 2:
        return None
    a, b = conv[0], conv[1]
    if not isinstance(a, str) or not isinstance(b, str):
        return None
    inst, comp = a.strip(), b.strip()
    if not inst or not comp:
        return None
    return inst, comp


def main(input_path: Path, output_path: Path) -> None:
    n = 150 # number of samples to sample
    seed = 42 # random seed
    min_completion_chars = 32 # minimum length of completion

    if not input_path.is_file():
        raise SystemExit(f"input not found: {input_path}")

    singles: list[dict] = []
    total = 0
    with input_path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[warn] line {line_no}: invalid json: {e}", file=sys.stderr)
                continue
            pair = is_single_turn_lima_row(obj)
            if pair is None:
                continue
            inst, comp = pair
            if len(comp) < min_completion_chars:
                continue
            singles.append(
                {
                    "lima_instruction": inst,
                    "lima_completion": comp,
                    "source": obj.get("source", ""),
                }
            )

    if len(singles) < n:
        print(
            f"[warn] 单轮样本仅 {len(singles)} 条，少于请求的 {n}；将全部写出。",
            file=sys.stderr,
        )
    rng = random.Random(seed)
    chosen = singles if len(singles) <= n else rng.sample(singles, n)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as out:
        for row in chosen:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(
        f"读取 {total} 行，单轮有效 {len(singles)} 条，写出 {len(chosen)} 条 -> {output_path}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    raw_dataset = Path("datasets/lima/train.jsonl")
    output_dataset = Path("step2/lima_step2_sample150.jsonl")
    main(raw_dataset, output_dataset)
