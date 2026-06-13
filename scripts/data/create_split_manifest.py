#!/usr/bin/env python3
"""Create a fixed random eval-index manifest for pretraining data."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from datasets import load_dataset, load_from_disk


def load_num_rows(data_path: str) -> int:
    path = Path(data_path)
    if path.is_dir() and (path / "dataset_info.json").exists() and (path / "state.json").exists():
        return len(load_from_disk(str(path)))
    if path.is_dir():
        parquet_files = sorted(str(p) for p in path.glob("*.parquet"))
        jsonl_files = sorted(str(p) for p in path.glob("*.jsonl"))
        if parquet_files:
            return len(load_dataset("parquet", data_files=parquet_files, split="train"))
        if jsonl_files:
            return len(load_dataset("json", data_files=jsonl_files, split="train"))
    if any(ch in data_path for ch in "*?[]"):
        format_name = "parquet" if ".parquet" in data_path else "json"
        return len(load_dataset(format_name, data_files=data_path, split="train"))
    if path.suffix.lower() == ".parquet":
        return len(load_dataset("parquet", data_files=data_path, split="train"))
    return len(load_dataset("json", data_files=data_path, split="train"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_path", required=True, help="Dataset path used by train/eval.")
    parser.add_argument("--output", required=True, help="Output JSON manifest path.")
    parser.add_argument("--eval_ratio", type=float, default=0.01, help="Fraction of rows held out for eval.")
    parser.add_argument("--seed", type=int, default=20260602, help="Random seed for eval index sampling.")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing manifest.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not 0 < args.eval_ratio < 1:
        raise ValueError("--eval_ratio must be in (0, 1)")

    output = Path(args.output)
    if output.exists() and not args.overwrite:
        print(f"[split] exists: {output}")
        return 0

    total_samples = load_num_rows(args.data_path)
    eval_count = max(1, int(total_samples * args.eval_ratio))
    rng = random.Random(args.seed)
    eval_indices = sorted(rng.sample(range(total_samples), eval_count))

    output.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "version": 1,
        "method": "fixed_random_eval_indices",
        "data_path": args.data_path,
        "total_samples": total_samples,
        "eval_ratio": args.eval_ratio,
        "eval_count": eval_count,
        "train_count": total_samples - eval_count,
        "seed": args.seed,
        "eval_indices": eval_indices,
    }
    output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[split] wrote {output}")
    print(f"[split] total={total_samples}, train={total_samples - eval_count}, eval={eval_count}, seed={args.seed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
