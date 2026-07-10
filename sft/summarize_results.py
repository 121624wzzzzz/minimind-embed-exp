import argparse
import csv
import json
import math
import os
import re
import statistics
from pathlib import Path


def split_values(value):
    return [item for item in re.split(r"[\s,]+", value.strip()) if item]


def load_jsonl(path):
    if not path.is_file():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}: {exc}") from exc
    return rows


def find_eval_row(result_root, seed, variant):
    seed_dir = result_root / f"seed{seed}"
    summary_path = seed_dir / "eval_after_seed" / f"seed{seed}_eval_summary.jsonl"
    summary_rows = load_jsonl(summary_path)
    matches = [
        row
        for row in summary_rows
        if row.get("event") == "seed-eval" and str(row.get("variant", "")).lower() == variant
    ]
    if matches:
        return matches[-1]

    metrics_path = seed_dir / f"{variant}_metrics.jsonl"
    metric_rows = load_jsonl(metrics_path)
    matches = [row for row in metric_rows if row.get("event") == "eval-final"]
    if matches:
        return matches[-1]
    return None


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize SFT eval JSONL files into one CSV")
    parser.add_argument("--result-root", required=True, type=Path)
    parser.add_argument("--output", default=None, type=Path)
    parser.add_argument("--seeds", required=True, type=str)
    parser.add_argument("--variants", required=True, type=str)
    return parser.parse_args()


def main():
    args = parse_args()
    result_root = args.result_root.resolve()
    output_path = args.output.resolve() if args.output is not None else result_root / "eval_summary.csv"
    seeds = split_values(args.seeds)
    variants = [variant.lower() for variant in split_values(args.variants)]
    if not seeds or not variants:
        raise ValueError("--seeds and --variants must not be empty")

    losses = {variant: {} for variant in variants}
    for seed in seeds:
        for variant in variants:
            row = find_eval_row(result_root, seed, variant)
            if row is None:
                raise FileNotFoundError(f"missing eval result for seed={seed}, variant={variant} under {result_root}")
            loss = float(row["eval_loss"])
            if not math.isfinite(loss):
                raise ValueError(f"non-finite eval loss for seed={seed}, variant={variant}: {loss}")
            losses[variant][seed] = loss

    means = {
        variant: statistics.fmean(losses[variant][seed] for seed in seeds)
        for variant in variants
    }
    baseline_losses = losses.get("s1")
    rows = []
    for variant in sorted(variants, key=lambda item: (means[item], item)):
        values = [losses[variant][seed] for seed in seeds]
        seed_cells = []
        deltas = []
        for seed, value in zip(seeds, values):
            seed_cells.append(f"{value:.6f}")
            if baseline_losses is None:
                seed_cells.append("")
            else:
                delta = value - baseline_losses[seed]
                deltas.append(delta)
                seed_cells.append(f"{delta:+.6f}")
        rows.append(
            [variant]
            + seed_cells
            + [
                f"{means[variant]:.6f}",
                f"{statistics.pstdev(values):.6f}",
                f"{math.exp(means[variant]):.6f}",
                "" if not deltas else f"{statistics.fmean(deltas):+.6f}",
            ]
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(
            ["variant"]
            + [column for seed in seeds for column in (f"seed{seed}_loss", f"seed{seed}_delta_vs_s1")]
            + ["mean_loss", "std_loss", "mean_ppl", "mean_delta_vs_s1"]
        )
        writer.writerows(rows)
    os.replace(tmp_path, output_path)
    print(f"[summary] wrote {output_path}")


if __name__ == "__main__":
    main()
