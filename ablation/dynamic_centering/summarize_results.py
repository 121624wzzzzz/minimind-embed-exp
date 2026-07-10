import argparse
import csv
import json
import math
import os


def load_baselines(path):
    with open(path, newline="", encoding="utf-8") as handle:
        rows = {row["variant"]: row for row in csv.DictReader(handle)}
    missing = {"s1", "s3"} - set(rows)
    if missing:
        raise ValueError(f"基线 CSV 缺少：{sorted(missing)}")
    return rows


def load_center_dynamic(path):
    with open(path, encoding="utf-8") as handle:
        rows = json.load(handle)
    matches = [row for row in rows if row["variant"] == "center_dynamic"]
    if len(matches) != 1:
        raise ValueError(f"期望唯一 center_dynamic 结果，实际为 {len(matches)} 条")
    return matches[0]


def main():
    parser = argparse.ArgumentParser(description="汇总 dynamic centering 与同 seed 的 S1/S3。")
    parser.add_argument("--baseline_csv", required=True)
    parser.add_argument("--center_json", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--output_json", required=True)
    args = parser.parse_args()

    baselines = load_baselines(args.baseline_csv)
    center = load_center_dynamic(args.center_json)
    losses = {
        "s1": float(baselines["s1"]["loss"]),
        "s3": float(baselines["s3"]["loss"]),
        "center_dynamic": float(center["loss"]),
    }
    metadata = {
        "s1": baselines["s1"],
        "s3": baselines["s3"],
        "center_dynamic": center,
    }
    rows = []
    for variant in ("s1", "center_dynamic", "s3"):
        loss = losses[variant]
        item = metadata[variant]
        rows.append({
            "variant": variant,
            "source": "existing_same_seed_baseline" if variant in {"s1", "s3"} else "new_ablation",
            "loss": f"{loss:.6f}",
            "ppl": f"{math.exp(loss):.6f}",
            "delta_vs_s1": f"{loss - losses['s1']:+.6f}",
            "delta_vs_s3": f"{loss - losses['s3']:+.6f}",
            "tokens": int(item["tokens"]),
            "sequences": int(item["sequences"]),
        })

    fieldnames = [
        "variant", "source", "loss", "ppl", "delta_vs_s1", "delta_vs_s3",
        "tokens", "sequences",
    ]
    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    with open(args.output_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    with open(args.output_json, "w", encoding="utf-8") as handle:
        json.dump(rows, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    for row in rows:
        print(
            f"[summary] {row['variant']}: loss={row['loss']}, "
            f"delta_vs_s1={row['delta_vs_s1']}, delta_vs_s3={row['delta_vs_s3']}"
        )
    print(f"[summary] wrote {args.output_csv}")
    print(f"[summary] wrote {args.output_json}")


if __name__ == "__main__":
    main()
