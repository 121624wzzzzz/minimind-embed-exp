import argparse
import csv
import json
import math
import os
import statistics


VARIANTS = ("s1", "center_dynamic", "s3")


def parse_seed_result(value):
    seed_text, path = value.split("=", 1)
    return int(seed_text), path


def load_comparison(path):
    with open(path, newline="", encoding="utf-8") as handle:
        rows = {row["variant"]: row for row in csv.DictReader(handle)}
    missing = set(VARIANTS) - set(rows)
    if missing:
        raise ValueError(f"{path} 缺少变体：{sorted(missing)}")
    return rows


def write_csv(path, fieldnames, rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="汇总 dynamic centering 三 seed 结果。")
    parser.add_argument(
        "--seed_result",
        action="append",
        required=True,
        help="格式为 seed=同 seed comparison CSV，可重复传入",
    )
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    seed_paths = dict(parse_seed_result(value) for value in args.seed_result)
    expected_seeds = {42, 123, 2026}
    if set(seed_paths) != expected_seeds:
        raise ValueError(f"必须恰好提供 seeds {sorted(expected_seeds)}，实际为 {sorted(seed_paths)}")

    by_seed = {seed: load_comparison(path) for seed, path in seed_paths.items()}
    per_seed_rows = []
    for seed in sorted(by_seed):
        s1_loss = float(by_seed[seed]["s1"]["loss"])
        s3_loss = float(by_seed[seed]["s3"]["loss"])
        for variant in VARIANTS:
            loss = float(by_seed[seed][variant]["loss"])
            per_seed_rows.append({
                "seed": seed,
                "variant": variant,
                "loss": f"{loss:.6f}",
                "ppl": f"{math.exp(loss):.6f}",
                "delta_vs_s1": f"{loss - s1_loss:+.6f}",
                "delta_vs_s3": f"{loss - s3_loss:+.6f}",
                "tokens": int(by_seed[seed][variant]["tokens"]),
                "sequences": int(by_seed[seed][variant]["sequences"]),
            })

    summary_rows = []
    for variant in VARIANTS:
        losses = [float(by_seed[seed][variant]["loss"]) for seed in sorted(by_seed)]
        s1_losses = [float(by_seed[seed]["s1"]["loss"]) for seed in sorted(by_seed)]
        deltas = [loss - baseline for loss, baseline in zip(losses, s1_losses)]
        mean_loss = statistics.mean(losses)
        summary_rows.append({
            "variant": variant,
            "seed42_loss": f"{losses[0]:.6f}",
            "seed123_loss": f"{losses[1]:.6f}",
            "seed2026_loss": f"{losses[2]:.6f}",
            "mean_loss": f"{mean_loss:.6f}",
            "std_loss": f"{statistics.pstdev(losses):.6f}",
            "mean_ppl": f"{math.exp(mean_loss):.6f}",
            "mean_delta_vs_s1": f"{statistics.mean(deltas):+.6f}",
        })

    mechanism_rows = []
    recovered_fractions = []
    for seed in sorted(by_seed):
        s1_loss = float(by_seed[seed]["s1"]["loss"])
        center_loss = float(by_seed[seed]["center_dynamic"]["loss"])
        s3_loss = float(by_seed[seed]["s3"]["loss"])
        s3_gain = s1_loss - s3_loss
        center_gain = s1_loss - center_loss
        fraction = center_gain / s3_gain if s3_gain != 0 else float("nan")
        recovered_fractions.append(fraction)
        mechanism_rows.append({
            "seed": seed,
            "s1_loss": f"{s1_loss:.6f}",
            "center_dynamic_loss": f"{center_loss:.6f}",
            "s3_loss": f"{s3_loss:.6f}",
            "s3_gain_vs_s1": f"{s3_gain:+.6f}",
            "center_gain_vs_s1": f"{center_gain:+.6f}",
            "fraction_of_s3_gain_recovered": f"{fraction:.6f}",
        })

    mean_s1 = statistics.mean(float(by_seed[s]["s1"]["loss"]) for s in by_seed)
    mean_center = statistics.mean(float(by_seed[s]["center_dynamic"]["loss"]) for s in by_seed)
    mean_s3 = statistics.mean(float(by_seed[s]["s3"]["loss"]) for s in by_seed)
    aggregate = {
        "seeds": sorted(by_seed),
        "fraction_from_mean_losses": (mean_s1 - mean_center) / (mean_s1 - mean_s3),
        "mean_of_per_seed_fractions": statistics.mean(recovered_fractions),
        "mean_losses": {"s1": mean_s1, "center_dynamic": mean_center, "s3": mean_s3},
    }

    write_csv(
        os.path.join(args.output_dir, "per_seed_comparison.csv"),
        ["seed", "variant", "loss", "ppl", "delta_vs_s1", "delta_vs_s3", "tokens", "sequences"],
        per_seed_rows,
    )
    write_csv(
        os.path.join(args.output_dir, "three_seed_summary.csv"),
        ["variant", "seed42_loss", "seed123_loss", "seed2026_loss", "mean_loss", "std_loss", "mean_ppl", "mean_delta_vs_s1"],
        summary_rows,
    )
    write_csv(
        os.path.join(args.output_dir, "mechanism_recovery.csv"),
        ["seed", "s1_loss", "center_dynamic_loss", "s3_loss", "s3_gain_vs_s1", "center_gain_vs_s1", "fraction_of_s3_gain_recovered"],
        mechanism_rows,
    )
    with open(os.path.join(args.output_dir, "three_seed_summary.json"), "w", encoding="utf-8") as handle:
        json.dump(
            {"summary": summary_rows, "mechanism_by_seed": mechanism_rows, "aggregate": aggregate},
            handle,
            ensure_ascii=False,
            indent=2,
        )
        handle.write("\n")

    print(json.dumps(aggregate, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
