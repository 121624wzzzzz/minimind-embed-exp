#!/usr/bin/env python3
"""
Collect Qwen3-0.6B FineEdu eval results from log directories and produce
summary CSVs in the same format as the GPT2 FineEdu summaries.

Usage:
  python results/generate_qwen3_fineedu_summary.py
  python results/generate_qwen3_fineedu_summary.py --dry-run

Output:
  results/summaries/fineedu-qwen3-0.6b-three-seed-summary.csv   (3-seed aggregate)
  results/summaries/fineedu-qwen3-0.6b-per-seed-delta-vs-s1.csv (per-seed detail)
"""

import csv
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOG_BASE = ROOT / "logs" / "fineedu-qwen3-0.6b"
OUT_DIR = ROOT / "results" / "summaries"
DATASET_NAME = "fineedu_qwen3"

# Output paths (consistent with GPT2 naming)
SUMMARY_3SEED = OUT_DIR / "fineedu-qwen3-0.6b-three-seed-summary.csv"
PER_SEED_DELTA = OUT_DIR / "fineedu-qwen3-0.6b-per-seed-delta-vs-s1.csv"

# Variant ordering (same as GPT2: s12 first, s2 last)
VARIANT_ORDER = ["s12", "s13", "s3", "s4", "s6", "s11", "s5", "s1", "s7", "s2"]

# Expected variant set for FineEdu Qwen3 (may be subset of full order)
QWEN3_VARIANTS = ["s1", "s3", "s12"]


def collect_eval_results():
    """Scan all eval JSONs in fineedu-qwen3 log dirs, return {seed: {variant: row}}."""
    results = defaultdict(dict)  # seed -> variant -> dict

    for json_path in sorted(LOG_BASE.rglob("eval_pretrain_loss*.json")):
        try:
            rows = json.loads(json_path.read_text())
        except (json.JSONDecodeError, ValueError) as e:
            print(f"[WARN] skipping {json_path}: {e}", file=sys.stderr)
            continue

        # Extract seed from path  (e.g., .../seed42/eval_pretrain_loss.json)
        try:
            seed = json_path.parent.name  # "seed42"
            if not seed.startswith("seed"):
                # Try parse seed from path differently
                for part in json_path.parts:
                    if part.startswith("seed"):
                        seed = part
                        break
                else:
                    continue
            seed_num = int(seed.replace("seed", ""))
        except (ValueError, StopIteration):
            print(f"[WARN] cannot determine seed from {json_path}", file=sys.stderr)
            continue

        for row in rows:
            variant = row.get("variant", "")
            tokens = int(row.get("tokens", 0))

            # Deduplicate: prefer the run with more tokens (full eval over sample)
            existing = results[seed_num].get(variant)
            if existing and int(existing.get("tokens", 0)) >= tokens:
                continue

            # Mark source for traceability
            row["_source"] = str(json_path.relative_to(ROOT))
            row["_seed"] = seed_num
            results[seed_num][variant] = row

    return results


def build_three_seed_csv(results):
    """
    Build the 3-seed aggregate summary.
    Format: variant, seed42, seed123, seed2026, mean, std, mean_ppl,
            delta_vs_s1_mean, delta_vs_s2_mean
    """
    all_variants = set()
    for seed_data in results.values():
        all_variants.update(seed_data.keys())

    # Keep only Qwen3 FineEdu variants, ordered
    variants = [v for v in VARIANT_ORDER if v in all_variants]
    if not variants:
        variants = sorted(all_variants, key=lambda v: QWEN3_VARIANTS.index(v) if v in QWEN3_VARIANTS else 999)

    # Determine all seeds available
    all_seeds = sorted(results.keys())

    rows = []
    for variant in variants:
        seed_losses = {}
        for seed in all_seeds:
            row = results[seed].get(variant)
            seed_losses[seed] = float(row["loss"]) if row else None

        # Compute mean across available seeds
        valid = [v for v in seed_losses.values() if v is not None]
        mean_loss = sum(valid) / len(valid) if valid else float("nan")
        std_loss = (
            math.sqrt(sum((v - mean_loss) ** 2 for v in valid) / len(valid))
            if len(valid) > 1 else 0.0
        )
        mean_ppl = math.exp(mean_loss) if not math.isnan(mean_loss) else float("nan")

        # delta_vs_s1_mean
        s1_mean = None
        s1_valid = [float(results[s]["s1"]["loss"]) for s in all_seeds if "s1" in results[s]]
        if s1_valid:
            s1_mean = sum(s1_valid) / len(s1_valid)
        delta_vs_s1 = (mean_loss - s1_mean) if (valid and s1_mean is not None) else float("nan")

        # delta_vs_s2_mean (if s2 available)
        s2_mean = None
        s2_valid = [float(results[s]["s2"]["loss"]) for s in all_seeds if "s2" in results[s]]
        if s2_valid:
            s2_mean = sum(s2_valid) / len(s2_valid)
        delta_vs_s2 = (mean_loss - s2_mean) if (valid and s2_mean is not None) else float("nan")

        row_dict = {"variant": variant}
        for seed in [42, 123, 2026]:
            if seed in seed_losses and seed_losses[seed] is not None:
                row_dict[f"seed{seed}"] = f"{seed_losses[seed]:.6f}"
            else:
                row_dict[f"seed{seed}"] = ""
        row_dict["mean"] = f"{mean_loss:.6f}" if not math.isnan(mean_loss) else ""
        row_dict["std"] = f"{std_loss:.6f}" if len(valid) > 1 else ""
        row_dict["mean_ppl"] = f"{mean_ppl:.6f}" if not math.isnan(mean_ppl) else ""
        row_dict["delta_vs_s1_mean"] = f"{delta_vs_s1:+.6f}" if not math.isnan(delta_vs_s1) else ""
        row_dict["delta_vs_s2_mean"] = f"{delta_vs_s2:+.6f}" if not math.isnan(delta_vs_s2) else ""
        rows.append(row_dict)

    return rows, all_seeds


def build_per_seed_csv(results):
    """
    Build per-seed delta-vs-s1 detail.
    Format: dataset, variant, seed, loss, ppl, delta_vs_s1_loss,
            delta_vs_s1_pct, ppl_delta_vs_s1, ppl_delta_vs_s1_pct, rank_in_seed
    """
    rows = []

    for seed in sorted(results.keys()):
        seed_data = results[seed]

        # Find s1 baseline for this seed
        s1_row = seed_data.get("s1")
        s1_loss = float(s1_row["loss"]) if s1_row else None
        s1_ppl = float(s1_row["ppl"]) if s1_row else None

        # Get all variants in this seed, sorted by loss
        variants = sorted(seed_data.keys(),
                          key=lambda v: float(seed_data[v]["loss"]))

        for rank, variant in enumerate(variants, 1):
            row = seed_data[variant]
            loss = float(row["loss"])
            ppl = float(row["ppl"])

            delta_loss = loss - s1_loss if s1_loss is not None else 0.0
            delta_ppl = ppl - s1_ppl if s1_ppl is not None else 0.0
            delta_loss_pct = (delta_loss / s1_loss * 100) if s1_loss and s1_loss != 0 else 0.0
            delta_ppl_pct = (delta_ppl / s1_ppl * 100) if s1_ppl and s1_ppl != 0 else 0.0

            rows.append({
                "dataset": DATASET_NAME,
                "variant": variant,
                "seed": seed,
                "loss": f"{loss:.6f}",
                "ppl": f"{ppl:.6f}",
                "delta_vs_s1_loss": f"{delta_loss:+.6f}",
                "delta_vs_s1_pct": f"{delta_loss_pct:+.3f}%",
                "ppl_delta_vs_s1": f"{delta_ppl:+.6f}",
                "ppl_delta_vs_s1_pct": f"{delta_ppl_pct:+.3f}%",
                "rank_in_seed": rank,
            })

    return rows


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"[OK] wrote {path}  ({len(rows)} rows)")


def main():
    dry_run = "--dry-run" in sys.argv

    print("Collecting Qwen3-0.6B FineEdu eval results...")
    results = collect_eval_results()

    if not results:
        print("[WARN] No eval results found!")
        return

    seeds = sorted(results.keys())
    variants = set()
    for sd in results.values():
        variants.update(sd.keys())
    print(f"  seeds found: {seeds}")
    print(f"  variants found: {sorted(variants)}")

    # ---- 3-seed aggregate ----
    agg_rows, agg_seeds = build_three_seed_csv(results)
    agg_fields = ["variant"] + [f"seed{s}" for s in sorted(agg_seeds)] + \
                 ["mean", "std", "mean_ppl", "delta_vs_s1_mean", "delta_vs_s2_mean"]
    # Use consistent column names even if some seeds missing
    agg_fields_out = ["variant", "seed42", "seed123", "seed2026",
                      "mean", "std", "mean_ppl", "delta_vs_s1_mean", "delta_vs_s2_mean"]
    for row in agg_rows:
        for s in [42, 123, 2026]:
            if s not in agg_seeds and f"seed{s}" not in row:
                row[f"seed{s}"] = ""
    if not dry_run:
        write_csv(SUMMARY_3SEED, agg_rows, agg_fields_out)

    # ---- Per-seed detail ----
    per_rows = build_per_seed_csv(results)
    per_fields = ["dataset", "variant", "seed", "loss", "ppl",
                  "delta_vs_s1_loss", "delta_vs_s1_pct",
                  "ppl_delta_vs_s1", "ppl_delta_vs_s1_pct", "rank_in_seed"]
    if not dry_run:
        write_csv(PER_SEED_DELTA, per_rows, per_fields)

    if dry_run:
        print("\n[Dry-run] 3-seed aggregate preview:")
        for row in agg_rows:
            print(f"  {row}")
        print(f"\n[Dry-run] Per-seed preview:")
        for row in per_rows:
            print(f"  {row}")


if __name__ == "__main__":
    main()
