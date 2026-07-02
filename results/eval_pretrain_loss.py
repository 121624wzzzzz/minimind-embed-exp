import argparse
import csv
import json
import math
import os
import sys
import time
import warnings
from contextlib import nullcontext

import torch
import torch.nn.functional as F
from datasets import load_dataset, load_from_disk
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from trainer.trainer_utils import get_model_params

warnings.filterwarnings("ignore")

VALID_VARIANTS = [f"s{i}" for i in range(1, 14)]
UNTIED_VARIANTS = {"s2", "s8", "s9", "s10"}


class PretrainEvalDataset(Dataset):
    def __init__(
        self,
        data_path,
        tokenizer,
        max_length=340,
        max_samples=10000,
        start_index=0,
        split_manifest_path="",
        shard_index=0,
        num_shards=1,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        if self.pad_token_id is None:
            raise ValueError("PretrainEvalDataset requires tokenizer.pad_token_id or tokenizer.eos_token_id")
        self.bos_token_id = tokenizer.bos_token_id
        self.eos_token_id = tokenizer.eos_token_id
        samples = self.load_samples(data_path)
        if split_manifest_path:
            with open(split_manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
            total_samples = int(manifest["total_samples"])
            if total_samples != len(samples):
                raise ValueError(f"split manifest total_samples={total_samples} != dataset size={len(samples)}")
            eval_indices = sorted(int(i) for i in manifest["eval_indices"])
            if max_samples > 0:
                eval_indices = eval_indices[:max_samples]
            if num_shards > 1:
                if not 0 <= shard_index < num_shards:
                    raise ValueError("--shard_index must be in [0, --num_shards)")
                eval_indices = eval_indices[shard_index::num_shards]
            self.samples = samples.select(eval_indices)
        else:
            if start_index < 0:
                start_index = max(len(samples) + start_index, 0)
            end_index = len(samples) if max_samples <= 0 else min(start_index + max_samples, len(samples))
            indices = list(range(start_index, end_index))
            if num_shards > 1:
                if not 0 <= shard_index < num_shards:
                    raise ValueError("--shard_index must be in [0, --num_shards)")
                indices = indices[shard_index::num_shards]
            self.samples = samples.select(indices)

    def load_samples(self, data_path):
        if os.path.isdir(data_path) and os.path.exists(os.path.join(data_path, "dataset_info.json")):
            return load_from_disk(data_path)
        if os.path.isdir(data_path):
            parquet_files = sorted(
                os.path.join(data_path, name)
                for name in os.listdir(data_path)
                if name.endswith(".parquet")
            )
            jsonl_files = sorted(
                os.path.join(data_path, name)
                for name in os.listdir(data_path)
                if name.endswith(".jsonl")
            )
            if parquet_files:
                return load_dataset("parquet", data_files=parquet_files, split="train")
            if jsonl_files:
                return load_dataset("json", data_files=jsonl_files, split="train")
        suffix = os.path.splitext(data_path)[1].lower()
        if suffix == ".parquet":
            return load_dataset("parquet", data_files=data_path, split="train")
        return load_dataset("json", data_files=data_path, split="train")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        if "input_ids" in sample:
            input_ids = list(sample["input_ids"])[:self.max_length]
            if len(input_ids) < self.max_length:
                input_ids += [self.pad_token_id] * (self.max_length - len(input_ids))
            input_ids = torch.tensor(input_ids, dtype=torch.long)
            labels = input_ids.clone()
            labels[input_ids == self.pad_token_id] = -100
            return input_ids, labels

        special_count = int(self.bos_token_id is not None) + int(self.eos_token_id is not None)
        tokens = self.tokenizer(
            str(sample["text"]),
            add_special_tokens=False,
            max_length=max(self.max_length - special_count, 0),
            truncation=True
        ).input_ids
        if self.bos_token_id is not None:
            tokens = [self.bos_token_id] + tokens
        if self.eos_token_id is not None:
            tokens = tokens + [self.eos_token_id]
        input_ids = tokens + [self.pad_token_id] * (self.max_length - len(tokens))
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        labels = input_ids.clone()
        labels[input_ids == self.pad_token_id] = -100
        return input_ids, labels


def variant_tie_word_embeddings(variant):
    return variant not in UNTIED_VARIANTS


def weight_filename(weight_prefix, variant, hidden_size, suffix):
    if weight_prefix:
        return f"{weight_prefix}_{variant}_{hidden_size}{suffix}.pth"
    return f"{variant}_{hidden_size}{suffix}.pth"


def load_variant_model(args, variant, vocab_size):
    intermediate_size = args.intermediate_size if args.intermediate_size > 0 else math.ceil(args.hidden_size * math.pi / 64) * 64
    config = MiniMindConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        num_attention_heads=args.num_attention_heads,
        num_key_value_heads=args.num_key_value_heads,
        head_dim=args.head_dim if args.head_dim > 0 else args.hidden_size // args.num_attention_heads,
        intermediate_size=intermediate_size,
        max_position_embeddings=args.max_position_embeddings,
        rope_theta=args.rope_theta,
        rms_norm_eps=args.rms_norm_eps,
        vocab_size=vocab_size,
        use_moe=bool(args.use_moe),
        tie_word_embeddings=variant_tie_word_embeddings(variant),
        lm_head_bias=bool(args.lm_head_bias),
        embedding_variant=variant,
        embedding_variant_rank=args.embedding_variant_rank,
    )
    model = MiniMindForCausalLM(config)
    suffix = "_moe" if args.use_moe else ""
    weight_path = os.path.join(args.save_dir, weight_filename(args.weight_prefix, variant, args.hidden_size, suffix))
    state_dict = torch.load(weight_path, map_location=args.device)
    model.load_state_dict(state_dict, strict=True)
    get_model_params(model, config)
    return model.eval().to(args.device)


@torch.no_grad()
def evaluate_model(model, loader, args):
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)
    total_nll = 0.0
    total_tokens = 0
    total_sequences = 0
    started = time.time()

    for step, (input_ids, labels) in enumerate(loader, start=1):
        input_ids = input_ids.to(args.device)
        labels = labels.to(args.device)
        with autocast_ctx:
            outputs = model(input_ids)
            logits = outputs.logits[..., :-1, :].contiguous()
            targets = labels[..., 1:].contiguous()
            token_loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-100,
                reduction="sum",
            )
        valid_tokens = targets.ne(-100).sum().item()
        total_nll += float(token_loss.item())
        total_tokens += valid_tokens
        total_sequences += input_ids.size(0)
        if args.max_batches > 0 and step >= args.max_batches:
            break

    loss = total_nll / max(total_tokens, 1)
    return {
        "loss": loss,
        "ppl": math.exp(loss) if loss < 50 else float("inf"),
        "tokens": total_tokens,
        "sequences": total_sequences,
        "seconds": time.time() - started,
    }


def write_results(results, args):
    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "variant", "loss", "ppl", "delta_vs_s1", "delta_vs_s2",
                "tokens", "sequences", "seconds", "data_path", "start_index", "max_samples"
            ],
        )
        writer.writeheader()
        writer.writerows(results)
    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Evaluate MiniMind pretrain loss for S1-S13 variants.")
    parser.add_argument("--data_path", default="dataset/minimind/pretrain_t2t.jsonl", type=str)
    parser.add_argument("--tokenizer_path", default="model", type=str)
    parser.add_argument("--save_dir", default="weights/final", type=str)
    parser.add_argument("--weight_prefix", default="pretrain_v2", type=str)
    parser.add_argument("--variants", default="s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11,s12,s13", type=str)
    parser.add_argument("--embedding_variant_rank", default=32, type=int)
    parser.add_argument("--hidden_size", default=768, type=int)
    parser.add_argument("--num_hidden_layers", default=8, type=int)
    parser.add_argument("--num_attention_heads", default=8, type=int)
    parser.add_argument("--num_key_value_heads", default=4, type=int)
    parser.add_argument("--head_dim", default=0, type=int)
    parser.add_argument("--intermediate_size", default=0, type=int)
    parser.add_argument("--max_position_embeddings", default=32768, type=int)
    parser.add_argument("--rope_theta", default=1e6, type=float)
    parser.add_argument("--rms_norm_eps", default=1e-6, type=float)
    parser.add_argument("--max_seq_len", default=340, type=int)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--max_samples", default=10000, type=int)
    parser.add_argument("--start_index", default=0, type=int)
    parser.add_argument("--tail_ratio", default=0.0, type=float, help="若大于0，则从数据尾部该比例开始评估")
    parser.add_argument("--split_manifest_path", default="", type=str, help="固定随机切分manifest路径；提供后评估manifest中的eval indices")
    parser.add_argument("--shard_index", default=0, type=int, help="当前eval分片编号，用于多进程并行eval")
    parser.add_argument("--num_shards", default=1, type=int, help="eval总分片数，用于多进程并行eval")
    parser.add_argument("--max_batches", default=0, type=int)
    parser.add_argument("--use_moe", default=0, type=int, choices=[0, 1])
    parser.add_argument("--lm_head_bias", default=1, type=int, choices=[0, 1])
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu", type=str)
    parser.add_argument("--output_csv", default="logs/eval_pretrain_loss.csv", type=str)
    parser.add_argument("--output_json", default="logs/eval_pretrain_loss.json", type=str)
    args = parser.parse_args()

    variants = [v.strip().lower() for v in args.variants.split(",") if v.strip()]
    invalid = sorted(set(variants) - set(VALID_VARIANTS))
    if invalid:
        raise ValueError(f"Invalid variants: {invalid}")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    start_index = args.start_index
    if args.split_manifest_path:
        with open(args.split_manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        total_samples = int(manifest["total_samples"])
        eval_count = len(manifest["eval_indices"])
        if args.max_samples <= 0:
            args.max_samples = eval_count
        print(f"[eval] split manifest: {args.split_manifest_path}, eval samples={min(args.max_samples, eval_count)} / total {total_samples}")
    elif args.tail_ratio > 0:
        if not 0 < args.tail_ratio < 1:
            raise ValueError("--tail_ratio 必须在 (0, 1) 范围内")
        if os.path.isdir(args.data_path) and os.path.exists(os.path.join(args.data_path, "dataset_info.json")):
            total_samples = len(load_from_disk(args.data_path))
        elif os.path.isdir(args.data_path):
            parquet_files = sorted(
                os.path.join(args.data_path, name)
                for name in os.listdir(args.data_path)
                if name.endswith(".parquet")
            )
            jsonl_files = sorted(
                os.path.join(args.data_path, name)
                for name in os.listdir(args.data_path)
                if name.endswith(".jsonl")
            )
            if parquet_files:
                total_samples = len(load_dataset("parquet", data_files=parquet_files, split="train"))
            elif jsonl_files:
                total_samples = len(load_dataset("json", data_files=jsonl_files, split="train"))
            else:
                raise FileNotFoundError(f"No parquet/jsonl files found in {args.data_path}")
        elif os.path.splitext(args.data_path)[1].lower() == ".parquet":
            total_samples = len(load_dataset("parquet", data_files=args.data_path, split="train"))
        else:
            total_samples = len(load_dataset("json", data_files=args.data_path, split="train"))
        start_index = int(total_samples * (1 - args.tail_ratio))
        if args.max_samples <= 0:
            args.max_samples = total_samples - start_index
        print(f"[eval] tail split: [{start_index}, {total_samples}) / total {total_samples} ({args.tail_ratio:.2%})")

    eval_ds = PretrainEvalDataset(
        args.data_path,
        tokenizer,
        max_length=args.max_seq_len,
        max_samples=args.max_samples,
        start_index=start_index,
        split_manifest_path=args.split_manifest_path,
        shard_index=args.shard_index,
        num_shards=args.num_shards,
    )
    loader = DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    shard_msg = f", shard={args.shard_index}/{args.num_shards}" if args.num_shards > 1 else ""
    print(f"[eval] data={args.data_path}, samples={len(eval_ds)}, max_seq_len={args.max_seq_len}, batch_size={args.batch_size}{shard_msg}")

    results_by_variant = {}
    for variant in variants:
        print(f"[eval] loading {variant}")
        model = load_variant_model(args, variant, len(tokenizer))
        result = evaluate_model(model, loader, args)
        result["variant"] = variant
        results_by_variant[variant] = result
        print(
            f"[eval] {variant}: loss={result['loss']:.4f}, ppl={result['ppl']:.2f}, "
            f"tokens={result['tokens']}, seconds={result['seconds']:.1f}"
        )
        del model
        if "cuda" in args.device:
            torch.cuda.empty_cache()

    s1_loss = results_by_variant.get("s1", {}).get("loss")
    s2_loss = results_by_variant.get("s2", {}).get("loss")
    rows = []
    for variant in variants:
        item = results_by_variant[variant]
        rows.append({
            "variant": variant,
            "loss": f"{item['loss']:.6f}",
            "ppl": f"{item['ppl']:.6f}",
            "delta_vs_s1": "" if s1_loss is None else f"{item['loss'] - s1_loss:+.6f}",
            "delta_vs_s2": "" if s2_loss is None else f"{item['loss'] - s2_loss:+.6f}",
            "tokens": item["tokens"],
            "sequences": item["sequences"],
            "seconds": f"{item['seconds']:.2f}",
            "data_path": args.data_path,
            "start_index": start_index,
            "max_samples": args.max_samples,
        })
    write_results(rows, args)
    print(f"[eval] wrote {args.output_csv}")
    if args.output_json:
        print(f"[eval] wrote {args.output_json}")


if __name__ == "__main__":
    main()
