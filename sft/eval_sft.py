import argparse
import json
import math
import os
import sys
import time
from contextlib import nullcontext
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / "sft/.hf_home"))
os.environ.setdefault("HF_DATASETS_CACHE", str(PROJECT_ROOT / "sft/.hf_datasets_cache"))
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

sys.path.append(str(PROJECT_ROOT))

from model.model_minimind import MiniMindForCausalLM
from model.variant_config import VALID_VARIANTS
from sft.train_sft import (
    DistributedEvalSampler,
    SmolTalkSFTDataset,
    build_config,
    collate_sft_batch,
    evaluate,
    init_distributed_mode,
    is_main_process,
    log,
    raw_model,
)
from trainer.trainer_utils import get_model_params, setup_seed


def parse_args():
    parser = argparse.ArgumentParser(description="MiniMind smol-smoltalk SFT eval")
    parser.add_argument("--eval_data_path", default=str(PROJECT_ROOT / "sft/smol-smoltalk"), type=str)
    parser.add_argument("--tokenizer_path", default="Qwen/Qwen3-0.6B", type=str)
    parser.add_argument("--weight_path", required=True, type=str)
    parser.add_argument("--metrics_path", default="", type=str)
    parser.add_argument("--run_tag", default="seed_eval", type=str)
    parser.add_argument("--batch_size", default=12, type=int)
    parser.add_argument("--max_seq_len", default=1024, type=int)
    parser.add_argument("--max_eval_samples", default=0, type=int)
    parser.add_argument("--eval_max_batches", default=0, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu", type=str)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--hidden_size", default=1024, type=int)
    parser.add_argument("--num_hidden_layers", default=28, type=int)
    parser.add_argument("--num_attention_heads", default=16, type=int)
    parser.add_argument("--num_key_value_heads", default=8, type=int)
    parser.add_argument("--head_dim", default=128, type=int)
    parser.add_argument("--intermediate_size", default=3072, type=int)
    parser.add_argument("--max_position_embeddings", default=40960, type=int)
    parser.add_argument("--rope_theta", default=1e6, type=float)
    parser.add_argument("--rms_norm_eps", default=1e-6, type=float)
    parser.add_argument("--use_moe", default=0, type=int, choices=[0, 1])
    parser.add_argument("--tie_word_embeddings", default=1, type=int, choices=[0, 1])
    parser.add_argument("--lm_head_bias", default=1, type=int, choices=[0, 1])
    parser.add_argument("--embedding_variant", default="s1", type=str, choices=sorted(VALID_VARIANTS))
    parser.add_argument("--embedding_variant_rank", default=32, type=int)
    parser.add_argument("--use_attention_mask", default=0, type=int, choices=[0, 1])
    return parser.parse_args()


def append_eval_metrics(args, row):
    if not is_main_process() or not args.metrics_path:
        return
    path = Path(args.metrics_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "run_tag": args.run_tag,
        "variant": args.embedding_variant,
        "seed": args.seed,
        "weight_path": args.weight_path,
        "eval_data_path": args.eval_data_path,
        **row,
    }
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def main():
    args = parse_args()
    args.embedding_variant = args.embedding_variant.lower()
    local_rank = init_distributed_mode()
    if dist.is_initialized():
        args.device = f"cuda:{local_rank}"
    setup_seed(args.seed + (dist.get_rank() if dist.is_initialized() else 0))

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    lm_config = build_config(args, len(tokenizer))
    log(
        f"[eval-config] variant={args.embedding_variant}, seed={args.seed}, "
        f"hidden={args.hidden_size}, layers={args.num_hidden_layers}, vocab={len(tokenizer)}"
    )

    model = MiniMindForCausalLM(lm_config)
    state = torch.load(args.weight_path, map_location="cpu")
    model.load_state_dict(state, strict=True)
    log(f"[eval-init] loaded {args.weight_path}")
    log("[eval-init] strict load passed")
    get_model_params(model, lm_config)
    model.to(args.device)

    if dist.is_initialized():
        model._ddp_params_and_buffers_to_ignore = {"model.freqs_cos", "model.freqs_sin"}
        model = DistributedDataParallel(model, device_ids=[local_rank], broadcast_buffers=False)

    device_type = "cuda" if "cuda" in args.device else "cpu"
    if args.dtype == "float32" or device_type == "cpu":
        autocast_ctx = nullcontext()
    else:
        amp_dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
        autocast_ctx = torch.cuda.amp.autocast(dtype=amp_dtype)

    eval_ds = SmolTalkSFTDataset(args.eval_data_path, "test", tokenizer, args.max_seq_len, args.max_eval_samples)
    eval_sampler = DistributedEvalSampler(eval_ds) if dist.is_initialized() else None
    eval_loader = DataLoader(
        eval_ds,
        batch_size=args.batch_size,
        sampler=eval_sampler,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_sft_batch,
    )

    eval_loss, eval_ppl = evaluate(args, model, eval_loader, autocast_ctx, args.eval_max_batches)
    if "cuda" in args.device:
        peak = torch.cuda.max_memory_allocated() / 1024**3
    else:
        peak = math.nan
    log(f"[eval-result] variant={args.embedding_variant}, seed={args.seed}, loss={eval_loss:.4f}, ppl={eval_ppl:.4f}, peak_alloc={peak:.2f}GiB")
    append_eval_metrics(args, {
        "event": "seed-eval",
        "eval_loss": eval_loss,
        "eval_ppl": eval_ppl,
        "eval_max_batches": args.eval_max_batches,
        "max_eval_samples": args.max_eval_samples,
    })

    raw_model(model).to("cpu")
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
