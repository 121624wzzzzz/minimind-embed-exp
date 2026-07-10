import argparse
import json
import math
import os
import sys
import time
import warnings
from contextlib import nullcontext
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / "sft/.hf_home"))
os.environ.setdefault("HF_DATASETS_CACHE", str(PROJECT_ROOT / "sft/.hf_datasets_cache"))
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import torch.distributed as dist
from datasets import load_dataset
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler, Dataset, Sampler
from transformers import AutoTokenizer

sys.path.append(str(PROJECT_ROOT))

from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from model.variant_config import VALID_VARIANTS, variant_tie_word_embeddings
from trainer.trainer_utils import get_lr, get_model_params, setup_seed

warnings.filterwarnings("ignore")


def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0


def log(message):
    if is_main_process():
        print(message, flush=True)


def init_distributed_mode():
    if int(os.environ.get("RANK", -1)) == -1:
        return 0
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def configure_deterministic_mode(strict_deterministic, sdp_backend="flash"):
    if strict_deterministic != 1:
        log("[determinism] strict mode disabled; optimized CUDA kernels may not be bitwise reproducible")
        return
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        if sdp_backend == "math":
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            torch.backends.cuda.enable_math_sdp(True)
        elif sdp_backend == "flash":
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            torch.backends.cuda.enable_math_sdp(False)
        else:
            raise ValueError(f"unsupported deterministic SDP backend: {sdp_backend}")
    log(
        f"[determinism] strict mode enabled: deterministic algorithms, "
        f"{sdp_backend} SDPA, TF32 disabled"
    )


def split_csv(value):
    return [item.strip() for item in value.split(",") if item.strip()]


def resolve_parquet_files(data_path, split):
    path = Path(data_path)
    if path.is_file():
        return [str(path)]
    if not path.exists():
        raise FileNotFoundError(f"data path does not exist: {data_path}")

    candidates = []
    if split == "train":
        candidates = sorted((path / "data").glob("train-*.parquet")) if (path / "data").is_dir() else []
        candidates = candidates or sorted(path.glob("train-*.parquet"))
    elif split == "test":
        candidates = sorted((path / "data").glob("test-*.parquet")) if (path / "data").is_dir() else []
        candidates = candidates or sorted(path.glob("test-*.parquet"))
    else:
        candidates = sorted((path / "data").glob("*.parquet")) if (path / "data").is_dir() else []
        candidates = candidates or sorted(path.glob("*.parquet"))

    if not candidates:
        raise FileNotFoundError(f"no parquet files found under {data_path} for split={split}")
    return [str(p) for p in candidates]


class SmolTalkSFTDataset(Dataset):
    def __init__(self, data_path, split, tokenizer, max_length=1024, max_samples=0):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        if self.pad_token_id is None:
            raise ValueError("tokenizer must define pad_token_id or eos_token_id")
        self.im_end = "<|im_end|>"
        files = resolve_parquet_files(data_path, split)
        self.samples = load_dataset("parquet", data_files=files, split="train")
        if max_samples and max_samples > 0:
            self.samples = self.samples.select(range(min(max_samples, len(self.samples))))
        log(f"[data] split={split}, files={len(files)}, samples={len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def _normalize_messages(self, messages):
        normalized = []
        for message in messages:
            role = str(message.get("role", "")).strip()
            content = "" if message.get("content") is None else str(message.get("content"))
            if not role:
                continue
            normalized.append({"role": role, "content": content})
        return normalized

    def _assistant_spans(self, text, messages):
        spans = []
        cursor = 0
        for message in messages:
            if message.get("role") != "assistant":
                continue
            content = message.get("content", "")
            if not content:
                continue

            header = "<|im_start|>assistant\n"
            header_idx = text.find(header, cursor)
            search_from = header_idx + len(header) if header_idx >= 0 else cursor
            start = text.find(content, search_from)
            if start < 0:
                start = text.find(content, cursor)
            if start < 0:
                continue

            content_end = start + len(content)
            end_marker = text.find(self.im_end, content_end)
            label_end = end_marker + len(self.im_end) if end_marker >= 0 else content_end
            spans.append((start, label_end))
            cursor = label_end
        return spans

    def __getitem__(self, index):
        sample = self.samples[index]
        messages = self._normalize_messages(sample["messages"])
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        spans = self._assistant_spans(text, messages)
        encoded = self.tokenizer(
            text,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_length,
            return_offsets_mapping=True,
        )
        input_ids = encoded["input_ids"]
        offsets = encoded["offset_mapping"]
        labels = [-100] * len(input_ids)

        span_idx = 0
        for pos, (start, end) in enumerate(offsets):
            while span_idx < len(spans) and start >= spans[span_idx][1]:
                span_idx += 1
            if span_idx < len(spans):
                span_start, span_end = spans[span_idx]
                if end > start and start < span_end and end > span_start:
                    labels[pos] = input_ids[pos]

        attention_mask = [1] * len(input_ids)
        pad_len = self.max_length - len(input_ids)
        if pad_len > 0:
            input_ids.extend([self.pad_token_id] * pad_len)
            labels.extend([-100] * pad_len)
            attention_mask.extend([0] * pad_len)

        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(labels, dtype=torch.long),
            torch.tensor(attention_mask, dtype=torch.long),
        )


def collate_sft_batch(features):
    input_ids, labels, attention_mask = zip(*features)
    attention_mask = torch.stack(attention_mask, dim=0)
    max_len = int(attention_mask.sum(dim=1).max().item())
    return (
        torch.stack(input_ids, dim=0)[:, :max_len],
        torch.stack(labels, dim=0)[:, :max_len],
        attention_mask[:, :max_len],
    )


class DistributedEvalSampler(Sampler):
    def __init__(self, dataset):
        self.dataset = dataset
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1

    def __iter__(self):
        return iter(range(self.rank, len(self.dataset), self.world_size))

    def __len__(self):
        if len(self.dataset) <= self.rank:
            return 0
        return (len(self.dataset) - 1 - self.rank) // self.world_size + 1


def build_config(args, vocab_size):
    required_tie = int(variant_tie_word_embeddings(args.embedding_variant))
    if args.tie_word_embeddings != required_tie:
        log(
            f"[variant] {args.embedding_variant} sets tie_word_embeddings={required_tie} "
            f"(was {args.tie_word_embeddings})"
        )
        args.tie_word_embeddings = required_tie
    intermediate_size = args.intermediate_size if args.intermediate_size > 0 else math.ceil(args.hidden_size * math.pi / 64) * 64
    return MiniMindConfig(
        vocab_size=vocab_size,
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        num_attention_heads=args.num_attention_heads,
        num_key_value_heads=args.num_key_value_heads,
        head_dim=args.head_dim if args.head_dim > 0 else args.hidden_size // args.num_attention_heads,
        intermediate_size=intermediate_size,
        max_position_embeddings=args.max_position_embeddings,
        rope_theta=args.rope_theta,
        rms_norm_eps=args.rms_norm_eps,
        use_moe=bool(args.use_moe),
        tie_word_embeddings=bool(args.tie_word_embeddings),
        lm_head_bias=bool(args.lm_head_bias),
        embedding_variant=args.embedding_variant,
        embedding_variant_rank=args.embedding_variant_rank,
    )


def raw_model(model):
    model = model.module if isinstance(model, DistributedDataParallel) else model
    return getattr(model, "_orig_mod", model)


def state_dict_for_save(model, save_dtype):
    dtype = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[save_dtype]
    return {
        key: value.detach().to(device="cpu", dtype=dtype) if value.is_floating_point() else value.detach().cpu()
        for key, value in raw_model(model).state_dict().items()
    }


def save_state(args, lm_config, model, optimizer, scaler, epoch, step, tag="final"):
    if not is_main_process():
        return
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    model_state = state_dict_for_save(model, args.save_dtype)
    is_final = str(tag) in {"final", "epoch_end"}
    if is_final:
        final_path = Path(args.output_dir) / f"{args.save_weight}_{lm_config.hidden_size}.pth"
        tmp_final = final_path.with_suffix(final_path.suffix + ".tmp")
        torch.save(model_state, tmp_final)
        os.replace(tmp_final, final_path)
        log(f"[save] final={final_path}")
    else:
        checkpoint_path = Path(args.checkpoint_dir) / f"{args.save_weight}_{lm_config.hidden_size}_{tag}.pth"
        tmp_checkpoint = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
        torch.save(model_state, tmp_checkpoint)
        os.replace(tmp_checkpoint, checkpoint_path)
        log(f"[save] checkpoint={checkpoint_path}")

    if args.save_resume == 1:
        resume_path = Path(args.checkpoint_dir) / f"{args.save_weight}_{lm_config.hidden_size}_resume.pth"
        resume = {
            "model": model_state,
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "step": step,
            "tag": tag,
            "args": vars(args),
            "world_size": dist.get_world_size() if dist.is_initialized() else 1,
        }
        tmp_resume = resume_path.with_suffix(resume_path.suffix + ".tmp")
        torch.save(resume, tmp_resume)
        os.replace(tmp_resume, resume_path)
        log(f"[save] resume={resume_path}")


def append_metrics(args, row):
    if not is_main_process() or not args.metrics_path:
        return
    path = Path(args.metrics_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "save_weight": args.save_weight,
        "variant": args.embedding_variant,
        "seed": args.seed,
        "train_data_path": args.train_data_path,
        "eval_data_path": args.eval_data_path,
        "init_weight_path": args.init_weight_path,
        "save_dtype": args.save_dtype,
        "strict_deterministic": bool(args.strict_deterministic),
        "sdp_backend": args.sdp_backend,
        **row,
    }
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


@torch.no_grad()
def evaluate(args, model, loader, autocast_ctx, max_batches=0):
    model.eval()
    device = args.device
    loss_sum = torch.zeros(2, device=device, dtype=torch.float64)
    for batch_idx, (input_ids, labels, attention_mask) in enumerate(loader, start=1):
        if max_batches > 0 and batch_idx > max_batches:
            break
        input_ids = input_ids.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        attention_mask = attention_mask.to(device, non_blocking=True)
        label_tokens = labels[:, 1:].ne(-100).sum()
        if label_tokens.item() == 0:
            continue
        with autocast_ctx:
            model_attention_mask = attention_mask if args.use_attention_mask == 1 else None
            outputs = model(input_ids, attention_mask=model_attention_mask, labels=labels)
            loss = outputs.loss
        if torch.isfinite(loss):
            loss_sum[0] += loss.detach().double() * label_tokens.double()
            loss_sum[1] += label_tokens.double()

    if dist.is_initialized():
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
    model.train()
    if loss_sum[1].item() == 0:
        return float("nan"), float("nan")
    loss = (loss_sum[0] / loss_sum[1]).item()
    return loss, math.exp(min(loss, 20.0))


def load_resume(args, lm_config, model, optimizer, scaler):
    if args.from_resume != 1:
        return 0, 0
    resume_path = Path(args.checkpoint_dir) / f"{args.save_weight}_{lm_config.hidden_size}_resume.pth"
    if not resume_path.exists():
        return 0, 0
    data = torch.load(resume_path, map_location="cpu")
    raw_model(model).load_state_dict(data["model"], strict=True)
    optimizer.load_state_dict(data["optimizer"])
    scaler.load_state_dict(data["scaler"])
    log(f"[resume] loaded {resume_path}")
    return int(data.get("epoch", 0)), int(data.get("step", 0))


def parse_args():
    parser = argparse.ArgumentParser(description="MiniMind smol-smoltalk SFT")
    parser.add_argument("--train_data_path", default=str(PROJECT_ROOT / "sft/smol-smoltalk"), type=str)
    parser.add_argument("--eval_data_path", default=str(PROJECT_ROOT / "sft/smol-smoltalk"), type=str)
    parser.add_argument("--tokenizer_path", default="Qwen/Qwen3-0.6B", type=str)
    parser.add_argument("--init_weight_path", required=True, type=str)
    parser.add_argument("--output_dir", required=True, type=str)
    parser.add_argument("--checkpoint_dir", required=True, type=str)
    parser.add_argument("--save_weight", required=True, type=str)
    parser.add_argument("--epochs", default=1, type=int)
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--accumulation_steps", default=1, type=int)
    parser.add_argument("--learning_rate", default=2e-5, type=float)
    parser.add_argument("--grad_clip", default=1.0, type=float)
    parser.add_argument("--max_seq_len", default=1024, type=int)
    parser.add_argument("--max_steps", default=0, type=int)
    parser.add_argument("--max_train_samples", default=0, type=int)
    parser.add_argument("--max_eval_samples", default=0, type=int)
    parser.add_argument("--eval_interval", default=0, type=int)
    parser.add_argument("--eval_max_batches", default=0, type=int)
    parser.add_argument("--save_interval", default=1000, type=int)
    parser.add_argument("--save_final", default=1, type=int, choices=[0, 1])
    parser.add_argument("--save_resume", default=1, type=int, choices=[0, 1])
    parser.add_argument(
        "--save_dtype",
        default="float16",
        choices=["float32", "float16", "bfloat16"],
        help="Checkpoint floating tensor dtype. Downstream experiments should keep float32 for small updates.",
    )
    parser.add_argument("--use_attention_mask", default=0, type=int, choices=[0, 1])
    parser.add_argument("--log_interval", default=20, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--strict_deterministic", default=0, type=int, choices=[0, 1])
    parser.add_argument("--sdp_backend", default="flash", choices=["math", "flash"])
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu", type=str)
    parser.add_argument("--from_resume", default=0, type=int, choices=[0, 1])
    parser.add_argument("--metrics_path", default="", type=str)
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
    return parser.parse_args()


def main():
    args = parse_args()
    args.embedding_variant = args.embedding_variant.lower()
    local_rank = init_distributed_mode()
    if dist.is_initialized():
        args.device = f"cuda:{local_rank}"
    setup_seed(args.seed + (dist.get_rank() if dist.is_initialized() else 0))
    configure_deterministic_mode(args.strict_deterministic, args.sdp_backend)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    lm_config = build_config(args, len(tokenizer))
    log(
        f"[config] variant={args.embedding_variant}, seed={args.seed}, "
        f"hidden={args.hidden_size}, layers={args.num_hidden_layers}, vocab={len(tokenizer)}"
    )

    model = MiniMindForCausalLM(lm_config)
    init_state = torch.load(args.init_weight_path, map_location="cpu")
    model.load_state_dict(init_state, strict=True)
    log(f"[init] loaded {args.init_weight_path}")
    log("[init] strict load passed")
    get_model_params(model, lm_config)
    log(f"Trainable Params: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f}M")
    model.to(args.device)

    train_ds = SmolTalkSFTDataset(args.train_data_path, "train", tokenizer, args.max_seq_len, args.max_train_samples)
    eval_ds = SmolTalkSFTDataset(args.eval_data_path, "test", tokenizer, args.max_seq_len, args.max_eval_samples)
    train_sampler = DistributedSampler(train_ds, shuffle=True, seed=args.seed) if dist.is_initialized() else None
    eval_sampler = DistributedEvalSampler(eval_ds) if dist.is_initialized() else None
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=train_sampler,
        shuffle=train_sampler is None,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_sft_batch,
    )
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

    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == "float16"))
    start_epoch, start_step = load_resume(args, lm_config, model, optimizer, scaler)

    if dist.is_initialized():
        model._ddp_params_and_buffers_to_ignore = {"model.freqs_cos", "model.freqs_sin"}
        model = DistributedDataParallel(model, device_ids=[local_rank], broadcast_buffers=False)

    device_type = "cuda" if "cuda" in args.device else "cpu"
    if args.dtype == "float32" or device_type == "cpu":
        autocast_ctx = nullcontext()
    else:
        amp_dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
        autocast_ctx = torch.cuda.amp.autocast(dtype=amp_dtype)

    total_batches = len(train_loader)
    target_batches = min(total_batches, args.max_steps) if args.max_steps > 0 else total_batches
    total_schedule_steps = max(args.epochs * target_batches, 1)
    log(
        f"[train] batches_per_epoch={total_batches}, target_batches={target_batches}, "
        f"batch_size_per_rank={args.batch_size}, accumulation={args.accumulation_steps}"
    )

    global_step = start_step
    for epoch in range(start_epoch, args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model.train()
        start_time = time.time()
        optimizer.zero_grad(set_to_none=True)
        no_label_sequences = 0
        seen_sequences = 0
        for batch_idx, (input_ids, labels, attention_mask) in enumerate(train_loader, start=1):
            if args.max_steps > 0 and batch_idx > args.max_steps:
                break
            input_ids = input_ids.to(args.device, non_blocking=True)
            labels = labels.to(args.device, non_blocking=True)
            attention_mask = attention_mask.to(args.device, non_blocking=True)
            per_sequence_labels = labels[:, 1:].ne(-100).sum(dim=1)
            no_label_sequences += int(per_sequence_labels.eq(0).sum().item())
            seen_sequences += int(labels.size(0))
            local_label_tokens = per_sequence_labels.sum()

            lr = get_lr(epoch * target_batches + batch_idx, total_schedule_steps, args.learning_rate)
            for group in optimizer.param_groups:
                group["lr"] = lr

            with autocast_ctx:
                model_attention_mask = attention_mask if args.use_attention_mask == 1 else None
                if local_label_tokens.item() > 0:
                    outputs = model(input_ids, attention_mask=model_attention_mask, labels=labels)
                    aux_loss = outputs.aux_loss if outputs.aux_loss is not None else input_ids.new_zeros(()).float()
                    if not torch.isfinite(outputs.loss):
                        raise RuntimeError(f"non-finite SFT loss at epoch={epoch + 1}, batch={batch_idx}: {outputs.loss.item()}")
                    loss = outputs.loss + aux_loss
                else:
                    outputs = model(input_ids, attention_mask=model_attention_mask, labels=None)
                    loss = outputs.logits.sum() * 0.0
                loss = loss / args.accumulation_steps
            scaler.scale(loss).backward()

            if batch_idx % args.accumulation_steps == 0:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                if not torch.isfinite(grad_norm):
                    raise RuntimeError(f"non-finite grad norm at epoch={epoch + 1}, batch={batch_idx}: {grad_norm.item()}")
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

            if batch_idx % args.log_interval == 0 or batch_idx == target_batches:
                elapsed = time.time() - start_time
                batches_done = max(batch_idx, 1)
                eta_min = elapsed / batches_done * max(target_batches - batch_idx, 0) / 60
                current_loss = loss.item() * args.accumulation_steps
                mem_info = ""
                if "cuda" in args.device:
                    mem_info = f", peak_alloc={torch.cuda.max_memory_allocated() / 1024**3:.2f}GiB"
                log(
                    f"Epoch:[{epoch + 1}/{args.epochs}]({batch_idx}/{target_batches}), "
                    f"loss={current_loss:.4f}, lr={lr:.8f}, eta={eta_min:.1f}min, "
                    f"no_label_seq={no_label_sequences}/{seen_sequences}{mem_info}"
                )

            if args.eval_interval > 0 and batch_idx % args.eval_interval == 0:
                eval_loss, eval_ppl = evaluate(args, model, eval_loader, autocast_ctx, args.eval_max_batches)
                log(f"[eval] epoch={epoch + 1}, batch={batch_idx}, loss={eval_loss:.4f}, ppl={eval_ppl:.4f}")
                append_metrics(args, {
                    "event": "eval",
                    "epoch": epoch + 1,
                    "batch": batch_idx,
                    "global_step": global_step,
                    "eval_loss": eval_loss,
                    "eval_ppl": eval_ppl,
                    "eval_max_batches": args.eval_max_batches,
                })

            if args.save_interval > 0 and batch_idx % args.save_interval == 0:
                save_state(args, lm_config, model, optimizer, scaler, epoch, global_step, tag=f"batch{batch_idx}")

        if args.save_final == 1:
            save_state(args, lm_config, model, optimizer, scaler, epoch + 1, global_step, tag="epoch_end")
        eval_loss, eval_ppl = evaluate(args, model, eval_loader, autocast_ctx, args.eval_max_batches)
        log(f"[eval-final] epoch={epoch + 1}, loss={eval_loss:.4f}, ppl={eval_ppl:.4f}")
        label_stats = torch.tensor([no_label_sequences, seen_sequences], device=args.device, dtype=torch.long)
        if dist.is_initialized():
            dist.all_reduce(label_stats, op=dist.ReduceOp.SUM)
        append_metrics(args, {
            "event": "eval-final",
            "epoch": epoch + 1,
            "global_step": global_step,
            "eval_loss": eval_loss,
            "eval_ppl": eval_ppl,
            "eval_max_batches": args.eval_max_batches,
            "no_label_sequences": int(label_stats[0].item()),
            "seen_sequences": int(label_stats[1].item()),
        })

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
