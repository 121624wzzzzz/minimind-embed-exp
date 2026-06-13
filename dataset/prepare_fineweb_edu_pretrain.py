#!/usr/bin/env python3
"""Prepare FineWeb-Edu parquet files as packed MiniMind pretrain blocks."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from datasets import Dataset, Features, Sequence, Value, concatenate_datasets, load_dataset
from transformers import AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "fineweb_edu" / "raw" / "sample" / "10BT",
        help="Directory containing FineWeb-Edu parquet files or cached Arrow shards.",
    )
    parser.add_argument(
        "--input-format",
        choices=("auto", "parquet", "arrow"),
        default="auto",
        help="How to read --input-dir. Arrow mode reads cached raw text shards.",
    )
    parser.add_argument(
        "--arrow-pattern",
        default="parquet-train-*.arrow",
        help="Arrow shard glob used when --input-format=arrow.",
    )
    parser.add_argument("--hf-dataset", default="", help="Hugging Face dataset name, e.g. HuggingFaceFW/fineweb-edu.")
    parser.add_argument("--hf-config", default="", help="Hugging Face dataset config, e.g. sample-100BT.")
    parser.add_argument("--hf-data-files", default="", help="Explicit HF/parquet data_files pattern for streaming mode.")
    parser.add_argument("--hf-split", default="train", help="Hugging Face dataset split.")
    parser.add_argument("--streaming", action="store_true", help="Stream from Hugging Face and write packed parquet shards.")
    parser.add_argument("--stream-local", action="store_true", help="Stream local parquet files directly and write packed parquet shards.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "fineweb_edu" / "gpt2_packed",
        help="Directory for the packed Hugging Face dataset.",
    )
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=Path("gpt2"),
        help="Tokenizer path or Hugging Face name.",
    )
    parser.add_argument("--max-seq-len", type=int, default=340, help="Packed training sequence length.")
    parser.add_argument("--num-proc", type=int, default=8, help="Parallel processes for dataset.map.")
    parser.add_argument("--batch-size", type=int, default=1000, help="Raw documents per map batch.")
    parser.add_argument("--writer-batch-size", type=int, default=1000, help="Arrow writer batch size.")
    parser.add_argument("--min-chars", type=int, default=20, help="Drop very short documents before tokenization.")
    parser.add_argument("--max-samples", type=int, default=0, help="Debug only: process first N raw rows.")
    parser.add_argument("--max-tokens", type=int, default=6_000_000_000, help="Keep first N packed tokens; 0 means no limit.")
    parser.add_argument("--file-limit", type=int, default=0, help="Debug only: process first N parquet files.")
    parser.add_argument("--parquet-shard-rows", type=int, default=50_000, help="Packed rows per parquet shard in streaming mode.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output directory if it exists.")
    return parser.parse_args()


def input_files(args: argparse.Namespace) -> tuple[str, list[str]]:
    input_format = args.input_format
    if input_format == "auto":
        if sorted(args.input_dir.glob("*.parquet")):
            input_format = "parquet"
        elif sorted(args.input_dir.glob(args.arrow_pattern)):
            input_format = "arrow"
        else:
            raise FileNotFoundError(
                f"No parquet files or {args.arrow_pattern!r} Arrow shards found in {args.input_dir}"
            )

    if input_format == "parquet":
        files = sorted(args.input_dir.glob("*.parquet"))
    else:
        files = sorted(args.input_dir.glob(args.arrow_pattern))

    if not files:
        raise FileNotFoundError(f"No {input_format} files found in {args.input_dir}")
    return input_format, [str(path) for path in files]


def load_raw_dataset(input_format: str, data_files: list[str]):
    if input_format == "parquet":
        return load_dataset("parquet", data_files=data_files, split="train")
    shards = [Dataset.from_file(path) for path in data_files]
    if len(shards) == 1:
        return shards[0]
    return concatenate_datasets(shards)


def token_id_dtype_for(tokenizer, eos_id: int) -> str:
    max_token_id = max(tokenizer.vocab_size - 1, len(tokenizer) - 1, eos_id)
    return "uint16" if max_token_id <= 65_535 else "uint32"


def write_streaming_packed(args: argparse.Namespace, tokenizer, eos_id: int, token_id_dtype: str) -> int:
    import pyarrow as pa
    import pyarrow.parquet as pq

    if not args.hf_dataset and not args.hf_data_files:
        raise ValueError("--streaming requires --hf-dataset or --hf-data-files")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    value_type = pa.uint16() if token_id_dtype == "uint16" else pa.uint32()
    schema = pa.schema([("input_ids", pa.list_(value_type))])
    if args.hf_data_files:
        hf_kwargs = {
            "path": "parquet",
            "data_files": args.hf_data_files,
            "split": args.hf_split,
            "streaming": True,
        }
    else:
        hf_kwargs = {
            "path": args.hf_dataset,
            "split": args.hf_split,
            "streaming": True,
        }
        if args.hf_config:
            hf_kwargs["name"] = args.hf_config
    raw = load_dataset(**hf_kwargs)

    max_blocks = args.max_tokens // args.max_seq_len if args.max_tokens > 0 else 0
    rows: list[list[int]] = []
    token_buffer: list[int] = []
    buffer_offset = 0
    text_batch: list[str] = []
    raw_rows = 0
    kept_rows = 0
    shard_index = 0

    def flush_rows() -> None:
        nonlocal rows, shard_index
        if not rows:
            return
        table = pa.Table.from_arrays([pa.array(rows, type=pa.list_(value_type))], schema=schema)
        out = args.output_dir / f"packed-{shard_index:05d}.parquet"
        pq.write_table(table, out, compression="zstd")
        print(f"[prepare] wrote {out.name}: rows={len(rows):,}, total_tokens={kept_rows * args.max_seq_len:,}", flush=True)
        rows = []
        shard_index += 1

    def consume_text_batch() -> None:
        nonlocal text_batch, raw_rows, kept_rows, token_buffer, buffer_offset
        if not text_batch:
            return
        encodings = tokenizer(text_batch, add_special_tokens=False).input_ids
        for ids in encodings:
            if ids:
                token_buffer.extend(ids)
                token_buffer.append(eos_id)
            while buffer_offset + args.max_seq_len <= len(token_buffer):
                if max_blocks > 0 and kept_rows >= max_blocks:
                    break
                rows.append(token_buffer[buffer_offset: buffer_offset + args.max_seq_len])
                buffer_offset += args.max_seq_len
                kept_rows += 1
                if len(rows) >= args.parquet_shard_rows:
                    flush_rows()
            if max_blocks > 0 and kept_rows >= max_blocks:
                break
            if buffer_offset > 1_000_000:
                token_buffer = token_buffer[buffer_offset:]
                buffer_offset = 0
        text_batch = []

    print(f"[prepare] hf dataset : {args.hf_dataset}")
    print(f"[prepare] hf config  : {args.hf_config or '(default)'}")
    print(f"[prepare] hf files   : {args.hf_data_files or '(from config)'}")
    print(f"[prepare] hf split   : {args.hf_split}")
    print(f"[prepare] output dir : {args.output_dir}")
    print(f"[prepare] tokenizer  : {args.tokenizer}")
    print(f"[prepare] seq len    : {args.max_seq_len}")
    print(f"[prepare] max tokens : {args.max_tokens:,}")
    print(f"[prepare] token dtype: {token_id_dtype}")

    for sample in raw:
        if args.max_samples > 0 and raw_rows >= args.max_samples:
            break
        raw_rows += 1
        text = sample.get("text")
        if not isinstance(text, str) or len(text.strip()) < args.min_chars:
            continue
        text_batch.append(text)
        if len(text_batch) >= args.batch_size:
            consume_text_batch()
        if max_blocks > 0 and kept_rows >= max_blocks:
            break

    consume_text_batch()
    flush_rows()

    metadata = {
        "source": args.hf_dataset,
        "hf_dataset": args.hf_dataset,
        "hf_config": args.hf_config,
        "hf_data_files": args.hf_data_files,
        "hf_split": args.hf_split,
        "streaming": True,
        "num_input_files": 0,
        "raw_rows_seen": raw_rows,
        "num_rows": kept_rows,
        "max_seq_len": args.max_seq_len,
        "tokenizer": str(args.tokenizer),
        "vocab_size": len(tokenizer),
        "token_id_dtype": token_id_dtype,
        "min_chars": args.min_chars,
        "max_samples": args.max_samples,
        "max_tokens": args.max_tokens,
        "kept_tokens": kept_rows * args.max_seq_len,
        "parquet_shard_rows": args.parquet_shard_rows,
        "shards": shard_index,
    }
    (args.output_dir / "preprocess_meta.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"[prepare] packed rows: {kept_rows:,}")
    print(f"[prepare] tokens     : {kept_rows * args.max_seq_len:,}")
    print(f"[prepare] shards     : {shard_index:,}")
    return 0


def write_local_streaming_packed(args: argparse.Namespace, tokenizer, eos_id: int, token_id_dtype: str) -> int:
    import pyarrow as pa
    import pyarrow.parquet as pq

    input_format, data_files = input_files(args)
    if input_format != "parquet":
        raise ValueError("--stream-local currently supports only parquet input")
    if args.file_limit > 0:
        data_files = data_files[: args.file_limit]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    value_type = pa.uint16() if token_id_dtype == "uint16" else pa.uint32()
    schema = pa.schema([("input_ids", pa.list_(value_type))])
    max_blocks = args.max_tokens // args.max_seq_len if args.max_tokens > 0 else 0
    rows: list[list[int]] = []
    token_buffer: list[int] = []
    buffer_offset = 0
    raw_rows = 0
    kept_rows = 0
    shard_index = 0

    def flush_rows() -> None:
        nonlocal rows, shard_index
        if not rows:
            return
        table = pa.Table.from_arrays([pa.array(rows, type=pa.list_(value_type))], schema=schema)
        out = args.output_dir / f"packed-{shard_index:05d}.parquet"
        pq.write_table(table, out, compression="zstd")
        print(f"[prepare] wrote {out.name}: rows={len(rows):,}, total_tokens={kept_rows * args.max_seq_len:,}", flush=True)
        rows = []
        shard_index += 1

    def consume_texts(texts: list[str]) -> bool:
        nonlocal kept_rows, token_buffer, buffer_offset
        encodings = tokenizer(texts, add_special_tokens=False).input_ids
        for ids in encodings:
            if ids:
                token_buffer.extend(ids)
                token_buffer.append(eos_id)
            while buffer_offset + args.max_seq_len <= len(token_buffer):
                if max_blocks > 0 and kept_rows >= max_blocks:
                    return True
                rows.append(token_buffer[buffer_offset: buffer_offset + args.max_seq_len])
                buffer_offset += args.max_seq_len
                kept_rows += 1
                if len(rows) >= args.parquet_shard_rows:
                    flush_rows()
            if buffer_offset > 1_000_000:
                token_buffer = token_buffer[buffer_offset:]
                buffer_offset = 0
        return max_blocks > 0 and kept_rows >= max_blocks

    print(f"[prepare] input files: {len(data_files)}")
    print("[prepare] input format: parquet-stream")
    print(f"[prepare] output dir : {args.output_dir}")
    print(f"[prepare] tokenizer  : {args.tokenizer}")
    print(f"[prepare] seq len    : {args.max_seq_len}")
    print(f"[prepare] max tokens : {args.max_tokens:,}")
    print(f"[prepare] token dtype: {token_id_dtype}")

    done = False
    for file_index, data_file in enumerate(data_files, start=1):
        parquet = pq.ParquetFile(data_file)
        for batch in parquet.iter_batches(batch_size=args.batch_size, columns=["text"]):
            texts: list[str] = []
            for value in batch.column(0).to_pylist():
                if args.max_samples > 0 and raw_rows >= args.max_samples:
                    done = True
                    break
                raw_rows += 1
                if isinstance(value, str) and len(value.strip()) >= args.min_chars:
                    texts.append(value)
            if texts and consume_texts(texts):
                done = True
            if done:
                break
        print(
            f"[prepare] scanned {file_index}/{len(data_files)} files, raw_rows={raw_rows:,}, tokens={kept_rows * args.max_seq_len:,}",
            flush=True,
        )
        if done:
            break

    flush_rows()

    metadata = {
        "source": str(args.input_dir),
        "input_format": "parquet-stream",
        "streaming": False,
        "local_streaming": True,
        "num_input_files": len(data_files),
        "raw_rows_seen": raw_rows,
        "num_rows": kept_rows,
        "max_seq_len": args.max_seq_len,
        "tokenizer": str(args.tokenizer),
        "vocab_size": len(tokenizer),
        "token_id_dtype": token_id_dtype,
        "min_chars": args.min_chars,
        "file_limit": args.file_limit,
        "max_samples": args.max_samples,
        "max_tokens": args.max_tokens,
        "kept_tokens": kept_rows * args.max_seq_len,
        "parquet_shard_rows": args.parquet_shard_rows,
        "shards": shard_index,
    }
    (args.output_dir / "preprocess_meta.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"[prepare] packed rows: {kept_rows:,}")
    print(f"[prepare] tokens     : {kept_rows * args.max_seq_len:,}")
    print(f"[prepare] shards     : {shard_index:,}")
    return 0


def main() -> int:
    args = parse_args()
    if args.output_dir.exists() and not args.overwrite:
        raise FileExistsError(f"{args.output_dir} already exists; pass --overwrite to replace it")
    if args.output_dir.exists() and args.overwrite:
        shutil.rmtree(args.output_dir)

    tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer), trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        raise ValueError("Tokenizer must define eos_token_id")

    token_id_dtype = token_id_dtype_for(tokenizer, eos_id)
    if args.streaming:
        return write_streaming_packed(args, tokenizer, eos_id, token_id_dtype)
    if args.stream_local:
        return write_local_streaming_packed(args, tokenizer, eos_id, token_id_dtype)

    input_format, data_files = input_files(args)
    if args.file_limit > 0:
        data_files = data_files[: args.file_limit]
    print(f"[prepare] input files: {len(data_files)}")
    print(f"[prepare] input format: {input_format}")
    print(f"[prepare] output dir : {args.output_dir}")
    print(f"[prepare] tokenizer  : {args.tokenizer}")
    print(f"[prepare] seq len    : {args.max_seq_len}")

    raw = load_raw_dataset(input_format, data_files)
    if args.max_samples > 0:
        raw = raw.select(range(min(args.max_samples, len(raw))))
        print(f"[prepare] debug rows : {len(raw)}")

    def tokenize_and_pack(batch):
        buffer: list[int] = []
        for text in batch["text"]:
            if not isinstance(text, str) or len(text.strip()) < args.min_chars:
                continue
            ids = tokenizer(text, add_special_tokens=False).input_ids
            if ids:
                buffer.extend(ids)
                buffer.append(eos_id)

        usable = len(buffer) // args.max_seq_len * args.max_seq_len
        blocks = [
            buffer[i : i + args.max_seq_len]
            for i in range(0, usable, args.max_seq_len)
        ]
        return {"input_ids": blocks}

    max_token_id = max(tokenizer.vocab_size - 1, len(tokenizer) - 1, eos_id)
    print(f"[prepare] token dtype: {token_id_dtype} (max token id: {max_token_id})")

    map_kwargs = {
        "batched": True,
        "batch_size": args.batch_size,
        "remove_columns": raw.column_names,
        "features": Features({"input_ids": Sequence(Value(token_id_dtype))}),
        "writer_batch_size": args.writer_batch_size,
        "desc": "tokenize and pack",
    }
    if args.num_proc > 1:
        map_kwargs["num_proc"] = args.num_proc
    packed = raw.map(tokenize_and_pack, **map_kwargs)
    if args.max_tokens > 0:
        max_blocks = args.max_tokens // args.max_seq_len
        if max_blocks <= 0:
            raise ValueError("--max-tokens must be at least --max-seq-len")
        if len(packed) > max_blocks:
            packed = packed.select(range(max_blocks))
    packed.save_to_disk(str(args.output_dir))

    metadata = {
        "source": str(args.input_dir),
        "input_format": input_format,
        "arrow_pattern": args.arrow_pattern if input_format == "arrow" else "",
        "num_input_files": len(data_files),
        "num_rows": len(packed),
        "max_seq_len": args.max_seq_len,
        "tokenizer": str(args.tokenizer),
        "vocab_size": len(tokenizer),
        "token_id_dtype": token_id_dtype,
        "min_chars": args.min_chars,
        "file_limit": args.file_limit,
        "max_samples": args.max_samples,
        "max_tokens": args.max_tokens,
        "kept_tokens": len(packed) * args.max_seq_len,
    }
    (args.output_dir / "preprocess_meta.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"[prepare] packed rows: {len(packed):,}")
    print(f"[prepare] tokens     : {len(packed) * args.max_seq_len:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
