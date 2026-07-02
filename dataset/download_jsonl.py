#!/usr/bin/env python3
"""Download MiniMind jsonl datasets into dataset/minimind."""

from __future__ import annotations

import argparse
import concurrent.futures
import os
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


REPO_ID = "jingyaogong/minimind_dataset"
DEFAULT_BASE_URL = "https://huggingface.co"

DATASET_FILES = (
    "agent_rl.jsonl",
    "agent_rl_math.jsonl",
    "dpo.jsonl",
    "lora_exam.jsonl",
    "lora_identity.jsonl",
    "lora_medical.jsonl",
    "pretrain_t2t.jsonl",
    "pretrain_t2t_mini.jsonl",
    "rlaif.jsonl",
    "sft_t2t.jsonl",
    "sft_t2t_mini.jsonl",
)


def dataset_url(base_url: str, filename: str) -> str:
    escaped_repo = urllib.parse.quote(REPO_ID, safe="/")
    escaped_file = urllib.parse.quote(filename)
    return f"{base_url.rstrip('/')}/datasets/{escaped_repo}/resolve/main/{escaped_file}"


def request(url: str, *, method: str = "GET", headers: dict[str, str] | None = None) -> urllib.request.Request:
    req_headers = {"User-Agent": "MiniMind dataset downloader"}
    if headers:
        req_headers.update(headers)
    return urllib.request.Request(url, method=method, headers=req_headers)


def remote_size(url: str) -> int:
    with urllib.request.urlopen(request(url, method="HEAD"), timeout=30) as response:
        return int(response.headers.get("Content-Length") or 0)


def download_file(url: str, destination: Path) -> None:
    tmp_path = destination.with_suffix(destination.suffix + ".part")

    with urllib.request.urlopen(request(url)) as response, tmp_path.open("wb") as output:
        total = int(response.headers.get("Content-Length") or 0)
        downloaded = 0

        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            output.write(chunk)
            downloaded += len(chunk)
            if total:
                percent = downloaded * 100 / total
                print(f"\r  {destination.name}: {percent:5.1f}%", end="", flush=True)

    if total:
        print()
    tmp_path.replace(destination)


def download_range(
    url: str,
    tmp_path: Path,
    start: int,
    end: int,
    progress: list[int],
    index: int,
    lock: threading.Lock,
    retries: int,
) -> None:
    headers = {"Range": f"bytes={start}-{end}"}
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request(url, headers=headers), timeout=60) as response:
                if response.status != 206:
                    raise OSError(f"server did not honor Range request, status={response.status}")
                with tmp_path.open("r+b") as output:
                    output.seek(start)
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        output.write(chunk)
                        with lock:
                            progress[index] += len(chunk)
            return
        except Exception:
            if attempt + 1 == retries:
                raise
            with lock:
                progress[index] = 0
            time.sleep(2)


def download_file_threaded(url: str, destination: Path, threads: int, retries: int) -> None:
    total = remote_size(url)
    if total <= 0:
        print("  unknown size; fallback to single-thread download")
        download_file(url, destination)
        return

    threads = min(threads, total)
    tmp_path = destination.with_suffix(destination.suffix + ".part")
    with tmp_path.open("wb") as output:
        output.truncate(total)

    ranges = []
    chunk_size = total // threads
    for index in range(threads):
        start = index * chunk_size
        end = start + chunk_size - 1 if index < threads - 1 else total - 1
        ranges.append((start, end))

    progress = [0] * threads
    lock = threading.Lock()
    stop = threading.Event()

    def print_progress() -> None:
        while not stop.wait(2):
            with lock:
                done = sum(progress)
            percent = done * 100 / total
            print(f"\r  {destination.name}: {done / 1e6:.0f}/{total / 1e6:.0f} MB ({percent:5.1f}%)", end="", flush=True)

    printer = threading.Thread(target=print_progress, daemon=True)
    printer.start()

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as pool:
            futures = [
                pool.submit(download_range, url, tmp_path, start, end, progress, index, lock, retries)
                for index, (start, end) in enumerate(ranges)
            ]
            for future in concurrent.futures.as_completed(futures):
                future.result()
    finally:
        stop.set()

    print(f"\r  {destination.name}: done ({total / 1e6:.0f} MB, {threads} threads)")
    tmp_path.replace(destination)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-url",
        default=os.environ.get("HF_ENDPOINT", DEFAULT_BASE_URL),
        help="Hugging Face compatible endpoint, e.g. https://hf-mirror.com",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "minimind",
        help="Directory to save jsonl files",
    )
    parser.add_argument(
        "--include",
        nargs="+",
        choices=DATASET_FILES,
        default=list(DATASET_FILES),
        help="Subset of dataset files to download",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing files")
    parser.add_argument("--list", action="store_true", help="List dataset files and exit")
    parser.add_argument("--threads", type=int, default=1, help="Parallel Range download threads; 1 disables it")
    parser.add_argument("--retries", type=int, default=3, help="Retries per Range chunk")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.threads < 1:
        raise ValueError("--threads must be >= 1")
    if args.retries < 1:
        raise ValueError("--retries must be >= 1")

    if args.list:
        print("\n".join(DATASET_FILES))
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for filename in args.include:
        destination = args.output_dir / filename
        if destination.exists() and not args.force:
            print(f"skip existing: {destination}")
            continue

        url = dataset_url(args.base_url, filename)
        print(f"download: {url}")
        try:
            if args.threads == 1:
                download_file(url, destination)
            else:
                download_file_threaded(url, destination, args.threads, args.retries)
        except (urllib.error.URLError, OSError) as exc:
            print(f"failed: {filename}: {exc}", file=sys.stderr)
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
