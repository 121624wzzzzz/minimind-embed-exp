# 实验脚本说明

本目录按功能存放 MiniMind embedding/head 变体实验脚本。脚本都会自动定位到项目根目录，因此推荐从项目根运行。

## 目录结构

| 目录 | 作用 |
| --- | --- |
| `experiments/` | 当前实验总入口，负责串起 split、训练、eval |
| `train/` | 训练执行脚本，逐个 variant 调用 `trainer/train_pretrain.py` |
| `eval/` | 并行 eval 脚本 |
| `data/` | 数据切分和数据准备工具 |

历史维护脚本已经移除，不再保留 backfill 和 large 训练自动重启脚本。

## 实验入口

`experiments/run_minimind_small_fixedinit.sh`

固定初始化顺序和随机 eval manifest 后的 MiniMind small 补跑入口。默认配置：

| 参数 | 默认值 |
| --- | --- |
| `SEEDS` | `42 123 2026` |
| `VARIANTS` | `s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11,s12,s13` |
| `WEIGHT_NAMESPACE` | `minimind-small-fixedinit-randsplit` |
| `SPLIT_SEED` | `20260602` |
| `EVAL_SPLIT_RATIO` | `0.01` |
| `BATCH_SIZE` | `224` |
| `EPOCHS` | `2` |
| `RUN_EVAL` | `1` |
| `EVAL_MODE` | `variants` |

常用命令：

```bash
bash scripts/experiments/run_minimind_small_fixedinit.sh
SEEDS="42" VARIANTS=s1,s3,s12 bash scripts/experiments/run_minimind_small_fixedinit.sh
RUN_EVAL=0 MAX_STEPS=10 bash scripts/experiments/run_minimind_small_fixedinit.sh
```

默认输出：

```text
weights/final/minimind-small-fixedinit-randsplit/seed42/s1_768.pth
weights/resume/minimind-small-fixedinit-randsplit/seed42/s1_768_resume.pth
logs/minimind-small-fixedinit/<run_id>/seed42/s1.log
logs/minimind-small-fixedinit/<run_id>/seed42/eval_pretrain_loss.csv
```

`experiments/run_minimind_large.sh`

MiniMind large 实验入口。默认 hidden=1024、layers=16，跑核心变体 `s1,s2,s3,s4,s6,s12`。

默认不传 `SPLIT_MANIFEST_PATH` 时使用旧版兼容切分：训练前 99%，eval 后 1%，并自动保存到 `WEIGHT_NAMESPACE=minimind-large-tail`。如果传入 `SPLIT_MANIFEST_PATH`，则使用固定随机 manifest split，并自动保存到 `WEIGHT_NAMESPACE=minimind-large-randsplit`。

```bash
bash scripts/experiments/run_minimind_large.sh
SEEDS="42" VARIANTS=s1,s3,s12 bash scripts/experiments/run_minimind_large.sh
RUN_EVAL=0 MAX_STEPS=10 BATCH_SIZE=48 bash scripts/experiments/run_minimind_large.sh
```

默认输出：

```text
weights/final/minimind-large-tail/seed42/s1_1024.pth
weights/resume/minimind-large-tail/seed42/s1_1024_resume.pth
logs/minimind-large/<run_id>/seed42/s1.log
logs/minimind-large/<run_id>/seed42/eval_pretrain_loss.csv
```

Manifest split 输出：

```text
weights/final/minimind-large-randsplit/seed42/s1_1024.pth
weights/resume/minimind-large-randsplit/seed42/s1_1024_resume.pth
```

`experiments/run_fineedu_gpt2.sh`

FineEdu packed 数据 + GPT-2 tokenizer 实验入口。默认跑 `s1,s2,s3,s6,s12`，训练后自动 eval。

```bash
bash scripts/experiments/run_fineedu_gpt2.sh
SEEDS="42" VARIANTS=s1,s3,s12 bash scripts/experiments/run_fineedu_gpt2.sh
RUN_EVAL=0 MAX_STEPS=5 bash scripts/experiments/run_fineedu_gpt2.sh
```

默认输出：

```text
weights/final/fineedu-gpt2/seed42/s1_768.pth
weights/resume/fineedu-gpt2/seed42/s1_768_resume.pth
logs/fineedu-gpt2/<run_id>/seed42/s1.log
logs/fineedu-gpt2/<run_id>/seed42/eval_pretrain_loss.csv
```

`experiments/run_minimind_qwen3_0_6b_config.sh`

MiniMind 数据上的 Qwen3-0.6B 尺寸配置入口。这个入口对齐 Qwen/Qwen3-0.6B 的主要结构参数：

| 参数 | 默认值 |
| --- | --- |
| `HIDDEN_SIZE` | `1024` |
| `NUM_HIDDEN_LAYERS` | `28` |
| `NUM_ATTENTION_HEADS` | `16` |
| `NUM_KEY_VALUE_HEADS` | `8` |
| `HEAD_DIM` | `128` |
| `INTERMEDIATE_SIZE` | `3072` |
| `MAX_POSITION_EMBEDDINGS` | `40960` |
| `ROPE_THETA` | `1000000` |
| `RMS_NORM_EPS` | `1e-6` |
| `LM_HEAD_BIAS` | `0` |
| `TOKENIZER_PATH` | `Qwen/Qwen3-0.6B` |

默认不传 `SPLIT_MANIFEST_PATH` 时使用旧版兼容切分，并自动保存到 `WEIGHT_NAMESPACE=minimind-qwen3-0.6b-config-tail`。传入 `SPLIT_MANIFEST_PATH` 时使用固定随机 manifest split，并自动保存到 `WEIGHT_NAMESPACE=minimind-qwen3-0.6b-config-randsplit`。

```bash
bash scripts/experiments/run_minimind_qwen3_0_6b_config.sh
SEEDS="42" VARIANTS=s1,s3,s12 bash scripts/experiments/run_minimind_qwen3_0_6b_config.sh
RUN_EVAL=0 MAX_STEPS=10 BATCH_SIZE=16 bash scripts/experiments/run_minimind_qwen3_0_6b_config.sh
```

默认输出：

```text
weights/final/minimind-qwen3-0.6b-config-tail/seed42/s1_1024.pth
weights/resume/minimind-qwen3-0.6b-config-tail/seed42/s1_1024_resume.pth
logs/minimind-qwen3-0.6b-config/<run_id>/seed42/s1.log
logs/minimind-qwen3-0.6b-config/<run_id>/seed42/eval_pretrain_loss.csv
```

如果需要只对齐结构但继续使用 MiniMind tokenizer，可以显式覆盖：

```bash
TOKENIZER_PATH=../model EVAL_TOKENIZER_PATH=model bash scripts/experiments/run_minimind_qwen3_0_6b_config.sh
```

## 训练脚本

`train/train_minimind_small_variants.sh`

MiniMind small dense variants 的通用训练脚本。默认 hidden=768、layers=8，可跑 `s1-s13`。它通常由 `experiments/run_minimind_small_fixedinit.sh` 调用，也可以单独运行。

```bash
bash scripts/train/train_minimind_small_variants.sh
VARIANTS=s3,s4 bash scripts/train/train_minimind_small_variants.sh
START=s4 END=s13 bash scripts/train/train_minimind_small_variants.sh
```

`train/train_large_pretrain.sh`

MiniMind large / Qwen3-size dense 配置共用的训练脚本。默认 hidden=1024、layers=16，核心变体为 `s1,s2,s3,s6,s12`；也支持由 experiment 入口传入 attention heads、FFN 中间层、RoPE 等结构参数。

```bash
bash scripts/train/train_large_pretrain.sh
VARIANTS=s1,s3 SEED=42 bash scripts/train/train_large_pretrain.sh
MAX_STEPS=10 BATCH_SIZE=48 bash scripts/train/train_large_pretrain.sh
```

`train/train_fineedu_gpt2_pretrain.sh`

FineWeb-Edu packed 数据 + GPT-2 tokenizer 的训练脚本。默认跑 `s1,s2,s3,s6,s12`，训练后自动 eval。

```bash
bash scripts/train/train_fineedu_gpt2_pretrain.sh
RUN_EVAL=0 VARIANTS=s1,s3 bash scripts/train/train_fineedu_gpt2_pretrain.sh
MAX_STEPS=5 DATA_PATH=../dataset/fineweb_edu/smoke bash scripts/train/train_fineedu_gpt2_pretrain.sh
```

## Eval 脚本

`eval/eval_parallel.sh`

统一的多 GPU loss/PPL eval 入口，不包含数据集、seed、variant 或模型尺寸的硬编码。
实验入口负责传入配置，脚本负责 GPU 调度、失败检查和结果合并，最终输出 seed 级别的
`eval_pretrain_loss.csv/json`。传入 `SUMMARY_DIR` 时，还会扫描其中的 `seed*/` 结果并更新
`eval_pretrain_loss_all_seeds.csv/json`。

- `EVAL_MODE=variants`：不同 GPU 评测不同 variants，适合 MiniMind 规模的 eval 集。
- `EVAL_MODE=shards`：所有 GPU 对同一 variant 做数据分片，适合 FineEdu 大规模 eval 集。

```bash
PY=/home/wz/anaconda3/envs/torch24/bin/python \
GPU_LIST=0,1,2,3,4,5,6,7 \
EVAL_MODE=variants \
VARIANTS=s1,s3,s12 \
SAVE_DIR=weights/final/minimind-large-tail/seed42 \
DATA_PATH=dataset/minimind/pretrain_t2t.jsonl \
TOKENIZER_PATH=model \
OUTPUT_DIR=logs/manual-eval/seed42 \
TAIL_RATIO=0.01 \
HIDDEN_SIZE=1024 \
NUM_HIDDEN_LAYERS=16 \
bash scripts/eval/eval_parallel.sh
```

`run_minimind_small_fixedinit.sh`、`run_minimind_large.sh` 和
`run_minimind_qwen3_0_6b_config.sh` 默认使用 `variants`；
`run_fineedu_qwen3_0_6b.sh` 默认使用 `shards`。可通过 `EVAL_MODE` 覆盖。

## 数据脚本

`dataset/prepare_fineweb_edu_pretrain.py`

将 FineWeb-Edu parquet 文本使用指定 tokenizer 打包成固定长度的预训练序列。当前
FineEdu GPT-2 6B 数据来自 `sample/10BT` 的 14 个 parquet，使用 GPT-2 tokenizer、
seq len `340`，并按顺序保留前 6B token。将同一组原始 parquet 放到下述 input 目录后，
可以先在临时目录复现并校验：

```bash
/home/wz/anaconda3/envs/torch24/bin/python dataset/prepare_fineweb_edu_pretrain.py \
  --input-dir dataset/fineweb_edu/raw/gpt2_sample_10bt \
  --input-format parquet \
  --output-dir /tmp/fineedu_gpt2_6b_rebuild \
  --tokenizer gpt2 \
  --max-seq-len 340 \
  --min-chars 20 \
  --max-tokens 6000000000 \
  --num-proc 8 \
  --batch-size 1000 \
  --overwrite
```

当前 packed 目录曾在项目整理时从同一次预处理的 Hugging Face datasets cache 恢复，
cache prefix 为 `cache-edfc1356ffbd6c6a`。恢复过程按原 shard 顺序截取 17,647,058 条，
并校验了 schema、长度、token ID 范围和 GPT-2 解码内容；这不是不同来源的数据替换。
统计见 `dataset/fineweb_edu/packed/gpt2_6b_seq340/preprocess_meta.json`。

`data/create_split_manifest.py`

生成固定随机 eval indices manifest。训练使用 manifest 的补集，eval 使用 manifest 中的 eval indices。

```bash
/home/wz/anaconda3/envs/torch24/bin/python scripts/data/create_split_manifest.py \
  --data_path dataset/minimind/pretrain_t2t.jsonl \
  --output data_splits/minimind_pretrain_t2t_random_eval_0.01_seed20260602.json \
  --eval_ratio 0.01 \
  --seed 20260602
```

## 通用训练逻辑

训练脚本的外层逻辑基本一致：

1. 设置 `torch24`、CUDA、HF、PIP 等环境变量。
2. 根据 `GPUS` 计算 `torchrun --nproc_per_node` 的进程数。
3. 解析 variants。
4. 按 variant 串行训练；每个 variant 内部使用 DDP 多 GPU。
5. 调用 `trainer/train_pretrain.py`，并把日志写入对应 `LOG_DIR`。

脚本层面不是多个 variant 并发跑，而是：

```text
s1 -> torchrun 多 GPU
s2 -> torchrun 多 GPU
s3 -> torchrun 多 GPU
...
```

## 保存逻辑

`train_pretrain.py` 有两类保存：

| 类型 | 内容 | 文件名 |
| --- | --- | --- |
| final 权重 | 仅模型 `state_dict` | `{save_dir}/{save_weight}_{hidden_size}.pth` |
| resume 快照 | model、optimizer、scaler、epoch、step、world_size、wandb_id | `{checkpoint_dir}/{save_weight}_{hidden_size}_resume.pth` |

当前实验入口统一使用 namespace 结构保存：

```text
weights/final/<namespace>/seed42/s1_768.pth
weights/resume/<namespace>/seed42/s1_768_resume.pth
```
