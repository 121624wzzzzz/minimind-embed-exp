# SFT 实验

本目录用于管理预训练模型的监督微调（SFT）实验。源码、日志、评测结果和模型权重按功能分开存放；模型权重保留在本地，但不纳入 Git。

## 目录结构

```text
sft/
├── train_sft.py
├── eval_sft.py
├── summarize_results.py
├── run_fineedu_qwen3_smol_sft.sh
├── smol-smoltalk/   # 本地数据集快照
├── logs/            # 训练与评测日志
├── results/         # Git 跟踪的 JSONL/CSV 评测结果
└── weights/         # final 与 resume 模型文件
```

## smol-smoltalk 数据

通过 ModelScope 下载 SFT 数据集：

```bash
/home/wz/anaconda3/envs/torch24/bin/python -c 'from modelscope.hub.snapshot_download import snapshot_download; snapshot_download(repo_id="HuggingFaceTB/smol-smoltalk", repo_type="dataset", local_dir="sft/smol-smoltalk", allow_patterns=["data/*.parquet"], max_workers=2)'
```

预期文件如下：

```text
sft/smol-smoltalk/data/test-00000-of-00001.parquet
sft/smol-smoltalk/data/train-00000-of-00004.parquet
sft/smol-smoltalk/data/train-00001-of-00004.parquet
sft/smol-smoltalk/data/train-00002-of-00004.parquet
sft/smol-smoltalk/data/train-00003-of-00004.parquet
```

## FineEdu Qwen3-0.6B SFT

单变体冒烟测试：

```bash
MAX_STEPS=2 MAX_TRAIN_SAMPLES=64 MAX_EVAL_SAMPLES=32 \
GPUS=0 SEEDS="42" VARIANTS=s1 \
bash sft/run_fineedu_qwen3_smol_sft.sh
```

运行三个预训练 seed、三个 embedding 变体的完整实验：

```bash
GPUS=0,1,2,3,4,5,6,7 BATCH_SIZE=12 SAVE_INTERVAL=0 \
USE_ATTENTION_MASK=0 SEEDS="42 123 2026" VARIANTS=s1,s3,s12 \
RUN_SEED_EVAL=1 \
bash sft/run_fineedu_qwen3_smol_sft.sh
```

在 A100 80GB、`max_seq_len=1024`、全参数 SFT 条件下，每卡 `BATCH_SIZE=12` 是已验证稳定的配置。右侧 padding 不传入模型 attention mask，从而继续使用 Flash/SDPA；因果注意力保证响应 token 不会看到位于其后的 padding。8 卡有效 batch 为 96，每个 epoch 约 4,795 个优化 step。双卡基准中的每卡 batch 14 和 16 均发生显存不足。

## 数据格式与 loss

- 从 parquet 的 `messages` 列读取对话。
- 使用 Qwen3 ChatML 模板渲染，`add_generation_prompt=False`。
- loss 只覆盖 assistant 内容以及紧随其后的 `<|im_end|>`；user、system、模板 token 和 padding 的 label 均为 `-100`。
- smol-smoltalk 不含 Qwen3 `<think>` 块。推理时使用默认 generation prompt；除非训练格式同步改变，否则不要人为加入空 think 前缀。

## 评测与汇总

- 训练使用 `train-*.parquet`，最终指标固定使用 `test-00000-of-00001.parquet`。
- 每个变体的训练内评测追加到 `sft/results/.../seed*/*_metrics.jsonl`。
- `RUN_SEED_EVAL=1` 时，一个 seed 的所有变体训练完成后，runner 会重新加载 final 权重，并将独立评测写入 `sft/results/.../seed*/eval_after_seed/seed*_eval_summary.jsonl`。
- 所有 seed 完成后，`sft/summarize_results.py` 生成 `eval_summary.csv`。CSV 包含逐 seed loss、逐 seed 相对 s1 的配对差值、三 seed 平均 loss、总体标准差、平均 PPL，以及相对 s1 的三 seed 平均配对差值。
- 差值定义为 `variant_loss - s1_loss`，因此负数优于 s1。
- 控制台输出和逐变体日志放在 `sft/logs/`；便于 Git 跟踪和查阅的 JSONL/CSV 放在 `sft/results/`。

当前正式结果位于：

```text
sft/results/main_3seed_assistant_loss/
```

## 序列长度审计

固定抽取 12,000 条训练样本得到：

```text
max_seq_len  label_retention  no_label_samples
512          47.84%           2.53%
640          59.04%           1.91%
768          69.23%           1.66%
896          78.22%           1.43%
1024         86.62%           1.20%
```

主实验建议保持 `MAX_SEQ_LEN=1024`。降到 896 会少保留约 8.4 个百分点的 assistant 监督 token。若更重视耗时，可使用已经测试过的速度优先配置：

```bash
GPUS=0,1 BATCH_SIZE=14 MAX_SEQ_LEN=896 SAVE_INTERVAL=0 \
USE_ATTENTION_MASK=0 SEEDS="42" VARIANTS=s1,s3,s12 \
bash sft/run_fineedu_qwen3_smol_sft.sh
```

双卡测试中，`MAX_SEQ_LEN=896` 配合每卡 batch 15 或 16 会显存不足。
