# Eval Result Archive

本目录保存从 `logs/` 和 `results/summaries/` 迁出的 eval 结果数据。`logs/` 被 git 忽略，后续需要引用结果时优先看这里。

项目级实验背景、数据指向和综合结论见 [`../实验.md`](../实验.md)；S1-S13 数学定义见
[`../s1_s13_model_variants.tex`](../s1_s13_model_variants.tex)。

## 目录说明

| 目录 | 状态 | 主要文件 |
| --- | --- | --- |
| `minimind-small-fixedinit-randsplit/` | 当前保留权重 | `all_seed_results.csv`, `three_seed_summary.csv`, `seed*/eval_pretrain_loss.*` |
| `minimind-large-tail/` | 当前保留权重 | `all_seed_results.csv`, `three_seed_summary.csv`, `per_variant/eval_seed*_s*.*` |
| `fineedu-qwen3-0.6b/` | 当前保留权重 | `all_seed_results.csv`, `three_seed_summary.csv`, `eval_pretrain_loss_qwen3_0_6b_fineedu20b_all3seeds.*` |
| `minimind-qwen3-0.6b-config-randsplit/` | 权重已删除，仅保留结果 | `all_seed_results.csv`, `three_seed_summary.csv`, `seed*/eval_pretrain_loss.*` |
| `historical-minimind-pretrain-v2-tail/` | 历史摘要，无当前整理权重 | `s1-s13-three-seed-summary.csv`, `s1-s13-per-seed-delta-vs-s1.csv` |
| `historical-fineedu-gpt2-tail/` | 历史摘要，无当前整理权重 | `fineedu-gpt2-6b-*.csv` |

进行中的 `fineedu-gpt2-6b-randsplit` 尚未完成三个 seed 和完整 eval，因此暂不创建结果归档目录。
其数据切分为 `data_splits/fineedu_gpt2_6b_random_eval_0.01_seed20260602.json`，运行入口为
`scripts/train/run_fineedu_gpt2_randsplit_6b.sh`。

## 实验索引

### minimind-small-fixedinit-randsplit

- 权重目录：`weights/final/minimind-small-fixedinit-randsplit/`
- 模型尺寸：MiniMind small dense，hidden `768`，layers `8`，约 `63.9M` 参数。
- 训练数据：`dataset/minimind/pretrain_t2t.jsonl`
- Tokenizer：本地 `model/`
- 切分方式：固定随机 eval manifest，eval ratio `0.01`，seed `20260602`。
- Split manifest：`data_splits/minimind_pretrain_t2t_random_eval_0.01_seed20260602.json`
- Seeds：`42`, `123`, `2026`
- Variants：`s1-s13`
- 训练时间：2026-06-02 到 2026-06-04。
- 结果文件：`minimind-small-fixedinit-randsplit/three_seed_summary.csv`
- 简要结论：random split 口径下 `s12` 和 `s3` 最好，untied 相关变体仍明显落后。

### minimind-large-tail

- 权重目录：`weights/final/minimind-large/`
- 模型尺寸：MiniMind large dense，hidden `1024`，layers `16`，约 `217M` 参数。
- 训练数据：`dataset/minimind/pretrain_t2t.jsonl`
- Tokenizer：本地 `model/`
- 切分方式：旧 tail split，训练前 `99%`，eval 后 `1%`。
- Eval 起点：`start_index=8384138`
- Seeds：`42`, `123`, `2026`
- Variants：`s1,s2,s3,s4,s6,s12`
- 训练时间：2026-05-30 到 2026-06-02。
- 结果文件：`minimind-large-tail/three_seed_summary.csv`
- 简要结论：三 seed 平均 `s12` 最好，`s6` 第二；`s2` 明显最差。seed42 单独看不稳定。

### minimind-qwen3-0.6b-config-randsplit

- 权重状态：原权重目录 `weights/final/minimind-qwen3-0.6b-config-randsplit/` 已删除，仅保留 eval 结果。
- 模型尺寸：Qwen3-0.6B 对齐配置，hidden `1024`，layers `28`，GQA `16/8`，约 `595.8M` 参数。
- 训练数据：`dataset/minimind/pretrain_t2t.jsonl`
- Tokenizer：`Qwen/Qwen3-0.6B`
- 切分方式：固定随机 eval manifest，eval ratio `0.01`，seed `20260602`。
- Split manifest：`data_splits/minimind_pretrain_t2t_random_eval_0.01_seed20260602.json`
- Seeds：`42`, `123`, `2026`
- Variants：`s1,s12`
- 训练时间：2026-06-06 到 2026-06-09。
- 结果文件：`minimind-qwen3-0.6b-config-randsplit/three_seed_summary.csv`
- 简要结论：MiniMind 数据配 Qwen3 tokenizer/config 时 `s12` 三个 seed 都差于 `s1`，这组权重已清理。

### fineedu-qwen3-0.6b

- 权重目录：`weights/final/fineedu-qwen3-0.6b/`
- 模型尺寸：Qwen3-0.6B 对齐配置，hidden `1024`，layers `28`，GQA `16/8`，约 `595.8M` 参数。
- 训练数据：`dataset/fineweb_edu/qwen3_modelscope_100bt_20b_packed_340`
- Tokenizer：`Qwen/Qwen3-0.6B`
- 切分方式：固定随机 eval manifest，eval ratio `0.01`，seed `20260602`。
- Split manifest：`data_splits/fineedu_qwen3_modelscope_20b_random_eval_0.01_seed20260602.json`
- Seeds：`42`, `123`, `2026`
- Variants：`s1,s3,s12`
- 训练时间：2026-06-11 到 2026-06-28。
- 结果文件：`fineedu-qwen3-0.6b/three_seed_summary.csv`
- 简要结论：换到 FineEdu 20B 后 `s12` 重新优于 `s1`，整体为 `s12 > s3 > s1`。

### historical-minimind-pretrain-v2-tail

- 权重状态：当前未保留对应整理后权重，仅保留结果摘要。
- 原始结果文件保存时间：约 2026-05-30 19:00 +0800；迁入 `docs/` 时间为 2026-06-28。
- 模型尺寸：MiniMind small dense，hidden `768`，layers `8`。
- 训练数据：`dataset/minimind/pretrain_t2t.jsonl`
- Tokenizer：本地 `model/`
- 切分方式：旧 tail split，训练前 `99%`，eval 后 `1%`。
- Eval 起点：`start_index=8384138`
- Seeds：`42`, `123`, `2026`
- Variants：`s1-s13`
- 结果文件：`historical-minimind-pretrain-v2-tail/s1-s13-three-seed-summary.csv`
- 简要结论：旧 tail 口径下 `s3` 最好，`s12` 第二；这组用于历史对照。

### historical-fineedu-gpt2-tail

- 权重状态：当前未保留对应整理后权重，仅保留结果摘要。
- 原始结果文件保存时间：约 2026-05-30 19:00 +0800；迁入 `docs/` 时间为 2026-06-28。
- 模型尺寸：MiniMind small dense，hidden `768`，layers `8`。
- 训练数据：FineWeb-Edu GPT-2 packed 6B token 数据。
- Tokenizer：`gpt2`
- 切分方式：旧 tail split，训练前 `99%`，eval 后 `1%`。
- Eval 起点：历史文档记录为 `start_index=17470587`
- Seeds：`42`, `123`, `2026`
- Variants：主要摘要覆盖 `s1,s2,s3,s4,s6,s7,s11,s12`，补充摘要包含 `s5,s13`。
- 结果文件：`historical-fineedu-gpt2-tail/fineedu-gpt2-6b-three-seed-summary.csv`
- 简要结论：主摘要旧 tail 口径下 `s12` 最好，`s3` 第二；补充的 `s1-s13` 摘要中 `s13` 三 seed 均值只比 `s3` 低约 `0.000410`，但 `s3` 赢了 2/3 个 seed，二者应视为基本持平的补充口径。

## 备注

- `three_seed_summary.csv` 是按 seed 聚合后的排序表，优先用于阅读结论。
- `all_seed_results.csv` 是从原始 eval CSV 标准化后的逐 seed/variant 明细。
- `seed*/eval_pretrain_loss.*` 或 `per_variant/eval_seed*_s*.*` 是从 `logs/` 原样迁出的结果文件。
- 已删除权重的实验仍保留 eval 结果，目的是记录“为什么不再保留权重”。
