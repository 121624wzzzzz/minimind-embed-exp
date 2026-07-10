# FineEdu Qwen3-0.6B 在 SmolTalk 上的 SFT 结果

本目录保存纯监督微调（SFT）实验的正式三 seed 结果。主指标为固定测试集上的
assistant-token cross-entropy（eval loss）；PPL 为三 seed 平均 loss 的指数变换。
该结果不包含任何生成式外部基准评测。

## 实验设置

- 初始模型：本项目从随机初始化预训练得到的 FineEdu Qwen3-0.6B 对齐配置权重，约
  595.8M 参数；预训练语料为 FineWeb-Edu 20B token。
- SFT 任务：`HuggingFaceTB/smol-smoltalk`。
- 比较变体：`s1`（tied baseline）、`s3`（输入 embedding 全局平移）和
  `s12`（`s3+s6`）。
- 随机种子：42、123、2026。每个 seed 的 SFT 从相同 seed、相同变体的预训练 final
  权重开始。
- 训练：全参数 SFT，1 epoch，AdamW，学习率 `2e-5`，bfloat16，最大序列长度 1024，
  梯度裁剪 1.0。
- 硬件与 batch：8 × A100 80GB；每卡 batch size 12、梯度累积 1，有效 batch size 96，
  因而每个 epoch 约 4,795 个优化 step。
- 对话与 loss：采用 Qwen3 ChatML 模板。仅对 assistant 内容和紧随其后的
  `<|im_end|>` 计算 loss；user、system、模板 token 与 padding 全部 mask 为 `-100`。

## 数据量与评测

| 划分 | 文件 | 样本数 | 用途 |
| --- | --- | ---: | --- |
| train | `train-00000-of-00004.parquet` 至 `train-00003-of-00004.parquet` | 460,341 | SFT 训练 |
| test | `test-00000-of-00001.parquet` | 24,229 | 最终 held-out 评测 |

所有变体均在完整 test split 上重新加载 final 权重后评测。`eval loss` 是每个被监督的
assistant token 的平均负对数似然；数值越低越好。`delta vs s1` 定义为
`variant_loss - s1_loss`，故负数表示优于 s1。

## 三 seed 最终结果

每个 seed 的括号内是 `variant - s1` 的 loss 配对差值；负数表示优于 s1。
`PPL` 为 `exp(mean loss)`。最后两列先在每个 seed 内计算配对差值，再报告三个 seed 的
`mean ± sample SD`（`ddof=1`）；PPL 差值按
`exp(loss_variant) - exp(loss_s1)` 逐 seed 计算，不能用平均 loss 的 PPL 直接相减。
配对差值由未显示的原始精度 loss 计算，故与表中六位小数 loss 直接相减时可能有末位差异。

| 变体 | seed42 loss (Δ vs s1) | seed123 loss (Δ vs s1) | seed2026 loss (Δ vs s1) | mean loss ± SD | PPL | paired loss Δ ± sample SD | paired PPL Δ ± sample SD |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| s1 | 1.348365 (0.000000) | 1.361640 (0.000000) | 1.349187 (0.000000) | 1.353064 ± 0.006073 | 3.869263 | 0.000000 ± 0.000000 | 0.000000 ± 0.000000 |
| s3 | 1.331062 (-0.017303) | 1.356669 (-0.004971) | 1.345507 (-0.003681) | 1.344412 ± 0.010482 | 3.835932 | -0.008652 ± 0.007520 | -0.033191 ± 0.028586 |
| s12 | 1.338309 (-0.010056) | 1.343713 (-0.017927) | 1.333366 (-0.015821) | **1.338463 ± 0.004225** | **3.813176** | **-0.014602 ± 0.004075** | **-0.056123 ± 0.015862** |

按 held-out assistant-token eval loss，三 seed 平均排序为 `s12 > s3 > s1`；其中 `s12`
在三个 seed 上均低于 s1；同时，s12 的平均配对改善大于 s3，跨 seed 波动也更小。

原始汇总见 [`eval_summary.csv`](eval_summary.csv)；各 seed 的重新加载 final 权重后的评测
明细位于 `seed*/eval_after_seed/`。
