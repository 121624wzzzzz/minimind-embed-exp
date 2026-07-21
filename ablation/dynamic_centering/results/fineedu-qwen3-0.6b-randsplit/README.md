# FineEdu20B × Qwen3-0.6B dynamic centering 三 seed 结果

本目录归档 `center_dynamic` 在当前最大规模配置上的正式复验。S1、S3、S12 直接复用同一
模型、数据、切分和训练超参数下已有的三个 seed 基线；本次只新增训练零参数
`center_dynamic`。

## 实验口径

| 项目 | 固定值 |
| --- | --- |
| 模型 | Qwen3-0.6B 对齐配置，约 595.93M 参数 |
| 架构 | hidden 1024，28 layers，GQA 16/8，head dim 128，FFN 3072 |
| 数据 | FineWeb-Edu 20B，Qwen3 tokenizer，packed seq len 340 |
| 切分 | fixed random 1% eval，split seed `20260602` |
| train / eval | 58,235,294 / 588,235 条序列 |
| seeds | 42、123、2026 |
| batch | 32/GPU × 8 GPU，梯度累积 1，有效 batch 256 |
| 训练 | 1 epoch，每个 seed 227,482 steps |
| 学习率 | `5e-4`，cosine schedule |
| lm-head | tied，带 vocab bias |
| dynamic 操作 | 输入 lookup 后减去当前共享 embedding 表的全词表均值；输出权重不变 |

固定 split manifest 为
`data_splits/fineedu_qwen3_20b_random_eval_0.01_seed20260602.json`。每项 held-out 评测均
覆盖 588,235 条序列、199,411,665 个有效预测 token。

## 三 seed 结果

| variant | seed42 | seed123 | seed2026 | mean loss | pop. std | PPL at mean loss | mean delta vs S1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `s1` | 2.755773 | 2.762714 | 2.757664 | 2.758717 | 0.002930 | 15.779585 | +0.000000 |
| `center_dynamic` | 2.771955 | 2.799285 | 2.799276 | 2.790172 | 0.012881 | 16.283820 | **+0.031455** |
| `s3` | 2.741402 | 2.758531 | 2.751138 | 2.750357 | 0.007015 | 15.648217 | -0.008360 |
| `s12` | 2.745604 | 2.751806 | 2.744375 | **2.747262** | 0.003252 | **15.599856** | -0.011455 |

这里的汇总 PPL 定义为 `exp(mean loss)`，逐 seed 的原始 PPL 见结果 CSV。

`center_dynamic` 在三个 seed 上均差于 S1，逐 seed loss 差值分别为 +0.016182、
+0.036571、+0.041612。三 seed 平均上，它相对 S1、S3、S12 分别高 +0.031455
(+1.140%)、+0.039815 (+1.448%)、+0.042910 (+1.562%)。

## 结论

MiniMind-small 上观察到的动态去均值收益没有扩展到 Qwen3-0.6B × FineEdu20B：在本次
更大模型、更大数据和更长训练口径下，零参数 `center_dynamic` 稳定退化，并且 seed 间方差
也高于三组基线。因此它可以继续作为解释小模型 S3 收益的机制消融，但当前结果不支持把它
当作跨规模稳定有效的独立训练技巧。

这种规模反转本身是重要结果，后续更值得检验的是允许保留一定可学习偏置或加入乘性自由度，
而不是把 embedding 均值严格固定为零。

## 文件

- `eval_pretrain_loss.csv`：本次三个 seed 的原始 `center_dynamic` held-out 指标；
- `eval_summary.csv`：四个变体的三 seed 汇总；
- `per_seed_comparison.csv`：逐 seed 的 S1 / dynamic / S3 / S12 对比。

未纳入 Git 的原始运行目录为
`logs/fineedu-qwen3-0.6b/20260715_fineedu20b_center_dynamic/`；最终权重与 resume checkpoint
分别位于 `weights/final/fineedu-qwen3-0.6b-center-dynamic/` 和
`weights/resume/fineedu-qwen3-0.6b-center-dynamic/`。正式运行从 2026-07-15 19:45:39 到
2026-07-21 10:39:16。
