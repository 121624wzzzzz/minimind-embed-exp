# Dynamic centering 零参数消融

本目录验证输入侧动态去均值能否解释 `s3` 相对 tied baseline `s1` 的收益。新增变体名为
`center_dynamic`，定义为

\[
\mu(W)=\frac{1}{V}\sum_{i=1}^{V}W_i,\qquad
E_{\mathrm{eff}}=W-\mathbf{1}\mu(W)^\top,\qquad
U_{\mathrm{eff}}=W.
\]

实现约束：

- 输入 lookup 后减去当前共享 embedding 表的全词表行均值；
- lm-head 仍直接使用同一个 `W`，输出路径不变；
- 不增加参数，参数名、形状和总量与 `s1` 完全一致；
- 每次 forward 重新计算均值；
- 均值先按 FP32 计算，再转换回 lookup 输出 dtype；
- `float()` 和 `to()` 均保留 autograd，不使用 `detach()`；
- special、pad 等所有词表行都参与均值。

## 正式实验口径

每个 seed 只新增训练 `center_dynamic`，直接复用已有完整记录的同 seed `s1`、`s3` 作为基线：

| 项目 | 固定值 |
| --- | --- |
| 数据 | `dataset/minimind/pretrain_t2t.jsonl` |
| 切分 | fixed random 1%，split seed `20260602` |
| 模型 | MiniMind dense，hidden 768，8 layers |
| attention | 8 query heads，4 KV heads |
| tokenizer / vocab | `model/`，全词表 6400 行参与均值 |
| seed | 默认 42；正式复验为 42、123、2026 |
| 序列长度 | 340 |
| batch | 224/GPU × 8 GPU，梯度累积 1 |
| 训练 | 2 epochs，完整 train split，不设 max steps |
| 学习率 | `5e-4`，复用原实验 cosine schedule |
| lm-head | tied，带 vocab bias，与 `s1` 相同 |
| 精度 | BF16 autocast，最终权重按原实验保存为 FP16 |

train split 含 8,384,139 条样本，每个 epoch 4,679 steps；两轮共 9,358 steps，
与已有同 seed `s1`、`s3` 的训练 token budget 相同。原实验单变体训练约 69 分钟，
本变体预计约 70–75 分钟，之后完整 eval split 评测约 1–2 分钟。

## 运行

默认运行 seed42 的正式完整实验：

```bash
bash ablation/dynamic_centering/run_seed42.sh
```

运行指定 seed：

```bash
SEED=123 bash ablation/dynamic_centering/run_seed42.sh
```

顺序补跑 seed123 和 seed2026：

```bash
bash ablation/dynamic_centering/run_remaining_seeds.sh
```

仅做训练链路 smoke test：

```bash
MAX_STEPS=2 RUN_EVAL=0 RUN_ID=smoke bash ablation/dynamic_centering/run_seed42.sh
```

脚本会先运行 `verify_implementation.py`，然后训练并评测，最后由
`summarize_results.py` 将新结果与已有同 seed `s1`、`s3` 合并为 CSV。模型权重和日志沿用
根目录的忽略规则；紧凑的正式结果保存在本目录 `results/` 下并加入 Git。

## 判读

- 若 `center_dynamic` 接近 `s3` 且明显优于 `s1`，说明强制去除当前 embedding 均值足以解释
  `s3` 的大部分收益。
- 若它接近 `s1`，而 `s3` 仍明显更好，说明收益更可能来自不受
  `beta = mean(W)` 约束的可学习自由方向。
- 若它优于 `s3`，说明训练自由 `beta` 可能没有持续跟踪最合适的动态中心。

这里只比较 held-out token NLL/PPL；单 seed 是机制消融，不用于替代三 seed 主结果。

## 三 seed 最终结果

seed42、123、2026 均已按完整口径完成。`center_dynamic` 在三个 seed 上都优于 S1，
平均 loss 为 1.733425，相对 S1 平均降低 0.004827；S3 的平均 loss 为 1.730690。
按三 seed 平均收益计算，零参数动态去均值复现了 S3 相对 S1 收益的 63.84%。完整结论见
`results/allseeds_full_20260710_2001_to_20260711_0014/README.md`。
