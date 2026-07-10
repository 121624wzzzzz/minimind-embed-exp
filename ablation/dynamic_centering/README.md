# Dynamic centering 零参数消融

状态：**三个 seed 的完整训练与评测均已完成**。

本实验用于判断：`s3` 相对 tied baseline `s1` 的收益，有多少可以由“去除共享 embedding
表的全词表均值”这一固定规则解释。

## 变体定义

新增变体 `center_dynamic`：

\[
\mu(W)=\frac{1}{V}\sum_{i=1}^{V}W_i,\qquad
E_{\mathrm{eff}}=W-\mathbf{1}\mu(W)^\top,\qquad
U_{\mathrm{eff}}=W.
\]

具体行为：

- 输入 lookup 后减去当前共享 embedding 表的全词表行均值；
- 输出 lm-head 仍直接使用同一个 `W`，logits 路径不变；
- 不增加参数，每次 forward 重新计算均值；
- 均值使用 FP32 计算，再转回 lookup 输出 dtype；
- 计算保留 autograd，不使用 `detach()`；
- special、pad 等全部 6,400 个词表行都参与均值。

它与其他变体的区别是：

- `s1`：输入直接使用 `W[id]`；
- `center_dynamic`：固定使用 `W[id] - mean(W)`，零参数；
- `s3`：使用 `W[id] - beta`，其中 `beta` 是不受 `beta = mean(W)` 约束的可学习参数。

## 等价性检查

正式 MiniMind 768×8 配置下，`s1` 与 `center_dynamic` 均为 **63,918,592** 个参数。
自动检查确认：

- 参数名、形状和总量与 `s1` 完全一致；
- 同 seed 构造时，初始参数逐 tensor 完全相同；
- embedding 与 lm-head 共享同一权重存储；
- 输出 logits 路径未改变；
- 未出现在 lookup 中的词表行也会通过均值项收到梯度；
- 修改任意未使用词表行后，下一次 forward 会重新得到新的均值。

这里的“初始化相同”指参数初始化相同；两者从第一次 forward 开始就因动态去均值而具有不同
的输入激活，这是本消融唯一有意引入的结构差异。

## 实验配置

| 项目 | 固定值 |
| --- | --- |
| 数据 | `dataset/minimind/pretrain_t2t.jsonl` |
| 切分 | fixed random 1%，split seed `20260602` |
| train / eval | 8,384,139 / 84,688 条序列 |
| 模型 | MiniMind dense，hidden 768，8 layers |
| attention | 8 query heads，4 KV heads |
| seeds | 42、123、2026 |
| 序列长度 | 340 |
| batch | 224/GPU × 8 GPU，梯度累积 1 |
| 训练 | 2 epochs，每轮 4,679 steps，共 9,358 steps |
| 学习率 | `5e-4`，复用原实验 cosine schedule |
| lm-head | tied，带 vocab bias，与 `s1` 相同 |
| 精度 | BF16 autocast，最终权重按原实验保存为 FP16 |

每个 seed 只新增训练 `center_dynamic`；`s1`、`s3` 直接复用相同 seed、相同配置的既有正式
结果。每项 held-out 评测覆盖 17,625,568 个有效预测 token。

## 三 seed 结果

| variant | seed42 | seed123 | seed2026 | mean loss | std | mean delta vs S1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `s1` | 1.741829 | 1.738371 | 1.734557 | 1.738252 | 0.002970 | +0.000000 |
| `center_dynamic` | **1.734228** | **1.734176** | **1.731871** | **1.733425** | 0.001099 | **-0.004827** |
| `s3` | 1.732819 | 1.726621 | 1.732631 | 1.730690 | 0.002878 | -0.007562 |

`center_dynamic` 在三个 seed 上都优于 `s1`，平均降低 0.004827 loss；`s3` 的三 seed
平均仍比它低 0.002735。

### S3 收益的恢复比例

| seed | S3 gain vs S1 | dynamic gain vs S1 | dynamic / S3 gain |
| --- | ---: | ---: | ---: |
| 42 | 0.009010 | 0.007601 | 84.36% |
| 123 | 0.011750 | 0.004195 | 35.70% |
| 2026 | 0.001926 | 0.002686 | 139.46% |

最终采用三 seed **平均收益之比**作为主结果：

\[
\frac{1.738252333-1.733425000}{1.738252333-1.730690333}=63.84\%.
\]

不使用三个逐 seed 比例的简单平均，因为 seed2026 的 S3 收益分母只有 0.001926，比例对
微小变化非常敏感。

## 结论

动态去除当前全词表均值本身即可稳定优于 tied baseline，并解释 `s3` 平均收益的约
**63.8%**。因此，embedding 公共均值方向是 `s3` 收益的重要来源，但不是完整解释。

`s3` 的自由可学习 `beta` 仍提供额外平均收益，而且这部分作用具有明显 seed 依赖：
seed123 中 `s3` 优势较大，seed2026 中固定动态中心反而略优于 `s3`。

## 结果目录

正式结果使用稳定的语义路径，运行时间只保留在 `run_metadata.json` 和日志中：

```text
results/minimind-small-fixedinit-randsplit/
├── README.md
├── eval_summary.csv
├── eval_summary.json
├── mechanism_recovery.csv
├── per_seed_comparison.csv
├── seed42/
├── seed123/
└── seed2026/
```

主要入口：

- [完整中文结果](results/minimind-small-fixedinit-randsplit/README.md)
- [三 seed 汇总](results/minimind-small-fixedinit-randsplit/eval_summary.csv)
- [机制恢复比例](results/minimind-small-fixedinit-randsplit/mechanism_recovery.csv)
- [逐 seed 对比](results/minimind-small-fixedinit-randsplit/per_seed_comparison.csv)

## 复现

运行单个 seed，默认 seed42：

```bash
bash ablation/dynamic_centering/run_seed.sh
SEED=123 bash ablation/dynamic_centering/run_seed.sh
```

顺序运行 seed123、seed2026，并在三个 seed 文件齐全后自动刷新汇总：

```bash
bash ablation/dynamic_centering/run_remaining_seeds.sh
```

`run_seed.sh` 会先执行实现约束检查，再完成训练、完整评测和同 seed S1/S3 对比。
最终权重保存在 `weights/final/minimind-small-dynamic-centering-randsplit/seed<seed>/`，不加入
Git；源码、检查报告和紧凑结果加入 Git。

## 运行记录

seed42、seed123、seed2026 的完整训练与评测时间分别约为 69、71、108 分钟，精确起止时间
记录在各 seed 的 `run_metadata.json` 中。
