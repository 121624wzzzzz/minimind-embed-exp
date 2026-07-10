# seed42 完整 dynamic centering 消融结果

状态：已完成。训练从 2026-07-10 20:01:01 运行至 21:10:26，包含完整训练、最终权重
保存、完整 fixed-random eval split 评测和结果汇总。

## 结果

三项均使用相同的 seed42、模型规模、训练数据与 split；`s1`、`s3` 直接复用已有正式记录，
本次只新增 `center_dynamic`。

| variant | held-out loss | PPL | delta vs S1 | delta vs S3 |
| --- | ---: | ---: | ---: | ---: |
| `s1` | 1.741829 | 5.707773 | +0.000000 | +0.009010 |
| `center_dynamic` | **1.734228** | **5.664553** | **-0.007601** | +0.001409 |
| `s3` | 1.732819 | 5.656577 | -0.009010 | +0.000000 |

完整评测覆盖 84,688 条 held-out 序列和 17,625,568 个有效预测 token。

以 S3 相对 S1 的 loss 收益为分母，dynamic centering 复现的比例为

\[
\frac{1.741829-1.734228}{1.741829-1.732819}=84.36\%.
\]

S3 仍比 dynamic centering 低 0.001409 loss，对应剩余约 15.64% 的 S3 收益。

## 结论

该结果支持“输入 embedding 的公共均值方向是 S3 收益的主要来源”：仅使用当前共享词表均值、
不增加任何参数，就复现了约 84.4% 的 seed42 收益。同时它没有完全追平 S3，说明自由训练且
不受 `beta = mean(W)` 约束的 `beta` 仍提供了小幅额外收益。

这是单 seed 机制消融。它足以说明 dynamic centering 不是无效对照，但 0.001409 的残余差异
不应在没有更多 seed 的情况下解释为稳定的总体效应。

## 等价性与实现检查

- 正式 768×8 配置下，S1 和 `center_dynamic` 均为 63,918,592 个参数；
- 参数名和 state-dict tensor 形状完全一致；
- 同随机种子构造时，初始参数逐 tensor 完全相同；
- embedding 与 lm-head 共享同一个权重存储；
- 输出 logits 路径不变；
- 全词表行参与均值，未被 lookup 的行也通过均值项收到梯度；
- 修改未使用词表行后，下一次 forward 会得到新的均值。

训练配置为 hidden 768、8 layers、8 卡、每卡 batch 224、seq len 340、2 epochs、9,358
steps、初始学习率 `5e-4`、BF16 autocast。峰值 allocated 显存为 35.93 GiB/GPU。

## 文件

- `comparison.csv/json`：S1、dynamic centering、S3 统一对比；
- `eval_pretrain_loss.csv/json`：新变体的原始完整评测；
- `implementation_checks.json`：自动化实现约束检查；
- `run_metadata.json`：训练、数据、权重哈希与关键数值。

最终权重位于被 Git 忽略的
`weights/final/minimind-small-dynamic-centering-randsplit/seed42/center_dynamic_768.pth`。
