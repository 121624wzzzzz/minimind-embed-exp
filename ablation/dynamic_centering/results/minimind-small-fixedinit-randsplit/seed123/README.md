# seed123 完整 dynamic centering 消融结果

状态：已完成。训练与评测从 2026-07-10 21:16:23 运行至 22:26:55，共 4,232 秒。

| variant | held-out loss | PPL | delta vs S1 | delta vs S3 |
| --- | ---: | ---: | ---: | ---: |
| `s1` | 1.738371 | 5.688070 | +0.000000 | +0.011750 |
| `center_dynamic` | **1.734176** | **5.664259** | **-0.004195** | +0.007555 |
| `s3` | 1.726621 | 5.621626 | -0.011750 | +0.000000 |

dynamic centering 在 seed123 上稳定优于 S1，但只复现了
`0.004195 / 0.011750 = 35.70%` 的 S3 收益，S3 仍明显领先。完整评测覆盖 84,688 条
序列、17,625,568 个有效预测 token。

最终权重：
`weights/final/minimind-small-dynamic-centering-randsplit/seed123/center_dynamic_768.pth`。
