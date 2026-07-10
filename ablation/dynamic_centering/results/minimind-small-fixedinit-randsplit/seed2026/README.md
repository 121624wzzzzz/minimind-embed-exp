# seed2026 完整 dynamic centering 消融结果

状态：已完成。训练与评测从 2026-07-10 22:26:55 运行至 2026-07-11 00:14:49，
共 6,474 秒。

| variant | held-out loss | PPL | delta vs S1 | delta vs S3 |
| --- | ---: | ---: | ---: | ---: |
| `s1` | 1.734557 | 5.666417 | +0.000000 | +0.001926 |
| `center_dynamic` | **1.731871** | **5.651217** | **-0.002686** | **-0.000760** |
| `s3` | 1.732631 | 5.655514 | -0.001926 | +0.000000 |

dynamic centering 在 seed2026 上同时优于 S1 和 S3。按 S3 相对 S1 的较小收益作分母，
恢复比例为 139.46%；该比例大于 100% 表示 dynamic centering 本 seed 超过了 S3，不代表可
跨 seed 直接外推。

最终权重：
`weights/final/minimind-small-dynamic-centering-randsplit/seed2026/center_dynamic_768.pth`。
