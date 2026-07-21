# Dynamic centering 结果

本目录只保存适合 Git 跟踪的实现检查、完整 eval 指标和对比 CSV/JSON。训练日志、最终权重和
resume checkpoint 分别保存在根目录的 `logs/`、`weights/final/`、`weights/resume/`，不加入 Git。

正式结果使用稳定的语义路径，不在目录名中编码运行时间。目前包含两个训练规模：

```text
results/
├── minimind-small-fixedinit-randsplit/
│   ├── README.md
│   ├── eval_summary.csv
│   ├── eval_summary.json
│   ├── mechanism_recovery.csv
│   ├── per_seed_comparison.csv
│   ├── seed42/
│   ├── seed123/
│   └── seed2026/
└── fineedu-qwen3-0.6b-randsplit/
    ├── README.md
    ├── eval_pretrain_loss.csv
    ├── eval_summary.csv
    └── per_seed_comparison.csv
```

MiniMind-small 实验根目录保存三 seed 汇总；每个 `seed<seed>/` 子目录保存该 seed 的完整
紧凑产物。Qwen3-0.6B 规模放大复验直接保存三 seed aggregate 和逐 seed 对比，避免复制
已有 S1/S3/S12 基线归档。具体运行日志和 checkpoint 仍只保存在被忽略的 `logs/`、
`weights/final/` 和 `weights/resume/` 中。
