# Dynamic centering 结果

本目录只保存适合 Git 跟踪的实现检查、完整 eval 指标和对比 CSV/JSON。训练日志、最终权重和
resume checkpoint 分别保存在根目录的 `logs/`、`weights/final/`、`weights/resume/`，不加入 Git。

正式结果使用稳定的语义路径，不在目录名中编码运行时间：

```text
minimind-small-fixedinit-randsplit/
├── README.md
├── eval_summary.csv
├── eval_summary.json
├── mechanism_recovery.csv
├── per_seed_comparison.csv
├── seed42/
├── seed123/
└── seed2026/
```

实验根目录保存三 seed 汇总；每个 `seed<seed>/` 子目录保存该 seed 的 `README.md`、
`comparison.csv/json`、`eval_pretrain_loss.csv/json`、`implementation_checks.json` 和
`run_metadata.json`。具体运行时间只记录在 `run_metadata.json` 和被忽略的日志目录中。
