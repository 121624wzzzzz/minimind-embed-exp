# Dynamic centering 结果

本目录只保存适合 Git 跟踪的实现检查、完整 eval 指标和对比 CSV/JSON。训练日志、最终权重和
resume checkpoint 分别保存在根目录的 `logs/`、`weights/final/`、`weights/resume/`，不加入 Git。

每个正式运行目录应至少包含：

- `implementation_checks.json`：零参数、共享权重、动态均值和 autograd 检查；
- `seed<seed>/eval_pretrain_loss.csv` 与 `.json`：新变体完整 held-out 评测；
- `comparison_seed<seed>.csv` 与 `.json`：与已有同 seed `s1`、`s3` 的统一对比。

已完成运行：

- `seed42_full_20260710_2001/`；
- `seed123_full_20260710_211623/`；
- `seed2026_full_20260710_222655/`；
- `allseeds_full_20260710_2001_to_20260711_0014/`：三 seed 总汇总与最终中文结论。
