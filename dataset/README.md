# 数据目录

`dataset/` 同时包含训练数据加载代码和本机大数据资产。大数据由 `.gitignore` 排除，
这里只提交加载/预处理代码与本说明。

`download_jsonl.py` 默认将 MiniMind 文件写入 `dataset/minimind/`；FineEdu 打包脚本要求
显式传入 `--input-dir` 和 `--output-dir`，避免误写到含义不明的默认目录。

## 目录结构

```text
dataset/
├── minimind/
│   └── pretrain_t2t.jsonl
└── fineweb_edu/
    ├── raw/
    │   └── modelscope_100bt_first25/
    │       └── data/*.parquet
    └── packed/
        ├── gpt2_6b_seq340/
        └── qwen3_20b_seq340/
```

命名规则：`raw/` 保存可重新分词的来源语料；`packed/` 保存训练直接读取的定长 token
序列。packed 目录名依次表达 tokenizer、token 规模和序列长度。

| 数据 | 路径 | 规模 |
| --- | --- | ---: |
| MiniMind 文本 | `dataset/minimind/pretrain_t2t.jsonl` | 8,468,827 条 |
| FineEdu GPT-2 packed | `dataset/fineweb_edu/packed/gpt2_6b_seq340` | 5,999,999,720 tokens |
| FineEdu Qwen3 packed | `dataset/fineweb_edu/packed/qwen3_20b_seq340` | 19,999,999,860 tokens |
| FineEdu Qwen3 原始语料 | `dataset/fineweb_edu/raw/modelscope_100bt_first25/data` | 25 个 parquet |

GPT-2 历史原始 parquet 当前未保存在仓库磁盘中；若重新获取，约定放在
`dataset/fineweb_edu/raw/gpt2_sample_10bt/`，不要重新引入 `sample/10BT` 等多层别名。

训练脚本从 `trainer/` 启动时使用 `../dataset/...`，从项目根目录运行的 eval 和 manifest
记录使用 `dataset/...`。两者指向同一规范目录，不再维护额外兼容软链接。

GPT-2 packed 数据的原始输入与 cache 历史位置记录在其 `preprocess_meta.json` 中；这些旧
绝对路径用于溯源，不是当前运行路径。历史 eval CSV/JSON 中的 `data_path` 也保留实验当时
的值，不应按当前目录机械改写。
