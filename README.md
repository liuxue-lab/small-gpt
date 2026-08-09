# Small GPT from Scratch

使用 PyTorch 从零实现并预训练一个约 34M 参数的 Decoder-only GPT，完整走通数据处理、Tokenizer、模型实现、预训练、评估与推理流程。

本项目的目标不是训练通用聊天助手，而是通过一个可运行、可验证、可复现的小型 GPT 项目，掌握现代语言模型的核心工程链路。

## 项目状态

| 项目 | 当前状态 |
| --- | --- |
| 当前阶段 | Day 4 已完成，下一阶段为训练数据编码与 Dataset/DataLoader |
| 自动测试 | 66 项测试通过 |
| 数据集 | `HuggingFaceFW/fineweb-edu` / `sample-10BT` |
| 数据访问方式 | 固定 revision 的流式读取 |
| 数据管线 | 支持清洗、去重、划分、分片、恢复与完整性校验 |
| 真实网络 Smoke | 53,004 provided tokens，2 个 shard groups |
| 2M Pilot | 2,000,083 provided tokens，4 个 shard groups |
| 项目 Tokenizer | 16,384 词表 ByteLevel BPE，已训练并验证 |
| Pilot 模型 tokens | 2,129,776，`<unk>` 为 0 |
| 正式语料 | 计划采集 350M provided tokens，尚未执行 |
| Git 分支 | `main` |

> FineWeb-Edu 的 `provided_token_count` 使用上游 Tokenizer 口径，只用于语料采集预算。当前项目 BPE 在全 Pilot 上产生 2,129,776 个模型 token，比 2,000,083 个 provided tokens 高约 6.48%。后续训练预算以项目 Tokenizer 的实际 token 数为准。

## 项目目标

- 清洗并划分英文预训练语料；
- 从训练语料训练 BPE Tokenizer；
- 从零实现 Causal Self-Attention；
- 从零实现 Decoder-only Transformer；
- 实现单卡混合精度训练；
- 支持 checkpoint 保存与恢复；
- 训练 3 亿～5 亿 tokens；
- 实现文本续写与采样；
- 完成至少一个消融实验。

## 模型基线

| 配置项 | 数值 |
| --- | ---: |
| Transformer layers | 8 |
| Attention heads | 8 |
| Hidden size | 512 |
| Head dimension | 64 |
| FFN hidden size | 2,048 |
| Context length | 512 |
| Vocabulary size | 16,384 |
| Approximate parameters | 33.82M |
| Minimum training tokens | 300M |
| Target range | 300M～500M |

项目同时保留一套约 2.51M 参数的 Debug 配置，用于本地快速验证模型、训练循环和 checkpoint。

## 数据集与处理策略

### 数据集身份

| 项目 | 值 |
| --- | --- |
| Dataset | `HuggingFaceFW/fineweb-edu` |
| Configuration | `sample-10BT` |
| Split | `train` |
| Revision | `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9` |
| Access | Streaming |

固定数据集 revision 可以降低上游数据更新对实验复现的影响。正式语料采集采用 streaming，不完整下载原始数据集。

### 清洗和划分配置

| 参数 | 数值 |
| --- | ---: |
| Minimum characters | 200 |
| Minimum language score | 0.65 |
| Minimum quality score | 3 |
| Split seed | 42 |
| Train ratio | 0.98 |
| Validation ratio | 0.01 |
| Test ratio | 0.01 |

文档经过 Unicode 和空白规范化、英文与质量过滤、SHA-256 精确去重后，再通过确定性规则划分到 train、validation 和 test。

### 采集规模

| Profile | Target provided tokens | Shard target | 状态 |
| --- | ---: | ---: | --- |
| Smoke | 50,000 | 25,000 | 已完成 |
| Pilot | 2,000,000 | 500,000 | 已完成 |
| Full | 350,000,000 | 5,000,000 | 尚未执行 |

Full 数据预计生成约 70 个 shard groups。项目为清洗后语料预留 5 GB 空间，不保存原始 full-corpus shard。

## Day 3 Pilot 结果

| 指标 | 结果 |
| --- | ---: |
| Status | `complete` |
| Input records | 1,852 |
| Kept records | 1,852 |
| Kept provided tokens | 2,000,083 |
| Shard groups | 4 |
| Train records | 1,824 |
| Validation records | 16 |
| Test records | 12 |
| Unique text hashes | 1,852 |
| Output files | 18 |
| Total disk size | 9.49 MiB |

Pilot 已通过以下验证：

- 所有 JSONL 文件可解析；
- 文件大小、SHA-256 和 metadata 一致；
- 三个 split 互斥且统计守恒；
- 跨 shard、跨 split 无精确重复文本；
- 不存在未完成的临时 shard；
- 同配置重跑的统计和全部文件哈希保持一致；
- 生成数据被 Git 正确忽略。

## Day 4 Tokenizer 结果

项目使用 Pilot 的 train split 训练了 16,384 词表的 ByteLevel BPE Tokenizer。validation 和 test 未参与 BPE 训练。

| 配置项 | 值 |
| --- | --- |
| 实现库 | `tokenizers==0.23.1` |
| 模型 | ByteLevel BPE |
| Normalizer | NFC |
| Vocabulary size | 16,384 |
| Minimum frequency | 2 |
| Maximum token length | 64 |
| 文档边界 | 每篇文档末尾追加一个 `<eos>` |
| 特殊 token ID | `<bos>=0`、`<eos>=1`、`<pad>=2`、`<unk>=3` |

真实编码结果：

| Split | 文档 | Provided tokens | 模型 tokens | Model/Provided | `<unk>` |
| --- | ---: | ---: | ---: | ---: | ---: |
| Train | 1,824 | 1,967,041 | 2,093,241 | 1.064157 | 0 |
| Validation | 16 | 12,600 | 13,933 | 1.105794 | 0 |
| Test | 12 | 20,442 | 22,602 | 1.105665 | 0 |
| **合计** | **1,852** | **2,000,083** | **2,129,776** | **1.064844** | **0** |

共有 1,122 篇文档超过 Baseline 的 512-token 上下文，占 60.58%。后续构建训练数据时需要正确切块，不能直接截断整篇长文档。

按照 Pilot train 的实际 token 比例估算，350M provided-token Full 语料的 train split 约产生 365,005,947 个项目模型 token。该数值仅用于规划，正式训练仍以实际编码统计为准。

## 数据管线能力

`scripts/build_fineweb_edu_corpus.py` 已实现：

- Hugging Face 流式读取和固定 revision；
- provided-token budget 停止条件；
- 文档边界感知的分片；
- train、validation、test 三路 JSONL 输出；
- 全局 SHA-256 精确去重；
- 原子写入 manifest、state 和 shard；
- 已完成 shard 恢复；
- 文件大小、SHA-256、JSON、split 和文本 hash 校验；
- 中断恢复、幂等重跑和文件损坏检测。

运行 2M Pilot：

```powershell
python .\scripts\build_fineweb_edu_corpus.py --config .\configs\data_fineweb_edu.yaml --profile pilot
```

运行后，默认输出到 `data/processed/fineweb_edu_corpus/`。数据目录不会提交到 Git。

## Tokenizer 使用

训练 Tokenizer：

```powershell
python .\scripts\train_tokenizer.py --config .\configs\tokenizer.yaml
```

统计三个 split 的真实模型 token：

```powershell
python .\scripts\evaluate_tokenizer.py --config .\configs\tokenizer.yaml
```

权威运行时产物为 `tokenizer/artifacts/tokenizer.json`。`vocab.json`、`merges.txt` 和 `tokenizer_config.json` 用于审计与复现。正式产物默认拒绝静默覆盖。

## 开发与训练分工

| 本地 Windows 电脑 | 租用的 Linux GPU |
| --- | --- |
| 编写和阅读代码 | 正式预训练 |
| Pytest 单元测试 | 完整数据处理 |
| 运行 2.51M Debug 模型 | 运行 33.82M Baseline 模型 |
| 训练和验证 Pilot Tokenizer | 编码正式训练语料 |
| 单批次过拟合 | 训练 3 亿～5 亿 tokens |
| 检查小规模样本 | 保存正式 checkpoint |
| 分析训练日志 | 长时间运行训练任务 |

本地 `.venv` 不复制到租用服务器。服务器将根据其 GPU、驱动和 CUDA 环境重新创建 Python 虚拟环境并安装 PyTorch。

## 当前配置

- `configs/debug.yaml`：本地快速调试配置；
- `configs/baseline.yaml`：租用 GPU 正式训练配置；
- `configs/data_fineweb_edu.yaml`：FineWeb-Edu 采集、清洗、划分和分片配置；
- `configs/tokenizer.yaml`：ByteLevel BPE 训练、产物和评估配置。

## 本地验证

激活项目虚拟环境后，在仓库根目录运行：

```powershell
python .\scripts\check_env.py
python .\scripts\check_config.py
python -m pytest -q
```

当前完整测试结果：

```text
66 passed
```

测试范围包括：

- PyTorch、CUDA 和 Autograd 环境检查；
- Debug 与 Baseline 模型配置检查；
- Tokenizer、模型词表大小和特殊 token ID 一致性检查；
- 小样本清洗、去重和确定性划分；
- FineWeb-Edu 数据采集配置检查；
- 流式分片、manifest 和 state 检查；
- 中断恢复和幂等重跑；
- 临时文件隔离与损坏检测；
- Tokenizer 配置、输入边界、NFC、Unicode、保存和重新加载；
- 三个 split 的真实 token 统计、EOS 守恒和原子报告写入。

## 项目结构

```text
small-gpt/
├── configs/
│   ├── baseline.yaml
│   ├── data_fineweb_edu.yaml
│   ├── debug.yaml
│   └── tokenizer.yaml
├── data/                          # 本地生成数据，不提交到 Git
├── eval/                          # 验证损失与文本生成
├── model/                         # Attention、Transformer Block 和 GPT
├── reports/
│   ├── daily-log.md
│   ├── day-02-data-audit.md
│   ├── day-02-inspection.json
│   ├── day-02-cleaning-stats.json
│   ├── day-03-data-pipeline-design.md
│   ├── day-03-execution-report.md
│   ├── day-04-tokenizer-report.md
│   └── day-04-tokenizer-stats.json
├── scripts/
│   ├── build_fineweb_edu_corpus.py
│   ├── check_config.py
│   ├── check_env.py
│   ├── evaluate_tokenizer.py
│   ├── inspect_dataset.py
│   ├── prepare_data.py
│   └── train_tokenizer.py
├── tests/
│   ├── test_config.py
│   ├── test_data_config.py
│   ├── test_data_pipeline.py
│   ├── test_environment.py
│   ├── test_streaming_data_pipeline.py
│   └── test_tokenizer.py
├── tokenizer/
│   ├── artifacts/
│   │   ├── merges.txt
│   │   ├── tokenizer.json
│   │   ├── tokenizer_config.json
│   │   └── vocab.json
│   ├── __init__.py
│   └── bpe.py
├── train/                         # 数据加载、训练循环和 checkpoint
├── .gitignore
├── README.md
└── requirements.txt
```

## 开发里程碑

- [x] Day 1：环境、项目结构、Debug/Baseline 配置和基础测试
- [x] Day 2：FineWeb-Edu 数据审计、小样本清洗和确定性划分
- [x] Day 3：可恢复流式数据管线、真实网络 Smoke 和 2M Pilot
- [x] Day 4：训练、保存并验证 16K ByteLevel BPE Tokenizer
- [ ] 构建 tokenized binary、Dataset 和 DataLoader
- [ ] 从零实现 Causal Self-Attention 和 Decoder-only GPT
- [ ] 完成单批次过拟合测试
- [ ] 实现混合精度训练、checkpoint 保存与恢复
- [ ] 采集并验证正式训练语料
- [ ] 在租用 GPU 上完成正式预训练
- [ ] 实现文本生成与模型评估
- [ ] 完成至少一个消融实验

## 文档

- [开发日志](reports/daily-log.md)
- [Day 2 数据审计](reports/day-02-data-audit.md)
- [Day 2 数据检查统计](reports/day-02-inspection.json)
- [Day 2 数据清洗统计](reports/day-02-cleaning-stats.json)
- [Day 3 数据管线设计](reports/day-03-data-pipeline-design.md)
- [Day 3 执行报告](reports/day-03-execution-report.md)
- [Day 4 Tokenizer 执行报告](reports/day-04-tokenizer-report.md)
- [Day 4 Tokenizer 统计](reports/day-04-tokenizer-stats.json)

## 当前阶段

Day 4 已完成。项目已经具备固定的 16,384 词表 ByteLevel BPE Tokenizer、可复用编码接口、四个权威产物和全 Pilot 真实 token 统计。

下一阶段将设计 tokenized binary 与索引格式，实现 Dataset/DataLoader，并按照 512-token 上下文正确处理长文档、文档边界、跨文档拼接和末尾残片。350M Full 语料仍未启动，AutoDL 继续保持关机。
