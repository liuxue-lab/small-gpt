# Small GPT from Scratch

使用 PyTorch 从零实现并预训练一个约 34M 参数的 Decoder-only GPT，完整走通数据处理、Tokenizer、模型实现、预训练、评估与推理流程。

本项目的目标不是训练通用聊天助手，而是通过一个可运行、可验证、可复现的小型 GPT 项目，掌握现代语言模型的核心工程链路。

## 项目状态

| 项目 | 当前状态 |
| --- | --- |
| 当前阶段 | Day 6 已完成，下一阶段为训练循环、优化器与 checkpoint |
| 自动测试 | 216 项测试通过 |
| 数据集 | `HuggingFaceFW/fineweb-edu` / `sample-10BT` |
| 数据访问方式 | 固定 revision 的流式读取 |
| 数据管线 | 支持清洗、去重、划分、分片、恢复与完整性校验 |
| 真实网络 Smoke | 53,004 provided tokens，2 个 shard groups |
| 2M Pilot | 2,000,083 provided tokens，4 个 shard groups |
| 项目 Tokenizer | 16,384 词表 ByteLevel BPE，已训练并验证 |
| Pilot 模型 tokens | 2,129,776，`<unk>` 为 0 |
| Tokenized Pilot | 5 个 storage shards，`<u2` 二进制与文档索引已验证 |
| Dataset/DataLoader | 跨 shard memory-map、因果 `T+1` 窗口和 Windows workers 已验证 |
| GPT 模型 | 手写 Decoder-only GPT；Debug 2,508,032 参数，Baseline 33,833,984 参数 |
| 模型诊断 | Synthetic/Pilot CUDA 前后向与固定批次过拟合均已通过 |
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
| Exact parameters | 33,833,984 |
| Minimum training tokens | 300M |
| Target range | 300M～500M |

项目同时保留一套 2,508,032 参数的 Debug 配置，用于本地快速验证模型、训练循环和 checkpoint。两套配置共用同一份模型实现。

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

## Day 5 Tokenized Data 与 Dataset 结果

Day 5 使用 Day 4 冻结的 `tokenizer.json` 将完整 Pilot 编码为训练可直接读取的二进制数据。存储层不按上下文长度截断文档，每篇文档只追加一个 `<eos>`，并通过独立索引保存文档偏移、长度和原始文本 SHA-256。

| Split | 文档 | 模型 tokens | Storage shards |
| --- | ---: | ---: | ---: |
| Train | 1,824 | 2,093,241 | 3 |
| Validation | 16 | 13,933 | 1 |
| Test | 12 | 22,602 | 1 |
| **合计** | **1,852** | **2,129,776** | **5** |

二进制契约：

| 项目 | 冻结值 |
| --- | --- |
| Token payload | little-endian `uint16`（`<u2`） |
| Token header | 64 bytes，magic `SGPTTOK1` |
| Index header | 128 bytes，magic `SGPTIDX1` |
| Index record | 48 bytes：offset、length、text SHA-256 |
| Token payload bytes | 4,259,552 |
| 发布策略 | staging 目录完成后原子发布；同身份完成态重跑为 no-op |
| Split 策略 | split 内可跨 storage shard；绝不跨 split |

因果窗口读取满足 `x = tokens[start:start+T]`、`y = tokens[start+1:start+T+1]`。真实 Pilot 验证结果：

| Context `T` | Train `all_starts` | Validation sequential | Test sequential |
| ---: | ---: | ---: | ---: |
| 128 | 2,093,113 | 108 | 176 |
| 512 | 2,092,729 | 27 | 44 |

Train 的两个真实分片边界 `989,147` 和 `1,987,969` 均通过跨 shard 的 512-token `T+1` 读取验证；Windows 下 `num_workers=2` 的三个 split 均成功构造 `(4, 512)`、`torch.int64` 批次。

## Day 6 Decoder-only GPT 结果

Day 6 在 Day 5 的因果 `x/y` 数据契约之上，从零组装了可训练的 Decoder-only GPT。模型直接计算同位置 logits 与 targets 的 next-token cross entropy，不在模型内部再次 shift。

| 项目 | 冻结实现 |
| --- | --- |
| Block | Pre-LayerNorm，两条 residual path |
| Position | Learned absolute position embedding |
| Attention | 手写 causal multi-head self-attention，融合 QKV |
| Attention 数值 | `1/sqrt(head_dim)` 缩放，FP32 softmax 后转回输入 dtype |
| Causal mask | boolean 下三角，对角线可见，非持久 buffer |
| MLP | `C -> 4C -> C`，GELU tanh approximation |
| Linear bias | QKV、attention output、MLP 和 LM head 均为 false |
| Output | final LayerNorm、tied token embedding / LM head |
| 初始化 | 基础 `Normal(0, 0.02)`；残差投影 `0.02/sqrt(2L)` |

| 配置 | Layers | Heads | Hidden | Context | 精确参数量 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Debug | 2 | 2 | 128 | 128 | 2,508,032 |
| Baseline | 8 | 8 | 512 | 512 | 33,833,984 |

模型专项测试共 113 项：

| 测试文件 | 结果 |
| --- | ---: |
| `tests/test_model_config.py` | 35 passed |
| `tests/test_layers.py` | 23 passed |
| `tests/test_attention.py` | 23 passed |
| `tests/test_model.py` | 32 passed |

CUDA 运行验证：

| 场景 | Batch / Sequence | 结果 |
| --- | --- | --- |
| Synthetic forward/backward | `4 x 128` | loss `9.726461`；20/20 梯度非零且有限 |
| 固定 synthetic batch 过拟合 | `4 x 32`，200 steps | loss `9.704769 -> 0.003591` |
| 真实 tokenized Pilot forward/backward | `4 x 128` | loss `9.744143`；`x/y` shift 正确；20/20 梯度非零且有限 |

自动测试还覆盖 Attention 与整模因果隔离、weight tying 的对象和 storage 共享、optimizer 参数去重、初始化复现、连续两次 backward，以及 strict state dict round-trip。完整项目回归为 `216 passed in 9.94s`。

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

## Tokenized Data 使用

编码 2M Pilot：

```powershell
python .\scripts\tokenize_corpus.py --config .\configs\tokenized_data.yaml --profile pilot
```

验证 manifest 并检查 512-token 因果样本与 DataLoader：

```powershell
python .\scripts\inspect_tokenized_data.py `
  --manifest .\data\tokenized\fineweb_edu_pilot\manifest.json `
  --context-length 512 `
  --samples 8 `
  --batch-size 4 `
  --num-workers 2
```

默认输出位于 `data/tokenized/fineweb_edu_pilot/`，由 `.gitignore` 排除。`reports/day-05-tokenized-data-stats.json` 是完成态 manifest 的可追踪统计快照，不包含二进制 payload。

## 模型诊断

检查模型公共 API：

```powershell
python -c "from model import GPT, GPTConfig, GPTOutput, TransformerBlock; print('Model import passed')"
```

使用 Debug 配置执行 synthetic CUDA 前后向：

```powershell
python .\scripts\inspect_model.py `
  --config .\configs\debug.yaml `
  --device auto `
  --batch-source synthetic `
  --batch-size 4 `
  --sequence-length 128 `
  --seed 42 `
  --backward
```

读取 Day 5 tokenized Pilot 并执行真实 batch 前后向：

```powershell
python .\scripts\inspect_model.py `
  --config .\configs\debug.yaml `
  --device cuda `
  --batch-source pilot `
  --manifest .\data\tokenized\fineweb_edu_pilot\manifest.json `
  --batch-size 4 `
  --sequence-length 128 `
  --num-workers 0 `
  --seed 42 `
  --backward
```

`scripts/inspect_model.py` 还提供 `--overfit-steps`，用于让 Debug 模型诊断性地过拟合同一个固定 batch。该模式不会保存权重，也不替代 Day 7 的正式训练系统。

## 开发与训练分工

| 本地 Windows 电脑 | 租用的 Linux GPU |
| --- | --- |
| 编写和阅读代码 | 正式预训练 |
| Pytest 单元测试 | 完整数据处理 |
| 运行 2,508,032 参数 Debug 模型 | 运行 33,833,984 参数 Baseline 模型 |
| 训练和验证 Pilot Tokenizer | 编码正式训练语料 |
| 单批次过拟合 | 训练 3 亿～5 亿 tokens |
| 检查小规模样本 | 保存正式 checkpoint |
| 分析训练日志 | 长时间运行训练任务 |

本地 `.venv` 不复制到租用服务器。服务器将根据其 GPU、驱动和 CUDA 环境重新创建 Python 虚拟环境并安装 PyTorch。

## 当前配置

- `configs/debug.yaml`：2,508,032 参数本地快速调试配置；
- `configs/baseline.yaml`：33,833,984 参数租用 GPU 正式训练配置；
- `configs/data_fineweb_edu.yaml`：FineWeb-Edu 采集、清洗、划分和分片配置；
- `configs/tokenizer.yaml`：ByteLevel BPE 训练、产物和评估配置；
- `configs/tokenized_data.yaml`：tokenized binary、索引、发布恢复和 Dataset 契约。

## 本地验证

激活项目虚拟环境后，在仓库根目录运行：

```powershell
python .\scripts\check_env.py
python .\scripts\check_config.py
python -m pytest -q
```

当前完整测试结果：

```text
216 passed in 9.94s
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
- 三个 split 的真实 token 统计、EOS 守恒和原子报告写入；
- `.bin/.idx` header、字节序、大小、哈希与索引连续性；
- 文档原子分片、恢复、完成态幂等和损坏检测；
- 跨 storage shard 的逻辑 token 流与 split 隔离；
- 因果 `T+1` Dataset、确定性 train sampler 和 Windows DataLoader workers；
- GPTConfig 严格字段校验与 Debug/Baseline 精确参数量；
- Embedding、GELU MLP、Pre-LN Block 和手写 causal self-attention；
- Attention 概率、因果 mask、未来信息隔离与 FP32 softmax；
- GPT logits/loss、无二次 shift、weight tying 和初始化；
- 全模型梯度、optimizer 去重、固定 seed 与 state dict round-trip。

## 项目结构

```text
small-gpt/
├── configs/
│   ├── baseline.yaml
│   ├── data_fineweb_edu.yaml
│   ├── debug.yaml
│   ├── tokenizer.yaml
│   └── tokenized_data.yaml
├── data/                          # 本地生成数据，不提交到 Git
├── data_pipeline/
│   ├── __init__.py
│   ├── binary_format.py
│   ├── dataset.py
│   └── tokenization.py
├── eval/                          # 验证损失与文本生成
├── model/
│   ├── __init__.py
│   ├── attention.py               # 手写 causal multi-head self-attention
│   ├── block.py                   # Pre-LN Transformer Block
│   ├── config.py                  # 严格模型配置与参数量计算
│   ├── gpt.py                     # Decoder-only GPT、loss 与初始化
│   └── layers.py                  # Embedding 与 MLP
├── reports/
│   ├── daily-log.md
│   ├── day-02-data-audit.md
│   ├── day-02-inspection.json
│   ├── day-02-cleaning-stats.json
│   ├── day-03-data-pipeline-design.md
│   ├── day-03-execution-report.md
│   ├── day-04-tokenizer-report.md
│   ├── day-04-tokenizer-stats.json
│   ├── day-05-execution-report.md
│   ├── day-05-tokenized-data-design.md
│   ├── day-05-tokenized-data-stats.json
│   ├── day-06-execution-report.md
│   └── day-06-model-design.md
├── scripts/
│   ├── build_fineweb_edu_corpus.py
│   ├── check_config.py
│   ├── check_env.py
│   ├── evaluate_tokenizer.py
│   ├── inspect_dataset.py
│   ├── inspect_model.py
│   ├── inspect_tokenized_data.py
│   ├── prepare_data.py
│   ├── tokenize_corpus.py
│   └── train_tokenizer.py
├── tests/
│   ├── test_attention.py
│   ├── test_binary_format.py
│   ├── test_config.py
│   ├── test_data_config.py
│   ├── test_data_pipeline.py
│   ├── test_dataset.py
│   ├── test_environment.py
│   ├── test_layers.py
│   ├── test_model.py
│   ├── test_model_config.py
│   ├── test_streaming_data_pipeline.py
│   ├── test_tokenization.py
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
- [x] Day 5：构建 tokenized binary、文档索引、Dataset 和 DataLoader
- [x] Day 6：手写 Causal Self-Attention、Decoder-only GPT 并完成模型验收
- [ ] Day 7：实现训练循环、优化器、调度器、评估与 checkpoint 恢复
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
- [Day 5 Tokenized Data 设计](reports/day-05-tokenized-data-design.md)
- [Day 5 执行报告](reports/day-05-execution-report.md)
- [Day 5 Tokenized Data 统计](reports/day-05-tokenized-data-stats.json)
- [Day 6 模型设计](reports/day-06-model-design.md)
- [Day 6 执行报告](reports/day-06-execution-report.md)

## 当前阶段

Day 6 已完成。项目现在具备一套从矩阵运算开始手写的 Decoder-only GPT：Pre-LN、learned absolute position、causal multi-head self-attention、GELU MLP、final LayerNorm、tied LM head 和 GPT 风格初始化均已实现。Debug 与 Baseline 精确参数量分别为 2,508,032 和 33,833,984。

113 个模型专项测试和 216 项完整回归全部通过。Debug 模型已在本地 CUDA 上完成 synthetic 与真实 tokenized Pilot 前后向，固定 batch loss 在 200 steps 内由 `9.704769` 降至 `0.003591`，证明数据对齐、因果 loss、梯度与参数更新闭环成立。

下一阶段将实现正式训练循环、AdamW 参数分组、学习率调度、gradient accumulation、混合精度、validation 和 checkpoint 原子保存/恢复，并继续先用 Debug 配置验收。350M Full 语料与正式 GPU 预训练仍未启动，AutoDL 继续保持关机。
