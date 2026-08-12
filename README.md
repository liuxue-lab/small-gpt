# Small GPT from Scratch

使用 PyTorch 从零实现并预训练一个约 34M 参数的 Decoder-only GPT，完整走通数据处理、Tokenizer、模型实现、预训练、评估与推理流程。

本项目的目标不是训练通用聊天助手，而是通过一个可运行、可验证、可复现的小型 GPT 项目，掌握现代语言模型的核心工程链路。

## 项目状态

| 项目 | 当前状态 |
| --- | --- |
| 当前阶段 | Day 8 RTX 5090 资源定标与本地回归已完成；等待 commit、push 与远端 hash 核对 |
| 自动测试 | Day 8 定向测试 `83 passed in 2.41s`；完整回归 `531 passed in 13.95s` |
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
| 训练系统 | AdamW、warmup/cosine、FP32/BF16、validation、JSONL、原子 checkpoint 与 strict resume 已完成 |
| Debug 训练 | Pilot 上 200 updates / 102,400 tokens；10 次验证；2 个 checkpoint；全程 finite |
| 恢复验证 | 下一 batch/LR/RNG 精确恢复；模型与 optimizer 在冻结 CUDA FP32 tolerance 内一致 |
| Baseline 资源 | RTX 5090 已冻结 `micro=16`、`accum=8`、`workers=4`、`pin_memory=false` |
| Baseline 短跑 | BF16 25 updates / 1,638,400 tokens；validation、checkpoint 与 20→25 resume 均通过 |
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

## Day 7 单卡训练系统结果

Day 7 在已冻结的 tokenized Pilot、因果 `x/y` 和 Decoder-only GPT 之上，实现了可验证、可记录、可中断恢复的正式单卡训练系统。

| 组件 | 冻结实现 |
| --- | --- |
| Training config | 严格字段、预算/warmup 互斥、完整 update token 预算 |
| Optimizer | AdamW；按参数维度分 decay/no-decay；tied Parameter 去重 |
| Scheduler | 20-update linear warmup + cosine decay，按 optimizer update 推进 |
| Update | accumulation、loss/N、finite checks、global grad-norm clip |
| Precision | FP32 与 CUDA BF16；BF16 autocast，不使用 GradScaler |
| Validation | sequential 非重叠窗口、token-weighted loss、固定 batch 数 |
| Logging | JSONL metrics、resolved config、run metadata、run-id 防覆盖 |
| Checkpoint | schema v1、原子保存、strict identity validation |
| Resume | model/optimizer/scheduler/trainer/RNG/data cursor 完整恢复 |
| CLI | dry-run、Pilot/synthetic、stop gate、显式 checkpoint resume |

Debug 模型在真实 tokenized Pilot 上完成 200 次 optimizer updates：

| 指标 | 结果 |
| --- | ---: |
| Model parameters | 2,508,032 |
| Device / precision | `cuda:0` / FP32 |
| Micro batch / context / accumulation | 4 / 128 / 1 |
| Tokens/update | 512 |
| Total updates / tokens | 200 / 102,400 |
| Initial / final train loss | 9.722784 / 7.447512 |
| First / final validation loss | 9.329641 / 7.517156 |
| First / final perplexity | 11,267.09 / 1,839.33 |
| Evaluation events | 10，每次 10,240 tokens |
| Checkpoints | step 100、step 200，各 30,138,029 bytes |
| Non-finite train/eval events | 0 / 0 |

CUDA BF16 5-step smoke 同样通过：autocast 已启用、GradScaler 未启用，5 个 train events 的 loss 和 grad norm 全部有限，并成功保存 step 5 checkpoint。

Checkpoint/resume 对照证明下一训练 batch、下一学习率、scheduler state 与 Python/NumPy/Torch RNG 可以恢复；模型和 optimizer continuation 在冻结的 CUDA FP32 tolerance 内一致。正式 CLI 的 step 5 → step 10 resume 在 step 10 得到与连续运行完全一致的 loss `9.656713485717773`。

Day 7 最终专项门为 `147 passed in 6.17s`，Day 1～Day 7 完整项目回归为：

```text
508 passed in 13.49s
```

以上仅证明 Debug/Pilot 训练工程闭环成立，不代表 33,833,984 参数 Baseline 已完成预训练。AutoDL、350M Full 语料和 300M-token 正式训练仍未启动。

## Day 8 RTX 5090 Baseline 资源定标结果

Day 8 在 AutoDL A57 的单卡 NVIDIA GeForce RTX 5090 上，使用真实 tokenized FineWeb-Edu Pilot 和 Day 7 正式训练路径完成隔离式资源探测。探测绑定 clean commit `2e3166c395d7057cb8509fda6f5768bd9b203537`，证据包已下载并通过远端/本地 SHA-256 对照。

冻结资源配置：

| 字段 | 值 |
| --- | ---: |
| `micro_batch_size` | 16 |
| `gradient_accumulation_steps` | 8 |
| `num_workers` | 4 |
| `pin_memory` | false |
| Context length | 512 |
| Tokens/micro-step | 8,192 |
| Tokens/update | 65,536 |
| Total / warmup updates | 4,578 / 92 |
| Planned tokens / overshoot | 300,023,808 / 23,808 |

Micro-batch 1、2、4、8、12、16 全部成功。b16 达到 241,017 tokens/s，peak reserved 5.053 GiB / 16.11%；没有为了寻找 OOM 边界继续扩大 batch，因此结论是“已验证到 16”，不是“最大只能为 16”。在 accumulation=8 时 peak reserved 为 5.553 GiB / 17.71%，仍保留充足运行时余量。

DataLoader 的 8 个候选全部成功，`workers=4, pin_memory=false` 以 246,578 tokens/s 位列本轮第一。`pin_memory=true` 没有一致收益。

正式训练入口随后完成：

| 验收项 | 结果 |
| --- | --- |
| Baseline BF16 dry-run | PASS；33,833,984 参数，输入/目标 `(16, 512)` |
| 20-step 短跑 | PASS；1,310,720 tokens；step 10/20 validation 有限 |
| 独立 resume 对照 | PASS；下一 batch exact，scheduler/RNG exact |
| 20 → 25 正式 resume | PASS；最终 1,638,400 tokens，JSONL steps 1～25 连续 |
| Checkpoint | step 20/25 各 406,108,827 bytes |
| Full / 300M 正式训练 | NOT STARTED |

从 JSONL 重算的 steady update throughput 约为 236,443 tokens/s。300M tokens 的计算时间规划范围约 21～42 分钟，按当时 ¥2.78/小时约为 ¥0.99～¥1.94；该范围不包含 Full 数据准备、云端抖动、热降频或故障重跑。

完整证据、选择理由、磁盘预算和正式训练硬门见 [Day 8 RTX 5090 Baseline 资源定标执行报告](reports/day-08-resource-calibration-report.md)。本轮只完成资源定标和短程工程验收，没有构建 Full，也没有启动 300M-token 正式训练。

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

`scripts/inspect_model.py` 还提供 `--overfit-steps`，用于让 Debug 模型诊断性地过拟合同一个固定 batch。该模式不会保存权重，也不替代正式训练入口。

## 正式训练入口

在不写日志或 checkpoint 的情况下检查完整 Debug/Pilot 组件：

```powershell
python .\scripts\train_gpt.py `
  --config .\configs\debug.yaml `
  --manifest .\data\tokenized\fineweb_edu_pilot\manifest.json `
  --run-id debug-dry-run `
  --device cuda `
  --precision fp32 `
  --batch-source pilot `
  --num-workers 0 `
  --dry-run
```

按照 `debug.yaml` 运行完整 200-update Debug 训练：

```powershell
python .\scripts\train_gpt.py `
  --config .\configs\debug.yaml `
  --manifest .\data\tokenized\fineweb_edu_pilot\manifest.json `
  --run-id debug-pilot-200 `
  --device cuda `
  --precision fp32 `
  --batch-source pilot `
  --num-workers 0
```

从显式 checkpoint 恢复到同一 scheduler horizon 的后续 stop gate：

```powershell
python .\scripts\train_gpt.py `
  --config .\configs\debug.yaml `
  --manifest .\data\tokenized\fineweb_edu_pilot\manifest.json `
  --run-id debug-resume `
  --device cuda `
  --precision fp32 `
  --batch-source pilot `
  --stop-at-step 20 `
  --num-workers 0 `
  --resume .\checkpoints\debug-resume\step-00000010.pt
```

`--stop-at-step` 只控制当前进程的停止位置，不修改 YAML 中的总训练步数或学习率曲线。新运行拒绝覆盖已有 run-id；resume 必须使用相同的 resolved config、模型、Tokenizer 和 dataset identity。

运行指标写入 `runs/<run-id>/metrics.jsonl`，checkpoint 写入 `checkpoints/<run-id>/`。两类运行产物均由 Git 忽略。

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
- `configs/baseline.yaml`：33,833,984 参数租用 GPU 正式训练配置；Day 8 已冻结 RTX 5090 资源字段 `16 / 8 / 4 / false`；
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

Day 8 四字段冻结后的完整测试结果：

```text
531 passed in 13.95s
PowerShell outer elapsed: 15.71s
Exit code: 0
```

`tests/test_training_config.py` 与 `tests/test_resource_probe.py` 定向门为 `83 passed in 2.41s`，PowerShell outer elapsed 为 3.20 秒；`git diff --check` exit code 为 0。

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
- 全模型梯度、optimizer 去重、固定 seed 与 state dict round-trip；
- 严格训练配置、update/token budget 与 Baseline RTX 5090 冻结资源计划；
- AdamW 参数分组、tied alias 去重与 warmup/cosine 边界；
- gradient accumulation、finite checks、梯度裁剪与 FP32/BF16 precision policy；
- 确定性训练数据流、固定 validation 和 token-weighted loss；
- JSONL 日志、run-id 防覆盖、resolved config 与 resume 日志校验；
- checkpoint schema、原子保存、identity mismatch 拒绝与 RNG/data cursor 恢复；
- 正式训练循环的 log/eval/save interval、final checkpoint 与 CLI 行为。
- isolated resource candidate、显存门、OOM 识别、原子探测报告与 loader 推荐。

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
│   ├── day-06-model-design.md
│   ├── day-07-execution-report.md
│   ├── day-07-training-system-design.md
│   ├── day-08-resource-calibration-design.md
│   └── day-08-resource-calibration-report.md
├── scripts/
│   ├── build_fineweb_edu_corpus.py
│   ├── check_checkpoint_resume.py
│   ├── check_config.py
│   ├── check_env.py
│   ├── check_training_config.py
│   ├── check_training_core.py
│   ├── check_training_pipeline.py
│   ├── check_training_update.py
│   ├── evaluate_tokenizer.py
│   ├── inspect_dataset.py
│   ├── inspect_model.py
│   ├── inspect_tokenized_data.py
│   ├── prepare_data.py
│   ├── probe_baseline_resources.py
│   ├── tokenize_corpus.py
│   ├── train_gpt.py
│   └── train_tokenizer.py
├── tests/
│   ├── test_attention.py
│   ├── test_binary_format.py
│   ├── test_checkpoint.py
│   ├── test_config.py
│   ├── test_data_config.py
│   ├── test_data_pipeline.py
│   ├── test_data_stream.py
│   ├── test_dataset.py
│   ├── test_environment.py
│   ├── test_evaluation.py
│   ├── test_layers.py
│   ├── test_model.py
│   ├── test_model_config.py
│   ├── test_optimizer.py
│   ├── test_precision.py
│   ├── test_run_logging.py
│   ├── test_resource_probe.py
│   ├── test_scheduler.py
│   ├── test_streaming_data_pipeline.py
│   ├── test_tokenization.py
│   ├── test_tokenizer.py
│   ├── test_train_gpt.py
│   ├── test_trainer.py
│   ├── test_trainer_state.py
│   ├── test_training_config.py
│   └── test_training_loop.py
├── tokenizer/
│   ├── artifacts/
│   │   ├── merges.txt
│   │   ├── tokenizer.json
│   │   ├── tokenizer_config.json
│   │   └── vocab.json
│   ├── __init__.py
│   └── bpe.py
├── train/
│   ├── __init__.py               # 稳定训练系统公共 API
│   ├── checkpoint.py             # 原子 checkpoint 与 strict resume
│   ├── config.py                 # 严格训练配置与 resolved plan
│   ├── data_stream.py            # 可恢复训练流与固定验证流
│   ├── evaluation.py             # token-weighted validation
│   ├── loop.py                   # log/eval/save interval 主循环
│   ├── optimizer.py              # AdamW 参数分组
│   ├── precision.py              # FP32 / CUDA BF16 policy
│   ├── resource_probe.py         # 隔离候选、显存/吞吐证据与推荐
│   ├── run_logging.py            # run 目录与 JSONL 指标
│   ├── scheduler.py              # warmup + cosine
│   ├── state.py                  # trainer counters
│   └── trainer.py                # accumulated optimizer update
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
- [x] Day 7：实现训练循环、优化器、调度器、评估、日志与 checkpoint 恢复
- [ ] Day 8：资源定标、配置冻结与 531 项回归已通过；commit、push 待完成
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
- [Day 7 训练系统设计](reports/day-07-training-system-design.md)
- [Day 7 执行报告](reports/day-07-execution-report.md)
- [Day 8 资源定标探针设计](reports/day-08-resource-calibration-design.md)
- [Day 8 RTX 5090 资源定标执行报告](reports/day-08-resource-calibration-report.md)

## 当前阶段

Day 8 的 GPU 实验阶段已经结束。33,833,984 参数 Baseline 已在 RTX 5090 上冻结 `micro_batch_size=16`、`gradient_accumulation_steps=8`、`num_workers=4`、`pin_memory=false`，形成 65,536 tokens/update、4,578 total updates 和 92 warmup updates 的可执行计划。

真实 Pilot 上的 isolated sweep、BF16 dry-run、25-step 短跑、step 10/20 validation、独立 resume 对照和正式入口 20 → 25 resume 均通过。A57 证据包已在本地核对 SHA，A57 与 F19 均由用户确认关机。

当前 Day 8 按 96% 记录：四字段冻结和报告已经应用，83 项定向测试、531 项完整回归与 `git diff --check` 均通过。剩余工作是最终审查 diff、commit、push 并确认 `HEAD == origin/main`。在 Git 门完成以及 Full corpus 独立验证前，不允许启动 300M-token 正式预训练。当前 Pilot 短跑结果也不能作为模型质量结论。
