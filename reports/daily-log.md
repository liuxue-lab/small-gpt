# Small GPT 开发日志

## Day 1：项目初始化与环境验证

日期：2026-08-06

### 今日目标

- 建立 Small GPT 项目目录结构；
- 创建独立 Python 虚拟环境；
- 安装支持 CUDA 的 PyTorch；
- 建立 Debug 和 Baseline 两套配置；
- 建立基础自动测试；
- 明确本地开发与租用 GPU 训练的分工。

### 已完成任务

- [x] 安装 Python 3.11.9
- [x] 安装 Git 2.55.0
- [x] 创建 `.venv` 虚拟环境
- [x] 安装 PyTorch 2.11.0+cu128
- [x] 验证 CUDA 可用
- [x] 验证 GPU 前向传播和反向传播
- [x] 创建项目目录结构
- [x] 创建 `.gitignore`
- [x] 创建 `debug.yaml`
- [x] 创建 `baseline.yaml`
- [x] 创建环境检查脚本
- [x] 创建配置检查脚本
- [x] 创建基础 PyTest 测试
- [x] 编写项目 README

### 本地开发环境

| 项目 | 当前结果 |
| --- | --- |
| 操作系统 | Windows 10 |
| Python | 3.11.9 |
| PyTorch | 2.11.0+cu128 |
| PyTorch CUDA Runtime | 12.8 |
| 本地 GPU | NVIDIA GeForce RTX 5060 Laptop GPU |
| GPU 显存 | 7.96 GiB |
| CUDA 是否可用 | True |
| BF16 是否支持 | True |

### 模型配置

#### Debug 模型

- 参数量：约 2.51M
- Transformer 层数：2
- 注意力头数：2
- 隐藏维度：128
- 上下文长度：128
- 用途：本地代码调试、单元测试和单批次过拟合。

#### Baseline 模型

- 参数量：约 33.82M
- Transformer 层数：8
- 注意力头数：8
- 隐藏维度：512
- 上下文长度：512
- 词表大小：16,384
- 用途：在租用的 Linux GPU 上进行正式预训练。

### 验证结果

运行环境检查：

```powershell
python scripts/check_env.py
```

配置检查结果：

- Debug 模型参数量约 2.51M；
- Baseline 模型参数量约 33.82M；
- 两套配置检查全部通过。

自动测试结果：

- `7 passed in 1.43s`。

### Day 1 验收结论

本地开发环境、模型配置、基础测试、Git 仓库和 GitHub 远程仓库均已建立，Day 1 验收通过。

## Day 2：AutoDL 验收与数据管线

日期：2026-08-07

### 今日目标

- 验收 AutoDL RTX 5090 训练环境；
- 在服务器数据盘克隆并验证项目；
- 明确英文预训练数据源及许可证；
- 使用流式读取审计真实小样本；
- 实现最小数据清洗、去重和划分管线；
- 为每条清洗规则增加自动测试。

### 已完成任务

- [x] 启动并验收 AutoDL RTX 5090 实例
- [x] 验证服务器 Python、PyTorch、CUDA 和 BF16
- [x] 将项目克隆到 `/root/autodl-tmp/small-gpt`
- [x] 在 AutoDL 安装项目依赖
- [x] 在 AutoDL 运行环境和配置检查
- [x] 在 AutoDL 运行原有 7 个自动测试
- [x] 选择 FineWeb-Edu `sample-10BT` 作为主要英文预训练数据源
- [x] 记录 ODC-By 1.0 许可证及 Common Crawl 使用条款
- [x] 固定 FineWeb-Edu 数据集版本
- [x] 安装并记录 `datasets==5.0.1`
- [x] 流式读取 10 条真实数据进行连通性测试
- [x] 流式读取 1,000 条真实数据进行字段和质量审计
- [x] 验证原始与处理后数据不会进入 Git
- [x] 实现 Unicode 和空白标准化
- [x] 实现文本、语言和质量过滤
- [x] 实现 SHA-256 精确去重
- [x] 实现固定种子的数据划分
- [x] 实现失败时清理临时输出文件
- [x] 为数据管线增加 14 个自动测试
- [x] 运行项目全部 21 个测试

### AutoDL 环境结果

| 项目 | 结果 |
| --- | --- |
| 操作系统 | Ubuntu 22.04 / Linux 5.15 |
| Python | 3.12.3 |
| PyTorch | 2.12.1+cu130 |
| PyTorch CUDA Runtime | 13.0 |
| NVIDIA 驱动 | 595.71.05 |
| GPU | NVIDIA GeForce RTX 5090 |
| GPU 显存 | 31.36 GiB |
| CUDA 是否可用 | True |
| BF16 是否支持 | True |
| GPU 前向与反向计算 | 通过 |

服务器验证结果：

- 环境检查通过；
- Debug 与 Baseline 配置检查通过；
- `7 passed in 1.12s`。

### 数据源决定

| 项目 | 结果 |
| --- | --- |
| 数据集 | `HuggingFaceFW/fineweb-edu` |
| 配置 | `sample-10BT` |
| 数据语言 | 英文 |
| 数据规模 | 约 10B GPT-2 tokens |
| 访问方式 | Streaming |
| 固定版本 | `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9` |
| 许可证 | ODC-By 1.0 |
| 上游来源 | Common Crawl |

选择该数据集的原因：

- 数据规模足够覆盖 300M～500M tokens 的训练目标；
- 已完成上游网页抽取、质量过滤和教育质量筛选；
- 支持流式读取，不需要下载完整数据集；
- 字段中保留 URL、语言分数、质量分数和 GPT-2 token 数量，便于审计。

### 1,000 条真实样本审计结果

| 指标 | 结果 |
| --- | ---: |
| 读取记录 | 1,000 |
| 空文本 | 0 |
| 精确重复文本 | 0 |
| 重复 ID | 0 |
| 英文记录 | 1,000 |
| `date` 为空 | 1,000 |
| 最短文档 | 307 characters |
| 最长文档 | 124,276 characters |
| 平均文档长度 | 4,893.27 characters |
| GPT-2 tokens 总数 | 1,057,560 |
| 平均 GPT-2 tokens | 1,057.56 |
| 最低语言置信度 | 0.665794 |
| 最高语言置信度 | 0.994643 |

教育质量分数分布：

| 分数 | 文档数 |
| ---: | ---: |
| 3 | 856 |
| 4 | 142 |
| 5 | 2 |

审计结论：

- `date` 字段在当前样本中不可用，后续作为可选字段；
- 沿用 FineWeb 上游的 `language_score >= 0.65` 标准；
- 长文档不因超过 512-token 上下文而删除，后续在分词后切分；
- 真实小样本未发现重复，但正式数据仍必须执行精确去重；
- FineWeb 的 `token_count` 只用于规模估算，不能代替自定义 BPE 的实际 token 数量。

### 清洗与划分规则

1. 正文必须是非空字符串；
2. 使用 Unicode NFC 标准化；
3. 统一换行和水平空白；
4. 删除少于 200 个标准化字符的文档；
5. 要求 `language == "en"`；
6. 要求 `language_score >= 0.65`；
7. 要求 `int_score >= 3`；
8. 对标准化文本执行 SHA-256 精确去重；
9. 使用种子 42 进行稳定哈希划分；
10. 文档级划分比例为 train 98%、validation 1%、test 1%；
11. Tokenizer 只能使用训练集训练；
12. 先写临时文件，全部成功后再替换正式输出。

### 1,000 条样本清洗结果

| 指标 | 结果 |
| --- | ---: |
| 输入文档 | 1,000 |
| 保留文档 | 1,000 |
| 删除空文本 | 0 |
| 删除短文本 | 0 |
| 删除非英文文本 | 0 |
| 删除低语言分数文本 | 0 |
| 删除低质量文本 | 0 |
| 删除精确重复文本 | 0 |
| Train | 985 |
| Validation | 7 |
| Test | 8 |
| 保留字符数 | 4,893,141 |
| 保留 GPT-2 tokens | 1,057,560 |
| 保留率 | 100% |

全部保留是合理结果，因为 FineWeb-Edu 上游已经进行了严格筛选。异常过滤分支通过合成单元测试验证。

### 自动测试结果

- 数据管线测试：`14 passed in 0.07s`；
- 项目全部测试：`21 passed in 1.65s`。

测试覆盖：

- Unicode、换行和空白标准化；
- 有效文档保留；
- 扁平和嵌套元数据兼容；
- 空文本、短文本、非英文、低语言分数和低质量数据过滤；
- 精确去重；
- 稳定划分及比例；
- 三个 JSONL 文件输出；
- 损坏 JSON 时不保留半成品；
- 非法配置拒绝。

### 新增或修改文件

- `requirements.txt`
- `data/README.md`
- `scripts/inspect_dataset.py`
- `scripts/prepare_data.py`
- `tests/test_data_pipeline.py`
- `reports/day-02-inspection.json`
- `reports/day-02-cleaning-stats.json`
- `reports/day-02-data-audit.md`
- `reports/daily-log.md`

原始样本和处理后的 JSONL 文件保存在 `data/` 下，并由 `.gitignore` 排除。

### 遇到的问题与处理

1. AutoDL 直接访问 GitHub 时克隆停滞：启用 `/etc/network_turbo` 后成功克隆；
2. `date` 字段全部为空：将其调整为可选元数据；
3. Python 浮点数将 1% 显示为长尾小数：在报告中进行稳定舍入；
4. 直接运行 `pytest` 时找不到项目根目录：统一使用 `python -m pytest`；
5. 测试文件中的 `é` 出现编码错误：改用 `\u00e9` 转义写法。

### Day 2 当前验收状态

- [x] AutoDL 硬件与软件环境通过
- [x] 服务器原有测试通过
- [x] 数据源与许可证已记录
- [x] 小样本可使用固定版本重复获取
- [x] 清洗前后数量可统计
- [x] 数据划分可复现
- [x] 新增测试全部通过
- [x] 确认 AutoDL 不使用时已经关机
- [x] 提交并推送 Day 2 代码

### Day 3 计划

- 将小样本脚本扩展为正式的流式数据获取管线；
- 按目标 token 数停止读取，而不是按固定文档数量停止；
- 支持分片输出，避免生成单个超大文件；
- 统计正式语料清洗前后的数量与大小；
- 为后续 BPE Tokenizer 训练生成稳定的 train、validation 和 test 文本。

## Day 4：16K ByteLevel BPE Tokenizer

日期：2026-08-09

### 今日目标

- 冻结项目 Tokenizer 的技术契约和特殊 token ID；
- 只使用 2M Pilot 的 train split 训练 ByteLevel BPE；
- 实现可复用的训练、加载、编码和统计接口；
- 保存并重新加载 Tokenizer 产物；
- 统计 train、validation、test 的真实模型 token 数；
- 完成专项测试、完整回归、报告和 Git 交付。

### 已完成任务

- [x] 固定 `tokenizers==0.23.1`
- [x] 新增 `configs/tokenizer.yaml`
- [x] 确认 Day 3 文本规范化使用 Unicode NFC
- [x] 固定 16,384 词表与 ByteLevel BPE 配置
- [x] 固定 `<bos>=0`、`<eos>=1`、`<pad>=2`、`<unk>=3`
- [x] 实现 Tokenizer 核心模块
- [x] 实现训练与评估命令行脚本
- [x] 为 Tokenizer 增加 37 项离线测试
- [x] 只使用 4 个 Pilot train shard 完成真实训练
- [x] 保存 `tokenizer.json`、`tokenizer_config.json`、`vocab.json` 和 `merges.txt`
- [x] 完成保存后重新加载验证
- [x] 流式统计三个 Pilot split
- [x] 验证普通语料 `<unk>` 为 0
- [x] 验证 EOS 数量与文档数量守恒
- [x] 验证全部 token ID 可安全存入 `uint16`
- [x] 运行完整项目回归，66 项测试全部通过
- [x] 确认 Pilot 数据继续被 Git 忽略
- [x] 生成 Day 4 机器可读统计和执行报告

### Tokenizer 配置

| 配置项 | 结果 |
| --- | --- |
| 实现库 | `tokenizers==0.23.1` |
| Tokenizer | ByteLevel BPE |
| 词表大小 | 16,384 |
| `min_frequency` | 2 |
| `max_token_length` | 64 |
| Normalizer | NFC |
| Pre-tokenizer | `ByteLevel(add_prefix_space=False, use_regex=True)` |
| BPE dropout | `None` |
| 文档边界 | 每篇文档末尾追加一个 `<eos>` |
| BOS 策略 | 预训练语料不自动添加 BOS |

### 真实训练结果

运行命令：

```powershell
python .\scripts\train_tokenizer.py --config .\configs\tokenizer.yaml
```

| 指标 | 结果 |
| --- | ---: |
| Train files | 4 |
| Train records | 1,824 |
| Train provided tokens | 1,967,041 |
| Vocabulary size | 16,384 |
| Compute merges | 16,124 |
| Elapsed | 1.789 秒 |
| Save/reload validation | 通过 |

Tokenizer 主产物 SHA-256：

```text
b26835e02eebf777a257c4732abdd6f9732a115967d2ad839f3a1a00e45ee8c5
```

### 全 Pilot 评估结果

运行命令：

```powershell
python .\scripts\evaluate_tokenizer.py --config .\configs\tokenizer.yaml
```

| Split | 文档 | Provided tokens | 模型 tokens | Model/Provided | `<unk>` |
| --- | ---: | ---: | ---: | ---: | ---: |
| Train | 1,824 | 1,967,041 | 2,093,241 | 1.064157 | 0 |
| Validation | 16 | 12,600 | 13,933 | 1.105794 | 0 |
| Test | 12 | 20,442 | 22,602 | 1.105665 | 0 |
| **合计** | **1,852** | **2,000,083** | **2,129,776** | **1.064844** | **0** |

守恒关系：

```text
records = eos_tokens = 1852
2127924 BPE tokens + 1852 EOS tokens = 2129776 model tokens
unknown_tokens = 0
```

### 关键结论

1. 项目模型 token 数比 FineWeb-Edu provided tokens 高约 6.48%，后续训练预算必须使用实际模型 token；
2. 按 Pilot train 比例估算，350M provided-token Full 语料的 train split 约有 365,005,947 个模型 token；
3. 1,122 篇文档超过 512-token 上下文，占 60.58%，下一阶段必须正确切块而不是整体截断；
4. 全部普通语料的 `<unk>` 数量为 0，ByteLevel 覆盖符合预期；
5. 最大 token ID 为 16,383，`uint16` 足以存储 tokenized binary；
6. Day 4 不生成 `.bin/.idx`，避免在 Dataset/DataLoader 设计前过早冻结二进制格式。

### 测试结果

Tokenizer 专项测试：

```text
37 passed in 0.23s
```

完整项目回归：

```text
66 passed in 44.71s
```

完整回归包含 Day 1～Day 3 的原有 29 项测试和 Day 4 新增的 37 项测试，没有失败或错误。

### 新增或修改文件

- `requirements.txt`
- `configs/tokenizer.yaml`
- `tokenizer/__init__.py`
- `tokenizer/bpe.py`
- `scripts/train_tokenizer.py`
- `scripts/evaluate_tokenizer.py`
- `tests/test_tokenizer.py`
- `tokenizer/artifacts/tokenizer.json`
- `tokenizer/artifacts/tokenizer_config.json`
- `tokenizer/artifacts/vocab.json`
- `tokenizer/artifacts/merges.txt`
- `reports/day-04-tokenizer-stats.json`
- `reports/day-04-tokenizer-report.md`
- `reports/daily-log.md`
- `README.md`

### 遇到的问题与处理

1. PowerShell 和聊天界面可能转义 `<bos>` 等字符串：通过 Python 读取 YAML 并打印替换后的显示值，确认文件内特殊 token 正确；
2. `tokenizer/__init__.py` 原本为空，复制代码时 Windows 提示同名文件：确认目标仅为零字节占位文件后安全替换；
3. `git diff --check` 显示 LF 将来可能转换为 CRLF：这是 Windows Git 行尾提示，不是空白错误；
4. Pilot 中 60.58% 文档超过上下文长度：保留原文，后续在 tokenized Dataset 阶段切块；
5. validation/test 的 Model/Provided 比例高于 train：两个 split 样本量很小，仅作真实统计，不据此调整 Tokenizer。

### Day 4 验收状态

- [x] Tokenizer 配置与依赖固定
- [x] Validation/test 未参与 BPE 训练
- [x] 16,384 词表训练完成
- [x] 特殊 token ID 固定
- [x] 保存和重新加载通过
- [x] 三个 split 统计完成
- [x] EOS、总量和 split 统计守恒
- [x] 普通语料 `<unk>` 为 0
- [x] 专项测试通过
- [x] 完整项目回归通过
- [x] 统计报告和执行报告生成

Git 精确暂存、提交和推送将在文档定稿后执行，并作为 Day 4 从 95% 更新到 100% 的最终交付门。

### 下一阶段

- 冻结 `uint16` tokenized binary 和索引格式；
- 使用当前权威 Tokenizer 编码语料；
- 实现 512-token 训练序列切块；
- 保留 `<eos>` 文档边界；
- 实现 memory-map Dataset/DataLoader；
- 为数据守恒、切块、抽样和恢复行为增加自动测试；
- 在正式预训练前保持 Tokenizer 词表与特殊 token ID 不变。

## Day 5：Tokenized Data、Dataset 与 DataLoader

日期：2026-08-10 ～ 2026-08-11

### 今日目标

- 冻结训练 token 二进制格式与文档索引格式；
- 使用 Day 4 的权威 Tokenizer 无损编码完整 2M Pilot；
- 实现文档原子分片、断点恢复、完整性校验和原子发布；
- 将多个物理 storage shard 映射为 split 内连续逻辑 token 流；
- 实现因果 `T+1` Dataset、训练采样器和验证/测试顺序窗口；
- 在 Windows 下验证 Dataset/DataLoader 与多 worker；
- 完成真实边界检查、专项测试、完整回归和报告。

### 已完成任务

- [x] 新增 `configs/tokenized_data.yaml` 并冻结 schema version 1
- [x] 固定 source manifest 与 Tokenizer SHA-256 身份
- [x] 固定 little-endian `uint16` token payload
- [x] 固定 64-byte token header
- [x] 固定 128-byte index header 与 48-byte document record
- [x] 实现 token/index header 的打包、解析与验证
- [x] 实现文档原子 tokenization 与 storage sharding
- [x] 每篇文档恰好追加一个 `<eos>`，不主动添加 BOS/PAD
- [x] 实现 staging、state checkpoint、resume 和原子发布
- [x] 实现完成态身份匹配 no-op 与拒绝静默覆盖
- [x] 实现 per-file 大小、SHA-256、索引连续性和 EOS 校验
- [x] 实现跨 storage shard 的 `SplitTokenStore`
- [x] 实现因果 `CausalWindowDataset`
- [x] 实现可复现的 epoch random train sampler
- [x] 实现 validation/test 顺序、非重叠窗口
- [x] 完成完整 Pilot 的真实编码并发布 5 个 storage shards
- [x] 验证 `T=128` 与 `T=512` 的窗口数量和 shift 关系
- [x] 验证两个真实 train shard 边界的跨 shard 读取
- [x] 验证三个 split 的 Windows `num_workers=2` DataLoader batch
- [x] Day 5 三组专项测试共 27 项全部通过
- [x] 完整项目回归 93 项全部通过
- [x] 确认 tokenized binary 与 manifest 继续被 Git 忽略
- [x] 将完成态 manifest 固化为机器可读统计报告
- [x] 生成设计文档、执行报告并更新 README

### 冻结身份

| 项目 | 结果 |
| --- | --- |
| Source manifest SHA-256 | `972c3c3e1733a3535aabb67b3153fc881d0c46f76849a3ff8e522ad18377a8ce` |
| Tokenizer SHA-256 | `b26835e02eebf777a257c4732abdd6f9732a115967d2ad839f3a1a00e45ee8c5` |
| Vocabulary size | 16,384 |
| Special IDs | `<bos>=0`、`<eos>=1`、`<pad>=2`、`<unk>=3` |
| Split order | train → validation → test |
| Tokenized manifest fingerprint | `a3eb6012c1cb3e2dab2a7839bebb04530563b19b4d5f7d8022e3c121b13ca7f3` |

### 二进制格式

| 项目 | 结果 |
| --- | --- |
| Token magic | `SGPTTOK1` |
| Token header | 64 bytes |
| Token dtype | little-endian `uint16`（`<u2`） |
| Index magic | `SGPTIDX1` |
| Index header | 128 bytes |
| Index record | 48 bytes |
| Index 内容 | shard-local offset、length、原始文本 SHA-256 |
| Shard target | 1,000,000 model tokens |
| Sharding | 文档原子，不拆分文档 |
| 跨界策略 | 可跨文档和 storage shard，不跨 split |

### 真实 Pilot 编码结果

运行命令：

```powershell
python .\scripts\tokenize_corpus.py --config .\configs\tokenized_data.yaml --profile pilot
```

| Split | 文档 | 模型 tokens | Storage shards |
| --- | ---: | ---: | ---: |
| Train | 1,824 | 2,093,241 | 3 |
| Validation | 16 | 13,933 | 1 |
| Test | 12 | 22,602 | 1 |
| **合计** | **1,852** | **2,129,776** | **5** |

Train storage shards：

| Shard | 文档 | Tokens |
| --- | ---: | ---: |
| `shard-00000` | 923 | 989,147 |
| `shard-00001` | 816 | 998,822 |
| `shard-00002` | 85 | 105,272 |

总 token payload 为 4,259,552 bytes，严格等于：

```text
2,129,776 model tokens × 2 bytes/token = 4,259,552 bytes
```

所有 split 的文档数和模型 token 数均与 Day 4 权威统计完全一致，没有丢 token、重复 EOS 或意外 BOS/PAD。

### Dataset 与 DataLoader 验证

| Context `T` | Train `all_starts` | Validation sequential | Test sequential |
| ---: | ---: | ---: | ---: |
| 128 | 2,093,113 | 108 | 176 |
| 512 | 2,092,729 | 27 | 44 |

Train split 的额外顺序窗口验证：

| Context `T` | Sequential windows | Remainder |
| ---: | ---: | ---: |
| 128 | 16,353 | 56 |
| 512 | 4,088 | 184 |

计数满足：

```text
all_starts = split_tokens - T
sequential = floor((split_tokens - 1) / T)
remainder = (split_tokens - 1) mod T
```

真实 train shard 边界验证：

| Boundary | Window start | Context | 结果 |
| ---: | ---: | ---: | --- |
| 989,147 | 988,891 | 512 | shape 正确，`shift=True` |
| 1,987,969 | 1,987,713 | 512 | shape 正确，`shift=True` |

在 `context_length=512`、`batch_size=4`、`num_workers=2` 下，train、validation、test 均成功产生：

```text
x=(4, 512), y=(4, 512), dtype=torch.int64, shift=True
```

### 测试结果

Day 5 专项测试：

```text
tests/test_binary_format.py  13 passed
tests/test_tokenization.py     6 passed
tests/test_dataset.py          8 passed
合计                          27 passed
```

完整项目回归：

```text
93 passed in 8.65s
```

完整回归包含 Day 1～Day 4 的原有 66 项测试和 Day 5 新增的 27 项测试，没有失败、错误或回退。

### 新增或修改文件

- `configs/tokenized_data.yaml`
- `data_pipeline/__init__.py`
- `data_pipeline/binary_format.py`
- `data_pipeline/tokenization.py`
- `data_pipeline/dataset.py`
- `scripts/tokenize_corpus.py`
- `scripts/inspect_tokenized_data.py`
- `tests/test_binary_format.py`
- `tests/test_tokenization.py`
- `tests/test_dataset.py`
- `reports/day-05-tokenized-data-design.md`
- `reports/day-05-tokenized-data-stats.json`
- `reports/day-05-execution-report.md`
- `reports/daily-log.md`
- `README.md`

### 关键结论

1. Pilot 的 2,129,776 个模型 token 已无损转换为训练可读取的 `<u2` payload；
2. storage shard 只是物理存储边界，不会切断 Dataset 的 split 内逻辑 token 流；
3. Dataset 使用 `T+1` token 构造严格右移一位的 `x/y`，不存在 off-by-one；
4. validation/test 使用确定性顺序窗口，train 使用可复现的随机起点采样；
5. tokenization 可在 shard 边界恢复，并在完成后原子发布；
6. 二进制数据由 Git 忽略，权威 manifest 的报告快照进入 Git，兼顾仓库体积和可审计性；
7. Day 5 未实现 GPT、未启动 Full 语料采集、未使用 AutoDL，范围保持清晰。

### Day 5 验收状态

- [x] 二进制和索引契约冻结
- [x] Source/Tokenizer 身份固定
- [x] Pilot 真实编码完成
- [x] Split、文档、EOS 和 token 总量守恒
- [x] Resume、no-op、原子发布和损坏检测通过测试
- [x] Memory-map 与跨 shard slice 通过
- [x] 因果 `T+1` 与两个真实 shard 边界通过
- [x] Windows DataLoader 多 worker 通过
- [x] 27 项专项测试通过
- [x] 93 项完整回归通过
- [x] 大型数据继续由 Git 忽略
- [x] 统计、设计、执行报告与 README 完成

### 下一阶段

- 从零实现 token embedding 与位置 embedding；
- 实现 multi-head causal self-attention；
- 实现 MLP、LayerNorm、残差连接和 Transformer Block；
- 组装 Decoder-only GPT 与 tied output head；
- 校验因果 mask、tensor shape、参数量和初始化；
- 使用 Debug 配置完成单批次前向、反向和过拟合测试；
- Full 语料采集与租用 GPU 正式训练继续保持未启动状态。

## Day 6：Decoder-only GPT 模型实现与验证

### 日期

2026-08-11

### 今日目标

在 Day 5 已冻结的 tokenized data 与因果 `x/y` 契约上，从零实现一套可验证、可反向传播的 Decoder-only GPT，并完成以下闭环：

1. 冻结模型结构、初始化、loss 与参数量契约；
2. 实现 Embedding、MLP、手写 Causal Self-Attention、Transformer Block 和完整 GPT；
3. 用单元测试证明 shape、因果性、weight tying、梯度、初始化和 state dict 行为；
4. 用 synthetic batch 完成 CUDA forward/backward；
5. 用固定 batch 过拟合证明优化方向和目标对齐正确；
6. 接入 Day 5 真实 tokenized Pilot 完成 forward/backward；
7. 运行全项目回归并保持 Git 工作区可审计。

Day 6 不包含正式训练循环、checkpoint、350M Full 语料采集或 AutoDL 训练。

### 冻结的架构契约

| 项目 | Day 6 冻结值 |
| --- | --- |
| Architecture | Decoder-only GPT |
| Block order | Pre-LayerNorm |
| Position encoding | Learned absolute position embedding |
| Attention | 手写 scaled dot-product causal MHA |
| QKV | 融合 `Linear(C, 3C)`，显式拆分 heads |
| Scale | `1/sqrt(head_dim)` |
| Causal mask | boolean 下三角，对角线可见 |
| Softmax | FP32 计算后转回输入 dtype |
| Linear bias | false |
| LayerNorm | affine，`eps=1e-5` |
| MLP | `C -> 4C -> C`，GELU tanh approximation |
| Output | final LayerNorm 与 bias-free LM head |
| Weight tying | token embedding 与 LM head 共享 Parameter |
| Base initialization | `Normal(0, 0.02)` |
| Residual projection std | `0.02/sqrt(2 * n_layer)` |
| Loss | 同位置 logits/targets cross entropy，不二次 shift |

### 配置与精确参数量

| 配置 | Layers | Heads | Hidden | FFN | Context | Vocab | 精确参数量 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Debug | 2 | 2 | 128 | 512 | 128 | 16,384 | 2,508,032 |
| Baseline | 8 | 8 | 512 | 2,048 | 512 | 16,384 | 33,833,984 |

配置检查器现在使用与真实实现一致的精确参数公式，并严格校验模型字段、维度关系、冻结选项和 Tokenizer/模型词表一致性。实际模型参数量与配置公式完全一致，tied LM head 不重复计数。

### 分阶段实现

#### Stage A～B：仓库基线与模型设计

- 确认 Day 5 最终提交为 `a569e2c`，本地与 `origin/main` 同步；
- 确认 Debug/Baseline、Day 5 manifest 和 Dataset 均存在；
- 冻结 Decoder-only GPT 的结构、数值边界、输入输出与验收条件；
- 更新 `configs/debug.yaml`、`configs/baseline.yaml`、配置检查器和配置测试；
- 生成 `reports/day-06-model-design.md`。

#### Stage C：配置、Embedding 与 MLP

- 新增严格的 `GPTConfig` 与 YAML 加载；
- 实现 token embedding 与 learned position embedding；
- 实现 bias-free `C -> 4C -> C` GELU MLP；
- 测试字段错误、维度错误、token 范围、最大上下文、dropout 和梯度。

#### Stage D：手写 Causal Self-Attention

- 使用单个融合 QKV projection；
- 显式执行 head split/merge、矩阵乘法、缩放、mask、softmax 和 value aggregation；
- 因果 mask 注册为 non-persistent buffer；
- 没有调用 PyTorch SDPA 替代首版核心 Attention；
- 测试未来概率、行和、前缀隔离、边界长度和 backward。

#### Stage E～F：Transformer Block 与完整 GPT

- 实现两条 residual path 的 Pre-LN Block；
- 组装 Embedding、Blocks、final LayerNorm 和 LM head；
- 实现 `GPTOutput(logits, loss)`；
- 完成 weight tying、基础初始化和残差投影缩放初始化；
- 测试整模因果性、loss、连续 backward、optimizer 去重和 state dict round-trip。

### 模型专项测试

| 测试文件 | 结果 |
| --- | ---: |
| `tests/test_model_config.py` | 35 passed |
| `tests/test_layers.py` | 23 passed |
| `tests/test_attention.py` | 23 passed |
| `tests/test_model.py` | 32 passed |
| **合计** | **113 passed** |

专项测试没有失败或错误。它们直接证明了：

- Attention 和整个 GPT 都无法观察未来 token；
- causal mask 对角线可见，未来概率为 0；
- logits 为 `[B,T,V]`，loss 为有限 scalar；
- Day 5 已 shift 的 targets 不会被模型二次 shift；
- tied weights 是同一个 Parameter 和同一块 storage；
- optimizer 参数遍历不会重复共享权重；
- 所有应训练参数均可获得有限梯度；
- 固定随机种子可复现初始化；
- strict state dict 加载后 tying 仍成立；
- causal mask 不进入参数列表或 state dict。

### Synthetic CUDA forward/backward

执行条件：Debug 配置、CUDA、seed 42、batch size 4、sequence length 128。

| 指标 | 实测结果 |
| --- | --- |
| Parameters / trainable | 2,508,032 / 2,508,032 |
| Weight tying | `True` |
| Input / target | `(4,128)` / `(4,128)`，`torch.int64` |
| Causal `x/y` shift | `True` |
| Input token range | `[9,16370]` |
| Logits | `(4,128,16384)` |
| Loss | `9.726461` |
| Logits finite | `True` |
| Gradient tensors | 20 |
| Nonzero gradients | 20 |
| Gradients finite | `True` |

该结果证明完整模型可在本地 GPU 上接收标准 token batch、生成有限 loss 并完成反向传播。

### 固定 batch 过拟合

执行条件：Debug 配置、CUDA、seed 42、固定 synthetic batch、batch size 4、sequence length 32、200 steps、learning rate 0.003。

| 指标 | 实测结果 |
| --- | ---: |
| Initial loss | 9.704769 |
| Final loss | 0.003591 |
| Final / Initial | 约 0.037% |
| Elapsed | 6.66 seconds |

最终 loss 远低于初始值的 50% 验收线，说明目标对齐、cross entropy、梯度路径和参数更新方向正确。此处 AdamW 只用于诊断，权重未写入磁盘。

### 真实 tokenized Pilot forward/backward

执行条件：Debug 配置、CUDA、Day 5 Pilot manifest、batch size 4、sequence length 128、`num_workers=0`、seed 42。

| 指标 | 实测结果 |
| --- | --- |
| Manifest exists | `True` |
| Input / target | `(4,128)` / `(4,128)`，`torch.int64` |
| Causal `x/y` shift | `True` |
| Input token range | `[15,15865]` |
| Logits | `(4,128,16384)` |
| Loss | `9.744143` |
| Logits finite | `True` |
| Gradient tensors | 20 |
| Nonzero gradients | 20 |
| Gradients finite | `True` |
| Elapsed | 4.97 seconds |

Day 5 的 Dataset/DataLoader 和 Day 6 的 GPT 已完成真实接口闭环。运行期间 Pilot manifest、二进制数据和 Tokenizer 均保持只读。

### 完整项目回归

最终执行：

```powershell
python .\scripts\check_config.py
python -m pytest -q
```

结果：

```text
All configuration checks passed.
216 passed in 9.94s
```

PowerShell 外层计时为 13.57 seconds。完整回归覆盖 Day 1～Day 6，没有 failed、error 或 skipped；执行后 Git 工作区仍然干净。

### Day 6 代码提交

| Commit | 内容 |
| --- | --- |
| `06efac4` | `feat: define decoder-only GPT model contract` |
| `4632aa2` | `feat: add GPT config embeddings and MLP` |
| `053ce40` | `feat: add handwritten causal self-attention` |
| `1bf1a52` | `feat: assemble decoder-only GPT model` |
| `3eddd2b` | `test: verify full GPT model contract` |

文档收口将作为独立提交精确暂存，便于将代码实现、测试增强和最终实测记录分别审计。

### 新增或修改文件

- `configs/baseline.yaml`
- `configs/debug.yaml`
- `model/__init__.py`
- `model/attention.py`
- `model/block.py`
- `model/config.py`
- `model/gpt.py`
- `model/layers.py`
- `scripts/check_config.py`
- `scripts/inspect_model.py`
- `tests/test_attention.py`
- `tests/test_config.py`
- `tests/test_layers.py`
- `tests/test_model.py`
- `tests/test_model_config.py`
- `reports/day-06-model-design.md`
- `reports/day-06-execution-report.md`
- `reports/daily-log.md`
- `README.md`

### 关键问题与处理

1. Day 5 已经提供右移一位的 targets，因此模型直接对同位置 logits/targets 求 loss，避免二次 shift；
2. 因果性不只检查 mask 形状，还检查未来概率、Attention 前缀隔离和整模 token 前缀隔离；
3. 首版 Attention 保留显式矩阵计算，确保缩放、mask、softmax dtype 与 head 变换均可审计；
4. weight tying 同时验证对象 identity、storage pointer、参数遍历和加载后行为；
5. 固定 batch 过拟合仅为模型诊断，不被误写成正式训练完成；
6. Git 的 LF/CRLF 提示不影响代码正确性；每个阶段均用 `git diff --check` 排除空白错误；
7. 模型验证只读复用 Pilot，没有修改或重新生成 Day 5 数据。

### Day 6 验收状态

- [x] 架构、配置、初始化与 loss 契约冻结
- [x] Token/position embedding 与 GELU MLP 完成
- [x] 手写 Causal Self-Attention 完成
- [x] Pre-LN Transformer Block 与完整 GPT 完成
- [x] Final LayerNorm 与真正的 weight tying 完成
- [x] Debug/Baseline 精确参数量验证通过
- [x] Attention 与整模因果行为验证通过
- [x] 梯度、optimizer 去重、初始化与 state dict 验证通过
- [x] 113 项模型专项测试通过
- [x] Synthetic CUDA forward/backward 通过
- [x] 固定 batch 过拟合通过
- [x] 真实 Pilot forward/backward 通过
- [x] 216 项完整回归通过
- [x] 正式训练、checkpoint、Full 语料和 AutoDL 保持未启动

### Day 6 结论

项目已经首次形成完整且可验证的模型链路：

```text
FineWeb-Edu Pilot
-> 16K ByteLevel BPE
-> uint16 tokenized corpus
-> reproducible causal x/y DataLoader
-> handwritten Decoder-only GPT
-> finite next-token loss
-> correct backward
-> fixed-batch overfit
```

这证明当前模型实现不仅能通过形状测试，也能接收真实语料 batch、保持因果性、产生有限 loss、传播梯度并被优化。

### 下一阶段

- 冻结 AdamW、weight decay 参数分组与学习率调度契约；
- 实现 gradient accumulation 和精确 token/step 计数；
- 实现 BF16/FP32 autocast 与 GradScaler 边界；
- 实现 validation loss 与评估频率；
- 实现 checkpoint 原子保存、strict 恢复以及 optimizer/scheduler/RNG 状态；
- 使用 Debug 模型完成短训练、中断和恢复一致性测试；
- 全部通过后再准备 Baseline 与租用 GPU；
- 350M Full 数据与正式预训练仍需独立验收后启动。
