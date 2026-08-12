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

## Day 7：单卡训练系统、Checkpoint 与精确恢复

### 日期

2026-08-11～2026-08-12

### 今日目标

在 Day 5 tokenized data 与 Day 6 Decoder-only GPT 的冻结契约上，实现一套可以用于后续正式预训练的单卡训练系统，并用本地 Debug 模型完成以下闭环：

1. 严格训练配置与 step/token 计数；
2. AdamW 参数分组与 warmup/cosine scheduler；
3. gradient accumulation、裁剪、有限值检查和 FP32/BF16；
4. 确定性训练数据流与固定 validation；
5. JSONL 日志、resolved config 和 run metadata；
6. 原子 checkpoint 与 model/optimizer/scheduler/RNG/data cursor 恢复；
7. 正式训练 CLI、dry-run、stop gate 和显式 resume；
8. Pilot 200-update Debug 训练与 Day 1～Day 7 完整回归。

Day 7 不启动 AutoDL，不构建 Full，不执行 33,833,984 参数 Baseline 的长时间训练。

### 冻结的训练计数

| 概念 | 定义 |
| --- | --- |
| Micro-step | 一个 micro-batch 的 forward + backward |
| Optimizer update | 累积完成后的一次 optimizer step |
| Global step | 已成功完成的 optimizer updates |
| Interval step | log/eval/save 均以 optimizer update 计数 |

Debug 计划：

```text
micro batch 4 × context 128 × accumulation 1
= 512 tokens/update

512 × 200 updates
= 102,400 tokens
```

### 已完成任务

- [x] 严格解析 Debug 与 Baseline training fields
- [x] 实现 max-steps/target-tokens 与 warmup-steps/ratio 互斥
- [x] 保持 Baseline 资源字段 unresolved，不进行本地猜测
- [x] 实现 TrainerState schema v1
- [x] 按 Parameter identity 去重 tied weights
- [x] 构建 AdamW decay/no-decay groups
- [x] 实现 linear warmup + cosine decay
- [x] 实现 accumulation、loss/N、finite checks 和 grad clipping
- [x] 实现具体 CUDA device resolution
- [x] 实现 FP32 与 CUDA BF16 precision policy
- [x] 实现可恢复 train data stream 与固定 validation stream
- [x] 实现 token-weighted validation loss/perplexity
- [x] 实现 JSONL metric logger 和 run directory 防覆盖
- [x] 实现 schema v1 原子 checkpoint
- [x] 保存并恢复模型、optimizer、scheduler、trainer state 和 RNG
- [x] 保存并恢复数据 sample cursor
- [x] 实现 checkpoint identity 与 strict mismatch rejection
- [x] 实现正式 interval training loop
- [x] 实现 `scripts/train_gpt.py`
- [x] 完成 dry-run、1/3/20/200-update Pilot gates
- [x] 完成 BF16 5-step smoke
- [x] 完成 continuous vs resume 一致性测试
- [x] 完成正式 CLI step 5 → step 10 resume
- [x] 完成 147 项最终核心专项测试
- [x] 完成 508 项全项目回归
- [x] 确认 runs/checkpoints 未进入 Git

### Training config 结果

Debug：

| 项目 | 结果 |
| --- | ---: |
| Execution ready | true |
| Tokens/micro-step | 512 |
| Tokens/update | 512 |
| Total updates | 200 |
| Warmup updates | 20 |
| Planned tokens | 102,400 |
| Token overshoot | 0 |

Baseline 静态解析通过，但以下资源字段保持 unresolved：

```text
micro_batch_size
gradient_accumulation_steps
num_workers
pin_memory
```

因此 Baseline `execution_ready=False`，Day 7 不允许直接启动。

### Optimizer 与 Scheduler

Debug 2,508,032 个参数被分为：

| Group | Tensors | Parameters |
| --- | ---: | ---: |
| Decay | 10 | 2,506,752 |
| No-decay | 10 | 1,280 |

`token_embedding.weight` 与 `lm_head.weight` 是 tied aliases，没有重复进入 optimizer。

LR 关键边界：

| Update index | LR |
| ---: | ---: |
| 0 | 0.000015 |
| 19 | 0.000300 |
| 20 | 0.000300 |
| 199 | 0.000030 |

### Precision 与单次 Update

Synthetic FP32 update 在 `cuda:0` 上通过：

| 指标 | 结果 |
| --- | ---: |
| Update index | 0 |
| Completed global step | 1 |
| Loss | 9.732769 |
| Learning rate | 0.000015 |
| Grad norm before clip | 1.261182 |
| Samples / tokens | 4 / 512 |
| Parameter changed | true |

BF16 使用 CUDA autocast，不使用 GradScaler；FP16 不在 Day 7 范围内。

### 数据、验证与日志

- train 使用 `all_starts` + deterministic random sampler；
- validation 使用 `sequential` non-overlapping windows；
- DataLoader 使用独立派生 generator，避免改变全局 Torch RNG；
- evaluation 不推进训练 sampler 或 data cursor；
- validation loss 按 token 加权；
- 每次运行保存 `metrics.jsonl`、`resolved-config.yaml` 和 `run-metadata.json`；
- resume 前严格检查已有日志不能领先于 checkpoint。

首次真实 3-step pipeline：

| Step | Train loss | LR |
| ---: | ---: | ---: |
| 1 | 9.722784 | 0.000015 |
| 2 | 9.745681 | 0.000030 |
| 3 | 9.725891 | 0.000045 |

step 3 validation 使用 1,024 tokens，loss 为 `9.711104`，perplexity 为 `16,499.814097`。

### Checkpoint 与 Resume

核心 CUDA FP32 对照运行：

```text
continuous 4 updates
vs.
2 updates → save → reload into new objects → 2 updates
```

结果：

- checkpoint `step-00000002.pt` 原子写入成功；
- 恢复 global step 2、tokens 1,024；
- 下一 batch exact；
- 下一 LR `4.5e-05`；
- scheduler state exact；
- Python/NumPy/Torch RNG exact；
- continuation metrics、model 和 optimizer 在 `rtol=1e-6, atol=1e-7` 内一致；
- 最终 global step 4、tokens 2,048。

正式 CLI 的 step 5 → step 10 resume 还验证了：

- data batches/samples 恢复为 5 / 20；
- step 10 loss `9.656713485717773` 与连续运行同一步完全相同；
- JSONL 事件连续追加，没有重复 train steps；
- step 5 与 step 10 checkpoint 均保留。

### BF16 Pilot smoke

运行 ID：`day07-debug-pilot-bf16-step5`。

| 项目 | 结果 |
| --- | --- |
| Device / precision | `cuda:0` / `bf16` |
| Autocast / GradScaler | true / false |
| Updates / tokens | 5 / 2,560 |
| Initial / final loss | 9.723145 / 9.727051 |
| Non-finite events | 0 |
| Wrong policy events | 0 |
| Checkpoint | `step-00000005.pt`，30,138,029 bytes |

该 smoke 只验证数值路径与状态系统，不要求 5 个随机 batch 的 loss 单调下降。

### 完整 200-update Debug/Pilot 运行

运行 ID：`day07-debug-pilot-200`。

| 指标 | 结果 |
| --- | ---: |
| Device / precision | `cuda:0` / FP32 |
| Updates / tokens | 200 / 102,400 |
| JSONL events | 213 |
| Initial train loss | 9.7227840423584 |
| Final train loss | 7.44751167297363 |
| Train loss reduction | 23.40% |
| Initial validation loss | 9.32964119911194 |
| Final validation loss | 7.51715626716614 |
| Validation loss reduction | 19.43% |
| Initial perplexity | 11,267.0881 |
| Final perplexity | 1,839.3293 |
| Perplexity reduction | 83.68% |
| Evaluations | 10 |
| Checkpoints | 2 |
| Non-finite train/eval events | 0 / 0 |

梯度范数记录的是裁剪前数值：minimum `0.664668`、maximum `3.552653`、mean `1.017803`。

吞吐日志：

| 指标 | tokens/s |
| --- | ---: |
| Aggregate update throughput | 62,137.05 |
| Steady-state step 2～200 | 73,061.11 |
| Minimum / maximum update | 2,020.43 / 89,516.75 |

该吞吐不包含 validation、checkpoint 和完整进程开销，不能外推为 Baseline 正式吞吐。

Validation 在 step 20、40、60、80、100、120、140、160、180、200 执行，loss 依次为：

```text
9.329641
8.760921
8.303771
7.986725
7.774237
7.652070
7.588548
7.552096
7.531419
7.517156
```

检查点：

```text
step-00000100.pt  30,138,029 bytes
step-00000200.pt  30,138,029 bytes
```

### 测试结果

最终核心专项门：

```text
147 passed in 6.17s
PowerShell outer elapsed: 7.91s
```

Day 1～Day 7 完整回归：

```text
508 passed in 13.49s
PowerShell outer elapsed: 15.26s
Exit code: 0
Skipped: 0
```

完整回归后 `git status --porcelain` 条目数为 0。

### Day 7 代码提交

| Commit | 内容 |
| --- | --- |
| `d679642` | `feat: add strict training configuration contract` |
| `2ee8a8a` | `feat: add training state AdamW and cosine scheduler` |
| `8391b28` | `feat: add accumulated training updates and precision` |
| `2bca55d` | `feat: add deterministic data evaluation and logging` |
| `7e80569` | `feat: add atomic checkpoint and exact resume` |
| `c01c087` | `feat: add formal training entry and interval loop` |

### 关键问题与处理

1. Scheduler 测试最初错误地把 post-warmup minimum LR 当成 warmup 下界；修正测试 phase 语义，没有修改正确的 warmup 公式；
2. 抽象 `torch.device("cuda")` 与实际参数设备 `cuda:0` 比较不一致；device resolution 改为具体当前 CUDA index；
3. DataLoader 构造可能消耗全局 Torch RNG；增加独立 loader generator，保证 resume RNG 轨迹；
4. Python 构造器不能直接粘贴到 PowerShell；后续通过脚本或 PyTest 执行；
5. `LF will be replaced by CRLF` 只是 Windows 换行提示，仍通过 `git diff --check` 检查真实空白错误；
6. 200-step 正式 run 在 Stage G 尚未 commit 时执行，因此 metadata 如实记录 `7e80569 + dirty=true`；同一组代码随后精确提交为 `c01c087`，最终完整回归在干净提交上通过。

### Day 7 验收状态

- [x] Training config、state、optimizer、scheduler
- [x] Accumulation、clip、finite checks
- [x] FP32 与 CUDA BF16
- [x] Deterministic train/validation streams
- [x] JSONL、resolved config、metadata
- [x] Atomic checkpoint 与 strict resume
- [x] Dry-run 与正式 CLI
- [x] Pilot 200 updates
- [x] BF16 smoke
- [x] Resume 一致性
- [x] 508 项完整回归
- [x] AutoDL/Full 未启动

### Day 7 结论

项目现已形成完整、可验证、可恢复的本地训练工程链路：

```text
FineWeb-Edu Pilot
→ frozen 16K BPE
→ recoverable token store
→ causal DataLoader
→ handwritten Decoder-only GPT
→ AdamW + warmup/cosine
→ FP32/BF16 accumulated updates
→ deterministic validation
→ JSONL metrics
→ atomic checkpoint
→ strict resume
```

这证明训练系统已经具备进入云端资源定标的工程基础，但不代表 34M Baseline 已经完成预训练或具备实用生成能力。

### 下一阶段

- 在 Day 7 文档提交并普通 push 后核对本地/远端 hash；
- 由用户明确授权后再启动 AutoDL；
- 在 RTX 5090 上探测 Baseline micro-batch、accumulation、显存和吞吐；
- 冻结 workers、pin memory 与 checkpoint 路径；
- 运行 Baseline BF16 短 smoke 和 resume；
- 完成 Full 数据与磁盘预算后，才允许启动 300M-token 正式预训练。

## Day 8：RTX 5090 Baseline 资源定标与短程验收

### 日期

2026-08-12

### 今日目标

在不构建 Full、不启动 300M-token 正式训练的前提下，使用 AutoDL RTX 5090 和真实 tokenized Pilot：

1. 建立隔离式资源探针；
2. 探测 Baseline micro batch、accumulation、workers 与 pin memory；
3. 冻结四个资源字段与精确 update/token 计划；
4. 运行 Baseline BF16 dry-run、短跑、validation 与 checkpoint；
5. 验证独立 resume 对照和正式训练入口 resume；
6. 下载完整证据并核对 SHA；
7. 关闭计费实例，生成本地配置与文档合入包。

### 授权边界

用户授权 AutoDL RTX 5090，本轮上限 60 分钟，允许远端只读检查、`git pull --ff-only`、Pilot 资源探测和后续短跑 checkpoint；明确禁止 Full 数据和 300M-token 正式训练。

执行过程中只上传 2M Pilot tokenized artifacts，最长训练路径为 25 updates / 1,638,400 tokens。没有构建 Full，没有启动正式预训练。证据下载并验证后，A57 与 F19 均由用户确认关机。

### Stage A 本地准备

新增资源探针：

- `train/resource_probe.py`
- `scripts/probe_baseline_resources.py`
- `tests/test_resource_probe.py`
- `reports/day-08-resource-calibration-design.md`

本地门：

```text
tests/test_resource_probe.py: 23 passed
full regression: 531 passed in 14.24s
```

Stage A 提交并推送：

```text
2e3166c395d7057cb8509fda6f5768bd9b203537
feat: add isolated baseline resource probe
HEAD == origin/main
```

### 远端身份

| 项目 | 结果 |
| --- | --- |
| GPU | NVIDIA GeForce RTX 5090 |
| Driver | 580.105.08 |
| VRAM | 32,607 MiB |
| Python | 3.12.3 |
| PyTorch | 2.12.1+cu130 |
| CUDA | 13.0 |
| cuDNN | 92000 |
| BF16 | supported |
| tokenizers | 0.23.1 |
| Commit | `2e3166c395d7057cb8509fda6f5768bd9b203537` |
| Git status | clean |
| 持久盘 | 50 GB；证据采集时剩余 49 GB |

AutoDL 控制台显示 25 CPU cores、90 GB RAM、¥2.78/小时。

### Pilot 上传与身份验证

| 项目 | 结果 |
| --- | --- |
| 上传 ZIP SHA-256 | `2bda0cd8fa495cba9314a06c633458d96f1130d1b1e240638f9e24fe528ba396` |
| Archive safety | PASS，16 entries |
| Manifest SHA-256 | `141a0c4626cb4f5ba8b041984514825b0d30a34ac8f51a8fcc2fdbb6e512f961` |
| Dataset fingerprint | `a3eb6012c1cb3e2dab2a7839bebb04530563b19b4d5f7d8022e3c121b13ca7f3` |
| Tokenizer SHA-256 | `b26835e02eebf777a257c4732abdd6f9732a115967d2ad839f3a1a00e45ee8c5` |
| Model tokens | 2,129,776 |
| Payload bytes | 4,259,552 |
| Storage shards | 5 |
| Shard scan | PASS |

Tokenizer metadata 的 SHA 差异被定位为 Windows CRLF 与远端 LF 的字节差异。严格转换为 CRLF 后精确匹配期望 SHA，因此记录 `PASS_CRLF`。source corpus manifest 没有随 Pilot 上传，身份检查标记 `DEFERRED_BY_SCOPE`；正式 Full/训练前仍是硬门。

### Micro-batch sweep

所有候选均成功：

| Batch | Tokens/s | Peak allocated GiB | Peak reserved GiB | Reserved |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 31,842 | 0.777 | 0.812 | 2.59% |
| 2 | 69,474 | 1.034 | 1.037 | 3.31% |
| 4 | 126,529 | 1.560 | 1.621 | 5.17% |
| 8 | 228,223 | 2.594 | 2.637 | 8.41% |
| 12 | 236,347 | 3.631 | 3.834 | 12.23% |
| 16 | 241,017 | 4.663 | 5.053 | 16.11% |

选择 batch 16。它是最大已测成功候选，不是已找到的 OOM 上限。b12 → b16 吞吐只提升约 1.98%，因此不继续追求更大 batch，保留 validation、allocator 和云端波动余量。

### Accumulation

| Accumulation | Tokens/update | Total updates | Warmup | Tokens/s | Peak reserved |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 16,384 | 18,311 | 367 | 252,141 | 5.553 GiB |
| 4 | 32,768 | 9,156 | 184 | 251,133 | 5.553 GiB |
| 8 | 65,536 | 4,578 | 92 | 249,709 | 5.553 GiB |

选择 accumulation 8。相对 a2 吞吐只低约 0.96%，显存不增加，并形成清晰的 65,536 tokens/update 正式计划。

### DataLoader sweep

| Workers | Pin | Tokens/s |
| ---: | --- | ---: |
| 0 | false | 238,838 |
| 0 | true | 237,098 |
| 2 | false | 228,000 |
| 2 | true | 230,212 |
| 4 | false | 246,578 |
| 4 | true | 224,355 |
| 8 | false | 235,713 |
| 8 | true | 236,980 |

选择 workers 4、pin memory false。所有候选 peak reserved 均为 5.553 GiB / 17.71%。

### 冻结配置

```yaml
micro_batch_size: 16
gradient_accumulation_steps: 8
num_workers: 4
pin_memory: false
```

精确计划：

```text
tokens/micro-step = 16 × 512 = 8,192
tokens/update = 8,192 × 8 = 65,536
total updates = ceil(300,000,000 / 65,536) = 4,578
warmup updates = ceil(4,578 × 0.02) = 92
planned tokens = 300,023,808
overshoot = 23,808
```

只冻结以上四字段。正式 Baseline 的 `eval_interval=500`、`eval_batches=100` 和 `save_interval=1000` 不变。短跑使用 10 / 3 / 20 只是为了在有限时间内增加 validation 与 checkpoint 覆盖。

### BF16 dry-run 与短跑

Dry-run 通过：

```text
Model parameters: 33,833,984
Device / precision: cuda:0 / bf16
Autocast: true
GradScaler: false
Sample input/target: (16, 512)
Tokens/update: 65,536
Total/warmup updates: 4,578 / 92
Source dirty: false
```

20-step 短跑和 20 → 25 resume：

| Step | Train loss | Validation loss | Perplexity | Grad norm |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 9.815502 |  |  | 7.472572 |
| 10 | 8.993008 | 8.953871 | 7,737.787 | 2.180461 |
| 20 | 8.685449 | 8.641006 | 5,659.020 | 1.769621 |
| 25 resumed | 8.486105 |  |  | 1.614367 |

最终 25 updates / 1,638,400 tokens；所有训练和验证数值有限。JSONL 共 31 events，train steps 1～25 连续，evaluation steps 为 10/20，checkpoint steps 为 20/25，resume step 为 20。

### Checkpoint 与 resume

独立 2 → 4 step 对照通过：

- 下一 batch exact；
- scheduler exact；
- Python/NumPy/Torch RNG exact；
- continuation metrics、model、optimizer 在 `rtol=0.0005, atol=0.00005` 内一致；
- 最终 step 4 / 262,144 tokens。

正式 checkpoint：

| Step | Bytes | SHA-256 |
| ---: | ---: | --- |
| 20 | 406,108,827 | `696d544311a74211b0bbe04d3d33a15eddebe969c9214ac23063ff483b794d80` |
| 25 | 406,108,827 | `7ed909498406151e22020df4ab5d7f27a55a1bfe09c688d91ef65f9f3d7f0e09` |

保存耗时均值 0.938043 秒。checkpoint load 没有单独计时，不能虚构 load latency。

### 吞吐、ETA 与费用

原始 JSONL 重算的 step 2～25 steady update throughput，排除 resume 后冷启动 step 21，为 236,442.75 tokens/s。按正式 interval 模型估算约 21.31 分钟 / ¥0.99；把 20-step 的 11 秒完整墙钟直接外推得到保守上界约 41.96 分钟 / ¥1.94。

规划范围写作 21～42 分钟、¥0.99～¥1.94，只是当前 RTX 5090/Pilot 环境的计算估算，不包含 Full 构建、云端抖动、热降频和重跑。

### 证据包与关机

远端证据包 SHA-256：

```text
8ebcb2b968a96ca1a7cfa950a5d2ff6188e9e41f44cb925e65f0eb19bef61966
```

本地下载后的 SHA 与远端一致，包内 `SHA256SUMS.txt` 记录的 24 个文件全部通过。A57 与 F19 随后由用户确认关机；实例未释放，持久盘数据仍保留。

### Stage I 本地应用与最终回归

Stage I 更新包通过源 commit、`origin/main`、clean worktree、六个源文件 SHA 和七个 payload SHA 门后完成应用：

```text
StageIPreflight=PASS
StageIApply=PASS
SourceCommit=2e3166c395d7057cb8509fda6f5768bd9b203537
```

本地最终验证：

```text
Targeted: 83 passed in 2.41s
Targeted outer elapsed: 3.20s
Targeted exit code: 0

Full regression: 531 passed in 13.95s
Full outer elapsed: 15.71s
Full exit code: 0

git diff --check exit code: 0
```

工作区范围精确为 6 个修改文件和 1 个新增报告，没有出现 `data/`、`runs/`、`checkpoints/` 或原始远端证据文件。Baseline diff 只有 `micro_batch_size=16`、`gradient_accumulation_steps=8`、`num_workers=4`、`pin_memory=false` 四项变化。

### Stage I commit 与远端闭环

最终功能冻结提交及远端核对结果：

```text
Commit=9a6f2fed495669b994aa1e85706fe461535e1883
Message=feat: freeze calibrated baseline resources
PushExitCode=0
HEAD=9a6f2fed495669b994aa1e85706fe461535e1883
origin/main=9a6f2fed495669b994aa1e85706fe461535e1883
RemoteMain=9a6f2fed495669b994aa1e85706fe461535e1883
HashesMatch=True
```

该提交包含 6 个修改文件和 1 个新增执行报告，共 `667 insertions(+), 37 deletions(-)`。普通 push 成功后，本地 `main`、远端跟踪分支 `origin/main` 与 GitHub `main` 已确认一致。

### Day 8 当前验收状态

- [x] Stage A probe 实现与 531 项回归
- [x] RTX 5090/BF16/Git 身份
- [x] Pilot 上传与数据身份扫描
- [x] Micro-batch sweep
- [x] Accumulation 选择
- [x] DataLoader sweep
- [x] Baseline BF16 dry-run
- [x] 20-step 短跑与 validation
- [x] 独立 checkpoint/resume 对照
- [x] 正式入口 20 → 25 resume
- [x] 证据下载与 SHA 对照
- [x] A57/F19 关机
- [x] 本地应用 Stage I 配置/文档更新
- [x] 83 项定向测试与 531 项完整项目回归
- [x] commit、普通 push、`HEAD == origin/main == RemoteMain`
- [x] Full 未启动
- [x] 300M-token 正式训练未启动

Day 8 当前进度为 100%。GPU 实验、配置冻结、本地回归、功能提交、普通 push 与远端 hash 核对全部完成。

### 下一阶段硬门

1. 已完成 Stage I 本地应用、83 项定向测试与 531 项完整回归；
2. 已确认 Baseline 只有四个资源字段变化；
3. 已确认 `git diff --check` exit code 为 0；
4. 已将功能冻结提交 `9a6f2fed495669b994aa1e85706fe461535e1883` 普通 push，并确认三方 hash 一致；
5. Full corpus 和 tokenized Full 必须独立构建、验证和测量磁盘；
6. 正式训练前重新核对 GPU/软件/数据身份并获得新的明确授权。

## Day 9：Full 数据、Tokenized Full 与单更新 smoke

### 日期

2026-08-12～2026-08-13

### 今日目标

在不启动 300M-token 正式预训练的前提下：

1. 把 Day 5 仅支持 Pilot 的 tokenization 扩展为 manifest-bound Full profile；
2. 固定 FineWeb-Edu Full 的具体 revision 与显式 Parquet 文件列表；
3. 在 AutoDL C14 RTX 5090 上构建 350M provided-token Full source corpus；
4. 用 Day 4 冻结 Tokenizer 构建 Tokenized Full；
5. 验证全部 shards、跨 shard `T+1` 与多 worker DataLoader；
6. 在 Full manifest 上运行 BF16 dry-run；
7. 只运行一次 optimizer update 并严格核验 checkpoint；
8. 下载小型证据包，完成 SHA 与 archive entries 核对。

### 授权与边界

用户明确授权进入 Day 9 Stage B，并允许产生 AutoDL 费用。实例为 C14 RTX 5090，控制台价格 ¥2.78/小时。

本轮明确不做：

- 不启动 4,578 updates / 300M tokens 正式训练；
- 不运行第二个 optimizer update；
- 不把单更新 loss 当作模型质量结论；
- 不提交 Full JSONL、tokenized binary、source cache 或 checkpoint；
- 不自动释放 AutoDL 实例。

### Day 9 初始 Git 状态

本地 Day 8 最终状态：

```text
b2ec0ebb4559cdf7cc1938d068742180f04fac35
docs: close Day 8 resource calibration
HEAD == origin/main
worktree clean
```

### Full tokenization 入口

新增/修改：

- `configs/tokenized_data_full.yaml`；
- `data_pipeline/tokenization.py`；
- `scripts/tokenize_corpus.py`；
- `tests/test_tokenization.py`；
- `tests/test_tokenize_corpus_cli.py`；
- `data_pipeline/__init__.py`。

第一阶段提交：

```text
5be64d3 feat: add verified full corpus tokenization
```

结果：

```text
targeted: 15 passed
full: 540 passed
profile choices: pilot, full
```

### Tokenizer metadata 跨平台身份

AutoDL Linux checkout 中：

```text
ExpectedSHA=8622711407aab3f299996b7d3009d4f4447ae35879ca8e50451b5f0adbdf5141
RawSHA=7525edf13e8f3082c9391307395a6e919272846d83fe90f152b8a0c32ab67ca8
CanonicalCRLFSHA=8622711407aab3f299996b7d3009d4f4447ae35879ca8e50451b5f0adbdf5141
IdentityResult=LINE_ENDING_ONLY
```

根因是 frozen digest 来自 Windows CRLF working tree，而 Git 在 Linux materialize 为 LF。修复保留 raw SHA 优先，只在内容可解码为 UTF-8 时同时计算 canonical LF/CRLF；只有换行不同才接受，任何其他内容变化仍拒绝。

提交：

```text
7c4ad5d fix: normalize tokenizer metadata line endings
targeted: 5 passed
full: 541 passed
```

### 远端环境

| 项目 | 结果 |
| --- | --- |
| 实例 | C14 |
| GPU | NVIDIA GeForce RTX 5090 |
| Driver | 580.105.08 |
| VRAM | 32,607 MiB |
| Python | 3.12.3 |
| PyTorch | 2.12.1+cu130 |
| CUDA | 13.0 |
| BF16 | supported |
| `datasets` | 5.0.1 |
| `pyarrow` | 25.0.1 |
| `tokenizers` | 0.23.1 |
| 数据盘 | 50 GB |

### Hugging Face 网络诊断

直接 `load_dataset()` 首先失败：

```text
[Errno 101] Network is unreachable
ConnectionError: Couldn't reach 'HuggingFaceFW/fineweb-edu'
```

AutoDL 加速脚本后，README HEAD 为 200，但 Hub tree pagination 间歇出现 503、server disconnect，且部分请求仍指向 `huggingface.co`。`hf-mirror.com` 的 README/API control requests 为 200，但 `datasets`/`huggingface_hub` 内部的分页和 range 请求仍不稳定。

本地电脑的 VPN 节点不会直接改变 AutoDL 容器出口，因此没有通过关闭本地 VPN 猜测解决远端问题。

### 显式 Full 源文件

通过 mirror API 固定 14 个文件：

```text
sample/10BT/000_00000.parquet
...
sample/10BT/013_00000.parquet
```

修改配置与实现后：

```text
4be46ee fix: freeze explicit full corpus sources
targeted gates: PASS
full: 546 passed
```

Full config fingerprint：

```text
555c15b1e851f567290fda3acf7e39245d3bc050c49259655bc071c8791aef84
```

### Lazy stream 与 local cache

一次显式源 probe 被 kill 137，但 cgroup 没有 OOM 记录。根因不是 GPU 或容器内存不足，而是多文件 eager stream 初始化/读取路径不适合该 2.15GB Parquet 网络链路。

改为每次只在消费到对应文件时调用 `load_dataset()`：

```text
c0b1df1 fix: stream frozen full sources lazily
targeted: 8 passed
full: 546 passed
```

远程 Parquet 首条记录读取仍可能超过 4 分钟并在 SSL body 中断。最终新增 verified local cache：

```text
c23876c feat: support verified local full source cache
targeted: 10 passed
full: 548 passed
```

下载第一个源文件时多次出现 `peer closed connection`，可恢复 downloader 保留 `.incomplete` 并从已下载 bytes 继续。完成结果：

```text
DownloadExitCode=0
DownloadStatus=COMPLETE_AND_VERIFIED
ResolvedPath=/root/autodl-tmp/day09-source-cache/.../sample/10BT/000_00000.parquet
ActualBytes=2152819114
ActualSHA256=b1ba7b2ce4cb5ea6ef42dca40263eabb85f37700d01693a68e9b30a31d78e871
SourceDownloadIdentity=PASS
```

断网 local-cache probe 成功读到第一条合格记录：

```text
AcceptedSourceIndex=1
AcceptedTextCharacters=3665
AcceptedProvidedTokens=845
AcceptedSplit=train
CleanOfflineLocalCacheProbe=PASS
```

### Full source corpus

正式构建：

```text
output=data/processed/fineweb_edu_full
start=2026-08-13T04:21:46+08:00
```

统计：

| 指标 | 结果 |
| --- | ---: |
| Input records | 339,027 |
| Kept records | 338,849 |
| Exact duplicates | 178 |
| Provided tokens | 350,000,812 |
| Shard groups | 70 |
| Train records/tokens | 332,112 / 343,299,897 |
| Validation records/tokens | 3,407 / 3,452,153 |
| Test records/tokens | 3,330 / 3,248,762 |
| Size | 约 1.7 GB |

身份：

```text
SourceManifestSHA256=14c69dc545838b426e29162c73132cfe444bb2cc56b72c80bb4929f3c65ca96a
FullFingerprint=555c15b1e851f567290fda3acf7e39245d3bc050c49259655bc071c8791aef84
```

### 已发布后进程 abort

构建已经写完 70 shards、state `complete` 和 manifest，但 generator teardown 触发 native abort，wrapper 得到 exit 134。

没有删除产物。先对 70 shards 做 canonical recovery：

```text
CorpusStatus=complete
VerifiedShards=70
VerifiedSourceRecords=339027
VerifiedKeptRecords=338849
VerifiedProvidedTokens=350000812
CanonicalRecoveryValidation=PASS
```

随后给 source stream 增加显式 `finally: close()`，并覆盖 success/failure/non-closeable 三种路径：

```text
23c63a6 fix: close full corpus source stream
targeted: 13 passed
local full: 551 passed in 42.46s
remote full: 551 passed, 2 warnings in 5.87s
```

真实 active Parquet 短构建：

```text
ActiveStreamProbeExit=0
ProbeStatus=complete
ProbeProvidedTokens=1900
ProbeLogContainsAbort=False
ActiveStreamLifecycleValidation=PASS
FormalCorpusStillComplete=PASS
```

### Tokenized Full

正式编码 preflight：

```text
profile=full
source manifest SHA-256=14c69dc54583...
vocabulary=16384
shard target=5000000
storage=<u2
```

运行：

```text
started=2026-08-13T04:42:43+08:00
finished=2026-08-13T04:53:38+08:00
exit=0
published=data/tokenized/fineweb_edu_full
staging=ABSENT
```

结果：

| Split | Records | Provided tokens | Model tokens | Storage shards |
| --- | ---: | ---: | ---: | ---: |
| Train | 332,112 | 343,299,897 | 372,328,191 | 75 |
| Validation | 3,407 | 3,452,153 | 3,741,345 | 1 |
| Test | 3,330 | 3,248,762 | 3,518,409 | 1 |
| **合计** | **338,849** | **350,000,812** | **379,587,945** | **77** |

```text
RawBpeTokens=379249096
AppendedEOS=338849
PayloadBytes=759175890
TokenizedManifestSHA256=ce7cd91075c7c666c427e1aaa286096a7f386643f3a76de3c26ef770d6cce67e
TokenizedConfigFingerprint=39dab5bacdf8719bbc849e85ddcd7422cba5777fc044b437d050a49b87ab174f
```

### Tokenized Full 验证

workers 0 全量 manifest/payload/index scan 通过。跨 shard 读取：

```text
BoundaryTokenOffset=4998983
BoundaryWindowStart=4998727
BoundaryReadLength=513
CrossStorageShardRead=PASS
CrossStorageShardCausalShift=PASS
ParentMemmapsClosedAfterBoundaryRead=PASS
```

workers 2 和 4 在三个 split 均得到 `(16, 512)`，causal shift 和 ID range 正确：

```text
Workers2And4BatchIdentity=PASS
AllParentMemmapsClosed=PASS
FullTokenizedLoaderValidation=PASS
TokenizedGitIgnoreExit=0
```

### Full dry-run

Run ID：

```text
day09-full-one-update-20260813-050032
```

计划：

```text
ModelParameters=33833984
MicroBatchSize=16
GradientAccumulationSteps=8
TokensPerUpdate=65536
TotalUpdates=4578
WarmupUpdates=92
PlannedTokens=300023808
```

Dry-run 确认 Full manifest、Tokenizer、fingerprint、commit、BF16、`(16,512)` 与 causal shift；没有写 run/checkpoint 目录。

### 唯一一次 optimizer update

只执行：

```text
--stop-at-step 1
```

结果：

```text
GlobalStep=1
TokensSeen=65536
MicroSteps=8
Samples=128
TrainLoss=9.816444397
LearningRate=3.26086956522e-06
GradNorm=7.308704853
TokensPerSecond=58906.051
Evaluations=0
Checkpoints=1
```

GPU 1 秒采样：

```text
Samples=5
PeakMemoryMiB=5783
PeakUtilizationPercent=31
PeakTemperatureC=35
PostRunMemoryMiB=0
```

Checkpoint：

```text
Path=checkpoints/day09-full-one-update-20260813-050032/step-00000001.pt
Bytes=406108827
SHA256=457d12600f143b400e2ab51af549d0bd020badafb5c32c38bb9502a0a254e7e4
```

CPU `weights_only=True` 核验通过，metrics event order 为：

```text
run_start -> train_update -> checkpoint
```

TrainerState 为 step 1 / 8 micro steps / 65,536 tokens / 128 samples / last save step 1 / last eval step 0。没有第二次 update。

### 证据下载

远端证据包：

```text
Archive=small-gpt-day09-evidence-20260813-051212.tar.gz
Bytes=63808
SHA256=0edd992562f64b1bdc156fd5bbb498f087b8de17097b974720b213075058e3a8
Entries=66
```

本地三方 hash、bytes 和 required entries 通过：

```text
ExpectedHash=0edd992562f64b1bdc156fd5bbb498f087b8de17097b974720b213075058e3a8
ActualHash=0edd992562f64b1bdc156fd5bbb498f087b8de17097b974720b213075058e3a8
SidecarHash=0edd992562f64b1bdc156fd5bbb498f087b8de17097b974720b213075058e3a8
ActualBytes=63808
ArchiveEntries=66
Day9EvidenceDownloadGate=PASS
```

证据位于仓库外：

```text
D:\code\small-gpt-day09-evidence
```

### Day 9 最终状态

- [x] Full source profile、显式源文件和 local cache 实现；
- [x] metadata 跨平台身份修复；
- [x] 2.15GB 源文件 bytes/SHA 验证；
- [x] 350,000,812 provided-token corpus；
- [x] 70 source shard groups canonical validation；
- [x] source stream lifecycle 修复与真实 active probe；
- [x] 379,587,945-token Tokenized Full；
- [x] 77 storage shards 完整验证；
- [x] 跨 shard `T+1` 与 workers 2/4；
- [x] Full BF16 dry-run；
- [x] 恰好一次 optimizer update；
- [x] 严格 checkpoint/identity 验证；
- [x] 本地/远端 551 项完整回归；
- [x] 证据下载与三方 SHA；
- [x] 正式 300M-token 训练未启动。

### 下一阶段硬门

1. 本地完成 Day 9 README、daily log 和执行报告提交；
2. 普通 push 后核对 `HEAD == origin/main == RemoteMain`；
3. 证据保全和 Git 闭环完成后关闭 C14，但不释放；
4. 正式预训练必须创建新 run ID；
5. 重新验证 Full/tokenizer/Git/GPU identity；
6. 获得新的 300M-token 长跑授权；
7. 再启动 4,578-update 正式训练。
