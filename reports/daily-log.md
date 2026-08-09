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
