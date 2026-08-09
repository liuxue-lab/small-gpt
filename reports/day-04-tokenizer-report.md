# Small GPT from Scratch：Day 4 Tokenizer 执行报告

> 项目：从零实现并预训练约 34M 参数的 Decoder-only GPT
>
> 阶段：Day 4——训练并验证项目专属 ByteLevel BPE Tokenizer
>
> 执行位置：Windows 本地 `D:\code\small-gpt`
>
> 输入语料：FineWeb-Edu 2M Pilot
>
> 最终状态：通过

## 1. 执行结论

Day 4 已完成从 Tokenizer 技术契约、实现、离线测试到真实训练和全 Pilot 评估的闭环：

- 使用 `tokenizers==0.23.1` 从 Pilot 的 **train split** 训练了一个 16,384 词表的 ByteLevel BPE Tokenizer；
- validation 和 test 数据未参与 BPE 训练；
- `<bos>`、`<eos>`、`<pad>`、`<unk>` 的 ID 分别固定为 `0`、`1`、`2`、`3`；
- 训练读取 4 个 train shard，共 1,824 篇文档和 1,967,041 个 provided tokens；
- Tokenizer 保存后重新加载验证通过；
- 对 train、validation、test 三个 split 共 1,852 篇文档完成真实编码统计；
- 全语料产生 2,129,776 个模型 token，其中包含 1,852 个文档结束 `<eos>`；
- 普通语料中的未知 token 数量为 `0`；
- 完整项目回归为 `66 passed in 44.71s`；
- Day 4 不生成正式训练用 `.bin/.idx`，该工作留给后续 Dataset/DataLoader 阶段。

## 2. 数据身份与训练边界

| 项目 | 实际值 |
| --- | --- |
| 数据集 | `HuggingFaceFW/fineweb-edu` |
| 配置 | `sample-10BT` |
| Hugging Face revision | `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9` |
| Pilot manifest | `data/processed/fineweb_edu_corpus/manifest.json` |
| Manifest SHA-256 | `972c3c3e1733a3535aabb67b3153fc881d0c46f76849a3ff8e522ad18377a8ce` |
| Tokenizer 训练 split | `train` |
| Train shard 数 | 4 |
| Train 文档数 | 1,824 |
| Train provided tokens | 1,967,041 |
| Validation 是否参与训练 | 否 |
| Test 是否参与训练 | 否 |

训练文件按 shard ID 升序发现，每个 JSONL 文件按原始行顺序流式读取。训练过程不把全部语料加载到 Python 列表，也不创建拼接后的大型临时文本文件。

## 3. Tokenizer 技术契约

| 配置项 | 冻结值 |
| --- | --- |
| 实现库 | `tokenizers==0.23.1` |
| 模型 | ByteLevel BPE |
| 词表大小 | 16,384 |
| `min_frequency` | 2 |
| `max_token_length` | 64 |
| BPE dropout | `None` |
| Normalizer | NFC |
| Pre-tokenizer | `ByteLevel(add_prefix_space=False, use_regex=True)` |
| 初始字母表 | `ByteLevel.alphabet()` |
| Decoder | ByteLevel |
| `byte_fallback` | `False` |
| 自动 PostProcessor | 不使用 |
| 预训练文档边界 | 每篇有效文档末尾追加一个 `<eos>` |
| BOS 策略 | 预训练语料不自动添加 `<bos>` |

### 3.1 特殊 token

| 名称 | 字符串 | ID | 用途 |
| --- | --- | ---: | --- |
| BOS | `<bos>` | 0 | 显式序列起点或后续生成 |
| EOS | `<eos>` | 1 | 每篇预训练文档结束标记 |
| PAD | `<pad>` | 2 | 后续批处理 padding |
| UNK | `<unk>` | 3 | 未知 token；真实评估中为零 |

这些 ID 已进入 Tokenizer 产物契约。后续 tokenized 数据、模型配置和 checkpoint 必须保持一致，不能随意重排。

## 4. 实现与交付文件

| 文件 | 作用 |
| --- | --- |
| `requirements.txt` | 固定 `tokenizers==0.23.1` |
| `configs/tokenizer.yaml` | 保存训练、输入、特殊 token 和评估配置 |
| `tokenizer/bpe.py` | 配置校验、输入发现、训练、加载、编码、保存和统计核心逻辑 |
| `tokenizer/__init__.py` | 暴露稳定的 Tokenizer 公共接口 |
| `scripts/train_tokenizer.py` | 训练真实 16K ByteLevel BPE 并验证保存/重载 |
| `scripts/evaluate_tokenizer.py` | 流式编码三个 split 并生成统计报告 |
| `tests/test_tokenizer.py` | 约束配置、数据边界、特殊 ID、EOS、Unicode、保存和评估行为 |
| `reports/day-04-tokenizer-stats.json` | 机器可读的真实 Tokenizer 统计 |

## 5. 真实训练结果

执行命令：

```powershell
python .\scripts\train_tokenizer.py --config .\configs\tokenizer.yaml
```

训练结果：

| 指标 | 结果 |
| --- | ---: |
| 训练文档数 | 1,824 |
| Train provided tokens | 1,967,041 |
| 最终词表大小 | 16,384 |
| Compute merges | 16,124 |
| 训练耗时 | 1.789 秒 |
| 保存/重载验证 | 通过 |

### 5.1 产物与哈希

| 产物 | 大小（bytes） | SHA-256 |
| --- | ---: | --- |
| `tokenizer.json` | 1,137,073 | `b26835e02eebf777a257c4732abdd6f9732a115967d2ad839f3a1a00e45ee8c5` |
| `vocab.json` | 248,219 | `0032f0364595f3b23d339bcb80027e5cabd13b3d8b52add066b83914f02f8f2e` |
| `merges.txt` | 143,681 | `825232ee99b639053f05485f7e8fa1692468fe3a94de7b1a0abb3b0b4e2413bc` |
| `tokenizer_config.json` | 2,988 | `8622711407aab3f299996b7d3009d4f4447ae35879ca8e50451b5f0adbdf5141` |

四个产物总大小为 1,531,961 bytes，可以安全纳入 Git。运行时以自包含的 `tokenizer.json` 为首选入口，`vocab.json` 与 `merges.txt` 用于审计，`tokenizer_config.json` 保存项目契约和数据身份。

## 6. 全 Pilot 真实编码统计

执行命令：

```powershell
python .\scripts\evaluate_tokenizer.py --config .\configs\tokenizer.yaml
```

### 6.1 分 split 结果

| Split | 文档 | Provided tokens | BPE（不含 EOS） | EOS | 模型 tokens | 模型/Provided | `<unk>` | 超过 512 tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Train | 1,824 | 1,967,041 | 2,091,417 | 1,824 | 2,093,241 | 1.064157 | 0 | 1,107（60.69%） |
| Validation | 16 | 12,600 | 13,917 | 16 | 13,933 | 1.105794 | 0 | 8（50.00%） |
| Test | 12 | 20,442 | 22,590 | 12 | 22,602 | 1.105665 | 0 | 7（58.33%） |
| **合计** | **1,852** | **2,000,083** | **2,127,924** | **1,852** | **2,129,776** | **1.064844** | **0** | **1,122（60.58%）** |

关键守恒关系全部成立：

```text
records = eos_tokens = 1852
2127924 raw BPE tokens + 1852 EOS tokens = 2129776 model tokens
unknown_tokens = 0
```

### 6.2 编码效率

| 指标 | 全 Pilot 结果 |
| --- | ---: |
| UTF-8 bytes | 9,254,127 |
| Unicode characters | 9,222,533 |
| Bytes / BPE token | 4.348899 |
| Characters / BPE token | 4.334052 |
| BPE / provided-token ratio（不含 EOS） | 1.063918 |
| Model / provided-token ratio（含 EOS） | 1.064844 |

数据集自带的 `provided_token_count` 不是本项目 Tokenizer 的精确 token 数。全 Pilot 中，本项目模型 token 数比 provided tokens 高约 **6.48%**。后续训练预算、学习率调度和 checkpoint 间隔应以实际模型 token 数为准。

## 7. 上下文长度分析

Baseline 模型的上下文长度为 512，而 1,852 篇文档中有 1,122 篇超过 512 个模型 token，占 60.58%。Train split 的单文档 token 分布如下：

| 指标 | Train 文档 tokens |
| --- | ---: |
| Minimum | 69 |
| P50 | 646.5 |
| Mean | 1,147.61 |
| P95 | 3,409.25 |
| P99 | 9,480.89 |
| Maximum | 27,029 |

这不是 Tokenizer 失败，而是后续数据加载设计的重要输入。正式构建训练序列时需要：

- 保留每篇文档末尾的 `<eos>`；
- 将连续 token 流切分为长度为 512 的训练块；
- 明确跨文档拼接、剩余 token 和 padding 策略；
- 避免把超过 512 token 的文档直接整体截断，从而浪费大部分语料。

## 8. 350M Full 规模推算

Pilot train split 的实际比例为：

```text
model tokens / provided tokens = 1.0641572798940133
```

按照 Full provided-token 目标 350,000,000、train split 比例 0.98 推算：

```text
350,000,000 × 0.98 × 1.0641572798940133
= 365,005,947 个 train model tokens
```

该值只是依据 Pilot 比例得到的估计，不是 Full 的真实编码结果。Full 语料的文本分布可能使最终比例发生变化，因此正式训练前仍需以实际 tokenization 统计为准。Day 4 没有启动 350M Full 采集。

## 9. 测试与安全检查

### 9.1 Tokenizer 专项测试

```text
37 passed in 0.23s
```

专项测试覆盖：

- 配置与版本契约；
- 只发现 train split，防止 validation/test 泄漏；
- JSONL schema、空文本和损坏输入处理；
- NFC、ByteLevel 和 Unicode round-trip；
- 特殊 token 固定 ID；
- 每篇文档恰好追加一个 EOS；
- 保存、重载、拒绝静默覆盖和半成品保护；
- 真实统计结构和原子 JSON 写入；
- 大型数据目录 Git 忽略规则。

### 9.2 完整项目回归

```text
66 passed in 44.71s
```

完整回归包含 Day 1～Day 3 的原有 29 项测试和 Day 4 新增的 37 项测试，没有 `failed` 或 `error`。

### 9.3 Git 与数据安全

- `git diff --check` 通过；
- Git 提示 `tokenizer/__init__.py` 将来可能由 LF 转换为 CRLF，这是 Windows 行尾提示，不是空白检查失败；
- `data/processed/fineweb_edu_corpus/manifest.json` 命中 `.gitignore` 中的 `data/processed/` 规则；
- Pilot JSONL、未来 tokenized binary、临时目录和缓存不进入 Git；
- Tokenizer 正式产物体积较小，可以提交。

## 10. 验收结果

| 验收项 | 结果 |
| --- | --- |
| 固定依赖与 Tokenizer 配置 | 通过 |
| 只使用 Pilot train split 训练 | 通过 |
| 词表大小为 16,384 | 通过 |
| 四个特殊 token ID 固定 | 通过 |
| 保存后重新加载一致 | 通过 |
| 三个 split 统计完整 | 通过 |
| EOS 数量与文档数相等 | 通过 |
| 普通文本 `<unk>` 为零 | 通过 |
| Token ID 可由 `uint16` 表示 | 通过 |
| 未生成 `.bin/.idx` | 符合 Day 4 范围 |
| Tokenizer 专项测试 | 37 passed |
| 完整项目回归 | 66 passed |
| Pilot 数据仍被 Git 忽略 | 通过 |

## 11. 后续阶段建议

下一阶段应围绕正式训练数据格式和 Dataset/DataLoader 建立闭环：

1. 冻结 `uint16` tokenized binary 与索引格式；
2. 使用当前权威 `tokenizer.json` 编码语料；
3. 正确保留 `<eos>` 文档边界并按 512 token 构造训练序列；
4. 实现 memory-map 读取、确定性划分和批次抽样；
5. 为跨文档拼接、末尾残片、恢复和数据守恒编写测试；
6. 在开始正式预训练前再次确认模型配置中的 `vocab_size=16384` 和特殊 token ID；
7. 一旦生成 tokenized 数据或 checkpoint，不再重排词表或特殊 token ID。

Day 4 到此已经完成技术验收；只有在报告、README、开发日志、Git 提交和远端推送全部完成后，项目进度才正式记为 100%。
