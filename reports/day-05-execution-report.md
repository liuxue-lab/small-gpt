# Small GPT from Scratch：Day 5 Tokenized Data、Dataset 与 DataLoader 执行报告

> 项目：从零实现并预训练约 34M 参数的 Decoder-only GPT
>
> 阶段：Day 5——构建训练二进制、Dataset 与 DataLoader
>
> 执行日期：2026-08-10 ～ 2026-08-11
>
> 执行位置：Windows 本地 `D:\code\small-gpt`
>
> 输入语料：FineWeb-Edu 2M Pilot
>
> 技术验收状态：通过

## 1. 执行结论

Day 5 已完成从格式设计、实现、自动测试到真实 Pilot 编码和 DataLoader 检查的完整闭环：

- 冻结了 schema version 1 的 token `.bin` 与文档 `.idx` 格式；
- 使用 Day 4 权威 `tokenizer.json` 编码 train、validation、test 共 1,852 篇文档；
- 生成 2,129,776 个模型 token，与 Day 4 的真实编码统计完全一致；
- token payload 使用 little-endian `uint16`，共 4,259,552 bytes；
- 生成 5 个 storage shards：train 3 个、validation 1 个、test 1 个；
- 每篇文档恰好追加一个 `<eos>`，没有主动插入 BOS 或 PAD；
- 实现了 shard 边界恢复、完整性校验、完成态 no-op 和原子发布；
- 实现了跨 storage shard 的 memory-map 逻辑 token 流；
- 实现了因果 `T+1` Dataset、可复现 train sampler 和顺序 evaluation windows；
- 在真实数据上验证了 `T=128`、`T=512`、两个 train shard 边界和 Windows 多 worker DataLoader；
- Day 5 新增 27 项测试全部通过，完整项目回归为 `93 passed in 8.65s`；
- tokenized binary 继续被 Git 忽略，只提交配置、代码、测试、设计和机器可读统计。

Day 5 没有实现 GPT 模型、没有启动 350M Full 语料、没有启动正式预训练，也没有使用 AutoDL。

## 2. 阶段目标与边界

Day 4 已经回答“文本如何转换为项目 token”，Day 5 需要回答“这些 token 如何可靠地保存，并以因果语言建模需要的形状送入 PyTorch”。

本阶段的核心不变量为：

1. 存储层无损：不按 128 或 512 上下文截断文档；
2. 文档边界明确：每篇文档恰好以一个 `<eos>` 结束；
3. split 完全隔离：任何窗口都不能从一个 split 读到另一个 split；
4. storage shard 只是物理边界：split 内的 Dataset 可以跨 shard 连续读取；
5. 训练标签严格右移一位：每个样本需要读取 `T+1` 个 token；
6. 运行可恢复：只在完整 shard 边界恢复，不信任半成品；
7. 输出可审计：header、大小、SHA-256、索引和统计必须互相一致；
8. 大型生成数据不进入 Git。

## 3. 冻结的上游身份

### 3.1 数据身份

| 项目 | 冻结值 |
| --- | --- |
| Dataset | `HuggingFaceFW/fineweb-edu` |
| Configuration | `sample-10BT` |
| Split | `train`（上游原始 split） |
| Revision | `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9` |
| Pilot corpus | `data/processed/fineweb_edu_corpus` |
| Source manifest SHA-256 | `972c3c3e1733a3535aabb67b3153fc881d0c46f76849a3ff8e522ad18377a8ce` |
| Source shard groups | 4 |
| 确定性读取顺序 | source shard ID → split → JSONL line |

### 3.2 Tokenizer 身份

| 项目 | 冻结值 |
| --- | --- |
| Tokenizer | 16,384 词表 ByteLevel BPE |
| 实现库 | `tokenizers==0.23.1` |
| Runtime file | `tokenizer/artifacts/tokenizer.json` |
| Tokenizer SHA-256 | `b26835e02eebf777a257c4732abdd6f9732a115967d2ad839f3a1a00e45ee8c5` |
| Normalizer | NFC |
| BOS | `<bos>=0` |
| EOS | `<eos>=1` |
| PAD | `<pad>=2` |
| UNK | `<unk>=3` |

编码启动前会验证 source manifest、Tokenizer、词表大小和特殊 ID。身份不一致时拒绝继续，避免静默生成不可复现的数据。

## 4. 冻结的二进制契约

配置文件：

```text
configs/tokenized_data.yaml
```

设计文档：

```text
reports/day-05-tokenized-data-design.md
```

### 4.1 Token `.bin`

| 字段 | 冻结值 |
| --- | --- |
| Magic | `SGPTTOK1` |
| Schema version | 1 |
| Header size | 64 bytes |
| Struct | `<8sHHBBHIB3xQQQIII4x` |
| Payload dtype | little-endian `uint16`（`<u2`） |
| Dtype code | 1 |
| Endian code | 1 |
| Token width | 2 bytes |
| Token ID range | `0..16383` |

选择 `uint16` 的依据是词表大小为 16,384，最大合法 ID 为 16,383，远小于 `uint16` 上限 65,535。磁盘字节序显式固定为 little-endian，不能依赖运行机器的原生字节序。

### 4.2 Document `.idx`

| 字段 | 冻结值 |
| --- | --- |
| Magic | `SGPTIDX1` |
| Schema version | 1 |
| Header size | 128 bytes |
| Header struct | `<8sHHHBBQQQQ32sQI36x` |
| Record size | 48 bytes |
| Record struct | `<QQ32s` |
| Offset | shard-local token offset，`uint64` |
| Length | 含末尾 EOS，`uint64` |
| Identity | 原始规范化文本 SHA-256，32 raw bytes |

索引记录必须满足无空洞、无重叠、单调递增：

```text
record[0].offset = 0
record[i + 1].offset = record[i].offset + record[i].length
last.offset + last.length = shard token count
```

每条文档索引指向的最后一个 token 必须为 EOS ID `1`。

### 4.3 分片与 split

| 项目 | 策略 |
| --- | --- |
| Target | 每个 storage shard 约 1,000,000 model tokens |
| 文档处理 | 文档原子；不会为了命中 target 拆分文档 |
| 超出 target | 当前文档写完后结束 shard |
| Split order | train → validation → test |
| 跨文档窗口 | 允许，以 EOS 保留边界 |
| 跨 storage shard 窗口 | 允许 |
| 跨 split 窗口 | 禁止 |

## 5. 实现结构

| 文件 | 作用 |
| --- | --- |
| `data_pipeline/binary_format.py` | header/record 打包解析、常量、文件和索引校验 |
| `data_pipeline/tokenization.py` | 输入发现、身份校验、编码、分片、状态、恢复、manifest 和发布 |
| `data_pipeline/dataset.py` | lazy memmap、跨 shard slice、因果 Dataset、sampler 和 DataLoader |
| `data_pipeline/__init__.py` | 对外暴露稳定的数据管线接口 |
| `scripts/tokenize_corpus.py` | Pilot tokenization 命令行入口 |
| `scripts/inspect_tokenized_data.py` | 完成态校验、样本检查和 DataLoader 检查入口 |
| `tests/test_binary_format.py` | 格式、header、index 和损坏检测测试 |
| `tests/test_tokenization.py` | 编码、分片、恢复、幂等和 manifest 测试 |
| `tests/test_dataset.py` | 跨 shard、因果窗口、sampler 和 DataLoader 测试 |

## 6. 原子发布、恢复与完整性

默认路径：

```text
完成态：data/tokenized/fineweb_edu_pilot
暂存态：data/tokenized/.fineweb_edu_pilot.inprogress
```

写入流程为：

1. 验证配置、source manifest 与 Tokenizer 身份；
2. 在 staging 目录写 `.part` 文件；
3. 完成一个 token/index shard；
4. flush、fsync，并计算大小与 SHA-256；
5. 验证 header、payload、index、文档 EOS 和统计；
6. 原子发布 shard，并更新 `state.json`；
7. 所有 split 完成后生成并验证 `manifest.json`；
8. 将 staging 目录原子重命名为最终输出目录。

恢复只接受身份一致、已经完整发布并通过校验的 shard。半成品 `.part` 不会被当作完成数据。最终目录已存在时：

- 身份和文件全部匹配：作为完成态 no-op；
- 身份不匹配或文件损坏：拒绝静默覆盖。

自动测试覆盖了中断后恢复、已完成 shard 哈希保持、one-shot 与 resume 结果一致、完成态幂等和损坏检测。真实 Pilot 使用正常 one-shot 路径完成发布。

## 7. 真实 Pilot 编码

执行命令：

```powershell
python .\scripts\tokenize_corpus.py --config .\configs\tokenized_data.yaml --profile pilot
```

### 7.1 分 split 结果

| Split | 文档 | Raw BPE | EOS | 模型 tokens | Storage shards |
| --- | ---: | ---: | ---: | ---: | ---: |
| Train | 1,824 | 2,091,417 | 1,824 | 2,093,241 | 3 |
| Validation | 16 | 13,917 | 16 | 13,933 | 1 |
| Test | 12 | 22,590 | 12 | 22,602 | 1 |
| **合计** | **1,852** | **2,127,924** | **1,852** | **2,129,776** | **5** |

### 7.2 Storage shard 结果

| Split | Shard | 文档 | Tokens |
| --- | --- | ---: | ---: |
| Train | `shard-00000` | 923 | 989,147 |
| Train | `shard-00001` | 816 | 998,822 |
| Train | `shard-00002` | 85 | 105,272 |
| Validation | `shard-00000` | 16 | 13,933 |
| Test | `shard-00000` | 12 | 22,602 |

Train 的前两个 shard 未精确达到 1,000,000，是因为分片策略不会拆分当前文档。这是文档原子规则的预期结果，不是 token 丢失。

### 7.3 守恒关系

全部真实统计满足：

```text
records = appended EOS = 1,852
2,127,924 raw BPE + 1,852 EOS = 2,129,776 model tokens
2,129,776 tokens × 2 bytes = 4,259,552 payload bytes
train + validation + test = totals
```

Day 5 的分 split 文档数和模型 token 数与 Day 4 完全一致。这证明存储写入没有截断长文档、丢失 EOS 或改变 Tokenizer 结果。

完成态 manifest fingerprint：

```text
a3eb6012c1cb3e2dab2a7839bebb04530563b19b4d5f7d8022e3c121b13ca7f3
```

机器可读统计保存在：

```text
reports/day-05-tokenized-data-stats.json
```

该报告是完成态 manifest 的可追踪快照；真实 `.bin/.idx` 与 `data/tokenized/` 下的运行 manifest 不进入 Git。

## 8. Dataset 语义

### 8.1 SplitTokenStore

`SplitTokenStore` 将同一 split 的多个物理 `.bin` 文件呈现为单个只读逻辑 token 流。它只在首次访问某个进程时创建 NumPy memmap，不把整个 split 复制到内存。

一次 slice 可以跨越 storage shard 边界，但 store 的实例只属于一个 split，因此不能跨越 train、validation、test。

### 8.2 因果 `T+1` 样本

对于起点 `s` 和上下文长度 `T`：

```text
window = tokens[s : s + T + 1]
x = window[:-1]
y = window[1:]
```

因此：

```text
x.shape = (T,)
y.shape = (T,)
y[:-1] == x[1:]
dtype = torch.int64
```

`uint16` 只用于节省磁盘空间；送入 embedding 前转换为 PyTorch `long`。

### 8.3 Train 与 evaluation 模式

| Split | Dataset 模式 | 行为 |
| --- | --- | --- |
| Train | `all_starts` | 每个合法 token 起点都可形成窗口 |
| Train sampler | epoch random with replacement | `base_seed + epoch` 可复现 |
| Validation | sequential non-overlapping | 从 0 开始，固定顺序 |
| Test | sequential non-overlapping | 从 0 开始，固定顺序 |

合法窗口数：

```text
all_starts = N - T
sequential = floor((N - 1) / T)
remainder = (N - 1) mod T
```

这里必须使用 `N - 1`，因为每个长度为 `T` 的输入还需要右侧一个 label token。

## 9. 真实 Dataset/DataLoader 验证

### 9.1 `T=128` 与 `T=512`

| Context `T` | Train `all_starts` | Validation sequential | Validation remainder | Test sequential | Test remainder |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 2,093,113 | 108 | 108 | 176 | 73 |
| 512 | 2,092,729 | 27 | 108 | 44 | 73 |

Train split 的顺序模式额外验证为：

| Context `T` | Sequential windows | Remainder |
| ---: | ---: | ---: |
| 128 | 16,353 | 56 |
| 512 | 4,088 | 184 |

所有抽样均满足：

- `x.shape == y.shape == (T,)`；
- `dtype == torch.int64`；
- token ID 位于 `[0, 16383]`；
- `shift=True`。

### 9.2 真实 storage shard 边界

Train 的累计物理边界为：

```text
boundary 1 = 989,147
boundary 2 = 989,147 + 998,822 = 1,987,969
```

验证结果：

| Boundary | Start | Shape | Shift |
| ---: | ---: | --- | --- |
| 989,147 | 988,891 | `(512,)` | `True` |
| 1,987,969 | 1,987,713 | `(512,)` | `True` |

两个起点均位于边界前 256 token，读取 `T+1=513` 个 token 时必然跨越物理 shard。这一验证直接证明逻辑 slice 没有在 shard 边界中断或产生 off-by-one。

### 9.3 Windows 多 worker

运行配置：

```text
context_length = 512
samples = 8
batch_size = 4
num_workers = 2
```

train、validation、test 三个 split 均成功构造：

```text
x=(4, 512), y=(4, 512), dtype=torch.int64, workers=2, shift=True
```

这验证了 Windows `spawn` worker 模式下 Dataset 对象可序列化、子进程能够延迟打开 memmap，并且 batch 的 shape、dtype 和因果右移关系正确。

## 10. 测试结果

### 10.1 Day 5 专项测试

| 测试文件 | 结果 |
| --- | ---: |
| `tests/test_binary_format.py` | 13 passed |
| `tests/test_tokenization.py` | 6 passed |
| `tests/test_dataset.py` | 8 passed |
| **合计** | **27 passed** |

覆盖范围包括：

- header 字节数、magic、schema、字节序和 reserved bytes；
- token 文件大小、payload hash 和截断检测；
- index record 大小、连续 offset、越界和 gap 检测；
- 每文档单 EOS、无 writer BOS/PAD、ID 范围；
- 文档原子 sharding 与精确统计；
- 中断恢复、完成态 no-op、输出身份和拒绝覆盖；
- one-shot 与 resume 确定性一致；
- lazy memmap 与跨 storage shard slice；
- split 隔离、`T+1` 因果 shift 和 remainder；
- train seed/epoch 可复现；
- validation/test 顺序稳定；
- DataLoader worker 行为。

### 10.2 完整项目回归

```text
93 passed in 8.65s
```

完整回归包含 Day 1～Day 4 的 66 项原有测试与 Day 5 的 27 项新增测试，没有 `failed`、`error` 或回退。

## 11. Git 与数据安全

真实 tokenized 输出位于：

```text
data/tokenized/fineweb_edu_pilot/
```

以下文件通过 `.gitignore` 中的 `data/tokenized/` 规则排除：

- 完成态 `manifest.json`；
- token `.bin`；
- document `.idx`；
- staging、state 和运行时中间产物。

Day 5 进入 Git 的内容只包括：

- 格式与 Dataset 配置；
- 设计文档；
- Python 实现；
- 自动测试；
- 机器可读 manifest 快照；
- 执行报告、开发日志和 README。

这样既避免把可再生成的大型数据提交到 Git，也保留了 source/Tokenizer 身份、每个 shard 的统计和 SHA-256 审计证据。

## 12. 验收结果

| 验收项 | 结果 |
| --- | --- |
| Source manifest 和 Tokenizer 身份固定 | 通过 |
| Token 与 index schema 冻结 | 通过 |
| little-endian `uint16` payload | 通过 |
| 每文档恰好一个 EOS | 通过 |
| 不添加 BOS/PAD | 通过 |
| Pilot 文档和 token 总量守恒 | 通过 |
| 文档原子分片 | 通过 |
| Resume、no-op 与原子发布 | 自动测试通过 |
| 大小、hash、index 与 EOS 校验 | 通过 |
| Split 完全隔离 | 通过 |
| 跨 storage shard slice | 通过 |
| 因果 `T+1` 与 shift | 通过 |
| 两个真实 train shard 边界 | 通过 |
| Windows `num_workers=2` | 通过 |
| Day 5 专项测试 | 27 passed |
| 完整项目回归 | 93 passed |
| Tokenized data 被 Git 忽略 | 通过 |
| Full/AutoDL 保持未启动 | 符合范围 |

## 13. 下一阶段

Day 6 可以进入 Decoder-only GPT 的最小正确实现：

1. 实现 token embedding 和位置 embedding；
2. 实现 multi-head causal self-attention；
3. 实现 MLP、LayerNorm、残差连接和 Transformer Block；
4. 组装 GPT 与 tied language-model head；
5. 固定权重初始化和因果 mask 行为；
6. 验证所有 tensor shape、参数量和梯度；
7. 使用 Debug 配置完成单批次前向、反向和小样本过拟合；
8. 在训练循环稳定前，不启动 350M Full 数据与正式 GPU 预训练。

Day 5 到此完成技术验收。精确暂存、语义提交和普通 push 由仓库历史记录最终交付状态；禁止提交 `data/tokenized/`，禁止使用 `git add .`、`git add -f` 或 force push。
