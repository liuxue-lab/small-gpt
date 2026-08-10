# Day 5 Tokenized Data、Dataset 与 DataLoader 设计

> 项目：Small GPT from Scratch
>
> 阶段：Day 5
>
> 状态：设计冻结候选；必须在配置解析、实现和真实 Pilot 验收后标记完成
>
> 执行环境：Windows 本地 `D:\code\small-gpt`

## 1. 设计目标

Day 5 将 Day 4 已冻结的文本到 token ID 层，扩展为可供 Decoder-only GPT 使用的训练数据层：

```text
tokenizer/artifacts/tokenizer.json
→ 按 split 流式 encode_document
→ 每篇追加一个 EOS
→ little-endian uint16 token shards
→ 固定宽度 document indexes
→ deterministic manifest
→ shard-level resume 与原子发布
→ split-level memory-mapped token store
→ T+1 next-token Dataset
→ 可复现 DataLoader
```

本阶段不重新训练或评估 Tokenizer，不启动 350M Full，不实现 GPT，不启动 AutoDL。

## 2. 已冻结的上游身份

### 2.1 数据身份

| 项目 | 值 |
| --- | --- |
| Dataset | `HuggingFaceFW/fineweb-edu` |
| Configuration | `sample-10BT` |
| Revision | `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9` |
| Pilot manifest | `data/processed/fineweb_edu_corpus/manifest.json` |
| Manifest SHA-256 | `972c3c3e1733a3535aabb67b3153fc881d0c46f76849a3ff8e522ad18377a8ce` |
| Source shard groups | 4 |
| Records | 1,852 |
| Provided tokens | 2,000,083 |

### 2.2 Tokenizer 身份

| 项目 | 值 |
| --- | --- |
| Runtime artifact | `tokenizer/artifacts/tokenizer.json` |
| Tokenizer SHA-256 | `b26835e02eebf777a257c4732abdd6f9732a115967d2ad839f3a1a00e45ee8c5` |
| Metadata | `tokenizer/artifacts/tokenizer_config.json` |
| Metadata SHA-256 | `8622711407aab3f299996b7d3009d4f4447ae35879ca8e50451b5f0adbdf5141` |
| Library | `tokenizers==0.23.1` |
| Type | ByteLevel BPE |
| Normalizer | NFC |
| Vocab size | 16,384 |
| BOS | `<bos>=0` |
| EOS | `<eos>=1` |
| PAD | `<pad>=2` |
| UNK | `<unk>=3` |

Day 5 必须直接加载正式 `tokenizer.json`，不能从 `vocab.json` 和 `merges.txt` 重新拼装另一套运行时 Tokenizer。

## 3. 范围与不变量

### 3.1 存储层必须无损

对每个 split：

```text
records == index entries
records == appended EOS tokens
raw BPE tokens + appended EOS tokens == model tokens
sum(output shard tokens) == split model tokens
sum(output shard payload bytes) == split model tokens * 2
storage dropped tokens == 0
unknown tokens == 0
```

全局：

```text
1824 + 16 + 12 = 1852 records
2127924 raw BPE + 1852 appended EOS = 2129776 model tokens
2129776 * 2 = 4259552 token payload bytes
```

### 3.2 Split 完全隔离

- train 只能来自 train JSONL；
- validation 只能来自 validation JSONL；
- test 只能来自 test JSONL；
- 同一 split 的 Dataset 可以跨物理 token shard；
- Dataset 不能跨 train、validation、test；
- 不能用下一个 split 给当前 split 凑满 `T+1`。

### 3.3 文档边界

每篇文档：

```python
document_ids = encode_document(tokenizer, text, eos_id=1)
```

约束：

- 末尾追加且只追加一个 EOS；
- writer 不主动插入 BOS；
- writer 不主动插入 PAD；
- 不对长文档整体截断；
- storage shard 只能在文档之间切换；
- 训练窗口允许跨文档，但必须看到中间的 EOS。

“writer 不主动插入特殊 token”与“正文恰好包含特殊 token 字面字符串”必须区分。正文中的真实编码不得被管线擅自过滤。

## 4. 输出布局与 sharding

正式 Pilot 输出：

```text
data/tokenized/fineweb_edu_pilot/
├── manifest.json
├── train/
│   ├── shard-00000.bin
│   ├── shard-00000.idx
│   └── ...
├── validation/
│   ├── shard-00000.bin
│   └── shard-00000.idx
└── test/
    ├── shard-00000.bin
    └── shard-00000.idx
```

构建中使用：

```text
data/tokenized/.fineweb_edu_pilot.inprogress/
```

Pilot target 为每个 storage shard 1,000,000 model tokens。

分片规则：

1. 文档保持原子性；
2. 当前 shard 非空，且加入下一文档会超过 target 时，先完成当前 shard；
3. 单篇文档超过 target 时，允许独立形成 oversize shard；
4. 不为满足 target 截断文档；
5. split 内文档顺序保持不变；
6. 物理 shard 边界不改变逻辑 token stream。

由于 Pilot train 有 2,093,241 model tokens，并且最长文档小于 1,000,000 tokens，预计生成 3 个 train shards。validation 和 test 各 1 个。

## 5. Token `.bin` 格式

每个 `.bin`：

```text
64-byte header + little-endian uint16 payload
```

Python struct：

```python
struct.Struct("<8sHHBBHIB3xQQQIII4x")
```

必须恰好为 64 bytes。

| Offset | 字段 | 类型 | 语义 |
| ---: | --- | --- | --- |
| 0 | magic | `8s` | `SGPTTOK1` |
| 8 | schema_version | `uint16` | 1 |
| 10 | header_bytes | `uint16` | 64 |
| 12 | dtype_code | `uint8` | 1 = unsigned 16-bit |
| 13 | endian_code | `uint8` | 1 = little-endian |
| 14 | flags | `uint16` | bit0 append EOS；bit1 add BOS；bit2 add PAD |
| 16 | vocab_size | `uint32` | 16,384 |
| 20 | split_code | `uint8` | train=0；validation=1；test=2 |
| 21 | reserved | 3 bytes | 必须为 0 |
| 24 | token_count | `uint64` | payload token 数 |
| 32 | document_count | `uint64` | 完整文档数 |
| 40 | payload_bytes | `uint64` | `token_count * 2` |
| 48 | eos_token_id | `uint32` | 1 |
| 52 | minimum_token_id | `uint32` | 实际最小 ID |
| 56 | maximum_token_id | `uint32` | 实际最大 ID |
| 60 | reserved | 4 bytes | 必须为 0 |

正式 flags 预期为 1：append EOS=true、add BOS=false、add PAD=false。

Payload 从 byte offset 64 开始：

```python
np.memmap(path, mode="r", dtype="<u2", offset=64, shape=(token_count,))
```

文件守恒：

```text
file_size == 64 + token_count * 2
payload_bytes == token_count * 2
0 <= minimum_token_id <= maximum_token_id < 16384
```

Reader 必须拒绝错误 magic/schema/dtype/endian、奇数 payload、截断文件、计数不一致和 token 越界。

## 6. Document `.idx` 格式

每个 `.idx`：

```text
128-byte header + document_count × 48-byte entries
```

Header struct：

```python
struct.Struct("<8sHHHBBQQQQ32sQI36x")
```

必须恰好为 128 bytes。

| Offset | 字段 | 类型 | 语义 |
| ---: | --- | --- | --- |
| 0 | magic | `8s` | `SGPTIDX1` |
| 8 | schema_version | `uint16` | 1 |
| 10 | header_bytes | `uint16` | 128 |
| 12 | record_bytes | `uint16` | 48 |
| 14 | offset_semantics | `uint8` | 1 = shard-local token offset |
| 15 | length_semantics | `uint8` | 1 = length includes EOS |
| 16 | token_count | `uint64` | 对应 `.bin` token 数 |
| 24 | document_count | `uint64` | entry 数 |
| 32 | global_token_start | `uint64` | 在 split 逻辑流中的起点 |
| 40 | global_document_start | `uint64` | split 内第一篇文档序号 |
| 48 | binary_sha256 | 32 bytes | 对应 `.bin` 完整文件 hash |
| 80 | entries_bytes | `uint64` | `document_count * 48` |
| 88 | eos_token_id | `uint32` | 1 |
| 92 | reserved | 36 bytes | 必须为 0 |

Entry struct：

```python
struct.Struct("<QQ32s")
```

| 字段 | 类型 | 语义 |
| --- | --- | --- |
| start_token | `uint64` | shard-local token 起点 |
| length_tokens | `uint64` | 含末尾 EOS 的文档长度 |
| text_sha256 | 32 bytes | 规范化正文的 hash |

索引守恒：

```text
index_size == 128 + document_count * 48
first.start_token == 0
next.start == current.start + current.length
all lengths >= 1
last.start + last.length == token_count
token_at(start + length - 1) == eos_id == 1
```

Index header 中的 binary hash 必须与对应 `.bin` 完全一致。

## 7. Manifest 契约

Dataset 只能从 `manifest.json` 的有序 shard 列表加载，不能使用无约束 glob 猜测数据身份。

顶层至少包含：

```text
schema_version
format_name
status
profile
config_fingerprint
source
tokenizer
encoding
binary_format
index_format
sharding
splits
totals
dataset_contract
```

每个 split 至少记录：

- records；
- provided tokens；
- raw BPE tokens；
- appended EOS tokens；
- total model tokens；
- unknown token occurrences；
- writer 主动插入 BOS/PAD 的数量，必须为 0；
- 实际 special ID 频次，仅用于审计正文；
- minimum/maximum token ID；
- source files；
- output shard 有序列表；
- shard global token/document start；
- `.bin/.idx` 相对路径、大小和 SHA-256；
- storage dropped tokens，必须为 0。

Manifest 不写入绝对路径、用户名、主机名、临时目录、耗时或自动变化时间戳。

`config_fingerprint` 只覆盖会改变数据内容或读取语义的配置。以下运行字段排除：

- output directory；
- staging directory；
- `--resume`；
- progress display。

因此相同 source、Tokenizer 和语义配置在不同输出位置可以生成相同内容身份。

## 8. 原子发布与恢复

### 8.1 Shard 完成顺序

```text
写 .bin.part
→ flush/fsync
→ 回填并验证 token header
→ 计算 .bin hash
→ 回填并验证 .idx.part
→ 计算 .idx hash
→ 原子 rename 两个 part 文件
→ 原子更新 state.json
```

只有 `state.json` 已记录的 shard 才是恢复检查点。

### 8.2 中断

中断后：

- 正式输出目录不得出现；
- staging 可以保留；
- completed shard 不重写；
- `.part` 文件重新构建；
- state 未登记的孤立 shard 不进入最终结果；
- resume 前重新验证 config/source/tokenizer identity；
- completed shard hash 不匹配时拒绝恢复。

### 8.3 完成态

- output complete 且 fingerprint 相同：no-op；
- output 损坏或 fingerprint 不同：拒绝；
- 不提供静默 overwrite；
- 新实验必须使用新输出目录和新身份。

## 9. SplitTokenStore 与 memmap

Train 预计有三个物理 token shards，但 Dataset 必须把它们暴露为一个长度为 2,093,241 的逻辑数组。

`SplitTokenStore`：

- 按 manifest 顺序读取 shard；
- 建立 token count prefix sums；
- 单 shard slice 直接读取 memmap；
- 跨 shard slice 只分配请求长度的临时数组；
- 不插入额外 EOS；
- 不删除 shard 尾部；
- 不越过 split 边界；
- pickle 时不保存已打开的 memmap/file handle；
- Windows worker 中第一次访问时 lazy reopen。

物理 shard 边界不能造成训练 token 丢失。

## 10. CausalWindowDataset

模型 context 为 `T` 时，一个训练样本必须读取 `T+1`：

```python
chunk = store.read(start, context_length + 1)
x = chunk[:-1]
y = chunk[1:]
```

必须满足：

```text
x.shape == y.shape == (T,)
x.dtype == y.dtype == torch.long
y[:-1] == x[1:]
0 <= token_id < 16384
```

合法随机起点数量：

```text
possible_starts = N - T
start range = [0, N-T)
```

Sequential 非重叠窗口：

```text
start_i = i * T
len = floor((N - 1) / T)
remainder = (N - 1) mod T
```

Remainder 只是没有进入该轮非重叠评估，磁盘 token 没有丢失。

### 10.1 Pilot 预期数量

| Split | N | T | Possible starts | Sequential windows | Remainder |
| --- | ---: | ---: | ---: | ---: | ---: |
| train | 2,093,241 | 128 | 2,093,113 | 16,353 | 56 |
| validation | 13,933 | 128 | 13,805 | 108 | 108 |
| test | 22,602 | 128 | 22,474 | 176 | 73 |
| train | 2,093,241 | 512 | 2,092,729 | 4,088 | 184 |
| validation | 13,933 | 512 | 13,421 | 27 | 108 |
| test | 22,602 | 512 | 22,090 | 44 | 73 |

这些值按 split 总 token 数计算，而不是对每个物理 shard 分别取整。

## 11. Train sampler

Train Dataset 使用 `all_starts` 模式：

```text
dataset index == token start
len(dataset) == N - T
```

随机性由 DataLoader 主进程中的 `EpochRandomWindowSampler` 管理：

- 接受 `base_seed`、`epoch` 和 `samples_per_epoch`；
- 使用独立 `torch.Generator`；
- 分块调用 `torch.randint(0, N-T, ...)`；
- 有放回采样；
- 不创建数亿长度的 `randperm`；
- 相同 seed/epoch 产生相同起点序列；
- epoch 改变时序列改变；
- worker 数和完成顺序不改变起点序列；
- DataLoader 显式传 sampler，`shuffle=False`。

Validation/test 使用确定性 sequential Dataset，不随机、不 padding、不跨 split。

## 12. DataLoader 与 Windows

Day 5 先验证：

- `num_workers=0`；
- 小规模 `num_workers=2`；
- `pin_memory=False`；
- train `drop_last=True`；
- validation/test `drop_last=False`。

Windows 要求：

- DataLoader 创建位于 `main()` 路径；
- CLI 使用 `if __name__ == "__main__"`；
- Dataset/Sampler 类定义在模块顶层；
- 不 pickle 已打开的 memmap；
- worker lazy reopen；
- `persistent_workers` 仅用于 workers>0；
- 进程结束后文件句柄可释放。

Day 5 不因为本地 GPU 可用就默认开启 pin memory。训练阶段再做吞吐基准。

## 13. 模块边界

推荐新增：

```text
configs/tokenized_data.yaml
data_pipeline/
├── __init__.py
├── binary_format.py
├── tokenization.py
└── dataset.py
scripts/
├── tokenize_corpus.py
└── inspect_tokenized_data.py
tests/
├── test_binary_format.py
├── test_tokenization.py
└── test_dataset.py
```

职责：

- `binary_format.py`：struct、header/index 读写和校验；
- `tokenization.py`：配置、writer、state、resume、manifest、发布；
- `dataset.py`：memmap store、Dataset、Sampler、DataLoader helper；
- CLI 只负责编排，不复制核心逻辑；
- import 不读真实语料、不访问网络、不写文件。

已有 `train/__init__.py` 为零字节占位，保留给未来 optimizer/training loop。Day 5 数据层使用独立 `data_pipeline/`，避免把数据格式和训练状态管理混在一起。

## 14. 测试边界

新增专项测试必须完全离线，使用临时目录和合成语料：

- token/index header round-trip；
- 错 magic/schema/endian/dtype；
- truncated/odd payload；
- index overlap/gap/out-of-range；
- split isolation；
- EOS 与 token 守恒；
- 长文档不截断；
- 强制多 storage shard；
- 跨 shard `T+1` slice；
- injected interruption 与 resume；
- resume 与一次性构建 hash 一致；
- random sampler seed/epoch 复现；
- sequential 数量公式；
- Windows worker pickle/lazy reopen。

合成专项通过前不运行真实 Pilot。新代码完成前不重复运行原有完整回归。

## 15. 真实 Pilot 验收门

| Split | Records | Provided | Raw BPE | Appended EOS | Model tokens | Unknown |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Train | 1,824 | 1,967,041 | 2,091,417 | 1,824 | 2,093,241 | 0 |
| Validation | 16 | 12,600 | 13,917 | 16 | 13,933 | 0 |
| Test | 12 | 20,442 | 22,590 | 12 | 22,602 | 0 |
| **Total** | **1,852** | **2,000,083** | **2,127,924** | **1,852** | **2,129,776** | **0** |

任何一个数不一致都必须停止验收，不能修改 Day 4 stats、使用容差或重训 Tokenizer 掩盖问题。

## 16. Git 与交付

进入 Git：

- 配置；
- 代码；
- 合成测试；
- 设计、统计和执行报告；
- README 和 daily log 更新。

不进入 Git：

- `data/tokenized/`；
- `.bin/.idx`；
- `.inprogress`；
- `.part`；
- cache 和临时 benchmark。

使用精确暂存，不使用 `git add .` 或 `git add -f`。

## 17. Day 5 完成标准

Day 5 只有在以下条件全部真实通过后才能标记 100%：

1. binary/index/manifest 配置与设计冻结；
2. writer、resume、Dataset、Sampler 和 DataLoader 实现；
3. 新增离线专项全部通过；
4. Pilot 真实 tokenized corpus 完成；
5. 2,129,776 model tokens 与 4,259,552 payload bytes 守恒；
6. context 128/512 的 Dataset 数量和 batch 正确；
7. Windows workers 0/2 通过；
8. 修改后的完整回归无失败；
9. 报告完成；
10. 大型数据仍被忽略；
11. commits 已普通 push；
12. `main == origin/main` 且工作区干净。
