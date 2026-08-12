# Day 9 Full 语料、Tokenized Full 与单更新验收报告

## 1. 执行结论

Day 9 已完成正式预训练前的数据闭环：在 AutoDL RTX 5090 C14 实例上，按冻结 revision 和显式 Parquet 文件列表构建了 350M provided-token Full 语料，用 Day 4 冻结的 16,384 词表 ByteLevel BPE 完成全量编码，并在真实 Full manifest 上执行了严格数据读取验证、BF16 dry-run 和恰好一次 optimizer update。

最终结论：

- Full source corpus 状态为 `complete`，共 70 个 shard groups；
- 338,849 篇保留文档提供 350,000,812 provided tokens；
- Tokenized Full 状态为 `complete`，共 379,587,945 个项目模型 tokens；
- Tokenized Full 共 77 个 storage shards，payload 为 759,175,890 bytes；
- workers 0/2/4、跨 storage shard 的 `T+1` 读取、因果 shift、token ID 范围和 memmap 释放全部通过；
- 33,833,984 参数 Baseline 在 Full manifest 上完成一次 BF16 optimizer update；
- 唯一真实更新停在 step 1 / 65,536 tokens，并保存一个原子 checkpoint；
- 没有触发 validation，没有启动 300M-token 正式预训练；
- 最终代码回归为本地和远端各 551 passed；
- 远端证据已封装为 63,808-byte archive，并在 Windows 本地完成字节数与三方 SHA-256 核对。

Day 9 的完成只证明 Full 数据与正式训练入口已经具备可执行、可验证和可恢复的工程条件，不代表模型已经完成预训练，也不能把单更新 loss 当作质量结论。

## 2. 范围与授权边界

用户明确授权进入 Day 9 Stage B，并允许产生 AutoDL 费用。本轮使用 C14 RTX 5090 实例，控制台价格为 ¥2.78/小时。

授权范围包括：

1. 更新 Full source/tokenization 可执行契约；
2. 访问冻结 FineWeb-Edu revision；
3. 下载构建 350M provided-token corpus 所需的源文件；
4. 构建、验证 source Full 与 tokenized Full；
5. 在 Full manifest 上执行 dry-run；
6. 只执行一次 optimizer update；
7. 保存并只读核验该 update 的日志、GPU 样本和 checkpoint；
8. 生成、下载并验证 Day 9 证据包。

本轮明确没有授权：

- 启动 4,578 updates / 300M tokens 正式预训练；
- 自动继续第二次 update；
- 用单更新结果评价模型语言能力；
- 释放 C14 实例或删除持久盘数据；
- 把 Full JSONL、tokenized `.bin/.idx`、source cache、run 或 checkpoint 提交 Git。

## 3. 权威环境与身份

### 3.1 AutoDL 环境

| 项目 | 结果 |
| --- | --- |
| 实例 | AutoDL C14 |
| GPU | NVIDIA GeForce RTX 5090 |
| VRAM | 32,607 MiB |
| Driver | 580.105.08 |
| CUDA Runtime | 13.0 |
| Python | 3.12.3 |
| PyTorch | 2.12.1+cu130 |
| `datasets` | 5.0.1 |
| `pyarrow` | 25.0.1 |
| `tokenizers` | 0.23.1 |
| BF16 | supported |
| CPU | 25 cores |
| 数据盘 | 50 GB |
| Day 9 最终空闲空间 | 44 GB |
| Day 9 最终 Git HEAD | `23c63a6b81c6f44e6cb7dc1208395f1b84c4f407` |
| Git 状态 | `main...origin/main`，clean |

### 3.2 数据集身份

| 项目 | 冻结值 |
| --- | --- |
| Dataset | `HuggingFaceFW/fineweb-edu` |
| Configuration | `sample-10BT` |
| Split | `train` |
| Revision | `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9` |
| Full target | 350,000,000 provided tokens |
| Source shard target | 5,000,000 provided tokens |
| Explicit source files | 14 |
| Full config fingerprint | `555c15b1e851f567290fda3acf7e39245d3bc050c49259655bc071c8791aef84` |

显式源文件固定为：

```text
sample/10BT/000_00000.parquet
...
sample/10BT/013_00000.parquet
```

350M 目标在第一个文件内已经达到，因此正式构建没有消费后续文件。第一个源文件的冻结身份为：

| 项目 | 结果 |
| --- | --- |
| 文件 | `sample/10BT/000_00000.parquet` |
| Bytes | 2,152,819,114 |
| SHA-256 | `b1ba7b2ce4cb5ea6ef42dca40263eabb85f37700d01693a68e9b30a31d78e871` |
| Local-cache offline probe | PASS |

### 3.3 Tokenizer 身份

| 项目 | 结果 |
| --- | --- |
| Tokenizer file | `tokenizer/artifacts/tokenizer.json` |
| SHA-256 | `b26835e02eebf777a257c4732abdd6f9732a115967d2ad839f3a1a00e45ee8c5` |
| Vocabulary | 16,384 |
| Special IDs | BOS 0 / EOS 1 / PAD 2 / UNK 3 |
| Metadata expected SHA | `8622711407aab3f299996b7d3009d4f4447ae35879ca8e50451b5f0adbdf5141` |

Windows working tree 的 metadata 使用 CRLF，Linux checkout 使用 LF。Day 9 保留 raw-byte strict check，同时只允许 LF/CRLF 规范化 fallback；任何非换行内容变化仍会导致 SHA-256 拒绝。

## 4. Day 9 代码演进

Day 9 没有把网络、生命周期和身份修复混在一个不可审计的大提交中，而是按问题边界形成以下提交：

| Commit | 主题 | 作用 |
| --- | --- | --- |
| `5be64d3` | `feat: add verified full corpus tokenization` | 增加 manifest-bound Full tokenization 配置、CLI 和测试 |
| `7c4ad5d` | `fix: normalize tokenizer metadata line endings` | 跨 Windows/Linux 严格处理 metadata 换行身份 |
| `4be46ee` | `fix: freeze explicit full corpus sources` | 冻结 14 个显式 Parquet 源文件，绕过不稳定 Hub 目录遍历 |
| `c0b1df1` | `fix: stream frozen full sources lazily` | 按需打开显式 Parquet，避免 eager stream 初始化 |
| `c23876c` | `feat: support verified local full source cache` | 支持通过环境变量解析经过身份验证的本地源缓存 |
| `23c63a6` | `fix: close full corpus source stream` | 在成功和异常路径上显式关闭 source generator，消除进程退出 abort |

以上提交均已普通 push，Day 9 远端技术阶段最终满足：

```text
HEAD == origin/main
HEAD = 23c63a6b81c6f44e6cb7dc1208395f1b84c4f407
worktree = clean
```

## 5. 网络访问与源缓存

### 5.1 直接 Hub 失败

最初通过 `datasets.load_dataset()` 打开固定 revision 时，AutoDL 到 `huggingface.co` 出现：

- `[Errno 101] Network is unreachable`；
- Hub API `503`；
- 连接在 response body 未完成时被对端关闭；
- 分页目录 API 即使设置 mirror，部分请求仍回到直接 Hub。

本地电脑是否连接 VPN 不会直接改变 AutoDL 容器的出口路径，因此不能用切换本地 VPN 代替远端链路诊断。

### 5.2 显式文件与 mirror

通过 mirror API 成功列出 `sample/10BT` 下 14 个 Parquet 文件，并验证第一个文件的 range request。直接 streaming 虽能创建 lazy object，但对 2.15GB Parquet 的随机 range 读取仍会长时间阻塞或被对端中断。

### 5.3 可恢复本地缓存

最终方案是：

1. 冻结 14 个文件名；
2. 以固定 dataset/revision/filename 下载；
3. 允许连接中断后从 `.incomplete` 文件恢复；
4. 下载完成后验证 2,152,819,114 bytes 与 SHA-256；
5. 通过 `SMALL_GPT_SOURCE_CACHE_DIR` 只读解析本地文件；
6. 设置 local-files-only，断网条件下打开真实 Parquet；
7. 在离开 probe 前显式关闭 stream。

下载完成门：

```text
DownloadExitCode=0
DownloadStatus=COMPLETE_AND_VERIFIED
ActualBytes=2152819114
ActualSHA256=b1ba7b2ce4cb5ea6ef42dca40263eabb85f37700d01693a68e9b30a31d78e871
SourceDownloadIdentity=PASS
```

## 6. Full source corpus

### 6.1 正式构建结果

正式输出：

```text
data/processed/fineweb_edu_full
```

构建从 2026-08-13 04:21:46+08:00 开始，在 04:23:30 前完成 70 个 shard groups 和 manifest 发布。统计如下：

| 指标 | 结果 |
| --- | ---: |
| Status | `complete` |
| Input/source records | 339,027 |
| Kept records | 338,849 |
| Exact duplicates removed | 178 |
| Kept provided tokens | 350,000,812 |
| Shard groups | 70 |
| Retention rate | 0.99947497 |
| On-disk size | 约 1.7 GB |

Split 统计：

| Split | Records | Provided tokens |
| --- | ---: | ---: |
| Train | 332,112 | 343,299,897 |
| Validation | 3,407 | 3,452,153 |
| Test | 3,330 | 3,248,762 |
| **合计** | **338,849** | **350,000,812** |

身份：

```text
Source manifest SHA-256 = 14c69dc545838b426e29162c73132cfe444bb2cc56b72c80bb4929f3c65ca96a
Config fingerprint      = 555c15b1e851f567290fda3acf7e39245d3bc050c49259655bc071c8791aef84
```

### 6.2 发布后 exit 134 的处理

第一次正式构建已经：

- 写完 70 个 shards；
- 把 state 标为 `complete`；
- 保存 manifest；
- 输出全部守恒统计。

但 Python 在销毁仍然活跃的 Parquet generator 时触发 native teardown abort，因此 wrapper 得到 exit 134。不能因为进程码非零就直接删除已经发布的数据，也不能因为 manifest 存在就忽略异常。

处理顺序为：

1. 确认没有存活 pipeline 进程；
2. 对 70 个 shard groups 执行 canonical recovery validation；
3. 验证每个 JSONL 的大小、SHA、文本 hash、split 和统计守恒；
4. 确认 339,027 source records、338,849 kept records、350,000,812 tokens 全部一致；
5. 将现有 corpus 判定为可保留的完整产物；
6. 在入口使用受控 lifecycle，在 `finally` 中关闭 source stream；
7. 增加成功、异常和 non-closeable source 的测试；
8. 对真实 Parquet 做 1,000-token active lifecycle probe。

修复后的真实 probe：

```text
ActiveStreamProbeExit=0
ProbeStatus=complete
ProbeProvidedTokens=1900
ProbeLogContainsAbort=False
ActiveStreamLifecycleValidation=PASS
FormalCorpusStillComplete=PASS
```

因此 Full corpus 既通过产物完整性门，也通过修复后的真实进程生命周期门。

## 7. Tokenized Full

### 7.1 正式编码

正式输出：

```text
data/tokenized/fineweb_edu_full
```

编码从 2026-08-13 04:42:43+08:00 开始，04:53:38+08:00 原子发布完成，exit code 0。发布后 staging 目录不存在。

总体统计：

| 指标 | 结果 |
| --- | ---: |
| Records | 338,849 |
| Raw BPE tokens | 379,249,096 |
| Appended EOS tokens | 338,849 |
| Model tokens | 379,587,945 |
| Token payload bytes | 759,175,890 |
| Storage shards | 77 |
| On-disk size | 约 741 MB |

Split 统计：

| Split | Records | Provided tokens | Raw BPE tokens | Model tokens | Storage shards |
| --- | ---: | ---: | ---: | ---: | ---: |
| Train | 332,112 | 343,299,897 | 371,996,079 | 372,328,191 | 75 |
| Validation | 3,407 | 3,452,153 | 3,737,938 | 3,741,345 | 1 |
| Test | 3,330 | 3,248,762 | 3,515,079 | 3,518,409 | 1 |
| **合计** | **338,849** | **350,000,812** | **379,249,096** | **379,587,945** | **77** |

实际 model/provided 比率约为：

```text
379,587,945 / 350,000,812 ≈ 1.084535
```

它高于 Pilot 的约 1.064844，说明正式训练预算必须继续以 Full manifest 的实际 model tokens 为准，不能沿用 Pilot 比率估算替代真实统计。

### 7.2 Tokenized 身份

```text
Tokenized manifest SHA-256 = ce7cd91075c7c666c427e1aaa286096a7f386643f3a76de3c26ef770d6cce67e
Config fingerprint         = 39dab5bacdf8719bbc849e85ddcd7422cba5777fc044b437d050a49b87ab174f
Tokenizer SHA-256          = b26835e02eebf777a257c4732abdd6f9732a115967d2ad839f3a1a00e45ee8c5
```

manifest contract、source manifest identity、token conservation、payload size、索引连续性和全部 split 统计均通过。

## 8. Dataset 与 DataLoader 实验

### 8.1 全量检查

workers 0 的完整 scan 通过，确认：

- 所有 77 个 `.bin/.idx` 文件 header 合法；
- little-endian `uint16` payload 大小与 token 数精确一致；
- 文档 index offset/length 连续；
- 每篇文档 EOS 与总统计守恒；
- train/validation/test 三个逻辑 token stream 隔离；
- 完成输出不存在 `.part` 或 staging 残留。

### 8.2 跨 storage shard 的真实 `T+1` 读取

```text
BoundaryTokenOffset=4998983
BoundaryWindowStart=4998727
BoundaryReadLength=513
CrossStorageShardRead=PASS
CrossStorageShardCausalShift=PASS
ParentMemmapsClosedAfterBoundaryRead=PASS
```

这证明 context 512 的 `(x, y)` 窗口能够跨 train shard 0/1 边界读取，并满足 `y[:-1] == x[1:]`。

### 8.3 多 worker

对 workers 2 和 4，分别在 train、validation、test 构造 `(16, 512)` batch：

- batch shape 正确；
- causal shift 正确；
- token IDs 全部在 `[0, 16384)`；
- workers 退出后文件句柄释放；
- workers 2 与 workers 4 的固定批次内容完全一致；
- 父进程 memmap 全部关闭。

最终门：

```text
FullTokenizedLoaderValidation=PASS
Workers2And4BatchIdentity=PASS
AllParentMemmapsClosed=PASS
TokenizedGitIgnoreExit=0
```

## 9. Full manifest 上的单更新 smoke

### 9.1 正式计划 preflight

| 项目 | 结果 |
| --- | ---: |
| Model parameters | 33,833,984 |
| Device | CUDA |
| Precision | BF16 |
| Micro batch | 16 |
| Gradient accumulation | 8 |
| Context length | 512 |
| Tokens/update | 65,536 |
| Total updates | 4,578 |
| Warmup updates | 92 |
| Planned tokens | 300,023,808 |
| Workers | 4 |

`--batch-source pilot` 是训练 CLI 的历史枚举名；该分支实际创建 `TrainingDataStream` 并读取传入 manifest。Full 身份由 manifest SHA 和 dataset fingerprint 严格锁定，并不表示本轮使用了 Pilot 数据。

### 9.2 Dry-run

Run ID：

```text
day09-full-one-update-20260813-050032
```

Dry-run 验证：

- `cuda:0` / BF16；
- autocast true / GradScaler false；
- 33,833,984 参数；
- sample input/target `(16, 512)`；
- causal shift exact；
- Full manifest/tokenizer/fingerprint/commit identity exact；
- run/checkpoint 目录均未创建。

### 9.3 唯一一次 optimizer update

执行参数固定为 `--stop-at-step 1`。结果：

| 指标 | 结果 |
| --- | ---: |
| Global step | 1 |
| Tokens seen | 65,536 |
| Micro steps | 8 |
| Samples | 128 |
| Train loss | 9.816444397 |
| Learning rate | 0.00000326086956522 |
| Grad norm before clip | 7.308704853 |
| Tokens/s | 58,906.051 |
| Evaluations | 0 |
| Checkpoints | 1 |
| CLI elapsed | 6 seconds |

训练日志、metrics 和 checkpoint 内部状态一致：

```text
event order = run_start -> train_update -> checkpoint
global_step = 1
micro_steps_seen = 8
tokens_seen = 65536
samples_consumed = 128
last_eval_step = 0
last_save_step = 1
```

没有执行第二次 update，也没有 resume 该 smoke run。

### 9.4 GPU 样本

| 指标 | 结果 |
| --- | ---: |
| Samples | 5 |
| Peak memory used | 5,783 MiB |
| Peak utilization | 31% |
| Peak temperature | 35°C |
| Post-run memory used | 0 MiB |

1 秒级 `nvidia-smi` 样本用于证明该更新确实在 RTX 5090 上执行；它不是 CUDA allocator 的精确 peak-memory 测量，不能替代 Day 8 隔离资源探针。

### 9.5 Checkpoint

```text
Path   = checkpoints/day09-full-one-update-20260813-050032/step-00000001.pt
Bytes  = 406108827
SHA256 = 457d12600f143b400e2ab51af549d0bd020badafb5c32c38bb9502a0a254e7e4
```

CPU `weights_only=True` 严格加载通过，确认：

- checkpoint format/schema 正确；
- resolved config 与 run snapshot 一致；
- model/optimizer/scheduler/RNG state 存在；
- CUDA RNG state 已保存；
- tokenizer、Full manifest、dataset fingerprint、source commit identity 全部一致；
- `source_dirty=false`；
- 只存在 `step-00000001.pt`，没有 `.tmp` 或 `.part`。

## 10. 测试与回归

Day 9 每个修复都先运行定向测试，再运行完整回归。关键门：

| 阶段 | 结果 |
| --- | --- |
| Full tokenization 初始功能 | 540 passed |
| Metadata newline 修复 | 541 passed |
| Explicit source files | 546 passed |
| Lazy source stream | 546 passed |
| Verified local source cache | 548 passed |
| Source lifecycle 最终版本，本地 | 551 passed in 42.46s |
| Source lifecycle 最终版本，AutoDL | 551 passed, 2 warnings in 5.87s |

两个远端 warning 均来自多线程进程中 DataLoader 测试调用 `fork()` 的 Python 3.12 deprecation warning；测试通过，worker 释放也有独立验证，不是 Day 9 功能失败。

最终新增覆盖包括：

- Full profile CLI 与默认 config 解析；
- source manifest-bound 身份与 placeholder 拒绝；
- metadata LF/CRLF-only fallback；
- 14 个显式源文件的配置 fingerprint；
- 显式文件按顺序、按需打开；
- 本地 cache 路径、bytes 和 SHA-256 验证；
- 缓存缺失、损坏与 fallback 拒绝；
- success/failure/non-closeable stream 生命周期；
- Full tokenization conservation、resume、publication 与 CLI；
- Pilot fingerprint 保持冻结不变。

## 11. 存储、证据与费用

### 11.1 最终存储

| 资产 | 大小 |
| --- | ---: |
| Full source corpus | 约 1.7 GB |
| Tokenized Full | 约 741 MB |
| Step-1 smoke checkpoint | 406,108,827 bytes |
| AutoDL 数据盘已用 | 约 6.2 GB / 50 GB |
| AutoDL 数据盘剩余 | 约 44 GB |

这些生成资产被 `.gitignore` 排除。关闭实例后它们仍依赖 AutoDL 持久盘；释放实例可能造成不可恢复的数据丢失。

### 11.2 证据包

远端证据包：

```text
small-gpt-day09-evidence-20260813-051212.tar.gz
Bytes  = 63808
SHA256 = 0edd992562f64b1bdc156fd5bbb498f087b8de17097b974720b213075058e3a8
Entries = 66
```

Windows 本地验证：

```text
ExpectedHash == ActualHash == SidecarHash
ExpectedBytes == ActualBytes == 63808
Day9EvidenceDownloadGate=PASS
```

证据保存在仓库外：

```text
D:\code\small-gpt-day09-evidence
```

证据包故意不包含 1.7GB source payload、741MB tokenized payload 或 406MB checkpoint，只包含配置、manifests、日志、验证 JSON、环境和哈希摘要。

### 11.3 时间与费用边界

C14 于 2026-08-12 22:23:53+08:00 启动，证据在 2026-08-13 05:12 左右封存，约 6 小时 48 分钟。按 ¥2.78/小时估算，截至证据封存约 ¥18.9。

该费用大部分来自网络诊断、2.15GB 源文件的低速可恢复下载、跨平台修复和完整验证，而不是语料构建或单更新本身：

- Full source 构建约 1 分 44 秒；
- Tokenized Full 编码约 10 分 55 秒；
- 唯一一次 optimizer update 的 CLI 墙钟约 6 秒。

最终费用应以 AutoDL 在关机时显示的实际账单为准。

## 12. 已踩问题与禁止重复

| 问题 | 根因 | 最终处理 |
| --- | --- | --- |
| Linux metadata SHA 不匹配 | Windows CRLF / Linux LF | 只允许换行规范化 fallback，非换行变化仍拒绝 |
| Hub `Errno 101` / 503 | AutoDL 远端出口与 Hub API 不稳定 | mirror 诊断、显式文件冻结、本地 cache |
| mirror 后请求又回直接 Hub | Hub 分页/文件系统内部 endpoint 行为 | 不再依赖运行时目录遍历 |
| Parquet streaming 长时间无首条记录 | 2.15GB 文件的远程随机 range 读取不稳定 | 可恢复完整下载后 local-only 读取 |
| `.incomplete` 长时间不增长 | 连接中断与重试 | 监控实际文件 bytes 和日志，不只看 PID |
| eager 显式源 stream 被 kill 137 | 多文件 stream 初始化生命周期过重 | 每次只在需要时打开一个文件 |
| corpus 已发布但 wrapper exit 134 | 未关闭 Parquet generator 的 native teardown | canonical recovery + `finally` close + 真实 probe |
| probe exit 124 但打印 PASS | generator teardown 卡住 | 显式 `close()` 后才允许 PASS |
| 核验脚本误找 `metadata.json` | 正式 run 文件名实际为 `run-metadata.json` | 按 `RunPaths` 契约读取 `run-metadata.json` 和 `resolved-config.yaml` |
| 本地 VPN 是否影响 AutoDL | 两者网络出口独立 | 直接诊断 AutoDL endpoint，不盲目切本地 VPN |

禁止重复：

- 不重新构建已通过 canonical validation 的 Full source；
- 不重新编码已原子发布且完整扫描通过的 Tokenized Full；
- 不再次执行 Day 9 单更新 run ID；
- 不把 exit 134 误判为 70-shard corpus 必然损坏；
- 不把 `--batch-source pilot` 字符串误读为实际使用 Pilot manifest；
- 不用本地 VPN 状态解释所有远端网络故障；
- 不把 source cache、Full、tokenized 或 checkpoint 加入 Git；
- 不在证据未保全前释放 C14。

## 13. 最终验收清单

- [x] Day 9 获得独立费用授权；
- [x] Git HEAD 与 origin/main 同步；
- [x] Full source profile 绑定显式源文件与 fingerprint；
- [x] 2.15GB 源文件完成 bytes/SHA 验证；
- [x] offline local-cache probe 通过；
- [x] 70-shard Full source corpus 完成；
- [x] source manifest 和全部 shards canonical validation 通过；
- [x] source stream success/failure 生命周期修复；
- [x] 379,587,945-token Tokenized Full 原子发布；
- [x] 77 个 storage shards 完整扫描；
- [x] 跨 storage shard `T+1` 读取通过；
- [x] workers 2/4 的三个 split batch 通过；
- [x] Full dry-run 通过且没有写运行产物；
- [x] 恰好一次 BF16 optimizer update；
- [x] 0 evaluations / 1 checkpoint；
- [x] checkpoint CPU 严格加载与身份验证；
- [x] 本地与远端最终 551 项回归；
- [x] 证据 archive 生成、下载、三方 SHA 与 entries 验证；
- [x] 300M-token 正式预训练未启动。

## 14. 下一阶段

Day 10 / 正式预训练必须获得新的独立授权。启动前仍需：

1. 确认 C14 只是关机而不是释放，Full 与 Tokenized Full 仍存在；
2. 启动后核对 Git、GPU、软件版本、manifest SHA 和 fingerprint；
3. 确认没有复用 Day 9 smoke run ID；
4. 创建新的正式 run ID；
5. 重新执行不写数据的正式 preflight；
6. 明确训练期间监控、checkpoint 下载和关机策略；
7. 再次获得 300M-token / 4,578-update 长跑授权。

在这些门全部通过前，不能因为 Day 9 已完成就自动启动正式训练。
