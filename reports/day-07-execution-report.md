# Day 7 训练系统执行报告

## 1. 执行结论

Day 7 已在本地 Windows + CUDA 环境中完成单卡 GPT 训练系统闭环，并通过真实 tokenized FineWeb-Edu Pilot 验收。

本日完成的链路为：

```text
严格训练配置
→ AdamW 参数分组
→ warmup + cosine 学习率
→ FP32 / CUDA BF16 精度策略
→ gradient accumulation、裁剪与有限值检查
→ 确定性训练和验证数据流
→ JSONL 指标日志
→ 原子 checkpoint
→ 模型、optimizer、scheduler、RNG 与数据游标恢复
→ 正式训练 CLI
→ 200-update Debug/Pilot 完整运行
```

最终完整项目回归结果：

```text
508 passed in 13.49s
PowerShell outer elapsed: 15.26s
Exit code: 0
Skipped: 0
Worktree entries after regression: 0
```

本报告记录的是 2,508,032 参数 Debug 模型在 Pilot 数据上的训练系统验收，不是 33,833,984 参数 Baseline 的正式预训练结果。AutoDL、350M Full 语料和 300M-token 正式训练在 Day 7 全程保持未启动。

## 2. 执行范围与边界

### 2.1 Day 7 已完成

- 严格解析 Debug 与 Baseline 的训练配置；
- 明确定义 micro-step、optimizer update、global step 与 token 计数；
- 按 Parameter identity 去重并构建 AdamW decay/no-decay 参数组；
- 实现按 optimizer update 推进的 linear warmup + cosine decay；
- 实现 FP32 与 CUDA BF16 精度策略；
- 实现 gradient accumulation、loss 缩放、有限值检查和全局梯度裁剪；
- 实现确定性训练数据流和固定 validation 数据流；
- 实现 token-weighted validation loss 与 perplexity；
- 实现 JSONL 指标、resolved config 与 run metadata；
- 实现 schema v1 原子 checkpoint 和 strict resume；
- 保存并恢复模型、optimizer、scheduler、trainer counters、RNG 与数据游标；
- 实现正式 `scripts/train_gpt.py` 入口、dry-run、stop gate 和显式 resume；
- 完成 FP32、BF16、validation、checkpoint、resume 与 200-update 集成验收；
- 完成 Day 1～Day 7 的 508 项完整回归。

### 2.2 Day 7 未执行

- 没有构建或下载 350M Full 语料；
- 没有启动 AutoDL；
- 没有探测 RTX 5090 上的 Baseline micro-batch；
- 没有为 Baseline 猜测 gradient accumulation、workers 或 pin memory；
- 没有执行 33,833,984 参数模型的长时间预训练；
- 没有实现 FP16、GradScaler、分布式训练、FlashAttention 或 `torch.compile`；
- 没有实现文本生成、采样或消融实验；
- 没有修改 Day 4 Tokenizer、Day 5 token store 或 Day 6 模型架构。

## 3. 上游身份与运行环境

### 3.1 冻结的项目身份

| 项目 | 值 |
| --- | --- |
| Tokenizer | 16,384 词表 ByteLevel BPE |
| Special IDs | `<bos>=0`、`<eos>=1`、`<pad>=2`、`<unk>=3` |
| Tokenizer SHA-256 | `b26835e02eebf777a257c4732abdd6f9732a115967d2ad839f3a1a00e45ee8c5` |
| Pilot manifest SHA-256 | `141a0c4626cb4f5ba8b041984514825b0d30a34ac8f51a8fcc2fdbb6e512f961` |
| Dataset fingerprint | `a3eb6012c1cb3e2dab2a7839bebb04530563b19b4d5f7d8022e3c121b13ca7f3` |
| Pilot model tokens | 2,129,776 |
| Debug model | 2,508,032 参数，context 128 |
| Baseline model | 33,833,984 参数，context 512 |
| 模型架构 | Pre-LN、learned absolute position、手写 causal MHA、tied LM head |

### 3.2 本地执行环境

| 项目 | 结果 |
| --- | --- |
| 操作系统 | Windows 10 PowerShell 环境 |
| Python | 3.11.9 |
| PyTorch | 2.11.0+cu128 |
| GPU | NVIDIA GeForce RTX 5060 Laptop GPU |
| 可用显存 | 约 7.96 GiB |
| Runtime device | `cuda:0` |
| CUDA BF16 | 支持并通过 smoke |
| AutoDL | 关机，未产生 Day 7 云端训练费用 |

## 4. 冻结训练配置

### 4.1 Debug 执行配置

| 字段 | 值 |
| --- | ---: |
| Seed | 1337 |
| Precision | FP32；另有 BF16 smoke |
| Micro batch size | 4 |
| Context length | 128 |
| Gradient accumulation | 1 |
| Tokens/micro-step | 512 |
| Tokens/update | 512 |
| Total updates | 200 |
| Planned tokens | 102,400 |
| Peak learning rate | 0.0003 |
| Minimum learning rate | 0.00003 |
| Warmup updates | 20 |
| Weight decay | 0.1 |
| Adam betas | 0.9 / 0.95 |
| Adam epsilon | 1e-8 |
| Gradient clip | 1.0 |
| Log interval | 10 |
| Evaluation interval | 20 |
| Evaluation batches | 20 |
| Save interval | 100 |
| DataLoader workers | 0 |
| Pin memory | false |
| Deterministic | true |
| TF32 | false |

Debug 的 200 updates 严格满足：

```text
4 samples × 128 tokens × 1 accumulation × 200 updates
= 102,400 training tokens
```

### 4.2 Baseline 资源配置保持未解析

Baseline 的模型、优化器、300M target-token budget 和 2% warmup ratio 已通过静态配置检查，但以下字段仍为 `null`：

```text
micro_batch_size
gradient_accumulation_steps
num_workers
pin_memory
```

因此 Baseline 配置的 `execution_ready=False`，正式解析会明确拒绝启动。Day 7 没有根据本地 8 GiB GPU 结果猜测 RTX 5090 的资源参数。

## 5. 训练系统实现结果

### 5.1 严格配置与状态

`train/config.py` 实现：

- missing 和 unknown 字段拒绝；
- bool 不允许冒充 int；
- `max_steps` 与 `target_tokens` 恰好一个非空；
- `warmup_steps` 与 `warmup_ratio` 恰好一个非空；
- 学习率、beta、epsilon、decay、clip 和 ratio 的有限值与范围检查；
- Debug 立即解析为可执行 `ResolvedTrainingPlan`；
- Baseline 可静态加载，但 unresolved 资源字段阻止执行；
- target-token 预算按完整 update 向上取整并记录 overshoot。

`TrainerState` 使用 schema v1，统一保存：

- `global_step`；
- `tokens_seen`；
- `micro_steps_completed`；
- `batches_consumed`；
- `samples_consumed`；
- 最近 train/eval loss；
- checkpoint 计数。

### 5.2 AdamW 参数分组

冻结规则：

```text
parameter.ndim >= 2 → decay
parameter.ndim < 2  → no-decay
```

Debug 实测：

| 参数组 | Tensor 数 | 参数量 |
| --- | ---: | ---: |
| Decay | 10 | 2,506,752 |
| No-decay | 10 | 1,280 |
| 合计 | 20 unique tensors | 2,508,032 |

`token_embedding.weight` 与 `lm_head.weight` 是 tied aliases，只按同一个 Parameter 计入一次。所有 trainable parameters 恰好属于一个参数组。

### 5.3 学习率调度

Scheduler 按 optimizer update 的 0-based index 计算本次学习率：

- update index 0：`1.5e-05`；
- warmup 最后一个 index 19：`0.0003`；
- cosine 起点 index 20：`0.0003`；
- 最后训练 update 的 index 199：`0.00003`。

`--stop-at-step` 只控制当前进程停止边界，不改变 200-update scheduler horizon，因此短跑与 resume 仍沿用同一条学习率曲线。

### 5.4 Update 与精度

每次完整 update 的顺序为：

```text
设置 LR
→ zero_grad
→ accumulation × (forward、loss/N、backward)
→ finite gradient check
→ global grad-norm clip
→ optimizer.step
→ 提交 trainer/data counters
→ 写 JSONL
→ evaluation/checkpoint gates
```

FP32 不使用 autocast。CUDA BF16 使用 `torch.autocast`，但不使用 GradScaler；FP16 不在 Day 7 范围内。模型 Attention 内部仍保持 Day 6 已冻结的 FP32 softmax。

### 5.5 数据流与验证

- train 使用 `all_starts` 数据集和确定性随机 sampler；
- validation 使用 `sequential` 非重叠窗口；
- train cursor 由 `samples_consumed` 重建并跳过已消费 sampler indices；
- DataLoader 使用独立派生 generator，避免 worker seed 消耗全局 Torch RNG；
- validation 在 `eval()` + `no_grad()` 下执行；
- validation loss 按有效 target token 数加权；
- evaluation 完成后恢复模型原训练模式；
- validation 不改变下一训练 batch、训练 cursor 或训练 RNG。

### 5.6 日志和 run 目录

每个正式运行目录包含：

```text
runs/<run-id>/
├── metrics.jsonl
├── resolved-config.yaml
└── run-metadata.json
```

日志只允许有限 JSON 数值，文件逐行写入并 flush。新运行拒绝覆盖已有 run-id；resume 会校验 metadata、resolved config、JSONL 完整性、事件单调性以及日志是否领先于所选 checkpoint。

### 5.7 Checkpoint 与恢复

Checkpoint format、checkpoint schema、trainer state 和 RNG state 均为 version 1。保存内容包括：

- model state；
- optimizer state；
- scheduler state；
- trainer counters；
- Python、NumPy、Torch CPU 与 CUDA RNG；
- training sample offset；
- resolved model/training config；
- Tokenizer、manifest 和 dataset identity；
- source commit/dirty metadata。

保存过程为临时文件写入、flush/fsync 和 `os.replace` 原子发布。恢复会先完整验证 payload 与 identity，再修改运行中对象，避免半加载状态。

## 6. 分阶段运行验收

| Gate | 场景 | 真实结果 |
| --- | --- | --- |
| Dry run | Pilot、CUDA FP32 | 构造真实模型/optimizer/data；读取 `(4,128)` 因果 batch；不创建 run/checkpoint |
| Single update | Pilot、CUDA FP32 | step 1、512 tokens、loss `9.722784`、checkpoint 成功 |
| 3 updates | Pilot、CUDA FP32 | step 3、1,536 tokens、5 个 JSONL events、final checkpoint |
| 20 updates | Pilot、CUDA FP32 | step 20、10,240 tokens、validation loss `9.329641`、checkpoint |
| Formal resume | Pilot、CUDA FP32 | step 5 checkpoint 恢复到 step 10，日志连续追加 |
| BF16 smoke | Pilot、CUDA BF16 | 5 updates 全有限、autocast true、GradScaler false |
| Full Debug | Pilot、CUDA FP32 | 200 updates、10 evaluations、2 checkpoints、无 NaN/Inf |

Dry run 前后均验证 `runs/<run-id>` 和 `checkpoints/<run-id>` 不存在，证明 `--dry-run` 不产生持久化副作用。

## 7. 200-update Debug/Pilot 结果

### 7.1 运行身份

| 项目 | 值 |
| --- | --- |
| Run ID | `day07-debug-pilot-200` |
| Device | `cuda:0` |
| Precision | `fp32` |
| Autocast / GradScaler | false / false |
| Model parameters | 2,508,032 |
| Batch source | tokenized Pilot |
| Metrics SHA-256 | `8AE895F344810D2FD9B7BC46D656F47CF14F593E07D88368C31C337D0E0F23CB` |
| JSONL events | 213 |
| Train updates | 200 |
| Evaluations | 10 |
| Checkpoints | 2 |

运行时 Stage G 尚未提交，因此 metadata 如实记录：

```text
source commit = 7e8056903ed480f93362e20d587d43d696901370
source dirty  = true
```

运行后没有再修改 Stage G 代码；同一组 7 个文件经精确暂存后提交为 `c01c087`。最终完整回归在干净的 `c01c087` 上执行并通过。这里保留 dirty metadata，而不是把事后 commit hash 伪装成运行时记录。

### 7.2 训练摘要

| 指标 | 结果 |
| --- | ---: |
| Updates | 200 |
| Total train tokens | 102,400 |
| Initial train loss | 9.7227840423584 |
| Final train loss | 7.44751167297363 |
| Train loss reduction | 23.40% |
| LR at step 1 | 0.000015 |
| LR at step 20 | 0.000300 |
| LR at step 200 | 0.000030 |
| Pre-clip grad norm minimum | 0.664668 |
| Pre-clip grad norm maximum | 3.552653 |
| Pre-clip grad norm mean | 1.017803 |
| Non-finite train events | 0 |

局部 train loss 允许因随机 batch 而波动；验收标准是计数、调度、有限性和总体下降，而不是要求每个 update 单调下降。

### 7.3 Validation 序列

每次 validation 固定评估 20 batches，即 10,240 target tokens。

| Step | Validation tokens | Validation loss | Perplexity |
| ---: | ---: | ---: | ---: |
| 20 | 10,240 | 9.329641199111938 | 11,267.088118963136 |
| 40 | 10,240 | 8.760921335220337 | 6,379.986977677267 |
| 60 | 10,240 | 8.303770971298217 | 4,039.074947308223 |
| 80 | 10,240 | 7.986725354194642 | 2,941.648313845773 |
| 100 | 10,240 | 7.774236798286438 | 2,378.527311078739 |
| 120 | 10,240 | 7.652069854736328 | 2,104.998123645612 |
| 140 | 10,240 | 7.588548064231873 | 1,975.443213844714 |
| 160 | 10,240 | 7.552095508575439 | 1,904.729930178450 |
| 180 | 10,240 | 7.531419062614441 | 1,865.751243714858 |
| 200 | 10,240 | 7.517156267166138 | 1,839.329288748652 |

汇总：

| 指标 | 起点 | 终点 | 降幅 |
| --- | ---: | ---: | ---: |
| Validation loss | 9.329641 | 7.517156 | 19.43% |
| Perplexity | 11,267.09 | 1,839.33 | 83.68% |

Validation loss 在十个观测点持续下降，没有反弹或非有限值。这证明当前训练闭环能够在 Pilot 上学习，但 102,400 tokens 只占正式目标的极小部分，perplexity 仍然很高，不能据此声称模型已完成预训练或具备实用生成质量。

### 7.4 吞吐

| 指标 | tokens/s |
| --- | ---: |
| Aggregate update throughput | 62,137.05 |
| Steady-state throughput（step 2～200） | 73,061.11 |
| Minimum per-update throughput | 2,020.43 |
| Maximum per-update throughput | 89,516.75 |
| Arithmetic mean per-update throughput | 73,560.68 |

200 个 update 在日志中累计耗时 1.647970 秒。该口径只累计每个 optimizer update 的计时，不含进程启动、首次 CUDA 初始化之外的整体准备、validation、checkpoint 和退出时间，因此不能当作 Baseline 或端到端正式训练吞吐。首个 update 的 2,020.43 tokens/s 明显包含初始化成本；steady-state 数值更适合用于本机 Debug 相对比较。

### 7.5 Checkpoint

| Step | 文件 | 大小 |
| ---: | --- | ---: |
| 100 | `step-00000100.pt` | 30,138,029 bytes |
| 200 | `step-00000200.pt` | 30,138,029 bytes |

step 100 和 step 200 的事件顺序均为：

```text
train_update → evaluation → checkpoint
```

step 200 的 interval checkpoint 同时作为 final checkpoint，没有重复写入第二个 step 200 文件。

## 8. BF16 smoke

运行：`day07-debug-pilot-bf16-step5`。

| 项目 | 结果 |
| --- | --- |
| Device | `cuda:0` |
| Precision | `bf16` |
| Autocast | true |
| GradScaler | false |
| Updates / tokens | 5 / 2,560 |
| Initial loss | 9.72314453125 |
| Final loss | 9.72705078125 |
| Train events | 5 |
| Non-finite events | 0 |
| Wrong device/precision events | 0 |
| Checkpoint | `step-00000005.pt`，30,138,029 bytes |

5-step BF16 smoke 的目的不是证明 loss 必须下降，而是验证 autocast 路径、finite loss/gradient、更新、日志和 checkpoint 均能工作。该验收全部通过。

## 9. Checkpoint/resume 一致性

### 9.1 核心连续运行对照

使用 CUDA FP32 对比：

```text
连续运行 4 updates
vs.
运行 2 updates → 保存 → 新对象加载 → 再运行 2 updates
```

真实结果：

| 项目 | 结果 |
| --- | --- |
| Saved checkpoint | `step-00000002.pt`，30,138,093 bytes |
| Saved/restored global step | 2 / 2 |
| Saved/restored tokens | 1,024 / 1,024 |
| Next LR | `4.5e-05` |
| Next training batch | exact |
| Scheduler state | exact |
| Python/NumPy/Torch RNG | exact |
| Continuation metrics | `rtol=1e-6, atol=1e-7` 内一致 |
| Model parameters | tolerance 内一致 |
| Optimizer state | tolerance 内一致 |
| Final global step/tokens | 4 / 2,048 |

### 9.2 正式 CLI resume

运行 `day07-debug-pilot-resume-5-to10`：

- 第一进程训练到 step 5 并保存；
- 第二进程通过显式 `--resume step-00000005.pt` 恢复；
- 恢复后 `global_step=5`、`tokens_seen=2,560`；
- data batches/samples consumed 为 5 / 20；
- 下一学习率为 `9e-05`；
- 继续训练至 step 10、5,120 tokens；
- step 10 loss 为 `9.656713485717773`，与连续运行同一步完全一致；
- 合并日志包含 1 个 `run_start`、10 个 `train_update`、2 个 `checkpoint` 和 1 个 `resume_start`；
- step 5 与 step 10 checkpoint 同时保留。

这些结果证明 resume 恢复的不只是权重，而是训练轨迹所需的全部关键状态。

## 10. 测试证据

以下测试门存在重叠，不应把数量直接相加：

| 阶段 | 测试集合 | 结果 |
| --- | --- | ---: |
| Training config compatibility | training config + existing config/model config | 109 passed |
| Optimizer/scheduler/state | trainer state、optimizer、scheduler、training config | 143 passed |
| Precision/update | precision + trainer | 42 passed |
| Data/eval/logging | data stream + evaluation + run logging | 59 passed |
| Checkpoint/data compatibility | checkpoint + data stream + dataset | 43 passed |
| Formal CLI/loop | run logging + training loop + train CLI | 49 passed |
| Stage G integration compatibility | CLI/loop/logging/checkpoint/trainer/evaluation/data stream | 124 passed in 6.89s |
| Final core targeted gate | config/optimizer/scheduler/trainer/checkpoint | 147 passed in 6.17s |
| **Day 1～Day 7 full regression** | **entire repository** | **508 passed in 13.49s** |

完整回归的 PowerShell 外层耗时为 15.26 秒，exit code 为 0；没有 failed、error、collection error 或 skipped。回归后 `git status --porcelain` 条目数为 0。

## 11. Git 提交

| Commit | 内容 |
| --- | --- |
| `d679642` | `feat: add strict training configuration contract` |
| `2ee8a8a` | `feat: add training state AdamW and cosine scheduler` |
| `8391b28` | `feat: add accumulated training updates and precision` |
| `2bca55d` | `feat: add deterministic data evaluation and logging` |
| `7e80569` | `feat: add atomic checkpoint and exact resume` |
| `c01c087` | `feat: add formal training entry and interval loop` |

Day 7 文档将作为独立提交，避免把实测结论混入实现提交。运行目录、checkpoint、metrics 和模型权重均由 `.gitignore` 排除。

## 12. 执行中发现并修复的问题

### 12.1 Warmup 下界测试错误

首次 scheduler 全曲线测试错误地要求 warmup 阶段所有 LR 都不低于 post-warmup `min_learning_rate`。Debug 的首个 warmup LR 按冻结公式应为 `0.0003/20=1.5e-05`，确实低于 `3e-05`。

处理方式是修正测试对不同 phase 的边界理解，没有为了迎合错误断言修改 scheduler 公式。修复后 scheduler 35 项测试和核心 143 项测试通过。

### 12.2 抽象 `cuda` 与具体 `cuda:0`

首次真实 CUDA pipeline 创建模型时，设备一致性检查发现 `torch.device("cuda")` 与参数实际设备 `torch.device("cuda:0")` 的表示不一致，从而拒绝训练。

修复后 device resolution 总是返回具体当前设备 `cuda:0`，模型先移动到该设备，再创建 Trainer。新增 precision/trainer 回归后，真实 Pilot 3-step pipeline 通过。

### 12.3 DataLoader RNG 隔离

为了让 resume 后的全局 Torch RNG 与连续运行保持一致，DataLoader 使用由 base seed 和 stream name 派生的独立 `torch.Generator`。这样创建 loader/worker seed 不会消耗模型训练使用的全局 RNG。

### 12.4 PowerShell 与 Python 语法边界

Python 构造器示例不能直接粘贴到 PowerShell 提示符执行。后续所有可运行检查均通过 `.py` 脚本、`python -c` 或 PyTest 入口执行，避免 shell 解析 Python 逗号和关键字参数。

### 12.5 Windows 换行提示

`LF will be replaced by CRLF` 是 Git 在 Windows 工作区的换行转换提示，不是测试失败。每个提交前仍使用 `git diff --check` 检查真实空白错误。

## 13. 验收结论

Day 7 的训练系统验收项全部达到本地完成标准：

- [x] Debug/Baseline 严格训练配置；
- [x] Baseline 未解析资源不被猜测；
- [x] AdamW 参数恰好分组且 tied weight 不重复；
- [x] warmup/cosine update 边界正确；
- [x] FP32 update、accumulation、裁剪和 finite checks；
- [x] CUDA BF16 autocast，且未使用 GradScaler；
- [x] 确定性 train stream 与固定 validation；
- [x] JSONL、resolved config 和 metadata；
- [x] schema v1 原子 checkpoint；
- [x] 模型/optimizer/scheduler/RNG/data cursor 恢复；
- [x] continuous vs resume 一致性；
- [x] dry-run、1/3/20/200-step Pilot gates；
- [x] 200 updates、102,400 tokens、10 evaluations、2 checkpoints；
- [x] 全程无 NaN/Inf；
- [x] 508 项完整回归；
- [x] 运行产物未进入 Git；
- [x] AutoDL 和 Full 保持未启动。

在文档提交、普通 push、远端 hash 同步和干净工作区得到真实终端证据前，Day 7 仍不能记为远端 100%。

## 14. 下一阶段

Day 7 完成并同步远端后，下一阶段才进入付费 GPU 资源定标：

1. 在 AutoDL checkout Day 7 的已验证 commit；
2. 验证 Python、PyTorch、CUDA、BF16 和磁盘路径；
3. 对 33,833,984 参数 Baseline 做 micro-batch/OOM 探测；
4. 冻结 accumulation、workers、pin memory 与 tokens/update；
5. 运行短 BF16 稳定性和 checkpoint/resume smoke；
6. 明确 Full 数据、磁盘和 checkpoint 保留预算；
7. 获得明确授权后才启动 300M-token 正式预训练。

正式训练开始前仍需坚持：

```text
不猜资源参数
不把 Debug/Pilot 结果写成 Baseline 质量
不在没有 checkpoint/resume 证据时启动长跑
不在未经确认时产生 AutoDL 费用
```
