# Day 7 训练系统设计与冻结契约

> 状态：执行前设计文档。
>
> 本文只冻结 Day 7 训练系统的接口、计数、配置和恢复语义。只有代码被复制到 Windows 仓库、测试真实通过并形成 Git 证据后，才能把对应内容写成已完成。

## 1. 上游权威状态

Day 7 直接复用已经完成的两层能力：

```text
Day 5 tokenized Pilot / Dataset / DataLoader
→ Day 6 Decoder-only GPT / causal loss / backward
```

冻结身份：

| 项目 | 值 |
| --- | --- |
| Tokenizer vocabulary | 16,384 |
| Special IDs | BOS=0、EOS=1、PAD=2、UNK=3 |
| Pilot model tokens | 2,129,776 |
| Token storage | 5 个 little-endian uint16 shards |
| Runtime batch dtype | `torch.int64` |
| Debug model | 2,508,032 parameters，context 128 |
| Baseline model | 33,833,984 parameters，context 512 |
| Model contract | Pre-LN、learned absolute position、tied LM head |

Day 7 不修改 Tokenizer、token store、causal x/y、Attention、Block 或 GPT。

## 2. 已核对的真实公共接口

### 2.1 Token store

```python
SplitTokenStore(
    manifest_path,
    split,
    verify_hashes=False,
)
```

训练器为 train/validation 分别打开 store，并在正常或异常退出时关闭。

### 2.2 Dataset

```python
CausalWindowDataset(
    store,
    context_length,
    mode="all_starts" | "sequential",
)
```

语义：

- `all_starts`：所有合法 token start，供随机训练 sampler 使用；
- `sequential`：start 为 `0,T,2T,...`，供固定验证使用；
- 每个 item 从底层读取 `T+1` tokens；
- 返回 `(tokens[:-1], tokens[1:])`；
- 模型和训练器不得再次 shift。

### 2.3 随机训练 sampler

```python
EpochRandomWindowSampler(
    dataset,
    samples_per_epoch,
    base_seed,
    epoch=0,
    chunk_size=65_536,
)
```

真实 sampler 使用 `base_seed + epoch` 的 SHA-256 派生种子，并通过独立 `torch.Generator` 生成随机 start。它不依赖全局 Torch RNG。

Day 7 首版将预训练视为一个由 token/update 预算确定的单一随机样本流：

```text
samples_required
= total_updates
 × micro_batch_size
 × gradient_accumulation_steps
```

Sampler 仍使用 `epoch=0`。Checkpoint 保存 `samples_consumed`。恢复时重建相同 sampler，并在 sampler index 层跳过已消费 start；被跳过的 index 不读取磁盘 token。

这样可以：

- 保留 Day 5 sampler；
- 不修改二进制数据格式；
- 不让 validation 改变训练顺序；
- 让下一 batch 可被严格比较。

### 2.4 DataLoader

```python
build_dataloader(
    dataset,
    batch_size=...,
    sampler=...,
    num_workers=0,
    pin_memory=False,
    drop_last=...,
    persistent_workers=False,
    prefetch_factor=2,
)
```

Day 7 Debug 首版：

```text
num_workers = 0
pin_memory = false
drop_last = true for train
drop_last = false for validation
```

先保证精确恢复，再单独优化 worker/pinning。

### 2.5 GPT

```python
output = model(input_ids, targets)
loss = output.loss
logits = output.logits
```

`GPT.forward()` 已经直接对全部 `B*T` logits/targets 计算 cross entropy。训练器不得切掉最后位置或再次 shift。

## 3. Step 与 token 口径

### 3.1 Micro-step

一个 micro-batch 的 forward + backward。

### 3.2 Optimizer update

累积 N 个 micro-step 后的一次 optimizer step。

Day 7 所有 `step`、warmup、eval interval、save interval 都以 optimizer update 为单位。

### 3.3 Global step

已成功完成的 optimizer update 数。

### 3.4 Token 计数

无 padding 时：

```text
tokens_per_micro_step
= micro_batch_size × context_length

tokens_per_update
= tokens_per_micro_step × gradient_accumulation_steps
```

Debug：

```text
4 × 128 × 1 = 512 tokens/update
200 updates = 102,400 planned tokens
```

Baseline 使用 `target_tokens=300,000,000`。资源参数解析后：

```text
total_updates = ceil(target_tokens / tokens_per_update)
planned_tokens = total_updates × tokens_per_update
overshoot = planned_tokens - target_tokens
```

首版只在完整 update 边界停止，因此允许记录明确的小幅 overshoot，不构造半个 accumulation。

## 4. 配置结构

### 4.1 预算互斥

以下恰好一个非 null：

```text
max_steps
target_tokens
```

Debug 使用 max steps，Baseline 使用 target tokens。

### 4.2 Warmup 互斥

以下恰好一个非 null：

```text
warmup_steps
warmup_ratio
```

Debug 使用 20 steps；Baseline 使用 0.02 ratio。

Ratio 转换：

```text
warmup_updates = ceil(total_updates × warmup_ratio)
```

### 4.3 Baseline 未解析资源

以下保持 null：

```text
micro_batch_size
gradient_accumulation_steps
num_workers
pin_memory
```

静态解析允许 null，但 `resolve()` 和正式启动必须拒绝 unresolved 配置。数值只能在 AutoDL 显存/吞吐探测后写入。

### 4.4 精度

Day 7 只支持：

- FP32；
- CUDA BF16。

不实现 FP16/GradScaler。显式 CPU+BF16 被拒绝，不能静默回退。

### 4.5 输出目录

```text
runs/
checkpoints/
```

两者已经被 `.gitignore` 覆盖。

## 5. 当前配置值

### 5.1 Debug

| 字段 | 值 |
| --- | ---: |
| seed | 1337 |
| device | auto |
| precision | fp32 |
| micro batch | 4 |
| accumulation | 1 |
| max steps | 200 |
| learning rate | 3e-4 |
| minimum LR | 3e-5 |
| weight decay | 0.1 |
| betas | 0.9 / 0.95 |
| Adam epsilon | 1e-8 |
| warmup steps | 20 |
| gradient clip | 1.0 |
| log interval | 10 |
| eval interval | 20 |
| eval batches | 20 |
| save interval | 100 |
| num workers | 0 |
| pin memory | false |
| deterministic | true |
| TF32 | false |

### 5.2 Baseline

| 字段 | 值 |
| --- | ---: |
| seed | 1337 |
| device | cuda |
| precision | bf16 |
| target tokens | 300,000,000 |
| micro batch | unresolved |
| accumulation | unresolved |
| learning rate | 3e-4 |
| minimum LR | 3e-5 |
| weight decay | 0.1 |
| betas | 0.9 / 0.95 |
| Adam epsilon | 1e-8 |
| warmup ratio | 0.02 |
| gradient clip | 1.0 |
| log interval | 10 |
| eval interval | 500 |
| eval batches | 100 |
| save interval | 1,000 |
| workers/pin memory | unresolved |
| deterministic | false |
| TF32 | true |

## 6. TrainingConfig 错误契约

严格拒绝：

- missing/unknown training field；
- bool 冒充 int；
- 非正 batch、accumulation、budget 或 interval；
- 负 worker；
- max_steps/target_tokens 同时有值或同时 null；
- warmup_steps/ratio 同时有值或同时 null；
- warmup 不小于 total updates；
- 非有限或非法 LR、decay、beta、epsilon、clip；
- min LR 大于 peak LR；
- 非布尔 deterministic/TF32/pin memory；
- 不支持的 device/precision；
- 空输出目录；
- Baseline 未解析资源直接 resolve。

## 7. Optimizer 冻结规则

Stage C 将按唯一规则分组：

```text
parameter.ndim >= 2 → weight decay
parameter.ndim < 2  → no decay
```

结果：

- Linear/Embedding matrices decay；
- LayerNorm affine no decay；
- tied token embedding/LM head 通过 Parameter identity 只出现一次；
- 所有 trainable Parameter 恰好属于一个 group。

AdamW：

```text
betas = (0.9, 0.95)
eps = 1e-8
weight_decay = 0.1 / 0.0 by group
```

## 8. Scheduler 冻结规则

Scheduler 以即将执行的 0-based update index `u` 计算本次 LR。

Warmup：

```text
lr(u) = lr_max × (u + 1) / W
```

Cosine：

```text
D = U - W
p = (u - W) / (D - 1)
lr(u) = lr_min + 0.5 × (lr_max - lr_min) × (1 + cos(pi × p))
```

Scheduler 每个 optimizer update 只推进一次，不随 micro-step 推进。

## 9. Update 冻结顺序

```text
设置本次 LR
→ zero_grad once
→ N × (forward, finite loss check, loss/N, backward)
→ finite gradient check
→ clip global norm once
→ optimizer.step once
→ commit global_step/tokens/sample cursor
→ JSONL event
→ completed-step eval/save gates
```

任何 NaN/Inf 都不能推进 optimizer/state，也不能覆盖最后有效 checkpoint。

## 10. Validation 冻结规则

- validation split；
- Dataset `mode="sequential"`；
- `model.eval()`；
- `torch.no_grad()`；
- 固定前 N batches 或全 split；
- 按 `targets.numel()` 聚合 token-weighted loss；
- eval 后恢复原 mode；
- 不推进训练 sampler/cursor/RNG。

## 11. Checkpoint 冻结规则

只在完整 optimizer update 边界保存：

```text
schema version
model / optimizer / scheduler
trainer counters
Python / NumPy / Torch / CUDA RNG
sample offset
resolved configs
model / tokenizer / dataset identity
source commit
```

保存：

```text
temporary file
→ flush / fsync
→ os.replace final
```

恢复必须验证下一 LR、下一 batch、step 和 token 计数。只通过 model state round-trip 不算完整 resume。

## 12. Stop gate 与 scheduler horizon

分层 smoke 使用操作性：

```text
--stop-at-step K
```

它不改变 YAML `max_steps=200`，也不改变 scheduler 总长度。

禁止用 `--max-steps 5` 保存后再把总长改成 10 做 resume 对照，因为两次运行的 LR 曲线已经不同。

到达 stop gate 时，若该 step 尚未按 interval 保存，应原子写一次 final checkpoint。

## 13. Stage B 验收门

- 两套 YAML 具有相同严格 training fields；
- 原有模型字段完全不变；
- 原有精确参数量不变；
- `TrainingConfig.from_yaml()` 对两套配置通过；
- Debug `resolve()` 得到 512 tokens/update、200 updates、20 warmup、102,400 planned tokens；
- Baseline 解析通过但 execution-ready 为 false；
- Baseline `resolve()` 明确拒绝；
- target-token ceil/overshoot 有测试；
- missing/unknown/type/range/互斥错误有测试；
- `check_training_config.py` 成功；
- 原有 config/model tests 无回退；
- 未启动训练、AutoDL 或 Full；
- 真实测试通过后 Day 7 才从 5% 更新到 18%。
