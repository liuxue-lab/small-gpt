# Day 10 300M-token 正式预训练执行报告

## 1. 执行结论

Day 10 已完成本项目第一轮正式预训练：33,833,984 参数的 Decoder-only GPT 在 Day 9 冻结的 FineWeb-Edu Tokenized Full 上，以单卡 NVIDIA GeForce RTX 5090、CUDA BF16、micro batch 16 和 gradient accumulation 8 完成 4,578 个 optimizer updates，共消费 300,023,808 个项目模型 tokens。

正式 run：

```text
baseline-full-300m-20260813-232952
```

最终结论：

- `global_step=4,578`；
- `tokens_seen=300,023,808`；
- 目标 300,000,000 tokens，因 update 粒度产生 23,808-token overshoot；
- 初始/final train loss 为 `9.816444 / 3.830253`；
- 最后一次 validation loss/perplexity 为 `3.832705 / 46.187310`；
- 9 次 validation loss 从 step 500 到 step 4,500 严格下降；
- 4,578 个 `train_update` events 连续，无重复、回退或 resume 拼接；
- 5 个 checkpoint 均为完整原子文件，无 `.part`；
- final checkpoint 为 406,108,827 bytes；
- final checkpoint SHA-256 为 `a39f8378ebe4012afb992be451d355e814b856ffb5e690ac011758f9db614b51`；
- final checkpoint 的 model、optimizer、scheduler、TrainerState、RNG、resolved config 和数据身份全部通过 CPU 严格加载；
- trainer exit code 为 0，known error count 为 0；
- 轻量证据包与 final checkpoint 均已下载到 Windows 仓库外并通过远端/本地 SHA-256 对照；
- D34 在训练、验证、证据和下载全部完成后关机，但未释放。

Day 10 证明正式预训练过程完整、连续、可审计、可恢复，并得到一个可用于后续评估和生成的 final checkpoint。Day 10 尚未执行冻结 test split 的最终 loss/perplexity，也尚未评价开放式文本生成质量或完成消融实验。

## 2. 范围、授权与停止策略

用户对 Day 10 正式训练给出独立授权：

```text
Authorization=FORMAL_300M_APPROVED
TargetUpdates=4578
RuntimeLimit=NONE
AutomaticResume=False
FailurePolicy=STOP_AND_REPORT
CompletionAction=REMIND_SHUTDOWN_NOT_RELEASE
```

授权范围包括：

1. 在新的 RTX 5090 实例上验证克隆资产；
2. 将远端代码精确推进到 Day 9 权威提交；
3. 执行正式 dry-run；
4. 创建新的唯一 run ID；
5. 完成 4,578 updates；
6. 每 500 steps validation，每 1,000 steps checkpoint，并保存 final checkpoint；
7. 记录 stdout、metrics、PID、exit code、GPU 和磁盘证据；
8. 完成 checkpoint/metrics 严格验收；
9. 下载轻量证据和 final checkpoint；
10. 当天所有 RTX 5090 任务完成后关机，但不释放实例。

失败策略明确要求：异常时停止并汇报，不能自动 resume。实际运行没有触发异常路径，也没有 resume。

本轮明确不做：

- 不修改模型架构或正式训练超参数；
- 不重建 Full source 或 Tokenized Full；
- 不把 checkpoint、Full、tokenized payload、source cache 或 evidence 加入 Git；
- 不用 validation 代替 test；
- 不在本轮得出生成质量或消融结论；
- 不释放 AutoDL 实例。

## 3. 实例迁移与环境

### 3.1 从 C14 迁移到 D34

Day 9 的 C14 在 Day 10 启动时没有可用 GPU 槽位。为避免重建已经严格验证的数据资产，使用 AutoDL 克隆功能将系统盘和数据盘迁移到同区域 D34。

D34 首次持久性检查：

| 资产 | 结果 |
| --- | ---: |
| Full source corpus | 约 1.7 GB |
| Tokenized Full | 约 741 MB |
| Day 9 source cache | 约 2.1 GB |
| Day 9 smoke run | 约 12 KB |
| Day 9 smoke checkpoint | 约 388 MiB |
| Source manifest | present |
| Tokenized manifest | present |
| Tokenization staging | absent |

关机后再次启动 D34 时，final checkpoint bytes/SHA、两个 Full manifests、Tokenizer、metrics 和 stdout 均保持不变，`RebootPersistenceGate=PASS`。

### 3.2 D34 环境

| 项目 | 结果 |
| --- | --- |
| Host | `autodl-container-8me3mxapw7-50c69103` |
| GPU | NVIDIA GeForce RTX 5090 |
| VRAM | 32,607 MiB |
| Driver | 580.95.05 |
| CUDA Runtime | 13.0 |
| Python | 3.12.3 |
| Python executable | `/root/autodl-tmp/day08-venv/bin/python` |
| PyTorch | 2.12.1+cu130 |
| cuDNN | 92000 |
| `tokenizers` | 0.23.1 |
| CUDA available | true |
| BF16 supported | true |
| 控制台 CPU / RAM | 25 vCPU / 92 GB |
| 数据盘 | 50 GB |
| 价格 | ¥2.78/小时 |

D34 driver `580.95.05` 与 Day 9 C14 的 `580.105.08` 存在小版本差异。由于 PyTorch/CUDA/cuDNN/BF16、正式 dry-run 和真实训练全部通过，因此接受该差异；没有把 driver 小版本不同当作必须重做 Day 8 sweep 的理由。

## 4. Git 身份与离线 bundle

### 4.1 权威代码

Day 10 正式训练绑定：

```text
SourceCommit=07c22a42a696e4d2bab7e6396fcb4c417dc5f63e
SourceDirty=False
```

该提交为：

```text
07c22a4 docs: close Day 9 full data pipeline
```

它相对于 Day 9 技术提交 `23c63a6` 只增加 README、daily log 和 Day 9 报告，没有改变训练代码或配置。

### 4.2 GitHub fetch 失败

D34 首次 fetch 得到：

```text
fatal: unable to access 'https://github.com/liuxue-lab/small-gpt.git/':
GnuTLS recv error (-110): The TLS connection was non-properly terminated.
FetchExit=128
```

限定 120 秒并强制 HTTP/1.1 的重试得到 `RetryFetchExit=124`。没有用 reset、手工复制源码或伪造 tracking ref 处理。

### 4.3 Bundle fast-forward

本地 Windows 权威仓库生成 Git bundle，上传 D34 后先执行 bundle fetch：

```text
BundleFetchExit=0
FETCH_HEAD=07c22a42a696e4d2bab7e6396fcb4c417dc5f63e
```

在确认差异只有 3 个 Day 9 文档文件后执行 fast-forward：

```text
MergeExit=0
HEAD=07c22a42a696e4d2bab7e6396fcb4c417dc5f63e
worktree entries=0
```

由于 GitHub 网络仍不可用，D34 的 `origin/main` 保持 `23c63a6`，状态显示 `[ahead 1]`。正式训练身份由 exact HEAD、clean worktree、run metadata 和 checkpoint identity 共同证明；没有把陈旧 tracking ref 隐藏成同步状态。

## 5. 冻结数据与模型身份

| 项目 | 冻结值 |
| --- | --- |
| Source manifest SHA-256 | `14c69dc545838b426e29162c73132cfe444bb2cc56b72c80bb4929f3c65ca96a` |
| Tokenized manifest SHA-256 | `ce7cd91075c7c666c427e1aaa286096a7f386643f3a76de3c26ef770d6cce67e` |
| Tokenized config fingerprint | `39dab5bacdf8719bbc849e85ddcd7422cba5777fc044b437d050a49b87ab174f` |
| Tokenizer JSON SHA-256 | `b26835e02eebf777a257c4732abdd6f9732a115967d2ad839f3a1a00e45ee8c5` |
| Model config SHA-256 | `ba82957a47a92cb0021f6e56b103f12dda2b4eb8cda3927c195c96d45d5c052e` |
| Source commit | `07c22a42a696e4d2bab7e6396fcb4c417dc5f63e` |
| Source dirty | false |

Tokenized Full：

| Split | Records | Model tokens | Storage shards |
| --- | ---: | ---: | ---: |
| Train | 332,112 | 372,328,191 | 75 |
| Validation | 3,407 | 3,741,345 | 1 |
| Test | 3,330 | 3,518,409 | 1 |
| **合计** | **338,849** | **379,587,945** | **77** |

300,023,808-token 正式预算小于 train split 的 372,328,191 tokens，能够在 `data_epoch=0` 内完成，不需要跨 epoch 重复数据。

## 6. 正式 preflight

Dry-run ID：

```text
day10-full-dry-run-20260813-232617
```

Dry-run 结果：

| 项目 | 结果 |
| --- | --- |
| Project | `small-gpt-baseline` |
| Device | `cuda:0` |
| Precision | BF16 |
| Autocast | enabled |
| GradScaler | disabled |
| Parameters | 33,833,984 |
| Optimizer decay | 34 tensors / 33,816,576 params |
| Optimizer no-decay | 34 tensors / 17,408 params |
| Tokens/update | 65,536 |
| Total updates | 4,578 |
| Warmup updates | 92 |
| Next LR | `3.26086956522e-06` |
| Input/target | `(16, 512)` / `(16, 512)` |
| Causal shift | true |
| Restored step/tokens | 0 / 0 |
| Run/checkpoint paths written | no |
| Exit code | 0 |

CLI 输出的 `Batch source: pilot` 来自历史枚举名称，不能单独用于判断数据规模。以下证据共同证明实际使用 Tokenized Full：

- `--manifest data/tokenized/fineweb_edu_full/manifest.json`；
- run metadata 中相同 manifest 路径；
- Tokenized manifest SHA 为 `ce7cd910...`；
- dataset fingerprint 为 `39dab5ba...`；
- checkpoint identity 与上述值相同。

## 7. 正式训练计划

| 配置项 | 数值 |
| --- | ---: |
| Model parameters | 33,833,984 |
| Layers / heads / hidden | 8 / 8 / 512 |
| FFN hidden | 2,048 |
| Context length | 512 |
| Vocabulary | 16,384 |
| Precision | CUDA BF16 autocast |
| Micro batch | 16 |
| Gradient accumulation | 8 |
| Tokens/micro-step | 8,192 |
| Tokens/update | 65,536 |
| Target tokens | 300,000,000 |
| Total updates | 4,578 |
| Planned tokens | 300,023,808 |
| Overshoot | 23,808 |
| Warmup updates | 92 |
| Max LR | 0.0003 |
| Min LR | 0.00003 |
| Weight decay | 0.1 |
| Gradient clip | 1.0 |
| Eval interval/batches | 500 / 100 |
| Save interval | 1,000 |
| Workers | 4 |
| Pin memory | false |
| Seed | 1337 |

计划满足：

```text
4578 × 65536 = 300023808
300023808 - 300000000 = 23808
```

## 8. 启动、进程与运行身份

```text
RunID=baseline-full-300m-20260813-232952
ControlStart=2026-08-13T23:30:11+08:00
LaunchTime=2026-08-13T23:32:45+08:00
RunMetadataUTC=2026-08-13T15:32:48.611052+00:00
FinalCheckpointMTime=2026-08-13T23:55:30.931311980+08:00
FinalLogMTime=2026-08-13T23:55:31.591283290+08:00
```

正式训练使用经过碰撞门保护的唯一 run ID。wrapper 记录 stdout 和 exit code，30 秒 GPU sampler 以 wrapper 存活状态为停止条件。

启动检查中：

- PID 2463 是唯一正式 trainer，也是唯一占用 GPU 的进程；
- PID 2518～2521 是 `num_workers=4` 产生的 DataLoader workers；
- 没有第二个 trainer；
- step 1 和 step 10 正常增长；
- GPU memory 开始占用；
- `TrainExit=PENDING` 仅表示运行尚未结束，不是错误。

训练自然完成后：

```text
TrainExit=0
TrainWrapperAlive=0
GPUSamplerAlive=0
TrainingProcessesRemaining=0
GPU memory used=0 MiB
```

## 9. 训练曲线

### 9.1 关键 update

| Step | Tokens seen | Train loss | Learning rate | Grad norm | Tokens/s |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 65,536 | 9.816444 | 0.00000326087 | 7.308705 | 50,808 |
| 10 | 655,360 | 9.027508 | 0.0000326087 | 2.090027 | 224,051 |
| 20 | 1,310,720 | 8.722315 | 0.0000652174 | 1.676427 | 222,394 |
| 92 | 6,029,312 | 6.701193 | 0.0003000000 | 0.483536 | 217,505 |
| 500 | 32,768,000 | 5.273014 | 0.0002945509 | 0.781865 | 226,532 |
| 1,000 | 65,536,000 | 4.582271 | 0.0002736588 | 1.080801 | 221,686 |
| 1,500 | 98,304,000 | 4.272860 | 0.0002395739 | 0.859399 | 220,898 |
| 2,000 | 131,072,000 | 4.028144 | 0.0001964347 | 0.788449 | 215,666 |
| 2,500 | 163,840,000 | 3.988490 | 0.0001494788 | 0.783063 | 242,893 |
| 3,000 | 196,608,000 | 3.908527 | 0.0001044074 | 0.744632 | 224,333 |
| 3,500 | 229,376,000 | 3.840983 | 0.0000666929 | 0.756915 | 227,653 |
| 4,000 | 262,144,000 | 3.845382 | 0.0000409142 | 0.749969 | 224,183 |
| 4,500 | 294,912,000 | 3.807149 | 0.0000302014 | 0.770310 | 225,460 |
| 4,578 | 300,023,808 | 3.830253 | 0.0000300000 | 0.814144 | 225,579 |

初始 update 包含 CUDA/DataLoader 冷启动，因此 step 1 的 throughput 不能代表稳定速度。step 2～4,578 的 update throughput 中位数约 223,595 tokens/s。

train loss 从 `9.816444` 降至 `3.830253`，下降约 60.98%。单 batch train loss 会随 batch 内容波动，因此最终 step 高于 step 4,500 并不表示训练反转。

### 9.2 学习率与梯度

- step 1 LR 为 `3e-4 / 92`；
- step 92 精确到达 max LR `3e-4`；
- step 93～4,578 按 cosine 非增衰减；
- step 4,578 到达 min LR `3e-5`；
- 所有 4,578 个 LR 和 grad norm 均为有限数；
- 日志中的 grad norm 是 clip 前值，因此超过 1.0 的日志值不等于裁剪失败。

## 10. Validation

每 500 updates 运行一次固定 validation；每次 100 batches / 819,200 tokens。

| Step | Tokens seen | Validation loss | Perplexity | Eval seconds |
| ---: | ---: | ---: | ---: | ---: |
| 500 | 32,768,000 | 5.235283 | 187.782316 | 1.345 |
| 1,000 | 65,536,000 | 4.583346 | 97.841230 | 1.361 |
| 1,500 | 98,304,000 | 4.259436 | 70.770092 | 1.401 |
| 2,000 | 131,072,000 | 4.105180 | 60.653632 | 1.370 |
| 2,500 | 163,840,000 | 4.004211 | 54.828556 | 1.395 |
| 3,000 | 196,608,000 | 3.935036 | 51.163982 | 1.369 |
| 3,500 | 229,376,000 | 3.884580 | 48.646510 | 1.345 |
| 4,000 | 262,144,000 | 3.852903 | 47.129668 | 1.342 |
| 4,500 | 294,912,000 | 3.832705 | 46.187310 | 1.369 |

验证结论：

- 9 个 validation loss 严格下降；
- 第一次到最后一次 validation loss 下降约 26.79%；
- perplexity 下降约 75.40%；
- final train loss 与最后 validation loss 接近；
- validation tokens 没有计入 `tokens_seen`；
- validation 后 scheduler 和训练 step 连续。

这些证据说明训练过程中没有观察到 validation 曲线恶化，但不能代替冻结 test split 的最终评估。validation 只覆盖每次 819,200 tokens，不能据此声称模型具备通用生成质量。

## 11. Metrics 连续性

`metrics.jsonl`：

```text
JSONObjects=4593
run_start=1
train_update=4578
evaluation=9
checkpoint=5
```

完整 validator 对全部记录执行以下检查：

- 首事件为 fresh `run_start`；
- `resume_checkpoint=null`；
- `stop_at_step=4578`；
- train step 精确为 1～4,578；
- 每步 `tokens=65,536`；
- 每步 `tokens_seen=step×65,536`；
- 每步 8 micro steps / 128 samples；
- device/precision 始终为 `cuda:0 / bf16`；
- train loss、grad norm、LR、throughput 和 elapsed time 全部有限；
- warmup 严格递增到 step 92；
- cosine decay 非增到 final step；
- evaluation steps 为 500～4,500；
- checkpoint steps 为 1,000、2,000、3,000、4,000、4,578；
- final event 为 step 4,578 checkpoint；
- 5 个 checkpoint 路径存在且 bytes 与记录一致。

结果：

```text
MetricsContinuity=PASS
```

## 12. 性能、GPU、磁盘与费用

### 12.1 训练时间

| 项目 | 结果 |
| --- | ---: |
| Launch 到 final log | 约 1,366.59 秒 / 22 分 47 秒 |
| 4,578 个 update elapsed 合计 | 1,340.724801 秒 |
| 9 次 evaluation 合计 | 12.297287 秒 |
| 5 次 checkpoint save 合计 | 5.139892 秒 |
| Aggregate update throughput | 223,777.323900 tokens/s |

### 12.2 GPU 采样

GPU sampler 从 23:35:16 到 23:55:20 每 30 秒记录，共 41 个样本。它没有覆盖最初约 2.5 分钟，因此统计只代表中后段稳定状态。

| 指标 | 最小 | 平均 | 中位数 | 最大 |
| --- | ---: | ---: | ---: | ---: |
| Utilization | 76% | 82.37% | 83% | 88% |
| Memory used | 6,295 MiB | 6,295 MiB | 6,295 MiB | 6,295 MiB |
| Temperature | 74°C | 75.39°C | 75°C | 77°C |
| Power | 427.35 W | 472.74 W | 471.81 W | 495.17 W |

训练完成后 GPU 为 0 MiB / 0%，没有残留 compute process。

### 12.3 磁盘

| 时间 | 已用 | 可用 | 使用率 |
| --- | ---: | ---: | ---: |
| 正式 run 前 | 6.2 GB | 44 GB | 13% |
| 正式 run 后 | 8.1 GB | 42 GB | 17% |

最终资产：

| 资产 | 大小 |
| --- | ---: |
| Full source | 约 1.7 GB |
| Tokenized Full | 约 741 MB |
| Source cache | 约 2.1 GB |
| Run artifacts | 约 1.5 MB |
| 5 个 checkpoints | 约 1.9 GB |

### 12.4 费用边界

按 ¥2.78/小时与 22 分 47 秒正式 trainer 墙钟估算，正式 run 本身约 ¥1.06。该数字不包含：

- D34 克隆与开机等待；
- Git bundle、环境与 SHA preflight；
- dry-run；
- 训练后重启持久性验证；
- checkpoint CPU load、metrics validator；
- evidence/checkpoint 下载时间。

总费用应以 AutoDL 账单为准，本报告没有把计算段估算伪装成完整账单。

## 13. Checkpoint 验收

### 13.1 文件与 SHA

| Step | Tokens seen | Bytes | SHA-256 |
| ---: | ---: | ---: | --- |
| 1,000 | 65,536,000 | 406,108,827 | `846565575f42253474b0e21ab162e97144ff1acee56d4efd51cecf0794ed5657` |
| 2,000 | 131,072,000 | 406,108,827 | `686fac55259d725452fe7a86e1f6ef450366412546a9a573965351cf05e2ea9f` |
| 3,000 | 196,608,000 | 406,108,827 | `3c3e8af0cb147cad49feb476c9ae213f368f1ef7fc24c6c9c0fa6c4a2fe9f895` |
| 4,000 | 262,144,000 | 406,108,827 | `6dbb5b1ffbfe4de60caabab45d11f883a90f4773457a1970483ef2f7c0b5d9c4` |
| 4,578 | 300,023,808 | 406,108,827 | `a39f8378ebe4012afb992be451d355e814b856ffb5e690ac011758f9db614b51` |

5 个文件均为正式 `.pt`，没有 `.part`。final checkpoint 是训练自然完成时保存，不是从 step 4,000 手工复制或改名。

### 13.2 Final checkpoint schema

顶层结构：

```text
created_at_utc
format_name
identity
model_state_dict
optimizer_state_dict
resolved_config
rng_state
scaler_state_dict
scheduler_state_dict
schema_version
trainer_state
```

严格 CPU load 结果：

| 检查 | 结果 |
| --- | --- |
| Format/schema | PASS |
| Model entries | 69 |
| Unique storages | 68 |
| Unique parameters | 33,833,984 |
| Model tensors finite | PASS |
| Embedding/head tied values | PASS |
| Optimizer states | 68 |
| Optimizer groups | 34 decay / 34 no-decay |
| Optimizer step | 4,578 |
| Optimizer tensors finite | PASS |
| Scheduler | total 4,578 / warmup 92 / final LR 3e-5 |
| TrainerState | step 4,578 / micro 36,624 / tokens 300,023,808 / samples 585,984 |
| Python/NumPy/Torch CPU/CUDA RNG | PASS |
| Resolved plan | PASS |
| Full/Tokenizer/source identity | PASS |
| BF16 scaler absent | expected / PASS |

守恒关系：

```text
4578 × 8 = 36624 micro steps
36624 × 16 = 585984 samples
4578 × 65536 = 300023808 tokens
```

结果：

```text
FailedChecks=NONE
FinalCheckpointInternalState=PASS
```

## 14. 证据、下载与灾备

### 14.1 轻量证据包

```text
File    = small-gpt-day10-evidence-baseline-full-300m-20260813-232952.tar.gz
Bytes   = 301228
SHA256  = a24371915c8de0b34eeabb23b62036b97bccbc83629981f6eccb22378483eb20
Entries = 31
```

内容包括：

- run/authorization/Git/manifest identity；
- before/after GPU 与磁盘快照；
- stdout、GPU CSV、metrics JSONL；
- resolved config 与 run metadata；
- checkpoint inventory 和 SHA 摘要；
- checkpoint schema 与 final verification；
- metrics schema 与 continuity verification；
- completion summary；
- `SHA256SUMS.txt`。

明确排除：

- 1.9 GB checkpoints；
- Full source payload；
- Tokenized Full `.bin/.idx`；
- source cache；
- virtual environment；
- credentials 或 tokens。

远端安全门：

```text
EvidenceSymlinks=0
ForbiddenPayloadFiles=0
OversizedEvidenceFiles=0
SecretPatternMatches=0
UnsafeArchiveEntries=0
EvidenceSafetyGate=PASS
ArchiveSafetyGate=PASS
```

Windows 下载验证：

```text
LocalEvidenceBytes=301228
LocalEvidenceSHA256=a24371915c8de0b34eeabb23b62036b97bccbc83629981f6eccb22378483eb20
LocalArchiveEntries=31
LocalArchiveReadExit=0
LocalEvidenceVerification=PASS
```

解压后：

```text
EvidenceFileCount=30
SHA256ManifestEntries=29
SHA256Failures=0
LocalExtractedEvidenceVerification=PASS
```

### 14.2 Final checkpoint 本地备份

JupyterLab 无法进入 `checkpoints/` 目录时，没有复制 406 MB 文件或改变原文件。远端在 `/root/autodl-tmp` 创建与 final checkpoint 相同 inode 的临时硬链接，下载并验证后再精确 `unlink`。原 checkpoint 最终 link count 回到 1。

Windows 本地验证：

```text
LocalCheckpointBytes=406108827
LocalCheckpointSHA256=a39f8378ebe4012afb992be451d355e814b856ffb5e690ac011758f9db614b51
LocalFinalCheckpointVerification=PASS
```

仓库外灾备目录：

```text
D:\model-backups\small-gpt\day10\baseline-full-300m-20260813-232952
```

该目录包含：

- `step-00004578.pt`；
- 301,228-byte evidence archive；
- 解压后的 evidence 文件。

这些文件没有放入 `D:\code\small-gpt`，不会提交 Git。

## 15. 问题与禁止重复

| 问题 | 根因/风险 | 处理 |
| --- | --- | --- |
| C14 无 GPU slot | 原实例无法启动 GPU | 克隆到 D34，先验证数据身份 |
| D34 GitHub GnuTLS/timeout | 远端 GitHub 网络不稳定 | 本地权威 bundle + diff gate + ff-only |
| `origin/main` 在 D34 落后 | fetch 失败导致 tracking ref 陈旧 | 记录真实状态；以 HEAD/clean/metadata 证明 source |
| driver 小版本变化 | 克隆目标宿主机不同 | 重新验证 PyTorch/CUDA/BF16 和 dry-run |
| `Batch source: pilot` | 历史 CLI enum 名称 | 以 Full manifest SHA/fingerprint 判断数据 |
| pgrep 显示 5 个 Python | 1 trainer + 4 DataLoader workers | 用 GPU process、PID/PPID 和 workers 配置区分 |
| grad norm 偶尔大于 1 | 日志记录 clip 前范数 | 不误判 gradient clipping 失效 |
| JupyterLab 无法进入 checkpoint 目录 | 文件浏览器目录访问异常 | 临时硬链接下载，验证后 unlink |
| 长 inline Python 被粘贴损坏 | 复制过程中字符串被改写 | SyntaxError 阶段没有修改资产；改短 validator |
| 证据命令出现 `>` prompt | 一条复制命令未闭合 | Ctrl+C，短命令验证已生成 archive，不重复打包 |
| 训练结束后过早关机 | 仍需 checkpoint/evidence/download 验收 | 重启做持久性门；以后当天所有 GPU 任务后再关机 |

禁止重复：

- 不重复运行 run ID `baseline-full-300m-20260813-232952`；
- 不从 step 0 再训练同一 300M budget；
- 不为了演示 resume 中断或修改已完成的正式 run；
- 不把 `pilot` 标签误判为实际数据 manifest；
- 不因 D34 tracking ref 陈旧而 reset 已验证的 exact HEAD；
- 不删除远端 final checkpoint，直到灾备策略明确；
- 不把 `.pt`、evidence、Full 或 tokenized payload 加入 Git；
- 不把 validation perplexity 写成 test perplexity；
- 不在生成入口未验收前声称模型生成质量达标；
- 不释放 D34，除非大数据和所有必要权重已有明确独立灾备且用户再次授权。

## 16. 测试与文档收口

Day 10 正式 run 使用 Day 9 最终冻结训练源码 `07c22a4`，没有在 GPU 主机上修改代码或配置。训练前所有 environment、identity、dry-run gates 通过；训练后 checkpoint 与 metrics 使用独立 validator 全量验证。

本地文档收口回归：

```text
551 passed in 68.12s
Exit code: 0
```

本次 Git 只应包含：

- `README.md`；
- `reports/daily-log.md`；
- `reports/day-10-pretraining-report.md`。

不应包含：

- checkpoint；
- evidence archive 或解压目录；
- run logs；
- Full/Tokenized Full；
- source cache；
- 下载用临时硬链接；
- Python cache 或临时 patch 文件。

## 17. 最终验收清单

- [x] 获得独立 300M-token 授权；
- [x] 失败时停止且不自动 resume 的策略明确；
- [x] C14 数据迁移到 D34 后身份不变；
- [x] D34 Git HEAD 精确为 `07c22a4`；
- [x] source dirty 为 false；
- [x] Full source manifest SHA 通过；
- [x] Tokenized Full manifest SHA/fingerprint 通过；
- [x] Tokenizer SHA 通过；
- [x] 正式 dry-run 不写 run/checkpoint；
- [x] 唯一正式 run ID；
- [x] 4,578 updates 连续；
- [x] 300,023,808 tokens 守恒；
- [x] warmup/cosine schedule 正确；
- [x] 所有 train loss/LR/grad norm finite；
- [x] 9 次 validation finite 且 step 正确；
- [x] 5 个 checkpoint 完整且无 `.part`；
- [x] final checkpoint SHA 记录；
- [x] final checkpoint CPU strict load 通过；
- [x] optimizer/scheduler/TrainerState/RNG 完整；
- [x] metrics 4,593 events 全量连续性通过；
- [x] trainer exit code 0；
- [x] 训练结束后 trainer、workers 和 GPU sampler 全部退出；
- [x] 轻量 evidence 不含大 payload 或 secret；
- [x] evidence 远端/本地 bytes、SHA、entries 一致；
- [x] final checkpoint 远端/本地 bytes、SHA 一致；
- [x] 本地仓库外灾备完成；
- [x] D34 最终关机；
- [x] D34 未释放；
- [x] 本地完整回归 `551 passed in 68.12s`；
- [ ] test split 最终评估尚未执行；
- [ ] 文本生成尚未执行；
- [ ] 消融实验尚未执行。

## 18. 下一阶段

Day 10 的正确后续不是重复训练，而是使用不可变 final checkpoint 做评估与生成：

1. 在本地或 GPU 实例只读加载 `step-00004578.pt`；
2. 验证模型构建、state dict strict load 和权重 tying；
3. 在冻结 validation/test split 上分别计算指标，明确 split 名称；
4. 实现或验收文本生成入口；
5. 使用固定 prompts 与 seed 对 temperature、top-k、top-p 做可复现实验；
6. 保留 greedy 结果作为基线；
7. 记录生成退化、重复、上下文截断和特殊 token 行为；
8. 设计至少一个只改变单一变量的消融；
9. 评估和生成期间不覆盖 final checkpoint；
10. 权重、生成大文件和证据继续保存在 Git 之外。

在 test 和生成完成前，项目状态应准确写为：正式预训练已完成，最终质量评估与生成尚未完成。
