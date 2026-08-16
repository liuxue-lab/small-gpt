# Day 12 Dropout 0.1 单变量消融执行报告

## 1. 执行结论

Day 12 已完成 Small GPT 的首个正式训练消融。实验严格比较：

- Control：`model.dropout=0.0`；
- Treatment：`model.dropout=0.1`；
- 唯一实验变量：`model.dropout`；
- 两组均使用 seed `1337`、4,578 updates、65,536 tokens/update 和 300,023,808 tokens；
- 模型结构、参数量、数据、Tokenizer、batch、optimizer、scheduler、评估协议和生成协议保持不变；
- Treatment 从随机初始化开始，没有 resume Control checkpoint；
- Treatment 完成完整 frozen validation、一次性 frozen test 和冻结生成套件；
- 结论仅限单 seed 描述性工程消融，不声明统计显著性。

正式 Treatment run：

```text
day12-dropout-01-full-300m-20260815-182244
```

最终 Treatment checkpoint：

```text
File=step-00004578.pt
Bytes=406108827
SHA256=29b15304f7e6f62b29b1ba4f4b5b6f591d4dcbc7e336ef79bd64486468fcb3ad
GlobalStep=4578
TokensSeen=300023808
```

最重要的结果：

| 指标 | Control | Treatment | Treatment - Control | 结论 |
| --- | ---: | ---: | ---: | --- |
| Final train loss | 3.830253452063 | 4.015954345465 | +0.185700893402 | Control 更低 |
| Final training validation loss | 3.832705090046 | 3.909346697330 | +0.076641607285 | Control 更低 |
| Frozen full validation loss | 3.819582318483 | 3.893882904755 | +0.074300586272 | Control 更低 |
| Frozen full validation perplexity | 45.585164263392 | 49.101172028751 | +3.516007765359 | Control 更低 |
| Frozen test loss | 3.830240146369 | 3.905327702658 | +0.075087556289 | Control 更低 |
| Frozen test perplexity | 46.073601313835 | 49.666353041570 | +3.592751727735 | Control 更低 |
| Aggregate update throughput | 223,777.323900 | 215,277.624224 | -8,499.699676 | Control 更高 |
| Update elapsed seconds | 1,340.724801 | 1,393.659973 | +52.935172 | Control 更短 |

Control 在 9/9 个匹配的训练过程 validation 节点上都取得更低 loss。Treatment 的 frozen validation loss 相对增加 `1.945254%`，frozen test loss 相对增加 `1.960388%`；对应 perplexity 分别增加 `7.713053%` 和 `7.797853%`。

冻结生成对比没有证明 Treatment 改善总体生成质量。Treatment 的 greedy 样本表面多样性略高，但 Control 与 Treatment 的 6/6 greedy 样本都进入明显循环；Treatment 在四种随机采样策略上的平均样本唯一 token 比例均低于 Control，跑题、幻觉、语义断裂和指令执行失败仍然存在。

正式工程决策：

```text
QuantitativeAblationOutcome=CONTROL_BETTER_ON_FROZEN_VALIDATION_AND_TEST
GenerationComparisonOutcome=NO_CLEAR_TREATMENT_IMPROVEMENT
OverallAblationDecision=RETAIN_CONTROL_DROPOUT_0_0
StatisticalScope=DESCRIPTIVE_SINGLE_SEED
StatisticalSignificanceClaim=False
```

在当前 34M 参数、300M-token、单 seed 预算下，不采用 `dropout=0.1` 替换基线，保留 `model.dropout=0.0`。

## 2. 日期、目标与问题边界

执行日期：2026-08-15 至 2026-08-16。

Day 12 的工程目标不是通过调整采样参数让少数文本“看起来更好”，而是完成一次真正的新训练 run，并在严格单变量条件下回答：加入 `0.1` dropout 是否能改善泛化或缓解生成重复。

冻结假设：

> Dropout 0.1 可能改善泛化或生成重复控制，但也可能在固定 300M-token 预算下减慢优化。

本轮能够回答：

- 在固定训练预算下，`dropout=0.1` 是否优于当前 `dropout=0.0`；
- 两组训练曲线、完整 validation/test 和固定生成协议是否一致支持某一方向；
- 当前差生成质量能否通过这一项正则化改动明显修复。

本轮不能回答：

- 多 seed 下结果是否稳定；
- 差异是否具有统计显著性；
- 增大模型、增加训练 token、改变数据配比或指令微调哪一种最有效；
- 不同 GPU/PyTorch/CUDA 环境能否产生 bitwise 相同生成；
- `dropout=0.1` 在更长训练预算下是否会反超。

## 3. 授权与操作边界

用户明确批准协议 `day12-dropout-01-ablation-v1`，并授权：

1. 本地创建 Treatment 配置、合同 validator 和测试；
2. 运行 dry-run、backward、定向测试和完整回归；
3. 创建本地功能提交并 push；
4. 在授权 RTX 5090 实例上从随机初始化训练 Treatment；
5. 训练完成后进行完整 validation、一次性 test 和固定 generation suite；
6. 任务全部结束后关机，但不释放实例。

明确禁止：

- 不 resume Baseline；
- 不改变第二个训练变量；
- 不覆盖 Baseline checkpoint；
- 不重建或替换 Tokenizer、Full 数据；
- 不用 test 反复调参；
- 不覆盖正式 generation suite；
- 不将跨机器结果描述为 bitwise 复现；
- 不因下载困难而删除云端唯一证据；
- 不释放任何未获释放授权的实例。

## 4. 冻结消融合同

合同文件：`configs/day12_ablation_contract.json`。

| 项目 | 冻结值 |
| --- | --- |
| Protocol ID | `day12-dropout-01-ablation-v1` |
| Contract status | `FROZEN` |
| Contract fingerprint | `35e70b57730b6cc8952c2b9a8dae49137aee7c23ac87b85db412a417ffbd7216` |
| Source mode | `CURRENT_MAIN` |
| Control config | `configs/baseline.yaml` |
| Treatment config | `configs/ablation_dropout_01.yaml` |
| Changed field | `model.dropout` |
| Control value | `0.0` |
| Treatment value | `0.1` |
| Allowed experimental diff count | 1 |
| Statistical scope | `single_seed_descriptive_engineering_ablation` |

### 4.1 精确单变量证明

Windows 原始文件核验显示：

```text
BaselineBytes=1258
TreatmentBytes=1258
ByteDiffCount=1
ChangedByteOffset=423
ChangedLineCount=1
ChangedLineNumber=29
ChangedFrom=dropout: 0.0
ChangedTo=dropout: 0.1
ExactByteDiff=True
ExactLineDiff=True
```

Baseline 配置没有被编辑：

```text
BaselineSHA256=ca8524c425e1e5e3a600de5773f9a526ef3674741040635bee91fe31f4b24c0e
TreatmentSHA256=35bbfd4624f5a4e8c224628e4995dd830112747b4686b8bcf3318debc869d4fb
```

### 4.2 保持不变的核心条件

| 条件 | Control 与 Treatment |
| --- | --- |
| Seed | 1337 |
| Model parameters | 33,833,984 |
| Layers / heads / hidden | 8 / 8 / 512 |
| FFN / context / vocab | 2,048 / 512 / 16,384 |
| Weight tying | true |
| Micro batch / accumulation | 16 / 8 |
| Tokens/update | 65,536 |
| Total updates | 4,578 |
| Planned/actual tokens | 300,023,808 |
| Warmup updates | 92 |
| Optimizer | AdamW |
| Peak/min LR | 3e-4 / 3e-5 |
| Weight decay | 0.1 |
| Betas / epsilon | 0.9, 0.95 / 1e-8 |
| Gradient clip | 1.0 |
| Training precision | CUDA BF16 |
| Full manifest SHA-256 | `ce7cd91075c7c666c427e1aaa286096a7f386643f3a76de3c26ef770d6cce67e` |
| Dataset fingerprint | `39dab5bacdf8719bbc849e85ddcd7422cba5777fc044b437d050a49b87ab174f` |
| Tokenizer SHA-256 | `b26835e02eebf777a257c4732abdd6f9732a115967d2ad839f3a1a00e45ee8c5` |
| Frozen evaluator | sequential non-overlapping full split |
| Generation protocol | `day11-baseline-generation-v1` |

合同 validator 的关键输出：

```text
ObservedExperimentalDiffCount=1
ObservedExperimentalDiffPaths=model.dropout
ParameterDelta=0
ControlTokensPerUpdate=65536
TreatmentTokensPerUpdate=65536
ControlTotalUpdates=4578
TreatmentTotalUpdates=4578
ControlPlannedTokens=300023808
TreatmentPlannedTokens=300023808
HeldConstantsMatch=True
Day12AblationContract=PASS
```

## 5. 本地实现、测试与 Git 功能提交

### 5.1 新增文件

Day 12 功能提交只增加四个文件：

```text
configs/ablation_dropout_01.yaml
configs/day12_ablation_contract.json
scripts/check_ablation_contract.py
tests/test_ablation_contract.py
```

变更量：1,438 insertions，无既有文件修改。

### 5.2 本地校验

定向校验：

```text
29 passed in 2.09s
Day12StageC4TargetedValidation=PASS
```

本地 CUDA backward：

```text
Configuration=configs/ablation_dropout_01.yaml
Device=cuda
Seed=1337
Parameters=33833984
TrainableParameters=33833984
InputShape=(1,64)
LogitsShape=(1,64,16384)
Loss=9.765543
GradientTensors=68
NonzeroGradients=68
GradientsFinite=True
Day12StageD2LocalDryRunAndBackward=PASS
```

完整回归：

```text
657 passed in 37.47s
Day11PassedReference=628
ExpectedAddedTests=29
PassedDeltaFromDay11=29
Day12StageD3FullRegression=PASS
```

dry-run 没有创建 run 或 checkpoint 目录。第一次输出 gate 把 `cuda:0` 错误地与 `cuda` 做精确文本比较而停止；模型 dry-run 本身成功。修正 gate 后，没有通过降低设备要求掩盖问题，而是用真实 CUDA backward 独立确认 Treatment 可执行。

### 5.3 功能提交与远端闭环

```text
Commit=e66a2bc3a4218ab3b28ec867d70327f9ac9f369e
Subject=feat: add verified Day 12 dropout ablation
Parent=0c1f0040d5bae891e4445b4039cf842990755e7c
```

PowerShell 将 `git push` 的标准错误流包装成 `NativeCommandError`，但远端实际已收到提交。后续没有盲目重复 push，而是重新查询 remote、fetch 并核对：

```text
LocalHead=e66a2bc3a4218ab3b28ec867d70327f9ac9f369e
TrackingHead=e66a2bc3a4218ab3b28ec867d70327f9ac9f369e
RemoteHead=e66a2bc3a4218ab3b28ec867d70327f9ac9f369e
AheadOriginMain=0
BehindOriginMain=0
ThreeWayIdentity=True
Day12StageD7GitClosure=PASS
```

## 6. F14 云端准备与离线源码同步

### 6.1 F14 身份

| 项目 | 值 |
| --- | --- |
| Instance | F14 |
| Hostname | `autodl-container-cvymrkm86b-f0a4f3ac` |
| GPU | NVIDIA GeForce RTX 5090 |
| GPU memory | 32,607 MiB |
| Python | 3.12.3 |
| PyTorch | 2.12.1+cu130 |
| CUDA runtime | 13.0 |
| Tokenizers | 0.23.1 |
| BF16 | supported |

F14 克隆持久化了 Baseline checkpoint、Tokenizer、Full 数据和 manifests。首次只读检查确认 GPU compute process 与 Small GPT process 都为 0，数据盘剩余空间超过 40 GiB，核心 artifact SHA 全部匹配。

### 6.2 GitHub 不可达与离线 bundle

F14 无法连接 GitHub，源码当时停在 `b8f8fc8`。没有伪造 tracking ref，也没有跳过 source identity。Windows 从当前完整历史创建：

```text
Bundle=small-gpt-day12-main-e66a2bc.bundle
Bytes=805927
SHA256=e17490a1c46a4d3ebfc37bf05875aff2a6fd94132061f9634f6c2598a87bebe1
BundleHead=e66a2bc3a4218ab3b28ec867d70327f9ac9f369e refs/heads/main
BundleCompleteHistory=True
```

F14 验证 bundle 后，先 fetch 到 `refs/offline/day12-main-e66a2bc`，再以 fast-forward 更新 main。原有 `origin/main` 没有被伪装为已同步。

### 6.3 云端回归与 Full dry-run

```text
657 passed, 2 warnings in 6.24s
FullRegressionExit=0
```

两个 warning 是 Python 3.12 多线程进程中 `fork()` 的弃用提示；对应 DataLoader 测试通过。

Full 数据 dry-run：

```text
ResolvedDevice=cuda:0
Precision=bf16
BatchSource=pilot
Parameters=33833984
TokensPerUpdate=65536
TotalUpdates=4578
WarmupUpdates=92
RestoredGlobalStep=0
RestoredTokensSeen=0
SourceCommit=e66a2bc3a4218ab3b28ec867d70327f9ac9f369e
SourceDirty=False
SampleInputShape=(16,512)
SampleTargetShape=(16,512)
NoWrite=True
```

dry-run 指向 Full manifest，但不会写 run、日志或 checkpoint。

## 7. 正式 Treatment 训练

### 7.1 Run identity

```text
ProtocolID=day12-dropout-01-ablation-v1
RunID=day12-dropout-01-full-300m-20260815-182244
FrozenAt=2026-08-15T18:22:44+08:00
SourceCommit=e66a2bc3a4218ab3b28ec867d70327f9ac9f369e
FreshInitialization=True
ResumeArgumentPresent=False
Seed=1337
Dropout=0.1
```

训练进程在 F14 上启动，命令显式使用 Treatment config、Full manifest、CUDA、BF16 和 pilot batch source。启动日志确认：

```text
RestoredGlobalStep=0
RestoredTokensSeen=0
DataBatchesConsumed=0
DataSamplesConsumed=0
NextLearningRate=3.26086956522e-06
```

这证明 Treatment 不是从 Baseline 权重或训练状态继续。

### 7.2 完成状态

```text
FinalGlobalStep=4578
FinalTokensSeen=300023808
EvaluationsThisRun=9
CheckpointsThisRun=5
TrainingProcessComplete=True
```

最终 update：

```text
step=4578
train_loss=4.015954345465
learning_rate=3e-05
grad_norm=0.662381887436
tokens_per_second=210093.539130
tokens_seen=300023808
```

### 7.3 Metrics 完整性

`metrics.jsonl`：

| 项目 | 值 |
| --- | ---: |
| Bytes | 1,462,596 |
| SHA-256 | `139fe92f73b6bc27467c847f7c9a27faf60d91c1ee2579381ee940de0929349c` |
| Total records | 4,593 |
| `run_start` | 1 |
| `train_update` | 4,578 |
| `evaluation` | 9 |
| `checkpoint` | 5 |
| Total micro steps | 36,624 |
| Total samples | 585,984 |
| All numeric values finite | true |

严格审计确认：

- step 1～4,578 连续；
- 每步正好 65,536 tokens；
- `tokens_seen=step×65,536`；
- learning-rate schedule 与配置完全匹配；
- evaluation 正好位于 500～4,500；
- checkpoint 正好位于 1,000/2,000/3,000/4,000/4,578；
- event ordering 有效；
- run start 的 `resume_checkpoint=null`；
- 所有指标 finite；
- metrics、checkpoint 和 smoke JSON 在审计前后 hash 稳定。

时间与吞吐：

```text
UpdateElapsedSeconds=1393.659973168280
EvaluationElapsedSeconds=11.599804
CheckpointSaveSeconds=3.462183
AggregateUpdateThroughput=215277.624224
```

### 7.4 Checkpoints

```text
step-00001000.pt
step-00002000.pt
step-00003000.pt
step-00004000.pt
step-00004578.pt
```

五个文件均为 406,108,827 bytes。最终 Treatment checkpoint SHA-256 为：

```text
29b15304f7e6f62b29b1ba4f4b5b6f591d4dcbc7e336ef79bd64486468fcb3ad
```

Control final checkpoint SHA-256 仍为：

```text
a39f8378ebe4012afb992be451d355e814b856ffb5e690ac011758f9db614b51
```

两者大小相同但 hash 不同，这是两次独立训练权重的正确结果。

## 8. F14 到 E85 的迁移

训练结束后 F14 出现机器问题，用户将实例克隆到 E85：

| 项目 | 值 |
| --- | --- |
| Instance | E85 |
| Hostname | `autodl-container-6jq3ca2nm5-edcb6602` |
| GPU | NVIDIA GeForce RTX 5090 |
| Source HEAD | `e66a2bc3a4218ab3b28ec867d70327f9ac9f369e` |

迁移后先验证而不是重跑训练：

```text
TreatmentMetricsSHA256=139fe92f73b6bc27467c847f7c9a27faf60d91c1ee2579381ee940de0929349c
TreatmentCheckpointSHA256=29b15304f7e6f62b29b1ba4f4b5b6f591d4dcbc7e336ef79bd64486468fcb3ad
BoundedSmokeJSONSHA256=e1f198cdda1b012e5b98ce6d942d8171879b5709a3bef53383354de25717e9be
FormalTrainingRerunRequired=False
```

完整 metrics audit 在 E85 上再次通过。F14 随后关机但未释放；所有后续只读评估与证据整理在 E85 完成。

## 9. Frozen full validation 与一次性 test

### 9.1 Validation

Treatment frozen full validation：

| 项目 | 值 |
| --- | ---: |
| Full split | true |
| Available/evaluated batches | 457 / 457 |
| Total windows | 7,307 |
| Evaluated tokens | 3,741,184 |
| Trailing tokens discarded | 160 |
| Loss | 3.893882904755 |
| Perplexity | 49.101172028751 |
| Elapsed seconds | 5.834124 |

证据文件：

```text
day12-dropout-01-full-300m-20260815-182244.frozen-full-validation.json
Bytes=3342
SHA256=6d8a03c86206ffded1040b7ed512303bd7ab95dd099ce400e09ea9222c944b44
```

### 9.2 One-shot test

只有 validation、checkpoint、data 和 evaluator identity 全部通过后，才执行一次正式 test：

| 项目 | 值 |
| --- | ---: |
| Full split | true |
| Available/evaluated batches | 430 / 430 |
| Total windows | 6,871 |
| Evaluated tokens | 3,517,952 |
| Trailing tokens discarded | 456 |
| Loss | 3.905327702658 |
| Perplexity | 49.666353041570 |
| Elapsed seconds | 5.455950 |

证据文件：

```text
day12-dropout-01-full-300m-20260815-182244.frozen-test.json
Bytes=3337
SHA256=6332dcc1f3c0f995ebcc77a73b38c95a832b3aaa29853430cdbbe2b2879d5653
```

```text
OneShotFormalTestConsumed=True
FormalTestMustNotBeRerun=True
```

## 10. Treatment 固定生成套件

Treatment 复用 Day 11 的冻结生成协议：

```text
ProtocolID=day11-baseline-generation-v1
ProtocolFileSHA256=bb6fd24c2d277d4369fcd21d551ff7023484b62d7bccfd7103beca6c71a8ce4a
ProtocolFingerprint=e60f3fb381b3efd8f00bd3f3fc3071c11645c78977dc7c6c40e0fd124b6d1ed0
PromptCount=6
DecodingCount=5
ExpectedSamples=30
MaxNewTokens=64
StochasticSeed=1337
```

正式 Treatment generation 使用 E85 RTX 5090、PyTorch 2.12.1+cu130 和 CUDA FP32：

```text
CompletedSamples=30
ArtifactsLoadedOnce=True
ModelLoads=1
GeneratedTokens=1920
ForwardPasses=1920
ContextCropEvents=0
EOSStops=0
MaxTokenStops=30
WallElapsedSeconds=6.219387
```

证据：

| 文件 | Bytes | SHA-256 |
| --- | ---: | --- |
| `manifest.json` | 5,318 | `03160e2196e5c0cf399698efe9fe60cae11c1fecf66b599e46364045e03cafdf` |
| `samples.jsonl` | 84,568 | `1f0a86d92022caa6fd50a00df59d19f46123e416e5b7c4225847230a3fc81f58` |

正式目录不允许覆盖。所有样本都保留 prompt、token IDs、解码参数、运行时、停止原因和 continuation text。

## 11. 训练过程对比

两组在完全相同的 evaluation steps 上各评估 100 batches / 819,200 tokens：

| Step | Control loss | Treatment loss | Treatment - Control | Winner |
| ---: | ---: | ---: | ---: | --- |
| 500 | 5.235283398628 | 5.284546904564 | +0.049263505936 | Control |
| 1,000 | 4.583346061707 | 4.689899873734 | +0.106553812027 | Control |
| 1,500 | 4.259436483383 | 4.338398759365 | +0.078962275982 | Control |
| 2,000 | 4.105179522038 | 4.175107874870 | +0.069928352833 | Control |
| 2,500 | 4.004211153984 | 4.072017548084 | +0.067806394100 | Control |
| 3,000 | 3.935035808086 | 4.004207334518 | +0.069171526432 | Control |
| 3,500 | 3.884580061436 | 3.957169308662 | +0.072589247227 | Control |
| 4,000 | 3.852902691364 | 3.926912159920 | +0.074009468555 | Control |
| 4,500 | 3.832705090046 | 3.909346697330 | +0.076641607285 | Control |

汇总：

```text
MatchedTrainingEvaluationCount=9
ControlTrainingEvaluationWins=9
TreatmentTrainingEvaluationWins=0
TrainingEvaluationTies=0
MeanTrainingValidationLossDelta=0.073880687820
```

Treatment 从第一个 500-step 节点开始就落后，并在剩余所有节点保持更高 validation loss。没有出现后期反超迹象。

## 12. Frozen validation/test 定量比较

| 指标 | Control | Treatment | 相对变化 | Winner |
| --- | ---: | ---: | ---: | --- |
| Frozen validation loss | 3.819582318483 | 3.893882904755 | +1.945254% | Control |
| Frozen validation perplexity | 45.585164263392 | 49.101172028751 | +7.713053% | Control |
| Frozen test loss | 3.830240146369 | 3.905327702658 | +1.960388% | Control |
| Frozen test perplexity | 46.073601313835 | 49.666353041570 | +7.797853% | Control |

Generalization gap：

```text
ControlFrozenTestMinusValidationLoss=0.010657827886
TreatmentFrozenTestMinusValidationLoss=0.011444797903
GeneralizationGapTreatmentMinusControl=0.000786970018
```

两组 test 都略高于 validation，且 Treatment 的 gap 没有改善。`dropout=0.1` 在固定预算下既没有取得更低 validation loss，也没有取得更低 test loss。

## 13. 生成结果对齐与比较

### 13.1 Identity gate

Baseline generation evidence：

```text
ArchiveBytes=11777
ArchiveSHA256=f1063bfcf048d5ffa8085be188203f3ed5638d1d31658e4d5962adf35214befa
ManifestSHA256=bd496565a4e669192bd660b9fc9cf546265b8a31419fd75cdab99858e5023953
SamplesSHA256=59decb14aff48fc52a6ee67247d8c15f9ad3a83066afc010eb5b957a9d1bc8cd
```

对齐检查全部通过：

- suite status、format 和 schema 一致；
- protocol ID、原始文件 SHA 和 fingerprint 一致；
- protocol definition 完全一致；
- Tokenizer SHA 一致；
- 参数量、global step 和 tokens seen 一致；
- 30 个 sample keys 的集合与顺序完全一致；
- prompt payload 与 per-sample protocol 完全一致；
- 所有 continuation 都是 64 tokens；
- 两组都是 30 次 `max_new_tokens`、0 次 EOS；
- 两组 context crop 都为 0；
- 唯一模型配置差异仍为 `dropout: 0.0 -> 0.1`。

### 13.2 运行环境边界

| 项目 | Control | Treatment |
| --- | --- | --- |
| GPU | RTX 5060 Laptop GPU | RTX 5090 |
| PyTorch | 2.11.0+cu128 | 2.12.1+cu130 |
| Precision | FP32 | FP32 |
| Deterministic algorithms | true | true |
| TF32 | false | false |
| cuDNN benchmark | false | false |

因此：

```text
CrossMachineBitwiseClaim=False
GenerationSpeedCausalComparisonAllowed=False
```

不同生成时间不能归因于 dropout；生成 token 是否逐位相同也不是本消融的验收条件。

### 13.3 Token-level 描述性指标

以下指标只用于描述 30 个固定样本，不能替代人工质量判断或统计检验：

| 指标 | Control | Treatment | Treatment - Control |
| --- | ---: | ---: | ---: |
| Corpus unique token types | 637 | 658 | +21 |
| Token entropy, bits | 7.791642 | 7.882842 | +0.091200 |
| Mean sample unique-token ratio | 0.624479 | 0.597917 | -0.026563 |
| Mean sample distinct-2 | 0.782540 | 0.777778 | -0.004762 |
| Mean sample distinct-3 | 0.818280 | 0.827419 | +0.009140 |
| Mean sample distinct-4 | 0.833333 | 0.846995 | +0.013661 |
| Mean longest non-overlap repeated span | 7.333333 | 7.466667 | +0.133333 |
| Samples with repeated span ≥4 tokens | 14 | 15 | +1 |
| Samples with repeated span ≥8 tokens | 6 | 6 | 0 |

逐样本 mean unique-token ratio：

```text
ControlWins=17
TreatmentWins=11
Ties=2
```

最长非重叠重复片段越短越好：

```text
ControlWins=10
TreatmentWins=6
Ties=14
```

按 decoding 的平均样本唯一 token 比例：

| Decoding | Control | Treatment | Treatment - Control |
| --- | ---: | ---: | ---: |
| Greedy | 0.156250 | 0.213542 | +0.057292 |
| Temperature 1.0 | 0.843750 | 0.804688 | -0.039063 |
| Temperature 0.7 | 0.645833 | 0.591146 | -0.054688 |
| Top-k 50 | 0.700521 | 0.638021 | -0.062500 |
| Top-p 0.9 | 0.776042 | 0.742188 | -0.033854 |

Treatment 在 greedy 上减少了一部分局部重复，但两组 6/6 greedy 样本仍全部达到“重复片段至少 8 tokens”的循环门。Treatment 在四种随机采样上的平均唯一 token 比例都更低。因此不能把 greedy 的局部指标改善扩写为总体生成质量改善。

### 13.4 人工文本观察

匹配样本显示：

- story/greedy：Control 围绕 `the sun` 循环，Treatment 围绕 `the students will be able` 循环；
- science/greedy：Control 重复 `the first time`，Treatment 重复 `the results ... the study`；
- technology/greedy：两者都围绕单一句式长循环；
- history/greedy：Control 重复 settlers，Treatment 重复 American Indian government；
- explanation/greedy：Control 重复 water，Treatment 重复 energy；
- instruction/greedy：两者都反复重述 tea，但没有给出可靠步骤；
- Treatment 的部分 technology sampling 更贴近 AI 主题；
- 两组 sampling 都存在虚构专名、事实拼接、主题漂移和句法断裂；
- 两组都没有可靠 instruction following；
- 没有一个模型通过 30 个样本证明达到聊天助手质量。

生成结论：

```text
GenerationComparisonOutcome=NO_CLEAR_TREATMENT_IMPROVEMENT
```

## 14. 对原始问题的回答

Day 12 的直接实验证据表明：

1. 当前差生成质量不是通过把 dropout 从 `0.0` 改为 `0.1` 就能修复；
2. Treatment 在固定预算下更难优化，training、validation 和 test 均落后；
3. 两组 validation/test gap 都小且方向一致，没有出现明显 train/test 管线断裂；
4. 两组都能稳定加载 checkpoint、完整评估并输出 30 个结构正确的生成样本，工程链路没有失败；
5. greedy 循环和 sampling 漂移在两组中都存在，更符合小模型容量、有限预训练预算、纯 next-token objective 和无指令微调的能力边界；
6. 本实验没有单独操纵模型规模、训练 token 或数据组成，因此不能在“训练不足、模型容量、数据质量”三者之间给出唯一因果归因。

最稳妥的结论是：

> 当前没有证据表明模型或数据管线存在导致评估失效的工程错误；`dropout=0.1` 在固定 300M-token 预算下使结果变差，且没有解决生成退化。若未来继续提升质量，应把增加训练预算、模型容量、数据配比或后训练分别作为新的单变量实验，而不是保留本次 Treatment。

## 15. 证据包与本地下载

Windows comparison input package：

```text
File=small-gpt-day12-comparison-inputs-e66a2bc-20260815-182244.zip
Bytes=603680
SHA256=1b5912c1122fabbca01532d64d9a33062c0208e1f94a50380d995883572f2ea4
ArchiveEntries=14
ArchiveCRCValid=True
ArchiveInternalHashesValid=True
```

该包包含：

- Control metrics；
- Control frozen evaluation archive；
- Treatment metrics；
- Treatment full validation/test；
- Treatment generation manifest/samples；
- Baseline/Treatment config；
- ablation contract、generation protocol 和 validator；
- package identity 与 SHA256SUMS。

该包明确不包含：

- checkpoint；
- tokenized dataset；
- Tokenizer binary。

Treatment final checkpoint 另行下载并完成本地验证：

```text
Bytes=406108827
SHA256=29b15304f7e6f62b29b1ba4f4b5b6f591d4dcbc7e336ef79bd64486468fcb3ad
Day12TreatmentCheckpointLocalVerification=PASS
```

Baseline generation archive 也在 Windows 重新定位并验证，外层 SHA、CRC、两个 entries 和内部 SHA 全部匹配。

## 16. 云端关机与保全

训练、评估、生成、证据打包和本地下载验证完成后：

- F14 已关机；
- E85 已关机；
- 两台机器都没有继续占用 RTX 5090 计算；
- 没有执行 release；
- 正式 Treatment test 没有重跑；
- 正式 generation suite 没有覆盖；
- 云端证据保留为恢复副本；
- 本地已保存 comparison package 和 final Treatment checkpoint。

## 17. 重要踩坑与永久纠正

| 问题 | 原因 | 永久纠正 |
| --- | --- | --- |
| Dry-run gate 把 `cuda:0` 判为非 CUDA | 文本匹配过窄 | 解析 resolved device 语义，并用真实 CUDA backward 独立确认 |
| `git push` 显示 PowerShell `NativeCommandError` | Git 将正常 remote 信息写到 stderr | 先查 remote SHA 和 tracking，再决定是否重推 |
| 首次云端 contract hash gate 停止 | hard-coded expected hash 写错 | 使用已冻结的实际 hash，任何不一致都停止 |
| Smoke JSON 两次被误判 | gate 猜测顶层 `full_split`，实际字段在嵌套 schema | 先做 schema inventory，再按真实字段审计 |
| Metrics audit 不能直接假设字段 | JSONL 有四种 event shape | 先统计 union keys、event counts 和 record shapes |
| F14 出现问题后担心要重训 | 没先区分机器故障与 artifact 损坏 | 在 E85 逐个核对 metrics/checkpoint/smoke SHA；相同则禁止重训 |
| Bash 把变量名当命令执行 | 长脚本粘贴/换行错误 | 在最近 PASS 点恢复，只读检查 hash 后继续，不重做 formal test |
| Baseline generation 在 E85 找不到 | 它原本在 Windows 本地生成并归档 | 先查 Day 11 Windows 权威路径，不开云端、不重跑生成 |
| Checkpoint hardlink 显示 23 小时前 | hardlink 共享 inode 与 mtime | 以 bytes/SHA 判断身份，不以复制显示时间判断 |
| `git archive` entry count 被误判 | ZIP 自动加入 `configs/`、`reports/` 目录 entries | 分开统计 file entries 与 directory entries，不把总数等同文件数 |

这些失败都发生在外围验证命令，没有改变训练权重、metrics、评估、生成或 Git 功能提交。

## 18. 最终验收清单

- [x] Baseline checkpoint 身份固定；
- [x] Control config 未修改；
- [x] Treatment 只改变 `model.dropout`；
- [x] ablation contract 冻结；
- [x] validator 验证唯一差异与所有 held constants；
- [x] Treatment 参数量不变；
- [x] Treatment 从随机初始化开始；
- [x] 不 resume Baseline；
- [x] 本地定向测试 29 passed；
- [x] 本地完整回归 657 passed；
- [x] 云端完整回归 657 passed；
- [x] Full dry-run 无写入；
- [x] 正式 4,578 updates；
- [x] 正式 300,023,808 tokens；
- [x] 9 次 training validation；
- [x] 5 个 checkpoints；
- [x] metrics 连续性与数值有限性通过；
- [x] final checkpoint 本地下载与 SHA 验证；
- [x] F14→E85 迁移不重训；
- [x] 完整 frozen validation；
- [x] 一次性 frozen test；
- [x] 同一 6×5 generation protocol；
- [x] 30 条 Control 与 30 条 Treatment samples 对齐；
- [x] 定量比较完成；
- [x] 生成内容比较完成；
- [x] 不声明统计显著性；
- [x] 不声明跨机器 bitwise 相同；
- [x] F14、E85 均关机；
- [x] 没有释放实例；
- [x] 最终决策保留 `dropout=0.0`。

## 19. 后续建议

本项目“至少完成一个消融实验”的目标已经满足。Day 12 文档闭环由本报告、README 和 daily log 共同构成；其后的优先任务是项目级总结，而不是无计划地继续租用 GPU。

若未来继续研究生成质量，应一次只选择一个新变量，例如：

1. 在模型结构不变时增加训练 token budget；
2. 在训练 token 不变时增加模型容量；
3. 在模型与预算不变时改变明确的数据配比；
4. 在冻结 base model 后开展独立的 instruction tuning。

任何后续实验都应重新冻结协议、预算、seed、评估和停止条件。Day 12 的一次性 test 和正式生成证据不得被覆盖或选择性重写。
