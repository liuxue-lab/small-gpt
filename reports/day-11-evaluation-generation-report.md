# Day 11 Frozen Evaluation 与文本生成执行报告

## 1. 执行结论

Day 11 已使用 Day 10 的不可变 final checkpoint 完成正式 checkpoint-to-model、frozen validation、frozen test 和固定文本生成协议闭环。

正式 run：

```text
baseline-full-300m-20260813-232952
```

权重身份：

```text
CheckpointPath=step-00004578.pt
CheckpointBytes=406108827
CheckpointSHA256=a39f8378ebe4012afb992be451d355e814b856ffb5e690ac011758f9db614b51
GlobalStep=4578
TokensSeen=300023808
TrainingSourceCommit=07c22a42a696e4d2bab7e6396fcb4c417dc5f63e
```

最终结果：

- 完整 frozen validation：457 batches、3,741,184 tokens；
- validation loss/perplexity：`3.819582318483078 / 45.58516426339153`；
- 完整 frozen test：430 batches、3,517,952 tokens；
- test loss/perplexity：`3.830240146368807 / 46.073601313835496`；
- 两个 split 都使用 sequential non-overlapping windows；
- 两个 split 都没有设置 `max_batches`；
- 评估前后 checkpoint、Tokenizer、Full manifests、tokenized tree 和 Git worktree 不变；
- 固定生成协议包含 6 prompts、5 decoding strategies 和 30 个样本；
- 30/30 样本成功发布，共生成 1,920 tokens；
- checkpoint/model/Tokenizer 在 generation suite 内只加载一次；
- 30 个样本均以 `max_new_tokens=64` 停止，EOS 次数为 0；
- generation/evaluation 两份证据均已归档、下载并通过本地 SHA/内容验证；
- Day 11 最终功能提交为 `b8f8fc854b76e5b73c091343a2234ad8521f8005`；
- 最终功能回归为 `628 passed in 18.73s`；
- A69 临时文件已精确清理，GPU/Small GPT/background jobs 均为 0；
- A69 已关机但未释放；
- D34 未释放。

Day 11 证明了正式权重能够在严格身份门下完成全 split 评估和可审计文本生成。生成结果同时显示：工程链路已经成立，但 34M 参数、300M-token、未指令微调的基础模型仍有严重 greedy 重复、主题漂移和语义不连贯，不能描述为聊天模型质量已经达标。

正式单变量消融尚未开始。

## 2. 范围、授权与停止策略

用户授权 A69 完成以下任务：

1. 验证从 D34 克隆的数据持久性和身份；
2. 仅在 clean worktree 下将代码 fast-forward 到 `b8f8fc8`；
3. 只读运行完整 frozen validation；
4. 只读运行完整 frozen test；
5. 保存、打包、下载并复核评估证据；
6. 任一 gate 失败立即停止和报告；
7. 所有云端任务结束后关机，但不释放 A69 或 D34。

本轮明确不做：

- 不 resume Day 10 正式训练；
- 不创建新训练 run；
- 不覆盖 final checkpoint；
- 不修改模型权重；
- 不恢复 optimizer/scheduler/RNG；
- 不使用 train split 做最终质量指标；
- 不重建 Full/Tokenized Full；
- 不重训 Tokenizer；
- 不通过改 prompt 或采样参数掩盖 baseline 质量；
- 不开展多变量实验；
- 不释放任何保留的 AutoDL 实例。

失败停止策略在代码、Git、云端数据、证据下载和关机门中全部保持。GitHub 网络失败时没有绕过 SHA 或 clean-tree 检查，而是改用 complete-history Git bundle。

## 3. 基线 artifact 与身份

### 3.1 Checkpoint

| 项目 | 值 |
| --- | --- |
| Run ID | `baseline-full-300m-20260813-232952` |
| File | `step-00004578.pt` |
| Bytes | 406,108,827 |
| SHA-256 | `a39f8378ebe4012afb992be451d355e814b856ffb5e690ac011758f9db614b51` |
| Global step | 4,578 |
| Tokens seen | 300,023,808 |
| Model parameters | 33,833,984 |
| Training source | `07c22a42a696e4d2bab7e6396fcb4c417dc5f63e` |
| Source dirty | false |

### 3.2 Tokenizer 与数据

| Artifact | Identity |
| --- | --- |
| Tokenizer SHA-256 | `b26835e02eebf777a257c4732abdd6f9732a115967d2ad839f3a1a00e45ee8c5` |
| Tokenized manifest SHA-256 | `ce7cd91075c7c666c427e1aaa286096a7f386643f3a76de3c26ef770d6cce67e` |
| Source manifest SHA-256 | `14c69dc545838b426e29162c73132cfe444bb2cc56b72c80bb4929f3c65ca96a` |
| Dataset fingerprint | `39dab5bacdf8719bbc849e85ddcd7422cba5777fc044b437d050a49b87ab174f` |
| Tokenized files | 156 |
| Tokenized tree bytes | 775,678,065 |
| Tokenized tree aggregate SHA-256 | `4106e0a7558b18a49edffce2f0436b1c6c553b81f88938caf5cbcdb3af3b25d6` |

### 3.3 Source identity 的两层含义

`07c22a4` 是 checkpoint 内冻结的训练源码身份；`b8f8fc8` 是 Day 11 evaluator/generator 的源码身份。二者不应强行改成同一个值：

- checkpoint 权重确实由 `07c22a4` 训练；
- Day 10 的 `27a5ef0` 只增加文档；
- Day 11 的五个提交增加 loader、evaluation 和 generation；
- Day 11 没有重新训练权重；
- evidence 同时记录 checkpoint training source 和 evaluator/generator source。

## 4. Day 11 Git 变更

### 4.1 提交链

| Commit | Subject | Changed files |
| --- | --- | ---: |
| `6ac89df` | `feat: add model-only checkpoint loading` | 3 |
| `fd23482` | `feat: add frozen split evaluation streams` | 3 |
| `863a721` | `feat: add frozen checkpoint evaluation` | 4 |
| `9cb3208` | `feat: add reproducible text generation` | 4 |
| `b8f8fc8` | `feat: add frozen generation suite` | 7 |

每个提交均满足：

- parent 是前一阶段权威 commit；
- staged files 与预期完全相等；
- 没有 unstaged/untracked 文件；
- `git diff --check` 通过；
- 定向测试通过；
- 完整回归通过；
- 普通 commit；
- push 前 remote/tracking 未漂移；
- push 后 local/tracking/remote 三方 SHA 一致；
- worktree 回到 clean。

### 4.2 最终功能范围

新增或增强：

```text
configs/day11_generation_protocol.json
eval/frozen_evaluation.py
eval/generation.py
eval/generation_suite.py
scripts/evaluate_checkpoint.py
scripts/generate_text.py
scripts/run_generation_suite.py
tests/test_checkpoint.py
tests/test_data_stream.py
tests/test_frozen_evaluation.py
tests/test_generation.py
tests/test_generation_suite.py
train/checkpoint.py
train/data_stream.py
```

checkpoint、dataset、Tokenizer、run output 和 generation/evaluation evidence 均不进入 Git。

## 5. Strict model-only checkpoint loading

### 5.1 为什么不能直接复用 full resume

完整训练 resume 会恢复：

- optimizer；
- scheduler；
- TrainerState；
- Python/NumPy/Torch RNG；
- CUDA RNG；
- data cursor。

评估和生成只需要模型权重。恢复训练态没有必要，还可能改变推理进程的随机状态或错误地表达“将继续训练”。因此新增独立 `load_model_checkpoint`。

### 5.2 Loader 契约

加载前验证：

1. checkpoint file 可读；
2. root field 集合精确；
3. checkpoint format/schema；
4. created timestamp；
5. checkpoint identity schema；
6. resolved config strict JSON；
7. model/training config 内部守恒；
8. model config canonical SHA 与 checkpoint identity 相等；
9. active model config 与 saved model config 相等；
10. TrainerState 有效且位于 update/save boundary；
11. expected run ID；
12. state dict keys；
13. tensor shapes；
14. tensor dtypes；
15. 所有浮点/复数 tensor finite。

只有以上全部通过后才执行 strict model state restore。

不恢复：

- optimizer state；
- scheduler state；
- scaler state；
- Python RNG；
- NumPy RNG；
- Torch CPU/CUDA RNG。

测试还验证这些未使用 payload 即使被替换为不可访问对象，也不会被 model-only loader 触碰。

## 6. Frozen evaluation stream

### 6.1 Split 约束

正式入口只允许：

```text
validation
test
```

`train`、拼写漂移或隐式默认 split 都会拒绝。

### 6.2 Window 语义

每个 split 使用：

- fixed token store；
- sequential non-overlapping windows；
- context length 512；
- 因果 `x/y` shift；
- 完整 tail batch；
- 不跨 split；
- 不重复 window；
- 不推进 training stream；
- 不改变 global Torch RNG。

窗口数：

```text
validation: floor((3741345 - 1) / 512) = 7307
test:       floor((3518409 - 1) / 512) = 6871
```

discarded remainder：

```text
validation: 3741345 - 1 - 7307 * 512 = 160
test:       3518409 - 1 - 6871 * 512 = 456
```

batch size 16，因此：

```text
validation batches = ceil(7307 / 16) = 457
test batches       = ceil(6871 / 16) = 430
```

## 7. Frozen evaluation 实现

### 7.1 CLI 必选 identity

`scripts/evaluate_checkpoint.py` 要求：

- `--checkpoint`；
- `--checkpoint-sha256`；
- `--manifest`；
- `--run-id`；
- `--split`；
- `--output`。

`--max-batches` 是可选正整数。只要设置了它，结果就必须如实区分 bounded 与 full。

### 7.2 执行门

实际执行：

1. checkpoint SHA 在模型加载前比对；
2. model-only strict load；
3. weight tying 验证；
4. manifest-derived active identity 与 checkpoint identity 比对；
5. explicit split stream；
6. token-weighted cross entropy；
7. finite loss/perplexity；
8. batches/tokens 与 frozen stream 计划比对；
9. 记录 runtime flags；
10. strict JSON serialization；
11. fsync + hard-link publish；
12. 既有 output 拒绝覆盖；
13. 临时文件在失败路径清理。

### 7.3 证据字段

结果 JSON 绑定：

- run ID；
- checkpoint path/bytes/SHA/global step/tokens；
- checkpoint identity；
- evaluator commit/dirty；
- model config/parameters/weight tying；
- manifest SHA/fingerprint/Tokenizer SHA；
- split token count；
- context/batch/workers/pin memory；
- available batches/windows/tail；
- full split flag；
- loss/perplexity/batches/tokens/time；
- CUDA/PyTorch/precision/TF32/cuDNN flags。

## 8. Frozen validation/test 正式执行

### 8.1 A69 环境

| 项目 | 值 |
| --- | --- |
| Instance | A69 |
| Hostname | `autodl-container-qcjwby1mc9-8befeb6c` |
| GPU | NVIDIA GeForce RTX 5090 |
| Python | 3.12.3 |
| PyTorch | 2.12.1+cu130 |
| CUDA runtime | 13.0 |
| BF16 | supported |

### 8.2 Evaluation 参数

| 项目 | 值 |
| --- | --- |
| Config | Baseline |
| Device | `cuda:0` |
| Precision | BF16 autocast |
| Batch size | 16 |
| Context | 512 |
| Workers | 4 |
| Pin memory | false |
| Window mode | sequential non-overlapping |
| Hash verification | true |
| max batches | null |
| Order | validation → test |

### 8.3 Validation

```text
Split=validation
FullSplit=True
Loss=3.819582318483078
Perplexity=45.58516426339153
EvaluatedBatches=457
AvailableBatches=457
EvaluatedTokens=3741184
FullEvaluationTokens=3741184
TrailingTokensDiscarded=160
ResultSHA256=3eaa78fa5c1340dfb24165fe48987026215013475c89330e57fac96736ea4307
```

### 8.4 Test

```text
Split=test
FullSplit=True
Loss=3.830240146368807
Perplexity=46.073601313835496
EvaluatedBatches=430
AvailableBatches=430
EvaluatedTokens=3517952
FullEvaluationTokens=3517952
TrailingTokensDiscarded=456
ResultSHA256=34d1a05139c92d5e7d860b3dfdac928a8f6df1f36ce97c894fe09253a3869e48
```

### 8.5 与 Day 10 validation 的边界

Day 10 的最后一次 training-time validation：

```text
step=4500
batches=100
tokens=819200
loss=3.832705090046
perplexity=46.187310231541
```

Day 11 frozen validation：

```text
batches=457
tokens=3741184
loss=3.819582318483078
perplexity=45.58516426339153
```

两者差异来自覆盖范围不同，不是 checkpoint 重新训练。Day 11 frozen test 是独立 test split，不能与 validation 混名。

## 9. 文本生成实现

### 9.1 解码策略

`GenerationSettings` 支持：

- greedy；
- seeded categorical sampling；
- temperature；
- top-k；
- top-p。

约束：

- `max_new_tokens > 0`；
- temperature finite 且大于 0；
- top-k 是正整数且不超过 vocab；
- top-p 位于 `(0, 1]`；
- seed 位于有效非负整数范围；
- greedy 只能使用 temperature 1.0，且不接受 seed/top-k/top-p；
- sample 必须显式 seed。

### 9.2 复现边界

生成运行时：

- `torch.inference_mode()`；
- deterministic algorithms 开启；
- TF32 关闭；
- cuDNN benchmark 关闭；
- sample 使用单独的 device-local generator；
- 不推进 global Torch RNG；
- 证据明确记录 `same_hardware_and_runtime` 复现范围。

这比只记录一个 seed 更严格，但不承诺跨 PyTorch/CUDA/GPU 的 bitwise 相同。

### 9.3 Context 与 EOS

- prompt 不自动添加 BOS；
- prompt 不自动追加 EOS；
- EOS token ID 为 1；
- 超过 512 tokens 时 conditioning window 从左侧裁剪；
- trace 保存完整原始 prompt 和全部生成 token；
- 每次 crop 计数；
- EOS 和 max-token 两种停止原因分开；
- 当前实现没有 KV cache，每个 token 一次完整 forward。

### 9.4 Artifact 与输出

加载 session 前验证：

- checkpoint expected SHA；
- Tokenizer expected SHA；
- 16,384 vocab；
- `<bos>=0`、`<eos>=1`、`<pad>=2`、`<unk>=3`；
- model config；
- run ID；
- weight tying。

输出包括：

- prompt text/decoded text/token IDs；
- initial conditioning IDs；
- generated/full token IDs；
- continuation/full text；
- exact decoding settings；
- EOS/stop reason；
- crop/forward/time；
- checkpoint/Tokenizer/model/runtime identity。

## 10. Frozen generation protocol

### 10.1 Protocol identity

```text
ProtocolID=day11-baseline-generation-v1
ProtocolFileBytes=1684
ProtocolFileSHA256=bb6fd24c2d277d4369fcd21d551ff7023484b62d7bccfd7103beca6c71a8ce4a
ProtocolFingerprint=e60f3fb381b3efd8f00bd3f3fc3071c11645c78977dc7c6c40e0fd124b6d1ed0
Ordering=prompt_then_decoding
MaxNewTokens=64
StochasticSeed=1337
```

raw file SHA 绑定执行时的文件 bytes；fingerprint 绑定规范化 JSON 语义，因此 JSON 格式变化不会改变 fingerprint。

### 10.2 Prompts

| ID | Text | 目的 |
| --- | --- | --- |
| story | `Once upon a time` | 开放式叙事 |
| science | `The experiment showed that` | 科学陈述 |
| technology | `The future of artificial intelligence is` | 技术主题 |
| history | `During the nineteenth century,` | 历史主题 |
| explanation | `The main reason this happens is` | 因果解释 |
| instruction | `To make a cup of tea,` | 程序性指令 |

### 10.3 Decodings

| Role | Strategy | Temperature | Top-k | Top-p | Seed |
| --- | --- | ---: | ---: | ---: | ---: |
| greedy | greedy | 1.0 | null | null | null |
| sample_temperature_1 | sample | 1.0 | null | null | 1337 |
| lower_temperature | sample | 0.7 | null | null | 1337 |
| top_k | sample | 1.0 | 50 | null | 1337 |
| top_p | sample | 1.0 | null | 0.9 | 1337 |

每种非 greedy decoding 只改变一个主要过滤变量，便于观察采样行为。但这些是 decoding 对比，不是模型训练消融。

### 10.4 Suite 发布语义

- exact schema；
- duplicate JSON key 拒绝；
- 5～12 prompts；
- exactly five required decoding roles；
- prompt/decoding ID 唯一；
- 所有 stochastic seeds 等于 protocol seed；
- 模型、checkpoint、Tokenizer 只加载一次；
- sample 顺序固定；
- 所有 sample identity 必须一致；
- strict JSONL；
- manifest 记录 samples bytes/SHA；
- staging 完整后再发布；
- 任何失败不留下部分 suite；
- output directory 已存在时拒绝。

## 11. 正式 generation suite

### 11.1 运行时

| 项目 | 值 |
| --- | --- |
| GPU | NVIDIA GeForce RTX 5060 Laptop GPU |
| Device | `cuda:0` |
| Precision | FP32 |
| PyTorch | 2.11.0+cu128 |
| Deterministic algorithms | true |
| CUDA matmul TF32 | false |
| cuDNN TF32 | false |
| cuDNN deterministic | true |
| cuDNN benchmark | false |
| Model loads | 1 |

### 11.2 汇总

```text
PromptCount=6
DecodingCount=5
ExpectedSamples=30
CompletedSamples=30
ArtifactsLoadedOnce=True
ModelLoads=1
SamplePromptTokens=165
GeneratedTokens=1920
ForwardPasses=1920
ContextCropEvents=0
EOSStops=0
MaxTokenStops=30
SummedGenerationSeconds=10.141866199439391
WallElapsedSeconds=12.773515000008047
```

### 11.3 Evidence

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| `manifest.json` | 5,280 | `bd496565a4e669192bd660b9fc9cf546265b8a31419fd75cdab99858e5023953` |
| `samples.jsonl` | 82,452 | `59decb14aff48fc52a6ee67247d8c15f9ad3a83066afc010eb5b957a9d1bc8cd` |

archive：

```text
File=small-gpt-day11-generation-evidence-b8f8fc8-20260814-050240601.zip
Bytes=11777
SHA256=f1063bfcf048d5ffa8085be188203f3ed5638d1d31658e4d5962adf35214befa
EntryCount=2
CRC=PASS
```

## 12. 生成质量分析

### 12.1 Greedy

六个 greedy 样本都出现明显重复：

| Prompt | 代表性重复 |
| --- | --- |
| story | `The sun is not visible ...` |
| science | `the first time` |
| technology | `not a matter of fact` |
| history | `the first settlers` |
| explanation | `The water is not water` |
| instruction | `make a cup of tea` |

greedy 的确定性使高概率局部循环持续自我强化。这个现象是模型分布与解码共同作用的结果，不能只靠工程入口修复。

### 12.2 Temperature 1.0

纯采样显著增加词汇多样性，但会产生：

- 不存在或拼接式专名；
- 句法尚可但语义关系不稳定；
- prompt 主题漂移；
- 类似网页片段的混杂风格；
- 缺乏事实约束。

### 12.3 Lower temperature

temperature 0.7 降低随机性，但没有稳定消除重复或漂移。有些输出更流畅，有些仍转向无关医学、宗教或历史内容。

### 12.4 Top-k / top-p

top-k=50 和 top-p=0.9 能限制候选集合，但没有让所有 prompts 获得一致的事实性或任务遵循。instruction 的 top-p 样本相对接近“茶”的主题，但步骤仍不可靠。

### 12.5 正确结论

可以证明：

- final checkpoint 可生成文本；
- decoding 参数真实生效；
- fixed seed 和 runtime contract 完整；
- raw tokens 与文本可审计；
- 同一 protocol 可以重复执行；
- 生成退化能够被证据化观察。

不能证明：

- 模型具备可靠事实知识；
- 模型具备指令遵循能力；
- 某个 decoding 在统计上显著优于另一个；
- 30 个样本足以代表所有 prompt；
- 当前模型适合作为生产聊天助手；
- 采样参数对比等于模型消融。

## 13. Evaluation 证据包

### 13.1 Archive identity

```text
File=frozen-evaluation-20260814T115751028276821Z.zip
Bytes=22200
SHA256=53ab948d13b2abbdac1e9bd5610c2b226743761a591701c65f4e23cdf2b62755
EntryCount=7
ReadableBytes=66353
```

entries：

```text
evidence-manifest.json
preflight.json
validation.json
validation.log
test.json
test.log
postflight.json
```

### 13.2 Manifest

```text
Status=complete
EvaluatorSourceCommit=b8f8fc854b76e5b73c091343a2234ad8521f8005
FullSplits=True
ReadOnlyStateVerified=True
EvidenceManifestBytes=1965
EvidenceManifestSHA256=e660b3f857e7501a6d000ecd0973d036cea752bc222a0db29083c4456c9f6363
```

### 13.3 Independent verification

下载后独立验证：

- archive bytes/SHA；
- exact entry inventory/order；
- 无 absolute path、`..`、backslash path；
- 无 symlink；
- 无 encrypted entry；
- 无 duplicate entry；
- ZIP CRC；
- manifest listed bytes/SHA；
- preflight/postflight artifact identity；
- Git branch/head/worktree；
- checkpoint/Tokenizer/manifest/tree identity；
- validation/test full coverage；
- `perplexity ≈ exp(loss)`；
- windows/tokens/remainder 算术；
- result与manifest一致；
- log split/full语义。

结果：

```text
ArchiveSafety=PASS
ArchiveCRC=PASS
ManifestInventoryHashes=PASS
PrePostReadOnlyIdentity=PASS
ValidationFullSplit=PASS
TestFullSplit=PASS
IndependentEvidenceVerification=PASS
```

## 14. A69 迁移、离线 Git 与只读门

### 14.1 克隆持久性

A69 在评估前验证：

- Full source 存在；
- Tokenized Full 存在；
- checkpoint bytes/SHA；
- Tokenizer SHA；
- source/tokenized manifest SHA；
- tokenized tree 完整扫描；
- worktree clean；
- GPU process 0。

初始 Git：

```text
LocalHead=07c22a42a696e4d2bab7e6396fcb4c417dc5f63e
TrackingHead=23c63a6b81c6f44e6cb7dc1208395f1b84c4f407
A69CloneGitState=EXACT_TRAINING_SOURCE_COMMIT
A69TrackingState=KNOWN_STALE_PRE_DAY9_DOCS_ANCESTOR
```

`07c22a4` 正是 checkpoint training source，因此初始克隆不是“错误代码”；`origin/main` 的 `23c63a6` 是网络不可用导致的 stale tracking ref。

### 14.2 GitHub 网络失败

观测到：

- HTTP/2 framing error；
- HTTP/1.1 port 443 timeout；
- `ls-remote` exit 128。

没有继续无限重试，也没有把网络失败写成 identity failure。

### 14.3 Offline bundle

Windows 权威仓库生成：

```text
BundleRef=b8f8fc854b76e5b73c091343a2234ad8521f8005 refs/heads/main
BundleRecordsCompleteHistory=True
BundleBytes=772317
BundleSHA256=aa3636c0aa0199b24a105460a98e07375babe95837eea71cb5870847c238605c
```

A69 执行：

1. bundle bytes/SHA；
2. `git bundle verify`；
3. bundle head；
4. clean worktree；
5. `07c22a4` 是 `b8f8fc8` ancestor；
6. fetch 到独立 `day11-offline/main`；
7. `git merge --ff-only`；
8. HEAD/tree/commit chain/file scope；
9. artifact 前后 bytes/SHA；
10. worktree/GPU process。

最终：

```text
LocalHeadAfter=b8f8fc854b76e5b73c091343a2234ad8521f8005
OfflineHead=b8f8fc854b76e5b73c091343a2234ad8521f8005
WorktreeEntriesAfter=0
ExactCommitChain=True
ExactChangedFileScope=True
NetworkAccess=NONE
```

## 15. 测试

| 阶段 | 定向测试 | 完整回归 |
| --- | ---: | ---: |
| D1 model-only loader | 28 passed | 559 passed |
| D2 frozen streams | 43 passed | 567 passed |
| D3 frozen evaluation | 84 passed | 580 passed |
| D4 generation | 23 passed | 603 passed |
| D5 generation suite | 48 passed | 628 passed |

最终功能回归：

```text
628 passed in 18.73s
FullTestExit=0
PreTestDiffCheckExit=0
PostTestDiffCheckExit=0
StagedAfter=0
TrailingWhitespaceAfter=0
```

测试覆盖重点：

- model-only load 前置拒绝和无部分 mutation；
- optimizer/scheduler/RNG 隔离；
- frozen stream split/coverage/close/RNG；
- checkpoint/manifest/shard hash drift；
- full/bounded split 语义；
- finite strict JSON；
- atomic no-overwrite；
- greedy/EOS/context crop；
- seeded sampling/global RNG；
- top-k/top-p；
- Tokenizer/checkpoint identity；
- protocol duplicate key/schema/role/fingerprint；
- single artifact load；
- ordered JSONL；
- failure does not publish partial output。

## 16. 云端清理、保全与关机

### 16.1 清理

在证据已下载且 SHA/内容通过后，A69 按 exact inventory 删除：

```text
DeletedTemporaryFiles=14
DeletedTemporaryDirectories=4
RemainingTemporaryCandidates=0
```

没有删除：

- repository；
- checkpoint；
- Full/Tokenized Full；
- Tokenizer；
- evaluation evidence directory/archive。

### 16.2 Final cloud closure

```text
Instance=A69
EvaluatorHead=b8f8fc854b76e5b73c091343a2234ad8521f8005
WorktreeEntries=0
FinalEvidenceStatus=complete
FinalEvidenceEntries=7
FinalEvidenceCRC=PASS
FinalEvidenceIntegrity=PASS
GPUComputeProcesses=0
SmallGPTBackgroundProcesses=0
ShellBackgroundJobs=0
CloudClosureReady=True
ReleaseAction=NONE
```

### 16.3 Shutdown

AutoDL 控制台最终状态：

```text
Instance=A69
PowerState=已关机
AvailableAction=开机
ReleaseAction=NONE
```

系统盘和数据盘使用量仍显示，证明实例被保留。D34 也没有执行 release。

## 17. 踩坑与不可重复错误

### 17.1 不要假设 Windows 有 `rg`

早期 inventory 命令因为本地没有 ripgrep 失败。门禁脚本应使用已确认存在的 PowerShell/Python/Git 工具，或先检查命令可用性。

### 17.2 不要在函数名中引入粘贴字符

`Normalize--Paths` 不是预期函数名。PowerShell 解析失败必须停止，不得仅凭后续空变量输出标记 PASS。

### 17.3 不要直接执行未验身份的长粘贴脚本

CRLF、末尾 newline 和复制损坏导致 bash 语法错误。正确顺序：

1. bytes；
2. SHA；
3. CRLF/LF 统计；
4. 规范化或重新传包；
5. `bash -n`；
6. 执行。

### 17.4 不要把 GitHub 不可达等同于 clone 身份失败

先用本地 Git graph、checkpoint source、manifest SHA 和 worktree 证明现状，再使用 complete-history bundle。网络不可用不能成为降低 identity 标准的理由。

### 17.5 不要把 stale `origin/main` 强行伪装成最新

A69 的 `origin/main=23c63a6` 被如实保留；新的 offline ref 指向 `b8f8fc8`。报告同时记录两者。

### 17.6 不要把 `::SKIP_COMPLETION::` 当终端结果

该文本来自异常提醒输出，不是 Small GPT gate。最终云端状态只认实际 terminal evidence 和 AutoDL 控制台。

### 17.7 不要在证据下载前清理

必须先：

- archive 完成；
- 下载；
- bytes/SHA；
- entries/CRC；
- manifest；
- 本地复核。

之后才能精确删除临时 package/helper/bundle。

### 17.8 不要把采样对比写成正式消融

temperature/top-k/top-p 改变的是 decoding，不是训练变量。正式消融需要新的训练实验和单变量控制。

## 18. 最终验收清单

- [x] Day 10 final checkpoint 身份固定；
- [x] strict model-only loader；
- [x] optimizer/scheduler/RNG 不恢复；
- [x] validation/test 显式 frozen streams；
- [x] 完整 validation；
- [x] 完整 test；
- [x] full/bounded 语义明确；
- [x] loss/perplexity finite；
- [x] checkpoint/Tokenizer/manifest identity 绑定；
- [x] 评估输出原子且拒绝覆盖；
- [x] greedy；
- [x] seeded temperature sampling；
- [x] top-k；
- [x] top-p；
- [x] EOS/context/global RNG tests；
- [x] 6 prompts × 5 decodings；
- [x] 30/30 samples；
- [x] generation model load once；
- [x] token IDs/text/runtime/stop reason 证据；
- [x] generation archive 下载验证；
- [x] A69 clone 持久性；
- [x] offline bundle `--ff-only`；
- [x] frozen validation/test evidence archive；
- [x] Windows 下载复核；
- [x] 独立 archive/manifest/pre-post/metrics 复核；
- [x] final code `b8f8fc8`；
- [x] 功能回归 `628 passed`；
- [x] checkpoint/data 全程未改变；
- [x] 临时文件清理；
- [x] GPU/Small GPT/shell jobs 为 0；
- [x] A69 关机；
- [x] A69 未释放；
- [x] D34 未释放；
- [ ] 单变量训练消融尚未执行。

## 19. 下一阶段

Day 11 已完成主模型的训练后 baseline。下一阶段应是正式单变量消融，而不是重复 test 或无限调采样参数。

开始消融前必须冻结：

1. baseline checkpoint 和三个核心 SHA；
2. data/Tokenizer；
3. 模型基线；
4. optimizer/LR schedule；
5. seed；
6. training token budget；
7. frozen validation/test evaluator；
8. generation protocol；
9. GPU/runtime；
10. artifact/evidence schema。

推荐先提出一个可解释假设，再选择唯一变量。例如 dropout、训练 token budget 或某一模型结构变量只能选一个；其余条件保持相同。资源成本和运行时必须在启动云端前另行计算并取得授权。

在消融完成前，项目状态应准确写为：

> 正式预训练、完整 frozen validation/test 和固定生成协议已经完成；主模型 baseline 已冻结；正式单变量消融尚未完成。
