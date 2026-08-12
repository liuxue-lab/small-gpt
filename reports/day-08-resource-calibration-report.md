# Day 8 RTX 5090 Baseline 资源定标执行报告

## 执行结论

Day 8 已在 AutoDL A57 的单卡 NVIDIA GeForce RTX 5090 上完成 Baseline 资源探测、BF16 短跑、checkpoint 保存、独立 resume 对照和正式训练入口的 20 → 25 step 恢复验证。

本轮冻结以下四个资源字段：

```yaml
micro_batch_size: 16
gradient_accumulation_steps: 8
num_workers: 4
pin_memory: false
```

对应训练计划为：

```text
16 sequences/micro-step × 512 tokens/sequence = 8,192 tokens/micro-step
8 micro-steps/update × 8,192 tokens/micro-step = 65,536 tokens/update
ceil(300,000,000 / 65,536) = 4,578 optimizer updates
ceil(4,578 × 0.02) = 92 warmup updates
4,578 × 65,536 = 300,023,808 planned tokens
token overshoot = 23,808
```

GPU 阶段结论为通过。它证明 33,833,984 参数 Baseline 在本次冻结环境与真实 Pilot 数据上具备可执行、数值有限、可验证、可保存并可恢复的工程路径，不代表模型已完成预训练，也不代表模型质量已经达标。

## 范围与授权边界

本轮用户授权：

1. 使用已启动的 AutoDL RTX 5090 实例，运行时间上限 60 分钟；
2. 远端只读身份检查、`git pull --ff-only`、Pilot 资源探测；
3. 在探测通过后运行有限 BF16 短跑与 checkpoint/resume；
4. 上传本地 `data/tokenized/fineweb_edu_pilot`；
5. 禁止 Full 数据构建和 300M-token 正式训练。

实际执行遵守以上边界：只使用 Pilot 数据，最长正式路径为 25 optimizer updates / 1,638,400 tokens；没有构建 Full，没有启动 300M-token 正式训练。A57 证据下载并核对后由用户确认关机，F19 旧实例也保持关机。

## 证据身份

| 项目 | 值 |
| --- | --- |
| 探测源码 commit | `2e3166c395d7057cb8509fda6f5768bd9b203537` |
| 远端 `origin/main` | `2e3166c395d7057cb8509fda6f5768bd9b203537` |
| 源工作区 | clean |
| Baseline 源 SHA-256 | `ffe7dac747eaef98421aa06767b6e5ca02c5372bd033f257836fcae7ca859fdd` |
| 证据包 SHA-256 | `8ebcb2b968a96ca1a7cfa950a5d2ff6188e9e41f44cb925e65f0eb19bef61966` |
| 证据包内部校验 | `SHA256SUMS.txt` 记录的 24/24 文件通过 |
| 远端证据采集时间 | `2026-08-12T19:15:05+08:00` |

证据包保留探测 JSON、stdout 日志、短跑 JSONL、resolved config、run metadata、环境快照和 checkpoint identity。原始探测数据、运行日志和 checkpoint 不进入 Git；Git 只提交冻结配置、测试和人类可读报告。

## 远端环境

| 项目 | 结果 |
| --- | --- |
| 实例 | AutoDL A57，单卡按量计费 |
| GPU | NVIDIA GeForce RTX 5090 |
| Driver | 580.105.08 |
| CUDA capability | 12.0 |
| 显存 | 32,607 MiB；PyTorch 记录 33,668,988,928 bytes / 31.357 GiB |
| Python | 3.12.3 |
| PyTorch | 2.12.1+cu130 |
| CUDA runtime | 13.0 |
| cuDNN | 92000 |
| BF16 | supported |
| tokenizers | 0.23.1 |
| 平台 | Linux 5.15 / glibc 2.35 |
| CPU / RAM | AutoDL 控制台显示 25 cores / 90 GB |
| 持久盘 | 50 GB；证据采集时使用 1.2 GB，剩余 49 GB |
| 控制台单价 | ¥2.78/小时 |

CPU 和 RAM 来自提供商控制台截图；其余运行时字段来自远端命令或探测 JSON。未保存账单明细，因此不虚构精确本轮费用。

## Pilot 数据身份

| 项目 | 结果 |
| --- | --- |
| Manifest SHA-256 | `141a0c4626cb4f5ba8b041984514825b0d30a34ac8f51a8fcc2fdbb6e512f961` |
| Dataset fingerprint | `a3eb6012c1cb3e2dab2a7839bebb04530563b19b4d5f7d8022e3c121b13ca7f3` |
| Tokenizer SHA-256 | `b26835e02eebf777a257c4732abdd6f9732a115967d2ad839f3a1a00e45ee8c5` |
| 模型 tokens | 2,129,776 |
| Payload bytes | 4,259,552 |
| Storage shards | 5 |
| Pilot shard scan | pass |

`tokenizer_config.json` 的 Windows 源文件为 CRLF。远端解压后的 raw/LF SHA 与清单中 CRLF SHA 不同，但严格换行归一化后精确得到期望 SHA，因此记录为 `TokenizerMetadataIdentity=PASS_CRLF`，不是内容漂移。

本轮没有上传 `data/processed/fineweb_edu_corpus/manifest.json`，所以 source corpus manifest identity 标记为 `DEFERRED_BY_SCOPE`。Pilot token shard、索引、Tokenizer 和 manifest 已完成扫描；正式训练前仍必须用完整数据链重新验证 source manifest identity，不能把本轮 defer 当作永久豁免。

## 探测方法

每个候选都在新的 Python 子进程中构造 Baseline 模型、AdamW、scheduler、Trainer 和真实 Pilot 数据流。候选先 warmup，再重置 CUDA peak counters，测量完整 optimizer updates，并在计时前后同步 CUDA。

阶段与采样：

1. single smoke：1 warmup + 1 measured update；
2. micro-batch sweep：每个候选 1 warmup + 3 measured updates；
3. accumulation：每个候选 1 warmup + 3 measured updates；
4. loader sweep：每个候选 2 warmup + 5 measured updates；
5. 正式 dry-run；
6. 20-step BF16 短跑；
7. 独立 2 → 4 step checkpoint/resume 对照；
8. 正式训练入口从 step 20 恢复到 step 25。

探测安全门使用 `peak_reserved / total VRAM <= 0.90`。所有候选均正常完成，无 OOM、timeout、非有限 loss 或非有限 gradient。

## Single smoke

`b1-a1-w0-p0` 成功完成，测得 21,558.47 tokens/s，peak allocated 0.777 GiB，peak reserved 0.812 GiB / 2.59%。该阶段只验证完整路径可运行，不用于最终吞吐决策。

## Micro-batch sweep

| Micro batch | Candidate | Tokens/s | Peak allocated GiB | Peak reserved GiB | Reserved | Loss mean | Max grad norm |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `b1-a1-w0-p0` | 31,842.12 | 0.777 | 0.812 | 2.59% | 9.805379 | 13.761248 |
| 2 | `b2-a1-w0-p0` | 69,473.71 | 1.034 | 1.037 | 3.31% | 9.823751 | 10.931164 |
| 4 | `b4-a1-w0-p0` | 126,528.88 | 1.560 | 1.621 | 5.17% | 9.815887 | 9.521565 |
| 8 | `b8-a1-w0-p0` | 228,223.25 | 2.594 | 2.637 | 8.41% | 9.808365 | 8.675487 |
| 12 | `b12-a1-w0-p0` | 236,346.55 | 3.631 | 3.834 | 12.23% | 9.800480 | 8.328429 |
| 16 | `b16-a1-w0-p0` | 241,017.23 | 4.663 | 5.053 | 16.11% | 9.795489 | 8.028246 |

选择 `micro_batch_size=16`。它是矩阵中最大的已测成功候选，reserved 仅占 16.11%。从 12 增至 16 的吞吐提升约 1.98%，已经接近平台区间，因此本轮没有为了寻找 OOM 边界继续扩大 batch。结论只能写成“已验证到 16”，不能声称 16 是 RTX 5090 的最大可行 micro batch。

## Gradient accumulation

| Accumulation | Tokens/update | Total updates | Warmup | Tokens/s | Peak reserved GiB | Reserved | Loss mean | Max grad norm |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 16,384 | 18,311 | 367 | 252,140.64 | 5.553 | 17.71% | 9.770172 | 7.672687 |
| 4 | 32,768 | 9,156 | 184 | 251,132.80 | 5.553 | 17.71% | 9.723078 | 7.351255 |
| 8 | 65,536 | 4,578 | 92 | 249,709.07 | 5.553 | 17.71% | 9.641564 | 7.327797 |

选择 `gradient_accumulation_steps=8`。相对 a2，a8 的探测吞吐只低约 0.96%，没有增加 peak reserved，却把一次 optimizer update 精确扩大到 65,536 tokens，将训练 horizon 缩短为 4,578 updates。该选择平衡吞吐、日志与 checkpoint 粒度以及 scheduler horizon；不是单纯追求表中最高 tokens/s。

## DataLoader sweep

| Workers | Pin memory | Candidate | Tokens/s | Peak reserved GiB | Reserved |
| ---: | --- | --- | ---: | ---: | ---: |
| 0 | false | `b16-a8-w0-p0` | 238,837.79 | 5.553 | 17.71% |
| 0 | true | `b16-a8-w0-p1` | 237,097.99 | 5.553 | 17.71% |
| 2 | false | `b16-a8-w2-p0` | 227,999.91 | 5.553 | 17.71% |
| 2 | true | `b16-a8-w2-p1` | 230,212.41 | 5.553 | 17.71% |
| 4 | false | `b16-a8-w4-p0` | 246,577.50 | 5.553 | 17.71% |
| 4 | true | `b16-a8-w4-p1` | 224,354.75 | 5.553 | 17.71% |
| 8 | false | `b16-a8-w8-p0` | 235,713.21 | 5.553 | 17.71% |
| 8 | true | `b16-a8-w8-p1` | 236,979.67 | 5.553 | 17.71% |

选择 `num_workers=4`、`pin_memory=false`。这是本轮 loader 矩阵最快组合。`pin_memory=true` 没有形成一致收益，在 workers=4 时反而明显变慢。该组合随后通过 25-step 正式路径与 resume 验证，提供了比单个 loader 测量更长的稳定性证据；但 loader sweep 本身仍只有一轮，未来更换实例、PyTorch、驱动或 Full 数据布局时必须复测。

## 冻结后的 Baseline 契约

只修改 `configs/baseline.yaml` 中四个字段：

| 字段 | 冻结值 | 依据 |
| --- | ---: | --- |
| `micro_batch_size` | 16 | 最大已测候选，5.053 GiB reserved，吞吐接近平台 |
| `gradient_accumulation_steps` | 8 | 65,536 tokens/update，吞吐损失小，显存无增长 |
| `num_workers` | 4 | loader sweep 最高端到端吞吐 |
| `pin_memory` | false | true 无一致收益，w4 下更慢 |

模型、优化器、学习率、warmup ratio、validation/save intervals 和 300M token 预算全部保持不变。探测时使用的短跑临时配置将 `eval_interval=10`、`eval_batches=3`、`save_interval=20` 仅用于加压验证，这三个短跑覆盖值不得写回正式 Baseline。

## Dry-run 与 25-step BF16 短跑

正式训练入口 dry-run 通过：

| 项目 | 结果 |
| --- | --- |
| Model parameters | 33,833,984 |
| Device / precision | `cuda:0` / BF16 |
| Autocast / GradScaler | true / false |
| Input / target | `(16, 512)` / `(16, 512)` |
| Causal shift | true |
| Tokens/update | 65,536 |
| Total / warmup updates | 4,578 / 92 |
| Source state | commit `2e3166c...`, clean |

短跑结果：

| Step | Train loss | Validation loss | Perplexity | Grad norm | LR |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 9.815502 |  |  | 7.472572 | 0.00000326087 |
| 10 | 8.993008 | 8.953871 | 7,737.787 | 2.180461 | 0.00003260870 |
| 20 | 8.685449 | 8.641006 | 5,659.020 | 1.769621 | 0.00006521739 |
| 25 resumed | 8.486105 |  |  | 1.614367 | 0.00008152174 |

最终完成 25 updates / 1,638,400 tokens。25 个 train loss、grad norm 和两次 validation 全部有限。JSONL 有 31 个事件：1 `run_start`、25 `train_update`、2 `evaluation`、2 `checkpoint`、1 `resume_start`；train steps 为连续的 1～25，resume 从 step 20 开始。

## Checkpoint 与恢复

| Checkpoint | Bytes | SHA-256 |
| --- | ---: | --- |
| 独立 resume gate step 2 | 406,108,891 | `73bf4fd7b61993b00239c5aeff61a050ed6593189f0d0d84cb5ea0bf1e913b0a` |
| 正式短跑 step 20 | 406,108,827 | `696d544311a74211b0bbe04d3d33a15eddebe969c9214ac23063ff483b794d80` |
| 正式短跑 step 25 | 406,108,827 | `7ed909498406151e22020df4ab5d7f27a55a1bfe09c688d91ef65f9f3d7f0e09` |

独立 2 → 4 step 对照结果：

1. 下一训练 batch exact；
2. 下一学习率为 `9.78260869565e-06`；
3. scheduler state exact；
4. Python、NumPy、Torch RNG exact；
5. continuation metrics、模型参数和 optimizer state 在 BF16 冻结容差 `rtol=0.0005, atol=0.00005` 内一致；
6. 最终 step 4 / 262,144 tokens；
7. 所有 checkpoint/resume 检查通过。

正式 run 从 step 20 checkpoint 恢复到 step 25，同一 run 的 metrics 连续追加，没有重复或缺失 step。两次 checkpoint 保存耗时分别为 0.932688 秒与 0.943397 秒，均值 0.938043 秒。证据没有单独计时 checkpoint load，因此本报告不提供虚构的 load latency。

## 吞吐、时长与费用估算

从原始 JSONL 重算：

| 口径 | Tokens/s | 300M update-only ETA | 计算费用估算 |
| --- | ---: | ---: | ---: |
| Step 2～20，排除首次冷启动 | 236,957.39 | 21.10 分钟 | ¥0.98 |
| Step 2～25，排除两次进程冷启动 step 1/21 | 236,442.75 | 21.15 分钟 | ¥0.98 |
| Step 1～20，包含首次冷启动 update | 208,172.11 | 24.02 分钟 | ¥1.11 |
| 20-step 进程墙钟 11 秒直接外推 | 119,156.36 | 41.96 分钟 | ¥1.94 |

推荐采用约 236,443 tokens/s 作为本次环境下的 steady update 吞吐。按正式 `eval_interval=500`、`save_interval=1000`、4,578 updates、实测平均 evaluation 0.108850 秒、平均 checkpoint save 0.938043 秒并加入约 4 秒启动开销，简化模型得到约 21.31 分钟 / ¥0.99。

因此正式规划写成约 21～42 分钟、计算费用约 ¥0.99～¥1.94。下界接近 update-only / interval 模型；上界来自故意提高 validation/checkpoint 频率的 20-step 短跑墙钟直接外推。该范围不包含 Full 数据准备、上传、下载、实例排队、共享云抖动、热降频或故障重跑。

AutoDL 控制台单价为 ¥2.78/小时，用户给出的本轮上限为 60 分钟，因此本次授权窗口的计算费用上界为 ¥2.78；由于没有账单明细，不记录更精细的实际金额。

## 磁盘预算

证据采集时持久盘剩余 49 GB。正式训练预估：

| 项目 | 估算 |
| --- | ---: |
| 5 个长期 checkpoint（1000/2000/3000/4000/final 4578） | 约 2.03 GB |
| 原子保存临时峰值，按 6 个 checkpoint image 保守计 | 约 2.44 GB |
| 34M 参数纯 FP32 权重理论值 | 约 135.3 MB |
| 34M 参数纯 BF16 权重理论值 | 约 67.7 MB |
| Full 项目 token payload，按约 365M tokens × uint16 粗估 | 约 730 MB |
| 清洗后 Full corpus 既有预留 | 5 GB |

权重导出大小是理论值，本轮没有生成独立推理权重；Full token payload 也只是基于 Day 4 比例的规划值，不含 `.idx`、header、manifest、staging 和源语料工作空间。正式 Full 构建前必须重新测量真实 token 数、文件数和 staging 峰值，并确认运行、checkpoint、Full 数据和临时文件全部位于持久盘。

## 风险与正式训练硬门

本地 Stage I 门已经通过；正式训练仍必须满足以下完整链路，任一未通过都不能启动 300M-token 正式训练：

1. [x] 应用本次配置/文档更新，完成 83 项定向测试与 531 项完整项目回归；
2. [x] `git diff --check` 通过，审查 Baseline 只有四个资源字段变化；
3. [ ] 提交并普通 push，确认 `HEAD == origin/main`；
4. [ ] 构建并验证 Full corpus，恢复 source manifest identity 硬门；
5. [ ] 使用冻结 Tokenizer 编码 Full，并扫描所有 payload/index identities；
6. [ ] 根据 Full 实际大小复核 50 GB 持久盘预算；
7. [ ] 正式训练前重新核对 GPU、驱动、PyTorch、CUDA、BF16、空闲显存和工作区身份；
8. [ ] 确认正式 run-id、输出路径、checkpoint 保留策略、费用和时长授权；
9. [ ] 若实例、驱动、PyTorch、数据布局或 Tokenizer 身份变化，重新执行资源定标；
10. [ ] 正式启动仍需用户单独明确授权。

## 本地合入状态

进入云端前，Stage A 资源探针专项为 `23 passed`，当时完整项目回归为 `531 passed in 14.24s`。Stage I 配置与文档随后通过 SHA/clean-worktree 门安全应用，并得到以下最终本地结果：

```text
Targeted: 83 passed in 2.41s
Targeted outer elapsed: 3.20s
Targeted exit code: 0

Full regression: 531 passed in 13.95s
Full outer elapsed: 15.71s
Full exit code: 0

git diff --check exit code: 0
```

应用后的工作区精确包含 6 个修改文件与 1 个新增报告；没有 `data/`、`runs/`、`checkpoints/` 或原始远端证据。Baseline diff 只有 `16 / 8 / 4 / false` 四项资源值变化。

Day 8 当前进度按 96% 记录。GPU 实验、配置冻结与本地回归均已结束，剩余工作只有最终 diff 审查、commit、普通 push 和远端 Git hash 核对。

## 最终判定

| 验收项 | 状态 |
| --- | --- |
| RTX 5090 / BF16 环境身份 | PASS |
| Pilot manifest / Tokenizer / shard scan | PASS |
| Micro-batch sweep | PASS |
| Accumulation 决策 | PASS |
| DataLoader sweep | PASS |
| Baseline dry-run | PASS |
| 20-step BF16 + validation | PASS |
| 独立 checkpoint/resume 对照 | PASS |
| 正式入口 20 → 25 resume | PASS |
| 证据包下载与 SHA 对照 | PASS |
| A57 / F19 关机 | PASS，用户确认 |
| Full 数据 | NOT STARTED |
| 300M-token 正式训练 | NOT STARTED |
| 本地 Stage I 应用与最终回归 | PASS；83 targeted / 531 full / diff-check 0 |
| commit、push 与远端 hash 核对 | PENDING |

结论：冻结 `16 / 8 / 4 / false` 有完整的真实 Pilot、显存、吞吐、数值、validation、resume 与本地回归证据支持，可以进入 commit/push；正式 Full 与 300M-token 训练仍是独立后续阶段。
