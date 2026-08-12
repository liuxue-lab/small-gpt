# Day 8 Baseline 资源定标探针设计

## 完成状态（2026-08-12）

本设计已经在 AutoDL A57 的 NVIDIA GeForce RTX 5090 上完成执行。探测源码绑定 commit `2e3166c395d7057cb8509fda6f5768bd9b203537`，远端工作区 clean，证据包 SHA-256 为 `8ebcb2b968a96ca1a7cfa950a5d2ff6188e9e41f44cb925e65f0eb19bef61966`。

经过 isolated micro-batch、accumulation、DataLoader、BF16 短跑、validation 与 checkpoint/resume 门，最终冻结：

```yaml
micro_batch_size: 16
gradient_accumulation_steps: 8
num_workers: 4
pin_memory: false
```

冻结计划为 65,536 tokens/update、4,578 total updates、92 warmup updates、300,023,808 planned tokens 和 23,808 token overshoot。峰值 reserved memory 为 5.553 GiB，占 PyTorch 记录总显存的 17.71%。

详细实测结果、决策理由、checkpoint 身份、ETA、费用与磁盘预算见 [Day 8 RTX 5090 Baseline 资源定标执行报告](day-08-resource-calibration-report.md)。本轮没有构建 Full，也没有启动 300M-token 正式训练。

下方保留执行前设计与命令，作为探针为何如此实现和审计原始假设的历史记录。配置冻结后，探针 CLI 会按设计拒绝把已 resolved 的正式 Baseline 当作“恰好四字段 unresolved”的新探测输入；如需复测，应使用绑定历史身份的临时配置和新输出文件，不应先把正式 Baseline 改回 `null`。

## 执行前状态（历史记录）

以下内容描述 Day 8 本地准备阶段当时新增的资源探针；在该历史时点，RTX 5090 尚未执行探测，Baseline 的四个资源字段也尚未冻结。

当时的权威 Baseline 保持以下未决字段：

```yaml
micro_batch_size: null
gradient_accumulation_steps: null
num_workers: null
pin_memory: null
```

探针不会改写 `configs/baseline.yaml`，不会创建正式 run，不会保存 checkpoint，也不会启动 300M token 正式训练。

## 目标

1. 在真实 Baseline 模型、BF16、context 512、真实 Pilot DataLoader 和 Day 7 `Trainer` 路径上测量候选资源组合。
2. 每个候选在独立子进程内运行，避免 OOM 后继续复用受污染的 CUDA 进程。
3. 每个候选只执行有限 warmup 和 measured updates。
4. 记录端到端 tokens/s、peak allocated、peak reserved、loss、gradient norm、学习率和运行时身份。
5. 每完成一个候选就原子更新 JSON 报告，使中断后的证据可恢复。
6. 只给出 preliminary recommendation；后续短跑、validation 和 checkpoint/resume 门通过前不得冻结配置。

## 新增文件

1. `train/resource_probe.py`

   提供严格候选类型、候选配置解析、accumulation 算术提案、结果校验、显存安全筛选、loader 吞吐筛选、SHA-256 和原子 JSON 写入。

2. `scripts/probe_baseline_resources.py`

   提供父进程编排与候选 worker。父进程自身不运行训练；每个候选由新的 Python 子进程构造模型、optimizer、scheduler、训练状态和真实数据流。

3. `tests/test_resource_probe.py`

   覆盖资源值边界、候选矩阵、Baseline 只改四字段、plan 数学、原子报告、结果校验、安全显存筛选、loader 推荐、OOM 识别和 CLI 参数契约。

4. `reports/day-08-resource-calibration-design.md`

   记录本设计、执行命令、验收门和故障处理方式。

## 不变契约

1. 模型结构、参数量、context、vocab、初始化、手写 attention 和 loss 契约不变。
2. AdamW 分组、scheduler、gradient accumulation、clipping 和状态计数复用 Day 7 实现。
3. 数据来自 `data/tokenized/fineweb_edu_pilot/manifest.json`，探针不修改 tokenized artifacts。
4. 每个输入 batch 仍是 `(input_ids, targets)`，形状为 `B × 512`，训练器和模型不再次 shift。
5. 候选解析只允许替换 `micro_batch_size`、`gradient_accumulation_steps`、`num_workers`、`pin_memory`。
6. Baseline YAML 只有在远端实测、短跑和 resume 门通过后才允许单独更新。

## 执行结构

```text
父进程只读校验 config、manifest、Git identity
→ 生成严格候选矩阵与 request fingerprint
→ 原子创建 JSON 报告
→ 为一个候选启动全新 Python worker
→ worker 构造 Baseline GPT、AdamW、scheduler、Trainer、Pilot stream
→ warmup update
→ 重置 CUDA peak memory counters
→ measured updates 与 CUDA synchronize
→ worker 原子写候选结果并退出
→ 父进程校验结果并原子更新总报告
→ 遇到首个 OOM 或非 OOM 错误立即停止候选序列
```

Micro-batch 候选必须严格递增。首个 OOM 后，父进程不会继续尝试更大的 batch。若候选 worker 发生 OOM，它记录 OOM 后直接退出；后续逻辑由未使用 CUDA 的父进程处理。

## 显存口径

1. `allocated_after_setup_bytes` 和 `reserved_after_setup_bytes` 记录模型、optimizer 对象已构造但 AdamW state 尚未经过首步完全建立时的状态。
2. 至少一个 warmup update 用于建立 AdamW state 和稳定 CUDA allocator。
3. warmup 后重置 peak counters。
4. `peak_allocated_bytes` 和 `peak_reserved_bytes` 来自 measured updates。
5. `peak_reserved_fraction = peak_reserved_bytes / total_device_memory_bytes`。
6. 默认安全门为 0.85。超过该比例的候选即使没有 OOM，也不会成为 micro-batch preliminary recommendation。

0.85 是探针的安全筛选阈值，不是最终结论。它为 validation、checkpoint、运行时波动和非 PyTorch GPU 占用保留余量。

## 吞吐口径

父进程采用 worker 中 measured updates 的外层墙钟时间：

```text
tokens_per_second = measured_tokens / measured_elapsed_seconds
```

计时前后执行 CUDA synchronize，因此不使用未同步的异步 kernel 提交时间。`Trainer.run_update` 在计时区间内从 DataLoader 拉取 batch，所以该数值包含数据等待，是端到端训练吞吐，不是纯 GPU kernel benchmark。

## Micro-batch 探测命令

获得 AutoDL 和费用明确授权、远端身份与 Pilot 校验通过后，先使用保守递增矩阵：

```bash
cd /root/autodl-tmp/small-gpt

python scripts/probe_baseline_resources.py \
  --config configs/baseline.yaml \
  --manifest data/tokenized/fineweb_edu_pilot/manifest.json \
  --output reports/day-08-microbatch-probe-rtx5090.json \
  --phase microbatch \
  --micro-batch-sizes 1 2 4 8 12 16 \
  --gradient-accumulation-steps 1 \
  --num-workers 0 \
  --pin-memory true \
  --warmup-updates 1 \
  --measured-updates 3 \
  --candidate-timeout-seconds 900 \
  --expected-device-name "RTX 5090" \
  --minimum-device-memory-gib 30 \
  --max-reserved-fraction 0.85
```

此命令最多执行 24 个 optimizer updates，但通常会在首个 OOM 或错误处提前停止。它不是正式训练，不创建 run/checkpoint。

如果 16 仍成功，结论只能是“已验证到 16”，不能声称 16 是最大值。扩大矩阵必须使用新的输出文件，并再次明确本轮时间边界。

## Accumulation 算术提案

探针不内置默认 tokens/update，避免把未经讨论的 effective batch 偷偷写入实验契约。只有显式提供 `--target-tokens-per-update` 时，报告才增加算术提案：

```text
tokens/micro-step = selected_micro_batch × 512
accumulation = ceil(requested_tokens/update ÷ tokens/micro-step)
tokens/update = tokens/micro-step × accumulation
total updates = ceil(300,000,000 ÷ tokens/update)
warmup updates = ceil(total updates × 0.02)
```

该输出仍标记为 proposal，不会写入 Baseline YAML。最终 accumulation 还需要结合 optimizer update 频率、gradient noise、训练总时长、validation/save 间隔和 resume 粒度确定。

## Workers 与 Pin Memory 探测命令

在 stable micro batch 通过后，用同一个 batch 和 accumulation 比较有限矩阵。将下例中的 `8` 替换为当时已通过安全门的值：

```bash
python scripts/probe_baseline_resources.py \
  --config configs/baseline.yaml \
  --manifest data/tokenized/fineweb_edu_pilot/manifest.json \
  --output reports/day-08-loader-probe-rtx5090.json \
  --phase loader \
  --micro-batch-size 8 \
  --gradient-accumulation-steps 1 \
  --num-workers 0 2 4 8 \
  --pin-memory both \
  --warmup-updates 1 \
  --measured-updates 3 \
  --candidate-timeout-seconds 900 \
  --expected-device-name "RTX 5090" \
  --minimum-device-memory-gib 30 \
  --max-reserved-fraction 0.85
```

此阶段的最快组合也只是 preliminary recommendation。至少需要重复测量，并在 Baseline BF16 短跑和 exact-resume 检查中证明稳定，才能冻结 workers/pin memory。

## 中断恢复

同一命令中加入 `--resume`，且所有请求参数、config SHA、manifest SHA 和候选矩阵必须与原报告完全一致：

```bash
python scripts/probe_baseline_resources.py <原参数> --resume
```

恢复时保留已经验证的 `ok` 结果。若原报告已经记录 OOM boundary，则不会跨过该边界继续尝试更大的 batch。请求 fingerprint 不一致时必须使用新输出文件，工具不会拼接不可比较的证据。

## JSON 报告关键字段

顶层包含：

1. schema version 与 probe kind。
2. probe status、创建时间、更新时间。
3. Git commit 与 dirty 状态。
4. config/manifest 路径和 SHA-256。
5. 原始未决 Baseline training snapshot。
6. 完整候选矩阵和 probe settings。
7. 每个 worker 的隔离结果。
8. preliminary recommendation。
9. stop reason 和未尝试候选。

成功候选包含：

1. Python、PyTorch、CUDA、cuDNN、GPU 名称、compute capability、总显存和 BF16 支持。
2. resolved plan 和模型参数量。
3. warmup/measured updates 与 measured tokens。
4. 端到端 tokens/s。
5. loss、gradient norm 和 learning rate 摘要。
6. setup、peak、measurement 后的 allocated/reserved memory。
7. 每个 measured update 的 Day 7 `UpdateMetrics`。

失败候选包含 status、异常类型、异常信息和有界 traceback/stdout/stderr 尾部。

## 本地验收命令

代码合入仓库但尚未启动 AutoDL 时运行：

```powershell
cd D:\code\small-gpt

python -m pytest -q .\tests\test_resource_probe.py
python -m pytest -q
git diff --check
git status --short
git diff -- .\configs\baseline.yaml
```

预期结果：

1. `tests/test_resource_probe.py` 全部通过。
2. 完整项目回归全部通过，不能少于 Day 7 已知的 508 项加新增测试数。
3. `git diff --check` 无输出。
4. `configs/baseline.yaml` 无 diff。
5. 工作区只出现本次明确新增的四个文件。

## 远端执行前硬门

1. 用户明确允许启动或使用正在计费的 AutoDL 实例，并给出本轮时长/预算边界。
2. 远端 commit、工作区、Python、PyTorch、CUDA、RTX 5090、VRAM、BF16 和磁盘身份通过。
3. Tokenizer SHA、Pilot manifest SHA 和 dataset fingerprint 与 Day 7 权威记录一致。
4. run/checkpoint 路径位于大容量持久盘。
5. 本地新增测试和完整回归通过。
6. Baseline YAML 仍为四字段 null。

任何一项未通过都不执行 probe worker。

## 停止条件

1. Micro batch 1 即 OOM。
2. GPU 不是预期 RTX 5090、显存容量异常或 BF16 不受支持。
3. config/manifest 身份变化。
4. loss 或 gradient 非有限。
5. worker timeout、进程非零退出或结果 JSON 无效。
6. 磁盘、RAM、CPU 或系统稳定性异常。
7. 超过用户批准的时间或预算边界。
8. 出现未知 Git 修改。

停止后保留 JSON 证据，不修改 Baseline，不自动重试更激进参数，不开始 Full 或正式训练。

## 完成定义

本地代码准备完成不等于 Day 8 资源定标完成。四个字段只有满足以下顺序后才能冻结：

```text
本地新增测试与全回归通过
→ AutoDL 身份和费用授权通过
→ micro-batch isolated probe 通过
→ accumulation 决策有明确数学与实验理由
→ workers/pin-memory 重复测量通过
→ Baseline BF16 分层短跑与 validation 通过
→ checkpoint/resume 对照通过
→ 报告审查通过
```

最终更新 `configs/baseline.yaml` 应是单独、可审计的后续动作，不属于本探针自动行为。
