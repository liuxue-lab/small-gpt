# Day 13 Jetson Orin Nano Super 推理部署执行报告

## 1. 执行结论

Day 13 已在一块真实 NVIDIA Jetson Orin Nano Engineering Reference Developer Kit Super 上完成 Small GPT Control 模型的原生 PyTorch CUDA 推理部署。部署对象是 Day 10 冻结的 `dropout=0.0` Control checkpoint，不是 Day 12 的 `dropout=0.1` Treatment。

本轮已经完成：

- Windows、GitHub `main` 与 Jetson 三端源码身份闭环；
- Control checkpoint、Baseline config 和 Tokenizer 的 bytes/SHA-256 闭环；
- Jetson 设备、操作系统、内存、磁盘、功耗模式、温度和原生 CUDA runtime 审计；
- 33,833,984 参数模型的 strict model-only load 与 CUDA finite forward；
- FP32、FP16 各 3 个固定 prompt 的 greedy smoke；
- FP32、FP16 同设备 benchmark；
- FP16 10 个串行请求稳定性验证；
- evidence archive 在 Jetson 冻结、传回 Windows 并独立验证；
- 推理与 `tegrastats` 进程归零。

核心结论：

```text
JetsonFP32Inference=PASS
JetsonFP16Inference=PASS
ModelOnlyLoad=PASS
StabilityRequests=10/10
OOMCount=0
NonFiniteCount=0
TrainingAttempted=False
CheckpointWritten=False
KVCacheImplemented=False
TensorRTImplemented=False
PerformanceClaimsScope=SAME_DEVICE_DESCRIPTIVE
```

这证明当前 checkpoint-to-text 工程链路能够在 Jetson Orin Nano Super 8GB 上运行，并不证明模型已经具备聊天助手质量、可靠机器人控制能力或通用的生产部署能力。

## 2. 目标与非目标

Day 13 的目标是回答：冻结的 34M Control 模型能否在 8GB Jetson 上被真实、可验证地加载并进行 CUDA 文本续写；FP32 与 FP16 在同一设备、同一协议下的速度、显存 allocator、温度和稳定性表现如何。

本轮目标：

1. 保持 Control checkpoint、Tokenizer 和 Baseline config 不变；
2. 新增只用于部署检查与 benchmark 的代码和协议；
3. 使用 Jetson 现有原生 PyTorch CUDA runtime，不在设备上训练；
4. 先做 model-only load，再做 FP32、FP16 smoke；
5. 用冻结 warmup/measured 协议做同设备描述性 benchmark；
6. 保存可独立验证且不包含 checkpoint、密钥或局域网地址的 evidence。

明确非目标：

- 不进行 300M-token 训练或任何训练恢复；
- 不修改权重、optimizer、scheduler 或 checkpoint；
- 不实现 KV Cache、TensorRT、INT8、动态 batching 或并发服务；
- 不把厂商标称 TOPS 直接换算为本模型 FP16 吞吐；
- 不证明 Windows 与 Jetson 的浮点计算逐位一致；
- 不把三个 smoke 文本当作生成质量评测；
- 不连接舵机、驱动电机或执行真实车辆控制。

## 3. Git 与源码身份

| 角色 | Git 身份 |
| --- | --- |
| Day 13 starting HEAD | `0319f80766991eead65556df564497036605d1a3` |
| Day 13 functional HEAD | `4c946adffc0e5ee24b1377662819491a86c40aa5` |
| Functional subject | `feat: add verified Jetson inference deployment` |
| Final docs HEAD | 包含本报告、README 与 daily log 的文档提交 |

final docs HEAD 的 40 位 SHA 必须在文档提交创建后由 `git rev-parse HEAD`、`origin/main` 和 `git ls-remote` 三方核验得到。提交内容不能可靠嵌入自身 SHA，因此本报告不写一个不可验证的自引用哈希；最终 gate 输出承担该身份记录。

Functional commit 只新增四个文件：

```text
configs/day13_jetson_deployment_protocol.json
scripts/benchmark_jetson_inference.py
scripts/check_jetson_deployment.py
tests/test_day13_jetson_deployment.py
```

变更量为 3,078 insertions。Day 13 定向测试 `55 passed`，完整回归 `712 passed`，其中原有测试 657 个、新增 Day 13 测试 55 个。正式 frozen test 没有重跑。

## 4. 冻结部署协议与 artifact 身份

协议：

```text
ProtocolID=day13-jetson-pytorch-inference-v1
ProtocolStatus=frozen_after_user_approval
ProtocolFingerprint=da361f335dba18e00d3f4caefb2ded8f7517e3032e58275d4be5fcae95f33ba3
RuntimeRoute=native_existing
TrainingAllowed=False
```

| Artifact | Bytes | SHA-256 |
| --- | ---: | --- |
| `configs/baseline.yaml` | 1,258 | `ca8524c425e1e5e3a600de5773f9a526ef3674741040635bee91fe31f4b24c0e` |
| Control `step-00004578.pt` | 406,108,827 | `a39f8378ebe4012afb992be451d355e814b856ffb5e690ac011758f9db614b51` |
| `tokenizer.json` | 1,137,073 | `b26835e02eebf777a257c4732abdd6f9732a115967d2ad839f3a1a00e45ee8c5` |
| `tokenizer_config.json` | 2,988 | `8622711407aab3f299996b7d3009d4f4447ae35879ca8e50451b5f0adbdf5141` |
| Deployment protocol | 3,506 | `0cb0eeeb47819351d44307afe54dc6dca270a6400a87c4cd22551988301985f3` |

Control checkpoint 元数据：

```text
ControlRunID=baseline-full-300m-20260813-232952
GlobalStep=4578
TokensSeen=300023808
CheckpointLoadMode=model_only
```

Tokenizer 运行时 probe：

```text
VocabularySize=16384
SpecialTokenIDs=<bos>:0,<eos>:1,<pad>:2,<unk>:3
ProbeText=Once upon a time
ProbeTokenIDs=[10235,2026,261,698]
ProbeUnknownTokenCount=0
ProbeDecodedText=Once upon a time
```

## 5. Jetson 硬件、系统与存储

| 项目 | 实测结果 |
| --- | --- |
| Device model | NVIDIA Jetson Orin Nano Engineering Reference Developer Kit Super |
| Architecture | `aarch64` |
| GPU name | `Orin` |
| CUDA compute capability | 8.7 |
| Ubuntu | 22.04.5 LTS |
| Kernel | `5.15.148-tegra` |
| Jetson Linux / L4T | R36.4.3 |
| `nvidia-jetpack` meta-package | `NOT_INSTALLED` |
| Physical memory | 7.44 GiB |
| Swap | 11.72 GiB |
| Storage device | 256 GB NVMe SSD |
| Root filesystem | `/dev/nvme0n1p1`, ext4, read-write |
| Root filesystem size | 87.64 GiB |
| Root available at inventory | 11.76 GiB |

`nvidia-jetpack` meta-package 未安装，因此本报告不用一个未被 `dpkg` 直接证明的 JetPack package version 替代设备证据。部署运行时以真实观测到的 Jetson Linux R36.4.3、PyTorch、CUDA 和 cuDNN 身份为准。

该板卡原本来自 ROSMASTER A1AI 阿克曼小车。部署测试前舵机等执行器已拆除，没有不可中断的机器人、采集或控制任务。板载散热器与风扇保持工作，板卡未放置在导电表面。

## 6. 网络、时间与主机链路

Windows 与 Jetson 通过 1 Gbps 有线链路通信，Jetson 部署地址固定为 `192.168.50.2`。正式 evidence archive 不记录该局域网地址。

厂商系统原有 Wi-Fi profile `ROSMASTER-A1` 工作在 AP/shared 模式，却错误提供 `192.168.1.1` 默认路由，导致外网和 NTP 不可达。使用管理员权限关闭该热点并连接真实 Internet Wi-Fi 后：

```text
InternetPing=PASS
DNSResolution=PASS
SystemClockSynchronized=yes
NTPService=active
TimeZone=Asia/Shanghai
```

SSH、SCP、tar 与 Windows 项目 Python 均通过主机 readiness gate。源码、runtime assets 与 406 MB checkpoint 分别传输，Jetson 端 bytes/SHA 与 Windows 源逐个一致。

## 7. 原生 runtime 路线

没有安装通用 x86 CUDA wheel，也没有在 Jetson 上无界源码编译。采用设备已有的 NVIDIA aarch64 PyTorch，并在继承 user site 的独立 venv 中只安装冻结的 aarch64 `tokenizers` wheel。

| Runtime 项目 | 身份 |
| --- | --- |
| Python | 3.10.12 |
| PyTorch | `2.5.0a0+872d972e41.nv24.08` |
| PyTorch CUDA build | 12.6 |
| cuDNN | 90600 |
| NumPy | 1.26.4 |
| PyYAML | 5.4.1 |
| Tokenizers | 0.23.1 |
| Runtime venv | `/home/jetson/small-gpt-day13/.venv-jetson` |
| Runtime identity SHA-256 | `44b65f0ac7be423affd46a4a0f80299c49a69ccd7f49385809ba3d5762d3ad40` |

冻结 wheel：

```text
Filename=tokenizers-0.23.1-cp310-abi3-manylinux_2_17_aarch64.manylinux2014_aarch64.whl
Bytes=3374081
SHA256=1bf13402aff9bc533c89cb849ec3b412dc3fbeacc9744840e423d7bf3f7dc0e3
```

原生 CUDA FP32 与 FP16 forward/backward probe 均为 finite，`torch.cuda.is_available()` 为 true，device count 为 1，BF16 capability probe 为 true。本轮部署实际只使用 FP32 与 FP16 推理。

## 8. 功耗与散热边界

设备观测到的功耗模式：

```text
NVPowerMode=MAXN_SUPER
PowerModeChanged=False
JetsonClocksAttempted=False
```

Day 13 没有调用 `nvpmodel` 改变模式，也没有启用 `jetson_clocks`。因此 benchmark 数字只描述当前 MAXN_SUPER、当前散热器/风扇、当前后台系统负载下的这一次设备状态，不能推广到 7W、15W、25W 或其他 Jetson。

空闲 `tegrastats` 观测约为 44.5～46.8°C、5.2～5.3 W 输入功耗。正式 smoke、benchmark 和 stability 期间最高温度不超过 49.406°C，没有记录到 OOM、nonfinite、主动关机或因温度导致的任务失败。

## 9. Model-only load gate

load-only 使用 `cuda:0` 与 FP32：

```text
StrictStateDictLoad=True
MissingKeys=0
UnexpectedKeys=0
OptimizerStateRestored=False
SchedulerStateRestored=False
TrainingResume=False
ModelTraining=False
InferenceMode=True
ModelLoadSeconds=2.524338
ForwardInputShape=[1,4]
ForwardLogitsShape=[1,4,16384]
ForwardLogitsDevice=cuda:0
ForwardLogitsDtype=torch.float32
ForwardLogitsFinite=True
```

load-only evidence：

```text
File=load-only.json
Bytes=3598
SHA256=ab9455fc3f6f8935549d89b859381095dc40e2dd1dbabb86a5ff9b19d0358518
```

该 gate 只反序列化冻结 checkpoint、构造模型、strict 加载 model state 并执行一个有限前向；没有恢复训练状态、生成文本或写 checkpoint。

## 10. 模型配置

| 配置 | 值 |
| --- | ---: |
| Parameters | 33,833,984 |
| Transformer layers | 8 |
| Attention heads | 8 |
| Hidden size | 512 |
| FFN hidden size | 2,048 |
| Context length | 512 |
| Vocabulary size | 16,384 |
| Dropout | 0.0 |

FP32 使用 float32 weights/ops；FP16 使用 float16 weights/ops。两条路线都处于 `model.eval()` 和 `torch.inference_mode()`。

## 11. FP32 smoke

冻结运行：

```text
RunID=day13-jetson-control-fp32-20260820T184158Z
ProtocolID=day13-jetson-greedy-smoke-v1
Precision=fp32
Device=cuda:0
Prompts=3
MaxNewTokensPerPrompt=64
CompletedSamples=3
GeneratedTokens=192
AllTokenIDsInRange=True
AllLogitsFinite=True
OOMCount=0
ArtifactsLoadedOnce=True
```

性能与热状态：

```text
MeanEndToEndTokensPerSecond=65.664218
MeanDecodeTokensPerSecond=69.931027
MaximumTemperatureC=49.406
MaximumGR3DPercent=77
```

三个 prompt 为 `Once upon a time`、`The solar system`、`To make a cup of tea`。continuation 均成功生成 64 tokens，但分别围绕 `the sun`、`the system` 和 `a cup of tea` 反复复述。这是工程 smoke，不是质量通过门。

## 12. FP16 smoke

冻结运行：

```text
RunID=day13-jetson-control-fp16-20260820T184158Z
ProtocolID=day13-jetson-greedy-smoke-v1
Precision=fp16
Device=cuda:0
Prompts=3
MaxNewTokensPerPrompt=64
CompletedSamples=3
GeneratedTokens=192
AllTokenIDsInRange=True
AllLogitsFinite=True
OOMCount=0
WeightDtype=torch.float16
ComputeDtype=torch.float16
```

性能与热状态：

```text
MeanEndToEndTokensPerSecond=71.903251
MeanDecodeTokensPerSecond=79.778462
MaximumTemperatureC=48.062
MaximumGR3DPercent=55
```

FP16 与 FP32 smoke 的 3/3 greedy token sequences 相同，但本报告明确保持：

```text
CrossPrecisionStructureAlignment=PASS
CrossPrecisionBitwiseClaim=False
```

相同输出不能推出所有输入、长度、kernel、runtime 或机器都逐位一致。

## 13. Benchmark 协议

FP32 与 FP16 benchmark 使用同一 Jetson、同一 functional HEAD、同一 checkpoint、Tokenizer、prompt 和时间戳：

```text
ProtocolID=day13-jetson-benchmark-v1
PromptID=prompt_02
Prompt=The solar system
BatchSize=1
WarmupRuns=3
MeasuredRuns=10
MaxNewTokensPerRequest=64
MeasuredGeneratedTokens=640
AllGeneratedTokens=832
Decoding=greedy
StopOnEOS=False
SynchronizeCUDA=True
WarmupExcludedFromMeasuredSummary=True
```

warmup 只用于稳定 kernel/allocator，不进入 10 次正式统计。model load 和第一请求分别记录，不混入 steady-state decode 指标。

## 14. FP32 与 FP16 性能结果

| 指标 | FP32 | FP16 | FP16 相对 FP32 |
| --- | ---: | ---: | ---: |
| Model load, s | 1.780663 | 1.843020 | +3.50% |
| First request wall, s | 1.128334 | 1.092088 | -3.21% |
| TTFT mean, ms | 11.717872 | 12.304402 | +5.01% |
| TTFT median, ms | 11.747540 | 12.305967 | +4.75% |
| TTFT p95, ms | 11.905624 | 12.460659 | +4.66% |
| Decode mean, tok/s | 78.719951 | 83.787412 | +6.437% |
| Decode median, tok/s | 78.751395 | 未作为跨精度主结论 | 不适用 |
| Decode p95, tok/s | 79.034659 | 未作为跨精度主结论 | 不适用 |
| End-to-end mean, tok/s | 78.815526 | 83.746914 | +6.257% |
| End-to-end median, tok/s | 78.843314 | 未作为跨精度主结论 | 不适用 |
| End-to-end p95, tok/s | 79.133604 | 未作为跨精度主结论 | 不适用 |
| CUDA peak allocated, bytes | 167,121,920 | 88,706,048 | -46.921% |
| CUDA peak reserved, bytes | 176,160,768 | 113,246,208 | -35.714% |
| Maximum temperature, °C | 49.187 | 47.062 | -2.125°C |
| Maximum GR3D | 76% | 55% | 描述性 |

本次单设备 run 中，FP16 steady-state mean decode 提升 `6.437%`，mean end-to-end 提升 `6.257%`，peak allocated 减少 `46.921%`。但 FP16 model load 稍慢、TTFT mean 稍高，说明“FP16 在所有指标上都更快”并不成立。

上述 CUDA bytes 是 PyTorch allocator 的 peak allocated/reserved，不是整块 8GB 统一内存的总占用，也不是“模型只使用几十 MB 内存”。Jetson CPU 与 GPU 共享系统内存，必须同时保留系统内存证据。

## 15. 系统内存与恢复

设备总内存为 7,802,708 KiB。正式 load/smoke/benchmark 前设置 2,097,152 KiB 可用内存门。

FP16 smoke 第一次尝试前，系统可用内存约比门槛少 9 MiB，因此没有启动生成。诊断发现 `update-manager` 使用约 226 MiB RSS；在确认没有 active package mutator 后，只对该进程发送正常终止，未强杀、未停止 ROS、未操作 PID 784，也未修改 package state。恢复后：

```text
MemAvailableKiB=2220452
MemoryThresholdMet=True
ForceKillAttempted=False
PackageMutationAttempted=False
ROSProcessMutationAttempted=False
```

稳定性 run 内第一请求前到最后请求后，系统 available memory 变化为 `-371,634,176` bytes，但请求后的 available 值并非严格单调下降；PyTorch allocated/reserved 在十次请求中 delta 与 range 都为 0。整个 Stage M 结束后的 `MemAvailableKiB=2238008`，高于运行前门槛。因此本轮没有观察到单调无界 allocator 增长，但十个请求不能证明长期服务绝无内存泄漏。

## 16. FP16 稳定性

```text
RunID=day13-jetson-control-stability-20260820T190445Z
ProtocolID=day13-jetson-stability-v1
Precision=fp16
Device=cuda:0
SequentialRequests=10
ConcurrentRequests=1
GeneratedTokensPerRequest=64
CompletedRequests=10
FailedRequests=0
GeneratedTokens=640
OOMCount=0
NonFiniteCount=0
AllTokenIDsInRange=True
AllLogitsFinite=True
MeanDecodeTokensPerSecond=82.815297
MeanEndToEndTokensPerSecond=80.661543
MaximumTemperatureC=46.781
MaximumGR3DPercent=55
```

allocator 稳态：

```text
CUDAPostRequestAllocatedFirstBytes=80481280
CUDAPostRequestAllocatedLastBytes=80481280
CUDAPostRequestAllocatedDeltaBytes=0
CUDAPostRequestReservedFirstBytes=113246208
CUDAPostRequestReservedLastBytes=113246208
CUDAPostRequestReservedDeltaBytes=0
NoObservedMonotonicUnboundedAllocatorGrowth=True
```

这支持“当前冻结协议下 10 个串行请求可靠通过”，不扩写为长时间、多用户或并发生产稳定性。

## 17. KV Cache 与解码实现

```text
KVCacheImplemented=False
DecodeImplementation=full_prefix_recompute
TensorRTImplemented=False
INT8Implemented=False
```

每生成一个新 token 都重新计算当前完整前缀。当前 tok/s 是这一简单实现的真实表现。若 Day 14 增加 KV Cache，必须创建新协议和新 run ID，重新测量 TTFT、decode、allocator、系统内存与温度，不能覆盖 Day 13 evidence。

TensorRT 未作为本轮运行路径，也没有因为设备可能带有相关库就宣称 TensorRT 部署完成。后续路线必须先审计设备上真实 TensorRT 版本和支持的 API。

## 18. 生成质量边界

FP32/FP16 都完成 checkpoint-to-text，但三个 greedy continuation 显示明显重复：

- story prompt 反复使用 `the sun`；
- solar-system prompt 反复使用 `the system is designed to provide the power`；
- tea prompt 反复使用 `a cup of tea`。

这些输出与 Day 11/12 的观察一致：34M 参数、300M-token、纯 next-token 预训练、无 instruction tuning 的 base LM 能生成结构合法文本，但容易循环、跑题、事实拼接和语义断裂。

因此：

```text
CheckpointToTextEngineeringPath=PASS
ChatAssistantQualityClaim=False
ReliableRobotControlClaim=False
DeploymentProvesModelQuality=False
```

固定 `stop_on_eos=false` 且每条恰好生成 64 tokens 是 smoke/benchmark 协议，不生成 EOS 不等于运行失败。

## 19. Evidence archive 与 Windows 独立复核

Jetson evidence archive：

```text
Filename=small-gpt-day13-jetson-evidence-4c946adf-20260820T190445Z.zip
Bytes=43975
SHA256=19e0e42454eb5a9e8329a014e112a4347dcaefc1ba7ab6005b9aad71c5357d0e
FileEntryCount=29
DirectoryEntryCount=0
UncompressedBytes=184212
CRCValid=True
InternalManifestRowCount=28
InternalHashesValid=True
FailureRecordCount=2
```

archive 包含 runtime inventory、pip freeze、load-only、FP32/FP16 smoke、FP32/FP16 benchmark、FP16 stability、各组 `tegrastats`、内部 manifest 和失败记录。它明确不包含：

- Control checkpoint；
- Tokenizer binary；
- Git bundle；
- SSH keys、密码或环境密钥；
- Jetson 局域网地址。

Windows 权威副本：

```text
D:\model-backups\small-gpt\day13\evidence\small-gpt-day13-jetson-evidence-4c946adf-20260820T190445Z.zip
```

Windows 独立验证结果：

```text
OuterTransferIdentity=PASS
FileEntryCount=29
DuplicateEntryCount=0
UnsafeEntryCount=0
SymlinkEntryCount=0
EncryptedEntryCount=0
BadCRCEntry=NONE
InternalHashMismatchCount=0
FrozenIdentityMismatchCount=0
JSONParseFailureCount=0
JSONLParseFailureCount=0
SecretPatternFindingCount=0
CheckpointFileIncluded=False
TokenizerBinaryIncluded=False
GitBundleIncluded=False
SemanticIdentityValidation=PASS
```

下载后 Jetson 原 archive bytes/SHA 保持不变，推理进程与 `tegrastats` 进程均为 0。没有自动删除 Jetson 文件，也没有自动关机。

## 20. 失败、恢复与未解决事项

| 问题 | 原因 | 恢复与结果 |
| --- | --- | --- |
| Runtime assets ZIP 初验失败 | Windows `Compress-Archive` 写入反斜杠 entry | 错误包移入 quarantine；用 Python ZIP 写正斜杠；CRC、entry set 与 SHA 全部通过 |
| Stage H1 SSH 返回 unexpected EOF | 远端长命令尾部语法闭合错误，但目录已创建 | 不盲目重建；执行只读 recovery，确认目录 owner/mode/空目录后通过 |
| Stage I2 `ssh.exe` 报文件名或扩展名太长 | Windows 将超长远端脚本作为单个命令行参数 | 改为通过 stdin 向远端 Bash 传入脚本，随后 runtime 安装与验证通过 |
| NTP 无法同步 | 厂商热点 profile 提供无效默认路由 | 关闭热点、连接真实 Wi-Fi，Internet、DNS、NTP 全部通过 |
| FP16 smoke 首次被内存门阻止 | 可用内存比 2 GiB 门槛少约 9 MiB | 只正常结束 idle `update-manager`，没有强杀或影响 ROS；随后原协议通过 |
| FP32 benchmark 后 validator 报 index mismatch | validator 假设 1-based，正式 JSONL 合同实际为 0-based | 不重跑、不改 evidence；只读 recovery 按真实 0-based contract 验证并冻结原输出 |
| `git push` 被 PowerShell 包装成异常 | Git 把正常 remote 消息写到 stderr | fetch 与 remote query 证明 push 已完成；没有盲目重推 |
| Stage O1 对缺失报告的预期检查报错 | `git cat-file -e` 的预期失败被 PowerShell 当终止错误 | 无文档变更；改用不报错的 `git ls-tree`，pre-mutation gate 通过 |

仍未解决或不在 Day 13 范围：

- 未实现 KV Cache，decode 仍为 full-prefix recompute；
- 未完成 TensorRT、INT8、量化或 engine build；
- 未测 batch size 大于 1、并发请求、长时间服务或 512-token 极限附近的完整矩阵；
- 未做不同功耗模式、`jetson_clocks` 或散热方案对比；
- 未完成 systemd/API 服务、容器化发布或 ROS 2 安全接口；
- 未改善 base LM 的重复与 instruction-following 能力；
- `nvidia-jetpack` meta-package 未安装，平台身份以 L4T 与真实 runtime 组件记录。

## 21. 安全与授权边界

本轮始终遵守：

- `FormalTestRerun=False`；
- `TrainingAttempted=False`；
- `CheckpointWritten=False`；
- 不恢复 optimizer/scheduler；
- 不改变 Baseline config、Tokenizer 或 Day 12 artifacts；
- 不改变 `nvpmodel`；
- 不运行 `jetson_clocks`；
- 不宽泛终止 Python/ROS 进程；
- 不删除 incoming、repo、runtime、checkpoint 或 evidence；
- 不自动重启或关机；
- 不把 Jetson 接入真实执行器；
- 不把局域网地址或凭据装入 evidence。

功能代码已经单独提交；文档闭环只允许修改 README、daily log 与本报告。最终暂存、提交和 push 都需要分别核对 exact file set。

## 22. 最终验收与 Day 14 建议

### 22.1 Day 13 验收

- [x] Windows 本地、tracking、remote functional HEAD 三方一致；
- [x] Control checkpoint bytes/SHA 固定；
- [x] Baseline config 与 Tokenizer 固定；
- [x] Jetson 型号、系统、内存、NVMe、功耗和温度完成审计；
- [x] 原生 PyTorch CUDA runtime identity 固定；
- [x] 55 个 Day 13 定向测试通过；
- [x] 712 个完整回归测试通过；
- [x] model-only strict load 与 finite CUDA forward 通过；
- [x] FP32 smoke 3/3；
- [x] FP16 smoke 3/3；
- [x] FP32 benchmark 3 warmup + 10 measured；
- [x] FP16 benchmark 3 warmup + 10 measured；
- [x] FP16 stability 10/10；
- [x] 无 OOM、nonfinite 或 allocator 单调无界增长；
- [x] evidence archive 在 Jetson 冻结；
- [x] Windows 外层 SHA、CRC、内部 hashes 与语义独立复核通过；
- [x] 推理与 `tegrastats` 进程归零；
- [x] 不训练、不写 checkpoint、不重跑 formal test；
- [x] 不实现或冒充 KV Cache/TensorRT；
- [x] 不删除 Jetson 文件、不自动关机。

Day 13 技术部署和 evidence 闭环已完成。文档提交的真实 40 位 SHA 与 `Day13Progress=100%` 只由最终普通 push 后的三方 Git gate 输出；本报告不写不可验证的自引用值。

### 22.2 Day 14 建议

建议 Day 14 选择一个新的、单独冻结的优化主题，优先级如下：

1. 在 PyTorch 路线实现可验证 KV Cache，对比 full-prefix recompute；
2. 固定短、中、长 prompt，分别测 prefill、TTFT、decode tok/s、allocator 与系统内存；
3. 审计设备真实 TensorRT 版本，再决定是否建立独立 TensorRT/ONNX 路线；
4. 若做 FP16、INT8 或不同功耗模式对比，创建新 protocol/run ID，不覆盖 Day 13；
5. 增加更长的稳定性与故障恢复测试，再考虑 systemd 或本地 API 服务；
6. 机器人集成必须建立独立的命令白名单、超时、急停和人工接管边界；
7. 若目标是改善回答质量，应在训练主机上设计独立 instruction-tuning 或更大训练预算，不把 Jetson 部署优化混同为模型质量优化。

在这些工作获得新授权前，保留 Jetson 上的 repo、runtime、Control checkpoint 与 evidence，保持当前可恢复状态。
