# Day 14 KV Cache 实现、v2 修订与证据恢复报告

## 1. 报告目的

本报告记录 Small GPT Day 14 的 KV Cache 功能、协议修订、Git 恢复、Jetson v2-r1 隔离部署和 F1 证据恢复状态。目标是把事实、历史报告和未知项严格分开，避免在缺少 runtime evidence 时宣称 Day 14 已完成。

本报告不是一次新的实验结果。文档恢复期间没有运行 Python、pytest、correctness、benchmark 或 stability，没有训练，没有读取、复制、链接或改写 checkpoint，也没有修改 Jetson。

## 2. 冻结身份

| 项目 | 冻结值 |
| --- | --- |
| Repository | `https://github.com/liuxue-lab/small-gpt.git` |
| Day 13 final docs head | `6ef391625a091fc652ea85478d1e72abdb1bb56e` |
| Day 14 v1 functional head | `74ff2619a6b20bb243d3e67c3bbb6a79a4cd54e3` |
| Day 14 v2 revision head | `c3076da038814f0d3da25d2030eb8201643c6e67` |
| Day 14 v2 repair head | `774cf358be9822cdeb6a5921bc9068c1312bc192` |
| Repair subject | `fix: align KV-cache v2 runtime branch contract` |
| Feature branch | `day14-kv-cache-v2` |
| Day 14 functional head | `774cf358be9822cdeb6a5921bc9068c1312bc192` |
| First Day 14 documentation commit | `bd760a526df713c22fa6c2d3ec3a0a1f0eecec1b` |
| Protocol | `day14-kv-cache-v2` / schema v2 |
| Required runtime branch | `day14-kv-cache-v2` |

功能代码、feature tracking 和 feature 远端在功能同步门禁后均为 `774cf358`。`main` 先 fast-forward 到该功能 head，随后增加首个文档提交 `bd760a5` 及纯文档修正提交。功能 head 与文档提交历史分开记录；实时 `main` SHA 以 Git 历史为准。所有同步均使用普通 push，没有 force push、amend 或 tag。

## 3. 功能实现范围

Day 14 为模型增加独立的 inference-only KV Cache 路径：

- `CausalSelfAttention.forward_cached`；
- `TransformerBlock.forward_cached`；
- `GPT.forward_cached`；
- `LayerKVCache`、`PastKeyValues` 与 `GPTCachedOutput`；
- full-prefix reference 与 KV-cache stepwise correctness 对照；
- paired benchmark、稳定性和 evidence validator；
- strict context-boundary、artifact identity 与禁止训练门禁。

缓存每层 key/value 的逻辑形状为 `[B, H, S, D]`。prefill 处理完整 prompt，decode 逐 token 追加；position IDs 从 past length 继续；`prompt_length + generated_token_count <= 512`。不实现 rolling window、eviction、跨请求复用或全局持久 cache。

原始训练 `forward`、训练调用点、optimizer、scheduler 和 checkpoint schema 保持不变。cache 不是 parameter、persistent buffer、optimizer state 或 checkpoint member。

## 4. 提交链与分支契约修复

| Commit | 角色 | 结果 |
| --- | --- | --- |
| `74ff2619` | KV Cache v1 功能 | 保留 |
| `c3076da0` | v2 数值决策合约和全新功能链 | 保留 |
| `774cf358` | required branch 与 v2-r1 namespace 修复 | 当前冻结 head |

`c3076da0` 的继承协议仍要求 `source.required_branch=main`，但实际 v2 功能链位于 `day14-kv-cache-v2`。这会在 checkpoint 加载前错误拒绝获批的运行。`774cf358` 将 required branch 修正为 feature branch，并将部署根隔离到 `/home/jetson/small-gpt-day14-v2-r1`，避免覆盖 v1 和已有 v2。

修复提交直接涉及：

```text
configs/day14_kv_cache_protocol.json
scripts/check_day14_kv_cache.py
tests/test_day14_kv_cache.py
```

Day 14 当前完整功能文件集合为：

```text
configs/day14_kv_cache_protocol.json
scripts/benchmark_day14_kv_cache.py
scripts/check_day14_kv_cache.py
tests/test_day14_kv_cache.py
```

## 5. v1 失败证据与 v2 科学边界

冻结 protocol 保留以下 v1 证据：

| Gate | 结果 |
| --- | --- |
| K2 FP32 model-only load | PASS |
| K3 FP32 correctness | PASS |
| K4 v1 FP16 original | 失败记录保留 |
| K4 v1 FP16 recovery | 失败记录保留 |

v1 FP16 观测值：

| 指标 | 观测 |
| --- | ---: |
| Comparison positions | 64 |
| Generated token IDs exact | true |
| All argmax exact | true |
| Minimum top-5 set overlap | 5 |
| All logits finite | true |
| Context boundaries | pass |
| Maximum absolute error | 0.0390625 |
| Mean absolute error | 0.0029780296463286504 |
| Legacy allclose failing positions | 43 |

因此 v1 不得声称完整 FP16 correctness PASS。v2 在任何 v2 correctness 执行前冻结 decision-and-bounded-drift 合约，其中最大误差门为 `<=0.05`、平均误差门为 `<=0.005`，并要求 token IDs、长度、argmax、top-5、finite、参数与 state-dict 稳定、context boundaries 和 OOM 等门全部通过。

这些阈值是预注册判据，不是运行结果。v2 不能倒写成 v1 通过，v2 执行后也不得继续放宽阈值。

## 6. Windows 测试环境恢复记录

旧会话冻结记录报告：Windows Code Integrity event 3077 曾阻止 `pyarrow\arrow_compute.dll`；之后 PyArrow 从 `25.0.0` 修复为 `23.0.1`，`datasets==5.0.1` 和 `torch==2.11.0+cu128` 保持不变，并报告 `3 / 110 / 822 passed`。

本次接续没有取得上述测试命令的完整 stdout/stderr，也没有重新运行测试。因此本报告将这些值分类为 `OLD_SESSION_REPORTED`，不写成当前接续重新验证的结果：

```text
Day14OldSessionReportedTests=3|110|822
Day14RecoveredRawTestStdout=UNKNOWN
TestsRerunDuringDocumentation=False
```

## 7. F0D v2-r1 隔离部署

冻结部署根：

```text
/home/jetson/small-gpt-day14-v2-r1
```

本地 v2-r1 runtime assets ZIP：

```text
Path=D:\model-backups\small-gpt\day14-v2-r1\packages\build-774cf358\small-gpt-day14-v2-r1-runtime-assets-774cf358.zip
Bytes=201013
SHA256=9ea9a979f9aa20465a1e1b9e108a6c1bdb6935f30152e9b927cc162ac65ea56e
EntryCount=5
```

只读 Jetson inventory 确认三份 F0D 文件：

```text
small-gpt-day14-v2-r1-transfer-manifest-774cf358.json
stage-f0d-isolation-inventory.json
stage-f0d-pip-freeze.txt
```

最后一个有直接证据的 PASS 门禁为：

```text
Day14V2StageF0DJetsonIsolatedDeployment=PASS
```

## 8. F1 只读证据清单

2026-08-22，用户仅授权重新给 Jetson 上电并执行 Day 14 F1 只读证据清单。连接使用既有专用有线 SSH 路径；没有使用显示器或键盘。

执行边界：

```text
PythonExecuted=False
CorrectnessExecuted=False
BenchmarkExecuted=False
StabilityExecuted=False
RemoteMutation=False
CheckpointRead=False
CheckpointCopied=False
CheckpointLinked=False
```

SSH 连通性结果：

```text
RemoteUser=jetson
RemoteHost=yahboom
SSHExit=0
```

inventory 实际输出：

```text
Directory=/home/jetson/small-gpt-day14-v2-r1|Exists=True
Directory=/home/jetson/small-gpt-day14-v2-r1/evidence|Exists=True
Directory=/home/jetson/small-gpt-day14-v2-r1/evidence/day14-v2|Exists=False
EvidenceFileCount=3
Day14F1RemoteEvidenceInventoryExit=0
```

三个文件全部属于 F0D。没有发现以下 F1 必需类别：

- v2 FP32/FP16 correctness summary 和 comparisons；
- full-prefix 与 KV-cache paired benchmark summary 和 samples；
- 30-request FP16 stability 输出；
- F1 tegrastats；
- 最终 F1 manifest 或 validation report。

## 9. 状态判定

| 项目 | 期望状态 | 当前证据 | 判定 |
| --- | --- | --- | --- |
| Day 14 功能代码 | 冻结并同步 | 功能 head 与 feature=`774cf358`；`main` 包含首个文档提交 `bd760a5` 及后续纯文档修正 | PASS |
| v1/v2 保留 | 不覆盖、不改写历史 | v1、v2、v2-r1 namespace 均保留 | PASS |
| F0D 隔离部署 | package/runtime/evidence 身份成立 | 三份 F0D evidence，既有 gate PASS | PASS |
| F1 evidence directory | `evidence/day14-v2` 存在 | 目录不存在 | FAIL |
| F1 correctness | 有可验证输出 | 未找到 | UNKNOWN |
| F1 paired benchmark | 有可验证输出 | 未找到 | UNKNOWN |
| F1 stability/thermal | 有可验证输出 | 未找到 | UNKNOWN |
| F1 final acceptance | final manifest/gate PASS | 未找到 | UNKNOWN |
| Day 14 整体 | 完整证据闭环 | F1 未闭环 | INCOMPLETE |

`F1 evidence directory=FAIL` 描述的是证据闭环缺失，不证明运行本身失败。证据不足的运行状态必须保留为 `UNKNOWN`。

## 10. 允许与禁止的结论

允许写入：

- KV Cache inference-only 功能已进入 `774cf358`；
- main 与 feature Git 身份已经对齐；
- v2-r1 F0D 隔离部署 PASS；
- F1 evidence 未找到，runtime acceptance UNKNOWN；
- Day 14 尚未完成。

禁止写入：

- Day 14 100% complete；
- v2 FP16 correctness PASS；
- KV Cache 达到任何 speedup；
- TTFT、decode latency、throughput 或 memory 的实测数字；
- 30-request stability 或 temperature PASS；
- 缺失证据等同于运行失败。

## 11. Jetson 最终状态

只读 inventory 后，用户明确授权安全关闭 Jetson。实际回报：关机命令已运行、SSH 显示连接关闭、物理电源已拔。

```text
JetsonShutdownCommand=PASS
SSHConnectionClosed=True
JetsonPhysicalPower=Disconnected
JetsonCurrentState=SAFE_POWERED_OFF
```

## 12. 文档闭环状态

本次授权只覆盖：

```text
README.md
reports/daily-log.md
reports/day-14-kv-cache-report.md
```

文档内容写入阶段没有执行测试、没有再次连接 Jetson，也没有 commit 或 push。随后用户单独授权精确暂存上述三个路径，创建首个文档提交 `bd760a526df713c22fa6c2d3ec3a0a1f0eecec1b` 并普通 push；未跟踪交接文档保持未跟踪且 SHA-256 不变。

首个文档版本中的写入前状态由后续新的纯文档提交修正，不 amend。为避免自引用，报告保留稳定的功能 head 和首个文档提交身份，不硬编码本次修正后的实时 `main` SHA。

```text
Day14FunctionalHead=774cf358be9822cdeb6a5921bc9068c1312bc192
Day14DocumentationLocalDraft=PASS
Day14DocumentationFirstCommit=bd760a526df713c22fa6c2d3ec3a0a1f0eecec1b
Day14DocumentationFirstPush=PASS
Day14DocumentationStatusCorrection=RECORDED_BY_THIS_COMMIT
PreservedUntrackedHandoff=True
Day14F1RuntimeAcceptance=UNKNOWN
Day14OverallStatus=INCOMPLETE
```

## 13. 最终结论

Day 14 的代码、协议、Git 同步和 F0D 隔离部署已有证据；F1 runtime acceptance 没有可恢复证据。最后一个有证据的 PASS 是 F0D，Day 14 整体状态必须保持 `INCOMPLETE`。

后续若不重新运行 F1，只能继续寻找既有、可验证且未被覆盖的原始证据；在找到之前不得补全任何性能或稳定性数据。
