# Small GPT from Scratch：Day 6 Decoder-only GPT 模型实现与验证执行报告

> 项目：从零实现并预训练约 34M 参数的 Decoder-only GPT
>
> 阶段：Day 6——模型结构、因果性、梯度与真实 Pilot 前后向验证
>
> 执行日期：2026-08-11
>
> 执行位置：Windows 本地 `D:\code\small-gpt`
>
> 本地 GPU：NVIDIA GeForce RTX 5060 Laptop GPU
>
> 上游数据：Day 5 FineWeb-Edu tokenized Pilot
>
> 技术验收状态：通过

## 1. 执行结论

Day 6 已完成从架构契约、手写实现、自动测试到真实 tokenized Pilot 前后向的完整模型闭环：

- 冻结并实现 Decoder-only GPT、Pre-LayerNorm、learned absolute position embedding；
- 从矩阵运算开始手写 causal multi-head self-attention，没有调用 PyTorch SDPA 代替核心计算；
- 实现融合 QKV projection、显式 head 拆分/合并、`1/sqrt(head_dim)` 缩放和下三角因果 mask；
- Attention softmax 显式使用 FP32 计算，再转换回输入 dtype；
- 实现 GELU MLP、两条 Pre-LN residual path、final LayerNorm 和 tied LM head；
- 实现严格的 `GPTConfig`、`GPTOutput(logits, loss)` 和输入错误检查；
- loss 直接使用 Day 5 已经右移一位的 targets，没有在模型内部二次 shift；
- 实现 GPT 风格初始化和残差投影缩放初始化；
- Debug 模型精确参数量为 2,508,032；
- Baseline 模型精确参数量为 33,833,984；
- 113 个模型专项用例全部通过；
- 完整项目回归为 `216 passed in 9.94s`；
- Debug synthetic batch 在本地 CUDA 上完成有限 forward/backward；
- 固定 synthetic batch 在 200 steps 内从 loss `9.704769` 降至 `0.003591`；
- Day 5 真实 Pilot batch 完成有限 forward/backward，因果 `x/y` 对齐保持正确；
- 没有生成 checkpoint、没有启动正式训练、没有启动 AutoDL，也没有采集 350M Full 语料。

Day 6 回答了“Day 5 DataLoader 输出能否被一个正确的 Decoder-only GPT 接收，并产生可优化的 next-token loss”。答案已经由自动测试、合成数据和真实 Pilot 三层证据共同确认。

## 2. 阶段目标与边界

Day 5 已经提供：

```text
x = [t0, t1, ..., t(T-1)]
y = [t1, t2, ..., tT]
dtype = torch.int64
```

Day 6 的输入边界因此固定为：

1. 模型接收二维 `torch.long` token ID；
2. `input_ids` 与 `targets` shape 必须完全一致；
3. 模型不再次 shift targets；
4. 模型不感知 `.bin/.idx`、memmap 或 storage shard；
5. Attention 只能查看当前位置和过去位置；
6. Debug/Baseline 使用同一套实现，不维护两份模型代码；
7. Day 6 只完成模型与诊断闭环，不实现正式 optimizer/scheduler/checkpoint 训练系统。

本阶段明确没有执行：

- 350M Full 数据采集或 tokenization；
- AutoDL 启动或远程长时间训练；
- 混合精度正式训练循环；
- optimizer/scheduler 状态保存；
- checkpoint 恢复；
- 文本生成、采样和评估；
- 正式预训练。

## 3. 冻结的模型架构

| 项目 | 冻结实现 |
| --- | --- |
| Architecture | Decoder-only GPT |
| Block order | Pre-LayerNorm |
| Position encoding | Learned absolute position embedding |
| Attention | 手写 scaled dot-product causal MHA |
| QKV | 单个 `Linear(C, 3C)` 后显式 split |
| Attention scale | `1 / sqrt(head_dim)` |
| Causal mask | boolean lower triangle，对角线可见 |
| Mask storage | non-persistent buffer |
| Attention softmax | FP32 计算后转换回 query dtype |
| Attention dropout | 配置值；Debug/Baseline 为 0.0 |
| Linear bias | QKV、attention output、MLP 均为 false |
| LayerNorm | affine=true，`eps=1e-5` |
| MLP | `C -> 4C -> C` |
| Activation | GELU，`approximate="tanh"` |
| Final norm | 有 |
| LM head bias | false |
| Weight tying | token embedding 与 LM head 共享同一 Parameter |
| Base initialization | `Normal(mean=0, std=0.02)` |
| Residual projections | `0.02 / sqrt(2 * n_layer)` |

### 3.1 Debug 与 Baseline

| 配置 | Layers | Heads | Hidden | FFN | Context | Vocab | 精确参数量 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Debug | 2 | 2 | 128 | 512 | 128 | 16,384 | 2,508,032 |
| Baseline | 8 | 8 | 512 | 2,048 | 512 | 16,384 | 33,833,984 |

两套配置的 `head_dim` 都为 64。项目配置额外冻结 `tie_embeddings=true`、`linear_bias=false`、`lm_head_bias=false` 和 `scale_residual_projections=true`，未知字段与拼写错误不会被静默忽略。

## 4. 实现结构

| 文件 | 作用 |
| --- | --- |
| `model/config.py` | 冻结配置 dataclass、YAML 加载、字段与跨字段校验、参数量公式 |
| `model/layers.py` | token/position embedding 组合层与 GELU MLP |
| `model/attention.py` | 手写 causal multi-head self-attention |
| `model/block.py` | 两条 residual 的 Pre-LN Transformer Block |
| `model/gpt.py` | embedding、blocks、final norm、tied head、初始化和 loss |
| `model/__init__.py` | 暴露稳定公共 API |
| `scripts/inspect_model.py` | synthetic/Pilot batch、backward 和固定 batch 过拟合诊断 |
| `tests/test_model_config.py` | 配置、错误输入和精确参数量契约 |
| `tests/test_layers.py` | Embedding、MLP、LayerNorm 与 Block 行为 |
| `tests/test_attention.py` | Attention shape、mask、因果性、概率和梯度 |
| `tests/test_model.py` | GPT、loss、tying、初始化、因果性、梯度和 state dict |

稳定公共 API：

```python
from model import GPT, GPTConfig, GPTOutput
```

可单独检查的构件：

```python
from model import CausalSelfAttention, MLP, TransformerBlock
```

导入 `model` 不会读取数据、访问网络、构造真实模型或启动训练。

## 5. Attention 实现与因果性

对 `X` 的核心计算为：

```text
[Q, K, V] = X W_qkv
Q, K, V   -> [B,H,T,D]
scores    = Q K^T / sqrt(D)
scores    = causal_mask(scores)
probs     = softmax(scores, FP32)
output    = concat(probs V) W_o
```

最大上下文 mask 在模块构造时创建为 `[1,1,Tmax,Tmax]` boolean buffer，并以 `persistent=False` 注册。前向只切片当前序列区域，不把 mask 注册成 Parameter，也不写入 state dict。

Attention 专项测试直接证明：

- QKV 和 output projection shape 正确且没有 bias；
- split/merge heads 可以无损往返；
- scale 使用 `head_dim`，不是 hidden size 或 head count；
- causal mask 为下三角且对角线可见；
- 被 mask 的未来概率精确为 0；
- 每行可见概率和为 1；
- sequence length 1 与最大上下文均无 NaN/Inf；
- 修改未来隐藏状态不会改变更早位置输出；
- backward 能到达输入、QKV 和 output projection；
- monkeypatch 禁止 SDPA 后前向仍能执行。

整模因果性测试进一步修改未来 token，并比较 GPT 的过去 logits，确认位置 embedding、Block、final norm 和 tied head 的组合没有引入未来信息泄漏。

## 6. Block、loss、tying 与初始化

### 6.1 Pre-LN Block

Block 严格执行：

```python
x = x + attention(norm1(x))
x = x + mlp(norm2(x))
```

自动测试验证了两个独立 LayerNorm、调用顺序、两条 residual、shape、全部梯度，以及 Attention/MLP 置零时 Block 精确退化为 identity。

### 6.2 Forward 与 loss

```python
output = model(input_ids, targets)
logits = output.logits  # [B,T,V]
loss = output.loss      # scalar
```

loss 为：

```python
cross_entropy(
    logits.reshape(-1, vocab_size),
    targets.reshape(-1),
)
```

模型没有执行以下错误操作：

- 不在内部再次 shift targets；
- 不丢弃最后一个位置；
- 不先对 logits 手动 softmax；
- 不使用 PAD ignore index；
- 不把 Day 5 storage 边界带入模型。

### 6.3 Weight tying

初始化完成后执行：

```python
lm_head.weight = token_embedding.weight
```

测试确认：

- 两侧是同一个 `Parameter`；
- 两侧 `data_ptr()` 相同；
- 修改 token embedding 会立即反映到 LM head；
- `named_parameters()` 和 optimizer 输入只包含一次共享权重；
- strict state dict round-trip 后 tying 仍然存在。

### 6.4 初始化顺序

初始化顺序固定为：

```text
Embedding/Linear/LayerNorm 基础初始化
-> Attention out_proj 与 MLP fc_out 残差缩放
-> 最后建立 LM head tying
```

自动测试验证所有参数有限、LayerNorm weight/bias 精确为 1/0、Linear 不存在意外 bias、残差投影经验标准差显著低于普通输入投影，以及固定 seed 可复现完整 state dict。

## 7. 精确参数量

在 tied embeddings、bias-free Linear、每层两个 affine LayerNorm 和 final LayerNorm 下：

```text
P = V*C
  + context*C
  + L*(4*C*C + 2*C*F + 4*C)
  + 2*C
```

### 7.1 Debug

```text
V=16384, C=128, context=128, L=2, F=512
P=2,508,032
```

### 7.2 Baseline

```text
V=16384, C=512, context=512, L=8, F=2048
P=33,833,984
```

配置公式和实际 `sum(p.numel() for p in model.parameters())` 对两套模型完全一致。LM head 没有额外增加 `V*C` 参数，causal mask 也没有被计入参数量。

## 8. 模型专项测试

Stage F 最终四组专项测试：

| 测试文件 | 结果 |
| --- | ---: |
| `tests/test_model_config.py` | 35 passed |
| `tests/test_layers.py` | 23 passed |
| `tests/test_attention.py` | 23 passed |
| `tests/test_model.py` | 32 passed |
| **合计** | **113 passed** |

模型专项测试覆盖：

- 两套正式配置构造和精确参数量；
- 非正维度、bool 伪装整数、dropout/eps/init std 与冻结字段错误；
- Embedding shape、position IDs、token 范围和最大上下文；
- MLP shape、bias、GELU、dropout 与 backward；
- Block Pre-LN、residual、identity 和全部参数梯度；
- Attention scale、mask、概率、因果性、FP32 softmax 和 backward；
- GPT logits/loss、无二次 shift、tying 和初始化；
- 整模未来信息隔离；
- 全模型连续两次 forward/backward；
- optimizer 参数去重；
- state dict strict round-trip；
- 加载后 tying；
- 固定 seed 初始化复现；
- causal mask 不进入参数或 state dict。

## 9. Synthetic forward/backward

执行命令：

```powershell
python .\scripts\inspect_model.py `
  --config .\configs\debug.yaml `
  --device auto `
  --batch-source synthetic `
  --batch-size 4 `
  --sequence-length 128 `
  --seed 42 `
  --backward
```

真实结果：

| 项目 | 结果 |
| --- | --- |
| Device | `cuda` |
| Parameters | 2,508,032 |
| Trainable parameters | 2,508,032 |
| Weight tying | `True` |
| Input/target | `(4,128)` / `torch.int64` |
| Causal `x/y` shift | `True` |
| Input token range | `[9,16370]` |
| Logits | `(4,128,16384)` |
| Loss | `9.726461` |
| Logits finite | `True` |
| Gradient tensors | 20 |
| Nonzero gradients | 20 |
| Gradients finite | `True` |

随机初始化 loss 接近 `ln(16384) ≈ 9.704`，符合未训练 16K 类 next-token 预测的预期量级。

## 10. 固定 batch 过拟合

执行设置：

| 项目 | 值 |
| --- | ---: |
| Device | CUDA |
| Seed | 42 |
| Batch source | synthetic，生成一次后固定复用 |
| Batch size | 4 |
| Sequence length | 32 |
| Optimizer | diagnostic AdamW |
| Learning rate | 0.003 |
| Weight decay | 0.0 |
| Steps | 200 |

真实结果：

| 指标 | 结果 |
| --- | ---: |
| Initial loss | 9.704769 |
| Final loss | 0.003591 |
| Final / Initial | 约 0.037% |
| 验收上限 | 50% |
| Gradient tensors | 20 |
| Nonzero gradients | 20 |
| Gradients finite | True |
| Elapsed | 6.66 seconds |

final loss 远低于 `0.5 * initial loss` 的最低门槛，证明 targets 对齐、loss 路径、反向传播和参数更新方向正确。该 optimizer 仅用于诊断，没有进入正式训练配置，也没有保存模型权重。

## 11. 真实 Pilot forward/backward

运行时 manifest：

```text
data/tokenized/fineweb_edu_pilot/manifest.json
```

执行命令：

```powershell
python .\scripts\inspect_model.py `
  --config .\configs\debug.yaml `
  --device cuda `
  --batch-source pilot `
  --manifest .\data\tokenized\fineweb_edu_pilot\manifest.json `
  --batch-size 4 `
  --sequence-length 128 `
  --num-workers 0 `
  --seed 42 `
  --backward
```

真实结果：

| 项目 | 结果 |
| --- | --- |
| Manifest exists | `True` |
| Device | `cuda` |
| Parameters | 2,508,032 |
| Weight tying | `True` |
| Input/target | `(4,128)` / `torch.int64` |
| Causal `x/y` shift | `True` |
| Input token range | `[15,15865]` |
| Logits | `(4,128,16384)` |
| Loss | `9.744143` |
| Logits finite | `True` |
| Gradient tensors | 20 |
| Nonzero gradients | 20 |
| Gradients finite | `True` |
| Elapsed | 4.97 seconds |

真实 Pilot loss 只用于验证未训练模型的数值与梯度闭环，不代表模型质量。`num_workers=0` 用于本次模型 smoke；Day 5 已经独立验证 Windows `num_workers=2`，无需在模型阶段重复证明数据 worker 契约。

`SplitTokenStore` 在取得 batch 后关闭。运行没有修改 manifest、`.bin/.idx`、Tokenizer 或任何数据文件。

## 12. 完整项目回归

最终执行：

```powershell
python .\scripts\check_config.py
python -m pytest -q
```

配置检查：

```text
Debug exact parameters    = 2,508,032
Baseline exact parameters = 33,833,984
All configuration checks passed.
```

完整回归：

```text
216 passed in 9.94s
```

PowerShell 外层计时为 13.57 seconds。没有 failed、error 或 skipped。回归包含 Day 1～Day 5 的环境、数据、Tokenizer、binary/Dataset 测试，以及 Day 6 新增的配置、层、Attention 和 GPT 测试。

## 13. Git 交付结构

截至执行报告生成前，Day 6 已创建五个语义提交：

| Commit | 内容 |
| --- | --- |
| `06efac4` | `feat: define decoder-only GPT model contract` |
| `4632aa2` | `feat: add GPT config embeddings and MLP` |
| `053ce40` | `feat: add handwritten causal self-attention` |
| `1bf1a52` | `feat: assemble decoder-only GPT model` |
| `3eddd2b` | `test: verify full GPT model contract` |

这些提交建立在 Day 5 最终提交 `a569e2c` 之上。执行报告、daily log 和 README 作为 Day 6 文档收口单独精确暂存；技术验收完成前没有 push 中间状态。

## 14. 关键问题与处理

### 14.1 不对 Day 5 targets 二次 shift

Day 5 已经输出右移后的 `x/y`。如果模型内部再执行 `logits[:, :-1]` 与 `targets[:, 1:]`，目标会变成预测下下个 token。实现直接 flatten 同位置 logits/targets，并用独立单元测试与固定 batch 过拟合证明对齐正确。

### 14.2 不用 mask shape 代替因果行为证明

仅检查三角形状无法发现 transpose、切片或广播错误。项目同时检查未来概率为 0、Attention 前缀隔离和整个 GPT 的 token-level 前缀隔离。

### 14.3 不调用 SDPA 隐藏首版核心实现

首版 Attention 显式执行 QKV、matmul、scale、mask、softmax、value aggregation 和 head merge。测试 monkeypatch 禁止 `scaled_dot_product_attention`，前向仍能运行。

### 14.4 Tied weight 必须是真共享

参数量接近不能证明 tying。项目同时验证对象 identity、storage pointer、参数遍历去重、optimizer 去重和 state dict 加载后的 tying。

### 14.5 Windows LF/CRLF 提示

Git 多次提示工作区 LF 将来可能转为 CRLF。每个阶段均运行 `git diff --check`，没有尾随空白或补丁错误；行尾提示没有被误判为测试失败。

### 14.6 测试与正式训练边界

固定 batch 过拟合只使用临时 AdamW，权重保存在内存中并随进程退出丢弃。没有把该 optimizer 误当作正式训练系统，也没有生成 `.pt/.ckpt`。

## 15. 安全与范围检查

| 项目 | 结果 |
| --- | --- |
| AutoDL | 未启动 |
| 350M Full corpus | 未采集 |
| Full tokenization | 未执行 |
| 正式预训练 | 未启动 |
| Checkpoint | 未生成 |
| Day 5 tokenized Pilot | 只读复用 |
| 数据/Tokenizer | 未修改 |
| 网络访问 | 模型测试和 smoke 不需要 |
| Git 大文件 | 未新增 |

## 16. Day 6 验收结果

- [x] 模型架构与配置契约冻结
- [x] GPTConfig 严格校验通过
- [x] Token 与 learned position embedding 完成
- [x] GELU MLP 完成
- [x] 手写 causal multi-head self-attention 完成
- [x] Pre-LN Transformer Block 完成
- [x] Final LayerNorm 与 tied LM head 完成
- [x] Next-token cross entropy 无二次 shift
- [x] GPT 风格初始化与残差缩放完成
- [x] Debug/Baseline 精确参数量通过
- [x] Attention 与整模因果行为通过
- [x] 全模型梯度、二次 backward 和 optimizer 去重通过
- [x] State dict strict round-trip 与固定 seed 复现通过
- [x] 113 个模型专项用例通过
- [x] Synthetic CUDA forward/backward 通过
- [x] 固定 batch 过拟合通过
- [x] 真实 Pilot forward/backward 通过
- [x] 完整项目 216 项回归通过
- [x] 未生成正式训练产物

## 17. 下一阶段

Day 7 开始进入正式训练系统，但仍先使用 Debug 配置闭环：

1. 冻结 optimizer、weight decay 参数分组和学习率调度契约；
2. 实现 gradient accumulation 与 token/step 计数；
3. 实现 BF16/FP32 autocast 与 GradScaler 边界；
4. 实现 validation loss 和评估频率；
5. 实现 checkpoint 原子保存、strict 恢复和 RNG/optimizer/scheduler 状态；
6. 使用 Debug 模型执行短训练与中断恢复测试；
7. 通过后再迁移 Baseline 到租用 GPU；
8. 350M Full 数据与正式预训练仍需独立验收后启动。

Day 6 的最终边界为：

```text
FineWeb-Edu Pilot
-> 16K ByteLevel BPE
-> uint16 tokenized corpus
-> reproducible x/y DataLoader
-> handwritten Decoder-only GPT
-> finite next-token loss
-> correct backward
-> fixed-batch overfit
```

项目已经第一次具备一个可接受真实语料 batch、产生有限 loss 并正确反向传播的完整 Decoder-only GPT。
