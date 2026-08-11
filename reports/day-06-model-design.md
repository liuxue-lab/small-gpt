# Day 6 Decoder-only GPT 模型设计契约

> 状态：设计已冻结，模型实现尚未开始。
>
> 上游基线：Day 5 endpoint `a569e2c`。
>
> 本文件定义 Day 6 的模型架构、接口、参数量和测试边界。后续实现若改变其中任何语义，必须先更新设计并解释兼容性，不能静默漂移。

## 1. 目标

Day 6 建立模型最小正确闭环：

```text
input_ids [B,T]
→ token embedding + learned position embedding
→ N 个 Pre-LN Decoder Blocks
→ final LayerNorm
→ tied language-model head
→ logits [B,T,V]
→ next-token cross entropy
→ backward
→ 固定小批次过拟合
```

Day 6 不实现正式训练循环、scheduler、AMP、gradient accumulation、checkpoint、生成、KV cache 或正式预训练。

## 2. 上游数据契约

模型接收 Day 5 `CausalWindowDataset` 的输出：

```text
x.shape = [B,T]
y.shape = [B,T]
x.dtype = y.dtype = torch.int64
y[:, :-1] == x[:, 1:]
0 <= token_id < 16384
```

Day 5 已经完成 next-token shift，因此模型不得再次切片或移动 targets。

## 3. 模型配置

### 3.1 Debug

| 字段 | 值 |
| --- | ---: |
| layers | 2 |
| heads | 2 |
| embedding width | 128 |
| head dimension | 64 |
| FFN hidden | 512 |
| context length | 128 |
| vocabulary | 16,384 |
| dropout | 0.0 |
| exact parameters | 2,508,032 |

### 3.2 Baseline

| 字段 | 值 |
| --- | ---: |
| layers | 8 |
| heads | 8 |
| embedding width | 512 |
| head dimension | 64 |
| FFN hidden | 2,048 |
| context length | 512 |
| vocabulary | 16,384 |
| dropout | 0.0 |
| exact parameters | 33,833,984 |

### 3.3 维度约束

```text
C = n_embd
H = n_head
D = C / H
F = ffn_hidden
V = vocab_size
T <= context_length
C % H == 0
F = 4C
```

## 4. 冻结的架构选择

| 决策 | 冻结值 |
| --- | --- |
| architecture | `decoder_only_gpt` |
| norm order | Pre-LN |
| normalization | LayerNorm |
| LayerNorm epsilon | `1e-5` |
| LayerNorm affine | true |
| position encoding | learned absolute |
| Attention | 手写 scaled dot-product causal MHA |
| QKV | 一个 `C → 3C` 融合投影，随后显式 split |
| Attention scale | `1/sqrt(D)` |
| causal mask | boolean lower triangle，包含对角线 |
| softmax | FP32 计算后转回输入 dtype |
| Linear bias | false |
| MLP | `C → 4C → C` |
| activation | GELU，`approximate="tanh"` |
| LM head bias | false |
| weight tying | true |
| embedding/Linear init | `Normal(0, 0.02)` |
| residual output init | `0.02 / sqrt(2*n_layer)` |
| dropout | 读取配置；Debug/Baseline 为 0.0 |
| loss | 全部 `B*T` 位置的 cross entropy |

## 5. 模块结构

```text
model/
├── __init__.py
├── config.py
├── layers.py
├── attention.py
├── block.py
└── gpt.py
```

稳定公共 API：

```python
from model import GPT, GPTConfig, GPTOutput
```

核心模块可选择公开：

```python
from model import CausalSelfAttention, MLP, TransformerBlock
```

所有模块 import 时不得读取数据、访问网络、创建模型实例或启动训练。

## 6. Embedding

### 6.1 Token embedding

```python
nn.Embedding(vocab_size, n_embd)
```

输入必须是二维 `torch.long`，token ID 必须位于 `[0, vocab_size)`。

### 6.2 Position embedding

```python
nn.Embedding(context_length, n_embd)
```

每次前向使用：

```python
position_ids = torch.arange(T, device=input_ids.device)
x = token_embedding(input_ids) + position_embedding(position_ids)
```

位置从每个训练窗口的 0 开始，不使用语料全局 offset。

模型在 lookup 前主动拒绝 `T > context_length`。

## 7. Causal Multi-Head Self-Attention

### 7.1 Shape 流

```text
x                              [B,T,C]
qkv_proj(x)                    [B,T,3C]
q, k, v                        3 × [B,T,C]
reshape + transpose            3 × [B,H,T,D]
q @ k.transpose(-2,-1)         [B,H,T,T]
softmax(masked scores)         [B,H,T,T]
probabilities @ v              [B,H,T,D]
merge heads                    [B,T,C]
out_proj                       [B,T,C]
```

### 7.2 数学

```text
scores = QKᵀ / sqrt(D)
scores[i,j] = masked, if j > i
P = softmax(scores)
output = P V
```

缩放使用 head dimension `D`，不是 embedding dimension `C`。

### 7.3 Causal mask

最大 mask：

```python
torch.tril(
    torch.ones(context_length, context_length, dtype=torch.bool)
)
```

注册为形状 `[1,1,context_length,context_length]` 的 non-persistent buffer。

语义：

```text
j <= i  可见
j > i   不可见
```

mask 必须在 softmax 前应用。对角线必须可见，确保任意 query 行至少有一个合法 key。

### 7.4 数值稳定

```python
probs = torch.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
```

未来 BF16 训练仍使用该语义。Day 6 首版不调用 SDPA、FlashAttention、xFormers 或 Triton。

## 8. MLP

```text
[B,T,C]
→ Linear(C,F,bias=False)
→ GELU(approximate="tanh")
→ Linear(F,C,bias=False)
→ Dropout
→ [B,T,C]
```

`F` 必须来自 `ffn_hidden`，当前正式配置要求 `F=4C`。

## 9. Transformer Block

使用 Pre-LN：

```python
x = x + attention(norm1(x))
x = x + mlp(norm2(x))
```

每个 Block 包含两个独立 LayerNorm。Block 输入输出 shape 相同。

如果 Attention 与 MLP 的输出投影权重均为 0，Block 应退化为 identity；该性质进入单元测试。

## 10. GPT 顶层

```text
GPT
├── token_embedding
├── position_embedding
├── embedding_dropout
├── blocks × n_layer
├── final_norm
└── lm_head
```

### 10.1 Forward

```python
def forward(
    input_ids: torch.Tensor,
    targets: torch.Tensor | None = None,
) -> GPTOutput:
    ...
```

```python
@dataclass
class GPTOutput:
    logits: torch.Tensor
    loss: torch.Tensor | None
```

| 情况 | logits | loss |
| --- | --- | --- |
| 无 targets | `[B,T,V]` | `None` |
| 有 targets | `[B,T,V]` | finite scalar |

### 10.2 Loss

```python
F.cross_entropy(
    logits.reshape(-1, vocab_size),
    targets.reshape(-1),
)
```

不能在模型内部再次 shift，也不能先对 logits 做 softmax。

### 10.3 输入错误契约

主动拒绝：

- input 不是二维；
- input 不是 `torch.long`；
- batch 或 sequence 为空；
- `T > context_length`；
- token ID 越界；
- targets shape 与 input 不同；
- targets dtype 错误；
- input/targets device 不一致。

## 11. Weight tying

初始化完成后：

```python
self.lm_head.weight = self.token_embedding.weight
```

验收必须同时满足：

```python
model.lm_head.weight is model.token_embedding.weight
model.lm_head.weight.data_ptr() == model.token_embedding.weight.data_ptr()
```

复制数值不等于共享 Parameter。参数统计和未来 optimizer 必须去重。

## 12. 初始化

### 12.1 基础初始化

| 模块 | 初始化 |
| --- | --- |
| Embedding | Normal(0, 0.02) |
| 普通 Linear | Normal(0, 0.02) |
| LayerNorm weight | 1 |
| LayerNorm bias | 0 |

### 12.2 残差投影

以下权重使用：

```text
std = 0.02 / sqrt(2*n_layer)
```

- Attention `out_proj.weight`；
- MLP `fc_out.weight`。

建议顺序：构造 → 基础初始化 → 覆盖残差投影初始化 → 建立 weight tying。

## 13. 精确参数量

在当前冻结架构下：

```text
token embedding      = V*C
position embedding   = T*C
每层 Attention       = 4*C*C
每层 MLP             = 2*C*F
每层两个 LayerNorm   = 4*C
final LayerNorm      = 2*C
LM head              = 0 extra parameters（tied）
```

公式：

```text
P = V*C + T*C + L*(4*C*C + 2*C*F + 4*C) + 2*C
```

结果：

```text
Debug   = 2,508,032
Baseline = 33,833,984
```

Day 1～Day 5 的旧估算 `L*12*C*C` 没有统计 LayerNorm，因此约 33.82M 是合理的早期近似。Day 6 开始以精确值为测试契约。

## 14. 因果性验收

仅检查 mask shape 不足以证明正确。

构造相同前缀、不同未来的两条序列：

```text
ids_a = [same prefix | future A]
ids_b = [same prefix | future B]
```

在 `eval()` 且 dropout=0 时，未来变化之前的 logits 必须一致。Attention 层和完整 GPT 都需要行为测试。

## 15. 梯度验收

对有限 scalar loss 执行 `backward()` 后：

- token/position embeddings 有有限梯度；
- 每层 QKV/out projections 有有限梯度；
- 每层 MLP 有有限梯度；
- 每层 LayerNorm 和 final norm 有有限梯度；
- tied weight 不重复计数；
- logits、loss 和 gradients 不含 NaN/Inf。

## 16. State dict 边界

Day 6 验证：

```text
save state_dict to temporary path
→ construct same config
→ strict load
→ compare eval logits
→ verify tying remains
```

这不是正式 checkpoint：不保存 optimizer、scheduler、step、token counter 或 RNG。

## 17. 小样本过拟合

使用固定 synthetic batch：

```text
Debug model
dropout=0
fixed seed
fixed batch
temporary AdamW
weight_decay=0
100～300 steps
```

最低验收门：

```text
final_loss < 0.5 * initial_loss
```

诊断 optimizer 不进入未来正式训练配置。

## 18. 真实 Pilot smoke

从 Day 5 DataLoader 取得：

```text
x/y [4,128], torch.long
```

执行 Debug forward/backward，要求：

- logits `[4,128,16384]`；
- loss finite；
- gradients finite；
- 不修改数据；
- 使用后关闭 store。

## 19. 测试文件

```text
tests/test_model_config.py
tests/test_layers.py
tests/test_attention.py
tests/test_model.py
```

重点覆盖：

- 配置校验；
- shape；
- bias 契约；
- mask 与因果行为；
- Pre-LN residual；
- exact parameter count；
- weight tying；
- loss；
- gradients；
- state dict round-trip；
- 输入错误。

单元测试默认使用小型配置且不访问网络或真实 Pilot。真实 Pilot 作为单独集成 smoke。

## 20. Day 6 完成边界

Day 6 完成必须同时具备：

```text
design frozen
→ implementation
→ model unit tests
→ exact counts
→ causal behavior
→ forward/backward
→ fixed-batch overfit
→ real Pilot smoke
→ full regression
→ report
→ commits
→ push
```

未完成正式训练循环、AMP、scheduler、checkpoint、生成或正式预训练。这些属于下一阶段。
