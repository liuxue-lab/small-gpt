# Small GPT from Scratch

使用 PyTorch 从零实现并预训练一个约 34M 参数的 Decoder-only GPT。

本项目的重点不是训练通用聊天助手，而是完整掌握语言模型的数据处理、Tokenizer、模型结构、预训练、评估和推理流程。

## 项目目标

- 清洗并划分英文预训练语料；
- 从训练语料训练 BPE Tokenizer；
- 从零实现 Causal Self-Attention；
- 从零实现 Decoder-only Transformer；
- 实现单卡混合精度训练；
- 支持 checkpoint 保存与恢复；
- 训练 3 亿～5 亿 tokens；
- 实现文本续写与采样；
- 完成至少一个消融实验。

## 模型基线

| 配置项 | 数值 |
|---|---:|
| Transformer layers | 8 |
| Attention heads | 8 |
| Hidden size | 512 |
| Head dimension | 64 |
| FFN hidden size | 2048 |
| Context length | 512 |
| Vocabulary size | 16,384 |
| Approximate parameters | 33.82M |
| Minimum training tokens | 300M |
| Target training tokens | 500M |

## 开发与训练分工

| 本地 Windows 电脑 | 租用的 Linux GPU |
|---|---|
| 编写和阅读代码 | 正式预训练 |
| PyTest 单元测试 | 完整数据处理 |
| 运行 2.51M Debug 模型 | 运行 33.82M Baseline 模型 |
| 单批次过拟合 | 训练 3 亿～5 亿 tokens |
| 检查少量数据 | 保存正式 checkpoint |
| 分析训练日志 | 长时间运行训练任务 |

本地 `.venv` 不复制到租用服务器。服务器会根据其 GPU、驱动和 CUDA 环境重新创建 Python 虚拟环境并安装 PyTorch。

## 当前配置

项目包含两套配置：

- `configs/debug.yaml`：本地快速调试；
- `configs/baseline.yaml`：租用 GPU 正式训练。

## 本地环境验证

检查 PyTorch、CUDA、GPU、前向传播和反向传播：

`python scripts/check_env.py`

检查模型配置和参数量：

`python scripts/check_config.py`

运行全部自动测试：

`pytest -q`

## 当前测试

- PyTorch 导入测试；
- 矩阵乘法维度测试；
- Autograd 梯度测试；
- Debug 模型配置测试；
- Baseline 模型配置测试；
- 数据集划分测试；
- Tokenizer 与模型词表一致性测试。

当前状态：7 个测试通过。

## 项目结构

- `configs/`：模型和训练配置；
- `data/`：数据说明与处理结果；
- `tokenizer/`：BPE Tokenizer；
- `model/`：Attention、Transformer Block 和 GPT；
- `train/`：数据加载、训练循环和 checkpoint；
- `eval/`：验证损失与文本生成；
- `tests/`：自动测试；
- `scripts/`：环境检查和任务入口；
- `reports/`：开发日志和实验报告。

## 开发里程碑

- [x] 创建独立 Python 环境
- [x] 安装 CUDA 版 PyTorch
- [x] 创建项目目录结构
- [x] 创建 Debug 和 Baseline 配置
- [x] 建立基础测试系统
- [ ] 数据集审计与清洗
- [ ] BPE Tokenizer
- [ ] GPT 模型
- [ ] 训练系统
- [ ] 单批次过拟合
- [ ] 正式预训练
- [ ] 模型评估
- [ ] 消融实验

## 当前阶段

Day 1：环境、项目结构、配置与测试系统。