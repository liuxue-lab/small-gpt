# Small GPT 开发日志

## Day 1：项目初始化与环境验证

日期：2026-08-06

### 今日目标

- 建立 Small GPT 项目目录结构；
- 创建独立 Python 虚拟环境；
- 安装支持 CUDA 的 PyTorch；
- 建立 Debug 和 Baseline 两套配置；
- 建立基础自动测试；
- 明确本地开发与租用 GPU 训练的分工。

### 已完成任务

- [x] 安装 Python 3.11.9
- [x] 安装 Git 2.55.0
- [x] 创建 `.venv` 虚拟环境
- [x] 安装 PyTorch 2.11.0+cu128
- [x] 验证 CUDA 可用
- [x] 验证 GPU 前向传播和反向传播
- [x] 创建项目目录结构
- [x] 创建 `.gitignore`
- [x] 创建 `debug.yaml`
- [x] 创建 `baseline.yaml`
- [x] 创建环境检查脚本
- [x] 创建配置检查脚本
- [x] 创建基础 PyTest 测试
- [x] 编写项目 README

### 本地开发环境

| 项目 | 当前结果 |
|---|---|
| 操作系统 | Windows 10 |
| Python | 3.11.9 |
| PyTorch | 2.11.0+cu128 |
| PyTorch CUDA Runtime | 12.8 |
| 本地 GPU | NVIDIA GeForce RTX 5060 Laptop GPU |
| GPU 显存 | 7.96 GiB |
| CUDA 是否可用 | True |
| BF16 是否支持 | True |

### 模型配置

#### Debug 模型

- 参数量：约 2.51M
- Transformer 层数：2
- 注意力头数：2
- 隐藏维度：128
- 上下文长度：128
- 用途：本地代码调试、单元测试和单批次过拟合。

#### Baseline 模型

- 参数量：约 33.82M
- Transformer 层数：8
- 注意力头数：8
- 隐藏维度：512
- 上下文长度：512
- 词表大小：16,384
- 用途：在租用的 Linux GPU 上进行正式预训练。

### 验证结果

运行环境检查：

```powershell
python scripts/check_env.py