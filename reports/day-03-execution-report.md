# Small GPT 从零训练：Day 3 执行报告

> 执行日期：2026-08-07 ～ 2026-08-08
> 项目：从零训练约 34M 参数 Decoder-only GPT
> 本地仓库：`D:\code\small-gpt`
> 分支：`main`

## 1. Day 3 目标

Day 3 的目标是把 Day 2 的小样本数据处理程序扩展为可用于正式语料采集的流式管线，并完成一个受控 Pilot 闭环。

核心要求：

- 固定 FineWeb-Edu 数据集版本；
- 使用 streaming，避免完整下载原始语料；
- 按 provided-token budget 停止；
- 清洗、过滤和确定性划分；
- 跨 shard、跨 split 的 SHA-256 精确去重；
- 原子 shard 输出；
- manifest、state 和文件校验；
- 中断恢复、幂等重跑和损坏检测；
- 生成语料不得进入 Git。

## 2. 仓库卫生修复

Day 3 开始时先完成两个遗留问题：

- 修复 `reports/day-02-data-audit.md` 的截断和未闭合代码围栏；
- 在 `.gitignore` 中加入 `data/tokenized/`。

提交：

```text
94c5d9d chore: repair Day 2 audit artifacts
```

## 3. 数据集与配置冻结

数据集身份：

| 项目 | 值 |
| --- | --- |
| Dataset | `HuggingFaceFW/fineweb-edu` |
| Configuration | `sample-10BT` |
| Split | `train` |
| Revision | `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9` |
| Access | Streaming |

清洗和划分：

| 参数 | 值 |
| --- | ---: |
| Minimum characters | 200 |
| Minimum language score | 0.65 |
| Minimum quality score | 3 |
| Split seed | 42 |
| Train ratio | 0.98 |
| Validation ratio | 0.01 |
| Test ratio | 0.01 |

采集配置：

| Profile | Target provided tokens | Shard target | Estimated shard groups |
| --- | ---: | ---: | ---: |
| Pilot | 2,000,000 | 500,000 | 4 |
| Full | 350,000,000 | 5,000,000 | 70 |

新增文件：

```text
configs/data_fineweb_edu.yaml
tests/test_data_config.py
```

数据配置测试和完整回归：

```text
4 passed in 0.03s
25 passed in 4.18s
```

提交：

```text
c5ba120 feat: add FineWeb-Edu collection configuration
```

## 4. 磁盘预算

根据 Day 2 的 1,000 条清洗样本：

| 项目 | 结果 |
| --- | ---: |
| 清洗后 JSONL | 5,279,467 bytes |
| Provided tokens | 1,057,560 |
| 平均占用 | 约 4.99 bytes/token |
| 2M Pilot 预计 | 约 10 MB |
| 350M Full 预计 | 约 1.75 GB |
| 冻结预留空间 | 5 GB |

正式采集只保存清洗后的 UTF-8 JSONL，不保存原始 full-corpus shard。

## 5. 管线设计

设计文档：

```text
reports/day-03-data-pipeline-design.md
```

输出结构：

```text
data/processed/fineweb_edu_corpus/
├── manifest.json
├── state.json
└── shards/
    └── shard-xxxxx/
        ├── train.jsonl
        ├── validation.jsonl
        ├── test.jsonl
        └── metadata.json
```

每个 shard group 先写入临时目录，完成文件刷新、统计和 SHA-256 后再原子重命名。恢复时只信任已完成 shard，并从完成的 JSONL 重建全局文本 hash 集合。

提交：

```text
311e509 docs: define streaming corpus pipeline design
```

## 6. 流式管线实现

新增文件：

```text
scripts/build_fineweb_edu_corpus.py
tests/test_streaming_data_pipeline.py
```

实现能力：

- 配置读取和配置指纹；
- Hugging Face streaming；
- provided-token budget；
- 文档边界感知的 shard；
- 三 split JSONL；
- 清洗和过滤复用；
- 全局 SHA-256 精确去重；
- 原子 JSON、manifest、state 和 shard；
- 完成 shard 恢复；
- 文件大小、SHA-256、JSON、split 和 hash 校验；
- 幂等完成态重跑；
- 文件损坏检测。

新增离线测试：

```text
4 passed in 0.27s
```

完整项目测试：

```text
29 passed in 32.22s
```

提交：

```text
17bda79 feat: add resumable FineWeb-Edu corpus builder
```

## 7. 真实网络 Smoke

Smoke 参数：

| 项目 | 值 |
| --- | ---: |
| Target provided tokens | 50,000 |
| Shard target | 25,000 |
| Output | `data/processed/fineweb_edu_smoke` |

结果：

| 指标 | 结果 |
| --- | ---: |
| Status | `complete` |
| Input records | 64 |
| Kept records | 64 |
| Kept provided tokens | 53,004 |
| Shard groups | 2 |
| Train records | 63 |
| Validation records | 0 |
| Test records | 1 |
| Unique text hashes | 64 |
| Removed records | 0 |

Smoke 验收结果：

- 固定 revision 可连接；
- manifest 和 state 正常；
- 所有 JSONL、文件大小和 SHA-256 通过恢复校验；
- 全局 hash 无重复；
- 第二次同配置运行后所有文件哈希保持不变；
- 输出目录被 Git 正确忽略。

Validation 为 0 是由 Smoke 样本量很小导致，不代表划分错误。

## 8. 2M Pilot

Pilot 参数：

| 项目 | 值 |
| --- | ---: |
| Target provided tokens | 2,000,000 |
| Shard target | 500,000 |
| Output | `data/processed/fineweb_edu_corpus` |

Pilot 结果：

| 指标 | 结果 |
| --- | ---: |
| Status | `complete` |
| Input records | 1,852 |
| Kept records | 1,852 |
| Kept provided tokens | 2,000,083 |
| Shard groups | 4 |
| Train records | 1,824 |
| Validation records | 16 |
| Test records | 12 |
| Train provided tokens | 1,967,041 |
| Validation provided tokens | 12,600 |
| Test provided tokens | 20,442 |
| Unique text hashes | 1,852 |
| Removed records | 0 |
| Output files | 18 |
| Total disk size | 9.49 MiB |

实际 token 数比目标多 83，是因为管线不会拆分文档，属于预期的文档边界超出。

Pilot 验收结果：

- 4 个连续 shard groups；
- 不存在临时 `.tmp` shard；
- 18 个输出文件齐全；
- 所有 JSONL 可解析；
- metadata、文件大小和 SHA-256 一致；
- `kept_records == unique_hashes == 1852`；
- 三个 split 互斥；
- 输入、保留、删除和 split 统计守恒；
- 同配置重跑统计完全一致；
- 重跑前后所有输出文件哈希一致；
- Pilot 数据未进入 Git。

## 9. 网络现象

真实 streaming 过程中出现过 Hugging Face 未认证请求警告，以及一次短暂的 `Server disconnected` 自动重试。两次任务均成功生成 manifest 并正常返回 PowerShell 提示符，未影响数据一致性。

当前 Pilot 规模不要求配置 `HF_TOKEN`。未来运行更大规模采集时，可根据限流和稳定性再决定是否配置认证。

## 10. Day 3 最终成果

Day 3 已完成：

- Day 2 遗留文档和忽略规则修复；
- FineWeb-Edu 数据身份、预算和磁盘空间冻结；
- 可恢复流式分片架构设计；
- 流式管线实现；
- 离线恢复、幂等和损坏测试；
- 真实网络 Smoke；
- 2M Pilot；
- manifest、SHA-256、split、去重和统计守恒验证；
- Git 忽略验证。

## 11. 尚未执行

以下内容不属于本次 Day 3 已完成范围：

- 350M provided-token Full 采集；
- 项目 BPE Tokenizer 训练；
- 使用项目 Tokenizer 重新统计真实训练 tokens；
- Decoder-only GPT 模型实现；
- AutoDL 正式训练。

## 12. 后续建议

下一阶段开始前，应先确定：

1. 是否立即运行 350M Full 采集，或先用 Pilot 数据验证 Tokenizer 全流程；
2. BPE Tokenizer 的训练语料范围、词表大小和特殊 token；
3. Full 采集放在本地还是 AutoDL 数据盘；
4. Full 采集完成后的独立校验和备份策略。

注意：FineWeb-Edu 的 `provided_token_count` 是 GPT-2 tokenizer 口径，只用于采集预算。项目 BPE 完成后必须重新统计真实 token 数。
