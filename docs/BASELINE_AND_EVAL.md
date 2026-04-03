# 基线模型与评估扩展（阶段 F）

本文档对应 `memory-bank/@prototype-survey-alignment-plan.md` **阶段 F**：第二基线、评估层级、挑战集评估与漏洞库数据准备度自检。

## 统一嵌入 Schema

以下路径产出的 JSON 均符合 `memory-bank/@architecture.md` 中 **EmbeddingDict / EmbeddingItem**（`name` 或 `function_id` 由脚本在写出时统一；`vector` 为浮点列表，基线默认为 **128 维**，与 `MultiModalFusionModel` 输出一致，便于 `scripts/eval_bcsd.py` 直接消费）。

| 模型 | 模块 / 脚本入口 |
|------|-----------------|
| SemPatch（多模态） | `features.inference.embed_batch`；`build_embeddings_db.py --model sempatch` |
| SAFE（指令级 pcode 序列） | `features.baselines.safe.embed_batch_safe`；`--model safe` |
| jTrans 风格（块序 `@块:opcode` 序列） | `features.baselines.jtrans_style.embed_batch_jtrans_style`；`--model jtrans_style` |

**说明**：`jtrans_style` 为仓库内 **近似实现**（Ghidra 多模态特征 + 轻量序列编码器），**不是** [vul337/jTrans](https://github.com/vul337/jTrans) 官方预训练模型。若需外部 jTrans 权重，请在其仓库生成向量后，编写独立转换脚本映射为本项目的 `EmbeddingDict`（字段对齐即可）。

## 对比脚本

```bash
# 已有嵌入：直接评估
python scripts/eval_bcsd.py --firmware-emb q.json --db-emb d.json -k 1 5 10

# 从 BinKit 索引构建嵌入再自评（示例）
python scripts/run_baseline_comparison.py --index-file data/binkit_functions.json --model sempatch -k 1 5
python scripts/run_baseline_comparison.py --index-file data/binkit_functions.json --model safe --model-path path/to/safe.pt -k 1 5
python scripts/run_baseline_comparison.py --index-file data/binkit_functions.json --model jtrans_style -k 1 5
```

从预计算特征侧车 / JSON 构建基线嵌入：

```bash
python scripts/build_embeddings_db.py --features-file data/two_stage/library_features.jsonl \
  -o output/lib_jtrans.json --model jtrans_style
```

## 评估层级 L0 / L1 / L2（F.2）

| 层级 | 含义 | 典型命令 / 条件 |
|------|------|-----------------|
| **L0** | 无 Ghidra：合成特征或现成嵌入 | `pytest -m "not ghidra"`；`generate_synthetic_features` + `train_multimodal --synthetic` |
| **L1** | 小规模真实二进制 / 测试库 | `data/vulnerability_db/` 样例；短索引 + `build_embeddings_db` |
| **L2** | BinKit 全量或论文级跨优化/跨架构统计 | 需完整下载 BinKit、磁盘与 Ghidra 批处理时间；**未就绪时**在 `progress.md` 注明阻塞原因（数据体积、下载限制等） |

## 挑战场景评估（F.3）

`scripts/eval_challenge.py` 支持对 query/db 两侧分别构建嵌入并调用 `eval_bcsd`：

```bash
python scripts/eval_challenge.py --query-index data/o0_index.json --db-index data/o3_index.json --model sempatch
python scripts/eval_challenge.py --query-emb q.json --db-emb d.json -k 1 10
```

`--model` 可选：`sempatch` | `safe` | `jtrans_style`（与 `build_embeddings_db` 一致）。

## 漏洞库二进制准备度（F.4）

```bash
python scripts/build_library_binary_index.py --scan-root data/your_vuln_elf_dir --validate-only
```

- exit **0**：目录存在且至少扫描到一个 `.elf/.bin/.so`。
- exit **2**：目录存在但未找到可分析二进制（路径错误或仅有非 ELF 资源如 `.arrow`）。
- 生成索引：去掉 `--validate-only` 后写入 `--output`（默认 `data/vuln_library_binary_index.json`）。完整流水线见 `docs/VULNERABILITY_LIBRARY.md`。

## 相关文件

| 文件 | 作用 |
|------|------|
| `src/features/baselines/jtrans_style.py` | jTrans 风格块序基线与 `embed_batch_jtrans_style` |
| `src/features/baselines/safe.py` | SAFE 风格基线 |
| `scripts/build_embeddings_db.py` | 统一构建嵌入库 CLI |
| `scripts/run_baseline_comparison.py` | 构建（可选）+ `eval_bcsd` 一站式对比 |
| `scripts/eval_challenge.py` | 跨索引 / 跨嵌入的挑战评估 |
| `scripts/build_library_binary_index.py` | 自备漏洞目录扫描与索引、`--validate-only` |
| `scripts/annotate_library_embeddings_cve.py` | 为库嵌入 JSON 写入 `cve` 字段 |
