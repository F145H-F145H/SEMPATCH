# SemPatch 整体设计

## 研究目标（survey 5.1 & 5.3）

- **5.1（已实现）**：**CFG 图模态** + **P-code 序列（含跳转语义）** + **跨模态注意力** → 函数嵌入（`MultiModalFusionModel` / `fuse_features` 的 `graph` 与 `sequence`）。
- **5.1（已实现，DFG 嵌入路径）**：LSIR 中的 **DFG** 经 `fuse_features` 写入 `multimodal.dfg`，由 `MultiModalFusionModel` 的 **独立 DFG 图分支** 与 CFG 图嵌入拼接融合后再与序列做跨模态注意力；设计说明见 [`docs/dfg_fusion_design.md`](dfg_fusion_design.md)。训练用 `train_multimodal.py --use-dfg`；旧检查点无 `meta.use_dfg` 时推理侧默认无 DFG 分支。
- **5.3 架构中立表示**：基于 Ghidra P-code 消除架构差异，并做规范化以抑制编译器优化噪声；可选汇编归一化路径。

## 架构分层

| 层次 | 模块 | 职责 |
|------|------|------|
| 编排层 | `sempatch.py` | CLI、配置、流水线构建、ctx 管理、输出 |
| 执行层 | `dag/` | 节点定义、图结构、调度、导出 |
| 服务层 | `utils/`、`features/`、`matcher/` | IR、P-code 规范化、特征、模型、检索 |

## 流水线（固件 vs 漏洞库）

按 `--strategy` 或 `pipeline_strategy` 配置选择：

| 策略 | 节点序列 |
|------|----------|
| semantic_embed | ghidra → lsir_build → feature_extract → embed → load_db → diff_bipartite |
| traditional_fuzzy | ghidra → lsir_build → fuzzy_hash → load_db(fuzzy) → diff_fuzzy |
| traditional_cfg | ghidra → lsir_build → load_db(lsir) → cfg_match |
| graph_embed | ghidra → lsir_build → acfg_extract → embed → load_db → diff_faiss |
| fusion | ghidra → lsir_build → feature_extract → embed → load_db → diff_bipartite |

**计划扩展**（5.3）：在 `lsir_build` 前接入 `pcode_normalize` 节点（独立节点可选；当前规范化已默认并入 `lsir_build`）。`feature_extract` 融合路径输出 **CFG + 序列 + dfg 子图槽位**（`multimodal.graph` / `sequence` / `dfg`；`dfg` 可为空图）。

### 节点

1. **GhidraNode**：分析固件，输出 lsir_raw（P-code、基本块、指令）
2. **LSIRBuildNode**：构建 CFG/DFG（LSIR 含 `cfg`/`dfg`）
3. **FeatureExtractNode**：提取图（CFG）/序列特征；融合路径输出多模态规范格式（`graph` + `sequence` + `dfg`）
4. **ACFGExtractNode**：ACFG 特征（Genius/Gemini）
5. **FuzzyHashNode**：模糊哈希（ssdeep/tlsh）
6. **EmbedNode**：转为向量（已接入 MultiModalFusionModel）
7. **LoadDBNode**：加载漏洞库（支持 embeddings/lsir/fuzzy_hashes）
8. **DiffNode / DiffFAISSNode / DiffBipartiteNode / DiffFuzzyNode / CFGMatchNode**：匹配

**待新增**：`PcodeNormalizeNode`（5.3）——独立节点；当前规范化已并入 LSIRBuildNode。
