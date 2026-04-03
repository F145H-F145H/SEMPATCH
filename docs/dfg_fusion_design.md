# DFG 融入多模态嵌入：设计说明（阶段 H）

本文档对应 `memory-bank/@prototype-survey-alignment-plan.md` 阶段 **H.2**，仅描述**一种**落地形态；实现代码不得并行实现多种互斥方案。

## 1. 目标

在保持 **CFG + P-code 序列 + 跨模态注意力** 的前提下，将 LSIR 中已由 `build_lsir` 构建的 **DFG** 作为**独立语义图**纳入 `fuse_features` → `MultiModalFusionModel` → `embed_batch` / 两阶段精排链路，并对**无 DFG 或空 DFG** 的旧数据保持降级（空图 + 零向量行为）。

## 2. 方案对比

| 方案 | 思路 | 优点 | 缺点 / 风险 |
|------|------|------|-------------|
| **A. 独立 DFG 图编码器** | `multimodal` 增加 `dfg` 子图（与 `graph` 同形）；模型内单独 GNN/嵌入后再与 CFG 图嵌入融合 | 与现有 `graph`/`sequence` 解耦清晰；易做 ablation（关 DFG 分支） | 参数量增加；需统一 `max_dfg_nodes` 截断 |
| **B. CFG+DFG 异构图** | 单一图含两类节点与类型化边 | 理论表达力强 | 需重写邻接与张量化；与当前 ACFG/CG 节点粒度不一致，改造成本高 |
| **C. 块级数据流特征拼到 CFG 节点** | 在 `node_features` 中附加每块的 def-use 统计 | 不增第二张图 | DFG 为 varnode 级，与块节点对齐启发式多、噪声大 |

## 3. 选定方案：**A（独立 DFG 图分支）**

- **特征侧**：`multimodal.dfg` 与 `multimodal.graph` 同 schema：`num_nodes`、`edge_index`（`[src_list, dst_list]`）、`node_list`、`node_features`。
- **节点特征**：DFG 节点字符串 `addr:varnode` 经 **稳定哈希** 映射到整数 id（与现有 `node_embed` 表尺寸兼容，见 `@architecture.md` §1.3），避免在 JSON 中膨胀词表。
- **模型侧**：DFG 分支结构与 CFG 图分支类似（节点嵌入 + 聚合）；将 **CFG 图嵌入** 与 **DFG 图嵌入** **拼接后经线性层** 压回 `output_dim`，再与序列分支做跨模态注意力（与当前 `MultiModalFusionModel` 衔接一致）。
- **空图语义**：`num_nodes == 0` 时张量侧使用 **零向量** 作为 DFG 图嵌入，不参与额外掩码分支失败。

## 4. 规模与配置

- **`max_dfg_nodes`**：与 `max_graph_nodes` 同量级（默认 128），超出则对 `node_list` 截断并过滤边端点落在截断集外的边（实现与 CFG 一致策略）。
- **`include_dfg`（fuse）**：默认 `True`；为 `False` 时不写入 `multimodal.dfg`（兼容极旧调用方）；为 `True` 且无 DFG 数据时写入**空图**（`num_nodes: 0`, 空 `edge_index`）。

## 5. 检查点与推理

- 训练保存 **`checkpoint` 字典**：含 `state_dict` 与 `meta`（至少 `use_dfg: bool`、`pcode_vocab_size`、可选 `max_dfg_nodes`）。
- 推理 / 精排：若 `meta.use_dfg` 为假或文件为**裸** `state_dict`（旧 `best_model.pth`），则构造**无 DFG 分支**的模型并忽略 `multimodal.dfg`，保证旧权重可用。

## 6. 验证口径

- 单元测试：带 DFG 的 LSIR fixture、`multimodal.dfg` schema、空 DFG 前向不抛错。
- 训练：`train_multimodal.py --synthetic` 在 `use_dfg` 开/关各至少 1 epoch 不崩溃。
