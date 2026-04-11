# 训练工作流

端到端：索引构建 → 多轮过滤 → 数据划分 → 特征提取 → vocab 构建 → 模型训练 → 嵌入库构建。

硬件参考：RTX 3050 4GB VRAM / 16GB RAM / R5 5500 6C12T。

---

## 0 端到端流程总览

```
┌─────────────────────────────────────────────────────────────────────┐
│  阶段一：数据准备                                                    │
│                                                                     │
│  [原始二进制]                                                        │
│      │                                                              │
│      ▼                                                              │
│  ① build_binkit_index.py ──→ binkit_functions.json                  │
│      │  Ghidra 提取 lsir_raw，推导 {name, entry}                     │
│      ▼                                                              │
│  ② filter_index_by_pcode_len.py ──→ binkit_functions_filtered.json  │
│      │  删除短函数（<16 pcode token）、CRT 样板符号                    │
│      │  同时可写出 filtered_features.jsonl（侧车，供后续复用）         │
│      ▼                                                              │
│  ③ filter_index_by_common_functions.py ──→ binkit_functions_common.json │
│      │  按 project_id 分组，仅保留同源所有变体共有的函数                │
│      ▼                                                              │
│  ④ prepare_two_stage_data.py ──→ data/two_stage/                    │
│      │  按二进制随机 80/20 划分 library / query                       │
│      │  产出 library_index.json + query_index.json + ground_truth.json │
│      ▼                                                              │
│  ⑤ build_library_features.py ──→ library_features.json              │
│      │  + query_features.json（可选）                                │
│      │  从 .training.jsonl 侧车读取或 Ghidra 动态提取                 │
│      ▼                                                              │
│  ⑥（可选）build_embeddings_db.py --emit-training-features           │
│        ──→ library_embeddings.training.jsonl                        │
│        一次遍历同时产出 SAFE tokens + MultiModal 完整特征             │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│  阶段二：训练                                                        │
│                                                                     │
│  .training.jsonl ──→ vocab 构建                                      │
│      │                                                              │
│      ├─→ ⑦ train_safe.py ──→ safe_best_model.pt（粗筛模型）          │
│      │                                                              │
│      └─→ ⑧ train_multimodal.py ──→ best_model.pth（精排模型）        │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│  阶段三：嵌入库构建                                                   │
│                                                                     │
│  ⑨ build_embeddings_db.py                                           │
│      ├─ model safe ──→ library_safe_embeddings.json（粗筛用）        │
│      └─ model sempatch ──→ library_mm_embeddings.json（精排用，可选）│
└─────────────────────────────────────────────────────────────────────┘
```

---

## 1 数据流细节

### 1.1 Ghidra 提取 → lsir_raw

每个二进制经 Ghidra 一次性导出全函数 `lsir_raw`，写入 `binary_cache`，后续步骤可直接读取缓存，避免重复 Ghidra 调用。

### 1.2 lsir_raw → multimodal 特征

```
lsir_raw
   │
   ├─ normalize_lsir_raw()
   ├─ build_lsir(include_cfg=True, include_dfg=True)
   │
   ├─ extract_graph_features()   → cfg/dfg 图结构
   ├─ extract_sequence_features() → {"pcode_tokens": [...], "jump_mask": [...]}
   ├─ extract_acfg_features()     → 块级 ACFG 属性
   └─ fuse_features()              → {"multimodal": {graph, sequence, dfg}}
```

### 1.3 `.training.jsonl` 侧车格式

每条记录：
```json
{
    "function_id": "<binary_rel>|<entry_hex>",
    "multimodal": { fused["multimodal"] },
    "safe_tokens": ["COPY", "LOAD", "INT_ADD", ...]
}
```

`multimodal` 字段结构：
```json
{
    "sequence": {
        "pcode_tokens": ["COPY", "LOAD", "INT_ADD", ...],
        "jump_mask": [0, 0, 1, 0, ...]
    },
    "graph": {
        "node_features": [{"pcode_opcodes": ["BRANCH"]}, {"pcode_opcodes": ["INT_ADD"]}, ...],
        "edge_index": [[src0, src1, ...], [dst0, dst1, ...]]
    },
    "dfg": {
        "node_features": [12, 5, 0, 3, ...],
        "edge_index": [[...], [...]]
    }
}
```

注意两种 `node_features` 的类型差异：
- **graph** `node_features`：**列表 of dict**（`{"pcode_opcodes": [...]}`），每个 dict 的第一个 opcode 经 `vocab.get()` 映射为整数 ID，喂入 `node_embed = nn.Embedding(pcode_vocab_size, embed_dim)`。
- **dfg** `node_features`：**列表 of int**（0-511），是特征提取管道产生的原始整数，直接喂入 `dfg_node_embed = nn.Embedding(512, embed_dim)`，不做 vocab 映射。

---

## 2 阶段一：数据准备（详细步骤）

### 前置条件

- Ghidra 12.0 已安装，`GHIDRA_HOME` 已设置
- Python venv：`source .venv/bin/activate`
- 原始二进制位于 `data/binkit_subset/`（.elf / .bin / .so）

### ① 构建函数索引

扫描所有二进制，用 Ghidra 提取 `lsir_raw`，推导函数名和入口地址。同时写入 `binary_cache`，后续步骤直接命中缓存。

```bash
PYTHONPATH=src python scripts/sidechain/build_binkit_index.py \
  --input-dir data/binkit_subset \
  -o data/binkit_functions.json
```

产出：`data/binkit_functions.json`
```json
[{"binary": "data/binkit_subset/xxx.elf", "functions": [{"name": "foo", "entry": "0x1234"}, ...]}]
```

### ② 过滤索引 — pcode 长度过滤 + CRT 符号排除

删除短函数（默认 <16 pcode token）和 CRT/启动胶水符号（`main`、`_start`、`__libc_start_main` 等），降低噪声。

```bash
PYTHONPATH=src python scripts/sidechain/filter_index_by_pcode_len.py \
  -i data/binkit_functions.json \
  -o data/binkit_functions_filtered.json \
  --filtered-features-output data/filtered_features.jsonl \
  --min-pcode-len 16 \
  --workers 6
```

关键参数：
- `--min-pcode-len 16`：最短 pcode 序列长度（太短的函数缺乏区分度）
- `--workers 6`：函数级多进程（建议 CPU 线程数的一半）
- `--filtered-features-output`：同时写出 `.jsonl` 侧车（供后续步骤直接读取，避免重跑 Ghidra）
- 默认排除 CRT 符号；加 `--no-exclude-runtime-symbols` 可保留

产出：
- `data/binkit_functions_filtered.json`（过滤后的索引）
- `data/filtered_features.jsonl`（侧车，含完整 multimodal 特征）

### ③ 跨变体公共函数过滤

对同源项目（`project_id` 相同，如 `coreutils-9.1` 的 gcc/clang/O2/O3 变体）做函数名交集，只保留在**所有变体中都存在**的函数。提高训练数据质量，消除因编译优化导致的函数缺失噪声。

```bash
PYTHONPATH=src python scripts/sidechain/filter_index_by_common_functions.py \
  -i data/binkit_functions_filtered.json \
  -o data/binkit_functions_common.json
```

关键参数：
- `--match-by name`（默认）：按函数名匹配（IPA 优化后缀如 `.isra.0` 自动剥离）
- `--min-variants 2`：最少变体数才进行交集（单变体 project 保留全部）
- `--min-ratio 1.0`（默认）：全交集（所有变体共有的函数）；0.5 = 多数投票

产出：`data/binkit_functions_common.json`

### ④ 准备两阶段数据（library / query 划分）

按二进制随机 80/20 划分，构建匹配评估用的 ground truth。

```bash
PYTHONPATH=src python scripts/sidechain/prepare_two_stage_data.py \
  --index-file data/binkit_functions_common.json \
  --output-dir data/two_stage \
  --min-queries 1000
```

产出：
- `data/two_stage/library_index.json`（库侧索引，~80% 二进制）
- `data/two_stage/query_index.json`（查询侧索引，~20% 二进制）
- `data/two_stage/ground_truth.json`（`{query_id: [positive_id, ...]}`）

### ⑤ 构建库/查询特征

从 `.training.jsonl` 侧车读取（或 Ghidra 回退提取）multimodal 特征。

```bash
PYTHONPATH=src python scripts/sidechain/build_library_features.py \
  --library-index data/two_stage/library_index.json \
  --query-index data/two_stage/query_index.json \
  --output-dir data/two_stage \
  --precomputed-multimodal data/filtered_features.jsonl
```

产出：
- `data/two_stage/library_features.json`（`{function_id: multimodal_dict}`）
- `data/two_stage/query_features.json`

### ⑥（可选）生成训练用 `.training.jsonl`

如果步骤②的侧车已经包含所有需要的函数，可跳过此步。否则用 `--emit-training-features` 从库特征重新生成完整训练侧车：

```bash
PYTHONPATH=src python scripts/sidechain/build_embeddings_db.py \
  --features-file data/two_stage/library_features.json \
  --model-path output/safe_best_model.pt \
  --emit-training-features \
  -o output/library_embeddings.json
```

产出：
- `output/library_embeddings.json`（嵌入）
- `output/library_embeddings.training.jsonl`（训练特征侧车）

> JSONL 追加写入（`open("a")`），支持中断恢复。同目录下 `processed_rels` 集合防止重复。

### `.training.jsonl` 兼容性

训练脚本通过 `_parse_jsonl_record()` 解析，只检查 `function_id` + `multimodal` 两个键。额外的 `safe_tokens` 字段会被自动忽略（MultiModal 训练不需要），SAFE 训练则通过 `multimodal.sequence.pcode_tokens` 获取 tokens。

---

## 3 Vocab 构建

训练 MultiModal 和 SAFE 都需要一个 `vocab: Dict[str, int]`（`""`=0, `[UNK]`=1, 后续 token 递增）。

### 从 `.training.jsonl` 流式构建

两个训练脚本均支持：
```bash
--vocab-from-features data/binkit_functions_common.training.jsonl
```

内部调用 `collect_vocab_from_features_jsonl()`（`src/features/baselines/safe.py`），逐行扫描 `multimodal.sequence.pcode_tokens` 和 `multimodal.graph.node_features[*].pcode_opcodes`，不加载整个文件到内存。

### Vocab 大小与 Embedding 尺寸

```python
vocab_size = max(len(vocab), 256)     # floor 256
```

典型值：`len(vocab) ≈ 62`，所以 `vocab_size = 256`。

这个值同时决定三个 Embedding：
| Embedding | 大小 | 输入源 |
|-----------|------|--------|
| `seq_embed` | `(pcode_vocab_size, embed_dim)` | sequence token IDs（vocab 映射） |
| `node_embed` | `(pcode_vocab_size, embed_dim)` | graph node opcode IDs（vocab 映射） |
| `dfg_node_embed` | **(512, embed_dim)** | DFG node 原始整数（0-511，不走 vocab） |

> **已知问题修复**：`node_embed` 曾硬编码 512，当 `pcode_vocab_size > 512` 时导致 CUDA device-side assert。已改为 `pcode_vocab_size`。`dfg_node_embed` 保留 512（DFG 特征来自管道原始整数，最大 511），`_tensorize_multimodal` 中对 DFG int 保留 `% 512` 保护。

---

## 4 特征加载：侧车 → Dataset

`PairwiseFunctionDataset`（`src/features/dataset.py`）是两个训练脚本共用的数据集。

### 懒加载索引

构造时：
1. 从 index 文件收集 `needed_ids = { "<binary>|<entry>", ... }`
2. 对 `.training.jsonl` 调用 `build_jsonl_sidecar_lazy_index()`：**单遍二进制扫描**，只为 `needed_ids` 内的函数记录 `(byte_offset, line_length_bytes)`，不解析 multimodal 内容
3. 运行时按 `function_id` 做 `f.seek(offset) → f.read(length) → json.loads` 单行解析

### 特征检索优先级（`_get_features`）

```
1. 内存缓存（memory_cache, 按 hash 索引）
2. 懒加载 JSONL（precomputed_lazy_index.get(fid) → seek + read + parse）
3. 磁盘缓存（cache_dir/*.json）
4. 动态提取（Ghidra，最后回退）
```

使用 `.training.jsonl` 时，第 2 层直接命中，不会触发 Ghidra。

---

## 5 `_tensorize_multimodal`：特征 dict → Tensor

`src/features/models/multimodal_fusion.py:235`

将 `multimodal` dict 转为 7 个 Tensor，喂入 `model.forward()`：

```
输出 (7-tuple)                    对应 forward 参数
─────────────────────────────────────────────────────
token_t  (1, max_seq_len)     → token_ids
jump_t   (1, max_seq_len)     → jump_mask
node_t   (1, max_graph_nodes) → graph_node_features
edge_t   (2, num_edges)       → edge_index
pad_mask (1, max_seq_len)     → padding_mask
dfg_node_t (1, max_dfg_nodes) → dfg_node_features
dfg_edge_t (2, num_edges)     → dfg_edge_index
```

关键映射逻辑：
- **sequence tokens**: `vocab.get(token, 1)` → 整数 ID → `token_t`（padding 用 0）
- **graph nodes**: 取 `nf.get("pcode_opcodes", [])[0]` → `vocab.get(opcode, 1)` → `node_t`
- **DFG nodes**: `int(x) % 512` → `dfg_node_t`（直接原始整数，模 512 保护）

---

## 6 阶段二：训练

### 前置条件

```bash
# 确保 .training.jsonl 与 index 文件同目录同前缀：
# data/binkit_functions_common.json  → data/binkit_functions_common.training.jsonl
ls data/binkit_functions_common.training.jsonl
```

训练脚本自动发现：如果 `--precomputed-features` 未指定，推导 `{index_stem}.training.jsonl`。

### RTX 3050 / 16GB RAM 参数

| 参数 | 默认 | 建议 | 原因 |
|------|------|------|------|
| `batch-size` | 8 | 4 | 显存正比于 batch |
| `max-seq-len` | 8192 | 512 | 注意力矩阵 O(L²) |
| `max-graph-nodes` | 512 | 128 | GNN 节点数 |
| `max-dfg-nodes` | 128 | 64 | DFG 分支节点数 |
| `num-workers` | 4 | 2 | DataLoader 进程 |

### ⑦ 训练 SAFE（粗筛模型）

SAFE 是轻量序列编码器（token embedding + mean 聚合），用于两阶段管线的**粗筛**阶段：将库函数快速编码为向量，用余弦相似度召回 Top-K 候选。

```bash
PYTHONPATH=src python scripts/sidechain/train_safe.py \
  --index-file data/binkit_functions_common.json \
  --precomputed-features data/binkit_functions_common.training.jsonl \
  --vocab-from-features data/binkit_functions_common.training.jsonl \
  --epochs 10 \
  --batch-size 4 \
  --num-pairs 10000 \
  --lr 1e-3 \
  --save-path output/safe_best_model.pt \
  --no-tb
```

训练后自动运行目标校验（coarse_recall / recall_at_1），未达标时自动扩样重训（最多 `--max-retries` 次）。加 `--skip-validation` 跳过校验。

产出：`output/safe_best_model.pt`（含 state_dict + vocab）

### ⑧ 训练 MultiModal（精排模型）

MultiModalFusionModel 是图分支 + 序列分支 + 跨模态注意力的多模态融合模型，用于两阶段管线的**精排**阶段：对 SAFE 召回的候选做精确相似度排序。

```bash
PYTHONPATH=src python scripts/sidechain/train_multimodal.py \
  --index-file data/binkit_functions_common.json \
  --precomputed-features data/binkit_functions_common.training.jsonl \
  --vocab-from-features data/binkit_functions_common.training.jsonl \
  --epochs 20 \
  --batch-size 4 \
  --num-pairs 20000 \
  --lr 1e-4 \
  --max-seq-len 512 \
  --max-graph-nodes 128 \
  --max-dfg-nodes 64 \
  --num-workers 2 \
  --pairing-mode binkit_refined \
  --save-path output/best_model.pth \
  --no-tb
```

关键选项：
- `--pairing-mode binkit_refined`：同源分层正负采样（比 `legacy` 跨二进制同名更精确）
- `--use-dfg` / `--no-use-dfg`：是否启用 DFG 图分支（默认开）
- `--init-weights`：从已有检查点热启（strict=False）
- `--retrieval-val-dir`：每 epoch 末跑 Recall@1 检索验证

产出：`output/best_model.pth`（含 `{state_dict, meta}`）

### 训练流程内部机制

两个训练脚本共用以下组件：

- **数据集**：`PairwiseFunctionDataset`，按 `positive_ratio` 随机采样正/负对
- **损失**：`ContrastiveLoss`（余弦相似度版），label=1 时惩罚 `(1-cos_sim)²`，label=0 时惩罚 `max(0, cos_sim - margin)²`
- **训练循环**：`Trainer.fit()`，按 epoch 跑 train → validate → 保存最佳权重
- **固定采样对**（`--fixed-pairs-per-epoch`）：每个 epoch 预生成 `num_pairs` 对站点坐标，提升 JSONL 缓存命中率
- **Epoch 间缓存清理**（默认开）：每个 epoch 后 `gc.collect()` + `torch.cuda.empty_cache()`

---

## 7 阶段三：构建嵌入库

训练完成后，用模型生成库嵌入供 `TwoStagePipeline` 匹配使用：

```bash
# SAFE 嵌入（TwoStage 粗筛用）
PYTHONPATH=src python scripts/sidechain/build_embeddings_db.py \
  --features-file data/two_stage/library_features.json \
  --model safe \
  --model-path output/safe_best_model.pt \
  -o data/two_stage/library_safe_embeddings.json

# MultiModal 嵌入（精排用，可选）
PYTHONPATH=src python scripts/sidechain/build_embeddings_db.py \
  --features-file data/two_stage/library_features.json \
  --model sempatch \
  --model-path output/best_model.pth \
  -o data/two_stage/library_mm_embeddings.json
```

---

## 8 OOM 应急

| 现象 | 处理 |
|------|------|
| CUDA OOM | `--batch-size` 4→2→1；`--max-seq-len` 512→256 |
| RAM OOM（训练） | `--num-workers` 2→0；减小 `--num-pairs` |
| RAM OOM（侧车扫描） | `build_jsonl_sidecar_lazy_index` 已是懒加载，不应 OOM；若仍 OOM 检查 `needed_ids` 是否过大 |
| Ghidra 超时 | 检查 `output/logs/ghidra.log`，增大 timeout |

---

## 9 附录：关键代码路径速查

| 步骤 | 入口 | 文件 |
|------|------|------|
| 索引构建 | `build_binkit_index.py` | `scripts/sidechain/build_binkit_index.py` |
| pcode 过滤 | `filter_index_by_pcode_len.py` | `scripts/sidechain/filter_index_by_pcode_len.py` |
| 交叉过滤 | `filter_common_functions()` | `scripts/sidechain/filter_index_by_common_functions.py` |
| 数据划分 | `prepare_two_stage_data.py` | `scripts/sidechain/prepare_two_stage_data.py` |
| 库特征构建 | `build_library_features.py` | `scripts/sidechain/build_library_features.py` |
| 侧车写出 | `_extract_training_features_from_raw()` | `scripts/sidechain/build_embeddings_db.py:71` |
| 侧车读取（流式） | `iter_jsonl_sidecar()` | `src/utils/precomputed_multimodal_io.py:150` |
| 侧车懒加载索引 | `build_jsonl_sidecar_lazy_index()` | `src/utils/precomputed_multimodal_io.py:346` |
| Vocab 构建（JSONL） | `collect_vocab_from_features_jsonl()` | `src/features/baselines/safe.py:69` |
| Dataset 特征加载 | `PairwiseFunctionDataset._get_features()` | `src/features/dataset.py:721` |
| tensorize（单条） | `_tensorize_multimodal()` | `src/features/models/multimodal_fusion.py:235` |
| tensorize（批量） | `tensorize_multimodal_many()` | `src/features/models/multimodal_fusion.py:320` |
| MultiModal forward | `MultiModalFusionModel.forward()` | `src/features/models/multimodal_fusion.py:196` |
| SAFE forward | `SafeEmbedder.embed_many()` | `src/features/baselines/safe.py:237` |
| 嵌入推断（批量） | `embed_batch()` / `embed_batch_safe()` | `src/features/inference.py` / `src/features/baselines/safe.py` |
| MultiModal 训练 | `scripts/train_multimodal.py` | `scripts/sidechain/train_multimodal.py` |
| SAFE 训练 | `scripts/train_safe.py` | `scripts/sidechain/train_safe.py` |
| 训练循环 | `Trainer.fit()` | `src/features/trainer.py` |
| 嵌入库构建 | `build_embeddings_db.py` | `scripts/sidechain/build_embeddings_db.py` |
