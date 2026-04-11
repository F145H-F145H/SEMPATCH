# 训练工作流

端到端：侧车 `.training.jsonl` 生成 → vocab 构建 → 模型训练 → 嵌入库构建。

硬件参考：RTX 3050 4GB VRAM / 16GB RAM / R5 5500 6C12T。

---

## 1 数据流总览

```
binary ──Ghidra──→ lsir_raw
                      │
                      ├─ _extract_training_features_from_raw()
                      │    ├─ build_lsir(include_cfg=True, include_dfg=True)
                      │    ├─ extract_graph_features()   → {"node_features": [{"pcode_opcodes": [...]}, ...], "edge_index": [...]}
                      │    ├─ extract_sequence_features() → {"pcode_tokens": [...], "jump_mask": [...]}
                      │    ├─ extract_acfg_features()     → {"normalized_degree": ..., ...}
                      │    ├─ extract_dfg_features()      → {"node_features": [int, ...], "edge_index": [...]}
                      │    └─ fuse_features()              → {"graph": ..., "sequence": ..., "acfg": ..., "dfg": ...}
                      │
                      └─ safe_tokens = pcode token 列表（给 SAFE 用）

每条 JSONL 记录（.training.jsonl）：
{
    "function_id": "<binary_rel>|<entry_hex>",
    "multimodal": { fused["multimodal"] },
    "safe_tokens": [ "COPY", "LOAD", "INT_ADD", ... ]
}
```

### 1.1 multimodal 字段结构

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
    },
    "acfg": { ... }
}
```

注意两种 node_features 的类型差异：
- **graph** `node_features`：**列表 of dict**（`{"pcode_opcodes": [...]}`），每个 dict 的第一个 opcode 经 `vocab.get()` 映射为整数 ID，喂入 `node_embed = nn.Embedding(pcode_vocab_size, embed_dim)`。
- **dfg** `node_features`：**列表 of int**（0-511），是特征提取管道产生的原始整数，直接喂入 `dfg_node_embed = nn.Embedding(512, embed_dim)`，不做 vocab 映射。

---

## 2 构建 `.training.jsonl`（侧车）

### 前置条件

- Ghidra 12.0 已安装，`GHIDRA_HOME` 已设置
- Python venv：`source .venv/bin/activate`
- 已有 `data/binkit_functions_common.json`（经 index → filter → common 三步产出）

### 一次性生成两个模型的训练特征

```bash
PYTHONPATH=src python scripts/sidechain/build_embeddings_db.py \
  --features-file data/two_stage/library_features.json \
  --model-path output/safe_best_model.pt \
  --emit-training-features \
  -o output/library_embeddings.json
```

`--emit-training-features` 额外写出 `{output_stem}.training.jsonl`（与 `-o` 同目录同前缀）。

产出文件（示例）：
```
output/library_embeddings.json           ← 嵌入（供匹配用）
output/library_embeddings.training.jsonl ← 训练特征（供两个训练脚本消费）
```

> JSONL 追加写入（`open("a")`），支持中断恢复。同目录下 `processed_rels` 集合防止重复。

### `.training.jsonl` 兼容性

训练脚本通过 `_parse_jsonl_record()` 解析，只检查 `function_id` + `multimodal` 两个键。额外的 `safe_tokens` 字段会被自动忽略（MultiModal 训练不需要），SAFE 训练则通过 `multimodal.sequence.pcode_tokens` 获取 tokens（或直接用 sidecar 内嵌的 `safe_tokens`）。

---

## 3 Vocab 构建

训练 MultiModal 和 SAFE 都需要一个 `vocab: Dict[str, int]`（`""`=0, `[UNK]`=1, 后续 token 递增）。

### 从 `.training.jsonl` 流式构建

```bash
# train_multimodal.py / train_safe.py 均支持：
--vocab-from-features data/binkit_functions_common.training.jsonl
```

内部调用 `collect_vocab_from_features_jsonl()`（`src/features/baselines/safe.py`），逐行扫描 `multimodal.sequence.pcode_tokens` 和 `multimodal.graph.node_features[*].pcode_opcodes`，不加载整个文件到内存。

### Vocab 大小与 Embedding 尺寸

```python
vocab_size = max(len(vocab), 256)     # floor 256
```

典型值：`len(vocab) ≈ 62`（filtered_features.jsonl），所以 `vocab_size = 256`。

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

## 6 训练

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

### 训练 MultiModal（精排模型）

```bash
PYTHONPATH=src python scripts/train_multimodal.py \
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
  --save-path /home/f145h/Documents/SemPatch/output/best_model.pth \
  --no-tb
```

> `scripts/train_multimodal.py` 是 `scripts/sidechain/train_multimodal.py` 的转发入口。默认 `--save-path`：`output/best_model.pth`。

### 训练 SAFE（粗筛模型）

```bash
PYTHONPATH=src python scripts/train_safe.py \
  --index-file data/binkit_functions_common.json \
  --precomputed-features data/binkit_functions_common.training.jsonl \
  --vocab-from-features data/binkit_functions_common.training.jsonl \
  --epochs 10 \
  --batch-size 4 \
  --num-pairs 10000 \
  --lr 1e-3 \
  --save-path /home/f145h/Documents/SemPatch/output/safe_best_model.pt \
  --no-tb
```

> `scripts/train_safe.py` 是 `scripts/sidechain/train_safe.py` 的转发入口。默认 `--save-path`：`output/safe_best_model.pt`。

---

## 7 构建嵌入库

训练完成后，用模型生成库嵌入供 `TwoStagePipeline` 匹配使用：

```bash
# SAFE 嵌入（TwoStage 粗筛用）
PYTHONPATH=src python scripts/sidechain/build_embeddings_db.py \
  --features-file data/two_stage/library_features.json \
  --model safe \
  --model-path /home/f145h/Documents/SemPatch/output/safe_best_model.pt \
  -o data/two_stage/library_safe_embeddings.json

# MultiModal 嵌入（精排用，可选）
PYTHONPATH=src python scripts/sidechain/build_embeddings_db.py \
  --features-file data/two_stage/library_features.json \
  --model sempatch \
  --model-path /home/f145h/Documents/SemPatch/output/best_model.pth \
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

## 附录：关键代码路径速查

| 步骤 | 入口 | 文件 |
|------|------|------|
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

---

old version:

### 1. 构建函数索引

扫描所有二进制，提取函数名和入口地址：

```bash
PYTHONPATH=src python scripts/sidechain/build_binkit_index.py \
  --input-dir data/binkit_subset \
  -o data/binkit_functions.json
```

### 2. 过滤索引（pcode 长度过滤 + 侧车特征提取）

删除短函数（<16 pcode token）和 CRT 样板代码，同时写入 `filtered_features.jsonl` 侧车：

```bash
PYTHONPATH=src python scripts/sidechain/filter_index_by_pcode_len.py \
  -i data/binkit_functions.json \
  -o data/binkit_functions_filtered.json \
  --filtered-features-output data/filtered_features.jsonl \
  --min-pcode-len 16 \
  --workers 6
```

> `--workers` 设为 CPU 线程数（R5 5500 = 12 线程，建议 6-8）。

### 3. 跨变体公共函数过滤（可选，推荐）

仅保留同一项目所有编译变体中都存在的函数，提高训练数据质量：

```bash
PYTHONPATH=src python scripts/sidechain/filter_index_by_common_functions.py \
  -i data/binkit_functions_filtered.json \
  -o data/binkit_functions_common.json
```

### 4. 准备两阶段数据（library / query 划分）

```bash
PYTHONPATH=src python scripts/sidechain/prepare_two_stage_data.py \
  --index-file data/binkit_functions_common.json \
  --output-dir data/two_stage \
  --min-queries 1000
```

输出：`data/two_stage/library_index.json`、`query_index.json`、`ground_truth.json`

### 5. 构建库特征

```bash
PYTHONPATH=src python scripts/sidechain/build_library_features.py \
  --library-index data/two_stage/library_index.json \
  --query-index data/two_stage/query_index.json \
  --output-dir data/two_stage \
  --precomputed-multimodal data/filtered_features.jsonl
```

输出：`data/two_stage/library_features.json`、`query_features.json`

---