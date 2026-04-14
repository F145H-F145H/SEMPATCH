# 训练工作流

端到端：索引构建 → 多轮过滤 → 数据划分 → 特征提取 → vocab 构建 → 模型训练 → 嵌入库构建。

硬件参考：RTX 3050 4GB VRAM / 16GB RAM / R5 5500 6C12T。

---

## 快速导航

| 我想... | 直接跳到 |
|---------|----------|
| 最快跑通一次训练（已有 JSONL） | [路径 A：JSONL → NPZ → 训练](#路径-ajsonl--npz--训练5-分钟上手) |
| 从原始二进制开始全流程 | [路径 B：从零全流程](#路径-b从零全流程) |
| 快速验证环境是否正常 | [smoke test（synthetic 数据）](#smoke-test) |
| 调参 / 解决 OOM | [性能调优指南](#9-性能调优指南) |
| 构建嵌入库用于匹配 | [阶段三：构建嵌入库](#8-阶段三构建嵌入库) |

---

## 0 环境准备

```bash
source .venv/bin/activate

#（可选）安装 orjson 加速 JSON 解析 5-10x
pip install orjson

# 验证环境
make env-info
```

### Smoke Test

最快验证环境和代码是否正常（无需真实数据，~30 秒）：

```bash
# synthetic 数据，2 epochs
PYTHONPATH=src python scripts/sidechain/train_multimodal.py --synthetic --epochs 2 --no-tb

# SAFE synthetic，1 epoch
PYTHONPATH=src python scripts/sidechain/train_safe.py --synthetic --epochs 1 --no-tb --skip-validation

# 或用 Makefile 一键跑（含测试）
make reproduce
```

---

## 路径 A：JSONL → NPZ → 训练（5 分钟上手）

**前提**：已有 `.training.jsonl` 和 `binkit_functions.json`（即步骤 ①-⑥ 已完成）。

### Step 1：JSONL → NPZ（一次性，~2 分钟/50k 函数）

```bash
PYTHONPATH=src python scripts/sidechain/jsonl_to_npz.py \
  --jsonl data/training/library_only.training.jsonl \
  --index data/two_stage/library_index.json \
  -o data/training/features_library_only.npz \
  --max-seq-len 512 --max-graph-nodes 128 --max-dfg-nodes 64

ls -la data/training/features_library_only*
```

产出 3 个文件：
```
data/training/
├── features.npz              # 预计算 tensor 数组
├── features.fid_map.json     # function_id → array_index 映射
└── features.vocab.json       # pcode vocab
```

> **重要**：`--max-dfg-nodes` 等维度参数在 NPZ 转换和训练时必须一致。
> `jsonl_to_npz.py` 默认 `--max-dfg-nodes=64`，而 `train_multimodal.py` 默认 `--max-dfg-nodes=128`。
> 始终显式传入相同值。

### Step 2：训练 SAFE（粗筛模型，~5-15 分钟）

```bash
PYTHONPATH=src python scripts/sidechain/train_safe.py \
  --npz data/training/features_library_only.npz \
  --fid-map data/training/features_library_only.fid_map.json \
  --vocab data/training/features_library_only.vocab.json \
  --index-file data/two_stage/library_index.json \
  --epochs 10 --batch-size 4 --num-pairs 10000 --lr 1e-3 \
  --save-path output/safe_library_only.pt --no-tb --skip-validation

```

训练后自动运行目标校验（coarse_recall / recall_at_1），未达标自动扩样重训。
加 `--skip-validation` 跳过。

产出：`output/safe_best_model.pt`

### Step 3：训练 MultiModal（精排模型，~20-60 分钟）

```bash
PYTHONPATH=src python scripts/sidechain/train_multimodal.py \
  --npz data/training/features_library_only.npz \
  --fid-map data/training/features_library_only.fid_map.json \
  --vocab data/training/features_library_only.vocab.json \
  --index-file data/two_stage/library_index.json \
  --epochs 50 --batch-size 4 --num-pairs 100000 --lr 3e-4 \
  --max-seq-len 512 --max-graph-nodes 128 --max-dfg-nodes 64 \
  --pairing-mode binkit_refined \
  --save-path output/best_model_library_only.pth --no-tb

```

产出：`output/best_model.pth`

### Step 4：构建嵌入库

```bash
# SAFE 嵌入（TwoStage 粗筛用）
PYTHONPATH=src python scripts/sidechain/build_embeddings_db.py \
  --features-file data/two_stage/library_features.json \
  --model safe --model-path output/safe_best_model.pt \
  -o data/two_stage/library_safe_embeddings.json

# MultiModal 嵌入（精排用，可选）
PYTHONPATH=src python scripts/sidechain/build_embeddings_db.py \
  --features-file data/two_stage/library_features.json \
  --model sempatch --model-path output/best_model.pth \
  -o data/two_stage/library_mm_embeddings.json
```

### Step 5：评估

```bash
# Step 5: 用新模型评估（query binaries 完全未见过）
PYTHONPATH=src python scripts/sidechain/eval_two_stage_fixed.py \
  --allow-large-inputs \
  --data-dir data/two_stage \
  --model-path output/best_model_library_only.pth \
  --safe-model-path output/safe_library_only.pt \
  --library-features-db data/two_stage/library_features.db \
  --coarse-k 100 \
  -k 1 5 10 20 50 \
  --output output/benchmarks/library_only_eval.json
```

---

## 路径 B：从零全流程

从原始二进制开始，按顺序执行 ①-⑧。

### 前置条件

- Ghidra 12.0 已安装，`GHIDRA_HOME` 已设置
- Python venv：`source .venv/bin/activate`
- 原始二进制位于 `data/binkit_subset/`（.elf / .bin / .so）

### 流程总览

```
[原始二进制]
    │
    ▼
① build_binkit_index          → binkit_functions.json
② filter_index_by_pcode_len   → binkit_functions_filtered.json + .jsonl
③ filter_index_by_common      → binkit_functions_common.json
④ prepare_two_stage_data      → data/two_stage/
⑤ build_library_features      → library_features.json
⑥ (可选) build_embeddings_db  → .training.jsonl
⑦ train_safe                  → safe_best_model.pt
⑧ train_multimodal            → best_model.pth
⑨ build_embeddings_db         → library_*_embeddings.json
```

### ① 构建函数索引

扫描所有二进制，用 Ghidra 提取 `lsir_raw`，推导函数名和入口地址。同时写入 `binary_cache`，后续步骤直接命中缓存。

```bash
PYTHONPATH=src python scripts/sidechain/build_binkit_index.py \
  --input-dir data/binkit_subset \
  -o data/binkit_functions.json
```

产出：`data/binkit_functions.json`

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
- `data/binkit_functions_filtered.json`
- `data/filtered_features.jsonl`

### ③ 跨变体公共函数过滤

对同源项目（`project_id` 相同，如 `coreutils-9.1` 的 gcc/clang/O2/O3 变体）做函数名交集，只保留在**所有变体中都存在**的函数。

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
- `data/two_stage/library_index.json`
- `data/two_stage/query_index.json`
- `data/two_stage/ground_truth.json`

### ⑤ 构建库/查询特征

```bash
PYTHONPATH=src python scripts/sidechain/build_library_features.py \
  --library-index data/two_stage/library_index.json \
  --query-index data/two_stage/query_index.json \
  --output-dir data/two_stage \
  --precomputed-multimodal data/filtered_features.jsonl
```

产出：
- `data/two_stage/library_features.json`
- `data/two_stage/query_features.json`

### ⑥（可选）生成训练用 `.training.jsonl`

如果步骤②的侧车已经包含所有需要的函数，可跳过此步。

```bash
# Step 1: 构建 library-only 训练特征
PYTHONPATH=src python scripts/sidechain/build_embeddings_db.py \
  --features-file data/two_stage/library_features.json \
  --emit-training-features \
  --training-features-output data/training/library_only.training.jsonl \
  --model-path output/safe_best_model.pt \
  -o /dev/null

wc -l data/training/library_only.training.jsonl

```
 
产出：`output/library_embeddings.training.jsonl`

### ⑦-⑧ 训练

PYTHONPATH=src python scripts/sidechain/train_safe.py \
  --npz data/training/features.npz \
  --fid-map data/training/features.fid_map.json \
  --index data/binkit_functions_common.json \
  --vocab data/training/features.vocab.json \
  --epochs 10 --batch-size 4 --num-pairs 10000 --lr 1e-3 \
  --save-path output/safe_best_model.pt --no-tb

PYTHONPATH=src python scripts/sidechain/train_multimodal.py \
  --npz data/training/features.npz \
  --fid-map data/training/features.fid_map.json \
  --index data/binkit_functions_common.json \
  --vocab data/training/features.vocab.json \
  --epochs 20 --batch-size 4 --num-pairs 20000 --lr 1e-4 \
  --save-path output/best_model.pth --no-tb




从步骤⑥（或步骤②的侧车）拿到 `.training.jsonl` 后，转到 [路径 A](#路径-ajsonl--npz--训练5-分钟上手) 执行训练。

或使用 JSONL 模式（跳过 NPZ，略慢）：

```bash
# SAFE
PYTHONPATH=src python scripts/sidechain/train_safe.py \
  --index-file data/binkit_functions_common.json \
  --precomputed-features data/binkit_functions_common.training.jsonl \
  --vocab-from-features data/binkit_functions_common.training.jsonl \
  --epochs 10 --batch-size 4 --num-pairs 10000 --lr 1e-3 \
  --save-path output/safe_best_model.pt --no-tb

# MultiModal
PYTHONPATH=src python scripts/sidechain/train_multimodal.py \
  --index-file data/binkit_functions_common.json \
  --precomputed-features data/binkit_functions_common.training.jsonl \
  --vocab-from-features data/binkit_functions_common.training.jsonl \
  --epochs 20 --batch-size 4 --num-pairs 20000 --lr 1e-4 \
  --max-seq-len 512 --max-graph-nodes 128 --max-dfg-nodes 64 \
  --num-workers 2 --pairing-mode binkit_refined \
  --save-path output/best_model.pth --no-tb
```

### ⑨ 构建嵌入库

同 [路径 A Step 4](#step-4构建嵌入库)。

---

## 1 端到端流程架构图

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
│      ▼                                                              │
│  jsonl_to_npz.py ──→ features.npz + fid_map.json + vocab.json       │
│      │  预计算为固定形状 NumPy 数组，训练时零 dict 操作               │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│  阶段二：训练                                                        │
│                                                                     │
│  features.npz ──→ PrecomputedTensorDataset (mmap, 纯数组索引)        │
│      │                                                              │
│      ├─→ train_safe.py ──→ safe_best_model.pt（粗筛模型）            │
│      │                                                              │
│      └─→ train_multimodal.py ──→ best_model.pth（精排模型）          │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│  阶段三：嵌入库构建                                                   │
│                                                                     │
│  build_embeddings_db.py                                             │
│      ├─ model safe ──→ library_safe_embeddings.json（粗筛用）        │
│      └─ model sempatch ──→ library_mm_embeddings.json（精排用，可选）│
└─────────────────────────────────────────────────────────────────────┘
```

---

## 2 数据流细节

### 2.1 Ghidra 提取 → lsir_raw

每个二进制经 Ghidra 一次性导出全函数 `lsir_raw`，写入 `binary_cache`，后续步骤可直接读取缓存，避免重复 Ghidra 调用。

### 2.2 lsir_raw → multimodal 特征

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

### 2.3 `.training.jsonl` 侧车格式

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

### 2.4 NPZ 预计算格式

`.training.jsonl` 经 `jsonl_to_npz.py` 转换后得到的 `.npz` 包含以下数组：

| 数组名 | dtype | shape | 说明 |
|--------|-------|-------|------|
| `token_ids` | int16 | `[N, max_seq_len]` | pcode token IDs（vocab 映射后，已 pad） |
| `jump_mask` | int8 | `[N, max_seq_len]` | 跳转标记掩码 |
| `node_ids` | int16 | `[N, max_graph_nodes]` | 图节点 opcode IDs |
| `edge_src` | int32 | `[N, max_edges]` | 图边源节点（0-padded） |
| `edge_dst` | int32 | `[N, max_edges]` | 图边目标节点（0-padded） |
| `graph_n_edges` | int16 | `[N]` | 实际图边数 |
| `dfg_node_ids` | int16 | `[N, max_dfg_nodes]` | DFG 节点 IDs |
| `dfg_edge_src` | int32 | `[N, max_dfg_edges]` | DFG 边源节点 |
| `dfg_edge_dst` | int32 | `[N, max_dfg_edges]` | DFG 边目标节点 |
| `dfg_n_edges` | int16 | `[N]` | 实际 DFG 边数 |
| `seq_lens` | int16 | `[N]` | 实际序列长度 |
| `node_counts` | int16 | `[N]` | 实际图节点数 |

NPZ 使用 `mmap_mode='r'` 加载，由 OS 按需换页，不会一次性占满 RAM。

### 2.5 训练时的特征增强（Vocab Enrichment）

训练脚本在 vocab 构建完成后、DataLoader 创建前，自动对已加载的特征注入预计算整数 ID：

```
原始 multimodal                     enrichment 后
─────────────────                   ────────────────────────────
sequence.pcode_tokens: ["COPY", ...]  + sequence.pcode_token_ids: [2, ...]
graph.node_features[0]: {"pcode_opcodes":["BRANCH"]}
                                      + graph.node_features[0]: {..., "opcode_id": 4}
```

Tensorize 优先读 `pcode_token_ids` / `opcode_id`（int list 切片，C 级速度），**消除每 batch ~24,000 次 `vocab.get()` 查表**。

兼容性：无预计算 IDs 时自动回退到 `vocab.get()` 路径。旧 `.training.jsonl` 无需重新生成。

---

## 3 Vocab 构建

训练 MultiModal 和 SAFE 都需要一个 `vocab: Dict[str, int]`（`""`=0, `[UNK]`=1，后续 token 递增）。

### 从 `.training.jsonl` 流式构建

两个训练脚本均支持：
```bash
--vocab-from-features data/binkit_functions_common.training.jsonl
```

内部调用 `collect_vocab_from_features_jsonl()`（`src/features/baselines/safe.py`），逐行扫描 `multimodal.sequence.pcode_tokens` 和 `multimodal.graph.node_features[*].pcode_opcodes`，不加载整个文件到内存。

安装 `orjson` 后此步骤加速 5-10x（`pip install orjson`）。

### 从 NPZ 自动获取

使用 NPZ 模式时，vocab 由 `jsonl_to_npz.py` 自动生成：
```bash
--vocab data/training/features.vocab.json
```

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
| `dfg_node_embed` | **(512, embed_dim)** | DFG 节点原始整数（0-511，不走 vocab） |

---

## 4 特征加载：侧车 → Dataset

### 4.1 NPZ 模式（推荐）

`PrecomputedTensorDataset`（`src/features/dataset.py`）加载 NPZ 后，`__getitem__` 纯数组索引，零 dict 操作：

```
__getitem__ → self._epoch_pairs[idx] → (a_idx, b_idx, label)
            → self._arrays['token_ids'][a_idx].copy()  # 纯 numpy 切片
            → ...
```

pair 采样数据结构：`_name_to_idx_list`（同名函数索引列表）+ `_binary_to_names`（二进制→函数名），`regenerate_epoch_pairs()` 预生成 `(N, 3)` int32 数组。

### 4.2 JSONL 模式（遗留）

`PairwiseFunctionDataset` 支持动态 Ghidra 提取、预计算 JSONL、磁盘/内存缓存、固定 pair per epoch。

特征检索优先级：
```
1. 内存缓存（memory_cache, 按 hash 索引）
2. 预计算特征（_precomputed_features dict, 内存中）
3. 懒加载 JSONL（precomputed_lazy_index.get(fid) → seek + read + parse）
4. 磁盘缓存（cache_dir/*.json）
5. 动态提取（Ghidra，最后回退）
```

### 4.3 训练时内部流程

```
┌─ DataLoader (num_workers=2, prefetch_factor=4+) ─────────────────┐
│                                                                   │
│  Worker 0                    Worker 1                             │
│  ┌──────────────────┐        ┌──────────────────┐                │
│  │ __getitem__ ×B   │        │ __getitem__ ×B   │                │
│  │  NPZ: 纯数组切片  │        │  NPZ: 纯数组切片  │                │
│  │  JSONL: dict 查找 │        │  JSONL: dict 查找 │                │
│  └────────┬─────────┘        └────────┬─────────┘                │
│           │                           │                           │
│           ▼                           ▼                           │
│  ┌──────────────────┐        ┌──────────────────┐                │
│  │ collate_fn       │        │ collate_fn       │                │
│  │  NPZ: numpy stack│        │  NPZ: numpy stack│  ← CPU 热路径  │
│  │  JSONL: tensorize│        │  JSONL: tensorize│                │
│  └────────┬─────────┘        └────────┬─────────┘                │
│           └──────────┬───────────────┘                            │
│                      ▼                                            │
│           ┌──────────────────┐                                    │
│           │  主进程           │                                    │
│           │  .to(device)     │                                    │
│           │  model.forward() │ ← GPU                              │
│           │  loss.backward() │                                    │
│           └──────────────────┘                                    │
└───────────────────────────────────────────────────────────────────┘
```

---

## 5 `_tensorize_multimodal`：特征 dict → Tensor

`src/features/models/multimodal_fusion.py`

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

NPZ 模式下此函数不被调用——collate 直接从预计算数组 stack tensor。

### JSONL 模式映射逻辑

**sequence tokens**（优先级从高到低）：
1. `sequence.pcode_token_ids`（int list）→ 直接切片 + clamp → `token_t`
2. `sequence.pcode_tokens`（str list）→ `vocab.get(token, 1)` → clamp → `token_t`

**graph nodes**（优先级从高到低）：
1. `node_features[i].opcode_id`（int）→ 直接读取 → `node_t`
2. `node_features[i].pcode_opcodes[0]（str）→ `vocab.get(opcode, 1)` → `node_t`

**DFG nodes**：`int(x) % 512` → `dfg_node_t`（直接原始整数，模 512 保护）

---

## 6 训练细节

### 损失函数

`ContrastiveLoss`（余弦相似度版）：
- label=1（正对）：惩罚 `(1 - cos_sim)²`
- label=0（负对）：惩罚 `max(0, cos_sim - margin)²`

### 训练内建机制

| 机制 | 说明 |
|------|------|
| 混合精度（AMP） | `--use-amp`（默认开），显存减半，~1.5x 加速 |
| 梯度累积 | `--accumulation-steps N`，等效 batch_size × N |
| 梯度裁剪 | `clip_grad_norm_(max_norm=1.0)`，每 step 执行 |
| OOM 自动恢复 | 捕获 `RuntimeError("out of memory")`，跳过 batch，清缓存 |
| 显存压力检测 | `_maybe_drain_gpu()`，reserved >90% 时自动清理 |
| 固定 pair 采样 | 每 epoch 预生成 pairs，提升缓存命中率 |
| Epoch 缓存清理 | 每 epoch 后 `gc.collect()` + `torch.cuda.empty_cache()` |
| SAFE 目标校验 | 训练后自动跑 Recall@1，不达标自动扩样重训（最多 `--max-retries` 次） |

### RTX 3050 / 16GB RAM 建议参数

| 参数 | 默认 | 建议 | 原因 |
|------|------|------|------|
| `--batch-size` | 8 | 4 | 显存正比于 batch |
| `--max-seq-len` | 8192 | 512 | 注意力矩阵 O(L²) |
| `--max-graph-nodes` | 512 | 128 | GNN 节点数 |
| `--max-dfg-nodes` | 128 | 64 | DFG 分支节点数 |
| `--num-workers` | 4 | 2 | DataLoader 进程 |

### 训练日志示例

```
PairwiseFunctionDataset: 预载 50000 个函数特征到内存（顺序读 JSONL）…
PairwiseFunctionDataset: 预载完成 50000/50000 条，耗时 8.3s（6024 条/s）
PairwiseFunctionDataset: vocab enrichment 完成 50000 条，耗时 1.2s（41667 条/s）
使用 worker 端 tensorize collate_fn（预计算特征模式，高吞吐）
[Trainer] 每 epoch: 训练 2250 batch | 验证 250 batch
Epoch 1/20  train_loss=0.1842  val_loss=0.1523  val_acc=0.8734
```

---

## 7 训练参数速查

### train_safe.py 关键参数

| 参数 | 默认 | 说明 |
|------|------|------|
| `--npz` | — | 预计算 NPZ 文件（推荐） |
| `--fid-map` | — | function_id → array_index 映射 |
| `--vocab` | — | 预计算 vocab JSON |
| `--precomputed-features` | — | JSONL 侧车（遗留模式） |
| `--vocab-from-features` | — | 从 JSONL 构建 vocab |
| `--index-file` | `data/binkit_functions.json` | 索引文件 |
| `--epochs` | 10 | 训练 epoch 数 |
| `--batch-size` | 16 | 批大小 |
| `--num-pairs` | 2000 | 每 epoch 采样对数 |
| `--lr` | 1e-3 | 学习率 |
| `--use-amp` | true | 混合精度 |
| `--skip-validation` | false | 跳过训练后目标校验 |
| `--target-coarse-recall` | 0.50 | 粗筛 recall 目标 |
| `--target-recall-at-1` | 0.45 | Recall@1 目标 |
| `--max-retries` | 3 | 达标失败最大重试次数 |

### train_multimodal.py 关键参数

| 参数 | 默认 | 说明 |
|------|------|------|
| `--config` | — | YAML 配置文件（CLI 覆盖 YAML） |
| `--npz` | — | 预计算 NPZ 文件（推荐） |
| `--fid-map` | — | function_id → array_index 映射 |
| `--vocab` | — | 预计算 vocab JSON |
| `--index-file` | `data/binkit_functions.json` | 索引文件 |
| `--epochs` | 20 | 训练 epoch 数 |
| `--batch-size` | 8 | 批大小 |
| `--num-pairs` | 2000 | 每 epoch 采样对数 |
| `--lr` | 1e-4 | 学习率 |
| `--max-seq-len` | 512 | 最大序列长度 |
| `--max-graph-nodes` | 128 | 最大图节点数 |
| `--max-dfg-nodes` | 128 | 最大 DFG 节点数（**注意**：jsonl_to_npz 默认 64） |
| `--pairing-mode` | `legacy` | `legacy` 或 `binkit_refined`（推荐后者） |
| `--use-dfg` | true | 启用 DFG 分支 |
| `--use-amp` | true | 混合精度 |
| `--accumulation-steps` | 1 | 梯度累积步数 |
| `--init-weights` | — | 从检查点热启（strict=False） |
| `--retrieval-val-dir` | — | 两阶段数据目录（启用 Recall@1 验证） |

---

## 8 阶段三：构建嵌入库

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

## 9 性能调优指南

### 9.1 训练加速（自动生效）

NPZ 模式下三项优化自动生效，无需额外配置：

| 优化 | 原理 | 加速 |
|------|------|------|
| 预计算 NPZ 数组 | `__getitem__` 纯数组索引，零 dict 操作 | **100x**（15-35μs → 0.1-0.5μs） |
| 纯 numpy collate | `numpy.stack` → `torch.from_numpy`，零 Python 循环 | **30x**（2-3ms → 0.05-0.1ms） |
| mmap 模式加载 | OS 按需换页，不占满 RAM | RAM 峰值降低 80% |

JSONL 模式额外优化：

| 优化 | 触发条件 | 效果 |
|------|----------|------|
| Vocab 预计算 IDs | `--vocab-from-features` + `--precomputed-features` | tensorize 4x 加速 |
| orjson 快速解析 | `pip install orjson` | JSONL 预载 6x 加速 |
| 合并 collate | 使用预计算特征时默认 | 消除重复 dict 访问 |

### 9.2 GPU 瓶颈与加速

| 手段 | 参数 | 效果 |
|------|------|------|
| 混合精度 | `--use-amp`（默认开） | 显存减半，训练加速 ~1.5x |
| 梯度累积 | `--accumulation-steps 4` | 等效大 batch 不增显存 |
| 减小序列长度 | `--max-seq-len 256` | 注意力矩阵 4x 缩小 |
| torch.compile | 自动（PyTorch 2.0+） | 算子融合，~1.2x |

### 9.3 数据加载调优

| 参数 | 默认 | 调优建议 |
|------|------|----------|
| `--num-workers` | -1 (auto) | CPU 核心多时增大到 4-8 |
| `--dataloader-prefetch-factor` | 4 | 增大到 8 让 CPU 提前跑 |
| `--memory-cache-max-items` | 16384 | 大库增大到 32768+ |

### 9.4 OOM 应急

| 现象 | 处理 |
|------|------|
| CUDA OOM | `--batch-size` 4→2→1；`--max-seq-len` 512→256 |
| RAM OOM（训练） | `--num-workers` 2→0；减小 `--num-pairs` |
| RAM OOM（侧车扫描） | `build_jsonl_sidecar_lazy_index` 已是懒加载，不应 OOM |
| Ghidra 超时 | 检查 `output/logs/ghidra.log`，增大 timeout |

### 9.5 维度一致性

`jsonl_to_npz.py` 和训练脚本的以下参数**必须一致**，否则训练时维度不匹配：

| 参数 | jsonl_to_npz 默认 | train_multimodal 默认 | 状态 |
|------|--------------------|-----------------------|------|
| `--max-seq-len` | 512 | 512 | ✅ 一致 |
| `--max-graph-nodes` | 128 | 128 | ✅ 一致 |
| `--max-dfg-nodes` | **64** | **128** | ⚠️ 不一致！ |
| `--max-edges` | 512 | — | 自动适配 |

**建议**：始终显式传入这些参数，不依赖默认值。

---

## 10 附录：关键代码路径速查

| 步骤 | 入口 | 文件 |
|------|------|------|
| 索引构建 | `build_binkit_index.py` | `scripts/sidechain/build_binkit_index.py` |
| pcode 过滤 | `filter_index_by_pcode_len.py` | `scripts/sidechain/filter_index_by_pcode_len.py` |
| 交叉过滤 | `filter_common_functions()` | `scripts/sidechain/filter_index_by_common_functions.py` |
| 数据划分 | `prepare_two_stage_data.py` | `scripts/sidechain/prepare_two_stage_data.py` |
| 库特征构建 | `build_library_features.py` | `scripts/sidechain/build_library_features.py` |
| 侧车写出 | `_extract_training_features_from_raw()` | `scripts/sidechain/build_embeddings_db.py` |
| JSONL→NPZ | `jsonl_to_npz.py` | `scripts/sidechain/jsonl_to_npz.py` |
| NPZ 构建 | `build_precomputed_npz()` | `src/utils/npz_features.py` |
| NPZ Dataset | `PrecomputedTensorDataset` | `src/features/dataset.py` |
| NPZ collate | `collate_multimodal_precomputed()` | `src/features/precomputed_collate.py` |
| 侧车读取（流式） | `iter_jsonl_sidecar()` | `src/utils/precomputed_multimodal_io.py` |
| orjson 快速解析 | `_json_loads()` | `src/utils/precomputed_multimodal_io.py` |
| 侧车懒加载索引 | `build_jsonl_sidecar_lazy_index()` | `src/utils/precomputed_multimodal_io.py` |
| Vocab enrichment | `enrich_multimodal_with_ids()` | `src/utils/precomputed_multimodal_io.py` |
| Dataset enrichment | `PairwiseFunctionDataset.enrich_with_vocab()` | `src/features/dataset.py` |
| Vocab 构建（JSONL） | `collect_vocab_from_features_jsonl()` | `src/features/baselines/safe.py` |
| Dataset 特征加载 | `PairwiseFunctionDataset._get_features()` | `src/features/dataset.py` |
| tensorize（单条） | `_tensorize_multimodal()` | `src/features/models/multimodal_fusion.py` |
| tensorize（批量） | `tensorize_multimodal_many()` | `src/features/models/multimodal_fusion.py` |
| collate（MultiModal JSONL） | `_collate_pairs_tensorized()` | `scripts/sidechain/train_multimodal.py` |
| collate（SAFE JSONL） | `_collate_pairs_safe_tensorized()` | `scripts/sidechain/train_safe.py` |
| collate（NPZ MultiModal） | `collate_multimodal_precomputed()` | `src/features/precomputed_collate.py` |
| collate（NPZ SAFE） | `collate_safe_precomputed()` | `src/features/precomputed_collate.py` |
| MultiModal forward | `MultiModalFusionModel.forward()` | `src/features/models/multimodal_fusion.py` |
| SAFE forward | `SafeEmbedder.embed_many()` | `src/features/baselines/safe.py` |
| MultiModal 训练 | `train_multimodal.py` | `scripts/sidechain/train_multimodal.py` |
| SAFE 训练 | `train_safe.py` | `scripts/sidechain/train_safe.py` |
| 训练循环 | `Trainer.fit()` | `src/features/trainer.py` |
| 嵌入库构建 | `build_embeddings_db.py` | `scripts/sidechain/build_embeddings_db.py` |
