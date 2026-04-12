# 训练工作流

端到端：索引构建 → 多轮过滤 → 数据划分 → 特征提取 → vocab 构建 → 模型训练 → 嵌入库构建。

硬件参考：RTX 3050 4GB VRAM / 16GB RAM / R5 5500 6C12T。

---

## 0 快速开始（TL;DR）

**已有 `.training.jsonl` 和 `binkit_functions.json`？直接从第 ⑦ ⑧ 步开始训练。**

```bash
# 前置：激活环境
source .venv/bin/activate

# （可选）安装 orjson 加速 JSON 解析 5-10x
pip install orjson

# ── 一键训练 SAFE 粗筛模型（~5-15 分钟，取决于数据量）──
PYTHONPATH=src python scripts/sidechain/train_safe.py \
  --index-file data/binkit_functions_common.json \
  --precomputed-features data/binkit_functions_common.training.jsonl \
  --vocab-from-features data/binkit_functions_common.training.jsonl \
  --epochs 10 --batch-size 4 --num-pairs 10000 --lr 1e-3 \
  --save-path output/safe_best_model.pt --no-tb

# ── 一键训练 MultiModal 精排模型（~20-60 分钟）──
PYTHONPATH=src python scripts/sidechain/train_multimodal.py \
  --index-file data/binkit_functions_common.json \
  --precomputed-features data/binkit_functions_common.training.jsonl \
  --vocab-from-features data/binkit_functions_common.training.jsonl \
  --epochs 20 --batch-size 4 --num-pairs 20000 --lr 1e-4 \
  --max-seq-len 512 --max-graph-nodes 128 --max-dfg-nodes 64 \
  --num-workers 2 --pairing-mode binkit_refined \
  --save-path output/best_model.pth --no-tb
```

### 训练加速自动生效（无需额外配置）

训练脚本内置三项 CPU 优化，自动触发：

| 优化 | 触发条件 | 效果 |
|------|----------|------|
| **Vocab 预计算** | `--vocab-from-features` + `--precomputed-features` 同时指定 | tensorize 4x 加速 |
| **orjson 快速解析** | `pip install orjson` | JSONL 预载 6x 加速 |
| **合并 collate** | 使用预计算特征时默认 | 消除重复 dict 访问 |

### 从零开始的完整流程

```
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

---

## 1 端到端流程总览

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

### 2.4 训练时的特征增强（Vocab Enrichment）

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

## 3 阶段一：数据准备（详细步骤）

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

## 4 Vocab 构建

训练 MultiModal 和 SAFE 都需要一个 `vocab: Dict[str, int]`（`""`=0, `[UNK]`=1, 后续 token 递增）。

### 从 `.training.jsonl` 流式构建

两个训练脚本均支持：
```bash
--vocab-from-features data/binkit_functions_common.training.jsonl
```

内部调用 `collect_vocab_from_features_jsonl()`（`src/features/baselines/safe.py`），逐行扫描 `multimodal.sequence.pcode_tokens` 和 `multimodal.graph.node_features[*].pcode_opcodes`，不加载整个文件到内存。

安装 `orjson` 后此步骤加速 5-10x（`pip install orjson`）。

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

## 5 特征加载：侧车 → Dataset

`PairwiseFunctionDataset`（`src/features/dataset.py`）是两个训练脚本共用的数据集。

### 懒加载索引 + 预载

构造时：
1. 从 index 文件收集 `needed_ids = { "<binary>|<entry>", ... }`
2. 对 `.training.jsonl` 调用 `build_jsonl_sidecar_lazy_index()`：**单遍二进制扫描**，只为 `needed_ids` 内的函数记录 `(byte_offset, line_length_bytes)`，不解析 multimodal 内容
3. 使用 `bulk_get_iter()` **按 offset 排序后单次顺序读 JSONL**，全部预载到内存（比逐条随机 seek 快 100x+）
4. 训练时直接 `dict[key]` 查找，无磁盘 I/O

### 特征检索优先级（`_get_features`）

```
1. 内存缓存（memory_cache, 按 hash 索引）
2. 预计算特征（_precomputed_features dict, 内存中）
3. 懒加载 JSONL（precomputed_lazy_index.get(fid) → seek + read + parse）
4. 磁盘缓存（cache_dir/*.json）
5. 动态提取（Ghidra，最后回退）
```

使用 `.training.jsonl` 时，第 2 层直接命中，不会触发 Ghidra。

### 训练时内部流程

```
┌─ DataLoader (num_workers=2, prefetch_factor=2) ──────────────────┐
│                                                                   │
│  Worker 0                    Worker 1                             │
│  ┌──────────────────┐        ┌──────────────────┐                │
│  │ __getitem__ ×4   │        │ __getitem__ ×4   │                │
│  │  → _get_features │        │  → _get_features │                │
│  │  → 正/负采样      │        │  → 正/负采样      │                │
│  └────────┬─────────┘        └────────┬─────────┘                │
│           │                           │                           │
│           ▼                           ▼                           │
│  ┌──────────────────┐        ┌──────────────────┐                │
│  │ collate_fn       │        │ collate_fn       │                │
│  │  → 空值检测       │        │  → 空值检测       │                │
│  │  → tensorize     │        │  → tensorize     │  ← CPU 热路径  │
│  │  → batched tensor│        │  → batched tensor│                │
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

## 6 `_tensorize_multimodal`：特征 dict → Tensor

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

### 映射逻辑（含预计算快速路径）

**sequence tokens**（优先级从高到低）：
1. `sequence.pcode_token_ids`（int list）→ 直接切片 + clamp → `token_t` ⚡
2. `sequence.pcode_tokens`（str list）→ `vocab.get(token, 1)` → clamp → `token_t`

**graph nodes**（优先级从高到低）：
1. `node_features[i].opcode_id`（int）→ 直接读取 → `node_t` ⚡
2. `node_features[i].pcode_opcodes[0]`（str）→ `vocab.get(opcode, 1)` → `node_t`

**DFG nodes**：`int(x) % 512` → `dfg_node_t`（直接原始整数，模 512 保护）

标 ⚡ 的路径跳过 Python dict 查表循环，使用 int list 切片（C 级速度）。

---

## 7 阶段二：训练

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
- `--use-amp`：混合精度训练（默认开，降低显存占用）
- `--accumulation-steps`：梯度累积（等效 batch_size × accumulation_steps）

产出：`output/best_model.pth`（含 `{state_dict, meta}`）

### 训练流程内部机制

两个训练脚本共用以下组件：

- **数据集**：`PairwiseFunctionDataset`，按 `positive_ratio` 随机采样正/负对
- **损失**：`ContrastiveLoss`（余弦相似度版），label=1 时惩罚 `(1-cos_sim)²`，label=0 时惩罚 `max(0, cos_sim - margin)²`
- **训练循环**：`Trainer.fit()`，按 epoch 跑 train → validate → 保存最佳权重
- **固定采样对**（`--fixed-pairs-per-epoch`）：每个 epoch 预生成 `num_pairs` 对站点坐标，提升 JSONL 缓存命中率
- **Epoch 间缓存清理**（默认开）：每个 epoch 后 `gc.collect()` + `torch.cuda.empty_cache()`

### 训练日志

```
# 正常训练输出示例：
PairwiseFunctionDataset: 预载 50000 个函数特征到内存（顺序读 JSONL）…
PairwiseFunctionDataset: 预载完成 50000/50000 条，耗时 8.3s（6024 条/s）       ← orjson 加速
PairwiseFunctionDataset: vocab enrichment 完成 50000 条，耗时 1.2s（41667 条/s）← 预计算 IDs
使用 worker 端 tensorize collate_fn（预计算特征模式，高吞吐）
[Trainer] 每 epoch: 训练 2250 batch | 验证 250 batch
Epoch 1/20  train_loss=0.1842  val_loss=0.1523  val_acc=0.8734
```

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

### 9.1 CPU 瓶颈与加速

训练循环中 GPU 常等 CPU 预处理（tensorize + 数据加载）。三项内置优化已自动生效：

| 优化 | 文件 | 原理 | 加速 |
|------|------|------|------|
| Vocab 预计算 IDs | `dataset.enrich_with_vocab()` | tensorize 跳过 `vocab.get()` 循环，直接读 int list | 4x |
| orjson 快速解析 | `precomputed_multimodal_io.py` | Rust 实现 JSON 解析替代纯 Python | 6x |
| 合并 collate | `_collate_pairs_tensorized()` | 单次遍历消除重复 dict 访问 | 1.3x |

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
| `--dataloader-prefetch-factor` | 2 | 增大到 4-8 让 CPU 提前跑 |
| `--memory-cache-max-items` | 16384 | 大库增大到 32768+ |

### 9.4 OOM 应急

| 现象 | 处理 |
|------|------|
| CUDA OOM | `--batch-size` 4→2→1；`--max-seq-len` 512→256 |
| RAM OOM（训练） | `--num-workers` 2→0；减小 `--num-pairs` |
| RAM OOM（侧车扫描） | `build_jsonl_sidecar_lazy_index` 已是懒加载，不应 OOM；若仍 OOM 检查 `needed_ids` 是否过大 |
| Ghidra 超时 | 检查 `output/logs/ghidra.log`，增大 timeout |

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
| 侧车读取（流式） | `iter_jsonl_sidecar()` | `src/utils/precomputed_multimodal_io.py` |
| orjson 快速解析 | `_json_loads()` | `src/utils/precomputed_multimodal_io.py` |
| 侧车懒加载索引 | `build_jsonl_sidecar_lazy_index()` | `src/utils/precomputed_multimodal_io.py` |
| Vocab enrichment | `enrich_multimodal_with_ids()` | `src/utils/precomputed_multimodal_io.py` |
| Dataset enrichment | `PairwiseFunctionDataset.enrich_with_vocab()` | `src/features/dataset.py` |
| Vocab 构建（JSONL） | `collect_vocab_from_features_jsonl()` | `src/features/baselines/safe.py` |
| Dataset 特征加载 | `PairwiseFunctionDataset._get_features()` | `src/features/dataset.py` |
| tensorize（单条） | `_tensorize_multimodal()` | `src/features/models/multimodal_fusion.py` |
| tensorize（批量） | `tensorize_multimodal_many()` | `src/features/models/multimodal_fusion.py` |
| collate（MultiModal） | `_collate_pairs_tensorized()` | `scripts/sidechain/train_multimodal.py` |
| collate（SAFE） | `_collate_pairs_safe_tensorized()` | `scripts/sidechain/train_safe.py` |
| MultiModal forward | `MultiModalFusionModel.forward()` | `src/features/models/multimodal_fusion.py` |
| SAFE forward | `SafeEmbedder.embed_many()` | `src/features/baselines/safe.py` |
| MultiModal 训练 | `train_multimodal.py` | `scripts/sidechain/train_multimodal.py` |
| SAFE 训练 | `train_safe.py` | `scripts/sidechain/train_safe.py` |
| 训练循环 | `Trainer.fit()` | `src/features/trainer.py` |
| 嵌入库构建 | `build_embeddings_db.py` | `scripts/sidechain/build_embeddings_db.py` |