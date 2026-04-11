# 训练工作流

面向中低端机器（RTX 3050 4GB / 16GB RAM / R5 5500 6C12T）的端到端流程。

## 前置条件

- Ghidra 12.0 已安装，`GHIDRA_HOME` 已设置
- Python venv 已激活：`source .venv/bin/activate`
- BinKit 数据集已下载至 `data/downloads/BinKit_normal/`

## 数据准备流水线

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

## 训练

### RTX 3050 / 16GB RAM 参数调整

`docs/train_default.yaml` 的默认参数（`max_seq_len=8192`, `max_graph_nodes=512`）针对大显存设计。
RTX 3050 仅 4GB 显存，需要降低以下参数：

| 参数 | 默认值 | RTX 3050 建议值 | 说明 |
|------|--------|----------------|------|
| `batch-size` | 8 | 4 | 显存占用正比于 batch |
| `max-seq-len` | 8192 | 512 | Transformer 注意力矩阵是 seq_len² |
| `max-graph-nodes` | 512 | 128 | GNN 节点数上限 |
| `max-dfg-nodes` | 128 | 64 | DFG 分支节点数 |
| `num-workers` | 4 | 2 | DataLoader 进程数，过多吃 RAM |

### 训练 MultiModal（精排模型）

```bash
PYTHONPATH=src python scripts/sidechain/train_multimodal.py \
  --index-file data/binkit_functions_common.json \
  --precomputed-features data/filtered_features.jsonl \
  --vocab-from-features data/filtered_features.jsonl \
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

### 训练 SAFE（粗筛模型）

```bash
PYTHONPATH=src python scripts/sidechain/train_safe.py \
  --index-file data/binkit_functions_common.json \
  --precomputed-features data/filtered_features.jsonl \
  --vocab-from-features data/filtered_features.jsonl \
  --epochs 10 \
  --batch-size 4 \
  --num-pairs 10000 \
  --lr 1e-3 \
  --save-path output/safe_best_model.pt \
  --no-tb
```

### 用 .training.jsonl 一次喂两个模型

如果已通过 `build_embeddings_db.py --emit-training-features` 生成了 `binkit_features.training.jsonl`，
训练脚本会**自动发现**同目录下 `{index_stem}.training.jsonl` 文件，无需手动指定：

```bash
# 确保 .training.jsonl 与 index 文件同目录同前缀
# data/binkit_functions_common.json  -> data/binkit_functions_common.training.jsonl

# 如果文件名不匹配，手动指定：
PYTHONPATH=src python scripts/sidechain/train_multimodal.py \
  --index-file data/binkit_functions_common.json \
  --precomputed-features data/binkit_features.training.jsonl \
  --vocab-from-features data/binkit_features.training.jsonl \
  --epochs 20 --batch-size 4 --num-pairs 20000 \
  --max-seq-len 512 --max-graph-nodes 128 --max-dfg-nodes 64 \
  --pairing-mode binkit_refined

PYTHONPATH=src python scripts/sidechain/train_safe.py \
  --index-file data/binkit_functions_common.json \
  --precomputed-features data/binkit_features.training.jsonl \
  --vocab-from-features data/binkit_features.training.jsonl \
  --epochs 10 --batch-size 4 --num-pairs 10000
```

---

## 构建 Embeddings 库

训练完成后，用模型生成库嵌入供匹配使用：

```bash
# SAFE 嵌入（TwoStage 粗筛用）
PYTHONPATH=src python scripts/sidechain/build_embeddings_db.py \
  --features-file data/two_stage/library_features.json \
  --model safe \
  --model-path output/safe_best_model.pt \
  -o data/two_stage/library_safe_embeddings.json
```

---

## OOM 应急

| 现象 | 处理 |
|------|------|
| CUDA OOM | 降低 `--batch-size`（4→2→1）或 `--max-seq-len`（512→256） |
| RAM OOM（训练） | 降低 `--num-workers`（2→0），降低 `--memory-cache-max-items` |
| RAM OOM（特征提取） | 降低 `--workers`，或分批处理 |
| Ghidra 超时 | 检查 `output/logs/ghidra.log`，增大 timeout |
