# SemPatch 训练流水线优化方案（v2 — GPU 利用率专项）

> **核心发现**：GPU 利用率 <5%，根本原因是数据流水线的 Python 开销远超 GPU 计算时间。
> RTX 3050 上一个 batch 的 forward+backward 仅 1-3ms，但 CPU 侧数据准备耗时 10-30ms。
> 优化目标：将数据准备从 Python dict 操作转为预计算 NumPy 数组索引，消除 per-sample 循环。

> ## ✅ 实施状态
> **Phase 1 已完成**（2026-04-13）：所有核心代码已实现并通过语法检查。
> 新增 3 个文件，修改 3 个文件，总计 ~400 行新代码。

---

## 一、问题根因分析

### 1.1 GPU 时序分解（RTX 3050, batch_size=4）

```
一个 batch 的生命周期（~25ms 周期）：
┌─────────────────────────────────────────────────────┐
│ DataLoader workers (CPU)                            │
│                                                     │
│  __getitem__ ×4 (串行 per worker)                    │
│    ├ _get_features(): dict lookup + cache_key(sha256)│  ~5-15μs × 2 calls = 10-30μs
│    ├ memory_cache dict lookup                        │  ~0.15μs
│    └ _put_memory_cache() dict insert                 │  ~0.2μs
│    合计 per sample: ~15-35μs                         │
│    4 samples: ~60-140μs                              │
│                                                     │
│  collate_fn (worker 进程内)                          │
│    tensorize_multimodal_many(f1_list, f2_list)       │
│    ├ Pass 1: 4 samples × 2 sides × (seq+graph+dfg)  │
│    │   dict.get × ~50次/sample = 200次 dict 操作     │  ~0.3ms
│    │   list comprehension + clamp                    │  ~0.5ms
│    │   edge_index _clamp_edge_index                  │  ~0.2ms
│    ├ Pass 2: numpy 填充 (seq/node/dfg arrays)        │  ~0.5ms
│    ├ Pass 3: edge index concat                       │  ~0.3ms
│    └ torch.from_numpy × 7 tensors × 2 batches        │  ~0.5ms
│    合计: ~2-3ms                                      │
│                                                     │
│  Total CPU per batch: ~3-5ms                        │
├─────────────────────────────────────────────────────┤
│ GPU (CUDA stream)                                   │
│   .to(device, non_blocking) × 7 tensors × 2         │  ~0.2ms (pin memory)
│   model.forward() × 2 (siamese)                     │  ~0.5-1ms
│   loss.backward()                                    │  ~0.5-1ms
│   optimizer.step()                                   │  ~0.2ms
│   Total GPU per batch: ~1.5-2.5ms                   │
└─────────────────────────────────────────────────────┘

GPU 利用率 = 1.5ms / (1.5ms + 3.5ms) ≈ 30% (理想情况)
实际: DataLoader prefetch 2 batches, 但 worker 串行处理，
      prefetch 不够深时 GPU 等待 → 实际 <5-10%
```

### 1.2 关键瓶颈识别

| 瓶颈 | 位置 | 量化影响 | 占比 |
|------|------|----------|------|
| **A. dict 遍历 + Python 循环** | `tensorize_multimodal_many` Pass 1 | ~1.5ms/batch (4 samples × ~0.4ms) | **40%** |
| **B. __getitem__ dict 操作** | `_get_features` + `_put_memory_cache` | ~0.1ms/batch (4 × 25μs) | 3% |
| **C. numpy → torch 转换** | `torch.from_numpy` × 14 | ~0.5ms/batch | 15% |
| **D. edge_index 处理** | `_clamp_edge_index` + list concat | ~0.5ms/batch | 15% |
| **E. DataLoader IPC 序列化** | worker → main process | ~0.5-1ms/batch | 15% |
| **F. GPU 计算** | forward + backward | ~1.5-2ms/batch | **~5-10%** |

**根本原因**: 数据以 Python dict 格式存储（`{sequence: {pcode_tokens: [...]}, graph: {node_features: [{pcode_opcodes: [...]}], edge_index: [[...],[...]]}}`），每 batch 都要遍历这些嵌套结构做 tensorize。

---

## 二、优化策略

### 策略 A（核心）：预计算 Tensor 数组 — 消除 per-batch tensorize

**原理**: 在数据准备阶段将每个函数的 multimodal 特征预转为固定形状的 NumPy 数组，训练时 `__getitem__` 纯数组索引，`collate_fn` 纯数组拼接。

#### 数据格式（准备阶段产出）

```python
# features_arrays.npz (或 LMDB)
# 每个函数一个条目，预计算为固定形状数组：

token_ids:     np.int16 [N, max_seq_len]        # pcode token IDs，已 pad
jump_mask:     np.int8  [N, max_seq_len]        # jump mask
node_ids:      np.int16 [N, max_graph_nodes]    # graph node opcode IDs
edge_src:      np.int32 [N, max_edges]          # graph edge sources (0-padded)
edge_dst:      np.int32 [N, max_edges]          # graph edge destinations (0-padded)
graph_n_edges: np.int16 [N]                     # 实际 graph 边数
dfg_node_ids:  np.int16 [N, max_dfg_nodes]      # DFG node IDs
dfg_edge_src:  np.int32 [N, max_dfg_edges]      # DFG edge sources (0-padded)
dfg_edge_dst:  np.int32 [N, max_dfg_edges]      # DFG edge destinations (0-padded)
dfg_n_edges:   np.int16 [N]                     # 实际 DFG 边数
seq_lens:      np.int16 [N]                     # 实际序列长度（用于 padding_mask）
node_counts:   np.int16 [N]                     # 实际节点数
```

#### 新 Dataset 实现

```python
class PrecomputedTensorDataset(Dataset):
    """纯数组索引数据集，零 Python dict 操作。"""
    
    def __init__(self, arrays_path: str, index_path: str, num_pairs: int, ...):
        # mmap 模式加载（不占 RAM）
        self._arrays = np.load(arrays_path, mmap_mode='r')
        # 构建 name→array_index 映射
        self._fid_to_idx: Dict[str, int] = ...  # 从 index.json 构建
        # pair 采样数据结构
        self._name_to_idx_list: Dict[str, List[int]] = ...  # 同名函数索引列表
    
    def __getitem__(self, idx: int) -> Tuple:
        a_idx, b_idx, label = self._epoch_pairs[idx]  # 纯 int 元组
        # 纯数组切片，无 dict 操作
        return (
            self._arrays['token_ids'][a_idx].copy(),    # (max_seq_len,) 切片
            self._arrays['jump_mask'][a_idx].copy(),
            self._arrays['node_ids'][a_idx].copy(),
            self._arrays['edge_src'][a_idx].copy(),
            self._arrays['edge_dst'][a_idx].copy(),
            int(self._arrays['graph_n_edges'][a_idx]),
            self._arrays['dfg_node_ids'][a_idx].copy(),
            self._arrays['dfg_edge_src'][a_idx].copy(),
            self._arrays['dfg_edge_dst'][a_idx].copy(),
            int(self._arrays['dfg_n_edges'][a_idx]),
            int(self._arrays['seq_lens'][a_idx]),
            int(self._arrays['node_counts'][a_idx]),
            # ... b_idx 同理 + label
        )
```

#### 新 Collate（纯 NumPy/Torch，无 Python 循环）

```python
def collate_precomputed(batch):
    """batch 是 tuple 列表，直接 numpy stack → torch tensor。"""
    # batch: [(a_tensors..., b_tensors..., label), ...]
    B = len(batch)
    # 一次 numpy stack，零 dict 遍历
    a_tokens = torch.from_numpy(np.stack([b[0] for b in batch]))  # (B, max_seq_len)
    a_jumps  = torch.from_numpy(np.stack([b[1] for b in batch]))
    # ... 
    return (a_tokens, a_jumps, ...), (b_tokens, b_jumps, ...), labels
```

**预期收益**: 
- `__getitem__`: 15-35μs → **0.1-0.5μs**（100x）
- `collate_fn`: 2-3ms → **0.05-0.1ms**（30x）
- GPU 利用率: <5% → **60-90%**

### 策略 B：预计算 pair 索引为整数元组

**当前**: `fixed_pairs_per_epoch` 存 `(binary_abs, entry, binary_abs, entry, label)`，`__getitem__` 仍需 `_get_features` 查找。
**优化**: 存 `(array_idx_a, array_idx_b, label)`，`__getitem__` 直接索引。

```python
# regenerate_epoch_pairs():
pairs = []
for _ in range(num_pairs):
    idx_a, idx_b, label = self._sample_pair_as_indices()  # 返回 int 索引
    pairs.append((idx_a, idx_b, label))
self._epoch_pairs = np.array(pairs, dtype=np.int32)  # (num_pairs, 3)
```

### 策略 C：消除 sha256 cache_key

**当前**: `_cache_key` 每次调用 `hashlib.sha256`（0.4μs），虽然不慢但没必要。
**优化**: 预计算模式下用整数数组索引，不需要 cache_key。

```python
# 旧: ck = _cache_key(binary_path, entry)  → sha256
# 新: idx = self._fid_to_idx[fid]           → dict.get (0.15μs)，或直接在 pair 中存 idx
```

### 策略 D：prepare_training_data 输出可直接训练的 .npz

合并数据准备 + 预计算 tensor 为一个步骤：

```bash
PYTHONPATH=src python scripts/sidechain/prepare_training_data.py \
  --input-dir data/binkit_subset \
  --output-dir data/training \
  --format npz \          # 输出预计算 tensor 数组
  --max-seq-len 512 \
  --max-graph-nodes 128 \
  --max-dfg-nodes 64 \
  --workers 6
```

输出:
```
data/training/
├── features.npz          # 预计算 tensor 数组 (~100-500MB for 50k functions)
├── vocab.json            # pcode vocab
├── index.json            # function_id → array_index 映射
├── binkit_functions_common.json  # 过滤后索引
└── two_stage/
    ├── library_index.json
    ├── query_index.json
    └── ground_truth.json
```

### 策略 E：DataLoader 深 prefetch

```python
# 当前: prefetch_factor=2, num_workers=2 → 预取 4 batches
# 优化: prefetch_factor=8, num_workers=2 → 预取 16 batches
# 配合 pin_memory + non_blocking, GPU 几乎不会空等

DataLoader(
    ...,
    num_workers=2,
    prefetch_factor=8,        # 深队列预取
    persistent_workers=True,
    pin_memory=True,
)
```

### 策略 F：OOM 全面防护

#### F1. npz mmap 模式（零 RAM 峰值）
```python
# mmap_mode='r'：OS 管理页缓存，按需加载，不会一次性占满 RAM
arrays = np.load('features.npz', mmap_mode='r')
# 50k 函数 × 512 tokens × 2 bytes = 50MB token_ids
# 实际驻留由 OS LRU 管理，通常 < 总大小的 20%
```

#### F2. Ghidra 超时保护
```python
# ghidra_runner.py 增加:
import signal

class GhidraTimeout(Exception): pass

def _run_with_timeout(cmd, timeout_sec=600):
    proc = subprocess.Popen(cmd, ...)
    try:
        stdout, stderr = proc.communicate(timeout=timeout_sec)
        return proc.returncode, stdout, stderr
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
        raise GhidraTimeout(f"Ghidra timed out after {timeout_sec}s")
```

#### F3. 显存压力前瞻检测
```python
# trainer.py _run_epoch 中 batch 开始前:
def _maybe_drain_gpu(self):
    if torch.cuda.is_available():
        ratio = torch.cuda.memory_reserved() / torch.cuda.get_device_properties(0).total_mem
        if ratio > 0.90:
            torch.cuda.empty_cache()
            gc.collect()
```

#### F4. 系统级 wrapper
```bash
#!/bin/bash
# scripts/run_training_safe.sh
set -e
MAX_MEM="${SEMPATCH_MAX_MEM:-12G}"
exec systemd-run --user --pty \
  -p "MemoryMax=${MAX_MEM}" \
  -p MemorySwapMax=0 \
  -p OOMPolicy=continue \
  "$@"
```

---

## 三、实施计划

### Phase 1: 预计算 Tensor 数组 + 新 Dataset（核心，最大收益） ✅ 已完成

| 任务 | 文件 | 状态 |
|------|------|------|
| 创建 `utils/npz_features.py`：multimodal dict → npz 数组 | `src/utils/npz_features.py` (280行) | ✅ |
| 创建 `PrecomputedTensorDataset` | `src/features/dataset.py` (+290行) | ✅ |
| 新 collate_fn（纯 numpy stack + edge rebuild） | `src/features/precomputed_collate.py` (197行) | ✅ |
| step_fn 适配（已有 fast path 直接兼容） | `train_*.py` | ✅ 无需改动 |
| JSONL→NPZ 转换器 | `scripts/sidechain/jsonl_to_npz.py` (179行) | ✅ |
| 训练脚本 --npz 参数集成 | `train_multimodal.py`, `train_safe.py` | ✅ |

### Phase 2: DataLoader 深度优化

| 任务 | 文件 | 工作量 |
|------|------|--------|
| 增大 prefetch_factor 默认值 | `train_*.py` | 0.5h |
| pair 预计算改为整数索引 | `dataset.py` | 0.5 天 |
| 消除 sha256 cache_key | `dataset.py` | 0.5h |

### Phase 3: OOM 防护

| 任务 | 文件 | 工作量 |
|------|------|--------|
| Ghidra 超时保护 | `utils/ghidra_runner.py` | 0.5 天 |
| 显存压力检测 | `src/features/trainer.py` | 0.5h |
| 安全运行 wrapper | `scripts/run_training_safe.sh` | 0.5h |

---

## 四、预期收益

| 指标 | 当前 | Phase 1 后 | Phase 1+2 |
|------|------|-----------|-----------|
| GPU 利用率 | <5% | 50-70% | 70-90% |
| `__getitem__` 耗时 | 15-35μs | 0.1-0.5μs | 0.1-0.5μs |
| `collate_fn` 耗时 | 2-3ms | 0.05-0.1ms | 0.05-0.1ms |
| 单 batch CPU 耗时 | 3-5ms | 0.2-0.5ms | 0.1-0.3ms |
| 单 batch GPU 耗时 | 1.5-2ms | 1.5-2ms | 1.5-2ms |
| GPU 利用率公式 | 1.5/(1.5+3.5)=30% | 1.5/(1.5+0.3)=83% | 1.5/(1.5+0.15)=91% |
| 数据准备命令 | 6+ | 1 | 1 |
| OOM 风险 | 中 | 低 | 极低 |

### 为什么当前只有 <5% 而不是 30%

DataLoader prefetch_factor=2, num_workers=2 → 最多预取 4 个 batch。
但 pair 采样可能失败重试（尤其是 binkit_refined 模式下 160 次循环），
导致某些 batch 准备时间极长（10-50ms），GPU 在等待中耗尽预取队列后空闲。

```
理想时间线: [CPU 3ms][GPU 2ms][CPU 3ms][GPU 2ms]... → GPU 40%
实际时间线: [CPU 3ms][GPU 2ms][CPU 3ms][GPU 2ms][CPU 50ms!!][GPU 2ms]... → GPU <5%
                                        ↑ 采样重试风暴
```

预计算 tensor 数组后：所有数据准备变为 O(1) 数组索引，无随机重试，
时间线变为完全均匀，GPU 持续饱和。

---

## 五、使用指南（Phase 1 已实现）

### 第一步：转换数据（一次性，~2分钟/50k函数）

```bash
# 已有 .training.jsonl + 索引？直接转换
PYTHONPATH=src python scripts/sidechain/jsonl_to_npz.py \
  --jsonl data/binkit_functions_common.training.jsonl \
  --index data/binkit_functions_common.json \
  -o data/training/features.npz
```

输出 3 个文件：
- `data/training/features.npz` — 预计算 tensor 数组
- `data/training/features.fid_map.json` — function_id → array_index
- `data/training/features.vocab.json` — pcode vocab

### 第二步：训练（使用 npz 模式）

```bash
# SAFE 训练
PYTHONPATH=src python scripts/sidechain/train_safe.py \
  --npz data/training/features.npz \
  --fid-map data/training/features.fid_map.json \
  --vocab data/training/features.vocab.json \
  --index-file data/binkit_functions_common.json \
  --epochs 10 --batch-size 4 --num-pairs 10000 --lr 1e-3 \
  --save-path output/safe_best_model.pt --no-tb --skip-validation

# MultiModal 训练
PYTHONPATH=src python scripts/sidechain/train_multimodal.py \
  --npz data/training/features.npz \
  --fid-map data/training/features.fid_map.json \
  --vocab data/training/features.vocab.json \
  --index-file data/binkit_functions_common.json \
  --epochs 20 --batch-size 4 --num-pairs 20000 --lr 1e-4 \
  --max-seq-len 512 --max-graph-nodes 128 --max-dfg-nodes 64 \
  --num-workers 2 --pairing-mode binkit_refined \
  --save-path output/best_model.pth --no-tb
```

### 向后兼容

不指定 `--npz` 时，训练脚本行为与修改前完全一致（使用 JSONL dict 格式）。
