# SemPatch 训练流水线优化方案（v3 — 可用性 + GPU 利用率）

> **目标**：让训练模型从 "需要理解 6 个脚本的复杂编排" 变成 "一条命令搞定"。
> 同时将 GPU 利用率从 <5% 提升到 60-90%。

> ## 实施状态
> - **Phase 1** ✅ 已完成（2026-04-13）：预计算 Tensor 数组 + 新 Dataset + Collate
> - **Phase 2** ✅ 已完成（2026-04-13）：DataLoader 优化（prefetch、pair 整数索引、消除 sha256）
> - **Phase 3** ✅ 已完成（2026-04-13）：OOM 防护（Ghidra 超时、显存检测、安全 wrapper）
> - **Phase 4** 🚧 规划中：训练可用性提升（一键训练、YAML 配置、Makefile 集成、维度校验）

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

### 1.3 可用性瓶颈

| 问题 | 影响 |
|------|------|
| 数据准备需手动编排 6 个脚本 | 新用户无法上手 |
| `--max-dfg-nodes` 默认值不一致（jsonl_to_npz=64, train=128） | 维度不匹配导致 silent failure |
| `train_safe.py` 无 `--config` 支持 | 配置管理割裂 |
| 无 Makefile 训练目标 | 命令行参数易出错 |
| TRAINING_WORKFLOW.md 侧重推理而非训练 | 文档与代码脱节 |

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

### 策略 G（新增）：训练可用性提升

#### G1. 统一 YAML 配置（SAFE + MultiModal）

当前 `train_safe.py` 无 `--config` 支持，训练超参只能通过命令行传递。新增统一配置：

```yaml
# configs/train_default.yaml — 两个脚本共用
model: multimodal           # multimodal | safe
epochs: 20
batch_size: 4
lr: 0.0001
num_pairs: 20000
seed: 42

# 架构
embed_dim: 64
hidden_dim: 128
output_dim: 128
num_gnn_layers: 2
num_transformer_layers: 2

# 数据维度（jsonl_to_npz 和训练必须一致）
max_seq_len: 512
max_graph_nodes: 128
max_dfg_nodes: 64

# 训练
use_dfg: true
use_amp: true
accumulation_steps: 2
num_workers: 2
pairing_mode: binkit_refined

# SAFE 专有
target_coarse_recall: 0.50
target_recall_at_1: 0.45
max_retries: 3
```

```python
# train_safe.py 增加 --config 支持（与 train_multimodal.py 一致）
parser.add_argument("--config", help="YAML 配置文件（CLI 覆盖 YAML）")
```

#### G2. 维度一致性校验

`jsonl_to_npz.py` 的 `--max-dfg-nodes` 默认 64，但 `train_multimodal.py` 默认 128 → silent mismatch。

```python
# PrecomputedTensorDataset.__init__() 增加校验:
def _validate_dims(self, expected: dict):
    """校验 NPZ 数组维度与模型期望一致。"""
    for key, expected_len in expected.items():
        actual = self._arrays[key].shape[1] if self._arrays[key].ndim > 1 else None
        if actual is not None and actual != expected_len:
            raise ValueError(
                f"NPZ 维度不匹配: {key} 实际 {actual} ≠ 期望 {expected_len}。"
                f"请确保 jsonl_to_npz 和训练脚本使用相同的 --max-* 参数。"
            )
```

#### G3. Makefile 训练目标

```makefile
# 训练便捷目标
train-safe-npz:
	@test -f $(NPZ) || (echo "先运行 jsonl_to_npz.py 生成 $(NPZ)"; exit 1)
	$(PY) scripts/sidechain/train_safe.py \
		--npz $(NPZ) --fid-map $(FID_MAP) --vocab $(VOCAB) \
		--index-file $(INDEX) \
		--epochs 10 --batch-size 4 --num-pairs 10000 --lr 1e-3 \
		--save-path output/safe_best_model.pt --no-tb

train-mm-npz:
	@test -f $(NPZ) || (echo "先运行 jsonl_to_npz.py 生成 $(NPZ)"; exit 1)
	$(PY) scripts/sidechain/train_multimodal.py \
		--npz $(NPZ) --fid-map $(FID_MAP) --vocab $(VOCAB) \
		--index-file $(INDEX) \
		--epochs 20 --batch-size 4 --num-pairs 20000 --lr 1e-4 \
		--max-seq-len 512 --max-graph-nodes 128 --max-dfg-nodes 64 \
		--pairing-mode binkit_refined \
		--save-path output/best_model.pth --no-tb

jsonl-to-npz:
	@test -f $(JSONL) || (echo "缺少 $(JSONL)"; exit 1)
	@test -f $(INDEX) || (echo "缺少 $(INDEX)"; exit 1)
	PYTHONPATH=src $(PYTHON) scripts/sidechain/jsonl_to_npz.py \
		--jsonl $(JSONL) --index $(INDEX) -o $(NPZ) \
		--max-seq-len 512 --max-graph-nodes 128 --max-dfg-nodes 64

train-all: jsonl-to-npz train-safe-npz train-mm-npz
```

#### G4. 一键数据准备脚本（`prepare_training_data.py`）

合并 ①-⑥ 步为单脚本，从原始二进制 → 过滤索引 + JSONL 侧车 + NPZ：

```bash
PYTHONPATH=src python scripts/sidechain/prepare_training_data.py \
  --input-dir data/binkit_subset \
  --output-dir data/training \
  --min-pcode-len 16 \
  --max-seq-len 512 --max-graph-nodes 128 --max-dfg-nodes 64 \
  --workers 6
```

内部调用：
1. `build_binkit_index.py` → `binkit_functions.json`
2. `filter_index_by_pcode_len.py` → `binkit_functions_filtered.json` + `.jsonl`
3. `filter_index_by_common_functions.py` → `binkit_functions_common.json`
4. `prepare_two_stage_data.py` → `two_stage/`
5. `build_library_features.py` → `library_features.json`
6. `jsonl_to_npz.py` → `features.npz` + `fid_map.json` + `vocab.json`

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

### Phase 2: DataLoader 深度优化 ✅ 已完成

| 任务 | 文件 | 状态 |
|------|------|------|
| 增大 prefetch_factor 默认值 | `train_*.py` | ✅ |
| pair 预计算改为整数索引 | `dataset.py` | ✅ |
| 消除 sha256 cache_key | `dataset.py` | ✅ |

### Phase 3: OOM 防护 ✅ 已完成

| 任务 | 文件 | 状态 |
|------|------|------|
| Ghidra 超时保护 | `utils/ghidra_runner.py` | ✅ |
| 显存压力检测 | `src/features/trainer.py` | ✅ |
| 安全运行 wrapper | `scripts/run_training_safe.sh` | ✅ |

### Phase 4: 训练可用性提升 🚧 规划中

| 任务 | 文件 | 优先级 |
|------|------|--------|
| `train_safe.py` 增加 `--config` YAML 支持 | `scripts/sidechain/train_safe.py` | P0 |
| 维度一致性校验（NPZ shape vs 模型期望） | `src/features/dataset.py` | P0 |
| `prepare_training_data.py` 一键数据准备 | `scripts/sidechain/prepare_training_data.py`（新建） | P0 |
| Makefile 训练目标 | `Makefile` | P1 |
| 统一 `configs/train_default.yaml` | `configs/train_default.yaml`（新建） | P1 |
| `jsonl_to_npz.py` 默认 `--max-dfg-nodes` 改为 128 | `scripts/sidechain/jsonl_to_npz.py` | P1 |
| CHECKPOINT 格式统一（SAFE vs MultiModal） | `train_safe.py`, `trainer.py` | P2 |

---

## 四、预期收益

| 指标 | 当前 | Phase 1+2+3 | Phase 4（可用性） |
|------|------|------------|-----------------|
| GPU 利用率 | <5% | 70-90% | 70-90% |
| `__getitem__` 耗时 | 15-35μs | 0.1-0.5μs | 0.1-0.5μs |
| `collate_fn` 耗时 | 2-3ms | 0.05-0.1ms | 0.05-0.1ms |
| 单 batch CPU 耗时 | 3-5ms | 0.1-0.3ms | 0.1-0.3ms |
| 数据准备命令数 | 6+ | 1 (jsonl_to_npz) | 1 (prepare_training_data) |
| OOM 风险 | 中 | 极低 | 极低 |
| **新手上手时间** | **数小时（理解 6 个脚本）** | 同左 | **~15 分钟（一条命令）** |
| 维度不匹配风险 | 高（无校验） | 高（无校验） | 低（自动校验） |

### 为什么当前 GPU 利用率只有 <5%

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

## 五、使用指南（Phase 1-3 已实现）

### 快速路径：已有 JSONL → 训练

```bash
source .venv/bin/activate

# 第一步：JSONL → NPZ（一次性，~2分钟/50k函数）
PYTHONPATH=src python scripts/sidechain/jsonl_to_npz.py \
  --jsonl data/binkit_functions_common.training.jsonl \
  --index data/binkit_functions_common.json \
  -o data/training/features.npz

# 第二步：训练 SAFE
PYTHONPATH=src python scripts/sidechain/train_safe.py \
  --npz data/training/features.npz \
  --fid-map data/training/features.fid_map.json \
  --vocab data/training/features.vocab.json \
  --index-file data/binkit_functions_common.json \
  --epochs 10 --batch-size 4 --num-pairs 10000 --lr 1e-3 \
  --save-path output/safe_best_model.pt --no-tb

# 第三步：训练 MultiModal
PYTHONPATH=src python scripts/sidechain/train_multimodal.py \
  --npz data/training/features.npz \
  --fid-map data/training/features.fid_map.json \
  --vocab data/training/features.vocab.json \
  --index-file data/binkit_functions_common.json \
  --epochs 20 --batch-size 4 --num-pairs 20000 --lr 1e-4 \
  --max-seq-len 512 --max-graph-nodes 128 --max-dfg-nodes 64 \
  --pairing-mode binkit_refined \
  --save-path output/best_model.pth --no-tb
```

### 向后兼容

不指定 `--npz` 时，训练脚本行为与修改前完全一致（使用 JSONL dict 格式）。

### 维度一致性要求

`jsonl_to_npz.py` 和训练脚本的以下参数**必须一致**：

| 参数 | jsonl_to_npz 默认 | train_multimodal 默认 | **注意** |
|------|--------------------|-----------------------|----------|
| `--max-seq-len` | 512 | 512 | ✅ 一致 |
| `--max-graph-nodes` | 128 | 128 | ✅ 一致 |
| `--max-dfg-nodes` | **64** | **128** | ⚠️ 不一致！需手动传相同值 |
| `--max-edges` | 512 | — | 无对应训练参数，自动适配 |

**建议**：始终显式传入这些参数，不依赖默认值。
