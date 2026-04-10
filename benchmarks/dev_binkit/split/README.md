# dev_binkit/split/ — 固定划分（可提交版）

由 `prepare_two_stage_data.py --seed 42` 从 BinKit 全量数据生成。本目录存放的是**小型替身 fixture**（4 个二进制、11 库函数、9 查询函数、6 条 ground truth），供 CI 和 `make eval-dev` 在无 BinKit 全量数据时冒烟验证。

## 完整性校验

```bash
sha256sum -c benchmarks/dev_binkit/CHECKSUMS.sha256
```

## 文件说明

| 文件 | 说明 |
|------|------|
| `ground_truth.json` | 查询函数 ID → 正例库函数 ID 列表 |
| `query_features.json` | 查询函数的图+序列特征 |
| `library_features.json` | 库函数的图+序列特征 |
| `library_safe_embeddings.json` | 库函数 SAFE 嵌入（128 维） |

## 全量 BinKit 评测

全量数据需本地构建（需 Ghidra + BinKit 数据集）：

```bash
# 1. 构建索引
PYTHONPATH=src python scripts/sidechain/build_binkit_index.py ...

# 2. 生成划分
PYTHONPATH=src python scripts/sidechain/prepare_two_stage_data.py \
  --seed 42 --min-queries 1000 --min-pcode-len 16 \
  --output-dir benchmarks/dev_binkit/split

# 3. 运行评测
make eval-dev
```

**修改约束：** 变更 seed/min_queries 或索引输入即为新基准，须在 PR 中说明并更新 CHECKSUMS。
