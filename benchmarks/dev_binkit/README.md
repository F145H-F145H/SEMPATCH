# Dev BinKit 基准（调参唯一对比组）

**固定契约：** `--seed 42`，`--min-queries 1000`，`--min-pcode-len 16`。变更任一参数即为新基准。

## split/ — 已提交的小型 fixture

`split/` 目录当前包含一个 **4 二进制 / 11 库函数 / 9 查询函数** 的替身 fixture，供 CI 和 `make eval-dev` 冒烟验证。数据由 `prepare_two_stage_data.py --seed 42` 的同一格式生成。

```bash
make eval-dev  # 直接可用，无需 BinKit 全量数据
```

## 完整性校验

```bash
sha256sum -c benchmarks/dev_binkit/CHECKSUMS.sha256
```

## 全量 BinKit 评测

全量数据需本地构建（需 Ghidra + BinKit 数据集）：

1. **索引**（需 Ghidra）
   ```bash
   PYTHONPATH=src python scripts/sidechain/build_binkit_index.py \
     --input-dir data/binkit_subset -o data/binkit_functions.json
   ```

2. **（可选）pcode 过滤**
   ```bash
   PYTHONPATH=src python scripts/sidechain/filter_index_by_pcode_len.py \
     -i data/binkit_functions.json -o data/binkit_functions_filtered.json \
     --min-pcode-len 16
   ```

3. **划分**（固定种子，覆盖 split/）
   ```bash
   PYTHONPATH=src python scripts/sidechain/prepare_two_stage_data.py \
     --index-file data/binkit_functions_filtered.json \
     --output-dir benchmarks/dev_binkit/split \
     --seed 42 --min-queries 1000
   ```

4. **特征与 SAFE 嵌入**（输出到 gitignored 的 `artifacts/`）
   按 `docs/DEVELOPMENT.md` 两阶段流水线，生成 `ground_truth.json`、`query_features.json`、`library_features.json`、`library_safe_embeddings.json`。

`split/` 内的 JSON 可被全量数据覆盖。覆盖后须更新 `CHECKSUMS.sha256` 并在 PR 中说明。

## manifest.json

冻结参数记录（不参与 CI），详见 [`manifest.json`](manifest.json)。
