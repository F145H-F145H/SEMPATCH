# real_cve/ — 真实 CVE 评测基准（最终价值判断）

用于 `make eval-real`（`eval_bcsd.py --mode cve`）。

## 示例数据

`example_data/` 包含一个小型 fixture（3 查询、5 库函数、含 CVE 交叉引用），供冒烟测试 `eval_bcsd`：

```bash
PYTHONPATH=src python scripts/eval_bcsd.py \
  --firmware-emb benchmarks/real_cve/example_data/query_embeddings.json \
  --db-emb benchmarks/real_cve/example_data/library_embeddings.json \
  --mode cve -k 1 2 3
```

## 完整性校验

```bash
sha256sum -c benchmarks/real_cve/CHECKSUMS.sha256
```

## 全量评测数据

嵌入文件体积大，默认 **gitignore**：

- `query_embeddings.json` — 待查/固件侧嵌入（`EmbeddingDict`，条目含可选 `cve` 列表）
- `library_embeddings.json` — 漏洞库嵌入（同上）

### CVE 映射侧车

构建漏洞库时可将 `cve_mapping.json` 放在扫描根目录，格式见 `cve_mapping.example.json`。

完整建库流程见 [`docs/VULNERABILITY_LIBRARY.md`](../../docs/VULNERABILITY_LIBRARY.md)。
