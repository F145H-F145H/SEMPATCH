# smoke/ — Commit 前快速验证

**用途：** `pytest -m fake_cve` 与 `eval_two_stage` 最小输入。无需 Ghidra，无需网络，< 5 秒完成。

## 数据文件

| 子目录 | 文件 | 说明 |
|--------|------|------|
| `fake_cve/` | `query_features.json` | 1 条查询函数 |
| | `library_features.json` | 2 条库函数 |
| | `library_embeddings.json` | 同上，含 CVE 标签 |
| `two_stage/` | `ground_truth.json` | 查询→库映射 |
| | `query_features.json` | 同 fake_cve 查询 |
| | `library_features.json` | 同 fake_cve 库函数 |
| | `library_safe_embeddings.json` | SAFE 嵌入（无 CVE 字段） |

## 完整性校验

```bash
sha256sum -c benchmarks/smoke/CHECKSUMS.sha256
```

**修改约束：** 除非特征提取格式发生破坏性变更，否则不要修改此目录下的 JSON 文件。任何变更须同步更新 `CHECKSUMS.sha256` 并在 PR 中说明。

## 关联测试

- `tests/test_fake_cve_match.py` — CVE 匹配 pipeline
- `tests/test_eval_two_stage_cli.py` — eval_two_stage CLI 冒烟
- `tests/test_demo_cve_match.py` — demo CVE 匹配
