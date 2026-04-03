# 侧链脚本（非产品入口）

**对外唯一推荐入口**：项目根目录 `python sempatch.py match ...`（TwoStage CVE 匹配，默认全函数）。

本目录包含：

- **兼容入口**：`run_cve_pipeline.py`、`demo_cve_match.py`（逻辑在 `src/cli/`，此处可继续用 `python scripts/<name>.py` 桩转发调用）。
- **漏洞库构建**：`build_library_*`、`build_embeddings_db.py`、`annotate_library_embeddings_cve.py` 等。
- **训练 / 评估 / 数据集**：`train_multimodal.py`、`train_safe.py`、`eval_*.py`、`download_*.py` 等。

`scripts/` 下与侧链同名的 `.py` 文件为 **转发桩**（`runpy` 至 `scripts/sidechain/`），便于旧文档中的路径仍可用。
