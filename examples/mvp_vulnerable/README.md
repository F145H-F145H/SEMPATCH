# MVP：自编译 vulnerable 与 SemPatch 匹配验证

含 `vuln_copy`（`strcpy` 栈溢出模式）与 `vuln_loop`（循环越界写入），用于本地验证 Ghidra 提取、漏洞库与匹配管线。

## 依赖

- 编译器：`gcc`
- Python：**CVE 管线与 SAFE 嵌入必须能 `import torch`**。不要用系统裸 `python` 若未装 PyTorch。建议使用 **`.venv/bin/python`**：`python3 -m venv .venv && .venv/bin/pip install -r requirements.txt`（若 `ssdeep` 构建失败可先 `.venv/bin/pip install networkx torch` 等核心依赖）
- **Java 17+**、**Ghidra**，`GHIDRA_HOME` 或 `sempatch.cfg` 已配置

## 编译

```bash
make -C examples/mvp_vulnerable
```

## 路径 A：CFG 结构匹配（无 CVE 字段）

```bash
export PYTHONPATH=src
./sempatch extract examples/mvp_vulnerable/vulnerable -o output/mvp_extract --force
python scripts/build_vuln_db.py output/mvp_extract/lsir_raw.json \
  -o data/vulnerability_db/mvp_vuln_lsir.json --filter vuln_
./sempatch compare examples/mvp_vulnerable/vulnerable \
  data/vulnerability_db/mvp_vuln_lsir.json \
  -o output/mvp_cfg --strategy traditional_cfg --force
```

验收：`output/mvp_cfg/diff_result.json` 的 `matches` 中含 `vuln_copy` / `vuln_loop` 与库中同名函数的高 `mcs_ratio` 配对。

## 路径 B：正式入口 `run_cve_pipeline.py`（推荐，「正常」CVE 库）

先写入与 `data/two_stage` 相同文件名的库（示例绑定文档中的 `CVE-2018-10822`）：

```bash
export PYTHONPATH=src
python scripts/build_mvp_vulnerable_cve_library.py \
  -o data/cve_quick_demo \
  --unified-cve CVE-2018-10822 \
  --write-library-safe-embeddings
python scripts/run_cve_pipeline.py \
  --query-binary examples/mvp_vulnerable/vulnerable \
  --two-stage-dir data/cve_quick_demo \
  --output-dir output/cve_quick_demo_run \
  --max-queries 8 --cpu
```

验收：`output/cve_quick_demo_run/matches.json` / `report.md` 中出现 `CVE-2018-10822` 即可（库与查询来自同一测试二进制，预匹配）。

## 路径 B'：直接调用 `demo_cve_match.py`（等价能力）

```bash
export PYTHONPATH=src
python scripts/build_mvp_vulnerable_cve_library.py -o output/mvp_cve_lib
python scripts/demo_cve_match.py \
  --query-binary examples/mvp_vulnerable/vulnerable \
  --library-emb output/mvp_cve_lib/library_embeddings.json \
  --library-features output/mvp_cve_lib/library_features.json \
  --output-dir output/mvp_cve_demo \
  --max-queries 30 --cpu
```

验收：`output/mvp_cve_demo/matches.json` 中 Top 候选含 `CVE-MVP-0001` / `CVE-MVP-0002`（默认未传 `--unified-cve` 时）。

## 一键脚本

```bash
scripts/run_mvp_vulnerable_demo.sh
```

可选环境变量：`SEMPATCH_PYTHON`（默认优先 `.venv/bin/python`，否则 `python3`）。
