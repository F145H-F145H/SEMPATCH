# 多二进制伪 CVE 匹配样例（`FAKE-CVE-*`）

本目录提供 **10 个独立 ELF**（各含全局符号 `vuln_fake_01` … `vuln_fake_10`）、**`fake_cve_labeled.c`**（同源命名空间说明）与 **`manifest.json`**，用于走与生产一致的建库与查询管线，验收「输入查询二进制 → 报告中出现库内绑定的伪 CVE」。

**无 Ghidra 自动化验收**（编造 JSON 漏洞库）：在项目根执行 `pytest -m fake_cve`（见 `tests/fixtures/fake_cve/` 与 `tests/test_fake_cve_match.py`）。

更通用的 CVE Demo 契约见 [docs/DEMO.md](../../docs/DEMO.md)。

## 前置条件

- **Ghidra**（`GHIDRA_HOME` 或 `sempatch.cfg`）与 **Java**
- **PyTorch**：请使用项目 venv（系统 `python3` 常无 `torch`，会导致嵌入异常或管线失败）。参见 [data/cve_quick_demo/README.md](../../data/cve_quick_demo/README.md)。

## 一键运行

```bash
# 在项目根目录
./examples/fake_cve_demo/run_demo.sh
```

脚本会依次：编译 `build/*.elf` 与 `build/query.elf`（`query.elf` 为 `vuln_fake_05.elf` 的副本，保证与库中第 5 条同源）→ 写入 `data/fake_cve_demo_lib/` → 调用 `sempatch.py match`。

**双参数快捷入口**（等价于 `match --query-binary … --two-stage-dir …`，须 venv + Ghidra）：

```bash
./sempatch examples/fake_cve_demo/build/query.elf data/fake_cve_demo_lib \
  --cpu --output-dir output/fake_cve_demo_run \
  --query-entry 0x401176
```

`--query-entry 0x401176` 只匹配 **`vuln_fake_05` 函数入口**（与 `vuln_fake_05.elf` 同源时 rank 1 为 `FAKE-CVE-0005`，相似度约 1）。不传时会对**全部**查询函数出报告，首条往往是 `_start`/CRT，Top-1 CVE 可能不是 `0005`，易被误认为「未匹配」。`pipeline_status.json` 内 **`match_summary`** 会摘录首条查询的 Top-1 候选与 CVE。

## 分步命令（等价）

```bash
make -C examples/fake_cve_demo all
PY="${SEMPATCH_PYTHON:-.venv/bin/python}"
PYTHONPATH=src "$PY" scripts/build_fake_cve_demo_library.py \
  --manifest examples/fake_cve_demo/manifest.json \
  -o data/fake_cve_demo_lib \
  --write-library-safe-embeddings
PYTHONPATH=src "$PY" sempatch.py match \
  --query-binary examples/fake_cve_demo/build/query.elf \
  --library-features data/fake_cve_demo_lib/library_features.json \
  --library-emb data/fake_cve_demo_lib/library_safe_embeddings.json \
  --output-dir output/fake_cve_demo_run \
  --cpu
```

## 关于 `--max-queries`

产品默认对查询二进制提取的**全部**函数做匹配（`--max-queries 0` 或不传）。若仅做快速试跑，可加 `--max-queries N` 截断。

## 高置信度匹配模式（`--match-filter`）

默认 `top_k` 与 `--top-k` 用于快速浏览。若希望**只输出通过阈值的候选**（并可在无通过项时得到明确的 `match_status: no_credible_match`），可使用：

- **`unique`**：确信全库至多一个真匹配时；要求精排最高分 ≥ `--min-similarity`，且与次高分的差 **>** `--tie-margin`（否则视为并列，不输出）。本样例库中多个函数分数常极度接近，`unique` 可能因并列而空结果，属预期。
- **`all_above`**：探索性查看所有 ≥ 阈值的候选及分数分布；可先于 `unique` 使用以调整阈值或 `tie-margin`。

阈值与唯一性均为**启发式**，不保证 CVE 标注正确，需人工复核。须在 **已安装 PyTorch 的 venv** 下运行，两段权重才会参与推理；若降级为 **coarse_only_safe**，报告顶部会警告**未应用**上述阈值策略。

示例：

```bash
.venv/bin/python sempatch.py match \
  --query-binary examples/fake_cve_demo/build/query.elf \
  --two-stage-dir data/fake_cve_demo_lib \
  --cpu --output-dir output/fake_cve_demo_run \
  --query-entry 0x401176 \
  --match-filter all_above --min-similarity 0.9 --coarse-k 500
```

详见 `sempatch.py match --help` 文末说明。

## 验收

打开 `output/fake_cve_demo_run/report.md` 或 `matches.json`，在 Top 候选中应能看到 **`FAKE-CVE-0005`**（与 `query.elf` 同源条目），以及其它库条目的 `FAKE-CVE-*`。

自动化（可选）：`pytest tests/test_sempatch_argv.py::test_sempatch_two_positional_fake_cve_demo_matches_cve_0005`（`@pytest.mark.integration`，需 Ghidra、已 `make`、已建库；若无候选会 skip）。

**`./sempatch` 说明**：首参为已知子命令（`match` / `compare` / `extract` / `unpack`）时直接转发；**两位置参数**「查询 ELF + 含 `library_*.json` 的目录」时自动转为上述 `match`。

## 产物路径说明

- `examples/fake_cve_demo/build/`：由 `make` 生成，根目录 `.gitignore` 中的 `build` 规则可能忽略该目录；克隆仓库后需先执行 `make`。
- `data/fake_cve_demo_lib/*.json`：由建库脚本生成，已列入 `.gitignore`。

## 可选：SAFE 权重

`build_fake_cve_demo_library.py` 支持 `--model-path` 指向训练后的 SAFE 检查点；不传时仍可用「查询与库条目二进制同源」的方式完成验收。
