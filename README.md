# SemPatch Replain

基于语义对比的固件漏洞分析原型。以 Ghidra P-code 为架构中立 IR，结合 LSIR（CFG/DFG）与多模态特征，实现函数级二进制相似性检测与 1-day 漏洞发现。

## 当前状态

**研究原型闭环，主链路稳定，部分大规模路径有工程限制。**

进度、里程碑与已知工程限制以仓库内 [`memory-bank/progress.md`](memory-bank/progress.md) 为准；数据结构与节点契约以 [`memory-bank/@architecture.md`](memory-bank/@architecture.md)、[`memory-bank/@design-document.md`](memory-bank/@design-document.md) 为准。能力边界请结合 **`pytest`**（如 `pytest -m fake_cve`、`pytest -m "not ghidra"`）与文档中的可复现命令核对，勿再以根目录 `TODO.md`、`项目现状.md` 为事实来源（二者已废弃，见其中说明）。

## Quick Start

```bash
# 1. Install
./scripts/setup.sh
source .venv/bin/activate

# 2. Quick demo (no Ghidra needed)
make eval-smoke
# 等价: pytest -m fake_cve -v

# 3. Two-stage match with preset profile
.venv/bin/python sempatch.py match \
  --query-binary path/to/binary \
  --two-stage-dir data/my_cve_lib \
  --profile standard \
  --cpu
```

## Documentation

| Document | Content |
|----------|---------|
| [memory-bank/progress.md](memory-bank/progress.md) | 实施进度、回归记录与工程限制（与代码/测试一并作为事实来源） |
| [memory-bank/@architecture.md](memory-bank/@architecture.md) | 数据结构、节点契约、JSON 形态（开发前必读） |
| [memory-bank/@design-document.md](memory-bank/@design-document.md) | 技术栈、目录、流程与评估计划（开发前必读） |
| [docs/QUICKSTART.md](docs/QUICKSTART.md) | Copy-paste commands for 3 common workflows |
| [docs/WORKFLOWS.md](docs/WORKFLOWS.md) | Detailed workflows: library building, two-stage matching, single-stage matching |
| [docs/DEMO.md](docs/DEMO.md) | CVE match demo with two-stage pipeline |
| [docs/VULNERABILITY_LIBRARY.md](docs/VULNERABILITY_LIBRARY.md) | Building CVE libraries from your own binaries |
| [benchmarks/README.md](benchmarks/README.md) | 固化评测基准（smoke / dev_binkit / real_cve）与 `make eval-*` |

## Key Commands

```bash
# Build CVE library from manifest
PYTHONPATH=src .venv/bin/python scripts/sidechain/extract_cve_library.py \
  --manifest examples/mvp_library/manifest.json -o data/my_cve_lib

# Match with profile preset (quick/standard/full)
.venv/bin/python sempatch.py match --query-binary <elf> --two-stage-dir <dir> --profile standard

# Match with inspection
.venv/bin/python sempatch.py match --query-binary <elf> --two-stage-dir <dir> --inspect

# Analyze match results
PYTHONPATH=src .venv/bin/python scripts/analyze_match_results.py output/matches.json
```

## Architecture

```
Binary → Ghidra P-code → LSIR (CFG/DFG) → Multimodal Features
                                                    ↓
Query Binary → [same pipeline] → SAFE Embedding → FAISS Coarse Retrieval
                                                    ↓
                                          Multimodal Reranking → matches.json / report.md
```

- **Two-stage pipeline**: SAFE coarse (FAISS IndexFlatIP) + multimodal rerank (MultiModalFusionModel)
- **Single-stage fallback**: SAFE + FAISS only, when multimodal features unavailable

## Project Structure

```
src/             — Source code (frontend/, dag/, utils/, features/, matcher/, cli/)
scripts/         — Build, train, evaluate scripts
examples/        — Sample binaries and manifests
data/            — Datasets, vulnerability libraries, caches (gitignored)
benchmarks/      — Versioned eval fixtures (smoke) + dev/real CVE benchmark layout
tests/           — Unit and integration tests
docs/            — Documentation
memory-bank/     — Architecture, design, progress tracking
```

## Development

Before writing code: read `memory-bank/@architecture.md` and `memory-bank/@design-document.md`.

```bash
# Run tests
pytest                           # all
pytest -m "not ghidra"           # skip Ghidra tests
pytest -m fake_cve               # fake CVE match tests
make eval-smoke                  # same as pytest -m fake_cve
make eval-dev                    # eval_two_stage on benchmarks/dev_binkit/artifacts (needs materialization)
make eval-real                   # eval_bcsd --mode cve on benchmarks/real_cve/*.json

# Lint
black src/ tests/ && isort src/ tests/ && ruff check src/ tests/
```

## Dependencies

- Python 3.8+
- Java 17+ (Ghidra 12.0)
- PyTorch (required for matching; optional for feature extraction)
- See `requirements.txt`

## License

Research prototype.
