# SemPatch Replain

基于语义对比的固件漏洞分析原型。以 Ghidra P-code 为架构中立 IR，结合 LSIR（CFG/DFG）与多模态特征，实现函数级二进制相似性检测与 1-day 漏洞发现。

## Quick Start

```bash
# 1. Install
./scripts/setup.sh
source .venv/bin/activate

# 2. Quick demo (no Ghidra needed)
pytest -m fake_cve -v

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
| [docs/QUICKSTART.md](docs/QUICKSTART.md) | Copy-paste commands for 3 common workflows |
| [docs/WORKFLOWS.md](docs/WORKFLOWS.md) | Detailed workflows: library building, two-stage matching, single-stage matching |
| [docs/DEMO.md](docs/DEMO.md) | CVE match demo with two-stage pipeline |
| [docs/VULNERABILITY_LIBRARY.md](docs/VULNERABILITY_LIBRARY.md) | Building CVE libraries from your own binaries |

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
data/            — Datasets, vulnerability libraries, caches
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
