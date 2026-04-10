# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SemPatch Replain — a firmware 1-day vulnerability discovery prototype. It decomposes binaries via Ghidra's P-code IR, builds LSIR (CFG/DFG) representations, extracts multimodal features (graph + sequence + optional DFG), and uses a two-stage pipeline (SAFE coarse retrieval via FAISS + multimodal reranking) for function-level binary similarity matching.

Primary language: Python 3.8+. External deps: Ghidra 12.0, PyTorch, FAISS, networkx, ssdeep, py-tlsh.

## Commands

```bash
# Activate venv
source .venv/bin/activate

# Run tests
pytest                          # all tests
pytest -m "not ghidra"          # skip Ghidra-dependent tests
pytest -m fake_cve              # fake CVE match pipeline tests (data: benchmarks/smoke/fake_cve)
make eval-smoke                 # same as pytest -m fake_cve
make eval-dev / make eval-real  # see benchmarks/README.md
pytest tests/test_dag.py        # single test file

# Lint/format (black line-length=100, isort profile=black, ruff select E/F/I/W/UP)
black src/ tests/
isort src/ tests/
ruff check src/ tests/

# Production match
python sempatch.py match --query-binary <elf> --two-stage-dir <dir> --output-dir <out>

# Training
PYTHONPATH=src python scripts/train_multimodal.py --synthetic --epochs 2
PYTHONPATH=src python scripts/train_safe.py --synthetic --epochs 1

# Setup
./scripts/setup.sh               # full (Python + Ghidra + binwalk)
./scripts/setup.sh --skip-ghidra # Python only
```

## Architecture

### DAG Execution Engine (`src/dag/`)

The core orchestrator. Pipelines are modeled as a `JobDAG` of `DAGNode` instances. Each node implements `execute(ctx)`. Nodes are registered in `NODE_TYPE_REGISTRY` (`src/dag/nodes/__init__.py`). The executor (`src/dag/executor.py`) runs nodes via `ThreadPoolExecutor` with semaphore-based concurrency (Ghidra nodes get separate thread slots). Builders in `src/dag/builders/` compose pipelines by adding typed nodes to the DAG.

### TwoStage Pipeline (`src/matcher/two_stage.py`)

Independent of the DAG engine. `TwoStagePipeline.retrieve()` does FAISS coarse filtering via SAFE embeddings, then `rerank()` applies multimodal scoring. Called directly from `src/cli/cve_match.py` (the `match` subcommand).

### Feature Extraction (`src/utils/feature_extractors/`)

- `graph_features.py` — ACFG (Attributed Control Flow Graph) features
- `sequence_features.py` — P-code sequence + jump encoding
- `fusion.py` — multimodal feature combination
- `multimodal_extraction.py` — end-to-end extraction from LSIR raw

### Models (`src/features/models/multimodal_fusion.py`)

`MultiModalFusionModel`: graph branch (GNN) + sequence branch (Transformer) + cross-modal attention + optional DFG branch. Baseline: `SafeEmbedder` in `src/features/baselines/safe.py`.

### Pipeline Strategies

Configured via `[dag] pipeline_strategy` in `sempatch.cfg`: `semantic_embed` (default), `fusion`, `graph_embed`, `traditional_fuzzy`, `traditional_cfg`.

### Configuration (`src/config.py`)

Loads from `sempatch.cfg` (INI format) with env var overrides. Config template: `sempatch.cfg.example`.

## Development Rules

- **Before writing code**: read `docs/WORKFLOWS.md` for workflow documentation.
- **After major milestones**: update relevant docs in `docs/`.
- Source root is `src/` (pytest `pythonpath = ["src"]`; imports use bare module names like `from config import ...`).
- Documentation is primarily in Chinese (Mandarin).
