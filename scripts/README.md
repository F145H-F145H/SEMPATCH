# Scripts

Build, train, evaluate, and utility scripts.

## Core Pipeline

| Script | Purpose |
|--------|---------|
| `build_binkit_index.py` | Build function index from binary directory |
| `build_library_features.py` | Extract multimodal features for library functions |
| `build_embeddings_db.py` | Generate SAFE/multimodal embeddings from features |
| `demo_cve_match.py` | CVE match demo (two-stage: SAFE coarse + multimodal rerank) |
| `run_cve_pipeline.py` | End-to-end pipeline: extract features → match → report |

## Training

| Script | Purpose |
|--------|---------|
| `train_multimodal.py` | Train MultiModalFusionModel (reranking). `--synthetic` for quick test |
| `train_safe.py` | Train SAFE model (coarse retrieval). `--synthetic` for quick test |

## Evaluation

| Script | Purpose |
|--------|---------|
| `eval_two_stage.py` | Two-stage pipeline evaluation (Recall@K) |
| `eval_bcsd.py` | Binary code similarity detection metrics |

## Sidechain Scripts (`sidechain/`)

| Script | Purpose |
|--------|---------|
| `extract_cve_library.py` | Build CVE library from manifest (Ghidra → features → SAFE model → embeddings) |
| `build_fake_cve_demo_library.py` | Build fake CVE demo library (no Ghidra) |

## Analysis

| Script | Purpose |
|--------|---------|
| `analyze_match_results.py` | Analyze match output: score distribution, threshold sweep |
| `filter_index_by_pcode_len.py` | Filter function index by P-code length |
