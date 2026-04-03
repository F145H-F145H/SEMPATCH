# Workflows

Three core workflows: CVE Library Building, Two-Stage Matching, and Single-Stage Matching.

## Prerequisites

```bash
# Install dependencies
./scripts/setup.sh

# Or Python-only (skip Ghidra)
./scripts/setup.sh --skip-ghidra

# Activate venv
source .venv/bin/activate
```

Ghidra is required for binary input (`--query-binary`). Pre-computed features (`--query-features`) bypass Ghidra.

---

## Workflow A: CVE Library Building

**Purpose**: Convert your binary collection into a searchable CVE library.

**Input**: Manifest JSON + compiled ELF binaries.
**Output**: `library_features.json`, `library_safe_embeddings.json`, `library_cve_map.json`, `safe_model.pt`.

### Step 1: Prepare binaries and manifest

Create a manifest mapping each binary + vulnerable function to CVE labels:

```json
[
  {"binary": "path/to/binary_a", "function_name": "vuln_func_a", "cve": ["CVE-2024-0001"]},
  {"binary": "path/to/binary_b", "function_name": "vuln_func_b", "cve": ["CVE-2024-0002"]}
]
```

Optional fields: `"entry": "0x401000"` (use when function name is ambiguous).

### Step 2: Build the library

```bash
PYTHONPATH=src .venv/bin/python scripts/sidechain/extract_cve_library.py \
  --manifest my_manifest.json \
  -o data/my_cve_lib
```

This runs: Ghidra extraction → multimodal features → SAFE model training → SAFE embeddings.

### Step 3: Verify output

```bash
ls data/my_cve_lib/
# library_features.json       — multimodal features for reranking
# library_safe_embeddings.json — SAFE vectors for coarse retrieval
# library_cve_map.json        — function_id → CVE mapping
# safe_model.pt               — trained SAFE model
```

### Troubleshooting

| Problem | Fix |
|---------|-----|
| `GhidraEnvironmentError` | Set `GHIDRA_HOME` or edit `sempatch.cfg` |
| `torch not found` | Use `.venv/bin/python` (not system python) |
| OOM on large libraries | Reduce `--safe-pairs` or process manifests in batches |

---

## Workflow B: Two-Stage Matching

**Purpose**: Match a query binary against a CVE library using SAFE coarse retrieval + multimodal reranking.

**Prerequisites**: A built CVE library (Workflow A output) + Ghidra for query binary.

### Basic usage

```bash
.venv/bin/python sempatch.py match \
  --query-binary path/to/query.elf \
  --two-stage-dir data/my_cve_lib \
  --output-dir output/match_run \
  --cpu
```

### With preset profile

```bash
.venv/bin/python sempatch.py match \
  --query-binary path/to/query.elf \
  --two-stage-dir data/my_cve_lib \
  --profile standard \
  --cpu
```

### With inspection

```bash
.venv/bin/python sempatch.py match \
  --query-binary path/to/query.elf \
  --two-stage-dir data/my_cve_lib \
  --inspect \
  --cpu
```

### Key parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--profile` | (none) | Preset: `quick`, `standard`, `full` |
| `--coarse-k` | 100 | FAISS candidates from SAFE coarse stage |
| `--top-k` | 10 | Results to report (top_k filter mode) |
| `--match-filter` | top_k | `top_k`, `unique`, `all_above` |
| `--min-similarity` | 0.95 | Threshold for unique/all_above |
| `--max-queries` | 0 (all) | Limit query functions processed |
| `--cpu` | false | Force CPU (no CUDA) |
| `--inspect` | false | Print score analysis after match |

### Output files

| File | Content |
|------|---------|
| `matches.json` | All query results with scores, CVE labels, filter metadata |
| `report.md` | Human-readable match report |
| `pipeline_status.json` | Execution status, warnings, artifact paths |

### Match filter modes

- **`top_k`**: Report top K results regardless of score. Best for exploration.
- **`unique`**: Report only if top score >= threshold AND gap to second > tie-margin. Best for confident matches.
- **`all_above`**: Report all candidates above threshold. Best for understanding score distribution.

---

## Workflow C: Single-Stage Matching

**Purpose**: FAST coarse-only matching without multimodal reranking. Use when `library_features.json` is unavailable or too large.

This mode is automatically triggered when:
- `library_features.json` is missing, OR
- `library_features.json` exceeds `--max-library-features-mb-for-two-stage` (default 2048 MB)

### Usage

```bash
.venv/bin/python sempatch.py match \
  --query-binary path/to/query.elf \
  --two-stage-dir data/my_cve_lib \
  --allow-coarse-fallback \
  --cpu
```

### Disable fallback (fail instead)

```bash
.venv/bin/python sempatch.py match \
  --query-binary path/to/query.elf \
  --two-stage-dir data/my_cve_lib \
  --no-allow-coarse-fallback \
  --cpu
```

### When to use single-stage

- Large libraries where loading multimodal features causes OOM
- Quick screening where coarse SAFE similarity is sufficient
- Libraries that only have SAFE embeddings (no multimodal features)

### Limitations

- No multimodal reranking — less accurate for function-level matching
- Score distribution may be less discriminative
- `--match-filter unique` may produce more false positives
