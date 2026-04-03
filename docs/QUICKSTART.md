# Quick Start

Three workflows from simplest to most complete. See [WORKFLOWS.md](WORKFLOWS.md) for full details.

## 1. Quick Demo (no Ghidra required)

Run the fake CVE match test to verify the pipeline works:

```bash
source .venv/bin/activate
pytest -m fake_cve -v
```

This uses pre-computed features — no Ghidra, no compilation needed.

## 2. Two-Stage Match (recommended)

Match a query binary against a CVE library using SAFE coarse retrieval + multimodal reranking:

```bash
# Build CVE library from manifest
PYTHONPATH=src .venv/bin/python scripts/sidechain/extract_cve_library.py \
  --manifest examples/mvp_library/manifest.json \
  -o data/mvp_library_cve

# Run two-stage match
.venv/bin/python sempatch.py match \
  --query-binary examples/mvp_library/build/mvp_vuln_01.elf \
  --two-stage-dir data/mvp_library_cve \
  --output-dir output/mvp_library_match \
  --cpu
```

Output: `output/mvp_library_match/matches.json`, `report.md`, `pipeline_status.json`.

Use `--profile quick|standard|full` to control search depth:

```bash
.venv/bin/python sempatch.py match \
  --query-binary path/to/binary \
  --two-stage-dir data/mvp_library_cve \
  --profile standard \
  --cpu
```

| Profile | coarse_k | top_k | filter | use case |
|---------|----------|-------|--------|----------|
| `quick` | 50 | 5 | top_k | fast exploration |
| `standard` | 100 | 10 | unique (0.95) | balanced |
| `full` | 500 | 50 | all_above (0.9) | thorough scan |

## 3. Build Your Own CVE Library

Create a manifest JSON mapping binaries + functions to CVEs:

```json
[
  {"binary": "path/to/libfoo.so", "function_name": "vuln_parse", "cve": ["CVE-2024-1234"]}
]
```

Then build:

```bash
PYTHONPATH=src .venv/bin/python scripts/sidechain/extract_cve_library.py \
  --manifest my_manifest.json \
  -o data/my_cve_lib
```

See [WORKFLOWS.md](WORKFLOWS.md) for details on prerequisites and troubleshooting.
