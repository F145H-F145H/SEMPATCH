#!/usr/bin/env bash
# 编译样例 ELF → 构建伪 CVE 库 → 对 query.elf 跑 sempatch match。
# 需要：Java/Ghidra、项目 venv 中已安装 PyTorch（见 README）。
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

PY="${SEMPATCH_PYTHON:-${ROOT}/.venv/bin/python}"
if [[ ! -x "${PY}" ]]; then
  PY="python3"
fi

export PYTHONPATH="${ROOT}/src"

echo "==> make (examples/fake_cve_demo)"
make -C "${ROOT}/examples/fake_cve_demo" all

LIB_OUT="${ROOT}/data/fake_cve_demo_lib"
MANIFEST="${ROOT}/examples/fake_cve_demo/manifest.json"
QUERY="${ROOT}/examples/fake_cve_demo/build/query.elf"
PIPE_OUT="${ROOT}/output/fake_cve_demo_run"

echo "==> build_fake_cve_demo_library.py -> ${LIB_OUT}"
"${PY}" "${ROOT}/scripts/build_fake_cve_demo_library.py" \
  --manifest "${MANIFEST}" \
  -o "${LIB_OUT}" \
  --write-library-safe-embeddings

echo "==> sempatch.py match (query = 与 vuln_fake_05 同源的二进制)"
"${PY}" "${ROOT}/sempatch.py" match \
  --query-binary "${QUERY}" \
  --library-features "${LIB_OUT}/library_features.json" \
  --library-emb "${LIB_OUT}/library_safe_embeddings.json" \
  --output-dir "${PIPE_OUT}" \
  --query-entry 0x401176 \
  --cpu

echo ""
echo "=== 验收：首条查询 Top-1 应为 vuln_fake_05 / FAKE-CVE-0005（见 pipeline_status match_summary）==="
if [[ -f "${PIPE_OUT}/pipeline_status.json" ]]; then
  "${PY}" -c "import json; d=json.load(open('${PIPE_OUT}/pipeline_status.json')); print(json.dumps(d.get('match_summary'), indent=2, ensure_ascii=False))"
fi
if [[ -f "${PIPE_OUT}/report.md" ]]; then
  grep -E "FAKE-CVE-0005" "${PIPE_OUT}/report.md" | head -5 || true
else
  echo "(无 report.md)"
fi
echo ""
echo "完整报告: ${PIPE_OUT}/report.md"
echo "JSON:     ${PIPE_OUT}/matches.json"
