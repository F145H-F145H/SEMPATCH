#!/usr/bin/env bash
# 一键跑通 MVP：编译 vulnerable → CFG 匹配 → sempatch match（测试库 + 示例 CVE）
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ -x "$ROOT/.venv/bin/python" ]]; then
  PY="${SEMPATCH_PYTHON:-$ROOT/.venv/bin/python}"
else
  PY="${SEMPATCH_PYTHON:-python3}"
fi

export PYTHONPATH="$ROOT/src"

echo "=== 使用 Python: $PY ==="
"$PY" -c "import networkx" 2>/dev/null || {
  echo "错误: 未安装 networkx，请: pip install networkx 或使用含依赖的 venv" >&2
  exit 1
}

echo "=== 1) 编译 examples/mvp_vulnerable ==="
make -C "$ROOT/examples/mvp_vulnerable"

BIN="$ROOT/examples/mvp_vulnerable/vulnerable"

echo "=== 2) 路径 A: extract → vuln_db → compare (traditional_cfg) ==="
"$PY" "$ROOT/sempatch.py" extract "$BIN" -o "$ROOT/output/mvp_extract" --force
"$PY" "$ROOT/scripts/build_vuln_db.py" "$ROOT/output/mvp_extract/lsir_raw.json" \
  -o "$ROOT/data/vulnerability_db/mvp_vuln_lsir.json" --filter vuln_
"$PY" "$ROOT/sempatch.py" compare "$BIN" "$ROOT/data/vulnerability_db/mvp_vuln_lsir.json" \
  -o "$ROOT/output/mvp_cfg" --strategy traditional_cfg --force

MATCHES_A="$ROOT/output/mvp_cfg/diff_result.json"
if "$PY" -c "import json; d=json.load(open('$MATCHES_A')); m=d.get('matches') or []; assert any(x.get('db_func','').startswith('vuln_') for x in m)" 2>/dev/null; then
  echo "路径 A 验收: diff_result 中含 vuln_* 匹配"
else
  echo "路径 A 警告: 未在 diff_result 中检测到预期匹配，请检查 networkx 与 Ghidra 日志" >&2
fi

echo "=== 3) 路径 B: 生成 data/cve_quick_demo 库 + sempatch.py match（正式入口）==="
"$PY" "$ROOT/scripts/build_mvp_vulnerable_cve_library.py" \
  -o "$ROOT/data/cve_quick_demo" \
  --unified-cve CVE-2018-10822 \
  --write-library-safe-embeddings
"$PY" "$ROOT/sempatch.py" match \
  --query-binary "$ROOT/examples/mvp_vulnerable/vulnerable" \
  --two-stage-dir "$ROOT/data/cve_quick_demo" \
  --output-dir "$ROOT/output/cve_quick_demo_run" \
  --cpu

MATCHES_B="$ROOT/output/cve_quick_demo_run/matches.json"
if grep -q "CVE-2018-10822" "$MATCHES_B" 2>/dev/null; then
  echo "路径 B 验收: matches.json 中含 CVE-2018-10822（与库预写一致）"
else
  echo "路径 B 警告: 未在 matches.json 中发现 CVE-2018-10822" >&2
fi

echo "=== 完成 ==="
echo "  CFG 结果: $MATCHES_A"
echo "  CVE 报告: $ROOT/output/cve_quick_demo_run/report.md（pipeline_status.json 同目录）"
echo "  库目录:   $ROOT/data/cve_quick_demo/（library_safe_embeddings.json + library_features.json）"
