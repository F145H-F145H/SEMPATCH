"""MVP 端到端冒烟测试：真实二进制 → Ghidra 提取 → CVE 库匹配 → 验证输出（无预计算特征）。

需要：Ghidra、PyTorch（.venv）、已编译 examples/mvp_vulnerable/vulnerable。
跳过条件：pytest -m "not ghidra"。
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_VENV_PYTHON = os.path.join(_PROJECT_ROOT, ".venv", "bin", "python")
_VULN_BINARY = os.path.join(_PROJECT_ROOT, "examples", "mvp_vulnerable", "vulnerable")
_CVE_QUICK_DIR = os.path.join(_PROJECT_ROOT, "data", "cve_quick_demo")

pytestmark = [
    pytest.mark.integration,
    pytest.mark.ghidra,
    pytest.mark.skipif(
        not os.path.isfile(_VENV_PYTHON),
        reason=".venv/bin/python not found (PyTorch required)",
    ),
    pytest.mark.skipif(
        not os.path.isfile(_VULN_BINARY),
        reason="examples/mvp_vulnerable/vulnerable not compiled",
    ),
]


def _collect_cves(data: dict) -> set[str]:
    cves: set[str] = set()
    for q in data.get("queries") or []:
        for c in q.get("candidates") or []:
            for x in c.get("cve") or []:
                if isinstance(x, str):
                    cves.add(x)
    return cves


def test_mvp_binary_to_cve(tmp_path) -> None:
    """sempatch.py match --query-binary <vulnerable ELF> → 匹配库中 CVE-2018-10822。"""
    out = tmp_path / "mvp_out"
    r = subprocess.run(
        [
            _VENV_PYTHON,
            os.path.join(_PROJECT_ROOT, "sempatch.py"),
            "match",
            "--query-binary",
            _VULN_BINARY,
            "--two-stage-dir",
            _CVE_QUICK_DIR,
            "--output-dir",
            str(out),
            "--coarse-k",
            "10",
            "--top-k",
            "5",
            "--max-queries",
            "5",
            "--cpu",
        ],
        cwd=_PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert r.returncode == 0, f"Pipeline failed:\n{r.stderr}\n{r.stdout}"

    # matches.json
    matches_path = out / "matches.json"
    assert matches_path.is_file(), "matches.json not produced"
    doc = json.loads(matches_path.read_text(encoding="utf-8"))
    assert "queries" in doc and len(doc["queries"]) >= 1
    found = _collect_cves(doc)
    assert "CVE-2018-10822" in found, f"Expected CVE-2018-10822, got: {found}"

    # report.md
    report_path = out / "report.md"
    assert report_path.is_file(), "report.md not produced"
    assert "CVE-2018-10822" in report_path.read_text(encoding="utf-8")

    # pipeline_status.json
    status_path = out / "pipeline_status.json"
    assert status_path.is_file()
    status = json.loads(status_path.read_text(encoding="utf-8"))
    assert status.get("ok") is True
