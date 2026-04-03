"""MVP 库提取器端到端测试：编译真实二进制 → 提取器构建 CVE 库 → two-stage 匹配 → 验证输出。

需要：Ghidra、PyTorch（.venv）、gcc。
跳过条件：无（Ghidra 必需）。
"""

from __future__ import annotations

import json
import os
import subprocess

import pytest

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_VENV_PYTHON = os.path.join(_PROJECT_ROOT, ".venv", "bin", "python")
_LIBRARY_DIR = os.path.join(_PROJECT_ROOT, "examples", "mvp_library")
_MANIFEST = os.path.join(_LIBRARY_DIR, "manifest.json")
_BINARY_01 = os.path.join(_LIBRARY_DIR, "build", "mvp_vuln_01.elf")

pytestmark = [
    pytest.mark.integration,
    pytest.mark.ghidra,
    pytest.mark.skipif(
        not os.path.isfile(_VENV_PYTHON),
        reason=".venv/bin/python not found (PyTorch required)",
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


def test_mvp_library_extractor_e2e(tmp_path) -> None:
    """编译 mvp_library → 提取器构建 CVE 库 → two-stage 匹配 → 验证 CVE-MVP-2024-0001。"""
    # 1) 编译
    make_r = subprocess.run(
        ["make", "-C", os.path.join(_PROJECT_ROOT, "examples", "mvp_library"), "all"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert make_r.returncode == 0, f"Make failed:\n{make_r.stderr}"
    assert os.path.isfile(_BINARY_01), f"Binary not compiled: {_BINARY_01}"

    # 2) 提取器构建 CVE 库
    lib_out = tmp_path / "mvp_library_cve"
    ext_r = subprocess.run(
        [
            _VENV_PYTHON,
            os.path.join(_PROJECT_ROOT, "scripts", "sidechain", "extract_cve_library.py"),
            "--manifest",
            _MANIFEST,
            "-o",
            str(lib_out),
        ],
        cwd=_PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=300,
        env={**os.environ, "PYTHONPATH": "src"},
    )
    assert ext_r.returncode == 0, f"Extractor failed:\n{ext_r.stderr}\n{ext_r.stdout}"

    # 验证提取器产物
    assert (lib_out / "library_features.json").is_file()
    assert (lib_out / "library_safe_embeddings.json").is_file()
    assert (lib_out / "safe_model.pt").is_file()

    with open(lib_out / "library_features.json", encoding="utf-8") as f:
        feats = json.load(f)
    assert len(feats) == 3, f"Expected 3 functions, got {len(feats)}"

    # 3) two-stage 匹配（使用提取器训练的 SAFE 模型）
    match_out = tmp_path / "match_result"
    safe_model = str(lib_out / "safe_model.pt")
    match_r = subprocess.run(
        [
            _VENV_PYTHON,
            os.path.join(_PROJECT_ROOT, "sempatch.py"),
            "match",
            "--query-binary",
            _BINARY_01,
            "--two-stage-dir",
            str(lib_out),
            "--safe-model-path",
            safe_model,
            "--output-dir",
            str(match_out),
            "--coarse-k",
            "10",
            "--top-k",
            "5",
            "--max-queries",
            "20",
            "--cpu",
        ],
        cwd=_PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert match_r.returncode == 0, f"Match failed:\n{match_r.stderr}\n{match_r.stdout}"

    # 4) 验证匹配结果
    matches_path = match_out / "matches.json"
    assert matches_path.is_file(), "matches.json not produced"
    doc = json.loads(matches_path.read_text(encoding="utf-8"))
    assert "queries" in doc and len(doc["queries"]) >= 1

    found = _collect_cves(doc)
    assert "CVE-MVP-2024-0001" in found, f"Expected CVE-MVP-2024-0001, got: {found}"

    report_path = match_out / "report.md"
    assert report_path.is_file(), "report.md not produced"
    assert "CVE-MVP-2024-0001" in report_path.read_text(encoding="utf-8")

    status_path = match_out / "pipeline_status.json"
    assert status_path.is_file()
    status = json.loads(status_path.read_text(encoding="utf-8"))
    assert status.get("ok") is True
