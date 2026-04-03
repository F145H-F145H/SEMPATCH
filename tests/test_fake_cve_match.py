"""人造 FAKE-CVE-* JSON 漏洞库 + 产品匹配验收（无 Ghidra、不调用 scripts/* 侧链脚本）。"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

from cli.cve_match import CveMatchOptions, run_cve_match_pipeline

pytestmark = pytest.mark.fake_cve


def _root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _fake_cve_dir() -> str:
    return os.path.join(_root(), "tests", "fixtures", "fake_cve")


def _collect_cves_from_matches(data: dict) -> set[str]:
    out: set[str] = set()
    for q in data.get("queries") or []:
        for c in q.get("candidates") or []:
            for x in c.get("cve") or []:
                if isinstance(x, str):
                    out.add(x)
    return out


def test_fake_cve_library_pipeline(tmp_path) -> None:
    """编造库 JSON → run_cve_match_pipeline；报告须带出库内 FAKE-CVE-*。"""
    fx = _fake_cve_dir()
    out = tmp_path / "pipeline_out"
    opts = CveMatchOptions(
        query_features=os.path.join(fx, "query_features.json"),
        two_stage_dir=fx,
        library_features=os.path.join(fx, "library_features.json"),
        library_emb=os.path.join(fx, "library_embeddings.json"),
        output_dir=str(out),
        coarse_k=10,
        top_k=5,
        max_queries=1,
        cpu=True,
        verbose=False,
    )
    assert run_cve_match_pipeline(opts) == 0
    status = json.loads((out / "pipeline_status.json").read_text(encoding="utf-8"))
    assert status.get("ok") is True
    doc = json.loads((out / "matches.json").read_text(encoding="utf-8"))
    cves = _collect_cves_from_matches(doc)
    assert "FAKE-CVE-0001" in cves
    report = (out / "report.md").read_text(encoding="utf-8")
    assert "FAKE-CVE-0001" in report


@pytest.mark.skipif(not os.path.isfile(os.path.join(_root(), "sempatch.py")), reason="no sempatch.py")
def test_fake_cve_sempatch_match_cli(tmp_path) -> None:
    """产品入口：sempatch.py match（不经过 scripts/demo_cve_match）。"""
    root = _root()
    fx = _fake_cve_dir()
    out = tmp_path / "cli_out"
    r = subprocess.run(
        [
            sys.executable,
            os.path.join(root, "sempatch.py"),
            "match",
            "--query-features",
            os.path.join(fx, "query_features.json"),
            "--library-features",
            os.path.join(fx, "library_features.json"),
            "--library-emb",
            os.path.join(fx, "library_embeddings.json"),
            "--output-dir",
            str(out),
            "--coarse-k",
            "10",
            "--top-k",
            "5",
            "--max-queries",
            "1",
            "--cpu",
        ],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert r.returncode == 0, r.stderr + r.stdout
    doc = json.loads((out / "matches.json").read_text(encoding="utf-8"))
    assert "FAKE-CVE-0001" in _collect_cves_from_matches(doc)
