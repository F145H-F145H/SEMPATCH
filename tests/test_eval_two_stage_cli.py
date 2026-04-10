"""eval_two_stage.py CLI：体积守卫与冒烟 fixture。"""

import json
import os
import subprocess
import sys

import pytest


def _project_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def test_eval_two_stage_smoke_fixture_runs() -> None:
    root = _project_root()
    script = os.path.join(root, "scripts", "eval_two_stage.py")
    data_dir = os.path.join(root, "benchmarks", "smoke", "two_stage")
    py = sys.executable
    r = subprocess.run(
        [
            py,
            script,
            "--data-dir",
            data_dir,
            "--max-queries",
            "1",
            "-k",
            "1",
        ],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert r.returncode == 0, r.stderr + r.stdout
    assert "Recall@K" in r.stdout


def test_eval_two_stage_rejects_huge_files(tmp_path) -> None:
    """任意输入 JSON 超过 --max-input-bytes 时立即退出，不 json.load。"""
    root = _project_root()
    script = os.path.join(root, "scripts", "eval_two_stage.py")
    d = tmp_path / "huge_guard"
    d.mkdir()
    tiny_mm = {
        "graph": {"num_nodes": 1, "edge_index": [[], []], "node_features": [[]]},
        "sequence": {
            "pcode_tokens": ["A"],
            "jump_mask": [0],
            "seq_len": 1,
        },
    }
    (d / "ground_truth.json").write_text(json.dumps({"q": ["a"]}), encoding="utf-8")
    (d / "query_features.json").write_text(json.dumps({"q": tiny_mm}), encoding="utf-8")
    (d / "library_features.json").write_text(
        json.dumps({"a": tiny_mm, "b": tiny_mm}), encoding="utf-8"
    )
    (d / "library_safe_embeddings.json").write_text(
        json.dumps({"functions": [{"function_id": "a", "vector": [0.1] * 128}]}),
        encoding="utf-8",
    )
    # 垫大 ground_truth 文件（无需合法 JSON 结构即可被 getsize 拦下）
    p_gt = d / "ground_truth.json"
    p_gt.write_bytes(b"{" + b'"q":[]' + b" " * 600 + b"}")

    py = sys.executable
    r = subprocess.run(
        [py, script, "--data-dir", str(d), "--max-input-bytes", "256", "-k", "1"],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert r.returncode == 2
    assert "过大" in r.stderr or "OOM" in r.stderr


def test_eval_two_stage_outputs_diagnostics(tmp_path) -> None:
    """输出 JSON 包含 metrics 与 diagnostics 字段，终端打印错误分析面板。"""
    root = _project_root()
    script = os.path.join(root, "scripts", "eval_two_stage.py")
    data_dir = os.path.join(root, "benchmarks", "smoke", "two_stage")
    out_file = str(tmp_path / "diag.json")
    py = sys.executable
    r = subprocess.run(
        [
            py,
            script,
            "--data-dir",
            data_dir,
            "--max-queries",
            "1",
            "-k",
            "1",
            "--output",
            out_file,
        ],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert r.returncode == 0, r.stderr + r.stdout

    # 终端输出包含错误分析面板
    assert "coarse_hit_rate" in r.stdout
    assert "fallback_rate" in r.stdout
    assert "tied_top_rate" in r.stdout

    # 输出 JSON 包含 diagnostics 字段
    data = json.loads(open(out_file, encoding="utf-8").read())
    assert "metrics" in data
    assert "diagnostics" in data
    diag = data["diagnostics"]
    assert "coarse_hit_rate" in diag
    assert "fallback_rate" in diag
    assert "tied_top_rate" in diag
    assert "total_queries" in diag
    assert diag["total_queries"] == 1
    assert 0.0 <= diag["coarse_hit_rate"] <= 1.0
    assert 0.0 <= diag["fallback_rate"] <= 1.0
    assert 0.0 <= diag["tied_top_rate"] <= 1.0
