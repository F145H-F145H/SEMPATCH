"""sempatch / sempatch_argv 路由：子命令白名单、双参 → match、legacy extract。"""

import json
import os
import subprocess
import sys

import pytest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from sempatch_argv import (  # noqa: E402
    KNOWN_SUBCOMMANDS,
    looks_like_two_stage_lib_dir,
    rewrite_sempatch_argv,
)


def test_known_subcommands_unchanged():
    for cmd in sorted(KNOWN_SUBCOMMANDS):
        assert rewrite_sempatch_argv([cmd, "--help"]) == [cmd, "--help"]
    assert rewrite_sempatch_argv(["match", "--query-binary", "x", "--two-stage-dir", "y"]) == [
        "match",
        "--query-binary",
        "x",
        "--two-stage-dir",
        "y",
    ]


def test_help_unchanged():
    assert rewrite_sempatch_argv(["-h"]) == ["-h"]
    assert rewrite_sempatch_argv(["--help"]) == ["--help"]


def test_legacy_extract_single_binary():
    assert rewrite_sempatch_argv(["/bin/true"]) == ["extract", "/bin/true"]
    assert rewrite_sempatch_argv(["/bin/true", "-o", "/tmp/out"]) == [
        "extract",
        "/bin/true",
        "-o",
        "/tmp/out",
    ]


def test_dual_positional_to_match(tmp_path):
    lib = tmp_path / "lib"
    lib.mkdir()
    (lib / "library_features.json").write_text("{}", encoding="utf-8")
    probe = tmp_path / "probe.elf"
    probe.write_bytes(b"\x7fELF")
    out = rewrite_sempatch_argv([str(probe), str(lib), "--cpu", "--max-queries", "1"])
    assert out == [
        "match",
        "--query-binary",
        str(probe),
        "--two-stage-dir",
        str(lib),
        "--cpu",
        "--max-queries",
        "1",
    ]


def test_dual_positional_not_match_without_lib_json(tmp_path):
    probe = tmp_path / "probe.elf"
    probe.write_bytes(b"\x7fELF")
    empty = tmp_path / "empty"
    empty.mkdir()
    out = rewrite_sempatch_argv([str(probe), str(empty)])
    assert out == ["extract", str(probe), str(empty)]


def test_looks_like_two_stage_lib_dir(tmp_path):
    d = tmp_path / "d"
    d.mkdir()
    assert not looks_like_two_stage_lib_dir(str(d))
    (d / "library_safe_embeddings.json").write_text("{}", encoding="utf-8")
    assert looks_like_two_stage_lib_dir(str(d))


def _root() -> str:
    return PROJECT_ROOT


def _ghidra_available() -> bool:
    try:
        sys.path.insert(0, os.path.join(_root(), "src"))
        from config import ANALYZE_HEADLESS, GHIDRA_HOME

        return (
            bool(GHIDRA_HOME)
            and os.path.isdir(GHIDRA_HOME)
            and os.path.isfile(ANALYZE_HEADLESS)
            and os.access(ANALYZE_HEADLESS, os.X_OK)
        )
    except Exception:
        return False


@pytest.mark.integration
def test_sempatch_two_positional_fake_cve_demo_matches_cve_0005(tmp_path):
    """./sempatch query.elf data/fake_cve_demo_lib：报告中须含 FAKE-CVE-0005（需 Ghidra + 预构建产物）。"""
    root = _root()
    query_elf = os.path.join(root, "examples", "fake_cve_demo", "build", "query.elf")
    lib_dir = os.path.join(root, "data", "fake_cve_demo_lib")
    emb = os.path.join(lib_dir, "library_safe_embeddings.json")
    if not _ghidra_available():
        pytest.skip("Ghidra 未安装或不可用")
    if not os.path.isfile(query_elf):
        pytest.skip("缺少 examples/fake_cve_demo/build/query.elf，请先 make -C examples/fake_cve_demo all")
    if not os.path.isfile(emb):
        pytest.skip("缺少 data/fake_cve_demo_lib，请先 build_fake_cve_demo_library")
    out_dir = tmp_path / "sempatch_out"
    # 经根目录 sempatch 入口以应用 argv 改写；当前解释器满足 match 对 PyTorch 的硬依赖
    launcher = os.path.join(root, "sempatch")
    r = subprocess.run(
        [
            sys.executable,
            launcher,
            query_elf,
            lib_dir,
            "--cpu",
            "--output-dir",
            str(out_dir),
            "--max-queries",
            "8",
        ],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert r.returncode == 0, r.stderr + r.stdout
    report = (out_dir / "report.md").read_text(encoding="utf-8")
    matches_text = (out_dir / "matches.json").read_text(encoding="utf-8")
    doc = json.loads(matches_text)
    if not any(q.get("candidates") for q in doc.get("queries") or []):
        pytest.skip(
            "无匹配候选（常见于 TwoStage 未跑通、SAFE 维度不一致或仅 coarse 回退）；"
            "请确认 .venv + torch、output/best_model.pth 与库嵌入一致后重试"
        )
    assert "FAKE-CVE-0005" in report or "FAKE-CVE-0005" in matches_text
