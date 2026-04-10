"""测试 config_validator 捕获缺失路径。"""

import pytest


def test_validate_config_catches_missing(monkeypatch):
    """monkeypatch 不存在的路径，validator 返回问题列表。"""
    import config_validator

    monkeypatch.setattr("config.GHIDRA_HOME", "/nonexistent/ghidra")
    monkeypatch.setattr("config.ANALYZE_HEADLESS", "/nonexistent/ghidra/support/analyzeHeadless")
    monkeypatch.setattr("config.DATA_DIR", "/nonexistent/data")
    monkeypatch.setattr("shutil.which", lambda _: None)  # binwalk missing

    problems = config_validator.validate_config()
    assert len(problems) >= 3
    assert any("GHIDRA_HOME" in p for p in problems)
    assert any("analyzeHeadless" in p for p in problems)
    assert any("DATA_DIR" in p for p in problems)
    assert any("binwalk" in p for p in problems)


def test_validate_config_clean(monkeypatch):
    """路径都存在时返回空列表（或仅 binwalk）。"""
    import os
    import tempfile

    import config_validator

    with tempfile.TemporaryDirectory() as tmp:
        ghidra_home = tmp
        analyze = os.path.join(tmp, "analyzeHeadless")
        with open(analyze, "w") as f:
            f.write("")
        data_dir = tmp

        monkeypatch.setattr("config.GHIDRA_HOME", ghidra_home)
        monkeypatch.setattr("config.ANALYZE_HEADLESS", analyze)
        monkeypatch.setattr("config.DATA_DIR", data_dir)
        monkeypatch.setattr("shutil.which", lambda cmd: "/usr/bin/" + cmd)

        problems = config_validator.validate_config()
        assert problems == []
