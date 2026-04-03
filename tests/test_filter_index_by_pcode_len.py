"""filter_index_by_pcode_len 脚本单元测试。使用 mock 避免依赖 Ghidra。"""
import copy
import importlib.util
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def _load_filter_module():
    spec = importlib.util.spec_from_file_location(
        "filter_script",
        PROJECT_ROOT / "scripts" / "sidechain" / "filter_index_by_pcode_len.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _make_multimodal(pcode_len: int, num_nodes: int = 2):
    tokens = ["COPY", "INT_ADD"] * (pcode_len // 2 + 1)
    tokens = tokens[:pcode_len]
    n = max(1, int(num_nodes))
    return {
        "graph": {
            "num_nodes": n,
            "edge_index": [[0], [1]] if n > 1 else [[], []],
            "node_list": [f"bb_{i}" for i in range(n)],
            "node_features": [[] for _ in range(n)],
        },
        "sequence": {"pcode_tokens": tokens, "jump_mask": [0] * len(tokens), "seq_len": len(tokens)},
    }


def _make_lsir_funcs(entries):
    return {"functions": [{"entry": e, "basic_blocks": []} for e in entries]}


@patch("utils.ghidra_runner.peek_binary_cache")
@patch("utils.ghidra_runner.run_ghidra_analysis")
@patch("utils.feature_extractors.extract_multimodal_from_lsir_raw")
def test_filter_index_keeps_long_pcode(extract_mock, ghidra_mock, peek_mock):
    mod = _load_filter_module()
    peek_mock.return_value = None
    ghidra_mock.return_value = _make_lsir_funcs(["0x401000"])
    extract_mock.return_value = _make_multimodal(20)

    index_items = [
        {"binary": "data/binkit_subset/test.elf", "functions": [{"name": "main", "entry": "0x401000"}]},
    ]
    with tempfile.TemporaryDirectory() as tmp:
        result = mod._filter_index(index_items, str(PROJECT_ROOT), tmp, min_pcode_len=16, workers=0)
    assert len(result) == 1
    assert len(result[0]["functions"]) == 1
    assert result[0]["functions"][0]["name"] == "main"
    extract_mock.assert_called_once()


@patch("utils.ghidra_runner.peek_binary_cache")
@patch("utils.ghidra_runner.run_ghidra_analysis")
@patch("utils.feature_extractors.extract_multimodal_from_lsir_raw")
def test_filter_index_drops_short_pcode(extract_mock, ghidra_mock, peek_mock):
    mod = _load_filter_module()
    peek_mock.return_value = None
    ghidra_mock.return_value = _make_lsir_funcs(["0x400000"])
    extract_mock.return_value = _make_multimodal(8)

    index_items = [
        {"binary": "data/binkit_subset/short.elf", "functions": [{"name": "tiny", "entry": "0x400000"}]},
    ]
    with tempfile.TemporaryDirectory() as tmp:
        result = mod._filter_index(index_items, str(PROJECT_ROOT), tmp, min_pcode_len=16, workers=0)
    assert len(result) == 0
    extract_mock.assert_called_once()


@patch("utils.ghidra_runner.peek_binary_cache")
@patch("utils.ghidra_runner.run_ghidra_analysis")
@patch("utils.feature_extractors.extract_multimodal_from_lsir_raw")
def test_filter_index_drops_low_basic_blocks(extract_mock, ghidra_mock, peek_mock):
    mod = _load_filter_module()
    peek_mock.return_value = None
    ghidra_mock.return_value = _make_lsir_funcs(["0x500000"])
    extract_mock.return_value = _make_multimodal(20, num_nodes=1)

    index_items = [
        {"binary": "data/binkit_subset/smallcfg.elf", "functions": [{"name": "f", "entry": "0x500000"}]},
    ]
    with tempfile.TemporaryDirectory() as tmp:
        result = mod._filter_index(
            index_items,
            str(PROJECT_ROOT),
            tmp,
            min_pcode_len=16,
            min_basic_blocks=3,
            workers=0,
        )
    assert len(result) == 0


@patch("utils.ghidra_runner.peek_binary_cache")
@patch("utils.ghidra_runner.run_ghidra_analysis")
@patch("utils.feature_extractors.extract_multimodal_from_lsir_raw")
def test_filter_index_collects_kept_features(extract_mock, ghidra_mock, peek_mock):
    mod = _load_filter_module()
    peek_mock.return_value = None
    ghidra_mock.return_value = _make_lsir_funcs(["0x1000", "0x2000"])

    def side_effect(_funcs, entry):
        return _make_multimodal(20 if entry == "0x1000" else 4)

    extract_mock.side_effect = side_effect
    kept_features = {}
    index_items = [
        {
            "binary": "rel/path.elf",
            "functions": [{"name": "f1", "entry": "0x1000"}, {"name": "f2", "entry": "0x2000"}],
        },
    ]
    with tempfile.TemporaryDirectory() as tmp:
        result = mod._filter_index(
            index_items,
            str(PROJECT_ROOT),
            tmp,
            min_pcode_len=16,
            workers=0,
            kept_features=kept_features,
        )

    assert len(result) == 1
    assert len(result[0]["functions"]) == 1
    assert "rel/path.elf|0x1000" in kept_features
    assert kept_features["rel/path.elf|0x1000"]["sequence"]["seq_len"] == 20
    assert "rel/path.elf|0x2000" not in kept_features


def test_filter_index_empty_input_returns_empty():
    mod = _load_filter_module()
    with tempfile.TemporaryDirectory() as tmp:
        result = mod._filter_index([], str(PROJECT_ROOT), tmp, min_pcode_len=16, workers=0)
    assert result == []


@patch("utils.ghidra_runner.peek_binary_cache")
@patch("utils.ghidra_runner.run_ghidra_analysis")
@patch("utils.feature_extractors.extract_multimodal_from_lsir_raw")
def test_filter_index_resume_from_checkpoint(extract_mock, ghidra_mock, peek_mock):
    mod = _load_filter_module()
    peek_mock.return_value = None

    def ghidra_side_effect(binary_path, **_kwargs):
        if binary_path.endswith("a.elf"):
            return _make_lsir_funcs(["0x1000"])
        return _make_lsir_funcs(["0x2000"])

    ghidra_mock.side_effect = ghidra_side_effect
    extract_mock.return_value = _make_multimodal(20)
    index_items = [
        {"binary": "data/a.elf", "functions": [{"name": "fa", "entry": "0x1000"}]},
        {"binary": "data/b.elf", "functions": [{"name": "fb", "entry": "0x2000"}]},
    ]
    staged_lines = []
    saved_state = {}

    def sink(fid, _mm):
        staged_lines.append(fid)

    def stop_after_first_binary(state):
        saved_state.update(copy.deepcopy(state))
        raise RuntimeError("stop")

    with tempfile.TemporaryDirectory() as tmp:
        try:
            mod._filter_index(
                index_items,
                str(PROJECT_ROOT),
                tmp,
                min_pcode_len=16,
                workers=0,
                feature_sink=sink,
                on_binary_done=stop_after_first_binary,
            )
        except RuntimeError as e:
            assert str(e) == "stop"

        assert len(staged_lines) == 1
        assert len(saved_state.get("completed_binaries", [])) == 1

        result = mod._filter_index(
            index_items,
            str(PROJECT_ROOT),
            tmp,
            min_pcode_len=16,
            workers=0,
            feature_sink=sink,
            resume_state=saved_state,
        )

    assert len(result) == 2
    assert len(staged_lines) == 2
    assert ghidra_mock.call_count == 2
