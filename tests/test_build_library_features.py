"""build_library_features 脚本单元测试。"""

import importlib.util
import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "build_lib_feat_script",
        PROJECT_ROOT / "scripts" / "sidechain" / "build_library_features.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _mock_mm(seq_len: int):
    return {
        "graph": {
            "num_nodes": 1,
            "edge_index": [[], []],
            "node_list": ["bb_0"],
            "node_features": [[]],
        },
        "sequence": {
            "pcode_tokens": ["COPY"] * seq_len,
            "jump_mask": [0] * seq_len,
            "seq_len": seq_len,
        },
    }


@patch("utils.ghidra_runner.peek_binary_cache")
@patch("utils.feature_extractors.extract_multimodal_from_lsir_raw")
def test_process_single_binary_uses_precomputed_first(extract_mock, peek_mock):
    mod = _load_module()
    peek_mock.return_value = {"functions": [{"entry": "0x1000", "basic_blocks": []}]}
    extract_mock.side_effect = AssertionError("命中预计算时不应走提取")

    item = {"binary": "rel/test.elf", "functions": [{"name": "f", "entry": "0x1000"}]}
    precomputed = {"rel/test.elf|0x1000": _mock_mm(9)}
    args = (1, item, str(PROJECT_ROOT), tempfile.gettempdir(), "lib", 1, precomputed, False, None)
    _idx, _bin_rel, features, hits, fallbacks = mod._process_single_binary(args)

    assert features["rel/test.elf|0x1000"]["sequence"]["seq_len"] == 9
    assert hits == 1
    assert fallbacks == 0


@patch("utils.ghidra_runner.peek_binary_cache")
@patch("utils.feature_extractors.extract_multimodal_from_lsir_raw")
def test_process_single_binary_partial_precomputed_fallback(extract_mock, peek_mock):
    mod = _load_module()
    peek_mock.return_value = {
        "functions": [
            {"entry": "0x1000", "basic_blocks": []},
            {"entry": "0x2000", "basic_blocks": []},
        ]
    }
    extract_mock.return_value = _mock_mm(7)

    item = {
        "binary": "rel/test.elf",
        "functions": [
            {"name": "f1", "entry": "0x1000"},
            {"name": "f2", "entry": "0x2000"},
        ],
    }
    precomputed = {"rel/test.elf|0x1000": _mock_mm(11)}
    args = (1, item, str(PROJECT_ROOT), tempfile.gettempdir(), "lib", 1, precomputed, False, None)
    _idx, _bin_rel, features, hits, fallbacks = mod._process_single_binary(args)

    assert hits == 1
    assert fallbacks == 1
    assert features["rel/test.elf|0x1000"]["sequence"]["seq_len"] == 11
    assert features["rel/test.elf|0x2000"]["sequence"]["seq_len"] == 7
    assert extract_mock.call_count == 1


@patch("utils.ghidra_runner.peek_binary_cache")
@patch("utils.feature_extractors.extract_multimodal_from_lsir_raw")
def test_process_single_binary_skips_lsir_when_fully_precomputed(extract_mock, peek_mock):
    mod = _load_module()
    item = {
        "binary": "rel/x.elf",
        "functions": [{"name": "f", "entry": "0x1000"}, {"name": "g", "entry": "0x2000"}],
    }
    precomputed = {
        "rel/x.elf|0x1000": _mock_mm(3),
        "rel/x.elf|0x2000": _mock_mm(4),
    }
    args = (1, item, str(PROJECT_ROOT), tempfile.gettempdir(), "lib", 1, precomputed, False, None)
    _idx, _bin_rel, features, hits, fallbacks = mod._process_single_binary(args)

    assert hits == 2 and fallbacks == 0
    peek_mock.assert_not_called()
    extract_mock.assert_not_called()


@patch("utils.ghidra_runner.peek_binary_cache")
@patch("utils.feature_extractors.extract_multimodal_from_lsir_raw")
def test_process_index_parallel_sidecar_stream_queue(extract_mock, peek_mock, tmp_path):
    mod = _load_module()
    from utils.precomputed_multimodal_io import build_jsonl_sidecar_lazy_index

    sidecar_path = tmp_path / "sidecar.jsonl"
    sidecar_rows = [
        {"function_id": "rel/a.elf|0x1000", "multimodal": _mock_mm(3)},
        {"function_id": "rel/a.elf|0x2000", "multimodal": _mock_mm(4)},
        {"function_id": "rel/b.elf|0x3000", "multimodal": _mock_mm(5)},
        {"function_id": "rel/b.elf|0x4000", "multimodal": _mock_mm(6)},
    ]
    with open(sidecar_path, "w", encoding="utf-8") as f:
        for row in sidecar_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    lazy = build_jsonl_sidecar_lazy_index(
        str(sidecar_path),
        {
            "rel/a.elf|0x1000",
            "rel/a.elf|0x2000",
            "rel/b.elf|0x3000",
            "rel/b.elf|0x4000",
        },
    )

    index_items = [
        {
            "binary": "rel/a.elf",
            "functions": [{"name": "a1", "entry": "0x1000"}, {"name": "a2", "entry": "0x2000"}],
        },
        {
            "binary": "rel/b.elf",
            "functions": [{"name": "b1", "entry": "0x3000"}, {"name": "b2", "entry": "0x4000"}],
        },
    ]
    out = tmp_path / "library_features.json"
    with open(out, "w", encoding="utf-8") as fp:
        with mod._JsonObjectStreamWriter(fp) as writer:
            total, hits, fallbacks, features = mod._process_index(
                index_items,
                str(PROJECT_ROOT),
                str(tmp_path),
                prefix="lib",
                workers=2,
                precomputed_multimodal=lazy,
                gc_after_each_binary=False,
                lsir_sem=None,
                stream_writer=writer,
            )

    assert total == 4
    assert hits == 4
    assert fallbacks == 0
    assert features is None
    with open(out, encoding="utf-8") as f:
        payload = json.load(f)
    assert payload["rel/a.elf|0x1000"]["sequence"]["seq_len"] == 3
    assert payload["rel/b.elf|0x4000"]["sequence"]["seq_len"] == 6
    peek_mock.assert_not_called()
    extract_mock.assert_not_called()


def test_load_precomputed_via_utils_filters_needed_and_invalid_keys():
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    from utils.precomputed_multimodal_io import load_precomputed_multimodal_map

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "pre.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"a|0x1": _mock_mm(3), "bad": [], "b|0x2": _mock_mm(4)}, f)
        loaded = load_precomputed_multimodal_map(str(path), needed_ids={"a|0x1"})
    assert "a|0x1" in loaded
    assert "b|0x2" not in loaded
    assert "bad" not in loaded
