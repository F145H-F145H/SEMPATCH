"""utils.precomputed_multimodal_io 单元测试。"""

import json
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from utils.precomputed_multimodal_io import (  # noqa: E402
    _extract_function_id_from_jsonl_line_bytes,
    build_jsonl_sidecar_lazy_index,
    is_jsonl_sidecar_path,
    load_precomputed_multimodal_map,
)


def _mm(n: int):
    return {
        "graph": {"num_nodes": 1, "edge_index": [[], []], "node_list": ["b"], "node_features": [[]]},
        "sequence": {"pcode_tokens": ["C"] * n, "jump_mask": [0] * n, "seq_len": n},
    }


def test_jsonl_load_stops_when_needed_satisfied():
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "f.jsonl"
        with open(p, "w", encoding="utf-8") as f:
            f.write(json.dumps({"function_id": "a|0x1", "multimodal": _mm(2)}, ensure_ascii=False) + "\n")
            f.write(json.dumps({"function_id": "b|0x2", "multimodal": _mm(3)}, ensure_ascii=False) + "\n")
        out = load_precomputed_multimodal_map(str(p), needed_ids={"a|0x1"})
    assert list(out.keys()) == ["a|0x1"]
    assert out["a|0x1"]["sequence"]["seq_len"] == 2


def test_is_jsonl_sidecar_path_by_extension():
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "x.jsonl"
        p.write_text("", encoding="utf-8")
        assert is_jsonl_sidecar_path(str(p)) is True


def test_lazy_index_bulk_get_order_independent(tmp_path):
    """bulk_get 按偏移顺序读，结果与逐条 get 一致。"""
    p = tmp_path / "f.jsonl"
    lines = [
        {"function_id": "b|0x2", "multimodal": _mm(3)},
        {"function_id": "a|0x1", "multimodal": _mm(2)},
        {"function_id": "c|0x3", "multimodal": _mm(4)},
    ]
    with open(p, "w", encoding="utf-8") as f:
        for obj in lines:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    lazy = build_jsonl_sidecar_lazy_index(str(p), {"a|0x1", "b|0x2", "c|0x3"})
    bulk = lazy.bulk_get(["c|0x3", "a|0x1", "b|0x2"])
    assert bulk["a|0x1"]["sequence"]["seq_len"] == 2
    assert bulk["b|0x2"]["sequence"]["seq_len"] == 3
    assert bulk["c|0x3"]["sequence"]["seq_len"] == 4
    assert lazy.get("a|0x1") == bulk["a|0x1"]


def test_lazy_index_loads_on_get(tmp_path):
    p = tmp_path / "f.jsonl"
    with open(p, "w", encoding="utf-8") as f:
        f.write(json.dumps({"function_id": "a|0x1", "multimodal": _mm(2)}, ensure_ascii=False) + "\n")
        f.write(json.dumps({"function_id": "b|0x2", "multimodal": _mm(3)}, ensure_ascii=False) + "\n")
    lazy = build_jsonl_sidecar_lazy_index(str(p), {"a|0x1"})
    assert len(lazy) == 1
    mm = lazy.get("a|0x1")
    assert mm is not None
    assert mm["sequence"]["seq_len"] == 2
    assert lazy.get("b|0x2") is None


def test_empty_needed_returns_empty_without_reading(tmp_path):
    p = tmp_path / "big.jsonl"
    p.write_text("should-not-parse\n", encoding="utf-8")
    assert load_precomputed_multimodal_map(str(p), needed_ids=set()) == {}


def test_extract_function_id_from_jsonl_line_bytes_fast_path():
    line = json.dumps(
        {"function_id": 'a|"quoted"|0x1', "multimodal": _mm(2), "tail": 1},
        ensure_ascii=False,
    ).encode("utf-8")
    fid = _extract_function_id_from_jsonl_line_bytes(line)
    assert fid == 'a|"quoted"|0x1'


def test_build_lazy_index_fallback_when_function_id_not_first(tmp_path):
    p = tmp_path / "f.jsonl"
    with open(p, "w", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {"multimodal": _mm(2), "function_id": "a|0x1"},
                ensure_ascii=False,
            )
            + "\n"
        )
    lazy = build_jsonl_sidecar_lazy_index(str(p), {"a|0x1"})
    assert len(lazy) == 1
    assert lazy.get("a|0x1") is not None
