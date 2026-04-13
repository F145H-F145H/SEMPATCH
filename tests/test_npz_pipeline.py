"""NPZ 预计算流水线单元测试。

覆盖: multimodal_to_arrays, build_precomputed_npz, PrecomputedTensorDataset,
      collate_multimodal_precomputed, collate_safe_precomputed。
"""

import json
import os
import sys
import tempfile

import numpy as np
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))


# ---------------------------------------------------------------------------
# 合成 multimodal dict 工厂
# ---------------------------------------------------------------------------

def _make_mm_dict(n_tokens=32, n_nodes=8, n_edges=12, n_dfg_nodes=4, n_dfg_edges=6):
    """构造一个合法的 multimodal 特征字典。"""
    return {
        "sequence": {
            "pcode_tokens": [f"INT_ADD_{i % 5}" for i in range(n_tokens)],
            "jump_mask": [0 if i % 4 != 0 else 1 for i in range(n_tokens)],
        },
        "graph": {
            "node_features": [
                {"pcode_opcodes": [f"COPY_{i % 3}"]} for i in range(n_nodes)
            ],
            "edge_index": [
                [i % n_nodes for i in range(n_edges)],
                [(i + 1) % n_nodes for i in range(n_edges)],
            ],
        },
        "dfg": {
            "node_features": list(range(n_dfg_nodes)),
            "edge_index": [
                [i % n_dfg_nodes for i in range(n_dfg_edges)],
                [(i + 1) % n_dfg_nodes for i in range(n_dfg_edges)],
            ],
        },
    }


def _make_vocab():
    """构造最小 vocab。"""
    vocab = {"[PAD]": 0, "[UNK]": 1}
    for i in range(5):
        vocab[f"INT_ADD_{i}"] = 10 + i
    for i in range(3):
        vocab[f"COPY_{i}"] = 20 + i
    return vocab


# ---------------------------------------------------------------------------
# Tests: multimodal_to_arrays
# ---------------------------------------------------------------------------

class TestMultimodalToArrays:
    def test_output_keys(self):
        from utils.npz_features import multimodal_to_arrays

        mm = _make_mm_dict()
        vocab = _make_vocab()
        result = multimodal_to_arrays(mm, vocab, max_seq_len=64, max_graph_nodes=16, max_dfg_nodes=8)

        expected_keys = {
            "token_ids", "jump_mask", "node_ids",
            "edge_src", "edge_dst", "graph_n_edges",
            "dfg_node_ids", "dfg_edge_src", "dfg_edge_dst", "dfg_n_edges",
            "seq_len", "node_count",
        }
        assert set(result.keys()) == expected_keys

    def test_array_shapes_and_dtypes(self):
        from utils.npz_features import multimodal_to_arrays

        mm = _make_mm_dict(n_tokens=32, n_nodes=8, n_dfg_nodes=4)
        vocab = _make_vocab()
        result = multimodal_to_arrays(
            mm, vocab, max_seq_len=64, max_graph_nodes=16,
            max_dfg_nodes=8, max_edges=32, max_dfg_edges=16,
        )

        assert result["token_ids"].shape == (64,)
        assert result["token_ids"].dtype == np.int16
        assert result["jump_mask"].shape == (64,)
        assert result["jump_mask"].dtype == np.int8
        assert result["node_ids"].shape == (16,)
        assert result["edge_src"].shape == (32,)
        assert result["edge_dst"].shape == (32,)
        assert result["edge_src"].dtype == np.int32
        assert result["dfg_node_ids"].shape == (8,)
        assert result["dfg_edge_src"].shape == (16,)
        assert result["dfg_edge_dst"].shape == (16,)

    def test_seq_len_and_node_count(self):
        from utils.npz_features import multimodal_to_arrays

        mm = _make_mm_dict(n_tokens=20, n_nodes=6)
        vocab = _make_vocab()
        result = multimodal_to_arrays(mm, vocab, max_seq_len=64, max_graph_nodes=16)

        assert int(result["seq_len"]) == 20
        assert int(result["node_count"]) == 6

    def test_edge_count(self):
        from utils.npz_features import multimodal_to_arrays

        mm = _make_mm_dict(n_nodes=8, n_edges=10)
        vocab = _make_vocab()
        result = multimodal_to_arrays(mm, vocab, max_graph_nodes=16, max_edges=32)

        assert int(result["graph_n_edges"]) == 10
        # 实际的 src/dst 值应有效（非零）在前 10 个位置
        assert np.any(result["edge_src"][:10] != 0) or np.any(result["edge_dst"][:10] != 0)

    def test_token_clamping(self):
        from utils.npz_features import multimodal_to_arrays

        mm = _make_mm_dict(n_tokens=8)
        vocab = _make_vocab()
        result = multimodal_to_arrays(mm, vocab, max_seq_len=16, pcode_vocab_size=15)

        # 所有 token_id 应在 [0, pcode_vocab_size-1] 范围内
        assert np.all(result["token_ids"] < 15)
        assert np.all(result["token_ids"] >= 0)


# ---------------------------------------------------------------------------
# Tests: build_precomputed_npz
# ---------------------------------------------------------------------------

class TestBuildPrecomputedNpz:
    def test_roundtrip(self, tmp_path):
        from utils.npz_features import build_precomputed_npz

        vocab = _make_vocab()
        multimodals = [
            ("bin1|0x1000", _make_mm_dict(n_tokens=16, n_nodes=4)),
            ("bin1|0x2000", _make_mm_dict(n_tokens=24, n_nodes=6)),
            ("bin2|0x3000", _make_mm_dict(n_tokens=32, n_nodes=8)),
        ]
        out_path = str(tmp_path / "test_features.npz")
        fid_to_idx = build_precomputed_npz(
            multimodals, vocab, out_path,
            max_seq_len=64, max_graph_nodes=16, max_dfg_nodes=8,
        )

        assert os.path.isfile(out_path)
        assert len(fid_to_idx) == 3
        assert fid_to_idx["bin1|0x1000"] == 0
        assert fid_to_idx["bin2|0x3000"] == 2

        # 验证 npz 可加载且形状正确
        arrays = np.load(out_path)
        assert arrays["token_ids"].shape == (3, 64)
        assert arrays["node_ids"].shape == (3, 16)
        assert arrays["dfg_node_ids"].shape == (3, 8)

    def test_empty_raises(self):
        from utils.npz_features import build_precomputed_npz

        with pytest.raises(ValueError, match="empty"):
            build_precomputed_npz([], _make_vocab(), "/tmp/should_not_exist.npz")


# ---------------------------------------------------------------------------
# Tests: PrecomputedTensorDataset
# ---------------------------------------------------------------------------

@pytest.fixture
def npz_dataset(tmp_path):
    """创建一个最小的 npz + index + fid_map，返回 PrecomputedTensorDataset。"""
    from utils.npz_features import build_precomputed_npz

    vocab = _make_vocab()
    funcs = [
        ("data/test.elf|0x1000", _make_mm_dict(n_tokens=16, n_nodes=4, n_dfg_nodes=2)),
        ("data/test.elf|0x2000", _make_mm_dict(n_tokens=16, n_nodes=4, n_dfg_nodes=2)),
        ("data/test2.elf|0x3000", _make_mm_dict(n_tokens=16, n_nodes=4, n_dfg_nodes=2)),
        ("data/test2.elf|0x4000", _make_mm_dict(n_tokens=16, n_nodes=4, n_dfg_nodes=2)),
    ]
    npz_path = str(tmp_path / "features.npz")
    fid_to_idx = build_precomputed_npz(funcs, vocab, npz_path)

    fid_map_path = str(tmp_path / "fid_map.json")
    with open(fid_map_path, "w") as f:
        json.dump(fid_to_idx, f)

    index_data = [
        {
            "binary": "data/test.elf",
            "functions": [
                {"name": "main", "entry": "0x1000"},
                {"name": "helper", "entry": "0x2000"},
            ],
        },
        {
            "binary": "data/test2.elf",
            "functions": [
                {"name": "main", "entry": "0x3000"},
                {"name": "util", "entry": "0x4000"},
            ],
        },
    ]
    index_path = str(tmp_path / "index.json")
    with open(index_path, "w") as f:
        json.dump(index_data, f)

    from features.dataset import PrecomputedTensorDataset

    ds = PrecomputedTensorDataset(
        npz_path=npz_path,
        index_path=index_path,
        fid_map_path=fid_map_path,
        num_pairs=10,
        seed=42,
        fixed_pairs_per_epoch=True,
    )
    return ds


class TestPrecomputedTensorDataset:
    def test_len(self, npz_dataset):
        assert len(npz_dataset) == 10

    def test_getitem_tuple_length(self, npz_dataset):
        item = npz_dataset[0]
        # 12 per side + 1 label = 25
        assert len(item) == 25

    def test_getitem_array_shapes(self, npz_dataset):
        item = npz_dataset[0]
        # npz defaults: max_seq_len=512, max_graph_nodes=128, max_dfg_nodes=64
        assert item[0].shape == (512,)   # token_ids
        assert item[1].shape == (512,)   # jump_mask
        assert item[2].shape == (128,)   # node_ids

    def test_label_type(self, npz_dataset):
        item = npz_dataset[0]
        label = item[24]
        assert isinstance(label, float)
        assert label in (0.0, 1.0)

    def test_get_safe_token_arrays(self, npz_dataset):
        token_ids, jump_mask, seq_len = npz_dataset.get_safe_token_arrays(0)
        assert token_ids.shape == (512,)
        assert jump_mask.shape == (512,)
        assert isinstance(seq_len, int)

    def test_regenerate_epoch_pairs(self, npz_dataset):
        pairs_before = npz_dataset._epoch_pairs.copy()
        npz_dataset.regenerate_epoch_pairs()
        pairs_after = npz_dataset._epoch_pairs
        # 形状不变
        assert pairs_after.shape == pairs_before.shape
        # 内容可能不同（随机种子连续但不一定相同）
        assert pairs_after.dtype == np.int32


# ---------------------------------------------------------------------------
# Tests: collate functions
# ---------------------------------------------------------------------------

class TestCollateMultimodalPrecomputed:
    def test_output_structure(self, npz_dataset):
        from features.precomputed_collate import collate_multimodal_precomputed

        batch = [npz_dataset[i] for i in range(min(4, len(npz_dataset)))]
        result = collate_multimodal_precomputed(batch)

        assert result["valid"] is True
        assert "batch1" in result
        assert "batch2" in result
        assert "labels" in result
        assert len(result["batch1"]) == 7
        assert len(result["batch2"]) == 7
        assert result["labels"].dim() == 1
        assert result["labels"].shape[0] == len(batch)

    def test_batch1_tensor_shapes(self, npz_dataset):
        from features.precomputed_collate import collate_multimodal_precomputed

        B = 2
        batch = [npz_dataset[i] for i in range(B)]
        result = collate_multimodal_precomputed(batch)

        token_t, jump_t, node_t, edge_t, pad_mask, dfg_node_t, dfg_edge_t = result["batch1"]
        assert token_t.shape[0] == B
        assert jump_t.shape[0] == B
        assert node_t.shape[0] == B
        assert pad_mask.shape[0] == B
        assert edge_t.shape[0] == 2  # (2, total_edges)
        assert dfg_edge_t.shape[0] == 2


class TestCollateSafePrecomputed:
    def test_output_structure(self, npz_dataset):
        from features.precomputed_collate import collate_safe_precomputed, make_safe_precomputed_pairs

        pairs = make_safe_precomputed_pairs(npz_dataset, 4, 0.5, 42)
        assert len(pairs) == 4

        result = collate_safe_precomputed(pairs)
        assert result["valid"] is True
        assert "t1" in result
        assert "p1" in result
        assert "t2" in result
        assert "p2" in result
        assert "labels" in result
        assert result["t1"].shape[0] == 4
        assert result["labels"].shape[0] == 4


# ---------------------------------------------------------------------------
# Tests: jsonl_to_npz 集成
# ---------------------------------------------------------------------------

class TestJsonlToNpzIntegration:
    def test_end_to_end(self, tmp_path):
        """合成 JSONL → NPZ → Dataset 加载，验证完整流水线。"""
        from utils.npz_features import build_precomputed_npz
        from features.dataset import PrecomputedTensorDataset

        vocab = _make_vocab()

        # 1. 写入合成 JSONL
        jsonl_path = str(tmp_path / "test.training.jsonl")
        funcs = []
        with open(jsonl_path, "w") as f:
            for i in range(4):
                fid = f"data/test.elf|0x{1000 + i * 0x1000:x}"
                mm = _make_mm_dict(n_tokens=16, n_nodes=4, n_dfg_nodes=2)
                funcs.append((fid, mm))
                line = json.dumps({"function_id": fid, "multimodal": mm})
                f.write(line + "\n")

        # 2. 转换为 NPZ
        npz_path = str(tmp_path / "features.npz")
        fid_to_idx = build_precomputed_npz(funcs, vocab, npz_path)

        fid_map_path = str(tmp_path / "fid_map.json")
        with open(fid_map_path, "w") as f:
            json.dump(fid_to_idx, f)

        # 3. 构建 index
        index_data = [{
            "binary": "data/test.elf",
            "functions": [
                {"name": f"func_{i}", "entry": f"0x{1000 + i * 0x1000:x}"}
                for i in range(4)
            ],
        }]
        index_path = str(tmp_path / "index.json")
        with open(index_path, "w") as f:
            json.dump(index_data, f)

        # 4. 加载 Dataset 并验证
        ds = PrecomputedTensorDataset(
            npz_path=npz_path,
            index_path=index_path,
            fid_map_path=fid_map_path,
            num_pairs=8,
            seed=42,
            fixed_pairs_per_epoch=True,
        )
        assert len(ds) == 8
        item = ds[0]
        assert len(item) == 25
