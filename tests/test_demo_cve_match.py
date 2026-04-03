"""TwoStage CVE 元数据、候选格式化与 run_demo 冒烟（无 Ghidra）；不调用 scripts/*。"""

import json
import os
import sys

_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(_project_root, "src"))
from cli import two_stage_demo as _demo  # noqa: E402


def test_parse_query_binary_from_function_id():
    assert _demo.parse_query_binary_from_function_id("data/foo.elf|0x401000") == "data/foo.elf"
    assert _demo.parse_query_binary_from_function_id("nobar") == "nobar"


def test_filter_query_function_ids_by_entry():
    qids = ["b|0x00401176", "b|0x401000", "b|0x00401000"]
    assert _demo.filter_query_function_ids_by_entry(qids, None) == qids
    assert _demo.filter_query_function_ids_by_entry(qids, "0x401176") == ["b|0x00401176"]
    assert _demo.filter_query_function_ids_by_entry(qids, "401176") == ["b|0x00401176"]


def test_load_library_metadata_cve_lists(tmp_path):
    p = tmp_path / "emb.json"
    p.write_text(
        json.dumps(
            {
                "functions": [
                    {
                        "function_id": "a|0x1",
                        "name": "f1",
                        "vector": [0.1, 0.2],
                        "cve": "CVE-1",
                    },
                    {
                        "function_id": "a|0x2",
                        "name": "f2",
                        "vector": [0.2, 0.1],
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    meta = _demo.load_library_metadata(str(p))
    assert meta["a|0x1"]["cve"] == ["CVE-1"]
    assert meta["a|0x2"]["cve"] == []


def test_build_candidates_no_dedupe_same_name():
    ranked = [("id1", 1.0), ("id2", 0.9)]
    meta = {
        "id1": {"name": "dup", "cve": ["CVE-A"]},
        "id2": {"name": "dup", "cve": ["CVE-B"]},
    }
    c = _demo.build_candidates_for_ranked(ranked[:10], meta)
    assert len(c) == 2
    assert c[0]["candidate_name"] == c[1]["candidate_name"] == "dup"
    assert c[0]["cve"] == ["CVE-A"]
    assert c[1]["cve"] == ["CVE-B"]
    assert all("similarity" in x and "rank" in x for x in c)


def test_two_stage_run_demo_fake_cve_fixture(tmp_path):
    """与 tests/fixtures/fake_cve 对齐：直接调 run_demo（等价原 demo_cve_match 子进程）。"""
    root = _project_root
    fx = os.path.join(root, "tests", "fixtures", "fake_cve")
    out = tmp_path / "demo_out"
    _demo.run_demo(
        query_features_path=os.path.join(fx, "query_features.json"),
        library_emb=os.path.join(fx, "library_embeddings.json"),
        library_features=os.path.join(fx, "library_features.json"),
        output_dir=str(out),
        rerank_model_path=None,
        safe_model_path=None,
        coarse_k=10,
        top_k=5,
        max_queries=1,
        prefer_cuda=False,
        query_mode="file",
        query_binary=None,
        rerank_use_dfg=None,
    )
    matches_path = out / "matches.json"
    assert matches_path.is_file()
    data = json.loads(matches_path.read_text(encoding="utf-8"))
    assert "config" in data and "queries" in data
    assert data["config"]["git_rev"]
    q0 = data["queries"][0]
    assert q0["query_function_id"]
    assert q0["top_k"] == 5
    for c in q0["candidates"]:
        assert "cve" in c and isinstance(c["cve"], list)
        assert "candidate_function_id" in c
    report = out / "report.md"
    assert report.is_file()
    assert "FAKE-CVE" in report.read_text(encoding="utf-8")


def test_filter_ranked_unique_all_above_and_apply_top_k():
    out, meta = _demo.filter_ranked_by_policy([], "unique", 0.95, 1e-5)
    assert out == [] and meta["reject_reason"] == "no_candidates"

    out, meta = _demo.filter_ranked_by_policy([("a", 0.94), ("b", 0.5)], "unique", 0.95, 1e-5)
    assert out == [] and meta["reject_reason"] == "below_threshold"

    out, meta = _demo.filter_ranked_by_policy([("a", 0.96), ("b", 0.96)], "unique", 0.95, 1e-5)
    assert out == [] and meta["reject_reason"] == "tied_top"

    out, meta = _demo.filter_ranked_by_policy([("a", 0.96), ("b", 0.90)], "unique", 0.95, 1e-5)
    assert out == [("a", 0.96)] and meta["reject_reason"] is None

    out, meta = _demo.filter_ranked_by_policy(
        [("x", 0.97), ("y", 0.94), ("z", 0.96)], "all_above", 0.95, 1e-5
    )
    assert [t[0] for t in out] == ["x", "z"] and meta["reject_reason"] is None

    out, meta, st = _demo.apply_output_policy([("a", 1.0), ("b", 0.5)], "top_k", 1, 0.95, 1e-5)
    assert out == [("a", 1.0)] and meta["mode"] == "top_k"
    assert meta["reject_reason"] is None and st == _demo.MATCH_STATUS_OK

    out, meta, st = _demo.apply_output_policy([], "top_k", 10, 0.95, 1e-5)
    assert out == [] and st == _demo.MATCH_STATUS_NO_MATCH


def test_format_match_explanation_zh():
    fm = {
        "reject_reason": "below_threshold",
        "max_similarity": 0.94,
        "min_similarity": 0.95,
    }
    s = _demo.format_match_explanation_zh(fm, _demo.MATCH_STATUS_NO_MATCH)
    assert "0.940000" in s and "0.950000" in s


def test_two_stage_dup_names_two_fake_cves_in_topk(tmp_path):
    """库中两函数同名 dup、CVE 为 FAKE-CVE-0001 / FAKE-CVE-0002；top_k=2 两条候选均保留。"""
    root = _project_root
    fx = os.path.join(root, "tests", "fixtures", "fake_cve")
    out = tmp_path / "demo_out2"
    _demo.run_demo(
        query_features_path=os.path.join(fx, "query_features.json"),
        library_emb=os.path.join(fx, "library_embeddings.json"),
        library_features=os.path.join(fx, "library_features.json"),
        output_dir=str(out),
        rerank_model_path=None,
        safe_model_path=None,
        coarse_k=10,
        top_k=2,
        max_queries=1,
        prefer_cuda=False,
        query_mode="file",
        query_binary=None,
        rerank_use_dfg=None,
    )
    data = json.loads((out / "matches.json").read_text(encoding="utf-8"))
    cands = data["queries"][0]["candidates"]
    assert len(cands) == 2
    names = [x["candidate_name"] for x in cands]
    assert names[0] == names[1] == "dup"
    cves = [tuple(x["cve"]) for x in cands]
    assert {("FAKE-CVE-0001",), ("FAKE-CVE-0002",)} == set(cves)
