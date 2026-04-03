"""
CVE 匹配生产线（TwoStage 优先，可降级 SAFE 粗筛）。
供 sempatch.py match 与侧链 run_cve_pipeline 调用。
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from config import DEFAULT_MATCH_OUTPUT_PATH, DEFAULT_TWO_STAGE_PATH, PROJECT_ROOT

from cli.two_stage_demo import (
    MATCH_STATUS_NO_MATCH,
    MATCH_STATUS_OK,
    build_query_features_from_binary,
    filter_query_function_ids_by_entry,
    parse_query_binary_from_function_id,
    run_demo,
    write_report_md,
)


def _torch_available() -> bool:
    try:
        import torch  # noqa: F401
    except ImportError:
        return False
    return True


def _python_hint_for_torch() -> str:
    venv_py = os.path.join(PROJECT_ROOT, ".venv", "bin", "python")
    if os.path.isfile(venv_py):
        return (
            f"请使用虚拟环境中的 Python（已检测到 {venv_py}），例如：\n"
            f"  {venv_py} sempatch.py match --query-binary ...\n"
        )
    return (
        "请在项目 venv 中安装依赖后运行，例如：\n"
        "  python3 -m venv .venv && .venv/bin/pip install -r requirements.txt\n"
        "  .venv/bin/python sempatch.py match ...\n"
    )


def _ghidra_hint() -> str:
    return (
        "请设置环境变量 GHIDRA_HOME 或在 sempatch.cfg 中配置 Ghidra 安装路径，"
        "并确保 analyzeHeadless 存在且可执行（需 Java）。\n"
    )


def _abs(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(PROJECT_ROOT, path))


def _embed_build_script(project_root: str) -> str:
    side = os.path.join(project_root, "scripts", "sidechain", "build_embeddings_db.py")
    if os.path.isfile(side):
        return side
    return os.path.join(project_root, "scripts", "build_embeddings_db.py")


def _run(cmd: List[str], *, cwd: str) -> None:
    print(f"$ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def _file_size_mb(path: str) -> float:
    try:
        return os.path.getsize(path) / (1024 * 1024)
    except OSError:
        return -1.0


def _normalize_cve_field(raw: object) -> List[str]:
    from cli.two_stage_demo import normalize_cve_field

    return normalize_cve_field(raw)


def _load_library_meta(emb_path: str) -> Dict[str, Dict[str, Any]]:
    with open(emb_path, encoding="utf-8") as f:
        data = json.load(f)
    funcs = data.get("functions") if isinstance(data, dict) else None
    if not isinstance(funcs, list):
        return {}
    meta: Dict[str, Dict[str, Any]] = {}
    for i, item in enumerate(funcs):
        if not isinstance(item, dict):
            continue
        vec = item.get("vector")
        if vec is None:
            continue
        cid = item.get("function_id", item.get("name", str(i)))
        meta[str(cid)] = {
            "name": item.get("name", str(cid)),
            "cve": _normalize_cve_field(item.get("cve")),
        }
    return meta


def _count_non_empty_cve(emb_path: str) -> int:
    try:
        with open(emb_path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return 0
    funcs = data.get("functions") if isinstance(data, dict) else None
    if not isinstance(funcs, list):
        return 0
    cnt = 0
    for item in funcs:
        if not isinstance(item, dict):
            continue
        cve = item.get("cve")
        if isinstance(cve, str) and cve.strip():
            cnt += 1
            continue
        if isinstance(cve, list) and any(str(x).strip() for x in cve if x is not None):
            cnt += 1
    return cnt


def _write_status(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _enrich_status_from_matches(status: Dict[str, Any], out_dir: str) -> None:
    """从 matches.json 写入简要 Top-1 摘要，便于对照 pipeline_status 判断「是否命中」。"""
    mp = os.path.join(out_dir, "matches.json")
    if not os.path.isfile(mp):
        return
    try:
        with open(mp, encoding="utf-8") as f:
            doc = json.load(f)
    except (OSError, json.JSONDecodeError):
        return
    queries = doc.get("queries") or []
    summ: Dict[str, Any] = {"queries_in_report": len(queries)}
    if queries:
        q0 = queries[0]
        cands = q0.get("candidates") or []
        summ["first_query_function_id"] = q0.get("query_function_id")
        summ["first_query_match_status"] = q0.get("match_status")
        if cands:
            top = cands[0]
            summ["first_query_rank1"] = {
                "candidate_function_id": top.get("candidate_function_id"),
                "candidate_name": top.get("candidate_name"),
                "similarity": top.get("similarity"),
                "cve": top.get("cve"),
            }
    status["match_summary"] = summ


MATCH_COMMAND_EPILOG = """
高置信度匹配示例（须在已安装 PyTorch 的 venv 中运行，使两段权重均参与推理）：
  .venv/bin/python sempatch.py match --query-binary QUERY.elf --two-stage-dir LIB_DIR \\
      --match-filter unique --min-similarity 0.95 --coarse-k 500
  .venv/bin/python sempatch.py match --query-binary QUERY.elf --two-stage-dir LIB_DIR \\
      --match-filter all_above --min-similarity 0.9

说明：--min-similarity 与「唯一最高分」均为启发式，不保证 CVE 标注正确，结果需人工复核。
并列高分导致 unique 无输出时，可先用 all_above 查看分数分布，再调整 --tie-margin 或阈值。
"""


def _normalize_query_entry_arg(raw: Optional[str]) -> Optional[str]:
    if raw is None:
        return None
    s = raw.strip()
    return s or None


def _effective_max_queries_cli(raw: Optional[int]) -> Optional[int]:
    """
    CLI: 默认 0 或未给出表示不限制（全函数）。
    正整数表示最多处理该数量的查询函数（按 function_id 排序后截断）。
    """
    if raw is None or raw <= 0:
        return None
    return int(raw)


def _load_query_multimodals_limited(
    query_features_path: str,
    max_queries: Optional[int],
    max_json_mb: int,
    query_entry: Optional[str] = None,
) -> List[Tuple[str, Dict[str, Any]]]:
    from utils.precomputed_multimodal_io import is_jsonl_sidecar_path, iter_jsonl_sidecar

    qpath = _abs(query_features_path)
    if is_jsonl_sidecar_path(qpath):
        out: List[Tuple[str, Dict[str, Any]]] = []
        for fid, mm in iter_jsonl_sidecar(qpath):
            out.append((fid, mm))
        fids = [x[0] for x in out]
        fids = filter_query_function_ids_by_entry(fids, query_entry)
        allow = set(fids)
        out = [(fid, mm) for fid, mm in out if fid in allow]
        if max_queries is not None:
            out = out[:max_queries]
        return out

    sz = _file_size_mb(qpath)
    if sz > float(max_json_mb):
        raise RuntimeError(
            f"query_features 为 JSON 且体积约 {sz:.1f}MB，超过阈值 {max_json_mb}MB，"
            "为避免 OOM 请改用 --query-binary 或 JSONL 侧车。"
        )
    with open(qpath, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("query_features 应为 {function_id: multimodal} 对象")
    qids = sorted(data.keys())
    qids = filter_query_function_ids_by_entry(qids, query_entry)
    if max_queries is not None:
        qids = qids[:max_queries]
    return [(qid, data[qid]) for qid in qids]


def _run_coarse_only_report(
    *,
    query_features_path: str,
    library_emb: str,
    output_dir: str,
    safe_model_path: Optional[str],
    top_k: int,
    max_queries: Optional[int],
    max_query_features_mb: int,
    query_entry: Optional[str] = None,
    match_filter_requested: str = "top_k",
) -> Dict[str, Any]:
    from matcher.faiss_library import LibraryFaissIndex
    from features.baselines.safe import embed_batch_safe

    library_meta = _load_library_meta(library_emb)
    if not library_meta:
        raise RuntimeError("库嵌入为空或无有效向量")
    query_items = _load_query_multimodals_limited(
        query_features_path,
        max_queries=max_queries,
        max_json_mb=max_query_features_mb,
        query_entry=query_entry,
    )
    if not query_items:
        raise RuntimeError("查询特征为空")

    index = LibraryFaissIndex(library_emb)
    queries_payload: List[Dict[str, Any]] = []
    for qid, mm in query_items:
        feats = {"functions": [{"name": "query", "features": {"multimodal": mm}}]}
        emb = embed_batch_safe(feats, model_path=safe_model_path)
        if not emb:
            ranked: List[Tuple[str, float]] = []
        else:
            ranked = index.search(emb[0]["vector"], k=top_k)
        candidates: List[Dict[str, Any]] = []
        for rank, (cid, score) in enumerate(ranked, start=1):
            info = library_meta.get(cid, {})
            candidates.append(
                {
                    "rank": rank,
                    "candidate_function_id": cid,
                    "candidate_name": info.get("name", cid),
                    "similarity": float(score),
                    "cve": list(info.get("cve", [])),
                }
            )
        rcount = len(ranked)
        fm: Dict[str, Any] = {
            "mode": "coarse_only_safe",
            "threshold_filter_applied": False,
            "reranked_count": rcount,
            "min_similarity": None,
            "tie_margin": None,
            "reject_reason": "no_candidates" if rcount == 0 else None,
            "max_similarity": None,
            "second_similarity": None,
            "top_k": top_k,
        }
        mst = MATCH_STATUS_OK if candidates else MATCH_STATUS_NO_MATCH
        queries_payload.append(
            {
                "query_function_id": qid,
                "query_binary": parse_query_binary_from_function_id(qid),
                "match_status": mst,
                "filter_meta": fm,
                "top_k": top_k,
                "candidates": candidates,
            }
        )

    config = {
        "mode": "coarse_only_safe",
        "query_features": _abs(query_features_path),
        "library_emb": _abs(library_emb),
        "safe_model_path": _abs(safe_model_path) if safe_model_path else None,
        "top_k": top_k,
        "max_queries": max_queries,
        "query_entry": query_entry,
        "match_filter_requested": match_filter_requested,
        "threshold_filter_applied": False,
        "note": "低内存降级路径：跳过精排（TwoStage）",
    }
    doc = {"config": config, "queries": queries_payload}
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "matches.json"), "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2, ensure_ascii=False)
    report_path = os.path.join(output_dir, "report.md")
    preamble: Optional[List[str]] = None
    if match_filter_requested != "top_k":
        preamble = [
            "## 警告",
            "",
            "当前为 **coarse_only_safe** 降级：未运行多模态精排，`--match-filter` / "
            "`--min-similarity` / `--tie-margin` **未应用**；下列为 SAFE 粗筛 Top 结果。",
            "",
        ]
    write_report_md(
        report_path,
        config,
        queries_payload,
        preamble_lines=preamble,
        report_title="# CVE 匹配报告",
    )
    return doc


@dataclass
class CveMatchOptions:
    query_binary: Optional[str] = None
    query_features: Optional[str] = None
    two_stage_dir: str = DEFAULT_TWO_STAGE_PATH
    library_features: Optional[str] = None
    library_emb: Optional[str] = None
    safe_model_path: Optional[str] = None
    model_path: Optional[str] = None
    output_dir: str = DEFAULT_MATCH_OUTPUT_PATH
    coarse_k: int = 100
    top_k: int = 10
    max_queries: Optional[int] = None
    build_missing: bool = True
    allow_coarse_fallback: bool = True
    max_library_features_mb_for_two_stage: int = 2048
    max_query_features_mb_for_json_load: int = 512
    cpu: bool = False
    rerank_use_dfg: Optional[bool] = None
    verbose: bool = True
    query_entry: Optional[str] = None
    match_filter: str = "top_k"
    min_similarity: float = 0.95
    tie_margin: float = 1e-5
    coarse_k_explicit: bool = False
    inspect: bool = False
    profile: Optional[str] = None


def run_cve_match_pipeline(opts: CveMatchOptions) -> int:
    """执行生产线；返回进程退出码（0 成功，1 失败）。"""
    if not _torch_available():
        print(
            "错误: match 子命令需要 PyTorch，当前解释器未安装 torch。\n"
            + _python_hint_for_torch(),
            file=sys.stderr,
            flush=True,
        )
        return 1

    if opts.query_binary:
        try:
            from utils.ghidra_runner import GhidraEnvironmentError, require_ghidra_environment

            require_ghidra_environment()
        except GhidraEnvironmentError as e:
            print(
                "错误: --query-binary 需要可用的 Ghidra 环境:\n"
                f"  {e}\n"
                + _ghidra_hint(),
                file=sys.stderr,
                flush=True,
            )
            return 1

    two_stage_dir = _abs(opts.two_stage_dir)
    out_dir = _abs(opts.output_dir)
    status_path = os.path.join(out_dir, "pipeline_status.json")

    library_features = _abs(
        opts.library_features
        if opts.library_features
        else os.path.join(two_stage_dir, "library_features.json")
    )
    library_emb = _abs(
        opts.library_emb
        if opts.library_emb
        else os.path.join(two_stage_dir, "library_safe_embeddings.json")
    )

    mq = opts.max_queries

    status: Dict[str, Any] = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "project_root": PROJECT_ROOT,
        "inputs": {
            "query_binary": opts.query_binary,
            "query_features": opts.query_features,
            "two_stage_dir": two_stage_dir,
            "library_features": library_features,
            "library_emb": library_emb,
            "safe_model_path": opts.safe_model_path,
            "model_path": opts.model_path,
            "max_queries": mq,
            "query_entry": opts.query_entry,
            "match_filter": opts.match_filter,
            "min_similarity": opts.min_similarity,
            "tie_margin": opts.tie_margin,
            "profile": opts.profile,
            "coarse_k": opts.coarse_k,
            "coarse_k_explicit": opts.coarse_k_explicit,
        },
        "steps": [],
        "artifacts": {},
        "missing": [],
        "warnings": [],
        "output": {
            "output_dir": out_dir,
            "matches_json": os.path.join(out_dir, "matches.json"),
            "report_md": os.path.join(out_dir, "report.md"),
        },
        "ok": False,
    }

    def mark_artifact(name: str, path: str) -> bool:
        exists = os.path.isfile(path)
        status["artifacts"][name] = {"path": path, "exists": exists}
        return exists

    os.makedirs(out_dir, exist_ok=True)
    mark_artifact("library_features", library_features)
    mark_artifact("library_emb", library_emb)
    if opts.query_features:
        mark_artifact("query_features", _abs(opts.query_features))

    query_features_path: Optional[str] = None
    if opts.query_binary:
        try:
            query_features_path = build_query_features_from_binary(
                opts.query_binary, PROJECT_ROOT, out_dir
            )
            status["steps"].append(
                {
                    "name": "build_query_features_from_binary",
                    "result": "ok",
                    "query_features": query_features_path,
                }
            )
            mark_artifact("query_features", query_features_path)
        except Exception as e:
            status["missing"].append(
                {
                    "artifact": "query_features",
                    "path": os.path.join(out_dir, "query_features.json"),
                    "reason": f"query-binary 提取失败: {e}",
                }
            )
    else:
        qf = _abs(opts.query_features or "")
        query_features_path = qf
        mark_artifact("query_features", qf)
        if not os.path.isfile(qf):
            status["missing"].append(
                {
                    "artifact": "query_features",
                    "path": qf,
                    "reason": "--query-features 文件不存在",
                }
            )

    if not os.path.isfile(library_emb):
        if not opts.build_missing:
            status["missing"].append(
                {
                    "artifact": "library_safe_embeddings",
                    "path": library_emb,
                    "reason": "文件缺失且 --no-build-missing 已启用",
                }
            )
        elif not os.path.isfile(library_features):
            status["missing"].append(
                {
                    "artifact": "library_safe_embeddings",
                    "path": library_emb,
                    "reason": "缺少 library_features.json，无法推理构建",
                }
            )
        else:
            lib_feat_mb = _file_size_mb(library_features)
            if library_features.lower().endswith(".json") and lib_feat_mb > 2048:
                status["missing"].append(
                    {
                        "artifact": "library_safe_embeddings",
                        "path": library_emb,
                        "reason": (
                            f"library_features.json 约 {lib_feat_mb:.1f}MB，"
                            "构建库嵌入易 OOM；请改用 JSONL 侧车或预先提供 library_emb"
                        ),
                    }
                )
                _write_status(status_path, status)
                if opts.verbose:
                    print(
                        f"[PARTIAL] 预检查发现高 OOM 风险，状态文件: {status_path}",
                        file=sys.stderr,
                    )
                return 1
            emb_script = _embed_build_script(PROJECT_ROOT)
            cmd = [
                sys.executable,
                emb_script,
                "--features-file",
                library_features,
                "--model",
                "safe",
                "-o",
                library_emb,
            ]
            if opts.safe_model_path:
                cmd.extend(["--model-path", _abs(opts.safe_model_path)])
            _run(cmd, cwd=PROJECT_ROOT)
            status["steps"].append(
                {
                    "name": "build_library_safe_embeddings",
                    "command": cmd,
                    "result": "ok",
                }
            )
            mark_artifact("library_emb", library_emb)

    if os.path.isfile(library_emb):
        non_empty_cve = _count_non_empty_cve(library_emb)
        status["artifacts"]["library_emb"]["non_empty_cve_items"] = non_empty_cve
        if non_empty_cve == 0:
            status["warnings"].append(
                "library_emb 中未发现非空 cve 字段；报告会输出 cve=[]（流程可运行但漏洞语义信息缺失）"
            )

    if opts.match_filter != "top_k" and not opts.coarse_k_explicit:
        status["warnings"].append(
            "当前为阈值匹配模式但未显式指定 --coarse-k：仍使用默认粗筛 K=100，可能漏召回；"
            "建议显式传入例如 --coarse-k 500。"
        )

    can_run = (
        os.path.isfile(library_emb)
        and query_features_path
        and os.path.isfile(query_features_path)
    )
    if not can_run and not status["missing"]:
        status["missing"].append(
            {
                "artifact": "runtime_inputs",
                "path": out_dir,
                "reason": "运行所需输入不完整",
            }
        )
    ran_two_stage = False

    if can_run and os.path.isfile(library_features):
        lib_feat_mb = _file_size_mb(library_features)
        if lib_feat_mb > float(opts.max_library_features_mb_for_two_stage):
            status["warnings"].append(
                (
                    f"library_features 体积约 {lib_feat_mb:.1f}MB，超过 TwoStage 阈值 "
                    f"{opts.max_library_features_mb_for_two_stage}MB，跳过精排以避免 OOM。"
                )
            )
        else:
            try:
                q_mode = "binary" if opts.query_binary else "file"
                run_demo(
                    query_features_path=query_features_path,
                    library_emb=library_emb,
                    library_features=library_features,
                    output_dir=out_dir,
                    rerank_model_path=_abs(opts.model_path) if opts.model_path else None,
                    safe_model_path=_abs(opts.safe_model_path) if opts.safe_model_path else None,
                    coarse_k=opts.coarse_k,
                    top_k=opts.top_k,
                    max_queries=mq,
                    prefer_cuda=not opts.cpu,
                    query_mode=q_mode,
                    query_binary=opts.query_binary,
                    rerank_use_dfg=opts.rerank_use_dfg,
                    query_entry=opts.query_entry,
                    match_filter=opts.match_filter,
                    min_similarity=opts.min_similarity,
                    tie_margin=opts.tie_margin,
                )
                status["steps"].append(
                    {"name": "two_stage_run_demo", "result": "ok"}
                )
                status["mode"] = "two_stage"
                ran_two_stage = True
            except Exception as e:
                status["warnings"].append(
                    f"TwoStage 执行失败（{e}），将尝试 coarse-only fallback。"
                )

    if can_run and not ran_two_stage:
        if not opts.allow_coarse_fallback:
            status["missing"].append(
                {
                    "artifact": "two_stage_execution",
                    "path": out_dir,
                    "reason": "TwoStage 未执行成功且已禁用 coarse fallback",
                }
            )
        else:
            try:
                _run_coarse_only_report(
                    query_features_path=query_features_path,
                    library_emb=library_emb,
                    output_dir=out_dir,
                    safe_model_path=_abs(opts.safe_model_path) if opts.safe_model_path else None,
                    top_k=opts.top_k,
                    max_queries=mq,
                    max_query_features_mb=opts.max_query_features_mb_for_json_load,
                    query_entry=opts.query_entry,
                    match_filter_requested=opts.match_filter,
                )
                status["steps"].append({"name": "coarse_only_fallback", "result": "ok"})
                status["mode"] = "coarse_only_safe"
            except Exception as e:
                status["missing"].append(
                    {
                        "artifact": "coarse_only_fallback",
                        "path": out_dir,
                        "reason": f"降级路径执行失败: {e}",
                    }
                )

    if not os.path.isfile(library_features):
        status["warnings"].append("library_features 不存在：已跳过 TwoStage 精排。")

    matches_ok = os.path.isfile(status["output"]["matches_json"])
    report_ok = os.path.isfile(status["output"]["report_md"])
    status["artifacts"]["matches_json"] = {
        "path": status["output"]["matches_json"],
        "exists": matches_ok,
    }
    status["artifacts"]["report_md"] = {
        "path": status["output"]["report_md"],
        "exists": report_ok,
    }
    status["ok"] = matches_ok and report_ok and not status["missing"]
    _enrich_status_from_matches(status, out_dir)
    _write_status(status_path, status)

    if opts.inspect and matches_ok:
        try:
            from cli.inspect import inspect_matches, print_inspect_report

            matches_path = status["output"]["matches_json"]
            info = inspect_matches(matches_path)
            print_inspect_report(info)
        except Exception as e:
            print(f"Warning: inspect failed: {e}", file=sys.stderr)

    if status["ok"]:
        if opts.verbose:
            print(f"[OK] 生产线执行完成，状态文件: {status_path}")
        return 0

    if opts.verbose:
        print(f"[PARTIAL] 生产线未完全达成，状态文件: {status_path}", file=sys.stderr)
        if status["missing"]:
            print("缺失项:", file=sys.stderr)
            for item in status["missing"]:
                print(
                    f"- {item.get('artifact')}: {item.get('reason')} ({item.get('path')})",
                    file=sys.stderr,
                )
    return 1


def add_match_arguments(parser: argparse.ArgumentParser, *, require_source: bool = True) -> None:
    """向子解析器或独立解析器挂载 match 参数。"""
    from cli.profiles import PROFILES

    parser.add_argument(
        "--profile",
        choices=list(PROFILES.keys()),
        default=None,
        help="预设配置：quick（快速探索）、standard（均衡）、full（全量扫描）",
    )
    parser.add_argument(
        "--inspect",
        action="store_true",
        help="匹配完成后输出分数分布与过滤分析",
    )
    if require_source:
        src = parser.add_mutually_exclusive_group(required=True)
    else:
        src = parser.add_mutually_exclusive_group(required=False)
    src.add_argument("--query-binary", help="查询 ELF（二进制输入）")
    src.add_argument(
        "--query-features",
        help="预计算查询特征 JSON（无 Ghidra 场景）",
    )
    parser.add_argument(
        "--two-stage-dir",
        default=DEFAULT_TWO_STAGE_PATH,
        help=f"两阶段数据目录（默认 {DEFAULT_TWO_STAGE_PATH}，来自 sempatch.cfg paths.data_dir/two_stage_rel）",
    )
    parser.add_argument(
        "--library-features",
        default=None,
        help="库特征文件；默认 <two-stage-dir>/library_features.json",
    )
    parser.add_argument(
        "--library-emb",
        default=None,
        help="库 SAFE 嵌入；默认 <two-stage-dir>/library_safe_embeddings.json",
    )
    parser.add_argument(
        "--safe-model-path",
        default=None,
        help="SAFE 权重（可选；缺失时仍可构建 baseline 风格嵌入）",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="精排模型权重（可选；缺失时 demo 走默认行为）",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_MATCH_OUTPUT_PATH,
        help=f"报告与状态输出目录（默认 {DEFAULT_MATCH_OUTPUT_PATH}，来自 paths.output_dir/match_run_subdir）",
    )
    parser.add_argument(
        "--coarse-k",
        type=int,
        default=argparse.SUPPRESS,
        help=(
            "SAFE 粗筛召回数量（默认 100；仅当本参数出现在命令行时视为「显式指定」。"
            "阈值匹配模式下建议 500+ 以降低漏召回"
        ),
    )
    parser.add_argument("--top-k", type=int, default=argparse.SUPPRESS, help="match-filter=top_k 时报告截断 Top-K")
    parser.add_argument(
        "--match-filter",
        choices=["top_k", "unique", "all_above"],
        default=argparse.SUPPRESS,
        help=(
            "结果策略：top_k=按 --top-k 截断、无阈值（默认，适合快速浏览）；"
            "unique=仅当最高分>=--min-similarity 且与次高分差>--tie-margin 时输出一条；"
            "all_above=输出所有分数>=阈值的候选（探索分布）"
        ),
    )
    parser.add_argument(
        "--min-similarity",
        type=float,
        default=argparse.SUPPRESS,
        metavar="S",
        help="unique / all_above 的余弦相似度阈值（启发式，默认 0.95；探索时可试 0.9）",
    )
    parser.add_argument(
        "--tie-margin",
        type=float,
        default=1e-5,
        metavar="EPS",
        help="unique：若 s0-s1<=EPS 则视为并列第一（默认 1e-5）",
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        default=argparse.SUPPRESS,
        help="最多处理查询函数数；0（默认）表示不限制、处理全部函数",
    )
    parser.add_argument(
        "--query-entry",
        default=None,
        metavar="ADDR",
        help=(
            "仅匹配 function_id 末尾入口地址等于该十六进制的查询（如 0x401176）；"
            "fake_cve_demo 中 query.elf 与 vuln_fake_05 同源时通常为 vuln 入口，报告首条即 Top-1 真阳性"
        ),
    )
    parser.add_argument(
        "--build-missing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否自动补齐可推理构建的缺失产物（默认开启）",
    )
    parser.add_argument(
        "--allow-coarse-fallback",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="TwoStage 失败或内存风险时，自动降级 SAFE 粗筛报告（默认开启）",
    )
    parser.add_argument(
        "--max-library-features-mb-for-two-stage",
        type=int,
        default=2048,
        help="当 library_features 体积超过该阈值时，默认不走 TwoStage（防 OOM）",
    )
    parser.add_argument(
        "--max-query-features-mb-for-json-load",
        type=int,
        default=512,
        help="coarse fallback 下 query_features 为 JSON 时的体积上限（MB）",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="强制 CPU",
    )
    parser.add_argument(
        "--use-dfg-model",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="精排是否使用 DFG 分支：默认按权重/meta 推断",
    )
    parser.epilog = MATCH_COMMAND_EPILOG


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="CVE 匹配生产线：二进制/特征 -> matches.json / report.md（不训练）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_match_arguments(parser, require_source=True)
    return parser


def namespace_to_options(args: argparse.Namespace) -> CveMatchOptions:
    from cli.profiles import resolve_with_profile

    profile_name = getattr(args, "profile", None)
    resolved = resolve_with_profile(args, profile_name)

    coarse_k = int(resolved["coarse_k"])
    coarse_k_explicit = hasattr(args, "coarse_k")
    return CveMatchOptions(
        query_binary=args.query_binary,
        query_features=args.query_features,
        two_stage_dir=args.two_stage_dir,
        library_features=args.library_features,
        library_emb=args.library_emb,
        safe_model_path=args.safe_model_path,
        model_path=args.model_path,
        output_dir=args.output_dir,
        coarse_k=coarse_k,
        coarse_k_explicit=coarse_k_explicit,
        top_k=int(resolved["top_k"]),
        max_queries=_effective_max_queries_cli(int(resolved["max_queries"])),
        build_missing=args.build_missing,
        allow_coarse_fallback=args.allow_coarse_fallback,
        max_library_features_mb_for_two_stage=args.max_library_features_mb_for_two_stage,
        max_query_features_mb_for_json_load=args.max_query_features_mb_for_json_load,
        cpu=args.cpu,
        rerank_use_dfg=args.use_dfg_model,
        query_entry=_normalize_query_entry_arg(getattr(args, "query_entry", None)),
        match_filter=resolved["match_filter"],
        min_similarity=float(resolved["min_similarity"]),
        tie_margin=float(args.tie_margin),
        inspect=getattr(args, "inspect", False),
        profile=profile_name,
    )


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    return run_cve_match_pipeline(namespace_to_options(args))
