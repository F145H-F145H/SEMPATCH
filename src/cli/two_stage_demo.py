"""
TwoStage CVE 匹配：库元数据、query_features 构建、run_demo 报告生成。
由 sempatch match 与侧链 demo_cve_match 共用。
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from config import PROJECT_ROOT

MATCH_STATUS_OK = "ok"
MATCH_STATUS_NO_MATCH = "no_credible_match"


def normalize_cve_field(raw: object) -> List[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        s = raw.strip()
        return [s] if s else []
    if isinstance(raw, list):
        out: List[str] = []
        for x in raw:
            if x is None:
                continue
            s = str(x).strip()
            if s:
                out.append(s)
        return out
    s = str(raw).strip()
    return [s] if s else []


def _norm_entry(entry: str) -> str:
    s = (entry or "").strip().lower()
    if s.startswith("0x"):
        return s
    return f"0x{s}" if s else "0x0"


def _parse_entry_as_int(entry: Union[str, int]) -> int:
    """将 0x401176 / 401176 / 00401176 统一为整型地址。"""
    if isinstance(entry, int):
        return entry
    s = (entry or "").strip()
    if s.lower().startswith("0x"):
        s = s[2:]
    return int(s, 16) if s else 0


def filter_query_function_ids_by_entry(
    qids: Sequence[str],
    query_entry: Optional[str],
) -> List[str]:
    """
    仅保留 function_id 中「入口地址」与 query_entry 数值相等的项（忽略 0x 与前导零差异）。
    query_entry 为空则原样返回。
    """
    if not (query_entry or "").strip():
        return list(qids)
    want = _parse_entry_as_int(query_entry)
    out: List[str] = []
    for qid in qids:
        if "|" not in qid:
            continue
        tail = qid.rsplit("|", 1)[-1]
        try:
            if _parse_entry_as_int(tail) == want:
                out.append(qid)
        except ValueError:
            continue
    if not out:
        raise RuntimeError(
            f"无与 --query-entry {query_entry!r}（数值 0x{want:x}）匹配的查询 function_id，"
            f"当前共 {len(qids)} 条；请核对 Ghidra 入口或省略该选项。"
        )
    return out


def _function_id(binary_rel: str, entry: str) -> str:
    return f"{binary_rel}|{_norm_entry(entry)}"


def parse_query_binary_from_function_id(fid: str) -> str:
    if "|" not in fid:
        return fid
    return fid.rsplit("|", 1)[0]


def load_library_metadata(embeddings_path: str) -> Dict[str, Dict[str, Any]]:
    with open(embeddings_path, encoding="utf-8") as f:
        data = json.load(f)
    funcs = data.get("functions") or []
    meta: Dict[str, Dict[str, Any]] = {}
    for i, item in enumerate(funcs):
        if not isinstance(item, dict):
            continue
        vec = item.get("vector")
        if vec is None:
            continue
        cid = item.get("function_id", item.get("name", str(i)))
        name = item.get("name", "") or str(cid)
        meta[str(cid)] = {
            "name": name,
            "cve": normalize_cve_field(item.get("cve")),
        }
    return meta


def build_candidates_for_ranked(
    ranked: Sequence[Tuple[str, float]],
    library_meta: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """将已截断/过滤后的 ranked 列表格式化为 candidates（rank 从 1 连续编号）。"""
    out: List[Dict[str, Any]] = []
    for rank, (cid, score) in enumerate(ranked, start=1):
        info = library_meta.get(cid, {})
        name = info.get("name", cid)
        cve = info.get("cve", [])
        if not isinstance(cve, list):
            cve = normalize_cve_field(cve)
        out.append(
            {
                "rank": rank,
                "candidate_function_id": cid,
                "candidate_name": name,
                "similarity": float(score),
                "cve": list(cve),
            }
        )
    return out


def filter_ranked_by_policy(
    ranked: Sequence[Tuple[str, float]],
    mode: str,
    min_similarity: float,
    tie_margin: float,
) -> Tuple[List[Tuple[str, float]], Dict[str, Any]]:
    """
    对精排后的全量列表按 unique / all_above 策略过滤。
    返回 (过滤后的 ranked 子列表, filter_meta)；不在此处理 top_k。
    """
    n = len(ranked)
    base_meta: Dict[str, Any] = {
        "mode": mode,
        "min_similarity": float(min_similarity),
        "tie_margin": float(tie_margin),
        "reranked_count": n,
        "reject_reason": None,
        "max_similarity": None,
        "second_similarity": None,
        "top_k": None,
    }
    if mode == "all_above":
        if n == 0:
            base_meta["reject_reason"] = "no_candidates"
            return [], base_meta
        s0 = float(ranked[0][1])
        passed = [(cid, float(s)) for cid, s in ranked if float(s) >= min_similarity]
        if not passed:
            base_meta["reject_reason"] = "below_threshold"
            base_meta["max_similarity"] = s0
            return [], base_meta
        return passed, base_meta

    if mode != "unique":
        raise ValueError(f"filter_ranked_by_policy 不支持 mode={mode!r}（应使用 apply_output_policy 处理 top_k）")

    if n == 0:
        base_meta["reject_reason"] = "no_candidates"
        return [], base_meta
    s0 = float(ranked[0][1])
    base_meta["max_similarity"] = s0
    if s0 < min_similarity:
        base_meta["reject_reason"] = "below_threshold"
        return [], base_meta
    if n >= 2:
        s1 = float(ranked[1][1])
        base_meta["second_similarity"] = s1
        if s0 - s1 <= tie_margin:
            base_meta["reject_reason"] = "tied_top"
            return [], base_meta
    return [ranked[0]], base_meta


def apply_output_policy(
    ranked: List[Tuple[str, float]],
    match_filter: str,
    top_k: int,
    min_similarity: float,
    tie_margin: float,
) -> Tuple[List[Tuple[str, float]], Dict[str, Any], str]:
    """
    粗筛+精排后的全量 ranked → 输出子列表、filter_meta、match_status。
    top_k 模式独立实现，不调用 filter_ranked_by_policy。
    """
    reranked_count = len(ranked)
    if match_filter == "top_k":
        if top_k <= 0:
            meta: Dict[str, Any] = {
                "mode": "top_k",
                "top_k": top_k,
                "reranked_count": reranked_count,
                "min_similarity": None,
                "tie_margin": None,
                "reject_reason": "no_candidates",
                "max_similarity": None,
                "second_similarity": None,
            }
            return [], meta, MATCH_STATUS_NO_MATCH
        take = min(top_k, reranked_count)
        out = ranked[:take]
        meta = {
            "mode": "top_k",
            "top_k": top_k,
            "reranked_count": reranked_count,
            "min_similarity": None,
            "tie_margin": None,
            "reject_reason": "no_candidates" if reranked_count == 0 else None,
            "max_similarity": None,
            "second_similarity": None,
        }
        if reranked_count == 0:
            return [], meta, MATCH_STATUS_NO_MATCH
        return out, meta, MATCH_STATUS_OK

    filtered, meta = filter_ranked_by_policy(
        ranked, match_filter, min_similarity, tie_margin
    )
    st = MATCH_STATUS_OK if filtered else MATCH_STATUS_NO_MATCH
    return filtered, meta, st


def format_match_explanation_zh(filter_meta: Dict[str, Any], match_status: str) -> str:
    """仅根据 filter_meta 生成人类可读原因（渲染层不重复策略判断）。"""
    if match_status == MATCH_STATUS_OK:
        return ""
    rr = filter_meta.get("reject_reason")
    if rr == "no_candidates":
        return "无精排候选（reranked_count=0）。"
    if rr == "below_threshold":
        mx = filter_meta.get("max_similarity")
        mn = filter_meta.get("min_similarity")
        if mx is not None and mn is not None:
            return f"最高分 {mx:.6f} 低于阈值 {mn:.6f}。"
        return "低于相似度阈值。"
    if rr == "tied_top":
        s0 = filter_meta.get("max_similarity")
        s1 = filter_meta.get("second_similarity")
        tm = filter_meta.get("tie_margin")
        if s0 is not None and s1 is not None and tm is not None:
            return (
                f"最高分 {s0:.6f} 与次高分 {s1:.6f} 的差距未超过 tie_margin ({tm:g})，判定为并列。"
            )
        return "最高分并列，未满足唯一性条件。"
    return f"reject_reason={rr!r}。"


def git_short_hash(project_root: str) -> str:
    try:
        r = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if r.returncode == 0 and r.stdout.strip():
            return r.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return "unknown"


def write_report_md(
    path: str,
    config: Dict[str, Any],
    queries_payload: List[Dict[str, Any]],
    *,
    preamble_lines: Optional[Sequence[str]] = None,
    report_title: str = "# CVE 匹配 Demo 报告",
) -> None:
    lines = [
        report_title,
        "",
        "## 配置摘要",
        "",
        "```json",
        json.dumps(config, indent=2, ensure_ascii=False),
        "```",
        "",
    ]
    if preamble_lines:
        lines.extend(list(preamble_lines))
        if lines[-1] != "":
            lines.append("")
    for q in queries_payload:
        qid = q.get("query_function_id", "")
        mst = q.get("match_status", MATCH_STATUS_OK)
        fm = q.get("filter_meta") or {}
        lines.append(f"## 查询 `{qid}`")
        lines.append("")
        lines.append(f"- query_binary: `{q.get('query_binary', '')}`")
        lines.append(f"- **match_status**: `{mst}`")
        if q.get("top_k") is not None:
            lines.append(f"- top_k: {q.get('top_k')}")
        if fm.get("mode") in ("unique", "all_above"):
            lines.append(
                f"- min_similarity: {fm.get('min_similarity')}, tie_margin: {fm.get('tie_margin')}"
            )
        lines.append(f"- reranked_count: {fm.get('reranked_count', '')}")
        expl = format_match_explanation_zh(fm, mst)
        if expl:
            lines.append(f"- 说明: {expl}")
        lines.append("")
        lines.append("| rank | candidate_id | name | similarity | cve |")
        lines.append("|------|--------------|------|------------|-----|")
        for c in q.get("candidates") or []:
            cve_s = json.dumps(c.get("cve", []), ensure_ascii=False)
            lines.append(
                f"| {c.get('rank')} | `{c.get('candidate_function_id')}` | "
                f"{c.get('candidate_name', '')} | {c.get('similarity', 0):.6f} | {cve_s} |"
            )
        if not (q.get("candidates") or []):
            lines.append("| — | — | — | — | — |")
        lines.append("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def build_query_features_from_binary(
    binary_path: str,
    project_root: str,
    output_dir: str,
) -> str:
    from utils.ghidra_runner import (
        peek_binary_cache,
        require_ghidra_environment,
        run_ghidra_analysis,
    )
    from utils.feature_extractors.multimodal_extraction import (
        extract_multimodal_from_lsir_raw,
    )

    require_ghidra_environment()

    if os.path.isabs(binary_path):
        binary_abs = binary_path
    else:
        binary_abs = os.path.abspath(os.path.join(project_root, binary_path))
    if not os.path.isfile(binary_abs):
        raise FileNotFoundError(f"查询二进制不存在: {binary_abs}")

    binary_rel = os.path.relpath(binary_abs, project_root)
    lsir_raw = peek_binary_cache(binary_abs)
    if lsir_raw is None:
        tmp = tempfile.mkdtemp(prefix="demo_cve_ghidra_")
        try:
            lsir_raw = run_ghidra_analysis(
                binary_path=binary_abs,
                output_dir=tmp,
                project_name="DemoCveMatch",
                script_name="extract_lsir_raw.java",
                script_output_name="lsir_raw.json",
                return_dict=True,
            )
        finally:
            import shutil

            shutil.rmtree(tmp, ignore_errors=True)

    if not lsir_raw or not lsir_raw.get("functions"):
        raise RuntimeError("未能从二进制得到 lsir_raw 或 functions 为空（检查 Ghidra 与缓存）")

    funcs = lsir_raw["functions"]
    query_features: Dict[str, Any] = {}
    errors = 0
    for fn in funcs:
        entry = fn.get("entry", "")
        if not entry:
            continue
        fid = _function_id(binary_rel, entry)
        try:
            mm = extract_multimodal_from_lsir_raw(funcs, entry)
            query_features[fid] = mm
        except (ValueError, RuntimeError) as e:
            print(f"警告: 跳过函数 entry={entry}: {e}", file=sys.stderr)
            errors += 1

    if not query_features:
        raise RuntimeError(
            f"未提取到任何查询特征（失败条目约 {errors}）；请检查二进制与 Ghidra 日志"
        )

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "query_features.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(query_features, f, ensure_ascii=False)
    return out_path


def run_demo(
    *,
    query_features_path: str,
    library_emb: str,
    library_features: str,
    output_dir: str,
    rerank_model_path: Optional[str],
    safe_model_path: Optional[str],
    coarse_k: int,
    top_k: int,
    max_queries: Optional[int],
    prefer_cuda: bool,
    query_mode: str,
    query_binary: Optional[str],
    rerank_use_dfg: Optional[bool],
    query_entry: Optional[str] = None,
    match_filter: str = "top_k",
    min_similarity: float = 0.95,
    tie_margin: float = 1e-5,
) -> Dict[str, Any]:
    from matcher.two_stage import TwoStagePipeline

    library_meta = load_library_metadata(library_emb)
    if not library_meta:
        raise RuntimeError("库嵌入为空或无有效 vector 条目")

    pipeline = TwoStagePipeline(
        library_safe_embeddings_path=library_emb,
        library_features_path=library_features,
        query_features_path=query_features_path,
        coarse_k=coarse_k,
        rerank_model_path=rerank_model_path,
        safe_model_path=safe_model_path,
        prefer_cuda=prefer_cuda,
        rerank_use_dfg=rerank_use_dfg,
    )

    with open(query_features_path, encoding="utf-8") as f:
        qfeats = json.load(f)
    if not isinstance(qfeats, dict):
        raise ValueError("query_features 应为 {function_id: multimodal} 对象")

    qids = sorted(qfeats.keys())
    qids = filter_query_function_ids_by_entry(qids, query_entry)
    if max_queries is not None and max_queries > 0:
        qids = qids[:max_queries]

    config = {
        "git_rev": git_short_hash(PROJECT_ROOT),
        "python": sys.executable,
        "project_root": PROJECT_ROOT,
        "library_emb": os.path.abspath(library_emb),
        "library_features": os.path.abspath(library_features),
        "query_features": os.path.abspath(query_features_path),
        "rerank_model_path": os.path.abspath(
            rerank_model_path or os.path.join(PROJECT_ROOT, "output", "best_model.pth")
        ),
        "safe_model_path": os.path.abspath(safe_model_path)
        if safe_model_path
        else None,
        "coarse_k": coarse_k,
        "top_k": top_k,
        "query_mode": query_mode,
        "query_binary": query_binary,
        "max_queries": max_queries,
        "prefer_cuda": prefer_cuda,
        "rerank_use_dfg": rerank_use_dfg,
        "query_entry": query_entry,
        "match_filter": match_filter,
        "min_similarity": min_similarity,
        "tie_margin": tie_margin,
    }

    queries_payload: List[Dict[str, Any]] = []
    for qid in qids:
        ranked = pipeline.retrieve_and_rerank(qid)
        out_ranked, filter_meta, match_status = apply_output_policy(
            ranked, match_filter, top_k, min_similarity, tie_margin
        )
        candidates = build_candidates_for_ranked(out_ranked, library_meta)
        qrow: Dict[str, Any] = {
            "query_function_id": qid,
            "query_binary": parse_query_binary_from_function_id(qid),
            "match_status": match_status,
            "filter_meta": filter_meta,
            "candidates": candidates,
        }
        if match_filter == "top_k":
            qrow["top_k"] = top_k
        else:
            qrow["top_k"] = None
        queries_payload.append(qrow)

    doc = {"config": config, "queries": queries_payload}
    os.makedirs(output_dir, exist_ok=True)
    matches_path = os.path.join(output_dir, "matches.json")
    with open(matches_path, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2, ensure_ascii=False)

    report_path = os.path.join(output_dir, "report.md")
    write_report_md(report_path, config, queries_payload, report_title="# CVE 匹配报告")
    return doc
