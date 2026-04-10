"""DAG 执行器：线程池调度、就绪队列、重试、防重复。"""

import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed, wait, FIRST_COMPLETED
from typing import Any, Dict, Optional, Set

from utils.config import DAG_GHIDRA_THREAD_SLOTS, DAG_MAX_WORKERS

from .model import JobDAG
from .node_exec import build_run_node_fn

logger = logging.getLogger(__name__)


def run_dag(
    dag: JobDAG,
    ctx: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    执行 DAG，在保证依赖顺序的前提下并行调度节点。
    返回更新后的 ctx（含各节点 output）。
    """
    # 空 dict 在布尔上下文中为 False，不能用 `ctx or {}`，否则调用方传入的 {} 会被替换为新 dict，外部永远看不到写入。
    if ctx is None:
        ctx = {}
    completed: Set[str] = set()
    pending: Set[str] = set()
    ready_queue: list = []
    lock = threading.Lock()
    max_workers = max(1, min(64, DAG_MAX_WORKERS))

    import multiprocessing

    sem_ghidra = multiprocessing.Semaphore(DAG_GHIDRA_THREAD_SLOTS)
    sem_cpu = multiprocessing.Semaphore(max(1, DAG_MAX_WORKERS))
    sem_by_type: Dict[str, Any] = {
        "ghidra": sem_ghidra,
        "lsir_build": sem_cpu,
        "feature_extract": sem_cpu,
        "embed": sem_cpu,
        "load_db": sem_cpu,
        "diff": sem_cpu,
        "unpack": sem_cpu,
        "fuzzy_hash": sem_cpu,
        "cfg_match": sem_cpu,
        "acfg_extract": sem_cpu,
        "diff_faiss": sem_cpu,
        "diff_bipartite": sem_cpu,
        "diff_fuzzy": sem_cpu,
    }

    def get_ready() -> list:
        with lock:
            newly = dag.get_ready(pending, completed)
            seen = set(ready_queue)
            for nid in newly:
                if nid not in seen:
                    seen.add(nid)
                    ready_queue.append(nid)
            return list(ready_queue)

    def get_next_node() -> Optional[str]:
        with lock:
            while ready_queue:
                nid = ready_queue.pop(0)
                if nid in completed:
                    continue
                pending.add(nid)
                return nid
            return None

    def on_node_done(nid: str, success: bool) -> None:
        with lock:
            pending.discard(nid)
            if success:
                completed.add(nid)
                node = dag.nodes[nid]
                newly = dag.get_ready(pending, completed)
                seen = set(ready_queue)
                for cnid in newly:
                    if cnid not in seen and cnid not in completed:
                        seen.add(cnid)
                        ready_queue.append(cnid)
            else:
                node = dag.nodes[nid]
                if (
                    getattr(node, "retriable", False)
                    and node.retry_count < node.max_retries
                ):
                    node.retry_count += 1
                    node.failed = False
                    ready_queue.append(nid)
                else:
                    completed.add(nid)

    run_fn = build_run_node_fn(dag, ctx, sem_by_type)

    # 初始就绪
    get_ready()

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {}
        while True:
            nid = get_next_node()
            if nid is None:
                if not futures:
                    break
                done_futures = [f for f in futures if f.done()]
                if not done_futures:
                    done, _ = wait(futures, timeout=2.0, return_when=FIRST_COMPLETED)
                    done_futures = list(done)
                for f in done_futures:
                    if f in futures:
                        fnid = futures.pop(f)
                        try:
                            ok = f.result()
                            on_node_done(fnid, ok)
                        except Exception:
                            logger.warning("节点 %s 执行失败", fnid, exc_info=True)
                            on_node_done(fnid, False)
                continue

            fut = pool.submit(run_fn, nid)
            futures[fut] = nid

        # 等待剩余
        for f in as_completed(futures):
            nid = futures[f]
            try:
                ok = f.result()
                on_node_done(nid, ok)
            except Exception:
                logger.warning("节点 %s 执行失败", nid, exc_info=True)
                on_node_done(nid, False)

    return ctx
