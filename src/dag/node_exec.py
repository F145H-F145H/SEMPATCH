"""节点执行：构造 run_node_fn，注入 ctx、信号量控制。"""

from typing import Any, Callable, Dict, Optional

from utils.concurrency import bounded_task_slots

from .model import DAGNode, JobDAG


def build_run_node_fn(
    dag: JobDAG,
    ctx: Dict[str, Any],
    sem_by_type: Dict[str, Any],
) -> Callable[[str], bool]:
    """返回 run_node_fn(node_id) -> 是否成功。"""

    def run_node_fn(node_id: str) -> bool:
        node = dag.nodes[node_id]
        sem = sem_by_type.get(node.node_type)
        if sem is None:
            # 无信号量则直接执行
            return _run_node(node, ctx)
        return bounded_task_slots(sem, node.thread_slots, _run_node, node, ctx)

    return run_node_fn


def _run_node(node: DAGNode, ctx: Dict[str, Any]) -> bool:
    """实际执行，返回是否成功。"""
    try:
        node.execute(ctx)
        node.done = True
        node.failed = False
        return True
    except Exception:
        node.failed = True
        node.done = False
        raise
