"""DAG 节点模型与图结构。"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type


class DAGNode(ABC):
    """DAG 节点抽象基类。子类实现 execute(ctx) 并注册到 NODE_TYPE_REGISTRY。"""

    retriable: bool = False

    def __init__(
        self,
        node_id: str,
        node_type: str,
        params: Dict[str, Any],
        deps: List[str],
        *,
        priority: int = 0,
        worker_id: Optional[int] = None,
        max_retries: int = 2,
        retry_count: int = 0,
        thread_slots: int = 1,
    ):
        self.node_id = node_id
        self.node_type = node_type
        self.params = params
        self.deps = deps
        self.priority = priority
        self.worker_id = worker_id
        self.max_retries = max_retries
        self.retry_count = retry_count
        self.thread_slots = max(1, thread_slots)
        self.done = False
        self.failed = False
        self.output: Any = None

    @abstractmethod
    def execute(self, ctx: Dict[str, Any]) -> None:
        """执行节点逻辑，可读写 ctx。"""
        pass

    def display_label(self, nid: Optional[str] = None) -> str:
        """用于 export 与日志的节点标签。"""
        label = getattr(self, "NODE_TYPE", self.node_type)
        nid = nid or self.node_id
        return f"{label}:{nid}"


class JobDAG:
    """DAG 图结构，通过 NODE_TYPE_REGISTRY 实例化节点。"""

    def __init__(self):
        self.nodes: Dict[str, DAGNode] = {}
        self._registry: Optional[Dict[str, Type[DAGNode]]] = None

    def _get_registry(self) -> Dict[str, Type[DAGNode]]:
        if self._registry is not None:
            return self._registry
        from .nodes import NODE_TYPE_REGISTRY

        return NODE_TYPE_REGISTRY

    def add_node(
        self,
        node_id: str,
        node_type: str,
        params: Dict[str, Any],
        deps: Optional[List[str]] = None,
        *,
        priority: int = 0,
        worker_id: Optional[int] = None,
        max_retries: int = 2,
        thread_slots: int = 1,
        **kwargs: Any,
    ) -> DAGNode:
        """添加节点，按 node_type 查表实例化。"""
        deps = deps or []
        registry = self._get_registry()
        if node_type not in registry:
            raise ValueError(f"Unknown node_type: {node_type}, registry keys: {list(registry.keys())}")
        cls = registry[node_type]
        node = cls(
            node_id=node_id,
            node_type=node_type,
            params=params,
            deps=deps,
            priority=priority,
            worker_id=worker_id,
            max_retries=max_retries,
            thread_slots=thread_slots,
            **kwargs,
        )
        self.nodes[node_id] = node
        return node

    def get_ready(
        self,
        pending: set,
        completed: set,
    ) -> List[str]:
        """返回依赖已全部完成且不在 pending 中的 node_id 列表。"""
        result = []
        for nid, node in self.nodes.items():
            if nid in pending or nid in completed:
                continue
            if all(d in completed for d in node.deps):
                result.append(nid)
        return sorted(result, key=lambda x: -self.nodes[x].priority)

    def get_dependencies(self, node_id: str) -> List[str]:
        return list(self.nodes[node_id].deps)
