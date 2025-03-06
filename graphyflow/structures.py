from __future__ import annotations
from typing import Callable, List, Optional, Union
from uuid import UUID
import uuid as uuid_lib
from graphyflow.datatypes import DataElement


class Node:
    """每个计算图的节点可以单或多输入，但是只有一样输出。任何将该节点作为pred_nodes的节点都将接受到该节点的输出的一份复制"""

    def __init__(
        self,
    ) -> None:
        self.uuid = uuid_lib.uuid4()
        self._pred_nodes = []

    def set_pred_nodes(self, pred_nodes: List[Node]) -> None:
        self._pred_nodes = pred_nodes

    @property
    def class_name(self) -> str:
        return self.__class__.__name__

    def __repr__(self) -> str:
        return f"Node(name={self.class_name}, preds={[node.class_name for node in self._pred_nodes]})"


class Inputer(Node):
    def __init__(self, input_type):
        self.input_type = input_type
        super().__init__()


class Filter(Node):
    def __init__(self, filter_func: Callable[[List[DataElement]], DataElement]):
        self.filter_func = filter_func
        super().__init__()


class Map_(Node):
    def __init__(self, map_func: Callable[[List[DataElement]], DataElement]):
        self.map_func = map_func
        super().__init__()


class ReduceBy(Node):
    def __init__(
        self,
        reduce_key: Callable[[List[DataElement]], DataElement],
        reduce_method: Callable[[List[DataElement]], DataElement],
    ):
        self.reduce_key = reduce_key
        self.reduce_method = reduce_method
        super().__init__()


class GlobalGraph:
    def __init__(self):
        self.nodes = {}  # Each node represents a method, nodes = {uuid: node}

    def pseudo_element(self, **kwargs) -> PseudoElement:
        return PseudoElement(graph=self, **kwargs)

    def assign_node(self, node: Node):
        assert node.uuid not in self.nodes
        self.nodes[node.uuid] = node

    def __repr__(self) -> str:
        return f"GlobalGraph(nodes={self.nodes}, edges={self.edges})"


class PseudoElement:
    def __init__(
        self,
        graph: GlobalGraph,
        cur_node: Optional[Node] = None,
    ):
        self.graph = graph
        self.cur_node = cur_node
        if cur_node:
            self.graph.assign_node(cur_node)

    def _assign_cur_node(self, cur_node: Node):
        assert self.cur_node is None
        self.graph.assign_node(cur_node)
        self.cur_node = cur_node

    def _assign_successor(self, succ_node: Node):
        if self.cur_node:
            succ_node.set_pred_nodes([self.cur_node])
        return self.graph.pseudo_element(cur_node=succ_node)

    def filter(self, **kvargs) -> PseudoElement:
        return self._assign_successor(Filter(**kvargs))

    def map_(self, **kvargs) -> PseudoElement:
        return self._assign_successor(Map_(**kvargs))

    def reduce_by(self, **kvargs) -> PseudoElement:
        return self._assign_successor(ReduceBy(**kvargs))
