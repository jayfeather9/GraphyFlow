from __future__ import annotations
from typing import Callable, List, Optional, Union
from uuid import UUID
import uuid as uuid_lib
from graphyflow.datatypes import *


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
    
    @property
    def preds(self) -> List[Node]:
        return self._pred_nodes

    def __repr__(self) -> str:
        return f"Node(name={self.class_name}, preds={[node.class_name for node in self._pred_nodes]})"


class Inputer(Node):
    def __init__(self, input_type):
        self.input_type = input_type
        super().__init__()


class Updater(Node):
    def __init__(self, type_, attr):
        assert type_ in ["node", "edge"]
        self.type_ = type_
        self.attr = attr
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
        self.input_nodes = [] # by UUID
        self.nodes = {}  # Each node represents a method, nodes = {uuid: node}

    def pseudo_element(self, **kwargs) -> PseudoElement:
        return PseudoElement(graph=self, **kwargs)
    
    def add_input(self, type_: str, **kwargs) -> PseudoElement:
        assert type_ in ["edge", "node"]
        return self.pseudo_element(cur_node=Inputer(input_type=BasicNode if type_ == "node" else BasicEdge))

    def assign_node(self, node: Node):
        assert node.uuid not in self.nodes
        self.nodes[node.uuid] = node
    
    def apply_all_edges(self, datas: PseudoElement, attr_name: str):
        datas._assign_successor(Updater("edge", attr_name))
    
    def topo_sort_nodes(self) -> List[Node]:
        result = []
        waitings = list(self.nodes.values())
        while waitings:
            new_ones = []
            for n in waitings:
                if all(pred in result for pred in n.preds):
                    new_ones.append(n)
                    waitings.remove(n)
            result.extend(new_ones)
        return result

    def __repr__(self) -> str:
        return f"GlobalGraph(nodes={[node for node in self.nodes.values()]})"


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
