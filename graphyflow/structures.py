from __future__ import annotations
from typing import Callable, List, Optional, Union
from uuid import UUID
import uuid as uuid_lib
from graphyflow.graph import DataElement


class Node:
    def __init__(
        self,
        multi_input: bool = False,
        multi_output: bool = False,
        input_number: int = 1,
        output_number: int = 1,
    ) -> None:
        self.uuid = uuid_lib.uuid4()
        self.multi_input = multi_input
        self.multi_output = multi_output
        assert (input_number == 1 and not multi_input) or (
            input_number > 1 and multi_input
        )
        assert (output_number == 1 and not multi_output) or (
            output_number > 1 and multi_output
        )
        self.input_number = input_number
        self.output_number = output_number

    def __repr__(self) -> str:
        return f"Node(name={self.__class__.__name__})"


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
        self.edges = {}  # directed edges

    def pseudo_element(self, **kwargs) -> PseudoElement:
        return PseudoElement(graph=self, **kwargs)

    def assign_node(self, node: Node):
        assert node.uuid not in self.nodes
        self.nodes[node.uuid] = node

    def add_connection(self, node1: Node, node2: Node):
        assert node1.uuid in self.nodes
        assert node2.uuid in self.nodes
        if node1.uuid not in self.edges:
            self.edges[node1.uuid] = []
        self.edges[node1.uuid].append(node2.uuid)

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
        self.succs = None

    def _assign_cur_node(self, cur_node: Node):
        self.graph.assign_node(cur_node)
        self.cur_node = cur_node
        if self.succs is not None:
            for succ in self.succs:
                self.graph.add_connection(self.cur_node, succ)

    def _assign_successor(self, succ_nodes: Union[List[Node], Node]):
        if isinstance(succ_nodes, Node):
            succ_nodes = [succ_nodes]
        self.succs = []
        for succ_node in succ_nodes:
            new_one = self.graph.pseudo_element(cur_node=succ_node)
            self.succs.append(new_one)
            if self.cur_node is not None:
                self.graph.add_connection(self.cur_node, succ_node)
        if len(self.succs) == 1:
            return self.succs[0]
        return self.succs

    def filter(self, **kvargs) -> PseudoElement:
        return self._assign_successor(Filter(**kvargs))

    def map_(self, **kvargs) -> PseudoElement:
        return self._assign_successor(Map_(**kvargs))

    def reduce_by(self, **kvargs) -> PseudoElement:
        return self._assign_successor(ReduceBy(**kvargs))
