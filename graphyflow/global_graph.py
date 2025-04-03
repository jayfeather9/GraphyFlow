from __future__ import annotations
from typing import Callable, List, Optional, Dict
from uuid import UUID
import uuid as uuid_lib
from graphyflow.graph_types import *
from graphyflow.lambda_func import parse_lambda, Tracer, format_lambda
import graphyflow.dataflow_ir as dfir


class Node:
    """每个计算图的节点可以单或多输入，但是只有一样输出。任何将该节点作为pred_nodes的节点都将接受到该节点的输出的一份复制"""

    def __init__(
        self,
    ) -> None:
        self.uuid = uuid_lib.uuid4()
        self.is_simple = False
        self._pred_nodes = []
        self._lambda_funcs = []

    def set_pred_nodes(self, pred_nodes: List[Node]) -> None:
        self._pred_nodes = pred_nodes

    @property
    def class_name(self) -> str:
        return self.__class__.__name__

    @property
    def preds(self) -> List[Node]:
        return self._pred_nodes

    @property
    def lambdas(self) -> List[Dict]:
        return self._lambda_funcs

    def __repr__(self) -> str:
        formatted_lambdas = "\n".join(
            format_lambda(lambda_func) for lambda_func in self._lambda_funcs
        )
        return f"Node(name={self.class_name}, preds={[node.class_name for node in self._pred_nodes]}, lambdas = {formatted_lambdas})"


class Inputer(Node):
    def __init__(self, input_type: Union[BasicNode, BasicEdge]):
        self.input_type = input_type
        super().__init__()

    def to_dfir(self) -> dfir.DfirNode:
        input_types = []
        if isinstance(self.input_type, BasicNode):
            input_types.append(dfir.SpecialType("node"))
        elif isinstance(self.input_type, BasicEdge):
            input_types.append(dfir.SpecialType("edge"))
        else:
            raise RuntimeError("Input type must be BasicNode or BasicEdge")
        input_types.extend(
            [data_type.to_dfir() for data_type in self.input_type.data_types]
        )
        return dfir.IOComponent(
            dfir.IOComponent.IOType.INPUT, dfir.TupleType(input_types)
        )


class Updater(Node):
    def __init__(self, type_, attr):
        assert type_ in ["node", "edge"]
        self.type_ = type_
        self.attr = attr
        super().__init__()


class GetLength(Node):
    def __init__(self) -> None:
        super().__init__()
        self.is_simple = True

    def to_dfir(self, input_type: dfir.DfirType) -> dfir.DfirNode:
        return dfir.UnaryOpComponent(dfir.UnaryOp.GET_LENGTH, input_type)


class Filter(Node):
    def __init__(self, filter_func: Callable[[List[DataElement]], DataElement]):
        self.filter_func = parse_lambda(filter_func)
        super().__init__()
        self.lambdas.append(self.filter_func)


class Map_(Node):
    def __init__(self, map_func: Callable[[List[DataElement]], DataElement]):
        self.map_func = parse_lambda(map_func)
        super().__init__()
        self.lambdas.append(self.map_func)


class ReduceBy(Node):
    def __init__(
        self,
        reduce_key: Callable[[List[DataElement]], DataElement],
        reduce_method: Callable[[List[DataElement]], DataElement],
    ):
        self.reduce_key = parse_lambda(reduce_key)
        self.reduce_method = parse_lambda(reduce_method)
        super().__init__()
        self.lambdas.extend([self.reduce_key, self.reduce_method])


class GlobalGraph:
    def __init__(self, properties: Optional[Dict[str, Dict[str, Any]]] = None):
        self.input_nodes = []  # by UUID
        self.nodes = {}  # Each node represents a method, nodes = {uuid: node}
        self.node_properties = {}
        self.edge_properties = {}
        self.added_input = False
        if properties:
            self.handle_properties(properties)

    def handle_properties(self, properties: Dict[str, Dict[str, Any]]):
        assert not self.added_input, "Properties must be set before adding input"
        for prop_name, prop_info in properties.items():
            assert prop_name in ["node", "edge"]
            if prop_name == "node":
                self.node_properties = prop_info
            else:
                self.edge_properties = prop_info

    def pseudo_element(self, **kwargs) -> PseudoElement:
        return PseudoElement(graph=self, **kwargs)

    def add_input(self, type_: str, **kwargs) -> PseudoElement:
        assert type_ in ["edge", "node"]
        self.added_input = True
        property_infos = (
            self.node_properties if type_ == "node" else self.edge_properties
        )
        data_types = [BasicData(prop_type) for _, prop_type in property_infos.items()]
        return self.pseudo_element(
            cur_node=Inputer(
                input_type=(
                    BasicNode(data_types) if type_ == "node" else BasicEdge(data_types)
                )
            )
        )

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

    def __repr__(self) -> str:
        return f"PseudoElement(cur_node={self.cur_node.class_name})"

    def _assign_cur_node(self, cur_node: Node):
        assert self.cur_node is None
        self.graph.assign_node(cur_node)
        self.cur_node = cur_node

    def _assign_successor(self, succ_node: Node):
        if self.cur_node:
            succ_node.set_pred_nodes([self.cur_node])
        return self.graph.pseudo_element(cur_node=succ_node)

    def length(self) -> PseudoElement:
        return self._assign_successor(GetLength())

    def filter(self, **kvargs) -> PseudoElement:
        return self._assign_successor(Filter(**kvargs))

    def map_(self, **kvargs) -> PseudoElement:
        return self._assign_successor(Map_(**kvargs))

    def reduce_by(self, **kvargs) -> PseudoElement:
        return self._assign_successor(ReduceBy(**kvargs))

    def to_tracer(self) -> Tracer:
        return Tracer(node_type="pseudo", pseudo_element=self)
