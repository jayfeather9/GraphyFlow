from __future__ import annotations
from typing import Callable, List, Optional, Dict
from uuid import UUID
import uuid as uuid_lib
from graphyflow.graph_types import *
from graphyflow.lambda_func import parse_lambda, Tracer, format_lambda, lambda_to_dfir
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
        formatted_lambdas = "\n".join(format_lambda(lambda_func) for lambda_func in self._lambda_funcs)
        return f"Node(name={self.class_name}, preds={[node.class_name for node in self._pred_nodes]}, lambdas = {formatted_lambdas})"


class Inputer(Node):
    def __init__(self, input_type: Union[BasicNode, BasicEdge]):
        self.input_type = input_type
        super().__init__()

    def to_dfir(
        self,
        placeholder: List[dfir.DfirType],
        props: Tuple[Dict[str, Any], Dict[str, Any]],
    ) -> dfir.DfirNode:
        if isinstance(self.input_type, BasicNode):
            input_type = dfir.SpecialType("node")
        elif isinstance(self.input_type, BasicEdge):
            input_type = dfir.SpecialType("edge")
        else:
            raise RuntimeError("Input type must be BasicNode or BasicEdge")
        input_type = dfir.ArrayType(input_type)
        io_comp = dfir.IOComponent(dfir.IOComponent.IOType.INPUT, input_type)
        return dfir.ComponentCollection([io_comp], [], [io_comp.ports[0]])


class Updater(Node):
    def __init__(self, type_: str, attrs: List[str]):
        assert type_ in ["node", "edge"]
        self.type_ = type_
        self.attrs = attrs
        super().__init__()


class GetLength(Node):
    def __init__(self) -> None:
        super().__init__()
        self.is_simple = True

    def to_dfir(self, input_type: dfir.DfirType) -> dfir.DfirNode:
        u_comp = dfir.UnaryOpComponent(dfir.UnaryOp.GET_LENGTH, input_type)
        return dfir.ComponentCollection([u_comp], [u_comp.in_ports[0]], [u_comp.out_ports[0]])


class Filter(Node):
    def __init__(self, filter_func: Callable[[List[DataElement]], DataElement]):
        self.filter_func = parse_lambda(filter_func)
        super().__init__()
        self._lambda_funcs.append(self.filter_func)

    def to_dfir(
        self, input_type: dfir.DfirType, props: Tuple[Dict[str, Any], Dict[str, Any]]
    ) -> dfir.ComponentCollection:
        assert isinstance(input_type, dfir.ArrayType)
        assert len(self._lambda_funcs) == 1
        if len(self._lambda_funcs[0]["input_ids"]) == 1:
            dfirs = lambda_to_dfir(self._lambda_funcs[0], [input_type], props[0], props[1])
            assert len(dfirs.outputs) == 1
            assert dfirs.outputs[0].data_type == dfir.ArrayType(dfir.BoolType())

            copy_comp = dfir.CopyComponent(input_type)
            cond_comp = dfir.ConditionalComponent(input_type)
            collect_comp = dfir.CollectComponent(cond_comp.output_type)

            assert len(dfirs.inputs) == 1
            copy_ports = {"o_0": dfirs.inputs[0]}
            dfirs.add_front(copy_comp, copy_ports)

            cond_ports = {"i_data": copy_comp.ports[2], "i_cond": dfirs.outputs[0]}
            dfirs.add_back(cond_comp, cond_ports)

            collect_ports = {"i_0": cond_comp.ports[2]}
            dfirs.add_back(collect_comp, collect_ports)

            return dfirs
        else:
            scatter_comp = dfir.ScatterComponent(input_type)
            scatter_out_types = [p.data_type for p in scatter_comp.ports[1:]]
            dfirs = lambda_to_dfir(self._lambda_funcs[0], scatter_out_types, props[0], props[1])
            assert len(dfirs.outputs) == 1
            assert dfirs.outputs[0].data_type == dfir.ArrayType(dfir.BoolType())

            copy_comp = dfir.CopyComponent(input_type)
            cond_comp = dfir.ConditionalComponent(input_type)
            collect_comp = dfir.CollectComponent(cond_comp.output_type)

            dfirs.add_front(
                scatter_comp,
                {f"o_{i}": dfirs.inputs[i] for i in range(len(scatter_out_types))},
            )

            copy_ports = {"o_0": scatter_comp.ports[0]}
            dfirs.add_front(copy_comp, copy_ports)

            cond_ports = {"i_data": copy_comp.ports[2], "i_cond": dfirs.outputs[0]}
            dfirs.add_back(cond_comp, cond_ports)

            collect_ports = {"i_0": cond_comp.ports[2]}
            dfirs.add_back(collect_comp, collect_ports)

            return dfirs


class Map_(Node):
    def __init__(self, map_func: Callable[[List[DataElement]], DataElement]):
        self.map_func = parse_lambda(map_func)
        super().__init__()
        self.lambdas.append(self.map_func)

    def to_dfir(
        self, input_type: dfir.DfirType, props: Tuple[Dict[str, Any], Dict[str, Any]]
    ) -> dfir.ComponentCollection:
        assert isinstance(input_type, dfir.ArrayType), f"{input_type} is not an array type"
        assert len(self._lambda_funcs) == 1
        if len(self._lambda_funcs[0]["input_ids"]) == 1:
            dfirs = lambda_to_dfir(self._lambda_funcs[0], [input_type], props[0], props[1])
        else:
            scatter_comp = dfir.ScatterComponent(input_type)
            scatter_out_types = [p.data_type for p in scatter_comp.ports[1:]]
            dfirs = lambda_to_dfir(self._lambda_funcs[0], scatter_out_types, props[0], props[1])
            dfirs.add_front(
                scatter_comp,
                {f"o_{i}": dfirs.inputs[i] for i in range(len(scatter_out_types))},
            )
        if len(dfirs.outputs) > 1:
            gather_comp = dfir.GatherComponent(dfirs.output_types)
            dfirs.add_back(
                gather_comp,
                {f"i_{i}": dfirs.outputs[i] for i in range(len(dfirs.outputs))},
            )
        return dfirs


class ReduceBy(Node):
    """
    reduce_key: x -> y, determine according to which element (is same) to group by and reduce, if want to reduce all, give lambda x: 1
    reduce_transform: x -> y, determine which element should be reduced
    reduce_method: a, b -> c, reduce two elements, should be commutative and associative
    """

    def __init__(
        self,
        reduce_key: Callable[[List[DataElement]], DataElement],
        reduce_transform: Callable[[List[DataElement]], DataElement],
        reduce_method: Callable[[List[DataElement]], DataElement],
    ):
        self.reduce_key = parse_lambda(reduce_key)
        self.reduce_transform = parse_lambda(reduce_transform)
        self.reduce_method = parse_lambda(reduce_method)
        super().__init__()
        self.lambdas.extend([self.reduce_key, self.reduce_transform, self.reduce_method])

    def to_dfir(
        self, input_type: dfir.DfirType, props: Tuple[Dict[str, Any], Dict[str, Any]]
    ) -> dfir.ComponentCollection:
        assert isinstance(input_type, dfir.ArrayType)
        assert len(self._lambda_funcs) == 3

        element_type = input_type.type_
        if len(self.reduce_key["input_ids"]) == 1:
            reduce_key_dfirs = lambda_to_dfir(self.reduce_key, [element_type], props[0], props[1])
        else:
            scatter_comp = dfir.ScatterComponent(element_type)
            scatter_out_types = [p.data_type for p in scatter_comp.ports[1:]]
            reduce_key_dfirs = lambda_to_dfir(self.reduce_key, scatter_out_types, props[0], props[1])
            reduce_key_dfirs.add_front(
                scatter_comp,
                {f"o_{i}": reduce_key_dfirs.inputs[i] for i in range(len(scatter_out_types))},
            )
        assert len(reduce_key_dfirs.inputs) == 1
        assert len(reduce_key_dfirs.outputs) == 1
        reduce_key_in_port = reduce_key_dfirs.inputs[0]
        reduce_key_out_port = reduce_key_dfirs.outputs[0]
        reduce_key_out_type = reduce_key_dfirs.output_types[0]

        if len(self.reduce_transform["input_ids"]) == 1:
            reduce_transform_dfirs = lambda_to_dfir(
                self.reduce_transform,
                [element_type],
                props[0],
                props[1],
                scatter_outputs=False,
            )
        else:
            scatter_comp = dfir.ScatterComponent(element_type)
            scatter_out_types = [p.data_type for p in scatter_comp.ports[1:]]
            reduce_transform_dfirs = lambda_to_dfir(
                self.reduce_transform,
                scatter_out_types,
                props[0],
                props[1],
                scatter_outputs=False,
            )
            reduce_transform_dfirs.add_front(
                scatter_comp,
                {f"o_{i}": reduce_transform_dfirs.inputs[i] for i in range(len(scatter_out_types))},
            )
        assert len(reduce_transform_dfirs.inputs) == 1
        assert len(reduce_transform_dfirs.outputs) == 1
        transform_input_port = reduce_transform_dfirs.inputs[0]
        transform_output_port = reduce_transform_dfirs.outputs[0]
        transform_type = transform_output_port.data_type

        assert len(self.reduce_method["input_ids"]) == 2
        reduce_method_dfirs = lambda_to_dfir(
            self.reduce_method,
            [transform_type, transform_type],
            props[0],
            props[1],
            scatter_outputs=False,
        )
        assert len(reduce_method_dfirs.inputs) == 2
        assert len(reduce_method_dfirs.outputs) == 1
        accumulate_input_port_0 = reduce_method_dfirs.inputs[0]
        accumulate_input_port_1 = reduce_method_dfirs.inputs[1]
        accumulate_output_port = reduce_method_dfirs.outputs[0]

        dfirs = reduce_transform_dfirs
        dfirs = dfirs.concat(reduce_method_dfirs, [])
        dfirs = dfirs.concat(reduce_key_dfirs, [])
        reduce_comp = dfir.ReduceComponent(input_type, transform_type, reduce_key_out_type)
        dfirs.add_front(
            reduce_comp,
            {
                "o_reduce_key_in": reduce_key_in_port,
                "o_reduce_transform_in": transform_input_port,
                "o_reduce_unit_start_0": accumulate_input_port_0,
                "o_reduce_unit_start_1": accumulate_input_port_1,
            },
        )
        dfirs.add_back(
            reduce_comp,
            {
                "i_reduce_key_out": reduce_key_out_port,
                "i_reduce_transform_out": transform_output_port,
                "i_reduce_unit_end": accumulate_output_port,
            },
        )
        return dfirs


class Merge(Node):
    def __init__(self):
        super().__init__()

    def to_dfir(
        self, input_type: dfir.DfirType, props: Tuple[Dict[str, Any], Dict[str, Any]]
    ) -> dfir.ComponentCollection:
        pass


class Append(Node):
    def __init__(self):
        super().__init__()

    def to_dfir(
        self, input_type: dfir.DfirType, props: Tuple[Dict[str, Any], Dict[str, Any]]
    ) -> dfir.ComponentCollection:
        pass


class GlobalGraph:
    def __init__(self, properties: Optional[Dict[str, Dict[str, Any]]] = None):
        self.input_nodes = []  # by UUID
        self.nodes = {}  # Each node represents a method, nodes = {uuid: node}
        self.node_properties = {}
        self.edge_properties = {}
        self.added_input = False
        self.input_names = ["edge", "node"]
        if properties:
            self.handle_properties(properties)

    def handle_properties(self, properties: Dict[str, Dict[str, Any]]):
        assert not self.added_input, "Properties must be set before adding any input"
        if "edge" not in properties.keys():
            properties["edge"] = {}
        if "node" not in properties.keys():
            properties["node"] = {}
        properties["edge"]["src"] = dfir.SpecialType("node")
        properties["edge"]["dst"] = dfir.SpecialType("node")
        properties["node"]["id"] = dfir.IntType()
        for prop_name, prop_info in properties.items():
            assert prop_name in ["node", "edge"]
            if prop_name == "node":
                self.node_properties = prop_info
            else:
                self.edge_properties = prop_info

    def pseudo_element(self, **kwargs) -> PseudoElement:
        return PseudoElement(graph=self, **kwargs)

    def add_graph_input(self, type_: str, **kwargs) -> PseudoElement:
        assert type_ in ["edge", "node"]
        self.added_input = True
        property_infos = self.node_properties if type_ == "node" else self.edge_properties
        data_types = [BasicData(prop_type) for _, prop_type in property_infos.items()]
        return self.pseudo_element(
            cur_node=Inputer(input_type=(BasicNode(data_types) if type_ == "node" else BasicEdge(data_types)))
        )

    def add_input(self, name: str, type_: str):
        assert name not in self.input_names
        self.input_names.append(name)
        self.input_types[name] = type_
        return self.pseudo_element(cur_node=Inputer(input_type=type_))

    def assign_node(self, node: Node):
        assert node.uuid not in self.nodes
        self.nodes[node.uuid] = node

    def finish_init(self, datas: PseudoElement, attrs: Dict[str, List[str]]):
        for attr_name, attr_types in attrs.items():
            datas._assign_successor(Updater(attr_name, attr_types))

    def finish_iter(
        self,
        datas: PseudoElement,
        attrs: Dict[str, List[str]],
        end_marker: PseudoElement,
    ):
        for attr_name, attr_types in attrs.items():
            datas._assign_successor(Updater(attr_name, attr_types))

    def topo_sort_nodes(self) -> List[Node]:
        result = []
        waitings = list(self.nodes.items())
        while waitings:
            new_ones = []
            for nid, n in waitings:
                if all((pred.uuid, pred) in result for pred in n.preds):
                    new_ones.append((nid, n))
            waitings = [w for w in waitings if w not in new_ones]
            result.extend(new_ones)
        return result

    def to_dfir(self) -> dfir.ComponentCollection:
        nodes = self.topo_sort_nodes()
        node_dfirs = {}
        added_nodes = []
        for nid, n in nodes:
            input_types = []
            if len(n.preds) > 0:
                assert len(n.preds) == 1
                input_types.extend(node_dfirs[n.preds[0].uuid].output_types)
                assert len(input_types) == 1, f"{n.class_name} {input_types}"
                input_types = input_types[0]
            node_dfirs[nid] = n.to_dfir(input_types, (self.node_properties, self.edge_properties))
            if len(n.preds) > 0:
                node_dfirs[nid] = node_dfirs[n.preds[0].uuid].concat(
                    node_dfirs[nid],
                    [
                        (
                            node_dfirs[n.preds[0].uuid].outputs[0],
                            node_dfirs[nid].inputs[0],
                        )
                    ],
                )
                added_nodes.append(n.preds[0].uuid)
        result_dfirs = [d for nid, d in node_dfirs.items() if nid not in added_nodes]
        return result_dfirs

    def __repr__(self) -> str:
        return f"GlobalGraph(nodes={self.nodes})"


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

    def merge(self, other: PseudoElement) -> PseudoElement:
        assert self.cur_node is not None and other.cur_node is not None
        self.cur_node.set_pred_nodes([other.cur_node])
        return self.graph.pseudo_element(cur_node=Merge())

    def append(self, other: PseudoElement) -> PseudoElement:
        assert self.cur_node is not None and other.cur_node is not None
        self.cur_node.set_pred_nodes([other.cur_node])
        return self.graph.pseudo_element(cur_node=Append())

    def to_tracer(self) -> Tracer:
        return Tracer(node_type="pseudo", pseudo_element=self)
