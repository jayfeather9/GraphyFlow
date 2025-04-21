import dataflow_ir as dfir
from global_graph import GlobalGraph
from typing import Dict, List, Tuple, Any, Callable, Optional


def _simulate_component(
    comp: dfir.Component, inputs: Dict[str, Any] = {}, attrs: Dict[str, Any] = {}
):
    # TODO: add parallel support
    support_components = [
        dfir.BinOpComponent,
        dfir.UnaryOpComponent,
        dfir.ConstantComponent,
        dfir.CopyComponent,
        dfir.GatherComponent,
        dfir.ScatterComponent,
        dfir.ConditionalComponent,
        dfir.CollectComponent,
        dfir.PlaceholderComponent,
    ]
    assert isinstance(comp, support_components)
    assert all(p.name in inputs.keys() for p in comp.ports)
    if isinstance(comp, dfir.PlaceholderComponent):
        return {"o_0": inputs["i_0"]}
    if isinstance(comp, dfir.ConstantComponent):
        return {"o_0": comp.value}
    if isinstance(comp, dfir.CopyComponent):
        return {"o_0": inputs["i_0"], "o_1": inputs["i_0"]}
    if isinstance(comp, dfir.GatherComponent):
        return {"o_0": sum([[inputs[p.name]] for p in comp.in_ports], [])}
    if isinstance(comp, dfir.ScatterComponent):
        return {p.name: inputs["i_0"][i] for i, p in enumerate(comp.out_ports)}
    if isinstance(comp, dfir.BinOpComponent):
        binop_calcs = {
            dfir.BinOp.ADD: lambda x, y: x + y,
            dfir.BinOp.SUB: lambda x, y: x - y,
            dfir.BinOp.MUL: lambda x, y: x * y,
            dfir.BinOp.DIV: lambda x, y: x / y,
            dfir.BinOp.AND: lambda x, y: x & y,
            dfir.BinOp.OR: lambda x, y: x | y,
            dfir.BinOp.EQ: lambda x, y: x == y,
            dfir.BinOp.NE: lambda x, y: x != y,
            dfir.BinOp.LT: lambda x, y: x < y,
            dfir.BinOp.GT: lambda x, y: x > y,
            dfir.BinOp.LE: lambda x, y: x <= y,
            dfir.BinOp.GE: lambda x, y: x >= y,
            dfir.BinOp.MIN: lambda x, y: min(x, y),
            dfir.BinOp.MAX: lambda x, y: max(x, y),
        }
        return {"o_0": binop_calcs[comp.op](inputs["i_0"], inputs["i_1"])}
    if isinstance(comp, dfir.UnaryOpComponent):
        unaryop_calcs = {
            dfir.UnaryOp.NOT: lambda x: not x,
            dfir.UnaryOp.NEG: lambda x: -x,
            dfir.UnaryOp.CAST_BOOL: lambda x: bool(x),
            dfir.UnaryOp.CAST_INT: lambda x: int(x),
            dfir.UnaryOp.CAST_FLOAT: lambda x: float(x),
            dfir.UnaryOp.SELECT: lambda x: x[comp.select_index],
            dfir.UnaryOp.GET_LENGTH: lambda x: len(x),
            dfir.UnaryOp.GET_ATTR: lambda x: None,  # TODO
        }
        return {"o_0": unaryop_calcs[comp.op](inputs["i_0"])}
    # TODO: conditional, collect


class DfirSimulator:
    def __init__(self, dfirs: dfir.ComponentCollection, g: GlobalGraph) -> None:
        self.dfirs = dfirs
        self.nodes = {}
        self.node_properties = g.node_properties
        self.edges = {}
        self.edge_properties = g.edge_properties

    def add_nodes(self, nodes: List[int], props: Dict[int, Dict[str, Any]]):
        assert len(set(nodes)) == len(nodes)
        self.nodes = {i: {} for i in nodes}
        for i in self.nodes.keys():
            assert i in props.keys()
            for prop in self.node_properties:
                if prop == "id":
                    self.nodes[i][prop] = i
                    continue
                assert prop in props[i].keys()
                self.nodes[i][prop] = props[i][prop]

    def add_edges(
        self, edges: Dict[int, Tuple[int, int]], props: Dict[int, Dict[str, Any]]
    ):
        assert len(set(edges.keys())) == len(edges.keys())
        self.edges = {i: {"src": v[0], "dst": v[1]} for i, v in edges.items()}
        for i in self.edges.keys():
            assert i in props.keys()
            for prop in self.edge_properties:
                if prop in ["src", "dst"]:
                    continue
                assert prop in props[i].keys()
                self.edges[i][prop] = props[i][prop]
