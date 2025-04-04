import inspect
from warnings import warn
import graphyflow.dataflow_ir as dfir
from typing import Dict, List, Tuple, Any


class Tracer:
    _tracer_id = 0  # for generating unique id

    def __init__(
        self,
        name=None,
        node_type="input",
        attr_name=None,
        operator=None,
        parents=None,
        value=None,
        pseudo_element=None,
    ):
        self._id = Tracer._tracer_id
        Tracer._tracer_id += 1
        self._name = name
        self._node_type = node_type
        self._attr_name = attr_name
        self._operator = operator
        self._parents = parents or []
        self._value = value
        self._pseudo_element = pseudo_element

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return Tracer(node_type="attr", attr_name=name, parents=[self])

    def __getitem__(self, index):
        assert type(index) == int
        return Tracer(node_type="idx", attr_name=index, parents=[self])

    def _bin_op(self, other, op):
        # Non-Tracer -> Constant
        if not isinstance(other, Tracer):
            other = Tracer(node_type="constant", value=other)
        return Tracer(node_type="operation", operator=op, parents=[self, other])

    def __add__(self, other):
        return self._bin_op(other, "+")

    def __radd__(self, other):
        return self._bin_op(other, "+")

    def __sub__(self, other):
        return self._bin_op(other, "-")

    def __rsub__(self, other):
        return self._bin_op(other, "-")

    def __mul__(self, other):
        return self._bin_op(other, "*")

    def __rmul__(self, other):
        return self._bin_op(other, "*")

    def __truediv__(self, other):
        return self._bin_op(other, "/")

    def __rtruediv__(self, other):
        return self._bin_op(other, "/")

    def __lt__(self, other):
        return self._bin_op(other, "<")

    def __gt__(self, other):
        return self._bin_op(other, ">")

    def __le__(self, other):
        return self._bin_op(other, "<=")

    def __ge__(self, other):
        return self._bin_op(other, ">=")

    def __eq__(self, other):
        return self._bin_op(other, "==")

    def __ne__(self, other):
        return self._bin_op(other, "!=")

    def to_dict(self):
        return {
            k: v
            for k, v in [
                ("type", self._node_type),
                ("name", self._name),
                ("attr", self._attr_name),
                ("operator", self._operator),
                ("value", self._value),
                ("pseudo_element", self._pseudo_element),
            ]
            if v is not None
        }


def parse_lambda(lambda_func):
    try:
        sig = inspect.signature(lambda_func)
        num_params = len(sig.parameters)
    except:
        raise RuntimeError("Lambda function analyze failed.")

    inputs = [Tracer(name=f"arg{i}", node_type="input") for i in range(num_params)]

    try:
        result = lambda_func(*inputs)
    except Exception as e:
        raise RuntimeError(f" {str(e)}")

    outputs = result if isinstance(result, (tuple, list)) else [result]

    all_nodes = []
    visited = set()

    def traverse(node):
        if node._id in visited:
            return
        visited.add(node._id)
        all_nodes.append(node)
        for parent in node._parents:
            traverse(parent)

    for node in outputs:
        traverse(node)

    edges = [(p._id, node._id) for node in all_nodes for p in node._parents]

    return {
        "nodes": {n._id: n.to_dict() for n in all_nodes},
        "edges": edges,
        "input_ids": [n._id for n in inputs],
        "output_ids": [n._id for n in outputs],
    }


def format_lambda(lambda_dict):
    nodes = lambda_dict["nodes"]
    node_strs = {}

    node_type_formats = {
        "input": lambda info: f"Input ({info.get('name', 'unnamed')})",
        "operation": lambda info: f"Operation ({info.get('operator', '?')})",
        "attr": lambda info: f"Attribute (.{info.get('attr', '?')})",
        "idx": lambda info: f"Index ([{info.get('attr', '?')}])",
        "constant": lambda info: f"Constant ({info.get('value', '?')})",
        "pseudo": lambda info: f"{info.get('pseudo_element', '?')}",
    }

    for node_id, node_info in sorted(nodes.items()):
        node_str = f"Node {node_id}: "
        node_type = node_info.get("type", "unknown")
        if node_type in node_type_formats:
            node_str += node_type_formats[node_type](node_info)
        else:
            node_str += str(node_info)
        if node_id in lambda_dict["input_ids"]:
            node_str += " [INPUT]"
        if node_id in lambda_dict["output_ids"]:
            node_str += " [OUTPUT]"
        node_strs[node_id] = node_str

    result = ["Lambda Repr: "]

    if len(lambda_dict["edges"]) == 0:
        result.extend(f"  {node_str}" for node_str in node_strs.values())
    else:
        max_src_len = max(len(node_strs[src]) for src, _ in lambda_dict["edges"])
        for src, dst in sorted(lambda_dict["edges"]):
            padding = " " * (max_src_len - len(node_strs[src]) + 2)
            result.append(f"  {node_strs[src]}{padding}→ {node_strs[dst]}")

    result.append(f"Input Nodes: {', '.join(map(str, lambda_dict['input_ids']))}")
    result.append(f"Output Nodes: {', '.join(map(str, lambda_dict['output_ids']))}")

    return "\n".join(result)


# TODO: fix the binop order problem
def lambda_to_dfir(
    lambda_dict: Dict[str, Any], input_types: List[dfir.DfirType]
) -> dfir.ComponentCollection:
    assert len(input_types) == len(lambda_dict["input_ids"])
    assert all(isinstance(t, dfir.DfirType) for t in input_types)
    is_parallel = isinstance(input_types[0], dfir.ArrayType) and len(input_types) == 1

    def translate_constant_val(node, pre_o_types):
        assert node["value"] is not None and type(node["value"]) in [bool, int, float]
        value_dfir_dict = {
            int: dfir.IntType(),
            bool: dfir.BoolType(),
            float: dfir.FloatType(),
        }
        dfir_type = value_dfir_dict[type(node["value"])]
        if is_parallel:
            dfir_type = dfir.ArrayType(dfir_type)
        return dfir.ConstantComponent(dfir_type, node["value"])

    def translate_bin_op(node, pre_o_types):
        assert node["operator"] in [
            "+",
            "-",
            "*",
            "/",
            "<",
            "<=",
            ">",
            ">=",
            "==",
            "!=",
        ]
        assert (
            len(pre_o_types) == 2 and pre_o_types[0] == pre_o_types[1]
        ), f"{pre_o_types} should be 2 equal types"
        op_dfir_dict = {
            "+": dfir.BinOp.ADD,
            "-": dfir.BinOp.SUB,
            "*": dfir.BinOp.MUL,
            "/": dfir.BinOp.DIV,
            "<": dfir.BinOp.LT,
            "<=": dfir.BinOp.LE,
            ">": dfir.BinOp.GT,
            ">=": dfir.BinOp.GE,
            "==": dfir.BinOp.EQ,
            "!=": dfir.BinOp.NE,
        }
        return dfir.BinOpComponent(op_dfir_dict[node["operator"]], pre_o_types[0])

    translate_dict = {
        "input": lambda node, pre_o_types: dfir.PlaceholderComponent(pre_o_types[0]),
        "attr": lambda node, pre_o_types: dfir.UnaryOpComponent(
            dfir.UnaryOp.GET_ATTR, pre_o_types[0]
        ),
        "idx": lambda node, pre_o_types: dfir.UnaryOpComponent(
            dfir.UnaryOp.SELECT, pre_o_types[0]
        ),
        "constant": translate_constant_val,
        "operation": translate_bin_op,
    }
    nodes, edges = lambda_dict["nodes"], lambda_dict["edges"]
    dfir_nodes = {}
    in_degree = {nid: len(list(dst for _, dst in edges if dst == nid)) for nid in nodes}
    start_queue = [nid for nid, deg in in_degree.items() if deg == 0]
    # delete the deg==0 nodes from in_degree
    for nid in start_queue:
        if nid in in_degree:
            del in_degree[nid]
    constants = [nid for nid in start_queue if nodes[nid]["type"] == "constant"]
    assert len(input_types) == len(start_queue) - len(constants)
    start_queue = [nid for nid in start_queue if nid not in constants]
    queue = []
    node_tmp_datas = {nid: [nid, [], {}] for nid in nodes.keys()}
    for nid in constants:
        queue.append([nid, [], {}])
    for name_id in range(len(input_types)):
        arg_name = f"arg{name_id}"
        target = [nid for nid in start_queue if nodes[nid]["name"] == arg_name]
        assert len(target) == 1
        queue.append(
            [target[0], [input_types[name_id]], {}]
        )  # node, prev_type, parent_ports
    while queue:
        nid, pre_o_types, p_ports = queue.pop(0)
        node_type = nodes[nid]["type"]
        dfir_nodes[nid] = translate_dict[node_type](nodes[nid], pre_o_types)
        dfir_nodes[nid].connect(p_ports)
        out_type = dfir_nodes[nid].output_type
        succ_node_ids = [dst for src, dst in edges if src == nid]
        for succ_nid in succ_node_ids:
            node_tmp_datas[succ_nid][1].append(out_type)
            for p in dfir_nodes[nid].out_ports:
                in_id = 0
                while f"i_{in_id}" in node_tmp_datas[succ_nid][2]:
                    in_id += 1
                node_tmp_datas[succ_nid][2][f"i_{in_id}"] = p
            in_degree[succ_nid] -= 1
            if in_degree[succ_nid] == 0:
                queue.append(node_tmp_datas[succ_nid])
                del in_degree[succ_nid]
    inputs = sum([dfir_nodes[nid].in_ports for nid in lambda_dict["input_ids"]], [])
    outputs = sum([dfir_nodes[nid].out_ports for nid in lambda_dict["output_ids"]], [])
    return dfir.ComponentCollection(list(dfir_nodes.values()), inputs, outputs)


if __name__ == "__main__":
    func = lambda a, b: a.x * a.y + 2 / b
    graph = parse_lambda(func)
    the_dfir = lambda_to_dfir(graph, [dfir.SpecialType("node"), dfir.IntType()])

    if graph:
        print("Nodes:")
        for nid, info in graph["nodes"].items():
            print(f"Node {nid}: {info}")

        print("\nEdges:")
        for src, dst in graph["edges"]:
            print(f"{src} -> {dst}")

        print("\nInput Node(s):", graph["input_ids"])
        print("Output Node(s)", graph["output_ids"])

    print(the_dfir)
