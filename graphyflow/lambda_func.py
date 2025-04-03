import inspect
from warnings import warn
from graphyflow import dataflow_ir as dfir
from typing import Dict, List, Tuple, Any


class Tracer:
    _id = 0  # for generating unique id

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
        self.id = Tracer._id
        Tracer._id += 1
        self.name = name
        self.node_type = node_type
        self.attr_name = attr_name
        self.operator = operator
        self.parents = parents or []
        self.value = value
        self.pseudo_element = pseudo_element

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

    def __sub__(self, other):
        return self._bin_op(other, "-")

    def __mul__(self, other):
        return self._bin_op(other, "*")

    def __truediv__(self, other):
        return self._bin_op(other, "/")

    def __radd__(self, other):
        return self._bin_op(other, "+")

    def __rsub__(self, other):
        return self._bin_op(other, "-")

    def __rmul__(self, other):
        return self._bin_op(other, "*")

    def __rtruediv__(self, other):
        return self._bin_op(other, "/")

    def to_dict(self):
        return {
            k: v
            for k, v in [
                ("type", self.node_type),
                ("name", self.name),
                ("attr", self.attr_name),
                ("operator", self.operator),
                ("value", self.value),
                ("pseudo_element", self.pseudo_element),
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
        if node.id in visited:
            return
        visited.add(node.id)
        all_nodes.append(node)
        for parent in node.parents:
            traverse(parent)

    for node in outputs:
        traverse(node)

    edges = [(p.id, node.id) for node in all_nodes for p in node.parents]

    return {
        "nodes": {n.id: n.to_dict() for n in all_nodes},
        "edges": edges,
        "input_ids": [n.id for n in inputs],
        "output_ids": [n.id for n in outputs],
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


def lambda_to_dfir(lambda_dict):
    pass


if __name__ == "__main__":
    func = lambda a, b: a.x * a.y + 2 / b
    graph = parse_lambda(func)

    if graph:
        print("Nodes:")
        for nid, info in graph["nodes"].items():
            print(f"Node {nid}: {info}")

        print("\nEdges:")
        for src, dst in graph["edges"]:
            print(f"{src} -> {dst}")

        print("\nInput Node(s):", graph["input_ids"])
        print("Output Node(s)", graph["output_ids"])
