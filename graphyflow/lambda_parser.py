import inspect
from warnings import warn


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
    ):
        self.id = Tracer._id
        Tracer._id += 1
        self.name = name
        self.node_type = node_type
        self.attr_name = attr_name
        self.operator = operator
        self.parents = parents or []
        self.value = value

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return Tracer(node_type="attr", attr_name=name, parents=[self])

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
