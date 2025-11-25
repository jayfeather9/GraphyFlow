from typing import Any, Callable, List, Optional, Dict
import inspect
from graphyflow import middleend_ir as mir


# Constants
INFINITY = float("inf")
ALL_EDGES = "ALL_EDGES"


# Data type classes
class DataType:
    """Base class for data types."""

    pass


class Int(DataType):
    """Integer data type with configurable width."""

    def __init__(self, width: int = 32, default: Any = 0):
        self.width = width
        self.default = default


# Operation Tracer - captures operations without executing
class OpTracer:
    """Traces operations in lambda functions without executing them."""

    _id_counter = 0

    def __init__(
        self,
        op_type: str,
        name: str = None,
        value: Any = None,
        operands: List["OpTracer"] = None,
        attr: str = None,
    ):
        self.id = OpTracer._id_counter
        OpTracer._id_counter += 1
        self.op_type = op_type  # 'input', 'attr', 'const', 'binop', 'unary'
        self.name = name
        self.value = value
        self.operands = operands or []
        self.attr = attr

    # Attribute access (e.g., e.src, e.dst, e.weight, node.dist)
    def __getattr__(self, name: str):
        if name in ["id", "op_type", "name", "value", "operands", "attr"]:
            return object.__getattribute__(self, name)
        return OpTracer("attr", attr=name, operands=[self])

    # Binary operations
    def __add__(self, other):
        return self._binary_op("+", other)

    def __radd__(self, other):
        return self._binary_op("+", other, reverse=True)

    def __sub__(self, other):
        return self._binary_op("-", other)

    def __rsub__(self, other):
        return self._binary_op("-", other, reverse=True)

    def __mul__(self, other):
        return self._binary_op("*", other)

    def __rmul__(self, other):
        return self._binary_op("*", other, reverse=True)

    def __truediv__(self, other):
        return self._binary_op("/", other)

    def __rtruediv__(self, other):
        return self._binary_op("/", other, reverse=True)

    def __lt__(self, other):
        return self._binary_op("<", other)

    def __le__(self, other):
        return self._binary_op("<=", other)

    def __gt__(self, other):
        return self._binary_op(">", other)

    def __ge__(self, other):
        return self._binary_op(">=", other)

    def __eq__(self, other):
        return self._binary_op("==", other)

    def __ne__(self, other):
        return self._binary_op("!=", other)

    def _binary_op(self, op: str, other, reverse: bool = False):
        if not isinstance(other, OpTracer):
            other = OpTracer("const", value=other)
        operands = [other, self] if reverse else [self, other]
        return OpTracer("binop", name=op, operands=operands)

    def __repr__(self):
        return self._to_str()

    def _to_str(self, indent: int = 0) -> str:
        prefix = "  " * indent
        if self.op_type == "input":
            return f"{prefix}Input({self.name})"
        elif self.op_type == "const":
            return f"{prefix}Const({self.value})"
        elif self.op_type == "attr":
            return f"{prefix}Attr(.{self.attr}, {self.operands[0]._to_str(0)})"
        elif self.op_type == "binop":
            left_str = self.operands[0]._to_str(indent + 1)
            right_str = self.operands[1]._to_str(indent + 1)
            return f"{prefix}BinOp({self.name}):\n{left_str}\n{right_str}"
        else:
            return f"{prefix}Unknown({self.op_type})"

    def print_tree(self):
        """Print the operation tree in a readable format."""
        print(self._to_str())


# Built-in functions that return OpTracers
def min(a, b):
    """Min function for use in reduce and update operations."""
    if not isinstance(a, OpTracer):
        a = OpTracer("const", value=a)
    if not isinstance(b, OpTracer):
        b = OpTracer("const", value=b)
    return OpTracer("binop", name="min", operands=[a, b])


class LambdaExpr:
    """Stores a captured lambda function with its traced operations."""

    def __init__(self, func: Callable, inputs: List[OpTracer], output: OpTracer):
        self.func = func
        self.inputs = inputs
        self.output = output

    def __repr__(self):
        return f"Lambda({len(self.inputs)} args) -> {self.output}"

    def print_operations(self):
        """Print the lambda function's operations."""
        print(f"Lambda with {len(self.inputs)} input(s):")
        for i, inp in enumerate(self.inputs):
            print(f"  Input {i}: {inp.name}")
        print(f"Output:")
        self.output.print_tree()


class MapExpr:
    """Represents a map operation with traced lambda."""

    def __init__(self, inputs: List, lambda_expr: LambdaExpr):
        self.inputs = inputs
        self.lambda_expr = lambda_expr

    def __repr__(self):
        return f"Map({self.lambda_expr})"

    def print_details(self):
        """Print detailed map operation."""
        print("=" * 50)
        print("MAP OPERATION")
        print("=" * 50)
        self.lambda_expr.print_operations()
        print()

    def to_ir(self, input_stream: mir.Stream, node_props: Dict = None, edge_props: Dict = None) -> mir.MapIR:
        """Convert this map expression to middle-end IR."""
        # Unwrap array type for lambda input
        if isinstance(input_stream.data_type, mir.ArrayType):
            element_type = input_stream.data_type.element_type
        else:
            element_type = input_stream.data_type

        ir_nodes, ir_inputs, ir_output = _convert_lambda_to_ir(
            self.lambda_expr, [element_type], node_props, edge_props
        )
        return mir.MapIR(input_stream, ir_nodes, ir_inputs, ir_output)


class ReduceExpr:
    """Represents a reduce operation with traced lambda."""

    def __init__(self, keys, values: List, lambda_expr: LambdaExpr, output_with_key: bool):
        self.keys = keys
        self.values = values
        self.lambda_expr = lambda_expr
        self.output_with_key = output_with_key

    def __repr__(self):
        return f"Reduce(output_with_key={self.output_with_key}, {self.lambda_expr})"

    def print_details(self):
        """Print detailed reduce operation."""
        print("=" * 50)
        print("REDUCE OPERATION")
        print("=" * 50)
        print(f"Output with key: {self.output_with_key}")
        self.lambda_expr.print_operations()
        print()

    def to_ir(
        self,
        key_stream: mir.Stream,
        value_streams: List[mir.Stream],
        node_props: Dict = None,
        edge_props: Dict = None,
    ) -> mir.ReduceIR:
        """Convert this reduce expression to middle-end IR."""
        # Reduce lambda takes 2 args of the unwrapped value type
        value_type = value_streams[0].data_type
        if isinstance(value_type, mir.ArrayType):
            value_type = value_type.element_type

        input_types = [value_type, value_type]
        ir_nodes, ir_inputs, ir_output = _convert_lambda_to_ir(
            self.lambda_expr, input_types, node_props, edge_props
        )
        return mir.ReduceIR(key_stream, value_streams, ir_nodes, ir_inputs, ir_output, self.output_with_key)


def _trace_lambda(func: Callable, num_args: int = None) -> LambdaExpr:
    """Trace a lambda function by executing it with OpTracers."""
    if num_args is None:
        sig = inspect.signature(func)
        num_args = len(sig.parameters)

    # Create input tracers
    inputs = [OpTracer("input", name=f"arg{i}") for i in range(num_args)]

    # Execute function with tracers
    result = func(*inputs)

    return LambdaExpr(func, inputs, result)


def map(inputs: List, func: Callable):
    """Map operation that traces the lambda function."""
    lambda_expr = _trace_lambda(func, num_args=1)
    return MapExpr(inputs, lambda_expr)


def reduce(keys, values: List, method: Callable, output_with_key: bool = False):
    """Reduce operation that traces the lambda function."""
    lambda_expr = _trace_lambda(method, num_args=2)
    return ReduceExpr(keys, values, lambda_expr, output_with_key)


def _convert_lambda_to_ir(
    lambda_expr: LambdaExpr, input_types: List[mir.IRType], node_props: Dict = None, edge_props: Dict = None
):
    """Convert a traced lambda expression to IR nodes."""
    tracer_to_ir: Dict[int, mir.IRNode] = {}
    all_ir_nodes: List[mir.IRNode] = []

    # Create input nodes
    for i, (tracer, ir_type) in enumerate(zip(lambda_expr.inputs, input_types)):
        ir_node = mir.InputNode(tracer.name, ir_type)
        tracer_to_ir[tracer.id] = ir_node
        all_ir_nodes.append(ir_node)

    # Convert traced operations to IR nodes using topological order
    def convert_tracer(tracer: OpTracer) -> mir.IRNode:
        if tracer.id in tracer_to_ir:
            return tracer_to_ir[tracer.id]

        if tracer.op_type == "const":
            # Infer type from value
            if isinstance(tracer.value, int):
                ir_type = mir.IntType()
            elif isinstance(tracer.value, float):
                ir_type = mir.FloatType()
            elif isinstance(tracer.value, bool):
                ir_type = mir.BoolType()
            else:
                ir_type = mir.IntType()

            ir_node = mir.ConstNode(tracer.value, ir_type)
            tracer_to_ir[tracer.id] = ir_node
            all_ir_nodes.append(ir_node)
            return ir_node

        elif tracer.op_type == "attr":
            # Convert operand first
            input_ir = convert_tracer(tracer.operands[0])

            # Determine output type based on attribute
            if isinstance(input_ir.output_type, mir.EdgeType):
                # Accessing edge attributes like src, dst, weight
                if tracer.attr in ["src", "dst"]:
                    output_type = mir.NodeType()
                elif tracer.attr == "weight":
                    output_type = mir.IntType()
                else:
                    output_type = mir.IntType()
            elif isinstance(input_ir.output_type, mir.NodeType):
                # Accessing node properties
                if node_props and tracer.attr in node_props:
                    prop = node_props[tracer.attr]
                    if isinstance(prop, Int):
                        output_type = mir.IntType(prop.width)
                    else:
                        output_type = mir.IntType()
                else:
                    output_type = mir.IntType()
            else:
                output_type = mir.IntType()

            ir_node = mir.AttrAccessNode(tracer.attr, input_ir.output_type, output_type)
            stream = input_ir.create_output() if not input_ir.output_stream else input_ir.output_stream
            ir_node.inputs[0] = stream
            stream.connect_to(ir_node, 0)

            tracer_to_ir[tracer.id] = ir_node
            all_ir_nodes.append(ir_node)
            return ir_node

        elif tracer.op_type == "binop":
            # Convert operands first
            left_ir = convert_tracer(tracer.operands[0])
            right_ir = convert_tracer(tracer.operands[1])

            # Use the type of the first operand (could be more sophisticated)
            operand_type = left_ir.output_type

            ir_node = mir.BinOpNode(tracer.name, operand_type)

            # Connect inputs
            left_stream = left_ir.create_output() if not left_ir.output_stream else left_ir.output_stream
            right_stream = right_ir.create_output() if not right_ir.output_stream else right_ir.output_stream

            ir_node.inputs[0] = left_stream
            ir_node.inputs[1] = right_stream
            left_stream.connect_to(ir_node, 0)
            right_stream.connect_to(ir_node, 1)

            tracer_to_ir[tracer.id] = ir_node
            all_ir_nodes.append(ir_node)
            return ir_node

        else:
            raise ValueError(f"Unknown tracer op_type: {tracer.op_type}")

    # Convert output
    output_ir = convert_tracer(lambda_expr.output)

    return all_ir_nodes, [tracer_to_ir[t.id] for t in lambda_expr.inputs], output_ir


# Main Graph class
class Graph:
    """Main graph computation class."""

    def __init__(self, node_props: Optional[dict] = None, edge_props: Optional[dict] = None):
        if node_props is None:
            node_props = {}
        if edge_props is None:
            edge_props = {}
        self.node_props = node_props
        self.edge_props = edge_props
        self.operations = []

    def start_iteration(self, edge_types: List[str]):
        """Start an iteration over specified edge types."""
        return edge_types

    def update_nodes(self, data: ReduceExpr, update_prop: str, update_method: Callable):
        """Update node properties with reduced data and trace the lambda."""
        lambda_expr = _trace_lambda(update_method, num_args=2)
        self.operations.append(
            {
                "type": "update_nodes",
                "data": data,
                "update_prop": update_prop,
                "update_method": update_method,
                "lambda_expr": lambda_expr,
            }
        )

    def print_all_operations(self):
        """Print all operations in the graph."""
        print("\n" + "=" * 60)
        print("GRAPH OPERATIONS SUMMARY")
        print("=" * 60)

        # Collect all map and reduce operations
        all_ops = []

        # Find maps and reduces from operations
        def collect_ops(obj):
            if isinstance(obj, MapExpr):
                all_ops.append(obj)
            elif isinstance(obj, ReduceExpr):
                all_ops.append(obj)
            elif isinstance(obj, list):
                for item in obj:
                    collect_ops(item)
            elif isinstance(obj, dict):
                for value in obj.values():
                    collect_ops(value)

        # Scan through operations
        for op in self.operations:
            collect_ops(op)

        # Print each operation
        for op in all_ops:
            op.print_details()

        # Print update operations
        for op in self.operations:
            if op["type"] == "update_nodes":
                print("=" * 50)
                print("UPDATE_NODES OPERATION")
                print("=" * 50)
                print(f"Update property: {op['update_prop']}")
                print(f"Update method lambda:")
                op["lambda_expr"].print_operations()
                print()

    def to_ir(self) -> mir.IRGraph:
        """Convert the frontend graph to middle-end IR."""
        ir_graph = mir.IRGraph()

        # Create initial edge stream (starting point)
        edge_stream = mir.Stream(mir.ArrayType(mir.EdgeType()))
        ir_graph.add_stream(edge_stream)

        # Track the streams as we build the graph
        expr_to_stream: Dict[int, mir.Stream] = {}

        # Process operations in order
        def get_or_create_stream(expr) -> mir.Stream:
            if isinstance(expr, MapExpr):
                if id(expr) not in expr_to_stream:
                    # Get input stream
                    input_stream = edge_stream  # Default to edge stream

                    # Convert to IR
                    map_ir = expr.to_ir(input_stream, self.node_props, self.edge_props)
                    ir_graph.add_node(map_ir)

                    # Create output stream
                    output_stream = map_ir.create_output()
                    ir_graph.add_stream(output_stream)
                    expr_to_stream[id(expr)] = output_stream

                return expr_to_stream[id(expr)]

            elif isinstance(expr, ReduceExpr):
                if id(expr) not in expr_to_stream:
                    # Get key and value streams
                    key_stream = get_or_create_stream(expr.keys)
                    value_streams = [get_or_create_stream(v) for v in expr.values]

                    # Convert to IR
                    reduce_ir = expr.to_ir(key_stream, value_streams, self.node_props, self.edge_props)
                    ir_graph.add_node(reduce_ir)

                    # Create output stream
                    output_stream = reduce_ir.create_output()
                    ir_graph.add_stream(output_stream)
                    expr_to_stream[id(expr)] = output_stream

                return expr_to_stream[id(expr)]

            elif isinstance(expr, list):
                # Handle list like edge types
                return edge_stream

            else:
                return edge_stream

        # Process all operations
        for op in self.operations:
            if op["type"] == "update_nodes":
                # Get the stream from the reduce
                get_or_create_stream(op["data"])

        return ir_graph
