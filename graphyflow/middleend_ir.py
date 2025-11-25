"""
Middle-end IR: Intermediate Representation for GraphyFlow
Represents computation as connected nodes with typed streams.
"""

from typing import List, Optional, Any, Dict, Set
from enum import Enum


class IRType:
    """Base class for IR types."""

    def __repr__(self):
        return self.__class__.__name__


class IntType(IRType):
    """Integer type."""

    def __init__(self, width: int = 32):
        self.width = width

    def __repr__(self):
        return f"Int{self.width}"

    def __eq__(self, other):
        return isinstance(other, IntType) and self.width == other.width


class FloatType(IRType):
    """Float type."""

    def __repr__(self):
        return "Float"

    def __eq__(self, other):
        return isinstance(other, FloatType)


class BoolType(IRType):
    """Boolean type."""

    def __repr__(self):
        return "Bool"

    def __eq__(self, other):
        return isinstance(other, BoolType)


class NodeType(IRType):
    """Graph node type."""

    def __repr__(self):
        return "Node"

    def __eq__(self, other):
        return isinstance(other, NodeType)


class EdgeType(IRType):
    """Graph edge type."""

    def __repr__(self):
        return "Edge"

    def __eq__(self, other):
        return isinstance(other, EdgeType)


class ArrayType(IRType):
    """Array/Stream type."""

    def __init__(self, element_type: IRType):
        self.element_type = element_type

    def __repr__(self):
        return f"Array<{self.element_type}>"

    def __eq__(self, other):
        return isinstance(other, ArrayType) and self.element_type == other.element_type


class TupleType(IRType):
    """Tuple type for multiple values."""

    def __init__(self, element_types: List[IRType]):
        self.element_types = element_types

    def __repr__(self):
        return f"Tuple<{', '.join(str(t) for t in self.element_types)}>"

    def __eq__(self, other):
        return isinstance(other, TupleType) and self.element_types == other.element_types


class Stream:
    """Represents a typed data stream between IR nodes."""

    _id_counter = 0

    def __init__(self, data_type: IRType, source: "IRNode" = None, source_port: int = 0):
        self.id = Stream._id_counter
        Stream._id_counter += 1
        self.data_type = data_type
        self.source = source
        self.source_port = source_port
        self.consumers: List[tuple["IRNode", int]] = []  # List of (node, port) pairs

    def connect_to(self, node: "IRNode", port: int = 0):
        """Connect this stream to a consumer node."""
        self.consumers.append((node, port))
        node.inputs[port] = self

    def __repr__(self):
        return f"Stream#{self.id}<{self.data_type}>"


class IRNode:
    """Base class for IR nodes."""

    _id_counter = 0

    def __init__(self, node_type: str, output_type: IRType, num_inputs: int = 0):
        self.id = IRNode._id_counter
        IRNode._id_counter += 1
        self.node_type = node_type
        self.output_type = output_type
        self.inputs: Dict[int, Optional[Stream]] = {i: None for i in range(num_inputs)}
        self.output_stream: Optional[Stream] = None

    def create_output(self) -> Stream:
        """Create an output stream for this node."""
        self.output_stream = Stream(self.output_type, source=self, source_port=0)
        return self.output_stream

    def __repr__(self):
        return f"{self.node_type}#{self.id}"

    def print_details(self, indent: int = 0):
        """Print node details."""
        prefix = "  " * indent
        print(f"{prefix}{self}")
        print(f"{prefix}  Output: {self.output_type}")
        for port, stream in self.inputs.items():
            if stream:
                print(f"{prefix}  Input[{port}]: {stream}")


class InputNode(IRNode):
    """Input node representing function arguments."""

    def __init__(self, name: str, data_type: IRType):
        super().__init__("Input", data_type, num_inputs=0)
        self.name = name

    def __repr__(self):
        return f"Input#{self.id}({self.name})"


class ConstNode(IRNode):
    """Constant value node."""

    def __init__(self, value: Any, data_type: IRType):
        super().__init__("Const", data_type, num_inputs=0)
        self.value = value

    def __repr__(self):
        return f"Const#{self.id}({self.value})"


class AttrAccessNode(IRNode):
    """Attribute access node (e.g., e.dst, node.dist)."""

    def __init__(self, attr_name: str, input_type: IRType, output_type: IRType):
        super().__init__("AttrAccess", output_type, num_inputs=1)
        self.attr_name = attr_name

    def __repr__(self):
        return f"AttrAccess#{self.id}(.{self.attr_name})"


class BinOpNode(IRNode):
    """Binary operation node."""

    def __init__(self, op: str, operand_type: IRType):
        super().__init__("BinOp", operand_type, num_inputs=2)
        self.op = op
        # Determine output type based on operation
        if op in ["<", "<=", ">", ">=", "==", "!="]:
            self.output_type = BoolType()
        else:
            self.output_type = operand_type

    def __repr__(self):
        return f"BinOp#{self.id}({self.op})"


class MapIR(IRNode):
    """Map operation IR node."""

    def __init__(
        self,
        input_stream: Stream,
        lambda_body: List[IRNode],
        lambda_inputs: List[IRNode],
        lambda_output: IRNode,
    ):
        # Output is array of lambda output type
        output_type = ArrayType(lambda_output.output_type)
        super().__init__("Map", output_type, num_inputs=1)
        self.inputs[0] = input_stream
        input_stream.connect_to(self, 0)

        self.lambda_body = lambda_body  # All nodes in the lambda
        self.lambda_inputs = lambda_inputs  # Input nodes of lambda
        self.lambda_output = lambda_output  # Output node of lambda

    def print_details(self, indent: int = 0):
        """Print map operation details."""
        prefix = "  " * indent
        print(f"{prefix}Map#{self.id}")
        print(f"{prefix}  Input Stream: {self.inputs[0]}")
        print(f"{prefix}  Output: {self.output_type}")
        print(f"{prefix}  Lambda Body ({len(self.lambda_body)} nodes):")
        for node in self.lambda_body:
            node.print_details(indent + 2)
        print(f"{prefix}  Lambda Output: {self.lambda_output}")


class ReduceIR(IRNode):
    """Reduce operation IR node."""

    def __init__(
        self,
        key_stream: Stream,
        value_streams: List[Stream],
        lambda_body: List[IRNode],
        lambda_inputs: List[IRNode],
        lambda_output: IRNode,
        output_with_key: bool = False,
    ):
        # Output type: unwrap array from value, optionally add key
        value_type = value_streams[0].data_type
        if isinstance(value_type, ArrayType):
            reduced_type = value_type.element_type
        else:
            reduced_type = value_type

        # Result is array of reduced values
        if output_with_key:
            key_type = key_stream.data_type
            if isinstance(key_type, ArrayType):
                key_type = key_type.element_type
            output_type = ArrayType(TupleType([reduced_type, key_type]))
        else:
            output_type = ArrayType(reduced_type)

        super().__init__("Reduce", output_type, num_inputs=1 + len(value_streams))

        # Connect inputs
        self.inputs[0] = key_stream
        key_stream.connect_to(self, 0)
        for i, stream in enumerate(value_streams):
            self.inputs[i + 1] = stream
            stream.connect_to(self, i + 1)

        self.output_with_key = output_with_key
        self.lambda_body = lambda_body
        self.lambda_inputs = lambda_inputs
        self.lambda_output = lambda_output

    def print_details(self, indent: int = 0):
        """Print reduce operation details."""
        prefix = "  " * indent
        print(f"{prefix}Reduce#{self.id}")
        print(f"{prefix}  Key Stream: {self.inputs[0]}")
        for i in range(1, len(self.inputs)):
            print(f"{prefix}  Value Stream[{i-1}]: {self.inputs[i]}")
        print(f"{prefix}  Output: {self.output_type}")
        print(f"{prefix}  Output with key: {self.output_with_key}")
        print(f"{prefix}  Lambda Body ({len(self.lambda_body)} nodes):")
        for node in self.lambda_body:
            node.print_details(indent + 2)
        print(f"{prefix}  Lambda Output: {self.lambda_output}")


class IRGraph:
    """Container for the entire IR graph."""

    def __init__(self):
        self.nodes: List[IRNode] = []
        self.streams: List[Stream] = []
        self.map_nodes: List[MapIR] = []
        self.reduce_nodes: List[ReduceIR] = []

    def add_node(self, node: IRNode):
        """Add a node to the graph."""
        self.nodes.append(node)
        if isinstance(node, MapIR):
            self.map_nodes.append(node)
        elif isinstance(node, ReduceIR):
            self.reduce_nodes.append(node)

    def add_stream(self, stream: Stream):
        """Add a stream to the graph."""
        self.streams.append(stream)

    def print_graph(self):
        """Print the entire IR graph."""
        print("\n" + "=" * 70)
        print("IR GRAPH")
        print("=" * 70)

        print(f"\nTotal Nodes: {len(self.nodes)}")
        print(f"Total Streams: {len(self.streams)}")
        print(f"Map Operations: {len(self.map_nodes)}")
        print(f"Reduce Operations: {len(self.reduce_nodes)}")

        print("\n" + "-" * 70)
        print("MAP OPERATIONS")
        print("-" * 70)
        for map_node in self.map_nodes:
            map_node.print_details()
            print()

        print("-" * 70)
        print("REDUCE OPERATIONS")
        print("-" * 70)
        for reduce_node in self.reduce_nodes:
            reduce_node.print_details()
            print()

        print("-" * 70)
        print("STREAM CONNECTIONS")
        print("-" * 70)
        for stream in self.streams:
            if stream.source:
                print(f"{stream} from {stream.source}")
                for consumer, port in stream.consumers:
                    print(f"  -> {consumer} [port {port}]")
        print()
