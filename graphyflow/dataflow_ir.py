from __future__ import annotations
import uuid as uuid_lib
from enum import Enum
from typing import List, Optional, Union, Dict, Any


class DfirType:
    def __init__(self, type_name, is_optional: bool = False) -> None:
        self.type_name = type_name
        self.is_optional = is_optional

    def __eq__(self, other: DfirType) -> bool:
        return self.type_name == other.type_name

    def __ne__(self, other: DfirType) -> bool:
        return self.type_name != other.type_name

    def __repr__(self) -> str:
        return self.type_name


class SpecialType(DfirType):
    """Node or Edge"""

    def __init__(self, type_name: str) -> None:
        assert type_name in ["node", "edge"]
        super().__init__(type_name)


class IntType(DfirType):
    def __init__(self) -> None:
        super().__init__("Int")


class FloatType(DfirType):
    def __init__(self) -> None:
        super().__init__("Float")


class BoolType(DfirType):
    def __init__(self) -> None:
        super().__init__("Bool")


class OptionalType(DfirType):
    def __init__(self, type_: DfirType) -> None:
        assert not isinstance(type_, OptionalType)
        super().__init__(f"Optional<{type_.type_name}>", True)
        self.type_ = type_


class TupleType(DfirType):
    def __init__(self, types: List[DfirType]) -> None:
        super().__init__(f"Tuple<{', '.join([t.type_name for t in types])}>")
        self.types = types


class ArrayType(DfirType):
    def __init__(self, type_: DfirType) -> None:
        super().__init__(f"Array<{type_.type_name}>")
        self.type_ = type_


class DfirNode:
    _readable_id = 0

    def __init__(self) -> None:
        self.uuid = uuid_lib.uuid4()
        self.readable_id = DfirNode._readable_id
        DfirNode._readable_id += 1


class PortType(Enum):
    IN = "in"
    OUT = "out"

    def pluggable(self, other: PortType) -> bool:
        return self != other


class Port(DfirNode):
    def __init__(self, name: str, parent: DfirNode) -> None:
        super().__init__()
        self.name = name
        self.port_type = PortType.OUT if name.startswith("o_") else PortType.IN
        self.data_type = (
            parent.input_type if self.port_type == PortType.IN else parent.output_type
        )
        self.parent = parent
        self.connection = None

    @property
    def connected(self) -> bool:
        return self.connection is not None

    def connect(self, other: Port) -> None:
        assert self.port_type.pluggable(other.port_type)
        self.connection = other
        other.connection = self

    def disconnect(self) -> None:
        assert self.connection is not None
        self.connection.connection = None
        self.connection = None

    def __repr__(self) -> str:
        if self.connection is None:
            return f"Port[{self.readable_id}] {self.name} ({self.data_type})"
        else:
            return f"Port[{self.readable_id}] {self.name} ({self.data_type}) <=> {self.connection.name} ({self.connection.data_type})"


class ComponentCollection(DfirNode):
    def __init__(
        self, components: List[Component], inputs: List[Port], outputs: List[Port]
    ) -> None:
        super().__init__()
        self.components = components
        self.inputs = inputs
        self.outputs = outputs
        in_and_out = inputs + outputs
        assert all(
            all(p.connected or p in in_and_out for p in c.ports)
            for c in self.components
        )

    def __repr__(self) -> str:
        return f"ComponentCollection(\n  components: {self.components},\n  inputs: {self.inputs},\n  outputs: {self.outputs}\n)"

    def add_front(self, component: Component, ports: Dict[str, Port]) -> None:
        assert all(p in self.inputs for p in ports.values())
        assert all(not p.connected for p in component.in_ports)
        component.connect(ports)
        self.components.append(component)
        self.inputs = [p for p in self.inputs if p not in ports.values()]
        self.inputs.extend(component.in_ports)
        self.outputs.extend([p for p in component.out_ports if not p.connected])

    def add_back(self, component: Component, ports: Dict[str, Port]) -> None:
        assert all(p in self.outputs for p in ports.values())
        assert all(not p.connected for p in component.out_ports)
        component.connect(ports)
        self.components.append(component)
        self.outputs = [p for p in self.outputs if p not in ports.values()]
        self.outputs.extend(component.out_ports)
        self.inputs.extend([p for p in component.in_ports if not p.connected])


class Component(DfirNode):
    def __init__(
        self,
        input_type: DfirType,
        output_type: DfirType,
        ports: List[str],
        parallel: bool = False,
        specific_port_types: Optional[Dict[str, DfirType]] = None,
    ) -> None:
        super().__init__()
        self.input_type = input_type
        self.output_type = output_type
        self.ports = [Port(port, self) for port in ports]
        self.in_ports = [p for p in self.ports if p.port_type == PortType.IN]
        self.input_port_num = len(self.in_ports)
        self.out_ports = [p for p in self.ports if p.port_type == PortType.OUT]
        self.output_port_num = len(self.out_ports)
        self.parallel = parallel
        if specific_port_types is not None:
            for port_name, data_type in specific_port_types.items():
                for port in self.ports:
                    if port.name == port_name:
                        port.data_type = data_type
                        break
                else:
                    raise ValueError(f"Port {port_name} not found in {self.ports}")

    def connect(self, ports: Union[List[Port], Dict[str, Port]]) -> None:
        if isinstance(ports, list):
            for i, port in enumerate(ports):
                assert port.data_type == self.ports[i].data_type
                self.ports[i].connect(port)
        elif isinstance(ports, dict):
            for port_name, port in ports.items():
                idx = None
                for i, p in enumerate(self.ports):
                    if p.name == port_name:
                        idx = i
                        break
                assert idx is not None
                assert port.data_type == self.ports[idx].data_type
                self.ports[idx].connect(port)

    def additional_info(self) -> str:
        return ""

    def __repr__(self) -> str:
        add_info = self.additional_info()
        add_info = add_info if isinstance(add_info, list) else [add_info]
        add_info = "\n  ".join(add_info)
        add_info = "\n  " + add_info if add_info else ""
        return (
            f"{self.__class__.__name__}(\n  input: {self.input_type},\n"
            + f"  output: {self.output_type},\n  ports:\n    "
            + "\n    ".join([str(p) for p in self.ports])
            + add_info
            + "\n)"
        )


class IOComponent(Component):
    class IOType(Enum):
        INPUT = "input"
        OUTPUT = "output"

    def __init__(self, io_type: IOType, data_type: DfirType) -> None:
        if self.io_type == self.IOType.INPUT:
            super().__init__(None, data_type, ["o_0"])
        else:
            super().__init__(data_type, None, ["i_0"])


class ConstantComponent(Component):
    def __init__(self, data_type: DfirType, value: Any) -> None:
        super().__init__(None, data_type, ["o_0"])
        self.value = value

    def additional_info(self) -> str:
        return f"value: {self.value}"


class CopyComponent(Component):
    def __init__(self, input_type: DfirType) -> None:
        super().__init__(input_type, input_type, ["i_0", "o_0", "o_1"])


class ScatterComponent(Component):
    def __init__(self, input_type: TupleType) -> None:
        assert isinstance(input_type, TupleType)
        ports = ["i_0"]
        for i in range(len(input_type.types)):
            ports.append(f"o_{i}")
        super().__init__(input_type, None, ports)
        self.output_types = input_type.types
        for i, type_ in enumerate(self.output_types):
            self.ports[i + 1].data_type = type_


class BinOp(Enum):
    ADD = "+"
    SUB = "-"
    MUL = "*"
    DIV = "/"
    AND = "&"
    OR = "|"
    EQ = "=="
    NE = "!="
    LT = "<"
    GT = ">"
    LE = "<="
    GE = ">="

    def __repr__(self) -> str:
        return self.value

    def can_reduce(self) -> bool:
        return self in [BinOp.ADD, BinOp.MUL, BinOp.AND, BinOp.OR, BinOp.EQ]

    def output_type(self, input_type: DfirType) -> DfirType:
        if self in [BinOp.ADD, BinOp.SUB, BinOp.MUL, BinOp.DIV, BinOp.AND, BinOp.OR]:
            return input_type
        elif self in [BinOp.EQ, BinOp.NE, BinOp.LT, BinOp.GT, BinOp.LE, BinOp.GE]:
            return BoolType()
        else:
            raise ValueError(f"Unsupported binary operation: {self}")


class BinOpComponent(Component):
    def __init__(self, op: BinOp, input_type: DfirType) -> None:
        if isinstance(input_type, ArrayType):
            parallel = True
            output_type = ArrayType(op.output_type(input_type.type_))
        else:
            parallel = False
            output_type = op.output_type(input_type)
        assert not isinstance(output_type, OptionalType) and not isinstance(
            output_type, TupleType
        )
        super().__init__(input_type, output_type, ["i_0", "i_1", "o_0"], parallel)
        self.op = op

    def additional_info(self) -> str:
        return f"op: {self.op}"


class UnaryOp(Enum):
    NOT = "!"
    NEG = "-"
    CAST_BOOL = "to_bool"
    CAST_INT = "to_int"
    CAST_FLOAT = "to_float"
    SELECT = "select"
    GET_LENGTH = "length"
    GET_ATTR = "get_attr"

    def __repr__(self) -> str:
        return self.value

    def output_type(self, input_type: DfirType) -> DfirType:
        input_available_dict = {
            UnaryOp.NOT: BoolType,
            UnaryOp.NEG: [IntType, FloatType],
            UnaryOp.CAST_BOOL: [IntType, FloatType, BoolType],
            UnaryOp.CAST_INT: [IntType, FloatType, BoolType],
            UnaryOp.CAST_FLOAT: [IntType, FloatType, BoolType],
            UnaryOp.SELECT: TupleType,
            UnaryOp.GET_LENGTH: ArrayType,
            UnaryOp.GET_ATTR: SpecialType,
        }
        assert isinstance(
            input_type, input_available_dict[self]
        ), f"{self}: input type {input_type} should be one of {input_available_dict[self]}"
        if self in [UnaryOp.NOT, UnaryOp.NEG]:
            return input_type
        elif self == UnaryOp.CAST_BOOL:
            return BoolType()
        elif self == UnaryOp.CAST_INT:
            return IntType()
        elif self == UnaryOp.CAST_FLOAT:
            return FloatType()
        elif self == UnaryOp.GET_LENGTH:
            return IntType()
        elif self == UnaryOp.GET_ATTR:
            # get node id from node or get src/dst node id from edge
            return IntType() if input_type.type_name == "node" else SpecialType("node")
        elif self == UnaryOp.SELECT:
            raise RuntimeError(
                "Output type for select operation should be decided outside of UnaryOp"
            )
        else:
            raise ValueError(f"Unsupported unary operation: {self}")


class UnaryOpComponent(Component):
    def __init__(
        self, op: UnaryOp, input_type: DfirType, select_index: Optional[int] = None
    ) -> None:
        if isinstance(input_type, ArrayType):
            parallel = True
            real_input_type = input_type.type_
        else:
            parallel = False
            real_input_type = input_type
        if op == UnaryOp.SELECT:
            assert isinstance(real_input_type, TupleType)
            assert select_index is not None
            inside_output_type = real_input_type.types[select_index]
        else:
            assert not isinstance(real_input_type, TupleType)
            inside_output_type = op.output_type(real_input_type)
        output_type = ArrayType(inside_output_type) if parallel else inside_output_type
        super().__init__(input_type, output_type, ["i_0", "o_0"], parallel)
        self.op = op
        self.select_index = select_index

    def additional_info(self) -> str:
        if self.op == UnaryOp.SELECT:
            return [f"op: {self.op}", f"select_index: {self.select_index}"]
        else:
            return f"op: {self.op}"


class ConditionalComponent(Component):
    def __init__(self, input_type: DfirType) -> None:
        if isinstance(input_type, ArrayType):
            parallel = True
            output_type = ArrayType(OptionalType(input_type.type_))
            cond_type = ArrayType(BoolType())
        else:
            parallel = False
            output_type = OptionalType(input_type)
            cond_type = BoolType()
        super().__init__(
            input_type,
            output_type,
            ["i_data", "i_cond", "o_0"],
            parallel,
            {"i_cond": cond_type},
        )


class CollectComponent(Component):
    def __init__(self, input_type: DfirType) -> None:
        output_type = (
            ArrayType(input_type)
            if not isinstance(input_type, OptionalType)
            else ArrayType(input_type.type_)
        )
        super().__init__(input_type, output_type, ["i_0", "o_0"])


class ReduceComponent(Component):
    def __init__(self, input_type: DfirType, reduce_op: BinOp) -> None:
        assert isinstance(input_type, ArrayType)
        assert reduce_op.can_reduce()
        output_type = reduce_op.output_type(input_type.type_)
        super().__init__(input_type, output_type, ["i_0", "o_0"])
        self.reduce_op = reduce_op


class PlaceholderComponent(Component):
    def __init__(self, data_type: DfirType) -> None:
        super().__init__(data_type, data_type, ["i_0", "o_0"])
