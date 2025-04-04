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
    def __init__(self) -> None:
        self.uuid = uuid_lib.uuid4()


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

    def connect(self, other: Port) -> None:
        assert self.port_type.pluggable(other.port_type)
        self.connection = other
        other.connection = self

    def disconnect(self) -> None:
        assert self.connection is not None
        self.connection.connection = None
        self.connection = None


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
            for port, data_type in specific_port_types.items():
                self.ports[port].data_type = data_type

    def connect(self, ports: Union[List[Port], Dict[str, Port]]) -> None:
        # 打印组件类名和端口信息
        print(f"组件类名: {self.__class__.__name__}")
        print("端口列表:")
        for port in self.ports:
            port_direction = "输入" if port.port_type == PortType.IN else "输出"
            print(f"  - {port.name} ({port_direction})")
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


class BinOp(Enum):
    ADD = "+"
    SUB = "-"
    MUL = "*"
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
        if self in [BinOp.ADD, BinOp.SUB, BinOp.MUL, BinOp.AND, BinOp.OR]:
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
        assert isinstance(input_type, input_available_dict[self])
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
            output_type = ArrayType(op.output_type(input_type.type_))
            real_input_type = input_type.type_
        else:
            parallel = False
            output_type = op.output_type(input_type)
            real_input_type = input_type
        if op == UnaryOp.SELECT:
            assert isinstance(real_input_type, TupleType)
            assert select_index is not None
            output_type = real_input_type.types[select_index]
        else:
            assert not isinstance(real_input_type, TupleType)
            output_type = op.output_type(real_input_type)
        super().__init__(input_type, output_type, ["i_0", "o_0"], parallel)
        self.op = op
        self.select_index = select_index


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
