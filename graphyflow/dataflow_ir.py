from __future__ import annotations
import uuid as uuid_lib
import copy
from enum import Enum
from typing import List, Optional, Union, Dict, Any, Tuple
from graphyflow.dataflow_ir_datatype import *


class DfirNode:
    _readable_id = 0

    def __init__(self) -> None:
        self.uuid = uuid_lib.uuid4()
        self.readable_id = DfirNode._readable_id
        DfirNode._readable_id += 1


class EmptyNode(DfirNode):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__()
        for key, value in kwargs.items():
            setattr(self, key, value)


class PortType(Enum):
    IN = "in"
    OUT = "out"

    def pluggable(self, other: PortType) -> bool:
        return self != other

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PortType):
            return NotImplemented
        return self.value == other.value


class Port(DfirNode):
    def __init__(self, name: str, parent: Component) -> None:
        super().__init__()
        self.name = name
        self.unique_name = f"{self.name}_{self.readable_id}"
        self.port_type = PortType.OUT if name.startswith("o_") else PortType.IN
        self.data_type = parent.input_type if self.port_type == PortType.IN else parent.output_type
        self.parent = parent
        self.connection = None

    def __eq__(self, other: object):
        if not isinstance(other, Port):
            return NotImplemented
        return (
            self.unique_name == other.unique_name
            and str(self.parent) == str(other.parent)
            and str(self) == str(other)
        )

    def __hash__(self) -> int:
        return hash(str(self) + self.unique_name)

    @property
    def connected(self) -> bool:
        return self.connection is not None

    @property
    def from_const(self) -> bool:
        assert self.port_type == PortType.IN
        assert self.connection is not None
        return isinstance(self.connection.parent, ConstantComponent)

    @property
    def from_const_val(self):
        assert self.from_const
        assert self.connection is not None
        assert isinstance(self.connection.parent, ConstantComponent)
        return self.connection.parent.value

    def copy(self, copy_comp: CopyComponent) -> Port:
        assert self.port_type == PortType.OUT
        assert self.connection is not None
        assert self.data_type == copy_comp.input_type
        assert all(not p.connected for p in copy_comp.ports)
        original_connection = self.connection
        self.disconnect()
        copy_comp.connect({"i_0": self, "o_0": original_connection})
        return copy_comp.out_ports[1]

    def connect(self, other: Port) -> None:
        assert not self.connected and not other.connected
        assert self.port_type.pluggable(other.port_type)
        self.connection = other
        other.connection = self

    def disconnect(self) -> None:
        assert self.connection is not None
        self.connection.connection = None
        self.connection = None

    def __repr__(self) -> str:
        my_repr = f"Port[{self.readable_id}] {self.name} ({self.data_type})"
        direction = "=>" if self.port_type == PortType.OUT else "<="
        tgt = self.connection
        if tgt is None:
            return my_repr
        else:
            return f"{my_repr} {direction} [{tgt.readable_id}] {tgt.name} ({tgt.data_type})"


class ComponentCollection(DfirNode):
    def __init__(self, components: List[Component], inputs: List[Port], outputs: List[Port]) -> None:
        super().__init__()
        self.components = components
        self.inputs = inputs
        self.outputs = outputs
        in_and_out = inputs + outputs
        assert all(all(p.connected or p in in_and_out for p in c.ports) for c in self.components)

    def __repr__(self) -> str:
        return f"ComponentCollection(\n  components: {self.components},\n  inputs: {self.inputs},\n  outputs: {self.outputs}\n)"

    @property
    def all_connected_ports(self) -> List[Port]:
        return [p for p in sum([comp.ports for comp in self.components], []) if p.connected]

    @property
    def output_types(self) -> List[DfirType]:
        assert all(p.data_type is not None for p in self.outputs)
        return [p.data_type for p in self.outputs if p.data_type is not None]

    def added(self, component: Component) -> bool:
        return component.readable_id in [c.readable_id for c in self.components]

    def update_ports(self) -> None:
        def remove_dup(ls: List[Port]) -> List[Port]:
            ls2 = []
            for p in ls:
                if p not in ls2:
                    ls2.append(p)
            return ls2

        # delete all connected ports in inputs and outputs, and delete replaced ports
        self.inputs = remove_dup([p for p in self.inputs if not p.connected])
        self.outputs = remove_dup([p for p in self.outputs if not p.connected])

    def add_front(self, component: Component, ports: Dict[str, Port]) -> None:
        assert all(p in self.inputs for p in ports.values())
        assert all(not p.connected or p in self.all_connected_ports for p in component.in_ports)
        component.connect(ports)
        if not self.added(component):
            self.components.insert(0, component)
        self.inputs = [p for p in self.inputs if p not in ports.values()]
        self.inputs.extend([p for p in component.in_ports])
        self.outputs.extend([p for p in component.out_ports])
        self.update_ports()

    def add_back(self, component: Component, ports: Dict[str, Port]) -> None:
        assert all(p in self.outputs for p in ports.values())
        assert all(not p.connected or p in self.all_connected_ports for p in component.out_ports)
        component.connect(ports)
        if not self.added(component):
            self.components.append(component)
        self.outputs = [p for p in self.outputs if p not in ports.values()]
        self.outputs.extend([p for p in component.out_ports])
        self.inputs.extend([p for p in component.in_ports])
        self.update_ports()

    def concat(
        self, other: ComponentCollection, port_connections: List[Tuple[Port, Port]]
    ) -> ComponentCollection:
        assert all(p in (self.inputs + self.outputs) for p, _ in port_connections)
        assert all(p in (other.inputs + other.outputs) for _, p in port_connections)
        for p, q in port_connections:
            p.connect(q)
        self.components.extend(other.components)
        for p in other.inputs:
            if p not in self.inputs:
                self.inputs.append(p)
        for p in other.outputs:
            if p not in self.outputs:
                self.outputs.append(p)
        for p, q in port_connections:
            for port in [p, q]:
                while port in self.inputs:
                    self.inputs.remove(port)
                while port in self.outputs:
                    self.outputs.remove(port)
        self.update_ports()
        return self

    def topo_sort(self) -> List[Component]:
        def port_solved(port: Port) -> bool:
            if not port.connected:
                assert port in (self.inputs + self.outputs)
                return True
            else:
                assert port.connection is not None
                return port.connection.parent in result

        def check_reduce(comp: Component) -> bool:
            if not isinstance(comp, ReduceComponent):
                return False
            return port_solved(comp.get_port("i_0"))

        result = []
        waitings = copy.deepcopy(self.components)
        while waitings:
            new_ones = []
            for comp in waitings:
                if all(port_solved(p) for p in comp.in_ports) or check_reduce(comp):
                    new_ones.append(comp)
            waitings = [w for w in waitings if w not in new_ones]
            result.extend(new_ones)
        return result


class Component(DfirNode):
    def __init__(
        self,
        input_type: Optional[DfirType],
        output_type: Optional[DfirType],
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

    def get_port(self, name: str) -> Port:
        for port in self.ports:
            if port.name == name:
                return port
        raise ValueError(f"Port {name} not found in {self.ports}")

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
                assert (
                    port.data_type == self.ports[idx].data_type
                ), f"{port.data_type} != {self.ports[idx].data_type}"
                self.ports[idx].connect(port)

    def additional_info(self) -> List[str]:
        return [""]

    @property
    def name(self) -> str:
        return f"{self.__class__.__name__[:5]}_{self.readable_id}"

    def __repr__(self) -> str:
        add_info = self.additional_info()
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
        self.io_type = io_type
        if self.io_type == self.IOType.INPUT:
            super().__init__(None, data_type, ["o_0"])
        else:
            super().__init__(data_type, None, ["i_0"])


class ConstantComponent(Component):
    def __init__(self, data_type: DfirType, value: Any) -> None:
        super().__init__(None, data_type, ["o_0"], parallel=isinstance(data_type, ArrayType))
        self.value = value

    def additional_info(self) -> List[str]:
        return [f"value: {self.value}"]


class CopyComponent(Component):
    def __init__(self, input_type: DfirType) -> None:
        super().__init__(input_type, input_type, ["i_0", "o_0", "o_1"])


class GatherComponent(Component):
    def __init__(self, input_types: List[DfirType]) -> None:
        parallel = all(isinstance(t, ArrayType) for t in input_types)
        ports = []
        specific_port_types = {}
        output_types = []
        for i in range(len(input_types)):
            ports.append(f"i_{i}")
            specific_port_types[f"i_{i}"] = input_types[i]
            if parallel:
                cur_input_type = input_types[i]
                assert isinstance(cur_input_type, ArrayType)
                output_types.append(cur_input_type.type_)
            else:
                output_types.append(input_types[i])
        output_type = TupleType(output_types)
        if parallel:
            output_type = ArrayType(output_type)
        ports.append("o_0")
        super().__init__(output_type, output_type, ports, parallel, specific_port_types)


class ScatterComponent(Component):
    def __init__(self, input_type: DfirType) -> None:
        assert isinstance(input_type, (TupleType, ArrayType))
        real_input_type = input_type
        parallel = False
        if isinstance(input_type, ArrayType):
            assert isinstance(input_type.type_, TupleType)
            real_input_type = input_type.type_
            parallel = True
        ports = ["i_0"]
        assert isinstance(real_input_type, TupleType)
        for i in range(len(real_input_type.types)):
            ports.append(f"o_{i}")
        # output_type = input_type just for assign
        super().__init__(
            input_type,
            input_type,
            ports,
            parallel,
            specific_port_types={
                f"o_{i}": ArrayType(type_) if parallel else type_
                for i, type_ in enumerate(real_input_type.types)
            },
        )


class BinOp(Enum):
    ADD = "+"
    SUB = "-"
    MUL = "*"
    DIV = "/"
    MOD = "%"
    AND = "&"
    OR = "|"
    EQ = "=="
    NE = "!="
    LT = "<"
    GT = ">"
    LE = "<="
    GE = ">="
    SL = "<<"
    SR = ">>"
    MIN = "min"
    MAX = "max"

    def __repr__(self) -> str:
        return self.value

    def gen_repr(self, part_a, part_b):
        if self.value in ["min", "max"]:
            translate_dict = {"min": "<", "max": ">"}
            repr = f"(({part_a}) {translate_dict[self.value]} ({part_b}) ? {part_a} : {part_b})"
        else:
            repr = f"{part_a} {self.value} {part_b}"
        return "{ " + repr + " }"

    def gen_repr_forbkd(self, part_a, part_b):
        if self.value in ["min", "max"]:
            translate_dict = {"min": "<", "max": ">"}
            repr = f"(({part_a}) {translate_dict[self.value]} ({part_b}) ? {part_a} : {part_b})"
        else:
            repr = f"{part_a} {self.value} {part_b}"
        return "(" + repr + ")"

    def output_type(self, input_type: DfirType) -> DfirType:
        if self in [
            BinOp.ADD,
            BinOp.SUB,
            BinOp.MUL,
            BinOp.DIV,
            BinOp.AND,
            BinOp.OR,
            BinOp.MIN,
            BinOp.MAX,
            BinOp.SL,
            BinOp.SR,
        ]:
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
        assert not isinstance(output_type, OptionalType) and not isinstance(output_type, TupleType)
        super().__init__(input_type, output_type, ["i_0", "i_1", "o_0"], parallel)
        self.op = op

    def additional_info(self) -> List[str]:
        return [f"op: {self.op}"]


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
            raise RuntimeError("Output type for select operation should be decided outside of UnaryOp")
        else:
            raise ValueError(f"Unsupported unary operation: {self}")


class UnaryOpComponent(Component):
    def __init__(
        self,
        op: UnaryOp,
        input_type: DfirType,
        select_index: Optional[int] = None,
        attr_type: Optional[DfirType] = None,
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
        elif op == UnaryOp.GET_ATTR:
            assert attr_type is not None
            assert select_index is not None
            inside_output_type = attr_type
        else:
            assert not isinstance(
                real_input_type, TupleType
            ), f"input type {real_input_type} is a tuple type for {op}"
            inside_output_type = op.output_type(real_input_type)
        output_type = ArrayType(inside_output_type) if parallel else inside_output_type
        super().__init__(input_type, output_type, ["i_0", "o_0"], parallel)
        self.op = op
        self.select_index = select_index

    def additional_info(self) -> List[str]:
        if self.op == UnaryOp.SELECT:
            return [f"op: {self.op}", f"select_index: {self.select_index}"]
        else:
            return [f"op: {self.op}"]


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
        assert isinstance(input_type, ArrayType)
        assert isinstance(input_type.type_, OptionalType)
        output_type = ArrayType(input_type.type_.type_)
        super().__init__(input_type, output_type, ["i_0", "o_0"], parallel=True)


class ReduceComponent(Component):
    def __init__(
        self,
        input_type: DfirType,
        accumulated_type: DfirType,
        reduce_key_out_type: DfirType,
    ) -> None:
        assert isinstance(input_type, ArrayType)
        real_input_type = input_type.type_
        super().__init__(
            input_type,
            ArrayType(accumulated_type),
            [
                "i_0",
                "o_0",
                "i_reduce_key_out",
                "i_reduce_transform_out",
                "i_reduce_unit_end",
                "o_reduce_key_in",
                "o_reduce_transform_in",
                "o_reduce_unit_start_0",
                "o_reduce_unit_start_1",
            ],
            parallel=False,
            specific_port_types={
                "i_reduce_key_out": reduce_key_out_type,
                "i_reduce_transform_out": accumulated_type,
                "i_reduce_unit_end": accumulated_type,
                "o_reduce_key_in": real_input_type,
                "o_reduce_transform_in": real_input_type,
                "o_reduce_unit_start_0": accumulated_type,
                "o_reduce_unit_start_1": accumulated_type,
            },
        )


class FusedOpComponent(Component):
    """
    A container that wraps a complex subgraph (ComponentCollection) into a single component.
    It exposes the inputs and outputs of the subgraph as its own ports.
    The subgraph is restricted to contain only pure computational components.
    """

    def __init__(self, name: str, sub_graph: ComponentCollection) -> None:
        self.sub_graph = sub_graph

        # --- Validation Step ---
        # Before creating the component, validate the types of components within the subgraph.
        # This ensures that a FusedOpComponent only contains pure, stateless computational logic.
        allowed_base_types = (
            ScatterComponent,
            GatherComponent,
            ConstantComponent,
            CopyComponent,
            BinOpComponent,
            UnusedEndMarkerComponent,
        )
        disallowed_unary_ops = (UnaryOp.SELECT, UnaryOp.GET_ATTR, UnaryOp.GET_LENGTH)
        for comp in self.sub_graph.components:
            if isinstance(comp, UnaryOpComponent):
                if comp.op in disallowed_unary_ops:
                    raise TypeError(
                        f"FusedOpComponent cannot contain a UnaryOpComponent with the operation '{comp.op.name}'. "
                        "Only pure arithmetic or casting operations are allowed."
                    )
            elif not isinstance(comp, allowed_base_types):
                raise TypeError(
                    f"Component type '{type(comp).__name__}' is not allowed inside a FusedOpComponent. "
                    "Allowed types are: Scatter, Gather, Constant, Copy, BinOp, UnusedEndMarker, and specific UnaryOps."
                )

        # --- Port Generation Step ---
        ports = []
        specific_port_types = {}

        for i, in_port in enumerate(sub_graph.inputs):
            port_name = f"i_{i}"
            ports.append(port_name)
            specific_port_types[port_name] = in_port.data_type

        for i, out_port in enumerate(sub_graph.outputs):
            port_name = f"o_{i}"
            ports.append(port_name)
            specific_port_types[port_name] = out_port.data_type

        super().__init__(
            input_type=None,
            output_type=None,
            ports=ports,
            parallel=False,
            specific_port_types=specific_port_types,
        )
        self._custom_name = name

    @property
    def name(self) -> str:
        return f"{self._custom_name}_{self.readable_id}"

    def additional_info(self) -> List[str]:
        return [f"sub_graph_components: {[c.name for c in self.sub_graph.components]}"]


class PlaceholderComponent(Component):
    def __init__(self, data_type: DfirType) -> None:
        super().__init__(data_type, data_type, ["i_0", "o_0"])


class UnusedEndMarkerComponent(Component):
    def __init__(self, input_type: DfirType) -> None:
        super().__init__(input_type, None, ["i_0"])
