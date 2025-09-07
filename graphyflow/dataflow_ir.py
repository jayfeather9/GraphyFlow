from __future__ import annotations
import uuid as uuid_lib
import copy
from enum import Enum
from typing import List, Optional, Union, Dict, Any, Tuple
from graphyflow.dataflow_ir_datatype import *
import graphyflow.hls_utils as hls


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

    def __eq__(self, other: PortType) -> bool:
        return self.value == other.value


class Port(DfirNode):
    def __init__(self, name: str, parent: DfirNode) -> None:
        super().__init__()
        self.name = name
        self.unique_name = f"{self.name}_{self.readable_id}"
        self.port_type = PortType.OUT if name.startswith("o_") else PortType.IN
        self.data_type = parent.input_type if self.port_type == PortType.IN else parent.output_type
        self.parent = parent
        self.connection = None

    def __eq__(self, other: Port):
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
        assert self.port_type == PortType.IN and self.connected
        return isinstance(self.connection.parent, ConstantComponent)

    @property
    def from_const_val(self):
        assert self.from_const
        return self.connection.parent.value

    def copy(self, copy_comp: CopyComponent) -> Port:
        assert self.port_type == PortType.OUT
        assert self.connected
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
        if self.connection is None:
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
        return [p.data_type for p in self.outputs]

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

    def additional_info(self) -> str:
        return ""

    @property
    def name(self) -> str:
        return f"{self.__class__.__name__[:5]}_{self.readable_id}"

    def to_hls(self) -> hls.HLSFunction:
        assert False, f"Abstract method to_hls() should be implemented for {self.__class__.__name__}"

    def get_hls_function(
        self,
        code_in_loop: List[str],
        code_before_loop: Optional[List[str]] = [],
        code_after_loop: Optional[List[str]] = [],
        name_tail: Optional[str] = None,
    ) -> hls.HLSFunction:
        return hls.HLSFunction(
            name=self.name + (f"_{name_tail}" if name_tail else ""),
            comp=self,
            code_in_loop=code_in_loop,
            code_before_loop=code_before_loop,
            code_after_loop=code_after_loop,
        )

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
        self.io_type = io_type
        if self.io_type == self.IOType.INPUT:
            super().__init__(None, data_type, ["o_0"])
        else:
            super().__init__(data_type, None, ["i_0"])

    def to_hls(self) -> hls.HLSFunction:
        assert False, "IOComponent should not be used in HLS"


class ConstantComponent(Component):
    def __init__(self, data_type: DfirType, value: Any) -> None:
        super().__init__(None, data_type, ["o_0"], parallel=isinstance(data_type, ArrayType))
        self.value = value

    def additional_info(self) -> str:
        return f"value: {self.value}"

    def to_hls(self) -> hls.HLSFunction:
        assert False, "ConstantComponent should not be used in HLS"


class CopyComponent(Component):
    def __init__(self, input_type: DfirType) -> None:
        super().__init__(input_type, input_type, ["i_0", "o_0", "o_1"])

    def to_hls(self) -> hls.HLSFunction:
        code_in_loop = [
            r"#type:i_0# copy_src = #read:i_0#;",
            r"bool #end_flag_val# = copy_src.end_flag;",
            # r"o_0.write(copy_src);",
            # r"o_1.write(copy_src);",
            r"#write_notrans:o_0,copy_src#",
            r"#write_notrans:o_1,copy_src#",
        ]
        return self.get_hls_function(code_in_loop)


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
                output_types.append(input_types[i].type_)
            else:
                output_types.append(input_types[i])
        output_type = TupleType(output_types)
        if parallel:
            output_type = ArrayType(output_type)
        ports.append("o_0")
        super().__init__(output_type, output_type, ports, parallel, specific_port_types)

    def to_hls(self) -> hls.HLSFunction:
        code_in_loop = ["bool #end_flag_val# = false;"]
        for i in range(len(self.in_ports)):
            code_in_loop.extend(
                [
                    f"#type:{self.in_ports[i].name}# gather_src_{i} = #read:i_{i}#;",
                    f"#end_flag_val# |= gather_src_{i}.end_flag;",
                    f"#peel:{self.in_ports[i].name},gather_src_{i},real_gather_src_{i}#",
                ]
            )
        code_in_loop += [
            r"#type:o_0# gather_result = {"
            + ", ".join(f"real_gather_src_{i}" for i in range(len(self.in_ports)))
            + ", #end_flag_val#"
            + r"};",
            r"#write_notrans:o_0,gather_result#",
        ]
        return self.get_hls_function(code_in_loop)


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

    def to_hls(self) -> hls.HLSFunction:
        code_in_loop = []
        code_in_loop.append(r"#type:i_0# scatter_src = #read:i_0#;")
        code_in_loop.append(r"bool #end_flag_val# = scatter_src.end_flag;")
        for i in range(len(self.out_ports)):
            code_in_loop.append(f"#write:o_{i},scatter_src.ele_{i}#")
        return self.get_hls_function(code_in_loop)


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

    def additional_info(self) -> str:
        return f"op: {self.op}"

    def to_hls(self) -> hls.HLSFunction:
        code_in_loop = [
            r"#type:i_0# binop_src_0 = #read:i_0#;",
            r"#type:i_1# binop_src_1 = #read:i_1#;",
            r"bool #end_flag_val# = binop_src_0.end_flag | binop_src_1.end_flag;",
            f"#type_inner:o_0# binop_out = {self.op.gen_repr('binop_src_0#may_ele:i_0#', 'binop_src_1#may_ele:i_1#')};",
            r"#write:o_0,binop_out#",
        ]
        return self.get_hls_function(code_in_loop)


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

    def additional_info(self) -> str:
        if self.op == UnaryOp.SELECT:
            return [f"op: {self.op}", f"select_index: {self.select_index}"]
        else:
            return f"op: {self.op}"

    def to_hls(self) -> hls.HLSFunction:
        if self.op == UnaryOp.GET_LENGTH:
            code_before_loop = [
                r"uint32_t length = 0;",
            ]
            code_in_loop = [
                r"#read:i_0#;",
                r"length++;",
            ]
            code_after_loop = [
                # r"#output_length# = 1;",
                r"#write:o_0,length#",
            ]
            return self.get_hls_function(code_in_loop, code_before_loop, code_after_loop)
        else:
            trans_dict = {
                UnaryOp.NOT: "{!unary_src#may_ele:i_0#}",
                UnaryOp.NEG: "{-unary_src#may_ele:i_0#}",
                UnaryOp.CAST_BOOL: "{(bool)(unary_src#may_ele:i_0#)}",
                UnaryOp.CAST_INT: "{(int32_t)(unary_src#may_ele:i_0#)}",
                UnaryOp.CAST_FLOAT: "{(ap_fixed<32, 16>)(unary_src#may_ele:i_0#)}",
                UnaryOp.SELECT: f"unary_src.ele_{self.select_index}",
                UnaryOp.GET_ATTR: f"unary_src.{self.select_index}",
            }
            code_in_loop = [
                f"#type:i_0# unary_src = #read:i_0#;",
                f"bool #end_flag_val# = unary_src.end_flag;",
                f"#type_inner:o_0# unary_out = {trans_dict[self.op]};",
                f"#write:o_0,unary_out#",
            ]
            return self.get_hls_function(code_in_loop)


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

    def to_hls(self) -> hls.HLSFunction:
        code_in_loop = [
            r"#type:i_data# cond_data = #read:i_data#;",
            r"#type:i_cond# cond = #read:i_cond#;",
            r"bool #end_flag_val# = cond_data.end_flag | cond.end_flag;",
            r"#peel:i_data,cond_data,real_cond_data#",
            r"#peel:i_cond,cond,real_cond#",
            r"#opt_type:o_0# cond_result = {real_cond_data, real_cond};",
            r"#write:o_0,cond_result#",
        ]
        return self.get_hls_function(code_in_loop)


class CollectComponent(Component):
    def __init__(self, input_type: DfirType) -> None:
        assert isinstance(input_type, ArrayType)
        assert isinstance(input_type.type_, OptionalType)
        output_type = ArrayType(input_type.type_.type_)
        super().__init__(input_type, output_type, ["i_0", "o_0"], parallel=True)

    def to_hls(self) -> hls.HLSFunction:
        # code_before_loop = [
        #     r"#output_length# = 0;",
        # ]
        code_in_loop = [
            r"#type:i_0# collect_src = #read:i_0#;",
            r"bool #end_flag_val# = collect_src.end_flag;",
            r"if (collect_src.valid.ele || #end_flag_val#) {",
            r"    #write:o_0,collect_src.data#",
            r"}",
        ]
        return self.get_hls_function(code_in_loop)


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

    def to_hls_list(
        self,
        func_key_name: str,
        func_transform_name: str,
        func_unit_name: str,
    ) -> List[hls.HLSFunction]:
        # Generate 1st func for key & transform pre-process
        code_in_loop = [
            r"#type:i_0# reduce_src = #read:i_0#;",
            r"bool #end_flag_val# = reduce_src.end_flag;",
            # f'hls::stream<#type:i_0#> reduce_key_in_stream("reduce_key_in_stream");',
            # r"#pragma HLS STREAM variable=reduce_key_in_stream depth=4",
            # f'hls::stream<#type:i_0#> reduce_transform_in_stream("reduce_transform_in_stream");',
            # r"#pragma HLS STREAM variable=reduce_transform_in_stream depth=4",
            # r"reduce_src.end_flag = true;",
            # f"reduce_key_in_stream.write(reduce_src);",
            # f"reduce_transform_in_stream.write(reduce_src);",
            # f'hls::stream<#type:i_reduce_key_out#> reduce_key_out_stream("reduce_key_out_stream");',
            # r"#pragma HLS STREAM variable=reduce_key_out_stream depth=4",
            # f'hls::stream<#type:i_reduce_transform_out#> reduce_transform_out_stream("reduce_transform_out_stream");',
            # r"#pragma HLS STREAM variable=reduce_transform_out_stream depth=4",
            f"#type:i_reduce_key_out# reduce_key_out;",
            f"#type:i_reduce_transform_out# reduce_transform_out;",
            f"#call_once:{func_key_name},reduce_src,reduce_key_out#;",
            f"#call_once:{func_transform_name},reduce_src,reduce_transform_out#;",
            # r"#type:i_reduce_key_out# reduce_key_out = reduce_key_out_stream.read();",
            r"reduce_key_out.end_flag = #end_flag_val#;",
            r"intermediate_key.write(reduce_key_out);",
            # r"#type:i_reduce_transform_out# reduce_transform_out = reduce_transform_out_stream.read();",
            r"reduce_transform_out.end_flag = #end_flag_val#;",
            r"intermediate_transform.write(reduce_transform_out);",
        ]
        stage_1_func = self.get_hls_function(code_in_loop, name_tail="pre_process")

        # Generate 2nd func for unit-reduce
        code_before_loop = [
            r"#reduce_key_struct# key_mem[MAX_NUM];",
            # r"#pragma HLS ARRAY_PARTITION variable=key_mem dim=0 complete",
            r"#pragma HLS BIND_STORAGE variable = key_mem type = RAM_2P impl = URAM",
            r"#pragma HLS dependence variable=key_mem inter false",
            r"#reduce_key_struct# key_buffer[L + 1];",
            r"#pragma HLS ARRAY_PARTITION variable=key_buffer dim=0 complete",
            r"uint32_t i_buffer[L + 1];",
            r"#pragma HLS ARRAY_PARTITION variable=i_buffer dim=0 complete",
            r"for (int i_clear_buffer = 0; i_clear_buffer < L + 1; i_clear_buffer++) {",
            r"#pragma HLS UNROLL",
            r"    i_buffer[i_clear_buffer] = MAX_NUM + 1;",
            r"}",
            # r"#pragma HLS ARRAY_PARTITION variable=key_mem complete dim=0",
            r"CLEAR_REDUCE_VALID: for (int i_reduce_clear = 0; i_reduce_clear < MAX_NUM; i_reduce_clear++) {",
            r"#pragma HLS UNROLL",
            r"    key_mem[i_reduce_clear].valid.ele = 0;",
            r"}",
        ]
        code_in_loop = [
            # the reduce_key_struct is {key, valid}, the loop uses one loop ahead
            # to clear the valid bit to 0 with pipeline
            f"#type:i_reduce_key_out# reduce_key_out = #read:intermediate_key#;",
            f"#type:i_reduce_transform_out# reduce_transform_out = #read:intermediate_transform#;",
            r"bool #end_flag_val# = reduce_key_out.end_flag | reduce_transform_out.end_flag;",
            # r"bool merged = false;",
            r"#peel:i_reduce_key_out,reduce_key_out,real_reduce_key_out#",
            r"#peel:i_reduce_transform_out,reduce_transform_out,real_reduce_transform_out#",
            # r"SCAN_BRAM_INTER_LOOP: for (int i_in_reduce = 0; i_in_reduce < MAX_NUM; i_in_reduce++) {",
            # r"#pragma HLS PIPELINE",
            # r"    #reduce_key_struct# cur_ele = key_mem[i_in_reduce];",
            # r"    if (!merged && !cur_ele.valid.ele) {",
            # r"        key_mem[i_in_reduce].valid.ele = 1;",
            # r"        key_mem[i_in_reduce].key#may_ele:i_reduce_key_out# = real_reduce_key_out;",
            # r"        key_mem[i_in_reduce].data#may_ele:i_reduce_transform_out# = real_reduce_transform_out;",
            # r"        merged = true;",
            # r"    } else if (!merged && cur_ele.valid.ele && #cmpeq:i_reduce_key_out,cur_ele.key,real_reduce_key_out#) {",
            # # new a stream to call the reduce unit
            # '        hls::stream<#type:o_reduce_unit_start_0#> reduce_unit_stream_0("reduce_unit_stream_0");',
            # r"#pragma HLS STREAM variable=reduce_unit_stream_0 depth=4",
            # '        hls::stream<#type:o_reduce_unit_start_1#> reduce_unit_stream_1("reduce_unit_stream_1");',
            # r"#pragma HLS STREAM variable=reduce_unit_stream_1 depth=4",
            # '        hls::stream<#type:i_reduce_unit_end#> reduce_unit_stream_out("reduce_unit_stream_out");',
            # r"#pragma HLS STREAM variable=reduce_unit_stream_out depth=4",
            # r"        #write:reduce_unit_stream_0,cur_ele.data#may_ele:i_reduce_transform_out#,#type:i_reduce_transform_out##",
            # r"        #write:reduce_unit_stream_1,real_reduce_transform_out,#type:i_reduce_transform_out##",
            # f"        #call_once:{func_unit_name},reduce_unit_stream_0,reduce_unit_stream_1,reduce_unit_stream_out#;",
            # r"        #type:i_reduce_unit_end# reduce_unit_out = #read:reduce_unit_stream_out#;",
            # r"        #peel:i_reduce_unit_end,reduce_unit_out,real_reduce_unit_out#",
            # r"        key_mem[i_in_reduce].data#may_ele:i_reduce_transform_out# = real_reduce_unit_out;",
            # r"        merged = true;",
            # r"    }",
            r"#reduce_key_struct# old_ele = key_mem[real_reduce_key_out#may_ele:i_reduce_key_out#];",
            r"for (int i_search_buffer = 0; i_search_buffer < L + 1; i_search_buffer++) {",
            r"#pragma HLS UNROLL",
            r"    {",
            r"        if (real_reduce_key_out#may_ele:i_reduce_key_out# == i_buffer[i_search_buffer]) old_ele = key_buffer[i_search_buffer];",
            r"    }",
            r"}",
            r"for (int i_move_buffer = 0; i_move_buffer < L; i_move_buffer++) {",
            r"#pragma HLS UNROLL",
            r"    {",
            r"        i_buffer[i_move_buffer] = i_buffer[i_move_buffer + 1];",
            r"        key_buffer[i_move_buffer] = key_buffer[i_move_buffer + 1];",
            r"    }",
            r"}",
            r"#reduce_key_struct# new_ele;",
            r"if (!old_ele.valid.ele) {",
            r"    new_ele.valid.ele = 1;",
            r"    new_ele.data = real_reduce_transform_out;",
            r"} else {",
            # '    hls::stream<#type:o_reduce_unit_start_0#> reduce_unit_stream_0("reduce_unit_stream_0");',
            # r"#pragma HLS STREAM variable=reduce_unit_stream_0 depth=1",
            # '    hls::stream<#type:o_reduce_unit_start_1#> reduce_unit_stream_1("reduce_unit_stream_1");',
            # r"#pragma HLS STREAM variable=reduce_unit_stream_1 depth=1",
            # '    hls::stream<#type:i_reduce_unit_end#> reduce_unit_stream_out("reduce_unit_stream_out");',
            # r"#pragma HLS STREAM variable=reduce_unit_stream_out depth=1",
            # r"    #write:reduce_unit_stream_0,old_ele.data,#type:i_reduce_transform_out##",
            # r"    #write:reduce_unit_stream_1,real_reduce_transform_out,#type:i_reduce_transform_out##",
            r"    #write_nostream:old_ele_data,old_ele.data,#type:i_reduce_transform_out##",
            r"    #write_nostream:new_ele_data,real_reduce_transform_out,#type:i_reduce_transform_out##",
            r"    #type:i_reduce_unit_end# reduce_unit_out;",
            f"    #call_once:{func_unit_name},old_ele_data,new_ele_data,reduce_unit_out#;",
            # r"    #type:i_reduce_unit_end# reduce_unit_out = #read:reduce_unit_stream_out#;",
            r"    #peel:i_reduce_unit_end,reduce_unit_out,real_reduce_unit_out#",
            r"    new_ele.data = real_reduce_unit_out;",
            r"}",
            r"key_mem[real_reduce_key_out#may_ele:i_reduce_key_out#] = new_ele;",
            r"key_buffer[L] = new_ele;",
            r"i_buffer[L] = real_reduce_key_out#may_ele:i_reduce_key_out#;",
        ]
        code_after_loop = [
            # r"#output_length# = 0;",
            r"WRITE_KEY_MEM_LOOP: for (int i_write_key_mem = 0; i_write_key_mem < MAX_NUM; i_write_key_mem++) {",
            r"#pragma HLS PIPELINE",
            r"    if (key_mem[i_write_key_mem].valid.ele) {",
            r"        #write_noend:o_0,key_mem[i_write_key_mem].data#may_ele:o_0##",
            # r"        #output_length#++;",
            r"    }",
            r"}",
            r"#write:o_0,key_mem[0].data#may_ele:o_0##",
        ]
        stage_2_func = self.get_hls_function(
            code_in_loop, code_before_loop, code_after_loop, name_tail="unit_reduce"
        )
        return [stage_1_func, stage_2_func]


class PlaceholderComponent(Component):
    def __init__(self, data_type: DfirType) -> None:
        super().__init__(data_type, data_type, ["i_0", "o_0"])

    def to_hls(self) -> hls.HLSFunction:
        assert False, "PlaceholderComponent should not be used in HLS"


class UnusedEndMarkerComponent(Component):
    def __init__(self, input_type: DfirType) -> None:
        super().__init__(input_type, None, ["i_0"])

    def to_hls(self) -> hls.HLSFunction:
        assert False, "UnusedEndMarkerComponent should not be used in HLS"
