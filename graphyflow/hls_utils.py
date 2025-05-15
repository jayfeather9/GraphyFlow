from __future__ import annotations
from enum import Enum
from typing import List, Optional, Union, Dict, Any, Tuple
import graphyflow.dataflow_ir_datatype as dftype
import graphyflow.dataflow_ir as dfir
import re


class HLSDataType(Enum):
    UINT = "uint16_t"
    INT = "int16_t"
    FLOAT = "ap_fixed<32, 16>"
    BOOL = "bool"
    EDGE = "edge_t"
    NODE = "node_t"

    def __repr__(self) -> str:
        return self.value


STD_TYPE_TRANSLATE_MAP = (
    (dftype.IntType(), HLSDataType.INT),
    (dftype.FloatType(), HLSDataType.FLOAT),
    (dftype.BoolType(), HLSDataType.BOOL),
    (dftype.SpecialType("edge"), HLSDataType.EDGE),
    (dftype.SpecialType("node"), HLSDataType.NODE),
)
STD_TYPES = ["uint16_t", "int16_t", "ap_fixed<32, 16>", "bool"]


class HLSDataTypeManager:
    _cnt = 0

    def __init__(
        self,
        node_props: Dict[str, dftype.DfirType],
        edge_props: Dict[str, dftype.DfirType],
    ) -> None:
        self.define_map = {}
        self.translate_map = {}
        self.type_preds = {}
        self.node_properties = {
            p_name: self.from_dfir_type(p_type) for p_name, p_type in node_props.items()
        }
        self.edge_properties = {
            p_name: self.from_dfir_type(p_type) for p_name, p_type in edge_props.items()
        }

    @classmethod
    def get_next_id(cls) -> int:
        cls._cnt += 1
        return cls._cnt

    def get_all_defines(self) -> List[str]:
        # first generate node_t & edge_t define, add to type_preds
        self.define_map[dftype.SpecialType("node")] = (
            "typedef struct {\n"
            + "\n".join(
                [f"    {t} {p_name};" for p_name, t in self.node_properties.items()]
            )
            + "\n} node_t;"
        )
        self.type_preds["node_t"] = list(self.node_properties.values())
        self.type_preds["node_t"] = [
            t for t in self.type_preds["node_t"] if t not in STD_TYPES
        ]
        self.translate_map[dftype.SpecialType("node")] = "node_t"
        self.define_map[dftype.SpecialType("edge")] = (
            "typedef struct {\n"
            + "\n".join(
                [f"    {t} {p_name};" for p_name, t in self.edge_properties.items()]
            )
            + "\n} edge_t;"
        )
        self.type_preds["edge_t"] = list(self.edge_properties.values())
        self.type_preds["edge_t"] = [
            t for t in self.type_preds["edge_t"] if t not in STD_TYPES
        ]
        self.translate_map[dftype.SpecialType("edge")] = "edge_t"
        # then iterate all translate_map
        all_defines = []
        waitings = list(self.translate_map.items())
        finished = []
        while waitings:
            dfir_type, type_name = waitings.pop(0)
            assert dfir_type in self.define_map
            if not all(t in finished for t in self.type_preds[type_name]):
                waitings.append((dfir_type, type_name))
                continue
            all_defines.append(self.define_map[dfir_type])
            finished.append(type_name)
        return all_defines

    def from_dfir_type(
        self, dfir_type: dftype.DfirType, sub_names: Optional[List[str]] = None
    ) -> str:
        if isinstance(dfir_type, dftype.ArrayType):
            dfir_type = dfir_type.type_
        for t, f in STD_TYPE_TRANSLATE_MAP:
            if dfir_type == t:
                return f.value
        if dfir_type in self.translate_map:
            return self.translate_map[dfir_type]
        else:
            assert isinstance(
                dfir_type, (dftype.TupleType, dftype.OptionalType)
            ), f"Unsupported type: {dfir_type}"
            assert dfir_type not in self.define_map, f"Type {dfir_type} already defined"
            if isinstance(dfir_type, dftype.TupleType):
                sub_types = [self.from_dfir_type(t) for t in dfir_type.types]
                name_id = HLSDataTypeManager.get_next_id()
                type_name = f'tuple_{"".join(st[:1] for st in sub_types)}_{name_id}_t'
                self.translate_map[dfir_type] = type_name
                self.type_preds[type_name] = [
                    t for t in sub_types if t not in STD_TYPES
                ]
                self.define_map[dfir_type] = (
                    (
                        "typedef struct {\n"
                        + "\n".join(
                            [f"    {t} ele_{i};" for i, t in enumerate(sub_types)]
                        )
                        + "\n} "
                        + type_name
                        + ";"
                    )
                    if sub_names is None
                    else (
                        "typedef struct {\n"
                        + "\n".join(
                            [
                                f"    {t} {sub_names[i]};"
                                for i, t in enumerate(sub_types)
                            ]
                        )
                        + "\n} "
                        + type_name
                        + ";"
                    )
                )
                return type_name
            elif isinstance(dfir_type, dftype.OptionalType):
                sub_type = self.from_dfir_type(dfir_type.type_)
                type_name = f"opt__of_{sub_type[:3]}_t"
                self.translate_map[dfir_type] = type_name
                self.type_preds[type_name] = [sub_type]
                self.define_map[dfir_type] = (
                    "typedef struct {\n"
                    + f"    {sub_type} data;\n"
                    + "    bool valid;\n"
                    + "} "
                    + type_name
                    + ";"
                )
                return type_name
            else:
                raise ValueError(f"Unsupported type: {dfir_type}")


class HLSFunction:
    def __init__(
        self,
        name: str,
        comp: dfir.Component,
        code_in_loop: List[str],
        code_before_loop: Optional[List[str]] = [],
        code_after_loop: Optional[List[str]] = [],
    ) -> None:
        assert (
            name not in global_hls_config.functions
        ), f"Function {name} already exists"
        self.name = name
        self.code_in_loop = code_in_loop
        self.code_before_loop = code_before_loop
        self.code_after_loop = code_after_loop
        self.comp = comp
        self.params = []
        self.change_length = any(
            "#output_length#" in line
            for line in (code_before_loop + code_in_loop + code_after_loop)
        )
        global_hls_config.functions[name] = self

    def __repr__(self) -> str:
        return f"HLSFunction(name={self.name}, inputs={self.inputs}, outputs={self.outputs}, code_in_loop={self.code_in_loop}, comp={self.comp})"


class HLSConfig:
    def __init__(self, header_name: str, source_name: str, top_name: str) -> None:
        self.header_name = header_name
        self.source_name = source_name
        self.top_name = top_name
        self.includes = []
        self.defines = []  # TODO: MAX_NUM
        self.structs = {}
        self.functions = {}
        self.PARTITION_FACTOR = 16
        self.STREAM_DEPTH = 4
        self.MAX_NUM = 32

    def __repr__(self) -> str:
        return (
            f"HLSConfig(header_name={self.header_name}, source_name={self.source_name})"
        )

    class ReduceSubFunc:
        def __init__(
            self, name: str, start_ports: List[dfir.Port], end_ports: List[dfir.Port]
        ) -> None:
            self.name = name
            self.start_ports = start_ports
            self.nxt_ports = start_ports
            self.end_ports = end_ports
            self.satisfieds = []
            self.sub_funcs: List[HLSFunction] = []

        def check_comp(self, comp: dfir.Component, sub_func: HLSFunction) -> None:
            if not any(p in self.nxt_ports for p in comp.in_ports):
                return
            assert all(
                p in self.nxt_ports
                or isinstance(p.connection.parent, dfir.ConstantComponent)
                for p in comp.in_ports
            )
            assert not isinstance(comp, dfir.ReduceComponent)
            self.sub_funcs.append(sub_func)
            self.nxt_ports = [p for p in self.nxt_ports if p not in comp.in_ports]
            for p in comp.out_ports:
                if p in self.end_ports:
                    self.satisfieds.append(p)
                else:
                    assert p.connected
                    self.nxt_ports.append(p.connection)
            self.nxt_ports = list(set(self.nxt_ports))

        def check_satisfied(self) -> bool:
            return self.satisfieds == self.end_ports

        def __repr__(self) -> str:
            return f"HLSConfig.ReduceSubFunc(name={self.name}, start_ports={self.start_ports}, end_ports={self.end_ports}, satisfieds={self.satisfieds}, sub_funcs={self.sub_funcs})"

        @classmethod
        def from_reduce(cls, comp: dfir.ReduceComponent) -> Tuple[
            Tuple[
                HLSConfig.ReduceSubFunc,
                HLSConfig.ReduceSubFunc,
                HLSConfig.ReduceSubFunc,
            ],
            Tuple[str, str, str],
        ]:
            reduce_key_func_name = f"{comp.name}_key_sub_func"
            reduce_transform_func_name = f"{comp.name}_transform_sub_func"
            reduce_unit_func_name = f"{comp.name}_unit_sub_func"
            key_func = cls(
                name=reduce_key_func_name,
                start_ports=[comp.get_port("o_reduce_key_in").connection],
                end_ports=[comp.get_port("i_reduce_key_out").connection],
            )
            transform_func = cls(
                name=reduce_transform_func_name,
                start_ports=[comp.get_port("o_reduce_transform_in").connection],
                end_ports=[comp.get_port("i_reduce_transform_out").connection],
            )
            unit_func = cls(
                name=reduce_unit_func_name,
                start_ports=[
                    comp.get_port("o_reduce_unit_start_0").connection,
                    comp.get_port("o_reduce_unit_start_1").connection,
                ],
                end_ports=[comp.get_port("i_reduce_unit_end").connection],
            )
            return (key_func, transform_func, unit_func), (
                reduce_key_func_name,
                reduce_transform_func_name,
                reduce_unit_func_name,
            )

    def generate_hls_code(
        self, global_graph, comp_col: dfir.ComponentCollection
    ) -> str:
        dt_manager = HLSDataTypeManager(
            global_graph.node_properties, global_graph.edge_properties
        )
        header_code = ""
        source_code = ""
        top_func_def = f"void {self.top_name}(\n"
        top_func_sub_funcs = []
        assert comp_col.inputs == []
        start_ports = []
        end_ports = comp_col.outputs
        constants_from_ports = {}
        source_code += f'#include "{self.header_name}"\n\n'
        source_code += "using namespace hls;\n\n"
        source_code_funcs_part = ""
        reduce_sub_funcs: List[HLSConfig.ReduceSubFunc] = []
        for comp in comp_col.topo_sort():
            assert not isinstance(comp, dfir.PlaceholderComponent)
            if isinstance(comp, dfir.UnusedEndMarkerComponent):
                pass
            elif isinstance(comp, dfir.ConstantComponent):
                assert comp.out_ports[0].connection not in constants_from_ports
                constants_from_ports[comp.out_ports[0].connection] = comp.value
            elif isinstance(comp, dfir.IOComponent):
                assert comp.io_type == dfir.IOComponent.IOType.INPUT
                for port in comp.ports:
                    assert port.port_type == dfir.PortType.OUT
                    start_ports.append(port.connection)
                    top_func_def += f"    stream<{dt_manager.from_dfir_type(port.data_type)}> &{port.connection.unique_name},\n"
            elif isinstance(comp, dfir.ReduceComponent):
                sub_funcs, sub_func_names = HLSConfig.ReduceSubFunc.from_reduce(comp)
                reduce_sub_funcs.extend(sub_funcs)
                (
                    reduce_key_func_name,
                    reduce_transform_func_name,
                    reduce_unit_func_name,
                ) = sub_func_names
                reduce_pre_func, reduce_unit_func = comp.to_hls_list(
                    func_key_name=reduce_key_func_name,
                    func_transform_name=reduce_transform_func_name,
                    func_unit_name=reduce_unit_func_name,
                )
                reduce_pre_func_str = f"static void {reduce_pre_func.name}(\n"
                in_port = comp.get_port("i_0")
                input_type = in_port.data_type
                key_out_type = comp.get_port("i_reduce_key_out").data_type
                accumulate_type = comp.get_port("i_reduce_unit_end").data_type
                reduce_pre_func_str += "".join(
                    [
                        f"    stream<{dt_manager.from_dfir_type(input_type)}> &i_0,\n"
                        f"    stream<{dt_manager.from_dfir_type(key_out_type)}> &intermediate_key,\n"
                        f"    stream<{dt_manager.from_dfir_type(accumulate_type)}> &intermediate_transform,\n"
                        "    uint16_t input_length\n"
                        ") {\n",
                        f"    LOOP_{reduce_pre_func.name}:\n",
                        "    for (uint16_t i = 0; i < input_length; i++) {\n",
                        "#pragma HLS PIPELINE\n",
                    ]
                )

                call_regex = r"#call:([\w_,]+)#"
                call_once_regex = r"#call_once:([\w_,]+)#"

                def manage_call(line: str) -> str:
                    match = re.search(call_regex, line)
                    if match:
                        args = match.group(1).split(",")
                        func_name = args[0]
                        args = args[1:]
                        return f"{func_name}({', '.join(args)}, input_length);"
                    match = re.search(call_once_regex, line)
                    if match:
                        args = match.group(1).split(",")
                        func_name = args[0]
                        args = args[1:]
                        return f"{func_name}({', '.join(args)}, 1);"
                    return line

                for line in reduce_pre_func.code_in_loop:
                    # replace #type# and #read#, only i_0 in reduce_pre_func
                    line = line.replace(
                        "#type:i_0#", dt_manager.from_dfir_type(input_type)
                    )
                    if in_port in constants_from_ports:
                        line = line.replace(
                            f"#read:i_0#", f"{constants_from_ports[in_port]}"
                        )
                    else:
                        line = line.replace(f"#read:i_0#", f"i_0.read()")
                    line = manage_call(line)
                    reduce_pre_func_str += f"        {line}\n"
                reduce_pre_func_str += "    }\n"
                reduce_pre_func_str += "}\n"
                intermediate_key_port = dfir.Port(
                    "o_intermediate_key", dfir.EmptyNode(output_type=key_out_type)
                )
                intermediate_key_i_port = dfir.Port(
                    "i_intermediate_key", dfir.EmptyNode(input_type=key_out_type)
                )
                intermediate_key_port.connect(intermediate_key_i_port)
                intermediate_transform_port = dfir.Port(
                    "o_intermediate_transform",
                    dfir.EmptyNode(output_type=accumulate_type),
                )
                intermediate_transform_i_port = dfir.Port(
                    "i_intermediate_transform",
                    dfir.EmptyNode(input_type=accumulate_type),
                )
                intermediate_transform_port.connect(intermediate_transform_i_port)
                reduce_pre_func.params = [
                    ("i_0", dt_manager.from_dfir_type(input_type), in_port, True),
                    (
                        "intermediate_key",
                        dt_manager.from_dfir_type(key_out_type),
                        intermediate_key_port,
                        False,
                    ),
                    (
                        "intermediate_transform",
                        dt_manager.from_dfir_type(accumulate_type),
                        intermediate_transform_port,
                        False,
                    ),
                    ("input_length", True),
                ]

                source_code_funcs_part += reduce_pre_func_str + "\n\n"

                # handle reduce_unit_func
                reduce_key_struct = dt_manager.from_dfir_type(
                    dftype.TupleType(
                        [
                            key_out_type,
                            accumulate_type,
                            dftype.BoolType(),
                        ]
                    ),
                    sub_names=["key", "data", "valid"],
                )
                codes_before_loop = [
                    line.replace("#reduce_key_struct#", reduce_key_struct).replace(
                        "#partition_factor#", str(self.PARTITION_FACTOR)
                    )
                    for line in reduce_unit_func.code_before_loop
                ]
                reduce_unit_func_str = "".join(
                    [
                        f"static void {reduce_unit_func.name}(\n",
                        f"    stream<{dt_manager.from_dfir_type(key_out_type)}> &intermediate_key,\n",
                        f"    stream<{dt_manager.from_dfir_type(accumulate_type)}> &intermediate_transform,\n",
                        f"    stream<{dt_manager.from_dfir_type(accumulate_type)}> &o_0,\n",
                        "    uint16_t &output_length,\n",
                        "    uint16_t input_length\n",
                        ") {\n",
                    ]
                    + [f"    {line}\n" for line in codes_before_loop]
                    + [
                        f"    LOOP_{reduce_unit_func.name}:\n",
                        "    for (uint16_t i = 0; i < input_length; i++) {\n",
                        "#pragma HLS PIPELINE\n",
                    ]
                )
                port2type = {
                    port.name: dt_manager.from_dfir_type(port.data_type)
                    for port in comp.ports
                }
                reduce_unit_func.params = [
                    (
                        "intermediate_key",
                        dt_manager.from_dfir_type(key_out_type),
                        intermediate_key_i_port,
                        True,
                    ),
                    (
                        "intermediate_transform",
                        dt_manager.from_dfir_type(accumulate_type),
                        intermediate_transform_i_port,
                        True,
                    ),
                    (
                        "o_0",
                        port2type["o_0"],
                        reduce_unit_func.comp.get_port("o_0"),
                        False,
                    ),
                    ("output_length", False),
                    ("input_length", True),
                ]
                for line in reduce_unit_func.code_in_loop:
                    for port, type in port2type.items():
                        line = line.replace(f"#type:{port}#", type)
                        line = line.replace("#reduce_key_struct#", reduce_key_struct)
                        # look for #cmpeq:type_port,a,b# and if type is edge, assert False, if node, use a.id == b.id, otherwise use a == b
                        cmp_regex = r"#cmpeq:([\w_]+),([\w_\.]+),([\w_\.]+)#"
                        match = re.search(cmp_regex, line)
                        if match and port == match.group(1):
                            _, cmp_a, cmp_b = match.groups()
                            if isinstance(
                                comp.get_port(port).data_type,
                                (dftype.BoolType, dftype.IntType),
                            ):
                                line = line.replace(
                                    match.group(0), f"{cmp_a} == {cmp_b}"
                                )
                            elif comp.get_port(port).data_type == dftype.SpecialType(
                                "node"
                            ):
                                line = line.replace(
                                    match.group(0), f"{cmp_a}.id == {cmp_b}.id"
                                )
                            else:
                                assert False, "Not supported type comparing"
                        line = manage_call(line)
                    # find #read:xxx#, change to xxx.read()
                    line = re.sub(r"#read:(\w+)#", r"\1.read()", line)
                    reduce_unit_func_str += f"        {line}\n"
                reduce_unit_func_str += "    }\n"
                for line in reduce_unit_func.code_after_loop:
                    line = line.replace("#output_length#", "output_length")
                    reduce_unit_func_str += f"    {line}\n"
                reduce_unit_func_str += "}\n"
                source_code_funcs_part += reduce_unit_func_str + "\n\n"
                top_func_sub_funcs.extend([reduce_pre_func, reduce_unit_func])
            else:
                func = comp.to_hls()
                # check if any port connected to EndMarker
                unused_ports = []
                for port in comp.ports:
                    if port.connected and isinstance(
                        port.connection.parent, dfir.UnusedEndMarkerComponent
                    ):
                        unused_ports.append(port.name)
                # check & add for reduce sub functions
                for sub_func in reduce_sub_funcs:
                    sub_func.check_comp(comp, func)
                # generate function str
                func_str = f"static void {func.name}(\n"
                # type, read, output_length, opt_type
                port2type = {}
                for port in func.comp.ports:
                    port2type[port.name] = dt_manager.from_dfir_type(port.data_type)
                    if port in constants_from_ports or port.name in unused_ports:
                        continue
                    func_str += f"    stream<{port2type[port.name]}> &{port.name},\n"
                    func.params.append(
                        (
                            port.unique_name,
                            port2type[port.name],
                            port,
                            port.port_type == dfir.PortType.IN,
                        )
                    )
                if any("#output_length#" in line for line in func.code_in_loop):
                    func_str += "    uint16_t &output_length,\n"
                    func.params.append(("output_length", False))
                func_str += "    uint16_t input_length\n"
                func.params.append(("input_length", True))
                func_str += ")"

                source_code += func_str + ";\n\n"
                func_str += " {\n"

                def manage_line(line: str) -> str:
                    for port_name in unused_ports:
                        if (
                            f"#read:{port_name}#" in line
                            or f"{port_name}.write" in line
                        ):
                            return ""
                    line = line.replace(f"#output_length#", "output_length")
                    for port, type in port2type.items():
                        line = line.replace(f"#type:{port}#", type)
                        if port in constants_from_ports:
                            line = line.replace(
                                f"#read:{port}#", f"{constants_from_ports[port]}"
                            )
                        else:
                            line = line.replace(f"#read:{port}#", f"{port}.read()")
                        line = line.replace(f"#opt_type:{port}#", port2type[port])
                    return line

                for line in func.code_before_loop:
                    func_str += f"    {manage_line(line)}\n"
                func_str += f"    LOOP_{func.name}:\n"
                func_str += "    for (uint16_t i = 0; i < input_length; i++) {\n"
                func_str += "#pragma HLS PIPELINE\n"
                for line in func.code_in_loop:
                    func_str += f"        {manage_line(line)}\n"
                func_str += "    }\n"
                for line in func.code_after_loop:
                    func_str += f"    {manage_line(line)}\n"
                func_str += "}\n"
                source_code_funcs_part += func_str + "\n\n"
                top_func_sub_funcs.append(func)

        # manage top function end ports & input len
        for port in end_ports:
            top_func_def += f"    stream<{dt_manager.from_dfir_type(port.data_type)}> &{port.unique_name},\n"
        top_func_def += "    uint16_t input_length\n"
        top_func_def += ")"

        # manage reduce sub functions
        assert all(sub_func.check_satisfied() for sub_func in reduce_sub_funcs)
        for sub_func in reduce_sub_funcs:
            sub_func_code = f"static void {sub_func.name}(\n"
            for port in sub_func.start_ports + sub_func.end_ports:
                sub_func_code += f"    stream<{dt_manager.from_dfir_type(port.data_type)}> &{port.unique_name},\n"
            if any(sub_sub_func.change_length for sub_sub_func in sub_func.sub_funcs):
                sub_func_code += "    uint16_t &output_length,\n"
            sub_func_code += "    uint16_t input_length\n"
            sub_func_code += ") {\n"
            sub_func_code += self.generate_sub_func_code(
                sub_func.start_ports, sub_func.end_ports, sub_func.sub_funcs
            )
            top_func_sub_funcs = [
                cur_sub_func
                for cur_sub_func in top_func_sub_funcs
                if cur_sub_func not in sub_func.sub_funcs
            ]
            sub_func_code += "}\n"
            source_code += sub_func_code + "\n\n"

        # add functions part
        source_code += source_code_funcs_part

        # manage structure defines
        all_defines = dt_manager.get_all_defines()
        header_code += "\n\n".join(all_defines)

        # add top module
        header_code += "\n\n" + top_func_def + ";"
        top_func_code = top_func_def + " {\n"
        top_func_code += "#pragma HLS dataflow\n"
        if any(sub_sub_func.change_length for sub_sub_func in top_func_sub_funcs):
            top_func_code += "    uint16_t output_length = input_length;\n"
            print(end_ports)
        top_func_code += self.generate_sub_func_code(
            start_ports, end_ports, top_func_sub_funcs
        )
        top_func_code += "}\n"
        source_code += top_func_code + "\n\n"

        # add start & end define for header
        header_name_for_define = f'__{self.header_name.upper().replace(".", "_")}__'
        header_code = (
            f"#ifndef {header_name_for_define}"
            + f"\n#define {header_name_for_define}\n\n"
            + "#include <stdint.h>\n#include <ap_int.h>\n#include <hls_stream.h>\n\n"
            + "using namespace hls;\nusing namespace std;\n\n"
            + f"#define MAX_NUM {self.MAX_NUM}\n\n"
            + header_code
            + "\n\n"
            + f"#endif // {header_name_for_define}\n"
        )

        return header_code, source_code

    def generate_sub_func_code(
        self,
        start_ports: List[dfir.Port],
        end_ports: List[dfir.Port],
        functions: List[HLSFunction],
    ) -> str:
        output_len_name = (
            "output_length"
            if any(sub_sub_func.change_length for sub_sub_func in functions)
            else "input_length"
        )
        port2var_name = {}
        adding_codes = ""
        for sub_sub_func in functions:
            adding_codes += (
                f"    uint16_t {sub_sub_func.name}_input_len = {output_len_name};\n"
            )
            call_code = f"    {sub_sub_func.name}(\n"
            call_params = []
            for param in sub_sub_func.params:
                if len(param) == 2 and param[1] == True:
                    call_params.append(f"{sub_sub_func.name}_input_len")
                elif len(param) == 2 and param[1] == False:
                    call_params.append("output_length")
                else:
                    port_name, port_type, cur_port, is_in = param
                    if is_in:
                        if any(cur_port == st_p for st_p in start_ports):
                            sub_sub_func_var_name = f"{cur_port.unique_name}"
                        else:
                            sub_sub_func_var_name = port2var_name[cur_port.connection]
                        call_params.append(sub_sub_func_var_name)
                    else:
                        if any(cur_port == ed_p for ed_p in end_ports):
                            sub_sub_func_var_name = f"{cur_port.unique_name}"
                        else:
                            sub_sub_func_var_name = (
                                f"{sub_sub_func.name}_{cur_port.unique_name}"
                            )
                            port2var_name[cur_port] = sub_sub_func_var_name
                            adding_codes += (
                                f"    stream<{port_type}> {sub_sub_func_var_name};\n"
                                f"    #pragma HLS STREAM variable={sub_sub_func_var_name} depth={self.STREAM_DEPTH} \n"
                            )
                        call_params.append(sub_sub_func_var_name)

            call_code += ",\n".join(f"        {param}" for param in call_params)
            call_code += "\n    );\n"
            adding_codes += call_code
        return adding_codes


global_hls_config = HLSConfig(
    header_name="graphyflow.h",
    source_name="graphyflow.cpp",
    top_name="graphyflow",
)
