from __future__ import annotations
from enum import Enum
from typing import List, Optional, Union, Dict, Any, Tuple
import graphyflow.dataflow_ir_datatype as dftype
import graphyflow.dataflow_ir as dfir
import re


class HLSDataType(Enum):
    UINT = "uint32_t"
    INT = "int32_t"
    FLOAT = "ap_fixed<32, 16>"
    BOOL = "bool"
    EDGE = "edge_t"
    NODE = "node_t"

    def __repr__(self) -> str:
        return self.value


class HLSDataTypeManager:
    _cnt = 0

    def __init__(self) -> None:
        self.define_map = {}
        self.translate_map = {}

    @classmethod
    def get_next_id(cls) -> int:
        cls._cnt += 1
        return cls._cnt

    def from_dfir_type(
        self, dfir_type: dftype.DfirType, sub_names: Optional[List[str]] = None
    ) -> str:
        translate_map = (
            (dftype.IntType(), HLSDataType.INT),
            (dftype.FloatType(), HLSDataType.FLOAT),
            (dftype.BoolType(), HLSDataType.BOOL),
            (dftype.SpecialType("edge"), HLSDataType.EDGE),
            (dftype.SpecialType("node"), HLSDataType.NODE),
        )
        if isinstance(dfir_type, dftype.ArrayType):
            dfir_type = dfir_type.type_
        for t, f in translate_map:
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
                self.define_map[dfir_type] = (
                    (
                        r"typedef struct {\n"
                        + "\n".join(
                            [f"    {t} ele_{i};" for i, t in enumerate(sub_types)]
                        )
                        + "\n} "
                        + type_name
                        + ";"
                    )
                    if sub_names is None
                    else (
                        r"typedef struct {\n"
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
                self.define_map[dfir_type] = (
                    r"typedef struct {\n"
                    + f"    {sub_type} data;\n"
                    + r"    bool valid;\n"
                    + r"} "
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
        global_hls_config.functions[name] = self

    def __repr__(self) -> str:
        return f"HLSFunction(name={self.name}, inputs={self.inputs}, outputs={self.outputs}, code_in_loop={self.code_in_loop}, comp={self.comp})"


class HLSConfig:
    def __init__(self, header_name: str, source_name: str) -> None:
        self.header_name = header_name
        self.source_name = source_name
        self.includes = []
        self.defines = []  # TODO: MAX_NUM
        self.structs = {}
        self.functions = {}
        self.PARTITION_FACTOR = 16

    def __repr__(self) -> str:
        return (
            f"HLSConfig(header_name={self.header_name}, source_name={self.source_name})"
        )

    def generate_hls_code(self, comp_col: dfir.ComponentCollection) -> str:
        dt_manager = HLSDataTypeManager()
        header_code = ""
        source_code = ""
        constants_from_ports = {}
        source_code += f'#include "{self.header_name}"\n\n'
        source_code += "using namespace hls;\n\n"
        reduce_sub_funcs = {}
        for comp in comp_col.topo_sort():

            assert not isinstance(comp, dfir.PlaceholderComponent)
            if isinstance(comp, dfir.UnusedEndMarkerComponent):
                pass
            elif isinstance(comp, dfir.ConstantComponent):
                assert comp.out_ports[0].connection not in constants_from_ports
                constants_from_ports[comp.out_ports[0].connection] = comp.value
            elif isinstance(comp, dfir.IOComponent):
                pass
            elif isinstance(comp, dfir.ReduceComponent):
                reduce_key_func_name = f"{comp.name}_key_sub_func"
                reduce_transform_func_name = f"{comp.name}_transform_sub_func"
                reduce_unit_func_name = f"{comp.name}_unit_sub_func"
                reduce_sub_funcs[comp.name] = {
                    "key": reduce_key_func_name,
                    "transform": reduce_transform_func_name,
                    "unit": reduce_unit_func_name,
                }
                inter_key_var_name = f"{comp.name}_key"
                inter_transform_var_name = f"{comp.name}_transform"
                out_stream_name = f"{comp.name}_out"
                reduce_pre_func, reduce_unit_func = comp.to_hls_list(
                    func_key_name=reduce_key_func_name,
                    func_transform_name=reduce_transform_func_name,
                    func_unit_name=reduce_unit_func_name,
                    inter_key_var_name=inter_key_var_name,
                    inter_transform_var_name=inter_transform_var_name,
                    out_stream_name=out_stream_name,
                )
                reduce_pre_func_str = f"static void {reduce_pre_func.name}(\n"
                in_port = comp.get_port("i_0")
                input_type = in_port.data_type
                key_out_type = comp.get_port("i_reduce_key_out").data_type
                accumulate_type = comp.get_port("i_reduce_unit_end").data_type
                reduce_pre_func_str += "".join(
                    [
                        f"    stream<{dt_manager.from_dfir_type(input_type)}> &i_0,\n"
                        f"    stream<{dt_manager.from_dfir_type(key_out_type)}> &{inter_key_var_name},\n"
                        f"    stream<{dt_manager.from_dfir_type(accumulate_type)}> &{inter_transform_var_name},\n"
                        "    uint32_t input_length\n"
                        ") {\n",
                        f"    LOOP_{reduce_pre_func.name}:\n",
                        "    for (uint32_t i = 0; i < input_length; i++) {\n",
                        "#pragma HLS PIPELINE\n",
                    ]
                )

                call_regex = r"#call:([\w_,]+)#"

                def manage_call(line: str) -> str:
                    match = re.search(call_regex, line)
                    if match:
                        args = match.group(1).split(",")
                        func_name = args[0]
                        args = args[1:]
                        return f"{func_name}({', '.join(args)}, input_length)"
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

                source_code += reduce_pre_func_str + "\n\n"

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
                        f"    stream<{dt_manager.from_dfir_type(key_out_type)}> &{inter_key_var_name},\n",
                        f"    stream<{dt_manager.from_dfir_type(accumulate_type)}> &{inter_transform_var_name},\n",
                        f"    stream<{dt_manager.from_dfir_type(accumulate_type)}> &{out_stream_name},\n",
                        "    uint32_t input_length\n",
                        ") {\n",
                    ]
                    + [f"    {line}\n" for line in codes_before_loop]
                    + [
                        f"    LOOP_{reduce_unit_func.name}:\n",
                        "    for (uint32_t i = 0; i < input_length; i++) {\n",
                        "#pragma HLS PIPELINE\n",
                    ]
                )
                port2type = {
                    port.name: dt_manager.from_dfir_type(port.data_type)
                    for port in reduce_unit_func.comp.ports
                }
                for line in reduce_unit_func.code_in_loop:
                    for port, type in port2type.items():
                        line = line.replace(f"#type:{port}#", type)
                        line = manage_call(line)
                    # find #read:xxx#, change to xxx.read()
                    line = re.sub(r"#read:(\w+)#", r"\1.read()", line)
                    reduce_unit_func_str += f"        {line}\n"
                reduce_unit_func_str += "    }\n"
                for line in reduce_unit_func.code_after_loop:
                    line = line.replace("#output_length#", "output_length")
                    reduce_unit_func_str += f"    {line}\n"
                reduce_unit_func_str += "}\n"
                source_code += reduce_unit_func_str + "\n\n"
            else:
                func = comp.to_hls()
                func_str = f"static void {func.name}(\n"
                # type, read, output_length, opt_type
                port2type = {}
                for port in func.comp.ports:
                    port2type[port.name] = dt_manager.from_dfir_type(port.data_type)
                    if port in constants_from_ports:
                        continue
                    func_str += f"    stream<{port2type[port.name]}> {port.name},\n"
                if any("#output_length#" in line for line in func.code_in_loop):
                    func_str += "    uint32_t &output_length,\n"
                func_str += "    uint32_t input_length\n"
                func_str += "){\n"

                def manage_line(line: str) -> str:
                    line = line.replace(f"#output_length#", "output_length")
                    for port, type in port2type.items():
                        line = line.replace(f"#type:{port}#", type)
                        if port in constants_from_ports:
                            line = line.replace(
                                f"#read:{port}#", f"{constants_from_ports[port]}"
                            )
                        else:
                            line = line.replace(f"#read:{port}#", f"{port}.read()")
                    return line

                for line in func.code_before_loop:
                    func_str += f"    {manage_line(line)}\n"
                func_str += f"    LOOP_{func.name}:\n"
                func_str += "    for (uint32_t i = 0; i < input_length; i++) {\n"
                func_str += "#pragma HLS PIPELINE\n"
                for line in func.code_in_loop:
                    func_str += f"        {manage_line(line)}\n"
                func_str += "    }\n"
                for line in func.code_after_loop:
                    func_str += f"    {manage_line(line)}\n"
                func_str += "}\n"
                source_code += func_str + "\n\n"
        return header_code, source_code


global_hls_config = HLSConfig("graphyflow.h", "graphyflow.cpp")
