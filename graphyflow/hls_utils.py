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

    def gen_basic_print(hls_name, type_name):
        if type_name == HLSDataType.INT or type_name == HLSDataType.BOOL:
            return f"""#ifdef PRINT_DEFS
#define print_{hls_name}(var) \\
    printf("%d", var.ele);
#endif"""
        elif type_name == HLSDataType.FLOAT:
            return f"""#ifdef PRINT_DEFS
#define print_{hls_name}(var) \\
    printf("%lf", (double)var.ele);
#endif"""
        else:
            assert False, f"Not supported for ({hls_name}, {type_name})"


STD_TYPE_TRANSLATE_MAP = (
    (dftype.IntType(), HLSDataType.INT),
    (dftype.FloatType(), HLSDataType.FLOAT),
    (dftype.BoolType(), HLSDataType.BOOL),
    (dftype.SpecialType("node"), HLSDataType.NODE),
    (dftype.SpecialType("edge"), HLSDataType.EDGE),
)
STD_TYPES = ["uint32_t", "int32_t", "ap_fixed<32, 16>", "bool"]


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
        self.to_outer_type = {}
        self.basic_basic_types = {}
        self.basic_type_names = []
        for dfir_type, hls_type in STD_TYPE_TRANSLATE_MAP:
            type_name = f"basic_{hls_type.value[:5]}_t"
            if not isinstance(dfir_type, dfir.SpecialType):
                self.basic_basic_types[type_name] = hls_type
            self.define_map[dfir_type] = (
                [f"    {hls_type.value} ele;"],
                f"{type_name}",
            )
            self.type_preds[type_name] = []
            self.translate_map[dfir_type] = type_name
            self.basic_type_names.append(type_name)
        self.node_properties = {
            p_name: self.from_dfir_type(p_type, outer=False) for p_name, p_type in node_props.items()
        }
        self.edge_properties = {
            p_name: self.from_dfir_type(p_type, outer=False) for p_name, p_type in edge_props.items()
        }

    @classmethod
    def get_next_id(cls) -> int:
        cls._cnt += 1
        return cls._cnt

    def get_all_defines(self) -> List[str]:
        def gen_define(def_map_ele):
            return "typedef struct {\n" + "\n".join(def_map_ele[0]) + "\n} " + f"{def_map_ele[1]};"

        # first generate node_t & edge_t define, add to type_preds
        self.define_map[dftype.SpecialType("node")] = (
            [f"    {t} {p_name};" for p_name, t in self.node_properties.items()],
            "basic_node__t",
        )
        self.type_preds["basic_node__t"] = list(self.node_properties.values())
        self.type_preds["basic_node__t"] = [t for t in self.type_preds["basic_node__t"] if t not in STD_TYPES]
        self.translate_map[dftype.SpecialType("node")] = "basic_node__t"
        self.define_map[dftype.SpecialType("edge")] = (
            [f"    {t} {p_name};" for p_name, t in self.edge_properties.items()],
            "basic_edge__t",
        )
        self.type_preds["basic_edge__t"] = list(self.edge_properties.values())
        self.type_preds["basic_edge__t"] = [t for t in self.type_preds["basic_edge__t"] if t not in STD_TYPES]
        self.translate_map[dftype.SpecialType("edge")] = "basic_edge__t"
        # then iterate all translate_map
        all_defines = []
        waitings = list(self.translate_map.items())
        finished = self.basic_type_names[:]

        # special deal with print_bool()
        all_defines.append(
            """#ifdef PRINT_DEFS
#define print_bool(var) \\
    printf("%d", var);
#endif"""
        )

        # print(finished)
        def gen_print_func(def_map_ele):
            if def_map_ele[1] in self.basic_basic_types.keys():
                return HLSDataType.gen_basic_print(def_map_ele[1], self.basic_basic_types[def_map_ele[1]])
            params = []
            for one_def_ele in def_map_ele[0]:
                print(f"{one_def_ele=}")
                one_def_ele = one_def_ele[:-1].strip().split(" ")
                assert len(one_def_ele) >= 2, f"{one_def_ele} len != 2"
                params.append(("".join(one_def_ele[:-1]), one_def_ele[-1]))
            LB, RB = "{", "}"
            print(def_map_ele[1])
            print(params)
            return (
                "#ifdef PRINT_DEFS\n"
                + f"#define print_{def_map_ele[1]}(var) \\\n"
                + f'    printf("{def_map_ele[1]} {LB}\\n"); \\\n'
                + ' printf(", "); \\\n'.join(
                    f"    print_{sub_type}(var.{param});" for sub_type, param in params
                )
                + f' \\\n    printf("{RB}\\n"); \n'
                + "#endif"
            )

        while waitings:
            dfir_type, type_name = waitings.pop(0)
            assert (
                dfir_type in self.define_map
            ), f"dfir_type {dfir_type} with type_name {type_name} not in define map"
            # print(dfir_type, type_name, self.type_preds[type_name])
            if not all(t in finished for t in self.type_preds[type_name]) and type_name not in [
                "basic_int32_t",
                "basic_ap_fi_t",
                "basic_bool_t",
            ]:
                waitings.append((dfir_type, type_name))
                continue
            all_defines.append(gen_define(self.define_map[dfir_type]))
            all_defines.append(gen_print_func(self.define_map[dfir_type]))
            finished.append(type_name)
        for dfir_type, outer_name in self.to_outer_type.items():
            ori_type_name = self.translate_map[dfir_type]
            assert ori_type_name in finished
            new_map_ele = list(self.define_map[dfir_type])
            ori_params = []
            for param_line in new_map_ele[0]:
                ori_params.append(param_line.split(" ")[-1][:-1])
            # add transition funcs
            transition_func1 = (
                f"#define {ori_type_name}_to_{outer_name}(origin_name, new_name, end_flag_val) \\\n"
                + f"    {outer_name} new_name;\\\n"
                + "\\\n".join(f"    new_name.{param} = origin_name.{param};" for param in ori_params)
                + "\\\n    new_name.end_flag = end_flag_val;\n\n"
            )
            transition_func2 = (
                f"#define {outer_name}_to_{ori_type_name}(origin_name, new_name) \\\n"
                + f"    {ori_type_name} new_name;\\\n"
                + "\\\n".join(f"    new_name.{param} = origin_name.{param};" for param in ori_params)
            )
            new_map_ele[0].append("    bool end_flag;")
            new_map_ele[1] = outer_name
            all_defines.append(gen_define(new_map_ele))
            all_defines.append(gen_print_func(new_map_ele))
            all_defines.append(transition_func1 + transition_func2)
        return all_defines

    def from_dfir_type(
        self,
        dfir_type: dftype.DfirType,
        outer=True,
        sub_names: Optional[List[str]] = None,
    ) -> str:
        if isinstance(dfir_type, dftype.ArrayType):
            dfir_type = dfir_type.type_
        if dfir_type in self.translate_map:
            if outer:
                if not dfir_type in self.to_outer_type:
                    self.to_outer_type[dfir_type] = "outer_" + self.translate_map[dfir_type]
                return self.to_outer_type[dfir_type]
            return self.translate_map[dfir_type]
        else:
            assert isinstance(
                dfir_type, (dftype.TupleType, dftype.OptionalType)
            ), f"Unsupported type: {dfir_type}"
            assert dfir_type not in self.define_map, f"Type {dfir_type} already defined"
            if isinstance(dfir_type, dftype.TupleType):
                sub_types = [self.from_dfir_type(t, outer=False) for t in dfir_type.types]
                name_id = HLSDataTypeManager.get_next_id()
                type_name = f'tuple_{"".join(st[:1] for st in sub_types)}_{name_id}_t'
                self.translate_map[dfir_type] = type_name
                self.type_preds[type_name] = [t for t in sub_types if t not in STD_TYPES]
                self.define_map[dfir_type] = (
                    ([f"    {t} ele_{i};" for i, t in enumerate(sub_types)], type_name)
                    if sub_names is None
                    else (
                        [f"    {t} {sub_names[i]};" for i, t in enumerate(sub_types)],
                        type_name,
                    )
                )
                if outer:
                    if not dfir_type in self.to_outer_type:
                        self.to_outer_type[dfir_type] = "outer_" + self.translate_map[dfir_type]
                    return self.to_outer_type[dfir_type]
                return type_name
            elif isinstance(dfir_type, dftype.OptionalType):
                sub_type = self.from_dfir_type(dfir_type.type_, False)
                type_name = f"opt__of_{sub_type[:3]}_t"
                self.translate_map[dfir_type] = type_name
                self.type_preds[type_name] = [sub_type]
                self.define_map[dfir_type] = (
                    [f"    {sub_type} data;", "    basic_bool_t valid;"],
                    type_name,
                )
                if outer:
                    if not dfir_type in self.to_outer_type:
                        self.to_outer_type[dfir_type] = "outer_" + self.translate_map[dfir_type]
                    return self.to_outer_type[dfir_type]
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
        assert name not in global_hls_config.functions, f"Function {name} already exists"
        self.name = name
        self.code_in_loop = code_in_loop
        self.code_before_loop = code_before_loop
        self.code_after_loop = code_after_loop
        self.comp = comp
        self.params = []
        # self.change_length = any(
        #     "#output_length#" in line
        #     for line in (code_before_loop + code_in_loop + code_after_loop)
        # )
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
        self.MAX_NUM = 256
        self.BUFFER_LENGTH = 4

    def __repr__(self) -> str:
        return f"HLSConfig(header_name={self.header_name}, source_name={self.source_name})"

    class ReduceSubFunc:
        def __init__(self, name: str, start_ports: List[dfir.Port], end_ports: List[dfir.Port]) -> None:
            self.name = name
            self.start_ports = start_ports
            self.nxt_ports = start_ports
            self.end_ports = end_ports
            self.satisfieds = []
            self.sub_funcs: List[HLSFunction] = []

        def check_comp(self, comp: dfir.Component, sub_func: HLSFunction) -> bool:
            if not any(p in self.nxt_ports for p in comp.in_ports):
                return False
            assert all(
                p in self.nxt_ports or isinstance(p.connection.parent, dfir.ConstantComponent)
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
            return True

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

    def generate_hls_code(self, global_graph, comp_col: dfir.ComponentCollection) -> str:
        def gen_read_name(name: str, port: dfir.Port, is_sub_func: bool = False) -> str:
            if is_sub_func:
                return name
            return f"{name}.read()"

        dt_manager = HLSDataTypeManager(global_graph.node_properties, global_graph.edge_properties)
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
                constants_from_ports[comp.out_ports[0].connection] = "{ " + f"{comp.value}" + ", false }"
            elif isinstance(comp, dfir.IOComponent):
                assert comp.io_type == dfir.IOComponent.IOType.INPUT
                for port in comp.ports:
                    assert port.port_type == dfir.PortType.OUT
                    start_ports.append(port.connection)
                    top_func_def += f"    stream<{dt_manager.from_dfir_type(port.data_type)}> &{port.connection.unique_name},\n"
            elif isinstance(comp, dfir.ReduceComponent):
                sub_funcs, sub_func_names = HLSConfig.ReduceSubFunc.from_reduce(comp)
                port2type = {port.name: dt_manager.from_dfir_type(port.data_type) for port in comp.ports}
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
                        f"    stream<{dt_manager.from_dfir_type(accumulate_type)}> &intermediate_transform\n"
                        # "    uint32_t input_length\n"
                        ") {\n",
                        f"    LOOP_{reduce_pre_func.name}:\n",
                        "    while (true) {\n",
                        "#pragma HLS PIPELINE\n",
                    ]
                )

                call_regex = r"#call:([\w_,]+)#"
                call_once_regex = r"#call_once:([\w_,]+)#"

                def manage_call(line: str) -> str:
                    i = 0
                    while i < len(line) and line[i] == " ":
                        i += 1
                    match = re.search(call_regex, line)
                    if match:
                        args = match.group(1).split(",")
                        func_name = args[0]
                        args = args[1:]
                        return " " * i + f"{func_name}({', '.join(args)});"
                    match = re.search(call_once_regex, line)
                    if match:
                        args = match.group(1).split(",")
                        func_name = args[0]
                        args = args[1:]
                        return " " * i + f"{func_name}({', '.join(args)});"
                    return line

                for line in reduce_pre_func.code_in_loop:
                    # replace #type# and #read#, only i_0 in reduce_pre_func
                    for port, p_type in port2type.items():
                        line = line.replace(f"#type:{port}#", p_type)
                    line = line.replace("#end_flag_val#", "end_flag_val")
                    if in_port in constants_from_ports:
                        line = line.replace(f"#read:i_0#", f"{constants_from_ports[in_port]}")
                    else:
                        line = line.replace(f"#read:i_0#", f"{gen_read_name('i_0', in_port)}")
                    line = manage_call(line)
                    reduce_pre_func_str += f"        {line}\n"
                reduce_pre_func_str += "        if (end_flag_val) break;\n"
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
                    # ("input_length", True),
                ]

                source_code_funcs_part += reduce_pre_func_str + "\n\n"

                # handle reduce_unit_func
                reduce_key_struct = dt_manager.from_dfir_type(
                    dftype.TupleType(
                        [
                            accumulate_type,
                            dftype.BoolType(),
                        ]
                    ),
                    sub_names=["data", "valid"],
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
                        f"    stream<{dt_manager.from_dfir_type(accumulate_type)}> &o_0\n",
                        # "    uint32_t &output_length,\n",
                        # "    uint32_t input_length\n",
                        ") {\n",
                    ]
                    + [f"    {line}\n" for line in codes_before_loop]
                    + [
                        f"    LOOP_{reduce_unit_func.name}:\n",
                        "    while (true) {\n",
                        "#pragma HLS PIPELINE\n",
                    ]
                )
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
                    # ("output_length", False),
                    # ("input_length", True),
                ]
                for line in reduce_unit_func.code_in_loop:
                    for inter_name, inter_p in [
                        ("intermediate_key", intermediate_key_i_port),
                        ("intermediate_transform", intermediate_transform_i_port),
                        (
                            "reduce_unit_stream_out",
                            reduce_unit_func.comp.get_port("i_reduce_unit_end"),
                        ),
                    ]:
                        line = line.replace(
                            f"#read:{inter_name}#",
                            f"{gen_read_name(inter_name, inter_p)}",
                        )
                    for port, p_type in port2type.items():
                        line = line.replace(f"#type:{port}#", p_type)
                        line = line.replace("#reduce_key_struct#", reduce_key_struct)
                        line = line.replace("#end_flag_val#", "end_flag_val")
                        # if f"#may_ele:{port}#" in line:
                        #     print(reduce_unit_func.comp.get_port(port))
                        #     print(line)
                        line = line.replace(
                            f"#may_ele:{port}#",
                            (".ele" if reduce_unit_func.comp.get_port(port).data_type.is_basic_type else ""),
                        )
                        # look for #cmpeq:type_port,a,b# and if type is edge, assert False, if node, use a.id == b.id, otherwise use a == b
                        line = line.replace(
                            f"#read:{port}#",
                            f"{gen_read_name(port, reduce_unit_func.comp.get_port(port))}",
                        )
                        cmp_regex = r"#cmpeq:([\w_]+),([\w_\.]+),([\w_\.]+)#"
                        match = re.search(cmp_regex, line)
                        if match and port == match.group(1):
                            _, cmp_a, cmp_b = match.groups()
                            if isinstance(
                                comp.get_port(port).data_type,
                                (dftype.BoolType, dftype.IntType),
                            ):
                                line = line.replace(match.group(0), f"{cmp_a} == {cmp_b}")
                            elif comp.get_port(port).data_type == dftype.SpecialType("node"):
                                line = line.replace(match.group(0), f"{cmp_a}.id.ele == {cmp_b}.id.ele")
                            else:
                                assert False, "Not supported type comparing"
                        line = manage_call(line)
                    # for write
                    if "#write" in line:
                        i = 0
                        while i < len(line) and line[i] == " ":
                            i += 1
                        indent_space = " " * (i + 8)
                        potential_match_str = line[i:]
                        pattern = re.compile(
                            r"^#write:([a-zA-Z\_0-9]+),([a-zA-Z\.\_0-9]+),([a-zA-Z\_0-9]+)#$"
                        )
                        match = pattern.match(potential_match_str)
                        pattern_nostream = re.compile(
                            r"^#write_nostream:([a-zA-Z\_0-9]+),([a-zA-Z\.\_0-9]+),([a-zA-Z\_0-9]+)#$"
                        )
                        match_nostream = pattern_nostream.match(potential_match_str)
                        assert (
                            match or match_nostream
                        ), f"Error: Line '{line.strip()}' include '#write' but with wrong format."
                        nostream = False
                        if match_nostream:
                            match = match_nostream
                            nostream = True

                        tgt_port_name, ori_var, tgt_outer_type = (
                            match.group(1),
                            match.group(2),
                            match.group(3),
                        )
                        assert tgt_outer_type[:6] == "outer_"
                        tgt_inner_type = tgt_outer_type[6:]
                        line = "" if not nostream else (indent_space + f"{tgt_outer_type} {tgt_port_name};\n")
                        line += indent_space + r"{" + "\n"
                        line += (
                            indent_space
                            + f"    {tgt_inner_type}_to_{tgt_outer_type}({ori_var}, tmp_{tgt_outer_type}_var, true);\n"
                        )
                        line += (
                            (indent_space + f"    {tgt_port_name}.write(tmp_{tgt_outer_type}_var);\n")
                            if not nostream
                            else (indent_space + f"    {tgt_port_name} = tmp_{tgt_outer_type}_var;\n")
                        )
                        line += indent_space + r"}" + "\n"
                        reduce_unit_func_str += line
                    elif "#peel" in line:
                        pattern = re.compile(r"^#peel:([a-zA-Z\_0-9]+),([a-zA-Z\.\_0-9]+),([a-zA-Z\_0-9]+)#$")
                        i = 0
                        while i < len(line) and line[i] == " ":
                            i += 1
                        indent_space = " " * (i + 8)
                        potential_match_str = line[i:]
                        match = pattern.match(potential_match_str)
                        assert match, f"Error: Line '{line.strip()}' include '#peel' but with wrong format."

                        tgt_port_name, ori_var, new_var = (
                            match.group(1),
                            match.group(2),
                            match.group(3),
                        )
                        tgt_outer_type = port2type[tgt_port_name]
                        assert tgt_outer_type[:6] == "outer_"
                        tgt_inner_type = tgt_outer_type[6:]
                        reduce_unit_func_str += (
                            indent_space + f"{tgt_outer_type}_to_{tgt_inner_type}({ori_var}, {new_var});\n"
                        )
                    else:
                        reduce_unit_func_str += f"        {line}\n"
                reduce_unit_func_str += "        if (end_flag_val) break;\n"
                reduce_unit_func_str += "    }\n"
                for line in reduce_unit_func.code_after_loop:
                    # line = line.replace("#output_length#", "output_length")
                    for port, p_type in port2type.items():
                        line = line.replace(
                            f"#may_ele:{port}#",
                            (".ele" if reduce_unit_func.comp.get_port(port).data_type.is_basic_type else ""),
                        )
                    if "#write" in line:
                        i = 0
                        while i < len(line) and line[i] == " ":
                            i += 1
                        indent_space = " " * (i + 4)
                        potential_match_str = line[i:]
                        pattern = re.compile(r"^#write:([a-zA-Z\_0-9]+),([a-zA-Z\.\[\]\_0-9]+)#$")
                        pattern_noend = re.compile(r"^#write_noend:([a-zA-Z\_0-9]+),([a-zA-Z\.\[\]\_0-9]+)#$")
                        pattern_notrans = re.compile(
                            r"^#write_notrans:([a-zA-Z\_0-9]+),([a-zA-Z\.\[\]\_0-9]+)#$"
                        )
                        match = pattern.match(potential_match_str)
                        match_noend = pattern_noend.match(potential_match_str)
                        match_notrans = pattern_notrans.match(potential_match_str)
                        assert (
                            match or match_noend or match_notrans
                        ), f"Error2: Line '{line.strip()}' include '#write' but with wrong format."
                        ending = "false" if match_noend else "true"
                        do_trans = False if match_notrans else True

                        if match_noend:
                            match = match_noend
                        elif match_notrans:
                            match = match_notrans
                        tgt_port_name, ori_var = match.group(1), match.group(2)
                        tgt_outer_type = port2type[tgt_port_name]
                        assert tgt_outer_type[:6] == "outer_"
                        tgt_inner_type = tgt_outer_type[6:]
                        line = indent_space + r"{" + "\n"
                        line += (
                            (
                                indent_space
                                + f"    {tgt_inner_type}_to_{tgt_outer_type}({ori_var}, tmp_{tgt_outer_type}_var, {ending});\n"
                            )
                            if do_trans
                            else ""
                        )
                        line += indent_space + f"    {tgt_port_name}.write(tmp_{tgt_outer_type}_var);\n"
                        line += indent_space + r"}" + "\n"
                        reduce_unit_func_str += line
                    else:
                        reduce_unit_func_str += f"    {line}\n"
                reduce_unit_func_str += "}\n"
                source_code_funcs_part += reduce_unit_func_str + "\n\n"
                top_func_sub_funcs.extend([reduce_pre_func, reduce_unit_func])
            else:
                func = comp.to_hls()
                # check if any port connected to EndMarker
                unused_ports = []
                for port in comp.ports:
                    if port.connected and isinstance(port.connection.parent, dfir.UnusedEndMarkerComponent):
                        unused_ports.append(port.name)
                is_sub_func = False
                # check & add for reduce sub functions
                for sub_func in reduce_sub_funcs:
                    is_sub_func |= sub_func.check_comp(comp, func)
                # generate function str
                func_str = f"static void {func.name}(\n"
                # type, read, output_length, opt_type
                port2type = {}
                for port in func.comp.ports:
                    port2type[port.name] = dt_manager.from_dfir_type(port.data_type)
                    if port in constants_from_ports or port.name in unused_ports:
                        continue
                    if not is_sub_func:
                        func_str += f"    stream<{port2type[port.name]}> &{port.name},\n"
                    else:
                        func_str += f"    {port2type[port.name]} &{port.name},\n"
                    func.params.append(
                        (
                            port.unique_name,
                            port2type[port.name],
                            port,
                            port.port_type == dfir.PortType.IN,
                        )
                    )
                # delete last ","
                if func_str[-2:] == ",\n":
                    func_str = func_str[:-2] + "\n"

                # if any("#output_length#" in line for line in func.code_in_loop):
                #     func_str += "    uint32_t &output_length,\n"
                #     func.params.append(("output_length", False))
                # func_str += "    uint32_t input_length\n"
                # func.params.append(("input_length", True))
                func_str += ")"

                source_code += func_str + ";\n\n"
                func_str += " {\n"
                if is_sub_func:
                    func_str += "#pragma HLS INLINE\n"
                has_end_flag = False

                def manage_line(line: str, indent: int) -> str:
                    if "write" in line:
                        print(func.name, line, unused_ports)
                    # for unused ports
                    for port_name in unused_ports:
                        if (
                            f"#read:{port_name}#" in line
                            or f"{port_name}.write" in line
                            or f"#write:{port_name}" in line
                        ):
                            return ""
                    # # output_length
                    # line = line.replace(f"#output_length#", "output_length")
                    # end flag val
                    nonlocal has_end_flag
                    if "#end_flag_val#" in line:
                        has_end_flag = True
                    line = line.replace("#end_flag_val#", "end_flag_val")
                    # type declaration & read
                    for port, type in port2type.items():
                        line = line.replace(f"#type:{port}#", type)
                        line = line.replace(f"#type_inner:{port}#", type[6:])
                        if comp.get_port(port) in constants_from_ports:
                            line = line.replace(
                                f"#read:{port}#",
                                f"{constants_from_ports[comp.get_port(port)]}",
                            )
                        else:
                            line = line.replace(
                                f"#read:{port}#",
                                f"{gen_read_name(port, comp.get_port(port), is_sub_func)}",
                            )
                        line = line.replace(f"#opt_type:{port}#", port2type[port][6:])

                        port_type = comp.get_port(port).data_type
                        if isinstance(port_type, dfir.ArrayType):
                            port_type = port_type.type_
                        line = line.replace(
                            f"#may_ele:{port}#",
                            (".ele" if port_type.is_basic_type else ""),
                        )
                    indent_space = " " * 4 * indent
                    # for write
                    if "#write" in line:
                        i = 0
                        while i < len(line) and line[i] == " ":
                            i += 1
                        indent_space += " " * i
                        potential_match_str = line[i:]
                        pattern = re.compile(r"^#write:([a-zA-Z\_0-9]+),([a-zA-Z\.\[\]\_0-9]+)#$")
                        pattern_notrans = re.compile(
                            r"^#write_notrans:([a-zA-Z\_0-9]+),([a-zA-Z\.\[\]\_0-9]+)#$"
                        )
                        match = pattern.match(potential_match_str)
                        match_notrans = pattern_notrans.match(potential_match_str)
                        assert (
                            match or match_notrans
                        ), f"Error2: Line '{line.strip()}' include '#write' but with wrong format."
                        do_trans = False if match_notrans else True

                        if match_notrans:
                            match = match_notrans

                        tgt_port_name, ori_var = match.group(1), match.group(2)
                        tgt_outer_type = port2type[tgt_port_name]
                        assert tgt_outer_type[:6] == "outer_"
                        tgt_inner_type = tgt_outer_type[6:]
                        line = indent_space + r"{" + "\n"
                        line += (
                            (
                                indent_space
                                + f"    {tgt_inner_type}_to_{tgt_outer_type}({ori_var}, tmp_{tgt_outer_type}_var, end_flag_val);\n"
                            )
                            if do_trans
                            else ""
                        )
                        target_final_var = f"tmp_{tgt_outer_type}_var" if do_trans else ori_var
                        if is_sub_func:
                            line += indent_space + f"    {tgt_port_name} = {target_final_var};\n"
                        else:
                            line += indent_space + f"    {tgt_port_name}.write({target_final_var});\n"
                        line += indent_space + r"}" + "\n"
                        return line
                    elif "#peel" in line:
                        pattern = re.compile(r"^#peel:([a-zA-Z\_0-9]+),([a-zA-Z\.\_0-9]+),([a-zA-Z\_0-9]+)#$")
                        i = 0
                        while i < len(line) and line[i] == " ":
                            i += 1
                        indent_space += " " * i
                        potential_match_str = line[i:]
                        match = pattern.match(potential_match_str)
                        assert match, f"Error: Line '{line.strip()}' include '#peel' but with wrong format."

                        tgt_port_name, ori_var, new_var = (
                            match.group(1),
                            match.group(2),
                            match.group(3),
                        )
                        tgt_outer_type = port2type[tgt_port_name]
                        assert tgt_outer_type[:6] == "outer_"
                        tgt_inner_type = tgt_outer_type[6:]
                        return indent_space + f"{tgt_outer_type}_to_{tgt_inner_type}({ori_var}, {new_var});\n"
                    else:
                        return indent_space + line + "\n"

                for line in func.code_before_loop:
                    func_str += f"{manage_line(line, 1)}"
                # sub func no loop
                if not is_sub_func:
                    func_str += f"    LOOP_{func.name}:\n"
                    # func_str += "    for (uint32_t i = 0; i < input_length; i++) {\n"
                    func_str += "    while (true) {\n"
                    func_str += "#pragma HLS PIPELINE\n"
                for line in func.code_in_loop:
                    func_str += f"{manage_line(line, 2)}"
                assert has_end_flag
                if not is_sub_func:
                    func_str += "        if (end_flag_val) break;\n"
                    func_str += "    }\n"
                for line in func.code_after_loop:
                    func_str += f"{manage_line(line, 1)}"
                func_str += "}\n"
                source_code_funcs_part += func_str + "\n\n"
                top_func_sub_funcs.append(func)

        # manage top function end ports & input len
        for port in end_ports:
            top_func_def += f"    stream<{dt_manager.from_dfir_type(port.data_type)}> &{port.unique_name},\n"
        # top_func_def += "    uint32_t input_length\n"
        if top_func_def[-2:] == ",\n":
            top_func_def = top_func_def[:-2] + "\n"
        top_func_def += ")"

        # manage reduce sub functions
        assert all(sub_func.check_satisfied() for sub_func in reduce_sub_funcs)
        for sub_func in reduce_sub_funcs:
            sub_func_code = f"static void {sub_func.name}(\n"
            for port in sub_func.start_ports + sub_func.end_ports:
                # sub_func_code += f"    stream<{dt_manager.from_dfir_type(port.data_type)}> &{port.unique_name},\n"
                sub_func_code += f"    {dt_manager.from_dfir_type(port.data_type)} &{port.unique_name},\n"
            # if any(sub_sub_func.change_length for sub_sub_func in sub_func.sub_funcs):
            #     sub_func_code += "    uint32_t &output_length,\n"
            # sub_func_code += "    uint32_t input_length\n"
            if sub_func_code[-2:] == ",\n":
                sub_func_code = sub_func_code[:-2] + "\n"
            sub_func_code += ") {\n#pragma HLS INLINE\n"
            sub_func_code += self.generate_sub_func_code(
                sub_func.start_ports, sub_func.end_ports, sub_func.sub_funcs, True
            )
            top_func_sub_funcs = [
                cur_sub_func for cur_sub_func in top_func_sub_funcs if cur_sub_func not in sub_func.sub_funcs
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
        # if any(sub_sub_func.change_length for sub_sub_func in top_func_sub_funcs):
        #     # top_func_code += "    uint32_t output_length = input_length;\n"
        #     print(end_ports)
        top_func_code += self.generate_sub_func_code(start_ports, end_ports, top_func_sub_funcs)
        top_func_code += "}\n"
        source_code += top_func_code + "\n\n"

        # add start & end define for header
        header_name_for_define = f'__{self.header_name.upper().replace(".", "_")}__'
        header_code = (
            f"#ifndef {header_name_for_define}"
            + f"\n#define {header_name_for_define}\n\n"
            + "#include <stdint.h>\n#include <ap_int.h>\n#include <hls_stream.h>\n\n"
            + "using namespace hls;\nusing namespace std;\n\n"
            + f"#define MAX_NUM {self.MAX_NUM}\n"
            + f"#define L {self.BUFFER_LENGTH}\n\n"
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
        is_sub_sub_func: bool = False,
    ) -> str:
        port2var_name = {}
        adding_codes = ""
        for sub_sub_func in functions:
            call_code = f"    {sub_sub_func.name}(\n"
            call_params = []
            for param in sub_sub_func.params:
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
                        sub_sub_func_var_name = f"{sub_sub_func.name}_{cur_port.unique_name}"
                        port2var_name[cur_port] = sub_sub_func_var_name
                        adding_codes += (
                            (
                                f"    stream<{port_type}> {sub_sub_func_var_name};\n"
                                f"    #pragma HLS STREAM variable={sub_sub_func_var_name} depth={self.STREAM_DEPTH}\n"
                            )
                            if not is_sub_sub_func
                            else (f"    {port_type} {sub_sub_func_var_name};\n")
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
