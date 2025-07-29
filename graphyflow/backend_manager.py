from __future__ import annotations
from enum import Enum
from typing import List, Optional, Union, Dict, Any, Tuple
import graphyflow.dataflow_ir_datatype as dftype
import graphyflow.dataflow_ir as dfir
import re


from graphyflow.backend_defines import (
    INDENT_UNIT,
    HLSType,
    HLSBasicType,
    HLSFunction,
    HLSCodeLine,
    HLSExpr,
    HLSExprT,
    HLSVar,
    CodeAssign,
    CodeBlock,
    CodeBreak,
    CodeCall,
    CodeComment,
    CodeFor,
    CodeIf,
    CodePragma,
    CodeVarDecl,
    CodeWhile,
    CodeWriteStream,
)
from graphyflow.backend_utils import generate_demux, generate_omega_network, generate_stream_zipper


class BackendManager:
    """Manages the entire HLS code generation process from a ComponentCollection."""

    def __init__(self):
        self.PE_NUM = 8
        self.STREAM_DEPTH = 4
        self.MAX_NUM = 256  # For ReduceComponent key_mem size
        self.L = 4  # For ReduceComponent buffer size
        # Mappings to store results of type analysis
        self.type_map: Dict[dftype.DfirType, HLSType] = {}
        self.batch_type_map: Dict[HLSType, HLSType] = {}
        self.struct_definitions: Dict[str, Tuple[HLSType, List[str]]] = {}
        self.unstreamed_funcs: set[str] = set()

        # State for Phase 2 & 3
        self.hls_functions: Dict[int, HLSFunction] = {}
        self.top_level_stream_decls: List[Tuple[CodeVarDecl, CodePragma]] = []

        self.reduce_internal_streams: Dict[int, Dict[str, HLSVar]] = {}

        # 存储所有由 utils 生成的辅助函数 (omega, demux, tree etc.)
        self.utility_functions: List[HLSFunction] = []
        # 将 ReduceComponent 的 ID 映射到其专属的辅助模块和中间流
        self.reduce_helpers: Dict[int, Dict[str, Any]] = {}

        self.global_graph_store = None
        self.comp_col_store = None

    def generate_backend(
        self, comp_col: dfir.ComponentCollection, global_graph: Any, top_func_name: str
    ) -> Tuple[str, str]:
        """
        Main entry point to generate HLS header and source files.
        """
        self.global_graph_store = global_graph
        self.comp_col_store = comp_col
        header_name = f"{top_func_name}.h"

        # 1. Correctly discover top-level I/O ports
        top_level_inputs = []
        for comp in comp_col.components:
            if (
                isinstance(comp, dfir.IOComponent)
                and comp.io_type == dfir.IOComponent.IOType.INPUT
            ):
                if comp.get_port("o_0").connected:
                    top_level_inputs.append(comp.get_port("o_0").connection)

        top_level_outputs = comp_col.outputs

        # --- 后续阶段不变 ---
        # Phase 1: Type Analysis
        self._analyze_and_map_types(comp_col)
        # Phase 2: Function Definition and Stream Instantiation
        self._define_functions_and_streams(comp_col, top_func_name)
        # Phase 3: Code Body Generation
        self._translate_functions()

        # --- Phase 4: Final Code Assembly (using corrected I/O ports) ---

        # 1. Build the top-level function signature string
        top_params = []
        for p in top_level_inputs:
            # Input ports are connections, so we use the port 'p' directly
            param_type = HLSType(
                HLSBasicType.STREAM, [self.batch_type_map[self.type_map[p.data_type]]]
            )
            top_params.append(f"hls::stream<{param_type.sub_types[0].name}>& {p.unique_name}")
        for p in top_level_outputs:
            param_type = HLSType(
                HLSBasicType.STREAM, [self.batch_type_map[self.type_map[p.data_type]]]
            )
            top_params.append(f"hls::stream<{param_type.sub_types[0].name}>& {p.unique_name}")

        top_func_sig = (
            f"void {top_func_name}(\n{INDENT_UNIT}" + f",\n{INDENT_UNIT}".join(top_params) + "\n)"
        )

        # 2. Generate file contents
        header_code = self._generate_header_file(top_func_name, top_func_sig)
        source_code = self._generate_source_file(header_name, top_func_name, top_func_sig)

        return header_code, source_code

    def debug_msgs(self, phases=[1, 2, 3]):
        if 1 in phases:
            # For demonstration, print discovered types
            print("--- Discovered Struct Definitions ---")
            for name, (hls_type, members) in self.struct_definitions.items():
                print(f"Struct: {name}")
                print(hls_type.gen_decl(members))

            print("\n--- Discovered Batch Type Mappings ---")
            for base, batch in self.batch_type_map.items():
                print(f"Base Type: {base.name} -> Batch Type: {batch.name}")
        if 2 in phases:
            # For demonstration, print discovered functions and streams
            print("\n--- Discovered HLS Functions and Signatures ---")
            for func in self.hls_functions.values():
                param_str = ", ".join([f"{p.type.name}& {p.name}" for p in func.params])
                stream_status = "Streamed" if func.streamed else "Unstreamed (by-ref)"
                print(f"Function: {func.name} ({stream_status})")
                print(f"  Signature: void {func.name}({param_str});")

            print("\n--- Intermediate Streams for Top-Level Function ---")
            for decl in self.top_level_stream_decls:
                print(decl.gen_code(indent_lvl=1).strip())
        if 3 in phases:
            print("\n--- Generated HLS Code Bodies (Phase 3) ---")
            for func in self.hls_functions.values():
                print(f"// ======== Code for function: {func.name} ========")
                for code_line in func.codes:
                    # The gen_code method of each HLSCodeLine object produces the C++ string
                    print(code_line.gen_code(indent_lvl=1), end="")
                print(f"// ======== End of function: {func.name} ========\n")

    def _find_unstreamed_funcs(self, comp_col: dfir.ComponentCollection):
        """
        Identifies all components that are part of a ReduceComponent's sub-graph.
        Their function signatures will use pass-by-reference instead of streams.
        """
        self.unstreamed_funcs.clear()
        q = []
        for comp in comp_col.components:
            if isinstance(comp, dfir.ReduceComponent):
                for port_name in [
                    "o_reduce_key_in",
                    "o_reduce_transform_in",
                    "o_reduce_unit_start_0",
                    "o_reduce_unit_start_1",
                ]:
                    if comp.get_port(port_name).connected:
                        q.append(comp.get_port(port_name).connection.parent)

        visited = set()
        while q:
            comp: dfir.Component = q.pop(0)
            if comp.readable_id in visited:
                continue
            visited.add(comp.readable_id)

            if isinstance(comp, dfir.ReduceComponent):
                continue

            self.unstreamed_funcs.add(f"{comp.__class__.__name__[:5]}_{comp.readable_id}")
            for port in comp.out_ports:
                if port.connected:
                    q.append(port.connection.parent)

    def _analyze_and_map_types(self, comp_col: dfir.ComponentCollection):
        """
        Phase 1: Traverse all components and ports to analyze and map DFIR types
        to HLS types, including special batching types for streams.
        """
        self.type_map.clear()
        self.batch_type_map.clear()
        self.struct_definitions.clear()

        # First, identify all functions that are part of a reduce operation
        self._find_unstreamed_funcs(comp_col)

        # Iterate all ports of all components to discover all necessary types
        for comp in comp_col.components:
            for port in comp.ports:
                dfir_type = port.data_type
                is_array_type = False
                if isinstance(dfir_type, dftype.ArrayType):
                    dfir_type = dfir_type.type_
                    is_array_type = True

                if dfir_type:
                    # Get the base HLS type (e.g., a struct without batching wrappers)
                    base_hls_type = self._to_hls_type(dfir_type, is_array_type)

                    # If it's a stream port (default case) and not for a reduce sub-function,
                    # create a corresponding batch type.
                    is_stream_port = comp.name not in self.unstreamed_funcs
                    if is_stream_port:
                        self._get_batch_type(base_hls_type)

    def _define_functions_and_streams(
        self, comp_col: dfir.ComponentCollection, top_func_name: str
    ):
        """
        Phase 2: Creates HLSFunction objects, defines their signatures, and
        identifies the intermediate streams needed for the top-level function.
        """
        self.hls_functions.clear()
        self.top_level_stream_decls.clear()
        self.reduce_internal_streams.clear()
        self.utility_functions.clear()
        self.reduce_helpers.clear()

        processed_sub_comp_ids = set()

        for comp in comp_col.components:
            if isinstance(comp, dfir.ReduceComponent):
                # 1. Create the two main HLS functions for the Reduce component
                pre_process_func = HLSFunction(name=f"{comp.name}_pre_process", comp=comp)
                unit_reduce_func = HLSFunction(name=f"{comp.name}_unit_reduce", comp=comp)

                # 2. Define their signatures
                # pre_process: stream in, two intermediate streams out
                in_type = self.type_map[comp.get_port("i_0").data_type]
                key_type = self.type_map[comp.get_port("i_reduce_key_out").data_type]
                transform_type = self.type_map[comp.get_port("i_reduce_transform_out").data_type]

                i_0_type = HLSType(
                    HLSBasicType.STREAM,
                    sub_types=[self.batch_type_map[in_type]],
                )
                i_key_type = HLSType(
                    HLSBasicType.STREAM,
                    sub_types=[self.batch_type_map[key_type]],
                )
                i_t_type = HLSType(
                    HLSBasicType.STREAM,
                    sub_types=[self.batch_type_map[transform_type]],
                )
                pre_process_func.params = [
                    HLSVar("i_0", i_0_type),
                    HLSVar("intermediate_key", i_key_type),
                    HLSVar("intermediate_transform", i_t_type),
                ]

                # 创建 kt_pair_t 类型
                kt_pair_type = HLSType(
                    HLSBasicType.STRUCT,
                    sub_types=[key_type, transform_type],
                    struct_name=f"kt_pair_{comp.readable_id}_t",
                    struct_prop_names=["key", "transform"],
                )
                self.struct_definitions[kt_pair_type.name] = (kt_pair_type, ["key", "transform"])

                member = [kt_pair_type, HLSType(HLSBasicType.BOOL), HLSType(HLSBasicType.UINT8)]
                member_names = ["data", "end_flag", "end_pos"]

                kt_wrap_type = HLSType(
                    HLSBasicType.STRUCT,
                    member,
                    struct_name=f"net_wrapper_{kt_pair_type.name}_t",
                    struct_prop_names=member_names,
                )
                self.struct_definitions[kt_wrap_type.name] = (kt_wrap_type, member_names)

                kt_wrap_stream = HLSType(HLSBasicType.STREAM, sub_types=[kt_wrap_type])
                kt_wrap_array = HLSType(
                    HLSBasicType.ARRAY, sub_types=[kt_wrap_stream], array_dims=["PE_NUM"]
                )

                # a. Define the un-batched, single stream types for parallel PEs
                # single_key_stream_type = HLSType(HLSBasicType.STREAM, sub_types=[key_type])
                # single_transform_stream_type = HLSType(
                #     HLSBasicType.STREAM, sub_types=[transform_type]
                # )

                out_dfir_type = comp.get_port("o_0").data_type
                base_data_type = self._get_batch_type(self.type_map[out_dfir_type.type_])
                single_data_stream_type = HLSType(HLSBasicType.STREAM, sub_types=[base_data_type])

                # b. Create ARRAY types for the stream arrays
                # key_stream_array_type = HLSType(
                #     HLSBasicType.ARRAY, sub_types=[single_key_stream_type], array_dims=["PE_NUM"]
                # )
                # transform_stream_array_type = HLSType(
                #     HLSBasicType.ARRAY,
                #     sub_types=[single_transform_stream_type],
                #     array_dims=["PE_NUM"],
                # )
                # output_stream_array_type = HLSType(
                #     HLSBasicType.ARRAY, sub_types=[single_data_stream_type], array_dims=["PE_NUM"]
                # )

                # c. Set the new parameters for unit_reduce_func
                unit_reduce_func.params = [
                    HLSVar("kt_wrap_item", kt_wrap_array),
                    # HLSVar("intermediate_key", key_stream_array_type),
                    # HLSVar("intermediate_transform", transform_stream_array_type),
                    HLSVar("o_0", single_data_stream_type),
                ]

                self.hls_functions[comp.readable_id] = pre_process_func
                self.hls_functions[comp.readable_id + 1] = unit_reduce_func  # Use a unique-ish ID

                # 2. 获取批处理类型
                key_batch_type = self.batch_type_map[key_type]
                transform_batch_type = self.batch_type_map[transform_type]
                kt_pair_batch_type = self._get_batch_type(kt_pair_type)

                # 3. 生成 Zipper
                zipper_func = generate_stream_zipper(
                    key_batch_type, transform_batch_type, kt_pair_batch_type
                )

                # 4. 生成 Demux (现在它处理 kt_pair_batch)
                demux_func = generate_demux(self.PE_NUM, kt_pair_batch_type, kt_wrap_type)

                # 5. 生成 Omega Network (现在它处理 kt_pair 并根据 'key' 路由)
                omega_funcs = generate_omega_network(
                    self.PE_NUM, kt_wrap_type, routing_key_member="key"
                )
                omega_func = next(f for f in omega_funcs if "omega_switch" in f.name)

                # 6. 生成 Unzipper
                # key_stream_array_type = unit_reduce_func.params[0].type
                # transform_stream_array_type = unit_reduce_func.params[1].type
                # unzipper_func = generate_stream_unzipper(self.PE_NUM, kt_wrap_type, key_stream_array_type, transform_stream_array_type)

                # 7. 生成 Reduction Tree (逻辑不变)
                # out_dfir_type = comp.get_port("o_0").data_type
                # base_data_type = self.type_map[out_dfir_type.type_]
                # merge_func = generate_merge_stream_2x1(base_data_type)
                # tree_func = generate_reduction_tree(self.PE_NUM, base_data_type, merge_func)

                # 8. 存储所有生成的辅助函数
                self.utility_functions.extend([zipper_func, demux_func] + omega_funcs)

                # 9. 声明所有中间流并存储辅助模块信息
                helpers = {
                    "zipper": zipper_func,
                    "demux": demux_func,
                    "omega": omega_func,
                    # "unzipper": unzipper_func,
                    # "tree": tree_func,
                    "unit_reduce": unit_reduce_func,
                }

                streams_to_declare = {
                    "zipper_to_demux": HLSVar(
                        f"reduce_{comp.readable_id}_z2d_pair",
                        HLSType(HLSBasicType.STREAM, sub_types=[kt_pair_batch_type]),
                    ),
                    "demux_to_omega": HLSVar(
                        f"reduce_{comp.readable_id}_d2o_pair", demux_func.params[1].type
                    ),
                    "omega_to_unit": HLSVar(
                        f"reduce_{comp.readable_id}_o2u_pair", omega_func.params[-1].type
                    ),
                    # "omega_to_unzipper": HLSVar(f"reduce_{comp.readable_id}_o2u_pair", unzipper_func.params[0].type),
                    # "unzipper_to_unit_key": HLSVar(f"reduce_{comp.readable_id}_u2u_key", key_stream_array_type),
                    # "unzipper_to_unit_transform": HLSVar(f"reduce_{comp.readable_id}_u2u_transform", transform_stream_array_type),
                    "unit_to_final": HLSVar(
                        f"reduce_{comp.readable_id}_uout_streams", unit_reduce_func.params[1].type
                    ),
                    # "unit_to_tree": HLSVar(f"reduce_{comp.readable_id}_u2t_streams", unit_reduce_func.params[2].type),
                    # "tree_to_final": HLSVar(f"reduce_{comp.readable_id}_t2f_stream", tree_func.params[2].type)
                }

                for stream_var in streams_to_declare.values():
                    decl = CodeVarDecl(stream_var.name, stream_var.type)
                    pragma = CodePragma(
                        f"STREAM variable={stream_var.name} depth={self.STREAM_DEPTH}"
                    )
                    self.top_level_stream_decls.append((decl, pragma))

                helpers["streams"] = streams_to_declare
                self.reduce_helpers[comp.readable_id] = helpers

                internal_streams: Dict[str, HLSVar] = {}
                for param_name in ["intermediate_key", "intermediate_transform"]:
                    # 从刚刚创建的函数签名中获取流的类型
                    stream_type = next(
                        p.type for p in pre_process_func.params if p.name == param_name
                    )
                    # 为这个流创建一个在顶层函数中唯一的变量名
                    stream_var = HLSVar(f"reduce_{comp.readable_id}_{param_name}", stream_type)
                    internal_streams[param_name] = stream_var

                    # 将声明和 pragma 添加到顶层函数体
                    decl = CodeVarDecl(stream_var.name, stream_var.type)
                    pragma = CodePragma(
                        f"STREAM variable={stream_var.name} depth={self.STREAM_DEPTH}"
                    )
                    self.top_level_stream_decls.append((decl, pragma))

                self.reduce_internal_streams[comp.readable_id] = internal_streams

                # 3. Mark all sub-graph components as processed
                for port_name in [
                    "o_reduce_key_in",
                    "o_reduce_transform_in",
                    "o_reduce_unit_start_0",
                    "o_reduce_unit_start_1",
                ]:
                    if comp.get_port(port_name).connected:
                        q = [comp.get_port(port_name).connection.parent]
                        visited_sub = set()
                        while q:
                            sub_comp = q.pop(0)
                            if sub_comp.readable_id in visited_sub:
                                continue
                            visited_sub.add(sub_comp.readable_id)
                            processed_sub_comp_ids.add(sub_comp.readable_id)
                            for p in sub_comp.out_ports:
                                if p.connected and not isinstance(
                                    p.connection.parent, dfir.ReduceComponent
                                ):
                                    q.append(p.connection.parent)

        # --- Original logic for non-reduce components ---
        for comp in comp_col.components:
            if comp.readable_id in processed_sub_comp_ids or isinstance(
                comp,
                (
                    dfir.IOComponent,
                    dfir.ConstantComponent,
                    dfir.UnusedEndMarkerComponent,
                    dfir.ReduceComponent,
                ),
            ):
                continue

            hls_func = HLSFunction(name=comp.name, comp=comp)
            # Normal components are always streamed
            for port in comp.ports:
                if port.connection and isinstance(
                    port.connection.parent, (dfir.UnusedEndMarkerComponent, dfir.ConstantComponent)
                ):
                    continue
                dfir_type = port.data_type
                if isinstance(dfir_type, dftype.ArrayType):
                    dfir_type = dfir_type.type_
                base_hls_type = self.type_map[dfir_type]
                batch_type = self.batch_type_map[base_hls_type]
                param_type = HLSType(HLSBasicType.STREAM, sub_types=[batch_type])
                hls_func.params.append(HLSVar(var_name=port.name, var_type=param_type))
            self.hls_functions[comp.readable_id] = hls_func

        # 2. Identify intermediate streams and add their declarations and pragmas
        visited_ports = set()
        for port in comp_col.all_connected_ports:
            if port.readable_id in visited_ports:
                continue
            conn = port.connection
            is_intermediate = (
                port.parent.readable_id in self.hls_functions
                and conn.parent.readable_id in self.hls_functions
                and self.hls_functions[port.parent.readable_id].streamed
                and self.hls_functions[conn.parent.readable_id].streamed
            )
            if is_intermediate:
                dfir_type = port.data_type
                if isinstance(dfir_type, dftype.ArrayType):
                    dfir_type = dfir_type.type_
                base_hls_type = self.type_map[dfir_type]
                batch_type = self.batch_type_map[base_hls_type]
                stream_type = HLSType(HLSBasicType.STREAM, sub_types=[batch_type])
                out_port = port if port.port_type == dfir.PortType.OUT else conn
                stream_name = f"stream_{out_port.unique_name}"

                decl = CodeVarDecl(stream_name, stream_type)
                pragma = CodePragma(f"STREAM variable={stream_name} depth={self.STREAM_DEPTH}")
                self.top_level_stream_decls.append((decl, pragma))

            visited_ports.add(port.readable_id)
            visited_ports.add(conn.readable_id)

    def _translate_functions(self):
        """Phase 3 Entry Point: Populates the .codes for all HLSFunctions."""
        # --- MODIFIED: Handle ReduceComponent first ---
        reduce_comps = [
            f.dfir_comp
            for f in self.hls_functions.values()
            if isinstance(f.dfir_comp, dfir.ReduceComponent)
        ]
        for comp in reduce_comps:
            pre_process_func = next(
                f for f in self.hls_functions.values() if f.name == f"{comp.name}_pre_process"
            )
            unit_reduce_func = next(
                f for f in self.hls_functions.values() if f.name == f"{comp.name}_unit_reduce"
            )

            pre_process_func.codes = self._translate_reduce_preprocess(pre_process_func)
            unit_reduce_func.codes = self._translate_reduce_unit_reduce(unit_reduce_func)

        # --- Translate other functions ---
        for func in self.hls_functions.values():
            if not isinstance(func.dfir_comp, dfir.ReduceComponent):
                if func.streamed:
                    self._translate_streamed_component(func)
                else:  # Should not happen with the new logic
                    assert False
                    self._translate_unstreamed_component(func)

    # ======================================================================== #
    #                            PHASE 1                                       #
    # ======================================================================== #

    def _get_batch_type(self, base_type: HLSType) -> HLSType:
        """
        Creates (or retrieves from cache) a batched version of a base HLSType.
        The batched type is a struct containing an array of the base type and control flags.
        """
        if base_type in self.batch_type_map:
            return self.batch_type_map[base_type]

        data_array_type = HLSType(HLSBasicType.ARRAY, sub_types=[base_type], array_dims=["PE_NUM"])
        end_flag_type = HLSType(HLSBasicType.BOOL)
        end_pos_type = HLSType(HLSBasicType.UINT8)

        member_types = [data_array_type, end_flag_type, end_pos_type]
        member_names = ["data", "end_flag", "end_pos"]

        batch_type = HLSType(HLSBasicType.STRUCT, member_types, struct_prop_names=member_names)

        self.batch_type_map[base_type] = batch_type
        if batch_type.name not in self.struct_definitions:
            self.struct_definitions[batch_type.name] = (batch_type, member_names)

        return batch_type

    def _to_hls_type(self, dfir_type: dftype.DfirType, is_array_type: bool = False) -> HLSType:
        """
        Recursively converts a DfirType to a base HLSType, using memoization.
        This handles basic types, tuples, optionals, and special graph types.
        """
        global_graph = self.global_graph_store
        if dfir_type in self.type_map:
            if is_array_type and dfir.ArrayType(dfir_type) not in self.type_map:
                self.type_map[dfir.ArrayType(dfir_type)] = self.type_map[dfir_type]
            return self.type_map[dfir_type]

        hls_type: HLSType

        # --- BASE CASES ---
        if isinstance(dfir_type, dftype.IntType):
            hls_type = HLSType(HLSBasicType.INT)
        elif isinstance(dfir_type, dftype.FloatType):
            hls_type = HLSType(HLSBasicType.FLOAT)
        elif isinstance(dfir_type, dftype.BoolType):
            hls_type = HLSType(HLSBasicType.BOOL)

        # --- RECURSIVE CASES ---
        elif isinstance(dfir_type, dftype.TupleType):
            sub_types = [self._to_hls_type(t, global_graph) for t in dfir_type.types]
            member_names = [f"ele_{i}" for i in range(len(sub_types))]
            hls_type = HLSType(HLSBasicType.STRUCT, sub_types, struct_prop_names=member_names)
            if hls_type.name not in self.struct_definitions:
                self.struct_definitions[hls_type.name] = (hls_type, member_names)

        elif isinstance(dfir_type, dftype.OptionalType):
            data_type = self._to_hls_type(dfir_type.type_, global_graph)
            valid_type = HLSType(HLSBasicType.BOOL)
            hls_type = HLSType(
                HLSBasicType.STRUCT,
                sub_types=[data_type, valid_type],
                struct_name=f"opt_{data_type.name}_t",
                struct_prop_names=["data", "valid"],
            )
            if hls_type.name not in self.struct_definitions:
                self.struct_definitions[hls_type.name] = (hls_type, ["data", "valid"])

        elif isinstance(dfir_type, dftype.SpecialType):
            # Assumes global_graph has node_properties and edge_properties dicts
            props = (
                global_graph.node_properties
                if dfir_type.type_name == "node"
                else global_graph.edge_properties
            )
            prop_names = list(props.keys())
            prop_types = [self._to_hls_type(t, global_graph) for t in props.values()]
            struct_name = f"{dfir_type.type_name}_t"
            hls_type = HLSType(HLSBasicType.STRUCT, prop_types, struct_name, prop_names)
            if hls_type.name not in self.struct_definitions:
                self.struct_definitions[hls_type.name] = (hls_type, prop_names)

        else:
            raise NotImplementedError(
                f"DFIR type conversion not implemented for {type(dfir_type)}"
            )

        # Cache the result before returning
        self.type_map[dfir_type] = hls_type
        if is_array_type:
            self.type_map[dfir.ArrayType(dfir_type)] = hls_type
        return hls_type

    # ======================================================================== #
    #                            PHASE 3                                       #
    # ======================================================================== #

    def _translate_streamed_component(self, hls_func: HLSFunction):
        """Translates a DFIR component into a standard streamed HLS function body."""
        comp = hls_func.dfir_comp

        # Dispatcher to select the correct translation logic
        if isinstance(comp, dfir.BinOpComponent):
            inner_logic = self._translate_binop_op(comp, "i")
        elif isinstance(comp, dfir.UnaryOpComponent):
            inner_logic = self._translate_unary_op(comp, "i")
        elif isinstance(comp, dfir.CopyComponent):
            inner_logic = self._translate_copy_op(comp, "i")
        elif isinstance(comp, dfir.GatherComponent):
            inner_logic = self._translate_gather_op(comp, "i")
        elif isinstance(comp, dfir.ScatterComponent):
            inner_logic = self._translate_scatter_op(comp, "i")
        elif isinstance(comp, dfir.ConditionalComponent):
            inner_logic = self._translate_conditional_op(comp, "i")
        elif isinstance(comp, dfir.CollectComponent):
            # Collect has a different boilerplate, handle it separately
            hls_func.codes = self._translate_collect_op(hls_func)
            return
        else:
            inner_logic = [
                CodePragma(
                    f"WARNING: Component {type(comp).__name__} translation not implemented."
                )
            ]

        # Wrap the core logic in the standard streaming boilerplate
        hls_func.codes = self._generate_streamed_function_boilerplate(hls_func, inner_logic)

    def _translate_unstreamed_component(self, hls_func: HLSFunction):
        """Translates a DFIR component for an unstreamed (pass-by-reference) function."""
        # This is a placeholder for now, as it's mainly for reduce sub-functions (key, transform, unit)
        hls_func.codes = [
            CodePragma("INLINE"),
            CodePragma(
                f"WARNING: Unstreamed func translation not fully implemented for {hls_func.name}"
            ),
        ]

    def _generate_streamed_function_boilerplate(
        self, hls_func: HLSFunction, inner_loop_logic: List[HLSCodeLine]
    ) -> List[HLSCodeLine]:
        """Creates the standard while/for loop structure for a streamed function."""
        body: List[HLSCodeLine] = []
        in_ports = hls_func.dfir_comp.in_ports.copy()
        out_ports = hls_func.dfir_comp.out_ports.copy()

        # 1. Declare local batch variables for inputs and outputs
        in_batch_vars: Dict[str, HLSVar] = {
            p.name: HLSVar(f"in_batch_{p.name}", p.type.sub_types[0])
            for p in hls_func.params
            if p.name in [ip.name for ip in in_ports]
        }
        out_batch_vars: Dict[str, HLSVar] = {
            p.name: HLSVar(f"out_batch_{p.name}", p.type.sub_types[0])
            for p in hls_func.params
            if p.name in [op.name for op in out_ports]
        }
        for var in list(in_batch_vars.values()) + list(out_batch_vars.values()):
            body.append(CodeVarDecl(var.name, var.type))
        end_flag_var_decl = CodeVarDecl("end_flag", HLSType(HLSBasicType.BOOL))
        end_flag_var = end_flag_var_decl.var
        body.append(end_flag_var_decl)
        end_pos_var_decl = CodeVarDecl("end_pos", HLSType(HLSBasicType.UINT8))
        end_pos_var = end_pos_var_decl.var
        body.append(end_pos_var_decl)

        # 2. Create the main while(true) loop
        while_loop_body: List[HLSCodeLine] = [CodePragma("PIPELINE")]

        # 3. Read from all input streams
        for p in hls_func.params:
            if p.name in in_batch_vars:
                read_expr = HLSExpr(HLSExprT.STREAM_READ, None, [HLSExpr(HLSExprT.VAR, p)])
                while_loop_body.append(CodeAssign(in_batch_vars[p.name], read_expr))

        # 4. Create the inner for loop
        for_loop = CodeFor(
            codes=[CodePragma("UNROLL")] + inner_loop_logic,
            iter_limit="PE_NUM",
            iter_name="i",
        )
        while_loop_body.append(for_loop)

        # 5. Get end flag value.
        assert in_batch_vars
        # Combine end flags from all inputs
        # For simplicity, we use the first input's end_flag. A real implementation might OR them.
        first_in_batch = list(in_batch_vars.values())[0]
        end_check_expr = HLSExpr(
            HLSExprT.VAR,
            HLSVar(f"{first_in_batch.name}.end_flag", end_flag_var.type),
        )
        assign_end_flag = CodeAssign(end_flag_var, end_check_expr)
        end_check_pos_expr = HLSExpr(
            HLSExprT.VAR,
            HLSVar(f"{first_in_batch.name}.end_pos", end_pos_var.type),
        )
        assign_end_pos = CodeAssign(end_pos_var, end_check_pos_expr)
        while_loop_body.extend([assign_end_flag, assign_end_pos])

        # 6. Write to all output streams
        for p in hls_func.params:
            if p.name in out_batch_vars:
                # assign end_flag & end pos
                while_loop_body.append(
                    CodeAssign(
                        HLSVar(f"{out_batch_vars[p.name].name}.end_flag", end_flag_var.type),
                        HLSExpr(HLSExprT.VAR, end_flag_var),
                    )
                )
                while_loop_body.append(
                    CodeAssign(
                        HLSVar(f"{out_batch_vars[p.name].name}.end_pos", end_pos_var.type),
                        HLSExpr(HLSExprT.VAR, end_pos_var),
                    )
                )
                while_loop_body.append(CodeWriteStream(p, out_batch_vars[p.name]))

        # 7. Check for end condition and break
        if in_batch_vars:
            break_if = CodeIf(HLSExpr(HLSExprT.VAR, end_flag_var), [CodeBreak()])
            while_loop_body.extend([break_if])

        body.append(CodeWhile(codes=while_loop_body, iter_expr=HLSExpr(HLSExprT.CONST, True)))
        return body

    # --- Component-Specific Translators for Inner Loop Logic ---

    def _translate_binop_op(self, comp: dfir.BinOpComponent, iterator: str) -> List[HLSCodeLine]:
        """Generates the core logic for a BinOpComponent."""
        # Assume i_0, i_1 are inputs and o_0 is output
        in0_type = self.batch_type_map[self.type_map[comp.input_type]].sub_types[0].sub_types[0]
        in1_type = self.batch_type_map[self.type_map[comp.input_type]].sub_types[0].sub_types[0]
        out_type = self.batch_type_map[self.type_map[comp.output_type]].sub_types[0].sub_types[0]

        # Operands from input batches, indexed by the iterator
        op1 = HLSExpr(HLSExprT.VAR, HLSVar(f"in_batch_i_0.data[{iterator}]", in0_type))
        op1 = HLSExpr.check_const(op1, comp.in_ports[0])
        op2 = HLSExpr(HLSExprT.VAR, HLSVar(f"in_batch_i_1.data[{iterator}]", in1_type))
        op2 = HLSExpr.check_const(op2, comp.in_ports[1])

        # The binary operation expression
        bin_expr = HLSExpr(HLSExprT.BINOP, comp.op, [op1, op2])

        # The variable to store the result in the output batch
        target_var = HLSVar(f"out_batch_o_0.data[{iterator}]", out_type)

        return [CodeAssign(target_var, bin_expr)]

    def _translate_unary_op(self, comp: dfir.UnaryOpComponent, iterator: str) -> List[HLSCodeLine]:
        """Generates the core logic for a UnaryOpComponent."""
        in_type = self.batch_type_map[self.type_map[comp.input_type]].sub_types[0].sub_types[0]
        out_type = self.batch_type_map[self.type_map[comp.output_type]].sub_types[0].sub_types[0]

        operand = HLSExpr(HLSExprT.VAR, HLSVar(f"in_batch_i_0.data[{iterator}]", in_type))
        operand = HLSExpr.check_const(operand, comp.in_ports[0])
        comp_op_var = comp.op
        if comp.op == dfir.UnaryOp.GET_ATTR:
            assert operand.val.type.type == HLSBasicType.STRUCT
            comp_op_var = (comp_op_var, comp.select_index)
        unary_expr = HLSExpr(HLSExprT.UOP, comp_op_var, [operand])
        target_var = HLSVar(f"out_batch_o_0.data[{iterator}]", out_type)

        return [CodeAssign(target_var, unary_expr)]

    def _translate_copy_op(self, comp: dfir.CopyComponent, iterator: str) -> List[HLSCodeLine]:
        """Generates the core logic for a CopyComponent."""
        in_type = self.type_map[comp.get_port("i_0").data_type]

        in_var_expr = HLSExpr(HLSExprT.VAR, HLSVar(f"in_batch_i_0.data[{iterator}]", in_type))
        in_var_expr = HLSExpr.check_const(in_var_expr, comp.in_ports[0])

        target_o0 = HLSVar(f"out_batch_o_0.data[{iterator}]", in_type)
        target_o1 = HLSVar(f"out_batch_o_1.data[{iterator}]", in_type)

        return [CodeAssign(target_o0, in_var_expr), CodeAssign(target_o1, in_var_expr)]

    def _translate_gather_op(self, comp: dfir.GatherComponent, iterator: str) -> List[HLSCodeLine]:
        """Generates the core logic for a GatherComponent."""
        out_port = comp.get_port("o_0")
        out_type = self.type_map[out_port.data_type]

        assignments = []
        for i, in_port in enumerate(comp.in_ports):
            in_type = self.type_map[in_port.data_type]
            in_var_expr = HLSExpr(
                HLSExprT.VAR,
                HLSVar(f"in_batch_{in_port.name}.data[{iterator}]", in_type),
            )
            in_var_expr = HLSExpr.check_const(in_var_expr, comp.in_ports[i])

            # Target is a member of the output struct
            target_member = HLSVar(f"out_batch_o_0.data[{iterator}].ele_{i}", in_type)
            assignments.append(CodeAssign(target_member, in_var_expr))

        return assignments

    def _translate_scatter_op(
        self, comp: dfir.ScatterComponent, iterator: str
    ) -> List[HLSCodeLine]:
        """Generates the core logic for a ScatterComponent."""
        in_port = comp.get_port("i_0")
        in_type = self.type_map[in_port.data_type]

        assignments = []
        for i, out_port in enumerate(comp.out_ports):
            if isinstance(out_port.connection.parent, dfir.UnusedEndMarkerComponent):
                continue
            out_type = self.type_map[out_port.data_type]

            ga_op = dfir.UnaryOp.GET_ATTR
            sub_name = in_type.get_nth_subname(i)
            # Source is a member of the input struct
            in_member_expr = HLSExpr(
                HLSExprT.UOP,
                (ga_op, sub_name),
                [
                    HLSExpr(
                        HLSExprT.VAR,
                        HLSVar(f"in_batch_i_0.data[{iterator}]", in_type),
                    )
                ],
            )

            target_var = HLSVar(f"out_batch_{out_port.name}.data[{iterator}]", out_type)
            assignments.append(CodeAssign(target_var, in_member_expr))

        return assignments

    def _translate_conditional_op(
        self, comp: dfir.ConditionalComponent, iterator: str
    ) -> List[HLSCodeLine]:
        """Generates the core logic for a ConditionalComponent."""
        data_port = comp.get_port("i_data")
        cond_port = comp.get_port("i_cond")
        out_port = comp.get_port("o_0")

        data_type = self.type_map[data_port.data_type]
        cond_type = self.type_map[cond_port.data_type]
        out_type = self.type_map[out_port.data_type]  # This is an Optional/Struct type

        # Source expressions
        data_expr = HLSExpr(HLSExprT.VAR, HLSVar(f"in_batch_i_data.data[{iterator}]", data_type))
        data_expr = HLSExpr.check_const(data_expr, data_port)
        cond_expr = HLSExpr(HLSExprT.VAR, HLSVar(f"in_batch_i_cond.data[{iterator}]", cond_type))
        cond_expr = HLSExpr.check_const(cond_expr, cond_port)

        # Target members of the output Optional struct
        target_data_member = HLSVar(f"out_batch_o_0.data[{iterator}].data", data_type)
        target_valid_member = HLSVar(f"out_batch_o_0.data[{iterator}].valid", cond_type)

        return [
            CodeAssign(target_data_member, data_expr),
            CodeAssign(target_valid_member, cond_expr),
        ]

    def _translate_collect_op(self, hls_func: HLSFunction) -> List[HLSCodeLine]:
        """Generates a custom function body for CollectComponent due to its filtering nature."""
        body: List[HLSCodeLine] = []
        comp = hls_func.dfir_comp
        in_port = comp.get_port("i_0")
        out_port = comp.get_port("o_0")

        # 1. Declare local batch variables
        in_batch_var = HLSVar(
            "in_batch_i_0", self.batch_type_map[self.type_map[in_port.data_type]]
        )
        out_batch_var = HLSVar(
            "out_batch_o_0", self.batch_type_map[self.type_map[out_port.data_type]]
        )
        body.extend(
            [
                CodeVarDecl(in_batch_var.name, in_batch_var.type),
                CodeVarDecl(out_batch_var.name, out_batch_var.type),
            ]
        )

        # 2. Main while(true) loop
        while_loop_body: List[HLSCodeLine] = [CodePragma("PIPELINE")]

        # 3. Read input batch and declare output index
        in_stream_param = hls_func.params[0]  # Assume i_0 is the first param
        read_expr = HLSExpr(HLSExprT.STREAM_READ, None, [HLSExpr(HLSExprT.VAR, in_stream_param)])
        while_loop_body.append(CodeAssign(in_batch_var, read_expr))

        out_idx_type = HLSType(HLSBasicType.UINT8)
        out_idx_var_decl = CodeVarDecl("out_idx", out_idx_type)
        while_loop_body.append(out_idx_var_decl)
        out_idx_var = out_idx_var_decl.var
        while_loop_body.append(CodeAssign(out_idx_var, HLSExpr(HLSExprT.CONST, 0)))

        # 4. Inner for loop for filtering
        in_elem_type = self.type_map[in_port.data_type]  # This is an Optional type
        out_elem_type = self.type_map[out_port.data_type]

        ga_op = dfir.UnaryOp.GET_ATTR
        # Condition: in_batch_i_0.data[i].valid
        cond_expr = HLSExpr(
            HLSExprT.UOP,
            (ga_op, "valid"),
            [HLSExpr(HLSExprT.VAR, HLSVar(f"in_batch_i_0.data[i]", in_elem_type))],
        )

        ga_op = dfir.UnaryOp.GET_ATTR
        # Assignment if valid: out_batch_o_0.data[out_idx++] = in_batch_i_0.data[i].data
        assign_data = CodeAssign(
            HLSVar(f"out_batch_o_0.data[{out_idx_var.name}]", out_elem_type),
            HLSExpr(
                HLSExprT.UOP,
                (ga_op, "data"),
                [HLSExpr(HLSExprT.VAR, HLSVar(f"in_batch_i_0.data[i]", in_elem_type))],
            ),
        )
        increment_idx = CodeAssign(
            out_idx_var,
            HLSExpr(
                HLSExprT.BINOP,
                dfir.BinOp.ADD,
                [HLSExpr(HLSExprT.VAR, out_idx_var), HLSExpr(HLSExprT.CONST, 1)],
            ),
        )

        if_block = CodeIf(cond_expr, [assign_data, increment_idx])
        for_loop = CodeFor(
            codes=[CodePragma("UNROLL"), if_block], iter_limit="PE_NUM", iter_name="i"
        )
        while_loop_body.append(for_loop)

        # 5. Set output batch metadata and write to stream
        while_loop_body.append(
            CodeAssign(
                HLSVar(f"{out_batch_var.name}.end_pos", out_idx_type),
                HLSExpr(HLSExprT.VAR, out_idx_var),
            )
        )
        ga_op = dfir.UnaryOp.GET_ATTR
        end_flag_expr = HLSExpr(
            HLSExprT.UOP, (ga_op, "end_flag"), [HLSExpr(HLSExprT.VAR, in_batch_var)]
        )
        while_loop_body.append(
            CodeAssign(
                HLSVar(f"{out_batch_var.name}.end_flag", HLSType(HLSBasicType.BOOL)),
                end_flag_expr,
            )
        )

        out_stream_param = hls_func.params[1]  # Assume o_0 is the second param
        while_loop_body.append(CodeWriteStream(out_stream_param, out_batch_var))

        # 6. Break condition
        while_loop_body.append(CodeIf(end_flag_expr, [CodeBreak()]))

        body.append(CodeWhile(codes=while_loop_body, iter_expr=HLSExpr(HLSExprT.CONST, True)))
        return body

    # --- Reduce Component Translation Logic ---

    def _inline_sub_graph_logic(
        self,
        start_ports: List[dfir.Port],
        end_port: dfir.Port,
        io_var_map: Dict[dfir.Port, HLSVar],
    ) -> List[HLSCodeLine]:
        """
        Traverses a sub-graph from start to end ports and generates the inlined logic.
        (This is the fully expanded version with more component types)
        """
        code_lines: List[HLSCodeLine] = []
        p2var_map = io_var_map.copy()
        code_lines.append(CodeComment(" -- Inline sub graph --"))

        # 1. Topologically sort the sub-graph
        q = [p.connection.parent for p in start_ports if p.connected]
        visited_ids = set([c.readable_id for c in q])

        head = 0
        while head < len(q):
            comp = q[head]
            head += 1

            code_lines.append(CodeComment(f"Starting for comp {comp.name}"))

            inputs_ready = all(p.connection in p2var_map for p in comp.in_ports)
            if not inputs_ready:
                q.append(comp)
                if head > len(q) * 2 + len(start_ports) * 2:  # More robust deadlock check
                    raise RuntimeError(
                        f"Deadlock in sub-graph topological sort, stuck at component {comp.name}"
                    )
                continue

            # --- Inputs are ready, process the component ---
            # a. Create HLSVars for the output ports
            for out_port in comp.out_ports:
                if out_port.connected:
                    if out_port.connection == end_port:
                        p2var_map[out_port] = p2var_map[end_port]
                    else:
                        temp_var = HLSVar(
                            f"temp_{out_port.parent.name}_{out_port.name}",
                            self.type_map[out_port.data_type],
                        )
                        code_lines.append(CodeVarDecl(temp_var.name, temp_var.type))
                        p2var_map[out_port] = temp_var

            # b. Translate the component's logic into CodeLine(s)
            # This is a non-batched, direct translation of the component's logic
            if isinstance(comp, dfir.BinOpComponent):
                op1 = HLSExpr(HLSExprT.VAR, p2var_map[comp.get_port("i_0").connection])
                op1 = HLSExpr.check_const(op1, comp.get_port("i_0"))
                op2 = HLSExpr(HLSExprT.VAR, p2var_map[comp.get_port("i_1").connection])
                op2 = HLSExpr.check_const(op2, comp.get_port("i_1"))
                expr = HLSExpr(HLSExprT.BINOP, comp.op, [op1, op2])
                code_lines.append(CodeAssign(p2var_map[comp.get_port("o_0")], expr))

            elif isinstance(comp, dfir.UnaryOpComponent):
                op1 = HLSExpr(HLSExprT.VAR, p2var_map[comp.get_port("i_0").connection])
                op1 = HLSExpr.check_const(op1, comp.get_port("i_0"))
                comp_op_var = comp.op
                if comp.op in [dfir.UnaryOp.GET_ATTR, dfir.UnaryOp.SELECT]:
                    assert op1.val.type.type == HLSBasicType.STRUCT
                    comp_op_var = (comp_op_var, comp.select_index)
                expr = HLSExpr(HLSExprT.UOP, comp_op_var, [op1])
                code_lines.append(CodeAssign(p2var_map[comp.get_port("o_0")], expr))

            elif isinstance(comp, dfir.CopyComponent):
                in_var_expr = HLSExpr(HLSExprT.VAR, p2var_map[comp.get_port("i_0").connection])
                in_var_expr = HLSExpr.check_const(in_var_expr, comp.get_port("i_0"))
                target_o0 = p2var_map[comp.get_port("o_0")]
                target_o1 = p2var_map[comp.get_port("o_1")]
                code_lines.append(CodeAssign(target_o0, in_var_expr))
                code_lines.append(CodeAssign(target_o1, in_var_expr))

            elif isinstance(comp, dfir.GatherComponent):
                target_struct_var = p2var_map[comp.get_port("o_0")]
                for i, in_port in enumerate(comp.in_ports):
                    in_var_expr = HLSExpr(HLSExprT.VAR, p2var_map[in_port.connection])
                    in_var_expr = HLSExpr.check_const(in_var_expr, in_port)
                    # Assign to a member of the target struct
                    member_var = HLSVar(f"{target_struct_var.name}.ele_{i}", in_var_expr.val.type)
                    code_lines.append(CodeAssign(member_var, in_var_expr))

            elif isinstance(comp, dfir.ScatterComponent):
                in_var = p2var_map[comp.get_port("i_0").connection]
                for i, out_port in enumerate(comp.out_ports):
                    ga_op = dfir.UnaryOp.GET_ATTR
                    sub_name = in_var.type.get_nth_subname(i)
                    expr = HLSExpr(
                        HLSExprT.UOP, (ga_op, sub_name), [HLSExpr(HLSExprT.VAR, in_var)]
                    )
                    code_lines.append(CodeAssign(p2var_map[out_port], expr))

            elif isinstance(comp, dfir.ConditionalComponent):
                data_expr = HLSExpr(HLSExprT.VAR, p2var_map[comp.get_port("i_data").connection])
                data_expr = HLSExpr.check_const(data_expr, comp.get_port("i_data"))
                cond_expr = HLSExpr(HLSExprT.VAR, p2var_map[comp.get_port("i_cond").connection])
                cond_expr = HLSExpr.check_const(cond_expr, comp.get_port("i_cond"))
                target_struct_var = p2var_map[comp.get_port("o_0")]

                # Assign to .data member
                assign_data = CodeAssign(
                    HLSVar(f"{target_struct_var.name}.data", data_expr.val.type),
                    data_expr,
                )
                # Assign to .valid member
                assign_valid = CodeAssign(
                    HLSVar(f"{target_struct_var.name}.valid", cond_expr.val.type),
                    cond_expr,
                )
                code_lines.extend([assign_data, assign_valid])

            elif isinstance(comp, dfir.CollectComponent):
                in_opt_var = p2var_map[comp.get_port("i_0").connection]
                out_var = p2var_map[comp.get_port("o_0")]

                # Condition: in_opt_var.valid
                valid_op = dfir.UnaryOp.GET_ATTR
                cond_expr = HLSExpr(
                    HLSExprT.UOP, (valid_op, "valid"), [HLSExpr(HLSExprT.VAR, in_opt_var)]
                )

                # Assignment: out_var = in_opt_var.data
                data_op = dfir.UnaryOp.GET_ATTR
                assign_expr = HLSExpr(
                    HLSExprT.UOP, (data_op, "data"), [HLSExpr(HLSExprT.VAR, in_opt_var)]
                )

                if_block = CodeIf(cond_expr, [CodeAssign(out_var, assign_expr)])
                code_lines.append(if_block)

            elif isinstance(comp, dfir.UnusedEndMarkerComponent):
                pass  # Do nothing for unused markers

            else:
                code_lines.append(CodeComment(f"Inlined logic for {comp.__class__} ({comp.name})"))

            # c. Add successors to the queue
            for p in comp.out_ports:
                if p.connected and not isinstance(
                    p.connection.parent,
                    (dfir.ReduceComponent, dfir.UnusedEndMarkerComponent),
                ):
                    successor_comp = p.connection.parent
                    if successor_comp.readable_id not in visited_ids:
                        q.append(successor_comp)
                        visited_ids.add(successor_comp.readable_id)

        code_lines.append(CodeComment(" -- Inline sub graph end --"))
        # print("One inline:")
        # print("".join(x.gen_code(0) for x in code_lines))
        return code_lines

    def _translate_reduce_preprocess_op(
        self, comp: dfir.ReduceComponent, iterator: str
    ) -> List[HLSCodeLine]:
        """
        Generates the inner-loop logic for ReduceComponent's pre_process stage.
        """
        in_type = self.type_map[comp.get_port("i_0").data_type]
        key_out_type = self.type_map[comp.get_port("i_reduce_key_out").data_type]
        transform_out_type = self.type_map[comp.get_port("i_reduce_transform_out").data_type]

        in_elem_var = HLSVar(f"in_batch_i_0.data[{iterator}]", in_type)
        key_out_elem_var = HLSVar("key_out_elem", key_out_type)
        transform_out_elem_var = HLSVar("transform_out_elem", transform_out_type)
        code_lines = [
            CodeVarDecl(key_out_elem_var.name, key_out_elem_var.type),
            CodeVarDecl(transform_out_elem_var.name, transform_out_elem_var.type),
        ]

        key_sub_graph_start = comp.get_port("o_reduce_key_in")
        key_sub_graph_end = comp.get_port("i_reduce_key_out")
        key_io_map = {key_sub_graph_start: in_elem_var, key_sub_graph_end: key_out_elem_var}
        code_lines.extend(
            self._inline_sub_graph_logic([key_sub_graph_start], key_sub_graph_end, key_io_map)
        )

        transform_sub_graph_start = comp.get_port("o_reduce_transform_in")
        transform_sub_graph_end = comp.get_port("i_reduce_transform_out")
        transform_io_map = {
            transform_sub_graph_start: in_elem_var,
            transform_sub_graph_end: transform_out_elem_var,
        }
        code_lines.extend(
            self._inline_sub_graph_logic(
                [transform_sub_graph_start], transform_sub_graph_end, transform_io_map
            )
        )

        assign_key = CodeAssign(
            HLSVar(f"out_batch_intermediate_key.data[{iterator}]", key_out_type),
            HLSExpr(HLSExprT.VAR, key_out_elem_var),
        )
        assign_transform = CodeAssign(
            HLSVar(f"out_batch_intermediate_transform.data[{iterator}]", transform_out_type),
            HLSExpr(HLSExprT.VAR, transform_out_elem_var),
        )
        code_lines.extend([assign_key, assign_transform])

        return code_lines

    def _translate_reduce_preprocess(self, hls_func: HLSFunction) -> List[HLSCodeLine]:
        """
        Generates a custom body for the pre_process stage, correctly handling its unique I/O.
        """
        body: List[HLSCodeLine] = []
        comp = hls_func.dfir_comp

        # 1. Get HLSVar for each parameter from the function signature
        in_stream, key_stream, transform_stream = hls_func.params

        # 2. Declare local batch variables for I/O
        in_batch_var = HLSVar("in_batch_i_0", in_stream.type.sub_types[0])
        key_out_batch_var = HLSVar("out_batch_intermediate_key", key_stream.type.sub_types[0])
        transform_out_batch_var = HLSVar(
            "out_batch_intermediate_transform", transform_stream.type.sub_types[0]
        )

        body.extend(
            [
                CodeVarDecl(in_batch_var.name, in_batch_var.type),
                CodeVarDecl(key_out_batch_var.name, key_out_batch_var.type),
                CodeVarDecl(transform_out_batch_var.name, transform_out_batch_var.type),
            ]
        )

        end_flag_var = HLSVar("end_flag", HLSType(HLSBasicType.BOOL))
        body.append(CodeVarDecl(end_flag_var.name, end_flag_var.type))

        # 3. Build the main while(true) loop
        while_loop_body: List[HLSCodeLine] = [CodePragma("PIPELINE")]

        # 4. Read input batch
        while_loop_body.append(
            CodeAssign(
                in_batch_var,
                HLSExpr(HLSExprT.STREAM_READ, None, [HLSExpr(HLSExprT.VAR, in_stream)]),
            )
        )

        # 5. Build the inner for-loop with the core logic
        inner_logic = self._translate_reduce_preprocess_op(comp, "i")
        for_loop = CodeFor(
            codes=[CodePragma("UNROLL")] + inner_logic, iter_limit="PE_NUM", iter_name="i"
        )
        while_loop_body.append(for_loop)

        # 6. Copy metadata (end_flag, end_pos) from input batch to both output batches
        ga_op = dfir.UnaryOp.GET_ATTR
        end_flag_expr = HLSExpr(
            HLSExprT.UOP, (ga_op, "end_flag"), [HLSExpr(HLSExprT.VAR, in_batch_var)]
        )
        end_pos_expr = HLSExpr(
            HLSExprT.UOP, (ga_op, "end_pos"), [HLSExpr(HLSExprT.VAR, in_batch_var)]
        )

        # Assign metadata to key_out_batch
        while_loop_body.append(
            CodeAssign(
                HLSVar(f"{key_out_batch_var.name}.end_flag", HLSType(HLSBasicType.BOOL)),
                end_flag_expr,
            )
        )
        while_loop_body.append(
            CodeAssign(
                HLSVar(f"{key_out_batch_var.name}.end_pos", HLSType(HLSBasicType.UINT8)),
                end_pos_expr,
            )
        )

        # Assign metadata to transform_out_batch
        while_loop_body.append(
            CodeAssign(
                HLSVar(f"{transform_out_batch_var.name}.end_flag", HLSType(HLSBasicType.BOOL)),
                end_flag_expr,
            )
        )
        while_loop_body.append(
            CodeAssign(
                HLSVar(f"{transform_out_batch_var.name}.end_pos", HLSType(HLSBasicType.UINT8)),
                end_pos_expr,
            )
        )

        # 7. Write both output batches to their streams
        while_loop_body.append(CodeWriteStream(key_stream, key_out_batch_var))
        while_loop_body.append(CodeWriteStream(transform_stream, transform_out_batch_var))

        # 8. Check for break condition
        while_loop_body.append(CodeAssign(end_flag_var, end_flag_expr))
        while_loop_body.append(CodeIf(HLSExpr(HLSExprT.VAR, end_flag_var), [CodeBreak()]))

        # 9. Finalize the function body
        body.append(CodeWhile(codes=while_loop_body, iter_expr=HLSExpr(HLSExprT.CONST, True)))
        return body

    # backend_manager.py

    # 在 backend_manager.py 中

    def _translate_reduce_unit_reduce(self, hls_func: HLSFunction) -> List[HLSCodeLine]:
        """
        Generates the body for the second stage of Reduce (stateful accumulation).
        """
        body: List[HLSCodeLine] = []
        comp = hls_func.dfir_comp

        # 1. Get types and variables from the new function signature
        kt_streams, out_streams = hls_func.params
        kt_type = kt_streams.type.sub_types[0].sub_types[0]
        key_type = kt_streams.type.sub_types[0].sub_types[0].sub_types[0].sub_types[0]
        transform_type = kt_streams.type.sub_types[0].sub_types[0].sub_types[0].sub_types[1]
        single_out_stream_type = out_streams.type.sub_types[0]
        out_data_type = single_out_stream_type.sub_types[0]

        bram_elem_type = self._to_hls_type(
            dftype.TupleType(
                [comp.get_port("i_reduce_transform_out").data_type, dftype.BoolType()]
            )
        )

        body.append(CodeComment("Stateful memories for PE_NUM parallel reduction units"))

        # 2. Declare 2D arrays for PE-local memories
        key_mem_type = HLSType(
            HLSBasicType.ARRAY, sub_types=[bram_elem_type], array_dims=["PE_NUM", "MAX_NUM"]
        )
        body.append(CodeVarDecl("key_mem", key_mem_type))
        body.append(CodePragma("BIND_STORAGE variable=key_mem type=RAM_2P impl=URAM"))
        body.append(CodePragma("ARRAY_PARTITION variable=key_mem complete dim=1"))

        key_buffer_type = HLSType(
            HLSBasicType.ARRAY, sub_types=[bram_elem_type], array_dims=["PE_NUM", "L + 1"]
        )
        body.append(CodeVarDecl("key_buffer", key_buffer_type))
        body.append(CodePragma("ARRAY_PARTITION variable=key_buffer complete dim=0"))

        i_buffer_base_type = HLSType(HLSBasicType.UINT)
        i_buffer_type = HLSType(
            HLSBasicType.ARRAY, sub_types=[i_buffer_base_type], array_dims=["PE_NUM", "L + 1"]
        )
        body.append(CodeVarDecl("i_buffer", i_buffer_type))
        body.append(CodePragma("ARRAY_PARTITION variable=i_buffer complete dim=0"))

        # 3. Initialize 2D memories using nested loops
        body.append(CodeComment("Memory initialization for all PEs"))
        uint_type = HLSType(HLSBasicType.UINT)
        max_num_var = HLSVar("MAX_NUM", uint_type)
        assign_val_expr = HLSExpr(
            HLSExprT.BINOP,
            dfir.BinOp.ADD,
            [HLSExpr(HLSExprT.VAR, max_num_var), HLSExpr(HLSExprT.CONST, 1)],
        )
        assign_ibuf = CodeAssign(HLSVar("i_buffer[pe][i]", uint_type), assign_val_expr)
        clear_ibuf_inner_loop = CodeFor(
            [CodePragma("UNROLL"), assign_ibuf], "L + 1", iter_name="i"
        )
        clear_ibuf_outer_loop = CodeFor(
            [CodePragma("UNROLL"), clear_ibuf_inner_loop], "PE_NUM", iter_name="pe"
        )
        body.append(clear_ibuf_outer_loop)

        target_valid_flag = HLSVar("key_mem[pe][i].ele_1", HLSType(HLSBasicType.BOOL))
        assign_valid_false = CodeAssign(target_valid_flag, HLSExpr(HLSExprT.CONST, False))
        clear_valid_inner_loop = CodeFor(
            [CodePragma("UNROLL"), assign_valid_false], "MAX_NUM", iter_name="i"
        )
        clear_valid_outer_loop = CodeFor(
            [CodePragma("UNROLL"), clear_valid_inner_loop], "PE_NUM", iter_name="pe"
        )
        body.append(clear_valid_outer_loop)

        # 4. Main Processing Loop - now handles parallel un-batched streams
        body.append(CodeComment("Main processing loop for aggregation across PEs"))
        end_flag_var = HLSVar("end_flag", HLSType(HLSBasicType.BOOL))
        body.append(CodeVarDecl(end_flag_var.name, end_flag_var.type))
        all_end_flag_var = HLSVar(
            "all_end_flags",
            HLSType(HLSBasicType.ARRAY, [end_flag_var.type], array_dims=["PE_NUM"]),
        )
        body.append(CodeVarDecl(all_end_flag_var.name, all_end_flag_var.type))
        body.append(CodePragma("ARRAY_PARTITION variable=all_end_flags complete dim=0"))
        assign_end_flag = CodeAssign(
            HLSVar(f"{all_end_flag_var.name}[i]", end_flag_var.type),
            HLSExpr(HLSExprT.CONST, False),
        )
        reset_end_loop = CodeFor(
            codes=[CodePragma("UNROLL"), assign_end_flag],
            iter_limit="PE_NUM",
            iter_name="i",
        )
        body.append(reset_end_loop)

        # This loop now assumes lock-step processing of the PE_NUM streams
        while_loop_body = [CodePragma("PIPELINE")]

        # Read one element from each PE's stream
        kt_elem_var = HLSVar("kt_elem", kt_type)
        key_elem_var = HLSVar("key_elem", key_type)
        transform_elem_var = HLSVar("transform_elem", transform_type)
        while_loop_body.extend(
            [
                CodeVarDecl(kt_elem_var.name, kt_elem_var.type),
                CodeVarDecl(key_elem_var.name, key_elem_var.type),
                CodeVarDecl(transform_elem_var.name, transform_elem_var.type),
            ]
        )

        # This logic is now inside a PE loop
        inner_loop_logic = self._translate_reduce_unit_inner_loop(
            comp, bram_elem_type, "i", key_elem_var, transform_elem_var
        )

        # Read from stream array and then process
        kt_stream_var_expr = HLSExpr(HLSExprT.VAR, HLSVar(f"{kt_streams.name}[i]", key_type))
        read_kt = CodeAssign(
            kt_elem_var,
            HLSExpr(
                HLSExprT.STREAM_READ,
                None,
                [kt_stream_var_expr],
            ),
        )
        read_key = CodeAssign(
            key_elem_var,
            HLSExpr(
                HLSExprT.UOP,
                (dfir.UnaryOp.GET_ATTR, "data.key"),
                [HLSExpr(HLSExprT.VAR, kt_elem_var)],
            ),
        )
        read_transform = CodeAssign(
            transform_elem_var,
            HLSExpr(
                HLSExprT.UOP,
                (dfir.UnaryOp.GET_ATTR, "data.transform"),
                [HLSExpr(HLSExprT.VAR, kt_elem_var)],
            ),
        )

        end_flag_expr = HLSExpr(
            HLSExprT.UOP,
            (dfir.UnaryOp.GET_ATTR, "end_flag"),
            [HLSExpr(HLSExprT.VAR, kt_elem_var)],
        )
        sub_end_flag = HLSExpr(
            HLSExprT.VAR, HLSVar(f"{all_end_flag_var.name}[i]", end_flag_var.type)
        )
        not_sub_end_expr = HLSExpr(HLSExprT.UOP, dfir.UnaryOp.NOT, [sub_end_flag])
        strm_empty = HLSExpr(HLSExprT.STREAM_EMPTY, None, [kt_stream_var_expr])
        strm_not_empty = HLSExpr(HLSExprT.UOP, dfir.UnaryOp.NOT, [strm_empty])
        both_conds = HLSExpr(HLSExprT.BINOP, dfir.BinOp.AND, [not_sub_end_expr, strm_not_empty])
        assign_flag2flag = CodeAssign(
            HLSVar(f"{all_end_flag_var.name}[i]", end_flag_var.type),
            end_flag_expr,
        )

        if_assign_block = CodeIf(
            expr=end_flag_expr,
            if_codes=[assign_flag2flag],
            else_codes=[read_key, read_transform] + inner_loop_logic,
        )

        # only if no end we process compute.
        if_not_end_block = CodeIf(
            expr=both_conds,
            if_codes=[read_kt, if_assign_block],
        )

        # We process one element per PE in an unrolled loop
        pe_processing_loop = CodeFor(
            codes=[CodePragma("UNROLL"), if_not_end_block],
            iter_limit="PE_NUM",
            iter_name="i",
        )
        while_loop_body.append(pe_processing_loop)

        # Check for end condition (can just check the first PE's stream)
        while_loop_body.append(CodeAssign(end_flag_var, HLSExpr(HLSExprT.CONST, True)))
        and_logic_for_flag = HLSExpr(
            HLSExprT.BINOP,
            dfir.BinOp.AND,
            [
                HLSExpr(HLSExprT.VAR, end_flag_var),
                HLSExpr(HLSExprT.VAR, HLSVar(f"{all_end_flag_var.name}[i]", end_flag_var.type)),
            ],
        )
        assign_end_flag_end = CodeAssign(
            end_flag_var,
            and_logic_for_flag,
        )
        set_end_loop = CodeFor(
            codes=[CodePragma("UNROLL"), assign_end_flag_end],
            iter_limit="PE_NUM",
            iter_name="i",
        )
        while_loop_body.append(set_end_loop)
        while_loop_body.append(CodeIf(HLSExpr(HLSExprT.VAR, end_flag_var), [CodeBreak()]))

        body.append(CodeWhile(codes=while_loop_body, iter_expr=HLSExpr(HLSExprT.CONST, True)))

        body.append(CodeComment("Final output loop to drain all PE memories with swapped loops"))
        cnt_var = HLSVar("data_cnt", HLSType(HLSBasicType.INT))
        body.append(CodeVarDecl(cnt_var.name, cnt_var.type))
        reset_data_cnt = CodeAssign(cnt_var, HLSExpr(HLSExprT.CONST, 0))
        body.append(reset_data_cnt)

        starting_flag = HLSVar("starting", HLSType(HLSBasicType.BOOL))
        body.append(CodeVarDecl(starting_flag.name, starting_flag.type))
        body.append(CodeAssign(starting_flag, HLSExpr(HLSExprT.CONST, False)))

        prev_data_var = HLSVar("prev_data", single_out_stream_type)
        body.append(CodeVarDecl(prev_data_var.name, prev_data_var.type))

        body.append(CodeVarDecl("data_to_write", single_out_stream_type))

        # 内层循环：遍历 PE，完全展开
        is_valid_expr = HLSExpr(HLSExprT.VAR, HLSVar(f"key_mem[pe][k].ele_1", bram_elem_type))
        data_expr = HLSExpr(HLSExprT.VAR, HLSVar(f"key_mem[pe][k].ele_0", bram_elem_type))

        assign_data = CodeAssign(
            HLSVar("data_to_write.data[data_cnt++]", out_data_type), data_expr
        )
        data_full_expr = HLSExpr(
            HLSExprT.BINOP,
            dfir.BinOp.EQ,
            [HLSExpr(HLSExprT.VAR, cnt_var), HLSExpr(HLSExprT.CONST, self.PE_NUM)],
        )
        set_data_false_flag = CodeAssign(
            HLSVar("data_to_write.end_flag", out_data_type),
            HLSExpr(HLSExprT.CONST, False),
        )
        set_data_pos = CodeAssign(
            HLSVar("data_to_write.end_pos", out_data_type),
            HLSExpr(HLSExprT.CONST, self.PE_NUM),
        )
        write_to_stream = CodeWriteStream(
            HLSVar(f"{out_streams.name}", single_out_stream_type),
            prev_data_var,
        )
        assign_prev_data = CodeAssign(
            prev_data_var,
            HLSExpr(HLSExprT.VAR, HLSVar("data_to_write", out_data_type)),
        )
        set_prev_data_flag1 = CodeAssign(starting_flag, HLSExpr(HLSExprT.CONST, True))
        if_set_cond = HLSExpr(
            HLSExprT.BINOP,
            dfir.BinOp.EQ,
            [HLSExpr(HLSExprT.VAR, starting_flag), HLSExpr(HLSExprT.CONST, False)],
        )
        if_set_data = CodeIf(
            if_set_cond,
            if_codes=[set_prev_data_flag1],
            else_codes=[write_to_stream],
        )
        if_data_full = CodeIf(
            data_full_expr,
            if_codes=[
                reset_data_cnt,
                set_data_false_flag,
                set_data_pos,
                if_set_data,
                assign_prev_data,
            ],
        )
        if_valid_block = CodeIf(
            is_valid_expr,
            [assign_data, if_data_full],
        )
        drain_inner_pe_loop = CodeFor(
            codes=[CodePragma("UNROLL"), if_valid_block], iter_limit="PE_NUM", iter_name="pe"
        )

        # 外层循环：遍历内存地址，进行流水线处理
        drain_outer_k_loop = CodeFor(
            codes=[
                CodePragma("PIPELINE"),
                drain_inner_pe_loop,
            ],
            iter_limit="MAX_NUM",
            iter_name="k",
        )
        body.append(drain_outer_k_loop)

        # 在所有数据都回写完毕后，发送结束标志
        body.append(CodeComment("Send end_flag to all PE output streams"))
        if_set_cond = HLSExpr(HLSExprT.VAR, starting_flag)
        if_set_data = CodeIf(
            if_set_cond,
            if_codes=[write_to_stream],
        )
        body.append(if_set_data)

        set_prev_data_flag_true = CodeAssign(
            HLSVar(f"{prev_data_var.name}.end_flag", HLSType(HLSBasicType.BOOL)),
            HLSExpr(HLSExprT.CONST, True),
        )
        set_data_pos = CodeAssign(
            HLSVar("prev_data.end_pos", out_data_type),
            HLSExpr(HLSExprT.VAR, cnt_var),
        )
        body.extend([assign_prev_data, set_prev_data_flag_true, set_data_pos, write_to_stream])

        return body

    def _translate_reduce_unit_inner_loop(
        self,
        comp: dfir.ReduceComponent,
        bram_elem_type: HLSType,
        pe_idx: str,
        key_var,
        val_var,
    ) -> List[HLSCodeLine]:
        """
        Helper to generate the complex logic inside unit_reduce's PE_NUM loop.
        *** MODIFIED to accept a PE index ***
        """
        key_type = self.type_map[comp.get_port("i_reduce_key_out").data_type]
        accum_type = self.type_map[comp.get_port("i_reduce_transform_out").data_type]
        bool_type = HLSType(HLSBasicType.BOOL)

        # # 1. Get current key and value from the batch using the PE index
        # key_var = HLSVar("current_key", key_type)
        # val_var = HLSVar("current_val", accum_type)
        # logic = [
        #     CodeVarDecl(key_var.name, key_var.type),
        #     CodeVarDecl(val_var.name, val_var.type),
        #     CodeAssign(
        #         key_var, HLSExpr(HLSExprT.VAR, HLSVar(f"in_key_batch.data[{pe_idx}]", key_type))
        #     ),
        #     CodeAssign(
        #         val_var,
        #         HLSExpr(HLSExprT.VAR, HLSVar(f"in_transform_batch.data[{pe_idx}]", accum_type)),
        #     ),
        # ]
        logic = []

        # *** 关键修改: 所有对内存的访问都使用 pe_idx 作为第一维度 ***
        # 2. Read old element from this PE's BRAM & buffer
        old_ele_var = HLSVar("old_ele", bram_elem_type)
        logic.append(CodeVarDecl(old_ele_var.name, old_ele_var.type))
        logic.append(
            CodeAssign(
                old_ele_var,
                HLSExpr(
                    HLSExprT.VAR, HLSVar(f"key_mem[{pe_idx}][{key_var.name}]", bram_elem_type)
                ),
            )
        )

        # 3. Buffer management for this PE
        buffer_elem_expr = HLSExpr(
            HLSExprT.VAR, HLSVar(f"i_buffer[{pe_idx}][i_search]", HLSType(HLSBasicType.UINT))
        )
        if_condition = HLSExpr(
            HLSExprT.BINOP, dfir.BinOp.EQ, [HLSExpr(HLSExprT.VAR, key_var), buffer_elem_expr]
        )
        value_to_assign = HLSExpr(
            HLSExprT.VAR, HLSVar(f"key_buffer[{pe_idx}][i_search]", bram_elem_type)
        )
        search_loop = CodeFor(
            [
                CodePragma("UNROLL"),
                CodeIf(if_condition, [CodeAssign(old_ele_var, value_to_assign)]),
            ],
            "L + 1",
            iter_name="i_search",
        )
        logic.append(search_loop)

        i_buffer_dest = HLSVar(f"i_buffer[{pe_idx}][i_move]", HLSType(HLSBasicType.UINT))
        i_buffer_src = HLSExpr(
            HLSExprT.VAR, HLSVar(f"i_buffer[{pe_idx}][i_move + 1]", HLSType(HLSBasicType.UINT))
        )
        key_buffer_dest = HLSVar(f"key_buffer[{pe_idx}][i_move]", bram_elem_type)
        key_buffer_src = HLSExpr(
            HLSExprT.VAR, HLSVar(f"key_buffer[{pe_idx}][i_move + 1]", bram_elem_type)
        )
        shift_loop = CodeFor(
            [
                CodePragma("UNROLL"),
                CodeBlock(
                    [
                        CodeAssign(i_buffer_dest, i_buffer_src),
                        CodeAssign(key_buffer_dest, key_buffer_src),
                    ]
                ),
            ],
            "L",
            iter_name="i_move",
        )
        logic.append(shift_loop)

        # 4. If/Else logic for aggregation (logic itself is unchanged)
        new_ele_var = HLSVar("new_ele", bram_elem_type)
        logic.append(CodeVarDecl(new_ele_var.name, new_ele_var.type))
        is_valid_expr = HLSExpr(
            HLSExprT.UOP, (dfir.UnaryOp.GET_ATTR, "ele_1"), [HLSExpr(HLSExprT.VAR, old_ele_var)]
        )
        if_codes = [
            CodeAssign(
                HLSVar(f"{new_ele_var.name}.ele_1", bool_type), HLSExpr(HLSExprT.CONST, True)
            ),
            CodeAssign(
                HLSVar(f"{new_ele_var.name}.ele_0", accum_type), HLSExpr(HLSExprT.VAR, val_var)
            ),
        ]
        old_data_var = HLSVar("old_data", accum_type)
        unit_res_var = HLSVar(f"{new_ele_var.name}.ele_0", accum_type)
        unit_starts = [
            comp.get_port("o_reduce_unit_start_0"),
            comp.get_port("o_reduce_unit_start_1"),
        ]
        unit_end = comp.get_port("i_reduce_unit_end")
        io_map = {unit_starts[0]: old_data_var, unit_starts[1]: val_var, unit_end: unit_res_var}
        else_codes = [
            CodeVarDecl(old_data_var.name, old_data_var.type),
            CodeAssign(
                old_data_var,
                HLSExpr(
                    HLSExprT.UOP,
                    (dfir.UnaryOp.GET_ATTR, "ele_0"),
                    [HLSExpr(HLSExprT.VAR, old_ele_var)],
                ),
            ),
            *self._inline_sub_graph_logic(unit_starts, unit_end, io_map),
            CodeAssign(
                HLSVar(f"{new_ele_var.name}.ele_1", bool_type), HLSExpr(HLSExprT.CONST, True)
            ),
        ]
        logic.append(CodeIf(is_valid_expr, if_codes=else_codes, else_codes=if_codes))

        # 5. Write back to this PE's BRAM and buffer
        logic.append(
            CodeAssign(
                HLSVar(f"key_mem[{pe_idx}][{key_var.name}]", bram_elem_type),
                HLSExpr(HLSExprT.VAR, new_ele_var),
            )
        )
        logic.append(
            CodeAssign(
                HLSVar(f"key_buffer[{pe_idx}][L]", bram_elem_type),
                HLSExpr(HLSExprT.VAR, new_ele_var),
            )
        )
        logic.append(
            CodeAssign(
                HLSVar(f"i_buffer[{pe_idx}][L]", HLSType(HLSBasicType.UINT)),
                HLSExpr(HLSExprT.VAR, key_var),
            )
        )

        return logic

    # ======================================================================== #
    #                            PHASE 4: Final Assembly                       #
    # ======================================================================== #

    def _topologically_sort_structs(self) -> List[Tuple[HLSType, List[str]]]:
        """Sorts struct definitions based on their member dependencies."""
        from collections import defaultdict

        adj = defaultdict(list)
        in_degree = defaultdict(int)

        # Build dependency graph
        for name, (hls_type, _) in self.struct_definitions.items():
            for sub_type in hls_type.sub_types:
                sub_basic_type = sub_type.type
                sub_type_name = sub_type.name
                if sub_basic_type == HLSBasicType.ARRAY:
                    sub_basic_type = sub_type.sub_types[0].type
                    sub_type_name = sub_type_name[:-8]
                if sub_basic_type == HLSBasicType.STRUCT:
                    adj[sub_type_name].append(name)
                    in_degree[name] += 1
                    # print(f"{sub_type_name} => {name}")

        # print(in_degree)
        # Kahn's algorithm for topological sort
        queue = [name for name in self.struct_definitions if in_degree[name] == 0]
        sorted_structs = []

        while queue:
            u = queue.pop(0)
            # print(f"Appending {u}")
            sorted_structs.append(self.struct_definitions[u])
            for v in adj[u]:
                in_degree[v] -= 1
                if in_degree[v] == 0:
                    queue.append(v)

        if len(sorted_structs) != len(self.struct_definitions):
            raise RuntimeError("A cycle was detected in the struct definitions.")

        return sorted_structs

    def _generate_header_file(self, top_func_name: str, top_func_sig: str) -> str:
        """Generates the full content of the .h header file."""
        header_guard = f"__GRAPHYFLOW_{top_func_name.upper()}_H__"
        code = f"#ifndef {header_guard}\n#define {header_guard}\n\n"
        code += "#include <hls_stream.h>\n#include <ap_fixed.h>\n#include <stdint.h>\n\n"
        code += f"#define PE_NUM {self.PE_NUM}\n"
        code += f"#define MAX_NUM {self.MAX_NUM}\n"
        code += f"#define L {self.L}\n\n"

        code += "// --- Struct Type Definitions ---\n"
        sorted_defs = self._topologically_sort_structs()
        for hls_type, members in sorted_defs:
            code += hls_type.gen_decl(members) + "\n"

        code += "// --- Function Prototypes ---\n"
        for func in self.hls_functions.values():
            params_str = ", ".join([p.type.get_upper_param(p.name, True) for p in func.params])
            code += f"void {func.name}({params_str});\n"

        code += f"\n// --- Top-Level Function Prototype ---\n"
        code += f"{top_func_sig};\n\n"

        code += f"#endif // {header_guard}\n"
        return code

    def _generate_top_level_function(self, top_func_name: str, top_func_sig: str) -> str:
        """Generates the implementation of the top-level dataflow function."""
        from collections import defaultdict

        body: List[HLSCodeLine] = [CodePragma("DATAFLOW")]

        # 1. Declare all intermediate streams (including those for Reduce)
        for decl, pragma in self.top_level_stream_decls:
            body.append(decl)
            if pragma:
                body.append(pragma)

        # 2. Prepare maps for top-level I/O and intermediate streams
        stream_map = {decl.var.name: decl.var for decl, _ in self.top_level_stream_decls}
        top_io_map: Dict[int, HLSVar] = {}

        top_level_inputs = []
        for comp in self.comp_col_store.components:
            if (
                isinstance(comp, dfir.IOComponent)
                and comp.io_type == dfir.IOComponent.IOType.INPUT
            ):
                if comp.get_port("o_0").connected:
                    top_level_inputs.append(comp.get_port("o_0").connection)
        top_level_outputs = self.comp_col_store.outputs

        for p in top_level_inputs + top_level_outputs:
            is_array = isinstance(p.data_type, dftype.ArrayType)
            dtype = p.data_type.type_ if is_array else p.data_type
            batch_type = self.batch_type_map[self.type_map[dtype]]
            # The variable name in the call must match the top function's signature
            top_io_map[p.readable_id] = HLSVar(
                p.unique_name, HLSType(HLSBasicType.STREAM, [batch_type])
            )

        # 3. Topologically sort the top-level streamed functions before generating calls
        stream_funcs = [f for f in self.hls_functions.values() if f.streamed]
        id_to_func = {f.readable_id: f for f in stream_funcs}
        comp_to_func = {f.dfir_comp: f for f in stream_funcs}
        reduce_comp_to_pre = {}

        adj = defaultdict(list)
        in_degree = {f.readable_id: 0 for f in stream_funcs}

        # manage all pre_process functions
        for func in stream_funcs:
            if isinstance(func.dfir_comp, dfir.ReduceComponent) and "pre_process" in func.name:
                in_port = func.dfir_comp.get_port("i_0")
                adj[comp_to_func[in_port.connection.parent].readable_id].append(func.readable_id)
                in_degree[func.readable_id] += 1
                reduce_comp_to_pre[func.dfir_comp] = func

        for func in stream_funcs:
            if isinstance(func.dfir_comp, dfir.ReduceComponent):
                if "pre_process" in func.name:
                    continue
                else:
                    assert func.dfir_comp in reduce_comp_to_pre
                    adj[reduce_comp_to_pre[func.dfir_comp].readable_id].append(func.readable_id)
                    in_degree[func.readable_id] += 1
                    continue
            for port in func.dfir_comp.in_ports:
                if port.connected:
                    predecessor_comp = port.connection.parent
                    # An edge exists if the predecessor is also a top-level streamed function
                    if predecessor_comp in comp_to_func:
                        adj[comp_to_func[predecessor_comp].readable_id].append(func.readable_id)
                        in_degree[func.readable_id] += 1

        queue = [fid for fid, degree in in_degree.items() if degree == 0]
        sorted_funcs = []

        while queue:
            func_id = queue.pop(0)
            sorted_funcs.append(id_to_func[func_id])

            for successor_id in adj[func_id]:
                in_degree[successor_id] -= 1
                if in_degree[successor_id] == 0:
                    queue.append(successor_id)

        if len(sorted_funcs) != len(stream_funcs):
            print("\n".join(str(x) for x in sorted_funcs))
            print()
            print("\n".join(str(x) for x in stream_funcs))
            raise RuntimeError("A cycle was detected in the top-level dataflow graph.")

        # 4. Generate calls to all top-level functions in topological order
        body.append(CodeComment("--- Function Calls (in topological order) ---"))
        handled_unit_reduce_ids = set()
        for func in sorted_funcs:
            if (
                "unit_reduce" in func.name
                and func.dfir_comp.readable_id in handled_unit_reduce_ids
            ):
                continue

            if "pre_process" in func.name and isinstance(func.dfir_comp, dfir.ReduceComponent):
                comp_id = func.dfir_comp.readable_id
                helpers = self.reduce_helpers[comp_id]
                streams = helpers["streams"]

                body.append(
                    CodeComment(f"--- Start of Reduce Super-Block for {func.dfir_comp.name} ---")
                )

                pre_process_call_params = []
                for func_param in func.params:
                    param_name = func_param.name
                    if param_name == "i_0":
                        port = func.dfir_comp.get_port(param_name)
                        if port.connected:
                            connection = port.connection
                            conn_parent = connection.parent
                            if isinstance(conn_parent, dfir.IOComponent):
                                if port.readable_id in top_io_map:
                                    pre_process_call_params.append(top_io_map[port.readable_id])
                            else:
                                stream_name = f"stream_{connection.unique_name}"
                                if stream_name in stream_map:
                                    pre_process_call_params.append(stream_map[stream_name])
                    elif param_name in ["intermediate_key", "intermediate_transform"]:
                        stream_var = self.reduce_internal_streams[func.dfir_comp.readable_id][
                            param_name
                        ]
                        pre_process_call_params.append(stream_var)
                body.append(CodeCall(func, pre_process_call_params))

                body.append(
                    CodeCall(
                        helpers["zipper"],
                        [
                            self.reduce_internal_streams[comp_id]["intermediate_key"],
                            self.reduce_internal_streams[comp_id]["intermediate_transform"],
                            streams["zipper_to_demux"],
                        ],
                    )
                )
                body.append(
                    CodeCall(
                        helpers["demux"], [streams["zipper_to_demux"], streams["demux_to_omega"]]
                    )
                )
                body.append(
                    CodeCall(
                        helpers["omega"], [streams["demux_to_omega"], streams["omega_to_unit"]]
                    )
                )
                # body.append(CodeCall(helpers["unzipper"], [streams["omega_to_unzipper"], streams["unzipper_to_unit_key"], streams["unzipper_to_unit_transform"]]))

                unit_reduce_func = helpers["unit_reduce"]
                body.append(
                    CodeCall(
                        unit_reduce_func, [streams["omega_to_unit"], streams["unit_to_final"]]
                    )
                )

                # body.append(CodeCall(helpers["tree"], [HLSExpr(HLSExprT.CONST, 0), streams["unit_to_tree"], streams["tree_to_final"]]))

                body.append(
                    CodeComment(f"--- End of Reduce Super-Block for {func.dfir_comp.name} ---")
                )
                handled_unit_reduce_ids.add(comp_id)

            else:
                call_params: List[HLSVar] = []
                for func_param in func.params:
                    port_name_to_find = func_param.name
                    port = func.dfir_comp.get_port(port_name_to_find)

                    if not port.connected:
                        if port.readable_id in top_io_map:
                            call_params.append(top_io_map[port.readable_id])
                        continue

                    conn_parent = port.connection.parent

                    if isinstance(conn_parent, dfir.ReduceComponent) and port.name == "i_0":
                        reduce_comp_id = conn_parent.readable_id
                        final_stream_var = self.reduce_helpers[reduce_comp_id]["streams"][
                            "unit_to_final"
                        ]
                        call_params.append(final_stream_var)

                    elif isinstance(conn_parent, dfir.IOComponent):
                        call_params.append(top_io_map[port.readable_id])
                    else:
                        out_port = port if port.port_type == dfir.PortType.OUT else port.connection
                        if not isinstance(out_port.parent, dfir.ConstantComponent):
                            stream_name = f"stream_{out_port.unique_name}"
                            if stream_name in stream_map:
                                call_params.append(stream_map[stream_name])

                body.append(CodeCall(func, call_params))

        code = f"{top_func_sig} " + "{\n"
        code += "".join([line.gen_code(1) for line in body])
        code += "}\n"
        return code

    def _generate_source_file(
        self, header_name: str, top_func_name: str, top_func_sig: str
    ) -> str:
        """Generates the full content of the .cpp source file."""
        code = f'#include "{header_name}"\n\n'

        if self.utility_functions:
            code += "// --- Utility Network Functions ---\n"
            for func in self.utility_functions:
                params_str = ", ".join(
                    [
                        p.type.get_upper_param(p.name, p.type.type != HLSBasicType.INT)
                        for p in func.params
                    ]
                )
                code += f"void {func.name}({params_str}) " + "{\n"
                code += "".join([line.gen_code(1) for line in func.codes])
                code += "}\n\n"

        # Generate implementations for all HLS functions
        code += "// --- DFIR Component Functions ---\n"
        for func in self.hls_functions.values():
            params_str = ", ".join([p.type.get_upper_param(p.name, True) for p in func.params])
            code += f"void {func.name}({params_str}) " + "{\n"
            code += "".join([line.gen_code(1) for line in func.codes])
            code += "}\n\n"

        # Generate top-level function implementation
        code += self._generate_top_level_function(top_func_name, top_func_sig)

        return code
