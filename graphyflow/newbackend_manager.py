from __future__ import annotations
from enum import Enum
from pathlib import Path
from typing import List, Optional, Union, Dict, Any, Tuple
import graphyflow.dataflow_ir_datatype as dftype
import graphyflow.dataflow_ir as dfir
import re
import copy

import graphyflow.passes as passes

import graphyflow.dataflow_ir_datatype as dftype
from graphyflow.dataflow_ir import BinOp, UnaryOp
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
    CodeOther,
)
from graphyflow.backend_utils import generate_demux, generate_omega_network, generate_stream_zipper
from graphyflow.backend_mem_manager import MemoryAndGraphManager


class BackendManager:
    """Manages the entire HLS code generation process from a ComponentCollection."""

    def __init__(self):
        self.PE_NUM = 8
        assert self.PE_NUM & (self.PE_NUM - 1) == 0, "PE_NUM must be a power of 2"
        self.LOG_PE_NUM = self.PE_NUM.bit_length() - 1
        self.STREAM_DEPTH = 4
        self.MAX_NUM = 32768  # For ReduceComponent key_mem size
        self.L = 4  # For ReduceComponent buffer size
        # Mappings to store results of type analysis
        self.type_map: Dict[dftype.DfirType, HLSType] = {}
        

        # State for code generation

        self.defines: Dict[str,str]={}
        # big_kernel
        self.big_scatter_funcs: List[HLSFunction] = []
        self.big_gather_funcs: List[HLSFunction] = []

        self.big_top_func = None
        self.big_top_dataflow_funcs =  []
        self.big_apply_top_func = None

        

        # little_kernel
        self.little_scatter_funcs: List[HLSFunction] = []
        self.little_gather_funcs: List[HLSFunction] = []

        self.little_top_func = None
        self.little_top_dataflow_funcs =  []
        self.little_apply_top_func = None

        self.struct_definitions: Dict[str, Tuple[HLSType, List[str]]] = {}
        # other funcs:
        self.apply_funcs : List[HLSFunction] = []
        self.big_helper_funcs = [
"""
template <typename T1, typename T2>
void stream2axistream(hls::stream<T1> &stream, hls::stream<T2> &axi_stream) {

stream2axistream:
    while (true) {

        T1 tmp_t1 = stream.read();

        T2 tmp_t2;
        tmp_t2.data = tmp_t1.idx;
        tmp_t2.dest = tmp_t1.dst;
        tmp_t2.last = tmp_t1.end_flag;

        axi_stream.write(tmp_t2);

        if (tmp_t1.end_flag)
            break;
    }
}
""",
"""
template <typename T1, typename T2>
void axistream2stream(hls::stream<T1> &axi_stream, hls::stream<T2> &stream) {

axistream2stream:
    while (true) {

        T1 tmp_t1 = axi_stream.read();

        T2 tmp_t2;
        tmp_t2.data = tmp_t1.data;
        tmp_t2.dst = tmp_t1.dest;
        tmp_t2.end_flag = tmp_t1.last;

        stream.write(tmp_t2);
        if (tmp_t2.end_flag)
            break;
    }
}
""",

"""
static ap_uint<4> count_end_ones(ap_uint<PE_NUM> valid_mask) {
#pragma HLS INLINE
    ap_uint<4> count = 0;
    switch (valid_mask) {
    case 0:
        count = 0;
        break;
    case 1:
        count = 1;
        break;
    case 3:
        count = 2;
        break;
    case 7:
        count = 3;
        break;
    case 15:
        count = 4;
        break;
    case 31:
        count = 5;
        break;
    case 63:
        count = 6;
        break;
    case 127:
        count = 7;
        break;
    case 255:
        count = 8;
        break;
    default:
        break;
    }
    return count;
}
"""

        ]

        self.little_helper_funcs = [
"""
template <typename T1, typename T2>
void stream2axistream(hls::stream<T1> &stream, hls::stream<T2> &axi_stream) {

stream2axistream:
    while (true) {

        T1 tmp_t1 = stream.read();

        T2 tmp_t2;
        tmp_t2.data = tmp_t1.request_round;
        tmp_t2.last = tmp_t1.end_flag;
        // write_to_stream(axi_stream, tmp_t2);
        axi_stream.write(tmp_t2);
        if (tmp_t1.end_flag)
            break;
    }
}
""",
"""
template <typename T1, typename T2>
void axistream2stream(hls::stream<T1> &axi_stream, hls::stream<T2> &stream) {

axistream2stream:
    while (true) {

        T1 tmp_t1 = axi_stream.read();

        T2 tmp_t2;
        tmp_t2.data = tmp_t1.data;
        tmp_t2.addr = tmp_t1.dest;
        tmp_t2.end_flag = tmp_t1.last;
        // write_to_stream(stream, tmp_t2);
        stream.write(tmp_t2);
        if (tmp_t2.end_flag)
            break;
    }
}
"""
        ]
        
        self.global_graph_store = None
        self.comp_col_store = None

        # New manager for memory and specific graph structures
        self.mem_manager: Optional[MemoryAndGraphManager] = None
        self.dataflow_core_func: Optional[HLSFunction] = None

        
    def _generate_apply(self):
        code = ""
        def write_func_body(func: HLSFunction,is_top):
            nonlocal code
            params_str = ",\n ".join(
                [p.type.get_upper_param(p.name, p.type.type != HLSBasicType.INT and p.type.type!= HLSBasicType.UINT) for p in func.params]
            )
            if is_top:
                code += f"extern \"C\" void\n {func.name}({params_str}) " + "{\n"
            else:
                code += f"static void\n{func.name}({params_str}) " + "{\n"
            code += "".join([line.gen_code(1) for line in func.codes])
            code += "}\n\n"

        for func in self.apply_funcs:
            write_func_body(func,False) 
        
        return code

    def _dfirtype_to_hlstype(self, dfir_type: dftype.DfirType) -> HLSType:
        node_id_type = HLSType(basic_type=HLSBasicType.NODE_ID)
        distance_type = HLSType(basic_type=HLSBasicType.AP_FIXED_POD)
        if isinstance(dfir_type , dftype.SpecialIdType):
            if dfir_type.type_name == "node_id":
                return node_id_type
            else:
                assert 0
        elif isinstance(dfir_type , dftype.FloatType):
            return distance_type
        elif isinstance(dfir_type , dftype.ArrayType):
            elem_type = dfir_type.type_
            if isinstance(elem_type , dftype.SpecialIdType):
                if elem_type.type_name == "node_id":
                    return node_id_type#HLSType(basic_type=HLSBasicType.ARRAY, sub_types=[node_id_type])
                else:
                    assert 0
            elif isinstance(elem_type , dftype.FloatType):
                return distance_type#HLSType(basic_type=HLSBasicType.ARRAY, sub_types=[distance_type])
        else:
            assert 0

    def _add_pod_to_float_cast(
        self, pod_expr: HLSExpr, code_list: List[HLSCodeLine], base_name: str
    ) -> HLSExpr:
        """
        Generates code to cast a POD type (int32_t) to a computational float (ap_fixed).
        Handles both variables and constants correctly by returning an HLSExpr.
        Appends prerequisite declarations to code_list for variables.
        """
        # If the expression is a constant, return a direct C++ cast expression.
        # This will generate "((ap_fixed<32, 16>)0.0)" which is legal C++.
        if pod_expr.type == HLSExprT.CONST:
            return HLSExpr(HLSExprT.UOP, UnaryOp.CAST_FLOAT, [pod_expr])

        # If the expression is a variable, generate the reinterpret_cast logic.
        elif pod_expr.type == HLSExprT.VAR:
            ap_fixed_type = HLSType(HLSBasicType.FLOAT)
            float_var = HLSVar(base_name, ap_fixed_type)

            # 1. Declare a new ap_fixed variable.
            # 2. Initialize it by reinterpreting the bits of the input POD variable.
            cast_str = f"*reinterpret_cast<ap_fixed<32, 16>*>(&{pod_expr.code})"
            code_list.append(CodeVarDecl(float_var.name, float_var.type, init_val=cast_str))

            # 3. Return an expression that refers to this new temporary variable.
            return HLSExpr(HLSExprT.VAR, float_var)

        else:
            raise TypeError(f"Unsupported HLSExpr type for casting: {pod_expr.type}")
        
    def _translate_inline_component(
        self, comp: dfir.Component, p2var_map: Dict[dfir.Port, HLSVar], code_lines: List[HLSCodeLine]
    ):

        if isinstance(comp, dfir.BinOpComponent):
            op1_expr = HLSExpr.check_const(
                HLSExpr(HLSExprT.VAR, p2var_map[comp.get_port("i_0").connection]), comp.get_port("i_0")
            )
            op2_expr = HLSExpr.check_const(
                HLSExpr(HLSExprT.VAR, p2var_map[comp.get_port("i_1").connection]), comp.get_port("i_1")
            )
            target_var = p2var_map[comp.get_port("o_0")]

            expr = HLSExpr(HLSExprT.BINOP, comp.op, [op1_expr, op2_expr])
            code_lines.append(CodeAssign(target_var, expr))

        else:
            print("unimplemented inline component type:", type(comp))
        pass

    def _topologically_sort_structs(self,struct_definitions) -> List[Tuple[HLSType, List[str]]]:
        """Sorts struct definitions based on their member dependencies."""
        from collections import defaultdict

        adj = defaultdict(list)
        in_degree = defaultdict(int)

        all_struct_names = struct_definitions.keys()

        # Build dependency graph
        for dependent_struct_name, (hls_type, _) in struct_definitions.items():
            if dependent_struct_name not in in_degree:
                in_degree[dependent_struct_name] = 0

            if not hls_type.sub_types:
                continue

            for member_type in hls_type.sub_types:

                base_member_type = member_type
                if base_member_type.type == HLSBasicType.ARRAY:
                    base_member_type = base_member_type.sub_types[0]

                if base_member_type.type == HLSBasicType.STRUCT:
                    dependency_struct_name = base_member_type.name

                    if dependency_struct_name in all_struct_names:
                        # --- *** 关键修正：修复依赖关系 *** ---
                        # The dependent_struct depends on the dependency_struct.
                        # The edge is: dependency_struct -> dependent_struct.
                        adj[dependency_struct_name].append(dependent_struct_name)
                        in_degree[dependent_struct_name] += 1

        # Kahn's algorithm for topological sort
        queue = [name for name in struct_definitions if in_degree[name] == 0]
        sorted_structs = []

        while queue:
            u = queue.pop(0)
            if u in struct_definitions:
                sorted_structs.append(struct_definitions[u])
                for v in adj[u]:
                    in_degree[v] -= 1
                    if in_degree[v] == 0:
                        queue.append(v)

        if len(sorted_structs) != len(struct_definitions):
            print("--- DEBUG: Cycle detected in struct dependencies ---")
            print("Total structs:", len(struct_definitions))
            print("Sorted structs:", len(sorted_structs))
            print("Remaining in_degrees:", {k: v for k, v in in_degree.items() if v > 0})
            raise RuntimeError("A cycle was detected in the struct definitions.")

        return sorted_structs
    
    def _generate_header_file(self):

        def write_func_sig(func: HLSFunction,header_file):
            params_str = ",\n ".join(
                [p.type.get_upper_param(p.name, p.type.type != HLSBasicType.INT and p.type.type != HLSBasicType.UINT) for p in func.params]
            )
            
            header_file += f"extern \"C\" void\n {func.name}({params_str} \n);\n\n"
            return header_file
            
        header_guard = f"__GRAPHYFLOW_GRAPHYFLOW_BIG_H__"
        big_header = f"#ifndef {header_guard}\n#define {header_guard}\n\n"
        little_header = f"#ifndef __GRAPHYFLOW_GRAPHYFLOW_LITTLE_H__\n#define __GRAPHYFLOW_GRAPHYFLOW_LITTLE_H__\n\n"

        shared_kernel_params = f"#ifndef __SHARED_KERNEL_PARAMS_H__\n#define __SHARED_KERNEL_PARAMS_H__\n\n"

        code = ""
        code += f"#include <ap_axi_sdata.h>\n"
        code += f"#include <ap_fixed.h>\n"
        code += f"#include <ap_int.h>\n"
        code += f"#include <hls_stream.h>\n"
        code += f"#include <stdint.h>\n"
        code += f"#include <stdio.h>\n"
        code += f"#include <string.h>\n"
        code += f"\n"
        code += f"#define PE_NUM {8}\n"
        code += f"#define DBL_PE_NUM {16}\n"
        code += f"#define LOG_PE_NUM {3}\n"
        code += f"#ifdef EMULATION\n"
        code += f"#define MAX_NUM {512}\n"
        code += f"#else\n"
        code += f"#define MAX_NUM {524288}\n"
        code += f"#endif\n"
        code += f"#define L {3}\n"
        code += f"\n"
        code += f"// --- New Bitwidth Definitions for HLS Synthesis ---\n"
        code += f"#define NODE_ID_BITWIDTH {32}\n"
        code += f"#define DISTANCE_BITWIDTH {32}\n"
        code += f"#define DISTANCE_INTEGER_PART {16}\n"
        code += f"#define WEIGHT_BITWIDTH DISTANCE_BITWIDTH\n"
        code += f"#define WEIGHT_INTEGER_PART DISTANCE_INTEGER_PART\n"
        code += f"#define OUT_END_MARKER_BITWIDTH {4}\n"
        code += f"#define DIST_PER_WORD {16} // AXI_BUS_WIDTH / DISTANCE_BITWIDTH = {512} / {32} = {16}\n"
        code += f"#define LOG_DIST_PER_WORD                                            \\\n"
        code += f"    {4} // log2(AXI_BUS_WIDTH / DISTANCE_BITWIDTH) = log2({512} / {32}) = log2({16}) = {4}\n"
        code += f"\n"
        code += f"// --- New Memory Word and Bus Definitions ---\n"
        code += f"#define AXI_BUS_WIDTH {512}\n"
        code += f"\n"
        code += f"#define REDUCE_MEM_WIDTH {64}\n"
        code += f"typedef ap_uint<AXI_BUS_WIDTH> bus_word_t;\n"
        code += f"typedef ap_uint<REDUCE_MEM_WIDTH> reduce_word_t;\n"
        code += f"\n"
        code += f"const int INFINITY_DIST = {16384};\n"
        code += f"\n"
        code += f"// --- New Packing-related Constants ---\n"
        code += f"// Number of distances that can be packed into a single reduce memory word.\n"
        code += f"#define DISTANCES_PER_REDUCE_WORD (REDUCE_MEM_WIDTH / DISTANCE_BITWIDTH)\n"
        code += f"\n"
        code += f"// --- Redefinition of Core Graph Types for HLS ---\n"
        code += f"// These typedefs override the standard integer types from common.h for\n"
        code += f"// synthesis.\n"
        code += f"typedef ap_uint<NODE_ID_BITWIDTH> node_id_t;\n"
        code += f"typedef ap_uint<{32}> edge_id_t; // edge_id_t is not customized yet, keep as is.\n"
        code += f"typedef ap_uint<DISTANCE_BITWIDTH>\n"
        code += f"    ap_fixed_pod_t; // Used to hold bit representation of ap_fixed types\n"
        code += f"typedef ap_fixed<DISTANCE_BITWIDTH, DISTANCE_INTEGER_PART> distance_t;\n"
        code += f"typedef ap_uint<OUT_END_MARKER_BITWIDTH> out_end_marker_t;\n"
        code += f"typedef ap_axiu<{256}, {0}, {0}, {0}> node_dist_pkt_t;\n"
        code += f"typedef ap_axiu<{512}, {0}, {0}, {0}> write_burst_pkt_t;\n"
        code += f"typedef ap_axiu<{32}, {0}, {0}, {8}> cacheline_request_pkt_t;\n"
        code += f"typedef ap_axiu<{512}, {0}, {0}, {8}> cacheline_response_pkt_t;\n"
        code += f"typedef ap_axiu<{512}, {0}, {0}, {0}> cacheline_data_pkt_t;\n"

        big_header += code

        # little part:
        code = ""
        code += f"#include <ap_axi_sdata.h>\n"
        code += f"#include <ap_fixed.h>\n"
        code += f"#include <ap_int.h>\n"
        code += f"#include <hls_stream.h>\n"
        code += f"#include <stdint.h>\n"
        code += f"#include <stdio.h>\n"
        code += f"#include <string.h>\n"
        code += f"\n"
        code += f"#define PE_NUM {8}\n"
        code += f"#define DBL_PE_NUM {16}\n"
        code += f"#define LOG_PE_NUM {3}\n"
        code += f"#ifdef EMULATION\n"
        code += f"#define MAX_NUM {512}\n"
        code += f"#else\n"
        code += f"#define MAX_NUM {65536}\n"
        code += f"#endif\n"
        code += f"#define L {3}\n"
        code += f"#define SRC_BUFFER_SIZE {4096}\n"
        code += f"#define LOG_SRC_BUFFER_SIZE {12}\n"
        code += f"\n"
        code += f"// --- New Bitwidth Definitions for HLS Synthesis ---\n"
        code += f"#define NODE_ID_BITWIDTH {32}\n"
        code += f"#define DISTANCE_BITWIDTH {32}\n"
        code += f"#define DISTANCE_INTEGER_PART {16}\n"
        code += f"#define WEIGHT_BITWIDTH DISTANCE_BITWIDTH\n"
        code += f"#define WEIGHT_INTEGER_PART DISTANCE_INTEGER_PART\n"
        code += f"#define OUT_END_MARKER_BITWIDTH {4}\n"
        code += f"#define DIST_PER_WORD {16} // AXI_BUS_WIDTH / DISTANCE_BITWIDTH = {512} / {32} = {16}\n"
        code += f"#define LOG_DIST_PER_WORD                                            \\\n"
        code += f"    {4} // log2(AXI_BUS_WIDTH / DISTANCE_BITWIDTH) = log2({512} / {32}) = log2({16}) = {4}\n"
        code += f"\n"
        code += f"// --- New Memory Word and Bus Definitions ---\n"
        code += f"#define AXI_BUS_WIDTH {512}\n"
        code += f"\n"
        code += f"#define REDUCE_MEM_WIDTH {64}\n"
        code += f"typedef ap_uint<AXI_BUS_WIDTH> bus_word_t;\n"
        code += f"typedef ap_uint<REDUCE_MEM_WIDTH> reduce_word_t;\n"
        code += f"\n"
        code += f"const int INFINITY_DIST = {16384};\n"
        code += f"\n"
        code += f"// --- New Packing-related Constants ---\n"
        code += f"// Number of distances that can be packed into a single reduce memory word.\n"
        code += f"#define DISTANCES_PER_REDUCE_WORD (REDUCE_MEM_WIDTH / DISTANCE_BITWIDTH)\n"
        code += f"\n"
        code += f"// --- Redefinition of Core Graph Types for HLS ---\n"
        code += f"// These typedefs override the standard integer types from common.h for\n"
        code += f"// synthesis.\n"
        code += f"typedef ap_uint<NODE_ID_BITWIDTH> node_id_t;\n"
        code += f"typedef ap_uint<{32}> edge_id_t; // edge_id_t is not customized yet, keep as is.\n"
        code += f"typedef ap_uint<DISTANCE_BITWIDTH>\n"
        code += f"    ap_fixed_pod_t; // Used to hold bit representation of ap_fixed types\n"
        code += f"typedef ap_fixed<DISTANCE_BITWIDTH, DISTANCE_INTEGER_PART> distance_t;\n"
        code += f"typedef ap_uint<OUT_END_MARKER_BITWIDTH> out_end_marker_t;\n"
        code += f"typedef ap_axiu<{256}, {0}, {0}, {0}> node_dist_pkt_t;\n"
        code += f"typedef ap_axiu<{512}, {0}, {0}, {0}> write_burst_pkt_t;\n"
        code += f"typedef ap_axiu<{64}, {0}, {0}, {0}> little_out_pkt_t;\n"
        code += f"typedef ap_axiu<{32}, {0}, {0}, {0}> ppb_request_pkt_t;\n"
        code += f"typedef ap_axiu<{512}, {0}, {0}, {32}> ppb_response_pkt_t;\n"
        code += f"typedef ap_axiu<{512}, {0}, {0}, {0}> cacheline_data_pkt_t;\n"
        little_header += code

        # shared params part:
        code = ""
        code += f"#include <ap_axi_sdata.h>\n"
        code += f"#include <ap_fixed.h>\n"
        code += f"#include <ap_int.h>\n"
        code += f"#include <hls_stream.h>\n"
        code += f"#include <stdint.h>\n"
        code += f"#include <stdio.h>\n"
        code += f"#include <string.h>\n"
        code += f"\n"
        code += f"#define PE_NUM {8}\n"
        code += f"#define DBL_PE_NUM {16}\n"
        code += f"#define LOG_PE_NUM {3}\n"
        code += f"#define L {4}\n"
        code += f"#define SRC_BUFFER_SIZE {4096}\n"
        code += f"#define LOG_SRC_BUFFER_SIZE {12}\n"
        code += f"\n"
        code += f"#define NODE_ID_BITWIDTH {32}\n"
        code += f"#define DISTANCE_BITWIDTH {32}\n"
        code += f"#define DISTANCE_INTEGER_PART {16}\n"
        code += f"#define WEIGHT_BITWIDTH DISTANCE_BITWIDTH\n"
        code += f"#define WEIGHT_INTEGER_PART DISTANCE_INTEGER_PART\n"
        code += f"#define OUT_END_MARKER_BITWIDTH {4}\n"
        code += f"#define DIST_PER_WORD {16} // AXI_BUS_WIDTH / DISTANCE_BITWIDTH = {512} / {32} = {16}\n"
        code += f"#define LOG_DIST_PER_WORD                                            \\\n"
        code += f"    {4} // log2(AXI_BUS_WIDTH / DISTANCE_BITWIDTH) = log2({512} / {32}) = log2({16}) = {4}\n"
        code += f"\n"
        code += f"// --- New Memory Word and Bus Definitions ---\n"
        code += f"#define AXI_BUS_WIDTH {512}\n"
        code += f"\n"
        code += f"#define BIG_MERGER_LENGTH {3}\n"
        code += f"#define LITTLE_MERGER_LENGTH {11}\n"
        code += f"\n"
        code += f"#define REDUCE_MEM_WIDTH {64}\n"
        code += f"typedef ap_uint<AXI_BUS_WIDTH> bus_word_t;\n"
        code += f"typedef ap_uint<REDUCE_MEM_WIDTH> reduce_word_t;\n"
        code += f"\n"
        code += f"const int INFINITY_DIST = {16384};\n"
        code += f"\n"
        code += f"#define DISTANCES_PER_REDUCE_WORD (REDUCE_MEM_WIDTH / DISTANCE_BITWIDTH)\n"
        code += f"\n"
        code += f"// --- Redefinition of Core Graph Types for HLS ---\n"
        code += f"// These typedefs override the standard integer types from common.h for\n"
        code += f"// synthesis.\n"
        code += f"typedef ap_uint<NODE_ID_BITWIDTH> node_id_t;\n"
        code += f"typedef ap_uint<{32}> edge_id_t; // edge_id_t is not customized yet, keep as is.\n"
        code += f"typedef ap_uint<DISTANCE_BITWIDTH>\n"
        code += f"    ap_fixed_pod_t; // Used to hold bit representation of ap_fixed types\n"
        code += f"typedef ap_fixed<DISTANCE_BITWIDTH, DISTANCE_INTEGER_PART> distance_t;\n"
        code += f"typedef ap_uint<OUT_END_MARKER_BITWIDTH> out_end_marker_t;\n"
        code += f"typedef ap_axiu<{256}, {0}, {0}, {0}> node_dist_pkt_t;\n"
        code += f"typedef ap_axiu<{512}, {0}, {0}, {0}> write_burst_pkt_t;\n"
        code += f"typedef ap_axiu<{64}, {0}, {0}, {0}> little_out_pkt_t;\n"
        code += f"typedef ap_axiu<{512}, {0}, {0}, {32}> write_burst_w_dst_pkt_t;\n"
        code += f"typedef ap_axiu<{32}, {0}, {0}, {8}> cacheline_request_pkt_t;\n"
        code += f"typedef ap_axiu<{512}, {0}, {0}, {8}> cacheline_response_pkt_t;\n"
        code += f"typedef ap_axiu<{32}, {0}, {0}, {0}> ppb_request_pkt_t;\n"
        code += f"typedef ap_axiu<{512}, {0}, {0}, {32}> ppb_response_pkt_t;\n"
        code += f"typedef ap_axiu<{512}, {0}, {0}, {0}> cacheline_data_pkt_t;\n"

        shared_kernel_params += code

        big_header += "// --- Struct Type Definitions ---\n"
        sorted_defs = self._topologically_sort_structs(self.struct_definitions)
        for hls_type, members in sorted_defs:
            big_header += hls_type.gen_decl(members) + "\n"
            #little_header += hls_type.gen_decl(members) + "\n"
            shared_kernel_params += hls_type.gen_decl(members) + "\n"

        sorted_defs = self._topologically_sort_structs(self.struct_definitions)
        for hls_type, members in sorted_defs:
            #big_header += hls_type.gen_decl(members) + "\n"
            little_header += hls_type.gen_decl(members) + "\n"

        big_header += "// --- Top-Level Function Prototypes ---\n"
        
        big_header = write_func_sig(self.big_top_func,big_header)
        little_header = write_func_sig(self.little_top_func,little_header)

        shared_kernel_params = write_func_sig(self.big_top_func,shared_kernel_params)
        shared_kernel_params = write_func_sig(self.little_top_func,shared_kernel_params)


        big_header += f"#endif // {header_guard}\n"
        little_header += f"#endif // __GRAPHYFLOW_GRAPHYFLOW_LITTLE_H__\n"
        
        return big_header,little_header,shared_kernel_params
    

    def _generate_top_func(self) -> Tuple[str, str]:

        graphyflow_big_func = HLSFunction(name="graphyflow_big", comp=None) # 假设 comp 存在
        params = []

        # --- 1. 定义类型和参数 (主要来自 graphyflow_big.h) ---

        # 基础类型
        int_type = HLSType(HLSBasicType.INT)       # int32_t
        uint_type = HLSType(HLSBasicType.UINT)      # uint32_t
        bus_word_t_type = HLSType(HLSBasicType.BUS_WORD_T)
        node_id_type = HLSType(HLSBasicType.NODE_ID)
        ap_fixed_pod_t_type = HLSType(HLSBasicType.AP_FIXED_POD)
        bool_type = HLSType(HLSBasicType.BOOL)
        # (根据你的澄清，使用 HLSBasicType.REDUCE_WORD_T)
        reduce_word_t_type = HLSType(HLSBasicType.REDUCE_WORD_T) 

        # .h 和 C++ 中使用的特定 ap_uint 类型
        ap_uint64_type = HLSType(basic_type=HLSBasicType.AP_UINT, width=64)
        ap_uint26_type = HLSType(basic_type=HLSBasicType.AP_UINT, width=26)
        ap_uint20_type = HLSType(basic_type=HLSBasicType.AP_UINT, width=20)
        ap_uint8_type = HLSType(basic_type=HLSBasicType.AP_UINT, width=8)
        ap_uint4_type = HLSType(basic_type=HLSBasicType.AP_UINT, width=4)

        # --- AXI Stream Packet Types (来自 .h typedefs) ---
        cacheline_request_pkt_t_type = HLSType(HLSBasicType.CACHELINE_REQUEST_PKT_T)
        cacheline_response_pkt_t_type = HLSType(HLSBasicType.CACHELINE_RESPONSE_PKT_T)
        write_burst_pkt_t_type = HLSType(HLSBasicType.WRITE_BURST_PKT_T)

        # --- 结构体定义 (严格按照 graphyflow_big.h) ---

        # struct node_id_burst_t
        node_id_array_type = HLSType(HLSBasicType.ARRAY, sub_types=[node_id_type], array_dims=["PE_NUM"])
        node_id_burst_t_type = HLSType(basic_type=HLSBasicType.STRUCT,
                                       struct_name="node_id_burst_t",
                                       struct_prop_names=["data"],
                                       sub_types=[node_id_array_type])
        if node_id_burst_t_type.name not in self.struct_definitions:
            self.struct_definitions[node_id_burst_t_type.name] = (node_id_burst_t_type, node_id_burst_t_type.struct_prop_names)

        # struct distance_req_pack_t
        ap_uint26_array_type = HLSType(HLSBasicType.ARRAY, sub_types=[ap_uint26_type], array_dims=["PE_NUM"])
        distance_req_pack_t_type = HLSType(basic_type=HLSBasicType.STRUCT,
                                           struct_name="distance_req_pack_t",
                                           struct_prop_names=["idx", "offset", "end_flag"],
                                           sub_types=[ap_uint26_array_type, ap_uint4_type, bool_type])
        if distance_req_pack_t_type.name not in self.struct_definitions:
            self.struct_definitions[distance_req_pack_t_type.name] = (distance_req_pack_t_type, distance_req_pack_t_type.struct_prop_names)

        # struct edge_t
        edge_t_type = HLSType(basic_type=HLSBasicType.STRUCT,
                              struct_name="edge_t",
                              struct_prop_names=["src_id", "dst_id"],
                              sub_types=[node_id_type, ap_uint20_type])
        if edge_t_type.name not in self.struct_definitions:
            self.struct_definitions[edge_t_type.name] = (edge_t_type, edge_t_type.struct_prop_names)

        # struct edge_descriptor_batch_t
        edge_array_type = HLSType(HLSBasicType.ARRAY, sub_types=[edge_t_type], array_dims=["PE_NUM"])
        edge_descriptor_batch_t_type = HLSType(basic_type=HLSBasicType.STRUCT,
                                               struct_name="edge_descriptor_batch_t",
                                               struct_prop_names=["edges"],
                                               sub_types=[edge_array_type])
        if edge_descriptor_batch_t_type.name not in self.struct_definitions:
            self.struct_definitions[edge_descriptor_batch_t_type.name] = (edge_descriptor_batch_t_type, edge_descriptor_batch_t_type.struct_prop_names)

        # struct update_t (使用 .h 中带 end_flag 的版本)
        update_t_type = HLSType(basic_type=HLSBasicType.STRUCT,
                                struct_name="update_t_big",
                                struct_prop_names=["node_id", "prop", "end_flag"],
                                sub_types=[ap_uint20_type, ap_fixed_pod_t_type, bool_type])
        if update_t_type.name not in self.struct_definitions:
            self.struct_definitions[update_t_type.name] = (update_t_type, update_t_type.struct_prop_names)

        # struct update_tuple_t
        update_t_array_type = HLSType(HLSBasicType.ARRAY, sub_types=[update_t_type], array_dims=["PE_NUM"])
        update_tuple_t_type = HLSType(basic_type=HLSBasicType.STRUCT,
                                      struct_name="update_tuple_t_big",
                                      struct_prop_names=["data"],
                                      sub_types=[update_t_array_type])
        if update_tuple_t_type.name not in self.struct_definitions:
            self.struct_definitions[update_tuple_t_type.name] = (update_tuple_t_type, update_tuple_t_type.struct_prop_names)

        # struct cacheline_req_t
        cacheline_req_t_type = HLSType(basic_type=HLSBasicType.STRUCT,
                                       struct_name="cacheline_req_t",
                                       struct_prop_names=["idx", "dst", "end_flag"],
                                       sub_types=[ap_uint26_type, ap_uint8_type, bool_type])
        if cacheline_req_t_type.name not in self.struct_definitions:
            self.struct_definitions[cacheline_req_t_type.name] = (cacheline_req_t_type, cacheline_req_t_type.struct_prop_names)

        # struct cacheline_resp_t
        cacheline_resp_t_type = HLSType(basic_type=HLSBasicType.STRUCT,
                                        struct_name="cacheline_resp_t",
                                        struct_prop_names=["data", "dst", "end_flag"],
                                        sub_types=[bus_word_t_type, ap_uint8_type, bool_type])
        if cacheline_resp_t_type.name not in self.struct_definitions:
            self.struct_definitions[cacheline_resp_t_type.name] = (cacheline_resp_t_type, cacheline_resp_t_type.struct_prop_names)


        # --- Stream 类型 (本地) ---
        stream_node_id_burst_type = HLSType(HLSBasicType.STREAM, sub_types=[node_id_burst_t_type])
        stream_dist_req_pack_type = HLSType(HLSBasicType.STREAM, sub_types=[distance_req_pack_t_type])
        stream_bus_word_type = HLSType(HLSBasicType.STREAM, sub_types=[bus_word_t_type])
        stream_edge_desc_batch_type = HLSType(HLSBasicType.STREAM, sub_types=[edge_descriptor_batch_t_type])
        stream_update_tuple_type = HLSType(HLSBasicType.STREAM, sub_types=[update_tuple_t_type])
        stream_cacheline_req_type = HLSType(HLSBasicType.STREAM, sub_types=[cacheline_req_t_type])
        stream_cacheline_resp_type = HLSType(HLSBasicType.STREAM, sub_types=[cacheline_resp_t_type])
        stream_update_t_type = HLSType(HLSBasicType.STREAM, sub_types=[update_t_type])
        stream_reduce_word_t_type = HLSType(HLSBasicType.STREAM, sub_types=[reduce_word_t_type]) # <--- 应用了你的澄清

        # --- Stream 数组类型 (本地) ---
        stream_cachelines_type = HLSType(HLSBasicType.ARRAY, sub_types=[stream_bus_word_type], array_dims=["PE_NUM"])
        reduce_105_d2o_pair_type = HLSType(HLSBasicType.ARRAY, sub_types=[stream_update_t_type], array_dims=[8])
        reduce_105_o2u_pair_type = HLSType(HLSBasicType.ARRAY, sub_types=[stream_update_t_type], array_dims=[8])
        stream_stage_0_type = HLSType(HLSBasicType.ARRAY, sub_types=[stream_update_t_type], array_dims=[8])
        stream_stage_1_type = HLSType(HLSBasicType.ARRAY, sub_types=[stream_update_t_type], array_dims=[8])
        pe_mem_out_streams_type = HLSType(HLSBasicType.ARRAY, sub_types=[stream_reduce_word_t_type], array_dims=["PE_NUM"]) # <--- 应用了你的澄清


        # --- 参数变量 ---
        ptr_bus_word_t_type = HLSType(HLSBasicType.POINTER, sub_types=[bus_word_t_type], is_const_ptr=True)
        edge_props_var = HLSVar(var_name="edge_props", var_type=ptr_bus_word_t_type)
        num_nodes_var = HLSVar(var_name="num_nodes", var_type=int_type)
        num_edges_var = HLSVar(var_name="num_edges", var_type=int_type)
        dst_num_var = HLSVar(var_name="dst_num", var_type=int_type)
        memory_offset_var = HLSVar(var_name="memory_offset", var_type=int_type)

        stream_cacheline_req_pkt_type = HLSType(HLSBasicType.STREAM, sub_types=[cacheline_request_pkt_t_type])
        cacheline_req_stream_var = HLSVar(var_name="cacheline_req_stream", var_type=stream_cacheline_req_pkt_type)

        stream_cacheline_resp_pkt_type = HLSType(HLSBasicType.STREAM, sub_types=[cacheline_response_pkt_t_type])
        cacheline_resp_stream_var = HLSVar(var_name="cacheline_resp_stream", var_type=stream_cacheline_resp_pkt_type)

        stream_write_burst_pkt_type = HLSType(HLSBasicType.STREAM, sub_types=[write_burst_pkt_t_type])
        kernel_out_stream_var = HLSVar(var_name="kernel_out_stream", var_type=stream_write_burst_pkt_type)

        params.extend([edge_props_var, num_nodes_var, num_edges_var, dst_num_var, memory_offset_var,
                        cacheline_req_stream_var, cacheline_resp_stream_var, kernel_out_stream_var])
        graphyflow_big_func.params = params

        # --- 2. 函数体 ---
        code_lines: List[HLSCodeLine] = []

        # --- 接口 Pragmas ---
        code_lines.append(CodePragma(content="INTERFACE m_axi port = edge_props offset = slave bundle = gmem0"))
        code_lines.append(CodePragma(content="INTERFACE s_axilite port = edge_props"))
        code_lines.append(CodePragma(content="INTERFACE s_axilite port = num_nodes"))
        code_lines.append(CodePragma(content="INTERFACE s_axilite port = num_edges"))
        code_lines.append(CodePragma(content="INTERFACE s_axilite port = dst_num"))
        code_lines.append(CodePragma(content="INTERFACE s_axilite port = memory_offset"))
        code_lines.append(CodePragma(content="INTERFACE s_axilite port = return"))
        code_lines.append(CodePragma(content="DATAFLOW"))
        code_lines.append(CodeOther(text=""))

        # --- 本地 Stream 声明 ---
        code_lines.append(CodeComment(text="Streams for the new COO-style property loading"))
        # hls::stream<node_id_burst_t> stream_src_ids;
        code_lines.append(CodeVarDecl(var_name="stream_src_ids", var_type=stream_node_id_burst_type))
        code_lines.append(CodePragma(content="STREAM variable = stream_src_ids depth = 16"))
        stream_src_ids_var = HLSVar(var_name="stream_src_ids", var_type=stream_node_id_burst_type)

        # hls::stream<distance_req_pack_t> stream_dist_req;
        code_lines.append(CodeVarDecl(var_name="stream_dist_req", var_type=stream_dist_req_pack_type))
        code_lines.append(CodePragma(content="STREAM variable = stream_dist_req depth = 16"))
        stream_dist_req_var = HLSVar(var_name="stream_dist_req", var_type=stream_dist_req_pack_type)

        # hls::stream<bus_word_t> stream_cachelines[PE_NUM];
        code_lines.append(CodeVarDecl(var_name="stream_cachelines", var_type=stream_cachelines_type))
        code_lines.append(CodePragma(content="STREAM variable = stream_cachelines depth = 16"))
        stream_cachelines_var = HLSVar(var_name="stream_cachelines", var_type=stream_cachelines_type)

        # hls::stream<edge_descriptor_batch_t> edge_stream;
        code_lines.append(CodeVarDecl(var_name="edge_stream", var_type=stream_edge_desc_batch_type))
        code_lines.append(CodePragma(content="STREAM variable = edge_stream depth = 16"))
        edge_stream_var = HLSVar(var_name="edge_stream", var_type=stream_edge_desc_batch_type)

        # hls::stream<update_tuple_t> stream_edge_data;
        code_lines.append(CodeVarDecl(var_name="stream_edge_data", var_type=stream_update_tuple_type))
        code_lines.append(CodePragma(content="STREAM variable = stream_edge_data depth = 16"))
        stream_edge_data_var = HLSVar(var_name="stream_edge_data", var_type=stream_update_tuple_type)

        # hls::stream<cacheline_req_t> cacheline_req;
        code_lines.append(CodeVarDecl(var_name="cacheline_req", var_type=stream_cacheline_req_type))
        code_lines.append(CodePragma(content="STREAM variable = cacheline_req depth = 32"))
        cacheline_req_var = HLSVar(var_name="cacheline_req", var_type=stream_cacheline_req_type)

        # hls::stream<cacheline_resp_t> cacheline_resp;
        code_lines.append(CodeVarDecl(var_name="cacheline_resp", var_type=stream_cacheline_resp_type))
        code_lines.append(CodePragma(content="STREAM variable = cacheline_resp depth = 32"))
        cacheline_resp_var = HLSVar(var_name="cacheline_resp", var_type=stream_cacheline_resp_type)

        # hls::stream<update_t> reduce_105_d2o_pair[8];
        code_lines.append(CodeVarDecl(var_name="reduce_105_d2o_pair", var_type=reduce_105_d2o_pair_type))
        code_lines.append(CodePragma(content="STREAM variable = reduce_105_d2o_pair depth = 8"))
        code_lines.append(CodePragma(content="ARRAY_PARTITION variable = reduce_105_d2o_pair complete dim = 0"))
        reduce_105_d2o_pair_var = HLSVar(var_name="reduce_105_d2o_pair", var_type=reduce_105_d2o_pair_type)

        # hls::stream<update_t> reduce_105_o2u_pair[8];
        code_lines.append(CodeVarDecl(var_name="reduce_105_o2u_pair", var_type=reduce_105_o2u_pair_type))
        code_lines.append(CodePragma(content="STREAM variable = reduce_105_o2u_pair depth = 2"))
        code_lines.append(CodePragma(content="ARRAY_PARTITION variable = reduce_105_o2u_pair complete dim = 0"))
        reduce_105_o2u_pair_var = HLSVar(var_name="reduce_105_o2u_pair", var_type=reduce_105_o2u_pair_type)

        code_lines.append(CodeOther(text=""))

        # --- 常量声明 ---
        # const uint32_t num_words = (dst_num + 1) / DISTANCES_PER_REDUCE_WORD;
        code_lines.append(CodeVarDecl(var_name="num_words", var_type=uint_type, init_val="((dst_num + 1) / DISTANCES_PER_REDUCE_WORD)", const=True))
        # const uint32_t num_word_per_pe = (num_words + PE_NUM - 1) >> LOG_PE_NUM;
        code_lines.append(CodeVarDecl(var_name="num_word_per_pe", var_type=uint_type, init_val="((num_words + PE_NUM - 1) >> LOG_PE_NUM)", const=True))
        num_word_per_pe_var = HLSVar(var_name="num_word_per_pe", var_type=uint_type)
        code_lines.append(CodeOther(text=""))

        # --- Data Loading ---
        code_lines.append(CodeComment(text="--- Data Loading ---"))
        # const int edges_per_word = ...
        code_lines.append(CodeVarDecl(var_name="edges_per_word", var_type=int_type, init_val="(AXI_BUS_WIDTH / (NODE_ID_BITWIDTH + NODE_ID_BITWIDTH))", const=True))
        edges_per_word_var = HLSVar(var_name="edges_per_word", var_type=int_type)
        # const int num_wide_reads = num_edges / edges_per_word;
        code_lines.append(CodeVarDecl(var_name="num_wide_reads", var_type=int_type, init_val="(num_edges / edges_per_word)", const=True))
        num_wide_reads_var = HLSVar(var_name="num_wide_reads", var_type=int_type)
        code_lines.append(CodeOther(text=""))

        # LOOP_EDL_READ:
        code_lines.append(CodeOther(text="LOOP_EDL_READ:"))
        # for (int i = 0; i < num_wide_reads; i++)
        for_1_codes: List[HLSCodeLine] = []
        # #pragma HLS PIPELINE II = 1
        for_1_codes.append(CodePragma(content="PIPELINE II = 1"))
        # bus_word_t wide_word = edge_props[i];
        for_1_codes.append(CodeVarDecl(var_name="wide_word", var_type=bus_word_t_type, init_val="edge_props[i]"))
        wide_word_var = HLSVar(var_name="wide_word", var_type=bus_word_t_type)
        # edge_descriptor_batch_t edge_batch;
        for_1_codes.append(CodeVarDecl(var_name="edge_batch", var_type=edge_descriptor_batch_t_type))
        edge_batch_var = HLSVar(var_name="edge_batch", var_type=edge_descriptor_batch_t_type)
        for_1_codes.append(CodeOther(text=""))

        # LOOP_EDL_UNPACK:
        for_1_codes.append(CodeOther(text="LOOP_EDL_UNPACK:"))
        # for (int j = 0; j < edges_per_word; j++)
        for_2_codes: List[HLSCodeLine] = []
        # #pragma HLS UNROLL
        for_2_codes.append(CodePragma(content="UNROLL"))
        # ap_uint<64> packed_edge = wide_word.range(63 + (j << 6), (j << 6));
        for_2_codes.append(CodeVarDecl(var_name="packed_edge", var_type=ap_uint64_type, init_val="wide_word.range(63 + (j << 6), (j << 6))"))
        # edge_t edge;
        for_2_codes.append(CodeVarDecl(var_name="edge", var_type=edge_t_type))
        edge_var = HLSVar(var_name="edge", var_type=edge_t_type)
        # edge.dst_id = packed_edge.range(19, 0);
        edge_dst_var = HLSVar(var_name="edge.dst_id", var_type=ap_uint20_type)
        for_2_codes.append(CodeAssign(var=edge_dst_var, expr=HLSExpr(HLSExprT.CONST, "packed_edge.range(19, 0)")))
        # edge.src_id = packed_edge.range(63, 32);
        edge_src_var = HLSVar(var_name="edge.src_id", var_type=node_id_type)
        for_2_codes.append(CodeAssign(var=edge_src_var, expr=HLSExpr(HLSExprT.CONST, "packed_edge.range(63, 32)")))
        # edge_batch.edges[j] = edge;
        edge_batch_j_var = HLSVar(var_name="edge_batch.edges[j]", var_type=edge_t_type)
        for_2_codes.append(CodeAssign(var=edge_batch_j_var, expr=HLSExpr(HLSExprT.VAR, edge_var)))
        # (构建 for_2)
        for_1_codes.append(CodeFor(codes=for_2_codes, iter_limit=edges_per_word_var, iter_name="j", iter_val_type=int_type))
        # edge_stream.write(edge_batch);
        for_1_codes.append(CodeWriteStream(stream_var=edge_stream_var, in_expr=edge_batch_var))
        for_1_codes.append(CodeOther(text=""))

        # node_id_burst_t src_id_burst;
        for_1_codes.append(CodeVarDecl(var_name="src_id_burst", var_type=node_id_burst_t_type))
        src_id_burst_var = HLSVar(var_name="src_id_burst", var_type=node_id_burst_t_type)
        # for (int j = 0; j < edges_per_word; j++)
        for_3_codes: List[HLSCodeLine] = []
        # #pragma HLS UNROLL
        for_3_codes.append(CodePragma(content="UNROLL"))
        # node_id_t src_id;
        for_3_codes.append(CodeVarDecl(var_name="src_id", var_type=node_id_type))
        src_id_var = HLSVar(var_name="src_id", var_type=node_id_type)
        # src_id = edge_batch.edges[j].src_id;
        for_3_codes.append(CodeAssign(var=src_id_var, expr=HLSExpr(HLSExprT.CONST, "edge_batch.edges[j].src_id")))
        # src_id_burst.data[j] = src_id;
        src_id_burst_j_var = HLSVar(var_name="src_id_burst.data[j]", var_type=node_id_type)
        for_3_codes.append(CodeAssign(var=src_id_burst_j_var, expr=HLSExpr(HLSExprT.VAR, src_id_var)))
        # (构建 for_3)
        for_1_codes.append(CodeFor(codes=for_3_codes, iter_limit=edges_per_word_var, iter_name="j", iter_val_type=int_type))
        # stream_src_ids.write(src_id_burst);
        for_1_codes.append(CodeWriteStream(stream_var=stream_src_ids_var, in_expr=src_id_burst_var))

        # (构建 for_1)
        code_lines.append(CodeFor(codes=for_1_codes, iter_limit=num_wide_reads_var, iter_name="i", iter_val_type=int_type))
        code_lines.append(CodeOther(text=""))

        # --- New COO-style Source Property Loading Pipeline ---
        code_lines.append(CodeComment(text="--- New COO-style Source Property Loading Pipeline ---"))
        # dist_req_packer(stream_src_ids, stream_dist_req, num_edges);
        call_params_1 = [stream_src_ids_var, stream_dist_req_var, num_edges_var]
        # code_lines.append(CodeCall(func=dist_req_packer_func, params=call_params_1))
        code_lines.append(CodeCall(func=self.big_top_dataflow_funcs[0], params=call_params_1))
        # cacheline_req_sender(stream_dist_req, cacheline_req, memory_offset);
        call_params_2 = [stream_dist_req_var, cacheline_req_var, memory_offset_var]
        # code_lines.append(CodeCall(func=cacheline_req_sender_func, params=call_params_2))
        code_lines.append(CodeCall(func=self.big_top_dataflow_funcs[1], params=call_params_2))


        # stream2axistream(cacheline_req, cacheline_req_stream);
        call_params_3 = [cacheline_req_var, cacheline_req_stream_var]
        #code_lines.append(CodeCall(func=stream2axistream_func, params=call_params_3))
        code_lines.append(CodeOther(text="stream2axistream(cacheline_req, cacheline_req_stream);"))
        # axistream2stream(cacheline_resp_stream, cacheline_resp);
        call_params_4 = [cacheline_resp_stream_var, cacheline_resp_var]
        #code_lines.append(CodeCall(func=axistream2stream_func, params=call_params_4))
        code_lines.append(CodeOther(text="axistream2stream(cacheline_resp_stream, cacheline_resp);"))
        # node_prop_resp_receiver(cacheline_resp, stream_cachelines);
        call_params_5 = [cacheline_resp_var, stream_cachelines_var]
        # code_lines.append(CodeCall(func=node_prop_resp_receiver_func, params=call_params_5))
        #code_lines.append(CodeCall(func=self.big_top_dataflow_funcs[2], params=call_params_5))

        # merge_node_props(stream_cachelines, edge_stream, stream_edge_data, num_edges);
        call_params_6 = [stream_cachelines_var, edge_stream_var, stream_edge_data_var, num_edges_var]
        # code_lines.append(CodeCall(func=merge_node_props_func, params=call_params_6))
        code_lines.append(CodeCall(func=self.big_top_dataflow_funcs[3], params=call_params_6))
        
        code_lines.append(CodeOther(text=""))

        # demux_1(stream_edge_data, reduce_105_d2o_pair, num_edges);
        call_params_7 = [stream_edge_data_var, reduce_105_d2o_pair_var, num_edges_var]
        # code_lines.append(CodeCall(func=demux_1_func, params=call_params_7))
        code_lines.append(CodeCall(func=self.big_top_dataflow_funcs[4], params=call_params_7))
        code_lines.append(CodeOther(text=""))

        # --- Benes Network (switch2x2_2 calls) ---
        # hls::stream<update_t> stream_stage_0[8];
        code_lines.append(CodeVarDecl(var_name="stream_stage_0", var_type=stream_stage_0_type))
        code_lines.append(CodePragma(content="STREAM variable = stream_stage_0 depth = 2"))
        code_lines.append(CodePragma(content="ARRAY_PARTITION variable = stream_stage_0 complete dim = 0"))
        stream_stage_0_var = HLSVar(var_name="stream_stage_0", var_type=stream_stage_0_type)

        # hls::stream<update_t> stream_stage_1[8];
        code_lines.append(CodeVarDecl(var_name="stream_stage_1", var_type=stream_stage_1_type))
        code_lines.append(CodePragma(content="STREAM variable = stream_stage_1 depth = 2"))
        code_lines.append(CodePragma(content="ARRAY_PARTITION variable = stream_stage_1 complete dim = 0"))
        stream_stage_1_var = HLSVar(var_name="stream_stage_1", var_type=stream_stage_1_type)

        # --- Stage 0 Calls ---
        call_s0_0_params = [HLSExpr(HLSExprT.CONST, 2), HLSVar("reduce_105_d2o_pair[0]", stream_update_t_type), HLSVar("reduce_105_d2o_pair[1]", stream_update_t_type), HLSVar("stream_stage_0[0]", stream_update_t_type), HLSVar("stream_stage_0[1]", stream_update_t_type)]
        code_lines.append(CodeCall(func=self.big_top_dataflow_funcs[5], params=call_s0_0_params))
        call_s0_1_params = [HLSExpr(HLSExprT.CONST, 2), HLSVar("reduce_105_d2o_pair[2]", stream_update_t_type), HLSVar("reduce_105_d2o_pair[3]", stream_update_t_type), HLSVar("stream_stage_0[2]", stream_update_t_type), HLSVar("stream_stage_0[3]", stream_update_t_type)]
        code_lines.append(CodeCall(func=self.big_top_dataflow_funcs[5], params=call_s0_1_params))
        call_s0_2_params = [HLSExpr(HLSExprT.CONST, 2), HLSVar("reduce_105_d2o_pair[4]", stream_update_t_type), HLSVar("reduce_105_d2o_pair[5]", stream_update_t_type), HLSVar("stream_stage_0[4]", stream_update_t_type), HLSVar("stream_stage_0[5]", stream_update_t_type)]
        code_lines.append(CodeCall(func=self.big_top_dataflow_funcs[5], params=call_s0_2_params))
        call_s0_3_params = [HLSExpr(HLSExprT.CONST, 2), HLSVar("reduce_105_d2o_pair[6]", stream_update_t_type), HLSVar("reduce_105_d2o_pair[7]", stream_update_t_type), HLSVar("stream_stage_0[6]", stream_update_t_type), HLSVar("stream_stage_0[7]", stream_update_t_type)]
        code_lines.append(CodeCall(func=self.big_top_dataflow_funcs[5], params=call_s0_3_params))

        # --- Stage 1 Calls ---
        call_s1_0_params = [HLSExpr(HLSExprT.CONST, 1), HLSVar("stream_stage_0[0]", stream_update_t_type), HLSVar("stream_stage_0[4]", stream_update_t_type), HLSVar("stream_stage_1[0]", stream_update_t_type), HLSVar("stream_stage_1[1]", stream_update_t_type)]
        code_lines.append(CodeCall(func=self.big_top_dataflow_funcs[5], params=call_s1_0_params))
        call_s1_1_params = [HLSExpr(HLSExprT.CONST, 1), HLSVar("stream_stage_0[1]", stream_update_t_type), HLSVar("stream_stage_0[5]", stream_update_t_type), HLSVar("stream_stage_1[2]", stream_update_t_type), HLSVar("stream_stage_1[3]", stream_update_t_type)]
        code_lines.append(CodeCall(func=self.big_top_dataflow_funcs[5], params=call_s1_1_params))
        call_s1_2_params = [HLSExpr(HLSExprT.CONST, 1), HLSVar("stream_stage_0[2]", stream_update_t_type), HLSVar("stream_stage_0[6]", stream_update_t_type), HLSVar("stream_stage_1[4]", stream_update_t_type), HLSVar("stream_stage_1[5]", stream_update_t_type)]
        code_lines.append(CodeCall(func=self.big_top_dataflow_funcs[5], params=call_s1_2_params))
        call_s1_3_params = [HLSExpr(HLSExprT.CONST, 1), HLSVar("stream_stage_0[3]", stream_update_t_type), HLSVar("stream_stage_0[7]", stream_update_t_type), HLSVar("stream_stage_1[6]", stream_update_t_type), HLSVar("stream_stage_1[7]", stream_update_t_type)]
        code_lines.append(CodeCall(func=self.big_top_dataflow_funcs[5], params=call_s1_3_params))

        # --- Stage 2 (Final) Calls ---
        call_s2_0_params = [HLSExpr(HLSExprT.CONST, 0), HLSVar("stream_stage_1[0]", stream_update_t_type), HLSVar("stream_stage_1[4]", stream_update_t_type), HLSVar("reduce_105_o2u_pair[0]", stream_update_t_type), HLSVar("reduce_105_o2u_pair[1]", stream_update_t_type)]
        code_lines.append(CodeCall(func=self.big_top_dataflow_funcs[5], params=call_s2_0_params))
        call_s2_1_params = [HLSExpr(HLSExprT.CONST, 0), HLSVar("stream_stage_1[1]", stream_update_t_type), HLSVar("stream_stage_1[5]", stream_update_t_type), HLSVar("reduce_105_o2u_pair[2]", stream_update_t_type), HLSVar("reduce_105_o2u_pair[3]", stream_update_t_type)]
        code_lines.append(CodeCall(func=self.big_top_dataflow_funcs[5], params=call_s2_1_params))
        call_s2_2_params = [HLSExpr(HLSExprT.CONST, 0), HLSVar("stream_stage_1[2]", stream_update_t_type), HLSVar("stream_stage_1[6]", stream_update_t_type), HLSVar("reduce_105_o2u_pair[4]", stream_update_t_type), HLSVar("reduce_105_o2u_pair[5]", stream_update_t_type)]
        code_lines.append(CodeCall(func=self.big_top_dataflow_funcs[5], params=call_s2_2_params))
        call_s2_3_params = [HLSExpr(HLSExprT.CONST, 0), HLSVar("stream_stage_1[3]", stream_update_t_type), HLSVar("stream_stage_1[7]", stream_update_t_type), HLSVar("reduce_105_o2u_pair[6]", stream_update_t_type), HLSVar("reduce_105_o2u_pair[7]", stream_update_t_type)]
        code_lines.append(CodeCall(func=self.big_top_dataflow_funcs[5], params=call_s2_3_params))

        # --- Reduce and Drain ---
        # // Reduc_105_unit_reduce(reduce_105_o2u_pair, stream_o_0_107, dst_num);
        code_lines.append(CodeComment(text="Reduc_105_unit_reduce(reduce_105_o2u_pair, stream_o_0_107, dst_num);"))
        # hls::stream<reduce_word_t> pe_mem_out_streams[PE_NUM];
        code_lines.append(CodeVarDecl(var_name="pe_mem_out_streams", var_type=pe_mem_out_streams_type))
        code_lines.append(CodePragma(content="STREAM variable = pe_mem_out_streams depth = 4"))
        pe_mem_out_streams_var = HLSVar(var_name="pe_mem_out_streams", var_type=pe_mem_out_streams_type)

        # LOOP_FOR_60:
        code_lines.append(CodeOther(text="LOOP_FOR_60:"))
        # for (int32_t pe_idx = 0; pe_idx < PE_NUM; pe_idx++)
        for_4_codes: List[HLSCodeLine] = []
        # #pragma HLS UNROLL
        for_4_codes.append(CodePragma(content="UNROLL"))
        # Reduc_105_unit_reduce_single_pe(...)
        reduce_call_params = [
            HLSVar("reduce_105_o2u_pair[pe_idx]", stream_update_t_type),
            HLSVar("pe_mem_out_streams[pe_idx]", stream_reduce_word_t_type),
            num_word_per_pe_var
        ]
        #for_4_codes.append(CodeCall(func=Reduc_105_unit_reduce_single_pe_func, params=reduce_call_params))
        for_4_codes.append(CodeCall(func=self.big_top_dataflow_funcs[6], params=reduce_call_params))
        # (构建 for_4)
        code_lines.append(CodeFor(codes=for_4_codes, iter_limit="PE_NUM", iter_name="pe_idx", iter_val_type=int_type))

        # Reduc_105_drain_multi_pe(...)
        drain_call_params = [pe_mem_out_streams_var, kernel_out_stream_var, num_word_per_pe_var]
        # code_lines.append(CodeCall(func=Reduc_105_drain_multi_pe_func, params=drain_call_params))
        code_lines.append(CodeCall(func=self.big_top_dataflow_funcs[7], params=drain_call_params))

        # --- 3. Finalize ---
        graphyflow_big_func.codes = code_lines

        self.big_top_func = graphyflow_big_func


        # little part:
    
        graphyflow_little_func = HLSFunction(name="graphyflow_little", comp=None) 
        params = []

        # --- 1. 定义类型和参数 (根据 graphyflow_little.h) ---

        # 基础类型
        int_type = HLSType(HLSBasicType.INT)       # int32_t
        uint_type = HLSType(HLSBasicType.UINT)      # uint32_t
        bus_word_t_type = HLSType(HLSBasicType.BUS_WORD_T)
        node_id_type = HLSType(HLSBasicType.NODE_ID)
        ap_fixed_pod_t_type = HLSType(HLSBasicType.AP_FIXED_POD)
        bool_type = HLSType(HLSBasicType.BOOL)
        # (根据你的澄清，使用 HLSBasicType.REDUCE_WORD_T)
        reduce_word_t_type = HLSType(HLSBasicType.REDUCE_WORD_T) 

        # .h 和 C++ 中使用的特定 ap_uint 类型
        ap_uint64_type = HLSType(basic_type=HLSBasicType.AP_UINT, width=64)
        ap_uint20_type = HLSType(basic_type=HLSBasicType.AP_UINT, width=20)

        # --- AXI Stream Packet Types (来自 .h typedefs) ---
        ppb_request_pkt_t_type = HLSType(HLSBasicType.PPB_REQUEST_PKT_T)
        ppb_response_pkt_t_type = HLSType(HLSBasicType.PPB_RESPONSE_PKT_T)
        little_out_pkt_t_type = HLSType(HLSBasicType.LITTLE_OUT_PKT_T)

        # --- 结构体定义 (严格按照 graphyflow_little.h) ---

        # struct edge_t
        edge_t_type = HLSType(basic_type=HLSBasicType.STRUCT,
                              struct_name="edge_t",
                              struct_prop_names=["src_id", "dst_id"],
                              sub_types=[node_id_type, ap_uint20_type])
        if edge_t_type.name not in self.struct_definitions:
            self.struct_definitions[edge_t_type.name] = (edge_t_type, edge_t_type.struct_prop_names)

        # struct edge_descriptor_batch_t
        edge_array_type = HLSType(HLSBasicType.ARRAY, sub_types=[edge_t_type], array_dims=["PE_NUM"])
        edge_descriptor_batch_t_type = HLSType(basic_type=HLSBasicType.STRUCT,
                                               struct_name="edge_descriptor_batch_t",
                                               struct_prop_names=["edges"],
                                               sub_types=[edge_array_type])
        if edge_descriptor_batch_t_type.name not in self.struct_definitions:
            self.struct_definitions[edge_descriptor_batch_t_type.name] = (edge_descriptor_batch_t_type, edge_descriptor_batch_t_type.struct_prop_names)


        update_t_type = HLSType(basic_type=HLSBasicType.STRUCT,
                                struct_name="update_t_little",
                                struct_prop_names=["node_id", "prop"],
                                sub_types=[ap_uint20_type, ap_fixed_pod_t_type]) 
        if update_t_type.name not in self.struct_definitions:
            self.struct_definitions[update_t_type.name] = (update_t_type, update_t_type.struct_prop_names)

        # struct update_tuple_t
        update_t_array_type = HLSType(HLSBasicType.ARRAY, sub_types=[update_t_type], array_dims=["PE_NUM"])
        update_tuple_t_type = HLSType(basic_type=HLSBasicType.STRUCT,
                                      struct_name="update_tuple_t_little",
                                      struct_prop_names=["data"],
                                      sub_types=[update_t_array_type])
        if update_tuple_t_type.name not in self.struct_definitions:
            self.struct_definitions[update_tuple_t_type.name] = (update_tuple_t_type, update_tuple_t_type.struct_prop_names)

        # struct ppb_request_t
        ppb_request_t_type = HLSType(basic_type=HLSBasicType.STRUCT,
                                     struct_name="ppb_request_t",
                                     struct_prop_names=["request_round", "end_flag"],
                                     sub_types=[HLSType(HLSBasicType.AP_UINT,width=32), bool_type])
        if ppb_request_t_type.name not in self.struct_definitions:
            self.struct_definitions[ppb_request_t_type.name] = (ppb_request_t_type, ppb_request_t_type.struct_prop_names)

        # struct ppb_response_t
        ppb_response_t_type = HLSType(basic_type=HLSBasicType.STRUCT,
                                      struct_name="ppb_response_t",
                                      struct_prop_names=["data", "addr", "end_flag"],
                                      sub_types=[bus_word_t_type, HLSType(HLSBasicType.AP_UINT,width=32), bool_type])
        if ppb_response_t_type.name not in self.struct_definitions:
            self.struct_definitions[ppb_response_t_type.name] = (ppb_response_t_type, ppb_response_t_type.struct_prop_names)


        # --- Stream 类型 (本地和参数) ---
        stream_edge_desc_batch_type = HLSType(HLSBasicType.STREAM, sub_types=[edge_descriptor_batch_t_type])
        stream_update_tuple_type = HLSType(HLSBasicType.STREAM, sub_types=[update_tuple_t_type])
        stream_reduce_word_t_type = HLSType(HLSBasicType.STREAM, sub_types=[reduce_word_t_type])
        stream_ppb_request_type = HLSType(HLSBasicType.STREAM, sub_types=[ppb_request_t_type])
        stream_ppb_response_type = HLSType(HLSBasicType.STREAM, sub_types=[ppb_response_t_type])

        stream_ppb_request_pkt_type = HLSType(HLSBasicType.STREAM, sub_types=[ppb_request_pkt_t_type])
        stream_ppb_response_pkt_type = HLSType(HLSBasicType.STREAM, sub_types=[ppb_response_pkt_t_type])
        stream_little_out_pkt_type = HLSType(HLSBasicType.STREAM, sub_types=[little_out_pkt_t_type])

        # --- Stream 数组类型 (本地) ---
        pe_mem_outs_type = HLSType(HLSBasicType.ARRAY, sub_types=[stream_reduce_word_t_type], array_dims=["PE_NUM"])

        # --- 参数变量 ---
        ptr_bus_word_t_type = HLSType(HLSBasicType.POINTER, sub_types=[bus_word_t_type], is_const_ptr=True)
        edge_props_var = HLSVar(var_name="edge_props", var_type=ptr_bus_word_t_type)
        num_nodes_var = HLSVar(var_name="num_nodes", var_type=int_type)
        num_edges_var = HLSVar(var_name="num_edges", var_type=int_type)
        dst_num_var = HLSVar(var_name="dst_num", var_type=int_type)
        memory_offset_var = HLSVar(var_name="memory_offset", var_type=int_type)
        ppb_req_stream_var = HLSVar(var_name="ppb_req_stream", var_type=stream_ppb_request_pkt_type)
        ppb_resp_stream_var = HLSVar(var_name="ppb_resp_stream", var_type=stream_ppb_response_pkt_type)
        kernel_out_stream_var = HLSVar(var_name="kernel_out_stream", var_type=stream_little_out_pkt_type)

        params.extend([edge_props_var, num_nodes_var, num_edges_var, dst_num_var, memory_offset_var,
                        ppb_req_stream_var, ppb_resp_stream_var, kernel_out_stream_var])
        graphyflow_little_func.params = params

        # --- 2. 函数体 ---
        code_lines: List[HLSCodeLine] = []

        # --- 接口 Pragmas ---
        code_lines.append(CodePragma(content="INTERFACE m_axi port = edge_props offset = slave bundle = gmem0"))
        code_lines.append(CodePragma(content="INTERFACE s_axilite port = edge_props"))
        code_lines.append(CodePragma(content="INTERFACE s_axilite port = num_nodes"))
        code_lines.append(CodePragma(content="INTERFACE s_axilite port = num_edges"))
        code_lines.append(CodePragma(content="INTERFACE s_axilite port = dst_num"))
        code_lines.append(CodePragma(content="INTERFACE s_axilite port = return"))
        code_lines.append(CodePragma(content="DATAFLOW"))
        code_lines.append(CodeOther(text=""))

        # --- 本地 Stream 声明 ---
        code_lines.append(CodeComment(text="Existing streams"))
        # hls::stream<edge_descriptor_batch_t> edge_stream;
        code_lines.append(CodeVarDecl(var_name="edge_stream", var_type=stream_edge_desc_batch_type))
        code_lines.append(CodePragma(content="STREAM variable = edge_stream depth = 8"))
        edge_stream_var = HLSVar(var_name="edge_stream", var_type=stream_edge_desc_batch_type)

        # hls::stream<update_tuple_t> stream_edge_data;
        code_lines.append(CodeVarDecl(var_name="stream_edge_data", var_type=stream_update_tuple_type))
        code_lines.append(CodePragma(content="STREAM variable = stream_edge_data depth = 8"))
        stream_edge_data_var = HLSVar(var_name="stream_edge_data", var_type=stream_update_tuple_type)

        # hls::stream<reduce_word_t> pe_mem_outs[PE_NUM];
        code_lines.append(CodeVarDecl(var_name="pe_mem_outs", var_type=pe_mem_outs_type))
        code_lines.append(CodePragma(content="STREAM variable = pe_mem_outs depth = 8"))
        pe_mem_outs_var = HLSVar(var_name="pe_mem_outs", var_type=pe_mem_outs_type)
        code_lines.append(CodeOther(text=""))

        # hls::stream<ppb_request_t> ppb_req_stream_internal;
        code_lines.append(CodeVarDecl(var_name="ppb_req_stream_internal", var_type=stream_ppb_request_type))
        code_lines.append(CodePragma(content="STREAM variable = ppb_req_stream_internal depth = 8"))
        ppb_req_stream_internal_var = HLSVar(var_name="ppb_req_stream_internal", var_type=stream_ppb_request_type)

        # hls::stream<ppb_response_t> ppb_resp_stream_internal;
        code_lines.append(CodeVarDecl(var_name="ppb_resp_stream_internal", var_type=stream_ppb_response_type))
        code_lines.append(CodePragma(content="STREAM variable = ppb_resp_stream_internal depth = 8"))
        ppb_resp_stream_internal_var = HLSVar(var_name="ppb_resp_stream_internal", var_type=stream_ppb_response_type)
        code_lines.append(CodeOther(text=""))

        # --- Data Loading ---
        code_lines.append(CodeComment(text="--- Data Loading ---"))
        code_lines.append(CodeComment(text="--- Data Loading ---"))
        # const int edges_per_word = ...
        code_lines.append(CodeVarDecl(var_name="edges_per_word", var_type=int_type, init_val="(AXI_BUS_WIDTH / (NODE_ID_BITWIDTH + NODE_ID_BITWIDTH))", const=True))
        edges_per_word_var = HLSVar(var_name="edges_per_word", var_type=int_type)
        # const int num_wide_reads = num_edges / edges_per_word;
        code_lines.append(CodeVarDecl(var_name="num_wide_reads", var_type=int_type, init_val="(num_edges / edges_per_word)", const=True))
        num_wide_reads_var = HLSVar(var_name="num_wide_reads", var_type=int_type)
        code_lines.append(CodeOther(text=""))

        # const uint32_t num_words = (dst_num + 1) >> 1;
        code_lines.append(CodeVarDecl(var_name="num_words", var_type=uint_type, init_val="((dst_num + 1) >> 1)", const=True))
        # const uint32_t rounded_num_words = ((num_words + 7) & ~7);
        code_lines.append(CodeVarDecl(var_name="rounded_num_words", var_type=uint_type, init_val="((num_words + 7) & ~7)", const=True))
        rounded_num_words_var = HLSVar(var_name="rounded_num_words", var_type=uint_type)
        code_lines.append(CodeOther(text=""))

        # LOOP_EDL_READ:
        code_lines.append(CodeOther(text="LOOP_EDL_READ:"))
        # for (int i = 0; i < num_wide_reads; i++)
        for_1_codes: List[HLSCodeLine] = []
        # #pragma HLS PIPELINE II = 1
        for_1_codes.append(CodePragma(content="PIPELINE II = 1"))
        # bus_word_t wide_word = edge_props[i];
        for_1_codes.append(CodeVarDecl(var_name="wide_word", var_type=bus_word_t_type, init_val="edge_props[i]"))
        wide_word_var = HLSVar(var_name="wide_word", var_type=bus_word_t_type)
        # edge_descriptor_batch_t edge_batch;
        for_1_codes.append(CodeVarDecl(var_name="edge_batch", var_type=edge_descriptor_batch_t_type))
        edge_batch_var = HLSVar(var_name="edge_batch", var_type=edge_descriptor_batch_t_type)

        # LOOP_EDL_UNPACK:
        for_1_codes.append(CodeOther(text="LOOP_EDL_UNPACK:"))
        # for (int j = 0; j < edges_per_word; j++)
        for_2_codes: List[HLSCodeLine] = []
        # #pragma HLS UNROLL
        for_2_codes.append(CodePragma(content="UNROLL"))
        # ap_uint<64> packed_edge = wide_word.range(63 + (j << 6), (j << 6));
        for_2_codes.append(CodeVarDecl(var_name="packed_edge", var_type=ap_uint64_type, init_val="wide_word.range(63 + (j << 6), (j << 6))"))
        # edge_batch.edges[j].dst_id = packed_edge.range(19, 0);
        lhs_dst = HLSVar(var_name="edge_batch.edges[j].dst_id", var_type=ap_uint20_type)
        for_2_codes.append(CodeAssign(var=lhs_dst, expr=HLSExpr(HLSExprT.CONST, "packed_edge.range(19, 0)")))
        # edge_batch.edges[j].src_id = packed_edge.range(63, 32);
        lhs_src = HLSVar(var_name="edge_batch.edges[j].src_id", var_type=node_id_type)
        for_2_codes.append(CodeAssign(var=lhs_src, expr=HLSExpr(HLSExprT.CONST, "packed_edge.range(63, 32)")))
        # (构建 for_2)
        for_1_codes.append(CodeFor(codes=for_2_codes, iter_limit=edges_per_word_var, iter_name="j", iter_val_type=int_type))
        # edge_stream.write(edge_batch);
        for_1_codes.append(CodeWriteStream(stream_var=edge_stream_var, in_expr=edge_batch_var))

        # (构建 for_1)
        code_lines.append(CodeFor(codes=for_1_codes, iter_limit=num_wide_reads_var, iter_name="i", iter_val_type=int_type))
        code_lines.append(CodeOther(text=""))

        # --- 函数调用 ---
        # stream2axistream(ppb_req_stream_internal, ppb_req_stream);
        call_params_1 = [ppb_req_stream_internal_var, ppb_req_stream_var]
        # code_lines.append(CodeCall(func=stream2axistream_func, params=call_params_1))
        code_lines.append(CodeOther(text="stream2axistream(ppb_req_stream_internal, ppb_req_stream);"))

        # axistream2stream(ppb_resp_stream, ppb_resp_stream_internal);
        call_params_2 = [ppb_resp_stream_var, ppb_resp_stream_internal_var]
        # code_lines.append(CodeCall(func=axistream2stream_func, params=call_params_2))
        code_lines.append(CodeOther(text="axistream2stream(ppb_resp_stream, ppb_resp_stream_internal);"))

        # request_manager(...)
        call_params_3 = [edge_stream_var, ppb_req_stream_internal_var, ppb_resp_stream_internal_var, 
                         stream_edge_data_var, memory_offset_var, num_edges_var]
        code_lines.append(CodeCall(func=self.little_top_dataflow_funcs[0], params=call_params_3))
        code_lines.append(CodeOther(text=""))

        # --- Reduction ---
        code_lines.append(CodeComment(text="--- Reduction ---"))
        # Reduc_105_unit_reduce(...)
        call_params_4 = [stream_edge_data_var, pe_mem_outs_var, num_edges_var, rounded_num_words_var]
        code_lines.append(CodeCall(func=self.little_top_dataflow_funcs[1], params=call_params_4))

        # Reduc_105_drain_multi_pe(...)
        call_params_5 = [pe_mem_outs_var, kernel_out_stream_var, rounded_num_words_var]
        code_lines.append(CodeCall(func=self.little_top_dataflow_funcs[2], params=call_params_5))

        # --- 3. Finalize ---
        graphyflow_little_func.codes = code_lines

        self.little_top_func = graphyflow_little_func


    def _generate_source_file(self) -> str:
        """Generates the full content of the .cpp source file with correct function order."""

        big_code = f'#include \"graphyflow_big.h\"\n\n'

        # --- Function Definition Order ---
        # 1. Memory helper functions (lowest level)
        # 2. Utility Network Functions (zipper, demux, etc.)
        # 3. DFIR Component Functions (computational logic)
        # 4. Top-level Memory/Dataflow functions (callers)
        # 5. Top-level AXI Kernel Wrapper (final orchestrator)

        def write_func_body(func: HLSFunction,is_top,code):
            
            if func is None:
                return code + "\nNot Implemented\n\n"
            params_str = ",\n ".join(
                [p.type.get_upper_param(p.name, p.type.type != HLSBasicType.INT and p.type.type != HLSBasicType.UINT) for p in func.params]
            )
            if len(func.codes) != 0:
                if is_top:
                    code += f"extern \"C\" void\n {func.name}({params_str}) " + "{\n"
                else:
                    code += f"static void \n{func.name}({params_str}) " + "{\n"
                code += "".join([line.gen_code(1) for line in func.codes])
                code += "}\n\n"
            return code

        for func in self.big_helper_funcs:
            big_code = big_code + func + "\n\n"

        if self.big_scatter_funcs:
            big_code += "// --- 1. scatter_funcs ---\n"
            for func in self.big_scatter_funcs:
                big_code = write_func_body(func,False,big_code)

        if self.big_gather_funcs:
            big_code += "// --- 2. gather_funcs ---\n"
            for func in self.big_gather_funcs:
                big_code = write_func_body(func,False,big_code)

        big_code += "// --- 4. top func ---\n"
        big_code = write_func_body(self.big_top_func,True,big_code)

        little_code = f'#include \"graphyflow_little.h\"\n\n'

        for func in self.little_helper_funcs:
            little_code = little_code + func + "\n\n"

        if self.little_scatter_funcs:
            little_code += "// --- 1. scatter_funcs ---\n"
            for func in self.little_scatter_funcs:
                little_code = write_func_body(func,False,little_code)

        if self.little_gather_funcs:
            little_code += "// --- 2. gather_funcs ---\n"
            for func in self.little_gather_funcs:
                little_code = write_func_body(func,False,little_code)

        little_code += "// --- 4. top func ---\n"
        little_code = write_func_body(self.little_top_func,True,little_code)

        return big_code,little_code



    def _translate_memory_read_op(self, comp: dfir.Component):
        
        # hard code 除了merge_node_props以外的函数 

        dist_req_packer_func = HLSFunction(name="dist_req_packer", comp=comp)
        params = []
        #
        # dist_req_packer(hls::stream<node_id_burst_t> &src_id_burst_stream,
        #         hls::stream<distance_req_pack_t> &distance_req_pack_stream,
        #         int32_t num_nodes)
        # --- 1. Define Types & Params ---
        
        # Basic Types
        int_type = HLSType(HLSBasicType.INT)
        bool_type = HLSType(HLSBasicType.BOOL)
        node_id_type = HLSType(HLSBasicType.NODE_ID)
        
        # Special ap_uint types
        cache_idx_elem_type = HLSType(basic_type=HLSBasicType.AP_UINT, 
                                      width="26")
        valid_mask_type = HLSType(basic_type=HLSBasicType.AP_UINT, width="PE_NUM") # Use string for define
        offset_type = HLSType(basic_type=HLSBasicType.AP_UINT, width=4)
        
        # Param 1: hls::stream<node_id_burst_t> &src_id_burst_stream
        node_id_array_pe_type = HLSType(HLSBasicType.ARRAY, sub_types=[node_id_type], array_dims=["PE_NUM"])
        node_id_burst_t_type = HLSType(basic_type=HLSBasicType.STRUCT,
                                       struct_name="node_id_burst_t",
                                       struct_prop_names=["data"],
                                       sub_types=[node_id_array_pe_type])
        if node_id_burst_t_type.name not in self.struct_definitions:
            self.struct_definitions[node_id_burst_t_type.name] = (node_id_burst_t_type, node_id_burst_t_type.struct_prop_names)

        src_id_burst_stream_type = HLSType(HLSBasicType.STREAM, sub_types=[node_id_burst_t_type])
        src_id_burst_stream = HLSVar(var_name="src_id_burst_stream", var_type=src_id_burst_stream_type)
        
        # Param 2: hls::stream<distance_req_pack_t> &distance_req_pack_stream
        # Note: distance_req_pack_t uses node_id_t idx[PE_NUM] which is node_id_array_pe_type
        distance_idx_array_type = HLSType(HLSBasicType.ARRAY, sub_types=[HLSType(HLSBasicType.AP_UINT,width=26)], array_dims=["PE_NUM"])
        distance_req_pack_t_type = HLSType(basic_type=HLSBasicType.STRUCT,
                                           struct_name="distance_req_pack_t",
                                           struct_prop_names=["idx", "offset", "end_flag"],
                                           sub_types=[distance_idx_array_type, offset_type, bool_type])
        
        if distance_req_pack_t_type.name not in self.struct_definitions:
            self.struct_definitions[distance_req_pack_t_type.name] = (distance_req_pack_t_type, distance_req_pack_t_type.struct_prop_names)

        distance_req_pack_stream_type = HLSType(HLSBasicType.STREAM, sub_types=[distance_req_pack_t_type])
        distance_req_pack_stream = HLSVar(var_name="distance_req_pack_stream", var_type=distance_req_pack_stream_type)
        
        # Param 3: int32_t num_nodes
        num_nodes = HLSVar(var_name="num_nodes", var_type=int_type)
        
        params.extend([src_id_burst_stream, distance_req_pack_stream, num_nodes])
        dist_req_packer_func.params = params
        
        # --- 2. Function Body ---
        code_lines: List[HLSCodeLine] = []
        
        # const int max_node_burst_idx = (num_nodes + PE_NUM - 1) / PE_NUM;
        code_lines.append(CodeVarDecl(var_name="max_node_burst_idx", var_type=int_type, init_val="(num_nodes + PE_NUM - 1) / PE_NUM", const=True))
        max_node_burst_idx_var = HLSVar(var_name="max_node_burst_idx", var_type=int_type)
        
        # ap_uint<...> last_idx_max = 0;
        code_lines.append(CodeVarDecl(var_name="last_idx_max", var_type=cache_idx_elem_type, init_val="0", const=False))
        code_lines.append(CodeOther(text=""))

        last_idx_max_var = HLSVar(var_name="last_idx_max", var_type=cache_idx_elem_type)
        
        # for (int32_t node_burst_idx = 0; ...
        for_loop_1_codes: List[HLSCodeLine] = []
        # (for_loop_1 object created and added to code_lines at the end)
        
        # --- Inside for(node_burst_idx) ---
        # #pragma HLS PIPELINE II = 1
        for_loop_1_codes.append(CodePragma(content="PIPELINE II = 1"))
        # node_id_burst_t node_id_burst = src_id_burst_stream.read();
        for_loop_1_codes.append(CodeVarDecl(var_name="node_id_burst", var_type=node_id_burst_t_type, init_val="src_id_burst_stream.read()"))
        
        for_loop_1_codes.append(CodeOther(text=""))
        # ap_uint<26> cache_idx[PE_NUM];
        cache_idx_type = HLSType(HLSBasicType.ARRAY, sub_types=[cache_idx_elem_type], array_dims=["PE_NUM"])
        for_loop_1_codes.append(CodeVarDecl(var_name="cache_idx", var_type=cache_idx_type))
        
        # #pragma HLS ARRAY_PARTITION variable = cache_idx complete dim = 0
        for_loop_1_codes.append(CodePragma(content="ARRAY_PARTITION variable = cache_idx complete dim = 0"))
        for_loop_1_codes.append(CodeOther(text=""))
        
        
        # --- Build for(pe_idx) 1 ---
        for_loop_2_codes: List[HLSCodeLine] = []
        # #pragma HLS UNROLL
        for_loop_2_codes.append(CodePragma(content="UNROLL"))
        # cache_idx[pe_idx] = node_id_burst.data[pe_idx] >> LOG_DIST_PER_WORD;
        cache_idx_pe_idx_var = HLSVar(var_name="cache_idx[pe_idx]", var_type=cache_idx_elem_type)
        assign_expr_1 = HLSExpr(HLSExprT.CONST, "node_id_burst.data[pe_idx].range(30,0) >> LOG_DIST_PER_WORD")
        
        for_loop_1_codes.append(CodeOther(text=""))

        for_loop_2_codes.append(CodeAssign(var=cache_idx_pe_idx_var, expr=assign_expr_1))
        # Create for loop 2
        for_loop_2 = CodeFor(codes=for_loop_2_codes, 
                             iter_limit="PE_NUM", 
                             iter_cmp="<", 
                             iter_name="pe_idx", 
                             iter_start="0", 
                             iter_step="pe_idx++", 
                             iter_val_type=int_type)
        for_loop_1_codes.append(for_loop_2)
        # --- End for(pe_idx) 1 ---
        
        # ap_uint<...> cache_idx_diffs[PE_NUM];
        cache_idx_diffs_type = HLSType(HLSBasicType.ARRAY, sub_types=[cache_idx_elem_type], array_dims=["PE_NUM"])
        
        for_loop_1_codes.append(CodeOther(text=""))
        for_loop_1_codes.append(CodeVarDecl(var_name="cache_idx_diffs", var_type=cache_idx_diffs_type))
        # #pragma HLS ARRAY_PARTITION variable = cache_idx_diffs complete dim = 0
        for_loop_1_codes.append(CodePragma(content="ARRAY_PARTITION variable = cache_idx_diffs complete dim = 0"))
        for_loop_1_codes.append(CodeOther(text=""))
    
        
        # --- Build for(pe_idx) 2 ---
        for_loop_3_codes: List[HLSCodeLine] = []
        # #pragma HLS UNROLL
        for_loop_3_codes.append(CodePragma(content="UNROLL"))
        # cache_idx_diffs[pe_idx] = cache_idx[pe_idx] - last_idx_max;
        cache_idx_diffs_pe_idx_var = HLSVar(var_name="cache_idx_diffs[pe_idx]", var_type=cache_idx_elem_type)
        assign_expr_2 = HLSExpr(HLSExprT.CONST, "cache_idx[pe_idx] - last_idx_max")
        for_loop_3_codes.append(CodeAssign(var=cache_idx_diffs_pe_idx_var, expr=assign_expr_2))
        # Create for loop 3
        for_loop_3 = CodeFor(codes=for_loop_3_codes, 
                             iter_limit="PE_NUM", 
                             iter_cmp="<", 
                             iter_name="pe_idx", 
                             iter_start="0", 
                             iter_step="pe_idx++", 
                             iter_val_type=int_type)

        for_loop_1_codes.append(for_loop_3)
        for_loop_1_codes.append(CodeOther(text=""))
        # --- End for(pe_idx) 2 ---
        
        # --- Build IF_1 (cache_idx_diffs[PE_NUM - 1]) ---
        if_1_codes: List[HLSCodeLine] = []
        if_1_codes.append(CodeComment(text="if not all diffs are zero, send a req_pack"))
        if_expr_1 = HLSExpr(HLSExprT.CONST, "cache_idx_diffs[PE_NUM - 1]")
        # (This IF_1 has no else block)
        
        # --- Build IF_1 Contents ---
        # ap_uint<PE_NUM> valid_mask;
        if_1_codes.append(CodeVarDecl(var_name="valid_mask", var_type=valid_mask_type))
        
        # --- Build for(pe_idx) 3 ---
        for_loop_4_codes: List[HLSCodeLine] = []
        # #pragma HLS UNROLL
        for_loop_4_codes.append(CodePragma(content="UNROLL"))
        
        # --- Build IF_2 (cache_idx_diffs[pe_idx] == 0) ---
        if_2_codes: List[HLSCodeLine] = []
        else_2_codes: List[HLSCodeLine] = []
        if_expr_2 = HLSExpr(HLSExprT.CONST, "cache_idx_diffs[pe_idx] == 0")
        # Build IF_2 Contents
        valid_mask_pe_idx_var = HLSVar(var_name="valid_mask[pe_idx].range(pe_idx, pe_idx)", var_type=HLSType(HLSBasicType.AP_UINT, width=1)) # Single bit assignment
        
        assign_expr_3 = HLSExpr(HLSExprT.CONST, 1)
        if_2_codes.append(CodeAssign(var=valid_mask_pe_idx_var, expr=assign_expr_3))
        # Build ELSE_2 Contents
        assign_expr_4 = HLSExpr(HLSExprT.CONST, 0)
        else_2_codes.append(CodeAssign(var=valid_mask_pe_idx_var, expr=assign_expr_4))
        # Create IF_2
        if_2 = CodeIf(expr=if_expr_2, if_codes=if_2_codes, else_codes=else_2_codes)
        for_loop_4_codes.append(if_2)
        # --- End IF_2 ---
        
        # Create for loop 4
        for_loop_4 = CodeFor(codes=for_loop_4_codes, 
                             iter_limit="PE_NUM", 
                             iter_cmp="<", 
                             iter_name="pe_idx", 
                             iter_start="0", 
                             iter_step="pe_idx++", 
                             iter_val_type=int_type)
        if_1_codes.append(for_loop_4)
        # --- End for(pe_idx) 3 ---
        if_1_codes.append(CodeOther(text=""))
        # ap_uint<4> num_unread = count_end_ones(valid_mask);
        if_1_codes.append(CodeVarDecl(var_name="num_unread", var_type=offset_type, init_val="count_end_ones(valid_mask)"))
        num_unread_var = HLSVar(var_name="num_unread", var_type=offset_type)
        if_1_codes.append(CodeOther(text=""))

        # distance_req_pack_t req_pack;
        if_1_codes.append(CodeVarDecl(var_name="req_pack", var_type=distance_req_pack_t_type))
        req_pack_var = HLSVar(var_name="req_pack", var_type=distance_req_pack_t_type)
        
        # req_pack.offset = num_unread;
        req_pack_offset_var = HLSVar(var_name="req_pack.offset", var_type=offset_type)
        if_1_codes.append(CodeAssign(var=req_pack_offset_var, expr=HLSExpr(HLSExprT.VAR, num_unread_var)))
        
        # req_pack.end_flag = false;
        req_pack_end_flag_var = HLSVar(var_name="req_pack.end_flag", var_type=bool_type)
        assign_expr_5 = HLSExpr(HLSExprT.CONST, False)
        if_1_codes.append(CodeAssign(var=req_pack_end_flag_var, expr=assign_expr_5))
        if_1_codes.append(CodeOther(text=""))

        # --- Build for(pe_idx) 4 ---
        for_loop_5_codes: List[HLSCodeLine] = []
        # #pragma HLS UNROLL
        for_loop_5_codes.append(CodePragma(content="UNROLL"))
        # req_pack.idx[pe_idx] = cache_idx[pe_idx];
        req_pack_idx_pe_idx_var = HLSVar(var_name="req_pack.idx[pe_idx]", var_type=node_id_type) # Assuming idx is node_id_t array
        assign_expr_6 = HLSExpr(HLSExprT.CONST, "cache_idx[pe_idx]")
        for_loop_5_codes.append(CodeAssign(var=req_pack_idx_pe_idx_var, expr=assign_expr_6))
        # Create for loop 5
        for_loop_5 = CodeFor(codes=for_loop_5_codes, 
                             iter_limit="PE_NUM", 
                             iter_cmp="<", 
                             iter_name="pe_idx", 
                             iter_start="0", 
                             iter_step="pe_idx++", 
                             iter_val_type=int_type)
        if_1_codes.append(for_loop_5)
        # --- End for(pe_idx) 4 ---
        
        # distance_req_pack_stream.write(req_pack);
        if_1_codes.append(CodeWriteStream(stream_var=distance_req_pack_stream, in_expr=req_pack_var))
        
        # Create IF_1 statement (no else)
        if_1 = CodeIf(expr=if_expr_1, if_codes=if_1_codes)
        for_loop_1_codes.append(if_1)
        # --- End IF_1 ---
        
        # last_idx_max = cache_idx[PE_NUM - 1];
        assign_expr_7 = HLSExpr(HLSExprT.CONST, "cache_idx[PE_NUM - 1]")
        for_loop_1_codes.append(CodeAssign(var=last_idx_max_var, expr=assign_expr_7))
        
        # --- Create For Loop 1 ---
        for_loop_1 = CodeFor(codes=for_loop_1_codes, 
                             iter_limit=max_node_burst_idx_var, 
                             iter_cmp="<", 
                             iter_name="node_burst_idx", 
                             iter_start="0", 
                             iter_step="node_burst_idx += 1", 
                             iter_val_type=int_type)
        code_lines.append(for_loop_1)
        # --- End For Loop 1 ---
        
        code_lines.append(CodeOther(text="{"))
        # distance_req_pack_t end_req_pack;
        code_lines.append(CodeVarDecl(var_name="end_req_pack", var_type=distance_req_pack_t_type))
        end_req_pack_var = HLSVar(var_name="end_req_pack", var_type=distance_req_pack_t_type)
        
        # end_req_pack.end_flag = true;
        end_req_pack_end_flag_var = HLSVar(var_name="end_req_pack.end_flag", var_type=bool_type)
        assign_expr_8 = HLSExpr(HLSExprT.CONST, True)
        code_lines.append(CodeAssign(var=end_req_pack_end_flag_var, expr=assign_expr_8))
        
        # end_req_pack.offset = 8;
        end_req_pack_offset_var = HLSVar(var_name="end_req_pack.offset", var_type=offset_type)
        assign_expr_9 = HLSExpr(HLSExprT.CONST, 7)
        code_lines.append(CodeAssign(var=end_req_pack_offset_var, expr=assign_expr_9))
        
        # distance_req_pack_stream.write(end_req_pack);
        code_lines.append(CodeWriteStream(stream_var=distance_req_pack_stream, in_expr=end_req_pack_var))
        code_lines.append(CodeOther(text="}"))

        # --- 3. Finalize ---
        dist_req_packer_func.codes = code_lines


        self.big_scatter_funcs.append(dist_req_packer_func)
        self.big_top_dataflow_funcs.append(dist_req_packer_func)
        # cacheline_req_sender(
        # hls::stream<distance_req_pack_t> &distance_req_pack_stream,
        # hls::stream<cacheline_request_pkt_t> &cacheline_req_stream)

        cacheline_req_sender_func = HLSFunction(name="cacheline_req_sender", comp=comp) # 假设 comp 存在
        params = []

        # --- 1. 定义类型和参数 (根据 graphyflow_big.h) ---

        # 基础类型
        int_type = HLSType(HLSBasicType.INT)       # int32_t
        bool_type = HLSType(HLSBasicType.BOOL)
        ap_uint26_type = HLSType(basic_type=HLSBasicType.AP_UINT, width=26)
        ap_uint8_type = HLSType(basic_type=HLSBasicType.AP_UINT, width=8)
        ap_uint4_type = HLSType(basic_type=HLSBasicType.AP_UINT, width=4)

        # --- 结构体定义 (严格按照 graphyflow_big.h) ---

        # struct distance_req_pack_t
        ap_uint26_array_type = HLSType(HLSBasicType.ARRAY, sub_types=[ap_uint26_type], array_dims=["PE_NUM"])
        distance_req_pack_t_type = HLSType(basic_type=HLSBasicType.STRUCT,
                                           struct_name="distance_req_pack_t",
                                           struct_prop_names=["idx", "offset", "end_flag"],
                                           sub_types=[ap_uint26_array_type, ap_uint4_type, bool_type])
        if distance_req_pack_t_type.name not in self.struct_definitions:
            self.struct_definitions[distance_req_pack_t_type.name] = (distance_req_pack_t_type, distance_req_pack_t_type.struct_prop_names)

        # struct cacheline_req_t
        cacheline_req_t_type = HLSType(basic_type=HLSBasicType.STRUCT,
                                       struct_name="cacheline_req_t",
                                       struct_prop_names=["idx", "dst", "end_flag"],
                                       sub_types=[ap_uint26_type, ap_uint8_type, bool_type])
        if cacheline_req_t_type.name not in self.struct_definitions:
            self.struct_definitions[cacheline_req_t_type.name] = (cacheline_req_t_type, cacheline_req_t_type.struct_prop_names)

        # --- Stream 类型 ---
        stream_dist_req_pack_type = HLSType(HLSBasicType.STREAM, sub_types=[distance_req_pack_t_type])
        stream_cacheline_req_type = HLSType(HLSBasicType.STREAM, sub_types=[cacheline_req_t_type])

        # --- 参数变量 ---
        # Param 1: hls::stream<distance_req_pack_t> &distance_req_pack_stream
        distance_req_pack_stream_var = HLSVar(var_name="distance_req_pack_stream", var_type=stream_dist_req_pack_type)
        # Param 2: hls::stream<cacheline_req_t> &cacheline_req_stream
        cacheline_req_stream_var = HLSVar(var_name="cacheline_req_stream", var_type=stream_cacheline_req_type)
        # Param 3: int32_t memory_offset
        memory_offset_var = HLSVar(var_name="memory_offset", var_type=int_type)

        params.extend([distance_req_pack_stream_var, cacheline_req_stream_var, memory_offset_var])
        cacheline_req_sender_func.params = params

        # --- 2. 函数体 ---
        code_lines: List[HLSCodeLine] = []

        # --- 初始请求块 ---
        block_1_codes: List[HLSCodeLine] = []
        # cacheline_req_t cache_req;
        block_1_codes.append(CodeVarDecl(var_name="cache_req", var_type=cacheline_req_t_type))
        cache_req_var = HLSVar(var_name="cache_req", var_type=cacheline_req_t_type)
        # cache_req.end_flag = false;
        cache_req_end_flag_var = HLSVar(var_name="cache_req.end_flag", var_type=bool_type)
        block_1_codes.append(CodeAssign(var=cache_req_end_flag_var, expr=HLSExpr(HLSExprT.CONST, False)))
        # cache_req.idx = memory_offset;
        cache_req_idx_var = HLSVar(var_name="cache_req.idx", var_type=ap_uint26_type)
        block_1_codes.append(CodeAssign(var=cache_req_idx_var, expr=HLSExpr(HLSExprT.VAR, memory_offset_var)))
        # cacheline_req_stream.write(cache_req);
        block_1_codes.append(CodeWriteStream(stream_var=cacheline_req_stream_var, in_expr=cache_req_var))
        # (构建 block_1)
        code_lines.append(CodeBlock(codes=block_1_codes))
        code_lines.append(CodeOther(text=""))

        # ap_uint<26> cacheline_idx[PE_NUM];
        cacheline_idx_type = HLSType(HLSBasicType.ARRAY, sub_types=[ap_uint26_type], array_dims=["PE_NUM"])
        code_lines.append(CodeVarDecl(var_name="cacheline_idx", var_type=cacheline_idx_type))
        cacheline_idx_var = HLSVar(var_name="cacheline_idx", var_type=cacheline_idx_type)
        # #pragma HLS ARRAY_PARTITION variable = cacheline_idx complete dim = 0
        code_lines.append(CodePragma(content="ARRAY_PARTITION variable = cacheline_idx complete dim = 0"))
        code_lines.append(CodeOther(text=""))

        # LOOP_SEND_CACHE_REQ:
        code_lines.append(CodeOther(text="LOOP_SEND_CACHE_REQ:"))
        # while (true)
        while_1_codes: List[HLSCodeLine] = []
        while_1_expr = HLSExpr(HLSExprT.CONST, True)

        # #pragma HLS PIPELINE II = 1
        while_1_codes.append(CodePragma(content="PIPELINE II = 1"))
        # #pragma HLS dependence variable = cacheline_idx inter false
        while_1_codes.append(CodePragma(content="dependence variable = cacheline_idx inter false"))
        while_1_codes.append(CodeOther(text=""))

        # distance_req_pack_t req_pack = distance_req_pack_stream.read();
        while_1_codes.append(CodeVarDecl(var_name="req_pack", var_type=distance_req_pack_t_type))
        req_pack_var = HLSVar(var_name="req_pack", var_type=distance_req_pack_t_type)
        read_expr = HLSExpr(HLSExprT.STREAM_READ, None, operands=[HLSExpr(HLSExprT.VAR, distance_req_pack_stream_var)])
        while_1_codes.append(CodeAssign(var=req_pack_var, expr=read_expr))

        # // #pragma HLS ARRAY_PARTITION variable = req_pack.idx complete dim = 0
        while_1_codes.append(CodeComment(text="#pragma HLS ARRAY_PARTITION variable = req_pack.idx complete dim = 0"))
        # for (int32_t pe_idx = 0; pe_idx < PE_NUM; pe_idx++)
        for_1_codes: List[HLSCodeLine] = []
        # #pragma HLS UNROLL
        for_1_codes.append(CodePragma(content="UNROLL"))
        # cacheline_idx[pe_idx] = req_pack.idx[pe_idx];
        lhs_var = HLSVar(var_name="cacheline_idx[pe_idx]", var_type=ap_uint26_type)
        rhs_expr = HLSExpr(HLSExprT.CONST, "req_pack.idx[pe_idx]")
        for_1_codes.append(CodeAssign(var=lhs_var, expr=rhs_expr))
        # (构建 for_1)
        while_1_codes.append(CodeFor(codes=for_1_codes, iter_limit="PE_NUM", iter_name="pe_idx", iter_val_type=int_type))
        while_1_codes.append(CodeOther(text=""))

        # --- 内部请求块 ---
        block_2_codes: List[HLSCodeLine] = []
        # LOOP_SEND_CACHE_REQ_INNER:
        block_2_codes.append(CodeOther(text="LOOP_SEND_CACHE_REQ_INNER:"))
        # for (ap_uint<4> i = req_pack.offset; i < PE_NUM; i++)
        for_2_codes: List[HLSCodeLine] = []
        # #pragma HLS PIPELINE II = 1 rewind
        for_2_codes.append(CodePragma(content="PIPELINE II = 1 rewind"))
        # #pragma HLS unroll factor = 1
        for_2_codes.append(CodePragma(content="unroll factor = 1"))
        # cacheline_req_t cache_req;
        for_2_codes.append(CodeVarDecl(var_name="cache_req", var_type=cacheline_req_t_type))
        cache_req_var_inner = HLSVar(var_name="cache_req", var_type=cacheline_req_t_type)
        # cache_req.idx = cacheline_idx[i] + memory_offset;
        cache_req_idx_var_inner = HLSVar(var_name="cache_req.idx", var_type=ap_uint26_type)
        rhs_expr_2 = HLSExpr(HLSExprT.BINOP, dfir.BinOp.ADD, 
                             operands=[HLSExpr(HLSExprT.CONST, "cacheline_idx[i]"), 
                                       HLSExpr(HLSExprT.VAR, memory_offset_var)])
        for_2_codes.append(CodeAssign(var=cache_req_idx_var_inner, expr=rhs_expr_2))
        # cache_req.dst = i;
        cache_req_dst_var_inner = HLSVar(var_name="cache_req.dst", var_type=ap_uint8_type)
        for_2_codes.append(CodeAssign(var=cache_req_dst_var_inner, expr=HLSExpr(HLSExprT.CONST, "i")))
        # cache_req.end_flag = req_pack.end_flag;
        cache_req_end_flag_var_inner = HLSVar(var_name="cache_req.end_flag", var_type=bool_type)
        rhs_expr_3 = HLSExpr(HLSExprT.UOP, (dfir.UnaryOp.GET_ATTR, "end_flag"), 
                             operands=[HLSExpr(HLSExprT.VAR, req_pack_var)])
        for_2_codes.append(CodeAssign(var=cache_req_end_flag_var_inner, expr=rhs_expr_3))
        # cacheline_req_stream.write(cache_req);
        for_2_codes.append(CodeWriteStream(stream_var=cacheline_req_stream_var, in_expr=cache_req_var_inner))
        # // printf(...)
        for_2_codes.append(CodeComment(text="printf(\"Sent cacheline req for idx %d to PE %d\\n\","))
        for_2_codes.append(CodeComment(text="// (int)cache_req.idx, (int)cache_req.target_pe); fflush(NULL);"))
        # (构建 for_2)
        block_2_codes.append(CodeFor(codes=for_2_codes, iter_limit="PE_NUM", iter_cmp="<", iter_name="i", iter_start="req_pack.offset", iter_val_type=ap_uint4_type))
        # (构建 block_2)
        while_1_codes.append(CodeBlock(codes=block_2_codes))
        while_1_codes.append(CodeOther(text=""))

        # if (req_pack.end_flag)
        if_3_codes: List[HLSCodeLine] = []
        if_3_expr = HLSExpr(HLSExprT.UOP, (dfir.UnaryOp.GET_ATTR, "end_flag"), operands=[HLSExpr(HLSExprT.VAR, req_pack_var)])
        # break;
        if_3_codes.append(CodeBreak())
        # (构建 if_3)
        while_1_codes.append(CodeIf(expr=if_3_expr, if_codes=if_3_codes))

        # (构建 while_1)
        code_lines.append(CodeWhile(codes=while_1_codes, iter_expr=while_1_expr))
        code_lines.append(CodeOther(text=""))

        # // cache_req.last = true;
        code_lines.append(CodeComment(text="cache_req.last = true;"))
        # // cacheline_req_stream.write(cache_req);
        code_lines.append(CodeComment(text="cacheline_req_stream.write(cache_req);"))

        # --- 3. Finalize ---
        cacheline_req_sender_func.codes = code_lines
        
        self.big_scatter_funcs.append(cacheline_req_sender_func)
        self.big_top_dataflow_funcs.append(cacheline_req_sender_func)
        #     node_prop_resp_receiver(
        # hls::stream<cacheline_response_pkt_t> &cacheline_resp_stream,
        # hls::stream<bus_word_t> (&cacheline_streams)[PE_NUM])
        node_prop_resp_receiver_func = HLSFunction(name="node_prop_resp_receiver", comp=comp)
        params = []

        # --- 1. Define Types & Params ---

        # Define sub-types for cacheline_response_pkt_t
        bus_word_t_type = HLSType(basic_type=HLSBasicType.BUS_WORD_T)
        dest_type = HLSType(basic_type=HLSBasicType.AP_UINT, width=8)
        bool_type = HLSType(basic_type=HLSBasicType.BOOL)

        # Define cacheline_response_pkt_t

        # Param 1: cacheline_resp_stream
        cacheline_resp_stream_type = HLSType(HLSBasicType.STREAM, sub_types=[HLSType(basic_type=HLSBasicType.CACHELINE_RESPONSE_PKT_T)])
        cacheline_resp_stream = HLSVar(var_name="cacheline_resp_stream", var_type=cacheline_resp_stream_type)

        # Param 2: cacheline_streams
        bus_word_stream_type = HLSType(HLSBasicType.STREAM, sub_types=[bus_word_t_type])
        cacheline_streams_type = HLSType(HLSBasicType.ARRAY, sub_types=[bus_word_stream_type], array_dims=["PE_NUM"])
        cacheline_streams = HLSVar(var_name="cacheline_streams", var_type=cacheline_streams_type)

        params.extend([cacheline_resp_stream, cacheline_streams])
        node_prop_resp_receiver_func.params = params

        # --- 2. Function Body ---
        code_lines: List[HLSCodeLine] = []

        # cacheline_response_pkt_t cache_resp = cacheline_resp_stream.read();
        code_lines.append(CodeVarDecl(var_name="cache_resp", var_type=HLSType(basic_type=HLSBasicType.CACHELINE_RESPONSE_PKT_T), init_val="cacheline_resp_stream.read()"))
        cache_resp_var = HLSVar(var_name="cache_resp", var_type=HLSType(basic_type=HLSBasicType.CACHELINE_RESPONSE_PKT_T))

        # bus_word_t first_line = cache_resp.data;
        code_lines.append(CodeVarDecl(var_name="first_line", var_type=bus_word_t_type, init_val="cache_resp.data"))
        first_line_var = HLSVar(var_name="first_line", var_type=bus_word_t_type)

        # for (int32_t pe_idx = 0; pe_idx < PE_NUM; pe_idx++) {
        for_loop_1_codes: List[HLSCodeLine] = []
        for_loop_1 = CodeFor(codes=for_loop_1_codes,
                             iter_limit="PE_NUM",
                             iter_cmp="<",
                             iter_name="pe_idx",
                             iter_start="0",
                             iter_step="pe_idx++",
                             iter_val_type=HLSType(HLSBasicType.INT))
        code_lines.append(for_loop_1)

        # --- Inside for(pe_idx) ---
        # #pragma HLS UNROLL
        for_loop_1_codes.append(CodePragma(content="UNROLL"))

        # cacheline_streams[pe_idx].write(first_line);
        cacheline_streams_pe_idx_var = HLSVar(var_name="cacheline_streams[pe_idx]", var_type=bus_word_stream_type)
        for_loop_1_codes.append(CodeWriteStream(stream_var=cacheline_streams_pe_idx_var, in_expr=first_line_var))
        # } (end for_loop_1)

        # (empty line)
        code_lines.append(CodeOther(text=""))

        # LOOP_RECEIVE_CACHE_RESP:
        code_lines.append(CodeComment(text="LOOP_RECEIVE_CACHE_RESP:"))

        # while (true) {
        while_loop_codes: List[HLSCodeLine] = []
        while_true_expr = HLSExpr(HLSExprT.CONST, True)
        while_loop = CodeWhile(codes=while_loop_codes, iter_expr=while_true_expr)
        code_lines.append(while_loop)

        # --- Inside while(true) ---
        # #pragma HLS PIPELINE II = 1
        while_loop_codes.append(CodePragma(content="PIPELINE II = 1"))

        # if (cacheline_resp_stream.read_nb(cache_resp)) {
        if_1_codes: List[HLSCodeLine] = []
        if_expr_1 = HLSExpr(HLSExprT.CONST, "cacheline_resp_stream.read_nb(cache_resp)")
        if_1 = CodeIf(expr=if_expr_1, if_codes=if_1_codes)
        while_loop_codes.append(if_1)

        # --- Inside if(read_nb) ---
        # if (cache_resp.last) {
        if_2_codes: List[HLSCodeLine] = []
        if_expr_2 = HLSExpr(HLSExprT.CONST, "cache_resp.last")
        if_2 = CodeIf(expr=if_expr_2, if_codes=if_2_codes)
        if_1_codes.append(if_2)

        # --- Inside if(cache_resp.last) ---
        # break;
        if_2_codes.append(CodeBreak())
        # } (end if_2)

        # bus_word_t resp_line = cache_resp.data;
        if_1_codes.append(CodeVarDecl(var_name="resp_line", var_type=bus_word_t_type, init_val="cache_resp.data"))
        resp_line_var = HLSVar(var_name="resp_line", var_type=bus_word_t_type)

        # ap_uint<8> target_pe = cache_resp.dest;
        if_1_codes.append(CodeVarDecl(var_name="target_pe", var_type=dest_type, init_val="cache_resp.dest"))
        target_pe_var = HLSVar(var_name="target_pe", var_type=dest_type)

        # cacheline_streams[target_pe].write(resp_line);
        cacheline_streams_target_pe_var = HLSVar(var_name="cacheline_streams[target_pe]", var_type=bus_word_stream_type)
        if_1_codes.append(CodeWriteStream(stream_var=cacheline_streams_target_pe_var, in_expr=resp_line_var))
        # } (end if_1)
        # } (end while_loop)

        # --- 3. Finalize ---
        node_prop_resp_receiver_func.codes = code_lines
        code_lines.append(CodeComment("End of node_prop_resp_receiver Function"))
        self.big_scatter_funcs.append(node_prop_resp_receiver_func)
        self.big_top_dataflow_funcs.append(node_prop_resp_receiver_func)




    def _translate_reduce_op(self, comp: dfir.Component):
        

        demux_1_func = HLSFunction(name="demux_1", comp=comp) # 假设 comp 存在
        params = []

        # 基础类型
        uint_type = HLSType(HLSBasicType.UINT)      # uint32_t
        ap_uint20_type = HLSType(basic_type=HLSBasicType.AP_UINT, width=20)
        ap_fixed_pod_t_type = HLSType(HLSBasicType.AP_FIXED_POD)
        bool_type = HLSType(HLSBasicType.BOOL)

        # --- 结构体定义 (来自 graphyflow_big.h) ---

        # struct update_t { ap_uint<20> node_id; ap_fixed_pod_t prop; bool end_flag; }
        update_t_type = HLSType(basic_type=HLSBasicType.STRUCT,
                                struct_name="update_t_big",
                                struct_prop_names=["node_id", "prop", "end_flag"],
                                sub_types=[ap_uint20_type, ap_fixed_pod_t_type, bool_type])
        if update_t_type.name not in self.struct_definitions:
            self.struct_definitions[update_t_type.name] = (update_t_type, update_t_type.struct_prop_names)

        # struct update_tuple_t { update_t data[PE_NUM]; }
        update_t_array_type = HLSType(HLSBasicType.ARRAY, sub_types=[update_t_type], array_dims=["PE_NUM"])
        update_tuple_t_type = HLSType(basic_type=HLSBasicType.STRUCT,
                                      struct_name="update_tuple_t_big",
                                      struct_prop_names=["data"],
                                      sub_types=[update_t_array_type])
        if update_tuple_t_type.name not in self.struct_definitions:
            self.struct_definitions[update_tuple_t_type.name] = (update_tuple_t_type, update_tuple_t_type.struct_prop_names)

        # --- Stream 类型 ---
        in_batch_stream_type = HLSType(HLSBasicType.STREAM, sub_types=[update_tuple_t_type])
        out_streams_elem_type = HLSType(HLSBasicType.STREAM, sub_types=[update_t_type])

        # --- 参数变量 ---
        # Param 1: hls::stream<update_tuple_t> &in_batch_stream
        in_batch_stream_var = HLSVar(var_name="in_batch_stream", var_type=in_batch_stream_type)

        # Param 2: hls::stream<update_t> (&out_streams)[8]
        # (注意: C++ 签名使用 [8], 而不是 [PE_NUM])
        out_streams_type = HLSType(HLSBasicType.ARRAY, sub_types=[out_streams_elem_type], array_dims=[8])
        out_streams_var = HLSVar(var_name="out_streams", var_type=out_streams_type)

        # Param 3: uint32_t edge_num
        edge_num_var = HLSVar(var_name="edge_num", var_type=uint_type)

        params.extend([in_batch_stream_var, out_streams_var, edge_num_var])
        demux_1_func.params = params

        # --- 2. 函数体 ---
        code_lines: List[HLSCodeLine] = []

        # const uint32_t scatter_size = edge_num / PE_NUM;
        code_lines.append(CodeVarDecl(var_name="scatter_size", var_type=uint_type, init_val="(edge_num / PE_NUM)", const=True))
        scatter_size_var = HLSVar(var_name="scatter_size", var_type=uint_type)
        code_lines.append(CodeOther(text=""))


        # for (uint32_t batch_idx = 0; batch_idx < scatter_size; batch_idx++)
        for_1_codes: List[HLSCodeLine] = []
        # #pragma HLS PIPELINE II = 1
        for_1_codes.append(CodePragma(content="PIPELINE II = 1"))
        # update_tuple_t in_batch;
        for_1_codes.append(CodeVarDecl(var_name="in_batch", var_type=update_tuple_t_type))
        in_batch_var = HLSVar(var_name="in_batch", var_type=update_tuple_t_type)
        # in_batch = in_batch_stream.read();
        read_expr = HLSExpr(HLSExprT.STREAM_READ, None, operands=[HLSExpr(HLSExprT.VAR, in_batch_stream_var)])
        for_1_codes.append(CodeAssign(var=in_batch_var, expr=read_expr))
        for_1_codes.append(CodeOther(text=""))

        # for (uint32_t i = 0; i < PE_NUM; i++)
        for_2_codes: List[HLSCodeLine] = []
        # #pragma HLS UNROLL
        for_2_codes.append(CodePragma(content="UNROLL"))
        # if (in_batch.data[i].node_id.range(19, 19) == 0)
        if_1_codes: List[HLSCodeLine] = []
        if_1_expr = HLSExpr(HLSExprT.CONST, "in_batch.data[i].node_id.range(19, 19) == 0")
        # out_streams[i].write(in_batch.data[i]);
        out_stream_i_var = HLSVar(var_name="out_streams[i]", var_type=out_streams_elem_type)
        in_batch_data_i_expr = HLSExpr(HLSExprT.CONST, "in_batch.data[i]")
        if_1_codes.append(CodeWriteStream(stream_var=out_stream_i_var, in_expr=in_batch_data_i_expr))
        # (构建 if_1)
        for_2_codes.append(CodeIf(expr=if_1_expr, if_codes=if_1_codes))
        # (构建 for_2)
        for_1_codes.append(CodeFor(codes=for_2_codes, iter_limit="PE_NUM", iter_name="i", iter_val_type=uint_type))
        # (构建 for_1)
        code_lines.append(CodeFor(codes=for_1_codes, iter_limit=scatter_size_var, iter_name="batch_idx", iter_val_type=uint_type))

        # // Propagate end_flag to all output streams
        code_lines.append(CodeComment(text="Propagate end_flag to all output streams"))

        # for (uint32_t i = 0; i < 8; i++)
        for_3_codes: List[HLSCodeLine] = []
        # #pragma HLS UNROLL
        for_3_codes.append(CodePragma(content="UNROLL"))
        # update_t end_wrapper;
        for_3_codes.append(CodeVarDecl(var_name="end_wrapper", var_type=update_t_type))
        end_wrapper_var = HLSVar(var_name="end_wrapper", var_type=update_t_type)
        # end_wrapper.end_flag = true;
        end_wrapper_flag_var = HLSVar(var_name="end_wrapper.end_flag", var_type=bool_type)
        for_3_codes.append(CodeAssign(var=end_wrapper_flag_var, expr=HLSExpr(HLSExprT.CONST, True)))
        # out_streams[i].write(end_wrapper);
        out_stream_i_var_2 = HLSVar(var_name="out_streams[i]", var_type=out_streams_elem_type)
        for_3_codes.append(CodeWriteStream(stream_var=out_stream_i_var_2, in_expr=end_wrapper_var))
        # (构建 for_3)
        code_lines.append(CodeFor(codes=for_3_codes, iter_limit="8", iter_name="i", iter_val_type=uint_type))

        # --- 3. Finalize ---
        demux_1_func.codes = code_lines
        self.big_gather_funcs.append(demux_1_func)
        self.big_top_dataflow_funcs.append(demux_1_func)



        sender_2_func = HLSFunction(name="sender_2", comp=comp) # 假设 comp 存在
        params = []

        # --- 1. 定义类型和参数 ---
        # (基于 C++ 代码对 'end_flag' 的使用, 
        #  我们必须使用 graphyflow_big.h 中的 'update_t' 定义)

        # 基础类型
        int_type = HLSType(HLSBasicType.INT)       # int32_t
        bool_type = HLSType(HLSBasicType.BOOL)
        ap_uint20_type = HLSType(basic_type=HLSBasicType.AP_UINT, width=20)
        ap_fixed_pod_t_type = HLSType(HLSBasicType.AP_FIXED_POD)

        # --- 结构体定义 (来自 graphyflow_big.h) ---

        # struct update_t { ap_uint<20> node_id; ap_fixed_pod_t prop; bool end_flag; }
        update_t_type = HLSType(basic_type=HLSBasicType.STRUCT,
                                struct_name="update_t_big",
                                struct_prop_names=["node_id", "prop", "end_flag"],
                                sub_types=[ap_uint20_type, ap_fixed_pod_t_type, bool_type])
        if update_t_type.name not in self.struct_definitions:
            self.struct_definitions[update_t_type.name] = (update_t_type, update_t_type.struct_prop_names)

        # --- Stream 类型 ---
        stream_type = HLSType(HLSBasicType.STREAM, sub_types=[update_t_type])

        # --- 参数变量 ---
        i_var = HLSVar(var_name="i", var_type=int_type)
        in1_var = HLSVar(var_name="in1", var_type=stream_type)
        in2_var = HLSVar(var_name="in2", var_type=stream_type)
        out1_var = HLSVar(var_name="out1", var_type=stream_type)
        out2_var = HLSVar(var_name="out2", var_type=stream_type)
        out3_var = HLSVar(var_name="out3", var_type=stream_type)
        out4_var = HLSVar(var_name="out4", var_type=stream_type)

        params.extend([i_var, in1_var, in2_var, out1_var, out2_var, out3_var, out4_var])
        sender_2_func.params = params

        # --- 2. 函数体 ---
        code_lines: List[HLSCodeLine] = []

        # 局部变量
        in1_end_flag_var = HLSVar(var_name="in1_end_flag", var_type=bool_type)
        in2_end_flag_var = HLSVar(var_name="in2_end_flag", var_type=bool_type)
        data1_var = HLSVar(var_name="data1", var_type=update_t_type)
        data2_var = HLSVar(var_name="data2", var_type=update_t_type)
        data_var = HLSVar(var_name="data", var_type=update_t_type)
        data_end_flag_var = HLSVar(var_name="data.end_flag", var_type=bool_type) # 用于赋值

        # #pragma HLS function_instantiate variable = i
        code_lines.append(CodePragma(content="function_instantiate variable = i"))

        # bool in1_end_flag = false;
        code_lines.append(CodeVarDecl(var_name="in1_end_flag", var_type=bool_type, init_val="false"))

        # bool in2_end_flag = false;
        code_lines.append(CodeVarDecl(var_name="in2_end_flag", var_type=bool_type, init_val="false"))

        # LOOP_WHILE_23:
        code_lines.append(CodeOther(text="LOOP_WHILE_23:"))

        # while (true) { ... }
        while_codes: List[HLSCodeLine] = []
        while_expr = HLSExpr(HLSExprT.CONST, True)

        # #pragma HLS PIPELINE II = 1
        while_codes.append(CodePragma(content="PIPELINE II = 1"))

        # --- if ((!in1.empty())) { ... } ---
        if_1_expr = HLSExpr(HLSExprT.UOP, dfir.UnaryOp.NOT, 
                            operands=[HLSExpr(HLSExprT.STREAM_EMPTY, None, 
                                              operands=[HLSExpr(HLSExprT.VAR, in1_var)])])
        if_1_codes: List[HLSCodeLine] = []

        # update_t data1;
        if_1_codes.append(CodeVarDecl(var_name="data1", var_type=update_t_type))

        # data1 = in1.read();
        if_1_codes.append(CodeAssign(var=data1_var, 
                                     expr=HLSExpr(HLSExprT.STREAM_READ, None, 
                                                  operands=[HLSExpr(HLSExprT.VAR, in1_var)])))

        # --- if ((!data1.end_flag)) { ... } else { ... } ---
        if_2_expr = HLSExpr(HLSExprT.UOP, dfir.UnaryOp.NOT, 
                            operands=[HLSExpr(HLSExprT.UOP, (dfir.UnaryOp.GET_ATTR, "end_flag"), 
                                              operands=[HLSExpr(HLSExprT.VAR, data1_var)])])
        if_2_codes: List[HLSCodeLine] = []
        else_2_codes: List[HLSCodeLine] = []

        # --- if (((data1.node_id >> i) & 1)) { ... } else { ... } ---
        if_3_expr = HLSExpr(HLSExprT.BINOP, dfir.BinOp.AND, 
                            operands=[
                                HLSExpr(HLSExprT.BINOP, dfir.BinOp.SR, 
                                        operands=[
                                            HLSExpr(HLSExprT.UOP, (dfir.UnaryOp.GET_ATTR, "node_id"), 
                                                    operands=[HLSExpr(HLSExprT.VAR, data1_var)]),
                                            HLSExpr(HLSExprT.VAR, i_var)
                                        ]),
                                HLSExpr(HLSExprT.CONST, 1)
                            ])
        if_3_codes: List[HLSCodeLine] = []
        else_3_codes: List[HLSCodeLine] = []

        # out2.write(data1);
        if_3_codes.append(CodeWriteStream(stream_var=out2_var, in_expr=data1_var))

        # out1.write(data1);
        else_3_codes.append(CodeWriteStream(stream_var=out1_var, in_expr=data1_var))

        # (构建 if_3)
        if_2_codes.append(CodeIf(expr=if_3_expr, if_codes=if_3_codes, else_codes=else_3_codes))

        # in1_end_flag = true;
        else_2_codes.append(CodeAssign(var=in1_end_flag_var, expr=HLSExpr(HLSExprT.CONST, True)))

        # (构建 if_2)
        if_1_codes.append(CodeIf(expr=if_2_expr, if_codes=if_2_codes, else_codes=else_2_codes))

        # (构建 if_1)
        while_codes.append(CodeIf(expr=if_1_expr, if_codes=if_1_codes, else_codes=None))


        # --- if ((!in2.empty())) { ... } ---
        if_4_expr = HLSExpr(HLSExprT.UOP, dfir.UnaryOp.NOT, 
                            operands=[HLSExpr(HLSExprT.STREAM_EMPTY, None, 
                                              operands=[HLSExpr(HLSExprT.VAR, in2_var)])])
        if_4_codes: List[HLSCodeLine] = []

        # update_t data2;
        if_4_codes.append(CodeVarDecl(var_name="data2", var_type=update_t_type))

        # data2 = in2.read();
        if_4_codes.append(CodeAssign(var=data2_var, 
                                     expr=HLSExpr(HLSExprT.STREAM_READ, None, 
                                                  operands=[HLSExpr(HLSExprT.VAR, in2_var)])))

        # --- if ((!data2.end_flag)) { ... } else { ... } ---
        if_5_expr = HLSExpr(HLSExprT.UOP, dfir.UnaryOp.NOT, 
                            operands=[HLSExpr(HLSExprT.UOP, (dfir.UnaryOp.GET_ATTR, "end_flag"), 
                                              operands=[HLSExpr(HLSExprT.VAR, data2_var)])])
        if_5_codes: List[HLSCodeLine] = []
        else_5_codes: List[HLSCodeLine] = []

        # --- if (((data2.node_id >> i) & 1)) { ... } else { ... } ---
        if_6_expr = HLSExpr(HLSExprT.BINOP, dfir.BinOp.AND, 
                            operands=[
                                HLSExpr(HLSExprT.BINOP, dfir.BinOp.SR, 
                                        operands=[
                                            HLSExpr(HLSExprT.UOP, (dfir.UnaryOp.GET_ATTR, "node_id"), 
                                                    operands=[HLSExpr(HLSExprT.VAR, data2_var)]),
                                            HLSExpr(HLSExprT.VAR, i_var)
                                        ]),
                                HLSExpr(HLSExprT.CONST, 1)
                            ])
        if_6_codes: List[HLSCodeLine] = []
        else_6_codes: List[HLSCodeLine] = []

        # out4.write(data2);
        if_6_codes.append(CodeWriteStream(stream_var=out4_var, in_expr=data2_var))

        # out3.write(data2);
        else_6_codes.append(CodeWriteStream(stream_var=out3_var, in_expr=data2_var))

        # (构建 if_6)
        if_5_codes.append(CodeIf(expr=if_6_expr, if_codes=if_6_codes, else_codes=else_6_codes))

        # in2_end_flag = true;
        else_5_codes.append(CodeAssign(var=in2_end_flag_var, expr=HLSExpr(HLSExprT.CONST, True)))

        # (构建 if_5)
        if_4_codes.append(CodeIf(expr=if_5_expr, if_codes=if_5_codes, else_codes=else_5_codes))

        # (构建 if_4)
        while_codes.append(CodeIf(expr=if_4_expr, if_codes=if_4_codes, else_codes=None))


        # --- if ((in1_end_flag & in2_end_flag)) { ... } ---
        if_7_expr = HLSExpr(HLSExprT.BINOP, dfir.BinOp.AND, 
                            operands=[
                                HLSExpr(HLSExprT.VAR, in1_end_flag_var),
                                HLSExpr(HLSExprT.VAR, in2_end_flag_var)
                            ])
        if_7_codes: List[HLSCodeLine] = []

        # update_t data;
        if_7_codes.append(CodeVarDecl(var_name="data", var_type=update_t_type))

        # data.end_flag = true;
        if_7_codes.append(CodeAssign(var=data_end_flag_var, expr=HLSExpr(HLSExprT.CONST, True)))

        # out1.write(data);
        if_7_codes.append(CodeWriteStream(stream_var=out1_var, in_expr=data_var))

        # out2.write(data);
        if_7_codes.append(CodeWriteStream(stream_var=out2_var, in_expr=data_var))

        # out3.write(data);
        if_7_codes.append(CodeWriteStream(stream_var=out3_var, in_expr=data_var))

        # out4.write(data);
        if_7_codes.append(CodeWriteStream(stream_var=out4_var, in_expr=data_var))

        # in1_end_flag = false;
        if_7_codes.append(CodeAssign(var=in1_end_flag_var, expr=HLSExpr(HLSExprT.CONST, False)))

        # in2_end_flag = false;
        if_7_codes.append(CodeAssign(var=in2_end_flag_var, expr=HLSExpr(HLSExprT.CONST, False)))

        # break;
        if_7_codes.append(CodeBreak())

        # (构建 if_7)
        while_codes.append(CodeIf(expr=if_7_expr, if_codes=if_7_codes, else_codes=None))


        # --- 
        # (构建 while)
        while_loop = CodeWhile(codes=while_codes, iter_expr=while_expr)
        code_lines.append(while_loop)

        # (完成函数)
        sender_2_func.codes = code_lines
  
        self.big_gather_funcs.append(sender_2_func)

        receiver_2_func = HLSFunction(name="receiver_2", comp=comp) # 假设 comp 存在
        params = []

        # --- 1. 定义类型和参数 ---
        # (基于 C++ 代码对 'end_flag' 的使用, 
        #  我们必须使用 graphyflow_big.h 中的 'update_t' 定义)

        # 基础类型
        int_type = HLSType(HLSBasicType.INT)       # int32_t
        bool_type = HLSType(HLSBasicType.BOOL)
        ap_uint20_type = HLSType(basic_type=HLSBasicType.AP_UINT, width=20)
        ap_fixed_pod_t_type = HLSType(HLSBasicType.AP_FIXED_POD)

        # --- 结构体定义 (来自 graphyflow_big.h) ---

        # struct update_t { ap_uint<20> node_id; ap_fixed_pod_t prop; bool end_flag; }
        update_t_type = HLSType(basic_type=HLSBasicType.STRUCT,
                                struct_name="update_t_big",
                                struct_prop_names=["node_id", "prop", "end_flag"],
                                sub_types=[ap_uint20_type, ap_fixed_pod_t_type, bool_type])
        if update_t_type.name not in self.struct_definitions:
            self.struct_definitions[update_t_type.name] = (update_t_type, update_t_type.struct_prop_names)

        # --- Stream 类型 ---
        stream_type = HLSType(HLSBasicType.STREAM, sub_types=[update_t_type])

        # --- 参数变量 ---
        i_var = HLSVar(var_name="i", var_type=int_type)
        out1_var = HLSVar(var_name="out1", var_type=stream_type)
        out2_var = HLSVar(var_name="out2", var_type=stream_type)
        in1_var = HLSVar(var_name="in1", var_type=stream_type)
        in2_var = HLSVar(var_name="in2", var_type=stream_type)
        in3_var = HLSVar(var_name="in3", var_type=stream_type)
        in4_var = HLSVar(var_name="in4", var_type=stream_type)

        params.extend([i_var, out1_var, out2_var, in1_var, in2_var, in3_var, in4_var])
        receiver_2_func.params = params

        # --- 2. 函数体 ---
        code_lines: List[HLSCodeLine] = []

        # 局部变量 (Flags)
        in1_end_flag_var = HLSVar(var_name="in1_end_flag", var_type=bool_type)
        in2_end_flag_var = HLSVar(var_name="in2_end_flag", var_type=bool_type)
        in3_end_flag_var = HLSVar(var_name="in3_end_flag", var_type=bool_type)
        in4_end_flag_var = HLSVar(var_name="in4_end_flag", var_type=bool_type)
        # 局部变量 (Data)
        data_var = HLSVar(var_name="data", var_type=update_t_type)
        data_end_flag_var = HLSVar(var_name="data.end_flag", var_type=bool_type) # 用于赋值

        # #pragma HLS function_instantiate variable = i
        code_lines.append(CodePragma(content="function_instantiate variable = i"))

        # bool in1_end_flag = false;
        code_lines.append(CodeVarDecl(var_name="in1_end_flag", var_type=bool_type, init_val="false"))
        # bool in2_end_flag = false;
        code_lines.append(CodeVarDecl(var_name="in2_end_flag", var_type=bool_type, init_val="false"))
        # bool in3_end_flag = false;
        code_lines.append(CodeVarDecl(var_name="in3_end_flag", var_type=bool_type, init_val="false"))
        # bool in4_end_flag = false;
        code_lines.append(CodeVarDecl(var_name="in4_end_flag", var_type=bool_type, init_val="false"))


        # while (true) { ... }
        while_codes: List[HLSCodeLine] = []
        while_expr = HLSExpr(HLSExprT.CONST, True)

        # #pragma HLS PIPELINE II = 1
        while_codes.append(CodePragma(content="PIPELINE II = 1"))

        # --- if ((!in1.empty())) { ... } else if ((!in3.empty())) { ... } ---
        if_1_expr = HLSExpr(HLSExprT.UOP, dfir.UnaryOp.NOT, 
                            operands=[HLSExpr(HLSExprT.STREAM_EMPTY, None, 
                                              operands=[HLSExpr(HLSExprT.VAR, in1_var)])])
        if_1_codes: List[HLSCodeLine] = []
        # update_t data;
        if_1_codes.append(CodeVarDecl(var_name="data", var_type=update_t_type))
        # data = in1.read();
        if_1_codes.append(CodeAssign(var=data_var, 
                                     expr=HLSExpr(HLSExprT.STREAM_READ, None, 
                                                  operands=[HLSExpr(HLSExprT.VAR, in1_var)])))
        # if ((!data.end_flag)) { ... } else { ... }
        if_2_expr = HLSExpr(HLSExprT.UOP, dfir.UnaryOp.NOT, 
                            operands=[HLSExpr(HLSExprT.UOP, (dfir.UnaryOp.GET_ATTR, "end_flag"), 
                                              operands=[HLSExpr(HLSExprT.VAR, data_var)])])
        if_2_codes = [CodeWriteStream(stream_var=out1_var, in_expr=data_var)]
        else_2_codes = [CodeAssign(var=in1_end_flag_var, expr=HLSExpr(HLSExprT.CONST, True))]
        if_1_codes.append(CodeIf(expr=if_2_expr, if_codes=if_2_codes, else_codes=else_2_codes))

        # Elif codes
        elif_1_expr = HLSExpr(HLSExprT.UOP, dfir.UnaryOp.NOT, 
                              operands=[HLSExpr(HLSExprT.STREAM_EMPTY, None, 
                                                operands=[HLSExpr(HLSExprT.VAR, in3_var)])])
        elif_1_codes: List[HLSCodeLine] = []
        # update_t data;
        elif_1_codes.append(CodeVarDecl(var_name="data", var_type=update_t_type))
        # data = in3.read();
        elif_1_codes.append(CodeAssign(var=data_var, 
                                     expr=HLSExpr(HLSExprT.STREAM_READ, None, 
                                                  operands=[HLSExpr(HLSExprT.VAR, in3_var)])))
        # if ((!data.end_flag)) { ... } else { ... }
        if_3_expr = HLSExpr(HLSExprT.UOP, dfir.UnaryOp.NOT, 
                            operands=[HLSExpr(HLSExprT.UOP, (dfir.UnaryOp.GET_ATTR, "end_flag"), 
                                              operands=[HLSExpr(HLSExprT.VAR, data_var)])])
        if_3_codes = [CodeWriteStream(stream_var=out1_var, in_expr=data_var)]
        else_3_codes = [CodeAssign(var=in3_end_flag_var, expr=HLSExpr(HLSExprT.CONST, True))]
        elif_1_codes.append(CodeIf(expr=if_3_expr, if_codes=if_3_codes, else_codes=else_3_codes))

        # (构建 if_1 / elif_1)
        while_codes.append(CodeIf(expr=if_1_expr, if_codes=if_1_codes, elifs=[(elif_1_expr, elif_1_codes)]))


        # --- if ((!in2.empty())) { ... } else if ((!in4.empty())) { ... } ---
        if_4_expr = HLSExpr(HLSExprT.UOP, dfir.UnaryOp.NOT, 
                            operands=[HLSExpr(HLSExprT.STREAM_EMPTY, None, 
                                              operands=[HLSExpr(HLSExprT.VAR, in2_var)])])
        if_4_codes: List[HLSCodeLine] = []
        # update_t data;
        if_4_codes.append(CodeVarDecl(var_name="data", var_type=update_t_type))
        # data = in2.read();
        if_4_codes.append(CodeAssign(var=data_var, 
                                     expr=HLSExpr(HLSExprT.STREAM_READ, None, 
                                                  operands=[HLSExpr(HLSExprT.VAR, in2_var)])))
        # if ((!data.end_flag)) { ... } else { ... }
        if_5_expr = HLSExpr(HLSExprT.UOP, dfir.UnaryOp.NOT, 
                            operands=[HLSExpr(HLSExprT.UOP, (dfir.UnaryOp.GET_ATTR, "end_flag"), 
                                              operands=[HLSExpr(HLSExprT.VAR, data_var)])])
        if_5_codes = [CodeWriteStream(stream_var=out2_var, in_expr=data_var)]
        else_5_codes = [CodeAssign(var=in2_end_flag_var, expr=HLSExpr(HLSExprT.CONST, True))]
        if_4_codes.append(CodeIf(expr=if_5_expr, if_codes=if_5_codes, else_codes=else_5_codes))

        # Elif codes
        elif_2_expr = HLSExpr(HLSExprT.UOP, dfir.UnaryOp.NOT, 
                              operands=[HLSExpr(HLSExprT.STREAM_EMPTY, None, 
                                                operands=[HLSExpr(HLSExprT.VAR, in4_var)])])
        elif_2_codes: List[HLSCodeLine] = []
        # update_t data;
        elif_2_codes.append(CodeVarDecl(var_name="data", var_type=update_t_type))
        # data = in4.read();
        elif_2_codes.append(CodeAssign(var=data_var, 
                                     expr=HLSExpr(HLSExprT.STREAM_READ, None, 
                                                  operands=[HLSExpr(HLSExprT.VAR, in4_var)])))
        # if ((!data.end_flag)) { ... } else { ... }
        if_6_expr = HLSExpr(HLSExprT.UOP, dfir.UnaryOp.NOT, 
                            operands=[HLSExpr(HLSExprT.UOP, (dfir.UnaryOp.GET_ATTR, "end_flag"), 
                                              operands=[HLSExpr(HLSExprT.VAR, data_var)])])
        if_6_codes = [CodeWriteStream(stream_var=out2_var, in_expr=data_var)]
        else_6_codes = [CodeAssign(var=in4_end_flag_var, expr=HLSExpr(HLSExprT.CONST, True))]
        elif_2_codes.append(CodeIf(expr=if_6_expr, if_codes=if_6_codes, else_codes=else_6_codes))

        # (构建 if_4 / elif_2)
        while_codes.append(CodeIf(expr=if_4_expr, if_codes=if_4_codes, elifs=[(elif_2_expr, elif_2_codes)]))


        # --- if ((((in1_end_flag & in2_end_flag) & in3_end_flag) & in4_end_flag)) { ... } ---
        if_7_expr_A = HLSExpr(HLSExprT.BINOP, dfir.BinOp.AND, 
                               operands=[HLSExpr(HLSExprT.VAR, in1_end_flag_var), 
                                         HLSExpr(HLSExprT.VAR, in2_end_flag_var)])
        if_7_expr_B = HLSExpr(HLSExprT.BINOP, dfir.BinOp.AND, 
                               operands=[if_7_expr_A, 
                                         HLSExpr(HLSExprT.VAR, in3_end_flag_var)])
        if_7_expr = HLSExpr(HLSExprT.BINOP, dfir.BinOp.AND, 
                             operands=[if_7_expr_B, 
                                       HLSExpr(HLSExprT.VAR, in4_end_flag_var)])
        if_7_codes: List[HLSCodeLine] = []

        # update_t data;
        if_7_codes.append(CodeVarDecl(var_name="data", var_type=update_t_type))
        # data.end_flag = true;
        if_7_codes.append(CodeAssign(var=data_end_flag_var, expr=HLSExpr(HLSExprT.CONST, True)))
        # out1.write(data);
        if_7_codes.append(CodeWriteStream(stream_var=out1_var, in_expr=data_var))
        # out2.write(data);
        if_7_codes.append(CodeWriteStream(stream_var=out2_var, in_expr=data_var))
        # break;
        if_7_codes.append(CodeBreak())

        # (构建 if_7)
        while_codes.append(CodeIf(expr=if_7_expr, if_codes=if_7_codes, else_codes=None))

        # --- 
        # (构建 while)
        while_loop = CodeWhile(codes=while_codes, iter_expr=while_expr)
        code_lines.append(while_loop)

        # (完成函数)
        receiver_2_func.codes = code_lines
        self.big_gather_funcs.append(receiver_2_func)

        switch2x2_2_func = HLSFunction(name="switch2x2_2", comp=comp) # 假设 comp 存在
        params = []
        
        # --- 1. 定义类型和参数 ---
        # (基于 graphyflow_big.h 中的 'update_t' 定义)
        
        # 基础类型
        int_type = HLSType(HLSBasicType.INT)       # int32_t
        bool_type = HLSType(HLSBasicType.BOOL)
        ap_uint20_type = HLSType(basic_type=HLSBasicType.AP_UINT, width=20)
        ap_fixed_pod_t_type = HLSType(HLSBasicType.AP_FIXED_POD)
        
        # --- 结构体定义 (来自 graphyflow_big.h) ---
        update_t_type = HLSType(basic_type=HLSBasicType.STRUCT,
                                struct_name="update_t_big",
                                struct_prop_names=["node_id", "prop", "end_flag"],
                                sub_types=[ap_uint20_type, ap_fixed_pod_t_type, bool_type])
        if update_t_type.name not in self.struct_definitions:
            self.struct_definitions[update_t_type.name] = (update_t_type, update_t_type.struct_prop_names)
        
        # --- Stream 类型 ---
        stream_type = HLSType(HLSBasicType.STREAM, sub_types=[update_t_type])
        
        # --- 参数变量 ---
        i_var = HLSVar(var_name="i", var_type=int_type)
        in1_var = HLSVar(var_name="in1", var_type=stream_type)
        in2_var = HLSVar(var_name="in2", var_type=stream_type)
        out1_var = HLSVar(var_name="out1", var_type=stream_type)
        out2_var = HLSVar(var_name="out2", var_type=stream_type)
        
        params.extend([i_var, in1_var, in2_var, out1_var, out2_var])
        switch2x2_2_func.params = params
        
        # --- 2. 函数体 ---
        code_lines: List[HLSCodeLine] = []
        
        # #pragma HLS DATAFLOW
        code_lines.append(CodePragma(content="DATAFLOW"))
        
        # --- 局部 Stream 变量 ---
        l1_1_var = HLSVar(var_name="l1_1", var_type=stream_type)
        l1_2_var = HLSVar(var_name="l1_2", var_type=stream_type)
        l1_3_var = HLSVar(var_name="l1_3", var_type=stream_type)
        l1_4_var = HLSVar(var_name="l1_4", var_type=stream_type)
        
        # hls::stream<update_t> l1_1;
        code_lines.append(CodeVarDecl(var_name="l1_1", var_type=stream_type))
        #pragma HLS STREAM variable = l1_1 depth = 2
        code_lines.append(CodePragma(content="STREAM variable = l1_1 depth = 2"))
        
        # hls::stream<update_t> l1_2;
        code_lines.append(CodeVarDecl(var_name="l1_2", var_type=stream_type))
        #pragma HLS STREAM variable = l1_2 depth = 2
        code_lines.append(CodePragma(content="STREAM variable = l1_2 depth = 2"))
        
        # hls::stream<update_t> l1_3;
        code_lines.append(CodeVarDecl(var_name="l1_3", var_type=stream_type))
        #pragma HLS STREAM variable = l1_3 depth = 2
        code_lines.append(CodePragma(content="STREAM variable = l1_3 depth = 2"))
        
        # hls::stream<update_t> l1_4;
        code_lines.append(CodeVarDecl(var_name="l1_4", var_type=stream_type))
        #pragma HLS STREAM variable = l1_4 depth = 2
        code_lines.append(CodePragma(content="STREAM variable = l1_4 depth = 2"))
        
        # --- 函数调用 ---
        
        # (修正点: 直接使用 sender_2_func 和 receiver_2_func 变量)
        
        # sender_2(i, in1, in2, l1_1, l1_2, l1_3, l1_4);
        sender_2_params = [i_var, in1_var, in2_var, l1_1_var, l1_2_var, l1_3_var, l1_4_var]
        # 假设 sender_2_func 是 HLSFunction 对象，已在作用域中定义
        code_lines.append(CodeCall(func=sender_2_func, params=sender_2_params)) 
        
        # receiver_2(i, out1, out2, l1_1, l1_2, l1_3, l1_4);
        receiver_2_params = [i_var, out1_var, out2_var, l1_1_var, l1_2_var, l1_3_var, l1_4_var]
        # 假设 receiver_2_func 是 HLSFunction 对象，已在作用域中定义
        code_lines.append(CodeCall(func=receiver_2_func, params=receiver_2_params))
        
        # --- 3. Finalize ---
        switch2x2_2_func.codes = code_lines
        self.big_gather_funcs.append(switch2x2_2_func)
        self.big_top_dataflow_funcs.append(switch2x2_2_func)
        

        Reduc_105_unit_reduce_single_pe_func = HLSFunction(name="Reduc_105_unit_reduce_single_pe", comp=comp) # 假设 comp 存在
        params = []

        # --- 1. 定义类型和参数 ---

        # 基础类型
        uint_type = HLSType(HLSBasicType.UINT)      # uint32_t
        int_type = HLSType(HLSBasicType.INT)       # int
        bool_type = HLSType(HLSBasicType.BOOL)
        ap_fixed_pod_t_type = HLSType(HLSBasicType.AP_FIXED_POD) # ap_uint<32>

        # .h 和 C++ 中使用的特定 ap_uint 类型
        # (来自 graphyflow_little.h: #define REDUCE_MEM_WIDTH 64)
        reduce_word_t_type = HLSType(basic_type=HLSBasicType.REDUCE_WORD_T)
        # (来自 C++: ap_uint<20> key / word_addr / cache_addr_buffer)
        ap_uint20_type = HLSType(basic_type=HLSBasicType.AP_UINT, width=20)

        # --- 结构体定义 (来自 graphyflow_big.h, 因为 .end_flag) ---

        # struct update_t { ap_uint<20> node_id; ap_fixed_pod_t prop; bool end_flag; }
        update_t_type = HLSType(basic_type=HLSBasicType.STRUCT,
                                struct_name="update_t_big",
                                struct_prop_names=["node_id", "prop", "end_flag"],
                                sub_types=[ap_uint20_type, ap_fixed_pod_t_type, bool_type])
        if update_t_type.name not in self.struct_definitions:
            self.struct_definitions[update_t_type.name] = (update_t_type, update_t_type.struct_prop_names)

        # --- Stream 类型 ---
        stream_update_t_type = HLSType(HLSBasicType.STREAM, sub_types=[update_t_type])
        stream_reduce_word_t_type = HLSType(HLSBasicType.STREAM, sub_types=[reduce_word_t_type])

        # --- 参数变量 ---
        # Param 1: hls::stream<update_t> &kt_wrap_item_single
        kt_wrap_item_single_var = HLSVar(var_name="kt_wrap_item_single", var_type=stream_update_t_type)
        # Param 2: hls::stream<reduce_word_t> &pe_mem_out
        pe_mem_out_var = HLSVar(var_name="pe_mem_out", var_type=stream_reduce_word_t_type)
        # Param 3: uint32_t num_word_per_pe
        num_word_per_pe_var = HLSVar(var_name="num_word_per_pe", var_type=uint_type)

        params.extend([kt_wrap_item_single_var, pe_mem_out_var, num_word_per_pe_var])
        Reduc_105_unit_reduce_single_pe_func.params = params

        # --- 2. 函数体 ---
        code_lines: List[HLSCodeLine] = []

        # // --- Phase 1: Memory Declaration ---
        code_lines.append(CodeComment(text="--- Phase 1: Memory Declaration ---"))
        # const int MEM_SIZE = (MAX_NUM >> LOG_PE_NUM) / DISTANCES_PER_REDUCE_WORD;
        code_lines.append(CodeVarDecl(var_name="MEM_SIZE", var_type=int_type, init_val="((MAX_NUM >> LOG_PE_NUM) / DISTANCES_PER_REDUCE_WORD)", const=True))

        # reduce_word_t prop_mem[MEM_SIZE];
        prop_mem_type = HLSType(HLSBasicType.ARRAY, sub_types=[reduce_word_t_type], array_dims=["MEM_SIZE"])
        code_lines.append(CodeVarDecl(var_name="prop_mem", var_type=prop_mem_type))
        prop_mem_var = HLSVar(var_name="prop_mem", var_type=prop_mem_type)

        # #pragma HLS BIND_STORAGE ...
        code_lines.append(CodePragma(content="BIND_STORAGE variable = prop_mem type = RAM_2P impl = URAM"))
        # #pragma HLS dependence ...
        code_lines.append(CodePragma(content="dependence variable = prop_mem inter false"))
        code_lines.append(CodeOther(text=""))

        # // Latency-hiding cache for recently accessed URAM words
        code_lines.append(CodeComment(text="Latency-hiding cache for recently accessed URAM words"))

        # reduce_word_t cache_data_buffer[L + 1];
        cache_data_type = HLSType(HLSBasicType.ARRAY, sub_types=[reduce_word_t_type], array_dims=["(L + 1)"])
        code_lines.append(CodeVarDecl(var_name="cache_data_buffer", var_type=cache_data_type))
        cache_data_buffer_var = HLSVar(var_name="cache_data_buffer", var_type=cache_data_type)

        # #pragma HLS ARRAY_PARTITION ...
        code_lines.append(CodePragma(content="ARRAY_PARTITION variable = cache_data_buffer complete dim = 0"))

        # ap_uint<20> cache_addr_buffer[L + 1];
        cache_addr_type = HLSType(HLSBasicType.ARRAY, sub_types=[ap_uint20_type], array_dims=["(L + 1)"])
        code_lines.append(CodeVarDecl(var_name="cache_addr_buffer", var_type=cache_addr_type))
        cache_addr_buffer_var = HLSVar(var_name="cache_addr_buffer", var_type=cache_addr_type)

        # #pragma HLS ARRAY_PARTITION ...
        code_lines.append(CodePragma(content="ARRAY_PARTITION variable = cache_addr_buffer complete dim = 0"))
        code_lines.append(CodeOther(text=""))

        #ifdef EMULATION
        code_lines.append(CodeOther(text="#ifdef EMULATION"))
        code_lines.append(CodeOther(text="    memset(prop_mem, 0, sizeof(reduce_word_t) * MEM_SIZE);"))
        code_lines.append(CodeOther(text="#endif"))
        code_lines.append(CodeOther(text=""))

        # LOOP_INIT_CACHE_ADDR:
        code_lines.append(CodeOther(text="LOOP_INIT_CACHE_ADDR:"))
        # for (int i = 0; i < L + 1; i++)
        for_1_codes: List[HLSCodeLine] = []
        # #pragma HLS UNROLL
        for_1_codes.append(CodePragma(content="UNROLL"))
        # cache_addr_buffer[i] = 0x0; // Invalidate cache
        cache_addr_i_var = HLSVar(var_name="cache_addr_buffer[i]", var_type=ap_uint20_type)
        for_1_codes.append(CodeAssign(var=cache_addr_i_var, expr=HLSExpr(HLSExprT.CONST, "0x0")))
        for_1_codes.append(CodeComment(text="Invalidate cache"))
        # cache_data_buffer[i] = 0;
        cache_data_i_var = HLSVar(var_name="cache_data_buffer[i]", var_type=reduce_word_t_type)
        for_1_codes.append(CodeAssign(var=cache_data_i_var, expr=HLSExpr(HLSExprT.CONST, 0)))
        # (构建 for_1)
        code_lines.append(CodeFor(codes=for_1_codes, iter_limit="(L + 1)", iter_name="i", iter_val_type=int_type))
        code_lines.append(CodeOther(text=""))

        # // --- Phase 3: Aggregation Loop ---
        code_lines.append(CodeComment(text="--- Phase 3: Aggregation Loop ---"))

        # LOOP_AGGREGATE:
        code_lines.append(CodeOther(text="LOOP_AGGREGATE:"))
        # while (true)
        while_1_codes: List[HLSCodeLine] = []
        while_1_expr = HLSExpr(HLSExprT.CONST, True)
        # #pragma HLS PIPELINE II = 1
        while_1_codes.append(CodePragma(content="PIPELINE II = 1"))

        # update_t kt_elem;
        while_1_codes.append(CodeVarDecl(var_name="kt_elem", var_type=update_t_type))
        kt_elem_var = HLSVar(var_name="kt_elem", var_type=update_t_type)

        # kt_elem = kt_wrap_item_single.read();
        read_expr = HLSExpr(HLSExprT.STREAM_READ, None, operands=[HLSExpr(HLSExprT.VAR, kt_wrap_item_single_var)])
        while_1_codes.append(CodeAssign(var=kt_elem_var, expr=read_expr))

        # if (kt_elem.end_flag)
        if_1_codes: List[HLSCodeLine] = []
        if_1_expr = HLSExpr(HLSExprT.UOP, (dfir.UnaryOp.GET_ATTR, "end_flag"), operands=[HLSExpr(HLSExprT.VAR, kt_elem_var)])
        # break;
        if_1_codes.append(CodeBreak())
        # (构建 if_1)
        while_1_codes.append(CodeIf(expr=if_1_expr, if_codes=if_1_codes))

        # ap_uint<20> key = (kt_elem.node_id >> LOG_PE_NUM);
        while_1_codes.append(CodeVarDecl(var_name="key", var_type=ap_uint20_type, init_val="(kt_elem.node_id >> LOG_PE_NUM)"))
        key_var = HLSVar(var_name="key", var_type=ap_uint20_type)
        # ap_fixed_pod_t incoming_dist_pod = kt_elem.prop;
        while_1_codes.append(CodeVarDecl(var_name="incoming_dist_pod", var_type=ap_fixed_pod_t_type, init_val="kt_elem.prop"))
        incoming_dist_pod_var = HLSVar(var_name="incoming_dist_pod", var_type=ap_fixed_pod_t_type)
        while_1_codes.append(CodeOther(text=""))

        # ap_uint<20> word_addr = (key >> 1);
        while_1_codes.append(CodeVarDecl(var_name="word_addr", var_type=ap_uint20_type, init_val="(key >> 1)"))
        word_addr_var = HLSVar(var_name="word_addr", var_type=ap_uint20_type)
        while_1_codes.append(CodeOther(text=""))

        # reduce_word_t current_word = prop_mem[word_addr];
        while_1_codes.append(CodeVarDecl(var_name="current_word", var_type=reduce_word_t_type, init_val="prop_mem[word_addr]"))
        current_word_var = HLSVar(var_name="current_word", var_type=reduce_word_t_type)
        while_1_codes.append(CodeOther(text=""))

        # // Check cache first
        while_1_codes.append(CodeComment(text="Check cache first"))
        # // for (int i = L; i >= 0; --i) {
        while_1_codes.append(CodeComment(text="for (int i = L; i >= 0; --i) {"))
        # for (int i = 0; i < L + 1; i++)
        for_2_codes: List[HLSCodeLine] = []
        # #pragma HLS UNROLL
        for_2_codes.append(CodePragma(content="UNROLL"))
        # if (cache_addr_buffer[i] == word_addr)
        if_2_codes: List[HLSCodeLine] = []
        if_2_expr = HLSExpr(HLSExprT.CONST, "cache_addr_buffer[i] == word_addr")
        # current_word = cache_data_buffer[i];
        if_2_codes.append(CodeAssign(var=current_word_var, expr=HLSExpr(HLSExprT.CONST, "cache_data_buffer[i]")))
        # // break;
        if_2_codes.append(CodeComment(text="break;"))
        # (构建 if_2)
        for_2_codes.append(CodeIf(expr=if_2_expr, if_codes=if_2_codes))
        # (构建 for_2)
        while_1_codes.append(CodeFor(codes=for_2_codes, iter_limit="(L + 1)", iter_name="i", iter_val_type=int_type))
        while_1_codes.append(CodeOther(text=""))

        # // Shift cache
        while_1_codes.append(CodeComment(text="Shift cache"))
        # for (int i = 0; i < L; i++)
        for_3_codes: List[HLSCodeLine] = []
        # #pragma HLS UNROLL
        for_3_codes.append(CodePragma(content="UNROLL"))
        # cache_addr_buffer[i] = cache_addr_buffer[i + 1];
        cache_addr_shift_var = HLSVar(var_name="cache_addr_buffer[i]", var_type=ap_uint20_type)
        for_3_codes.append(CodeAssign(var=cache_addr_shift_var, expr=HLSExpr(HLSExprT.CONST, "cache_addr_buffer[i + 1]")))
        # cache_data_buffer[i] = cache_data_buffer[i + 1];
        cache_data_shift_var = HLSVar(var_name="cache_data_buffer[i]", var_type=reduce_word_t_type)
        for_3_codes.append(CodeAssign(var=cache_data_shift_var, expr=HLSExpr(HLSExprT.CONST, "cache_data_buffer[i + 1]")))
        # (构建 for_3)
        while_1_codes.append(CodeFor(codes=for_3_codes, iter_limit="L", iter_name="i", iter_val_type=int_type))
        while_1_codes.append(CodeOther(text=""))

        # reduce_word_t tmp_cur_word = current_word;
        while_1_codes.append(CodeVarDecl(var_name="tmp_cur_word", var_type=reduce_word_t_type, init_val="current_word"))
        while_1_codes.append(CodeOther(text=""))

        # ap_fixed_pod_t msb = tmp_cur_word.range(63, 32);
        while_1_codes.append(CodeVarDecl(var_name="msb", var_type=ap_fixed_pod_t_type, init_val="tmp_cur_word.range(63, 32)"))
        # ap_fixed_pod_t lsb = tmp_cur_word.range(31, 0);
        while_1_codes.append(CodeVarDecl(var_name="lsb", var_type=ap_fixed_pod_t_type, init_val="tmp_cur_word.range(31, 0)"))
        while_1_codes.append(CodeOther(text=""))

        # ap_fixed_pod_t msb_out = ...
        msb_out_init = "(msb < incoming_dist_pod && msb != 0x0) ? msb : incoming_dist_pod"
        while_1_codes.append(CodeVarDecl(var_name="msb_out", var_type=ap_fixed_pod_t_type, init_val=msb_out_init))
        # ap_fixed_pod_t lsb_out = ...
        lsb_out_init = "(lsb < incoming_dist_pod && lsb != 0x0) ? lsb : incoming_dist_pod"
        while_1_codes.append(CodeVarDecl(var_name="lsb_out", var_type=ap_fixed_pod_t_type, init_val=lsb_out_init))
        while_1_codes.append(CodeOther(text=""))

        # reduce_word_t accumulated_msb;
        while_1_codes.append(CodeVarDecl(var_name="accumulated_msb", var_type=reduce_word_t_type))
        accumulated_msb_var = HLSVar(var_name="accumulated_msb", var_type=reduce_word_t_type)
        # reduce_word_t accumulated_lsb;
        while_1_codes.append(CodeVarDecl(var_name="accumulated_lsb", var_type=reduce_word_t_type))
        accumulated_lsb_var = HLSVar(var_name="accumulated_lsb", var_type=reduce_word_t_type)
        while_1_codes.append(CodeOther(text=""))

        # accumulated_msb.range(63, 32) = msb_out;
        acc_msb_hi_var = HLSVar(var_name="accumulated_msb.range(63, 32)", var_type=ap_fixed_pod_t_type)
        while_1_codes.append(CodeAssign(var=acc_msb_hi_var, expr=HLSExpr(HLSExprT.CONST, "msb_out")))
        # accumulated_msb.range(31, 0) = tmp_cur_word.range(31, 0);
        acc_msb_lo_var = HLSVar(var_name="accumulated_msb.range(31, 0)", var_type=ap_fixed_pod_t_type)
        while_1_codes.append(CodeAssign(var=acc_msb_lo_var, expr=HLSExpr(HLSExprT.CONST, "tmp_cur_word.range(31, 0)")))
        while_1_codes.append(CodeOther(text=""))

        # accumulated_lsb.range(63, 32) = tmp_cur_word.range(63, 32);
        acc_lsb_hi_var = HLSVar(var_name="accumulated_lsb.range(63, 32)", var_type=ap_fixed_pod_t_type)
        while_1_codes.append(CodeAssign(var=acc_lsb_hi_var, expr=HLSExpr(HLSExprT.CONST, "tmp_cur_word.range(63, 32)")))
        # accumulated_lsb.range(31, 0) = lsb_out;
        acc_lsb_lo_var = HLSVar(var_name="accumulated_lsb.range(31, 0)", var_type=ap_fixed_pod_t_type)
        while_1_codes.append(CodeAssign(var=acc_lsb_lo_var, expr=HLSExpr(HLSExprT.CONST, "lsb_out")))
        while_1_codes.append(CodeOther(text=""))

        # if (key & 0x01)
        if_3_codes: List[HLSCodeLine] = []
        else_3_codes: List[HLSCodeLine] = []
        if_3_expr = HLSExpr(HLSExprT.BINOP, dfir.BinOp.AND, 
                             operands=[HLSExpr(HLSExprT.VAR, key_var), 
                                       HLSExpr(HLSExprT.CONST, "0x01")])
        # if codes
        prop_mem_addr_var = HLSVar(var_name="prop_mem[word_addr]", var_type=reduce_word_t_type)
        cache_data_L_var = HLSVar(var_name="cache_data_buffer[L]", var_type=reduce_word_t_type)
        if_3_codes.append(CodeAssign(var=prop_mem_addr_var, expr=accumulated_msb_var))
        if_3_codes.append(CodeAssign(var=cache_data_L_var, expr=accumulated_msb_var))
        # else codes
        else_3_codes.append(CodeAssign(var=prop_mem_addr_var, expr=accumulated_lsb_var))
        else_3_codes.append(CodeAssign(var=cache_data_L_var, expr=accumulated_lsb_var))
        # (构建 if_3)
        while_1_codes.append(CodeIf(expr=if_3_expr, if_codes=if_3_codes, else_codes=else_3_codes))

        # cache_addr_buffer[L] = word_addr;
        cache_addr_L_var = HLSVar(var_name="cache_addr_buffer[L]", var_type=ap_uint20_type)
        while_1_codes.append(CodeAssign(var=cache_addr_L_var, expr=HLSExpr(HLSExprT.VAR, word_addr_var)))

        # (构建 while_1)
        code_lines.append(CodeWhile(codes=while_1_codes, iter_expr=while_1_expr))
        code_lines.append(CodeOther(text=""))

        # // --- Phase 4: Stream out aggregated memory ---
        code_lines.append(CodeComment(text="--- Phase 4: Stream out aggregated memory ---"))

        # LOOP_STREAM_OUT:
        code_lines.append(CodeOther(text="LOOP_STREAM_OUT:"))
        # for (int i = 0; i < num_word_per_pe; i++)
        for_4_codes: List[HLSCodeLine] = []
        # #pragma HLS UNROLL factor = 1
        for_4_codes.append(CodePragma(content="UNROLL factor = 1"))
        # reduce_word_t tmp_word = prop_mem[i];
        for_4_codes.append(CodeVarDecl(var_name="tmp_word", var_type=reduce_word_t_type, init_val="prop_mem[i]"))
        tmp_word_var = HLSVar(var_name="tmp_word", var_type=reduce_word_t_type)
        # prop_mem[i] = 0;
        prop_mem_i_var = HLSVar(var_name="prop_mem[i]", var_type=reduce_word_t_type)
        for_4_codes.append(CodeAssign(var=prop_mem_i_var, expr=HLSExpr(HLSExprT.CONST, 0)))
        # pe_mem_out.write(tmp_word);
        for_4_codes.append(CodeWriteStream(stream_var=pe_mem_out_var, in_expr=tmp_word_var))
        # (构建 for_4)
        code_lines.append(CodeFor(codes=for_4_codes, iter_limit=num_word_per_pe_var, iter_name="i", iter_val_type=int_type))

        # --- 3. Finalize ---
        Reduc_105_unit_reduce_single_pe_func.codes = code_lines
        self.big_gather_funcs.append(Reduc_105_unit_reduce_single_pe_func)
        self.big_top_dataflow_funcs.append(Reduc_105_unit_reduce_single_pe_func)

        Reduc_105_drain_multi_pe_func_b = HLSFunction(name="Reduc_105_drain_multi_pe", comp=comp) # 假设 comp 存在
        params = []

        # --- 1. 定义类型和参数 ---
        # (基于 graphyflow_little.h 和 graphyflow_big.h)

        # 基础类型
        int_type = HLSType(HLSBasicType.INT)       # int32_t
        uint_type = HLSType(HLSBasicType.UINT)      # uint32_t
        # (来自 .h: #define REDUCE_MEM_WIDTH 64)
        reduce_word_t_type = HLSType(basic_type=HLSBasicType.REDUCE_WORD_T)
        # (来自 .h: #define AXI_BUS_WIDTH 512)
        bus_word_t_type = HLSType(HLSBasicType.BUS_WORD_T) 
        # (来自 .h: #define DISTANCE_BITWIDTH 32)
        ap_fixed_pod_t_type = HLSType(HLSBasicType.AP_FIXED_POD) 

        # --- 结构体定义 (基于 C++ 用法推断) ---

        # C++ 中 write_burst_pkt_t 被当作 ap_axiu<512,...> 使用
        # 它有一个 .data 成员，类型为 ap_uint<512> (即 bus_word_t)
        write_burst_pkt_t_type = HLSType(basic_type=HLSBasicType.STRUCT,
                                         struct_name="write_burst_pkt_t",
                                         struct_prop_names=["data"],
                                         sub_types=[bus_word_t_type])
        if write_burst_pkt_t_type.name not in self.struct_definitions:
            self.struct_definitions[write_burst_pkt_t_type.name] = (write_burst_pkt_t_type, write_burst_pkt_t_type.struct_prop_names)

        # --- Stream 类型 ---
        pe_mem_in_elem_type = HLSType(HLSBasicType.STREAM, sub_types=[reduce_word_t_type])
        kernel_out_stream_type = HLSType(HLSBasicType.STREAM, sub_types=[write_burst_pkt_t_type])

        # --- 参数变量 ---
        # Param 1: hls::stream<reduce_word_t> (&pe_mem_in)[PE_NUM]
        pe_mem_in_type = HLSType(HLSBasicType.ARRAY, sub_types=[pe_mem_in_elem_type], array_dims=["PE_NUM"])
        pe_mem_in_var = HLSVar(var_name="pe_mem_in", var_type=pe_mem_in_type)

        # Param 2: hls::stream<write_burst_pkt_t> &kernel_out_stream
        kernel_out_stream_var = HLSVar(var_name="kernel_out_stream", var_type=kernel_out_stream_type)

        # Param 3: uint32_t num_word_per_pe
        num_word_per_pe_var = HLSVar(var_name="num_word_per_pe", var_type=uint_type)

        params.extend([pe_mem_in_var, kernel_out_stream_var, num_word_per_pe_var])
        Reduc_105_drain_multi_pe_func_b.params = params

        # --- 2. 函数体 ---
        code_lines: List[HLSCodeLine] = []

        # LOOP_DRAIN_ADDR:
        code_lines.append(CodeOther(text="LOOP_DRAIN_ADDR:"))

        # for (int32_t i = 0; i < num_word_per_pe; i++)
        for_1_codes: List[HLSCodeLine] = []
        # #pragma HLS PIPELINE II = 1
        for_1_codes.append(CodePragma(content="PIPELINE II = 1"))
        # write_burst_pkt_t one_write_burst;
        for_1_codes.append(CodeVarDecl(var_name="one_write_burst", var_type=write_burst_pkt_t_type))
        one_write_burst_var = HLSVar(var_name="one_write_burst", var_type=write_burst_pkt_t_type)
        # reduce_word_t tmp_data[PE_NUM];
        tmp_data_type = HLSType(HLSBasicType.ARRAY, sub_types=[reduce_word_t_type], array_dims=["PE_NUM"])
        for_1_codes.append(CodeVarDecl(var_name="tmp_data", var_type=tmp_data_type))
        tmp_data_var = HLSVar(var_name="tmp_data", var_type=tmp_data_type)

        # for (uint32_t pe_idx = 0; pe_idx < PE_NUM; pe_idx++)
        for_2_codes: List[HLSCodeLine] = []
        # #pragma HLS UNROLL
        for_2_codes.append(CodePragma(content="UNROLL"))
        # tmp_data[pe_idx] = pe_mem_in[pe_idx].read();
        tmp_data_pe_idx_var = HLSVar(var_name="tmp_data[pe_idx]", var_type=reduce_word_t_type)
        read_expr = HLSExpr(HLSExprT.CONST, "pe_mem_in[pe_idx].read()")
        for_2_codes.append(CodeAssign(var=tmp_data_pe_idx_var, expr=read_expr))
        for_2_codes.append(CodeOther(text=""))

        # one_write_burst.data.range(31 + (pe_idx << 5), (pe_idx << 5)) = ...
        assign_1_lhs = HLSVar(var_name="one_write_burst.data.range(31 + (pe_idx << 5), (pe_idx << 5))", var_type=ap_fixed_pod_t_type)
        assign_1_rhs_expr = HLSExpr(HLSExprT.CONST, "tmp_data[pe_idx].range(31, 0)")
        for_2_codes.append(CodeAssign(var=assign_1_lhs, expr=assign_1_rhs_expr))
        # one_write_burst.data.range(..., ... + 256) = ...
        assign_2_lhs = HLSVar(var_name="one_write_burst.data.range(31 + (pe_idx << 5) + 256, (pe_idx << 5) + 256)", var_type=ap_fixed_pod_t_type)
        assign_2_rhs_expr = HLSExpr(HLSExprT.CONST, "tmp_data[pe_idx].range(63, 32)")
        for_2_codes.append(CodeAssign(var=assign_2_lhs, expr=assign_2_rhs_expr))

        # (构建 for_2)
        for_1_codes.append(CodeFor(codes=for_2_codes, iter_limit="PE_NUM", iter_name="pe_idx", iter_val_type=uint_type))
        # kernel_out_stream.write(one_write_burst);
        for_1_codes.append(CodeWriteStream(stream_var=kernel_out_stream_var, in_expr=one_write_burst_var))

        # (构建 for_1)
        code_lines.append(CodeFor(codes=for_1_codes, iter_limit=num_word_per_pe_var, iter_name="i", iter_val_type=int_type))

        # --- 3. Finalize ---
        Reduc_105_drain_multi_pe_func_b.codes = code_lines
        self.big_gather_funcs.append(Reduc_105_drain_multi_pe_func_b)
        self.big_top_dataflow_funcs.append(Reduc_105_drain_multi_pe_func_b)

        Reduc_105_unit_reduce_func = HLSFunction(name="Reduc_105_unit_reduce", comp=comp) # 假设 comp 存在
        params = []
        
        # --- 1. 定义类型和参数 (根据 .h 文件) ---
        
        # 基础类型
        int_type = HLSType(HLSBasicType.INT)       # int32_t / int
        uint_type = HLSType(HLSBasicType.UINT)      # uint32_t
        ap_fixed_pod_t_type = HLSType(HLSBasicType.AP_FIXED_POD)
        # (根据你的澄清，使用 HLSBasicType.REDUCE_WORD_T)
        reduce_word_t_type = HLSType(HLSBasicType.REDUCE_WORD_T)
        
        # .h 和 C++ 中使用的特定 ap_uint 类型
        # (来自 C++: ap_uint<20> key / cache_addr_buffer)
        ap_uint20_type = HLSType(basic_type=HLSBasicType.AP_UINT, width=20)
        # (来自 C++: ap_uint<15> word_addr)
        ap_uint15_type = HLSType(basic_type=HLSBasicType.AP_UINT, width=15)
        
        # --- 结构体定义 (来自 graphyflow_little.h, 因为此函数不使用 end_flag) ---
        
        # struct update_t
        update_t_type = HLSType(basic_type=HLSBasicType.STRUCT,
                                struct_name="update_t_little",
                                struct_prop_names=["node_id", "prop"], # .h: ap_uint<20>, ap_fixed_pod_t
                                sub_types=[ap_uint20_type, ap_fixed_pod_t_type]) 
        if update_t_type.name not in self.struct_definitions:
            self.struct_definitions[update_t_type.name] = (update_t_type, update_t_type.struct_prop_names)
        
        # struct update_tuple_t
        update_t_array_type = HLSType(HLSBasicType.ARRAY, sub_types=[update_t_type], array_dims=["PE_NUM"])
        update_tuple_t_type = HLSType(basic_type=HLSBasicType.STRUCT,
                                      struct_name="update_tuple_t_little",
                                      struct_prop_names=["data"], # .h: update_t data[PE_NUM]
                                      sub_types=[update_t_array_type])
        if update_tuple_t_type.name not in self.struct_definitions:
            self.struct_definitions[update_tuple_t_type.name] = (update_tuple_t_type, update_tuple_t_type.struct_prop_names)
        
        # --- Stream 类型 ---
        update_set_stm_type = HLSType(HLSBasicType.STREAM, sub_types=[update_tuple_t_type])
        pe_mem_outs_elem_type = HLSType(HLSBasicType.STREAM, sub_types=[reduce_word_t_type])
        
        # --- 参数变量 ---
        # Param 1: hls::stream<update_tuple_t> &update_set_stm
        update_set_stm_var = HLSVar(var_name="update_set_stm", var_type=update_set_stm_type)
        
        # Param 2: hls::stream<reduce_word_t> (&pe_mem_outs)[PE_NUM]
        pe_mem_outs_type = HLSType(HLSBasicType.ARRAY, sub_types=[pe_mem_outs_elem_type], array_dims=["PE_NUM"])
        pe_mem_outs_var = HLSVar(var_name="pe_mem_outs", var_type=pe_mem_outs_type)
        
        # Param 3: int32_t edge_num
        edge_num_var = HLSVar(var_name="edge_num", var_type=int_type)
        
        # Param 4: uint32_t rounded_num_words
        rounded_num_words_var = HLSVar(var_name="rounded_num_words", var_type=uint_type)
        
        params.extend([update_set_stm_var, pe_mem_outs_var, edge_num_var, rounded_num_words_var])
        Reduc_105_unit_reduce_func.params = params
        
        # --- 2. 函数体 ---
        code_lines: List[HLSCodeLine] = []
        
        # // --- Phase 1: Memory Declaration ---
        code_lines.append(CodeComment(text="--- Phase 1: Memory Declaration ---"))
        
        # const int MEM_SIZE = MAX_NUM / DISTANCES_PER_REDUCE_WORD;
        code_lines.append(CodeVarDecl(var_name="MEM_SIZE", var_type=int_type, init_val="(MAX_NUM / DISTANCES_PER_REDUCE_WORD)", const=True))
        
        # reduce_word_t prop_mem[PE_NUM][MEM_SIZE];
        prop_mem_elem_type = HLSType(HLSBasicType.ARRAY, sub_types=[reduce_word_t_type], array_dims=["MEM_SIZE"])
        prop_mem_type = HLSType(HLSBasicType.ARRAY, sub_types=[prop_mem_elem_type], array_dims=["PE_NUM"])
        code_lines.append(CodeVarDecl(var_name="prop_mem", var_type=prop_mem_type))
        prop_mem_var = HLSVar(var_name="prop_mem", var_type=prop_mem_type)
        
        # #pragma HLS ARRAY_PARTITION ...
        code_lines.append(CodePragma(content="ARRAY_PARTITION variable = prop_mem complete dim = 1"))
        code_lines.append(CodePragma(content="BIND_STORAGE variable = prop_mem type = RAM_S2P impl = URAM"))
        code_lines.append(CodePragma(content="dependence variable = prop_mem inter false"))
        code_lines.append(CodeOther(text=""))
        
        # // Latency-hiding cache for recently accessed URAM words
        code_lines.append(CodeComment(text="Latency-hiding cache for recently accessed URAM words"))
        
        # reduce_word_t cache_data_buffer[PE_NUM][L + 1];
        cache_data_elem_type = HLSType(HLSBasicType.ARRAY, sub_types=[reduce_word_t_type], array_dims=["(L + 1)"])
        cache_data_type = HLSType(HLSBasicType.ARRAY, sub_types=[cache_data_elem_type], array_dims=["PE_NUM"])
        code_lines.append(CodeVarDecl(var_name="cache_data_buffer", var_type=cache_data_type))
        cache_data_buffer_var = HLSVar(var_name="cache_data_buffer", var_type=cache_data_type)
        
        # #pragma HLS ARRAY_PARTITION ...
        code_lines.append(CodePragma(content="ARRAY_PARTITION variable = cache_data_buffer complete dim = 0"))
        
        # ap_uint<20> cache_addr_buffer[PE_NUM][L + 1];
        cache_addr_elem_type = HLSType(HLSBasicType.ARRAY, sub_types=[ap_uint20_type], array_dims=["(L + 1)"])
        cache_addr_type = HLSType(HLSBasicType.ARRAY, sub_types=[cache_addr_elem_type], array_dims=["PE_NUM"])
        code_lines.append(CodeVarDecl(var_name="cache_addr_buffer", var_type=cache_addr_type))
        cache_addr_buffer_var = HLSVar(var_name="cache_addr_buffer", var_type=cache_addr_type)
        
        # #pragma HLS ARRAY_PARTITION ...
        code_lines.append(CodePragma(content="ARRAY_PARTITION variable = cache_addr_buffer complete dim = 0"))
        code_lines.append(CodeOther(text=""))
        
        #ifdef EMULATION
        code_lines.append(CodeOther(text="#ifdef EMULATION"))
        code_lines.append(CodeOther(text="    memset(prop_mem, 0, sizeof(reduce_word_t) * PE_NUM * MEM_SIZE);"))
        code_lines.append(CodeOther(text="#endif"))
        code_lines.append(CodeOther(text=""))
        
        # LOOP_INIT_CACHE_ADDR:
        code_lines.append(CodeOther(text="LOOP_INIT_CACHE_ADDR:"))
        # for (int i = 0; i < L + 1; i++)
        for_1_codes: List[HLSCodeLine] = []
        for_1_codes.append(CodePragma(content="UNROLL"))
        # for (int pe = 0; pe < PE_NUM; pe++)
        for_2_codes: List[HLSCodeLine] = []
        for_2_codes.append(CodePragma(content="UNROLL"))
        # cache_addr_buffer[pe][i] = 0x0;
        cache_addr_pe_i_var = HLSVar(var_name="cache_addr_buffer[pe][i]", var_type=ap_uint20_type)
        for_2_codes.append(CodeAssign(var=cache_addr_pe_i_var, expr=HLSExpr(HLSExprT.CONST, "0x0")))
        for_2_codes.append(CodeComment(text="Invalidate cache"))
        # cache_data_buffer[pe][i] = 0x0;
        cache_data_pe_i_var = HLSVar(var_name="cache_data_buffer[pe][i]", var_type=reduce_word_t_type)
        for_2_codes.append(CodeAssign(var=cache_data_pe_i_var, expr=HLSExpr(HLSExprT.CONST, "0x0")))
        # (构建 for_2)
        for_1_codes.append(CodeFor(codes=for_2_codes, iter_limit="PE_NUM", iter_name="pe", iter_val_type=int_type))
        # (构建 for_1)
        code_lines.append(CodeFor(codes=for_1_codes, iter_limit="(L + 1)", iter_name="i", iter_val_type=int_type))
        code_lines.append(CodeOther(text=""))
        
        # const uint32_t total_updates = ...
        code_lines.append(CodeVarDecl(var_name="total_updates", var_type=uint_type, init_val="(edge_num >> LOG_PE_NUM)", const=True))
        total_updates_var = HLSVar(var_name="total_updates", var_type=uint_type)
        # // Assuming one update per node
        code_lines.append(CodeComment(text="Assuming one update per node"))
        # // --- Phase 3: Aggregation Loop ---
        code_lines.append(CodeComment(text="--- Phase 3: Aggregation Loop ---"))
        
        # LOOP_AGGREGATE:
        code_lines.append(CodeOther(text="LOOP_AGGREGATE:"))
        # for (int update_idx = 0; update_idx < total_updates; update_idx++)
        for_3_codes: List[HLSCodeLine] = []
        for_3_codes.append(CodePragma(content="PIPELINE II = 1"))
        # update_tuple_t one_update;
        for_3_codes.append(CodeVarDecl(var_name="one_update", var_type=update_tuple_t_type))
        one_update_var = HLSVar(var_name="one_update", var_type=update_tuple_t_type)
        # one_update = update_set_stm.read();
        read_expr = HLSExpr(HLSExprT.STREAM_READ, None, operands=[HLSExpr(HLSExprT.VAR, update_set_stm_var)])
        for_3_codes.append(CodeAssign(var=one_update_var, expr=read_expr))
        for_3_codes.append(CodeOther(text=""))
        
        # for (int pe = 0; pe < PE_NUM; pe++)
        for_4_codes: List[HLSCodeLine] = []
        for_4_codes.append(CodePragma(content="UNROLL"))
        # if ((one_update.data[pe].node_id.range(19, 19) == 0))
        if_1_codes: List[HLSCodeLine] = []
        if_1_expr = HLSExpr(HLSExprT.CONST, "(one_update.data[pe].node_id.range(19, 19) == 0)")
        
        # ap_uint<20> key = one_update.data[pe].node_id;
        if_1_codes.append(CodeVarDecl(var_name="key", var_type=ap_uint20_type, init_val="one_update.data[pe].node_id"))
        key_var = HLSVar(var_name="key", var_type=ap_uint20_type)
        # ap_fixed_pod_t incoming_dist_pod = one_update.data[pe].prop;
        if_1_codes.append(CodeVarDecl(var_name="incoming_dist_pod", var_type=ap_fixed_pod_t_type, init_val="one_update.data[pe].prop"))
        incoming_dist_pod_var = HLSVar(var_name="incoming_dist_pod", var_type=ap_fixed_pod_t_type)
        if_1_codes.append(CodeOther(text=""))
        
        # ap_uint<15> word_addr = key.range(15, 1);
        if_1_codes.append(CodeVarDecl(var_name="word_addr", var_type=ap_uint15_type, init_val="key.range(15, 1)"))
        word_addr_var = HLSVar(var_name="word_addr", var_type=ap_uint15_type)
        if_1_codes.append(CodeOther(text=""))
        
        # reduce_word_t current_word = prop_mem[pe][word_addr];
        if_1_codes.append(CodeVarDecl(var_name="current_word", var_type=reduce_word_t_type, init_val="prop_mem[pe][word_addr]"))
        current_word_var = HLSVar(var_name="current_word", var_type=reduce_word_t_type)
        if_1_codes.append(CodeOther(text=""))
        
        # // Check cache first
        if_1_codes.append(CodeComment(text="Check cache first"))
        # for (int i = L; i >= 0; --i)
        for_5_codes: List[HLSCodeLine] = []
        for_5_codes.append(CodePragma(content="UNROLL"))
        # if (cache_addr_buffer[pe][i] == word_addr)
        if_2_codes: List[HLSCodeLine] = []
        if_2_expr = HLSExpr(HLSExprT.CONST, "cache_addr_buffer[pe][i] == word_addr")
        # current_word = cache_data_buffer[pe][i];
        if_2_codes.append(CodeAssign(var=current_word_var, expr=HLSExpr(HLSExprT.CONST, "cache_data_buffer[pe][i]")))
        # break;
        if_2_codes.append(CodeBreak())
        # (构建 if_2)
        for_5_codes.append(CodeIf(expr=if_2_expr, if_codes=if_2_codes))
        # (构建 for_5)
        if_1_codes.append(CodeFor(codes=for_5_codes, iter_limit="-1", iter_cmp=">=", iter_name="i", iter_start="L", iter_step="--i", iter_val_type=int_type))
        if_1_codes.append(CodeOther(text=""))
        
        # // Shift cache
        if_1_codes.append(CodeComment(text="Shift cache"))
        # for (int i = 0; i < L; i++)
        for_6_codes: List[HLSCodeLine] = []
        for_6_codes.append(CodePragma(content="UNROLL"))
        # cache_addr_buffer[pe][i] = cache_addr_buffer[pe][i + 1];
        cache_addr_shift_var = HLSVar(var_name="cache_addr_buffer[pe][i]", var_type=ap_uint20_type)
        cache_addr_shift_expr = HLSExpr(HLSExprT.CONST, "cache_addr_buffer[pe][i + 1]")
        for_6_codes.append(CodeAssign(var=cache_addr_shift_var, expr=cache_addr_shift_expr))
        # cache_data_buffer[pe][i] = cache_data_buffer[pe][i + 1];
        cache_data_shift_var = HLSVar(var_name="cache_data_buffer[pe][i]", var_type=reduce_word_t_type)
        cache_data_shift_expr = HLSExpr(HLSExprT.CONST, "cache_data_buffer[pe][i + 1]")
        for_6_codes.append(CodeAssign(var=cache_data_shift_var, expr=cache_data_shift_expr))
        # (构建 for_6)
        if_1_codes.append(CodeFor(codes=for_6_codes, iter_limit="L", iter_name="i", iter_val_type=int_type))
        if_1_codes.append(CodeOther(text=""))
        
        # reduce_word_t tmp_cur_word = current_word;
        if_1_codes.append(CodeVarDecl(var_name="tmp_cur_word", var_type=reduce_word_t_type, init_val="current_word"))
        tmp_cur_word_var = HLSVar(var_name="tmp_cur_word", var_type=reduce_word_t_type)
        if_1_codes.append(CodeOther(text=""))
        
        # ap_fixed_pod_t msb = current_word.range(63, 32);
        if_1_codes.append(CodeVarDecl(var_name="msb", var_type=ap_fixed_pod_t_type, init_val="current_word.range(63, 32)"))
        # ap_fixed_pod_t lsb = current_word.range(31, 0);
        if_1_codes.append(CodeVarDecl(var_name="lsb", var_type=ap_fixed_pod_t_type, init_val="current_word.range(31, 0)"))
        if_1_codes.append(CodeOther(text=""))
        
        # ap_fixed_pod_t msb_out = ...
        msb_out_init = "(msb < incoming_dist_pod && msb != 0x0) ? msb : incoming_dist_pod"
        if_1_codes.append(CodeVarDecl(var_name="msb_out", var_type=ap_fixed_pod_t_type, init_val=msb_out_init))
        # ap_fixed_pod_t lsb_out = ...
        lsb_out_init = "(lsb < incoming_dist_pod && lsb != 0x0) ? lsb : incoming_dist_pod"
        if_1_codes.append(CodeVarDecl(var_name="lsb_out", var_type=ap_fixed_pod_t_type, init_val=lsb_out_init))
        if_1_codes.append(CodeOther(text=""))
        
        # reduce_word_t accumulate_msb;
        if_1_codes.append(CodeVarDecl(var_name="accumulate_msb", var_type=reduce_word_t_type))
        accumulate_msb_var = HLSVar(var_name="accumulate_msb", var_type=reduce_word_t_type)
        # reduce_word_t accumulate_lsb;
        if_1_codes.append(CodeVarDecl(var_name="accumulate_lsb", var_type=reduce_word_t_type))
        accumulate_lsb_var = HLSVar(var_name="accumulate_lsb", var_type=reduce_word_t_type)
        if_1_codes.append(CodeOther(text=""))
        
        # accumulate_msb.range(63, 32) = msb_out;
        acc_msb_hi_var = HLSVar(var_name="accumulate_msb.range(63, 32)", var_type=ap_fixed_pod_t_type)
        if_1_codes.append(CodeAssign(var=acc_msb_hi_var, expr=HLSExpr(HLSExprT.CONST, "msb_out")))
        # accumulate_msb.range(31, 0) = tmp_cur_word.range(31, 0);
        acc_msb_lo_var = HLSVar(var_name="accumulate_msb.range(31, 0)", var_type=ap_fixed_pod_t_type)
        if_1_codes.append(CodeAssign(var=acc_msb_lo_var, expr=HLSExpr(HLSExprT.CONST, "tmp_cur_word.range(31, 0)")))
        if_1_codes.append(CodeOther(text=""))
        
        # accumulate_lsb.range(63, 32) = tmp_cur_word.range(63, 32);
        acc_lsb_hi_var = HLSVar(var_name="accumulate_lsb.range(63, 32)", var_type=ap_fixed_pod_t_type)
        if_1_codes.append(CodeAssign(var=acc_lsb_hi_var, expr=HLSExpr(HLSExprT.CONST, "tmp_cur_word.range(63, 32)")))
        # accumulate_lsb.range(31, 0) = lsb_out;
        acc_lsb_lo_var = HLSVar(var_name="accumulate_lsb.range(31, 0)", var_type=ap_fixed_pod_t_type)
        if_1_codes.append(CodeAssign(var=acc_lsb_lo_var, expr=HLSExpr(HLSExprT.CONST, "lsb_out")))
        if_1_codes.append(CodeOther(text=""))
        
        # if (key & 0x01)
        if_3_codes: List[HLSCodeLine] = []
        else_3_codes: List[HLSCodeLine] = []
        if_3_expr = HLSExpr(HLSExprT.BINOP, dfir.BinOp.AND, 
                             operands=[HLSExpr(HLSExprT.VAR, key_var), 
                                       HLSExpr(HLSExprT.CONST, "0x01")])
        # if codes
        prop_mem_pe_addr_var = HLSVar(var_name="prop_mem[pe][word_addr]", var_type=reduce_word_t_type)
        cache_data_pe_L_var = HLSVar(var_name="cache_data_buffer[pe][L]", var_type=reduce_word_t_type)
        if_3_codes.append(CodeAssign(var=prop_mem_pe_addr_var, expr=accumulate_msb_var))
        if_3_codes.append(CodeAssign(var=cache_data_pe_L_var, expr=accumulate_msb_var))
        # else codes
        else_3_codes.append(CodeAssign(var=prop_mem_pe_addr_var, expr=accumulate_lsb_var))
        else_3_codes.append(CodeAssign(var=cache_data_pe_L_var, expr=accumulate_lsb_var))
        # (构建 if_3)
        if_1_codes.append(CodeIf(expr=if_3_expr, if_codes=if_3_codes, else_codes=else_3_codes))
        
        # cache_addr_buffer[pe][L] = word_addr;
        cache_addr_pe_L_var = HLSVar(var_name="cache_addr_buffer[pe][L]", var_type=ap_uint20_type)
        if_1_codes.append(CodeAssign(var=cache_addr_pe_L_var, expr=HLSExpr(HLSExprT.VAR, word_addr_var)))
        
        # (构建 if_1)
        for_4_codes.append(CodeIf(expr=if_1_expr, if_codes=if_1_codes))
        # (构建 for_4)
        for_3_codes.append(CodeFor(codes=for_4_codes, iter_limit="PE_NUM", iter_name="pe", iter_val_type=int_type))
        # (构建 for_3)
        code_lines.append(CodeFor(codes=for_3_codes, iter_limit=total_updates_var, iter_name="update_idx", iter_val_type=int_type))
        code_lines.append(CodeOther(text=""))
        
        # // --- Phase 4: Stream out aggregated memory ---
        code_lines.append(CodeComment(text="--- Phase 4: Stream out aggregated memory ---"))
        # LOOP_STREAM_OUT:
        code_lines.append(CodeOther(text="LOOP_STREAM_OUT:"))
        # for (int i = 0; i < rounded_num_words; i++)
        for_7_codes: List[HLSCodeLine] = []
        for_7_codes.append(CodePragma(content="PIPELINE"))
        # for (int pe = 0; pe < PE_NUM; pe++)
        for_8_codes: List[HLSCodeLine] = []
        for_8_codes.append(CodePragma(content="UNROLL"))
        # reduce_word_t word = prop_mem[pe][i];
        for_8_codes.append(CodeVarDecl(var_name="word", var_type=reduce_word_t_type, init_val="prop_mem[pe][i]"))
        word_var = HLSVar(var_name="word", var_type=reduce_word_t_type)
        # prop_mem[pe][i] = 0;
        prop_mem_pe_i_var = HLSVar(var_name="prop_mem[pe][i]", var_type=reduce_word_t_type)
        for_8_codes.append(CodeAssign(var=prop_mem_pe_i_var, expr=HLSExpr(HLSExprT.CONST, 0)))
        # pe_mem_outs[pe].write(word);
        pe_mem_out_var = HLSVar(var_name="pe_mem_outs[pe]", var_type=pe_mem_outs_elem_type)
        for_8_codes.append(CodeWriteStream(stream_var=pe_mem_out_var, in_expr=word_var))
        # (构建 for_8)
        for_7_codes.append(CodeFor(codes=for_8_codes, iter_limit="PE_NUM", iter_name="pe", iter_val_type=int_type))
        # (构建 for_7)
        code_lines.append(CodeFor(codes=for_7_codes, iter_limit=rounded_num_words_var, iter_name="i", iter_val_type=int_type))
        
        # --- 3. Finalize ---
        Reduc_105_unit_reduce_func.codes = code_lines
        self.little_gather_funcs.append(Reduc_105_unit_reduce_func)
        self.little_top_dataflow_funcs.append(Reduc_105_unit_reduce_func)

        Reduc_105_drain_multi_pe_func = HLSFunction(name="Reduc_105_drain_multi_pe", comp=comp) # 假设 comp 存在
        params = []

        # --- 1. 定义类型和参数 (根据 .h 文件) ---

        # 基础类型
        int_type = HLSType(HLSBasicType.INT)       # int32_t
        uint_type = HLSType(HLSBasicType.UINT)      # uint32_t
        bool_type = HLSType(HLSBasicType.BOOL)
        # (根据你的澄清，使用 HLSBasicType.REDUCE_WORD_T)
        reduce_word_t_type = HLSType(HLSBasicType.REDUCE_WORD_T)
        distance_t_type = HLSType(HLSBasicType.DISTANCE_T)
        ap_fixed_pod_t_type = HLSType(HLSBasicType.AP_FIXED_POD)

        # --- 结构体定义 (来自 graphyflow_little.h) ---

        # struct little_out_pkt_t (ap_axiu<64, 0, 0, 0>)
        # (根据 C++ 代码用法推断: .data (64-bit), .last (bool))
        little_out_pkt_t_type = HLSType(basic_type=HLSBasicType.STRUCT,
                                        struct_name="little_out_pkt_t",
                                        struct_prop_names=["data", "last"],
                                        sub_types=[reduce_word_t_type, bool_type])
        if little_out_pkt_t_type.name not in self.struct_definitions:
            self.struct_definitions[little_out_pkt_t_type.name] = (little_out_pkt_t_type, little_out_pkt_t_type.struct_prop_names)


        # --- Stream 类型 ---
        pe_mem_in_elem_type = HLSType(HLSBasicType.STREAM, sub_types=[reduce_word_t_type])
        kernel_out_stream_type = HLSType(HLSBasicType.STREAM, sub_types=[little_out_pkt_t_type])

        # --- 参数变量 ---
        # Param 1: hls::stream<reduce_word_t> (&pe_mem_in)[PE_NUM]
        pe_mem_in_type = HLSType(HLSBasicType.ARRAY, sub_types=[pe_mem_in_elem_type], array_dims=["PE_NUM"])
        pe_mem_in_var = HLSVar(var_name="pe_mem_in", var_type=pe_mem_in_type)

        # Param 2: hls::stream<little_out_pkt_t> &kernel_out_stream
        kernel_out_stream_var = HLSVar(var_name="kernel_out_stream", var_type=kernel_out_stream_type)

        # Param 3: uint32_t rounded_num_words
        rounded_num_words_var = HLSVar(var_name="rounded_num_words", var_type=uint_type)

        params.extend([pe_mem_in_var, kernel_out_stream_var, rounded_num_words_var])
        Reduc_105_drain_multi_pe_func.params = params

        # --- 2. 函数体 ---
        code_lines: List[HLSCodeLine] = []

        # // --- Phase 2: High-Performance Drain Loop ---
        code_lines.append(CodeComment(text="--- Phase 2: High-Performance Drain Loop ---"))
        # little_out_pkt_t one_write_burst;
        code_lines.append(CodeVarDecl(var_name="one_write_burst", var_type=little_out_pkt_t_type))
        one_write_burst_var = HLSVar(var_name="one_write_burst", var_type=little_out_pkt_t_type)
        # one_write_burst.last = 0;
        one_write_burst_last_var = HLSVar(var_name="one_write_burst.last", var_type=bool_type)
        code_lines.append(CodeAssign(var=one_write_burst_last_var, expr=HLSExpr(HLSExprT.CONST, 0)))
        # distance_t max_val = (distance_t)(16384.0);
        code_lines.append(CodeVarDecl(var_name="max_val", var_type=distance_t_type, init_val="(distance_t)(16384.0)"))
        # ap_fixed_pod_t max_pod = *reinterpret_cast<ap_fixed_pod_t *>(&max_val);
        code_lines.append(CodeVarDecl(var_name="max_pod", var_type=ap_fixed_pod_t_type, init_val="*reinterpret_cast<ap_fixed_pod_t *>(&max_val)"))
        max_pod_var = HLSVar(var_name="max_pod", var_type=ap_fixed_pod_t_type)
        code_lines.append(CodeOther(text=""))

        # LOOP_DRAIN_ADDR:
        code_lines.append(CodeOther(text="LOOP_DRAIN_ADDR:"))
        # for (int32_t i = 0; i < rounded_num_words; i++)
        for_1_codes: List[HLSCodeLine] = []
        # #pragma HLS PIPELINE II = 1
        for_1_codes.append(CodePragma(content="PIPELINE II = 1"))
        # ap_fixed_pod_t uram_res_low = max_pod;
        for_1_codes.append(CodeVarDecl(var_name="uram_res_low", var_type=ap_fixed_pod_t_type, init_val="max_pod"))
        uram_res_low_var = HLSVar(var_name="uram_res_low", var_type=ap_fixed_pod_t_type)
        # ap_fixed_pod_t uram_res_high = max_pod;
        for_1_codes.append(CodeVarDecl(var_name="uram_res_high", var_type=ap_fixed_pod_t_type, init_val="max_pod"))
        uram_res_high_var = HLSVar(var_name="uram_res_high", var_type=ap_fixed_pod_t_type)

        # for (uint32_t pe_idx = 0; pe_idx < PE_NUM; pe_idx++)
        for_2_codes: List[HLSCodeLine] = []
        # #pragma HLS UNROLL
        for_2_codes.append(CodePragma(content="UNROLL"))
        # reduce_word_t word = pe_mem_in[pe_idx].read();
        for_2_codes.append(CodeVarDecl(var_name="word", var_type=reduce_word_t_type))
        word_var = HLSVar(var_name="word", var_type=reduce_word_t_type)
        read_expr = HLSExpr(HLSExprT.CONST, "pe_mem_in[pe_idx].read()")
        for_2_codes.append(CodeAssign(var=word_var, expr=read_expr))
        for_2_codes.append(CodeOther(text=""))

        # ap_fixed_pod_t incoming_dist_pod_low = word.range(31, 0);
        for_2_codes.append(CodeVarDecl(var_name="incoming_dist_pod_low", var_type=ap_fixed_pod_t_type, init_val="word.range(31, 0)"))
        incoming_dist_pod_low_var = HLSVar(var_name="incoming_dist_pod_low", var_type=ap_fixed_pod_t_type)
        # ap_fixed_pod_t incoming_dist_pod_high = word.range(63, 32);
        for_2_codes.append(CodeVarDecl(var_name="incoming_dist_pod_high", var_type=ap_fixed_pod_t_type, init_val="word.range(63, 32)"))
        incoming_dist_pod_high_var = HLSVar(var_name="incoming_dist_pod_high", var_type=ap_fixed_pod_t_type)

        # uram_res_low = (uram_res_low < incoming_dist_pod_low || ...
        min_low_expr = HLSExpr(HLSExprT.CONST, "(uram_res_low < incoming_dist_pod_low || incoming_dist_pod_low == 0x0) ? uram_res_low : incoming_dist_pod_low")
        for_2_codes.append(CodeAssign(var=uram_res_low_var, expr=min_low_expr))
        # uram_res_high = (uram_res_high < incoming_dist_pod_high || ...
        min_high_expr = HLSExpr(HLSExprT.CONST, "(uram_res_high < incoming_dist_pod_high || incoming_dist_pod_high == 0x0) ? uram_res_high : incoming_dist_pod_high")
        for_2_codes.append(CodeAssign(var=uram_res_high_var, expr=min_high_expr))
        # (构建 for_2)
        for_1_codes.append(CodeFor(codes=for_2_codes, iter_limit="PE_NUM", iter_name="pe_idx", iter_val_type=uint_type))

        # reduce_word_t merged_word;
        for_1_codes.append(CodeVarDecl(var_name="merged_word", var_type=reduce_word_t_type))
        merged_word_var = HLSVar(var_name="merged_word", var_type=reduce_word_t_type)
        # merged_word.range(31, 0) = uram_res_low;
        merged_word_lo_var = HLSVar(var_name="merged_word.range(31, 0)", var_type=ap_fixed_pod_t_type)
        for_1_codes.append(CodeAssign(var=merged_word_lo_var, expr=uram_res_low_var))
        # merged_word.range(63, 32) = uram_res_high;
        merged_word_hi_var = HLSVar(var_name="merged_word.range(63, 32)", var_type=ap_fixed_pod_t_type)
        for_1_codes.append(CodeAssign(var=merged_word_hi_var, expr=uram_res_high_var))
        # one_write_burst.data = merged_word;
        one_write_burst_data_var = HLSVar(var_name="one_write_burst.data", var_type=reduce_word_t_type)
        for_1_codes.append(CodeAssign(var=one_write_burst_data_var, expr=merged_word_var))
        # kernel_out_stream.write(one_write_burst);
        for_1_codes.append(CodeWriteStream(stream_var=kernel_out_stream_var, in_expr=one_write_burst_var))

        # (构建 for_1)
        code_lines.append(CodeFor(codes=for_1_codes, iter_limit=rounded_num_words_var, iter_name="i", iter_val_type=int_type))

        # --- 3. Finalize ---
        Reduc_105_drain_multi_pe_func.codes = code_lines
        self.little_gather_funcs.append(Reduc_105_drain_multi_pe_func)
        self.little_top_dataflow_funcs.append(Reduc_105_drain_multi_pe_func)

    def _analyse_reduce(self,comp,group:str,port_property:Dict[dfir.Port,Any],port_to_var,top_vars,target_codes):
        port_already_analysed = []
        q = []
        visited_ids = set()

        global_key_inport =[]
        global_transform_inport =[]

        for port in comp._port_groups["global"]:
            if port.port_type == dfir.PortType.IN:
                if comp.glb_grp(port) == "key":
                    global_key_inport.append(port)
                elif comp.glb_grp(port) == "transform":
                    global_transform_inport.append(port)

        idx = 0
        for port in comp._port_groups["transform"]:
            
            if port.port_type == dfir.PortType.OUT:
                conn = port.connection
                parent = conn.parent
            
                port_property[port] = port_property[global_transform_inport[idx]]
                port_to_var[port] = port_to_var[global_transform_inport[idx]]
                idx += 1
                port_already_analysed.append(port)
                if parent not in q:
                    q.append(parent)
                        
        head = 0
        while head < len(q):
            cur = q[head]
            head+=1
            inputs_ready = all(p.connection in port_already_analysed for p in cur.in_ports)
            if not inputs_ready:
                q.append(cur)
                if head > len(q) * 2:
                    raise RuntimeError(f"Deadlock in sub-graph topological sort at component {cur.name}")
                continue
            else:
                port_property ,target_codes = self._scatter_analyze(cur,port_property,port_to_var,top_vars,target_codes)
            
            end = False
            for p in cur.out_ports:
                if p.connected and not isinstance(
                    p.connection.parent,
                    (dfir.ReduceComponent, dfir.UnusedEndMarkerComponent),
                ):
                    successor_comp = p.connection.parent
                    if successor_comp.readable_id not in visited_ids:
                        q.append(successor_comp)
                        visited_ids.add(successor_comp.readable_id)
                else:# cur是最后一个组件
                    end = True
            if end:
                for p in cur.out_ports:
                    if p.port_type == dfir.PortType.OUT:
                        for prop in port_property[p]:
                            if prop[0] == "edge":
                                if prop[1][0] == "weight":
                                    target_codes.append(CodeAssign(top_vars["FINAL_PROP_VAR"], port_to_var[p][0]))
                                elif prop[1][0] == "dst":
                                    target_codes.append(CodeAssign(top_vars["FINAL_DST_ID_VAR"], port_to_var[p][1]))
                                else:
                                    assert 0
                            
        
        return port_property,target_codes
    
        # hard code
    def _scatter_analyze(self,comp,port_property,port_to_var,top_vars,target_codes):

        if isinstance(comp, dfir.MemoryReadComponent):
            for port in comp.ports:
                if port.port_type == dfir.PortType.OUT: # 仅适用于现在的写法
                    port_access_pattern = comp.pname_to_pattern[port.name]
                    port_property[port] = port_access_pattern[1]
                    if port_property[port][0] == "edge":
                        if port_property[port][1][0] == "weight":
                            port_to_var[port] = top_vars["EDGE_WEIGHT_VAR"]
                        elif port_property[port][1][0] == "src":
                            if port_property[port][1][1] == "distance":
                                port_to_var[port] = top_vars["SRC_PROP_VAR"]
                            else:
                                assert 0
                        elif port_property[port][1][0] == "dst":
                            port_to_var[port] = top_vars["DST_ID_VAR"]
                    elif port_property[port][0] == "node":
                        port_to_var[port] = None
        
        elif isinstance(comp, dfir.FusedOpComponent):
            for port in comp.ports:
                if port.port_type == dfir.PortType.IN:
                    conn = port.connection
                    parent = conn.parent
                    port_property[port] = port_property[conn]
                    port_to_var[port] = port_to_var[conn]

            sub_graph_components = comp.sub_graph.topo_sort()
            for port in comp.sub_graph.inputs:
                if port.port_type == dfir.PortType.IN:
                    parent_port = comp.port_mapping[port.readable_id]
                    port_property[port] = port_property[parent_port]
                    port.connection = parent_port
                    port_to_var[port] = port_to_var[parent_port]

            for sub_c in sub_graph_components:
                port_property,target_codes = self._scatter_analyze(sub_c,port_property,port_to_var,top_vars,target_codes)
                
            for port in comp.sub_graph.outputs:
                if port.port_type == dfir.PortType.OUT:
                    child_port = comp.port_mapping[port.readable_id]
                    port_property[child_port] = port_property[port]
                    port_to_var[child_port] = port_to_var[port]
                
        elif isinstance(comp,dfir.ScatterComponent):
            idx = 0
            for port in comp.ports:
                if port.port_type == dfir.PortType.IN:
                    conn = port.connection
                    port_property[port] = port_property[conn]
                    in_properties = port_property[port]
                    in_vars = port_to_var[conn]
                elif port.port_type == dfir.PortType.OUT:
                    port_property[port] = in_properties[idx]
                    port_to_var[port] = in_vars[idx]
                    idx += 1
        elif isinstance(comp,dfir.GatherComponent):
            gather_out_property = []
            gather_out_vars = []
            for port in comp.ports:
                if port.port_type == dfir.PortType.IN:
                    if port in port_property:
                        if port_property[port] is not None:
                            gather_out_property.append(port_property[port])
                            gather_out_vars.append(port_to_var[port])
                    else:
                        conn = port.connection
                        port_property[port] = port_property[conn]
                        gather_out_property.append(port_property[port])
                        gather_out_vars.append(port_to_var[conn])
                elif port.port_type == dfir.PortType.OUT:
                    port_property[port] = gather_out_property
                    port_to_var[port] = gather_out_vars
        elif isinstance(comp,dfir.ConstantComponent):
            tmp_var = HLSVar(var_name=f"constant_{comp.readable_id}", var_type=HLSType(HLSBasicType.AP_FIXED_POD))
            target_codes.append(CodeVarDecl(var_name=f"constant_{comp.readable_id}", var_type=HLSType(HLSBasicType.AP_FIXED_POD), init_val=str(comp.value)))
            for port in comp.ports:
                if port.port_type == dfir.PortType.OUT:
                    port_property[port] = None
                    port_to_var[port] = tmp_var
            
            
        elif isinstance(comp,dfir.BinOpComponent):
            # lhs_var = HLSVar(var_name=f"BinOp_{comp.readable_id}_lhs", var_type=HLSType(HLSBasicType.AP_FIXED_POD))
            # rhs_var = HLSVar(var_name=f"BinOp_{comp.readable_id}_rhs", var_type=HLSType(HLSBasicType.AP_FIXED_POD))
            
            is_op1 = True
            for port in comp.ports:
                if port.port_type == dfir.PortType.IN:
                    conn = port.connection

                    if port in port_property:
                        if port_property[port] is not None:
                            inproperty = port_property[port]
                            
                    else:
                        port_property[port] = port_property[conn]
                        if port_property[conn] is not None:
                            inproperty = port_property[conn]

                    if is_op1:
                        is_op1 = False
                        op1_var = port_to_var[conn]
                        op1_expr = HLSExpr(HLSExprT.VAR, op1_var)
                    else:
                        op2_var = port_to_var[conn]
                        op2_expr = HLSExpr(HLSExprT.VAR, op2_var)
                elif port.port_type == dfir.PortType.OUT:
                    result_var = HLSVar(var_name=f"BinOp_{comp.readable_id}_res", var_type=HLSType(HLSBasicType.AP_FIXED_POD)) 
                    target_codes.append(CodeVarDecl(var_name=f"BinOp_{comp.readable_id}_res", var_type=HLSType(HLSBasicType.AP_FIXED_POD)))
                    tmp_expr = HLSExpr(HLSExprT.BINOP, comp.op, [op1_expr, op2_expr])
                    target_codes.append(CodeAssign(result_var, tmp_expr))

                    port_property[port] = inproperty
                    port_to_var[port] = result_var
            

        elif isinstance(comp,dfir.UnaryOpComponent):
            for port in comp.ports:
                if port in port_property:
                    if port_property[port] is not None:
                        inproperty = port_property[port]
                    continue
                if port.port_type == dfir.PortType.IN:
                    conn = port.connection
                    port_property[port] = port_property[conn]
                    if port_property[conn] is not None:
                        inproperty = port_property[conn]
                elif port.port_type == dfir.PortType.OUT:
                    port_property[port] = inproperty
        elif isinstance(comp,dfir.CopyComponent):
            
            for port in comp.ports:
                if port.port_type == dfir.PortType.IN:
                    conn = port.connection
                    port_property[port] = port_property[conn]
                    in_property = port_property[port]
                    in_var = port_to_var[conn]
                elif port.port_type == dfir.PortType.OUT:
                    port_property[port] = in_property
                    port_to_var[port] = in_var
        elif isinstance(comp,dfir.ReduceComponent):

            for port in comp._port_groups["global"]:
                if port.port_type == dfir.PortType.IN:
                    conn = port.connection
                    port_property[port] = port_property[conn]       
                    port_to_var[port] = port_to_var[conn]
            port_property,target_codes = self._analyse_reduce(comp,group="transform",port_property=port_property,port_to_var=port_to_var,top_vars=top_vars,target_codes=target_codes)
            # 约定reduce的key一定是 dst 的nodeIid
                
        else:
            pass

        return port_property, target_codes
    

    def _apply_analyze(self,comp,port_property,port_to_var,top_vars,target_codes):

        if isinstance(comp,dfir.ReduceComponent):
            for port in comp._port_groups["global"]:
                if port.port_type == dfir.PortType.OUT:
                    #HLSVar(var_name=f"reduce_{comp.readable_id}_key_out", var_type=HLSType(HLSBasicType.NODE_ID))
                    #HLSVar(var_name=f"reduce_{comp.readable_id}_transform_out", var_type=HLSType(HLSBasicType.AP_FIXED_POD))

                    out_vars = []
                    
                    out_vars.append(top_vars["init_val_2_var"])
                    out_vars.append(None)

                    port_to_var[port] = out_vars
                    
        elif isinstance(comp, dfir.MemoryReadComponent):
            for port in comp.ports:
                if port.port_type == dfir.PortType.OUT: # 仅适用于现在的写法
                    port_access_pattern = comp.pname_to_pattern[port.name]
                    port_property[port] = port_access_pattern[1]
                    if port_property[port][0] == "edge":
                        if port_property[port][1][0] == "weight":
                            port_to_var[port] = top_vars["EDGE_WEIGHT_VAR"]
                        elif port_property[port][1][0] == "src":
                            if port_property[port][1][1] == "distance":
                                port_to_var[port] = top_vars["SRC_PROP_VAR"]
                            else:
                                assert 0
                        elif port_property[port][1][0] == "dst":
                            port_to_var[port] = top_vars["DST_ID_VAR"]
                    elif port_property[port][0] == "node":
                        if port_property[port][1][0] == "distance":
                            port_to_var[port] = top_vars["init_val_1_var"]
                        else:
                            assert 0
        
        elif isinstance(comp, dfir.FusedOpComponent):
            for port in comp.ports:
                if port.port_type == dfir.PortType.IN:
                    conn = port.connection
                    parent = conn.parent
                    # port_property[port] = port_property[conn]
                    port_to_var[port] = port_to_var[conn]

            sub_graph_components = comp.sub_graph.topo_sort()
            for port in comp.sub_graph.inputs:
                if port.port_type == dfir.PortType.IN:
                    parent_port = comp.port_mapping[port.readable_id]
                    # port_property[port] = port_property[parent_port]
                    port.connection = parent_port
                    port_to_var[port] = port_to_var[parent_port]

            for sub_c in sub_graph_components:
                port_property,target_codes = self._apply_analyze(sub_c,port_property,port_to_var,top_vars,target_codes)
                
            for port in comp.sub_graph.outputs:
                if port.port_type == dfir.PortType.OUT:
                    child_port = comp.port_mapping[port.readable_id]
                    # port_property[child_port] = port_property[port]
                    port_to_var[child_port] = port_to_var[port]
                
        elif isinstance(comp,dfir.ScatterComponent):
            idx = 0
            for port in comp.ports:
                if port.port_type == dfir.PortType.IN:
                    conn = port.connection
                    # port_property[port] = port_property[conn]
                    # in_properties = port_property[port]
                    in_vars = port_to_var[conn]
                elif port.port_type == dfir.PortType.OUT:
                    # port_property[port] = in_properties[idx]
                    port_to_var[port] = in_vars[idx]
                    idx += 1
        elif isinstance(comp,dfir.GatherComponent):
            gather_out_property = []
            gather_out_vars = []
            for port in comp.ports:
                if port.port_type == dfir.PortType.IN:
                    # if port in port_property:
                    #     if port_property[port] is not None:
                    #         #gather_out_property.append(port_property[port])
                    #         gather_out_vars.append(port_to_var[port])
                    # else:
                    #     conn = port.connection
                    #     #port_property[port] = port_property[conn]
                    #     gather_out_property.append(port_property[port])
                    #     gather_out_vars.append(port_to_var[conn])
                    conn = port.connection
                    gather_out_vars.append(port_to_var[conn])
                elif port.port_type == dfir.PortType.OUT:
                    # port_property[port] = gather_out_property
                    target_codes.append(CodeAssign(var = top_vars["result_val3"], expr=HLSExpr(HLSExprT.VAR, gather_out_vars[0])))
                    port_to_var[port] = gather_out_vars

        elif isinstance(comp,dfir.ConstantComponent):
            tmp_var = HLSVar(var_name=f"constant_{comp.readable_id}", var_type=HLSType(HLSBasicType.AP_FIXED_POD))
            target_codes.append(CodeVarDecl(var_name=f"constant_{comp.readable_id}", var_type=HLSType(HLSBasicType.AP_FIXED_POD), init_val=str(comp.value)))
            for port in comp.ports:
                if port.port_type == dfir.PortType.OUT:
                    #port_property[port] = None
                    port_to_var[port] = tmp_var
            
            
        elif isinstance(comp,dfir.BinOpComponent):
            # lhs_var = HLSVar(var_name=f"BinOp_{comp.readable_id}_lhs", var_type=HLSType(HLSBasicType.AP_FIXED_POD))
            # rhs_var = HLSVar(var_name=f"BinOp_{comp.readable_id}_rhs", var_type=HLSType(HLSBasicType.AP_FIXED_POD))
            
            is_op1 = True
            for port in comp.ports:
                if port.port_type == dfir.PortType.IN:
                    conn = port.connection

                    # if port in port_property:
                    #     if port_property[port] is not None:
                    #         inproperty = port_property[port]
                    #         
                    # else:
                    #     port_property[port] = port_property[conn]
                    #     if port_property[conn] is not None:
                    #         inproperty = port_property[conn]
                    if port_to_var[conn] is None: # 没有用到node_id变量，因此也没有chushihua 
                        continue
                    if is_op1:
                        is_op1 = False
                        op1_var = port_to_var[conn]
                        op1_expr = HLSExpr(HLSExprT.VAR, op1_var)
                    else:
                        op2_var = port_to_var[conn]
                        op2_expr = HLSExpr(HLSExprT.VAR, op2_var)
            for port in comp.ports:
                if port.port_type == dfir.PortType.OUT:
                    result_var = HLSVar(var_name=f"BinOp_{comp.readable_id}_res", var_type=HLSType(HLSBasicType.AP_FIXED_POD)) 
                    target_codes.append(CodeVarDecl(var_name=f"BinOp_{comp.readable_id}_res", var_type=HLSType(HLSBasicType.AP_FIXED_POD)))
                    tmp_expr = HLSExpr(HLSExprT.BINOP, comp.op, [op1_expr, op2_expr])
                    target_codes.append(CodeAssign(result_var, tmp_expr))

                    # port_property[port] = inproperty
                    port_to_var[port] = result_var
            

        elif isinstance(comp,dfir.UnaryOpComponent):
            for port in comp.ports:
                if port in port_property:
                    if port_property[port] is not None:
                        inproperty = port_property[port]
                    continue
                if port.port_type == dfir.PortType.IN:
                    conn = port.connection
                    port_property[port] = port_property[conn]
                    if port_property[conn] is not None:
                        inproperty = port_property[conn]
                elif port.port_type == dfir.PortType.OUT:
                    port_property[port] = inproperty
        elif isinstance(comp,dfir.CopyComponent):
            
            for port in comp.ports:
                if port.port_type == dfir.PortType.IN:
                    conn = port.connection
                    #port_property[port] = port_property[conn]
                    #in_property = port_property[port]
                    in_var = port_to_var[conn]
                elif port.port_type == dfir.PortType.OUT:
                    #port_property[port] = in_property
                    port_to_var[port] = in_var

        else:
            pass

        return port_property, target_codes
    
    
    def process_scatter(self,scatter_stage_comps : List[dfir.Component],ReduceComp : dfir.ReduceComponent):

            self._translate_memory_read_op(None) #这部分访存应该全是hard code

            merge_node_props_func = HLSFunction(name="merge_node_props", comp=None) # 假设 comp 存在
            params = []


            # 基础类型
            node_id_type = HLSType(HLSBasicType.NODE_ID) # ap_uint<32>
            bus_word_t_type = HLSType(HLSBasicType.BUS_WORD_T) # ap_uint<512>
            ap_fixed_pod_t_type = HLSType(HLSBasicType.AP_FIXED_POD) # ap_uint<32>
            distance_t_type = HLSType(HLSBasicType.DISTANCE_T) # ap_fixed<32, 16>
            int_type = HLSType(HLSBasicType.INT) # int32_t
            uint_type = HLSType(HLSBasicType.UINT) # uint32_t
            bool_type = HLSType(HLSBasicType.BOOL)

            # .h 和 C++ 代码中定义的特定 ap_uint 类型
            ap_uint26_type = HLSType(basic_type=HLSBasicType.AP_UINT, width=26) # 匹配 C++ ap_uint<26>
            ap_uint20_type = HLSType(basic_type=HLSBasicType.AP_UINT, width=20) # 匹配 .h ap_uint<20>
            ap_uint4_type = HLSType(basic_type=HLSBasicType.AP_UINT, width=4) # 匹配 C++ ap_uint<4>
            ap_uint9_type = HLSType(basic_type=HLSBasicType.AP_UINT, width=9) # 匹配 C++ (ap_uint<9>)

            # --- 结构体定义 (严格按照 .h 文件) ---

            # struct edge_t
            edge_t_type = HLSType(basic_type=HLSBasicType.STRUCT,
                                struct_name="edge_t",
                                struct_prop_names=["src_id", "dst_id"],
                                sub_types=[node_id_type, ap_uint20_type]) # .h: dst_id 是 ap_uint<20>
            if edge_t_type.name not in self.struct_definitions:
                self.struct_definitions[edge_t_type.name] = (edge_t_type, edge_t_type.struct_prop_names)

            # struct edge_descriptor_batch_t
            edge_array_type = HLSType(HLSBasicType.ARRAY, sub_types=[edge_t_type], array_dims=["PE_NUM"])
            edge_descriptor_batch_t_type = HLSType(basic_type=HLSBasicType.STRUCT,
                                                struct_name="edge_descriptor_batch_t",
                                                struct_prop_names=["edges"], # .h: 只有 'edges'
                                                sub_types=[edge_array_type])
            
            if edge_descriptor_batch_t_type.name not in self.struct_definitions:
                self.struct_definitions[edge_descriptor_batch_t_type.name] = (edge_descriptor_batch_t_type, edge_descriptor_batch_t_type.struct_prop_names)

            # struct update_t
            update_t_type = HLSType(basic_type=HLSBasicType.STRUCT,
                                    struct_name="update_t_big",
                                    struct_prop_names=["node_id", "prop", "end_flag"],
                                    sub_types=[ap_uint20_type, ap_fixed_pod_t_type, bool_type]) # .h: node_id 是 ap_uint<20>
            if update_t_type.name not in self.struct_definitions:
                self.struct_definitions[update_t_type.name] = (update_t_type, update_t_type.struct_prop_names)

            # struct update_tuple_t
            update_t_array_type = HLSType(HLSBasicType.ARRAY, sub_types=[update_t_type], array_dims=["PE_NUM"])
            update_tuple_t_type = HLSType(basic_type=HLSBasicType.STRUCT,
                                        struct_name="update_tuple_t_big",
                                        struct_prop_names=["data"], # .h: 只有 'data'
                                        sub_types=[update_t_array_type])
            if update_tuple_t_type.name not in self.struct_definitions:
                self.struct_definitions[update_tuple_t_type.name] = (update_tuple_t_type, update_tuple_t_type.struct_prop_names)

            # --- 参数定义 ---

            # Param 1: hls::stream<bus_word_t> (&cacheline_streams)[PE_NUM]
            bus_word_stream_type = HLSType(HLSBasicType.STREAM, sub_types=[bus_word_t_type])
            cacheline_streams_type = HLSType(HLSBasicType.ARRAY, sub_types=[bus_word_stream_type], array_dims=["PE_NUM"])
            cacheline_streams = HLSVar(var_name="cacheline_streams", var_type=cacheline_streams_type)

            # Param 2: hls::stream<edge_descriptor_batch_t> &edge_stream
            edge_stream_type = HLSType(HLSBasicType.STREAM, sub_types=[edge_descriptor_batch_t_type])
            edge_stream = HLSVar(var_name="edge_stream", var_type=edge_stream_type)

            # Param 3: hls::stream<update_tuple_t> &edge_batch_stream
            edge_batch_stream_type = HLSType(HLSBasicType.STREAM, sub_types=[update_tuple_t_type])
            edge_batch_stream = HLSVar(var_name="edge_batch_stream", var_type=edge_batch_stream_type)

            # Param 4: uint32_t edge_num
            edge_num_var = HLSVar(var_name="edge_num", var_type=uint_type)

            params.extend([cacheline_streams, edge_stream, edge_batch_stream, edge_num_var])
            merge_node_props_func.params = params

            # --- 2. 函数体 ---
            code_lines: List[HLSCodeLine] = []

            # bus_word_t last_cacheline[PE_NUM] = {0};
            last_cacheline_type = HLSType(HLSBasicType.ARRAY, sub_types=[bus_word_t_type], array_dims=["PE_NUM"])
            code_lines.append(CodeVarDecl(var_name="last_cacheline", var_type=last_cacheline_type, init_val="{0}"))
            last_cacheline_var = HLSVar(var_name="last_cacheline", var_type=last_cacheline_type)

            # #pragma HLS ARRAY_PARTITION variable = last_cacheline complete dim = 0
            code_lines.append(CodePragma(content="ARRAY_PARTITION variable = last_cacheline complete dim = 0"))

            # ap_uint<26> last_cache_idx[PE_NUM] = {0};
            last_cache_idx_type = HLSType(HLSBasicType.ARRAY, sub_types=[ap_uint26_type], array_dims=["PE_NUM"])
            code_lines.append(CodeVarDecl(var_name="last_cache_idx", var_type=last_cache_idx_type, init_val="{0}"))
            last_cache_idx_var = HLSVar(var_name="last_cache_idx", var_type=last_cache_idx_type)

            # #pragma HLS ARRAY_PARTITION variable = last_cache_idx complete dim = 0
            code_lines.append(CodePragma(content="ARRAY_PARTITION variable = last_cache_idx complete dim = 0"))

            code_lines.append(CodeOther(text=""))

            # // Init first cacheline for each PE
            code_lines.append(CodeComment(text="Init first cacheline for each PE"))
            # LOOP_INIT_CACHELINE:
            code_lines.append(CodeOther(text="LOOP_INIT_CACHELINE:"))

            # --- Build for(pe_idx) 1 ---
            for_loop_1_codes: List[HLSCodeLine] = []
            # #pragma HLS UNROLL
            for_loop_1_codes.append(CodePragma(content="UNROLL"))
            # last_cacheline[pe_idx] = cacheline_streams[pe_idx].read();
            last_cacheline_pe_idx_var = HLSVar(var_name="last_cacheline[pe_idx]", var_type=bus_word_t_type)
            assign_expr_1 = HLSExpr(HLSExprT.CONST, "cacheline_streams[pe_idx].read()")
            for_loop_1_codes.append(CodeAssign(var=last_cacheline_pe_idx_var, expr=assign_expr_1))
            # last_cache_idx[pe_idx] = 0x0;
            last_cache_idx_pe_idx_var = HLSVar(var_name="last_cache_idx[pe_idx]", var_type=ap_uint26_type)
            assign_expr_2 = HLSExpr(HLSExprT.CONST, "0x0")
            for_loop_1_codes.append(CodeAssign(var=last_cache_idx_pe_idx_var, expr=assign_expr_2))
            # Create for loop 1
            for_loop_1 = CodeFor(codes=for_loop_1_codes,
                                iter_limit="PE_NUM",
                                iter_cmp="<",
                                iter_name="pe_idx",
                                iter_start="0",
                                iter_step="pe_idx++",
                                iter_val_type=int_type) # C++: int32_t
            code_lines.append(for_loop_1)
            code_lines.append(CodeOther(text=""))
            # --- End for(pe_idx) 1 ---

            # const uint32_t scatter_size = (edge_num >> LOG_PE_NUM);
            code_lines.append(CodeVarDecl(var_name="scatter_size", var_type=uint_type, init_val="(edge_num >> LOG_PE_NUM)", const=True))
            scatter_size_var = HLSVar(var_name="scatter_size", var_type=uint_type)

            # distance_t real_edge_weight = 1.0; 
            code_lines.append(CodeVarDecl(var_name="real_edge_weight", var_type=distance_t_type, init_val="1.0", const=False))
            # // All edge weights are 1.0 in unweighted graph
            code_lines.append(CodeComment(text="All edge weights are 1.0 in unweighted graph"))

            # const ap_fixed_pod_t edge_weight = (*reinterpret_cast<...>(&real_edge_weight));
            edge_weight_init_val = "(*reinterpret_cast<ap_fixed_pod_t *>(&real_edge_weight))"
            code_lines.append(CodeVarDecl(var_name="edge_weight", var_type=ap_fixed_pod_t_type, init_val=edge_weight_init_val, const=True))
            edge_weight_var = HLSVar(var_name="edge_weight", var_type=ap_fixed_pod_t_type)
            code_lines.append(CodeOther(text=""))

            # LOOP_SCATTER_EDGES:
            code_lines.append(CodeOther(text="LOOP_SCATTER_EDGES:"))

            # --- Build for(edge_batch_idx) ---
            for_loop_2_codes: List[HLSCodeLine] = []
            # #pragma HLS PIPELINE II = 1
            for_loop_2_codes.append(CodePragma(content="PIPELINE II = 1"))

            # edge_descriptor_batch_t edge_batch;
            for_loop_2_codes.append(CodeVarDecl(var_name="edge_batch", var_type=edge_descriptor_batch_t_type))
            edge_batch_var = HLSVar(var_name="edge_batch", var_type=edge_descriptor_batch_t_type)

            # edge_batch = edge_stream.read();
            assign_expr_3 = HLSExpr(HLSExprT.CONST, "edge_stream.read()")
            for_loop_2_codes.append(CodeAssign(var=edge_batch_var, expr=assign_expr_3))

            for_loop_2_codes.append(CodeOther(text=""))
            # update_tuple_t out_batch;
            for_loop_2_codes.append(CodeVarDecl(var_name="out_batch", var_type=update_tuple_t_type))
            out_batch_var = HLSVar(var_name="out_batch", var_type=update_tuple_t_type)
            for_loop_2_codes.append(CodeOther(text=""))

            # --- Build for(pe_idx) 2 ---
            for_loop_3_codes: List[HLSCodeLine] = []
            # #pragma HLS UNROLL
            for_loop_3_codes.append(CodePragma(content="UNROLL"))

            # ap_uint<26> cacheline_idx = ...
            cacheline_idx_init_val = "edge_batch.edges[pe_idx].src_id.range(29, 4)"
            for_loop_3_codes.append(CodeVarDecl(var_name="cacheline_idx", var_type=ap_uint26_type, init_val=cacheline_idx_init_val))
            cacheline_idx_var = HLSVar(var_name="cacheline_idx", var_type=ap_uint26_type)

            # ap_uint<4> offset = ...
            offset_init_val = "edge_batch.edges[pe_idx].src_id.range(3, 0)"
            for_loop_3_codes.append(CodeVarDecl(var_name="offset", var_type=ap_uint4_type, init_val=offset_init_val))
            offset_var = HLSVar(var_name="offset", var_type=ap_uint4_type)

            # bus_word_t cacheline;
            cacheline_var = HLSVar(var_name="cacheline", var_type=bus_word_t_type)
            for_loop_3_codes.append(CodeVarDecl(var_name="cacheline", var_type=cacheline_var.type))

            # --- Build IF_2 (cacheline_idx == last_cache_idx[pe_idx]) ---
            if_2_codes: List[HLSCodeLine] = []
            else_2_codes: List[HLSCodeLine] = []
            if_expr_2 = HLSExpr(HLSExprT.CONST, "cacheline_idx == last_cache_idx[pe_idx]")
            # Build IF_2 Contents
            assign_expr_6 = HLSExpr(HLSExprT.CONST, "last_cacheline[pe_idx]")
            if_2_codes.append(CodeAssign(var=cacheline_var, expr=assign_expr_6))
            # Build ELSE_2 Contents
            assign_expr_7 = HLSExpr(HLSExprT.CONST, "cacheline_streams[pe_idx].read()")
            else_2_codes.append(CodeAssign(var=cacheline_var, expr=assign_expr_7))
            # Create IF_2
            if_2 = CodeIf(expr=if_expr_2, if_codes=if_2_codes, else_codes=else_2_codes)
            for_loop_3_codes.append(if_2)
            for_loop_3_codes.append(CodeOther(text="")) # 空行

            # ap_fixed_pod_t prop = ...
            prop_init_val = "cacheline.range(31 + ((ap_uint<9>)offset << 5), ((ap_uint<9>)offset << 5))"
            for_loop_3_codes.append(CodeVarDecl(var_name="prop", var_type=ap_fixed_pod_t_type, init_val=prop_init_val))
            prop_var = HLSVar(var_name="prop", var_type=ap_fixed_pod_t_type)

            # ap_fixed_pod_t update = prop + edge_weight;
            update_init_expr = HLSExpr(HLSExprT.BINOP, dfir.BinOp.ADD,
                                    operands=[HLSExpr(HLSExprT.VAR, prop_var),
                                                HLSExpr(HLSExprT.VAR, edge_weight_var)])
            for_loop_3_codes.append(CodeVarDecl(var_name="update", var_type=ap_fixed_pod_t_type, init_val=update_init_expr.code))
            update_var = HLSVar(var_name="update", var_type=ap_fixed_pod_t_type)
            for_loop_3_codes.append(CodeOther(text="")) # 空行

            # out_batch.data[pe_idx].node_id = edge_batch.edges[pe_idx].dst_id;
            # (注意 HLSVar 的 name 属性现在匹配 .h 结构, 类型是 ap_uint20_type)
            out_batch_node_id_var = HLSVar(var_name="out_batch.data[pe_idx].node_id", var_type=ap_uint20_type)
            #assign_expr_8 = HLSExpr(HLSExprT.CONST, "edge_batch.edges[pe_idx].dst_id")
            #for_loop_3_codes.append(CodeAssign(var=out_batch_node_id_var, expr=assign_expr_8))

            # out_batch.data[pe_idx].prop = update;
            out_batch_prop_var = HLSVar(var_name="out_batch.data[pe_idx].prop", var_type=ap_fixed_pod_t_type)
            #assign_expr_9 = HLSExpr(HLSExprT.VAR, update_var)
            #for_loop_3_codes.append(CodeAssign(var=out_batch_prop_var, expr=assign_expr_9))


            # ============= begin inline logic ===============

            port_to_var = {}
            for_loop_3_codes.append(CodeOther(text="// Begin inline logic"))
            # DST_ID_VAR = HLSVar(var_name="DST_ID_VAR", var_type=node_id_type)
            # for_2_codes.append(CodeVarDecl(var_name="DST_ID_VAR", var_type=node_id_type, init_val="an_edge_burst.edges[u].dst_id"))
            # SRC_PROP_VAR = src_prop_var
            # EDGE_WEIGHT_VAR = edge_weight_var
            top_vars = {
                "DST_ID_VAR": out_batch_node_id_var,
                "SRC_PROP_VAR": prop_var,
                "EDGE_WEIGHT_VAR": edge_weight_var
            }

            top_vars["FINAL_PROP_VAR"] = out_batch_prop_var
            top_vars["FINAL_DST_ID_VAR"] = out_batch_node_id_var
            port_property = {} # 只能是src , dst, edge_prop这三项或组合
            inlinecodes = []
            for comp in scatter_stage_comps:
                port_property,inlinecodes = self._scatter_analyze(comp,port_property,port_to_var,top_vars,target_codes=inlinecodes)
            for_loop_3_codes.extend(inlinecodes)
            for_loop_3_codes.append(CodeOther(text="// End inline logic"))
            # =========== end inline logic ==============


            # out_batch.data[pe_idx].end_flag = 0;
            out_batch_end_flag_var = HLSVar(var_name="out_batch.data[pe_idx].end_flag", var_type=bool_type)
            assign_expr_10 = HLSExpr(HLSExprT.CONST, 0)
            for_loop_3_codes.append(CodeAssign(var=out_batch_end_flag_var, expr=assign_expr_10))
            for_loop_3_codes.append(CodeOther(text="")) # 空行

            # --- Build IF_3 (pe_idx == PE_NUM - 1) ---
            if_3_codes: List[HLSCodeLine] = []
            if_expr_3 = HLSExpr(HLSExprT.CONST, "pe_idx == (PE_NUM - 1)")
            # Build IF_3 Contents
            # last_cacheline[pe_idx] = cacheline;
            last_cacheline_pe_idx_var_if3 = HLSVar(var_name="last_cacheline[pe_idx]", var_type=bus_word_t_type)
            if_3_codes.append(CodeAssign(var=last_cacheline_pe_idx_var_if3, expr=HLSExpr(HLSExprT.VAR, cacheline_var)))
            # last_cache_idx[pe_idx] = cacheline_idx;
            last_cache_idx_pe_idx_var_if3 = HLSVar(var_name="last_cache_idx[pe_idx]", var_type=ap_uint26_type)
            if_3_codes.append(CodeAssign(var=last_cache_idx_pe_idx_var_if3, expr=HLSExpr(HLSExprT.VAR, cacheline_idx_var)))
            # Create IF_3
            if_3 = CodeIf(expr=if_expr_3, if_codes=if_3_codes)
            for_loop_3_codes.append(if_3)
            # --- End IF_3 ---

            # Create for loop 3
            for_loop_3 = CodeFor(codes=for_loop_3_codes,
                                iter_limit="PE_NUM",
                                iter_cmp="<",
                                iter_name="pe_idx",
                                iter_start="0",
                                iter_step="pe_idx++",
                                iter_val_type=int_type)
            for_loop_2_codes.append(for_loop_3)
            # --- End for(pe_idx) 2 ---

            # edge_batch_stream.write(out_batch);
            for_loop_2_codes.append(CodeWriteStream(stream_var=edge_batch_stream, in_expr=out_batch_var))
            for_loop_2_codes.append(CodeOther(text="")) # 空行

            # --- Build for(pe_idx) 3 (Broadcast loop) ---
            for_loop_4_codes: List[HLSCodeLine] = []
            # #pragma HLS UNROLL
            for_loop_4_codes.append(CodePragma(content="UNROLL"))
            # last_cacheline[pe_idx] = last_cacheline[PE_NUM - 1];
            last_cacheline_pe_idx_var_4 = HLSVar(var_name="last_cacheline[pe_idx]", var_type=bus_word_t_type)
            assign_expr_4_1 = HLSExpr(HLSExprT.CONST, "last_cacheline[PE_NUM - 1]")
            for_loop_4_codes.append(CodeAssign(var=last_cacheline_pe_idx_var_4, expr=assign_expr_4_1))
            # last_cache_idx[pe_idx] = last_cache_idx[PE_NUM - 1];
            last_cache_idx_pe_idx_var_4 = HLSVar(var_name="last_cache_idx[pe_idx]", var_type=ap_uint26_type)
            assign_expr_4_2 = HLSExpr(HLSExprT.CONST, "last_cache_idx[PE_NUM - 1]")
            for_loop_4_codes.append(CodeAssign(var=last_cache_idx_pe_idx_var_4, expr=assign_expr_4_2))
            # Create for loop 4
            for_loop_4 = CodeFor(codes=for_loop_4_codes,
                                iter_limit="(PE_NUM - 1)", # C++: pe_idx < (PE_NUM - 1)
                                iter_cmp="<",
                                iter_name="pe_idx",
                                iter_start="0",
                                iter_step="pe_idx++",
                                iter_val_type=int_type)
            for_loop_2_codes.append(for_loop_4)
            # --- End for(pe_idx) 3 ---

            # Create for loop 2
            for_loop_2 = CodeFor(codes=for_loop_2_codes,
                                iter_limit=scatter_size_var,
                                iter_cmp="<",
                                iter_name="edge_batch_idx",
                                iter_start="0",
                                iter_step="edge_batch_idx++",
                                iter_val_type=int_type)
            code_lines.append(for_loop_2)
            # --- End for(edge_batch_idx) ---

            # --- 3. Finalize ---
            merge_node_props_func.codes = code_lines
            self.big_scatter_funcs.append(merge_node_props_func)
            self.big_top_dataflow_funcs.append(merge_node_props_func)


            request_manager_func = HLSFunction(name="request_manager", comp=None) # 假设 comp 存在
            params = []

            # --- 1. 定义类型和参数 (根据 graphyflow_little.h) ---

            # 基础类型
            node_id_type = HLSType(HLSBasicType.NODE_ID) # ap_uint<32>
            bus_word_t_type = HLSType(HLSBasicType.BUS_WORD_T) # ap_uint<512>
            ap_fixed_pod_t_type = HLSType(HLSBasicType.AP_FIXED_POD) # ap_uint<32>
            distance_t_type = HLSType(HLSBasicType.DISTANCE_T) # ap_fixed<32, 16>
            int_type = HLSType(HLSBasicType.INT)       # int32_t
            uint_type = HLSType(HLSBasicType.UINT)      # uint32_t / ap_uint<32>
            bool_type = HLSType(HLSBasicType.BOOL)

            # C++ 中使用的特定 ap_uint 类型
            ap_uint22_type = HLSType(basic_type=HLSBasicType.AP_UINT, width=22)
            ap_uint20_type = HLSType(basic_type=HLSBasicType.AP_UINT, width=20) # 来自 .h (edge_t.dst_id)
            ap_uint12_type = HLSType(basic_type=HLSBasicType.AP_UINT, width=12)
            ap_uint8_type = HLSType(basic_type=HLSBasicType.AP_UINT, width=8)
            ap_uint4_type = HLSType(basic_type=HLSBasicType.AP_UINT, width=4)
            ap_uint9_type = HLSType(basic_type=HLSBasicType.AP_UINT, width=9)

            # --- 结构体定义 (严格按照 graphyflow_little.h) ---

            # struct edge_t
            edge_t_type = HLSType(basic_type=HLSBasicType.STRUCT,
                                  struct_name="edge_t",
                                  struct_prop_names=["src_id", "dst_id"],
                                  sub_types=[node_id_type, ap_uint20_type])
            if edge_t_type.name not in self.struct_definitions:
                self.struct_definitions[edge_t_type.name] = (edge_t_type, edge_t_type.struct_prop_names)

            # struct edge_descriptor_batch_t
            edge_array_type = HLSType(HLSBasicType.ARRAY, sub_types=[edge_t_type], array_dims=["PE_NUM"])
            edge_descriptor_batch_t_type = HLSType(basic_type=HLSBasicType.STRUCT,
                                                   struct_name="edge_descriptor_batch_t",
                                                   struct_prop_names=["edges"],
                                                   sub_types=[edge_array_type])
            if edge_descriptor_batch_t_type.name not in self.struct_definitions:
                self.struct_definitions[edge_descriptor_batch_t_type.name] = (edge_descriptor_batch_t_type, edge_descriptor_batch_t_type.struct_prop_names)

            # struct update_t
            update_t_type = HLSType(basic_type=HLSBasicType.STRUCT,
                                    struct_name="update_t_little",
                                    struct_prop_names=["node_id", "prop"],
                                    sub_types=[HLSType(HLSBasicType.AP_UINT,width=20), HLSType(HLSBasicType.AP_FIXED_POD)]) # .h: node_id 是 ap_uint<20>
            if update_t_type.name not in self.struct_definitions:
                self.struct_definitions[update_t_type.name] = (update_t_type, update_t_type.struct_prop_names)

            # struct update_tuple_t
            update_t_array_type = HLSType(HLSBasicType.ARRAY, sub_types=[update_t_type], array_dims=["PE_NUM"])
            update_tuple_t_type = HLSType(basic_type=HLSBasicType.STRUCT,
                                          struct_name="update_tuple_t_little",
                                          struct_prop_names=["data"],
                                          sub_types=[update_t_array_type])
            if update_tuple_t_type.name not in self.struct_definitions:
                self.struct_definitions[update_tuple_t_type.name] = (update_tuple_t_type, update_tuple_t_type.struct_prop_names)

            # struct ppb_request_t
            ppb_request_t_type = HLSType(basic_type=HLSBasicType.STRUCT,
                                         struct_name="ppb_request_t",
                                         struct_prop_names=["request_round", "end_flag"],
                                         sub_types=[HLSType(HLSBasicType.AP_UINT,width=32), bool_type]) # .h: ap_uint<32>, bool
            if ppb_request_t_type.name not in self.struct_definitions:
                self.struct_definitions[ppb_request_t_type.name] = (ppb_request_t_type, ppb_request_t_type.struct_prop_names)

            # struct ppb_response_t
            ppb_response_t_type = HLSType(basic_type=HLSBasicType.STRUCT,
                                          struct_name="ppb_response_t",
                                          struct_prop_names=["data", "addr", "end_flag"],
                                          sub_types=[bus_word_t_type, HLSType(HLSBasicType.AP_UINT,width=32), bool_type]) # .h: bus_word_t, ap_uint<32>, bool
            if ppb_response_t_type.name not in self.struct_definitions:
                self.struct_definitions[ppb_response_t_type.name] = (ppb_response_t_type, ppb_response_t_type.struct_prop_names)

            # --- Stream 类型 ---
            edge_burst_stm_type = HLSType(HLSBasicType.STREAM, sub_types=[edge_descriptor_batch_t_type])
            ppb_request_stm_type = HLSType(HLSBasicType.STREAM, sub_types=[ppb_request_t_type])
            ppb_response_stm_type = HLSType(HLSBasicType.STREAM, sub_types=[ppb_response_t_type])
            update_set_stm_type = HLSType(HLSBasicType.STREAM, sub_types=[update_tuple_t_type])

            # --- 参数变量 ---
            edge_burst_stm_var = HLSVar(var_name="edge_burst_stm", var_type=edge_burst_stm_type)
            ppb_request_stm_var = HLSVar(var_name="ppb_request_stm", var_type=ppb_request_stm_type)
            ppb_response_stm_var = HLSVar(var_name="ppb_response_stm", var_type=ppb_response_stm_type)
            update_set_stm_var = HLSVar(var_name="update_set_stm", var_type=update_set_stm_type)
            memory_offset_var = HLSVar(var_name="memory_offset", var_type=uint_type)
            part_edge_num_var = HLSVar(var_name="part_edge_num", var_type=uint_type)

            params.extend([edge_burst_stm_var, ppb_request_stm_var, ppb_response_stm_var, 
                            update_set_stm_var, memory_offset_var, part_edge_num_var])
            request_manager_func.params = params

            # --- 2. 函数体 ---
            code_lines: List[HLSCodeLine] = []

            # // as we can buffer two vertices...
            code_lines.append(CodeComment(text="as we can buffer two vertices in one row with width of 64-bit, we can let"))
            code_lines.append(CodeComment(text="the depth go as MAX_VERTICES_IN_ONE_PARTITION / 2."))

            # bus_word_t src_prop_buffer[PE_NUM][2][SRC_BUFFER_SIZE >> 4];
            # (SRC_BUFFER_SIZE >> 4) == (4096 >> 4) == 256
            src_buffer_elem_type = HLSType(HLSBasicType.ARRAY, sub_types=[bus_word_t_type], array_dims=["(SRC_BUFFER_SIZE >> 4)"])
            src_buffer_mid_type = HLSType(HLSBasicType.ARRAY, sub_types=[src_buffer_elem_type], array_dims=[2])
            src_prop_buffer_type = HLSType(HLSBasicType.ARRAY, sub_types=[src_buffer_mid_type], array_dims=["PE_NUM"])
            code_lines.append(CodeVarDecl(var_name="src_prop_buffer", var_type=src_prop_buffer_type))
            src_prop_buffer_var = HLSVar(var_name="src_prop_buffer", var_type=src_prop_buffer_type)

            # #pragma HLS ARRAY_PARTITION ...
            code_lines.append(CodePragma(content="ARRAY_PARTITION variable = src_prop_buffer dim = 1 complete"))
            code_lines.append(CodePragma(content="BIND_STORAGE variable = src_prop_buffer type = RAM_S2P impl = BRAM"))
            code_lines.append(CodePragma(content="dependence variable = src_prop_buffer inter false"))
            code_lines.append(CodeOther(text=""))

            # ap_uint<22> pp_read_round = 0;
            code_lines.append(CodeVarDecl(var_name="pp_read_round", var_type=ap_uint22_type, init_val="0"))
            pp_read_round_var = HLSVar(var_name="pp_read_round", var_type=ap_uint22_type)
            # ap_uint<22> pp_write_round = 0;
            code_lines.append(CodeVarDecl(var_name="pp_write_round", var_type=ap_uint22_type, init_val="0"))
            pp_write_round_var = HLSVar(var_name="pp_write_round", var_type=ap_uint22_type)
            code_lines.append(CodeOther(text=""))
            # ap_uint<22> pp_request_round = 0;
            code_lines.append(CodeVarDecl(var_name="pp_request_round", var_type=ap_uint22_type, init_val="0"))
            pp_request_round_var = HLSVar(var_name="pp_request_round", var_type=ap_uint22_type)
            code_lines.append(CodeOther(text=""))
            # int32_t edge_set_cnt = 0;
            code_lines.append(CodeVarDecl(var_name="edge_set_cnt", var_type=int_type, init_val="0"))
            edge_set_cnt_var = HLSVar(var_name="edge_set_cnt", var_type=int_type)
            # const int32_t total_edge_sets = part_edge_num >> LOG_PE_NUM;
            code_lines.append(CodeVarDecl(var_name="total_edge_sets", var_type=int_type, init_val="(part_edge_num >> LOG_PE_NUM)", const=True))
            total_edge_sets_var = HLSVar(var_name="total_edge_sets", var_type=int_type)
            code_lines.append(CodeOther(text=""))
            # bool wait_flag = 0;
            code_lines.append(CodeVarDecl(var_name="wait_flag", var_type=bool_type, init_val="0"))
            wait_flag_var = HLSVar(var_name="wait_flag", var_type=bool_type)
            code_lines.append(CodeOther(text=""))
            # edge_descriptor_batch_t an_edge_burst;
            code_lines.append(CodeVarDecl(var_name="an_edge_burst", var_type=edge_descriptor_batch_t_type))
            an_edge_burst_var = HLSVar(var_name="an_edge_burst", var_type=edge_descriptor_batch_t_type)
            code_lines.append(CodeOther(text=""))
            # distance_t real_edge_weight = 1.0;
            code_lines.append(CodeVarDecl(var_name="real_edge_weight", var_type=distance_t_type, init_val="1.0"))
            code_lines.append(CodeComment(text="All edge weights are 1.0 in unweighted graph"))
            # const ap_fixed_pod_t edge_weight = ...
            edge_weight_init_val = "(*reinterpret_cast<ap_fixed_pod_t *>(&real_edge_weight))"
            code_lines.append(CodeVarDecl(var_name="edge_weight", var_type=ap_fixed_pod_t_type, init_val=edge_weight_init_val, const=True))
            edge_weight_var = HLSVar(var_name="edge_weight", var_type=ap_fixed_pod_t_type)
            code_lines.append(CodeOther(text=""))

            # scatterLoop:
            code_lines.append(CodeOther(text="scatterLoop:"))
            # while (true)
            while_1_codes: List[HLSCodeLine] = []
            while_1_expr = HLSExpr(HLSExprT.CONST, True)

            # #pragma HLS PIPELINE II = 1
            while_1_codes.append(CodePragma(content="PIPELINE II = 1"))
            # // logic to fill the ping-pong buffer.
            while_1_codes.append(CodeComment(text="logic to fill the ping-pong buffer."))

            # if ((pp_request_round - pp_read_round) <= 1)
            if_1_codes: List[HLSCodeLine] = []
            if_1_expr = HLSExpr(HLSExprT.CONST, "(pp_request_round - pp_read_round) <= 1")

            # if (pp_request_round < pp_read_round)
            if_2_codes: List[HLSCodeLine] = []
            if_2_expr = HLSExpr(HLSExprT.CONST, "pp_request_round < pp_read_round")
            # pp_request_round = pp_read_round;
            if_2_codes.append(CodeAssign(var=pp_request_round_var, expr=HLSExpr(HLSExprT.VAR, pp_read_round_var)))
            if_1_codes.append(CodeIf(expr=if_2_expr, if_codes=if_2_codes))

            # ppb_request_t one_ppb_request;
            if_1_codes.append(CodeVarDecl(var_name="one_ppb_request", var_type=ppb_request_t_type))
            one_ppb_request_var = HLSVar(var_name="one_ppb_request", var_type=ppb_request_t_type)
            # one_ppb_request.request_round = pp_request_round + memory_offset;
            req_round_var = HLSVar(var_name="one_ppb_request.request_round", var_type=uint_type) # 匹配 .h
            req_round_expr = HLSExpr(HLSExprT.BINOP, dfir.BinOp.ADD,
                                     operands=[HLSExpr(HLSExprT.VAR, pp_request_round_var),
                                               HLSExpr(HLSExprT.VAR, memory_offset_var)])
            if_1_codes.append(CodeAssign(var=req_round_var, expr=req_round_expr))
            # one_ppb_request.end_flag = 0;
            req_end_flag_var = HLSVar(var_name="one_ppb_request.end_flag", var_type=bool_type) # 匹配 .h
            if_1_codes.append(CodeAssign(var=req_end_flag_var, expr=HLSExpr(HLSExprT.CONST, 0)))
            # ppb_request_stm.write(one_ppb_request);
            if_1_codes.append(CodeWriteStream(stream_var=ppb_request_stm_var, in_expr=one_ppb_request_var))
            # pp_request_round++;
            pp_req_plus_expr = HLSExpr(HLSExprT.BINOP, dfir.BinOp.ADD,
                                       operands=[HLSExpr(HLSExprT.VAR, pp_request_round_var),
                                                 HLSExpr(HLSExprT.CONST, 1)])
            if_1_codes.append(CodeAssign(var=pp_request_round_var, expr=pp_req_plus_expr))
            # (构建 if_1)
            while_1_codes.append(CodeIf(expr=if_1_expr, if_codes=if_1_codes))
            while_1_codes.append(CodeOther(text=""))

            # ppb_response_t one_ppb_response;
            while_1_codes.append(CodeVarDecl(var_name="one_ppb_response", var_type=ppb_response_t_type))
            one_ppb_response_var = HLSVar(var_name="one_ppb_response", var_type=ppb_response_t_type)
            # if (ppb_response_stm.read_nb(one_ppb_response))
            if_3_codes: List[HLSCodeLine] = []
            if_3_expr = HLSExpr(HLSExprT.CONST, "ppb_response_stm.read_nb(one_ppb_response)")

            # pp_write_round = ...
            # (LOG_SRC_BUFFER_SIZE == 12)
            assign_expr_3 = HLSExpr(HLSExprT.CONST, "(one_ppb_response.addr << 4 >> LOG_SRC_BUFFER_SIZE) - memory_offset")
            if_3_codes.append(CodeAssign(var=pp_write_round_var, expr=assign_expr_3))
            while_1_codes.append(CodeOther(text=""))
            # bool write_buffer = pp_write_round.range(0, 0);
            if_3_codes.append(CodeVarDecl(var_name="write_buffer", var_type=bool_type, init_val="pp_write_round.range(0, 0)"))
            write_buffer_var = HLSVar(var_name="write_buffer", var_type=bool_type)

            # ap_uint<8> write_idx = one_ppb_response.addr.range(7, 0);
            if_3_codes.append(CodeVarDecl(var_name="write_idx", var_type=ap_uint8_type, init_val="one_ppb_response.addr.range(7, 0)"))
            write_idx_var = HLSVar(var_name="write_idx", var_type=ap_uint8_type)
            # // one_ppb_response.addr & ...
            if_3_codes.append(CodeComment(text="one_ppb_response.addr & ((SRC_BUFFER_SIZE >> 4) - 1);"))
            if_3_codes.append(CodeComment(text="// 4096 >> 4 = 256 - 1 = 255 = 2^8 -1"))

            # bus_word_t one_read_burst = one_ppb_response.data;
            if_3_codes.append(CodeVarDecl(var_name="one_read_burst", var_type=bus_word_t_type, init_val="one_ppb_response.data"))
            one_read_burst_var = HLSVar(var_name="one_read_burst", var_type=bus_word_t_type)

            # for (int u = 0; u < PE_NUM; u++)
            for_1_codes: List[HLSCodeLine] = []
            # #pragma HLS UNROLL
            for_1_codes.append(CodePragma(content="UNROLL"))
            # src_prop_buffer[u][write_buffer][write_idx] = one_read_burst;
            assign_target_var = HLSVar(var_name="src_prop_buffer[u][write_buffer][write_idx]", var_type=bus_word_t_type)
            for_1_codes.append(CodeAssign(var=assign_target_var, expr=HLSExpr(HLSExprT.VAR, one_read_burst_var)))
            # (构建 for_1)
            if_3_codes.append(CodeFor(codes=for_1_codes, iter_limit="PE_NUM", iter_name="u", iter_val_type=int_type))
            # (构建 if_3)
            while_1_codes.append(CodeIf(expr=if_3_expr, if_codes=if_3_codes))


            # // logic to read the ping-pong buffer and synchronization.
            while_1_codes.append(CodeComment(text="logic to read the ping-pong buffer and synchronization."))
            # if (!wait_flag)
            if_4_codes: List[HLSCodeLine] = []
            if_4_expr = HLSExpr(HLSExprT.UOP, dfir.UnaryOp.NOT, operands=[HLSExpr(HLSExprT.VAR, wait_flag_var)])
            # an_edge_burst = edge_burst_stm.read();
            read_expr_2 = HLSExpr(HLSExprT.STREAM_READ, None, operands=[HLSExpr(HLSExprT.VAR, edge_burst_stm_var)])
            if_4_codes.append(CodeAssign(var=an_edge_burst_var, expr=read_expr_2))
            # (构建 if_4)
            while_1_codes.append(CodeIf(expr=if_4_expr, if_codes=if_4_codes))
            while_1_codes.append(CodeOther(text=""))
            # pp_read_round = (an_edge_burst.edges[0].src_id / SRC_BUFFER_SIZE);
            assign_expr_4 = HLSExpr(HLSExprT.CONST, "(an_edge_burst.edges[0].src_id / SRC_BUFFER_SIZE)")
            while_1_codes.append(CodeAssign(var=pp_read_round_var, expr=assign_expr_4))

            # wait_flag = (pp_read_round >= pp_write_round) ? 1 : 0;
            assign_expr_5 = HLSExpr(HLSExprT.CONST, "(pp_read_round >= pp_write_round) ? 1 : 0")
            while_1_codes.append(CodeAssign(var=wait_flag_var, expr=assign_expr_5))

            # bool exit_flag = ...
            exit_flag_init_val = "(wait_flag == 0) ? (edge_set_cnt + 1 >= total_edge_sets) : (edge_set_cnt >= total_edge_sets)"
            while_1_codes.append(CodeVarDecl(var_name="exit_flag", var_type=bool_type, init_val=exit_flag_init_val))
            exit_flag_var = HLSVar(var_name="exit_flag", var_type=bool_type)


            # if (!wait_flag)
            if_5_codes: List[HLSCodeLine] = []
            if_5_expr = HLSExpr(HLSExprT.UOP, dfir.UnaryOp.NOT, operands=[HLSExpr(HLSExprT.VAR, wait_flag_var)])

            # bool read_buffer = pp_read_round.range(0, 0);
            if_5_codes.append(CodeVarDecl(var_name="read_buffer", var_type=bool_type, init_val="pp_read_round.range(0, 0)"))
            read_buffer_var = HLSVar(var_name="read_buffer", var_type=bool_type)

            # update_tuple_t an_update_set;
            if_5_codes.append(CodeVarDecl(var_name="an_update_set", var_type=update_tuple_t_type))
            an_update_set_var = HLSVar(var_name="an_update_set", var_type=update_tuple_t_type)

            # for (int u = 0; u < PE_NUM; u++)
            for_2_codes: List[HLSCodeLine] = []
            # #pragma HLS UNROLL
            for_2_codes.append(CodePragma(content="UNROLL"))
            # ap_uint<12> idx = an_edge_burst.edges[u].src_id.range(11, 0);
            for_2_codes.append(CodeVarDecl(var_name="idx", var_type=ap_uint12_type, init_val="an_edge_burst.edges[u].src_id.range(11, 0)"))
            idx_var = HLSVar(var_name="idx", var_type=ap_uint12_type)
            # ap_uint<8> uram_row_idx = idx.range(11, 4);
            for_2_codes.append(CodeVarDecl(var_name="uram_row_idx", var_type=ap_uint8_type, init_val="idx.range(11, 4)"))
            uram_row_idx_var = HLSVar(var_name="uram_row_idx", var_type=ap_uint8_type)
            # ap_uint<4> uram_row_offset = idx.range(3, 0);
            for_2_codes.append(CodeVarDecl(var_name="uram_row_offset", var_type=ap_uint4_type, init_val="idx.range(3, 0)"))
            uram_row_offset_var = HLSVar(var_name="uram_row_offset", var_type=ap_uint4_type)
            while_1_codes.append(CodeOther(text=""))
            # bus_word_t uram_row = src_prop_buffer[u][read_buffer][uram_row_idx];
            uram_row_init_val = "src_prop_buffer[u][read_buffer][uram_row_idx]"
            for_2_codes.append(CodeVarDecl(var_name="uram_row", var_type=bus_word_t_type, init_val=uram_row_init_val))
            uram_row_var = HLSVar(var_name="uram_row", var_type=bus_word_t_type)
            # ap_fixed_pod_t src_prop = ...
            src_prop_init_val = "uram_row.range(31 + ((ap_uint<9>)uram_row_offset << 5), ((ap_uint<9>)uram_row_offset << 5))"
            for_2_codes.append(CodeVarDecl(var_name="src_prop", var_type=ap_fixed_pod_t_type, init_val=src_prop_init_val))
            src_prop_var = HLSVar(var_name="src_prop", var_type=ap_fixed_pod_t_type)
            # ap_fixed_pod_t update = src_prop + edge_weight;
            update_expr = HLSExpr(HLSExprT.BINOP, dfir.BinOp.ADD,
                                  operands=[HLSExpr(HLSExprT.VAR, src_prop_var),
                                            HLSExpr(HLSExprT.VAR, edge_weight_var)])
            #for_2_codes.append(CodeVarDecl(var_name="update", var_type=ap_fixed_pod_t_type, init_val=update_expr.code))
            #update_var = HLSVar(var_name="update", var_type=ap_fixed_pod_t_type)
           
            # an_update_set.data[u].node_id = an_edge_burst.edges[u].dst_id;
            update_node_id_var = HLSVar(var_name="an_update_set.data[u].node_id", var_type=ap_uint20_type)
            dst_id_var = HLSVar(var_name="an_edge_burst.edges[u].dst_id", var_type=ap_uint20_type)
            # assign_expr_10 = HLSExpr(HLSExprT.CONST, "an_edge_burst.edges[u].dst_id")
            # for_2_codes.append(CodeAssign(var=update_node_id_var, expr=assign_expr_10))
            # an_update_set.data[u].prop = update;
            update_prop_var = HLSVar(var_name="an_update_set.data[u].prop", var_type=ap_fixed_pod_t_type)
            # for_2_codes.append(CodeAssign(var=update_prop_var, expr=HLSExpr(HLSExprT.VAR, update_var)))
            # (构建 for_2)
            
            #============= begin inline logic ===============
            port_to_var = {}
            for_2_codes.append(CodeOther(text="// Begin inline logic"))
            
            for_2_codes.append(CodeVarDecl(var_name="DST_ID_VAR", var_type=node_id_type, init_val="an_edge_burst.edges[u].dst_id"))

            top_vars = {
                "DST_ID_VAR": dst_id_var,
                "SRC_PROP_VAR": src_prop_var,
                "EDGE_WEIGHT_VAR": edge_weight_var
            }

            top_vars["FINAL_PROP_VAR"] = update_prop_var
            top_vars["FINAL_DST_ID_VAR"] = update_node_id_var
            port_property = {} # 只能是src , dst, edge_prop这三项或组合
            inlinecodes = []
            for comp in scatter_stage_comps:
                port_property,inlinecodes = self._scatter_analyze(comp,port_property,port_to_var,top_vars,target_codes=inlinecodes)
            for_2_codes.extend(inlinecodes)
            for_2_codes.append(CodeOther(text="// End inline logic"))
            # =========== end inline logic ==============
            
            
            if_5_codes.append(CodeFor(codes=for_2_codes, iter_limit="PE_NUM", iter_name="u", iter_val_type=int_type))
            # update_set_stm.write(an_update_set);
            if_5_codes.append(CodeWriteStream(stream_var=update_set_stm_var, in_expr=an_update_set_var))
            while_1_codes.append(CodeOther(text=""))
            # edge_set_cnt++;
            edge_set_cnt_plus_expr = HLSExpr(HLSExprT.BINOP, dfir.BinOp.ADD,
                                             operands=[HLSExpr(HLSExprT.VAR, edge_set_cnt_var),
                                                       HLSExpr(HLSExprT.CONST, 1)])
            if_5_codes.append(CodeAssign(var=edge_set_cnt_var, expr=edge_set_cnt_plus_expr))
            # (构建 if_5)
            while_1_codes.append(CodeIf(expr=if_5_expr, if_codes=if_5_codes))
            while_1_codes.append(CodeOther(text=""))

            # if (exit_flag)
            if_6_codes: List[HLSCodeLine] = []
            if_6_expr = HLSExpr(HLSExprT.VAR, exit_flag_var)

            # ppb_request_t one_ppb_request;
            if_6_codes.append(CodeVarDecl(var_name="one_ppb_request", var_type=ppb_request_t_type))
            one_ppb_request_var_inner = HLSVar(var_name="one_ppb_request", var_type=ppb_request_t_type)
            # one_ppb_request.end_flag = 1;
            req_end_flag_var_inner = HLSVar(var_name="one_ppb_request.end_flag", var_type=bool_type)
            if_6_codes.append(CodeAssign(var=req_end_flag_var_inner, expr=HLSExpr(HLSExprT.CONST, 1)))
            # ppb_request_stm.write(one_ppb_request);
            if_6_codes.append(CodeWriteStream(stream_var=ppb_request_stm_var, in_expr=one_ppb_request_var_inner))
            # exitscatter:
            if_6_codes.append(CodeOther(text="exitscatter:"))
            # while (true)
            while_2_codes: List[HLSCodeLine] = []
            while_2_expr = HLSExpr(HLSExprT.CONST, True)
            # ppb_response_stm.read(one_ppb_response);
            read_expr_3 = HLSExpr(HLSExprT.STREAM_READ, None, operands=[HLSExpr(HLSExprT.VAR, ppb_response_stm_var)])
            while_2_codes.append(CodeAssign(var=one_ppb_response_var, expr=read_expr_3))
            # if (one_ppb_response.end_flag)
            if_7_codes: List[HLSCodeLine] = []
            if_7_expr = HLSExpr(HLSExprT.UOP, (dfir.UnaryOp.GET_ATTR, "end_flag"),
                                 operands=[HLSExpr(HLSExprT.VAR, one_ppb_response_var)])
            # break;
            if_7_codes.append(CodeBreak())
            # (构建 if_7)
            while_2_codes.append(CodeIf(expr=if_7_expr, if_codes=if_7_codes))
            # (构建 while_2)
            if_6_codes.append(CodeWhile(codes=while_2_codes, iter_expr=while_2_expr))
            # break;
            if_6_codes.append(CodeBreak())
            # (构建 if_6)
            while_1_codes.append(CodeIf(expr=if_6_expr, if_codes=if_6_codes))

            # (构建 while_1)
            code_lines.append(CodeWhile(codes=while_1_codes, iter_expr=while_1_expr))

            # --- 3. Finalize ---
            request_manager_func.codes = code_lines

            self.little_scatter_funcs.append(request_manager_func)
            self.little_top_dataflow_funcs.append(request_manager_func)
#             ## Little part:
# 
# 
#             request_manager_func = HLSFunction(name="request_manager", comp=comp)
#             params_req: List[HLSVar] = []
# 
#             # --- 1. 定义类型和参数 ---
# 
#             # 基本类型
#             int_type = HLSType(HLSBasicType.INT)
#             uint_type = HLSType(HLSBasicType.UINT)
#             uint8_type = HLSType(HLSBasicType.UINT8)
#             bool_type = HLSType(HLSBasicType.BOOL)
# 
#             # Typedefs (来自 HLSBasicType)
#             bus_word_t_type = HLSType(HLSBasicType.BUS_WORD_T)
#             distance_t_type = HLSType(HLSBasicType.DISTANCE_T)
#             ap_fixed_pod_t_type = HLSType(HLSBasicType.AP_FIXED_POD)
#             ppb_request_pkt_t_type = HLSType(HLSBasicType.PPB_REQUEST_PKT_T)
#             ppb_response_pkt_t_type = HLSType(HLSBasicType.PPB_RESPONSE_PKT_T)
#             node_id_t_type = HLSType(HLSBasicType.NODE_ID)
# 
#             # --- 详细的 Struct 定义 (基于 graphyflow_little.h) ---
# 
#             # 依赖: struct edge_t
#             edge_t_sub_types = [node_id_t_type, node_id_t_type]
#             edge_t_prop_names = ["src_id", "dst_id"]
#             edge_t_type = HLSType(HLSBasicType.STRUCT, 
#                                   sub_types=edge_t_sub_types, 
#                                   struct_name="edge_t", 
#                                   struct_prop_names=edge_t_prop_names)
#             if edge_t_type.name not in self.struct_definitions:
#                 self.struct_definitions[edge_t_type.name] = (
#                     edge_t_type,
#                     edge_t_type.struct_prop_names,
#                 )
#             # 1. struct edge_descriptor_batch_t
#             edge_array_type = HLSType(HLSBasicType.ARRAY, sub_types=[edge_t_type], array_dims=["PE_NUM"])
#             edge_descriptor_batch_t_type = HLSType(basic_type=HLSBasicType.STRUCT,
#                                                 struct_name="edge_descriptor_batch_t",
#                                                 struct_prop_names=["edges"], # .h: 只有 'edges'
#                                                 sub_types=[edge_array_type])
#             if edge_descriptor_batch_t_type.name not in self.struct_definitions:
#                 self.struct_definitions[edge_descriptor_batch_t_type.name] = (
#                     edge_descriptor_batch_t_type,
#                     edge_descriptor_batch_t_type.struct_prop_names,
#                 )
# 
#             # 2. struct update_tuple_t
#             node_id_array_type = HLSType(HLSBasicType.ARRAY, sub_types=[node_id_t_type], array_dims=["PE_NUM"])
#             prop_array_type = HLSType(HLSBasicType.ARRAY, sub_types=[ap_fixed_pod_t_type], array_dims=["PE_NUM"])
#             update_tuple_sub_types = [node_id_array_type, prop_array_type, bool_type, uint8_type]
#             update_tuple_prop_names = ["node_id", "prop", "end_flag", "end_pos"]
#             update_tuple_t_type = HLSType(HLSBasicType.STRUCT, 
#                                           sub_types=update_tuple_sub_types, 
#                                           struct_name="update_tuple_t", 
#                                           struct_prop_names=update_tuple_prop_names)
#             if update_tuple_t_type.name not in self.struct_definitions:
#                 self.struct_definitions[update_tuple_t_type.name] = (
#                     update_tuple_t_type,
#                     update_tuple_t_type.struct_prop_names,
#                 )
#             # --- 结束 Struct 定义 ---
# 
# 
#             # Param 1: hls::stream<edge_descriptor_batch_t> &edge_burst_stm
#             edge_burst_stm_type = HLSType(HLSBasicType.STREAM, sub_types=[edge_descriptor_batch_t_type])
#             param_edge_burst_stm = HLSVar(var_name="edge_burst_stm", var_type=edge_burst_stm_type)
# 
#             # Param 2: hls::stream<ppb_request_pkt_t> &ppb_request_stm
#             ppb_request_stm_type = HLSType(HLSBasicType.STREAM, sub_types=[ppb_request_pkt_t_type])
#             param_ppb_request_stm = HLSVar(var_name="ppb_request_stm", var_type=ppb_request_stm_type)
# 
#             # Param 3: hls::stream<ppb_response_pkt_t> &ppb_response_stm
#             ppb_response_stm_type = HLSType(HLSBasicType.STREAM, sub_types=[ppb_response_pkt_t_type])
#             param_ppb_response_stm = HLSVar(var_name="ppb_response_stm", var_type=ppb_response_stm_type)
# 
#             # Param 4: hls::stream<update_tuple_t> &update_set_stm
#             update_set_stm_type = HLSType(HLSBasicType.STREAM, sub_types=[update_tuple_t_type])
#             param_update_set_stm = HLSVar(var_name="update_set_stm", var_type=update_set_stm_type)
# 
#             # Param 5: int32_t part_edge_num
#             param_part_edge_num = HLSVar(var_name="part_edge_num", var_type=int_type)
# 
#             params_req.extend([param_edge_burst_stm, param_ppb_request_stm, param_ppb_response_stm, param_update_set_stm, param_part_edge_num])
#             request_manager_func.params = params_req
# 
#             # --- 2. 函数体 (构建本地列表) ---
#             code_lines_req: List[HLSCodeLine] = []
# 
#             # bus_word_t src_prop_buffer[PE_NUM][2][SRC_BUFFER_SIZE >> 4];
#             src_buffer_dims = ["PE_NUM", "2", "(SRC_BUFFER_SIZE >> 4)"] # 宏/表达式作为字符串
#             src_prop_buffer_type = HLSType(HLSBasicType.ARRAY, sub_types=[bus_word_t_type], array_dims=src_buffer_dims)
#             code_lines_req.append(CodeVarDecl(var_name="src_prop_buffer", var_type=src_prop_buffer_type))
# 
#             # Pragmas for src_prop_buffer
#             code_lines_req.append(CodePragma(content="ARRAY_PARTITION variable = src_prop_buffer dim = 1 complete"))
#             code_lines_req.append(CodePragma(content="BIND_STORAGE variable = src_prop_buffer type = RAM_S2P impl = BRAM"))
#             code_lines_req.append(CodePragma(content="dependence variable = src_prop_buffer inter false"))
# 
#             # int32_t pp_read_idx = 0;
#             code_lines_req.append(CodeVarDecl(var_name="pp_read_idx", var_type=int_type, init_val="0"))
#             pp_read_idx_var = HLSVar(var_name="pp_read_idx", var_type=int_type)
# 
#             # int32_t pp_write_idx = 0;
#             code_lines_req.append(CodeVarDecl(var_name="pp_write_idx", var_type=int_type, init_val="0"))
#             pp_write_idx_var = HLSVar(var_name="pp_write_idx", var_type=int_type)
# 
#             # int32_t pp_reponse_idx = 0;
#             code_lines_req.append(CodeVarDecl(var_name="pp_reponse_idx", var_type=int_type, init_val="0"))
#             pp_reponse_idx_var = HLSVar(var_name="pp_reponse_idx", var_type=int_type)
# 
#             # int32_t pp_read_round = 0;
#             code_lines_req.append(CodeVarDecl(var_name="pp_read_round", var_type=int_type, init_val="0"))
#             pp_read_round_var = HLSVar(var_name="pp_read_round", var_type=int_type)
# 
#             # int32_t pp_write_round = 0;
#             code_lines_req.append(CodeVarDecl(var_name="pp_write_round", var_type=int_type, init_val="0"))
#             pp_write_round_var = HLSVar(var_name="pp_write_round", var_type=int_type)
# 
#             # int32_t pp_request_round = 0;
#             code_lines_req.append(CodeVarDecl(var_name="pp_request_round", var_type=int_type, init_val="0"))
#             pp_request_round_var = HLSVar(var_name="pp_request_round", var_type=int_type)
# 
#             # int32_t edge_set_cnt = 0;
#             code_lines_req.append(CodeVarDecl(var_name="edge_set_cnt", var_type=int_type, init_val="0"))
#             edge_set_cnt_var = HLSVar(var_name="edge_set_cnt", var_type=int_type)
# 
#             # const int32_t total_edge_sets = (part_edge_num + PE_NUM - 1) / PE_NUM;
#             code_lines_req.append(CodeVarDecl(var_name="total_edge_sets", var_type=int_type, init_val="(part_edge_num + PE_NUM - 1) / PE_NUM", const=True))
#             total_edge_sets_var = HLSVar(var_name="total_edge_sets", var_type=int_type)
# 
#             # bool wait_flag = 0;
#             code_lines_req.append(CodeVarDecl(var_name="wait_flag", var_type=bool_type, init_val="0"))
#             wait_flag_var = HLSVar(var_name="wait_flag", var_type=bool_type)
# 
#             # edge_descriptor_batch_t an_edge_burst;
#             code_lines_req.append(CodeVarDecl(var_name="an_edge_burst", var_type=edge_descriptor_batch_t_type))
#             an_edge_burst_var = HLSVar(var_name="an_edge_burst", var_type=edge_descriptor_batch_t_type)
# 
#             #pragma HLS ARRAY_PARTITION variable = an_edge_burst.edges complete dim = 0
#             code_lines_req.append(CodePragma(content="ARRAY_PARTITION variable = an_edge_burst.edges complete dim = 0"))
# 
#             # ppb_request_pkt_t one_ppb_request;
#             code_lines_req.append(CodeVarDecl(var_name="one_ppb_request", var_type=ppb_request_pkt_t_type))
#             one_ppb_request_var = HLSVar(var_name="one_ppb_request", var_type=ppb_request_pkt_t_type)
# 
#             # ppb_response_pkt_t one_ppb_response;
#             code_lines_req.append(CodeVarDecl(var_name="one_ppb_response", var_type=ppb_response_pkt_t_type))
#             one_ppb_response_var = HLSVar(var_name="one_ppb_response", var_type=ppb_response_pkt_t_type)
# 
#             # distance_t real_edge_weight = 1.0;
#             code_lines_req.append(CodeVarDecl(var_name="real_edge_weight", var_type=distance_t_type, init_val="1.0"))
#             real_edge_weight_var = HLSVar(var_name="real_edge_weight", var_type=distance_t_type)
# 
#             # const ap_fixed_pod_t edge_weight = (*reinterpret_cast<ap_fixed_pod_t *>(&real_edge_weight));
#             code_lines_req.append(CodeVarDecl(var_name="edge_weight", var_type=ap_fixed_pod_t_type, init_val="(*reinterpret_cast<ap_fixed_pod_t *>(&real_edge_weight))", const=True))
#             edge_weight_var = HLSVar(var_name="edge_weight", var_type=ap_fixed_pod_t_type)
# 
#             # const uint32_t total_rounds = (part_edge_num + SRC_BUFFER_SIZE - 1) / SRC_BUFFER_SIZE;
#             code_lines_req.append(CodeVarDecl(var_name="total_rounds", var_type=uint_type, init_val="(part_edge_num + SRC_BUFFER_SIZE - 1) / SRC_BUFFER_SIZE", const=True))
#             total_rounds_var = HLSVar(var_name="total_rounds", var_type=uint_type)
# 
# 
#             # while (true)
#             while_1_codes: List[HLSCodeLine] = []
#             while_1_expr = HLSExpr(HLSExprT.CONST, True) # 使用 Python bool
#             while_1 = CodeWhile(codes=while_1_codes, iter_expr=while_1_expr)
#             code_lines_req.append(while_1)
# 
#             # --- 在 while(true) 循环内部 ---
#             # #pragma HLS PIPELINE II = 1
#             while_1_codes.append(CodePragma(content="PIPELINE II = 1"))
# 
#             # if ((pp_request_round - pp_read_round) <= 1) { ... }
#             if_1_codes: List[HLSCodeLine] = []
#             # HLSExpr(HLSExprT.CONST, "...") 用于 CodeIf 的表达式
#             if_1_expr = HLSExpr(HLSExprT.CONST, "((pp_request_round - pp_read_round) <= 1)")
#             if_1 = CodeIf(expr=if_1_expr, if_codes=if_1_codes)
#             while_1_codes.append(if_1)
# 
#             # --- 在 if( (pp_request_round - pp_read_round) <= 1 ) 内部 ---
#             # if (pp_request_round < pp_read_round)
#             if_2_codes: List[HLSCodeLine] = []
#             if_2_expr = HLSExpr(HLSExprT.CONST, "(pp_request_round < pp_read_round)")
#             if_2 = CodeIf(expr=if_2_expr, if_codes=if_2_codes)
#             if_1_codes.append(if_2)
#             # pp_request_round = pp_read_round;
#             if_2_codes.append(CodeAssign(var=pp_request_round_var, expr=HLSExpr(HLSExprT.VAR, pp_read_round_var)))
# 
#             # one_ppb_request.data = pp_request_round;
#             one_ppb_request_data_var = HLSVar(var_name="one_ppb_request.data", var_type=int_type) # 假设 data 匹配 round 类型
#             if_1_codes.append(CodeAssign(var=one_ppb_request_data_var, expr=HLSExpr(HLSExprT.VAR, pp_request_round_var)))
# 
#             # one_ppb_request.last = 0;
#             one_ppb_request_last_var = HLSVar(var_name="one_ppb_request.last", var_type=bool_type) # 假设
#             if_1_codes.append(CodeAssign(var=one_ppb_request_last_var, expr=HLSExpr(HLSExprT.CONST, 0))) # 使用 Python int
# 
#             # ppb_request_stm.write(one_ppb_request);
#             if_1_codes.append(CodeWriteStream(stream_var=param_ppb_request_stm, in_expr=one_ppb_request_var))
# 
#             # pp_request_round++;
#             if_1_codes.append(CodeOther(text="pp_request_round++;"))
# 
# 
#             # if (ppb_response_stm.read_nb(one_ppb_response)) { ... }
#             # (修正：直接在 CodeIf 中使用 read_nb 表达式字符串)
#             if_3_codes: List[HLSCodeLine] = []
#             if_3_expr = HLSExpr(HLSExprT.CONST, "ppb_response_stm.read_nb(one_ppb_response)")
#             if_3 = CodeIf(expr=if_3_expr, if_codes=if_3_codes)
#             while_1_codes.append(if_3)
# 
#             # --- 在 if( read_nb ) 内部 ---
#             # pp_write_round = one_ppb_response.dest << 4 >> LOG_SRC_BUFFER_SIZE;
#             # (修正：CodeAssign 和 HLSExprT.CONST 字符串)
#             if_3_codes.append(CodeAssign(var=pp_write_round_var, expr=HLSExpr(HLSExprT.CONST, "one_ppb_response.dest << 4 >> LOG_SRC_BUFFER_SIZE")))
# 
#             # bool write_buffer = pp_write_round & 0x1;
#             if_3_codes.append(CodeVarDecl(var_name="write_buffer", var_type=bool_type, init_val="(pp_write_round & 0x1)"))
#             write_buffer_var = HLSVar(var_name="write_buffer", var_type=bool_type)
# 
#             # int32_t write_idx = one_ppb_response.dest & ((SRC_BUFFER_SIZE >> 4) - 1);
#             if_3_codes.append(CodeVarDecl(var_name="write_idx", var_type=int_type, init_val="one_ppb_response.dest & ((SRC_BUFFER_SIZE >> 4) - 1)"))
#             write_idx_var = HLSVar(var_name="write_idx", var_type=int_type)
# 
#             # bus_word_t one_read_burst = one_ppb_response.data;
#             if_3_codes.append(CodeVarDecl(var_name="one_read_burst", var_type=bus_word_t_type, init_val="one_ppb_response.data"))
#             one_read_burst_var = HLSVar(var_name="one_read_burst", var_type=bus_word_t_type)
# 
#             # for (int u = 0; u < PE_NUM; u++) { ... }
#             for_1_codes: List[HLSCodeLine] = []
#             for_1 = CodeFor(codes=for_1_codes, iter_limit="PE_NUM", iter_cmp="<", iter_name="u", iter_start="0", iter_step="u++", iter_val_type=int_type)
#             if_3_codes.append(for_1)
#             # #pragma HLS UNROLL
#             for_1_codes.append(CodePragma(content="UNROLL"))
#             # src_prop_buffer[u][write_buffer][write_idx] = one_read_burst;
#             for_1_codes.append(CodeOther(text="src_prop_buffer[u][write_buffer][write_idx] = one_read_burst;"))
# 
#             # --- 结束 if( read_nb ) ---
# 
#             # if (!wait_flag)
#             if_4_codes: List[HLSCodeLine] = []
#             if_4_expr = HLSExpr(HLSExprT.UOP, dfir.UnaryOp.NOT, operands=[HLSExpr(HLSExprT.VAR, wait_flag_var)])
#             if_4 = CodeIf(expr=if_4_expr, if_codes=if_4_codes)
#             while_1_codes.append(if_4)
#             # an_edge_burst = edge_burst_stm.read();
#             # (修正：添加 expr_val=None)
#             if_4_codes.append(CodeAssign(var=an_edge_burst_var, expr=HLSExpr(HLSExprT.STREAM_READ, expr_val=None, operands=[HLSExpr(HLSExprT.VAR, param_edge_burst_stm)])))
# 
#             # pp_read_round = (an_edge_burst.edges[0].src_id / SRC_BUFFER_SIZE);
#             while_1_codes.append(CodeOther(text="pp_read_round = (an_edge_burst.edges[0].src_id / SRC_BUFFER_SIZE);"))
# 
#             # wait_flag = (pp_read_round >= pp_write_round) ? 1 : 0;
#             # (CodeAssign 和 HLSExprT.CONST 字符串)
#             while_1_codes.append(CodeAssign(var=wait_flag_var, expr=HLSExpr(HLSExprT.CONST, "(pp_read_round >= pp_write_round) ? 1 : 0")))
# 
#             # bool exit_flag = (wait_flag == 0) ? (edge_set_cnt + 1 >= total_edge_sets) : (edge_set_cnt >= total_edge_sets);
#             while_1_codes.append(CodeVarDecl(var_name="exit_flag", var_type=bool_type, init_val="(wait_flag == 0) ? (edge_set_cnt + 1 >= total_edge_sets) : (edge_set_cnt >= total_edge_sets)"))
#             exit_flag_var = HLSVar(var_name="exit_flag", var_type=bool_type)
# 
#             # if (!wait_flag) { ... }
#             if_5_codes: List[HLSCodeLine] = []
#             if_5_expr = HLSExpr(HLSExprT.UOP, dfir.UnaryOp.NOT, operands=[HLSExpr(HLSExprT.VAR, wait_flag_var)])
#             if_5 = CodeIf(expr=if_5_expr, if_codes=if_5_codes)
#             while_1_codes.append(if_5)
# 
#             # --- 在 if( !wait_flag ) 内部 ---
#             # bool read_buffer = pp_read_round & 0x1;
#             if_5_codes.append(CodeVarDecl(var_name="read_buffer", var_type=bool_type, init_val="(pp_read_round & 0x1)"))
#             read_buffer_var = HLSVar(var_name="read_buffer", var_type=bool_type)
# 
#             # update_tuple_t an_update_set;
#             if_5_codes.append(CodeVarDecl(var_name="an_update_set", var_type=update_tuple_t_type))
#             an_update_set_var = HLSVar(var_name="an_update_set", var_type=update_tuple_t_type)
# 
#             # Pragmas for an_update_set
#             if_5_codes.append(CodePragma(content="ARRAY_PARTITION variable = an_update_set.prop complete dim = 0"))
#             if_5_codes.append(CodePragma(content="ARRAY_PARTITION variable = an_update_set.node_id complete dim = 0"))
# 
#             # for (int u = 0; u < PE_NUM; u++) { ... }
#             for_2_codes: List[HLSCodeLine] = []
#             for_2 = CodeFor(codes=for_2_codes, iter_limit="PE_NUM", iter_cmp="<", iter_name="u", iter_start="0", iter_step="u++", iter_val_type=int_type)
#             if_5_codes.append(for_2)
# 
#             # --- 在 for(u) 内部 ---
#             # #pragma HLS UNROLL
#             for_2_codes.append(CodePragma(content="UNROLL"))
# 
#             # ap_uint<31> idx = (an_edge_burst.edges[u].src_id % SRC_BUFFER_SIZE);
#             ap_uint_31_type = HLSType(basic_type=HLSBasicType.AP_UINT, width=31)
#             for_2_codes.append(CodeVarDecl(var_name="idx", var_type=ap_uint_31_type, init_val="(an_edge_burst.edges[u].src_id % SRC_BUFFER_SIZE)"))
#             idx_var = HLSVar(var_name="idx", var_type=ap_uint_31_type)
# 
#             # ap_uint<30> uram_row_idx = idx >> 4;
#             ap_uint_30_type = HLSType(basic_type=HLSBasicType.AP_UINT, width=30)
#             for_2_codes.append(CodeVarDecl(var_name="uram_row_idx", var_type=ap_uint_30_type, init_val="(idx >> 4)"))
#             uram_row_idx_var = HLSVar(var_name="uram_row_idx", var_type=ap_uint_30_type)
# 
#             # ap_uint<30> uram_row_offset = (idx & 0xf);
#             for_2_codes.append(CodeVarDecl(var_name="uram_row_offset", var_type=ap_uint_30_type, init_val="(idx & 0xf)"))
#             uram_row_offset_var = HLSVar(var_name="uram_row_offset", var_type=ap_uint_30_type)
# 
#             # bus_word_t uram_row = src_prop_buffer[u][read_buffer][uram_row_idx];
#             for_2_codes.append(CodeOther(text="bus_word_t uram_row = src_prop_buffer[u][read_buffer][uram_row_idx];"))
# 
#             # ap_fixed_pod_t src_prop = get_val_from_bus(uram_row, uram_row_offset);
#             for_2_codes.append(CodeVarDecl(var_name="src_prop", var_type=ap_fixed_pod_t_type, init_val="get_val_from_bus(uram_row, uram_row_offset)"))
#             src_prop_var = HLSVar(var_name="src_prop", var_type=ap_fixed_pod_t_type)
# 
# 
#             #============= begin inline logic ===============
#             port_to_var = {}
#             for_2_codes.append(CodeOther(text="// Begin inline logic"))
#             DST_ID_VAR = HLSVar(var_name="DST_ID_VAR", var_type=node_id_type)
#             for_2_codes.append(CodeVarDecl(var_name="DST_ID_VAR", var_type=node_id_type, init_val="an_edge_burst.edges[u].dst_id"))
#             SRC_PROP_VAR = src_prop_var
#             EDGE_WEIGHT_VAR = edge_weight_var
#             top_vars = {
#                 "DST_ID_VAR": DST_ID_VAR,
#                 "SRC_PROP_VAR": SRC_PROP_VAR,
#                 "EDGE_WEIGHT_VAR": EDGE_WEIGHT_VAR
#             }
#             # an_update_set.prop[u] = (src_prop + edge_weight);
#             #for_2_codes.append(CodeOther(text="an_update_set.prop[u] = (src_prop + edge_weight);"))
# 
#             # an_update_set.node_id[u] = an_edge_burst.edges[u].dst_id;
#             #for_2_codes.append(CodeOther(text="an_update_set.node_id[u] = an_edge_burst.edges[u].dst_id;"))
#             an_update_set_prop_var = HLSVar(var_name="an_update_set.prop[u]", var_type=ap_fixed_pod_t_type)
#             an_update_set_node_id_var = HLSVar(var_name="an_update_set.node_id[u]", var_type=node_id_type)
#             top_vars["FINAL_PROP_VAR"] = an_update_set_prop_var
#             top_vars["FINAL_DST_ID_VAR"] = an_update_set_node_id_var
#             port_property = {} # 只能是src , dst, edge_prop这三项或组合
#             inlinecodes = []
#             for comp in scatter_stage_comps:
#                 port_property,inlinecodes = self._scatter_analyze(comp,port_property,port_to_var,top_vars,target_codes=inlinecodes)
#             for_2_codes.extend(inlinecodes)
#             for_2_codes.append(CodeOther(text="// End inline logic"))
#             # =========== end inline logic ==============
#             # --- 结束 for(u) ---
# 
#             # update_set_stm.write(an_update_set);
#             if_5_codes.append(CodeWriteStream(stream_var=param_update_set_stm, in_expr=an_update_set_var))
# 
#             # edge_set_cnt++;
#             if_5_codes.append(CodeOther(text="edge_set_cnt++;"))
# 
#             # --- 结束 if( !wait_flag ) ---
# 
#             # if (exit_flag) { ... }
#             if_6_codes: List[HLSCodeLine] = []
#             if_6_expr = HLSExpr(HLSExprT.VAR, exit_flag_var)
#             if_6 = CodeIf(expr=if_6_expr, if_codes=if_6_codes)
#             while_1_codes.append(if_6)
# 
#             # --- 在 if( exit_flag ) 内部 ---
#             # one_ppb_request.last = 1;
#             if_6_codes.append(CodeAssign(var=one_ppb_request_last_var, expr=HLSExpr(HLSExprT.CONST, 1))) # 使用 Python int
# 
#             # ppb_request_stm.write(one_ppb_request);
#             if_6_codes.append(CodeWriteStream(stream_var=param_ppb_request_stm, in_expr=one_ppb_request_var))
# 
#             # while (true) { ... } (内部退出循环)
#             while_2_codes: List[HLSCodeLine] = []
#             while_2_expr = HLSExpr(HLSExprT.CONST, True) # 使用 Python bool
#             while_2 = CodeWhile(codes=while_2_codes, iter_expr=while_2_expr)
#             if_6_codes.append(while_2)
# 
#             # --- 在内部 while(true) 循环 ---
#             # ppb_response_stm.read(one_ppb_response);
#             while_2_codes.append(CodeOther(text="ppb_response_stm.read(one_ppb_response);"))
# 
#             # if (one_ppb_response.last)
#             if_7_codes: List[HLSCodeLine] = []
#             if_7_expr = HLSExpr(HLSExprT.CONST, "one_ppb_response.last") # 使用 C++ 表达式字符串
#             if_7 = CodeIf(expr=if_7_expr, if_codes=if_7_codes)
#             while_2_codes.append(if_7)
#             # break;
#             if_7_codes.append(CodeBreak())
# 
#             # --- 结束内部 while(true) ---
# 
#             # break; (退出外部 while 循环)
#             if_6_codes.append(CodeBreak())
# 
#             # --- 结束 if( exit_flag ) ---
#             # --- 结束 while(true) ---
# 
# 
#             # --- 3. 显式地将所有代码行赋值给函数 ---
#             request_manager_func.codes = code_lines_req
# 
#             self.little_scatter_funcs.append(request_manager_func)
#             self.little_top_dataflow_funcs.append(request_manager_func)
# 
# 
#             set_word_in_bus_func = HLSFunction(name="set_word_in_bus", comp=comp)
#             params_swib: List[HLSVar] = []
# 
#             # --- 1. 定义类型和参数 ---
# 
#             # 基本类型
#             int_type = HLSType(HLSBasicType.INT)
#             bus_word_t_type = HLSType(HLSBasicType.BUS_WORD_T)
#             ap_fixed_pod_t_type = HLSType(HLSBasicType.AP_FIXED_POD)
# 
#             # Param 1: bus_word_t &bus_word
#             # (HLSType 不需要 &，gen_code 会处理引用)
#             param_bus_word = HLSVar(var_name="bus_word", var_type=bus_word_t_type)
# 
#             # Param 2: int idx
#             param_idx = HLSVar(var_name="idx", var_type=int_type)
# 
#             # Param 3: ap_fixed_pod_t pod_low
#             param_pod_low = HLSVar(var_name="pod_low", var_type=ap_fixed_pod_t_type)
# 
#             # Param 4: ap_fixed_pod_t pod_high
#             param_pod_high = HLSVar(var_name="pod_high", var_type=ap_fixed_pod_t_type)
# 
#             params_swib.extend([param_bus_word, param_idx, param_pod_low, param_pod_high])
#             set_word_in_bus_func.params = params_swib
# 
#             # --- 2. 函数体 (构建本地列表) ---
#             code_lines_swib: List[HLSCodeLine] = []
# 
#             # #pragma HLS INLINE
#             code_lines_swib.append(CodePragma(content="INLINE"))
# 
#             # --- switch (idx) ---
#             # (按照要求，使用 CodeOther 实现整个 switch 块)
#             switch_block_text = """switch (idx) {
#                 case 0:
#                     bus_word.range(31, 0) = pod_low;
#                     bus_word.range(63, 32) = pod_high;
#                     break;
#                 case 1:
#                     bus_word.range(95, 64) = pod_low;
#                     bus_word.range(127, 96) = pod_high;
#                     ;
#                     break;
#                 case 2:
#                     bus_word.range(159, 128) = pod_low;
#                     bus_word.range(191, 160) = pod_high;
#                     break;
#                 case 3:
#                     bus_word.range(223, 192) = pod_low;
#                     bus_word.range(255, 224) = pod_high;
#                     break;
#                 case 4:
#                     bus_word.range(287, 256) = pod_low;
#                     bus_word.range(319, 288) = pod_high;
#                     break;
#                 case 5:
#                     bus_word.range(351, 320) = pod_low;
#                     bus_word.range(383, 352) = pod_high;
#                     break;
#                 case 6:
#                     bus_word.range(415, 384) = pod_low;
#                     bus_word.range(447, 416) = pod_high;
#                     break;
#                 case 7:
#                     bus_word.range(479, 448) = pod_low;
#                     bus_word.range(511, 480) = pod_high;
#                     break;
#                 default:
#                     break;
#                 }"""
#             code_lines_swib.append(CodeOther(text=switch_block_text))
# 
# 
#             # --- 3. 显式地将所有代码行赋值给函数 ---
#             set_word_in_bus_func.codes = code_lines_swib
#             self.little_scatter_funcs.append(set_word_in_bus_func)
# 
        
    def process_gather(self,gather_stage_comps : List[dfir.Component]):
        assert len(gather_stage_comps) == 1
        reduce_comp = gather_stage_comps[0]
        self._translate_reduce_op(reduce_comp)
            
    def process_apply(self,apply_stage_comps : List[dfir.Component]):

        apply_func = HLSFunction(name="apply_func", comp=None)
        params = []

        bus_word_t_type = HLSType(HLSBasicType.BUS_WORD_T)
        uint32_type = HLSType(HLSBasicType.UINT)
        bool_type = HLSType(HLSBasicType.BOOL)
        ap_fixed_pod_t_type = HLSType(HLSBasicType.AP_FIXED_POD)
        int_type = HLSType(HLSBasicType.INT)

        in_write_burst_w_dst_pkt_t_type = HLSType(basic_type=HLSBasicType.STRUCT,
                                                  struct_name="in_write_burst_w_dst_pkt_t",
                                                  struct_prop_names=["data", "dest_addr", "end_flag"],
                                                  sub_types=[bus_word_t_type, uint32_type, bool_type])
        if in_write_burst_w_dst_pkt_t_type.name not in self.struct_definitions:
            self.struct_definitions[in_write_burst_w_dst_pkt_t_type.name] = (
                in_write_burst_w_dst_pkt_t_type,
                in_write_burst_w_dst_pkt_t_type.struct_prop_names,
            )


        write_burst_w_dst_pkt_t_type = HLSType(basic_type=HLSBasicType.WRITE_BURST_W_DST_PKT_T)
                                                

        # Stream 类型
        stream_in_type = HLSType(HLSBasicType.STREAM, sub_types=[in_write_burst_w_dst_pkt_t_type])
        stream_out_type = HLSType(HLSBasicType.STREAM, sub_types=[write_burst_w_dst_pkt_t_type])

        # 指针类型
        ptr_bus_word_t_type = HLSType(HLSBasicType.POINTER, sub_types=[bus_word_t_type])

        # 参数变量
        node_props_var = HLSVar(var_name="node_props", var_type=ptr_bus_word_t_type)
        write_burst_stream_var = HLSVar(var_name="write_burst_stream", var_type=stream_in_type)
        kernel_out_stream_var = HLSVar(var_name="kernel_out_stream", var_type=stream_out_type)

        params.extend([node_props_var, write_burst_stream_var, kernel_out_stream_var])
        apply_func.params = params

        # --- 2. 函数体 ---
        code_lines: List[HLSCodeLine] = []

        # while (true)
        while_codes: List[HLSCodeLine] = []
        while_expr = HLSExpr(HLSExprT.CONST, True)

        # in_write_burst_w_dst_pkt_t in_pkt = write_burst_stream.read();
        in_pkt_var = HLSVar(var_name="in_pkt", var_type=in_write_burst_w_dst_pkt_t_type)
        read_expr = HLSExpr(HLSExprT.STREAM_READ, None, operands=[HLSExpr(HLSExprT.VAR, write_burst_stream_var)])
        while_codes.append(CodeVarDecl(var_name="in_pkt", var_type=in_write_burst_w_dst_pkt_t_type, init_val=read_expr.code))

        # if (in_pkt.end_flag)
        if_1_codes: List[HLSCodeLine] = []
        if_1_expr = HLSExpr(HLSExprT.UOP, (dfir.UnaryOp.GET_ATTR, "end_flag"), operands=[HLSExpr(HLSExprT.VAR, in_pkt_var)])

        # write_burst_w_dst_pkt_t end_pkt;
        end_pkt_var = HLSVar(var_name="end_pkt", var_type=write_burst_w_dst_pkt_t_type)
        if_1_codes.append(CodeVarDecl(var_name="end_pkt", var_type=write_burst_w_dst_pkt_t_type))

        # end_pkt.last = true;
        end_pkt_last_var = HLSVar(var_name="end_pkt.last", var_type=bool_type)
        if_1_codes.append(CodeAssign(var=end_pkt_last_var, expr=HLSExpr(HLSExprT.CONST, True)))

        # kernel_out_stream.write(end_pkt);
        if_1_codes.append(CodeWriteStream(stream_var=kernel_out_stream_var, in_expr=end_pkt_var))

        # break;
        if_1_codes.append(CodeBreak())

        # (构建 if_1)
        while_codes.append(CodeIf(expr=if_1_expr, if_codes=if_1_codes, else_codes=None))

        # uint32_t dest_addr = in_pkt.dest_addr;
        dest_addr_var = HLSVar(var_name="dest_addr", var_type=uint32_type)
        dest_addr_init_expr = HLSExpr(HLSExprT.UOP, (dfir.UnaryOp.GET_ATTR, "dest_addr"), operands=[HLSExpr(HLSExprT.VAR, in_pkt_var)])
        while_codes.append(CodeVarDecl(var_name="dest_addr", var_type=uint32_type, init_val=dest_addr_init_expr.code))

        # bus_word_t ori_props = node_props[dest_addr];
        ori_props_var = HLSVar(var_name="ori_props", var_type=bus_word_t_type)
        # HLSExpr 不支持数组访问，因此将其作为 opaque 字符串 init_val
        while_codes.append(CodeVarDecl(var_name="ori_props", var_type=bus_word_t_type, init_val="node_props[dest_addr]"))

        # bus_word_t new_props;
        new_props_var = HLSVar(var_name="new_props", var_type=bus_word_t_type)
        while_codes.append(CodeVarDecl(var_name="new_props", var_type=bus_word_t_type))

        # write_burst_w_dst_pkt_t out_pkt;
        out_pkt_var = HLSVar(var_name="out_pkt", var_type=write_burst_w_dst_pkt_t_type)
        while_codes.append(CodeVarDecl(var_name="out_pkt", var_type=write_burst_w_dst_pkt_t_type))

        # out_pkt.dest = dest_addr;
        out_pkt_dest_var = HLSVar(var_name="out_pkt.dest", var_type=uint32_type)
        while_codes.append(CodeAssign(var=out_pkt_dest_var, expr=HLSExpr(HLSExprT.VAR, dest_addr_var)))

        # out_pkt.last = false;
        out_pkt_last_var = HLSVar(var_name="out_pkt.last", var_type=bool_type)
        while_codes.append(CodeAssign(var=out_pkt_last_var, expr=HLSExpr(HLSExprT.CONST, False)))

        # for (int i = 0; i < 16; i++)
        for_codes: List[HLSCodeLine] = []
        i_var = HLSVar(var_name="i", var_type=int_type) # 循环变量

        # #pragma HLS UNROLL
        for_codes.append(CodePragma(content="UNROLL"))

        # ap_fixed_pod_t update = in_pkt.data.range(31 + (i << 5), (i << 5));
        update_var = HLSVar(var_name="update", var_type=ap_fixed_pod_t_type)
        # .range() 作为 opaque 字符串 init_val
        for_codes.append(CodeVarDecl(var_name="update", var_type=ap_fixed_pod_t_type, init_val="in_pkt.data.range(31 + (i << 5), (i << 5))"))

        # ap_fixed_pod_t old = ori_props.range(31 + (i << 5), (i << 5));
        old_var = HLSVar(var_name="old", var_type=ap_fixed_pod_t_type)
        for_codes.append(CodeVarDecl(var_name="old", var_type=ap_fixed_pod_t_type, init_val="ori_props.range(31 + (i << 5), (i << 5))"))

        # ap_fixed_pod_t new_prop = (old < update) ? old : update;
        new_prop_var = HLSVar(var_name="new_prop", var_type=ap_fixed_pod_t_type)

        # ================ begin inline fused op =============================
        for_codes.append(CodeOther(text="// Begin inline fused op"))
        inline_codes = []
        port_property = {}
        port_to_var = {}
        top_vars = {
            "init_val_1_var": update_var,
            "init_val_2_var": old_var,
            "result_val3": new_prop_var
        }
        for comp in apply_stage_comps:
            port_property,inline_codes = self._apply_analyze(comp,port_property,port_to_var,top_vars,target_codes=inline_codes)
        for_codes.extend(inline_codes)
        for_codes.append(CodeOther(text="// End inline fused op"))
        
            
        # ========================= end inline fused op ==========================
        # new_props.range(31 + (i << 5), (i << 5)) = new_prop;
        # .range() 作为 HLSVar 的 name 属性 (LHS)
        lhs_range_var = HLSVar(var_name="new_props.range(31 + (i << 5), (i << 5))", var_type=ap_fixed_pod_t_type)
        for_codes.append(CodeAssign(var=lhs_range_var, expr=HLSExpr(HLSExprT.VAR, new_prop_var)))

        # (构建 for 循环)
        while_codes.append(CodeFor(codes=for_codes, 
                                   iter_limit="16", 
                                   iter_cmp="<", 
                                   iter_name="i", 
                                   iter_start="0", 
                                   iter_val_type=int_type))

        # out_pkt.data = new_props;
        out_pkt_data_var = HLSVar(var_name="out_pkt.data", var_type=bus_word_t_type)
        while_codes.append(CodeAssign(var=out_pkt_data_var, expr=HLSExpr(HLSExprT.VAR, new_props_var)))

        # kernel_out_stream.write(out_pkt);
        while_codes.append(CodeWriteStream(stream_var=kernel_out_stream_var, in_expr=out_pkt_var))

        # (构建 while 循环)
        code_lines.append(CodeWhile(codes=while_codes, iter_expr=while_expr))

        # (完成函数)
        apply_func.codes = code_lines
        self.apply_funcs.append(apply_func)


    # 识别出S G A 三部分计算逻辑
    def generate_backend(
        self, comp_col: dfir.ComponentCollection, global_graph: Any, top_func_name: str
    ) -> Tuple[str, str]:
        
        self.global_graph_store = global_graph
        self.comp_col_store = comp_col
        
        self.big_scatter_funcs.clear()
        self.big_gather_funcs.clear()
        self.apply_funcs.clear()
        self.little_scatter_funcs.clear()
        self.little_gather_funcs.clear()
        component_list = comp_col.topo_sort()

        scatter_stage_comps = []        
        gather_stage_comps = []
        apply_stage_comps = []

        reduce_found = False
        reduce_out_ports = []

        for comp in component_list:
            for port in comp.ports:
                conn_id = port.connection.readable_id if port.connection else "None"
                print(f"    Port: {port.readable_id}, name:{port.name},Type: {port.port_type}, Connected to: {conn_id}")
            if isinstance(
                    comp,
                    (
                        dfir.IOComponent, 
                        dfir.ConstantComponent,
                        dfir.UnusedEndMarkerComponent,
                    )
            ):
                continue
            if reduce_found:
                is_post_reduce = True

                for port in comp.ports:
                    if port.port_type == dfir.PortType.IN:
                        if not (port.connection in reduce_out_ports):
                            is_post_reduce = False
                            break
                if is_post_reduce:
                    for port in comp.ports:
                        if port.port_type == dfir.PortType.OUT:
                             reduce_out_ports.append(port)
                    apply_stage_comps.append(comp)


            elif isinstance(comp, dfir.ReduceComponent):
                gather_stage_comps.append(comp)
                reduce_found = True  
                ReduceComp = comp
                scatter_stage_comps.append(comp)
                apply_stage_comps.append(comp)
                for port in comp._port_groups["global"]:
                    if port.port_type == dfir.PortType.OUT:
                        reduce_out_ports.append(port)


            else:
                scatter_stage_comps.append(comp)

        self.process_scatter(scatter_stage_comps,ReduceComp)

        self.process_gather(gather_stage_comps)

        self.process_apply(apply_stage_comps)

        self._generate_top_func()
        
        big_source_code , little_source_code = self._generate_source_file()

        big_header_code , little_header_code, shared_kernel_params= self._generate_header_file()
        apply_kernel = self._generate_apply()
        

        return big_header_code, little_header_code, shared_kernel_params, big_source_code, little_source_code , apply_kernel
        
    