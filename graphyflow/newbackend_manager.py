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
        self.struct_definitions: Dict[str, Tuple[HLSType, List[str]]] = {}

        # State for code generation

        self.defines: Dict[str,str]={}
        # funcs_for_big_kernel
        self.big_scatter_funcs: List[HLSFunction] = []
        self.big_gather_funcs: List[HLSFunction] = []
        self.big_apply_funcs: List[HLSFunction] = []
        self.big_top_func = None
        self.big_top_dataflow_funcs =  []
        self.big_apply_top_func = None
        self.big_hbm_writer_func = None

        # funcs_for_little_kernel
        self.little_scatter_funcs: List[HLSFunction] = []
        self.little_gather_funcs: List[HLSFunction] = []
        self.little_apply_funcs: List[HLSFunction] = []
        self.little_top_func = None
        self.little_top_dataflow_funcs =  []
        self.little_apply_top_func = None
        self.little_hbm_writer_func = None

        # other funcs:
        self.helper_funcs = [
"""
static ap_fixed_pod_t get_val_from_bus(const bus_word_t bus, int offset) {
#pragma HLS INLINE
    switch (offset) {
    case 0:
        return bus.range(31, 0);
    case 1:
        return bus.range(63, 32);
    case 2:
        return bus.range(95, 64);
    case 3:
        return bus.range(127, 96);
    case 4:
        return bus.range(159, 128);
    case 5:
        return bus.range(191, 160);
    case 6:
        return bus.range(223, 192);
    case 7:
        return bus.range(255, 224);
    case 8:
        return bus.range(287, 256);
    case 9:
        return bus.range(319, 288);
    case 10:
        return bus.range(351, 320);
    case 11:
        return bus.range(383, 352);
    case 12:
        return bus.range(415, 384);
    case 13:
        return bus.range(447, 416);
    case 14:
        return bus.range(479, 448);
    case 15:
        return bus.range(511, 480);
    default:
        return 0;
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
""",
"""
inline ap_fixed_pod_t get_raw_val(reduce_word_t word, int idx) {
#pragma HLS INLINE
    ap_uint<DISTANCE_BITWIDTH> bits;
    switch (idx) {
    case 0:
        bits = word.range(DISTANCE_BITWIDTH - 1, 0);
        break;
    case 1:
        bits = word.range((DISTANCE_BITWIDTH << 1) - 1, DISTANCE_BITWIDTH);
        break;
    case 2:
        bits =
            word.range((DISTANCE_BITWIDTH * 3) - 1, (DISTANCE_BITWIDTH << 1));
        break;
    default:
        bits = 0;
        break;
    }
    return bits;
}
""",
"""
inline distance_t get_val(reduce_word_t word, int idx) {
#pragma HLS INLINE
    ap_fixed_pod_t raw_val = get_raw_val(word, idx);
    distance_t val = *reinterpret_cast<distance_t *>(&raw_val);
    return val;
}
""",
"""
inline void set_val(reduce_word_t &word, int idx, distance_t val) {
#pragma HLS INLINE
    ap_uint<DISTANCE_BITWIDTH> val_bits =
        *reinterpret_cast<ap_uint<DISTANCE_BITWIDTH> *>(&val);
    switch (idx) {
    case 0:
        word.range(DISTANCE_BITWIDTH - 1, 0) = val_bits;
        break;
    case 1:
        word.range((DISTANCE_BITWIDTH << 1) - 1, DISTANCE_BITWIDTH) = val_bits;
        break;
    case 2:
        word.range((DISTANCE_BITWIDTH * 3) - 1, (DISTANCE_BITWIDTH << 1)) =
            val_bits;
        break;
    default:
        break;
    }
}

""",

"""
inline void set_raw_val(reduce_word_t &word, int idx, ap_fixed_pod_t pod_val) {
#pragma HLS INLINE
    ap_uint<DISTANCE_BITWIDTH> val_bits = pod_val;
    switch (idx) {
    case 0:
        word.range(DISTANCE_BITWIDTH - 1, 0) = val_bits;
        break;
    case 1:
        word.range((DISTANCE_BITWIDTH << 1) - 1, DISTANCE_BITWIDTH) = val_bits;
        break;
    case 2:
        word.range((DISTANCE_BITWIDTH * 3) - 1, (DISTANCE_BITWIDTH << 1)) =
            val_bits;
        break;
    default:
        break;
    }
}
"""

        ]

        
        #
        self.global_graph_store = None
        self.comp_col_store = None

        # New manager for memory and specific graph structures
        self.mem_manager: Optional[MemoryAndGraphManager] = None
        self.dataflow_core_func: Optional[HLSFunction] = None

        # reduce_mode
        self.REDUCE_MODE = "big_pipeline"
    def _generate_hbm_writer(self):
        hbm_writer_little_func = HLSFunction(name="hbm_writer_little", comp=None)
        params_hwl: List[HLSVar] = []

        # --- 1. 为此函数独立定义类型 ---

        # 基本类型
        bus_word_t_type = HLSType(HLSBasicType.BUS_WORD_T)
        uint_type = HLSType(HLSBasicType.UINT)
        ppb_request_pkt_t_type = HLSType(HLSBasicType.PPB_REQUEST_PKT_T)
        ppb_response_pkt_t_type = HLSType(HLSBasicType.PPB_RESPONSE_PKT_T)
        write_burst_pkt_t_type = HLSType(HLSBasicType.WRITE_BURST_PKT_T)

        # --- 2. 定义参数 HLSVars ---

        # Param 1: bus_word_t *node_props
        param_node_props = HLSVar(var_name="node_props", 
                                  var_type=HLSType(HLSBasicType.POINTER, sub_types=[bus_word_t_type]))

        # Param 2: bus_word_t *output
        param_output = HLSVar(var_name="output", 
                              var_type=HLSType(HLSBasicType.POINTER, sub_types=[bus_word_t_type]))

        # Param 3: uint32_t dst_num
        param_dst_num = HLSVar(var_name="dst_num", var_type=uint_type)

        # Param 4: hls::stream<ppb_request_pkt_t> &ppb_req_stream
        param_ppb_req_stream = HLSVar(var_name="ppb_req_stream", 
                                      var_type=HLSType(HLSBasicType.STREAM, sub_types=[ppb_request_pkt_t_type]))

        # Param 5: hls::stream<ppb_response_pkt_t> &ppb_resp_stream
        param_ppb_resp_stream = HLSVar(var_name="ppb_resp_stream", 
                                       var_type=HLSType(HLSBasicType.STREAM, sub_types=[ppb_response_pkt_t_type]))

        # Param 6: hls::stream<write_burst_pkt_t> &write_burst_stream
        param_write_burst_stream = HLSVar(var_name="write_burst_stream", 
                                          var_type=HLSType(HLSBasicType.STREAM, sub_types=[write_burst_pkt_t_type]))


        params_hwl.extend([param_node_props, param_output, param_dst_num, 
                           param_ppb_req_stream, param_ppb_resp_stream, param_write_burst_stream])
        hbm_writer_little_func.params = params_hwl

        # --- 3. 函数体 (构建本地列表) ---
        code_lines_hwl: List[HLSCodeLine] = []

        # (函数体为空)

        # --- 4. 显式地将所有代码行赋值给函数 ---
        hbm_writer_little_func.codes = code_lines_hwl
        self.little_hbm_writer_func = hbm_writer_little_func

        hbm_writer_big_func = HLSFunction(name="hbm_writer_big", comp=None)
        params_hwb: List[HLSVar] = []

        # --- 1. 为此函数独立定义类型 ---

        # 基本类型
        bus_word_t_type = HLSType(HLSBasicType.BUS_WORD_T)
        uint_type = HLSType(HLSBasicType.UINT)
        cacheline_request_pkt_t_type = HLSType(HLSBasicType.CACHELINE_REQUEST_PKT_T)
        cacheline_response_pkt_t_type = HLSType(HLSBasicType.CACHELINE_RESPONSE_PKT_T)
        write_burst_pkt_t_type = HLSType(HLSBasicType.WRITE_BURST_PKT_T)

        # --- 2. 定义参数 HLSVars ---

        # Param 1: bus_word_t *node_props
        param_node_props = HLSVar(var_name="node_props", 
                                  var_type=HLSType(HLSBasicType.POINTER, sub_types=[bus_word_t_type]))

        # Param 2: bus_word_t *output
        param_output = HLSVar(var_name="output", 
                              var_type=HLSType(HLSBasicType.POINTER, sub_types=[bus_word_t_type]))

        # Param 3: uint32_t dst_num
        param_dst_num = HLSVar(var_name="dst_num", var_type=uint_type)

        # Param 4: hls::stream<cacheline_request_pkt_t> &cacheline_req_stream
        param_cacheline_req_stream = HLSVar(var_name="cacheline_req_stream", 
                                            var_type=HLSType(HLSBasicType.STREAM, sub_types=[cacheline_request_pkt_t_type]))

        # Param 5: hls::stream<cacheline_response_pkt_t> &cacheline_resp_stream
        param_cacheline_resp_stream = HLSVar(var_name="cacheline_resp_stream", 
                                             var_type=HLSType(HLSBasicType.STREAM, sub_types=[cacheline_response_pkt_t_type]))

        # Param 6: hls::stream<write_burst_pkt_t> &write_burst_stream
        param_write_burst_stream = HLSVar(var_name="write_burst_stream", 
                                          var_type=HLSType(HLSBasicType.STREAM, sub_types=[write_burst_pkt_t_type]))


        params_hwb.extend([param_node_props, param_output, param_dst_num, 
                           param_cacheline_req_stream, param_cacheline_resp_stream, param_write_burst_stream])
        hbm_writer_big_func.params = params_hwb

        # --- 3. 函数体 (构建本地列表) ---
        code_lines_hwb: List[HLSCodeLine] = []

        # (函数体为空)

        # --- 4. 显式地将所有代码行赋值给函数 ---
        hbm_writer_big_func.codes = code_lines_hwb
        self.big_hbm_writer_func = hbm_writer_big_func

        
    def _generate_apply(self):
        code = "#include \"shared_kernel_params.h\"\n\n"
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

        for func in self.big_apply_funcs:
            write_func_body(func,False) 
        
        write_func_body(self.big_apply_top_func,True)
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

    def _topologically_sort_structs(self) -> List[Tuple[HLSType, List[str]]]:
        """Sorts struct definitions based on their member dependencies."""
        from collections import defaultdict

        adj = defaultdict(list)
        in_degree = defaultdict(int)

        all_struct_names = self.struct_definitions.keys()

        # Build dependency graph
        for dependent_struct_name, (hls_type, _) in self.struct_definitions.items():
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
        queue = [name for name in self.struct_definitions if in_degree[name] == 0]
        sorted_structs = []

        while queue:
            u = queue.pop(0)
            if u in self.struct_definitions:
                sorted_structs.append(self.struct_definitions[u])
                for v in adj[u]:
                    in_degree[v] -= 1
                    if in_degree[v] == 0:
                        queue.append(v)

        if len(sorted_structs) != len(self.struct_definitions):
            print("--- DEBUG: Cycle detected in struct dependencies ---")
            print("Total structs:", len(self.struct_definitions))
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

        code = "#include <hls_stream.h>\n#include <ap_fixed.h>\n#include <ap_int.h>\n"
        code += "#include <stdint.h>\n#include <ap_axi_sdata.h>\n"
        code += "#include <string.h>\n\n"

        code += f"#define PE_NUM {8}\n"
        code += f"#define DBL_PE_NUM {16}\n"
        code += f"#define LOG_PE_NUM {3}\n"
        code += f"#define MAX_NUM {524288}\n"
        code += f"#define L {4}\n\n"

        code += f"// --- New Bitwidth Definitions for HLS Synthesis ---\n"
        code += f"#define NODE_ID_BITWIDTH {32}\n"
        code += f"#define DISTANCE_BITWIDTH {32}\n"
        code += f"#define DISTANCE_INTEGER_PART {16}\n"
        code += f"#define WEIGHT_BITWIDTH DISTANCE_BITWIDTH\n"
        code += f"#define WEIGHT_INTEGER_PART DISTANCE_INTEGER_PART\n"
        code += f"#define OUT_END_MARKER_BITWIDTH 4\n"
        code += f"#define DIST_PER_WORD {16}\n"
        code += f"#define LOG_DIST_PER_WORD {4}\n"



        code += f"// --- New Memory Word and Bus Definitions ---\n"
        code += f"#define AXI_BUS_WIDTH 512\n"
        code += f"#define REDUCE_MEM_WIDTH 64\n"
        code += f"typedef ap_uint<AXI_BUS_WIDTH> bus_word_t;\n "
        code += f"typedef ap_uint<REDUCE_MEM_WIDTH> reduce_word_t;\n"
        code += f"const int INFINITY_DIST = 16384;\n\n"

        code += f"#define DISTANCES_PER_REDUCE_WORD (REDUCE_MEM_WIDTH / DISTANCE_BITWIDTH)"

        code += "// --- Graph Type Definitions ---\n"
        code += "typedef uint32_t edge_id_t;\n"
        code += "typedef uint32_t node_id_t;\n"
        code += "typedef uint32_t ap_fixed_pod_t;\n\n"

        code += "typedef ap_fixed<DISTANCE_BITWIDTH, DISTANCE_INTEGER_PART> distance_t;\n"
        code += "typedef ap_uint<OUT_END_MARKER_BITWIDTH> out_end_marker_t;\n"
        code += "typedef ap_axiu<256, 0, 0, 0> node_dist_pkt_t;\n"
        code += "typedef ap_axiu<512, 0, 0, 0> write_burst_pkt_t;"
        code += "typedef ap_axiu<32, 0, 0, 8> cacheline_request_pkt_t;\n"
        code += "typedef ap_axiu<512, 0, 0, 8> cacheline_response_pkt_t;\n"
        code += "typedef ap_axiu<512, 0, 0, 0> cacheline_data_pkt_t;\n"

        big_header += code

        # little part:
        code = ""
        code += "#include <ap_axi_sdata.h>\n"
        code += "#include <ap_fixed.h>\n"
        code += "#include <ap_int.h>\n"
        code += "#include <hls_stream.h>\n"
        code += "#include <stdint.h>\n"
        code += "// #include <stdio.h>\n"
        code += "#include <string.h>\n"
        code += "\n"
        code += "#define PE_NUM 8\n"
        code += "#define DBL_PE_NUM 16\n"
        code += "#define LOG_PE_NUM 3\n"
        code += "#ifdef EMULATION\n"
        code += "#define MAX_NUM 512\n"
        code += "#else\n"
        code += "#define MAX_NUM 65536\n"
        code += "#endif\n"
        code += "#define L 4\n"
        code += "#define SRC_BUFFER_SIZE 4096\n"
        code += "#define LOG_SRC_BUFFER_SIZE 12\n"
        code += "\n"
        code += "// --- New Bitwidth Definitions for HLS Synthesis ---\n"
        code += "#define NODE_ID_BITWIDTH 32\n"
        code += "#define DISTANCE_BITWIDTH 32\n"
        code += "#define DISTANCE_INTEGER_PART 16\n"
        code += "#define WEIGHT_BITWIDTH DISTANCE_BITWIDTH\n"
        code += "#define WEIGHT_INTEGER_PART DISTANCE_INTEGER_PART\n"
        code += "#define OUT_END_MARKER_BITWIDTH 4\n"
        code += "#define DIST_PER_WORD 16 // AXI_BUS_WIDTH / DISTANCE_BITWIDTH = 512 / 32 = 16\n"
        code += "#define LOG_DIST_PER_WORD                                                 \\\n"
        code += "    4 // log2(AXI_BUS_WIDTH / DISTANCE_BITWIDTH) = log2(512 / 32) = log2(16) = 4\n"
        code += "\n"
        code += "// --- New Memory Word and Bus Definitions ---\n"
        code += "#define AXI_BUS_WIDTH 512\n"
        code += "\n"
        code += "#define REDUCE_MEM_WIDTH 64\n"
        code += "typedef ap_uint<AXI_BUS_WIDTH> bus_word_t;\n"
        code += "typedef ap_uint<REDUCE_MEM_WIDTH> reduce_word_t;\n"
        code += "\n"
        code += "const int INFINITY_DIST = 16384;\n"
        code += "\n"
        code += "// --- New Packing-related Constants ---\n"
        code += "// Number of distances that can be packed into a single reduce memory word.\n"
        code += "#define DISTANCES_PER_REDUCE_WORD (REDUCE_MEM_WIDTH / DISTANCE_BITWIDTH)\n"
        code += "\n"
        code += "// --- Redefinition of Core Graph Types for HLS ---\n"
        code += "// These typedefs override the standard integer types from common.h for\n"
        code += "// synthesis.\n"
        code += "typedef ap_uint<NODE_ID_BITWIDTH> node_id_t;\n"
        code += "typedef ap_uint<32> edge_id_t; // edge_id_t is not customized yet, keep as is.\n"
        code += "typedef ap_uint<DISTANCE_BITWIDTH>\n"
        code += "    ap_fixed_pod_t; // Used to hold bit representation of ap_fixed types\n"
        code += "typedef ap_fixed<DISTANCE_BITWIDTH, DISTANCE_INTEGER_PART> distance_t;\n"
        code += "typedef ap_uint<OUT_END_MARKER_BITWIDTH> out_end_marker_t;\n"
        code += "typedef ap_axiu<256, 0, 0, 0> node_dist_pkt_t;\n"
        code += "typedef ap_axiu<512, 0, 0, 0> write_burst_pkt_t;\n"
        code += "typedef ap_axiu<32, 0, 0, 0> ppb_request_pkt_t;\n"
        code += "typedef ap_axiu<512, 0, 0, 32> ppb_response_pkt_t;\n"
        code += "typedef ap_axiu<512, 0, 0, 0> cacheline_data_pkt_t;\n"
        little_header += code

        # shared params part:


        code = ""
        code += "#include <ap_axi_sdata.h>\n"
        code += "#include <ap_fixed.h>\n"
        code += "#include <ap_int.h>\n"
        code += "#include <hls_stream.h>\n"
        code += "#include <stdint.h>\n"
        code += "#include <string.h>\n"
        code += "\n"
        code += "#define PE_NUM 8\n"
        code += "#define DBL_PE_NUM 16\n"
        code += "#define LOG_PE_NUM 3\n"
        code += "#define L 4\n"
        code += "#define SRC_BUFFER_SIZE 4096\n"
        code += "#define LOG_SRC_BUFFER_SIZE 12\n"
        code += "\n"
        code += "#define NODE_ID_BITWIDTH 32\n"
        code += "#define DISTANCE_BITWIDTH 32\n"
        code += "#define DISTANCE_INTEGER_PART 16\n"
        code += "#define WEIGHT_BITWIDTH DISTANCE_BITWIDTH\n"
        code += "#define WEIGHT_INTEGER_PART DISTANCE_INTEGER_PART\n"
        code += "#define OUT_END_MARKER_BITWIDTH 4\n"
        code += "#define DIST_PER_WORD 16 // AXI_BUS_WIDTH / DISTANCE_BITWIDTH = 512 / 32 = 16\n"
        code += "#define LOG_DIST_PER_WORD                                                 \\\n"
        code += "    4 // log2(AXI_BUS_WIDTH / DISTANCE_BITWIDTH) = log2(512 / 32) = log2(16) = 4\n"
        code += "\n"
        code += "// --- New Memory Word and Bus Definitions ---\n"
        code += "#define AXI_BUS_WIDTH 512\n"
        code += "\n"
        code += "#define REDUCE_MEM_WIDTH 64\n"
        code += "typedef ap_uint<AXI_BUS_WIDTH> bus_word_t;\n"
        code += "typedef ap_uint<REDUCE_MEM_WIDTH> reduce_word_t;\n"
        code += "\n"
        code += "const int INFINITY_DIST = 16384;\n"
        code += "\n"
        code += "#define DISTANCES_PER_REDUCE_WORD (REDUCE_MEM_WIDTH / DISTANCE_BITWIDTH)\n"
        code += "\n"
        code += "// --- Redefinition of Core Graph Types for HLS ---\n"
        code += "// These typedefs override the standard integer types from common.h for\n"
        code += "// synthesis.\n"
        code += "typedef ap_uint<NODE_ID_BITWIDTH> node_id_t;\n"
        code += "typedef ap_uint<32> edge_id_t; // edge_id_t is not customized yet, keep as is.\n"
        code += "typedef ap_uint<DISTANCE_BITWIDTH>\n"
        code += "    ap_fixed_pod_t; // Used to hold bit representation of ap_fixed types\n"
        code += "typedef ap_fixed<DISTANCE_BITWIDTH, DISTANCE_INTEGER_PART> distance_t;\n"
        code += "typedef ap_uint<OUT_END_MARKER_BITWIDTH> out_end_marker_t;\n"
        code += "typedef ap_axiu<256, 0, 0, 0> node_dist_pkt_t;\n"
        code += "typedef ap_axiu<512, 0, 0, 0> write_burst_pkt_t;\n"
        code += "typedef ap_axiu<32, 0, 0, 8> cacheline_request_pkt_t;\n"
        code += "typedef ap_axiu<512, 0, 0, 8> cacheline_response_pkt_t;\n"
        code += "typedef ap_axiu<32, 0, 0, 0> ppb_request_pkt_t;\n"
        code += "typedef ap_axiu<512, 0, 0, 32> ppb_response_pkt_t;\n"
        code += "typedef ap_axiu<512, 0, 0, 0> cacheline_data_pkt_t;\n"

        shared_kernel_params += code

        big_header += "// --- Struct Type Definitions ---\n"
        sorted_defs = self._topologically_sort_structs()
        for hls_type, members in sorted_defs:
            big_header += hls_type.gen_decl(members) + "\n"
            little_header += hls_type.gen_decl(members) + "\n"
            shared_kernel_params += hls_type.gen_decl(members) + "\n"

        big_header += "// --- Top-Level Function Prototypes ---\n"
        
        big_header = write_func_sig(self.big_top_func,big_header)
        little_header = write_func_sig(self.little_top_func,little_header)
        shared_kernel_params = write_func_sig(self.big_top_func,shared_kernel_params)
        shared_kernel_params = write_func_sig(self.little_top_func,shared_kernel_params)
        shared_kernel_params = write_func_sig(self.big_hbm_writer_func,shared_kernel_params)
        shared_kernel_params = write_func_sig(self.little_hbm_writer_func,shared_kernel_params)

        big_header += f"#endif // {header_guard}\n"
        little_header += f"#endif // __GRAPHYFLOW_GRAPHYFLOW_LITTLE_H__\n"
        shared_kernel_params += f"#endif // __SHARED_KERNEL_PARAMS_H__\n"
        
        return big_header,little_header,shared_kernel_params
    

    def _generate_top_func(self) -> Tuple[str, str]:

        graphyflow_big_func = HLSFunction(name="graphyflow_big", comp=None)
        params = []

        # --- 1. Define Types & Params ---

        # Basic Types
        bus_word_t_type = HLSType(HLSBasicType.BUS_WORD_T)
        int_type = HLSType(HLSBasicType.INT)
        node_id_type = HLSType(HLSBasicType.NODE_ID)
        ap_fixed_pod_t_type = HLSType(HLSBasicType.AP_FIXED_POD)
        bool_type = HLSType(HLSBasicType.BOOL)
        uint8_type = HLSType(HLSBasicType.UINT8)
        ap_uint_4_type = HLSType(basic_type=HLSBasicType.AP_UINT, width=4)

        # Param 1: const bus_word_t *edge_props
        edge_props_type = HLSType(HLSBasicType.POINTER, sub_types=[bus_word_t_type], is_const_ptr=True)
        edge_props_var = HLSVar(var_name="edge_props", var_type=edge_props_type)

        # Param 2: int32_t num_nodes
        num_nodes_var = HLSVar(var_name="num_nodes", var_type=int_type)

        # Param 3: int32_t num_edges
        num_edges_var = HLSVar(var_name="num_edges", var_type=int_type)

        # Param 4: int32_t dst_num
        dst_num_var = HLSVar(var_name="dst_num", var_type=int_type)

        # Param 5: hls::stream<cacheline_request_pkt_t> &cacheline_req_stream
        cacheline_request_pkt_t_type = HLSType(HLSBasicType.CACHELINE_REQUEST_PKT_T)
        cacheline_req_stream_type = HLSType(HLSBasicType.STREAM, sub_types=[cacheline_request_pkt_t_type])
        cacheline_req_stream_var = HLSVar(var_name="cacheline_req_stream", var_type=cacheline_req_stream_type)

        # Param 6: hls::stream<cacheline_response_pkt_t> &cacheline_resp_stream
        cacheline_response_pkt_t_type = HLSType(HLSBasicType.CACHELINE_RESPONSE_PKT_T)
        cacheline_resp_stream_type = HLSType(HLSBasicType.STREAM, sub_types=[cacheline_response_pkt_t_type])
        cacheline_resp_stream_var = HLSVar(var_name="cacheline_resp_stream", var_type=cacheline_resp_stream_type)

        # Param 7: hls::stream<write_burst_pkt_t> &kernel_out_stream
        write_burst_pkt_t_type = HLSType(HLSBasicType.WRITE_BURST_PKT_T)
        kernel_out_stream_type = HLSType(HLSBasicType.STREAM, sub_types=[write_burst_pkt_t_type])
        kernel_out_stream_var = HLSVar(var_name="kernel_out_stream", var_type=kernel_out_stream_type)

        params.extend([edge_props_var, num_nodes_var, num_edges_var, dst_num_var, cacheline_req_stream_var, cacheline_resp_stream_var, kernel_out_stream_var])
        graphyflow_big_func.params = params

        # --- 2. Define Internal Types for Streams ---

        # Type: node_id_burst_t
        node_id_array_pe_type = HLSType(HLSBasicType.ARRAY, sub_types=[node_id_type], array_dims=["PE_NUM"])
        node_id_burst_t_type = HLSType(basic_type=HLSBasicType.STRUCT,
                                       struct_name="node_id_burst_t",
                                       struct_prop_names=["data"],
                                       sub_types=[node_id_array_pe_type])

        # Type: distance_req_pack_t
        distance_req_pack_t_type = HLSType(basic_type=HLSBasicType.STRUCT,
                                           struct_name="distance_req_pack_t",
                                           struct_prop_names=["idx", "offset", "end_flag"],
                                           sub_types=[node_id_array_pe_type, ap_uint_4_type, bool_type])
        if distance_req_pack_t_type.name not in self.struct_definitions:
            self.struct_definitions[distance_req_pack_t_type.name] = (distance_req_pack_t_type, distance_req_pack_t_type.struct_prop_names)

        # Type: edge_descriptor_batch_t
        edge_t_type = HLSType(basic_type=HLSBasicType.STRUCT,
                              struct_name="edge_t",
                              struct_prop_names=["src_id", "dst_id"],
                              sub_types=[node_id_type, node_id_type])
        if edge_t_type.name not in self.struct_definitions:
            self.struct_definitions[edge_t_type.name] = (edge_t_type, edge_t_type.struct_prop_names)

        edge_array_pe_type = HLSType(HLSBasicType.ARRAY, sub_types=[edge_t_type], array_dims=["PE_NUM"])
        edge_descriptor_batch_t_type = HLSType(basic_type=HLSBasicType.STRUCT,
                                               struct_name="edge_descriptor_batch_t",
                                               struct_prop_names=["edges", "end_pos"],
                                               sub_types=[edge_array_pe_type, int_type])
        if edge_descriptor_batch_t_type.name not in self.struct_definitions:
            self.struct_definitions[edge_descriptor_batch_t_type.name] = (edge_descriptor_batch_t_type, edge_descriptor_batch_t_type.struct_prop_names)

        # Type: update_tuple_t
        prop_array_pe_type = HLSType(HLSBasicType.ARRAY, sub_types=[ap_fixed_pod_t_type], array_dims=["PE_NUM"])
        update_tuple_t_type = HLSType(basic_type=HLSBasicType.STRUCT,
                                    struct_name="update_tuple_t",
                                    struct_prop_names=["node_id", "prop", "end_flag", "end_pos"],
                                    sub_types=[node_id_array_pe_type, prop_array_pe_type, bool_type, uint8_type])
        if update_tuple_t_type.name not in self.struct_definitions:
            self.struct_definitions[update_tuple_t_type.name] = (update_tuple_t_type, update_tuple_t_type.struct_prop_names)

        # --- 3. Function Body ---
        code_lines: List[HLSCodeLine] = []

        # #pragma HLS INTERFACE m_axi port = edge_props offset = slave bundle = gmem0
        code_lines.append(CodePragma(content="INTERFACE m_axi port = edge_props offset = slave bundle = gmem0"))
        # #pragma HLS INTERFACE s_axilite port = edge_props
        code_lines.append(CodePragma(content="INTERFACE s_axilite port = edge_props"))
        # #pragma HLS INTERFACE s_axilite port = num_nodes
        code_lines.append(CodePragma(content="INTERFACE s_axilite port = num_nodes"))
        # #pragma HLS INTERFACE s_axilite port = num_edges
        code_lines.append(CodePragma(content="INTERFACE s_axilite port = num_edges"))
        # #pragma HLS INTERFACE s_axilite port = dst_num
        code_lines.append(CodePragma(content="INTERFACE s_axilite port = dst_num"))
        # #pragma HLS INTERFACE s_axilite port = return
        code_lines.append(CodePragma(content="INTERFACE s_axilite port = return"))
        # #pragma HLS DATAFLOW
        code_lines.append(CodePragma(content="DATAFLOW"))

        code_lines.append(CodeOther(text="")) # Blank line

        # hls::stream<node_id_burst_t> stream_src_ids;
        stream_src_ids_type = HLSType(HLSBasicType.STREAM, sub_types=[node_id_burst_t_type])
        code_lines.append(CodeVarDecl(var_name="stream_src_ids", var_type=stream_src_ids_type))
        stream_src_ids_var = HLSVar(var_name="stream_src_ids", var_type=stream_src_ids_type)
        # #pragma HLS STREAM variable = stream_src_ids depth = 16
        code_lines.append(CodePragma(content="STREAM variable = stream_src_ids depth = 16"))

        # hls::stream<distance_req_pack_t> stream_dist_req;
        stream_dist_req_type = HLSType(HLSBasicType.STREAM, sub_types=[distance_req_pack_t_type])
        code_lines.append(CodeVarDecl(var_name="stream_dist_req", var_type=stream_dist_req_type))
        stream_dist_req_var = HLSVar(var_name="stream_dist_req", var_type=stream_dist_req_type)
        # #pragma HLS STREAM variable = stream_dist_req depth = 32
        code_lines.append(CodePragma(content="STREAM variable = stream_dist_req depth = 32"))

        # hls::stream<bus_word_t> stream_cachelines[PE_NUM];
        bus_word_stream_type = HLSType(HLSBasicType.STREAM, sub_types=[bus_word_t_type])
        stream_cachelines_type = HLSType(HLSBasicType.ARRAY, sub_types=[bus_word_stream_type], array_dims=["PE_NUM"])
        code_lines.append(CodeVarDecl(var_name="stream_cachelines", var_type=stream_cachelines_type))
        stream_cachelines_var = HLSVar(var_name="stream_cachelines", var_type=stream_cachelines_type)
        # #pragma HLS STREAM variable = stream_cachelines depth = 32
        code_lines.append(CodePragma(content="STREAM variable = stream_cachelines depth = 32"))

        code_lines.append(CodeOther(text="")) # Blank line

        # hls::stream<edge_descriptor_batch_t> edge_stream;
        edge_stream_type = HLSType(HLSBasicType.STREAM, sub_types=[edge_descriptor_batch_t_type])
        code_lines.append(CodeVarDecl(var_name="edge_stream", var_type=edge_stream_type))
        edge_stream_var = HLSVar(var_name="edge_stream", var_type=edge_stream_type)
        # #pragma HLS STREAM variable = edge_stream depth = 32
        code_lines.append(CodePragma(content="STREAM variable = edge_stream depth = 32"))

        # hls::stream<update_tuple_t> stream_edge_data;
        stream_edge_data_type = HLSType(HLSBasicType.STREAM, sub_types=[update_tuple_t_type])
        code_lines.append(CodeVarDecl(var_name="stream_edge_data", var_type=stream_edge_data_type))
        stream_edge_data_var = HLSVar(var_name="stream_edge_data", var_type=stream_edge_data_type)
        # #pragma HLS STREAM variable = stream_edge_data depth = 16
        code_lines.append(CodePragma(content="STREAM variable = stream_edge_data depth = 16"))

        code_lines.append(CodeOther(text="")) # Blank line

        # --- Data Loading ---
        # edge_descriptor_loader(edge_props, stream_src_ids, edge_stream, num_edges);
        code_lines.append(CodeCall(func=self.big_top_dataflow_funcs[0], params=[edge_props_var, stream_src_ids_var, edge_stream_var, num_edges_var]))

        code_lines.append(CodeOther(text="")) # Blank line

        # --- New COO-style Source Property Loading Pipeline ---
        # dist_req_packer(stream_src_ids, stream_dist_req, num_edges);
        code_lines.append(CodeCall(func=self.big_top_dataflow_funcs[1], params=[stream_src_ids_var, stream_dist_req_var, num_edges_var]))

        # cacheline_req_sender(stream_dist_req, cacheline_req_stream);
        code_lines.append(CodeCall(func=self.big_top_dataflow_funcs[2], params=[stream_dist_req_var, cacheline_req_stream_var]))

        # node_prop_resp_receiver(cacheline_resp_stream, stream_cachelines);
        code_lines.append(CodeCall(func=self.big_top_dataflow_funcs[3], params=[cacheline_resp_stream_var, stream_cachelines_var]))

        # merge_node_props(stream_cachelines, edge_stream, stream_edge_data, num_edges);
        code_lines.append(CodeCall(func=self.big_top_dataflow_funcs[4], params=[stream_cachelines_var, edge_stream_var, stream_edge_data_var, num_edges_var]))

        code_lines.append(CodeOther(text="")) # Blank line

        # --- Main Dataflow Processing ---
        # graphyflow_big_dataflow(stream_edge_data, kernel_out_stream, dst_num);
        code_lines.append(CodeCall(func=self.big_top_dataflow_funcs[5], params=[stream_edge_data_var, kernel_out_stream_var, dst_num_var]))

        code_lines.append(CodeOther(text="")) # Blank line

        # --- 4. Finalize ---
        graphyflow_big_func.codes = code_lines
        self.big_top_func = graphyflow_big_func


        # little part:
    
        graphyflow_little_func = HLSFunction(name="graphyflow_little", comp=None)
        params_gl: List[HLSVar] = []

        # --- 1. 为此函数独立定义类型 ---

        # 基本类型
        int_type = HLSType(HLSBasicType.INT)
        bus_word_t_type = HLSType(HLSBasicType.BUS_WORD_T)
        ppb_request_pkt_t_type = HLSType(HLSBasicType.PPB_REQUEST_PKT_T)
        ppb_response_pkt_t_type = HLSType(HLSBasicType.PPB_RESPONSE_PKT_T)
        write_burst_pkt_t_type = HLSType(HLSBasicType.WRITE_BURST_PKT_T)
        node_id_t_type = HLSType(HLSBasicType.NODE_ID)
        ap_fixed_pod_t_type = HLSType(HLSBasicType.AP_FIXED_POD)
        bool_type = HLSType(HLSBasicType.BOOL)
        uint8_type = HLSType(HLSBasicType.UINT8)
        reduce_word_t_type = HLSType(HLSBasicType.REDUCE_WORD_T)

        # --- 2. 定义参数 HLSVars ---

        # Param 1: const bus_word_t *edge_props
        param_edge_props = HLSVar(var_name="edge_props", 
                                  var_type=HLSType(HLSBasicType.POINTER, sub_types=[bus_word_t_type], is_const_ptr=True))

        # Param 2: int32_t num_nodes
        param_num_nodes = HLSVar(var_name="num_nodes", var_type=int_type)

        # Param 3: int32_t num_edges
        param_num_edges = HLSVar(var_name="num_edges", var_type=int_type)

        # Param 4: int32_t dst_num
        param_dst_num = HLSVar(var_name="dst_num", var_type=int_type)

        # Param 5: hls::stream<ppb_request_pkt_t> &ppb_req_stream
        param_ppb_req_stream = HLSVar(var_name="ppb_req_stream", 
                                      var_type=HLSType(HLSBasicType.STREAM, sub_types=[ppb_request_pkt_t_type]))

        # Param 6: hls::stream<ppb_response_pkt_t> &ppb_resp_stream
        param_ppb_resp_stream = HLSVar(var_name="ppb_resp_stream", 
                                       var_type=HLSType(HLSBasicType.STREAM, sub_types=[ppb_response_pkt_t_type]))

        # Param 7: hls::stream<write_burst_pkt_t> &kernel_out_stream
        param_kernel_out_stream = HLSVar(var_name="kernel_out_stream", 
                                         var_type=HLSType(HLSBasicType.STREAM, sub_types=[write_burst_pkt_t_type]))

        params_gl.extend([param_edge_props, param_num_nodes, param_num_edges, param_dst_num, 
                          param_ppb_req_stream, param_ppb_resp_stream, param_kernel_out_stream])
        graphyflow_little_func.params = params_gl

        # --- 3. 函数体 (构建本地列表) ---
        code_lines_gl: List[HLSCodeLine] = []

        # Pragmas
        code_lines_gl.append(CodePragma(content="INTERFACE m_axi port = edge_props offset = slave bundle = gmem0"))
        code_lines_gl.append(CodePragma(content="INTERFACE s_axilite port = edge_props"))
        code_lines_gl.append(CodePragma(content="INTERFACE s_axilite port = num_nodes"))
        code_lines_gl.append(CodePragma(content="INTERFACE s_axilite port = num_edges"))
        code_lines_gl.append(CodePragma(content="INTERFACE s_axilite port = dst_num"))
        code_lines_gl.append(CodePragma(content="INTERFACE s_axilite port = return"))
        code_lines_gl.append(CodePragma(content="DATAFLOW"))
        code_lines_gl.append(CodeOther(text="")) # 空行

        code_lines_gl.append(CodeComment(text="--- Existing streams ---"))

        # --- 定义内部流类型 ---

        # 内部类型 1: edge_descriptor_batch_t
        edge_t_type = HLSType(basic_type=HLSBasicType.STRUCT, 
                              struct_name="edge_t", 
                              struct_prop_names=["src_id", "dst_id"], 
                              sub_types=[node_id_t_type, node_id_t_type])
        edge_array_type = HLSType(HLSBasicType.ARRAY, sub_types=[edge_t_type], array_dims=["PE_NUM"])
        edge_descriptor_batch_t_type = HLSType(basic_type=HLSBasicType.STRUCT, 
                                               struct_name="edge_descriptor_batch_t", 
                                               struct_prop_names=["edges", "end_pos"], 
                                               sub_types=[edge_array_type, int_type])

        # hls::stream<edge_descriptor_batch_t> edge_stream;
        edge_stream_type = HLSType(HLSBasicType.STREAM, sub_types=[edge_descriptor_batch_t_type])
        code_lines_gl.append(CodeVarDecl(var_name="edge_stream", var_type=edge_stream_type))
        code_lines_gl.append(CodePragma(content="STREAM variable = edge_stream depth = 32"))
        # (创建 HLSVar 供 CodeCall 使用)
        edge_stream_var = HLSVar(var_name="edge_stream", var_type=edge_stream_type)


        # 内部类型 2: update_tuple_t
        node_id_array_type_internal = HLSType(HLSBasicType.ARRAY, sub_types=[node_id_t_type], array_dims=["PE_NUM"])
        prop_array_type_internal = HLSType(HLSBasicType.ARRAY, sub_types=[ap_fixed_pod_t_type], array_dims=["PE_NUM"])
        update_tuple_t_type = HLSType(basic_type=HLSBasicType.STRUCT, 
                                      struct_name="update_tuple_t", 
                                      struct_prop_names=["node_id", "prop", "end_flag", "end_pos"], 
                                      sub_types=[node_id_array_type_internal, prop_array_type_internal, bool_type, uint8_type])

        # hls::stream<update_tuple_t> stream_edge_data;
        stream_edge_data_type = HLSType(HLSBasicType.STREAM, sub_types=[update_tuple_t_type])
        code_lines_gl.append(CodeVarDecl(var_name="stream_edge_data", var_type=stream_edge_data_type))
        code_lines_gl.append(CodePragma(content="STREAM variable = stream_edge_data depth = 8"))
        # (创建 HLSVar 供 CodeCall 使用)
        stream_edge_data_var = HLSVar(var_name="stream_edge_data", var_type=stream_edge_data_type)


        # 内部类型 3: hls::stream<reduce_word_t> pe_mem_outs[PE_NUM];
        pe_mem_out_stream_type = HLSType(HLSBasicType.STREAM, sub_types=[reduce_word_t_type])
        pe_mem_outs_type = HLSType(HLSBasicType.ARRAY, sub_types=[pe_mem_out_stream_type], array_dims=["PE_NUM"])
        code_lines_gl.append(CodeVarDecl(var_name="pe_mem_outs", var_type=pe_mem_outs_type))
        code_lines_gl.append(CodePragma(content="STREAM variable = pe_mem_outs depth = 8"))
        # (创建 HLSVar 供 CodeCall 使用)
        pe_mem_outs_var = HLSVar(var_name="pe_mem_outs", var_type=pe_mem_outs_type)

        code_lines_gl.append(CodeOther(text="")) # 空行

        # --- Data Loading ---
        code_lines_gl.append(CodeComment(text="--- Data Loading ---"))
        code_lines_gl.append(CodeOther(text="edge_descriptor_loader(edge_props, edge_stream, num_edges);"))
        code_lines_gl.append(CodeOther(text="request_manager(edge_stream, ppb_req_stream, ppb_resp_stream, stream_edge_data, num_edges);"))
        code_lines_gl.append(CodeOther(text="")) # 空行

        # --- Reduction ---
        code_lines_gl.append(CodeComment(text="--- Reduction ---"))
        code_lines_gl.append(CodeOther(text="Reduc_105_unit_reduce(stream_edge_data, pe_mem_outs, num_edges, dst_num);"))
        code_lines_gl.append(CodeOther(text="Reduc_105_drain_multi_pe(pe_mem_outs, kernel_out_stream, dst_num);"))


        # --- 4. 显式地将所有代码行赋值给函数 ---
        graphyflow_little_func.codes = code_lines_gl

        self.little_top_func = graphyflow_little_func


    def _generate_source_file(self, header_name: str) -> str:
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

        for func in self.helper_funcs:
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

        for func in self.helper_funcs:
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
        # edge_descriptor_loader(const bus_word_t *edge_props_ddr,
        #                hls::stream<node_id_burst_t> &stream_src_ids,
        #                hls::stream<edge_descriptor_batch_t> &edge_stream,
        #                int32_t num_edges)

 
        edge_descriptor_loader_func_b = HLSFunction(name="edge_descriptor_loader", comp=comp)
        params = []

        # 2. 定义函数参数
        edge_props_ddr = HLSVar(var_name="edge_props_ddr", var_type=HLSType(HLSBasicType.POINTER,sub_types=[
            HLSType(basic_type=HLSBasicType.BUS_WORD_T)
            ],is_const_ptr=True))

        # 2a. 定义 node_id_burst_t 类型 (用于 stream_src_ids)
        # struct node_id_burst_t { node_id_t data[PE_NUM]; }
        node_id_array_type = HLSType(HLSBasicType.ARRAY,sub_types=[
            HLSType(basic_type=HLSBasicType.NODE_ID)
            ],array_dims=["PE_NUM"])
        node_id_burst_t_type = HLSType(basic_type=HLSBasicType.STRUCT,
            struct_name="node_id_burst_t",
            struct_prop_names=["data"],
            sub_types=[node_id_array_type])
        if node_id_burst_t_type.name not in self.struct_definitions:
                self.struct_definitions[node_id_burst_t_type.name] = (node_id_burst_t_type, ["data"])


        stream_src_ids = HLSVar(var_name="stream_src_ids", var_type=HLSType(HLSBasicType.STREAM,sub_types=[
            node_id_burst_t_type
            ]))

        # 2b. 定义 edge_descriptor_batch_t 类型 (用于 edge_stream)
        # struct edge_t { node_id_t src_id; node_id_t dst_id; }
        edge_t_type = HLSType(basic_type=HLSBasicType.STRUCT,
            struct_name="edge_t",
            struct_prop_names=["src_id", "dst_id"],
            sub_types=[
                HLSType(basic_type=HLSBasicType.NODE_ID),
                HLSType(basic_type=HLSBasicType.NODE_ID)
            ])
        if edge_t_type.name not in self.struct_definitions:
                self.struct_definitions[edge_t_type.name] = (edge_t_type, edge_t_type.struct_prop_names)

        # struct edge_descriptor_batch_t { edge_t edges[PE_NUM]; int end_pos; }
        edges_array_type = HLSType(HLSBasicType.ARRAY,sub_types=[
            edge_t_type
            ],array_dims=["PE_NUM"])
        end_pos_type = HLSType(basic_type=HLSBasicType.INT)
        edge_descriptor_batch_t_type = HLSType(HLSBasicType.STRUCT,
            struct_name="edge_descriptor_batch_t",
            struct_prop_names=["edges", "end_pos"],
            sub_types=[
                edges_array_type,
                end_pos_type
            ])
        if edge_descriptor_batch_t_type.name not in self.struct_definitions:
            self.struct_definitions[edge_descriptor_batch_t_type.name] = (edge_descriptor_batch_t_type, edge_descriptor_batch_t_type.struct_prop_names)
        

        edge_stream = HLSVar(var_name="edge_stream", var_type=HLSType(HLSBasicType.STREAM,sub_types=[
            edge_descriptor_batch_t_type
            ]))

        # 2c. 定义最后一个参数
        num_edges = HLSVar(var_name="num_edges", var_type=HLSType(HLSBasicType.INT))

        # 2d. 组合参数
        params.extend([edge_props_ddr, stream_src_ids, edge_stream, num_edges])
        edge_descriptor_loader_func_b.params = params

        # 3. 设置相关的 #defines
        self.defines["PE_NUM"] = "8" 
        self.defines["NODE_ID_BITWIDTH"] = "32"
        self.defines["WEIGHT_BITWIDTH"] = "32"
        self.defines["AXI_BUS_WIDTH"] = "512"

        # 4. 开始构建函数体 (CodeLine 列表)
        code_lines: List[HLSCodeLine] = []

        # const int bits_per_edge = ...
        code_lines.append(CodeVarDecl(var_name="bits_per_edge", var_type=HLSType(HLSBasicType.INT),init_val = "NODE_ID_BITWIDTH + WEIGHT_BITWIDTH",const=True))
        # const int edges_per_word = ...
        code_lines.append(CodeVarDecl(var_name="edges_per_word", var_type=HLSType(HLSBasicType.INT), init_val="AXI_BUS_WIDTH / bits_per_edge",const=True))
        # const int num_wide_reads = ...
        code_lines.append(CodeVarDecl(var_name="num_wide_reads", var_type=HLSType(HLSBasicType.INT), init_val="(num_edges + edges_per_word - 1) / edges_per_word",const=True))

        # int edges_read = 0;
        code_lines.append(CodeVarDecl(var_name="edges_read", var_type=HLSType(HLSBasicType.INT), init_val="0",const=False))
        # edge_descriptor_batch_t edge_batch;
        # (使用缓存的类型)
        edge_batch_type_from_cache = HLSType._full_to_type[HLSType._name_to_full["edge_descriptor_batch_t"]]
        code_lines.append(CodeVarDecl(var_name="edge_batch", var_type=edge_batch_type_from_cache, init_val=None,const=False))

        # #pragma ...
        code_lines.append(CodePragma(content="ARRAY_PARTITION variable = edge_batch.edges complete dim = 0"))

        # edge_batch.end_pos = 0;
        # (为局部变量创建 HLSVar 句柄以便在表达式中使用)
        edge_batch_end_pos_var = HLSVar(var_name="edge_batch.end_pos", var_type=HLSType(HLSBasicType.INT))
        const_0_expr = HLSExpr(HLSExprT.CONST, 0)
        code_lines.append(CodeAssign(var=edge_batch_end_pos_var, expr=const_0_expr))

        # node_id_burst_t src_id_burst;
        src_id_burst_type_from_cache = HLSType._full_to_type[HLSType._name_to_full["node_id_burst_t"]]
        code_lines.append(CodeVarDecl(var_name="src_id_burst", var_type=src_id_burst_type_from_cache, init_val=None,const=False))

        # #pragma ...
        code_lines.append(CodePragma(content="ARRAY_PARTITION variable = src_id_burst.data complete dim = 0"))

        # #if ...
        code_lines.append(CodeOther(text="#if (NODE_ID_BITWIDTH == 32) && (WEIGHT_BITWIDTH == 32)"))

        # --- 创建第一个 For 循环 (i) ---
        for_loop_i_codes: List[HLSCodeLine] = []
        code_lines.append(CodeComment("LOOP_EDL_READ:"))
        # (创建 HLSVar 句柄)
        num_wide_reads_var = HLSVar(var_name="num_wide_reads", var_type=HLSType(HLSBasicType.INT))
        for_loop_i = CodeFor(codes=for_loop_i_codes, 
                             iter_limit=num_wide_reads_var, # 使用 HLSVar
                             iter_name="i", 
                             iter_val_type=HLSType(HLSBasicType.INT))
        code_lines.append(for_loop_i)

        # --- 循环 (i) 内部 ---
        # #pragma PIPELINE
        for_loop_i_codes.append(CodePragma(content="PIPELINE II = 1"))

        # bus_word_t wide_word = edge_props_ddr[i];
        # (init_val 是一个字符串, 可以处理数组访问)
        for_loop_i_codes.append(CodeVarDecl(var_name="wide_word", var_type=HLSType(HLSBasicType.BUS_WORD_T), init_val="edge_props_ddr[i]", const=False))

        # --- 创建第二个 For 循环 (j) ---
        for_loop_j_codes: List[HLSCodeLine] = []
        for_loop_i_codes.append(CodeComment("LOOP_EDL_UNPACK:"))
        # (iter_limit 可以是字符串)
        edges_per_word_str = "edges_per_word" 
        for_loop_j = CodeFor(codes=for_loop_j_codes, 
                             iter_limit=edges_per_word_str, # 使用字符串
                             iter_name="j", 
                             iter_val_type=HLSType(HLSBasicType.INT))
        for_loop_i_codes.append(for_loop_j)

        # --- 循环 (j) 内部 ---
        # #pragma UNROLL
        for_loop_j_codes.append(CodePragma(content="UNROLL"))

        # --- 创建 If 语句 ---
        if_codes: List[HLSCodeLine] = []
        # (创建 HLSVar 句柄)
        edges_read_var = HLSVar(var_name="edges_read", var_type=HLSType(HLSBasicType.INT))
        j_var = HLSVar(var_name="j", var_type=HLSType(HLSBasicType.INT))
        # num_edges (来自参数)
        # (构建表达式: edges_read + j < num_edges)
        expr_add = HLSExpr(HLSExprT.BINOP, dfir.BinOp.ADD, operands=[HLSExpr(HLSExprT.VAR, edges_read_var), HLSExpr(HLSExprT.VAR, j_var)])
        expr_lt = HLSExpr(HLSExprT.BINOP, dfir.BinOp.LT, operands=[expr_add, HLSExpr(HLSExprT.VAR, num_edges)])

        if_statement = CodeIf(expr=expr_lt, if_codes=if_codes)
        for_loop_j_codes.append(if_statement)

        # --- If 语句内部 ---
        # ap_uint<bits_per_edge> packed_edge = ...
        # (这是一个技巧, 用于处理模板参数是变量的情况)
        packed_edge_type = HLSType(basic_type=HLSBasicType.AP_UINT, width=1) # 假的 width
        packed_edge_type.name = "ap_uint<bits_per_edge>" # 手动覆盖
        init_val_str = "wide_word.range((j + 1) * bits_per_edge - 1, j * bits_per_edge)"
        if_codes.append(CodeVarDecl(var_name="packed_edge", var_type=packed_edge_type, init_val=init_val_str))

        # edge_t edge;
        edge_t_type_from_cache = HLSType._full_to_type[HLSType._name_to_full["edge_t"]]
        if_codes.append(CodeVarDecl(var_name="edge", var_type=edge_t_type_from_cache))
        edge_var = HLSVar(var_name="edge", var_type=edge_t_type_from_cache)

        # node_id_t src_id;
        node_id_type = HLSType(basic_type=HLSBasicType.NODE_ID)
        if_codes.append(CodeVarDecl(var_name="src_id", var_type=node_id_type))
        src_id_var = HLSVar(var_name="src_id", var_type=node_id_type)


        # edge.dst_id = packed_edge.range(NODE_ID_BITWIDTH - 1, 0);
        edge_dst_id_var = HLSVar(var_name="edge.dst_id", var_type=node_id_type)
        expr_dst_id = HLSExpr(HLSExprT.CONST, "packed_edge.range(NODE_ID_BITWIDTH - 1, 0)")
        if_codes.append(CodeAssign(var=edge_dst_id_var, expr=expr_dst_id))

        # edge.src_id = packed_edge.range(bits_per_edge - 1, NODE_ID_BITWIDTH);
        edge_src_id_var = HLSVar(var_name="edge.src_id", var_type=node_id_type)
        expr_src_id = HLSExpr(HLSExprT.CONST, "packed_edge.range(bits_per_edge - 1, NODE_ID_BITWIDTH)")
        if_codes.append(CodeAssign(var=edge_src_id_var, expr=expr_src_id))

        # src_id = edge.src_id;
        if_codes.append(CodeAssign(var=src_id_var, expr=HLSExpr(HLSExprT.VAR, edge_src_id_var)))

        # edge_batch.edges[j] = edge;
        # (HLSVar 的 name 字段可以包含数组访问)
        edge_batch_edges_j_var = HLSVar(var_name="edge_batch.edges[j]", var_type=edge_t_type_from_cache)
        if_codes.append(CodeAssign(var=edge_batch_edges_j_var, expr=HLSExpr(HLSExprT.VAR, edge_var)))

        # src_id_burst.data[j] = src_id;
        src_id_burst_data_j_var = HLSVar(var_name="src_id_burst.data[j]", var_type=node_id_type)
        if_codes.append(CodeAssign(var=src_id_burst_data_j_var, expr=HLSExpr(HLSExprT.VAR, src_id_var)))


        # --- 回到循环 (i) 内部 (在循环 j 之后) ---
        # stream_src_ids.write(src_id_burst);
        # (stream_src_ids 来自参数)
        src_id_burst_var = HLSVar(var_name="src_id_burst", var_type=src_id_burst_type_from_cache)
        for_loop_i_codes.append(CodeWriteStream(stream_var=stream_src_ids, in_expr=src_id_burst_var))

        # edges_read += edges_per_word; (即 edges_read = edges_read + edges_per_word)
        edges_per_word_hls_var = HLSVar(var_name="edges_per_word", var_type=HLSType(HLSBasicType.INT))
        expr_add_edges_read = HLSExpr(HLSExprT.BINOP, dfir.BinOp.ADD, operands=[HLSExpr(HLSExprT.VAR, edges_read_var), HLSExpr(HLSExprT.VAR, edges_per_word_hls_var)])
        for_loop_i_codes.append(CodeAssign(var=edges_read_var, expr=expr_add_edges_read))

        # edge_batch.end_pos = (edges_read <= num_edges) ...
        # (再次使用 HLSExprT.CONST 技巧处理三元运算符)
        expr_ternary = HLSExpr(HLSExprT.CONST, "(edges_read <= num_edges) ? edges_per_word : (num_edges % edges_per_word)")
        for_loop_i_codes.append(CodeAssign(var=edge_batch_end_pos_var, expr=expr_ternary))

        # edge_stream.write(edge_batch);
        # (edge_stream 来自参数)
        edge_batch_var = HLSVar(var_name="edge_batch", var_type=edge_batch_type_from_cache)
        for_loop_i_codes.append(CodeWriteStream(stream_var=edge_stream, in_expr=edge_batch_var))

        # edge_batch.end_pos = 0;
        for_loop_i_codes.append(CodeAssign(var=edge_batch_end_pos_var, expr=const_0_expr))


        # --- 回到主代码 (在循环 i 之后) ---
        # #else
        code_lines.append(CodeOther(text="#else"))
        # #error ...
        code_lines.append(CodeOther(text='#error "edge_descriptor_loader currently only supports 32-bit node_id and 32-bit weight."'))
        # #endif
        code_lines.append(CodeOther(text="#endif"))

        # 5. 完成函数体
        edge_descriptor_loader_func_b.codes = code_lines

        # (可选：添加一个结束注释)
        code_lines.append(CodeComment("End of Edge Descriptor Loader Function"))
        

        edge_descriptor_loader_func_l = HLSFunction(name="edge_descriptor_loader", comp=comp)
        params_edl: List[HLSVar] = []

        # --- 1. 为此函数独立定义类型 ---

        # 基本类型
        int_type = HLSType(HLSBasicType.INT)
        bus_word_t_type = HLSType(HLSBasicType.BUS_WORD_T)
        node_id_t_type = HLSType(HLSBasicType.NODE_ID)

        # 结构体: edge_t
        edge_t_type = HLSType(basic_type=HLSBasicType.STRUCT, 
                              struct_name="edge_t", 
                              struct_prop_names=["src_id", "dst_id"], 
                              sub_types=[node_id_t_type, node_id_t_type])

        # 结构体: edge_descriptor_batch_t
        edge_array_type = HLSType(HLSBasicType.ARRAY, sub_types=[edge_t_type], array_dims=["PE_NUM"])
        edge_descriptor_batch_t_type = HLSType(basic_type=HLSBasicType.STRUCT, 
                                               struct_name="edge_descriptor_batch_t", 
                                               struct_prop_names=["edges", "end_pos"], 
                                               sub_types=[edge_array_type, int_type])

        # 结构体: node_id_burst_t
        node_id_array_type = HLSType(HLSBasicType.ARRAY, sub_types=[node_id_t_type], array_dims=["PE_NUM"])
        node_id_burst_t_type = HLSType(basic_type=HLSBasicType.STRUCT,
                                       struct_name="node_id_burst_t",
                                       struct_prop_names=["data"],
                                       sub_types=[node_id_array_type])

        # 特殊 ap_uint 类型 (用于 packed_edge)
        # 我们将使用 CodeOther 来声明它，因为它的宽度是一个变量
        # ap_uint_bits_per_edge_type = HLSType(HLSBasicType.AP_UINT, width=1) 
        # ap_uint_bits_per_edge_type.name = "ap_uint<bits_per_edge>"


        # --- 2. 定义参数 HLSVars ---

        # Param 1: const bus_word_t *edge_props_ddr
        param_edge_props_ddr = HLSVar(var_name="edge_props_ddr", 
                                      var_type=HLSType(HLSBasicType.POINTER, sub_types=[bus_word_t_type], is_const_ptr=True))

        # Param 2: hls::stream<edge_descriptor_batch_t> &edge_stream
        param_edge_stream = HLSVar(var_name="edge_stream", 
                                   var_type=HLSType(HLSBasicType.STREAM, sub_types=[edge_descriptor_batch_t_type]))

        # Param 3: int32_t num_edges
        param_num_edges = HLSVar(var_name="num_edges", var_type=int_type)

        params_edl.extend([param_edge_props_ddr, param_edge_stream, param_num_edges])
        edge_descriptor_loader_func_l.params = params_edl

        # --- 3. 函数体 (构建本地列表) ---
        code_lines_edl: List[HLSCodeLine] = []

        # const int bits_per_edge = ...
        code_lines_edl.append(CodeVarDecl(var_name="bits_per_edge", var_type=int_type, init_val="NODE_ID_BITWIDTH + WEIGHT_BITWIDTH", const=True))
        # const int edges_per_word = ...
        code_lines_edl.append(CodeVarDecl(var_name="edges_per_word", var_type=int_type, init_val="AXI_BUS_WIDTH / bits_per_edge", const=True))
        # const int num_wide_reads = ...
        code_lines_edl.append(CodeVarDecl(var_name="num_wide_reads", var_type=int_type, init_val="(num_edges + edges_per_word - 1) / edges_per_word", const=True))
        num_wide_reads_var = HLSVar(var_name="num_wide_reads", var_type=int_type) # 用于 for 循环

        # int edges_read = 0;
        code_lines_edl.append(CodeVarDecl(var_name="edges_read", var_type=int_type, init_val=0))
        edges_read_var = HLSVar(var_name="edges_read", var_type=int_type)

        # edge_descriptor_batch_t edge_batch;
        code_lines_edl.append(CodeVarDecl(var_name="edge_batch", var_type=edge_descriptor_batch_t_type))
        edge_batch_var = HLSVar(var_name="edge_batch", var_type=edge_descriptor_batch_t_type)

        # #pragma HLS ARRAY_PARTITION ...
        code_lines_edl.append(CodePragma(content="ARRAY_PARTITION variable = edge_batch.edges complete dim = 0"))

        # edge_batch.end_pos = 0;
        edge_batch_end_pos_var = HLSVar(var_name="edge_batch.end_pos", var_type=int_type)
        code_lines_edl.append(CodeAssign(var=edge_batch_end_pos_var, expr=HLSExpr(HLSExprT.CONST, 0)))

        # node_id_burst_t src_id_burst;
        code_lines_edl.append(CodeVarDecl(var_name="src_id_burst", var_type=node_id_burst_t_type))
        # (HLSVar 供 write 使用)
        src_id_burst_var = HLSVar(var_name="src_id_burst", var_type=node_id_burst_t_type)


        # #pragma HLS ARRAY_PARTITION ...
        code_lines_edl.append(CodePragma(content="ARRAY_PARTITION variable = src_id_burst.data complete dim = 0"))
        code_lines_edl.append(CodeOther(text="")) # 空行

        # #if (NODE_ID_BITWIDTH == 32) && (WEIGHT_BITWIDTH == 32)
        code_lines_edl.append(CodeOther(text="#if (NODE_ID_BITWIDTH == 32) && (WEIGHT_BITWIDTH == 32)"))

        # LOOP_EDL_READ: for (int i = 0; ...
        for_loop_1_codes: List[HLSCodeLine] = []
        for_loop_1 = CodeFor(codes=for_loop_1_codes,
                             iter_limit=num_wide_reads_var, # HLSVar
                             iter_cmp="<",
                             iter_name="i",
                             iter_start="0",
                             iter_step="i++",
                             iter_val_type=int_type)
        code_lines_edl.append(CodeComment(text="LOOP_EDL_READ:"))
        code_lines_edl.append(for_loop_1)

        # --- 内部循环 1 ---
        # #pragma HLS PIPELINE II = 1
        for_loop_1_codes.append(CodePragma(content="PIPELINE II = 1"))

        # bus_word_t wide_word = edge_props_ddr[i];
        for_loop_1_codes.append(CodeVarDecl(var_name="wide_word", var_type=bus_word_t_type, init_val="edge_props_ddr[i]"))

        # LOOP_EDL_UNPACK: for (int j = 0; ...
        for_loop_2_codes: List[HLSCodeLine] = []
        for_loop_2 = CodeFor(codes=for_loop_2_codes,
                             iter_limit="edges_per_word", # 字符串
                             iter_cmp="<",
                             iter_name="j",
                             iter_start="0",
                             iter_step="j++",
                             iter_val_type=int_type)
        for_loop_1_codes.append(CodeComment(text="LOOP_EDL_UNPACK:"))
        for_loop_1_codes.append(for_loop_2)

        # --- 内部循环 2 ---
        # #pragma HLS UNROLL
        for_loop_2_codes.append(CodePragma(content="UNROLL"))

        # if (edges_read + j < num_edges)
        if_1_codes: List[HLSCodeLine] = []
        if_1_expr = HLSExpr(HLSExprT.CONST, "edges_read + j < num_edges") # C++ 表达式
        if_1 = CodeIf(expr=if_1_expr, if_codes=if_1_codes)
        for_loop_2_codes.append(if_1)

        # --- 内部 If ---
        # ap_uint<bits_per_edge> packed_edge = ...
        if_1_codes.append(CodeOther(text="ap_uint<bits_per_edge> packed_edge = wide_word.range((j + 1) * bits_per_edge - 1, j * bits_per_edge);"))

        # edge_t edge;
        if_1_codes.append(CodeVarDecl(var_name="edge", var_type=edge_t_type))
        edge_var = HLSVar(var_name="edge", var_type=edge_t_type)

        # node_id_t src_id;
        if_1_codes.append(CodeVarDecl(var_name="src_id", var_type=node_id_t_type))
        src_id_var = HLSVar(var_name="src_id", var_type=node_id_t_type)

        # edge.dst_id = ...
        if_1_codes.append(CodeOther(text="edge.dst_id = packed_edge.range(NODE_ID_BITWIDTH - 1, 0);"))
        # edge.src_id = ...
        if_1_codes.append(CodeOther(text="edge.src_id = packed_edge.range(bits_per_edge - 1, NODE_ID_BITWIDTH);"))

        # src_id = edge.src_id;
        edge_src_id_var = HLSVar(var_name="edge.src_id", var_type=node_id_t_type)
        if_1_codes.append(CodeAssign(var=src_id_var, expr=HLSExpr(HLSExprT.VAR, edge_src_id_var)))

        # edge_batch.edges[j] = edge;
        if_1_codes.append(CodeOther(text="edge_batch.edges[j] = edge;"))
        # src_id_burst.data[j] = src_id;
        if_1_codes.append(CodeOther(text="src_id_burst.data[j] = src_id;"))
        # --- 结束 If ---
        # --- 结束 内部循环 2 ---

        # edges_read += edges_per_word;
        for_loop_1_codes.append(CodeOther(text="edges_read += edges_per_word;"))

        # edge_batch.end_pos = (edges_read <= num_edges) ? ...
        expr_ternary = HLSExpr(HLSExprT.CONST, "(edges_read <= num_edges) ? edges_per_word : (num_edges % edges_per_word)")
        for_loop_1_codes.append(CodeAssign(var=edge_batch_end_pos_var, expr=expr_ternary))

        # edge_stream.write(edge_batch);
        for_loop_1_codes.append(CodeWriteStream(stream_var=param_edge_stream, in_expr=edge_batch_var))

        # edge_batch.end_pos = 0;
        for_loop_1_codes.append(CodeAssign(var=edge_batch_end_pos_var, expr=HLSExpr(HLSExprT.CONST, 0)))
        # --- 结束 内部循环 1 ---

        # #else
        code_lines_edl.append(CodeOther(text="#else"))
        # #error ...
        code_lines_edl.append(CodeOther(text="#error \"edge_descriptor_loader currently only supports 32-bit node_id and 32-bit weight.\""))
        # #endif
        code_lines_edl.append(CodeOther(text="#endif"))


        # --- 4. 显式地将所有代码行赋值给函数 ---
        edge_descriptor_loader_func_l.codes = code_lines_edl

        self.big_scatter_funcs.append(edge_descriptor_loader_func_b)
        self.little_scatter_funcs.append(edge_descriptor_loader_func_l)

        self.big_top_dataflow_funcs.append(edge_descriptor_loader_func_b)
        self.little_top_dataflow_funcs.append(edge_descriptor_loader_func_l)



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
                                      width="NODE_ID_BITWIDTH - LOG_DIST_PER_WORD")
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
        distance_req_pack_t_type = HLSType(basic_type=HLSBasicType.STRUCT,
                                           struct_name="distance_req_pack_t",
                                           struct_prop_names=["idx", "offset", "end_flag"],
                                           sub_types=[node_id_array_pe_type, offset_type, bool_type])
        
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
        last_idx_max_var = HLSVar(var_name="last_idx_max", var_type=cache_idx_elem_type)
        
        # for (int32_t node_burst_idx = 0; ...
        for_loop_1_codes: List[HLSCodeLine] = []
        # (for_loop_1 object created and added to code_lines at the end)
        
        # --- Inside for(node_burst_idx) ---
        # #pragma HLS PIPELINE II = 1
        for_loop_1_codes.append(CodePragma(content="PIPELINE II = 1"))
        
        # ap_uint<...> cache_idx[PE_NUM];
        cache_idx_type = HLSType(HLSBasicType.ARRAY, sub_types=[cache_idx_elem_type], array_dims=["PE_NUM"])
        for_loop_1_codes.append(CodeVarDecl(var_name="cache_idx", var_type=cache_idx_type))
        
        # #pragma HLS ARRAY_PARTITION variable = cache_idx complete dim = 0
        for_loop_1_codes.append(CodePragma(content="ARRAY_PARTITION variable = cache_idx complete dim = 0"))
        
        # node_id_burst_t node_id_burst = src_id_burst_stream.read();
        for_loop_1_codes.append(CodeVarDecl(var_name="node_id_burst", var_type=node_id_burst_t_type, init_val="src_id_burst_stream.read()"))
        
        # #pragma HLS ARRAY_PARTITION variable = node_id_burst.data complete dim = 0
        for_loop_1_codes.append(CodePragma(content="ARRAY_PARTITION variable = node_id_burst.data complete dim = 0"))
        
        # --- Build for(pe_idx) 1 ---
        for_loop_2_codes: List[HLSCodeLine] = []
        # #pragma HLS UNROLL
        for_loop_2_codes.append(CodePragma(content="UNROLL"))
        # cache_idx[pe_idx] = node_id_burst.data[pe_idx] >> LOG_DIST_PER_WORD;
        cache_idx_pe_idx_var = HLSVar(var_name="cache_idx[pe_idx]", var_type=cache_idx_elem_type)
        assign_expr_1 = HLSExpr(HLSExprT.CONST, "node_id_burst.data[pe_idx] >> LOG_DIST_PER_WORD")
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
        for_loop_1_codes.append(CodeVarDecl(var_name="cache_idx_diffs", var_type=cache_idx_diffs_type))
        
        # #pragma HLS ARRAY_PARTITION variable = cache_idx_diffs complete dim = 0
        for_loop_1_codes.append(CodePragma(content="ARRAY_PARTITION variable = cache_idx_diffs complete dim = 0"))
        
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
        # --- End for(pe_idx) 2 ---
        
        # --- Build IF_1 (cache_idx_diffs[PE_NUM - 1]) ---
        if_1_codes: List[HLSCodeLine] = []
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
        valid_mask_pe_idx_var = HLSVar(var_name="valid_mask[pe_idx]", var_type=HLSType(HLSBasicType.AP_UINT, width=1)) # Single bit assignment
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
        
        # ap_uint<4> num_unread = count_end_ones(valid_mask);
        if_1_codes.append(CodeVarDecl(var_name="num_unread", var_type=offset_type, init_val="count_end_ones(valid_mask)"))
        num_unread_var = HLSVar(var_name="num_unread", var_type=offset_type)
        
        # distance_req_pack_t req_pack;
        if_1_codes.append(CodeVarDecl(var_name="req_pack", var_type=distance_req_pack_t_type))
        req_pack_var = HLSVar(var_name="req_pack", var_type=distance_req_pack_t_type)
        
        # #pragma HLS ARRAY_PARTITION variable = req_pack.idx complete dim = 0
        if_1_codes.append(CodePragma(content="ARRAY_PARTITION variable = req_pack.idx complete dim = 0"))
        
        # req_pack.offset = num_unread;
        req_pack_offset_var = HLSVar(var_name="req_pack.offset", var_type=offset_type)
        if_1_codes.append(CodeAssign(var=req_pack_offset_var, expr=HLSExpr(HLSExprT.VAR, num_unread_var)))
        
        # req_pack.end_flag = false;
        req_pack_end_flag_var = HLSVar(var_name="req_pack.end_flag", var_type=bool_type)
        assign_expr_5 = HLSExpr(HLSExprT.CONST, False)
        if_1_codes.append(CodeAssign(var=req_pack_end_flag_var, expr=assign_expr_5))
        
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
        
        # distance_req_pack_t end_req_pack;
        code_lines.append(CodeVarDecl(var_name="end_req_pack", var_type=distance_req_pack_t_type))
        end_req_pack_var = HLSVar(var_name="end_req_pack", var_type=distance_req_pack_t_type)
        
        # end_req_pack.end_flag = true;
        end_req_pack_end_flag_var = HLSVar(var_name="end_req_pack.end_flag", var_type=bool_type)
        assign_expr_8 = HLSExpr(HLSExprT.CONST, True)
        code_lines.append(CodeAssign(var=end_req_pack_end_flag_var, expr=assign_expr_8))
        
        # end_req_pack.offset = 8;
        end_req_pack_offset_var = HLSVar(var_name="end_req_pack.offset", var_type=offset_type)
        assign_expr_9 = HLSExpr(HLSExprT.CONST, 8)
        code_lines.append(CodeAssign(var=end_req_pack_offset_var, expr=assign_expr_9))
        
        # distance_req_pack_stream.write(end_req_pack);
        code_lines.append(CodeWriteStream(stream_var=distance_req_pack_stream, in_expr=end_req_pack_var))
        
        # --- 3. Finalize ---
        dist_req_packer_func.codes = code_lines


        self.big_scatter_funcs.append(dist_req_packer_func)
        self.big_top_dataflow_funcs.append(dist_req_packer_func)
        # cacheline_req_sender(
        # hls::stream<distance_req_pack_t> &distance_req_pack_stream,
        # hls::stream<cacheline_request_pkt_t> &cacheline_req_stream)


        cacheline_req_sender_func = HLSFunction(name="cacheline_req_sender", comp=comp)
        params = []

        # --- 1. Define Types & Params ---

        # Param 1: distance_req_pack_stream
        # (Assume distance_req_pack_t_type is cached from previous function)
        distance_req_pack_t_type = HLSType._full_to_type[HLSType._name_to_full["distance_req_pack_t"]]
        distance_req_pack_stream = HLSVar(var_name="distance_req_pack_stream", var_type=HLSType(HLSBasicType.STREAM, sub_types=[
            distance_req_pack_t_type
        ]))

        # Param 2: cacheline_req_stream
        # (Define cacheline_request_pkt_t and its sub-types)

        # Sub-type: ap_uint<NODE_ID_BITWIDTH - LOG_DIST_PER_WORD>
        # (Assume cached from previous function, e.g., as 'cache_idx_elem_type')
        try:
            cache_idx_elem_type = HLSType._full_to_type[HLSType._name_to_full["ap_uint<NODE_ID_BITWIDTH - LOG_DIST_PER_WORD>"]]
        except KeyError:
            cache_idx_elem_type = HLSType(basic_type=HLSBasicType.AP_UINT, width=1)
            cache_idx_elem_type.name = "ap_uint<NODE_ID_BITWIDTH - LOG_DIST_PER_WORD>"

        # Sub-type: ap_uint<4>
        # (Assume cached from previous function, e.g., as 'offset_type')
        try:
            offset_type = HLSType._full_to_type[HLSType._name_to_full["ap_uint<4>"]]
        except KeyError:
            offset_type = HLSType(basic_type=HLSBasicType.AP_UINT, width=4)

        # Sub-type: bool
        end_flag_type = HLSType(basic_type=HLSBasicType.BOOL)


        cacheline_req_stream = HLSVar(var_name="cacheline_req_stream", var_type=HLSType(HLSBasicType.STREAM, sub_types=[
           HLSType(basic_type=HLSBasicType.CACHELINE_REQUEST_PKT_T)
        ]))

        params.extend([distance_req_pack_stream, cacheline_req_stream])
        cacheline_req_sender_func.params = params

        # --- 2. Function Body ---
        code_lines: List[HLSCodeLine] = []

        # cacheline_request_pkt_t cache_req;
        code_lines.append(CodeVarDecl(var_name="cache_req", var_type=HLSType(basic_type=HLSBasicType.CACHELINE_REQUEST_PKT_T)))
        cache_req_var = HLSVar(var_name="cache_req", var_type=HLSType(basic_type=HLSBasicType.CACHELINE_REQUEST_PKT_T))

        # cache_req.last = false;
        cache_req_last_var = HLSVar(var_name="cache_req.last", var_type=end_flag_type)
        const_false_expr = HLSExpr(HLSExprT.CONST, False)
        code_lines.append(CodeAssign(var=cache_req_last_var, expr=const_false_expr))

        # cache_req.data = 0;
        cache_req_data_var = HLSVar(var_name="cache_req.data", var_type=cache_idx_elem_type)
        const_0_expr = HLSExpr(HLSExprT.CONST, 0)
        code_lines.append(CodeAssign(var=cache_req_data_var, expr=const_0_expr))

        # cache_req.dest = 0;
        cache_req_dest_var = HLSVar(var_name="cache_req.dest", var_type=offset_type)
        code_lines.append(CodeAssign(var=cache_req_dest_var, expr=const_0_expr))

        # cacheline_req_stream.write(cache_req);
        code_lines.append(CodeWriteStream(stream_var=cacheline_req_stream, in_expr=cache_req_var))

        # (empty line)
        code_lines.append(CodeOther(text=""))

        # ap_uint<NODE_ID_BITWIDTH - LOG_DIST_PER_WORD> cacheline_idx[PE_NUM];
        cacheline_idx_type = HLSType(HLSBasicType.ARRAY, sub_types=[HLSType(basic_type=HLSBasicType.AP_UINT,width="NODE_ID_BITWIDTH-LOG_DIST_PER_WORD")  ], array_dims=["PE_NUM"])
        code_lines.append(CodeVarDecl(var_name="cacheline_idx", var_type=cacheline_idx_type))

        # #pragma HLS ARRAY_PARTITION variable = cacheline_idx complete dim = 0
        code_lines.append(CodePragma(content="ARRAY_PARTITION variable = cacheline_idx complete dim = 0"))

        # (empty line)
        code_lines.append(CodeOther(text=""))

        # LOOP_SEND_CACHE_REQ:
        code_lines.append(CodeComment(text="LOOP_SEND_CACHE_REQ:"))

        # while (true) {
        while_loop_codes: List[HLSCodeLine] = []
        while_true_expr = HLSExpr(HLSExprT.CONST, True)
        while_loop = CodeWhile(codes=while_loop_codes, iter_expr=while_true_expr)
        code_lines.append(while_loop)

        # --- Inside while(true) ---
        # #pragma HLS PIPELINE II = 1
        while_loop_codes.append(CodePragma(content="PIPELINE II = 1"))

        # #pragma HLS dependence variable = cacheline_idx inter false
        while_loop_codes.append(CodePragma(content="dependence variable = cacheline_idx inter false"))

        # distance_req_pack_t req_pack = distance_req_pack_stream.read();
        while_loop_codes.append(CodeVarDecl(var_name="req_pack", var_type=distance_req_pack_t_type, init_val="distance_req_pack_stream.read()"))

        # #pragma HLS ARRAY_PARTITION variable = req_pack.idx complete dim = 0
        while_loop_codes.append(CodePragma(content="ARRAY_PARTITION variable = req_pack.idx complete dim = 0"))

        # for (int32_t pe_idx = 0; pe_idx < PE_NUM; pe_idx++) {
        for_loop_1_codes: List[HLSCodeLine] = []
        for_loop_1 = CodeFor(codes=for_loop_1_codes,
                             iter_limit="PE_NUM",
                             iter_cmp="<",
                             iter_name="pe_idx",
                             iter_start="0",
                             iter_step="pe_idx++",
                             iter_val_type=HLSType(HLSBasicType.INT))
        while_loop_codes.append(for_loop_1)

        # --- Inside for(pe_idx) ---
        # #pragma HLS UNROLL
        for_loop_1_codes.append(CodePragma(content="UNROLL"))

        # cacheline_idx[pe_idx] = req_pack.idx[pe_idx];
        cacheline_idx_pe_idx_var = HLSVar(var_name="cacheline_idx[pe_idx]", var_type=cache_idx_elem_type)
        assign_expr_1 = HLSExpr(HLSExprT.CONST, "req_pack.idx[pe_idx]")
        for_loop_1_codes.append(CodeAssign(var=cacheline_idx_pe_idx_var, expr=assign_expr_1))
        # } (end for_loop_1)

        # (empty line)
        while_loop_codes.append(CodeOther(text=""))

        # { (CodeBlock starts)
        block_codes: List[HLSCodeLine] = []
        code_block = CodeBlock(codes=block_codes)
        while_loop_codes.append(code_block)

        # --- Inside CodeBlock ---
        # LOOP_SEND_CACHE_REQ_INNER:
        block_codes.append(CodeComment(text="LOOP_SEND_CACHE_REQ_INNER:"))

        # for (ap_uint<4> i = req_pack.offset; i < PE_NUM; i++) {
        for_loop_2_codes: List[HLSCodeLine] = []
        for_loop_2 = CodeFor(codes=for_loop_2_codes,
                             iter_limit="PE_NUM",
                             iter_cmp="<",
                             iter_name="i",
                             iter_start="req_pack.offset",
                             iter_step="i++",
                             iter_val_type=offset_type) 
        block_codes.append(for_loop_2)

        # --- Inside for(i) ---
        # #pragma HLS PIPELINE II = 1 rewind
        for_loop_2_codes.append(CodePragma(content="PIPELINE II = 1 rewind"))

        # #pragma HLS unroll factor = 1
        for_loop_2_codes.append(CodePragma(content="unroll factor = 1"))

        # cache_req.data = cacheline_idx[i];
        assign_expr_2 = HLSExpr(HLSExprT.CONST, "cacheline_idx[i]")
        for_loop_2_codes.append(CodeAssign(var=cache_req_data_var, expr=assign_expr_2))

        # cache_req.dest = i;
        i_var = HLSVar(var_name="i", var_type=offset_type)
        for_loop_2_codes.append(CodeAssign(var=cache_req_dest_var, expr=HLSExpr(HLSExprT.VAR, i_var)))

        # cache_req.last = req_pack.end_flag;
        assign_expr_3 = HLSExpr(HLSExprT.CONST, "req_pack.end_flag")
        for_loop_2_codes.append(CodeAssign(var=cache_req_last_var, expr=assign_expr_3))

        # cacheline_req_stream.write(cache_req);
        for_loop_2_codes.append(CodeWriteStream(stream_var=cacheline_req_stream, in_expr=cache_req_var))

        # // printf("Sent cacheline req for idx %d to PE %d\n",
        for_loop_2_codes.append(CodeComment(text=" printf(\"Sent cacheline req for idx %d to PE %d\\n\","))
        # // (int)cache_req.idx, (int)cache_req.target_pe); fflush(NULL);
        for_loop_2_codes.append(CodeComment(text=" (int)cache_req.idx, (int)cache_req.target_pe); fflush(NULL);"))
        # } (end for_loop_2)
        # } (end code_block)

        # (empty line)
        while_loop_codes.append(CodeOther(text=""))

        # if (req_pack.end_flag) {
        if_1_codes: List[HLSCodeLine] = []
        if_expr_1 = HLSExpr(HLSExprT.CONST, "req_pack.end_flag")
        if_1 = CodeIf(expr=if_expr_1, if_codes=if_1_codes)
        while_loop_codes.append(if_1)

        # --- Inside if(end_flag) ---
        # break;
        if_1_codes.append(CodeBreak())
        # } (end if_1)
        # } (end while_loop)

        # cache_req.last = true;
        const_true_expr = HLSExpr(HLSExprT.CONST, True)
        code_lines.append(CodeAssign(var=cache_req_last_var, expr=const_true_expr))

        # cacheline_req_stream.write(cache_req);
        code_lines.append(CodeWriteStream(stream_var=cacheline_req_stream, in_expr=cache_req_var))

        # --- 3. Finalize ---
        cacheline_req_sender_func.codes = code_lines
        code_lines.append(CodeComment("End of cacheline_req_sender Function"))
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




        
        
    def _translate_fused_op(self, comp: dfir.FusedOpComponent,memory_read_outpattern):
    
        # 分析fused op
        node_id_type = HLSType(HLSBasicType.NODE_ID)
        distance_type = HLSType(HLSBasicType.DISTANCE_T)

        inline_code = []
        sub_graph_inports = comp.sub_graph.inputs
        port_to_var: Dict[dfir.Port, HLSVar] = {}


        sub_graph_components = comp.sub_graph.topo_sort()

        top = 0
        inline_code.append(CodeComment("Begin inline code for fused op: " + comp.name))
        for c in sub_graph_components:
            if isinstance(c,dfir.ConstantComponent):
                constvar = HLSVar(var_name=c.name+"out", var_type=distance_type)
                inline_code.append(CodeVarDecl(var_name=constvar.name, var_type=distance_type, init_val=str(c.value)))

                for port in c.ports:
                    if port.port_type == dfir.PortType.OUT:
                        port_to_var[port] = constvar
            elif isinstance(c,dfir.BinOpComponent):
                lhs_port = c.get_port("i_0")
                rhs_port = c.get_port("i_1")
                if not lhs_port in sub_graph_inports:
                    lhs = port_to_var[lhs_port.connection]
                else:
                    lhs = HLSVar(var_name="top" + str(top), var_type=distance_type)
                    top = top+1
                if not rhs_port in sub_graph_inports:
                    rhs = port_to_var[rhs_port.connection]
                else:
                    rhs = HLSVar(var_name="top" + str(top), var_type=distance_type)
                    top =top +1
                result_var = HLSVar(var_name=c.name+"out", var_type=distance_type)
                binop_expr = HLSExpr(HLSExprT.BINOP, c.op, [HLSExpr(HLSExprT.VAR, lhs), HLSExpr(HLSExprT.VAR, rhs)] )
                inline_code.append(CodeVarDecl(var_name=result_var.name, var_type=distance_type))
                inline_code.append(CodeAssign(var=result_var, expr=binop_expr))
            else:
                pass
        
        inline_code.append(CodeComment("end inline code for fused op: " + comp.name))


        # 还缺少pre reduce的翻译
        # 翻译merge_node_props函数
        merge_node_props_func = HLSFunction(name="merge_node_props", comp=comp)
        params = []

        # --- 1. Define Types & Params ---

        # Basic Types
        node_id_type = HLSType(HLSBasicType.NODE_ID)
        bus_word_t_type = HLSType(HLSBasicType.BUS_WORD_T)
        ap_fixed_pod_t_type = HLSType(HLSBasicType.AP_FIXED_POD)
        distance_t_type = HLSType(HLSBasicType.DISTANCE_T)
        int_type = HLSType(HLSBasicType.INT)
        uint_type = HLSType(HLSBasicType.UINT)
        uint8_type = HLSType(HLSBasicType.UINT8)
        bool_type = HLSType(HLSBasicType.BOOL)

        # Special ap_uint type
        cache_idx_elem_type = HLSType(basic_type=HLSBasicType.AP_UINT, 
                                      width="NODE_ID_BITWIDTH - LOG_DIST_PER_WORD")

        # Param 1: hls::stream<bus_word_t> (&cacheline_streams)[PE_NUM]
        bus_word_stream_type = HLSType(HLSBasicType.STREAM, sub_types=[bus_word_t_type])
        cacheline_streams_type = HLSType(HLSBasicType.ARRAY, sub_types=[bus_word_stream_type], array_dims=["PE_NUM"])
        cacheline_streams = HLSVar(var_name="cacheline_streams", var_type=cacheline_streams_type)

        # Param 2: hls::stream<edge_descriptor_batch_t> &edge_stream
        edge_t_type = HLSType(basic_type=HLSBasicType.STRUCT,
                              struct_name="edge_t",
                              struct_prop_names=["src_id", "dst_id"],
                              sub_types=[node_id_type, node_id_type])
        if edge_t_type.name not in self.struct_definitions:
            self.struct_definitions[edge_t_type.name] = (edge_t_type, edge_t_type.struct_prop_names)

        edge_array_type = HLSType(HLSBasicType.ARRAY, sub_types=[edge_t_type], array_dims=["PE_NUM"])
        edge_descriptor_batch_t_type = HLSType(basic_type=HLSBasicType.STRUCT,
                                               struct_name="edge_descriptor_batch_t",
                                               struct_prop_names=["edges", "end_pos"],
                                               sub_types=[edge_array_type, int_type])
        if edge_descriptor_batch_t_type.name not in self.struct_definitions:
            self.struct_definitions[edge_descriptor_batch_t_type.name] = (edge_descriptor_batch_t_type, edge_descriptor_batch_t_type.struct_prop_names)

        edge_stream_type = HLSType(HLSBasicType.STREAM, sub_types=[edge_descriptor_batch_t_type])
        edge_stream = HLSVar(var_name="edge_stream", var_type=edge_stream_type)

        # Param 3: hls::stream<update_tuple_t> &edge_batch_stream
        node_id_array_pe_type = HLSType(HLSBasicType.ARRAY, sub_types=[node_id_type], array_dims=["PE_NUM"]) # Reused type
        prop_array_type = HLSType(HLSBasicType.ARRAY, sub_types=[ap_fixed_pod_t_type], array_dims=["PE_NUM"])
        update_tuple_t_type = HLSType(basic_type=HLSBasicType.STRUCT,
                                    struct_name="update_tuple_t",
                                    struct_prop_names=["node_id", "prop", "end_flag", "end_pos"],
                                    sub_types=[node_id_array_pe_type, prop_array_type, bool_type, uint8_type])

        if update_tuple_t_type.name not in self.struct_definitions:
            self.struct_definitions[update_tuple_t_type.name] = (update_tuple_t_type, update_tuple_t_type.struct_prop_names)


        edge_batch_stream_type = HLSType(HLSBasicType.STREAM, sub_types=[update_tuple_t_type])
        edge_batch_stream = HLSVar(var_name="edge_batch_stream", var_type=edge_batch_stream_type)

        # Param 4: uint32_t edge_num
        edge_num = HLSVar(var_name="edge_num", var_type=uint_type)

        params.extend([cacheline_streams, edge_stream, edge_batch_stream, edge_num])
        merge_node_props_func.params = params

        # --- 2. Function Body ---
        code_lines: List[HLSCodeLine] = inline_code

        # bus_word_t last_cacheline[PE_NUM];
        last_cacheline_type = HLSType(HLSBasicType.ARRAY, sub_types=[bus_word_t_type], array_dims=["PE_NUM"])
        code_lines.append(CodeVarDecl(var_name="last_cacheline", var_type=last_cacheline_type))
        last_cacheline_var = HLSVar(var_name="last_cacheline", var_type=last_cacheline_type)

        # #pragma HLS ARRAY_PARTITION variable = last_cacheline complete dim = 0
        code_lines.append(CodePragma(content="ARRAY_PARTITION variable = last_cacheline complete dim = 0"))

        # ap_uint<...> last_cache_idx[PE_NUM];
        last_cache_idx_type = HLSType(HLSBasicType.ARRAY, sub_types=[cache_idx_elem_type], array_dims=["PE_NUM"])
        code_lines.append(CodeVarDecl(var_name="last_cache_idx", var_type=last_cache_idx_type))
        last_cache_idx_var = HLSVar(var_name="last_cache_idx", var_type=last_cache_idx_type)

        # #pragma HLS ARRAY_PARTITION variable = last_cache_idx complete dim = 0
        code_lines.append(CodePragma(content="ARRAY_PARTITION variable = last_cache_idx complete dim = 0"))

        # --- Build for(pe_idx) 1 ---
        for_loop_1_codes: List[HLSCodeLine] = []
        # #pragma HLS UNROLL
        for_loop_1_codes.append(CodePragma(content="UNROLL"))
        # last_cacheline[pe_idx] = cacheline_streams[pe_idx].read();
        last_cacheline_pe_idx_var = HLSVar(var_name="last_cacheline[pe_idx]", var_type=bus_word_t_type)
        assign_expr_1 = HLSExpr(HLSExprT.CONST, "cacheline_streams[pe_idx].read()")
        for_loop_1_codes.append(CodeAssign(var=last_cacheline_pe_idx_var, expr=assign_expr_1))
        # last_cache_idx[pe_idx] = 0;
        last_cache_idx_pe_idx_var = HLSVar(var_name="last_cache_idx[pe_idx]", var_type=cache_idx_elem_type)
        assign_expr_2 = HLSExpr(HLSExprT.CONST, 0)
        for_loop_1_codes.append(CodeAssign(var=last_cache_idx_pe_idx_var, expr=assign_expr_2))
        # Create for loop 1
        for_loop_1 = CodeFor(codes=for_loop_1_codes,
                             iter_limit="PE_NUM",
                             iter_cmp="<",
                             iter_name="pe_idx",
                             iter_start="0",
                             iter_step="pe_idx++",
                             iter_val_type=int_type)
        code_lines.append(for_loop_1)
        # --- End for(pe_idx) 1 ---

        # const uint32_t scatter_size = (edge_num + PE_NUM - 1) / PE_NUM;
        code_lines.append(CodeVarDecl(var_name="scatter_size", var_type=uint_type, init_val="(edge_num + PE_NUM - 1) / PE_NUM", const=True))
        scatter_size_var = HLSVar(var_name="scatter_size", var_type=uint_type)

        # distance_t real_edge_weight = 1.0; 
        code_lines.append(CodeVarDecl(var_name="real_edge_weight", var_type=distance_t_type, init_val="1.0", const=False))

        # const ap_fixed_pod_t edge_weight = (*reinterpret_cast<...>(&real_edge_weight));
        edge_weight_init_val = "(*reinterpret_cast<ap_fixed_pod_t *>(&real_edge_weight))"
        code_lines.append(CodeVarDecl(var_name="edge_weight", var_type=ap_fixed_pod_t_type, init_val=edge_weight_init_val, const=True))
        edge_weight_var = HLSVar(var_name="edge_weight", var_type=ap_fixed_pod_t_type)

        # --- Build for(edge_batch_idx) ---
        for_loop_2_codes: List[HLSCodeLine] = []
        # #pragma HLS PIPELINE II = 1
        for_loop_2_codes.append(CodePragma(content="PIPELINE II = 1"))

        # edge_descriptor_batch_t edge_batch;
        for_loop_2_codes.append(CodeVarDecl(var_name="edge_batch", var_type=edge_descriptor_batch_t_type))
        edge_batch_var = HLSVar(var_name="edge_batch", var_type=edge_descriptor_batch_t_type)

        # #pragma HLS ARRAY_PARTITION variable = edge_batch.edges complete dim = 0
        for_loop_2_codes.append(CodePragma(content="ARRAY_PARTITION variable = edge_batch.edges complete dim = 0"))

        # edge_batch = edge_stream.read();
        assign_expr_3 = HLSExpr(HLSExprT.CONST, "edge_stream.read()")
        for_loop_2_codes.append(CodeAssign(var=edge_batch_var, expr=assign_expr_3))

        # update_tuple_t out_batch;
        for_loop_2_codes.append(CodeVarDecl(var_name="out_batch", var_type=update_tuple_t_type))
        out_batch_var = HLSVar(var_name="out_batch", var_type=update_tuple_t_type)

        # #pragma HLS ARRAY_PARTITION variable = out_batch.node_id complete dim = 0
        for_loop_2_codes.append(CodePragma(content="ARRAY_PARTITION variable = out_batch.node_id complete dim = 0"))

        # #pragma HLS ARRAY_PARTITION variable = out_batch.prop complete dim = 0
        for_loop_2_codes.append(CodePragma(content="ARRAY_PARTITION variable = out_batch.prop complete dim = 0"))

        # out_batch.end_flag = false;
        out_batch_end_flag_var = HLSVar(var_name="out_batch.end_flag", var_type=bool_type)
        assign_expr_4 = HLSExpr(HLSExprT.CONST, False)
        for_loop_2_codes.append(CodeAssign(var=out_batch_end_flag_var, expr=assign_expr_4))

        # out_batch.end_pos = edge_batch.end_pos;
        out_batch_end_pos_var = HLSVar(var_name="out_batch.end_pos", var_type=uint8_type)
        assign_expr_5 = HLSExpr(HLSExprT.CONST, "edge_batch.end_pos")
        for_loop_2_codes.append(CodeAssign(var=out_batch_end_pos_var, expr=assign_expr_5))

        # bus_word_t cur_last_cacheline;
        for_loop_2_codes.append(CodeVarDecl(var_name="cur_last_cacheline", var_type=bus_word_t_type))
        cur_last_cacheline_var = HLSVar(var_name="cur_last_cacheline", var_type=bus_word_t_type)

        # ap_uint<...> cur_last_cache_idx;
        for_loop_2_codes.append(CodeVarDecl(var_name="cur_last_cache_idx", var_type=cache_idx_elem_type))
        cur_last_cache_idx_var = HLSVar(var_name="cur_last_cache_idx", var_type=cache_idx_elem_type)

        # --- Build for(pe_idx) 2 ---
        for_loop_3_codes: List[HLSCodeLine] = []
        # #pragma HLS UNROLL
        for_loop_3_codes.append(CodePragma(content="UNROLL"))

        # ap_uint<...> cacheline_idx = ...
        cacheline_idx_init_val = "(edge_batch.edges[pe_idx].src_id >> LOG_DIST_PER_WORD)"
        for_loop_3_codes.append(CodeVarDecl(var_name="cacheline_idx", var_type=cache_idx_elem_type, init_val=cacheline_idx_init_val))

        # uint32_t offset = ...
        offset_init_val = "(edge_batch.edges[pe_idx].src_id & (DIST_PER_WORD - 1))"
        for_loop_3_codes.append(CodeVarDecl(var_name="offset", var_type=uint_type, init_val=offset_init_val))

        # --- Build IF_1 (pe_idx < edge_batch.end_pos) ---
        if_1_codes: List[HLSCodeLine] = []
        if_expr_1 = HLSExpr(HLSExprT.CONST, "pe_idx < edge_batch.end_pos")
        # (This IF_1 has no else block)

        # --- Build IF_1 Contents ---
        cacheline_var = HLSVar(var_name="cacheline", var_type=bus_word_t_type)
        if_1_codes.append(CodeVarDecl(var_name="cacheline", var_type=cacheline_var.type))

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
        if_1_codes.append(if_2)
        # --- End IF_2 ---

        # ap_fixed_pod_t prop = get_val_from_bus(cacheline, offset);
        if_1_codes.append(CodeVarDecl(var_name="prop", var_type=ap_fixed_pod_t_type, init_val="get_val_from_bus(cacheline, offset)"))

        # out_batch.node_id[pe_idx] = edge_batch.edges[pe_idx].dst_id;

    
        
        assign_expr_8 = HLSExpr(HLSExprT.CONST, "edge_batch.edges[pe_idx].dst_id")


        out_batch_node_id_pe_idx_var = HLSVar(var_name="out_batch.node_id[pe_idx]", var_type=node_id_type)        
        if_1_codes.append(CodeAssign(var=out_batch_node_id_pe_idx_var, expr=assign_expr_8))

        # out_batch.prop[pe_idx] = (prop + edge_weight);
        out_batch_prop_pe_idx_var = HLSVar(var_name="out_batch.prop[pe_idx]", var_type=ap_fixed_pod_t_type)
        assign_expr_9 = HLSExpr(HLSExprT.CONST, "(prop + edge_weight)")
    
        
        
        
        
        if_1_codes.append(CodeAssign(var=out_batch_prop_pe_idx_var, expr=assign_expr_9))

        # --- Build IF_3 (pe_idx == PE_NUM - 1) ---
        if_3_codes: List[HLSCodeLine] = []
        if_expr_3 = HLSExpr(HLSExprT.CONST, "pe_idx == PE_NUM - 1")
        # (This IF_3 has no else block)
        # Build IF_3 Contents
        if_3_codes.append(CodeAssign(var=cur_last_cacheline_var, expr=HLSExpr(HLSExprT.VAR, cacheline_var)))
        assign_expr_10 = HLSExpr(HLSExprT.CONST, "cacheline_idx")
        if_3_codes.append(CodeAssign(var=cur_last_cache_idx_var, expr=assign_expr_10))
        # Create IF_3
        if_3 = CodeIf(expr=if_expr_3, if_codes=if_3_codes)
        if_1_codes.append(if_3)
        # --- End IF_3 ---

        # Create IF_1
        if_1 = CodeIf(expr=if_expr_1, if_codes=if_1_codes)
        for_loop_3_codes.append(if_1)
        # --- End IF_1 ---

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

        # --- Build for(pe_idx) 3 ---
        for_loop_4_codes: List[HLSCodeLine] = []
        # #pragma HLS UNROLL
        for_loop_4_codes.append(CodePragma(content="UNROLL"))
        # last_cacheline[pe_idx] = cur_last_cacheline;
        last_cacheline_pe_idx_var_2 = HLSVar(var_name="last_cacheline[pe_idx]", var_type=bus_word_t_type)
        for_loop_4_codes.append(CodeAssign(var=last_cacheline_pe_idx_var_2, expr=cur_last_cacheline_var))
        # last_cache_idx[pe_idx] = cur_last_cache_idx;
        last_cache_idx_pe_idx_var_2 = HLSVar(var_name="last_cache_idx[pe_idx]", var_type=cache_idx_elem_type)
        for_loop_4_codes.append(CodeAssign(var=last_cache_idx_pe_idx_var_2, expr=cur_last_cache_idx_var))
        # Create for loop 4
        for_loop_4 = CodeFor(codes=for_loop_4_codes,
                             iter_limit="PE_NUM",
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

        # update_tuple_t end_batch;
        code_lines.append(CodeVarDecl(var_name="end_batch", var_type=update_tuple_t_type))
        end_batch_var = HLSVar(var_name="end_batch", var_type=update_tuple_t_type)

        # end_batch.end_flag = true;
        end_batch_end_flag_var = HLSVar(var_name="end_batch.end_flag", var_type=bool_type)
        assign_expr_11 = HLSExpr(HLSExprT.CONST, True)
        code_lines.append(CodeAssign(var=end_batch_end_flag_var, expr=assign_expr_11))

        # end_batch.end_pos = 0;
        end_batch_end_pos_var = HLSVar(var_name="end_batch.end_pos", var_type=uint8_type)
        assign_expr_12 = HLSExpr(HLSExprT.CONST, 0)
        code_lines.append(CodeAssign(var=end_batch_end_pos_var, expr=assign_expr_12))

        # edge_batch_stream.write(end_batch);
        code_lines.append(CodeWriteStream(stream_var=edge_batch_stream, in_expr=end_batch_var))

        # --- 3. Finalize ---
        merge_node_props_func.codes = code_lines

        self.big_scatter_funcs.append(merge_node_props_func)
        self.big_top_dataflow_funcs.append(merge_node_props_func)
        # TODO 识别计算逻辑




        ## 辅助函数：



    def _translate_reduce_op(self, comp: dfir.Component):
        
        #     static void graphyflow_big_dataflow(
        # hls::stream<update_tuple_t> &input_to_demux,
        # hls::stream<write_burst_pkt_t> &kernel_out_stream, int32_t dst_num) 

        demux_1_func = HLSFunction(name="demux_1", comp=comp)
        params = []

        # --- 1. Define Types & Params ---

        # Basic Types
        node_id_type = HLSType(HLSBasicType.NODE_ID)
        ap_fixed_pod_t_type = HLSType(HLSBasicType.AP_FIXED_POD)
        bool_type = HLSType(HLSBasicType.BOOL)
        uint8_type = HLSType(HLSBasicType.UINT8)
        uint_type = HLSType(HLSBasicType.UINT)

        # Param 1 Type: hls::stream<update_tuple_t>
        # (based on user's definition)
        node_id_array_type = HLSType(HLSBasicType.ARRAY, sub_types=[node_id_type], array_dims=["PE_NUM"])
        prop_array_type = HLSType(HLSBasicType.ARRAY, sub_types=[ap_fixed_pod_t_type], array_dims=["PE_NUM"])
        update_tuple_t_type = HLSType(basic_type=HLSBasicType.STRUCT,
                                    struct_name="update_tuple_t",
                                    struct_prop_names=["node_id", "prop", "end_flag", "end_pos"],
                                    sub_types=[node_id_array_type, prop_array_type, bool_type, uint8_type])
        
        if update_tuple_t_type.name not in self.struct_definitions:
            self.struct_definitions[update_tuple_t_type.name] = (update_tuple_t_type, update_tuple_t_type.struct_prop_names)

        if update_tuple_t_type.name not in self.struct_definitions:
            self.struct_definitions[update_tuple_t_type.name] = (update_tuple_t_type, update_tuple_t_type.struct_prop_names)
        in_batch_stream_type = HLSType(HLSBasicType.STREAM, sub_types=[update_tuple_t_type])
        in_batch_stream = HLSVar(var_name="in_batch_stream", var_type=in_batch_stream_type)

        # Param 2 Type: hls::stream<net_wrapper_kt_pair_105_t_t> (&out_streams)[8]
        # (based on user's definition)
        net_wrapper_kt_pair_105_t_t_type = HLSType(basic_type=HLSBasicType.STRUCT,
                                                  struct_name="net_wrapper_kt_pair_105_t_t",
                                                  struct_prop_names=["node_id", "prop", "end_flag"],
                                                  sub_types=[node_id_type, ap_fixed_pod_t_type, bool_type])
        if net_wrapper_kt_pair_105_t_t_type.name not in self.struct_definitions:
            self.struct_definitions[net_wrapper_kt_pair_105_t_t_type.name] = (net_wrapper_kt_pair_105_t_t_type, net_wrapper_kt_pair_105_t_t_type.struct_prop_names)
        out_stream_element_type = HLSType(HLSBasicType.STREAM, sub_types=[net_wrapper_kt_pair_105_t_t_type])
        out_streams_type = HLSType(HLSBasicType.ARRAY, sub_types=[out_stream_element_type], array_dims=[8])
        out_streams = HLSVar(var_name="out_streams", var_type=out_streams_type)

        params.extend([in_batch_stream, out_streams])
        demux_1_func.params = params

        # --- 2. Function Body ---
        code_lines: List[HLSCodeLine] = []

        # update_tuple_t in_batch;
        code_lines.append(CodeVarDecl(var_name="in_batch", var_type=update_tuple_t_type))
        in_batch_var = HLSVar(var_name="in_batch", var_type=update_tuple_t_type)

        # #pragma HLS ARRAY_PARTITION variable = in_batch.node_id complete dim = 0
        code_lines.append(CodePragma(content="ARRAY_PARTITION variable = in_batch.node_id complete dim = 0"))

        # #pragma HLS ARRAY_PARTITION variable = in_batch.prop complete dim = 0
        code_lines.append(CodePragma(content="ARRAY_PARTITION variable = in_batch.prop complete dim = 0"))

        # while (true) {
        while_loop_codes: List[HLSCodeLine] = []
        while_true_expr = HLSExpr(HLSExprT.CONST, True)
        while_loop = CodeWhile(codes=while_loop_codes, iter_expr=while_true_expr)
        code_lines.append(while_loop)

        # --- Inside while(true) ---
        # #pragma HLS PIPELINE
        while_loop_codes.append(CodePragma(content="PIPELINE"))

        # in_batch = in_batch_stream.read();
        assign_expr_1 = HLSExpr(HLSExprT.CONST, "in_batch_stream.read()")
        while_loop_codes.append(CodeAssign(var=in_batch_var, expr=assign_expr_1))

        # net_wrapper_kt_pair_105_t_t wrapper_data;
        while_loop_codes.append(CodeVarDecl(var_name="wrapper_data", var_type=net_wrapper_kt_pair_105_t_t_type))
        wrapper_data_var = HLSVar(var_name="wrapper_data", var_type=net_wrapper_kt_pair_105_t_t_type)

        # #pragma HLS ARRAY_PARTITION variable = wrapper_data.node_id complete dim = 0
        while_loop_codes.append(CodePragma(content="ARRAY_PARTITION variable = wrapper_data.node_id complete dim = 0"))

        # #pragma HLS ARRAY_PARTITION variable = wrapper_data.prop complete dim = 0
        while_loop_codes.append(CodePragma(content="ARRAY_PARTITION variable = wrapper_data.prop complete dim = 0"))

        # for (uint32_t i = 0; i < PE_NUM; i++) {
        for_loop_1_codes: List[HLSCodeLine] = []
        for_loop_1 = CodeFor(codes=for_loop_1_codes,
                             iter_limit="PE_NUM",
                             iter_cmp="<",
                             iter_name="i",
                             iter_start="0",
                             iter_step="i++",
                             iter_val_type=uint_type)
        while_loop_codes.append(for_loop_1)

        # --- Inside for(i) ---
        # #pragma HLS UNROLL
        for_loop_1_codes.append(CodePragma(content="UNROLL"))

        # if ((i < in_batch.end_pos)) {
        if_1_codes: List[HLSCodeLine] = []
        if_expr_1 = HLSExpr(HLSExprT.CONST, "(i < in_batch.end_pos)")
        if_1 = CodeIf(expr=if_expr_1, if_codes=if_1_codes)
        for_loop_1_codes.append(if_1)

        # --- Inside if(i < end_pos) ---
        # wrapper_data.node_id = in_batch.node_id[i];
        wrapper_data_node_id_var = HLSVar(var_name="wrapper_data.node_id", var_type=node_id_type)
        assign_expr_2 = HLSExpr(HLSExprT.CONST, "in_batch.node_id[i]")
        if_1_codes.append(CodeAssign(var=wrapper_data_node_id_var, expr=assign_expr_2))

        # wrapper_data.prop = in_batch.prop[i];
        wrapper_data_prop_var = HLSVar(var_name="wrapper_data.prop", var_type=ap_fixed_pod_t_type)
        assign_expr_3 = HLSExpr(HLSExprT.CONST, "in_batch.prop[i]")
        if_1_codes.append(CodeAssign(var=wrapper_data_prop_var, expr=assign_expr_3))

        # wrapper_data.end_flag = false;
        wrapper_data_end_flag_var = HLSVar(var_name="wrapper_data.end_flag", var_type=bool_type)
        assign_expr_4 = HLSExpr(HLSExprT.CONST, False)
        if_1_codes.append(CodeAssign(var=wrapper_data_end_flag_var, expr=assign_expr_4))

        # out_streams[i].write(wrapper_data);
        out_stream_i_var = HLSVar(var_name="out_streams[i]", var_type=out_stream_element_type)
        if_1_codes.append(CodeWriteStream(stream_var=out_stream_i_var, in_expr=wrapper_data_var))
        # } (end if_1)
        # } (end for_loop_1)

        # if (in_batch.end_flag) {
        if_2_codes: List[HLSCodeLine] = []
        if_expr_2 = HLSExpr(HLSExprT.CONST, "in_batch.end_flag")
        if_2 = CodeIf(expr=if_expr_2, if_codes=if_2_codes)
        while_loop_codes.append(if_2)

        # --- Inside if(in_batch.end_flag) ---
        # break;
        if_2_codes.append(CodeBreak())
        # } (end if_2)
        # } (end while_loop)

        # net_wrapper_kt_pair_105_t_t end_wrapper;
        code_lines.append(CodeVarDecl(var_name="end_wrapper", var_type=net_wrapper_kt_pair_105_t_t_type))
        end_wrapper_var = HLSVar(var_name="end_wrapper", var_type=net_wrapper_kt_pair_105_t_t_type)

        # end_wrapper.end_flag = true;
        end_wrapper_end_flag_var = HLSVar(var_name="end_wrapper.end_flag", var_type=bool_type)
        assign_expr_5 = HLSExpr(HLSExprT.CONST, True)
        code_lines.append(CodeAssign(var=end_wrapper_end_flag_var, expr=assign_expr_5))

        # for (uint32_t i = 0; i < 8; i++) {
        for_loop_2_codes: List[HLSCodeLine] = []
        for_loop_2 = CodeFor(codes=for_loop_2_codes,
                             iter_limit="8",
                             iter_cmp="<",
                             iter_name="i",
                             iter_start="0",
                             iter_step="i++",
                             iter_val_type=uint_type)
        code_lines.append(for_loop_2)

        # --- Inside for(i) 2 ---
        # #pragma HLS UNROLL
        for_loop_2_codes.append(CodePragma(content="UNROLL"))

        # out_streams[i].write(end_wrapper);
        out_stream_i_var_2 = HLSVar(var_name="out_streams[i]", var_type=out_stream_element_type)
        for_loop_2_codes.append(CodeWriteStream(stream_var=out_stream_i_var_2, in_expr=end_wrapper_var))
        # } (end for_loop_2)

        # --- 3. Finalize ---
        demux_1_func.codes = code_lines
        self.big_gather_funcs.append(demux_1_func)




        sender_2_func = HLSFunction(name="sender_2", comp=comp)
        params = []

        # --- 1. Define Types & Params ---

        # Basic Types
        int_type = HLSType(HLSBasicType.INT)
        bool_type = HLSType(HLSBasicType.BOOL)
        node_id_type = HLSType(HLSBasicType.NODE_ID)
        ap_fixed_pod_t_type = HLSType(HLSBasicType.AP_FIXED_POD)

        # Param 1: i
        i_var = HLSVar(var_name="i", var_type=int_type)

        # Param 2-7 Types: hls::stream<net_wrapper_kt_pair_105_t_t>
        net_wrapper_kt_pair_105_t_t_type = HLSType(basic_type=HLSBasicType.STRUCT,
                                                  struct_name="net_wrapper_kt_pair_105_t_t",
                                                  struct_prop_names=["node_id", "prop", "end_flag"],
                                                  sub_types=[node_id_type, ap_fixed_pod_t_type, bool_type])
        if net_wrapper_kt_pair_105_t_t_type.name not in self.struct_definitions:
            self.struct_definitions[net_wrapper_kt_pair_105_t_t_type.name] = (net_wrapper_kt_pair_105_t_t_type, net_wrapper_kt_pair_105_t_t_type.struct_prop_names)

        stream_type = HLSType(HLSBasicType.STREAM, sub_types=[net_wrapper_kt_pair_105_t_t_type])

        in1_var = HLSVar(var_name="in1", var_type=stream_type)
        in2_var = HLSVar(var_name="in2", var_type=stream_type)
        out1_var = HLSVar(var_name="out1", var_type=stream_type)
        out2_var = HLSVar(var_name="out2", var_type=stream_type)
        out3_var = HLSVar(var_name="out3", var_type=stream_type)
        out4_var = HLSVar(var_name="out4", var_type=stream_type)

        params.extend([i_var, in1_var, in2_var, out1_var, out2_var, out3_var, out4_var])
        sender_2_func.params = params

        # --- 2. Function Body ---
        code_lines: List[HLSCodeLine] = []

        # #pragma HLS function_instantiate variable = i
        code_lines.append(CodePragma(content="function_instantiate variable = i"))

        # bool in1_end_flag = false;
        code_lines.append(CodeVarDecl(var_name="in1_end_flag", var_type=bool_type, init_val="false"))
        in1_end_flag_var = HLSVar(var_name="in1_end_flag", var_type=bool_type)

        # bool in2_end_flag = false;
        code_lines.append(CodeVarDecl(var_name="in2_end_flag", var_type=bool_type, init_val="false"))
        in2_end_flag_var = HLSVar(var_name="in2_end_flag", var_type=bool_type)

        # while (true) {
        while_loop_codes: List[HLSCodeLine] = []
        while_true_expr = HLSExpr(HLSExprT.CONST, True)
        # (We append the while_loop at the end, after filling while_loop_codes)

        # --- Inside while(true) ---
        # #pragma HLS PIPELINE II = 1
        while_loop_codes.append(CodePragma(content="PIPELINE II = 1"))

        # --- Build IF_1 Block ---
        if_1_codes: List[HLSCodeLine] = []
        if_1_expr = HLSExpr(HLSExprT.UOP, dfir.UnaryOp.NOT, operands=[
            HLSExpr(HLSExprT.STREAM_EMPTY, expr_val=None, operands=[HLSExpr(HLSExprT.VAR, in1_var)])
        ])
        # (This IF_1 has no else block)

        # --- Build IF_1 Contents ---
        data1_var = HLSVar(var_name="data1", var_type=net_wrapper_kt_pair_105_t_t_type)
        if_1_codes.append(CodeVarDecl(var_name="data1", var_type=net_wrapper_kt_pair_105_t_t_type))
        assign_expr_1 = HLSExpr(HLSExprT.STREAM_READ, expr_val=None, operands=[HLSExpr(HLSExprT.VAR, in1_var)])
        if_1_codes.append(CodeAssign(var=data1_var, expr=assign_expr_1))

        # --- Build IF_2 (nested) Block ---
        if_2_codes: List[HLSCodeLine] = []
        else_2_codes: List[HLSCodeLine] = []
        if_2_expr = HLSExpr(HLSExprT.CONST, "(!data1.end_flag)")

        # --- Build IF_2 Contents ---
        # --- Build IF_3 (nested) Block ---
        if_3_codes: List[HLSCodeLine] = []
        else_3_codes: List[HLSCodeLine] = []
        if_3_expr = HLSExpr(HLSExprT.CONST, "((data1.node_id >> i) & 1)")
        # Build IF_3 Contents
        if_3_codes.append(CodeWriteStream(stream_var=out2_var, in_expr=data1_var))
        # Build ELSE_3 Contents
        else_3_codes.append(CodeWriteStream(stream_var=out1_var, in_expr=data1_var))
        # Create IF_3
        if_3 = CodeIf(expr=if_3_expr, if_codes=if_3_codes, else_codes=else_3_codes)
        if_2_codes.append(if_3)
        # --- End IF_3 ---

        # --- Build ELSE_2 Contents ---
        assign_expr_2 = HLSExpr(HLSExprT.CONST, True)
        else_2_codes.append(CodeAssign(var=in1_end_flag_var, expr=assign_expr_2))

        # Create IF_2
        if_2 = CodeIf(expr=if_2_expr, if_codes=if_2_codes, else_codes=else_2_codes)
        if_1_codes.append(if_2)
        # --- End IF_2 ---

        # Create IF_1
        if_1 = CodeIf(expr=if_1_expr, if_codes=if_1_codes) # else_codes is default None
        while_loop_codes.append(if_1)
        # --- End IF_1 ---


        # --- Build IF_4 Block ---
        if_4_codes: List[HLSCodeLine] = []
        if_4_expr = HLSExpr(HLSExprT.UOP, dfir.UnaryOp.NOT, operands=[
            HLSExpr(HLSExprT.STREAM_EMPTY, expr_val=None, operands=[HLSExpr(HLSExprT.VAR, in2_var)])
        ])
        # (This IF_4 has no else block)

        # --- Build IF_4 Contents ---
        data2_var = HLSVar(var_name="data2", var_type=net_wrapper_kt_pair_105_t_t_type)
        if_4_codes.append(CodeVarDecl(var_name="data2", var_type=net_wrapper_kt_pair_105_t_t_type))
        assign_expr_3 = HLSExpr(HLSExprT.STREAM_READ, expr_val=None, operands=[HLSExpr(HLSExprT.VAR, in2_var)])
        if_4_codes.append(CodeAssign(var=data2_var, expr=assign_expr_3))

        # --- Build IF_5 (nested) Block ---
        if_5_codes: List[HLSCodeLine] = []
        else_5_codes: List[HLSCodeLine] = []
        if_5_expr = HLSExpr(HLSExprT.CONST, "(!data2.end_flag)")

        # --- Build IF_5 Contents ---
        # --- Build IF_6 (nested) Block ---
        if_6_codes: List[HLSCodeLine] = []
        else_6_codes: List[HLSCodeLine] = []
        if_6_expr = HLSExpr(HLSExprT.CONST, "((data2.node_id >> i) & 1)")
        # Build IF_6 Contents
        if_6_codes.append(CodeWriteStream(stream_var=out4_var, in_expr=data2_var))
        # Build ELSE_6 Contents
        else_6_codes.append(CodeWriteStream(stream_var=out3_var, in_expr=data2_var))
        # Create IF_6
        if_6 = CodeIf(expr=if_6_expr, if_codes=if_6_codes, else_codes=else_6_codes)
        if_5_codes.append(if_6)
        # --- End IF_6 ---

        # --- Build ELSE_5 Contents ---
        assign_expr_4 = HLSExpr(HLSExprT.CONST, True)
        else_5_codes.append(CodeAssign(var=in2_end_flag_var, expr=assign_expr_4))

        # Create IF_5
        if_5 = CodeIf(expr=if_5_expr, if_codes=if_5_codes, else_codes=else_5_codes)
        if_4_codes.append(if_5)
        # --- End IF_5 ---

        # Create IF_4
        if_4 = CodeIf(expr=if_4_expr, if_codes=if_4_codes) # else_codes is default None
        while_loop_codes.append(if_4)
        # --- End IF_4 ---


        # --- Build IF_7 Block ---
        if_7_codes: List[HLSCodeLine] = []
        if_7_expr = HLSExpr(HLSExprT.CONST, "(in1_end_flag & in2_end_flag)")
        # (This IF_7 has no else block)

        # --- Build IF_7 Contents ---
        data_var = HLSVar(var_name="data", var_type=net_wrapper_kt_pair_105_t_t_type)
        if_7_codes.append(CodeVarDecl(var_name="data", var_type=data_var.type))
        data_end_flag_var = HLSVar(var_name="data.end_flag", var_type=bool_type)
        assign_expr_5 = HLSExpr(HLSExprT.CONST, True)
        if_7_codes.append(CodeAssign(var=data_end_flag_var, expr=assign_expr_5))

        if_7_codes.append(CodeWriteStream(stream_var=out1_var, in_expr=data_var))
        if_7_codes.append(CodeWriteStream(stream_var=out2_var, in_expr=data_var))
        if_7_codes.append(CodeWriteStream(stream_var=out3_var, in_expr=data_var))
        if_7_codes.append(CodeWriteStream(stream_var=out4_var, in_expr=data_var))

        assign_expr_6 = HLSExpr(HLSExprT.CONST, False)
        if_7_codes.append(CodeAssign(var=in1_end_flag_var, expr=assign_expr_6))
        assign_expr_7 = HLSExpr(HLSExprT.CONST, False)
        if_7_codes.append(CodeAssign(var=in2_end_flag_var, expr=assign_expr_7))

        if_7_codes.append(CodeBreak())

        # Create IF_7
        if_7 = CodeIf(expr=if_7_expr, if_codes=if_7_codes) # else_codes is default None
        while_loop_codes.append(if_7)
        # --- End IF_7 ---

        # --- Create While Loop (now that contents are ready) ---
        while_loop = CodeWhile(codes=while_loop_codes, iter_expr=while_true_expr)
        code_lines.append(while_loop)

        # --- 3. Finalize ---
        sender_2_func.codes = code_lines
  
        self.big_gather_funcs.append(sender_2_func)

        receiver_2_func = HLSFunction(name="receiver_2", comp=comp)
        params = []
        
        # --- 1. Define Types & Params ---
        
        # Basic Types
        int_type = HLSType(HLSBasicType.INT)
        bool_type = HLSType(HLSBasicType.BOOL)
        node_id_type = HLSType(HLSBasicType.NODE_ID)
        ap_fixed_pod_t_type = HLSType(HLSBasicType.AP_FIXED_POD)
        
        # Param 1: i
        i_var = HLSVar(var_name="i", var_type=int_type)
        
        # Param 2-7 Types: hls::stream<net_wrapper_kt_pair_105_t_t>
        net_wrapper_kt_pair_105_t_t_type = HLSType(basic_type=HLSBasicType.STRUCT,
                                                  struct_name="net_wrapper_kt_pair_105_t_t",
                                                  struct_prop_names=["node_id", "prop", "end_flag"],
                                                  sub_types=[node_id_type, ap_fixed_pod_t_type, bool_type])
        if net_wrapper_kt_pair_105_t_t_type.name not in self.struct_definitions:
            self.struct_definitions[net_wrapper_kt_pair_105_t_t_type.name] = (net_wrapper_kt_pair_105_t_t_type, net_wrapper_kt_pair_105_t_t_type.struct_prop_names)

        stream_type = HLSType(HLSBasicType.STREAM, sub_types=[net_wrapper_kt_pair_105_t_t_type])
        
        out1_var = HLSVar(var_name="out1", var_type=stream_type)
        out2_var = HLSVar(var_name="out2", var_type=stream_type)
        in1_var = HLSVar(var_name="in1", var_type=stream_type)
        in2_var = HLSVar(var_name="in2", var_type=stream_type)
        in3_var = HLSVar(var_name="in3", var_type=stream_type)
        in4_var = HLSVar(var_name="in4", var_type=stream_type)
        
        params.extend([i_var, out1_var, out2_var, in1_var, in2_var, in3_var, in4_var])
        receiver_2_func.params = params
        
        # --- 2. Function Body ---
        code_lines: List[HLSCodeLine] = []
        
        # #pragma HLS function_instantiate variable = i
        code_lines.append(CodePragma(content="function_instantiate variable = i"))
        
        # bool in1_end_flag = false;
        code_lines.append(CodeVarDecl(var_name="in1_end_flag", var_type=bool_type, init_val="false"))
        in1_end_flag_var = HLSVar(var_name="in1_end_flag", var_type=bool_type)
        
        # bool in2_end_flag = false;
        code_lines.append(CodeVarDecl(var_name="in2_end_flag", var_type=bool_type, init_val="false"))
        in2_end_flag_var = HLSVar(var_name="in2_end_flag", var_type=bool_type)
        
        # bool in3_end_flag = false;
        code_lines.append(CodeVarDecl(var_name="in3_end_flag", var_type=bool_type, init_val="false"))
        in3_end_flag_var = HLSVar(var_name="in3_end_flag", var_type=bool_type)
        
        # bool in4_end_flag = false;
        code_lines.append(CodeVarDecl(var_name="in4_end_flag", var_type=bool_type, init_val="false"))
        in4_end_flag_var = HLSVar(var_name="in4_end_flag", var_type=bool_type)
        
        # while (true) {
        while_loop_codes: List[HLSCodeLine] = []
        while_true_expr = HLSExpr(HLSExprT.CONST, True)
        # (while_loop object created and added to code_lines at the end)
        
        # --- Inside while(true) ---
        # #pragma HLS PIPELINE II = 1
        while_loop_codes.append(CodePragma(content="PIPELINE II = 1"))
        
        # --- Build IF_1 (!in1.empty) ---
        if_1_codes: List[HLSCodeLine] = []
        if_1_expr = HLSExpr(HLSExprT.UOP, dfir.UnaryOp.NOT, operands=[
            HLSExpr(HLSExprT.STREAM_EMPTY, expr_val=None, operands=[HLSExpr(HLSExprT.VAR, in1_var)])
        ])
        # Build nested IF_2 (!data.end_flag)
        if_2_codes: List[HLSCodeLine] = []
        else_2_codes: List[HLSCodeLine] = []
        data_var_1 = HLSVar(var_name="data", var_type=net_wrapper_kt_pair_105_t_t_type)
        if_1_codes.append(CodeVarDecl(var_name="data", var_type=data_var_1.type))
        assign_expr_1 = HLSExpr(HLSExprT.STREAM_READ, expr_val=None, operands=[HLSExpr(HLSExprT.VAR, in1_var)])
        if_1_codes.append(CodeAssign(var=data_var_1, expr=assign_expr_1))
        
        if_2_expr = HLSExpr(HLSExprT.CONST, "(!data.end_flag)")
        if_2_codes.append(CodeWriteStream(stream_var=out1_var, in_expr=data_var_1))
        assign_expr_2 = HLSExpr(HLSExprT.CONST, True)
        else_2_codes.append(CodeAssign(var=in1_end_flag_var, expr=assign_expr_2))
        if_2 = CodeIf(expr=if_2_expr, if_codes=if_2_codes, else_codes=else_2_codes)
        if_1_codes.append(if_2)
        
        # --- Build ELIF_1 (!in3.empty) ---
        elif_1_codes: List[HLSCodeLine] = []
        elif_1_expr = HLSExpr(HLSExprT.UOP, dfir.UnaryOp.NOT, operands=[
            HLSExpr(HLSExprT.STREAM_EMPTY, expr_val=None, operands=[HLSExpr(HLSExprT.VAR, in3_var)])
        ])
        # Build nested IF_3 (!data.end_flag)
        if_3_codes: List[HLSCodeLine] = []
        else_3_codes: List[HLSCodeLine] = []
        data_var_2 = HLSVar(var_name="data", var_type=net_wrapper_kt_pair_105_t_t_type)
        elif_1_codes.append(CodeVarDecl(var_name="data", var_type=data_var_2.type))
        assign_expr_3 = HLSExpr(HLSExprT.STREAM_READ, expr_val=None, operands=[HLSExpr(HLSExprT.VAR, in3_var)])
        elif_1_codes.append(CodeAssign(var=data_var_2, expr=assign_expr_3))
        
        if_3_expr = HLSExpr(HLSExprT.CONST, "(!data.end_flag)")
        if_3_codes.append(CodeWriteStream(stream_var=out1_var, in_expr=data_var_2))
        assign_expr_4 = HLSExpr(HLSExprT.CONST, True)
        else_3_codes.append(CodeAssign(var=in3_end_flag_var, expr=assign_expr_4))
        if_3 = CodeIf(expr=if_3_expr, if_codes=if_3_codes, else_codes=else_3_codes)
        elif_1_codes.append(if_3)
        
        # Create IF_1 statement (with if and elif)
        if_1_statement = CodeIf(expr=if_1_expr, if_codes=if_1_codes, elifs=[(elif_1_expr, elif_1_codes)])
        while_loop_codes.append(if_1_statement)
        # --- End IF_1 Block ---
        
        
        # --- Build IF_4 (!in2.empty) ---
        if_4_codes: List[HLSCodeLine] = []
        if_4_expr = HLSExpr(HLSExprT.UOP, dfir.UnaryOp.NOT, operands=[
            HLSExpr(HLSExprT.STREAM_EMPTY, expr_val=None, operands=[HLSExpr(HLSExprT.VAR, in2_var)])
        ])
        # Build nested IF_5 (!data.end_flag)
        if_5_codes: List[HLSCodeLine] = []
        else_5_codes: List[HLSCodeLine] = []
        data_var_3 = HLSVar(var_name="data", var_type=net_wrapper_kt_pair_105_t_t_type)
        if_4_codes.append(CodeVarDecl(var_name="data", var_type=data_var_3.type))
        assign_expr_5 = HLSExpr(HLSExprT.STREAM_READ, expr_val=None, operands=[HLSExpr(HLSExprT.VAR, in2_var)])
        if_4_codes.append(CodeAssign(var=data_var_3, expr=assign_expr_5))
        
        if_5_expr = HLSExpr(HLSExprT.CONST, "(!data.end_flag)")
        if_5_codes.append(CodeWriteStream(stream_var=out2_var, in_expr=data_var_3))
        assign_expr_6 = HLSExpr(HLSExprT.CONST, True)
        else_5_codes.append(CodeAssign(var=in2_end_flag_var, expr=assign_expr_6))
        if_5 = CodeIf(expr=if_5_expr, if_codes=if_5_codes, else_codes=else_5_codes)
        if_4_codes.append(if_5)
        
        # --- Build ELIF_2 (!in4.empty) ---
        elif_2_codes: List[HLSCodeLine] = []
        elif_2_expr = HLSExpr(HLSExprT.UOP, dfir.UnaryOp.NOT, operands=[
            HLSExpr(HLSExprT.STREAM_EMPTY, expr_val=None, operands=[HLSExpr(HLSExprT.VAR, in4_var)])
        ])
        # Build nested IF_6 (!data.end_flag)
        if_6_codes: List[HLSCodeLine] = []
        else_6_codes: List[HLSCodeLine] = []
        data_var_4 = HLSVar(var_name="data", var_type=net_wrapper_kt_pair_105_t_t_type)
        elif_2_codes.append(CodeVarDecl(var_name="data", var_type=data_var_4.type))
        assign_expr_7 = HLSExpr(HLSExprT.STREAM_READ, expr_val=None, operands=[HLSExpr(HLSExprT.VAR, in4_var)])
        elif_2_codes.append(CodeAssign(var=data_var_4, expr=assign_expr_7))
        
        if_6_expr = HLSExpr(HLSExprT.CONST, "(!data.end_flag)")
        if_6_codes.append(CodeWriteStream(stream_var=out2_var, in_expr=data_var_4))
        assign_expr_8 = HLSExpr(HLSExprT.CONST, True)
        else_6_codes.append(CodeAssign(var=in4_end_flag_var, expr=assign_expr_8))
        if_6 = CodeIf(expr=if_6_expr, if_codes=if_6_codes, else_codes=else_6_codes)
        elif_2_codes.append(if_6)
        
        # Create IF_4 statement (with if and elif)
        if_4_statement = CodeIf(expr=if_4_expr, if_codes=if_4_codes, elifs=[(elif_2_expr, elif_2_codes)])
        while_loop_codes.append(if_4_statement)
        # --- End IF_4 Block ---
        
        
        # --- Build IF_7 (all flags) ---
        if_7_codes: List[HLSCodeLine] = []
        if_7_expr = HLSExpr(HLSExprT.CONST, "(((in1_end_flag & in2_end_flag) & in3_end_flag) & in4_end_flag)")
        
        data_var_end = HLSVar(var_name="data", var_type=net_wrapper_kt_pair_105_t_t_type)
        if_7_codes.append(CodeVarDecl(var_name="data", var_type=data_var_end.type))
        
        data_end_flag_var = HLSVar(var_name="data.end_flag", var_type=bool_type)
        assign_expr_9 = HLSExpr(HLSExprT.CONST, True)
        if_7_codes.append(CodeAssign(var=data_end_flag_var, expr=assign_expr_9))
        
        if_7_codes.append(CodeWriteStream(stream_var=out1_var, in_expr=data_var_end))
        if_7_codes.append(CodeWriteStream(stream_var=out2_var, in_expr=data_var_end))
        if_7_codes.append(CodeBreak())
        
        # Create IF_7 statement (no else)
        if_7 = CodeIf(expr=if_7_expr, if_codes=if_7_codes)
        while_loop_codes.append(if_7)
        # --- End IF_7 Block ---
        
        # --- Create While Loop ---
        while_loop = CodeWhile(codes=while_loop_codes, iter_expr=while_true_expr)
        code_lines.append(while_loop)
        
        # --- 3. Finalize ---
        receiver_2_func.codes = code_lines
        self.big_gather_funcs.append(receiver_2_func)

        switch2x2_2_func = HLSFunction(name="switch2x2_2", comp=comp)
        params = []

        # --- 1. Define Types & Params ---

        # Basic Types
        int_type = HLSType(HLSBasicType.INT)
        node_id_type = HLSType(HLSBasicType.NODE_ID)
        ap_fixed_pod_t_type = HLSType(HLSBasicType.AP_FIXED_POD)
        bool_type = HLSType(HLSBasicType.BOOL)

        # Param 1: i
        i_var = HLSVar(var_name="i", var_type=int_type)

        # Param 2-5 Types: hls::stream<net_wrapper_kt_pair_105_t_t>
        # (Redefining net_wrapper_kt_pair_105_t_t)
        net_wrapper_kt_pair_105_t_t_type = HLSType(basic_type=HLSBasicType.STRUCT,
                                                  struct_name="net_wrapper_kt_pair_105_t_t",
                                                  struct_prop_names=["node_id", "prop", "end_flag"],
                                                  sub_types=[node_id_type, ap_fixed_pod_t_type, bool_type])
        if net_wrapper_kt_pair_105_t_t_type.name not in self.struct_definitions:
            self.struct_definitions[net_wrapper_kt_pair_105_t_t_type.name] = (net_wrapper_kt_pair_105_t_t_type, net_wrapper_kt_pair_105_t_t_type.struct_prop_names)

        stream_type = HLSType(HLSBasicType.STREAM, sub_types=[net_wrapper_kt_pair_105_t_t_type])

        in1_var = HLSVar(var_name="in1", var_type=stream_type)
        in2_var = HLSVar(var_name="in2", var_type=stream_type)
        out1_var = HLSVar(var_name="out1", var_type=stream_type)
        out2_var = HLSVar(var_name="out2", var_type=stream_type)

        params.extend([i_var, in1_var, in2_var, out1_var, out2_var])
        switch2x2_2_func.params = params

        # --- 2. Function Body ---
        code_lines: List[HLSCodeLine] = []

        # #pragma HLS DATAFLOW
        code_lines.append(CodePragma(content="DATAFLOW"))

        # hls::stream<net_wrapper_kt_pair_105_t_t> l1_1;
        code_lines.append(CodeVarDecl(var_name="l1_1", var_type=stream_type))
        l1_1_var = HLSVar(var_name="l1_1", var_type=stream_type)

        # #pragma HLS STREAM variable = l1_1 depth = 2
        code_lines.append(CodePragma(content="STREAM variable = l1_1 depth = 2"))

        # hls::stream<net_wrapper_kt_pair_105_t_t> l1_2;
        code_lines.append(CodeVarDecl(var_name="l1_2", var_type=stream_type))
        l1_2_var = HLSVar(var_name="l1_2", var_type=stream_type)

        # #pragma HLS STREAM variable = l1_2 depth = 2
        code_lines.append(CodePragma(content="STREAM variable = l1_2 depth = 2"))

        # hls::stream<net_wrapper_kt_pair_105_t_t> l1_3;
        code_lines.append(CodeVarDecl(var_name="l1_3", var_type=stream_type))
        l1_3_var = HLSVar(var_name="l1_3", var_type=stream_type)

        # #pragma HLS STREAM variable = l1_3 depth = 2
        code_lines.append(CodePragma(content="STREAM variable = l1_3 depth = 2"))

        # hls::stream<net_wrapper_kt_pair_105_t_t> l1_4;
        code_lines.append(CodeVarDecl(var_name="l1_4", var_type=stream_type))
        l1_4_var = HLSVar(var_name="l1_4", var_type=stream_type)

        # #pragma HLS STREAM variable = l1_4 depth = 2
        code_lines.append(CodePragma(content="STREAM variable = l1_4 depth = 2"))

        # sender_2(i, in1, in2, l1_1, l1_2, l1_3, l1_4);
        # (Assuming sender_2_func exists from previous generation)
        code_lines.append(CodeCall(func=sender_2_func, params=[i_var, in1_var, in2_var, l1_1_var, l1_2_var, l1_3_var, l1_4_var]))

        # receiver_2(i, out1, out2, l1_1, l1_2, l1_3, l1_4);
        # (Assuming receiver_2_func exists from previous generation)
        code_lines.append(CodeCall(func=receiver_2_func, params=[i_var, out1_var, out2_var, l1_1_var, l1_2_var, l1_3_var, l1_4_var]))

        # --- 3. Finalize ---
        switch2x2_2_func.codes = code_lines
        self.big_gather_funcs.append(switch2x2_2_func)



        omega_switch_2_func = HLSFunction(name="omega_switch_2", comp=comp)
        params = []

        # --- 1. Define Types & Params ---

        # Basic Types
        int_type = HLSType(HLSBasicType.INT)
        node_id_type = HLSType(HLSBasicType.NODE_ID)
        ap_fixed_pod_t_type = HLSType(HLSBasicType.AP_FIXED_POD)
        bool_type = HLSType(HLSBasicType.BOOL)

        # Param Types: hls::stream<net_wrapper_kt_pair_105_t_t> (&streams)[8]
        # (Redefining net_wrapper_kt_pair_105_t_t)
        net_wrapper_kt_pair_105_t_t_type = HLSType(basic_type=HLSBasicType.STRUCT,
                                                  struct_name="net_wrapper_kt_pair_105_t_t",
                                                  struct_prop_names=["node_id", "prop", "end_flag"],
                                                  sub_types=[node_id_type, ap_fixed_pod_t_type, bool_type])

        if net_wrapper_kt_pair_105_t_t_type.name not in self.struct_definitions:
            self.struct_definitions[net_wrapper_kt_pair_105_t_t_type.name] = (net_wrapper_kt_pair_105_t_t_type, net_wrapper_kt_pair_105_t_t_type.struct_prop_names)

        stream_type = HLSType(HLSBasicType.STREAM, sub_types=[net_wrapper_kt_pair_105_t_t_type])
        stream_array_type = HLSType(HLSBasicType.ARRAY, sub_types=[stream_type], array_dims=[8])

        in_streams_var = HLSVar(var_name="in_streams", var_type=stream_array_type)
        out_streams_var = HLSVar(var_name="out_streams", var_type=stream_array_type)

        params.extend([in_streams_var, out_streams_var])
        omega_switch_2_func.params = params

        # --- 2. Function Body ---
        code_lines: List[HLSCodeLine] = []

        # #pragma HLS DATAFLOW
        code_lines.append(CodePragma(content="DATAFLOW"))

        # hls::stream<net_wrapper_kt_pair_105_t_t> stream_stage_0[8];
        code_lines.append(CodeVarDecl(var_name="stream_stage_0", var_type=stream_array_type))

        # #pragma HLS STREAM variable = stream_stage_0 depth = 2
        code_lines.append(CodePragma(content="STREAM variable = stream_stage_0 depth = 2"))

        # #pragma HLS ARRAY_PARTITION variable = stream_stage_0 complete dim = 0
        code_lines.append(CodePragma(content="ARRAY_PARTITION variable = stream_stage_0 complete dim = 0"))

        # hls::stream<net_wrapper_kt_pair_105_t_t> stream_stage_1[8];
        code_lines.append(CodeVarDecl(var_name="stream_stage_1", var_type=stream_array_type))

        # #pragma HLS STREAM variable = stream_stage_1 depth = 2
        code_lines.append(CodePragma(content="STREAM variable = stream_stage_1 depth = 2"))

        # #pragma HLS ARRAY_PARTITION variable = stream_stage_1 complete dim = 0
        code_lines.append(CodePragma(content="ARRAY_PARTITION variable = stream_stage_1 complete dim = 0"))

        # --- Create HLSVar handles for function calls ---
        # (Assuming switch2x2_2_func is available from previous step)

        # Integer constants
        i_2_var = HLSVar(var_name="2", var_type=int_type)
        i_1_var = HLSVar(var_name="1", var_type=int_type)
        i_0_var = HLSVar(var_name="0", var_type=int_type)

        # Array element handles
        in_streams_vars = [HLSVar(var_name=f"in_streams[{i}]", var_type=stream_type) for i in range(8)]
        stream_stage_0_vars = [HLSVar(var_name=f"stream_stage_0[{i}]", var_type=stream_type) for i in range(8)]
        stream_stage_1_vars = [HLSVar(var_name=f"stream_stage_1[{i}]", var_type=stream_type) for i in range(8)]
        out_streams_elem_vars = [HLSVar(var_name=f"out_streams[{i}]", var_type=stream_type) for i in range(8)]


        # --- Stage 0 Calls ---
        # switch2x2_2(2, in_streams[0], in_streams[1], stream_stage_0[0], stream_stage_0[1]);
        code_lines.append(CodeCall(func=switch2x2_2_func, params=[i_2_var, in_streams_vars[0], in_streams_vars[1], stream_stage_0_vars[0], stream_stage_0_vars[1]]))
        # switch2x2_2(2, in_streams[2], in_streams[3], stream_stage_0[2], stream_stage_0[3]);
        code_lines.append(CodeCall(func=switch2x2_2_func, params=[i_2_var, in_streams_vars[2], in_streams_vars[3], stream_stage_0_vars[2], stream_stage_0_vars[3]]))
        # switch2x2_2(2, in_streams[4], in_streams[5], stream_stage_0[4], stream_stage_0[5]);
        code_lines.append(CodeCall(func=switch2x2_2_func, params=[i_2_var, in_streams_vars[4], in_streams_vars[5], stream_stage_0_vars[4], stream_stage_0_vars[5]]))
        # switch2x2_2(2, in_streams[6], in_streams[7], stream_stage_0[6], stream_stage_0[7]);
        code_lines.append(CodeCall(func=switch2x2_2_func, params=[i_2_var, in_streams_vars[6], in_streams_vars[7], stream_stage_0_vars[6], stream_stage_0_vars[7]]))

        # --- Stage 1 Calls ---
        # switch2x2_2(1, stream_stage_0[0], stream_stage_0[4], stream_stage_1[0], stream_stage_1[1]);
        code_lines.append(CodeCall(func=switch2x2_2_func, params=[i_1_var, stream_stage_0_vars[0], stream_stage_0_vars[4], stream_stage_1_vars[0], stream_stage_1_vars[1]]))
        # switch2x2_2(1, stream_stage_0[1], stream_stage_0[5], stream_stage_1[2], stream_stage_1[3]);
        code_lines.append(CodeCall(func=switch2x2_2_func, params=[i_1_var, stream_stage_0_vars[1], stream_stage_0_vars[5], stream_stage_1_vars[2], stream_stage_1_vars[3]]))
        # switch2x2_2(1, stream_stage_0[2], stream_stage_0[6], stream_stage_1[4], stream_stage_1[5]);
        code_lines.append(CodeCall(func=switch2x2_2_func, params=[i_1_var, stream_stage_0_vars[2], stream_stage_0_vars[6], stream_stage_1_vars[4], stream_stage_1_vars[5]]))
        # switch2x2_2(1, stream_stage_0[3], stream_stage_0[7], stream_stage_1[6], stream_stage_1[7]);
        code_lines.append(CodeCall(func=switch2x2_2_func, params=[i_1_var, stream_stage_0_vars[3], stream_stage_0_vars[7], stream_stage_1_vars[6], stream_stage_1_vars[7]]))

        # --- Stage 2 Calls (Output) ---
        # switch2x2_2(0, stream_stage_1[0], stream_stage_1[4], out_streams[0], out_streams[1]);
        code_lines.append(CodeCall(func=switch2x2_2_func, params=[i_0_var, stream_stage_1_vars[0], stream_stage_1_vars[4], out_streams_elem_vars[0], out_streams_elem_vars[1]]))
        # switch2x2_2(0, stream_stage_1[1], stream_stage_1[5], out_streams[2], out_streams[3]);
        code_lines.append(CodeCall(func=switch2x2_2_func, params=[i_0_var, stream_stage_1_vars[1], stream_stage_1_vars[5], out_streams_elem_vars[2], out_streams_elem_vars[3]]))
        # switch2x2_2(0, stream_stage_1[2], stream_stage_1[6], out_streams[4], out_streams[5]);
        code_lines.append(CodeCall(func=switch2x2_2_func, params=[i_0_var, stream_stage_1_vars[2], stream_stage_1_vars[6], out_streams_elem_vars[4], out_streams_elem_vars[5]]))
        # switch2x2_2(0, stream_stage_1[3], stream_stage_1[7], out_streams[6], out_streams[7]);
        code_lines.append(CodeCall(func=switch2x2_2_func, params=[i_0_var, stream_stage_1_vars[3], stream_stage_1_vars[7], out_streams_elem_vars[6], out_streams_elem_vars[7]]))

        # --- 3. Finalize ---
        omega_switch_2_func.codes = code_lines
        self.big_gather_funcs.append(omega_switch_2_func)

        get_raw_val_func = HLSFunction(name="get_raw_val", comp=comp)
        params = []

        # --- 1. Define Types & Params ---

        # Basic Types
        ap_fixed_pod_t_type = HLSType(HLSBasicType.AP_FIXED_POD)
        int_type = HLSType(HLSBasicType.INT)

        # Param 1: reduce_word_t
        reduce_word_t_type = HLSType(basic_type=HLSBasicType.AP_UINT, width="REDUCE_MEM_WIDTH")
        word_var = HLSVar(var_name="word", var_type=reduce_word_t_type)

        # Param 2: int
        idx_var = HLSVar(var_name="idx", var_type=int_type)

        params.extend([word_var, idx_var])
        get_raw_val_func.params = params

        get_val_func = HLSFunction(name="get_val", comp=comp)
        params = []

        # --- 1. Define Types & Params ---

        # Basic Types
        distance_t_type = HLSType(HLSBasicType.DISTANCE_T)
        ap_fixed_pod_t_type = HLSType(HLSBasicType.AP_FIXED_POD)
        int_type = HLSType(HLSBasicType.INT)

        # Param 1: reduce_word_t
        reduce_word_t_type = HLSType(basic_type=HLSBasicType.AP_UINT, width="REDUCE_MEM_WIDTH")
        word_var = HLSVar(var_name="word", var_type=reduce_word_t_type)

        # Param 2: int
        idx_var = HLSVar(var_name="idx", var_type=int_type)

        params.extend([word_var, idx_var])
        get_val_func.params = params


        set_val_func = HLSFunction(name="set_val", comp=comp)
        params = []

        # --- 1. Define Types & Params ---

        # Basic Types
        distance_t_type = HLSType(HLSBasicType.DISTANCE_T)
        int_type = HLSType(HLSBasicType.INT)

        # Param 1: reduce_word_t &word
        reduce_word_t_type = HLSType(basic_type=HLSBasicType.AP_UINT, width="REDUCE_MEM_WIDTH")
        # The '&' is handled by HLSFunction's code generation based on HLSType, 
        # so we just pass the base type.
        word_var = HLSVar(var_name="word", var_type=reduce_word_t_type) 

        # Param 2: int idx
        idx_var = HLSVar(var_name="idx", var_type=int_type)

        # Param 3: distance_t val
        val_var = HLSVar(var_name="val", var_type=distance_t_type)

        params.extend([word_var, idx_var, val_var])
        set_val_func.params = params

        set_raw_val_func = HLSFunction(name="set_raw_val", comp=comp)
        params = []

        # --- 1. Define Types & Params ---

        # Basic Types
        ap_fixed_pod_t_type = HLSType(HLSBasicType.AP_FIXED_POD)
        int_type = HLSType(HLSBasicType.INT)

        # Param 1: reduce_word_t &word
        reduce_word_t_type = HLSType(basic_type=HLSBasicType.AP_UINT, width="REDUCE_MEM_WIDTH")
        word_var = HLSVar(var_name="word", var_type=reduce_word_t_type) 

        # Param 2: int idx
        idx_var = HLSVar(var_name="idx", var_type=int_type)

        # Param 3: ap_fixed_pod_t pod_val
        pod_val_var = HLSVar(var_name="pod_val", var_type=ap_fixed_pod_t_type)

        params.extend([word_var, idx_var, pod_val_var])
        set_raw_val_func.params = params

        

        Reduc_105_unit_reduce_single_pe_func = HLSFunction(name="Reduc_105_unit_reduce_single_pe", comp=comp)
        params = []

        # --- 1. Define Types & Params ---

        # Basic Types
        int_type = HLSType(HLSBasicType.INT)
        bool_type = HLSType(HLSBasicType.BOOL)
        node_id_type = HLSType(HLSBasicType.NODE_ID)
        ap_fixed_pod_t_type = HLSType(HLSBasicType.AP_FIXED_POD)

        # Param 1 Type: hls::stream<net_wrapper_kt_pair_105_t_t>
        net_wrapper_kt_pair_105_t_t_type = HLSType(basic_type=HLSBasicType.STRUCT,
                                                  struct_name="net_wrapper_kt_pair_105_t_t",
                                                  struct_prop_names=["node_id", "prop", "end_flag"],
                                                  sub_types=[node_id_type, ap_fixed_pod_t_type, bool_type])
        
        if net_wrapper_kt_pair_105_t_t_type.name not in self.struct_definitions:
            self.struct_definitions[net_wrapper_kt_pair_105_t_t_type.name] = (net_wrapper_kt_pair_105_t_t_type, net_wrapper_kt_pair_105_t_t_type.struct_prop_names)


        kt_wrap_item_single_stream_type = HLSType(HLSBasicType.STREAM, sub_types=[net_wrapper_kt_pair_105_t_t_type])
        kt_wrap_item_single = HLSVar(var_name="kt_wrap_item_single", var_type=kt_wrap_item_single_stream_type)

        # Param 2 Type: hls::stream<reduce_word_t>
        reduce_word_t_type = HLSType(basic_type=HLSBasicType.REDUCE_WORD_T)
        pe_mem_out_stream_type = HLSType(HLSBasicType.STREAM, sub_types=[reduce_word_t_type])
        pe_mem_out = HLSVar(var_name="pe_mem_out", var_type=pe_mem_out_stream_type)

        # Param 3: pe_id
        pe_id = HLSVar(var_name="pe_id", var_type=int_type)

        # Param 4: dst_num
        dst_num = HLSVar(var_name="dst_num", var_type=int_type)

        params.extend([kt_wrap_item_single, pe_mem_out, pe_id, dst_num])
        Reduc_105_unit_reduce_single_pe_func.params = params

        # --- 2. Function Body ---
        code_lines: List[HLSCodeLine] = []

        # const int MEM_SIZE = (MAX_NUM >> LOG_PE_NUM) / DISTANCES_PER_REDUCE_WORD;
        code_lines.append(CodeVarDecl(var_name="MEM_SIZE", var_type=int_type, init_val="(MAX_NUM >> LOG_PE_NUM) / DISTANCES_PER_REDUCE_WORD", const=True))

        # reduce_word_t prop_mem[MEM_SIZE];
        prop_mem_type = HLSType(HLSBasicType.ARRAY, sub_types=[reduce_word_t_type], array_dims=["MEM_SIZE"])
        code_lines.append(CodeVarDecl(var_name="prop_mem", var_type=prop_mem_type))
        prop_mem_var = HLSVar(var_name="prop_mem", var_type=prop_mem_type)

        # #pragma HLS BIND_STORAGE variable = prop_mem type = RAM_2P impl = URAM
        code_lines.append(CodePragma(content="BIND_STORAGE variable = prop_mem type = RAM_2P impl = URAM"))

        # #pragma HLS dependence variable = prop_mem inter false
        code_lines.append(CodePragma(content="dependence variable = prop_mem inter false"))

        # reduce_word_t cache_data_buffer[L + 1];
        cache_data_buffer_type = HLSType(HLSBasicType.ARRAY, sub_types=[reduce_word_t_type], array_dims=["L + 1"])
        code_lines.append(CodeVarDecl(var_name="cache_data_buffer", var_type=cache_data_buffer_type))
        cache_data_buffer_var = HLSVar(var_name="cache_data_buffer", var_type=cache_data_buffer_type)

        # #pragma HLS ARRAY_PARTITION variable = cache_data_buffer complete dim = 0
        code_lines.append(CodePragma(content="ARRAY_PARTITION variable = cache_data_buffer complete dim = 0"))

        # int32_t cache_addr_buffer[L + 1];
        cache_addr_buffer_type = HLSType(HLSBasicType.ARRAY, sub_types=[int_type], array_dims=["L + 1"])
        code_lines.append(CodeVarDecl(var_name="cache_addr_buffer", var_type=cache_addr_buffer_type))
        cache_addr_buffer_var = HLSVar(var_name="cache_addr_buffer", var_type=cache_addr_buffer_type)

        # #pragma HLS ARRAY_PARTITION variable = cache_addr_buffer complete dim = 0
        code_lines.append(CodePragma(content="ARRAY_PARTITION variable = cache_addr_buffer complete dim = 0"))

        # const int32_t num_words = (dst_num + DISTANCES_PER_REDUCE_WORD - 1) / DISTANCES_PER_REDUCE_WORD;
        code_lines.append(CodeVarDecl(var_name="num_words", var_type=int_type, init_val="(dst_num + DISTANCES_PER_REDUCE_WORD - 1) / DISTANCES_PER_REDUCE_WORD", const=True))

        # const int32_t num_word_per_pe = (num_words + PE_NUM - 1) / PE_NUM;
        code_lines.append(CodeVarDecl(var_name="num_word_per_pe", var_type=int_type, init_val="(num_words + PE_NUM - 1) / PE_NUM", const=True))
        num_word_per_pe_var = HLSVar(var_name="num_word_per_pe", var_type=int_type)

        # for (int i = 0; i < L + 1; i++) {
        for_loop_1_codes: List[HLSCodeLine] = []
        for_loop_1 = CodeFor(codes=for_loop_1_codes,
                             iter_limit="L + 1",
                             iter_cmp="<",
                             iter_name="i",
                             iter_start="0",
                             iter_step="i++",
                             iter_val_type=int_type)
        code_lines.append(for_loop_1)

        # --- Inside for(i) 1 ---
        # #pragma HLS UNROLL
        for_loop_1_codes.append(CodePragma(content="UNROLL"))

        # cache_addr_buffer[i] = -1;
        cache_addr_buffer_i_var = HLSVar(var_name="cache_addr_buffer[i]", var_type=int_type)
        assign_expr_1 = HLSExpr(HLSExprT.CONST, -1)
        for_loop_1_codes.append(CodeAssign(var=cache_addr_buffer_i_var, expr=assign_expr_1))

        # bool end_flag = false;
        code_lines.append(CodeVarDecl(var_name="end_flag", var_type=bool_type, init_val="false"))

        # while (true) {
        while_loop_codes: List[HLSCodeLine] = []
        while_true_expr = HLSExpr(HLSExprT.CONST, True)
        while_loop = CodeWhile(codes=while_loop_codes, iter_expr=while_true_expr)
        code_lines.append(while_loop)

        # --- Inside while(true) ---
        # #pragma HLS PIPELINE II = 1
        while_loop_codes.append(CodePragma(content="PIPELINE II = 1"))

        # net_wrapper_kt_pair_105_t_t kt_elem;
        while_loop_codes.append(CodeVarDecl(var_name="kt_elem", var_type=net_wrapper_kt_pair_105_t_t_type))
        kt_elem_var = HLSVar(var_name="kt_elem", var_type=net_wrapper_kt_pair_105_t_t_type)

        # kt_elem = kt_wrap_item_single.read();
        assign_expr_2 = HLSExpr(HLSExprT.STREAM_READ, expr_val=None, operands=[HLSExpr(HLSExprT.VAR, kt_wrap_item_single)])
        while_loop_codes.append(CodeAssign(var=kt_elem_var, expr=assign_expr_2))

        # if (kt_elem.end_flag) {
        if_1_codes: List[HLSCodeLine] = []
        if_1_expr = HLSExpr(HLSExprT.CONST, "kt_elem.end_flag")
        if_1 = CodeIf(expr=if_1_expr, if_codes=if_1_codes)
        while_loop_codes.append(if_1)

        # --- Inside if(kt_elem.end_flag) ---
        # break;
        if_1_codes.append(CodeBreak())

        # int32_t key = kt_elem.node_id >> LOG_PE_NUM;
        while_loop_codes.append(CodeVarDecl(var_name="key", var_type=int_type, init_val="kt_elem.node_id >> LOG_PE_NUM"))

        # ap_fixed_pod_t incoming_dist_pod = kt_elem.prop;
        while_loop_codes.append(CodeVarDecl(var_name="incoming_dist_pod", var_type=ap_fixed_pod_t_type, init_val="kt_elem.prop"))
        incoming_dist_pod_var = HLSVar(var_name="incoming_dist_pod", var_type=ap_fixed_pod_t_type)

        # int32_t word_addr = (key >> 1);
        while_loop_codes.append(CodeVarDecl(var_name="word_addr", var_type=int_type, init_val="(key >> 1)"))
        word_addr_var = HLSVar(var_name="word_addr", var_type=int_type)

        # int32_t pack_idx = (key & 1);
        while_loop_codes.append(CodeVarDecl(var_name="pack_idx", var_type=int_type, init_val="(key & 1)"))
        pack_idx_var = HLSVar(var_name="pack_idx", var_type=int_type)

        # reduce_word_t current_word = prop_mem[word_addr];
        while_loop_codes.append(CodeVarDecl(var_name="current_word", var_type=reduce_word_t_type, init_val="prop_mem[word_addr]"))
        current_word_var = HLSVar(var_name="current_word", var_type=reduce_word_t_type)

        # for (int i = L; i >= 0; --i) {
        for_loop_2_codes: List[HLSCodeLine] = []
        for_loop_2 = CodeFor(codes=for_loop_2_codes,
                             iter_limit="0",
                             iter_cmp=">=",
                             iter_name="i",
                             iter_start="L",
                             iter_step="--i",
                             iter_val_type=int_type)
        while_loop_codes.append(for_loop_2)

        # --- Inside for(i) 2 ---
        # #pragma HLS UNROLL
        for_loop_2_codes.append(CodePragma(content="UNROLL"))

        # if (cache_addr_buffer[i] == word_addr) {
        if_2_codes: List[HLSCodeLine] = []
        if_2_expr = HLSExpr(HLSExprT.CONST, "cache_addr_buffer[i] == word_addr")
        if_2 = CodeIf(expr=if_2_expr, if_codes=if_2_codes)
        for_loop_2_codes.append(if_2)

        # --- Inside if(cache_addr_buffer[i] == word_addr) ---
        # current_word = cache_data_buffer[i];
        assign_expr_3 = HLSExpr(HLSExprT.CONST, "cache_data_buffer[i]")
        if_2_codes.append(CodeAssign(var=current_word_var, expr=assign_expr_3))

        # break;
        if_2_codes.append(CodeBreak())

        # for (int i = 0; i < L; i++) {
        for_loop_3_codes: List[HLSCodeLine] = []
        for_loop_3 = CodeFor(codes=for_loop_3_codes,
                             iter_limit="L",
                             iter_cmp="<",
                             iter_name="i",
                             iter_start="0",
                             iter_step="i++",
                             iter_val_type=int_type)
        while_loop_codes.append(for_loop_3)

        # --- Inside for(i) 3 ---
        # #pragma HLS UNROLL
        for_loop_3_codes.append(CodePragma(content="UNROLL"))

        # cache_addr_buffer[i] = cache_addr_buffer[i + 1];
        cache_addr_buffer_i_var_2 = HLSVar(var_name="cache_addr_buffer[i]", var_type=int_type)
        assign_expr_4 = HLSExpr(HLSExprT.CONST, "cache_addr_buffer[i + 1]")
        for_loop_3_codes.append(CodeAssign(var=cache_addr_buffer_i_var_2, expr=assign_expr_4))

        # cache_data_buffer[i] = cache_data_buffer[i + 1];
        cache_data_buffer_i_var = HLSVar(var_name="cache_data_buffer[i]", var_type=reduce_word_t_type)
        assign_expr_5 = HLSExpr(HLSExprT.CONST, "cache_data_buffer[i + 1]")
        for_loop_3_codes.append(CodeAssign(var=cache_data_buffer_i_var, expr=assign_expr_5))

        # ap_fixed_pod_t old_dist_pod = get_raw_val(current_word, pack_idx);
        # (Assuming get_raw_val_func exists from previous step)
        while_loop_codes.append(CodeVarDecl(var_name="old_dist_pod", var_type=ap_fixed_pod_t_type, init_val="get_raw_val(current_word, pack_idx)"))

        # ap_fixed_pod_t new_dist_pod = ...
        new_dist_pod_init_val = "(old_dist_pod < incoming_dist_pod && old_dist_pod != 0x0) ? old_dist_pod : incoming_dist_pod"
        while_loop_codes.append(CodeVarDecl(var_name="new_dist_pod", var_type=ap_fixed_pod_t_type, init_val=new_dist_pod_init_val))
        new_dist_pod_var = HLSVar(var_name="new_dist_pod", var_type=ap_fixed_pod_t_type)

        # set_raw_val(current_word, pack_idx, new_dist_pod);
        # (Assuming set_raw_val_func exists from previous step)
        call_params_1 = [current_word_var, pack_idx_var, new_dist_pod_var]
        while_loop_codes.append(CodeCall(func=set_raw_val_func, params=call_params_1))

        # prop_mem[word_addr] = current_word;
        prop_mem_word_addr_var = HLSVar(var_name="prop_mem[word_addr]", var_type=reduce_word_t_type)
        while_loop_codes.append(CodeAssign(var=prop_mem_word_addr_var, expr=current_word_var))

        # cache_addr_buffer[L] = word_addr;
        cache_addr_buffer_L_var = HLSVar(var_name="cache_addr_buffer[L]", var_type=int_type)
        while_loop_codes.append(CodeAssign(var=cache_addr_buffer_L_var, expr=word_addr_var))

        # cache_data_buffer[L] = current_word;
        cache_data_buffer_L_var = HLSVar(var_name="cache_data_buffer[L]", var_type=reduce_word_t_type)
        while_loop_codes.append(CodeAssign(var=cache_data_buffer_L_var, expr=current_word_var))
        # } (end while_loop)

        # for (int i = 0; i < num_word_per_pe; i++) {
        for_loop_4_codes: List[HLSCodeLine] = []
        for_loop_4 = CodeFor(codes=for_loop_4_codes,
                             iter_limit=num_word_per_pe_var,
                             iter_cmp="<",
                             iter_name="i",
                             iter_start="0",
                             iter_step="i++",
                             iter_val_type=int_type)
        code_lines.append(for_loop_4)

        # --- Inside for(i) 4 ---
        # #pragma HLS UNROLL factor = 1
        for_loop_4_codes.append(CodePragma(content="UNROLL factor = 1"))

        # pe_mem_out.write(prop_mem[i]);
        prop_mem_i_var = HLSVar(var_name="prop_mem[i]", var_type=reduce_word_t_type)
        for_loop_4_codes.append(CodeWriteStream(stream_var=pe_mem_out, in_expr=prop_mem_i_var))

        # --- 3. Finalize ---
        Reduc_105_unit_reduce_single_pe_func.codes = code_lines
        self.big_gather_funcs.append(Reduc_105_unit_reduce_single_pe_func)

        Reduc_105_drain_multi_pe_func = HLSFunction(name="Reduc_105_drain_multi_pe", comp=comp)
        params = []

        # --- 1. Define Types & Params ---

        # Basic Types
        int_type = HLSType(HLSBasicType.INT)
        uint_type = HLSType(HLSBasicType.UINT)

        # Param 1: hls::stream<reduce_word_t> (&pe_mem_in)[PE_NUM]
        reduce_word_t_type = HLSType(basic_type=HLSBasicType.REDUCE_WORD_T)
        pe_mem_in_stream_type = HLSType(HLSBasicType.STREAM, sub_types=[reduce_word_t_type])
        pe_mem_in_type = HLSType(HLSBasicType.ARRAY, sub_types=[pe_mem_in_stream_type], array_dims=["PE_NUM"])
        pe_mem_in = HLSVar(var_name="pe_mem_in", var_type=pe_mem_in_type)

        # Param 2: hls::stream<write_burst_pkt_t> &kernel_out_stream
        # (write_burst_pkt_t is defined in backend_defines.py)
        write_burst_pkt_t_type = HLSType(HLSBasicType.WRITE_BURST_PKT_T)

        for p in comp.ports:
            if p.name in ["i_0", "o_0"]:
                self.type_map[p] = write_burst_pkt_t_type
        
        kernel_out_stream_type = HLSType(HLSBasicType.STREAM, sub_types=[write_burst_pkt_t_type])
        kernel_out_stream = HLSVar(var_name="kernel_out_stream", var_type=kernel_out_stream_type)

        # Param 3: int32_t dst_num
        dst_num = HLSVar(var_name="dst_num", var_type=int_type)

        params.extend([pe_mem_in, kernel_out_stream, dst_num])
        Reduc_105_drain_multi_pe_func.params = params

        # --- 2. Function Body ---
        code_lines: List[HLSCodeLine] = []

        # write_burst_pkt_t one_write_burst;
        code_lines.append(CodeVarDecl(var_name="one_write_burst", var_type=write_burst_pkt_t_type))
        one_write_burst_var = HLSVar(var_name="one_write_burst", var_type=write_burst_pkt_t_type)

        # one_write_burst.last = 0;
        # (ap_axiu.last is ap_uint<1>)
        one_write_burst_last_type = HLSType(basic_type=HLSBasicType.AP_UINT, width=1)
        one_write_burst_last_var = HLSVar(var_name="one_write_burst.last", var_type=one_write_burst_last_type)
        assign_expr_1 = HLSExpr(HLSExprT.CONST, 0)
        code_lines.append(CodeAssign(var=one_write_burst_last_var, expr=assign_expr_1))

        # for (int32_t base_addr = 0; ...
        for_loop_1_codes: List[HLSCodeLine] = []
        for_loop_1 = CodeFor(codes=for_loop_1_codes,
                             iter_limit=dst_num, # HLSVar from params
                             iter_cmp="<",
                             iter_name="base_addr",
                             iter_start="0",
                             iter_step="base_addr += (PE_NUM << 1)",
                             iter_val_type=int_type)
        code_lines.append(for_loop_1)

        # --- Inside for(base_addr) ---
        # #pragma HLS PIPELINE II = 1
        for_loop_1_codes.append(CodePragma(content="PIPELINE II = 1"))

        # for (uint32_t pe_idx = 0; ...
        for_loop_2_codes: List[HLSCodeLine] = []
        for_loop_2 = CodeFor(codes=for_loop_2_codes,
                             iter_limit="PE_NUM",
                             iter_cmp="<",
                             iter_name="pe_idx",
                             iter_start="0",
                             iter_step="pe_idx++",
                             iter_val_type=uint_type)
        for_loop_1_codes.append(for_loop_2)

        # --- Inside for(pe_idx) ---
        # #pragma HLS UNROLL
        for_loop_2_codes.append(CodePragma(content="UNROLL"))

        # reduce_word_t word = pe_mem_in[pe_idx].read();
        for_loop_2_codes.append(CodeVarDecl(var_name="word", var_type=reduce_word_t_type, init_val="pe_mem_in[pe_idx].read()"))

        # one_write_burst.data.range(...) = word.range(31, 0);
        # (Using CodeOther for LHS .range())
        for_loop_2_codes.append(CodeOther(text="one_write_burst.data.range(31 + (pe_idx << 5), (pe_idx << 5)) ="))
        for_loop_2_codes.append(CodeOther(text="    word.range(31, 0);"))

        # one_write_burst.data.range(...) = word.range(63, 32);
        # (Using CodeOther for LHS .range())
        for_loop_2_codes.append(CodeOther(text="one_write_burst.data.range(31 + (pe_idx << 5) + 256,"))
        for_loop_2_codes.append(CodeOther(text="    (pe_idx << 5) + 256) ="))
        for_loop_2_codes.append(CodeOther(text="    word.range(63, 32);"))

        # } (end for_loop_2)

        # kernel_out_stream.write(one_write_burst);
        for_loop_1_codes.append(CodeWriteStream(stream_var=kernel_out_stream, in_expr=one_write_burst_var))
        # } (end for_loop_1)

        # --- 3. Finalize ---
        Reduc_105_drain_multi_pe_func.codes = code_lines
        self.big_gather_funcs.append(Reduc_105_drain_multi_pe_func)

        graphyflow_big_dataflow_func = HLSFunction(name="graphyflow_big_dataflow", comp=comp)
        params = []

        # --- 1. Define Types & Params ---

        # Basic Types
        node_id_type = HLSType(HLSBasicType.NODE_ID)
        ap_fixed_pod_t_type = HLSType(HLSBasicType.AP_FIXED_POD)
        bool_type = HLSType(HLSBasicType.BOOL)
        uint8_type = HLSType(HLSBasicType.UINT8)
        write_burst_pkt_t_type = HLSType(HLSBasicType.WRITE_BURST_PKT_T)
        int_type = HLSType(HLSBasicType.INT)

        # Param 1 Type: hls::stream<update_tuple_t>
        node_id_array_type = HLSType(HLSBasicType.ARRAY, sub_types=[node_id_type], array_dims=["PE_NUM"])
        prop_array_type = HLSType(HLSBasicType.ARRAY, sub_types=[ap_fixed_pod_t_type], array_dims=["PE_NUM"])
        update_tuple_t_type = HLSType(basic_type=HLSBasicType.STRUCT,
                                    struct_name="update_tuple_t",
                                    struct_prop_names=["node_id", "prop", "end_flag", "end_pos"],
                                    sub_types=[node_id_array_type, prop_array_type, bool_type, uint8_type])
        
        if update_tuple_t_type.name not in self.struct_definitions:
            self.struct_definitions[update_tuple_t_type.name] = (update_tuple_t_type, update_tuple_t_type.struct_prop_names)

        input_to_demux_stream_type = HLSType(HLSBasicType.STREAM, sub_types=[update_tuple_t_type])
        input_to_demux = HLSVar(var_name="input_to_demux", var_type=input_to_demux_stream_type)

        # Param 2 Type: hls::stream<write_burst_pkt_t>
        kernel_out_stream_type = HLSType(HLSBasicType.STREAM, sub_types=[write_burst_pkt_t_type])
        kernel_out_stream = HLSVar(var_name="kernel_out_stream", var_type=kernel_out_stream_type)

        # Param 3 Type: int32_t
        dst_num = HLSVar(var_name="dst_num", var_type=int_type)

        params.extend([input_to_demux, kernel_out_stream, dst_num])
        graphyflow_big_dataflow_func.params = params

        # --- 2. Define Internal Types ---

        # Internal Type: net_wrapper_kt_pair_105_t_t
        net_wrapper_kt_pair_105_t_t_type = HLSType(basic_type=HLSBasicType.STRUCT,
                                                  struct_name="net_wrapper_kt_pair_105_t_t",
                                                  struct_prop_names=["node_id", "prop", "end_flag"],
                                                  sub_types=[node_id_type, ap_fixed_pod_t_type, bool_type])
        if net_wrapper_kt_pair_105_t_t_type.name not in self.struct_definitions:
            self.struct_definitions[net_wrapper_kt_pair_105_t_t_type.name] = (net_wrapper_kt_pair_105_t_t_type, net_wrapper_kt_pair_105_t_t_type.struct_prop_names)


        # Internal Type: hls::stream<net_wrapper_kt_pair_105_t_t>
        reduce_stream_type = HLSType(HLSBasicType.STREAM, sub_types=[net_wrapper_kt_pair_105_t_t_type])
        reduce_stream_array_type = HLSType(HLSBasicType.ARRAY, sub_types=[reduce_stream_type], array_dims=[8])


        # Internal Type: reduce_word_t (ap_uint<REDUCE_MEM_WIDTH>)
        reduce_word_t_type = HLSType(basic_type=HLSBasicType.REDUCE_WORD_T)

        # Internal Type: hls::stream<reduce_word_t>
        pe_mem_stream_type = HLSType(HLSBasicType.STREAM, sub_types=[reduce_word_t_type])
        pe_mem_stream_array_type = HLSType(HLSBasicType.ARRAY, sub_types=[pe_mem_stream_type], array_dims=["PE_NUM"])


        # --- 3. Function Body ---
        code_lines: List[HLSCodeLine] = []

        # #pragma HLS DATAFLOW
        code_lines.append(CodePragma(content="DATAFLOW"))

        # hls::stream<net_wrapper_kt_pair_105_t_t> reduce_105_d2o_pair[8];
        code_lines.append(CodeVarDecl(var_name="reduce_105_d2o_pair", var_type=reduce_stream_array_type))
        reduce_105_d2o_pair_var = HLSVar(var_name="reduce_105_d2o_pair", var_type=reduce_stream_array_type)

        # #pragma HLS STREAM variable = reduce_105_d2o_pair depth = 16
        code_lines.append(CodePragma(content="STREAM variable = reduce_105_d2o_pair depth = 16"))

        # #pragma HLS ARRAY_PARTITION variable = reduce_105_d2o_pair complete dim = 0
        code_lines.append(CodePragma(content="ARRAY_PARTITION variable = reduce_105_d2o_pair complete dim = 0"))

        # hls::stream<net_wrapper_kt_pair_105_t_t> reduce_105_o2u_pair[8];
        code_lines.append(CodeVarDecl(var_name="reduce_105_o2u_pair", var_type=reduce_stream_array_type))
        reduce_105_o2u_pair_var = HLSVar(var_name="reduce_105_o2u_pair", var_type=reduce_stream_array_type)

        # #pragma HLS STREAM variable = reduce_105_o2u_pair depth = 2
        code_lines.append(CodePragma(content="STREAM variable = reduce_105_o2u_pair depth = 2"))

        # #pragma HLS ARRAY_PARTITION variable = reduce_105_o2u_pair complete dim = 0
        code_lines.append(CodePragma(content="ARRAY_PARTITION variable = reduce_105_o2u_pair complete dim = 0"))

        code_lines.append(CodeOther(text="")) # Blank line

        # demux_1(input_to_demux, reduce_105_d2o_pair);
        # (Assuming demux_1_func is defined)
        code_lines.append(CodeCall(func=demux_1_func, params=[input_to_demux, reduce_105_d2o_pair_var]))

        # omega_switch_2(reduce_105_d2o_pair, reduce_105_o2u_pair);
        # (Assuming omega_switch_2_func is defined)
        code_lines.append(CodeCall(func=omega_switch_2_func, params=[reduce_105_d2o_pair_var, reduce_105_o2u_pair_var]))

        # hls::stream<reduce_word_t> pe_mem_out_streams[PE_NUM];
        code_lines.append(CodeVarDecl(var_name="pe_mem_out_streams", var_type=pe_mem_stream_array_type))
        pe_mem_out_streams_var = HLSVar(var_name="pe_mem_out_streams", var_type=pe_mem_stream_array_type)

        # #pragma HLS STREAM variable = pe_mem_out_streams depth = 4
        code_lines.append(CodePragma(content="STREAM variable = pe_mem_out_streams depth = 4"))

        # for (int32_t pe_idx = 0; pe_idx < PE_NUM; pe_idx++) {
        for_loop_1_codes: List[HLSCodeLine] = []
        for_loop_1 = CodeFor(codes=for_loop_1_codes,
                             iter_limit="PE_NUM",
                             iter_cmp="<",
                             iter_name="pe_idx",
                             iter_start="0",
                             iter_step="pe_idx++",
                             iter_val_type=int_type)
        code_lines.append(for_loop_1)

        # --- Inside for(pe_idx) ---
        # #pragma HLS UNROLL
        for_loop_1_codes.append(CodePragma(content="UNROLL"))

        # Reduc_105_unit_reduce_single_pe(...)
        # (Assuming Reduc_105_unit_reduce_single_pe_func is defined)
        param_reduce_stream_pe = HLSVar(var_name="reduce_105_o2u_pair[pe_idx]", var_type=reduce_stream_type)
        param_pe_mem_stream_pe = HLSVar(var_name="pe_mem_out_streams[pe_idx]", var_type=pe_mem_stream_type)
        param_pe_idx = HLSVar(var_name="pe_idx", var_type=int_type)
        # dst_num is already defined as a function parameter HLSVar

        call_params = [param_reduce_stream_pe, param_pe_mem_stream_pe, param_pe_idx, dst_num]
        for_loop_1_codes.append(CodeCall(func=Reduc_105_unit_reduce_single_pe_func, params=call_params))
        # } (end for_loop_1)

        # Reduc_105_drain_multi_pe(pe_mem_out_streams, kernel_out_stream, dst_num);
        # (Assuming Reduc_105_drain_multi_pe_func is defined)
        code_lines.append(CodeCall(func=Reduc_105_drain_multi_pe_func, params=[pe_mem_out_streams_var, kernel_out_stream, dst_num]))

        # --- 4. Finalize ---
        graphyflow_big_dataflow_func.codes = code_lines
        
        
        
        
        self.big_gather_funcs.append(graphyflow_big_dataflow_func)
        self.big_top_dataflow_funcs.append(graphyflow_big_dataflow_func)


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
            print(comp.port_mapping)
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
            print(comp.port_mapping)
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

        merge_node_props_func = HLSFunction(name="merge_node_props", comp=None)
        params = []

        # --- 1. Define Types & Params ---

        # Basic Types
        node_id_type = HLSType(HLSBasicType.NODE_ID)
        bus_word_t_type = HLSType(HLSBasicType.BUS_WORD_T)
        ap_fixed_pod_t_type = HLSType(HLSBasicType.AP_FIXED_POD)
        distance_t_type = HLSType(HLSBasicType.DISTANCE_T)
        int_type = HLSType(HLSBasicType.INT)
        uint_type = HLSType(HLSBasicType.UINT)
        uint8_type = HLSType(HLSBasicType.UINT8)
        bool_type = HLSType(HLSBasicType.BOOL)

        # Special ap_uint type
        cache_idx_elem_type = HLSType(basic_type=HLSBasicType.AP_UINT, 
                                      width="NODE_ID_BITWIDTH - LOG_DIST_PER_WORD")

        # Param 1: hls::stream<bus_word_t> (&cacheline_streams)[PE_NUM]
        bus_word_stream_type = HLSType(HLSBasicType.STREAM, sub_types=[bus_word_t_type])
        cacheline_streams_type = HLSType(HLSBasicType.ARRAY, sub_types=[bus_word_stream_type], array_dims=["PE_NUM"])
        cacheline_streams = HLSVar(var_name="cacheline_streams", var_type=cacheline_streams_type)

        # Param 2: hls::stream<edge_descriptor_batch_t> &edge_stream
        edge_t_type = HLSType(basic_type=HLSBasicType.STRUCT,
                              struct_name="edge_t",
                              struct_prop_names=["src_id", "dst_id"],
                              sub_types=[node_id_type, node_id_type])
        if edge_t_type.name not in self.struct_definitions:
            self.struct_definitions[edge_t_type.name] = (edge_t_type, edge_t_type.struct_prop_names)

        edge_array_type = HLSType(HLSBasicType.ARRAY, sub_types=[edge_t_type], array_dims=["PE_NUM"])
        edge_descriptor_batch_t_type = HLSType(basic_type=HLSBasicType.STRUCT,
                                               struct_name="edge_descriptor_batch_t",
                                               struct_prop_names=["edges", "end_pos"],
                                               sub_types=[edge_array_type, int_type])
        if edge_descriptor_batch_t_type.name not in self.struct_definitions:
            self.struct_definitions[edge_descriptor_batch_t_type.name] = (edge_descriptor_batch_t_type, edge_descriptor_batch_t_type.struct_prop_names)

        edge_stream_type = HLSType(HLSBasicType.STREAM, sub_types=[edge_descriptor_batch_t_type])
        edge_stream = HLSVar(var_name="edge_stream", var_type=edge_stream_type)

        # Param 3: hls::stream<update_tuple_t> &edge_batch_stream
        node_id_array_pe_type = HLSType(HLSBasicType.ARRAY, sub_types=[node_id_type], array_dims=["PE_NUM"]) # Reused type
        prop_array_type = HLSType(HLSBasicType.ARRAY, sub_types=[ap_fixed_pod_t_type], array_dims=["PE_NUM"])
        update_tuple_t_type = HLSType(basic_type=HLSBasicType.STRUCT,
                                    struct_name="update_tuple_t",
                                    struct_prop_names=["node_id", "prop", "end_flag", "end_pos"],
                                    sub_types=[node_id_array_pe_type, prop_array_type, bool_type, uint8_type])

        if update_tuple_t_type.name not in self.struct_definitions:
            self.struct_definitions[update_tuple_t_type.name] = (update_tuple_t_type, update_tuple_t_type.struct_prop_names)


        edge_batch_stream_type = HLSType(HLSBasicType.STREAM, sub_types=[update_tuple_t_type])
        edge_batch_stream = HLSVar(var_name="edge_batch_stream", var_type=edge_batch_stream_type)

        # Param 4: uint32_t edge_num
        edge_num = HLSVar(var_name="edge_num", var_type=uint_type)

        params.extend([cacheline_streams, edge_stream, edge_batch_stream, edge_num])
        merge_node_props_func.params = params

        # --- 2. Function Body ---
        code_lines: List[HLSCodeLine] = []

        # bus_word_t last_cacheline[PE_NUM];
        last_cacheline_type = HLSType(HLSBasicType.ARRAY, sub_types=[bus_word_t_type], array_dims=["PE_NUM"])
        code_lines.append(CodeVarDecl(var_name="last_cacheline", var_type=last_cacheline_type))
        last_cacheline_var = HLSVar(var_name="last_cacheline", var_type=last_cacheline_type)

        # #pragma HLS ARRAY_PARTITION variable = last_cacheline complete dim = 0
        code_lines.append(CodePragma(content="ARRAY_PARTITION variable = last_cacheline complete dim = 0"))

        # ap_uint<...> last_cache_idx[PE_NUM];
        last_cache_idx_type = HLSType(HLSBasicType.ARRAY, sub_types=[cache_idx_elem_type], array_dims=["PE_NUM"])
        code_lines.append(CodeVarDecl(var_name="last_cache_idx", var_type=last_cache_idx_type))
        last_cache_idx_var = HLSVar(var_name="last_cache_idx", var_type=last_cache_idx_type)

        # #pragma HLS ARRAY_PARTITION variable = last_cache_idx complete dim = 0
        code_lines.append(CodePragma(content="ARRAY_PARTITION variable = last_cache_idx complete dim = 0"))

        # --- Build for(pe_idx) 1 ---
        for_loop_1_codes: List[HLSCodeLine] = []
        # #pragma HLS UNROLL
        for_loop_1_codes.append(CodePragma(content="UNROLL"))
        # last_cacheline[pe_idx] = cacheline_streams[pe_idx].read();
        last_cacheline_pe_idx_var = HLSVar(var_name="last_cacheline[pe_idx]", var_type=bus_word_t_type)
        assign_expr_1 = HLSExpr(HLSExprT.CONST, "cacheline_streams[pe_idx].read()")
        for_loop_1_codes.append(CodeAssign(var=last_cacheline_pe_idx_var, expr=assign_expr_1))
        # last_cache_idx[pe_idx] = 0;
        last_cache_idx_pe_idx_var = HLSVar(var_name="last_cache_idx[pe_idx]", var_type=cache_idx_elem_type)
        assign_expr_2 = HLSExpr(HLSExprT.CONST, 0)
        for_loop_1_codes.append(CodeAssign(var=last_cache_idx_pe_idx_var, expr=assign_expr_2))
        # Create for loop 1
        for_loop_1 = CodeFor(codes=for_loop_1_codes,
                             iter_limit="PE_NUM",
                             iter_cmp="<",
                             iter_name="pe_idx",
                             iter_start="0",
                             iter_step="pe_idx++",
                             iter_val_type=int_type)
        code_lines.append(for_loop_1)
        # --- End for(pe_idx) 1 ---

        # const uint32_t scatter_size = (edge_num + PE_NUM - 1) / PE_NUM;
        code_lines.append(CodeVarDecl(var_name="scatter_size", var_type=uint_type, init_val="(edge_num + PE_NUM - 1) / PE_NUM", const=True))
        scatter_size_var = HLSVar(var_name="scatter_size", var_type=uint_type)

        # distance_t real_edge_weight = 1.0; 
        code_lines.append(CodeVarDecl(var_name="real_edge_weight", var_type=distance_t_type, init_val="1.0", const=False))

        # const ap_fixed_pod_t edge_weight = (*reinterpret_cast<...>(&real_edge_weight));
        edge_weight_init_val = "(*reinterpret_cast<ap_fixed_pod_t *>(&real_edge_weight))"
        code_lines.append(CodeVarDecl(var_name="edge_weight", var_type=ap_fixed_pod_t_type, init_val=edge_weight_init_val, const=True))
        edge_weight_var = HLSVar(var_name="edge_weight", var_type=ap_fixed_pod_t_type)

        # --- Build for(edge_batch_idx) ---
        for_loop_2_codes: List[HLSCodeLine] = []
        # #pragma HLS PIPELINE II = 1
        for_loop_2_codes.append(CodePragma(content="PIPELINE II = 1"))

        # edge_descriptor_batch_t edge_batch;
        for_loop_2_codes.append(CodeVarDecl(var_name="edge_batch", var_type=edge_descriptor_batch_t_type))
        edge_batch_var = HLSVar(var_name="edge_batch", var_type=edge_descriptor_batch_t_type)

        # #pragma HLS ARRAY_PARTITION variable = edge_batch.edges complete dim = 0
        for_loop_2_codes.append(CodePragma(content="ARRAY_PARTITION variable = edge_batch.edges complete dim = 0"))

        # edge_batch = edge_stream.read();
        assign_expr_3 = HLSExpr(HLSExprT.CONST, "edge_stream.read()")
        for_loop_2_codes.append(CodeAssign(var=edge_batch_var, expr=assign_expr_3))

        # update_tuple_t out_batch;
        for_loop_2_codes.append(CodeVarDecl(var_name="out_batch", var_type=update_tuple_t_type))
        out_batch_var = HLSVar(var_name="out_batch", var_type=update_tuple_t_type)

        # #pragma HLS ARRAY_PARTITION variable = out_batch.node_id complete dim = 0
        for_loop_2_codes.append(CodePragma(content="ARRAY_PARTITION variable = out_batch.node_id complete dim = 0"))

        # #pragma HLS ARRAY_PARTITION variable = out_batch.prop complete dim = 0
        for_loop_2_codes.append(CodePragma(content="ARRAY_PARTITION variable = out_batch.prop complete dim = 0"))

        # out_batch.end_flag = false;
        out_batch_end_flag_var = HLSVar(var_name="out_batch.end_flag", var_type=bool_type)
        assign_expr_4 = HLSExpr(HLSExprT.CONST, False)
        for_loop_2_codes.append(CodeAssign(var=out_batch_end_flag_var, expr=assign_expr_4))

        # out_batch.end_pos = edge_batch.end_pos;
        out_batch_end_pos_var = HLSVar(var_name="out_batch.end_pos", var_type=uint8_type)
        assign_expr_5 = HLSExpr(HLSExprT.CONST, "edge_batch.end_pos")
        for_loop_2_codes.append(CodeAssign(var=out_batch_end_pos_var, expr=assign_expr_5))

        # bus_word_t cur_last_cacheline;
        for_loop_2_codes.append(CodeVarDecl(var_name="cur_last_cacheline", var_type=bus_word_t_type))
        cur_last_cacheline_var = HLSVar(var_name="cur_last_cacheline", var_type=bus_word_t_type)

        # ap_uint<...> cur_last_cache_idx;
        for_loop_2_codes.append(CodeVarDecl(var_name="cur_last_cache_idx", var_type=cache_idx_elem_type))
        cur_last_cache_idx_var = HLSVar(var_name="cur_last_cache_idx", var_type=cache_idx_elem_type)

        # --- Build for(pe_idx) 2 ---
        for_loop_3_codes: List[HLSCodeLine] = []
        # #pragma HLS UNROLL
        for_loop_3_codes.append(CodePragma(content="UNROLL"))

        # ap_uint<...> cacheline_idx = ...
        cacheline_idx_init_val = "(edge_batch.edges[pe_idx].src_id >> LOG_DIST_PER_WORD)"
        for_loop_3_codes.append(CodeVarDecl(var_name="cacheline_idx", var_type=cache_idx_elem_type, init_val=cacheline_idx_init_val))

        # uint32_t offset = ...
        offset_init_val = "(edge_batch.edges[pe_idx].src_id & (DIST_PER_WORD - 1))"
        for_loop_3_codes.append(CodeVarDecl(var_name="offset", var_type=uint_type, init_val=offset_init_val))

        # --- Build IF_1 (pe_idx < edge_batch.end_pos) ---
        if_1_codes: List[HLSCodeLine] = []
        if_expr_1 = HLSExpr(HLSExprT.CONST, "pe_idx < edge_batch.end_pos")
        # (This IF_1 has no else block)

        # --- Build IF_1 Contents ---
        cacheline_var = HLSVar(var_name="cacheline", var_type=bus_word_t_type)
        if_1_codes.append(CodeVarDecl(var_name="cacheline", var_type=cacheline_var.type))

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
        if_1_codes.append(if_2)
        # --- End IF_2 ---

        # ap_fixed_pod_t prop = get_val_from_bus(cacheline, offset);
        if_1_codes.append(CodeVarDecl(var_name="prop", var_type=ap_fixed_pod_t_type, init_val="get_val_from_bus(cacheline, offset)"))


        # ============= begin inline logic ==============
        # out_batch.node_id[pe_idx] = edge_batch.edges[pe_idx].dst_id;
        port_to_var = {}
        if_1_codes.append(CodeOther(text="// Begin inline logic"))
        DST_ID_VAR = HLSVar(var_name="DST_ID_VAR", var_type=node_id_type)
        SRC_PROP_VAR = HLSVar(var_name="SRC_PROP_VAR", var_type=ap_fixed_pod_t_type)
        EDGE_WEIGHT_VAR = HLSVar(var_name="EDGE_WEIGHT_VAR", var_type=ap_fixed_pod_t_type)
        top_vars = {
            "DST_ID_VAR": DST_ID_VAR,
            "SRC_PROP_VAR": SRC_PROP_VAR,
            "EDGE_WEIGHT_VAR": EDGE_WEIGHT_VAR
        }
        if_1_codes.append(CodeVarDecl(var_name="EDGE_WEIGHT_VAR", var_type=ap_fixed_pod_t_type, init_val="edge_weight"))
        if_1_codes.append(CodeVarDecl(var_name="SRC_PROP_VAR", var_type=ap_fixed_pod_t_type, init_val="prop"))
        if_1_codes.append(CodeVarDecl(var_name="DST_ID_VAR", var_type=node_id_type, init_val="edge_batch.edges[pe_idx].dst_id"))
        
        
        assign_expr_8 = HLSExpr(HLSExprT.CONST, "edge_batch.edges[pe_idx].dst_id")
        assign_expr_9 = HLSExpr(HLSExprT.CONST, "(prop + edge_weight)")


        out_batch_node_id_pe_idx_var = HLSVar(var_name="out_batch.node_id[pe_idx]", var_type=node_id_type)
        out_batch_prop_pe_idx_var = HLSVar(var_name="out_batch.prop[pe_idx]", var_type=ap_fixed_pod_t_type)

        top_vars["FINAL_DST_ID_VAR"] = out_batch_node_id_pe_idx_var
        top_vars["FINAL_PROP_VAR"] = out_batch_prop_pe_idx_var
        port_property = {} # 只能是src , dst, edge_prop这三项或组合
        inlinecodes = []
        for comp in scatter_stage_comps:
            port_property,inlinecodes = self._scatter_analyze(comp,port_property,port_to_var,top_vars,target_codes=inlinecodes)
        
        if_1_codes.extend(inlinecodes)
        
        #if_1_codes.append(CodeAssign(var=out_batch_node_id_pe_idx_var, expr=assign_expr_8))
        # out_batch.prop[pe_idx] = (prop + edge_weight);
        
        #if_1_codes.append(CodeAssign(var=out_batch_prop_pe_idx_var, expr=assign_expr_9))

        if_1_codes.append(CodeOther(text="// end inline logic"))
        # ============= end inline logic ==============


        # --- Build IF_3 (pe_idx == PE_NUM - 1) ---
        if_3_codes: List[HLSCodeLine] = []
        if_expr_3 = HLSExpr(HLSExprT.CONST, "pe_idx == PE_NUM - 1")
        # (This IF_3 has no else block)
        # Build IF_3 Contents
        if_3_codes.append(CodeAssign(var=cur_last_cacheline_var, expr=HLSExpr(HLSExprT.VAR, cacheline_var)))
        assign_expr_10 = HLSExpr(HLSExprT.CONST, "cacheline_idx")
        if_3_codes.append(CodeAssign(var=cur_last_cache_idx_var, expr=assign_expr_10))
        # Create IF_3
        if_3 = CodeIf(expr=if_expr_3, if_codes=if_3_codes)
        if_1_codes.append(if_3)
        # --- End IF_3 ---

        # Create IF_1
        if_1 = CodeIf(expr=if_expr_1, if_codes=if_1_codes)
        for_loop_3_codes.append(if_1)
        # --- End IF_1 ---

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

        # --- Build for(pe_idx) 3 ---
        for_loop_4_codes: List[HLSCodeLine] = []
        # #pragma HLS UNROLL
        for_loop_4_codes.append(CodePragma(content="UNROLL"))
        # last_cacheline[pe_idx] = cur_last_cacheline;
        last_cacheline_pe_idx_var_2 = HLSVar(var_name="last_cacheline[pe_idx]", var_type=bus_word_t_type)
        for_loop_4_codes.append(CodeAssign(var=last_cacheline_pe_idx_var_2, expr=cur_last_cacheline_var))
        # last_cache_idx[pe_idx] = cur_last_cache_idx;
        last_cache_idx_pe_idx_var_2 = HLSVar(var_name="last_cache_idx[pe_idx]", var_type=cache_idx_elem_type)
        for_loop_4_codes.append(CodeAssign(var=last_cache_idx_pe_idx_var_2, expr=cur_last_cache_idx_var))
        # Create for loop 4
        for_loop_4 = CodeFor(codes=for_loop_4_codes,
                             iter_limit="PE_NUM",
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

        # update_tuple_t end_batch;
        code_lines.append(CodeVarDecl(var_name="end_batch", var_type=update_tuple_t_type))
        end_batch_var = HLSVar(var_name="end_batch", var_type=update_tuple_t_type)

        # end_batch.end_flag = true;
        end_batch_end_flag_var = HLSVar(var_name="end_batch.end_flag", var_type=bool_type)
        assign_expr_11 = HLSExpr(HLSExprT.CONST, True)
        code_lines.append(CodeAssign(var=end_batch_end_flag_var, expr=assign_expr_11))

        # end_batch.end_pos = 0;
        end_batch_end_pos_var = HLSVar(var_name="end_batch.end_pos", var_type=uint8_type)
        assign_expr_12 = HLSExpr(HLSExprT.CONST, 0)
        code_lines.append(CodeAssign(var=end_batch_end_pos_var, expr=assign_expr_12))

        # edge_batch_stream.write(end_batch);
        code_lines.append(CodeWriteStream(stream_var=edge_batch_stream, in_expr=end_batch_var))

        # --- 3. Finalize ---
        merge_node_props_func.codes = code_lines

        self.big_scatter_funcs.append(merge_node_props_func)
        self.big_top_dataflow_funcs.append(merge_node_props_func)
        
        print("========= Scatter Stage =========")

        ## Little part:

        request_manager_func = HLSFunction(name="request_manager", comp=comp)
        params_req: List[HLSVar] = []

        # --- 1. 定义类型和参数 ---

        # 基本类型
        int_type = HLSType(HLSBasicType.INT)
        uint_type = HLSType(HLSBasicType.UINT)
        uint8_type = HLSType(HLSBasicType.UINT8)
        bool_type = HLSType(HLSBasicType.BOOL)

        # Typedefs (来自 HLSBasicType)
        bus_word_t_type = HLSType(HLSBasicType.BUS_WORD_T)
        distance_t_type = HLSType(HLSBasicType.DISTANCE_T)
        ap_fixed_pod_t_type = HLSType(HLSBasicType.AP_FIXED_POD)
        ppb_request_pkt_t_type = HLSType(HLSBasicType.PPB_REQUEST_PKT_T)
        ppb_response_pkt_t_type = HLSType(HLSBasicType.PPB_RESPONSE_PKT_T)
        node_id_t_type = HLSType(HLSBasicType.NODE_ID)

        # --- 详细的 Struct 定义 (基于 graphyflow_little.h) ---

        # 依赖: struct edge_t
        edge_t_sub_types = [node_id_t_type, node_id_t_type]
        edge_t_prop_names = ["src_id", "dst_id"]
        edge_t_type = HLSType(HLSBasicType.STRUCT, 
                              sub_types=edge_t_sub_types, 
                              struct_name="edge_t", 
                              struct_prop_names=edge_t_prop_names)

        # 1. struct edge_descriptor_batch_t
        edge_t_array_type = HLSType(HLSBasicType.ARRAY, sub_types=[edge_t_type], array_dims=["PE_NUM"])
        edge_desc_sub_types = [edge_t_array_type, int_type]
        edge_desc_prop_names = ["edges", "end_pos"]
        edge_descriptor_batch_t_type = HLSType(HLSBasicType.STRUCT, 
                                               sub_types=edge_desc_sub_types, 
                                               struct_name="edge_descriptor_batch_t", 
                                               struct_prop_names=edge_desc_prop_names)

        # 2. struct update_tuple_t
        node_id_array_type = HLSType(HLSBasicType.ARRAY, sub_types=[node_id_t_type], array_dims=["PE_NUM"])
        prop_array_type = HLSType(HLSBasicType.ARRAY, sub_types=[ap_fixed_pod_t_type], array_dims=["PE_NUM"])
        update_tuple_sub_types = [node_id_array_type, prop_array_type, bool_type, uint8_type]
        update_tuple_prop_names = ["node_id", "prop", "end_flag", "end_pos"]
        update_tuple_t_type = HLSType(HLSBasicType.STRUCT, 
                                      sub_types=update_tuple_sub_types, 
                                      struct_name="update_tuple_t", 
                                      struct_prop_names=update_tuple_prop_names)

        # --- 结束 Struct 定义 ---


        # Param 1: hls::stream<edge_descriptor_batch_t> &edge_burst_stm
        edge_burst_stm_type = HLSType(HLSBasicType.STREAM, sub_types=[edge_descriptor_batch_t_type])
        param_edge_burst_stm = HLSVar(var_name="edge_burst_stm", var_type=edge_burst_stm_type)

        # Param 2: hls::stream<ppb_request_pkt_t> &ppb_request_stm
        ppb_request_stm_type = HLSType(HLSBasicType.STREAM, sub_types=[ppb_request_pkt_t_type])
        param_ppb_request_stm = HLSVar(var_name="ppb_request_stm", var_type=ppb_request_stm_type)

        # Param 3: hls::stream<ppb_response_pkt_t> &ppb_response_stm
        ppb_response_stm_type = HLSType(HLSBasicType.STREAM, sub_types=[ppb_response_pkt_t_type])
        param_ppb_response_stm = HLSVar(var_name="ppb_response_stm", var_type=ppb_response_stm_type)

        # Param 4: hls::stream<update_tuple_t> &update_set_stm
        update_set_stm_type = HLSType(HLSBasicType.STREAM, sub_types=[update_tuple_t_type])
        param_update_set_stm = HLSVar(var_name="update_set_stm", var_type=update_set_stm_type)

        # Param 5: int32_t part_edge_num
        param_part_edge_num = HLSVar(var_name="part_edge_num", var_type=int_type)

        params_req.extend([param_edge_burst_stm, param_ppb_request_stm, param_ppb_response_stm, param_update_set_stm, param_part_edge_num])
        request_manager_func.params = params_req

        # --- 2. 函数体 (构建本地列表) ---
        code_lines_req: List[HLSCodeLine] = []

        # bus_word_t src_prop_buffer[PE_NUM][2][SRC_BUFFER_SIZE >> 4];
        src_buffer_dims = ["PE_NUM", "2", "(SRC_BUFFER_SIZE >> 4)"] # 宏/表达式作为字符串
        src_prop_buffer_type = HLSType(HLSBasicType.ARRAY, sub_types=[bus_word_t_type], array_dims=src_buffer_dims)
        code_lines_req.append(CodeVarDecl(var_name="src_prop_buffer", var_type=src_prop_buffer_type))

        # Pragmas for src_prop_buffer
        code_lines_req.append(CodePragma(content="ARRAY_PARTITION variable = src_prop_buffer dim = 1 complete"))
        code_lines_req.append(CodePragma(content="BIND_STORAGE variable = src_prop_buffer type = RAM_S2P impl = BRAM"))
        code_lines_req.append(CodePragma(content="dependence variable = src_prop_buffer inter false"))

        # int32_t pp_read_idx = 0;
        code_lines_req.append(CodeVarDecl(var_name="pp_read_idx", var_type=int_type, init_val="0"))
        pp_read_idx_var = HLSVar(var_name="pp_read_idx", var_type=int_type)

        # int32_t pp_write_idx = 0;
        code_lines_req.append(CodeVarDecl(var_name="pp_write_idx", var_type=int_type, init_val="0"))
        pp_write_idx_var = HLSVar(var_name="pp_write_idx", var_type=int_type)

        # int32_t pp_reponse_idx = 0;
        code_lines_req.append(CodeVarDecl(var_name="pp_reponse_idx", var_type=int_type, init_val="0"))
        pp_reponse_idx_var = HLSVar(var_name="pp_reponse_idx", var_type=int_type)

        # int32_t pp_read_round = 0;
        code_lines_req.append(CodeVarDecl(var_name="pp_read_round", var_type=int_type, init_val="0"))
        pp_read_round_var = HLSVar(var_name="pp_read_round", var_type=int_type)

        # int32_t pp_write_round = 0;
        code_lines_req.append(CodeVarDecl(var_name="pp_write_round", var_type=int_type, init_val="0"))
        pp_write_round_var = HLSVar(var_name="pp_write_round", var_type=int_type)

        # int32_t pp_request_round = 0;
        code_lines_req.append(CodeVarDecl(var_name="pp_request_round", var_type=int_type, init_val="0"))
        pp_request_round_var = HLSVar(var_name="pp_request_round", var_type=int_type)

        # int32_t edge_set_cnt = 0;
        code_lines_req.append(CodeVarDecl(var_name="edge_set_cnt", var_type=int_type, init_val="0"))
        edge_set_cnt_var = HLSVar(var_name="edge_set_cnt", var_type=int_type)

        # const int32_t total_edge_sets = (part_edge_num + PE_NUM - 1) / PE_NUM;
        code_lines_req.append(CodeVarDecl(var_name="total_edge_sets", var_type=int_type, init_val="(part_edge_num + PE_NUM - 1) / PE_NUM", const=True))
        total_edge_sets_var = HLSVar(var_name="total_edge_sets", var_type=int_type)

        # bool wait_flag = 0;
        code_lines_req.append(CodeVarDecl(var_name="wait_flag", var_type=bool_type, init_val="0"))
        wait_flag_var = HLSVar(var_name="wait_flag", var_type=bool_type)

        # edge_descriptor_batch_t an_edge_burst;
        code_lines_req.append(CodeVarDecl(var_name="an_edge_burst", var_type=edge_descriptor_batch_t_type))
        an_edge_burst_var = HLSVar(var_name="an_edge_burst", var_type=edge_descriptor_batch_t_type)

        #pragma HLS ARRAY_PARTITION variable = an_edge_burst.edges complete dim = 0
        code_lines_req.append(CodePragma(content="ARRAY_PARTITION variable = an_edge_burst.edges complete dim = 0"))

        # ppb_request_pkt_t one_ppb_request;
        code_lines_req.append(CodeVarDecl(var_name="one_ppb_request", var_type=ppb_request_pkt_t_type))
        one_ppb_request_var = HLSVar(var_name="one_ppb_request", var_type=ppb_request_pkt_t_type)

        # ppb_response_pkt_t one_ppb_response;
        code_lines_req.append(CodeVarDecl(var_name="one_ppb_response", var_type=ppb_response_pkt_t_type))
        one_ppb_response_var = HLSVar(var_name="one_ppb_response", var_type=ppb_response_pkt_t_type)

        # distance_t real_edge_weight = 1.0;
        code_lines_req.append(CodeVarDecl(var_name="real_edge_weight", var_type=distance_t_type, init_val="1.0"))
        real_edge_weight_var = HLSVar(var_name="real_edge_weight", var_type=distance_t_type)

        # const ap_fixed_pod_t edge_weight = (*reinterpret_cast<ap_fixed_pod_t *>(&real_edge_weight));
        code_lines_req.append(CodeVarDecl(var_name="edge_weight", var_type=ap_fixed_pod_t_type, init_val="(*reinterpret_cast<ap_fixed_pod_t *>(&real_edge_weight))", const=True))
        edge_weight_var = HLSVar(var_name="edge_weight", var_type=ap_fixed_pod_t_type)

        # const uint32_t total_rounds = (part_edge_num + SRC_BUFFER_SIZE - 1) / SRC_BUFFER_SIZE;
        code_lines_req.append(CodeVarDecl(var_name="total_rounds", var_type=uint_type, init_val="(part_edge_num + SRC_BUFFER_SIZE - 1) / SRC_BUFFER_SIZE", const=True))
        total_rounds_var = HLSVar(var_name="total_rounds", var_type=uint_type)


        # while (true)
        while_1_codes: List[HLSCodeLine] = []
        while_1_expr = HLSExpr(HLSExprT.CONST, True) # 使用 Python bool
        while_1 = CodeWhile(codes=while_1_codes, iter_expr=while_1_expr)
        code_lines_req.append(while_1)

        # --- 在 while(true) 循环内部 ---
        # #pragma HLS PIPELINE II = 1
        while_1_codes.append(CodePragma(content="PIPELINE II = 1"))

        # if ((pp_request_round - pp_read_round) <= 1) { ... }
        if_1_codes: List[HLSCodeLine] = []
        # HLSExpr(HLSExprT.CONST, "...") 用于 CodeIf 的表达式
        if_1_expr = HLSExpr(HLSExprT.CONST, "((pp_request_round - pp_read_round) <= 1)")
        if_1 = CodeIf(expr=if_1_expr, if_codes=if_1_codes)
        while_1_codes.append(if_1)

        # --- 在 if( (pp_request_round - pp_read_round) <= 1 ) 内部 ---
        # if (pp_request_round < pp_read_round)
        if_2_codes: List[HLSCodeLine] = []
        if_2_expr = HLSExpr(HLSExprT.CONST, "(pp_request_round < pp_read_round)")
        if_2 = CodeIf(expr=if_2_expr, if_codes=if_2_codes)
        if_1_codes.append(if_2)
        # pp_request_round = pp_read_round;
        if_2_codes.append(CodeAssign(var=pp_request_round_var, expr=HLSExpr(HLSExprT.VAR, pp_read_round_var)))

        # one_ppb_request.data = pp_request_round;
        one_ppb_request_data_var = HLSVar(var_name="one_ppb_request.data", var_type=int_type) # 假设 data 匹配 round 类型
        if_1_codes.append(CodeAssign(var=one_ppb_request_data_var, expr=HLSExpr(HLSExprT.VAR, pp_request_round_var)))

        # one_ppb_request.last = 0;
        one_ppb_request_last_var = HLSVar(var_name="one_ppb_request.last", var_type=bool_type) # 假设
        if_1_codes.append(CodeAssign(var=one_ppb_request_last_var, expr=HLSExpr(HLSExprT.CONST, 0))) # 使用 Python int

        # ppb_request_stm.write(one_ppb_request);
        if_1_codes.append(CodeWriteStream(stream_var=param_ppb_request_stm, in_expr=one_ppb_request_var))

        # pp_request_round++;
        if_1_codes.append(CodeOther(text="pp_request_round++;"))


        # if (ppb_response_stm.read_nb(one_ppb_response)) { ... }
        # (修正：直接在 CodeIf 中使用 read_nb 表达式字符串)
        if_3_codes: List[HLSCodeLine] = []
        if_3_expr = HLSExpr(HLSExprT.CONST, "ppb_response_stm.read_nb(one_ppb_response)")
        if_3 = CodeIf(expr=if_3_expr, if_codes=if_3_codes)
        while_1_codes.append(if_3)

        # --- 在 if( read_nb ) 内部 ---
        # pp_write_round = one_ppb_response.dest << 4 >> LOG_SRC_BUFFER_SIZE;
        # (修正：CodeAssign 和 HLSExprT.CONST 字符串)
        if_3_codes.append(CodeAssign(var=pp_write_round_var, expr=HLSExpr(HLSExprT.CONST, "one_ppb_response.dest << 4 >> LOG_SRC_BUFFER_SIZE")))

        # bool write_buffer = pp_write_round & 0x1;
        if_3_codes.append(CodeVarDecl(var_name="write_buffer", var_type=bool_type, init_val="(pp_write_round & 0x1)"))
        write_buffer_var = HLSVar(var_name="write_buffer", var_type=bool_type)

        # int32_t write_idx = one_ppb_response.dest & ((SRC_BUFFER_SIZE >> 4) - 1);
        if_3_codes.append(CodeVarDecl(var_name="write_idx", var_type=int_type, init_val="one_ppb_response.dest & ((SRC_BUFFER_SIZE >> 4) - 1)"))
        write_idx_var = HLSVar(var_name="write_idx", var_type=int_type)

        # bus_word_t one_read_burst = one_ppb_response.data;
        if_3_codes.append(CodeVarDecl(var_name="one_read_burst", var_type=bus_word_t_type, init_val="one_ppb_response.data"))
        one_read_burst_var = HLSVar(var_name="one_read_burst", var_type=bus_word_t_type)

        # for (int u = 0; u < PE_NUM; u++) { ... }
        for_1_codes: List[HLSCodeLine] = []
        for_1 = CodeFor(codes=for_1_codes, iter_limit="PE_NUM", iter_cmp="<", iter_name="u", iter_start="0", iter_step="u++", iter_val_type=int_type)
        if_3_codes.append(for_1)
        # #pragma HLS UNROLL
        for_1_codes.append(CodePragma(content="UNROLL"))
        # src_prop_buffer[u][write_buffer][write_idx] = one_read_burst;
        for_1_codes.append(CodeOther(text="src_prop_buffer[u][write_buffer][write_idx] = one_read_burst;"))

        # --- 结束 if( read_nb ) ---

        # if (!wait_flag)
        if_4_codes: List[HLSCodeLine] = []
        if_4_expr = HLSExpr(HLSExprT.UOP, dfir.UnaryOp.NOT, operands=[HLSExpr(HLSExprT.VAR, wait_flag_var)])
        if_4 = CodeIf(expr=if_4_expr, if_codes=if_4_codes)
        while_1_codes.append(if_4)
        # an_edge_burst = edge_burst_stm.read();
        # (修正：添加 expr_val=None)
        if_4_codes.append(CodeAssign(var=an_edge_burst_var, expr=HLSExpr(HLSExprT.STREAM_READ, expr_val=None, operands=[HLSExpr(HLSExprT.VAR, param_edge_burst_stm)])))

        # pp_read_round = (an_edge_burst.edges[0].src_id / SRC_BUFFER_SIZE);
        while_1_codes.append(CodeOther(text="pp_read_round = (an_edge_burst.edges[0].src_id / SRC_BUFFER_SIZE);"))

        # wait_flag = (pp_read_round >= pp_write_round) ? 1 : 0;
        # (CodeAssign 和 HLSExprT.CONST 字符串)
        while_1_codes.append(CodeAssign(var=wait_flag_var, expr=HLSExpr(HLSExprT.CONST, "(pp_read_round >= pp_write_round) ? 1 : 0")))

        # bool exit_flag = (wait_flag == 0) ? (edge_set_cnt + 1 >= total_edge_sets) : (edge_set_cnt >= total_edge_sets);
        while_1_codes.append(CodeVarDecl(var_name="exit_flag", var_type=bool_type, init_val="(wait_flag == 0) ? (edge_set_cnt + 1 >= total_edge_sets) : (edge_set_cnt >= total_edge_sets)"))
        exit_flag_var = HLSVar(var_name="exit_flag", var_type=bool_type)

        # if (!wait_flag) { ... }
        if_5_codes: List[HLSCodeLine] = []
        if_5_expr = HLSExpr(HLSExprT.UOP, dfir.UnaryOp.NOT, operands=[HLSExpr(HLSExprT.VAR, wait_flag_var)])
        if_5 = CodeIf(expr=if_5_expr, if_codes=if_5_codes)
        while_1_codes.append(if_5)

        # --- 在 if( !wait_flag ) 内部 ---
        # bool read_buffer = pp_read_round & 0x1;
        if_5_codes.append(CodeVarDecl(var_name="read_buffer", var_type=bool_type, init_val="(pp_read_round & 0x1)"))
        read_buffer_var = HLSVar(var_name="read_buffer", var_type=bool_type)

        # update_tuple_t an_update_set;
        if_5_codes.append(CodeVarDecl(var_name="an_update_set", var_type=update_tuple_t_type))
        an_update_set_var = HLSVar(var_name="an_update_set", var_type=update_tuple_t_type)

        # Pragmas for an_update_set
        if_5_codes.append(CodePragma(content="ARRAY_PARTITION variable = an_update_set.prop complete dim = 0"))
        if_5_codes.append(CodePragma(content="ARRAY_PARTITION variable = an_update_set.node_id complete dim = 0"))

        # for (int u = 0; u < PE_NUM; u++) { ... }
        for_2_codes: List[HLSCodeLine] = []
        for_2 = CodeFor(codes=for_2_codes, iter_limit="PE_NUM", iter_cmp="<", iter_name="u", iter_start="0", iter_step="u++", iter_val_type=int_type)
        if_5_codes.append(for_2)

        # --- 在 for(u) 内部 ---
        # #pragma HLS UNROLL
        for_2_codes.append(CodePragma(content="UNROLL"))

        # ap_uint<31> idx = (an_edge_burst.edges[u].src_id % SRC_BUFFER_SIZE);
        ap_uint_31_type = HLSType(basic_type=HLSBasicType.AP_UINT, width=31)
        for_2_codes.append(CodeVarDecl(var_name="idx", var_type=ap_uint_31_type, init_val="(an_edge_burst.edges[u].src_id % SRC_BUFFER_SIZE)"))
        idx_var = HLSVar(var_name="idx", var_type=ap_uint_31_type)

        # ap_uint<30> uram_row_idx = idx >> 4;
        ap_uint_30_type = HLSType(basic_type=HLSBasicType.AP_UINT, width=30)
        for_2_codes.append(CodeVarDecl(var_name="uram_row_idx", var_type=ap_uint_30_type, init_val="(idx >> 4)"))
        uram_row_idx_var = HLSVar(var_name="uram_row_idx", var_type=ap_uint_30_type)

        # ap_uint<30> uram_row_offset = (idx & 0xf);
        for_2_codes.append(CodeVarDecl(var_name="uram_row_offset", var_type=ap_uint_30_type, init_val="(idx & 0xf)"))
        uram_row_offset_var = HLSVar(var_name="uram_row_offset", var_type=ap_uint_30_type)

        # bus_word_t uram_row = src_prop_buffer[u][read_buffer][uram_row_idx];
        for_2_codes.append(CodeOther(text="bus_word_t uram_row = src_prop_buffer[u][read_buffer][uram_row_idx];"))

        # ap_fixed_pod_t src_prop = get_val_from_bus(uram_row, uram_row_offset);
        for_2_codes.append(CodeVarDecl(var_name="src_prop", var_type=ap_fixed_pod_t_type, init_val="get_val_from_bus(uram_row, uram_row_offset)"))
        src_prop_var = HLSVar(var_name="src_prop", var_type=ap_fixed_pod_t_type)


        #============= begin inline logic ===============
        port_to_var = {}
        for_2_codes.append(CodeOther(text="// Begin inline logic"))
        DST_ID_VAR = HLSVar(var_name="DST_ID_VAR", var_type=node_id_type)
        for_2_codes.append(CodeVarDecl(var_name="DST_ID_VAR", var_type=node_id_type, init_val="an_edge_burst.edges[u].dst_id"))
        SRC_PROP_VAR = src_prop_var
        EDGE_WEIGHT_VAR = edge_weight_var
        top_vars = {
            "DST_ID_VAR": DST_ID_VAR,
            "SRC_PROP_VAR": SRC_PROP_VAR,
            "EDGE_WEIGHT_VAR": EDGE_WEIGHT_VAR
        }
        # an_update_set.prop[u] = (src_prop + edge_weight);
        #for_2_codes.append(CodeOther(text="an_update_set.prop[u] = (src_prop + edge_weight);"))

        # an_update_set.node_id[u] = an_edge_burst.edges[u].dst_id;
        #for_2_codes.append(CodeOther(text="an_update_set.node_id[u] = an_edge_burst.edges[u].dst_id;"))
        an_update_set_prop_var = HLSVar(var_name="an_update_set.prop[u]", var_type=ap_fixed_pod_t_type)
        an_update_set_node_id_var = HLSVar(var_name="an_update_set.node_id[u]", var_type=node_id_type)
        top_vars["FINAL_PROP_VAR"] = an_update_set_prop_var
        top_vars["FINAL_DST_ID_VAR"] = an_update_set_node_id_var
        port_property = {} # 只能是src , dst, edge_prop这三项或组合
        inlinecodes = []
        for comp in scatter_stage_comps:
            port_property,inlinecodes = self._scatter_analyze(comp,port_property,port_to_var,top_vars,target_codes=inlinecodes)
        for_2_codes.extend(inlinecodes)
        for_2_codes.append(CodeOther(text="// End inline logic"))
        # =========== end inline logic ==============
        # --- 结束 for(u) ---

        # update_set_stm.write(an_update_set);
        if_5_codes.append(CodeWriteStream(stream_var=param_update_set_stm, in_expr=an_update_set_var))

        # edge_set_cnt++;
        if_5_codes.append(CodeOther(text="edge_set_cnt++;"))

        # --- 结束 if( !wait_flag ) ---

        # if (exit_flag) { ... }
        if_6_codes: List[HLSCodeLine] = []
        if_6_expr = HLSExpr(HLSExprT.VAR, exit_flag_var)
        if_6 = CodeIf(expr=if_6_expr, if_codes=if_6_codes)
        while_1_codes.append(if_6)

        # --- 在 if( exit_flag ) 内部 ---
        # one_ppb_request.last = 1;
        if_6_codes.append(CodeAssign(var=one_ppb_request_last_var, expr=HLSExpr(HLSExprT.CONST, 1))) # 使用 Python int

        # ppb_request_stm.write(one_ppb_request);
        if_6_codes.append(CodeWriteStream(stream_var=param_ppb_request_stm, in_expr=one_ppb_request_var))

        # while (true) { ... } (内部退出循环)
        while_2_codes: List[HLSCodeLine] = []
        while_2_expr = HLSExpr(HLSExprT.CONST, True) # 使用 Python bool
        while_2 = CodeWhile(codes=while_2_codes, iter_expr=while_2_expr)
        if_6_codes.append(while_2)

        # --- 在内部 while(true) 循环 ---
        # ppb_response_stm.read(one_ppb_response);
        while_2_codes.append(CodeOther(text="ppb_response_stm.read(one_ppb_response);"))

        # if (one_ppb_response.last)
        if_7_codes: List[HLSCodeLine] = []
        if_7_expr = HLSExpr(HLSExprT.CONST, "one_ppb_response.last") # 使用 C++ 表达式字符串
        if_7 = CodeIf(expr=if_7_expr, if_codes=if_7_codes)
        while_2_codes.append(if_7)
        # break;
        if_7_codes.append(CodeBreak())

        # --- 结束内部 while(true) ---

        # break; (退出外部 while 循环)
        if_6_codes.append(CodeBreak())

        # --- 结束 if( exit_flag ) ---
        # --- 结束 while(true) ---


        # --- 3. 显式地将所有代码行赋值给函数 ---
        request_manager_func.codes = code_lines_req

        self.little_scatter_funcs.append(request_manager_func)
        self.little_top_dataflow_funcs.append(request_manager_func)


        set_word_in_bus_func = HLSFunction(name="set_word_in_bus", comp=comp)
        params_swib: List[HLSVar] = []

        # --- 1. 定义类型和参数 ---

        # 基本类型
        int_type = HLSType(HLSBasicType.INT)
        bus_word_t_type = HLSType(HLSBasicType.BUS_WORD_T)
        ap_fixed_pod_t_type = HLSType(HLSBasicType.AP_FIXED_POD)

        # Param 1: bus_word_t &bus_word
        # (HLSType 不需要 &，gen_code 会处理引用)
        param_bus_word = HLSVar(var_name="bus_word", var_type=bus_word_t_type)

        # Param 2: int idx
        param_idx = HLSVar(var_name="idx", var_type=int_type)

        # Param 3: ap_fixed_pod_t pod_low
        param_pod_low = HLSVar(var_name="pod_low", var_type=ap_fixed_pod_t_type)

        # Param 4: ap_fixed_pod_t pod_high
        param_pod_high = HLSVar(var_name="pod_high", var_type=ap_fixed_pod_t_type)

        params_swib.extend([param_bus_word, param_idx, param_pod_low, param_pod_high])
        set_word_in_bus_func.params = params_swib

        # --- 2. 函数体 (构建本地列表) ---
        code_lines_swib: List[HLSCodeLine] = []

        # #pragma HLS INLINE
        code_lines_swib.append(CodePragma(content="INLINE"))

        # --- switch (idx) ---
        # (按照要求，使用 CodeOther 实现整个 switch 块)
        switch_block_text = """switch (idx) {
            case 0:
                bus_word.range(31, 0) = pod_low;
                bus_word.range(63, 32) = pod_high;
                break;
            case 1:
                bus_word.range(95, 64) = pod_low;
                bus_word.range(127, 96) = pod_high;
                ;
                break;
            case 2:
                bus_word.range(159, 128) = pod_low;
                bus_word.range(191, 160) = pod_high;
                break;
            case 3:
                bus_word.range(223, 192) = pod_low;
                bus_word.range(255, 224) = pod_high;
                break;
            case 4:
                bus_word.range(287, 256) = pod_low;
                bus_word.range(319, 288) = pod_high;
                break;
            case 5:
                bus_word.range(351, 320) = pod_low;
                bus_word.range(383, 352) = pod_high;
                break;
            case 6:
                bus_word.range(415, 384) = pod_low;
                bus_word.range(447, 416) = pod_high;
                break;
            case 7:
                bus_word.range(479, 448) = pod_low;
                bus_word.range(511, 480) = pod_high;
                break;
            default:
                break;
            }"""
        code_lines_swib.append(CodeOther(text=switch_block_text))


        # --- 3. 显式地将所有代码行赋值给函数 ---
        set_word_in_bus_func.codes = code_lines_swib
        self.little_scatter_funcs.append(set_word_in_bus_func)

        Reduc_105_unit_reduce_func = HLSFunction(name="Reduc_105_unit_reduce", comp=comp)
        params_rur: List[HLSVar] = []

        # --- 1. 定义类型和参数 ---

        # 基本类型
        int_type = HLSType(HLSBasicType.INT)
        bool_type = HLSType(HLSBasicType.BOOL)
        uint8_type = HLSType(HLSBasicType.UINT8)
        node_id_t_type = HLSType(HLSBasicType.NODE_ID)
        ap_fixed_pod_t_type = HLSType(HLSBasicType.AP_FIXED_POD)
        reduce_word_t_type = HLSType(HLSBasicType.REDUCE_WORD_T)

        # 依赖: struct update_tuple_t
        # (假设已在作用域中定义或从 'graphyflow_little.h' 导入)
        node_id_array_type = HLSType(HLSBasicType.ARRAY, sub_types=[node_id_t_type], array_dims=["PE_NUM"])
        prop_array_type = HLSType(HLSBasicType.ARRAY, sub_types=[ap_fixed_pod_t_type], array_dims=["PE_NUM"])
        update_tuple_sub_types = [node_id_array_type, prop_array_type, bool_type, uint8_type]
        update_tuple_prop_names = ["node_id", "prop", "end_flag", "end_pos"]
        update_tuple_t_type = HLSType(HLSBasicType.STRUCT, 
                                      sub_types=update_tuple_sub_types, 
                                      struct_name="update_tuple_t", 
                                      struct_prop_names=update_tuple_prop_names)

        # Param 1: hls::stream<update_tuple_t> &update_set_stm
        update_set_stm_type = HLSType(HLSBasicType.STREAM, sub_types=[update_tuple_t_type])
        param_update_set_stm = HLSVar(var_name="update_set_stm", var_type=update_set_stm_type)

        # Param 2: hls::stream<reduce_word_t> (&pe_mem_outs)[PE_NUM]
        pe_mem_out_stream_type = HLSType(HLSBasicType.STREAM, sub_types=[reduce_word_t_type])
        pe_mem_outs_type = HLSType(HLSBasicType.ARRAY, sub_types=[pe_mem_out_stream_type], array_dims=["PE_NUM"])
        param_pe_mem_outs = HLSVar(var_name="pe_mem_outs", var_type=pe_mem_outs_type)

        # Param 3: int32_t edge_num
        param_edge_num = HLSVar(var_name="edge_num", var_type=int_type)

        # Param 4: int32_t dst_num
        param_dst_num = HLSVar(var_name="dst_num", var_type=int_type)

        params_rur.extend([param_update_set_stm, param_pe_mem_outs, param_edge_num, param_dst_num])
        Reduc_105_unit_reduce_func.params = params_rur

        # --- 2. 函数体 (构建本地列表) ---
        code_lines_rur: List[HLSCodeLine] = []

        code_lines_rur.append(CodeComment(text="--- Phase 1: Memory Declaration ---"))

        # const int MEM_SIZE = MAX_NUM / DISTANCES_PER_REDUCE_WORD;
        code_lines_rur.append(CodeVarDecl(var_name="MEM_SIZE", var_type=int_type, init_val="MAX_NUM / DISTANCES_PER_REDUCE_WORD", const=True))

        # reduce_word_t prop_mem[PE_NUM][MEM_SIZE];
        prop_mem_type = HLSType(HLSBasicType.ARRAY, sub_types=[reduce_word_t_type], array_dims=["PE_NUM", "MEM_SIZE"])
        code_lines_rur.append(CodeVarDecl(var_name="prop_mem", var_type=prop_mem_type))

        # Pragmas for prop_mem
        code_lines_rur.append(CodePragma(content="ARRAY_PARTITION variable = prop_mem complete dim = 1"))
        code_lines_rur.append(CodePragma(content="BIND_STORAGE variable = prop_mem type = RAM_S2P impl = URAM"))
        code_lines_rur.append(CodePragma(content="dependence variable = prop_mem inter false"))

        # reduce_word_t cache_data_buffer[PE_NUM][L + 1];
        cache_data_buffer_type = HLSType(HLSBasicType.ARRAY, sub_types=[reduce_word_t_type], array_dims=["PE_NUM", "L + 1"])
        code_lines_rur.append(CodeVarDecl(var_name="cache_data_buffer", var_type=cache_data_buffer_type))
        code_lines_rur.append(CodePragma(content="ARRAY_PARTITION variable = cache_data_buffer complete dim = 0"))

        # int32_t cache_addr_buffer[PE_NUM][L + 1];
        cache_addr_buffer_type = HLSType(HLSBasicType.ARRAY, sub_types=[int_type], array_dims=["PE_NUM", "L + 1"])
        code_lines_rur.append(CodeVarDecl(var_name="cache_addr_buffer", var_type=cache_addr_buffer_type))
        code_lines_rur.append(CodePragma(content="ARRAY_PARTITION variable = cache_addr_buffer complete dim = 0"))

        # const int32_t num_words = ...
        code_lines_rur.append(CodeVarDecl(var_name="num_words", var_type=int_type, init_val="(dst_num + DISTANCES_PER_REDUCE_WORD - 1) / DISTANCES_PER_REDUCE_WORD", const=True))
        num_words_var = HLSVar(var_name="num_words", var_type=int_type)

        #ifdef EMULATION
        code_lines_rur.append(CodeOther(text="#ifdef EMULATION"))
        code_lines_rur.append(CodeOther(text="    memset(prop_mem, 0, sizeof(reduce_word_t) * PE_NUM * MEM_SIZE);"))
        code_lines_rur.append(CodeOther(text="#endif"))

        # LOOP_INIT_CACHE_ADDR: for (int i = 0; ...
        for_1_codes: List[HLSCodeLine] = []
        for_1 = CodeFor(codes=for_1_codes, iter_limit="L + 1", iter_cmp="<", iter_name="i", iter_start="0", iter_step="i++", iter_val_type=int_type)

        # --- Inside for(i) ---
        for_1_codes.append(CodePragma(content="UNROLL"))
        # for (int pe = 0; ...
        for_2_codes: List[HLSCodeLine] = []
        for_2 = CodeFor(codes=for_2_codes, iter_limit="PE_NUM", iter_cmp="<", iter_name="pe", iter_start="0", iter_step="pe++", iter_val_type=int_type)

        # --- Inside for(pe) ---
        for_2_codes.append(CodePragma(content="UNROLL"))
        # cache_addr_buffer[pe][i] = -1;
        for_2_codes.append(CodeOther(text="cache_addr_buffer[pe][i] = -1;"))

        # Assemble init loops
        for_1_codes.append(for_2)
        code_lines_rur.append(CodeOther(text="LOOP_INIT_CACHE_ADDR:"))
        code_lines_rur.append(for_1)

        # const int32_t total_updates = ...
        code_lines_rur.append(CodeVarDecl(var_name="total_updates", var_type=int_type, init_val="(edge_num + PE_NUM - 1) / PE_NUM", const=True))
        total_updates_var = HLSVar(var_name="total_updates", var_type=int_type)

        # const int32_t last_pack_size = ...
        code_lines_rur.append(CodeVarDecl(var_name="last_pack_size", var_type=int_type, init_val="(edge_num % PE_NUM == 0) ? PE_NUM : (edge_num % PE_NUM)", const=True))
        last_pack_size_var = HLSVar(var_name="last_pack_size", var_type=int_type)

        code_lines_rur.append(CodeComment(text="--- Phase 3: Aggregation Loop ---"))

        # LOOP_AGGREGATE: for (int update_idx = 0; ...
        for_3_codes: List[HLSCodeLine] = []
        for_3 = CodeFor(codes=for_3_codes, iter_limit=total_updates_var, iter_cmp="<", iter_name="update_idx", iter_start="0", iter_step="update_idx++", iter_val_type=int_type)

        # --- Inside for(update_idx) ---
        for_3_codes.append(CodePragma(content="PIPELINE II = 1"))

        # update_tuple_t one_update;
        for_3_codes.append(CodeVarDecl(var_name="one_update", var_type=update_tuple_t_type))
        one_update_var = HLSVar(var_name="one_update", var_type=update_tuple_t_type)
        for_3_codes.append(CodePragma(content="ARRAY_PARTITION variable = one_update.prop complete dim = 0"))
        for_3_codes.append(CodePragma(content="ARRAY_PARTITION variable = one_update.node_id complete dim = 0"))

        # one_update = update_set_stm.read();
        read_expr = HLSExpr(HLSExprT.STREAM_READ, expr_val=None, operands=[HLSExpr(HLSExprT.VAR, param_update_set_stm)])
        for_3_codes.append(CodeAssign(var=one_update_var, expr=read_expr))

        # int32_t cur_pe_end = ...
        for_3_codes.append(CodeVarDecl(var_name="cur_pe_end", var_type=int_type, init_val="(update_idx == total_updates - 1) ? last_pack_size : PE_NUM"))
        cur_pe_end_var = HLSVar(var_name="cur_pe_end", var_type=int_type)

        # for (int pe = 0; ...
        for_4_codes: List[HLSCodeLine] = []
        for_4 = CodeFor(codes=for_4_codes, iter_limit="PE_NUM", iter_cmp="<", iter_name="pe", iter_start="0", iter_step="pe++", iter_val_type=int_type)

        # --- Inside for(pe) ---
        for_4_codes.append(CodePragma(content="UNROLL"))

        # int32_t key = one_update.node_id[pe];
        for_4_codes.append(CodeVarDecl(var_name="key", var_type=int_type, init_val="one_update.node_id[pe]"))

        # if (pe < cur_pe_end && (key & 0x40000000) == 0) { ... }
        if_1_codes: List[HLSCodeLine] = []
        if_1_expr = HLSExpr(HLSExprT.CONST, "(pe < cur_pe_end && (key & 0x40000000) == 0)")
        if_1 = CodeIf(expr=if_1_expr, if_codes=if_1_codes)

        # --- Inside if(pe < cur_pe_end...) ---
        # ap_fixed_pod_t incoming_dist_pod = one_update.prop[pe];
        if_1_codes.append(CodeVarDecl(var_name="incoming_dist_pod", var_type=ap_fixed_pod_t_type, init_val="one_update.prop[pe]"))

        # int32_t word_addr = (key >> 1);
        if_1_codes.append(CodeVarDecl(var_name="word_addr", var_type=int_type, init_val="(key >> 1)"))
        word_addr_var = HLSVar(var_name="word_addr", var_type=int_type)

        # int32_t pack_idx = (key & 1);
        if_1_codes.append(CodeVarDecl(var_name="pack_idx", var_type=int_type, init_val="(key & 1)"))

        # reduce_word_t current_word = prop_mem[pe][word_addr];
        if_1_codes.append(CodeOther(text="reduce_word_t current_word = prop_mem[pe][word_addr];"))
        current_word_var = HLSVar(var_name="current_word", var_type=reduce_word_t_type)

        # for (int i = L; i >= 0; --i) { ... } (Cache check)
        for_5_codes: List[HLSCodeLine] = []
        for_5 = CodeFor(codes=for_5_codes, iter_limit="0", iter_cmp=">=", iter_name="i", iter_start="L", iter_step="--i", iter_val_type=int_type)
        for_5_codes.append(CodePragma(content="UNROLL"))

        # if (cache_addr_buffer[pe][i] == word_addr)
        if_2_codes: List[HLSCodeLine] = []
        if_2_expr = HLSExpr(HLSExprT.CONST, "cache_addr_buffer[pe][i] == word_addr")
        if_2 = CodeIf(expr=if_2_expr, if_codes=if_2_codes)
        # current_word = cache_data_buffer[pe][i];
        if_2_codes.append(CodeOther(text="current_word = cache_data_buffer[pe][i];"))
        # break;
        if_2_codes.append(CodeBreak())
        for_5_codes.append(if_2)
        if_1_codes.append(for_5)

        # for (int i = 0; i < L; i++) { ... } (Cache shift)
        for_6_codes: List[HLSCodeLine] = []
        for_6 = CodeFor(codes=for_6_codes, iter_limit="L", iter_cmp="<", iter_name="i", iter_start="0", iter_step="i++", iter_val_type=int_type)
        for_6_codes.append(CodePragma(content="UNROLL"))
        # cache_addr_buffer[pe][i] = cache_addr_buffer[pe][i + 1];
        for_6_codes.append(CodeOther(text="cache_addr_buffer[pe][i] = cache_addr_buffer[pe][i + 1];"))
        # cache_data_buffer[pe][i] = cache_data_buffer[pe][i + 1];
        for_6_codes.append(CodeOther(text="cache_data_buffer[pe][i] = cache_data_buffer[pe][i + 1];"))
        if_1_codes.append(for_6)

        # ap_fixed_pod_t old_dist_pod = get_raw_val(current_word, pack_idx);
        if_1_codes.append(CodeVarDecl(var_name="old_dist_pod", var_type=ap_fixed_pod_t_type, init_val="get_raw_val(current_word, pack_idx)"))

        # ap_fixed_pod_t new_dist_pod = ...
        if_1_codes.append(CodeVarDecl(var_name="new_dist_pod", var_type=ap_fixed_pod_t_type, init_val="(old_dist_pod < incoming_dist_pod && old_dist_pod != 0x0) ? old_dist_pod : incoming_dist_pod"))

        # set_raw_val(current_word, pack_idx, new_dist_pod);
        if_1_codes.append(CodeOther(text="set_raw_val(current_word, pack_idx, new_dist_pod);"))

        # prop_mem[pe][word_addr] = current_word;
        if_1_codes.append(CodeOther(text="prop_mem[pe][word_addr] = current_word;"))

        # cache_addr_buffer[pe][L] = word_addr;
        if_1_codes.append(CodeOther(text="cache_addr_buffer[pe][L] = word_addr;"))

        # cache_data_buffer[pe][L] = current_word;
        if_1_codes.append(CodeOther(text="cache_data_buffer[pe][L] = current_word;"))

        # Assemble aggregate loops
        for_4_codes.append(if_1)
        for_3_codes.append(for_4)
        code_lines_rur.append(CodeOther(text="LOOP_AGGREGATE:"))
        code_lines_rur.append(for_3)


        code_lines_rur.append(CodeComment(text="--- Phase 4: Stream out aggregated memory ---"))

        # LOOP_STREAM_OUT: for (int i = 0; i < num_words; i++) {
        for_7_codes: List[HLSCodeLine] = []
        for_7 = CodeFor(codes=for_7_codes, iter_limit=num_words_var, iter_cmp="<", iter_name="i", iter_start="0", iter_step="i++", iter_val_type=int_type)

        # --- Inside for(i) ---
        for_7_codes.append(CodePragma(content="PIPELINE II = 1"))

        # for (int pe = 0; pe < PE_NUM; pe++) {
        for_8_codes: List[HLSCodeLine] = []
        for_8 = CodeFor(codes=for_8_codes, iter_limit="PE_NUM", iter_cmp="<", iter_name="pe", iter_start="0", iter_step="pe++", iter_val_type=int_type)

        # --- Inside for(pe) ---
        for_8_codes.append(CodePragma(content="UNROLL"))

        # reduce_word_t word = prop_mem[pe][i];
        for_8_codes.append(CodeVarDecl(var_name="word", var_type=reduce_word_t_type, init_val="prop_mem[pe][i]"))
        word_var = HLSVar(var_name="word", var_type=reduce_word_t_type)

        # pe_mem_outs[pe].write(word);
        pe_mem_out_pe_var = HLSVar(var_name="pe_mem_outs[pe]", var_type=pe_mem_out_stream_type)
        for_8_codes.append(CodeWriteStream(stream_var=pe_mem_out_pe_var, in_expr=word_var))

        # prop_mem[pe][i] = 0;
        for_8_codes.append(CodeOther(text="prop_mem[pe][i] = 0;"))

        # Assemble stream out loops
        for_7_codes.append(for_8)
        code_lines_rur.append(CodeOther(text="LOOP_STREAM_OUT:"))
        code_lines_rur.append(for_7)


        # --- 3. 显式地将所有代码行赋值给函数 ---
        Reduc_105_unit_reduce_func.codes = code_lines_rur
        self.little_scatter_funcs.append(Reduc_105_unit_reduce_func)

        Reduc_105_drain_multi_pe_func = HLSFunction(name="Reduc_105_drain_multi_pe", comp=comp)
        params_rur_drain: List[HLSVar] = []

        # --- 1. 定义类型和参数 ---

        # (复用之前定义的基本类型)
        # int_type = HLSType(HLSBasicType.INT)
        # uint_type = HLSType(HLSBasicType.UINT)
        # reduce_word_t_type = HLSType(HLSBasicType.REDUCE_WORD_T)
        # ap_fixed_pod_t_type = HLSType(HLSBasicType.AP_FIXED_POD)

        # 新增类型
        write_burst_pkt_t_type = HLSType(HLSBasicType.WRITE_BURST_PKT_T)
        distance_t_type = HLSType(HLSBasicType.DISTANCE_T)
        uint_type = HLSType(HLSBasicType.UINT) # for uint32_t

        # Param 1: hls::stream<reduce_word_t> (&pe_mem_in)[PE_NUM]
        pe_mem_in_stream_type = HLSType(HLSBasicType.STREAM, sub_types=[reduce_word_t_type])
        pe_mem_in_type = HLSType(HLSBasicType.ARRAY, sub_types=[pe_mem_in_stream_type], array_dims=["PE_NUM"])
        param_pe_mem_in = HLSVar(var_name="pe_mem_in", var_type=pe_mem_in_type)

        # Param 2: hls::stream<write_burst_pkt_t> &kernel_out_stream
        kernel_out_stream_type = HLSType(HLSBasicType.STREAM, sub_types=[write_burst_pkt_t_type])
        param_kernel_out_stream = HLSVar(var_name="kernel_out_stream", var_type=kernel_out_stream_type)

        # Param 3: int32_t dst_num
        # (param_dst_num 已在上面定义)
        # param_dst_num = HLSVar(var_name="dst_num", var_type=int_type)

        params_rur_drain.extend([param_pe_mem_in, param_kernel_out_stream, param_dst_num])
        Reduc_105_drain_multi_pe_func.params = params_rur_drain

        # --- 2. 函数体 (构建本地列表) ---
        code_lines_rur_drain: List[HLSCodeLine] = []

        code_lines_rur_drain.append(CodeComment(text="--- Phase 2: High-Performance Drain Loop ---"))

        # write_burst_pkt_t one_write_burst;
        code_lines_rur_drain.append(CodeVarDecl(var_name="one_write_burst", var_type=write_burst_pkt_t_type))
        one_write_burst_var = HLSVar(var_name="one_write_burst", var_type=write_burst_pkt_t_type)
        one_write_burst_data_var = HLSVar(var_name="one_write_burst.data", var_type=HLSType(HLSBasicType.BUS_WORD_T)) # for set_word_in_bus

        # one_write_burst.last = 0;
        one_write_burst_last_var = HLSVar(var_name="one_write_burst.last", var_type=HLSType(HLSBasicType.AP_UINT, width=1))
        code_lines_rur_drain.append(CodeAssign(var=one_write_burst_last_var, expr=HLSExpr(HLSExprT.CONST, 0)))

        # uint32_t waiting_count = 0;
        code_lines_rur_drain.append(CodeVarDecl(var_name="waiting_count", var_type=uint_type, init_val="0"))
        waiting_count_var = HLSVar(var_name="waiting_count", var_type=uint_type)

        # distance_t max_val = (distance_t)(16384.0);
        code_lines_rur_drain.append(CodeVarDecl(var_name="max_val", var_type=distance_t_type, init_val="(distance_t)(16384.0)"))
        max_val_var = HLSVar(var_name="max_val", var_type=distance_t_type)

        # ap_fixed_pod_t max_pod = *reinterpret_cast<ap_fixed_pod_t *>(&max_val);
        code_lines_rur_drain.append(CodeVarDecl(var_name="max_pod", var_type=ap_fixed_pod_t_type, init_val="*reinterpret_cast<ap_fixed_pod_t *>(&max_val)"))
        max_pod_var = HLSVar(var_name="max_pod", var_type=ap_fixed_pod_t_type)

        # LOOP_DRAIN_ADDR: for (int32_t base_addr = 0; ...
        for_loop_drain_addr_codes: List[HLSCodeLine] = []
        for_loop_drain_addr = CodeFor(codes=for_loop_drain_addr_codes,
                                        iter_limit=param_dst_num, # HLSVar from params
                                        iter_cmp="<",
                                        iter_name="base_addr",
                                        iter_start="0",
                                        iter_step="base_addr += DISTANCES_PER_REDUCE_WORD",
                                        iter_val_type=int_type)

        # --- Inside for(base_addr) ---
        for_loop_drain_addr_codes.append(CodePragma(content="PIPELINE II = 1"))

        # ap_fixed_pod_t uram_res_low = max_pod;
        for_loop_drain_addr_codes.append(CodeVarDecl(var_name="uram_res_low", var_type=ap_fixed_pod_t_type, init_val="max_pod"))
        uram_res_low_var = HLSVar(var_name="uram_res_low", var_type=ap_fixed_pod_t_type)

        # ap_fixed_pod_t uram_res_high = max_pod;
        for_loop_drain_addr_codes.append(CodeVarDecl(var_name="uram_res_high", var_type=ap_fixed_pod_t_type, init_val="max_pod"))
        uram_res_high_var = HLSVar(var_name="uram_res_high", var_type=ap_fixed_pod_t_type)

        # LOOP_FOR_57: for (uint32_t pe_idx = 0; ...
        for_loop_57_codes: List[HLSCodeLine] = []
        for_loop_57 = CodeFor(codes=for_loop_57_codes,
                                iter_limit="PE_NUM",
                                iter_cmp="<",
                                iter_name="pe_idx",
                                iter_start="0",
                                iter_step="pe_idx++",
                                iter_val_type=uint_type)

        # --- Inside for(pe_idx) ---
        for_loop_57_codes.append(CodePragma(content="UNROLL"))

        # reduce_word_t word = pe_mem_in[pe_idx].read();
        for_loop_57_codes.append(CodeVarDecl(var_name="word", var_type=reduce_word_t_type, init_val="pe_mem_in[pe_idx].read()"))

        # ap_fixed_pod_t incoming_dist_pod_low = word.range(31, 0);
        for_loop_57_codes.append(CodeVarDecl(var_name="incoming_dist_pod_low", var_type=ap_fixed_pod_t_type, init_val="word.range(31, 0)"))

        # ap_fixed_pod_t incoming_dist_pod_high = word.range(63, 32);
        for_loop_57_codes.append(CodeVarDecl(var_name="incoming_dist_pod_high", var_type=ap_fixed_pod_t_type, init_val="word.range(63, 32)"))

        # uram_res_low = ... (ternary op)
        # 使用 CodeOther 处理复杂的原地更新
        for_loop_57_codes.append(CodeOther(text="uram_res_low = (uram_res_low < incoming_dist_pod_low || incoming_dist_pod_low == 0x0) ? uram_res_low : incoming_dist_pod_low;"))

        # uram_res_high = ... (ternary op)
        for_loop_57_codes.append(CodeOther(text="uram_res_high = (uram_res_high < incoming_dist_pod_high || incoming_dist_pod_high == 0x0) ? uram_res_high : incoming_dist_pod_high;"))

        # Assemble for(pe_idx)
        for_loop_drain_addr_codes.append(CodeOther(text="LOOP_FOR_57:"))
        for_loop_drain_addr_codes.append(for_loop_57)

        # set_word_in_bus(...)
        # (假设 set_word_in_bus_func 已在作用域中定义)
        call_params_swib = [one_write_burst_data_var, waiting_count_var, uram_res_low_var, uram_res_high_var]
        # 假设 set_word_in_bus_func 是一个 HLSFunction 实例
        # set_word_in_bus_func = HLSFunction(name="set_word_in_bus", comp=comp) # 占位符
        # for_loop_drain_addr_codes.append(CodeCall(func=set_word_in_bus_func, params=call_params_swib))
        # 鉴于 set_word_in_bus_func 可能未定义，先使用 CodeOther
        for_loop_drain_addr_codes.append(CodeOther(text="set_word_in_bus(one_write_burst.data, waiting_count, uram_res_low, uram_res_high);"))


        # waiting_count++;
        for_loop_drain_addr_codes.append(CodeOther(text="waiting_count++;"))

        # if (waiting_count == 8) { ... }
        if_wc8_codes: List[HLSCodeLine] = []
        if_wc8_expr = HLSExpr(HLSExprT.CONST, "waiting_count == 8")
        if_wc8 = CodeIf(expr=if_wc8_expr, if_codes=if_wc8_codes)

        # waiting_count = 0;
        if_wc8_codes.append(CodeAssign(var=waiting_count_var, expr=HLSExpr(HLSExprT.CONST, 0)))
        # kernel_out_stream.write(one_write_burst);
        if_wc8_codes.append(CodeWriteStream(stream_var=param_kernel_out_stream, in_expr=one_write_burst_var))

        for_loop_drain_addr_codes.append(if_wc8)

        # Assemble for(base_addr)
        code_lines_rur_drain.append(CodeOther(text="LOOP_DRAIN_ADDR:"))
        code_lines_rur_drain.append(for_loop_drain_addr)

        # if (waiting_count != 0) { ... }
        if_wc_not0_codes: List[HLSCodeLine] = []
        if_wc_not0_expr = HLSExpr(HLSExprT.CONST, "waiting_count != 0")
        if_wc_not0 = CodeIf(expr=if_wc_not0_expr, if_codes=if_wc_not0_codes)

        # kernel_out_stream.write(one_write_burst);
        if_wc_not0_codes.append(CodeWriteStream(stream_var=param_kernel_out_stream, in_expr=one_write_burst_var))

        code_lines_rur_drain.append(if_wc_not0)

        # --- 3. 显式地将所有代码行赋值给函数 ---
        Reduc_105_drain_multi_pe_func.codes = code_lines_rur_drain
        self.little_scatter_funcs.append(Reduc_105_drain_multi_pe_func)

        


    def process_gather(self,gather_stage_comps : List[dfir.Component]):
        assert len(gather_stage_comps) == 1
        reduce_comp = gather_stage_comps[0]
        print("========= Gather Stage =========")
        self._translate_reduce_op(reduce_comp)
            
    def process_apply(self,apply_stage_comps : List[dfir.Component]):

        """
        # translate funstions
        apply_kernel_inter_func = HLSFunction(name="apply_kernel_inter", comp=None)
        params = []

        # --- 1. Define Types & Params ---

        # Basic Types
        uint_type = HLSType(HLSBasicType.UINT)
        int_type = HLSType(HLSBasicType.INT)
        bus_word_t_type = HLSType(HLSBasicType.BUS_WORD_T)
        ap_fixed_pod_t_type = HLSType(HLSBasicType.AP_FIXED_POD)
        bool_type = HLSType(HLSBasicType.BOOL) # For .last field

        # Param 1: uint32_t dst_num
        dst_num_var = HLSVar(var_name="dst_num", var_type=uint_type)

        # Param 2: hls::stream<cacheline_data_pkt_t> &cacheline_data_stream
        cacheline_data_pkt_t_type = HLSType(HLSBasicType.CACHELINE_DATA_PKT_T)
        cacheline_data_stream_type = HLSType(HLSBasicType.STREAM, sub_types=[cacheline_data_pkt_t_type])
        cacheline_data_stream = HLSVar(var_name="cacheline_data_stream", var_type=cacheline_data_stream_type)

        # Param 3: hls::stream<write_burst_pkt_t> &node_distance_burst_stream
        write_burst_pkt_t_type = HLSType(HLSBasicType.WRITE_BURST_PKT_T)
        write_burst_stream_type = HLSType(HLSBasicType.STREAM, sub_types=[write_burst_pkt_t_type])
        node_distance_burst_stream = HLSVar(var_name="node_distance_burst_stream", var_type=write_burst_stream_type)

        # Param 4: hls::stream<write_burst_pkt_t> &write_burst_stream
        write_burst_stream = HLSVar(var_name="write_burst_stream", var_type=write_burst_stream_type)

        params.extend([dst_num_var, cacheline_data_stream, node_distance_burst_stream, write_burst_stream])
        apply_kernel_inter_func.params = params

        # --- 2. Function Body ---
        code_lines: List[HLSCodeLine] = []

        # uint32_t write_idx = 0;
        code_lines.append(CodeVarDecl(var_name="write_idx", var_type=uint_type, init_val="0"))
        write_idx_var = HLSVar(var_name="write_idx", var_type=uint_type)

        # for (uint32_t addr = 0; addr < dst_num; addr += (PE_NUM << 1)) {
        for_loop_1_codes: List[HLSCodeLine] = []
        for_loop_1 = CodeFor(codes=for_loop_1_codes,
                             iter_limit=dst_num_var,
                             iter_cmp="<",
                             iter_name="addr",
                             iter_start="0",
                             iter_step="addr += (PE_NUM << 1)",
                             iter_val_type=uint_type)
        code_lines.append(for_loop_1)

        # --- Inside for(addr) ---
        # #pragma HLS PIPELINE II = 1
        for_loop_1_codes.append(CodePragma(content="PIPELINE II = 1"))

        # write_burst_pkt_t pkt = node_distance_burst_stream.read();
        for_loop_1_codes.append(CodeVarDecl(var_name="pkt", var_type=write_burst_pkt_t_type, init_val="node_distance_burst_stream.read()"))

        # cacheline_data_pkt_t cache_pkt = cacheline_data_stream.read();
        for_loop_1_codes.append(CodeVarDecl(var_name="cache_pkt", var_type=cacheline_data_pkt_t_type, init_val="cacheline_data_stream.read()"))

        # bus_word_t wide_word = pkt.data;
        for_loop_1_codes.append(CodeVarDecl(var_name="wide_word", var_type=bus_word_t_type, init_val="pkt.data"))
        wide_word_var = HLSVar(var_name="wide_word", var_type=bus_word_t_type)

        # bus_word_t node_prop = cache_pkt.data;
        for_loop_1_codes.append(CodeVarDecl(var_name="node_prop", var_type=bus_word_t_type, init_val="cache_pkt.data"))
        node_prop_var = HLSVar(var_name="node_prop", var_type=bus_word_t_type)

        # bus_word_t new_node_prop;
        for_loop_1_codes.append(CodeVarDecl(var_name="new_node_prop", var_type=bus_word_t_type))
        new_node_prop_var = HLSVar(var_name="new_node_prop", var_type=bus_word_t_type)

        # for (int i = 0; i < DBL_PE_NUM; i++) {
        for_loop_2_codes: List[HLSCodeLine] = []
        for_loop_2 = CodeFor(codes=for_loop_2_codes,
                             iter_limit="DBL_PE_NUM",
                             iter_cmp="<",
                             iter_name="i",
                             iter_start="0",
                             iter_step="i++",
                             iter_val_type=int_type)
        for_loop_1_codes.append(for_loop_2)

        # --- Inside for(i) ---
        # #pragma HLS UNROLL
        for_loop_2_codes.append(CodePragma(content="UNROLL"))


        
        # ap_fixed_pod_t update_dist = wide_word.range(31 + (i << 5), (i << 5));
        init_val_1 = "wide_word.range(31 + (i << 5), (i << 5))"
        for_loop_2_codes.append(CodeVarDecl(var_name="update_dist", var_type=ap_fixed_pod_t_type, init_val=init_val_1))
        init_val_1_var = HLSVar(var_name="update_dist", var_type=ap_fixed_pod_t_type)

        # ap_fixed_pod_t current_dist = node_prop.range(31 + (i << 5), (i << 5));
        init_val_2 = "node_prop.range(31 + (i << 5), (i << 5))"
        for_loop_2_codes.append(CodeVarDecl(var_name="current_dist", var_type=ap_fixed_pod_t_type, init_val=init_val_2))
        init_val_2_var = HLSVar(var_name="current_dist", var_type=ap_fixed_pod_t_type)


        # ap_fixed_pod_t new_dist = (update_dist < current_dist) ? update_dist : current_dist;
        # init_val_3 = "(update_dist < current_dist) ? update_dist : current_dist"
        for_loop_2_codes.append(CodeVarDecl(var_name="new_dist", var_type=ap_fixed_pod_t_type))
        result_val3 = HLSVar(var_name="new_dist", var_type=ap_fixed_pod_t_type)

        # new_node_prop.range(31 + (i << 5), (i << 5)) = new_dist;
        # (Using CodeOther for LHS .range())
        
        # } (end for_loop_2)
        # ================ begin inline fused op =============================
        for_loop_2_codes.append(CodeOther(text="// Begin inline fused op"))
        inline_codes = []
        port_property = {}
        port_to_var = {}
        top_vars = {
            "init_val_1_var": init_val_1_var,
            "init_val_2_var": init_val_2_var,
            "result_val3": result_val3
        }
        for comp in apply_stage_comps:
            port_property,inline_codes = self._apply_analyze(comp,port_property,port_to_var,top_vars,target_codes=inline_codes)
        for_loop_2_codes.extend(inline_codes)
        for_loop_2_codes.append(CodeOther(text="// End inline fused op"))
        
            
        # ========================= end inline fused op ===========================
        for_loop_2_codes.append(CodeOther(text="new_node_prop.range(31 + (i << 5), (i << 5)) = new_dist;"))
        
        
        # write_burst_pkt_t out_pkt;
        for_loop_1_codes.append(CodeVarDecl(var_name="out_pkt", var_type=write_burst_pkt_t_type))
        out_pkt_var = HLSVar(var_name="out_pkt", var_type=write_burst_pkt_t_type)

        # out_pkt.data = new_node_prop;
        # (Assuming out_pkt.data is compatible with bus_word_t)
        out_pkt_data_var = HLSVar(var_name="out_pkt.data", var_type=bus_word_t_type)
        for_loop_1_codes.append(CodeAssign(var=out_pkt_data_var, expr=new_node_prop_var))

        # out_pkt.last = false;
        # (Assuming out_pkt.last is compatible with bool)
        out_pkt_last_var = HLSVar(var_name="out_pkt.last", var_type=bool_type)
        assign_expr_last = HLSExpr(HLSExprT.CONST, False)
        for_loop_1_codes.append(CodeAssign(var=out_pkt_last_var, expr=assign_expr_last))

        # write_burst_stream.write(out_pkt);
        for_loop_1_codes.append(CodeWriteStream(stream_var=write_burst_stream, in_expr=out_pkt_var))

        # write_idx++;
        # (Representing as write_idx = write_idx + 1)
        assign_expr_inc = HLSExpr(HLSExprT.BINOP, dfir.BinOp.ADD, operands=[
            HLSExpr(HLSExprT.VAR, write_idx_var),
            HLSExpr(HLSExprT.CONST, 1)
        ])
        for_loop_1_codes.append(CodeAssign(var=write_idx_var, expr=assign_expr_inc))
        # } (end for_loop_1)

        # --- 3. Finalize ---
        apply_kernel_inter_func.codes = code_lines

        """
        apply_kernel_inter_func = HLSFunction(name="apply_kernel_inter", comp=None)
        params = []

        # --- 1. 定义类型和参数 ---

        # 基本类型
        uint_type = HLSType(HLSBasicType.UINT)
        bool_type = HLSType(HLSBasicType.BOOL)
        int_type = HLSType(HLSBasicType.INT) # 用于 for 循环的 'i'

        # Typedefs (来自 HLSBasicType)
        bus_word_t_type = HLSType(HLSBasicType.BUS_WORD_T)
        write_burst_pkt_t_type = HLSType(HLSBasicType.WRITE_BURST_PKT_T)
        ap_fixed_pod_t_type = HLSType(HLSBasicType.AP_FIXED_POD)

        # Param 1: bus_word_t *node_props
        node_props_type = HLSType(HLSBasicType.POINTER, sub_types=[bus_word_t_type])
        node_props = HLSVar(var_name="node_props", var_type=node_props_type)

        # Param 2: uint32_t dst_num
        dst_num = HLSVar(var_name="dst_num", var_type=uint_type)

        # Param 3: hls::stream<write_burst_pkt_t> &node_distance_burst_stream
        node_distance_burst_stream_type = HLSType(HLSBasicType.STREAM, sub_types=[write_burst_pkt_t_type])
        node_distance_burst_stream = HLSVar(var_name="node_distance_burst_stream", var_type=node_distance_burst_stream_type)

        # Param 4: hls::stream<write_burst_pkt_t> &write_burst_stream
        write_burst_stream_type = HLSType(HLSBasicType.STREAM, sub_types=[write_burst_pkt_t_type])
        write_burst_stream = HLSVar(var_name="write_burst_stream", var_type=write_burst_stream_type)

        params.extend([node_props, dst_num, node_distance_burst_stream, write_burst_stream])
        apply_kernel_inter_func.params = params

        # --- 2. 函数体 ---
        code_lines: List[HLSCodeLine] = []
        apply_kernel_inter_func.codes = code_lines

        # uint32_t read_idx = 0;
        code_lines.append(CodeVarDecl(var_name="read_idx", var_type=uint_type, init_val="0"))
        read_idx_var = HLSVar(var_name="read_idx", var_type=uint_type)

        # uint32_t addr = 0;
        code_lines.append(CodeVarDecl(var_name="addr", var_type=uint_type, init_val="0"))
        addr_var = HLSVar(var_name="addr", var_type=uint_type)

        # bool pkt_ready = false;
        code_lines.append(CodeVarDecl(var_name="pkt_ready", var_type=bool_type, init_val="false"))
        pkt_ready_var = HLSVar(var_name="pkt_ready", var_type=bool_type)

        # write_burst_pkt_t pkt;
        code_lines.append(CodeVarDecl(var_name="pkt", var_type=write_burst_pkt_t_type))
        pkt_var = HLSVar(var_name="pkt", var_type=write_burst_pkt_t_type)

        # while (true)
        while_loop_codes: List[HLSCodeLine] = []
        while_expr = HLSExpr(HLSExprT.CONST, "true")
        while_loop = CodeWhile(codes=while_loop_codes, iter_expr=while_expr)
        code_lines.append(while_loop)

        # --- 在 while(true) 循环内部 ---
        # #pragma HLS PIPELINE II = 1
        while_loop_codes.append(CodePragma(content="PIPELINE II = 1"))

        # if (!pkt_ready) { ... }
        if_1_codes: List[HLSCodeLine] = []
        not_pkt_ready_expr = HLSExpr(HLSExprT.UOP, dfir.UnaryOp.NOT, operands=[HLSExpr(HLSExprT.VAR, pkt_ready_var)])
        if_1 = CodeIf(expr=not_pkt_ready_expr, if_codes=if_1_codes)
        while_loop_codes.append(if_1)

        # pkt_ready = node_distance_burst_stream.read_nb(pkt);
        # read_nb 是一个非阻塞读，返回bool并修改参数。
        # 这不适合 CodeAssign 或 CodeWriteStream。使用 CodeOther。
        if_1_codes.append(CodeOther(text="pkt_ready = node_distance_burst_stream.read_nb(pkt);"))

        # if (pkt_ready) { ... }
        if_2_codes: List[HLSCodeLine] = []
        pkt_ready_expr = HLSExpr(HLSExprT.VAR, pkt_ready_var)
        if_2 = CodeIf(expr=pkt_ready_expr, if_codes=if_2_codes)
        while_loop_codes.append(if_2)

        # --- 在 if(pkt_ready) 内部 ---

        # pkt_ready = false;
        assign_pkt_ready_false = CodeAssign(var=pkt_ready_var, expr=HLSExpr(HLSExprT.CONST, "false"))
        if_2_codes.append(assign_pkt_ready_false)

        # bus_word_t wide_word = pkt.data;
        # (假设 pkt.data 是 bus_word_t 类型)
        if_2_codes.append(CodeVarDecl(var_name="wide_word", var_type=bus_word_t_type, init_val="pkt.data"))
        wide_word_var = HLSVar(var_name="wide_word", var_type=bus_word_t_type)

        # bus_word_t node_prop = node_props[read_idx];
        # (指针/数组访问作为 init_val 字符串)
        if_2_codes.append(CodeVarDecl(var_name="node_prop", var_type=bus_word_t_type, init_val="node_props[read_idx]"))
        node_prop_var = HLSVar(var_name="node_prop", var_type=bus_word_t_type)

        # bus_word_t new_node_prop;
        if_2_codes.append(CodeVarDecl(var_name="new_node_prop", var_type=bus_word_t_type))
        new_node_prop_var = HLSVar(var_name="new_node_prop", var_type=bus_word_t_type)

        # for (int i = 0; i < DBL_PE_NUM; i++) { ... }
        for_loop_1_codes: List[HLSCodeLine] = []
        for_loop_1 = CodeFor(codes=for_loop_1_codes,
                               iter_limit="DBL_PE_NUM", # 假设 DBL_PE_NUM 是一个宏
                               iter_cmp="<",
                               iter_name="i",
                               iter_start="0",
                               iter_step="i++",
                               iter_val_type=int_type)
        if_2_codes.append(for_loop_1)

        # --- 在 for(i) 循环内部 ---
        # #pragma HLS UNROLL
        for_loop_1_codes.append(CodePragma(content="UNROLL"))

        # ap_fixed_pod_t update_dist = wide_word.range(31 + (i << 5), (i << 5));
        # (使用 init_val 字符串处理 .range() 初始化)
        for_loop_1_codes.append(CodeVarDecl(var_name="update_dist", var_type=ap_fixed_pod_t_type, init_val="wide_word.range(31 + (i << 5), (i << 5))"))
        update_dist_var = HLSVar(var_name="update_dist", var_type=ap_fixed_pod_t_type)

        # ap_fixed_pod_t current_dist = node_prop.range(31 + (i << 5), (i << 5));
        for_loop_1_codes.append(CodeVarDecl(var_name="current_dist", var_type=ap_fixed_pod_t_type, init_val="node_prop.range(31 + (i << 5), (i << 5))"))
        current_dist_var = HLSVar(var_name="current_dist", var_type=ap_fixed_pod_t_type)

        result_var3 = HLSVar(var_name="new_dist", var_type=ap_fixed_pod_t_type)
        for_loop_1_codes.append(CodeVarDecl(var_name="new_dist", var_type=ap_fixed_pod_t_type))
        # ================ begin inline fused op =============================
        for_loop_1_codes.append(CodeOther(text="// Begin inline fused op"))
        inline_codes = []
        port_property = {}
        port_to_var = {}
        top_vars = {
            "init_val_1_var": update_dist_var,
            "init_val_2_var": current_dist_var,
            "result_val3": result_var3
        }
        for comp in apply_stage_comps:
            port_property,inline_codes = self._apply_analyze(comp,port_property,port_to_var,top_vars,target_codes=inline_codes)
        for_loop_1_codes.extend(inline_codes)
        for_loop_1_codes.append(CodeOther(text="// End inline fused op"))
        
            
        # ========================= end inline fused op ===========================

        # ap_fixed_pod_t new_dist = (update_dist < current_dist) ? update_dist : current_dist;
        # (三元运算符作为 init_val 字符串)
        
        # new_dist_var = HLSVar(var_name="new_dist", var_type=ap_fixed_pod_t_type)

        # (跳过注释)

        # new_node_prop.range(31 + (i << 5), (i << 5)) = new_dist;
        # (对 .range() 的赋值使用 CodeOther，遵循示例)
        for_loop_1_codes.append(CodeOther(text=f"new_node_prop.range(31 + (i << 5), (i << 5)) = {result_var3.name};"))

        # --- 结束 for(i) 循环 ---

        # write_burst_pkt_t out_pkt;
        if_2_codes.append(CodeVarDecl(var_name="out_pkt", var_type=write_burst_pkt_t_type))
        out_pkt_var = HLSVar(var_name="out_pkt", var_type=write_burst_pkt_t_type)

        # out_pkt.data = new_node_prop;
        # (假设 out_pkt.data 也是 bus_word_t 类型)
        out_pkt_data_var = HLSVar(var_name="out_pkt.data", var_type=bus_word_t_type)
        assign_out_pkt_data = CodeAssign(var=out_pkt_data_var, expr=HLSExpr(HLSExprT.VAR, new_node_prop_var))
        if_2_codes.append(assign_out_pkt_data)

        # out_pkt.last = false;
        # (遵循示例，.last 是 ap_uint<1>)
        out_pkt_last_type = HLSType(basic_type=HLSBasicType.AP_UINT, width=1)
        out_pkt_last_var = HLSVar(var_name="out_pkt.last", var_type=out_pkt_last_type)
        assign_out_pkt_last = CodeAssign(var=out_pkt_last_var, expr=HLSExpr(HLSExprT.CONST, "false"))
        if_2_codes.append(assign_out_pkt_last)

        # write_burst_stream.write(out_pkt);
        write_stream_code = CodeWriteStream(stream_var=write_burst_stream, in_expr=out_pkt_var)
        if_2_codes.append(write_stream_code)

        # read_idx++;
        if_2_codes.append(CodeOther(text="read_idx++;"))

        # addr += (PE_NUM << 1);
        if_2_codes.append(CodeOther(text="addr += (PE_NUM << 1);")) # 假设 PE_NUM 是宏

        # if (addr >= dst_num) { break; }
        if_3_codes: List[HLSCodeLine] = []
        if_3_codes.append(CodeBreak())

        addr_var_expr = HLSExpr(HLSExprT.VAR, addr_var)
        dst_num_var_expr = HLSExpr(HLSExprT.VAR, dst_num) # 这是来自参数的 HLSVar
        if_3_expr = HLSExpr(HLSExprT.BINOP, dfir.BinOp.GE, operands=[addr_var_expr, dst_num_var_expr])

        if_3 = CodeIf(expr=if_3_expr, if_codes=if_3_codes)
        if_2_codes.append(if_3)

        self.big_apply_funcs.append(apply_kernel_inter_func)

        apply_kernel_func = HLSFunction(name="apply_kernel", comp=comp)
        params_kernel = []

        # --- 1. 定义类型和参数 (for apply_kernel) ---
        # (重用上面定义的类型和 HLSVar)

        # Param 1: bus_word_t *node_props (重用 'node_props' HLSVar)
        # Param 2: uint32_t dst_num (重用 'dst_num' HLSVar)

        # Param 3: hls::stream<write_burst_pkt_t> &kernel_out_stream
        kernel_out_stream_type = HLSType(HLSBasicType.STREAM, sub_types=[write_burst_pkt_t_type])
        kernel_out_stream = HLSVar(var_name="kernel_out_stream", var_type=kernel_out_stream_type)

        # Param 4: hls::stream<write_burst_pkt_t> &write_burst_stream (重用 'write_burst_stream' HLSVar)

        params_kernel.extend([node_props, dst_num, kernel_out_stream, write_burst_stream])
        apply_kernel_func.params = params_kernel

        # --- 2. 函数体 (for apply_kernel) ---
        code_lines_kernel: List[HLSCodeLine] = []
        apply_kernel_func.codes = code_lines_kernel

        # #pragma HLS INTERFACE m_axi port = node_props offset = slave bundle = gmem1
        code_lines_kernel.append(CodePragma(content="INTERFACE m_axi port = node_props offset = slave bundle = gmem1"))

        # #pragma HLS INTERFACE s_axilite port = node_props bundle = control
        code_lines_kernel.append(CodePragma(content="INTERFACE s_axilite port = node_props bundle = control"))

        # #pragma HLS INTERFACE s_axilite port = dst_num bundle = control
        code_lines_kernel.append(CodePragma(content="INTERFACE s_axilite port = dst_num bundle = control"))

        # #pragma HLS INTERFACE s_axilite port = return bundle = control
        code_lines_kernel.append(CodePragma(content="INTERFACE s_axilite port = return bundle = control"))

        # #pragma HLS DATAFLOW
        code_lines_kernel.append(CodePragma(content="DATAFLOW"))

        # apply_kernel_inter(node_props, dst_num, kernel_out_stream, write_burst_stream);
        # (注意：这里我们将 'kernel_out_stream' 作为第3个参数传递)
        call_params = [node_props, dst_num, kernel_out_stream, write_burst_stream]
        code_lines_kernel.append(CodeCall(func=apply_kernel_inter_func, params=call_params))

        self.big_apply_top_func = apply_kernel_func

    # 识别出S G A 三部分计算逻辑
    def generate_backend(
        self, comp_col: dfir.ComponentCollection, global_graph: Any, top_func_name: str
    ) -> Tuple[str, str]:
        
        self.global_graph_store = global_graph
        self.comp_col_store = comp_col
        
        self.big_scatter_funcs.clear()
        self.big_gather_funcs.clear()
        self.big_apply_funcs.clear()
        self.little_scatter_funcs.clear()
        self.little_gather_funcs.clear()
        self.little_apply_funcs.clear()
        component_list = comp_col.topo_sort()

        scatter_stage_comps = []        
        gather_stage_comps = []
        apply_stage_comps = []

        reduce_found = False
        reduce_out_ports = []

        for comp in component_list:
            # print(f"{type(comp)}: id:{comp.readable_id}")
            for port in comp.ports:
                conn_id = port.connection.readable_id if port.connection else "None"
                print(f"    Port: {port.readable_id}, name:{port.name},Type: {port.port_type}, Connected to: {conn_id}")
            if isinstance(
                    comp,
                    (
                        dfir.IOComponent,  # Still exclude IOComponent just in case
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
        
        self._generate_hbm_writer()

        header_name = "graphyflow_big.h"
        big_source_code , little_source_code = self._generate_source_file(header_name)

        big_header_code , little_header_code, shared_kernel_params= self._generate_header_file()
        apply_kernel = self._generate_apply()
        

        return big_header_code, little_header_code, shared_kernel_params, big_source_code, little_source_code, apply_kernel
        
    