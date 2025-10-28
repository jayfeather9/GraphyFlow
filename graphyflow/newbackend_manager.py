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
        self.batch_type_map: Dict[HLSType, HLSType] = {}
        self.struct_definitions: Dict[str, Tuple[HLSType, List[str]]] = {}
        self.unstreamed_funcs: set[str] = set()

        # State for code generation
        self.hls_functions: Dict[int, HLSFunction] = {}
        self.top_level_stream_decls: List[Tuple[CodeVarDecl, CodePragma]] = []

        self.typedefs: Dict[HLSType,str] ={}
        self.defines: Dict[str,str]={}
        # new
        self.scatter_funcs: List[HLSFunction] = []
        self.gather_funcs: List[HLSFunction] = []
        self.apply_funcs: List[HLSFunction] = []
        self.top_func = None
        self.top_dataflow_funcs =  []
        self.apply_top_func = None
        self.hbm_writer_func = None
        self.helper_funcs = [
"""
ap_fixed_pod_t get_val_from_bus(const bus_word_t bus, int offset) {
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
ap_uint<4> count_end_ones(ap_uint<PE_NUM> valid_mask) {
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
        hbm_writer_func = HLSFunction(name="hbm_writer", comp=None)
        params = []

        # --- 1. Define Types ---

        bus_word_t_type = HLSType(HLSBasicType.BUS_WORD_T)
        bus_word_ptr_type = HLSType(HLSBasicType.POINTER, sub_types=[bus_word_t_type])
        uint_type = HLSType(HLSBasicType.UINT)

        cacheline_request_pkt_t_type = HLSType(HLSBasicType.CACHELINE_REQUEST_PKT_T)
        cacheline_req_stream_type = HLSType(HLSBasicType.STREAM, sub_types=[cacheline_request_pkt_t_type])

        cacheline_response_pkt_t_type = HLSType(HLSBasicType.CACHELINE_RESPONSE_PKT_T)
        cacheline_resp_stream_type = HLSType(HLSBasicType.STREAM, sub_types=[cacheline_response_pkt_t_type])

        cacheline_data_pkt_t_type = HLSType(HLSBasicType.CACHELINE_DATA_PKT_T)
        cacheline_data_stream_type = HLSType(HLSBasicType.STREAM, sub_types=[cacheline_data_pkt_t_type])

        write_burst_pkt_t_type = HLSType(HLSBasicType.WRITE_BURST_PKT_T)
        write_burst_stream_type = HLSType(HLSBasicType.STREAM, sub_types=[write_burst_pkt_t_type])

        # --- 2. Define Params ---

        # bus_word_t *
        params.append(HLSVar(var_name="node_props_1", var_type=bus_word_ptr_type))
        params.append(HLSVar(var_name="node_props_2", var_type=bus_word_ptr_type))
        params.append(HLSVar(var_name="node_props_3", var_type=bus_word_ptr_type))
        params.append(HLSVar(var_name="output_1", var_type=bus_word_ptr_type))
        params.append(HLSVar(var_name="output_2", var_type=bus_word_ptr_type))
        params.append(HLSVar(var_name="output_3", var_type=bus_word_ptr_type))

        # uint32_t
        params.append(HLSVar(var_name="dst_num_1", var_type=uint_type))
        params.append(HLSVar(var_name="dst_num_2", var_type=uint_type))
        params.append(HLSVar(var_name="dst_num_3", var_type=uint_type))

        # hls::stream<cacheline_request_pkt_t> &
        params.append(HLSVar(var_name="cacheline_req_stream_1", var_type=cacheline_req_stream_type))
        params.append(HLSVar(var_name="cacheline_req_stream_2", var_type=cacheline_req_stream_type))
        params.append(HLSVar(var_name="cacheline_req_stream_3", var_type=cacheline_req_stream_type))

        # hls::stream<cacheline_response_pkt_t> &
        params.append(HLSVar(var_name="cacheline_resp_stream_1", var_type=cacheline_resp_stream_type))
        params.append(HLSVar(var_name="cacheline_resp_stream_2", var_type=cacheline_resp_stream_type))
        params.append(HLSVar(var_name="cacheline_resp_stream_3", var_type=cacheline_resp_stream_type))

        # hls::stream<cacheline_data_pkt_t> &
        params.append(HLSVar(var_name="cacheline_data_stream_1", var_type=cacheline_data_stream_type))
        params.append(HLSVar(var_name="cacheline_data_stream_2", var_type=cacheline_data_stream_type))
        params.append(HLSVar(var_name="cacheline_data_stream_3", var_type=cacheline_data_stream_type))

        # hls::stream<write_burst_pkt_t> &
        params.append(HLSVar(var_name="write_burst_stream_1", var_type=write_burst_stream_type))
        params.append(HLSVar(var_name="write_burst_stream_2", var_type=write_burst_stream_type))
        params.append(HLSVar(var_name="write_burst_stream_3", var_type=write_burst_stream_type))

        # --- 3. Finalize ---
        hbm_writer_func.params = params
        self.hbm_writer_func = hbm_writer_func
    def _generate_apply(self):
        code = "#include \"graphyflow_big.h\"\n\n"
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
        
        write_func_body(self.apply_top_func,True)
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

        def write_func_sig(func: HLSFunction):
            nonlocal code
            params_str = ",\n ".join(
                [p.type.get_upper_param(p.name, p.type.type != HLSBasicType.INT and p.type.type != HLSBasicType.UINT) for p in func.params]
            )
            
            code += f"extern \"C\" void\n {func.name}({params_str} \n);\n\n"
            
        header_guard = f"__GRAPHYFLOW_GRAPHYFLOW_BIG_H__"
        code = f"#ifndef {header_guard}\n#define {header_guard}\n\n"
        code += "#include <hls_stream.h>\n#include <ap_fixed.h>\n#include <ap_int.h>\n"
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



        code += "// --- Struct Type Definitions ---\n"


        sorted_defs = self._topologically_sort_structs()
        for hls_type, members in sorted_defs:
            code += hls_type.gen_decl(members) + "\n"

        code += "// --- Top-Level Function Prototypes ---\n"
        
        write_func_sig(self.top_func)
        write_func_sig(self.hbm_writer_func)
        write_func_sig(self.apply_top_func)

        code += f"#endif // {header_guard}\n"
        return code
    

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
        code_lines.append(CodeCall(func=self.top_dataflow_funcs[0], params=[edge_props_var, stream_src_ids_var, edge_stream_var, num_edges_var]))

        code_lines.append(CodeOther(text="")) # Blank line

        # --- New COO-style Source Property Loading Pipeline ---
        # dist_req_packer(stream_src_ids, stream_dist_req, num_edges);
        code_lines.append(CodeCall(func=self.top_dataflow_funcs[1], params=[stream_src_ids_var, stream_dist_req_var, num_edges_var]))

        # cacheline_req_sender(stream_dist_req, cacheline_req_stream);
        code_lines.append(CodeCall(func=self.top_dataflow_funcs[2], params=[stream_dist_req_var, cacheline_req_stream_var]))

        # node_prop_resp_receiver(cacheline_resp_stream, stream_cachelines);
        code_lines.append(CodeCall(func=self.top_dataflow_funcs[3], params=[cacheline_resp_stream_var, stream_cachelines_var]))

        # merge_node_props(stream_cachelines, edge_stream, stream_edge_data, num_edges);
        code_lines.append(CodeCall(func=self.top_dataflow_funcs[4], params=[stream_cachelines_var, edge_stream_var, stream_edge_data_var, num_edges_var]))

        code_lines.append(CodeOther(text="")) # Blank line

        # --- Main Dataflow Processing ---
        # graphyflow_big_dataflow(stream_edge_data, kernel_out_stream, dst_num);
        code_lines.append(CodeCall(func=self.top_dataflow_funcs[5], params=[stream_edge_data_var, kernel_out_stream_var, dst_num_var]))

        code_lines.append(CodeOther(text="")) # Blank line

        # --- 4. Finalize ---
        graphyflow_big_func.codes = code_lines
        self.top_func = graphyflow_big_func
    
    def _generate_source_file(self, header_name: str) -> str:
        """Generates the full content of the .cpp source file with correct function order."""

        code = f'#include "{header_name}"\n\n'

        # --- Function Definition Order ---
        # 1. Memory helper functions (lowest level)
        # 2. Utility Network Functions (zipper, demux, etc.)
        # 3. DFIR Component Functions (computational logic)
        # 4. Top-level Memory/Dataflow functions (callers)
        # 5. Top-level AXI Kernel Wrapper (final orchestrator)

        def write_func_body(func: HLSFunction,is_top):
            nonlocal code
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

        for func in self.helper_funcs:
            code = code + func + "\n\n"

        if self.scatter_funcs:
            code += "// --- 1. scatter_funcs ---\n"
            for func in self.scatter_funcs:
                write_func_body(func,False)

        if self.gather_funcs:
            code += "// --- 2. gather_funcs ---\n"
            for func in self.gather_funcs:
                write_func_body(func,False)

        code += "// --- 4. top func ---\n"
        write_func_body(self.top_func,True)


        return code



    def _translate_memory_read_op(self, comp: dfir.Component):
        
        # hard code 除了merge_node_props以外的函数 
        # edge_descriptor_loader(const bus_word_t *edge_props_ddr,
        #                hls::stream<node_id_burst_t> &stream_src_ids,
        #                hls::stream<edge_descriptor_batch_t> &edge_stream,
        #                int32_t num_edges)

 
        edge_descriptor_loader_func = HLSFunction(name="edge_descriptor_loader", comp=comp)
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
        edge_descriptor_loader_func.params = params

        # 3. 设置相关的 #defines
        # (假设 self.defines 是一个在此上下文中可用的 dict)
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
        edge_descriptor_loader_func.codes = code_lines

        # (可选：添加一个结束注释)
        code_lines.append(CodeComment("End of Edge Descriptor Loader Function"))

        #
        # dist_req_packer(hls::stream<node_id_burst_t> &src_id_burst_stream,
        #         hls::stream<distance_req_pack_t> &distance_req_pack_stream,
        #         int32_t num_nodes)
        self.scatter_funcs.append(edge_descriptor_loader_func)
        self.top_dataflow_funcs.append(edge_descriptor_loader_func)

        dist_req_packer_func = HLSFunction(name="dist_req_packer", comp=comp)
        params = []
        
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


        self.scatter_funcs.append(dist_req_packer_func)
        self.top_dataflow_funcs.append(dist_req_packer_func)
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
        self.scatter_funcs.append(cacheline_req_sender_func)
        self.top_dataflow_funcs.append(cacheline_req_sender_func)
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
        self.scatter_funcs.append(node_prop_resp_receiver_func)
        self.top_dataflow_funcs.append(node_prop_resp_receiver_func)

        
        
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

        self.scatter_funcs.append(merge_node_props_func)
        self.top_dataflow_funcs.append(merge_node_props_func)
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
        self.gather_funcs.append(demux_1_func)




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
  
        self.gather_funcs.append(sender_2_func)

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
        self.gather_funcs.append(receiver_2_func)

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
        self.gather_funcs.append(switch2x2_2_func)



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
        self.gather_funcs.append(omega_switch_2_func)

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
        self.gather_funcs.append(Reduc_105_unit_reduce_single_pe_func)

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
        self.gather_funcs.append(Reduc_105_drain_multi_pe_func)

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
        
        
        
        
        self.gather_funcs.append(graphyflow_big_dataflow_func)
        self.top_dataflow_funcs.append(graphyflow_big_dataflow_func)


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
                port_property ,target_codes = self._scatter_type_analyze(cur,port_property,port_to_var,top_vars,target_codes)
            
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
    def _scatter_type_analyze(self,comp,port_property,port_to_var,top_vars,target_codes):

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
            # self._translate_memory_read_op(comp) #这部分访存应该全是hard code
        
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
                port_property,target_codes = self._scatter_type_analyze(sub_c,port_property,port_to_var,top_vars,target_codes)
                
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
    def _build_reduce_subgraph(
        self,
        start_ports: List[dfir.Port],
        end_port: dfir.Port,
        io_var_map: Dict[dfir.Port, HLSVar],
    ) -> List[HLSCodeLine]:
        """
        Traverses a sub-graph from start to end ports and generates the inlined logic.
        (Refactored to use the _translate_inline_component helper).
        """

        q = []
        visited_ids = set()
        for p in start_ports:
            assert p.connected
            comp = p.connection.parent
            if comp.readable_id not in visited_ids:
                q.append(comp)
                visited_ids.add(comp.readable_id)
        head = 0

        while head < len(q):
            comp = q[head]
            head += 1

            inputs_ready = all(p.connection in p2var_map for p in comp.in_ports)
            if not inputs_ready:
                q.append(comp)
                if head > len(q) * 2 + len(start_ports) * 2:
                    raise RuntimeError(f"Deadlock in sub-graph topological sort at component {comp.name}")
                continue


            for p in comp.out_ports:
                if p.connected and not isinstance(
                    p.connection.parent,
                    (dfir.ReduceComponent, dfir.UnusedEndMarkerComponent),
                ):
                    successor_comp = p.connection.parent
                    if successor_comp.readable_id not in visited_ids:
                        q.append(successor_comp)
                        visited_ids.add(successor_comp.readable_id)

        return q
    
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
            port_property,inlinecodes = self._scatter_type_analyze(comp,port_property,port_to_var,top_vars,target_codes=inlinecodes)
        
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

        self.scatter_funcs.append(merge_node_props_func)
        self.top_dataflow_funcs.append(merge_node_props_func)
        
        print("========= Scatter Stage =========")
        
        


    def process_gather(self,gather_stage_comps : List[dfir.Component]):
        assert len(gather_stage_comps) == 1
        reduce_comp = gather_stage_comps[0]
        print("========= Gather Stage =========")
        self._translate_reduce_op(reduce_comp)
            
    def process_apply(self,apply_stage_comps : List[dfir.Component]):

 
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


        # ================ begin inline fused op =============================
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

        node_id_type = HLSType(HLSBasicType.NODE_ID)
        distance_type = HLSType(HLSBasicType.AP_FIXED_POD)
        print("========= apply Stage =========")
        for comp in apply_stage_comps:
            print(f"{type(comp)} id : {comp.readable_id}")
            for port in comp.ports:
                print(f"Port: {port.name}, Type: {port.port_type}, id:{port.readable_id},Connection: {port.connection if port.connection else 'None'}")
            
            if isinstance(comp, dfir.ScatterComponent):
                for port in comp.ports:
                    in_port = None
                    conn = port.connection
                    parent = conn.parent
                    idx = 0
                    if port.port_type == dfir.PortType.IN:
                        in_port = port
                        self.type_map[port] = self.type_map[conn]
                        
                    elif port.port_type == dfir.PortType.OUT:
                        if isinstance(port.data_type , dftype.SpecialIdType):
                            if port.data_type.type_name == "node_id":
                                 self.type_map[port] = node_id_type
                            else:
                                assert 0
                        elif isinstance(port.data_type , dftype.FloatType):
                            self.type_map[port] = distance_type
                        elif isinstance(port.data_type , dftype.ArrayType):
                            elem_type = port.data_type.type_
                            if isinstance(elem_type , dftype.SpecialIdType):
                                if elem_type.type_name == "node_id":
                                     self.type_map[port] = node_id_type
                                else:
                                    assert 0
                            elif isinstance(elem_type , dftype.FloatType):
                                self.type_map[port] = distance_type
                        else:
                            assert 0
                               

            elif isinstance(comp, dfir.CopyComponent):
                for port in comp.ports:
                    conn = port.connection
    
                    if port.port_type == dfir.PortType.IN:
                        in_port = port
                        self.type_map[port] = self.type_map[conn]
                    elif port.port_type == dfir.PortType.OUT:
                        self.type_map[port] = self.type_map[in_port]
            elif isinstance(comp,dfir.MemoryReadComponent):
                for port in comp.ports:
                    conn = port.connection

                    if port.port_type == dfir.PortType.IN:
                        in_port = port
                        self.type_map[port] = self.type_map[conn]
                    elif port.port_type == dfir.PortType.OUT:
                        if isinstance(port.data_type , dftype.SpecialIdType):
                            if port.data_type.type_name == "node_id":
                                 self.type_map[port] = node_id_type
                            else:
                                assert 0
                        elif isinstance(port.data_type , dftype.FloatType):
                            self.type_map[port] = distance_type
                        elif isinstance(port.data_type , dftype.ArrayType):
                            elem_type = port.data_type.type_
                            if isinstance(elem_type , dftype.SpecialIdType):
                                if elem_type.type_name == "node_id":
                                     self.type_map[port] = node_id_type
                                else:
                                    assert 0
                            elif isinstance(elem_type , dftype.FloatType):
                                self.type_map[port] = distance_type
                        else:
                            assert 0
            elif isinstance(comp,dfir.FusedOpComponent):
                 
                # 连接到memor的是旧distance 连接到scatter的是新distance
                for port in comp.ports:
                    conn = port.connection
    
                    if port.port_type == dfir.PortType.IN:
                        in_port = port
                        self.type_map[port] = self.type_map[conn]
                    elif port.port_type == dfir.PortType.OUT:
                        self.type_map[port] = distance_type
                        print(port.data_type)
                
                inline_code = []

                inline_code.append(CodeComment(f" -- Inlining FusedOp {comp.name} -- "))


                for c in comp.sub_graph.components:
                    if isinstance(c, dfir.BinOpComponent):
                        op1_expr = HLSExpr(HLSExprT.VAR, init_val_1_var)

                        op2_expr = HLSExpr(HLSExprT.VAR, init_val_2_var)

                        target_var = result_val3

                        expr = HLSExpr(HLSExprT.BINOP, c.op, [op1_expr, op2_expr])
                        inline_code.append(CodeAssign(target_var, expr))


                inline_code.append(CodeComment(f" -- End Inlining FusedOp {comp.name} -- "))


                for_loop_2_codes.extend(inline_code)

            else:
                assert 0

            
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
        self.apply_funcs.append(apply_kernel_inter_func)

        apply_kernel_func = HLSFunction(name="apply_kernel", comp=None)
        params = []

        # --- 1. Define Types & Params ---

        # Basic Types
        uint_type = HLSType(HLSBasicType.UINT)

        # Param 1: uint32_t dst_num
        dst_num_var = HLSVar(var_name="dst_num", var_type=uint_type)

        # Param 2: hls::stream<cacheline_data_pkt_t> &cacheline_data_stream
        cacheline_data_pkt_t_type = HLSType(HLSBasicType.CACHELINE_DATA_PKT_T)
        cacheline_data_stream_type = HLSType(HLSBasicType.STREAM, sub_types=[cacheline_data_pkt_t_type])
        cacheline_data_stream = HLSVar(var_name="cacheline_data_stream", var_type=cacheline_data_stream_type)

        # Param 3: hls::stream<write_burst_pkt_t> &kernel_out_stream
        write_burst_pkt_t_type = HLSType(HLSBasicType.WRITE_BURST_PKT_T)
        write_burst_stream_type = HLSType(HLSBasicType.STREAM, sub_types=[write_burst_pkt_t_type])
        kernel_out_stream = HLSVar(var_name="kernel_out_stream", var_type=write_burst_stream_type)

        # Param 4: hls::stream<write_burst_pkt_t> &write_burst_stream
        write_burst_stream = HLSVar(var_name="write_burst_stream", var_type=write_burst_stream_type)

        params.extend([dst_num_var, cacheline_data_stream, kernel_out_stream, write_burst_stream])
        apply_kernel_func.params = params

        # --- 2. Function Body ---
        code_lines: List[HLSCodeLine] = []

        # #pragma HLS INTERFACE s_axilite port = dst_num bundle = control
        code_lines.append(CodePragma(content="INTERFACE s_axilite port = dst_num bundle = control"))

        # #pragma HLS INTERFACE s_axilite port = return bundle = control
        code_lines.append(CodePragma(content="INTERFACE s_axilite port = return bundle = control"))

        # #pragma HLS DATAFLOW
        code_lines.append(CodePragma(content="DATAFLOW"))

        code_lines.append(CodeOther(text="")) # Blank line

        # apply_kernel_inter(dst_num, cacheline_data_stream, kernel_out_stream, write_burst_stream);
        # (Assuming apply_kernel_inter_func is defined)
        code_lines.append(CodeCall(func=apply_kernel_inter_func, params=[dst_num_var, cacheline_data_stream, kernel_out_stream, write_burst_stream]))

        # --- 3. Finalize ---
        apply_kernel_func.codes = code_lines
        self.apply_top_func = apply_kernel_func

    # 识别出S G A 三部分计算逻辑
    def generate_backend(
        self, comp_col: dfir.ComponentCollection, global_graph: Any, top_func_name: str
    ) -> Tuple[str, str]:
        
        
        self.global_graph_store = global_graph
        self.comp_col_store = comp_col
        

        self.scatter_funcs.clear()
        self.gather_funcs.clear()
        self.apply_funcs.clear()

        component_list = comp_col.topo_sort()

        scatter_stage_comps = []        
        gather_stage_comps = []
        apply_stage_comps = []

        reduce_found = False
        reduce_out_ports = []

        for comp in component_list:
            print(f"{type(comp)}: id:{comp.readable_id}")
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
        source_code = self._generate_source_file(header_name)
        header_code = self._generate_header_file()
        apply_kernel = self._generate_apply()
        
        return header_code,source_code,apply_kernel
    