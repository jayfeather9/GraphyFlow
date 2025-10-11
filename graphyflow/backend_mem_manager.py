from __future__ import annotations
from typing import List, Optional, Tuple, Dict, Any
import copy

import graphyflow.dataflow_ir as dfir
from graphyflow.global_graph import GlobalGraph
from graphyflow.backend_defines import (
    HLSBasicType,
    HLSType,
    HLSVar,
    HLSFunction,
    CodeVarDecl,
    CodePragma,
    CodeFor,
    CodeAssign,
    CodeIf,
    CodeWriteStream,
    CodeCall,
    CodeOther,
    HLSExpr,
    HLSExprT,
    CodeBreak,
    CodeWhile,
    HLSCodeLine,
    CodeComment,
)
from graphyflow.dataflow_ir_datatype import ArrayType, SpecialType


class MemoryAndGraphManager:
    """
    Manages validation of graph structure and programmatic generation of memory modules.
    """

    def __init__(self, comp_col: dfir.ComponentCollection, g: GlobalGraph):
        self.comp_col = comp_col
        self.g = g

        self.io_comp: Optional[dfir.IOComponent] = None
        self.reduce_comp: Optional[dfir.ReduceComponent] = None
        self.pre_reduce_mem_read: Optional[dfir.MemoryReadComponent] = None
        self.post_reduce_mem_read: Optional[dfir.MemoryReadComponent] = None
        self.go_through_mem_reads: List[dfir.MemoryReadComponent] = []
        self.mem_top_io_map: Dict[int, HLSVar] = {}

        self.mem_types: Dict[str, HLSType] = {}
        self.helper_funcs: List[HLSFunction] = []
        self.memory_loader_func: Optional[HLSFunction] = None
        self.final_writeback_func: Optional[HLSFunction] = None

        print("--- Initializing MemoryAndGraphManager ---")
        self.validate_and_analyze()
        self._build_all_memory_functions()

        self.kernel_params: Dict[str, HLSVar] = {
            "edge_batches": self.memory_loader_func.params[6],
            "node_distances": self.memory_loader_func.params[7],
            "writeback_stream": self.final_writeback_func.params[1],
        }

    def validate_and_analyze(self):
        """
        Validates the graph structure against specific architectural requirements
        and extracts key components for code generation. This version defines
        substantive MemRead components as those directly connected to the graph's IO source.
        """
        print("[M&G Manager] Phase 1: Finding Key Components...")
        # --- Phase 1: Find all component instances ---
        io_components = [c for c in self.comp_col.components if isinstance(c, dfir.IOComponent)]
        reduce_components = [c for c in self.comp_col.components if isinstance(c, dfir.ReduceComponent)]
        all_mem_read_components = [
            c for c in self.comp_col.components if isinstance(c, dfir.MemoryReadComponent)
        ]

        print(
            f"[M&G Manager] Found {len(io_components)} IO, {len(reduce_components)} Reduce, {len(all_mem_read_components)} total MemRead components."
        )

        # --- Phase 2: Graph Structure Validation (Part 1) ---
        print("[M&G Manager] Phase 2: Validating Graph Structure...")

        # 2.1: Check for a single 'edge' input
        assert len(io_components) == 1, "Graph must have exactly one IOComponent as input."
        io_comp = io_components[0]
        assert io_comp.io_type == dfir.IOComponent.IOType.INPUT, "The IOComponent must be of type INPUT."
        assert (
            isinstance(io_comp.output_type, ArrayType)
            and isinstance(io_comp.output_type.type_, SpecialType)
            and io_comp.output_type.type_.type_name == "edge"
        ), "The single graph input must be of type Array<SpecialType('edge')>."
        self.io_comp = io_comp
        print("[M&G Manager] -> Input validation successful.")

        # 2.2: Check for a single ReduceComponent
        assert len(reduce_components) == 1, "Graph must have exactly one ReduceComponent."
        self.reduce_comp = reduce_components[0]
        print("[M&G Manager] -> Reduce component validation successful.")

        # --- Phase 2.1: Redefine and Filter Substantive Memory Readers ---
        substantive_mem_reads = []
        io_comp_id = self.io_comp.readable_id
        for mem_read in all_mem_read_components:
            # If a mem reader only read node and output node_id, then is not a substantive memory reader
            is_substantive = True
            # check for mem_access_pattern
            if len(mem_read.access_pattern) == 1:
                _, base_type, access_path = mem_read.access_pattern[0]
                if base_type == "node" and len(access_path) == 1 and access_path[0] == "id":
                    is_substantive = False
                    self.go_through_mem_reads.append(mem_read)

            if is_substantive:
                print(
                    f"[M&G Manager] -> Identified Substantive MemoryRead (ID: {mem_read.readable_id}) connected to IO."
                )
                substantive_mem_reads.append(mem_read)
            else:
                print(
                    f"[M&G Manager] -> Classifying MemoryRead (ID: {mem_read.readable_id}) as computational component."
                )

        # 2.3: Check substantive MemoryReadComponent constraints
        assert (
            len(substantive_mem_reads) <= 2
        ), "Graph must have at most two substantive (IO-connected) MemoryReadComponents."

        # --- Phase 3: Classify Substantive MemoryRead Components ---
        print("[M&G Manager] Phase 3: Classifying substantive MemoryRead Components...")
        for mem_read in substantive_mem_reads:
            base_types = {pattern[1] for pattern in mem_read.access_pattern}
            assert (
                len(base_types) == 1
            ), f"Substantive MemoryReadComponent {mem_read.readable_id} must read from a single base type, but it reads from {base_types}."

            base_type = base_types.pop()
            if base_type == "edge":
                assert (
                    self.pre_reduce_mem_read is None
                ), "Multiple substantive MemoryReadComponents read from 'edge'. Only one is allowed."
                self.pre_reduce_mem_read = mem_read
            elif base_type == "node":
                assert (
                    self.post_reduce_mem_read is None
                ), "Multiple substantive MemoryReadComponents read from 'node'. Only one is allowed."
                self.post_reduce_mem_read = mem_read
            else:
                raise ValueError(f"Invalid base type '{base_type}' in a substantive MemoryReadComponent.")

        print("[M&G Manager] -> MemoryRead component validation successful.")

        # --- Phase 4: Final Analysis Summary ---
        print("[M&G Manager] Analysis complete and validation passed.")
        print(f"  - Input IO: {'Found' if self.io_comp else 'None'}")
        print(f"  - Reduce Comp: {'Found' if self.reduce_comp else 'None'}")
        print(f"  - Pre-Reduce MemRead: {'Found' if self.pre_reduce_mem_read else 'None'}")
        print(f"  - Post-Reduce MemRead: {'Found' if self.post_reduce_mem_read else 'None'}")
        if self.post_reduce_mem_read:
            print(f"    - Access Pattern: {self.post_reduce_mem_read.access_pattern}")

    def _build_all_memory_functions(self):
        print("[M&G Manager] Phase 4: Building HLSFunction representations for memory modules...")
        self._define_memory_types()

        helpers = [
            self._build_node_property_loader_func(),
            self._build_edge_descriptor_loader_func(),
            self._build_src_offset_loader_func(),
            self._build_edge_property_loader_and_dispatcher_func(),
            self._build_node_property_responder_func(),
            self._build_final_convert_func(),
            self._build_final_write_func(),
        ]
        self.helper_funcs.extend(helpers)

        self.memory_loader_func = self._build_memory_loader_func()
        self.final_writeback_func = self._build_final_writeback_func()
        print("[M&G Manager] -> HLSFunction objects built successfully.")

    def _define_memory_types(self):
        int_t = HLSType(HLSBasicType.INT)
        ap_fixed_pod_t = HLSType(HLSBasicType.AP_FIXED_POD)
        bool_t = HLSType(HLSBasicType.BOOL)
        uint8_t = HLSType(HLSBasicType.UINT8)
        float_t = HLSType(HLSBasicType.REAL_FLOAT)
        node_id_t = HLSType(HLSBasicType.NODE_ID)  # Use the correct type for node IDs

        # --- Type Correction: Use node_id_t for ID fields ---
        node_with_prop_t = HLSType(
            HLSBasicType.STRUCT, [ap_fixed_pod_t, node_id_t], "node_with_prop_t", ["prop", "node_id"]
        )

        edge_desc_arr_t = HLSType(HLSBasicType.ARRAY, [node_with_prop_t], array_dims=["PE_NUM"])
        edge_des_burst_t = HLSType(HLSBasicType.STRUCT, [edge_desc_arr_t], "edge_des_burst_t", ["edges"])

        apf_arr_t = HLSType(HLSBasicType.ARRAY, [ap_fixed_pod_t], array_dims=["PE_NUM"])
        node_dist_burst_t = HLSType(HLSBasicType.STRUCT, [apf_arr_t], "node_distance_burst_t", ["data"])

        edge_desc_batch_t = HLSType(
            HLSBasicType.STRUCT, [edge_desc_arr_t, int_t], "edge_descriptor_batch_t", ["edges", "end_pos"]
        )

        node_id_arr_t = HLSType(HLSBasicType.ARRAY, [node_id_t], array_dims=["PE_NUM"])
        edge_batch_t = HLSType(
            HLSBasicType.STRUCT,
            [apf_arr_t, apf_arr_t, node_id_arr_t, int_t, bool_t],
            "edge_batch_t",
            ["weights", "src_distances", "dsts", "end_pos", "end_flag"],
        )

        node_dist_batch_t = HLSType(
            HLSBasicType.STRUCT,
            [apf_arr_t, uint8_t, bool_t],
            "node_dist_batch_t",
            ["data", "end_pos", "end_flag"],
        )

        # --- Type Correction: Use node_id_t for ID fields ---
        kernel_out_data_t = HLSType(
            HLSBasicType.STRUCT, [float_t, node_id_t], "KernelOutputData", ["distance", "id"]
        )

        kernel_out_data_arr_t = HLSType(HLSBasicType.ARRAY, [kernel_out_data_t], array_dims=["PE_NUM"])
        kernel_out_batch_t = HLSType(
            HLSBasicType.STRUCT,
            [kernel_out_data_arr_t, bool_t, uint8_t],
            "KernelOutputBatch",
            ["data", "end_flag", "end_pos"],
        )

        # --- Type Correction: Use node_id_t for ID fields ---
        internal_end_data_arr_t = HLSType(HLSBasicType.ARRAY, [node_with_prop_t], array_dims=["PE_NUM"])
        internal_end_data_batch_t = HLSType(
            HLSBasicType.STRUCT,
            [internal_end_data_arr_t, bool_t, uint8_t],
            "internal_end_data_batch_t",
            ["data", "end_flag", "end_pos"],
        )

        self.mem_types = {
            "node_with_prop_t": node_with_prop_t,
            "edge_des_burst_t": edge_des_burst_t,
            "node_distance_burst_t": node_dist_burst_t,
            "edge_descriptor_batch_t": edge_desc_batch_t,
            "edge_batch_t": edge_batch_t,
            "node_dist_batch_t": node_dist_batch_t,
            "KernelOutputData": kernel_out_data_t,
            "KernelOutputBatch": kernel_out_batch_t,
            "internal_end_data_batch_t": internal_end_data_batch_t,
        }

    def check_type_exists(self, hls_type: HLSType) -> Optional[HLSType]:
        for defined_type in self.mem_types.values():
            if hls_type.compare_struct(defined_type):
                return defined_type
        return None

    # --- Functions to build HLSFunction objects programmatically ---

    def _build_node_property_loader_func(self) -> HLSFunction:
        T = self.mem_types
        int_ptr_const = HLSType(HLSBasicType.POINTER, [HLSType(HLSBasicType.INT)], is_const_ptr=True)
        stream_t = HLSType(HLSBasicType.STREAM, [T["node_distance_burst_t"]])
        int_t = HLSType(HLSBasicType.INT)
        bool_t = HLSType(HLSBasicType.BOOL)

        func = HLSFunction("node_property_loader", comp=None)
        func.params = [
            HLSVar("node_distances_ddr", int_ptr_const),
            HLSVar("node_distance_burst_stream_0", stream_t),
            HLSVar("node_distance_burst_stream_1", stream_t),
            HLSVar("num_nodes", int_t),
        ]

        body: List[HLSCodeLine] = [
            CodeOther(
                "const ap_uint<AXI_BUS_WIDTH> *wide_bus_ptr = reinterpret_cast<const ap_uint<AXI_BUS_WIDTH> *>(node_distances_ddr);"
            ),
            CodeVarDecl(
                "num_wide_reads", int_t, init_val="(num_nodes + NUM_WORDS_PER_BUS - 1) / NUM_WORDS_PER_BUS"
            ),
        ]
        body.extend(
            [
                CodeVarDecl("sent_pack_cnt", int_t, init_val="0"),
                CodeVarDecl("total_pack_cnt", int_t, init_val="(num_nodes + PE_NUM - 1) / PE_NUM"),
            ]
        )

        for stream_idx in range(2):
            body.append(CodeAssign(HLSVar("sent_pack_cnt", int_t), HLSExpr(HLSExprT.CONST, 0)))
            pe_loop_body = [
                CodePragma("UNROLL"),
                CodeIf(
                    HLSExpr(HLSExprT.VAR, HLSVar("i * NUM_WORDS_PER_BUS + j + pe < num_nodes", bool_t)),
                    [
                        CodeAssign(
                            HLSVar("burst.data[pe]", int_t),
                            HLSExpr(
                                HLSExprT.VAR,
                                HLSVar(
                                    "wide_word.range((j + pe + 1) * DATA_TYPE_WIDTH - 1, (j + pe) * DATA_TYPE_WIDTH)",
                                    int_t,
                                ),
                            ),
                        ),
                    ],
                ),
            ]
            pe_loop = CodeFor(pe_loop_body, "PE_NUM", iter_name="pe")

            middle_if = CodeIf(
                HLSExpr(HLSExprT.VAR, HLSVar("sent_pack_cnt < total_pack_cnt", bool_t)),
                [
                    CodeWriteStream(
                        HLSVar(f"node_distance_burst_stream_{stream_idx}", stream_t),
                        HLSVar("burst", T["node_distance_burst_t"]),
                    ),
                    CodeAssign(
                        HLSVar("sent_pack_cnt", int_t),
                        HLSExpr(HLSExprT.VAR, HLSVar("sent_pack_cnt + 1", int_t)),
                    ),
                ],
            )

            inner_loop_body = [
                CodePragma("UNROLL"),
                pe_loop,
                middle_if,
            ]

            # Use CodeFor with a custom step for the middle loop
            inner_loop = CodeFor(inner_loop_body, "NUM_WORDS_PER_BUS", iter_name="j", iter_step="j += PE_NUM")

            main_loop_body = [
                CodePragma("PIPELINE II=1"),
                CodeVarDecl("wide_word", HLSType(HLSBasicType.AP_UINT, width="AXI_BUS_WIDTH")),
                CodeAssign(
                    HLSVar("wide_word", HLSType(HLSBasicType.AP_UINT, width="AXI_BUS_WIDTH")),
                    HLSExpr(
                        HLSExprT.VAR,
                        HLSVar("wide_bus_ptr[i]", HLSType(HLSBasicType.AP_UINT, width="AXI_BUS_WIDTH")),
                    ),
                ),
                CodeVarDecl("burst", T["node_distance_burst_t"]),
                inner_loop,
            ]
            body.append(CodeFor(main_loop_body, "num_wide_reads", iter_name="i"))

        func.codes = body
        return func

    def _build_edge_descriptor_loader_func(self) -> HLSFunction:
        T = self.mem_types
        func = HLSFunction("edge_descriptor_loader", comp=None)
        func.params = [
            HLSVar(
                "edge_des_bursts", HLSType(HLSBasicType.POINTER, [T["edge_des_burst_t"]], is_const_ptr=True)
            ),
            HLSVar("edge_stream", HLSType(HLSBasicType.STREAM, [T["edge_descriptor_batch_t"]])),
            HLSVar("num_edges", HLSType(HLSBasicType.INT)),
        ]

        num_batches_decl = CodeVarDecl(
            "num_batches",
            HLSType(HLSBasicType.INT),
            init_val="(num_edges + PE_NUM - 1) / PE_NUM",
            const=True,
        )
        num_batches_var = num_batches_decl.var

        inner_loop = CodeFor(
            [
                CodePragma("UNROLL"),
                CodeIf(
                    HLSExpr(HLSExprT.VAR, HLSVar("i * PE_NUM + j < num_edges", bool)),
                    [
                        CodeAssign(
                            HLSVar("edge_batch.edges[edge_batch.end_pos]", T["node_with_prop_t"]),
                            HLSExpr(HLSExprT.VAR, HLSVar("burst.edges[j]", T["node_with_prop_t"])),
                        ),
                        CodeAssign(
                            HLSVar("edge_batch.end_pos", HLSType(HLSBasicType.INT)),
                            HLSExpr(
                                HLSExprT.VAR, HLSVar("edge_batch.end_pos + 1", HLSType(HLSBasicType.INT))
                            ),
                        ),
                    ],
                ),
            ],
            "PE_NUM",
            iter_name="j",
        )

        main_loop = CodeFor(
            [
                CodePragma("PIPELINE II=1"),
                CodeVarDecl("edge_batch", T["edge_descriptor_batch_t"]),
                CodePragma("ARRAY_PARTITION variable=edge_batch.edges complete dim=0"),
                CodePragma("dependence variable = edge_batch inter false direction = WAW"),
                CodeVarDecl("burst", T["edge_des_burst_t"]),
                CodeAssign(
                    HLSVar("burst", T["edge_des_burst_t"]),
                    HLSExpr(HLSExprT.VAR, HLSVar("edge_des_bursts[i]", T["edge_des_burst_t"])),
                ),
                CodeAssign(
                    HLSVar("edge_batch.end_pos", HLSType(HLSBasicType.INT)), HLSExpr(HLSExprT.CONST, 0)
                ),
                inner_loop,
                CodeWriteStream(func.params[1], HLSVar("edge_batch", T["edge_descriptor_batch_t"])),
            ],
            num_batches_var,
            iter_name="i",
        )

        func.codes = [
            CodePragma("dependence variable=edge_des_bursts inter false"),
            num_batches_decl,
            main_loop,
        ]
        return func

    def _build_src_offset_loader_func(self) -> HLSFunction:
        func = HLSFunction("src_offset_loader", comp=None)
        int_t = HLSType(HLSBasicType.INT)
        func.params = [
            HLSVar("src_offsets_ddr", HLSType(HLSBasicType.POINTER, [int_t], is_const_ptr=True)),
            HLSVar("src_offsets_stream", HLSType(HLSBasicType.STREAM, [int_t])),
            HLSVar("num_nodes", int_t),
        ]

        inner_if = CodeIf(
            HLSExpr(HLSExprT.VAR, HLSVar("i * NUM_WORDS_PER_BUS + j <= num_nodes", bool)),
            [
                CodeVarDecl(
                    "cur_data",
                    int_t,
                    init_val="wide_word.range((j + 1) * DATA_TYPE_WIDTH - 1, j * DATA_TYPE_WIDTH)",
                ),
                CodeWriteStream(func.params[1], HLSVar("cur_data", int_t)),
            ],
        )
        inner_loop = CodeFor([CodePragma("UNROLL"), inner_if], "NUM_WORDS_PER_BUS", iter_name="j")

        num_wide_reads_decl = CodeVarDecl(
            "num_wide_reads",
            int_t,
            init_val="(num_nodes + 1 + NUM_WORDS_PER_BUS - 1) / NUM_WORDS_PER_BUS",
            const=True,
        )
        num_wide_reads_var = num_wide_reads_decl.var

        main_loop = CodeFor(
            [
                CodePragma("PIPELINE II=1"),
                CodeVarDecl("wide_word", HLSType(HLSBasicType.AP_UINT, width="AXI_BUS_WIDTH")),
                CodeAssign(
                    HLSVar("wide_word", HLSType(HLSBasicType.AP_UINT, width="AXI_BUS_WIDTH")),
                    HLSExpr(
                        HLSExprT.VAR,
                        HLSVar("wide_bus_ptr[i]", HLSType(HLSBasicType.AP_UINT, width="AXI_BUS_WIDTH")),
                    ),
                ),
                inner_loop,
            ],
            num_wide_reads_var,
            iter_cmp="<=",
            iter_name="i",
        )

        func.codes = [
            CodeOther(
                "const ap_uint<AXI_BUS_WIDTH> *wide_bus_ptr = reinterpret_cast<const ap_uint<AXI_BUS_WIDTH> *>(src_offsets_ddr);"
            ),
            CodePragma("dependence variable=wide_bus_ptr inter false"),
            num_wide_reads_decl,
            main_loop,
        ]
        return func

    def _build_edge_property_loader_and_dispatcher_func(self) -> HLSFunction:
        # This is a complex function, translated faithfully
        T = self.mem_types
        int_t = HLSType(HLSBasicType.INT)
        func = HLSFunction("edge_property_loader_and_dispatcher", comp=None)
        func.params = [
            HLSVar("src_offsets_cache_stream", HLSType(HLSBasicType.STREAM, [int_t])),
            HLSVar("edge_stream", HLSType(HLSBasicType.STREAM, [T["edge_descriptor_batch_t"]])),
            HLSVar("node_distance_burst_stream", HLSType(HLSBasicType.STREAM, [T["node_distance_burst_t"]])),
            HLSVar("num_nodes", int_t),
            HLSVar("response_stream", HLSType(HLSBasicType.STREAM, [T["edge_batch_t"]])),
        ]

        edge_loop = CodeFor(
            [
                CodePragma("PIPELINE II=1"),
                CodeIf(
                    HLSExpr(HLSExprT.VAR, HLSVar("edge_batch_pos == edge_batch.end_pos", bool)),
                    [
                        CodeAssign(
                            HLSVar("edge_batch", T["edge_descriptor_batch_t"]),
                            HLSExpr(HLSExprT.STREAM_READ, None, [HLSExpr(HLSExprT.VAR, func.params[1])]),
                        ),
                        CodeAssign(HLSVar("edge_batch_pos", int_t), HLSExpr(HLSExprT.CONST, 0)),
                    ],
                ),
                CodeVarDecl("edge", T["node_with_prop_t"]),
                CodeAssign(
                    HLSVar("edge", T["node_with_prop_t"]),
                    HLSExpr(
                        HLSExprT.VAR, HLSVar("edge_batch.edges[edge_batch_pos++]", T["node_with_prop_t"])
                    ),
                ),
                CodeAssign(
                    HLSVar("current_batch.weights[current_batch.end_pos]", int_t),
                    HLSExpr(HLSExprT.VAR, HLSVar("edge.prop", int_t)),
                ),
                CodeAssign(
                    HLSVar("current_batch.src_distances[current_batch.end_pos]", int_t),
                    HLSExpr(HLSExprT.VAR, HLSVar("src_dist", int_t)),
                ),
                CodeAssign(
                    HLSVar("current_batch.dsts[current_batch.end_pos]", HLSType(HLSBasicType.NODE_ID)),
                    HLSExpr(HLSExprT.VAR, HLSVar("edge.node_id", HLSType(HLSBasicType.NODE_ID))),
                ),
                CodeAssign(
                    HLSVar("current_batch.end_pos", int_t),
                    HLSExpr(HLSExprT.VAR, HLSVar("current_batch.end_pos + 1", int_t)),
                ),
                CodeIf(
                    HLSExpr(HLSExprT.VAR, HLSVar("current_batch.end_pos == PE_NUM", bool)),
                    [
                        CodeWriteStream(func.params[4], HLSVar("current_batch", T["edge_batch_t"])),
                        CodeAssign(HLSVar("current_batch.end_pos", int_t), HLSExpr(HLSExprT.CONST, 0)),
                    ],
                ),
            ],
            "end_edge_idx",
            iter_name="e_idx",
            iter_cmp="<",
            iter_start="start_edge_idx",
        )

        base_idx_decl = CodeVarDecl(
            "base_idx",
            int_t,
            init_val="(node_burst_idx << LOG_PE_NUM)",
            const=True,
        )

        pe_loop = CodeFor(
            [
                CodeIf(HLSExpr(HLSExprT.VAR, HLSVar("base_idx + pe_idx >= num_nodes", bool)), [CodeBreak()]),
                CodeVarDecl("src_dist", int_t),
                CodeAssign(
                    HLSVar("src_dist", int_t),
                    HLSExpr(HLSExprT.VAR, HLSVar("node_distance_burst.data[pe_idx]", int_t)),
                ),
                CodeAssign(
                    HLSVar("end_edge_idx", int_t),
                    HLSExpr(HLSExprT.STREAM_READ, None, [HLSExpr(HLSExprT.VAR, func.params[0])]),
                ),
                edge_loop,
                CodeAssign(
                    HLSVar("start_edge_idx", int_t), HLSExpr(HLSExprT.VAR, HLSVar("end_edge_idx", int_t))
                ),
            ],
            "PE_NUM",
            iter_name="pe_idx",
        )

        max_node_burst_idx_decl = CodeVarDecl(
            "max_node_burst_idx",
            int_t,
            init_val="(num_nodes + PE_NUM - 1) / PE_NUM",
            const=True,
        )
        max_node_burst_idx_var = max_node_burst_idx_decl.var

        main_loop = CodeFor(
            [
                CodeAssign(
                    HLSVar("node_distance_burst", T["node_distance_burst_t"]),
                    HLSExpr(HLSExprT.STREAM_READ, None, [HLSExpr(HLSExprT.VAR, func.params[2])]),
                ),
                base_idx_decl,
                pe_loop,
            ],
            max_node_burst_idx_var,
            iter_name="node_burst_idx",
        )

        func.codes = [
            CodeVarDecl("current_batch", T["edge_batch_t"]),
            CodePragma("ARRAY_PARTITION variable=current_batch.weights complete dim=0"),
            CodePragma("ARRAY_PARTITION variable=current_batch.src_distances complete dim=0"),
            CodePragma("ARRAY_PARTITION variable=current_batch.dsts complete dim=0"),
            CodePragma("dependence variable=current_batch inter false direction=WAW"),
            CodeAssign(HLSVar("current_batch.end_pos", int_t), HLSExpr(HLSExprT.CONST, 0)),
            CodeAssign(HLSVar("current_batch.end_flag", bool), HLSExpr(HLSExprT.CONST, False)),
            CodeVarDecl("edge_batch", T["edge_descriptor_batch_t"]),
            CodeAssign(HLSVar("edge_batch.end_pos", int_t), HLSExpr(HLSExprT.CONST, 0)),
            CodePragma("ARRAY_PARTITION variable=edge_batch.edges complete dim=0"),
            CodePragma("dependence variable=edge_batch inter false direction=WAW"),
            CodeVarDecl("edge_batch_pos", int_t, init_val=0),
            CodeVarDecl("node_distance_burst", T["node_distance_burst_t"]),
            CodePragma("ARRAY_PARTITION variable=node_distance_burst.data complete dim=0"),
            CodePragma("dependence variable=node_distance_burst inter false"),
            CodeVarDecl("start_edge_idx", int_t),
            CodeVarDecl("end_edge_idx", int_t),
            CodeAssign(
                HLSVar("start_edge_idx", int_t),
                HLSExpr(HLSExprT.STREAM_READ, None, [HLSExpr(HLSExprT.VAR, func.params[0])]),
            ),
            max_node_burst_idx_decl,
            main_loop,
            CodeAssign(HLSVar("current_batch.end_flag", bool), HLSExpr(HLSExprT.CONST, True)),
            CodeWriteStream(func.params[4], HLSVar("current_batch", T["edge_batch_t"])),
        ]
        return func

    def _build_node_property_responder_func(self) -> HLSFunction:
        T = self.mem_types
        func = HLSFunction("node_property_responder", comp=None)
        func.params = [
            HLSVar("node_distance_burst_stream", HLSType(HLSBasicType.STREAM, [T["node_distance_burst_t"]])),
            HLSVar("num_nodes", HLSType(HLSBasicType.INT)),
            HLSVar(
                "all_distances_stream", HLSType(HLSBasicType.STREAM, [T["node_dist_batch_t"]])
            ),  # TODO: make this dynamic
        ]

        pe_loop = CodeFor(
            [
                CodePragma("UNROLL"),
                CodeIf(HLSExpr(HLSExprT.VAR, HLSVar("base_idx + pe_idx >= num_nodes", bool)), [CodeBreak()]),
                CodeAssign(
                    HLSVar("dist_batch.data[dist_batch.end_pos]", HLSType(HLSBasicType.INT)),
                    HLSExpr(
                        HLSExprT.VAR, HLSVar("node_distance_burst.data[pe_idx]", HLSType(HLSBasicType.INT))
                    ),
                ),
                CodeAssign(
                    HLSVar("dist_batch.end_pos", HLSType(HLSBasicType.INT)),
                    HLSExpr(HLSExprT.VAR, HLSVar("dist_batch.end_pos + 1", HLSType(HLSBasicType.INT))),
                ),
            ],
            "PE_NUM",
            iter_name="pe_idx",
        )

        max_node_burst_idx_decl = CodeVarDecl(
            "max_node_burst_idx",
            HLSType(HLSBasicType.INT),
            init_val="(num_nodes + PE_NUM - 1) / PE_NUM",
            const=True,
        )
        max_node_burst_idx_var = max_node_burst_idx_decl.var

        main_loop = CodeFor(
            [
                CodePragma("PIPELINE II=1"),
                CodeVarDecl("node_distance_burst", T["node_distance_burst_t"]),
                CodeAssign(
                    HLSVar("node_distance_burst", T["node_distance_burst_t"]),
                    HLSExpr(HLSExprT.STREAM_READ, None, [HLSExpr(HLSExprT.VAR, func.params[0])]),
                ),
                CodeVarDecl(
                    "base_idx",
                    HLSType(HLSBasicType.INT),
                    init_val="(node_burst_idx << LOG_PE_NUM)",
                    const=True,
                ),
                pe_loop,
                CodeWriteStream(func.params[2], HLSVar("dist_batch", T["node_dist_batch_t"])),
                CodeAssign(
                    HLSVar("dist_batch.end_pos", HLSType(HLSBasicType.INT)), HLSExpr(HLSExprT.CONST, 0)
                ),
            ],
            max_node_burst_idx_var,
            iter_name="node_burst_idx",
        )

        func.codes = [
            CodeVarDecl("dist_batch", T["node_dist_batch_t"]),
            CodeAssign(HLSVar("dist_batch.end_pos", HLSType(HLSBasicType.INT)), HLSExpr(HLSExprT.CONST, 0)),
            CodeAssign(HLSVar("dist_batch.end_flag", bool), HLSExpr(HLSExprT.CONST, False)),
            max_node_burst_idx_decl,
            main_loop,
            CodeAssign(HLSVar("dist_batch.end_flag", bool), HLSExpr(HLSExprT.CONST, True)),
            CodeWriteStream(func.params[2], HLSVar("dist_batch", T["node_dist_batch_t"])),
        ]
        return func

    def _build_final_convert_func(self) -> HLSFunction:
        T = self.mem_types
        func = HLSFunction("final_convert", comp=None)
        func.params = [
            HLSVar("in_stream", HLSType(HLSBasicType.STREAM, [T["internal_end_data_batch_t"]])),
            HLSVar("converted_stream", HLSType(HLSBasicType.STREAM, [T["KernelOutputBatch"]])),
        ]

        loop = CodeFor(
            [
                CodePragma("UNROLL"),
                CodeOther(
                    "ap_fixed<32, 16> dist_fp = *reinterpret_cast<ap_fixed<32, 16> *>(&in_batch.data[i].prop);"
                ),
                CodeAssign(
                    HLSVar("out_batch.data[i].distance", HLSType(HLSBasicType.REAL_FLOAT)),
                    HLSExpr(HLSExprT.VAR, HLSVar("(float)dist_fp", HLSType(HLSBasicType.REAL_FLOAT))),
                ),
                CodeAssign(
                    HLSVar("out_batch.data[i].id", HLSType(HLSBasicType.NODE_ID)),
                    HLSExpr(HLSExprT.VAR, HLSVar("in_batch.data[i].node_id", HLSType(HLSBasicType.NODE_ID))),
                ),
            ],
            "PE_NUM",
            iter_name="i",
        )

        func.codes = [
            CodeWhile(
                codes=[
                    CodePragma("PIPELINE II=1"),
                    CodeVarDecl("in_batch", T["internal_end_data_batch_t"]),
                    CodeAssign(
                        HLSVar("in_batch", T["internal_end_data_batch_t"]),
                        HLSExpr(HLSExprT.STREAM_READ, None, [HLSExpr(HLSExprT.VAR, func.params[0])]),
                    ),
                    CodeVarDecl("out_batch", T["KernelOutputBatch"]),
                    CodeAssign(
                        HLSVar("out_batch.end_flag", bool),
                        HLSExpr(HLSExprT.VAR, HLSVar("in_batch.end_flag", bool)),
                    ),
                    CodeAssign(
                        HLSVar("out_batch.end_pos", int),
                        HLSExpr(HLSExprT.VAR, HLSVar("in_batch.end_pos", int)),
                    ),
                    loop,
                    CodeWriteStream(func.params[1], HLSVar("out_batch", T["KernelOutputBatch"])),
                    CodeIf(HLSExpr(HLSExprT.VAR, HLSVar("in_batch.end_flag", bool)), [CodeBreak()]),
                ],
                iter_expr=HLSExpr(HLSExprT.CONST, True),
            )
        ]
        return func

    def _build_final_write_func(self) -> HLSFunction:
        T = self.mem_types
        func = HLSFunction("final_write", comp=None)
        func.params = [
            HLSVar("converted_stream", HLSType(HLSBasicType.STREAM, [T["KernelOutputBatch"]])),
            HLSVar("out_o_0_342", HLSType(HLSBasicType.POINTER, [T["KernelOutputBatch"]])),
        ]
        func.codes = [
            CodePragma("dependence variable=out_o_0_342 inter false"),
            CodeVarDecl("i", HLSType(HLSBasicType.INT), init_val=0),
            CodeWhile(
                codes=[
                    CodePragma("PIPELINE II=1"),
                    CodeVarDecl("out_batch", T["KernelOutputBatch"]),
                    CodeAssign(
                        HLSVar("out_batch", T["KernelOutputBatch"]),
                        HLSExpr(HLSExprT.STREAM_READ, None, [HLSExpr(HLSExprT.VAR, func.params[0])]),
                    ),
                    CodeAssign(
                        HLSVar("out_o_0_342[i]", T["KernelOutputBatch"]),
                        HLSExpr(HLSExprT.VAR, HLSVar("out_batch", T["KernelOutputBatch"])),
                    ),
                    CodeIf(HLSExpr(HLSExprT.VAR, HLSVar("out_batch.end_flag", bool)), [CodeBreak()]),
                    CodeAssign(HLSVar("i", int), HLSExpr(HLSExprT.VAR, HLSVar("(i + 1)", int))),
                ],
                iter_expr=HLSExpr(HLSExprT.CONST, True),
            ),
        ]
        return func

    def _build_memory_loader_func(self) -> HLSFunction:
        T = self.mem_types
        func = HLSFunction("memory_loader", comp=None)

        params = [
            HLSVar("instantiate_idx", HLSType(HLSBasicType.INT)),
            HLSVar(
                "src_offsets", HLSType(HLSBasicType.POINTER, [HLSType(HLSBasicType.INT)], is_const_ptr=True)
            ),
            HLSVar(
                "edge_des_bursts", HLSType(HLSBasicType.POINTER, [T["edge_des_burst_t"]], is_const_ptr=True)
            ),
            HLSVar(
                "node_distances",
                HLSType(HLSBasicType.POINTER, [HLSType(HLSBasicType.INT)], is_const_ptr=True),
            ),
            HLSVar("num_nodes", HLSType(HLSBasicType.INT)),
            HLSVar("num_edges", HLSType(HLSBasicType.INT)),
            HLSVar("response_to_318", HLSType(HLSBasicType.STREAM, [T["edge_batch_t"]])),
            HLSVar("all_node_distances_to_343", HLSType(HLSBasicType.STREAM, [T["node_dist_batch_t"]])),
        ]
        func.params = params

        body = [CodePragma("function_instantiate variable=instantiate_idx"), CodePragma("DATAFLOW")]

        stream_decls = [
            ("node_distance_burst_stream_0", HLSType(HLSBasicType.STREAM, [T["node_distance_burst_t"]]), 12),
            ("node_distance_burst_stream_1", HLSType(HLSBasicType.STREAM, [T["node_distance_burst_t"]]), 12),
            ("edge_stream", HLSType(HLSBasicType.STREAM, [T["edge_descriptor_batch_t"]]), 12),
            ("src_offsets_cache_stream", HLSType(HLSBasicType.STREAM, [HLSType(HLSBasicType.INT)]), 32),
        ]
        for name, type, depth in stream_decls:
            body.append(CodeVarDecl(name, type))
            body.append(CodePragma(f"STREAM variable={name} depth={depth}"))

        # Get HLSFunction objects for helpers
        (
            node_prop_loader,
            edge_desc_loader,
            src_offset_loader,
            edge_prop_dispatcher,
            node_prop_responder,
            _,
            _,
        ) = self.helper_funcs

        body.extend(
            [
                CodeCall(
                    src_offset_loader,
                    [params[1], HLSVar("src_offsets_cache_stream", stream_decls[3][1]), params[4]],
                ),
                CodeCall(
                    node_prop_loader,
                    [
                        params[3],
                        HLSVar("node_distance_burst_stream_0", stream_decls[0][1]),
                        HLSVar("node_distance_burst_stream_1", stream_decls[1][1]),
                        params[4],
                    ],
                ),
                CodeCall(edge_desc_loader, [params[2], HLSVar("edge_stream", stream_decls[2][1]), params[5]]),
                CodeCall(
                    edge_prop_dispatcher,
                    [
                        HLSVar("src_offsets_cache_stream", stream_decls[3][1]),
                        HLSVar("edge_stream", stream_decls[2][1]),
                        HLSVar("node_distance_burst_stream_0", stream_decls[0][1]),
                        params[4],
                        params[6],
                    ],
                ),
                CodeCall(
                    node_prop_responder,
                    [HLSVar("node_distance_burst_stream_1", stream_decls[1][1]), params[4], params[7]],
                ),
            ]
        )

        func.codes = body
        return func

    def _build_final_writeback_func(self) -> HLSFunction:
        T = self.mem_types
        func = HLSFunction("final_writeback", comp=None)

        func.params = [
            HLSVar("instantiate_idx", HLSType(HLSBasicType.INT)),
            HLSVar("internal_end_stream", HLSType(HLSBasicType.STREAM, [T["internal_end_data_batch_t"]])),
            HLSVar("out_o_0_342", HLSType(HLSBasicType.POINTER, [T["KernelOutputBatch"]])),
        ]

        converted_stream_t = HLSType(HLSBasicType.STREAM, [T["KernelOutputBatch"]])
        converted_stream_v = HLSVar("converted_stream", converted_stream_t)

        final_convert_func, final_write_func = self.helper_funcs[5], self.helper_funcs[6]

        func.codes = [
            CodePragma("function_instantiate variable=instantiate_idx"),
            CodePragma("DATAFLOW"),
            CodeVarDecl(converted_stream_v.name, converted_stream_v.type),
            CodePragma(f"STREAM variable={converted_stream_v.name} depth=12"),
            CodeCall(final_convert_func, [func.params[1], converted_stream_v]),
            CodeCall(final_write_func, [converted_stream_v, func.params[2]]),
        ]
        return func

    def gen_mem_read_func(
        self,
        comp: dfir.MemoryReadComponent,
        type_map: Dict[dfir.DfirType, HLSType],
        batch_type_map: Dict[HLSType, HLSType],
    ) -> HLSFunction:
        func = HLSFunction(name=comp.name, comp=comp)
        remain_ports = copy.deepcopy(comp.ports)
        if comp == self.pre_reduce_mem_read:
            assert len(comp.in_ports) == 1
            in_port = comp.in_ports[0]
            remain_ports.remove(in_port)
            # for pre_reduce read, input type is edge_batch_t
            func.params.append(
                HLSVar(
                    in_port.name,
                    HLSType(HLSBasicType.STREAM, [self.mem_types["edge_batch_t"]]),
                )
            )
            self.mem_top_io_map[in_port.readable_id] = self.kernel_params["edge_batches"]
        elif comp == self.post_reduce_mem_read:
            new_port = dfir.Port("i_all_node_distances", comp)
            comp.ports.append(new_port)
            comp.in_ports.append(new_port)
            # for post_reduce read, input type is node_dist_batch_t
            func.params.append(
                HLSVar(
                    new_port.name,
                    HLSType(HLSBasicType.STREAM, [self.mem_types["node_dist_batch_t"]]),
                )
            )
            self.mem_top_io_map[new_port.readable_id] = self.kernel_params["node_distances"]
        for port in remain_ports:
            if port.connection and isinstance(
                port.connection.parent, (dfir.UnusedEndMarkerComponent, dfir.ConstantComponent)
            ):
                continue
            dfir_type = port.data_type.type_ if isinstance(port.data_type, dfir.ArrayType) else port.data_type
            base_hls_type = type_map[dfir_type]
            batch_type = batch_type_map[base_hls_type]
            param_type = HLSType(HLSBasicType.STREAM, sub_types=[batch_type])
            func.params.append(HLSVar(var_name=port.name, var_type=param_type))
        return func

    def gen_mem_read_op(
        self, comp: dfir.Component, iterator: str, type_map: Dict[dfir.DfirType, HLSType]
    ) -> List[HLSCodeLine]:
        print("[M&G Manager] Generating mem read op for", comp.name)
        assert isinstance(comp, dfir.MemoryReadComponent)
        assert self.pre_reduce_mem_read is not None
        assert self.post_reduce_mem_read is not None

        if comp in self.go_through_mem_reads:
            assert (
                len(comp.in_ports) == 1
            ), "Go-through memory read components should have exactly one input port."
            assert (
                len(comp.out_ports) == 1
            ), "Go-through memory read components should have exactly one output port."
            codelines = []
            in_port = comp.in_ports[0]
            out_port = comp.out_ports[0]
            in_var = HLSVar(f"in_batch_{in_port.name}.data[{iterator}]", type_map[in_port.data_type])
            in_var_expr = HLSExpr(HLSExprT.VAR, in_var)
            out_var = HLSVar(f"out_batch_{out_port.name}.data[{iterator}]", type_map[out_port.data_type])
            codelines.append(CodeAssign(out_var, in_var_expr))
            return codelines
        if comp == self.pre_reduce_mem_read:
            # iterate over output ports, each port looks like "o_X_edge_a_b" where a_b_... is access path
            # extract access path and add "s" at the end, assert edge_batch_t have corresponding field
            port2field = {port: "_".join(port.name.split("_")[3:]) + "s" for port in comp.out_ports}
            codes = []
            for port, field in port2field.items():
                if field in self.mem_types["edge_batch_t"].struct_prop_names:
                    out_var = HLSVar(f"out_batch_{port.name}.data[{iterator}]", type_map[port.data_type])
                    out_var_expr = HLSExpr(HLSExprT.VAR, out_var)
                    field_var = HLSVar(
                        f"in_batch_{comp.in_ports[0].name}.{field}[{iterator}]", type_map[port.data_type]
                    )
                    field_var_expr = HLSExpr(HLSExprT.VAR, field_var)
                    codes.append(CodeAssign(out_var, field_var_expr))
                else:
                    raise ValueError(f"Field {field} not found in edge_batch_t")
            return codes
        assert False, "Only go-through and pre-reduce memory read components are supported."

    def check_post_reduce_mem_read(
        self,
        func: HLSFunction,
        comp: dfir.Component,
        type_map: Dict[dfir.DfirType, HLSType],
        batch_type_map: Dict[HLSType, HLSType],
    ) -> Optional[List[HLSCodeLine]]:
        if comp != self.post_reduce_mem_read:
            return None
        """
        Generates the HLSCodeLine list for the body of the Memor_343 C++ function.
        This version uses a flatter, more readable Python structure for generation.
        """

        type_map = copy.deepcopy(type_map)
        # --- Phase 1: Type and Variable Definitions ---
        # Define the necessary HLSType objects based on their usage in the C++ code.
        int_t = HLSType(HLSBasicType.INT)
        uint_t = HLSType(HLSBasicType.UINT)
        bool_t = HLSType(HLSBasicType.BOOL)
        node_id_t = HLSType(HLSBasicType.NODE_ID)

        assert len(comp.in_ports) == 2, "Post-reduce memory read should have exactly two input ports."
        assert len(comp.out_ports) == 1, "Post-reduce memory read should have exactly one output port."
        i_all_node_distances_port = None
        i_node_id_port = None
        o_node_distance_port = None
        for port in comp.ports:
            if port.name == "i_all_node_distances":
                i_all_node_distances_port = port
            elif port.name == "i_0_node_id":
                i_node_id_port = port
            elif port.name == "o_0_node_distance":
                o_node_distance_port = port
        assert i_all_node_distances_port is not None, "Input port 'i_all_node_distances' not found."
        assert i_node_id_port is not None, "Input port 'i_node_id' not found."
        assert o_node_distance_port is not None, "Output port 'o_node_distance' not found."

        # get the corresponding function param HLSVar for each port by comparing names
        param_map = {param.name: param for param in func.params}
        i_all_node_dist_param = param_map[i_all_node_distances_port.name]
        i_node_id_param = param_map[i_node_id_port.name]
        o_node_distance_param = param_map[o_node_distance_port.name]

        node_dist_batch_t = self.mem_types["node_dist_batch_t"]
        out_node_dist_t = type_map[o_node_distance_port.data_type]
        out_node_dist_batch_t = batch_type_map[out_node_dist_t]
        in_node_id_t = type_map[i_node_id_port.data_type]
        in_node_id_batch_t = batch_type_map[in_node_id_t]

        # Define local variables
        in_node_id_batch_v = HLSVar("in_node_id_batch", in_node_id_batch_t)
        in_dist_batch_v = HLSVar("in_dist_batch", node_dist_batch_t)
        out_dist_batch_v = HLSVar("out_dist_batch", out_node_dist_batch_t)
        id_idx_v = HLSVar("id_idx", uint_t)
        in_node_base_id_v = HLSVar("in_node_base_id", uint_t)
        in_node_end_id_v = HLSVar("in_node_end_id", uint_t)
        current_batch_len_v = HLSVar("current_batch_len", uint_t)
        target_node_id_v = HLSVar("target_node_id", node_id_t)
        final_batch_v = HLSVar("final_batch", out_node_dist_batch_t)

        # --- Phase 2: Code Generation (Flatter Structure) ---

        body: List[HLSCodeLine] = []

        # --- Initial declarations and setup ---
        body.extend(
            [
                CodeComment(
                    "Efficiently filters a stream of all node distances against a stream of requested node IDs."
                ),
                CodeVarDecl(in_node_id_batch_v.name, in_node_id_batch_v.type),
                CodeVarDecl(in_dist_batch_v.name, in_dist_batch_v.type),
                CodeVarDecl(out_dist_batch_v.name, out_dist_batch_v.type),
                CodePragma("ARRAY_PARTITION variable=in_node_id_batch.data complete dim=0"),
                CodePragma("ARRAY_PARTITION variable=in_dist_batch.data complete dim=0"),
                CodePragma("ARRAY_PARTITION variable=out_dist_batch.data complete dim=0"),
                CodeAssign(
                    HLSVar(f"{out_dist_batch_v.name}.end_flag", bool_t), HLSExpr(HLSExprT.CONST, False)
                ),
                CodeComment("Initial reads to prime the pipeline"),
                CodeAssign(
                    in_node_id_batch_v,
                    HLSExpr(HLSExprT.STREAM_READ, None, [HLSExpr(HLSExprT.VAR, i_node_id_param)]),
                ),
                CodeAssign(
                    in_dist_batch_v,
                    HLSExpr(HLSExprT.STREAM_READ, None, [HLSExpr(HLSExprT.VAR, i_all_node_dist_param)]),
                ),
                CodeVarDecl(in_node_base_id_v.name, in_node_base_id_v.type, init_val=0),
                CodeVarDecl(id_idx_v.name, id_idx_v.type, init_val=0),
                CodeVarDecl(in_node_end_id_v.name, in_node_end_id_v.type),
                CodeAssign(
                    in_node_end_id_v,
                    HLSExpr(
                        HLSExprT.UOP,
                        (dfir.UnaryOp.GET_ATTR, "end_pos"),
                        [HLSExpr(HLSExprT.VAR, in_dist_batch_v)],
                    ),
                ),
                CodePragma("BIND_STORAGE variable=in_node_end_id type=register impl=srl"),
            ]
        )

        # --- Build the main `while(true)` loop body from the inside out ---

        # Logic to handle finishing a batch of requested IDs
        handle_id_batch_finish_codes = [
            CodeIf(
                expr=HLSExpr(
                    HLSExprT.BINOP,
                    dfir.BinOp.GT,
                    [HLSExpr(HLSExprT.VAR, id_idx_v), HLSExpr(HLSExprT.CONST, 0)],
                ),
                if_codes=[
                    CodeAssign(
                        HLSVar(f"{out_dist_batch_v.name}.end_pos", uint_t), HLSExpr(HLSExprT.VAR, id_idx_v)
                    ),
                    CodeWriteStream(o_node_distance_param, out_dist_batch_v),
                ],
            ),
            CodeIf(
                expr=HLSExpr(
                    HLSExprT.UOP,
                    (dfir.UnaryOp.GET_ATTR, "end_flag"),
                    [HLSExpr(HLSExprT.VAR, in_node_id_batch_v)],
                ),
                if_codes=[CodeBreak()],
            ),
            CodeAssign(
                in_node_id_batch_v,
                HLSExpr(HLSExprT.STREAM_READ, None, [HLSExpr(HLSExprT.VAR, i_node_id_param)]),
            ),
            CodeAssign(id_idx_v, HLSExpr(HLSExprT.CONST, 0)),
            CodeOther("continue;"),
        ]

        # Logic to fetch the next batch of distances if the target ID is out of range
        fetch_dist_batch_codes = [
            CodeComment("Target is in a future batch, load the next distance batch"),
            CodeAssign(in_node_base_id_v, HLSExpr(HLSExprT.VAR, in_node_end_id_v)),
            CodeAssign(
                in_dist_batch_v,
                HLSExpr(HLSExprT.STREAM_READ, None, [HLSExpr(HLSExprT.VAR, i_all_node_dist_param)]),
            ),
            CodeVarDecl("batch_len", uint_t, init_val=f"{in_dist_batch_v.name}.end_pos"),
            CodeAssign(
                in_node_end_id_v,
                HLSExpr(
                    HLSExprT.BINOP,
                    dfir.BinOp.ADD,
                    [
                        HLSExpr(HLSExprT.VAR, in_node_base_id_v),
                        HLSExpr(HLSExprT.VAR, HLSVar("batch_len", uint_t)),
                    ],
                ),
            ),
            CodePragma("BIND_OP variable=in_node_end_id op=add impl=fabric latency=0"),
            CodeOther("continue;"),
        ]

        # Assemble the main while loop's body
        while_loop_codes = [
            CodePragma("PIPELINE II=1"),
            CodePragma("expression_balance"),
            CodeVarDecl(
                current_batch_len_v.name,
                current_batch_len_v.type,
                init_val=f"{in_node_id_batch_v.name}.end_pos",
            ),
            CodeIf(
                expr=HLSExpr(
                    HLSExprT.BINOP,
                    dfir.BinOp.OR,
                    [
                        HLSExpr(
                            HLSExprT.BINOP,
                            dfir.BinOp.EQ,
                            [HLSExpr(HLSExprT.VAR, current_batch_len_v), HLSExpr(HLSExprT.CONST, 0)],
                        ),
                        HLSExpr(
                            HLSExprT.BINOP,
                            dfir.BinOp.GE,
                            [HLSExpr(HLSExprT.VAR, id_idx_v), HLSExpr(HLSExprT.VAR, current_batch_len_v)],
                        ),
                    ],
                ),
                if_codes=handle_id_batch_finish_codes,
            ),
            CodeVarDecl(
                target_node_id_v.name,
                target_node_id_v.type,
                init_val=f"{in_node_id_batch_v.name}.data[{id_idx_v.name}]",
            ),
            CodeIf(
                expr=HLSExpr(
                    HLSExprT.BINOP,
                    dfir.BinOp.GE,
                    [HLSExpr(HLSExprT.VAR, target_node_id_v), HLSExpr(HLSExprT.VAR, in_node_end_id_v)],
                ),
                if_codes=fetch_dist_batch_codes,
            ),
            CodeComment("Target found, calculate index and copy distance"),
            CodeAssign(
                var=HLSVar(f"{out_dist_batch_v.name}.data[{id_idx_v.name}]", int_t),
                expr=HLSExpr(
                    HLSExprT.VAR,
                    HLSVar(
                        f"{in_dist_batch_v.name}.data[{target_node_id_v.name} - {in_node_base_id_v.name}]",
                        int_t,
                    ),
                ),
            ),
            CodeAssign(
                id_idx_v,
                HLSExpr(
                    HLSExprT.BINOP,
                    dfir.BinOp.ADD,
                    [HLSExpr(HLSExprT.VAR, id_idx_v), HLSExpr(HLSExprT.CONST, 1)],
                ),
            ),
        ]

        body.append(CodeWhile(codes=while_loop_codes, iter_expr=HLSExpr(HLSExprT.CONST, True)))

        # --- Finalization and Stream Draining ---
        body.extend(
            [
                CodeComment("Send the final (empty) output batch with the end flag"),
                CodeVarDecl(final_batch_v.name, final_batch_v.type),
                CodeAssign(HLSVar(f"{final_batch_v.name}.end_flag", bool_t), HLSExpr(HLSExprT.CONST, True)),
                CodeAssign(HLSVar(f"{final_batch_v.name}.end_pos", uint_t), HLSExpr(HLSExprT.CONST, 0)),
                CodeWriteStream(o_node_distance_param, final_batch_v),
                CodeComment("Drain any remaining batches from the all_distances stream to prevent deadlock"),
                CodeWhile(
                    iter_expr=HLSExpr(
                        HLSExprT.UOP,
                        dfir.UnaryOp.NOT,
                        [
                            HLSExpr(
                                HLSExprT.UOP,
                                (dfir.UnaryOp.GET_ATTR, "end_flag"),
                                [HLSExpr(HLSExprT.VAR, in_dist_batch_v)],
                            )
                        ],
                    ),
                    codes=[
                        CodeAssign(
                            in_dist_batch_v,
                            HLSExpr(
                                HLSExprT.STREAM_READ, None, [HLSExpr(HLSExprT.VAR, i_all_node_dist_param)]
                            ),
                        )
                    ],
                ),
            ]
        )
        return body
