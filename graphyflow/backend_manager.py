from __future__ import annotations
from enum import Enum
from typing import List, Optional, Union, Dict, Any, Tuple
import graphyflow.dataflow_ir_datatype as dftype
import graphyflow.dataflow_ir as dfir
import re


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
        self.top_level_io_ports: List[dfir.Port] = []

        self.reduce_internal_streams: Dict[int, Dict[str, HLSVar]] = {}

        self.utility_functions: List[HLSFunction] = []
        self.reduce_helpers: Dict[int, Dict[str, Any]] = {}

        self.global_graph_store = None
        self.comp_col_store = None

        # 新增: 存储AXI相关的函数和类型信息
        self.axi_input_ports: List[dfir.Port] = []
        self.axi_output_ports: List[dfir.Port] = []
        self.mem_to_stream_func: Optional[HLSFunction] = None
        self.stream_to_mem_func: Optional[HLSFunction] = None
        self.dataflow_core_func: Optional[HLSFunction] = None

    def generate_backend(
        self, comp_col: dfir.ComponentCollection, global_graph: Any, top_func_name: str
    ) -> Tuple[str, str]:
        """
        Main entry point to generate HLS header and source files.
        """
        self.global_graph_store = global_graph
        self.comp_col_store = comp_col
        header_name = f"{top_func_name}.h"

        # 1. Discover top-level I/O ports
        self.axi_input_ports = [
            p.connection
            for p in comp_col.inputs
            if isinstance(p.parent, dfir.IOComponent)
            and p.parent.io_type == dfir.IOComponent.IOType.INPUT
            and p.connected
        ]
        self.axi_output_ports = comp_col.outputs
        self.top_level_io_ports = self.axi_input_ports + self.axi_output_ports

        # Phase 1: Type Analysis
        self._analyze_and_map_types(comp_col)
        # Phase 2: Function Definition and Stream Instantiation
        self._define_functions_and_streams(comp_col, top_func_name)
        # Phase 3: Code Body Generation
        self._translate_functions()

        # Phase 4: AXI Wrapper Generation
        self.mem_to_stream_func = self._generate_mem_to_stream_func()
        self.stream_to_mem_func = self._generate_stream_to_mem_func()
        self.dataflow_core_func = self._generate_dataflow_core_func(top_func_name)

        axi_wrapper_func_str, top_func_sig = self._generate_axi_kernel_wrapper(top_func_name)

        # 2. Generate file contents
        header_code = self._generate_header_file(top_func_name, top_func_sig)
        source_code = self._generate_source_file(header_name, axi_wrapper_func_str)

        return header_code, source_code

    def generate_host_codes(self, top_func_name: str) -> Tuple[str, str]:
        """生成 generated_host.h 和 generated_host.cpp 的内容"""

        # ----- 生成 generated_host.h -----
        header_code = f"""
#ifndef __GENERATED_HOST_H__
#define __GENERATED_HOST_H__

#include "common.h"
#include "{top_func_name}.h" // 包含内核数据类型定义
#include <vector>

class AlgorithmHost {{
public:
    AlgorithmHost(cl::Context& context, cl::Kernel& kernel, cl::CommandQueue& q);
    void setup_buffers(const GraphCSR& graph, int start_node);
    void transfer_data_to_fpga();
    void execute_kernel_iteration(cl::Event& event);
    void transfer_data_from_fpga();
    const std::vector<int>& get_results() const; // 保持接口兼容，即使内容可能为空
    std::vector<int>& get_host_memory_for_verification();
    int get_stop_flag() const;

private:
    cl::Context& m_context;
    cl::Kernel& m_kernel;
    cl::CommandQueue& m_q;
    int m_num_vertices;
    
    // Host-side memory
"""
        # 为每个内核IO参数声明 host vector 和 device buffer
        for port in self.top_level_io_ports:
            hls_type = self.batch_type_map[self.type_map[port.data_type]]
            header_code += f"    std::vector<{hls_type.name}, aligned_allocator<{hls_type.name}>> h_{port.unique_name};\n"

        header_code += "\n    // Device-side OpenCL buffers\n"
        for port in self.top_level_io_ports:
            header_code += f"    cl::Buffer d_{port.unique_name};\n"

        header_code += """
}};

#endif // __GENERATED_HOST_H__
"""

        # ----- 生成 generated_host.cpp -----
        cpp_code = f"""
#include "generated_host.h"
#include <iostream>

// 构造函数
AlgorithmHost::AlgorithmHost(cl::Context& context, cl::Kernel& kernel, cl::CommandQueue& q)
    : m_context(context), m_kernel(kernel), m_q(q), m_num_vertices(0) {{
}}

void AlgorithmHost::setup_buffers(const GraphCSR& graph, int start_node) {{
    m_num_vertices = graph.num_vertices;
    cl_int err;

    // --- 模拟数据填充 ---
    // 这是需要根据具体应用修改的部分。
    // 这里我们仅为buffer分配空间并填充一些默认值。
    size_t num_elements = 1024; // 示例: 假设我们的流中有1024个元素
"""
        for port in self.top_level_io_ports:
            hls_type = self.batch_type_map[self.type_map[port.data_type]]
            is_input = port.port_type == dfir.PortType.IN

            cpp_code += f"""
    // Buffer for {port.unique_name}
    h_{port.unique_name}.resize(num_elements);
    // TODO: 使用有意义的数据填充 h_{port.unique_name}
    for(size_t i = 0; i < num_elements; ++i) {{
        // 示例填充
        h_{port.unique_name}[i] = {{}}; // 使用默认构造函数
    }}
    h_{port.unique_name}[num_elements - 1].end_flag = true; // 设置最后一个元素的结束标志

    cl_mem_flags flag = CL_MEM_{'READ_ONLY' if is_input else 'WRITE_ONLY'};
    OCL_CHECK(err, d_{port.unique_name} = cl::Buffer(m_context, flag, h_{port.unique_name}.size() * sizeof({hls_type.name}), NULL, &err));
"""

        cpp_code += "}\n\n"

        # transfer_data_to_fpga
        cpp_code += "void AlgorithmHost::transfer_data_to_fpga() {\n    cl_int err;\n"
        for port in self.top_level_io_ports:
            if port.port_type == dfir.PortType.IN:
                hls_type = self.batch_type_map[self.type_map[port.data_type]]
                cpp_code += f"    OCL_CHECK(err, err = m_q.enqueueWriteBuffer(d_{port.unique_name}, CL_TRUE, 0, h_{port.unique_name}.size() * sizeof({hls_type.name}), h_{port.unique_name}.data(), nullptr, nullptr));\n"
        cpp_code += "}\n\n"

        # execute_kernel_iteration
        cpp_code += "void AlgorithmHost::execute_kernel_iteration(cl::Event& event) {\n    cl_int err;\n    int arg_idx = 0;\n"
        cpp_code += "    uint16_t input_length = 1024; // 示例长度\n"
        cpp_code += "    OCL_CHECK(err, err = m_kernel.setArg(arg_idx++, input_length));\n"
        for port in self.top_level_io_ports:
            cpp_code += f"    OCL_CHECK(err, err = m_kernel.setArg(arg_idx++, d_{port.unique_name}));\n"
        cpp_code += "\n    OCL_CHECK(err, err = m_q.enqueueTask(m_kernel, nullptr, &event));\n}\n\n"

        # transfer_data_from_fpga
        cpp_code += "void AlgorithmHost::transfer_data_from_fpga() {\n    cl_int err;\n"
        for port in self.top_level_io_ports:
            if port.port_type == dfir.PortType.OUT:
                hls_type = self.batch_type_map[self.type_map[port.data_type]]
                cpp_code += f"    OCL_CHECK(err, err = m_q.enqueueReadBuffer(d_{port.unique_name}, CL_TRUE, 0, h_{port.unique_name}.size() * sizeof({hls_type.name}), h_{port.unique_name}.data(), nullptr, nullptr));\n"
        cpp_code += "}\n\n"

        # get_results (保持兼容性)
        cpp_code += """
const std::vector<int>& AlgorithmHost::get_results() const {
    // 这个函数是为了与旧的验证流程兼容。
    // 对于流式内核，可能没有一个单一的“距离”向量。
    // 我们返回一个空的静态向量来满足接口要求。
    static const std::vector<int> empty_results;
    std::cout << "Warning: get_results() called on a streaming kernel. Returning empty vector." << std::endl;
    // TODO: 实现有意义的结果提取逻辑
    return empty_results;
}

std::vector<int>& AlgorithmHost::get_host_memory_for_verification() {
    static std::vector<int> empty_vector;
    return empty_vector;
}

int AlgorithmHost::get_stop_flag() const {
    // 流式内核通常执行一次就结束，不像迭代算法。
    // 返回1表示“停止”，这样主循环在fpga_executor中执行一次后就会退出。
    return 1;
}
"""
        return header_code, cpp_code

    def generate_build_system_files(self, kernel_name: str, executable_name: str) -> Dict[str, str]:
        """生成所有需要的构建和执行文件的内容"""

        files = {}

        # Makefile
        files[
            "Makefile"
        ] = f"""
# This file is auto-generated by the GraphyFlow backend.
KERNEL_NAME := {kernel_name}
EXECUTABLE := {executable_name}

TARGET := sw_emu
DEVICE := /opt/xilinx/platforms/xilinx_u55c_gen3x16_xdma_3_202210_1/xilinx_u55c_gen3x16_xdma_3_202210_1.xpfm

.PHONY: all clean cleanall exe check

include scripts/main.mk

check: all
	./run.sh
"""
        # fpga_executor.cpp (修改内核名称宏)
        files[
            "fpga_executor.cpp"
        ] = f"""
#include "fpga_executor.h"
#include "generated_host.h" 
#include <iostream>

#define KERNEL_NAME "{kernel_name}"

// ... (fpga_executor.cpp 的其余内容保持不变) ...
std::vector<int> run_fpga_kernel(
    const std::string& xclbin_path,
    const GraphCSR& graph,
    int start_node,
    double& total_kernel_time_sec)
{{
    cl_int err;
    auto devices = xcl::get_xil_devices();
    auto device = devices[0];
    OCL_CHECK(err, cl::Context context(device, NULL, NULL, NULL, &err));
    OCL_CHECK(err, cl::CommandQueue q(context, device, CL_QUEUE_PROFILING_ENABLE, &err));
    auto fileBuf = xcl::read_binary_file(xclbin_path);
    cl::Program::Binaries bins{{{{fileBuf.data(), fileBuf.size()}}}};
    OCL_CHECK(err, cl::Program program(context, {{device}}, bins, NULL, &err));
    OCL_CHECK(err, cl::Kernel kernel(program, KERNEL_NAME, &err));

    AlgorithmHost algo_host(context, kernel, q);
    algo_host.setup_buffers(graph, start_node);
    total_kernel_time_sec = 0;
    int max_iterations = graph.num_vertices;
    int iter = 0;
    std::cout << "\\nStarting FPGA execution..." << std::endl;

    // 对于流式内核, 这个循环只会执行一次 (因为 get_stop_flag 返回 1)
    while (iter < max_iterations) {{
        algo_host.transfer_data_to_fpga();
        cl::Event event;
        algo_host.execute_kernel_iteration(event);
        event.wait();
        algo_host.transfer_data_from_fpga();

        unsigned long start = 0, end = 0;
        event.getProfilingInfo(CL_PROFILING_COMMAND_START, &start);
        event.getProfilingInfo(CL_PROFILING_COMMAND_END, &end);
        double iteration_time_ns = end - start;
        total_kernel_time_sec += iteration_time_ns * 1.0e-9;
        double mteps = (double)graph.num_edges / (iteration_time_ns * 1.0e-9) / 1.0e6;

        std::cout << "FPGA Iteration " << iter << ": "
                  << "Time = " << (iteration_time_ns * 1.0e-6) << " ms, "
                  << "Throughput = " << mteps << " MTEPS" << std::endl;
        
        iter++;
        
        if (algo_host.get_stop_flag() == 1) {{
            std::cout << "FPGA computation finished after " << iter << " execution(s)." << std::endl;
            break;
        }}
    }}

    const std::vector<int>& final_results_ref = algo_host.get_results();
    std::vector<int> final_results = final_results_ref;
    
    return final_results;
}}
"""
        # run.sh
        files[
            "run.sh"
        ] = f"""
#!/bin/bash
# This file is auto-generated by the GraphyFlow backend.
source /home/feiyang/set_env.sh
export XCL_EMULATION_MODE=sw_emu
DATASET="./graph.txt"
./{executable_name} ./xclbin/{kernel_name}.sw_emu.xclbin $DATASET
"""
        # system.cfg
        files[
            "system.cfg"
        ] = f"""
# This file is auto-generated by the GraphyFlow backend.
[connectivity]
nk={kernel_name}:1:{kernel_name}_1
"""
        # scripts/kernel/kernel.mk
        files[
            "kernel.mk"
        ] = """
# Makefile for the Vitis Kernel
VPP := v++
# KERNEL_NAME is passed from the top Makefile
KERNEL_SRC := scripts/kernel/$(KERNEL_NAME).cpp
XCLBIN_DIR := ./xclbin
XCLBIN_FILE := $(XCLBIN_DIR)/$(KERNEL_NAME).$(TARGET).xclbin
KERNEL_XO := $(XCLBIN_DIR)/$(KERNEL_NAME).$(TARGET).xo
EMCONFIG_FILE := ./emconfig.json
CLFLAGS += --kernel $(KERNEL_NAME)
CLFLAGS += -Iscripts/kernel
LDFLAGS_VPP += --config ./system.cfg
LDFLAGS_VPP += -Iscripts/kernel

$(KERNEL_XO): $(KERNEL_SRC)
	@mkdir -p $(XCLBIN_DIR)
	$(VPP) -c -t $(TARGET) --platform $(DEVICE) $(CLFLAGS) -o $@ $<

$(XCLBIN_FILE): $(KERNEL_XO)
	$(VPP) -l -t $(TARGET) --platform $(DEVICE) $(LDFLAGS_VPP) -o $@ $<

emconfig:
	emconfigutil --platform $(DEVICE) --od .

.PHONY: emconfig
"""
        return files

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

    def generate_common_header(self, top_func_name: str) -> str:
        """
        Generates the full content of the common.h header file.
        """
        # *** 关键：在排序前调用此方法，以确保 KernelOutput 类型被注册 ***
        self._get_host_output_type()

        header_guard = "__COMMON_H__"
        code = f"#ifndef {header_guard}\n#define {header_guard}\n\n"

        code += "#include <limits>\n"
        code += "#include <string>\n"
        code += "#include <vector>\n\n"
        code += '#ifndef __SYNTHESIS__\n#include "xcl2.h"\n#endif\n\n'
        code += "#include <ap_fixed.h>\n#include <stdint.h>\n\n"
        code += f"#define PE_NUM {self.PE_NUM}\n\n"

        code += "// --- Struct Type Definitions ---\n"
        sorted_defs = self._topologically_sort_structs()
        for hls_type, members in sorted_defs:
            code += hls_type.gen_decl(members) + "\n"

        code += f"#endif // {header_guard}\n"
        return code

    def _define_functions_and_streams(self, comp_col: dfir.ComponentCollection, top_func_name: str):
        """
        Phase 2: Creates HLSFunction objects, defines their signatures, and
        identifies the intermediate streams needed for the top-level function.
        """
        self.hls_functions.clear()
        self.top_level_stream_decls.clear()
        self.reduce_helpers.clear()
        self.utility_functions.clear()

        # This map is used to track which ReduceComponent sub-graph components have been processed
        # to avoid creating duplicate HLSFunction objects for them.
        processed_sub_comp_ids = set()

        for comp in comp_col.components:
            if isinstance(comp, dfir.ReduceComponent):
                # 1. Create HLS functions for the Reduce component's stages
                pre_process_func = HLSFunction(name=f"{comp.name}_pre_process", comp=comp)
                unit_reduce_func = HLSFunction(name=f"{comp.name}_unit_reduce", comp=comp)

                # 2. Define function signatures
                in_type = self.type_map[comp.get_port("i_0").data_type]
                key_type = self.type_map[comp.get_port("i_reduce_key_out").data_type]
                transform_type = self.type_map[comp.get_port("i_reduce_transform_out").data_type]

                pre_process_func.params = [
                    HLSVar("i_0", HLSType(HLSBasicType.STREAM, [self.batch_type_map[in_type]])),
                    HLSVar("intermediate_key", HLSType(HLSBasicType.STREAM, [self.batch_type_map[key_type]])),
                    HLSVar(
                        "intermediate_transform",
                        HLSType(HLSBasicType.STREAM, [self.batch_type_map[transform_type]]),
                    ),
                ]

                kt_pair_type = HLSType(
                    HLSBasicType.STRUCT,
                    sub_types=[key_type, transform_type],
                    struct_name=f"kt_pair_{comp.readable_id}_t",
                    struct_prop_names=["key", "transform"],
                )
                self.struct_definitions[kt_pair_type.name] = (kt_pair_type, ["key", "transform"])

                net_wrapper_type = HLSType(
                    HLSBasicType.STRUCT,
                    [kt_pair_type, HLSType(HLSBasicType.BOOL)],
                    struct_name=f"net_wrapper_{kt_pair_type.name}_t",
                    struct_prop_names=["data", "end_flag"],
                )
                self.struct_definitions[net_wrapper_type.name] = (net_wrapper_type, ["data", "end_flag"])

                out_dfir_type = comp.get_port("o_0").data_type
                base_out_batch_type = self._get_batch_type(self.type_map[out_dfir_type])

                unit_reduce_func.params = [
                    HLSVar(
                        "kt_wrap_item",
                        HLSType(
                            HLSBasicType.ARRAY,
                            [HLSType(HLSBasicType.STREAM, [net_wrapper_type])],
                            array_dims=["PE_NUM"],
                        ),
                    ),
                    HLSVar("o_0", HLSType(HLSBasicType.STREAM, [base_out_batch_type])),
                ]

                self.hls_functions[pre_process_func.readable_id] = pre_process_func
                self.hls_functions[unit_reduce_func.readable_id] = unit_reduce_func

                # 3. Generate utility functions
                key_batch_type = self.batch_type_map[key_type]
                transform_batch_type = self.batch_type_map[transform_type]
                kt_pair_batch_type = self._get_batch_type(kt_pair_type)

                zipper_func = generate_stream_zipper(key_batch_type, transform_batch_type, kt_pair_batch_type)
                demux_func = generate_demux(self.PE_NUM, kt_pair_batch_type, net_wrapper_type)
                omega_funcs = generate_omega_network(self.PE_NUM, net_wrapper_type, routing_key_member="key")
                omega_func = next(f for f in omega_funcs if "omega_switch" in f.name)

                self.utility_functions.extend([zipper_func, demux_func] + omega_funcs)

                # 4. Store helpers and declare all necessary streams
                helpers = {
                    "pre_process": pre_process_func,
                    "unit_reduce": unit_reduce_func,
                    "zipper": zipper_func,
                    "demux": demux_func,
                    "omega": omega_func,
                }

                streams_to_declare = {
                    "zipper_to_demux": HLSVar(
                        f"reduce_{comp.readable_id}_z2d_pair",
                        HLSType(HLSBasicType.STREAM, [kt_pair_batch_type]),
                    ),
                    "demux_to_omega": HLSVar(
                        f"reduce_{comp.readable_id}_d2o_pair", demux_func.params[1].type
                    ),
                    "omega_to_unit": HLSVar(f"reduce_{comp.readable_id}_o2u_pair", omega_func.params[1].type),
                }

                # --- *** 关键修正：在此处添加 'intermediate' 流到字典中 *** ---
                for param_name in ["intermediate_key", "intermediate_transform"]:
                    stream_type = next(p.type for p in pre_process_func.params if p.name == param_name)
                    stream_var = HLSVar(f"reduce_{comp.readable_id}_{param_name}", stream_type)
                    streams_to_declare[param_name] = stream_var
                # -----------------------------------------------------------

                for stream_var in streams_to_declare.values():
                    decl = CodeVarDecl(stream_var.name, stream_var.type)
                    pragma = CodePragma(f"STREAM variable={stream_var.name} depth={self.STREAM_DEPTH}")
                    self.top_level_stream_decls.append((decl, pragma))

                helpers["streams"] = streams_to_declare
                self.reduce_helpers[comp.readable_id] = helpers

                internal_streams: Dict[str, HLSVar] = {}
                for param_name in ["intermediate_key", "intermediate_transform"]:
                    # 从刚刚创建的函数签名中获取流的类型
                    stream_type = next(p.type for p in pre_process_func.params if p.name == param_name)
                    # 为这个流创建一个在顶层函数中唯一的变量名
                    stream_var = HLSVar(f"reduce_{comp.readable_id}_{param_name}", stream_type)
                    internal_streams[param_name] = stream_var

                    # 将声明和 pragma 添加到顶层函数体
                    decl = CodeVarDecl(stream_var.name, stream_var.type)
                    pragma = CodePragma(f"STREAM variable={stream_var.name} depth={self.STREAM_DEPTH}")
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
                                if p.connected and not isinstance(p.connection.parent, dfir.ReduceComponent):
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
            for port in comp.ports:
                if port.connection and isinstance(
                    port.connection.parent, (dfir.UnusedEndMarkerComponent, dfir.ConstantComponent)
                ):
                    continue

                dfir_type = (
                    port.data_type.type_ if isinstance(port.data_type, dftype.ArrayType) else port.data_type
                )
                base_hls_type = self.type_map[dfir_type]
                batch_type = self.batch_type_map[base_hls_type]
                param_type = HLSType(HLSBasicType.STREAM, sub_types=[batch_type])
                hls_func.params.append(HLSVar(var_name=port.name, var_type=param_type))
            self.hls_functions[comp.readable_id] = hls_func

        # --- *** 关键修正：使用更健壮的逻辑来声明所有中间流 *** ---
        # 2. Identify intermediate streams and add their declarations and pragmas
        all_stream_comp_ids = {f.dfir_comp.readable_id for f in self.hls_functions.values()}

        visited_ports = set()
        for port in comp_col.all_connected_ports:
            if port.readable_id in visited_ports:
                continue

            conn = port.connection

            # A connection is intermediate if it's between two components that will be turned into HLS functions.
            # This is simpler and more robust than the previous check.
            is_intermediate = (
                port.parent.readable_id in all_stream_comp_ids
                and conn.parent.readable_id in all_stream_comp_ids
            )

            if is_intermediate:
                dfir_type = (
                    port.data_type.type_ if isinstance(port.data_type, dftype.ArrayType) else port.data_type
                )
                base_hls_type = self.type_map[dfir_type]
                batch_type = self.batch_type_map[base_hls_type]
                stream_type = HLSType(HLSBasicType.STREAM, sub_types=[batch_type])

                # The stream is always named after the output port that produces the data.
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
            f.dfir_comp for f in self.hls_functions.values() if isinstance(f.dfir_comp, dfir.ReduceComponent)
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

    def _generate_mem_to_stream_func(self) -> HLSFunction:
        """Generates the function that reads from AXI pointers into streams."""
        func = HLSFunction("mem_to_stream_func", comp=None)

        params = []
        body = []

        for port in self.axi_input_ports:
            batch_type = self.batch_type_map[self.type_map[port.data_type]]

            axi_ptr_type = HLSType(HLSBasicType.POINTER, sub_types=[batch_type], is_const_ptr=True)
            axi_param = HLSVar(f"in_{port.unique_name}", axi_ptr_type)
            params.append(axi_param)

            stream_type = HLSType(HLSBasicType.STREAM, [batch_type])
            stream_param = HLSVar(f"out_{port.unique_name}_stream", stream_type)
            params.append(stream_param)

            loop = CodeFor(
                codes=[
                    CodePragma("PIPELINE"),
                    CodeWriteStream(
                        stream_param, HLSExpr(HLSExprT.VAR, HLSVar(f"{axi_param.name}[i]", batch_type))
                    ),
                ],
                iter_limit="num_batches",
                iter_name="i",
            )
            body.append(loop)

        num_batches_type = HLSType(HLSBasicType.UINT16)
        params.append(HLSVar("num_batches", num_batches_type))

        func.params = params
        func.codes = body
        return func

    def _generate_stream_to_mem_func(self) -> HLSFunction:
        """Generates the function that writes from streams to AXI pointers."""
        func = HLSFunction("stream_to_mem_func", comp=None)

        params = []
        body = []

        for port in self.axi_output_ports:
            batch_type = self.batch_type_map[self.type_map[port.data_type]]

            stream_param = HLSVar(f"in_{port.unique_name}_stream", HLSType(HLSBasicType.STREAM, [batch_type]))
            params.append(stream_param)

            host_output_type = self._get_host_output_type()

            axi_ptr_type = HLSType(HLSBasicType.POINTER, sub_types=[host_output_type], is_const_ptr=False)
            axi_param = HLSVar(f"out_{port.unique_name}", axi_ptr_type)
            params.append(axi_param)

            i_var = HLSVar("i", HLSType(HLSBasicType.INT))
            internal_batch_var = HLSVar("internal_batch", batch_type)
            output_batch_var = HLSVar("output_batch", host_output_type)

            kernel_output_data_type = host_output_type.sub_types[0].sub_types[0]

            conversion_loop = CodeFor(
                [
                    CodePragma("UNROLL"),
                    CodeAssign(
                        HLSVar(
                            f"{output_batch_var.name}.data[k].distance", kernel_output_data_type.sub_types[0]
                        ),
                        HLSExpr(
                            HLSExprT.VAR,
                            HLSVar(
                                f"(float){internal_batch_var.name}.data[k].ele_0", HLSType(HLSBasicType.FLOAT)
                            ),
                        ),
                    ),
                    CodeAssign(
                        HLSVar(f"{output_batch_var.name}.data[k].id", kernel_output_data_type.sub_types[1]),
                        HLSExpr(
                            HLSExprT.VAR,
                            HLSVar(f"{internal_batch_var.name}.data[k].ele_1.id", HLSType(HLSBasicType.INT)),
                        ),
                    ),
                ],
                "PE_NUM",
                iter_name="k",
            )
            while_body = [
                CodePragma("PIPELINE"),
                CodeIf(
                    HLSExpr(
                        HLSExprT.UOP,
                        UnaryOp.NOT,
                        [HLSExpr(HLSExprT.STREAM_EMPTY, None, [HLSExpr(HLSExprT.VAR, stream_param)])],
                    ),
                    if_codes=[
                        CodeAssign(
                            internal_batch_var,
                            HLSExpr(HLSExprT.STREAM_READ, None, [HLSExpr(HLSExprT.VAR, stream_param)]),
                        ),
                        CodeVarDecl(output_batch_var.name, output_batch_var.type),
                        conversion_loop,
                        CodeAssign(
                            HLSVar(f"{output_batch_var.name}.end_flag", HLSType(HLSBasicType.BOOL)),
                            HLSExpr(
                                HLSExprT.UOP,
                                (UnaryOp.GET_ATTR, "end_flag"),
                                [HLSExpr(HLSExprT.VAR, internal_batch_var)],
                            ),
                        ),
                        CodeAssign(
                            HLSVar(f"{output_batch_var.name}.end_pos", HLSType(HLSBasicType.UINT8)),
                            HLSExpr(
                                HLSExprT.UOP,
                                (UnaryOp.GET_ATTR, "end_pos"),
                                [HLSExpr(HLSExprT.VAR, internal_batch_var)],
                            ),
                        ),
                        CodeAssign(
                            HLSVar(f"out_{port.unique_name}[i]", host_output_type),
                            HLSExpr(HLSExprT.VAR, output_batch_var),
                        ),
                        CodeIf(
                            HLSExpr(
                                HLSExprT.VAR,
                                HLSVar(f"out_{port.unique_name}[i].end_flag", HLSType(HLSBasicType.BOOL)),
                            ),
                            [CodeBreak()],
                        ),
                        CodeAssign(
                            i_var,
                            HLSExpr(
                                HLSExprT.BINOP,
                                BinOp.ADD,
                                [HLSExpr(HLSExprT.VAR, i_var), HLSExpr(HLSExprT.CONST, 1)],
                            ),
                        ),
                    ],
                ),
            ]
            body.extend(
                [
                    CodeVarDecl(i_var.name, i_var.type, init_val=0),
                    CodeWhile(while_body, HLSExpr(HLSExprT.CONST, True)),
                ]
            )

        func.params = params
        func.codes = body
        return func

    def _generate_dataflow_core_func(self, top_func_name: str) -> HLSFunction:
        """Phase 4.3: Generates the implementation of the core dataflow function (the old top-level)."""
        # This function is largely the same as the old _generate_top_level_function
        # but with a new name and a signature composed of only streams.
        func = HLSFunction(f"{top_func_name}_dataflow", comp=None)

        params = []
        for port in self.axi_input_ports + self.axi_output_ports:
            batch_type = self.batch_type_map[self.type_map[port.data_type]]
            stream_type = HLSType(HLSBasicType.STREAM, [batch_type])
            params.append(HLSVar(f"{port.unique_name}_stream", stream_type))
        func.params = params

        # The body generation is the same as the original _generate_top_level_function
        func.codes = self._generate_top_level_function_body()  # Delegate body generation
        return func

    def _generate_axi_kernel_wrapper(self, top_func_name: str) -> Tuple[str, str]:
        """Generates the final extern "C" kernel with AXI pragmas."""
        params_str_list = []
        param_vars = []

        # 1. Build parameter lists for the top-level function signature
        for port in self.axi_input_ports:
            batch_type = self.batch_type_map[self.type_map[port.data_type]]
            axi_ptr_type = HLSType(HLSBasicType.POINTER, sub_types=[batch_type], is_const_ptr=True)
            axi_param = HLSVar(port.unique_name, axi_ptr_type)
            params_str_list.append(axi_param.type.get_upper_decl(axi_param.name))
            param_vars.append(axi_param)

        for port in self.axi_output_ports:
            host_output_type = self._get_host_output_type()
            axi_ptr_type = HLSType(HLSBasicType.POINTER, sub_types=[host_output_type], is_const_ptr=False)
            axi_param = HLSVar(port.unique_name, axi_ptr_type)
            params_str_list.append(axi_param.type.get_upper_decl(axi_param.name))
            param_vars.append(axi_param)

        params_str_list.append("int* stop_flag")
        param_vars.append(HLSVar("stop_flag", HLSType(HLSBasicType.POINTER, [HLSType(HLSBasicType.INT)])))

        params_str_list.append("uint16_t input_length_in_batches")
        param_vars.append(
            HLSVar("input_length_in_batches", HLSType(HLSBasicType.UINT8, struct_name="uint16_t"))
        )

        top_func_sig = (
            f'extern "C" void {top_func_name}(\n{INDENT_UNIT}'
            + f",\n{INDENT_UNIT}".join(params_str_list)
            + "\n)"
        )

        # 2. Generate Pragmas
        pragmas = []
        for i, var in enumerate(param_vars):
            bundle = f"gmem{i}"
            if "input_length" not in var.name:
                pragmas.append(f"#pragma HLS INTERFACE m_axi port={var.name} offset=slave bundle={bundle}")

        for var in param_vars:
            pragmas.append(f"#pragma HLS INTERFACE s_axilite port={var.name}")
        pragmas.append("#pragma HLS INTERFACE s_axilite port=return")

        # 3. Generate Body
        body = []

        # Declare internal streams
        internal_streams = []
        for port in self.top_level_io_ports:
            batch_type = self.batch_type_map[self.type_map[port.data_type]]
            stream_var = HLSVar(
                f"{port.unique_name}_internal_stream", HLSType(HLSBasicType.STREAM, [batch_type])
            )
            internal_streams.append(stream_var)

            decl = CodeVarDecl(stream_var.name, stream_var.type)
            # Use 'static' for streams inside a function to ensure they are properly implemented by HLS
            setattr(decl, "is_static", True)
            body.append(decl)
            body.append(CodePragma(f"STREAM variable={stream_var.name} depth={self.STREAM_DEPTH}"))

        body.append(CodePragma("DATAFLOW"))

        # --- *** 关键修正：使用更清晰的逻辑构建函数调用参数 *** ---
        # 4. Generate calls to sub-functions

        # Params for mem_to_stream_func
        m2s_axi_params = [p for p in param_vars if p.type.is_const_ptr]
        m2s_stream_params = [
            s for s in internal_streams if any(p.unique_name in s.name for p in self.axi_input_ports)
        ]
        m2s_len_param = [p for p in param_vars if "input_length" in p.name]
        body.append(CodeCall(self.mem_to_stream_func, m2s_axi_params + m2s_stream_params + m2s_len_param))

        # Params for dataflow_core_func
        df_core_input_streams = [
            s for s in internal_streams if any(p.unique_name in s.name for p in self.axi_input_ports)
        ]
        df_core_output_streams = [
            s for s in internal_streams if any(p.unique_name in s.name for p in self.axi_output_ports)
        ]
        body.append(CodeCall(self.dataflow_core_func, df_core_input_streams + df_core_output_streams))

        # Params for stream_to_mem_func
        s2m_stream_params = [
            s for s in internal_streams if any(p.unique_name in s.name for p in self.axi_output_ports)
        ]
        s2m_axi_params = [
            p
            for p in param_vars
            if p.type.type == HLSBasicType.POINTER and not p.type.is_const_ptr and "stop_flag" not in p.name
        ]
        body.append(CodeCall(self.stream_to_mem_func, s2m_stream_params + s2m_axi_params))

        # 5. Assemble final C++ string
        code = "\n".join(pragmas) + "\n"
        code += f"{top_func_sig} " + "{\n"
        # Manually handle static declaration for CodeVarDecl
        for line in body:
            if isinstance(line, CodeVarDecl) and getattr(line, "is_static", False):
                code += f"{INDENT_UNIT}static {line.gen_code().lstrip()}"
            else:
                code += line.gen_code(1)
        code += "}\n"

        return code, top_func_sig

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
            raise NotImplementedError(f"DFIR type conversion not implemented for {type(dfir_type)}")

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
                CodePragma(f"WARNING: Component {type(comp).__name__} translation not implemented.")
            ]

        # Wrap the core logic in the standard streaming boilerplate
        hls_func.codes = self._generate_streamed_function_boilerplate(hls_func, inner_logic)

    def _translate_unstreamed_component(self, hls_func: HLSFunction):
        """Translates a DFIR component for an unstreamed (pass-by-reference) function."""
        # This is a placeholder for now, as it's mainly for reduce sub-functions (key, transform, unit)
        hls_func.codes = [
            CodePragma("INLINE"),
            CodePragma(f"WARNING: Unstreamed func translation not fully implemented for {hls_func.name}"),
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
        if comp.op == UnaryOp.GET_ATTR:
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

    def _translate_scatter_op(self, comp: dfir.ScatterComponent, iterator: str) -> List[HLSCodeLine]:
        """Generates the core logic for a ScatterComponent."""
        in_port = comp.get_port("i_0")
        in_type = self.type_map[in_port.data_type]

        assignments = []
        for i, out_port in enumerate(comp.out_ports):
            if isinstance(out_port.connection.parent, dfir.UnusedEndMarkerComponent):
                continue
            out_type = self.type_map[out_port.data_type]

            ga_op = UnaryOp.GET_ATTR
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

    def _translate_conditional_op(self, comp: dfir.ConditionalComponent, iterator: str) -> List[HLSCodeLine]:
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
        in_batch_var = HLSVar("in_batch_i_0", self.batch_type_map[self.type_map[in_port.data_type]])
        out_batch_var = HLSVar("out_batch_o_0", self.batch_type_map[self.type_map[out_port.data_type]])
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
        out_idx_var_decl = CodeVarDecl("out_idx", out_idx_type, init_val=0)  # Initialize to 0
        while_loop_body.append(out_idx_var_decl)
        out_idx_var = out_idx_var_decl.var
        # The initialization is now part of the declaration

        # 4. Inner for loop for filtering
        in_elem_type = self.type_map[in_port.data_type]  # This is an Optional type
        out_elem_type = self.type_map[out_port.data_type]

        ga_op = UnaryOp.GET_ATTR
        # Part 1: in_batch_i_0.data[i].valid
        valid_check_expr = HLSExpr(
            HLSExprT.UOP,
            (ga_op, "valid"),
            [HLSExpr(HLSExprT.VAR, HLSVar(f"in_batch_i_0.data[i]", in_elem_type))],
        )

        # Part 2: i < in_batch_i_0.end_pos
        end_pos_check_expr = HLSExpr(
            HLSExprT.BINOP,
            BinOp.LT,
            [
                HLSExpr(HLSExprT.VAR, HLSVar("i", HLSType(HLSBasicType.UINT))),
                HLSExpr(HLSExprT.UOP, (ga_op, "end_pos"), [HLSExpr(HLSExprT.VAR, in_batch_var)]),
            ],
        )

        # Combined Condition: data.valid && i < end_pos
        cond_expr = HLSExpr(HLSExprT.BINOP, BinOp.AND, [valid_check_expr, end_pos_check_expr])

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
                BinOp.ADD,
                [HLSExpr(HLSExprT.VAR, out_idx_var), HLSExpr(HLSExprT.CONST, 1)],
            ),
        )

        if_block = CodeIf(cond_expr, [assign_data, increment_idx])
        for_loop = CodeFor(codes=[CodePragma("UNROLL"), if_block], iter_limit="PE_NUM", iter_name="i")
        while_loop_body.append(for_loop)

        # 5. Set output batch metadata and write to stream
        while_loop_body.append(
            CodeAssign(
                HLSVar(f"{out_batch_var.name}.end_pos", out_idx_type),
                HLSExpr(HLSExprT.VAR, out_idx_var),
            )
        )
        ga_op = UnaryOp.GET_ATTR
        end_flag_expr = HLSExpr(HLSExprT.UOP, (ga_op, "end_flag"), [HLSExpr(HLSExprT.VAR, in_batch_var)])
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
                if comp.op in [UnaryOp.GET_ATTR, UnaryOp.SELECT]:
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
                    ga_op = UnaryOp.GET_ATTR
                    sub_name = in_var.type.get_nth_subname(i)
                    expr = HLSExpr(HLSExprT.UOP, (ga_op, sub_name), [HLSExpr(HLSExprT.VAR, in_var)])
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
                valid_op = UnaryOp.GET_ATTR
                cond_expr = HLSExpr(HLSExprT.UOP, (valid_op, "valid"), [HLSExpr(HLSExprT.VAR, in_opt_var)])

                # Assignment: out_var = in_opt_var.data
                data_op = UnaryOp.GET_ATTR
                assign_expr = HLSExpr(HLSExprT.UOP, (data_op, "data"), [HLSExpr(HLSExprT.VAR, in_opt_var)])

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

    def _translate_reduce_preprocess_op(self, comp: dfir.ReduceComponent, iterator: str) -> List[HLSCodeLine]:
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
        code_lines.extend(self._inline_sub_graph_logic([key_sub_graph_start], key_sub_graph_end, key_io_map))

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
        for_loop = CodeFor(codes=[CodePragma("UNROLL")] + inner_logic, iter_limit="PE_NUM", iter_name="i")
        while_loop_body.append(for_loop)

        # 6. Copy metadata (end_flag, end_pos) from input batch to both output batches
        ga_op = UnaryOp.GET_ATTR
        end_flag_expr = HLSExpr(HLSExprT.UOP, (ga_op, "end_flag"), [HLSExpr(HLSExprT.VAR, in_batch_var)])
        end_pos_expr = HLSExpr(HLSExprT.UOP, (ga_op, "end_pos"), [HLSExpr(HLSExprT.VAR, in_batch_var)])

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

    def _translate_reduce_unit_reduce(self, hls_func: HLSFunction) -> List[HLSCodeLine]:
        """
        Generates the body for the second stage of Reduce (stateful accumulation).
        This version is updated to use a more performant circular buffer for draining.
        """
        body: List[HLSCodeLine] = []
        comp = hls_func.dfir_comp

        # 1. Get types and variables from the new function signature
        kt_streams, out_streams = hls_func.params
        kt_type = kt_streams.type.sub_types[0].sub_types[0]
        key_type = kt_streams.type.sub_types[0].sub_types[0].sub_types[0].sub_types[0]
        transform_type = kt_streams.type.sub_types[0].sub_types[0].sub_types[0].sub_types[1]
        single_out_stream_type = out_streams.type
        # The output from this unit is a batched stream
        out_batch_type = single_out_stream_type.sub_types[0]
        out_data_type = out_batch_type.sub_types[0].sub_types[0]

        bram_elem_type = self._to_hls_type(
            dftype.TupleType([comp.get_port("i_reduce_transform_out").data_type, dftype.BoolType()])
        )

        body.append(CodeComment("1. Stateful memories for PE_NUM parallel reduction units"))
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

        body.append(CodeComment("2. Memory initialization for all PEs"))
        uint_type = HLSType(HLSBasicType.UINT)
        max_num_var = HLSVar("MAX_NUM", uint_type)
        assign_val_expr = HLSExpr(
            HLSExprT.BINOP,
            BinOp.ADD,
            [HLSExpr(HLSExprT.VAR, max_num_var), HLSExpr(HLSExprT.CONST, 1)],
        )
        assign_ibuf = CodeAssign(HLSVar("i_buffer[pe][i]", uint_type), assign_val_expr)
        clear_ibuf_inner_loop = CodeFor([CodePragma("UNROLL"), assign_ibuf], "L + 1", iter_name="i")
        clear_ibuf_outer_loop = CodeFor(
            [CodePragma("UNROLL"), clear_ibuf_inner_loop], "PE_NUM", iter_name="pe"
        )
        body.append(clear_ibuf_outer_loop)

        target_valid_flag = HLSVar("key_mem[pe][i].ele_1", HLSType(HLSBasicType.BOOL))
        assign_valid_false = CodeAssign(target_valid_flag, HLSExpr(HLSExprT.CONST, False))
        clear_valid_inner_loop = CodeFor([CodePragma("UNROLL"), assign_valid_false], "MAX_NUM", iter_name="i")
        clear_valid_outer_loop = CodeFor(
            [CodePragma("UNROLL"), clear_valid_inner_loop], "PE_NUM", iter_name="pe"
        )
        body.append(clear_valid_outer_loop)

        body.append(CodeComment("3. Main processing loop for aggregation across PEs"))
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

        while_loop_body = [CodePragma("PIPELINE")]
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
        inner_loop_logic = self._translate_reduce_unit_inner_loop(
            comp, bram_elem_type, "i", key_elem_var, transform_elem_var
        )

        kt_stream_expr = HLSExpr(HLSExprT.VAR, HLSVar(f"{kt_streams.name}[i]", kt_type))
        read_kt = CodeAssign(kt_elem_var, HLSExpr(HLSExprT.STREAM_READ, None, [kt_stream_expr]))
        key_from_wrapper = HLSExpr(
            HLSExprT.UOP,
            (UnaryOp.GET_ATTR, "data.key"),
            [HLSExpr(HLSExprT.VAR, kt_elem_var)],
        )
        transform_from_wrapper = HLSExpr(
            HLSExprT.UOP,
            (UnaryOp.GET_ATTR, "data.transform"),
            [HLSExpr(HLSExprT.VAR, kt_elem_var)],
        )
        read_key = CodeAssign(key_elem_var, key_from_wrapper)
        read_transform = CodeAssign(transform_elem_var, transform_from_wrapper)
        end_flag_from_wrapper = HLSExpr(
            HLSExprT.UOP,
            (UnaryOp.GET_ATTR, "end_flag"),
            [HLSExpr(HLSExprT.VAR, kt_elem_var)],
        )
        end_flag_in_array = HLSVar(f"{all_end_flag_var.name}[i]", end_flag_var.type)

        process_logic = CodeIf(
            expr=end_flag_from_wrapper,
            if_codes=[CodeAssign(end_flag_in_array, end_flag_from_wrapper)],
            else_codes=[read_key, read_transform] + inner_loop_logic,
        )
        read_and_process_block = CodeIf(
            expr=HLSExpr(
                HLSExprT.BINOP,
                BinOp.AND,
                [
                    HLSExpr(HLSExprT.UOP, UnaryOp.NOT, [HLSExpr(HLSExprT.VAR, end_flag_in_array)]),
                    HLSExpr(
                        HLSExprT.UOP, UnaryOp.NOT, [HLSExpr(HLSExprT.STREAM_EMPTY, None, [kt_stream_expr])]
                    ),
                ],
            ),
            if_codes=[read_kt, process_logic],
        )
        pe_processing_loop = CodeFor(
            codes=[CodePragma("UNROLL"), read_and_process_block],
            iter_limit="PE_NUM",
            iter_name="i",
        )
        while_loop_body.append(pe_processing_loop)

        # Check for end condition (all PEs must have seen an end flag)
        check_end_flag_logic = [CodeAssign(end_flag_var, HLSExpr(HLSExprT.CONST, True))]
        and_comp_vars = [
            HLSExpr(HLSExprT.VAR, end_flag_var),
            HLSExpr(HLSExprT.VAR, end_flag_in_array),
        ]
        end_flag_agg_loop = CodeFor(
            codes=[
                CodePragma("UNROLL"),
                CodeAssign(
                    end_flag_var,
                    HLSExpr(HLSExprT.BINOP, BinOp.AND, and_comp_vars),
                ),
            ],
            iter_limit="PE_NUM",
            iter_name="i",
        )
        check_end_flag_logic.append(end_flag_agg_loop)
        check_end_flag_logic.append(CodeIf(HLSExpr(HLSExprT.VAR, end_flag_var), [CodeBreak()]))
        while_loop_body.extend(check_end_flag_logic)

        body.append(CodeWhile(codes=while_loop_body, iter_expr=HLSExpr(HLSExprT.CONST, True)))

        body.append(CodeComment("4. Final output loop to drain all PE memories with swapped loops"))

        # New variable declarations for circular buffer draining
        cnt_var = HLSVar("data_cnt", HLSType(HLSBasicType.UINT))
        body.append(CodeVarDecl(cnt_var.name, cnt_var.type, init_val=0))
        body.append(CodeAssign(cnt_var, HLSExpr(HLSExprT.CONST, 0)))

        start_pos_var = HLSVar("start_pos", HLSType(HLSBasicType.UINT))
        body.append(CodeVarDecl(start_pos_var.name, start_pos_var.type, init_val=0))
        body.append(CodeAssign(start_pos_var, HLSExpr(HLSExprT.CONST, 0)))

        data_pack_var = HLSVar("data_pack", out_batch_type)
        body.append(CodeVarDecl(data_pack_var.name, data_pack_var.type))
        body.append(
            CodeAssign(
                HLSVar(f"{data_pack_var.name}.end_flag", HLSType(HLSBasicType.BOOL)),
                HLSExpr(HLSExprT.CONST, False),
            )
        )

        drain_buffer_type = HLSType(
            HLSBasicType.ARRAY, sub_types=[out_data_type], array_dims=[f"((PE_NUM << 1))"]
        )
        drain_buffer_var = HLSVar("data_to_write", drain_buffer_type)
        body.append(CodeVarDecl(drain_buffer_var.name, drain_buffer_var.type))
        body.append(CodePragma(f"ARRAY_PARTITION variable={drain_buffer_var.name} complete dim=0"))

        # Since CodeFor doesn't support custom increments, we simulate it with a while loop
        k_var = HLSVar("k", HLSType(HLSBasicType.UINT))
        body.append(CodeVarDecl(k_var.name, k_var.type, init_val=0))
        body.append(CodeAssign(k_var, HLSExpr(HLSExprT.CONST, 0)))

        # Inner logic of the tiled draining loop
        pe_var = HLSVar("pe", HLSType(HLSBasicType.UINT))
        k_plus_pe_expr = HLSExpr(
            HLSExprT.BINOP,
            BinOp.ADD,
            [HLSExpr(HLSExprT.VAR, k_var), HLSExpr(HLSExprT.VAR, pe_var)],
        )

        is_valid_expr = HLSExpr(
            HLSExprT.VAR,
            HLSVar(f"key_mem[pe][{k_plus_pe_expr.code}].ele_1", HLSType(HLSBasicType.BOOL)),
        )
        data_from_mem = HLSExpr(
            HLSExprT.VAR, HLSVar(f"key_mem[pe][{k_plus_pe_expr.code}].ele_0", out_data_type)
        )

        write_to_buffer_expr = HLSExpr(
            HLSExprT.BINOP,
            BinOp.MOD,
            [HLSExpr(HLSExprT.VAR, start_pos_var), HLSExpr(HLSExprT.CONST, "((PE_NUM << 1))")],
        )
        assign_to_buffer = CodeAssign(
            HLSVar(f"{drain_buffer_var.name}[{write_to_buffer_expr.code}]", out_data_type),
            data_from_mem,
        )

        increment_cnt = CodeAssign(
            cnt_var,
            HLSExpr(
                HLSExprT.BINOP,
                BinOp.ADD,
                [HLSExpr(HLSExprT.VAR, cnt_var), HLSExpr(HLSExprT.CONST, 1)],
            ),
        )
        increment_pos = CodeAssign(
            start_pos_var,
            HLSExpr(
                HLSExprT.BINOP,
                BinOp.ADD,
                [HLSExpr(HLSExprT.VAR, start_pos_var), HLSExpr(HLSExprT.CONST, 1)],
            ),
        )

        if_valid_block = CodeIf(is_valid_expr, [assign_to_buffer, increment_cnt, increment_pos])

        drain_inner_pe_loop = CodeFor(
            codes=[CodePragma("UNROLL"), if_valid_block], iter_limit="PE_NUM", iter_name="pe"
        )

        # Logic to pack and write a full batch
        pack_and_write_logic = []
        pack_and_write_logic.append(
            CodeAssign(
                HLSVar(f"{data_pack_var.name}.end_pos", HLSType(HLSBasicType.UINT8)),
                HLSExpr(HLSExprT.CONST, "PE_NUM"),
            )
        )

        i_var = HLSVar("i", HLSType(HLSBasicType.UINT))

        read_from_buffer_offset = HLSExpr(
            HLSExprT.BINOP,
            BinOp.ADD,
            [
                HLSExpr(
                    HLSExprT.BINOP,
                    BinOp.SUB,
                    [HLSExpr(HLSExprT.VAR, start_pos_var), HLSExpr(HLSExprT.VAR, cnt_var)],
                ),
                HLSExpr(HLSExprT.VAR, i_var),
            ],
        )
        read_from_buffer_expr = HLSExpr(
            HLSExprT.BINOP,
            BinOp.MOD,
            [read_from_buffer_offset, HLSExpr(HLSExprT.CONST, "(PE_NUM << 1)")],
        )

        packing_loop_body = [
            CodePragma("UNROLL"),
            CodeAssign(
                HLSVar(f"{data_pack_var.name}.data[i]", out_data_type),
                HLSExpr(
                    HLSExprT.VAR,
                    HLSVar(f"{drain_buffer_var.name}[{read_from_buffer_expr.code}]", out_data_type),
                ),
            ),
        ]
        packing_loop = CodeFor(packing_loop_body, iter_limit="PE_NUM", iter_name="i")
        pack_and_write_logic.append(packing_loop)
        pack_and_write_logic.append(CodeWriteStream(out_streams, data_pack_var))
        pack_and_write_logic.append(
            CodeAssign(
                cnt_var,
                HLSExpr(
                    HLSExprT.BINOP,
                    BinOp.SUB,
                    [HLSExpr(HLSExprT.VAR, cnt_var), HLSExpr(HLSExprT.CONST, "PE_NUM")],
                ),
            )
        )

        if_data_full = CodeIf(
            HLSExpr(
                HLSExprT.BINOP,
                BinOp.GE,
                [HLSExpr(HLSExprT.VAR, cnt_var), HLSExpr(HLSExprT.CONST, "PE_NUM")],
            ),
            pack_and_write_logic,
        )

        # Main draining while loop
        drain_while_body = [
            CodePragma("PIPELINE"),
            drain_inner_pe_loop,
            if_data_full,
            CodeAssign(
                k_var,
                HLSExpr(
                    HLSExprT.BINOP,
                    BinOp.ADD,
                    [HLSExpr(HLSExprT.VAR, k_var), HLSExpr(HLSExprT.CONST, "PE_NUM")],
                ),
            ),
        ]
        body.append(
            CodeWhile(
                codes=drain_while_body,
                iter_expr=HLSExpr(
                    HLSExprT.BINOP,
                    BinOp.LT,
                    [HLSExpr(HLSExprT.VAR, k_var), HLSExpr(HLSExprT.CONST, "MAX_NUM")],
                ),
            )
        )

        body.append(CodeComment("5. Drain any remaining data and send final batch with end_flag"))

        # Final packing logic
        final_pack_logic = []
        final_pack_logic.append(
            CodeAssign(
                HLSVar(f"{data_pack_var.name}.end_flag", HLSType(HLSBasicType.BOOL)),
                HLSExpr(HLSExprT.CONST, True),
            )
        )
        final_pack_logic.append(
            CodeAssign(
                HLSVar(f"{data_pack_var.name}.end_pos", HLSType(HLSBasicType.UINT8)),
                HLSExpr(HLSExprT.VAR, cnt_var),
            )
        )
        d_buffer_sub = HLSVar(f"{drain_buffer_var.name}[{read_from_buffer_expr.code}]", out_data_type)
        data_assign = CodeAssign(
            HLSVar(f"{data_pack_var.name}.data[i]", out_data_type),
            HLSExpr(HLSExprT.VAR, d_buffer_sub),
        )
        lt_cmp_vars = [HLSExpr(HLSExprT.VAR, i_var), HLSExpr(HLSExprT.VAR, cnt_var)]
        final_packing_loop_body = [
            CodePragma("UNROLL"),
            CodeIf(
                HLSExpr(HLSExprT.BINOP, BinOp.LT, lt_cmp_vars),
                [data_assign],
            ),
        ]
        final_packing_loop = CodeFor(final_packing_loop_body, iter_limit="PE_NUM", iter_name="i")

        final_pack_logic.append(final_packing_loop)
        final_pack_logic.append(CodeWriteStream(out_streams, data_pack_var))

        body.extend(final_pack_logic)

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
                HLSExpr(HLSExprT.VAR, HLSVar(f"key_mem[{pe_idx}][{key_var.name}]", bram_elem_type)),
            )
        )

        # 3. Buffer management for this PE
        buffer_elem_expr = HLSExpr(
            HLSExprT.VAR, HLSVar(f"i_buffer[{pe_idx}][i_search]", HLSType(HLSBasicType.UINT))
        )
        if_condition = HLSExpr(HLSExprT.BINOP, BinOp.EQ, [HLSExpr(HLSExprT.VAR, key_var), buffer_elem_expr])
        value_to_assign = HLSExpr(HLSExprT.VAR, HLSVar(f"key_buffer[{pe_idx}][i_search]", bram_elem_type))
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
        key_buffer_src = HLSExpr(HLSExprT.VAR, HLSVar(f"key_buffer[{pe_idx}][i_move + 1]", bram_elem_type))
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
            HLSExprT.UOP, (UnaryOp.GET_ATTR, "ele_1"), [HLSExpr(HLSExprT.VAR, old_ele_var)]
        )
        if_codes = [
            CodeAssign(HLSVar(f"{new_ele_var.name}.ele_1", bool_type), HLSExpr(HLSExprT.CONST, True)),
            CodeAssign(HLSVar(f"{new_ele_var.name}.ele_0", accum_type), HLSExpr(HLSExprT.VAR, val_var)),
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
                    (UnaryOp.GET_ATTR, "ele_0"),
                    [HLSExpr(HLSExprT.VAR, old_ele_var)],
                ),
            ),
            *self._inline_sub_graph_logic(unit_starts, unit_end, io_map),
            CodeAssign(HLSVar(f"{new_ele_var.name}.ele_1", bool_type), HLSExpr(HLSExprT.CONST, True)),
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

    def _get_host_output_type(self) -> HLSType:
        """
        A helper function to define and return the host-friendly output batch type.
        This centralizes the definition of KernelOutputBatch and registers the types.
        """
        kernel_output_data_type = HLSType(
            HLSBasicType.STRUCT,
            sub_types=[HLSType(HLSBasicType.REAL_FLOAT), HLSType(HLSBasicType.INT)],
            struct_prop_names=["distance", "id"],
            struct_name="KernelOutputData",
        )
        if kernel_output_data_type.name not in self.struct_definitions:
            self.struct_definitions[kernel_output_data_type.name] = (
                kernel_output_data_type,
                ["distance", "id"],
            )

        output_data_array_type = HLSType(
            HLSBasicType.ARRAY, sub_types=[kernel_output_data_type], array_dims=["PE_NUM"]
        )
        host_output_type = HLSType(
            HLSBasicType.STRUCT,
            sub_types=[output_data_array_type, HLSType(HLSBasicType.BOOL), HLSType(HLSBasicType.UINT8)],
            struct_prop_names=["data", "end_flag", "end_pos"],
            struct_name="KernelOutputBatch",
        )
        if host_output_type.name not in self.struct_definitions:
            self.struct_definitions[host_output_type.name] = (
                host_output_type,
                ["data", "end_flag", "end_pos"],
            )

        return host_output_type

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

    def _generate_top_level_function_body(self) -> List[HLSCodeLine]:
        """Generates the implementation of the dataflow core function body,
        based on the proven logic from the original implementation."""
        from collections import defaultdict

        body: List[HLSCodeLine] = [CodePragma("DATAFLOW")]

        # 1. Declare all intermediate streams
        for decl, pragma in self.top_level_stream_decls:
            body.append(decl)
            if pragma:
                body.append(pragma)

        # 2. Prepare maps for I/O and intermediate streams
        stream_map = {decl.var.name: decl.var for decl, _ in self.top_level_stream_decls}
        top_io_map: Dict[int, HLSVar] = {}

        # --- *** 关键修正：在此处重新发现顶层IO端口，确保状态最新 *** ---
        # This logic is taken directly from the user-provided working code.
        top_level_inputs = []
        for comp in self.comp_col_store.components:
            if isinstance(comp, dfir.IOComponent) and comp.io_type == dfir.IOComponent.IOType.INPUT:
                if comp.get_port("o_0").connected:
                    top_level_inputs.append(comp.get_port("o_0").connection)
        top_level_outputs = self.comp_col_store.outputs

        current_top_level_io_ports = top_level_inputs + top_level_outputs
        # ----------------------------------------------------------------

        # Populate top_io_map using the fresh list of ports
        for p in current_top_level_io_ports:
            # Note: For ArrayType, we need its inner type for batching.
            dfir_type = p.data_type.type_ if isinstance(p.data_type, dftype.ArrayType) else p.data_type
            batch_type = self.batch_type_map[self.type_map[dfir_type]]
            top_io_map[p.readable_id] = HLSVar(
                f"{p.unique_name}_stream", HLSType(HLSBasicType.STREAM, [batch_type])
            )

        # 3. Topologically sort functions (using the user-provided, proven logic)
        stream_funcs = [f for f in self.hls_functions.values() if f.streamed]
        id_to_func = {f.readable_id: f for f in stream_funcs}
        comp_to_func = {f.dfir_comp: f for f in stream_funcs}  # This is safe within this function's context
        reduce_comp_to_pre = {}

        adj = defaultdict(list)
        in_degree = {f.readable_id: 0 for f in stream_funcs}

        for func in stream_funcs:
            if isinstance(func.dfir_comp, dfir.ReduceComponent) and "pre_process" in func.name:
                in_port = func.dfir_comp.get_port("i_0")
                if in_port.connected and in_port.connection.parent in comp_to_func:
                    adj[comp_to_func[in_port.connection.parent].readable_id].append(func.readable_id)
                    in_degree[func.readable_id] += 1
                reduce_comp_to_pre[func.dfir_comp] = func

        for func in stream_funcs:
            comp = func.dfir_comp
            if isinstance(comp, dfir.ReduceComponent):
                if "pre_process" in func.name:
                    continue
                else:  # unit_reduce
                    adj[reduce_comp_to_pre[comp].readable_id].append(func.readable_id)
                    in_degree[func.readable_id] += 1
                    continue
            for port in comp.in_ports:
                if port.connected:
                    predecessor_comp = port.connection.parent
                    if predecessor_comp in comp_to_func:
                        # For ReduceComponent, comp_to_func correctly points to unit_reduce, which produces the output
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
            raise RuntimeError("A cycle was detected in the top-level dataflow graph.")

        # 4. Generate calls (using the user-provided, proven logic)
        body.append(CodeComment("--- Function Calls (in topological order) ---"))
        handled_unit_reduce_ids = set()
        for func in sorted_funcs:
            comp = func.dfir_comp
            if "unit_reduce" in func.name and comp.readable_id in handled_unit_reduce_ids:
                continue

            if "pre_process" in func.name and isinstance(comp, dfir.ReduceComponent):
                comp_id = comp.readable_id
                helpers = self.reduce_helpers[comp_id]
                streams = helpers["streams"]
                unit_reduce_func = helpers["unit_reduce"]

                body.append(CodeComment(f"--- Start of Reduce Super-Block for {comp.name} ---"))

                in_port = comp.get_port("i_0")
                pred_port = in_port.connection
                in_stream_var = (
                    top_io_map[in_port.readable_id]
                    if isinstance(pred_port.parent, dfir.IOComponent)
                    else stream_map[f"stream_{pred_port.unique_name}"]
                )

                pre_process_call_params = [
                    in_stream_var,
                    streams["intermediate_key"],
                    streams["intermediate_transform"],
                ]
                body.append(CodeCall(func, pre_process_call_params))

                body.append(
                    CodeCall(
                        helpers["zipper"],
                        [
                            streams["intermediate_key"],
                            streams["intermediate_transform"],
                            streams["zipper_to_demux"],
                        ],
                    )
                )
                body.append(
                    CodeCall(helpers["demux"], [streams["zipper_to_demux"], streams["demux_to_omega"]])
                )
                body.append(CodeCall(helpers["omega"], [streams["demux_to_omega"], streams["omega_to_unit"]]))

                out_port = unit_reduce_func.dfir_comp.get_port("o_0")
                out_stream_var = (
                    top_io_map[out_port.readable_id]
                    if out_port.connection is None
                    else stream_map[f"stream_{out_port.unique_name}"]
                )

                body.append(CodeCall(unit_reduce_func, [streams["omega_to_unit"], out_stream_var]))

                body.append(CodeComment(f"--- End of Reduce Super-Block for {comp.name} ---"))
                handled_unit_reduce_ids.add(comp_id)

            elif not isinstance(comp, dfir.ReduceComponent):
                call_params: List[HLSVar] = []
                for func_param in func.params:
                    port = comp.get_port(func_param.name)

                    if port.port_type == dfir.PortType.IN:
                        predecessor_port = port.connection
                        if isinstance(predecessor_port.parent, dfir.IOComponent):
                            call_params.append(top_io_map[port.readable_id])
                        else:
                            call_params.append(stream_map[f"stream_{predecessor_port.unique_name}"])
                    else:  # OUT port
                        if port.connection is None:
                            call_params.append(top_io_map[port.readable_id])
                        else:
                            call_params.append(stream_map[f"stream_{port.unique_name}"])
                body.append(CodeCall(func, call_params))

        return body

    def _generate_source_file(self, header_name: str, axi_wrapper_func_str: str) -> str:
        """Generates the full content of the .cpp source file."""
        code = f'#include "{header_name}"\n\n'

        # Generate AXI helper functions
        for func in [self.mem_to_stream_func, self.stream_to_mem_func, self.dataflow_core_func]:
            if func:
                # Manually construct signature for static functions
                params_str_list = []
                for p in func.params:
                    # Special handling for pointer names in m2s/s2m
                    if "*" in p.type.name:
                        params_str_list.append(p.type.name.replace("*", f" *{p.name}"))
                    elif "uint16_t" in p.type.name:
                        params_str_list.append(f"uint16_t {p.name}")
                    else:
                        params_str_list.append(p.type.get_upper_param(p.name, True))

                params_str = ", ".join(params_str_list)
                code += f"static void {func.name}({params_str}) " + "{\n"
                code += "".join([line.gen_code(1) for line in func.codes])
                code += "}\n\n"

        if self.utility_functions:
            code += "// --- Utility Network Functions ---\n"
            for func in self.utility_functions:
                params_str = ", ".join(
                    [p.type.get_upper_param(p.name, p.type.type != HLSBasicType.INT) for p in func.params]
                )
                code += f"void {func.name}({params_str}) " + "{\n"
                code += "".join([line.gen_code(1) for line in func.codes])
                code += "}\n\n"

        # Generate implementations for all HLS component functions
        code += "// --- DFIR Component Functions ---\n"
        for func in self.hls_functions.values():
            params_str = ", ".join([p.type.get_upper_param(p.name, True) for p in func.params])
            code += f"void {func.name}({params_str}) " + "{\n"
            code += "".join([line.gen_code(1) for line in func.codes])
            code += "}\n\n"

        # Generate top-level AXI kernel wrapper
        code += axi_wrapper_func_str

        return code
