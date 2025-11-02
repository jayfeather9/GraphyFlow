import shutil
from pathlib import Path
from typing import Any, Dict,List, Optional
import copy
import sys
from .global_graph import GlobalGraph
from .dataflow_ir import ComponentCollection
# from .backend_manager import BackendManager
from .newbackend_manager import BackendManager


# --- CONFIGURATION SECTION ---
# You can change the number of kernels and their HBM mapping here.
NUM_BIG_KERNELS = 2
NUM_LITTLE_KERNELS = 1

# HBM channel IDs for Big Kernels. The list length must match NUM_BIG_KERNELS.
big_kernel_hbm_edge_id = [28,30]
big_kernel_hbm_node_id = [11,12]

little_kernel_hbm_edge_id = [0]
little_kernel_hbm_node_id = [9]

big_kernel_slr = ["SLR1", "SLR2"]
little_kernel_slr = ["SLR0"]
apply_kernel_slr = ["SLR0", "SLR1","SLR2"]
hbm_writer_slr = ["SLR0","SLR1","SLR2"]

# --- END CONFIGURATION SECTION ---


def _copy_and_template(src: Path, dest: Path, replacements: Dict[str, str]):
    """Reads a file, replaces placeholders, and writes to a new location."""
    content = src.read_text()
    for placeholder, value in replacements.items():
        content = content.replace(placeholder, value)
    dest.write_text(content)

def _generate_hbm_writer(n: int, file_path: Path) -> None:
    """
    Generates HLS C++ code for an hbm_writer kernel with 'n' parallel channels
    and appends it to the specified file.

    Args:
        n: The number of parallel channels to generate.
        file_path: The pathlib.Path object of the file to append the code to.
    """
    if n <= 0:
        print(f"Error: n must be greater than 0. No code written to {file_path}", file=sys.stderr)
        return

    # Use a list to build the code string efficiently
    code = []

    # --- 1. Function Signature ---
    code.append("extern \"C\" void\n")
    code.append(f"hbm_writer(\n") # Changed name to be unique for n
    
    args = []
    # Pointers (node_props_ and output_)
    for i in range(1, n + 1):
        args.append(f"    bus_word_t *node_props_{i}")
    for i in range(1, n + 1):
        args.append(f"    bus_word_t *output_{i}")
    
    # Scalars (dst_num_)
    for i in range(1, n + 1):
        args.append(f"    uint32_t dst_num_{i}")

    # Streams
    for i in range(1, n + 1):
        args.append(f"    hls::stream<cacheline_request_pkt_t> &cacheline_req_stream_{i}")
    for i in range(1, n + 1):
        args.append(f"    hls::stream<cacheline_response_pkt_t> &cacheline_resp_stream_{i}")
    for i in range(1, n + 1):
        args.append(f"    hls::stream<cacheline_data_pkt_t> &cacheline_data_stream_{i}")
    for i in range(1, n + 1):
        # Last argument set for this channel
        args.append(f"    hls::stream<write_burst_pkt_t> &write_burst_stream_{i}")

    # Join all arguments with commas
    code.append(",\n".join(args))
    code.append("\n) {\n")

    # --- 2. Interface Pragmas (m_axi) ---
    code.append("\n    // --- Interface Pragmas (m_axi) ---\n")
    code.append("    // Map pointers to separate AXI memory interfaces (gmem1...gmemN),\n")
    code.append("    // allowing for parallel access to different HBM banks.\n")
    for i in range(1, n + 1):
        code.append(f"#pragma HLS INTERFACE m_axi port = node_props_{i} offset = slave bundle = gmem{i}\n")
        code.append(f"#pragma HLS INTERFACE m_axi port = output_{i}     offset = slave bundle = gmem{i}\n\n")

    # --- 3. Interface Pragmas (s_axilite) ---
    code.append("    // --- Interface Pragmas (s_axilite) ---\n")
    code.append("    // All scalar arguments and pointer addresses are mapped to a single control bus.\n")
    
    # Pointers
    for i in range(1, n + 1):
        code.append(f"#pragma HLS INTERFACE s_axilite port = node_props_{i} bundle = control\n")
    for i in range(1, n + 1):
        code.append(f"#pragma HLS INTERFACE s_axilite port = output_{i}     bundle = control\n")
    
    # Scalars
    for i in range(1, n + 1):
        code.append(f"#pragma HLS INTERFACE s_axilite port = dst_num_{i}      bundle = control\n")
    
    # Return
    code.append("#pragma HLS INTERFACE s_axilite port = return         bundle = control\n")

    # --- 4. Dataflow Pragma ---
    code.append("\n    // --- Dataflow Pragma ---\n")
    code.append("    // This pragma enables task-level parallelism.\n")
    code.append("#pragma HLS DATAFLOW\n\n")

    # --- 5. Function Instantiations ---
    code.append("    // --- Function Instantiations ---\n")
    code.append(f"    // Instantiate the processing logic for each of the {n} parallel channels.\n")
    code.append("    // The first argument (0...N-1) is a constant integer used by HLS to create\n")
    code.append("    // distinct hardware instances of each function.\n\n")

    # node_property_loader instances
    for i in range(1, n + 1):
        template_id = i - 1  # Create IDs 0, 1, 2, ... n-1
        code.append(f"    node_property_loader({template_id}, node_props_{i}, dst_num_{i}, cacheline_req_stream_{i},\n")
        code.append(f"                         cacheline_resp_stream_{i}, cacheline_data_stream_{i});\n")
    
    code.append("\n") # Spacer

    # write_out instances
    for i in range(1, n + 1):
        template_id = i - 1 # Create IDs 0, 1, 2, ... n-1
        code.append(f"    write_out({template_id}, output_{i}, dst_num_{i}, write_burst_stream_{i});\n")

    # --- 6. Closing Brace ---
    code.append("}\n")

    # --- 7. Generate String and Append to File ---
    final_code = "".join(code)

    try:
        # Open the file in append mode ('a')
        with open(file_path, 'a', encoding='utf-8') as f:
            f.write(f"\n\n// --- Automatically generated HLS kernel for n={n} ---\n")
            f.write(final_code)
    except Exception as e:
        print(f"error: {e}", file=sys.stderr)

def _create_cfg(dest: Path, kernel_name: str):
    """
    根据 *全局配置变量* 动态生成 Vitis 连接配置文件，
    严格仿照示例文件的格式和注释。

    Args:
        dest (Path): 要写入配置文件的目标路径。
        kernel_name (str): 内核的基本名称 (未使用, 但保留接口)。
    """

    # --- 2. 派生配置和验证 ---
    # (函数将直接读取上面定义的全局变量)
    TOTAL_APPLY_KERNELS = NUM_BIG_KERNELS + NUM_LITTLE_KERNELS

    try:
        # 验证 Big Kernels
        assert len(big_kernel_hbm_edge_id) == NUM_BIG_KERNELS
        assert len(big_kernel_hbm_node_id) == NUM_BIG_KERNELS
        assert len(big_kernel_slr) == NUM_BIG_KERNELS

        # 验证 Little Kernels
        assert len(little_kernel_hbm_edge_id) == NUM_LITTLE_KERNELS
        assert len(little_kernel_hbm_node_id) == NUM_LITTLE_KERNELS
        assert len(little_kernel_slr) == NUM_LITTLE_KERNELS
        
        # 验证合并后的列表
        assert len(apply_kernel_slr) == TOTAL_APPLY_KERNELS
        assert len(hbm_writer_slr) == TOTAL_APPLY_KERNELS
        
    except AssertionError as e:
        print(f"配置错误: 全局列表长度与 NUM_... 变量不匹配。", file=sys.stderr)
        print(f"错误详情: {e}", file=sys.stderr)
        return

    # --- 3. 字符串构建 ---
    content = []
    content.append("[connectivity]")

    # --- 1. Kernel Instantiation (nk) ---
    content.append("\n# --- 1. Kernel Instantiation (nk) ---")
    content.append(f"nk=graphyflow_little:{NUM_LITTLE_KERNELS}")
    content.append(f"nk=graphyflow_big:{NUM_BIG_KERNELS}")
    content.append(f"nk=hbm_writer_little:{NUM_LITTLE_KERNELS}")
    content.append(f"nk=hbm_writer_big:{NUM_BIG_KERNELS}")
    content.append(f"nk=apply_kernel:{TOTAL_APPLY_KERNELS}")

    # --- 2. HBM Port Mapping (sp) ---
    content.append("\n# --- 2. HBM Port Mapping (sp) ---")

    # -- graphyflow_little (edge_props) --
    for i in range(NUM_LITTLE_KERNELS):
        instance_num = i + 1
        hbm_id = little_kernel_hbm_edge_id[i]
        content.append(f"\n# -- Mapping for instance: graphyflow_little_{instance_num} --")
        content.append(f"sp=graphyflow_little_{instance_num}.edge_props:HBM[{hbm_id}]")

    # -- graphyflow_big (edge_props) --
    for i in range(NUM_BIG_KERNELS):
        instance_num = i + 1
        hbm_id = big_kernel_hbm_edge_id[i]
        content.append(f"\n# -- Mapping for instance: graphyflow_big_{instance_num} --")
        content.append(f"sp=graphyflow_big_{instance_num}.edge_props:HBM[{hbm_id}]")

    # -- hbm_writer_little & apply_kernel --
    for i in range(NUM_LITTLE_KERNELS):
        instance_num = i + 1
        apply_instance_num = instance_num # apply_kernel 实例 1, 2, ...
        hbm_id = little_kernel_hbm_node_id[i]
        content.append(f"\n# -- Mapping for instance: hbm_writer_little_{instance_num} --")
        content.append(f"sp=hbm_writer_little_{instance_num}.node_props:HBM[{hbm_id}]")
        content.append(f"sp=hbm_writer_little_{instance_num}.output:HBM[{hbm_id}]")
        content.append(f"sp=apply_kernel_{apply_instance_num}.node_props:HBM[{hbm_id}]")

    # -- hbm_writer_big & apply_kernel --
    for i in range(NUM_BIG_KERNELS):
        instance_num = i + 1
        apply_instance_num = instance_num + NUM_LITTLE_KERNELS 
        hbm_id = big_kernel_hbm_node_id[i]
        content.append(f"\n# -- Mapping for instance: hbm_writer_big_{instance_num} --")
        content.append(f"sp=hbm_writer_big_{instance_num}.node_props:HBM[{hbm_id}]")
        content.append(f"sp=hbm_writer_big_{instance_num}.output:HBM[{hbm_id}]")
        content.append(f"sp=apply_kernel_{apply_instance_num}.node_props:HBM[{hbm_id}]")

    # --- 3. Stream Connections ---
    content.append("\n# --- 3. Stream Connections ---")

    # -- Stream connections for little --
    for i in range(NUM_LITTLE_KERNELS):
        instance_num = i + 1
        apply_instance_num = instance_num
        content.append(f"\n# -- Stream connections for graphyflow_little_{instance_num} and apply_kernel_{apply_instance_num} --")
        content.append(f"stream_connect=graphyflow_little_{instance_num}.ppb_req_stream:hbm_writer_little_{instance_num}.ppb_req_stream:16")
        content.append(f"stream_connect=hbm_writer_little_{instance_num}.ppb_resp_stream:graphyflow_little_{instance_num}.ppb_resp_stream:16")
        content.append(f"stream_connect=graphyflow_little_{instance_num}.kernel_out_stream:apply_kernel_{apply_instance_num}.kernel_out_stream:16")
        content.append(f"stream_connect=apply_kernel_{apply_instance_num}.write_burst_stream:hbm_writer_little_{instance_num}.write_burst_stream:16")

    # -- Stream connections for big --
    for i in range(NUM_BIG_KERNELS):
        instance_num = i + 1
        apply_instance_num = instance_num + NUM_LITTLE_KERNELS
        content.append(f"\n# -- Stream connections for graphyflow_big_{instance_num} and apply_kernel_{apply_instance_num} --")
        content.append(f"stream_connect=graphyflow_big_{instance_num}.cacheline_req_stream:hbm_writer_big_{instance_num}.cacheline_req_stream:16")
        content.append(f"stream_connect=hbm_writer_big_{instance_num}.cacheline_resp_stream:graphyflow_big_{instance_num}.cacheline_resp_stream:16")
        content.append(f"stream_connect=graphyflow_big_{instance_num}.kernel_out_stream:apply_kernel_{apply_instance_num}.kernel_out_stream:16")
        content.append(f"stream_connect=apply_kernel_{apply_instance_num}.write_burst_stream:hbm_writer_big_{instance_num}.write_burst_stream:16")

    # --- 4. SLR Placement ---
    # (严格按照示例文件的顺序)
    content.append("\n# --- 4. SLR Placement ---")
    
    for i in range(NUM_LITTLE_KERNELS):
        content.append(f"slr=graphyflow_little_{i+1}:{little_kernel_slr[i]}")
        
    for i in range(NUM_BIG_KERNELS):
        content.append(f"slr=graphyflow_big_{i+1}:{big_kernel_slr[i]}")

    # (使用合并后的 hbm_writer_slr 列表)
    for i in range(NUM_LITTLE_KERNELS):
        # 索引 0, 1
        content.append(f"slr=hbm_writer_little_{i+1}:{hbm_writer_slr[i]}")

    for i in range(NUM_BIG_KERNELS):
        # 索引 2, 3 (i + NUM_LITTLE_KERNELS)
        content.append(f"slr=hbm_writer_big_{i+1}:{hbm_writer_slr[i + NUM_LITTLE_KERNELS]}")

    # (使用合并后的 apply_kernel_slr 列表)
    for i in range(NUM_LITTLE_KERNELS):
        instance_num = i + 1
        # 索引 0, 1
        content.append(f"slr=apply_kernel_{instance_num}:{apply_kernel_slr[i]}")
        
    for i in range(NUM_BIG_KERNELS):
        instance_num = i + 1 + NUM_LITTLE_KERNELS
        # 索引 2, 3 (i + NUM_LITTLE_KERNELS)
        content.append(f"slr=apply_kernel_{instance_num}:{apply_kernel_slr[i + NUM_LITTLE_KERNELS]}")

    # --- 4. 写入文件 ---
    try:
        with open(dest, 'w', encoding='utf-8') as f:
            f.write("\n".join(content))
            f.write("\n") # 确保文件末尾有换行符
        print(f"配置文件已成功生成 (读取全局变量): {dest}")
    except IOError as e:
        print(f"写入文件时出错 {dest}: {e}", file=sys.stderr)





def fill_host_config(file_to_modify: Path):
    """
    Reads the specified file, replaces placeholders with the content of global variables,
    and writes the modified content back to the original file.
    """
    try:
        if not file_to_modify.is_file():
            print(f"Error: File not found '{file_to_modify}'")
            return

        replacements = {
            "{{BIG_KERNEL_NUM}}": NUM_BIG_KERNELS,
            "{{LITTLE_KERNEL_NUM}}": NUM_LITTLE_KERNELS,
            "{{BIG_KERNEL_HBM_EDGE_ID}}": big_kernel_hbm_edge_id,
            "{{BIG_KERNEL_HBM_NODE_ID}}": big_kernel_hbm_node_id,

            "{{LITTLE_KERNEL_HBM_EDGE_ID}}": little_kernel_hbm_edge_id,
            "{{LITTLE_KERNEL_HBM_NODE_ID}}": little_kernel_hbm_node_id,
        }

        original_content = file_to_modify.read_text(encoding="utf-8")
        modified_content = original_content

        for placeholder, value in replacements.items():
            replacement_string = ""
            if isinstance(value, list):
                replacement_string = f"{{{', '.join(map(str, value))}}}"
            elif isinstance(value, int):
                replacement_string = str(value)

            if replacement_string:
                modified_content = modified_content.replace(placeholder, replacement_string)

        file_to_modify.write_text(modified_content, encoding="utf-8")

    except Exception as e:
        print(f"An error occurred while processing file '{file_to_modify}': {e}")


def generate_project(
    comp_col: ComponentCollection,
    global_graph: Any,
    kernel_name: str,
    output_dir: Path,
    executable_name: str = "host",
    template_dir_override: Optional[Path] = None,
):
    """
    Generates a complete Vitis project directory from a DFG-IR.
    This version includes templating for build and run scripts.
    """
    print(f"--- Starting Project Generation for Kernel '{kernel_name}' ---")

    # 1. Define paths
    if template_dir_override:
        template_dir = template_dir_override
    else:
        # Assuming this script is in graphyflow/
        template_dir = Path(__file__).parent / "project_template"

    if not template_dir.exists():
        raise FileNotFoundError(f"Project template directory not found at: {template_dir}")

    # 2. Create output directory structure
    print(f"[1/6] Setting up Output Directory: '{output_dir}'")
    if output_dir.exists():
        shutil.rmtree(output_dir)

    scripts_dir = output_dir / "scripts"
    host_script_dir = scripts_dir / "host"
    kernel_script_dir = scripts_dir / "kernel"

    # Copy the entire template directory first
    shutil.copytree(template_dir, output_dir, dirs_exist_ok=True)

    # Ensure specific directories exist after copy
    host_script_dir.mkdir(parents=True, exist_ok=True)
    kernel_script_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "xclbin").mkdir(exist_ok=True)

    # 3. Fill dynamic configurations
    print("[2/6] Filling Dynamic Configuration Files...")
    fill_host_config(host_script_dir / "host_config.h")

    # Call the new dynamic system.cfg generator
    _create_cfg(output_dir/"system.cfg", kernel_name)
    # _generate_hbm_writer(NUM_BIG_KERNELS,output_dir / "scripts" / "kernel" / "hbm_writer.cpp")
    # Template the Makefile and run.sh (placeholders might be simple)
    replacements = {
        "{{EXECUTABLE_NAME}}": executable_name,
    }
    _copy_and_template(template_dir / "Makefile", output_dir / "Makefile", replacements)
    _copy_and_template(template_dir / "run.sh", output_dir / "run.sh", replacements)
    (output_dir / "run.sh").chmod(0o755)

    # 4. Instantiate backend and generate all dynamic code
    print("[3/6] Generating Dynamic Source Code via BackendManager...")
    bkd_mng = BackendManager()

    # Perform type analysis once
    # bkd_mng.analyze_graph_types(comp_col, global_graph)

    # Generate Big Kernel
    bkd_mng.REDUCE_MODE = "big_pipeline"
    kernel_h_big, kernel_h_little, shared_kernel_params,kernel_cpp_big,kernel_cpp_little,apply_kernel_cpp = bkd_mng.generate_backend(
        copy.deepcopy(comp_col), global_graph, f"{kernel_name}_big"
    )

    # Generate Little Kernel
    # bkd_mng.REDUCE_MODE = "little_pipeline"
    # kernel_h_little, kernel_cpp_little = bkd_mng.generate_backend(
    #     copy.deepcopy(comp_col), global_graph, f"{kernel_name}_little"
    # )

    #common_h = bkd_mng.generate_common_header(kernel_name)
    #host_h, host_cpp = bkd_mng.generate_host_codes(kernel_name, template_dir / "scripts" / "host")

    # 5. Deploy all dynamically generated files
    print(f"[4/6] Deploying Generated Kernel Files to '{kernel_script_dir}'")
    (kernel_script_dir / f"{kernel_name}_big.h").write_text(kernel_h_big)
    (kernel_script_dir / f"{kernel_name}_big.cpp").write_text(kernel_cpp_big)
    (kernel_script_dir / f"{kernel_name}_little.h").write_text(kernel_h_little)
    (kernel_script_dir / f"{kernel_name}_little.cpp").write_text(kernel_cpp_little)
    (kernel_script_dir / f"apply_kernel.cpp").write_text(apply_kernel_cpp)
    (kernel_script_dir / f"shared_kernel_params.h").write_text(shared_kernel_params)
    #(kernel_script_dir / f"{kernel_name}_little.h").write_text(kernel_h_little)
    #(kernel_script_dir / f"{kernel_name}_little.cpp").write_text(kernel_cpp_little)

    #print(f"[5/6] Deploying Generated Host Files to '{host_script_dir}'")
    #(host_script_dir / "common.h").write_text(common_h)
    #(host_script_dir / "generated_host.h").write_text(host_h)
    #(host_script_dir / "generated_host.cpp").write_text(host_cpp)

    # 6. Clean up template files that were replaced by generated ones
    #(host_script_dir / "generated_host.h.template").unlink()
    #(host_script_dir / "generated_host.cpp.template").unlink()

    #print("[6/6] Project Generation Complete!")
