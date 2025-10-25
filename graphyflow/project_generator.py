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
NUM_BIG_KERNELS = 3
NUM_LITTLE_KERNELS = 0

# HBM channel IDs for Big Kernels. The list length must match NUM_BIG_KERNELS.
big_kernel_hbm_edge_id = [0, 1,2]
big_kernel_hbm_node_id = [20,21,22]

big_kernel_slr = ["SLR1", "SLR2", "SLR1"]
apply_kernel_slr = ["SLR2", "SLR2", "SLR2"]
hbm_writer_slr = "SLR0"

# --- END CONFIGURATION SECTION ---


def _copy_and_template(src: Path, dest: Path, replacements: Dict[str, str]):
    """Reads a file, replaces placeholders, and writes to a new location."""
    content = src.read_text()
    for placeholder, value in replacements.items():
        content = content.replace(placeholder, value)
    dest.write_text(content)


def _create_cfg(dest: Path, kernel_name: str):
    """
    根据此文件顶部定义的 Kernel 配置，
    以编程方式生成 system.cfg 文件。
    """
    
    # --- PHASE 1: Validation (验证) ---
    # 确保 HBM ID 和 SLR 列表的长度与 Kernel 实例数匹配。
    assert (
        len(big_kernel_hbm_edge_id) == NUM_BIG_KERNELS
    ), "Mismatch between NUM_BIG_KERNELS and big_kernel_hbm_edge_id list."
    assert (
        len(big_kernel_hbm_node_id) == NUM_BIG_KERNELS
    ), "Mismatch between NUM_BIG_KERNELS and big_kernel_hbm_node_id list."
    assert (
        len(big_kernel_slr) == NUM_BIG_KERNELS
    ), "Mismatch between NUM_BIG_KERNELS and big_kernel_slr list."
    assert (
        len(apply_kernel_slr) == NUM_BIG_KERNELS
    ), "Mismatch between NUM_BIG_KERNELS and apply_kernel_slr list."

    print(
        f"[INFO] Generating system.cfg for {NUM_BIG_KERNELS} big kernel(s) and {NUM_LITTLE_KERNELS} little kernel(s)."
    )

    # --- PHASE 2: Content Generation (内容生成) ---
    
    # 定义基础名称
    big_kernel_base_name = f"{kernel_name}_big"
    little_kernel_base_name = f"{kernel_name}_little"
    apply_kernel_base_name = "apply_kernel"
    hbm_writer_base_name = "hbm_writer"

    # 生成实例名称列表
    big_instance_names = [f"{big_kernel_base_name}_{i+1}" for i in range(NUM_BIG_KERNELS)]
    little_instance_names = [f"{little_kernel_base_name}_{i+1}" for i in range(NUM_LITTLE_KERNELS)]
    # apply_kernel 的实例数跟随 big_kernel
    apply_instance_names = [f"{apply_kernel_base_name}_{i+1}" for i in range(NUM_BIG_KERNELS)]
    # 只有一个 hbm_writer
    hbm_writer_instance_name = f"{hbm_writer_base_name}_1"


    content = ["[connectivity]"]
    content.append("# --- 1. Kernel Instantiation (nk) ---")

    # 定义每个 Kernel 类型的计算单元 (nk) 数量
    if NUM_BIG_KERNELS > 0:
        content.append(f"nk={big_kernel_base_name}:{NUM_BIG_KERNELS}:{'.'.join(big_instance_names)}")
        # 添加 apply_kernel，其数量与 big_kernel 相同
        content.append(f"nk={apply_kernel_base_name}:{NUM_BIG_KERNELS}:{'.'.join(apply_instance_names)}")
        
    if NUM_LITTLE_KERNELS > 0:
        content.append(f"nk={little_kernel_base_name}:{NUM_LITTLE_KERNELS}:{'.'.join(little_instance_names)}")
    
    # 添加 hbm_writer
    content.append(f"nk={hbm_writer_base_name}:1:{hbm_writer_instance_name}")

    # --- 2. HBM Port Mapping (sp) ---
    content.append("\n# --- 2. HBM Port Mapping (sp) ---")

    # -- 映射 big kernel 实例 --
    content.append(f"\n# -- Mapping for instance: {big_kernel_base_name} --")
    for i in range(NUM_BIG_KERNELS):
        instance_name = big_instance_names[i]
        hbm_id = big_kernel_hbm_edge_id[i]
        content.append(f"sp={instance_name}.edge_props:HBM[{hbm_id}]")

    # -- 映射 hbm_writer 实例 --
    # hbm_writer 的端口 (_1, _2, _3) 对应于 big kernel 的 HBM ID
    content.append(f"\n# -- Mapping for instance: {hbm_writer_instance_name} --")
    for i in range(NUM_BIG_KERNELS):
        port_index = i + 1
        hbm_id = big_kernel_hbm_node_id[i]
        content.append(f"sp={hbm_writer_instance_name}.node_props_{port_index}:HBM[{hbm_id}]")
        content.append(f"sp={hbm_writer_instance_name}.output_{port_index}:HBM[{hbm_id}]")


    # --- 3. Stream Connections (stream_connect) ---
    content.append("\n# --- 3. Stream Connections (stream_connect) ---")

    # -- graphyflow_big <-> hbm_writer_1 : cacheline streams --
    content.append(f"\n# {big_kernel_base_name} <-> {hbm_writer_instance_name} : cacheline streams")
    for i in range(NUM_BIG_KERNELS):
        big_instance = big_instance_names[i]
        port_index = i + 1
        content.append(f"stream_connect={big_instance}.cacheline_req_stream:{hbm_writer_instance_name}.cacheline_req_stream_{port_index}:16")
        content.append(f"stream_connect={hbm_writer_instance_name}.cacheline_resp_stream_{port_index}:{big_instance}.cacheline_resp_stream:16")

    # -- graphyflow_big -> apply_kernel --
    content.append(f"\n# {big_kernel_base_name} -> {apply_kernel_base_name}")
    for i in range(NUM_BIG_KERNELS):
        big_instance = big_instance_names[i]
        apply_instance = apply_instance_names[i]
        content.append(f"stream_connect={big_instance}.kernel_out_stream:{apply_instance}.kernel_out_stream:16")

    # -- hbm_writer_1 -> apply_kernel : cacheline data streams --
    content.append(f"\n# {hbm_writer_instance_name} -> {apply_kernel_base_name} : cacheline data streams")
    for i in range(NUM_BIG_KERNELS):
        apply_instance = apply_instance_names[i]
        port_index = i + 1
        content.append(f"stream_connect={hbm_writer_instance_name}.cacheline_data_stream_{port_index}:{apply_instance}.cacheline_data_stream:16")

    # -- apply_kernel -> hbm_writer_1 : write burst streams --
    content.append(f"\n# {apply_kernel_base_name} -> {hbm_writer_instance_name} : write burst streams")
    for i in range(NUM_BIG_KERNELS):
        apply_instance = apply_instance_names[i]
        port_index = i + 1
        content.append(f"stream_connect={apply_instance}.write_burst_stream:{hbm_writer_instance_name}.write_burst_stream_{port_index}:16")

    # --- 4. SLR Assignments (slr) ---
    content.append("\n# --- 4. SLR Assignments (slr) ---")
    
    # -- 分配 big kernels --
    for i in range(NUM_BIG_KERNELS):
        content.append(f"slr={big_instance_names[i]}:{big_kernel_slr[i]}")

    # -- 分配 apply kernels --
    for i in range(NUM_BIG_KERNELS):
        content.append(f"slr={apply_instance_names[i]}:{apply_kernel_slr[i]}")

    # -- 分配 hbm_writer --
    content.append(f"slr={hbm_writer_instance_name}:{hbm_writer_slr}")


    # --- PHASE 3: File Writing (文件写入) ---
    final_content_str = "\n".join(content)
    try:
        dest.write_text(final_content_str)
        print(f"[INFO] Successfully generated config file at: {dest}")
    except IOError as e:
        print(f"[ERROR] Failed to write config file: {e}", file=sys.stderr)
# --- MODIFICATION END ---


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
    _create_cfg(output_dir, kernel_name)

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
    kernel_h_big, kernel_cpp_big,apply_kernel_cpp = bkd_mng.generate_backend(
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
    (kernel_script_dir / f"apply_kernel.cpp").write_text(apply_kernel_cpp)
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
