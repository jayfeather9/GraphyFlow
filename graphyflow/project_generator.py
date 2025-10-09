import shutil
from pathlib import Path
from typing import Any, Dict, Optional

from .global_graph import GlobalGraph
from .dataflow_ir import ComponentCollection
from .backend_manager import BackendManager


# --- 在这里配置内核数量 ---
NUM_BIG_KERNELS = 2
NUM_LITTLE_KERNELS = 3

# Big Kernels
big_kernel_hbm_input_id = [1, 3]
big_kernel_hbm_output_id = [2, 4]

# Little Kernels
little_kernel_hbm_input_id = [5, 7, 9]
little_kernel_hbm_output_id = [6, 8, 10]


def _copy_and_template(src: Path, dest: Path, replacements: Dict[str, str]):
    """Reads a file, replaces placeholders, and writes to a new location."""
    content = src.read_text()
    for placeholder, value in replacements.items():
        content = content.replace(placeholder, value)
    dest.write_text(content)


def _create_cfg(dest: Path, kernel_name: str):

    big_kernel_base_name = f"{kernel_name}_big"
    little_kernel_base_name = f"{kernel_name}_little"

    big_instance_names = [f"{big_kernel_base_name}_{i+1}" for i in range(NUM_BIG_KERNELS)]
    little_instance_names = [f"{little_kernel_base_name}_{i+1}" for i in range(NUM_LITTLE_KERNELS)]

    content = ["[connectivity]"]

    # --- 添加内核实例化 (nk) 配置 -
    content.append("# --- 1. 内核实例化 ---")
    if NUM_BIG_KERNELS > 0:
        content.append(f"# 创建 {NUM_BIG_KERNELS} 个 big 内核实例")
        content.append(f"nk={big_kernel_base_name}:{NUM_BIG_KERNELS}:{'.'.join(big_instance_names)}")

    if NUM_LITTLE_KERNELS > 0:
        content.append(f"# 创建 {NUM_LITTLE_KERNELS} 个 little 内核实例")
        content.append(f"nk={little_kernel_base_name}:{NUM_LITTLE_KERNELS}:{'.'.join(little_instance_names)}")

    content.append("")

    content.append("# --- 2. HBM 通道映射 ---")

    # 遍历所有 Big Kernel 实例
    for i, instance_name in enumerate(big_instance_names):
        content.append(f"# 将 {instance_name} 的端口连接到 HBM bank")
        # gmem0 是输入, gmem1 是输出
        content.append(f"sp={instance_name}.m_axi_gmem0:HBM[{big_kernel_hbm_input_id[i]}]")
        content.append(f"sp={instance_name}.m_axi_gmem1:HBM[{big_kernel_hbm_output_id[i]}]")
        content.append(f"sp={instance_name}.m_axi_gmem2:HBM[{big_kernel_hbm_output_id[i]}]")
        content.append("")

    # 遍历所有 Little Kernel 实例
    for i, instance_name in enumerate(little_instance_names):
        content.append(f"# 将 {instance_name} 的端口连接到 HBM bank")
        # gmem0 是输入, gmem1 是输出
        content.append(f"sp={instance_name}.m_axi_gmem0:HBM[{little_kernel_hbm_input_id[i]}]")
        content.append(f"sp={instance_name}.m_axi_gmem1:HBM[{little_kernel_hbm_output_id[i]}]")
        content.append(f"sp={instance_name}.m_axi_gmem2:HBM[{little_kernel_hbm_output_id[i]}]")
        content.append("")

    output_file = dest / "system.cfg"
    dest.mkdir(parents=True, exist_ok=True)
    file_content = "\n".join(content)

    with open(output_file, "w") as f:
        f.write(file_content)


def fill_host_config(file_to_modify: Path):
    """
    读取指定文件，将其中的占位符替换为全局列表的内容，
    然后将修改后的内容写回原文件（就地修改）。

    Args:
        file_to_modify (Path): 需要就地修改的文件的路径。
    """
    try:
        # 检查文件是否存在
        if not file_to_modify.is_file():
            print(f"错误: 文件未找到 '{file_to_modify}'")
            return

        # 定义占位符与对应全局变量的映射关系
        replacements = {
            "{{BIG_KERNEL_NUM}}": NUM_BIG_KERNELS,
            "{{LITTLE_KERNEL_NUM}}": NUM_LITTLE_KERNELS,
            "{{BIG_KERNEL_HBM_INPUT_ID}}": big_kernel_hbm_input_id,
            "{{BIG_KERNEL_HBM_OUTPUT_ID}}": big_kernel_hbm_output_id,
            "{{LITTLE_KERNEL_HBM_INPUT_ID}}": little_kernel_hbm_input_id,
            "{{LITTLE_KERNEL_HBM_OUTPUT_ID}}": little_kernel_hbm_output_id,
        }

        # 1. 读取文件的全部内容
        original_content = file_to_modify.read_text(encoding="utf-8")
        modified_content = original_content

        # 2. 遍历所有需要替换的占位符
        for placeholder, value in replacements.items():

            replacement_string = ""
            if isinstance(value, list):
                # 列表 -> "{item1, item2}"
                replacement_string = f"{{{', '.join(map(str, value))}}}"
            elif isinstance(value, int):
                # 整数 -> "1"
                replacement_string = str(value)

            if replacement_string:
                modified_content = modified_content.replace(placeholder, replacement_string)

        # 3. 将替换后的内容写回到同一个文件
        file_to_modify.write_text(modified_content, encoding="utf-8")

    except Exception as e:
        print(f"在处理文件 '{file_to_modify}' 时发生错误: {e}")


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

    # 1. 定义路径
    if template_dir_override:
        template_dir = template_dir_override
    else:
        project_root = Path(__file__).parent.parent.resolve()
        template_dir = project_root / "graphyflow" / "project_template"

    if not template_dir.exists():
        raise FileNotFoundError(f"Project template directory not found at: {template_dir}")

    # 2. 创建输出目录结构
    print(f"[1/6] Setting up Output Directory: '{output_dir}'")
    if output_dir.exists():
        shutil.rmtree(output_dir)

    scripts_dir = output_dir / "scripts"
    host_script_dir = scripts_dir / "host"
    kernel_script_dir = scripts_dir / "kernel"
    xclbin_dir = output_dir / "xclbin"

    host_script_dir.mkdir(parents=True, exist_ok=True)
    kernel_script_dir.mkdir(parents=True, exist_ok=True)
    xclbin_dir.mkdir(exist_ok=True)

    # 3. 复制静态文件
    print(f"[2/6] Copying Static Files from Template: '{template_dir}'")
    static_files_to_ignore = ["Makefile", "run.sh", "system.cfg", "*.template"]
    shutil.copytree(
        template_dir, output_dir, dirs_exist_ok=True, ignore=shutil.ignore_patterns(*static_files_to_ignore)
    )

    # 3.1
    fill_host_config(host_script_dir / "host_config.h")

    # 4. 动态生成需要模板化的脚本文件
    print("[3/6] Generating Templated Scripts...")
    replacements = {
        "{{KERNEL_NAME_little}}": kernel_name + "_little",
        "{{KERNEL_NAME_big}}": kernel_name + "_big",
        "{{KERNEL_NAME_big}}": kernel_name,
        "{{EXECUTABLE_NAME}}": executable_name,
    }

    _copy_and_template(template_dir / "Makefile", output_dir / "Makefile", replacements)
    _copy_and_template(template_dir / "run.sh", output_dir / "run.sh", replacements)
    (output_dir / "run.sh").chmod(0o755)
    # _copy_and_template(template_dir / "system.cfg", output_dir / "system.cfg", replacements)
    _create_cfg(output_dir, kernel_name)

    _copy_and_template(
        template_dir / "scripts/kernel/kernel.mk", output_dir / "scripts/kernel/kernel.mk", replacements
    )

    # 5. 实例化后端并生成所有动态代码
    print("[4/6] Generating Dynamic Source Code via BackendManager...")
    bkd_mng = BackendManager()

    # Perform type analysis once, before any kernel generation.
    bkd_mng.analyze_graph_types(comp_col, global_graph)

    bkd_mng.REDUCE_MODE = "big_pipeline"
    kernel_h_big, kernel_cpp_big = bkd_mng.generate_backend(
        comp_col,
        global_graph,
        kernel_name,
    )
    bkd_mng.REDUCE_MODE = "little_pipeline"
    kernel_h_little, kernel_cpp_little = bkd_mng.generate_backend(comp_col, global_graph, kernel_name)

    common_h = bkd_mng.generate_common_header(kernel_name)

    host_h, host_cpp = bkd_mng.generate_host_codes(kernel_name, template_dir / "scripts" / "host")

    # 6. 部署所有动态生成的文件
    print(f"[5/6] Deploying Generated Files to '{output_dir}'")
    with open(kernel_script_dir / f"{kernel_name}_big.h", "w") as f:
        f.write(kernel_h_big)
    with open(kernel_script_dir / f"{kernel_name}_big.cpp", "w") as f:
        f.write(kernel_cpp_big)

    with open(kernel_script_dir / f"{kernel_name}_little.h", "w") as f:
        f.write(kernel_h_little)
    with open(kernel_script_dir / f"{kernel_name}_little.cpp", "w") as f:
        f.write(kernel_cpp_little)

    with open(host_script_dir / "common.h", "w") as f:
        f.write(common_h)
    with open(host_script_dir / "generated_host.h", "w") as f:
        f.write(host_h)
    with open(host_script_dir / "generated_host.cpp", "w") as f:
        f.write(host_cpp)

    print("[6/6] Project Generation Complete!")
