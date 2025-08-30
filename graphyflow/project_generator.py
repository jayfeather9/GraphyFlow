import shutil
from pathlib import Path
from typing import Any

# 导入我们需要的GraphyFlow模块
from .global_graph import GlobalGraph
from .dataflow_ir import ComponentCollection
from .backend_manager import BackendManager


def generate_project(
    comp_col: ComponentCollection,
    global_graph: Any,
    kernel_name: str,
    output_dir: Path,
    executable_name: str = "host",
):
    """
    Generates a complete Vitis project directory from a DFG-IR.

    Args:
        comp_col: The component collection representing the dataflow graph.
        global_graph: The original GlobalGraph object (for type information).
        kernel_name: The base name for the kernel files (e.g., 'graphyflow').
        output_dir: The path to the directory where the project will be created.
        executable_name: The name of the final host executable.
    """
    print(f"--- Starting Project Generation for Kernel '{kernel_name}' ---")

    # 1. 定义路径
    project_root = Path(__file__).parent.parent.resolve()
    template_dir = project_root / "graphyflow" / "project_template"

    if not template_dir.exists():
        raise FileNotFoundError(f"Project template directory not found at: {template_dir}")

    # 2. 创建输出目录结构
    print(f"[1/5] Setting up Output Directory: '{output_dir}'")
    if output_dir.exists():
        shutil.rmtree(output_dir)

    host_script_dir = output_dir / "scripts" / "host"
    kernel_script_dir = output_dir / "scripts" / "kernel"
    xclbin_dir = output_dir / "xclbin"

    host_script_dir.mkdir(parents=True, exist_ok=True)
    kernel_script_dir.mkdir(parents=True, exist_ok=True)
    xclbin_dir.mkdir(exist_ok=True)

    # 3. 复制所有静态文件
    print(f"[2/5] Copying Static Files from Template: '{template_dir}'")
    # 使用 ignore 来排除 .template 文件，因为它们将被动态生成
    shutil.copytree(template_dir, output_dir, dirs_exist_ok=True, ignore=shutil.ignore_patterns("*.template"))

    # 4. 实例化后端并生成所有动态代码
    print("[3/5] Generating Dynamic Source Code via BackendManager...")
    bkd_mng = BackendManager()

    # 生成内核 C++ 代码
    kernel_h, kernel_cpp = bkd_mng.generate_backend(comp_col, global_graph, kernel_name)

    # 生成共享头文件
    common_h = bkd_mng.generate_common_header(kernel_name)

    # 生成主机代码 (从模板)
    host_h, host_cpp = bkd_mng.generate_host_codes(kernel_name, template_dir / "scripts" / "host")

    # 5. 部署所有动态生成的文件
    print(f"[4/5] Deploying Generated Files to '{output_dir}'")
    with open(kernel_script_dir / f"{kernel_name}.h", "w") as f:
        f.write(kernel_h)
    with open(kernel_script_dir / f"{kernel_name}.cpp", "w") as f:
        f.write(kernel_cpp)
    with open(host_script_dir / "common.h", "w") as f:
        f.write(common_h)
    with open(host_script_dir / "generated_host.h", "w") as f:
        f.write(host_h)
    with open(host_script_dir / "generated_host.cpp", "w") as f:
        f.write(host_cpp)

    print("[5/5] Project Generation Complete!")
