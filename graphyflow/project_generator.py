import shutil
from pathlib import Path
from typing import Any, Dict, Optional

from .global_graph import GlobalGraph
from .dataflow_ir import ComponentCollection
from .backend_manager import BackendManager


def _copy_and_template(src: Path, dest: Path, replacements: Dict[str, str]):
    """Reads a file, replaces placeholders, and writes to a new location."""
    content = src.read_text()
    for placeholder, value in replacements.items():
        content = content.replace(placeholder, value)
    dest.write_text(content)


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

    # 4. 动态生成需要模板化的脚本文件
    print("[3/6] Generating Templated Scripts...")
    replacements = {"{{KERNEL_NAME}}": kernel_name, "{{EXECUTABLE_NAME}}": executable_name}
    _copy_and_template(template_dir / "Makefile", output_dir / "Makefile", replacements)
    _copy_and_template(template_dir / "run.sh", output_dir / "run.sh", replacements)
    (output_dir / "run.sh").chmod(0o755)
    _copy_and_template(template_dir / "system.cfg", output_dir / "system.cfg", replacements)

    # 5. 实例化后端并生成所有动态代码
    print("[4/6] Generating Dynamic Source Code via BackendManager...")
    bkd_mng = BackendManager()
    kernel_h, kernel_cpp = bkd_mng.generate_backend(comp_col, global_graph, kernel_name)
    common_h = bkd_mng.generate_common_header(kernel_name)
    host_h, host_cpp = bkd_mng.generate_host_codes(kernel_name, template_dir / "scripts" / "host")

    # 6. 部署所有动态生成的文件
    print(f"[5/6] Deploying Generated Files to '{output_dir}'")
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

    print("[6/6] Project Generation Complete!")
