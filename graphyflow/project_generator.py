import shutil
from pathlib import Path
from typing import Any, Dict, Optional
import copy

from .global_graph import GlobalGraph
from .dataflow_ir import ComponentCollection
from .backend_manager import BackendManager


# --- CONFIGURATION SECTION ---
# You can change the number of kernels and their HBM mapping here.
NUM_BIG_KERNELS = 8
NUM_LITTLE_KERNELS = 0

# HBM channel IDs for Big Kernels. The list length must match NUM_BIG_KERNELS.
big_kernel_hbm_input_id = [0, 2, 4, 6, 8, 10, 12, 14]
big_kernel_hbm_output_id = [1, 3, 5, 7, 9, 11, 13, 15]

# HBM channel IDs for Little Kernels. The list length must match NUM_LITTLE_KERNELS.
little_kernel_hbm_input_id = []
little_kernel_hbm_output_id = []
# --- END CONFIGURATION SECTION ---


def _copy_and_template(src: Path, dest: Path, replacements: Dict[str, str]):
    """Reads a file, replaces placeholders, and writes to a new location."""
    content = src.read_text()
    for placeholder, value in replacements.items():
        content = content.replace(placeholder, value)
    dest.write_text(content)


def _create_cfg(dest: Path, kernel_name: str):
    """
    Programmatically generates the system.cfg file based on the
    kernel configuration defined at the top of this file.
    """
    # --- PHASE 1: Validation ---
    # Ensure the number of HBM IDs matches the number of kernel instances.
    assert (
        len(big_kernel_hbm_input_id) == NUM_BIG_KERNELS
    ), "Mismatch between NUM_BIG_KERNELS and the length of big_kernel_hbm_input_id list."
    assert (
        len(big_kernel_hbm_output_id) == NUM_BIG_KERNELS
    ), "Mismatch between NUM_BIG_KERNELS and the length of big_kernel_hbm_output_id list."
    assert (
        len(little_kernel_hbm_input_id) == NUM_LITTLE_KERNELS
    ), "Mismatch between NUM_LITTLE_KERNELS and the length of little_kernel_hbm_input_id list."
    assert (
        len(little_kernel_hbm_output_id) == NUM_LITTLE_KERNELS
    ), "Mismatch between NUM_LITTLE_KERNELS and the length of little_kernel_hbm_output_id list."

    print(
        f"[INFO] Generating system.cfg for {NUM_BIG_KERNELS} big kernel(s) and {NUM_LITTLE_KERNELS} little kernel(s)."
    )

    # --- PHASE 2: Content Generation ---
    big_kernel_base_name = f"{kernel_name}_big"
    little_kernel_base_name = f"{kernel_name}_little"

    big_instance_names = [f"{big_kernel_base_name}_{i+1}" for i in range(NUM_BIG_KERNELS)]
    little_instance_names = [f"{little_kernel_base_name}_{i+1}" for i in range(NUM_LITTLE_KERNELS)]

    content = ["[connectivity]"]
    content.append("# --- 1. Kernel Instantiation (nk) ---")

    # Define number of compute units (nk) for each kernel type
    if NUM_BIG_KERNELS > 0:
        content.append(f"nk={big_kernel_base_name}:{NUM_BIG_KERNELS}:{'.'.join(big_instance_names)}")
    if NUM_LITTLE_KERNELS > 0:
        content.append(f"nk={little_kernel_base_name}:{NUM_LITTLE_KERNELS}:{'.'.join(little_instance_names)}")

    content.append("\n# --- 2. HBM Port Mapping (sp) ---")

    # Generate port mappings for all Big Kernel instances
    for i, instance_name in enumerate(big_instance_names):
        input_hbm = big_kernel_hbm_input_id[i]
        output_hbm = big_kernel_hbm_output_id[i]
        content.append(f"\n# -- Mapping for instance: {instance_name} --")
        content.append(f"# All 3 input ports (gmem0, gmem1, gmem2) connect to HBM[{input_hbm}]")
        content.append(f"sp={instance_name}.m_axi_gmem0:HBM[{input_hbm}]")
        content.append(f"sp={instance_name}.m_axi_gmem1:HBM[{input_hbm}]")
        content.append(f"sp={instance_name}.m_axi_gmem2:HBM[{input_hbm}]")
        content.append(f"# Output port (gmem3) connects to HBM[{output_hbm}]")
        content.append(f"sp={instance_name}.m_axi_gmem3:HBM[{output_hbm}]")

    # Generate port mappings for all Little Kernel instances
    for i, instance_name in enumerate(little_instance_names):
        input_hbm = little_kernel_hbm_input_id[i]
        output_hbm = little_kernel_hbm_output_id[i]
        content.append(f"\n# -- Mapping for instance: {instance_name} --")
        content.append(f"# All 3 input ports (gmem0, gmem1, gmem2) connect to HBM[{input_hbm}]")
        content.append(f"sp={instance_name}.m_axi_gmem0:HBM[{input_hbm}]")
        content.append(f"sp={instance_name}.m_axi_gmem1:HBM[{input_hbm}]")
        content.append(f"sp={instance_name}.m_axi_gmem2:HBM[{input_hbm}]")
        content.append(f"# Output port (gmem3) connects to HBM[{output_hbm}]")
        content.append(f"sp={instance_name}.m_axi_gmem3:HBM[{output_hbm}]")

    # --- PHASE 3: File Writing ---
    output_file = dest / "system.cfg"
    file_content = "\n".join(content)

    output_file.write_text(file_content)
    print(f"[SUCCESS] Successfully generated dynamic system.cfg at '{output_file}'")


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
            "{{BIG_KERNEL_HBM_INPUT_ID}}": big_kernel_hbm_input_id,
            "{{BIG_KERNEL_HBM_OUTPUT_ID}}": big_kernel_hbm_output_id,
            "{{LITTLE_KERNEL_HBM_INPUT_ID}}": little_kernel_hbm_input_id,
            "{{LITTLE_KERNEL_HBM_OUTPUT_ID}}": little_kernel_hbm_output_id,
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
    bkd_mng.analyze_graph_types(comp_col, global_graph)

    # Generate Big Kernel
    bkd_mng.REDUCE_MODE = "big_pipeline"
    kernel_h_big, kernel_cpp_big = bkd_mng.generate_backend(
        copy.deepcopy(comp_col), global_graph, f"{kernel_name}_big"
    )

    # Generate Little Kernel
    bkd_mng.REDUCE_MODE = "little_pipeline"
    kernel_h_little, kernel_cpp_little = bkd_mng.generate_backend(
        copy.deepcopy(comp_col), global_graph, f"{kernel_name}_little"
    )

    common_h = bkd_mng.generate_common_header(kernel_name)
    host_h, host_cpp = bkd_mng.generate_host_codes(kernel_name, template_dir / "scripts" / "host")

    # 5. Deploy all dynamically generated files
    print(f"[4/6] Deploying Generated Kernel Files to '{kernel_script_dir}'")
    (kernel_script_dir / f"{kernel_name}_big.h").write_text(kernel_h_big)
    (kernel_script_dir / f"{kernel_name}_big.cpp").write_text(kernel_cpp_big)
    (kernel_script_dir / f"{kernel_name}_little.h").write_text(kernel_h_little)
    (kernel_script_dir / f"{kernel_name}_little.cpp").write_text(kernel_cpp_little)

    print(f"[5/6] Deploying Generated Host Files to '{host_script_dir}'")
    (host_script_dir / "common.h").write_text(common_h)
    (host_script_dir / "generated_host.h").write_text(host_h)
    (host_script_dir / "generated_host.cpp").write_text(host_cpp)

    # 6. Clean up template files that were replaced by generated ones
    (host_script_dir / "generated_host.h.template").unlink()
    (host_script_dir / "generated_host.cpp.template").unlink()

    print("[6/6] Project Generation Complete!")
