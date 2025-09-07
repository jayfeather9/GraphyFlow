import os
import sys
from pathlib import Path
import tempfile

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from graphyflow.global_graph import *
import graphyflow.dataflow_ir as dfir
from graphyflow.passes import delete_placeholder_components_pass
from graphyflow.project_generator import generate_project


def test_script_templating():
    print("--- Testing Task 3.2: Build Script Templating ---")

    # 1. Create a temporary directory for the project and a mock template
    with tempfile.TemporaryDirectory() as temp_dir:
        output_path = Path(temp_dir) / "my_templated_project"
        mock_template_dir = Path(temp_dir) / "mock_template"
        (mock_template_dir / "scripts" / "host").mkdir(parents=True, exist_ok=True)
        (mock_template_dir / "scripts" / "kernel").mkdir(exist_ok=True)  # Ensure kernel dir exists

        # Create mock template files with placeholders
        with open(mock_template_dir / "Makefile", "w") as f:
            f.write("KERNEL_NAME := {{KERNEL_NAME}}\nEXECUTABLE := {{EXECUTABLE_NAME}}")
        with open(mock_template_dir / "run.sh", "w") as f:
            f.write("./{{EXECUTABLE_NAME}} ./xclbin/{{KERNEL_NAME}}.sw_emu.xclbin")
        with open(mock_template_dir / "system.cfg", "w") as f:
            f.write("nk={{KERNEL_NAME}}:1:{{KERNEL_NAME}}_1")
        (mock_template_dir / "scripts" / "host" / "generated_host.h.template").touch()
        (mock_template_dir / "scripts" / "host" / "generated_host.cpp.template").touch()

        # 2. Define a dummy graph
        g = GlobalGraph(properties={"node": {"d": dfir.FloatType()}})
        edges = g.add_graph_input("edge")
        dfirs = g.to_dfir()
        comp_col = delete_placeholder_components_pass(dfirs[0])

        kernel_name = "my_custom_kernel"
        executable_name = "my_custom_host"

        # 3. Call generate_project, passing the mock template directory
        generate_project(
            comp_col, g, kernel_name, output_path, executable_name, template_dir_override=mock_template_dir
        )

        # 4. Verify the contents of the generated files
        errors = []

        makefile_content = (output_path / "Makefile").read_text()
        run_sh_content = (output_path / "run.sh").read_text()
        system_cfg_content = (output_path / "system.cfg").read_text()

        if f"KERNEL_NAME := {kernel_name}" not in makefile_content:
            errors.append("FAIL: KERNEL_NAME was not replaced in Makefile.")
        if f"EXECUTABLE := {executable_name}" not in makefile_content:
            errors.append("FAIL: EXECUTABLE_NAME was not replaced in Makefile.")

        if f"./{executable_name}" not in run_sh_content:
            errors.append("FAIL: EXECUTABLE_NAME was not replaced in run.sh.")

        # --- *** 关键修正：检查正确的文件名 *** ---
        if f"{kernel_name}.sw_emu.xclbin" not in run_sh_content:
            errors.append("FAIL: KERNEL_NAME was not correctly replaced in run.sh's xclbin path.")

        if f"nk={kernel_name}:1:{kernel_name}_1" not in system_cfg_content:
            errors.append("FAIL: KERNEL_NAME was not replaced in system.cfg.")

        # 5. Report results
        if not errors:
            print("Build Script Templating Test: SUCCESS")
            return True
        else:
            print("Build Script Templating Test: FAILED")
            for error in errors:
                print(f"  - {error}")
            return False


if __name__ == "__main__":
    test_dir = project_root / "implement_tests" / "task_3_2"
    test_dir.mkdir(exist_ok=True)

    print(f"Running tests for Task 3.2. Test script location: {test_dir}")

    success = test_script_templating()

    if success:
        print("\nAll tests for Task 3.2 PASSED.")
        with open(test_dir / "SUCCESS", "w") as f:
            f.write("Task 3.2 completed and verified.")
        sys.exit(0)
    else:
        print("\nSome tests for Task 3.2 FAILED.")
        sys.exit(1)
