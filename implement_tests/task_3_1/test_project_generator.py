import os
import sys
from pathlib import Path
import tempfile
import shutil

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from graphyflow.global_graph import *
import graphyflow.dataflow_ir as dfir
from graphyflow.passes import delete_placeholder_components_pass
from graphyflow.lambda_func import lambda_min

# *** 关键：导入新的顶层函数 ***
from graphyflow.project_generator import generate_project


def test_full_project_generation():
    print("--- Testing Task 3.1: Full Project Generation API ---")

    # 1. Setup a temporary directory for the generated project
    with tempfile.TemporaryDirectory() as temp_dir:
        output_path = Path(temp_dir) / "my_generated_project"

        # 2. Define a standard test graph
        g = GlobalGraph(properties={"node": {"d": dfir.FloatType()}, "edge": {"w": dfir.FloatType()}})
        edges = g.add_graph_input("edge")
        min_dist = edges.reduce_by(
            reduce_key=lambda e: e.dst.id,
            reduce_transform=lambda e: e.src.d + e.w,
            reduce_method=lambda x, y: lambda_min(x, y),
        )
        dfirs = g.to_dfir()
        comp_col = delete_placeholder_components_pass(dfirs[0])

        kernel_name = "my_kernel"

        # 3. Call the new high-level generate_project function
        try:
            generate_project(comp_col, g, kernel_name, output_path)
        except Exception as e:
            assert False, f"generate_project raised an unexpected exception: {e}"

        # 4. Verify that the project structure and key files were created
        errors = []

        # Check directories
        if not (output_path / "scripts" / "host").exists():
            errors.append("FAIL: scripts/host directory was not created.")
        if not (output_path / "scripts" / "kernel").exists():
            errors.append("FAIL: scripts/kernel directory was not created.")

        # Check static copied files
        if not (output_path / "Makefile").exists():
            errors.append("FAIL: Static file 'Makefile' was not copied.")
        if not (output_path / "scripts" / "host" / "fpga_executor.cpp").exists():
            errors.append("FAIL: Static file 'fpga_executor.cpp' was not copied.")

        # Check dynamically generated files
        if not (output_path / "scripts" / "kernel" / f"{kernel_name}.cpp").exists():
            errors.append(f"FAIL: Dynamic kernel file '{kernel_name}.cpp' was not created.")
        if not (output_path / "scripts" / "host" / "common.h").exists():
            errors.append("FAIL: Dynamic file 'common.h' was not created.")
        if not (output_path / "scripts" / "host" / "generated_host.cpp").exists():
            errors.append("FAIL: Dynamic file 'generated_host.cpp' was not created.")

        # Quick check on generated host code to see if placeholder was replaced
        with open(output_path / "scripts" / "host" / "generated_host.h", "r") as f:
            host_h_content = f.read()
        if "{{GRAPHYFLOW_HOST_BUFFER_DECLARATIONS}}" in host_h_content:
            errors.append("FAIL: Placeholder was not replaced in generated_host.h.")

        # 5. Report results
        if not errors:
            print("Full Project Generation Test: SUCCESS")
            return True
        else:
            print("Full Project Generation Test: FAILED")
            for error in errors:
                print(f"  - {error}")
            return False


if __name__ == "__main__":
    test_dir = project_root / "implement_tests" / "task_3_1"
    test_dir.mkdir(exist_ok=True)

    print(f"Running tests for Task 3.1. Test script location: {test_dir}")

    success = test_full_project_generation()

    if success:
        print("\nAll tests for Task 3.1 PASSED.")
        with open(test_dir / "SUCCESS", "w") as f:
            f.write("Task 3.1 completed and verified.")
        sys.exit(0)
    else:
        print("\nSome tests for Task 3.1 FAILED.")
        sys.exit(1)
