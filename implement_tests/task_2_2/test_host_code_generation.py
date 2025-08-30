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
from graphyflow.backend_manager import BackendManager
from graphyflow.lambda_func import lambda_min


def test_host_code_generation():
    print("--- Testing Task 2.2: Host Code Generation from Template ---")

    # 1. Create a temporary directory for our templates
    with tempfile.TemporaryDirectory() as temp_dir:
        template_path = Path(temp_dir)

        # 2. Create dummy template files inside the temp directory
        with open(template_path / "generated_host.h.template", "w") as f:
            f.write(
                """#ifndef __GENERATED_HOST_H__
#define __GENERATED_HOST_H__
// {{GRAPHYFLOW_HOST_BUFFER_DECLARATIONS}}
// {{GRAPHYFLOW_DEVICE_BUFFER_DECLARATIONS}}
#endif"""
            )

        with open(template_path / "generated_host.cpp.template", "w") as f:
            f.write(
                """#include "generated_host.h"
void AlgorithmHost::setup_buffers() {
// {{GRAPHYFLOW_BUFFER_INITIALIZATION}}
}
void AlgorithmHost::execute_kernel_iteration() {
// {{GRAPHYFLOW_SET_KERNEL_ARGS}}
}"""
            )

        # 3. Setup the BackendManager with a standard graph
        g = GlobalGraph(properties={"node": {"d": dfir.FloatType()}, "edge": {"w": dfir.FloatType()}})
        edges = g.add_graph_input("edge")
        min_dist = edges.reduce_by(
            reduce_key=lambda e: e.dst.id,
            reduce_transform=lambda e: e.src.d + e.w,
            reduce_method=lambda x, y: lambda_min(x, y),
        )
        dfirs = g.to_dfir()
        comp_col = delete_placeholder_components_pass(dfirs[0])

        bkd_mng = BackendManager()
        # Run full backend to populate all necessary internal states (like axi_input_ports)
        bkd_mng.generate_backend(comp_col, g, "graphyflow_test")

        # 4. Call the generate_host_codes function
        host_h, host_cpp = bkd_mng.generate_host_codes("graphyflow_test", template_path)

        # 5. Perform robust checks
        errors = []

        # --- *** 关键修正：动态获取端口名称用于测试 *** ---
        # Get the dynamically generated names from the backend manager
        input_port_name = bkd_mng.axi_input_ports[0].unique_name
        output_port_name = bkd_mng.axi_output_ports[0].unique_name

        # Build expected strings using the dynamic names
        expected_h_decl = (
            f"std::vector<struct_ebu_7_t, aligned_allocator<struct_ebu_7_t>> h_{input_port_name};"
        )
        expected_d_decl = f"cl::Buffer d_{output_port_name};"
        expected_set_arg = f"m_kernel.setArg(arg_idx++, d_{input_port_name})"
        # -----------------------------------------------------------

        if "// {{GRAPHYFLOW_HOST_BUFFER_DECLARATIONS}}" in host_h:
            errors.append("FAIL: Placeholder for host buffers was not replaced in .h file.")

        if "".join(expected_h_decl.split()) not in "".join(host_h.split()):
            errors.append("FAIL: Input host buffer declaration is missing or incorrect in .h file.")

        if "".join(expected_d_decl.split()) not in "".join(host_h.split()):
            errors.append("FAIL: Output device buffer declaration is missing or incorrect in .h file.")

        if "// {{GRAPHYFLOW_BUFFER_INITIALIZATION}}" in host_cpp:
            errors.append("FAIL: Placeholder for buffer initialization was not replaced in .cpp file.")

        if "".join(expected_set_arg.split()) not in "".join(host_cpp.split()):
            errors.append("FAIL: setArg for input buffer is missing or incorrect in .cpp file.")

        # 6. Report results
        if not errors:
            print("Host Code Generation Test: SUCCESS")
            return True
        else:
            print("Host Code Generation Test: FAILED")
            for error in errors:
                print(f"  - {error}")
            print("\n--- Generated host.h ---\n", host_h)
            print("\n--- Generated host.cpp ---\n", host_cpp)
            return False


if __name__ == "__main__":
    test_dir = project_root / "implement_tests" / "task_2_2"
    test_dir.mkdir(exist_ok=True)

    print(f"Running tests for Task 2.2. Test script location: {test_dir}")

    success = test_host_code_generation()

    if success:
        print("\nAll tests for Task 2.2 PASSED.")
        with open(test_dir / "SUCCESS", "w") as f:
            f.write("Task 2.2 completed and verified.")
        sys.exit(0)
    else:
        print("\nSome tests for Task 2.2 FAILED.")
        sys.exit(1)
