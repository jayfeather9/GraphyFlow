import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from graphyflow.global_graph import *
import graphyflow.dataflow_ir as dfir
from graphyflow.passes import delete_placeholder_components_pass
from graphyflow.backend_manager import BackendManager
from graphyflow.lambda_func import lambda_min


def test_axi_kernel_generation():
    print("--- Testing Task 1.2: AXI Kernel Generation ---")

    # 1. Define a simple graph algorithm (similar to dist.py)
    g = GlobalGraph(
        properties={
            "node": {"distance": dfir.FloatType()},
            "edge": {"weight": dfir.FloatType()},
        }
    )
    edges = g.add_graph_input("edge")
    pdu = edges.map_(map_func=lambda edge: (edge.src.distance, edge.dst, edge.weight))
    min_dist = pdu.reduce_by(
        reduce_key=lambda src_dist, dst, edge_w: dst.id,
        reduce_transform=lambda src_dist, dst, edge_w: (src_dist + edge_w, dst),
        reduce_method=lambda x, y: (lambda_min(x[0], y[0]), x[1]),
    )

    # 2. Run frontend and passes
    dfirs = g.to_dfir()
    comp_col = delete_placeholder_components_pass(dfirs[0])

    # 3. Instantiate BackendManager and generate code
    bkd_mng = BackendManager()
    kernel_name = "graphyflow_test_kernel"
    header_code, source_code = bkd_mng.generate_backend(comp_col, g, kernel_name)

    # 4. Perform checks on the generated source code
    errors = []

    # Check 1: Presence of the extern "C" AXI wrapper
    if f'extern "C" void {kernel_name}(' not in source_code:
        errors.append("FAIL: extern 'C' AXI wrapper function not found.")

    # Check 2: Presence of AXI interface pragmas
    if "#pragma HLS INTERFACE m_axi" not in source_code:
        errors.append("FAIL: #pragma HLS INTERFACE m_axi not found.")
    if "#pragma HLS INTERFACE s_axilite" not in source_code:
        errors.append("FAIL: #pragma HLS INTERFACE s_axilite not found.")

    # Check 3: Presence of the DATAFLOW pragma in the AXI wrapper
    if "#pragma HLS DATAFLOW" not in source_code:
        # A bit more specific check to ensure it's in the right place
        try:
            axi_wrapper_part = source_code.split(f'extern "C" void {kernel_name}(')[1]
            if "#pragma HLS DATAFLOW" not in axi_wrapper_part:
                errors.append("FAIL: #pragma HLS DATAFLOW not found inside the AXI wrapper.")
        except IndexError:
            pass  # The wrapper itself is missing, already caught by check 1

    # Check 4: Presence of the three key static functions
    if "static void mem_to_stream_func(" not in source_code:
        errors.append("FAIL: mem_to_stream_func not found.")
    if f"static void {kernel_name}_dataflow(" not in source_code:
        errors.append("FAIL: dataflow core function not found.")
    if "static void stream_to_mem_func(" not in source_code:
        errors.append("FAIL: stream_to_mem_func not found.")

    # Check 5: Calls to the three key functions inside the AXI wrapper
    try:
        axi_wrapper_part = source_code.split(f'extern "C" void {kernel_name}(')[1]
        if "mem_to_stream_func(" not in axi_wrapper_part:
            errors.append("FAIL: Call to mem_to_stream_func not found in AXI wrapper.")
        if f"{kernel_name}_dataflow(" not in axi_wrapper_part:
            errors.append("FAIL: Call to dataflow core function not found in AXI wrapper.")
        if "stream_to_mem_func(" not in axi_wrapper_part:
            errors.append("FAIL: Call to stream_to_mem_func not found in AXI wrapper.")
    except IndexError:
        pass

    # 5. Report results
    if not errors:
        print("AXI Kernel Generation Test: SUCCESS")
        return True
    else:
        print("AXI Kernel Generation Test: FAILED")
        for error in errors:
            print(f"  - {error}")
        print("\n--- Generated Source Code for Debugging ---")
        print(source_code)
        return False


if __name__ == "__main__":
    test_dir = project_root / "implement_tests" / "task_1_2"
    test_dir.mkdir(exist_ok=True)

    print(f"Running tests for Task 1.2. Test script location: {test_dir}")

    success = test_axi_kernel_generation()

    if success:
        print("\nAll tests for Task 1.2 PASSED.")
        with open(test_dir / "SUCCESS", "w") as f:
            f.write("Task 1.2 completed and verified.")
        sys.exit(0)
    else:
        print("\nSome tests for Task 1.2 FAILED.")
        sys.exit(1)
