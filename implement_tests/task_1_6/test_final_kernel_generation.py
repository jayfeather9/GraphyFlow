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


def strip_whitespace(text):
    """A helper function to remove all whitespace and newlines for comparison."""
    return "".join(text.split())


def test_final_axi_kernel_generation():
    print("--- Testing Task 1.6: Final AXI Kernel Generation ---")

    # 1. Define the standard test graph
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

    # 3. Instantiate BackendManager and run the full backend generation
    bkd_mng = BackendManager()
    kernel_name = "graphyflow"
    header_code, source_code = bkd_mng.generate_backend(comp_col, g, kernel_name)

    # 4. Perform a series of checks on the generated source code
    errors = []

    # Get names for checking
    input_port = bkd_mng.axi_input_ports[0]
    output_port = bkd_mng.axi_output_ports[0]
    input_batch_type = bkd_mng.batch_type_map[bkd_mng.type_map[input_port.data_type]]

    # Check 1: Complete and correct extern "C" function signature
    expected_signature_pattern = f'extern"C"void{kernel_name}(const{input_batch_type.name}*{input_port.unique_name},KernelOutputBatch*{output_port.unique_name},int*stop_flag,uint16_tinput_length_in_batches)'
    if strip_whitespace(expected_signature_pattern) not in strip_whitespace(source_code):
        errors.append(f"FAIL: Top-level AXI wrapper signature is incorrect.")
        print(f"DEBUG: Expected signature pattern: {expected_signature_pattern}")

    # Check 2: Correct call to mem_to_stream_func
    expected_m2s_call = f"mem_to_stream_func({input_port.unique_name},{input_port.unique_name}_internal_stream,input_length_in_batches);"
    if strip_whitespace(expected_m2s_call) not in strip_whitespace(source_code):
        errors.append(f"FAIL: Call to mem_to_stream_func is incorrect.")

    # Check 3: Correct call to graphyflow_dataflow
    expected_df_call = f"{kernel_name}_dataflow({input_port.unique_name}_internal_stream,{output_port.unique_name}_internal_stream);"
    if strip_whitespace(expected_df_call) not in strip_whitespace(source_code):
        errors.append(f"FAIL: Call to {kernel_name}_dataflow is incorrect.")

    # Check 4: Correct call to stream_to_mem_func
    expected_s2m_call = (
        f"stream_to_mem_func({output_port.unique_name}_internal_stream,{output_port.unique_name});"
    )
    if strip_whitespace(expected_s2m_call) not in strip_whitespace(source_code):
        errors.append(f"FAIL: Call to stream_to_mem_func is incorrect.")

    # 5. Report results
    if not errors:
        print("Final AXI Kernel Generation Test: SUCCESS")
        return True
    else:
        print("Final AXI Kernel Generation Test: FAILED")
        for error in errors:
            print(f"  - {error}")
        print("\n--- Generated Source Code for Debugging ---")
        print(source_code)
        return False


if __name__ == "__main__":
    test_dir = project_root / "implement_tests" / "task_1_6"
    test_dir.mkdir(exist_ok=True)

    print(f"Running tests for Task 1.6. Test script location: {test_dir}")

    success = test_final_axi_kernel_generation()

    if success:
        print("\nAll tests for Task 1.6 PASSED.")
        print("\n==============================================")
        print("      PHASE 1 HAS BEEN SUCCESSFULLY COMPLETED!      ")
        print("==============================================")
        with open(test_dir / "SUCCESS", "w") as f:
            f.write("Task 1.6 and all of Phase 1 completed and verified.")
        sys.exit(0)
    else:
        print("\nSome tests for Task 1.6 FAILED.")
        sys.exit(1)
